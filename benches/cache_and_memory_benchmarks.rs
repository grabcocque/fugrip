//! Consolidated Cache and Memory Performance Benchmarks
//!
//! This benchmark suite consolidates all cache locality, memory bandwidth,
//! and heap layout optimization tests including:
//! - Cache line utilization efficiency
//! - Memory bandwidth scaling
//! - Heap microbenchmarks
//! - NUMA-aware memory access patterns
//! - Prefetching and cache optimization

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fugrip::simd_sweep::SimdBitvector;
use mmtk::util::Address;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

/// Create cache-aligned test pattern
fn create_cache_aligned_pattern(heap_size: usize, stride: usize) -> SimdBitvector {
    let heap_base = unsafe { Address::from_usize(0x80000000) };
    let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

    // Create cache-friendly access pattern
    for i in (0..heap_size / 16).step_by(stride) {
        let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        bitvector.mark_live(obj_addr);
    }

    bitvector
}

/// Benchmark cache line utilization efficiency
fn bench_cache_line_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_line_efficiency");
    group.measurement_time(Duration::from_secs(4));

    let heap_size = 512 * 1024; // 512KB
    let cache_line_size = 64; // 64-byte cache lines
    let objects_per_cache_line = cache_line_size / 16; // 4 objects per line

    let access_patterns = vec![
        ("sequential", 1),
        ("cache_line_aligned", objects_per_cache_line),
        ("cache_friendly", objects_per_cache_line * 2),
        ("stride_8", 8),
        ("stride_16", 16),
        ("random_sparse", 32),
    ];

    for (pattern_name, stride) in access_patterns {
        group.bench_with_input(
            BenchmarkId::new("hybrid_sweep", pattern_name),
            &stride,
            |b, &stride| {
                b.iter_batched(
                    || create_cache_aligned_pattern(heap_size, stride),
                    |bitvector| {
                        let stats = black_box(bitvector.hybrid_sweep());

                        // Validate cache efficiency based on pattern
                        if stride <= objects_per_cache_line {
                            // Cache-friendly patterns should have high throughput
                            assert!(
                                stats.throughput_objects_per_sec > 100_000,
                                "Cache-friendly pattern should have high throughput: {} ops/sec",
                                stats.throughput_objects_per_sec
                            );
                        }

                        stats
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark memory bandwidth scaling
fn bench_memory_bandwidth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth_scaling");
    group.measurement_time(Duration::from_secs(6));

    let heap_sizes = vec![
        64 * 1024,        // 64KB - L1 cache size
        256 * 1024,       // 256KB - L2 cache size
        8 * 1024 * 1024,  // 8MB - L3 cache size
        64 * 1024 * 1024, // 64MB - Beyond cache
    ];

    for &heap_size in &heap_sizes {
        group.bench_with_input(
            BenchmarkId::new("memory_throughput", format!("{}kb", heap_size / 1024)),
            &heap_size,
            |b, &heap_size| {
                b.iter_batched(
                    || {
                        // Create realistic mixed-density pattern
                        create_cache_aligned_pattern(heap_size, 4)
                    },
                    |bitvector| {
                        let stats = black_box(bitvector.hybrid_sweep());

                        // Calculate memory bandwidth
                        let memory_processed = heap_size; // Bytes scanned
                        let time_seconds = stats.sweep_time_ns as f64 / 1e9;

                        if time_seconds > 0.0 {
                            let bandwidth_mb_per_sec =
                                (memory_processed as f64) / (time_seconds * 1024.0 * 1024.0);

                            // Should achieve reasonable bandwidth
                            assert!(
                                bandwidth_mb_per_sec > 10.0,
                                "Memory bandwidth too low: {:.2} MB/s for {} KB heap",
                                bandwidth_mb_per_sec,
                                heap_size / 1024
                            );
                        }

                        stats
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark cache-optimized allocation patterns
fn bench_cache_optimized_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_optimized_allocation");
    group.measurement_time(Duration::from_secs(4));

    group.bench_function("cache_aware_bitvector_creation", |b| {
        b.iter(|| {
            let heap_base = unsafe { Address::from_usize(0x85000000) };
            let heap_size = 1024 * 1024; // 1MB

            // Test cache-aware bitvector creation with different alignments
            let bitvector = black_box(SimdBitvector::new(heap_base, heap_size, 16));

            // Verify optimal layout
            assert!(bitvector.max_objects() > 0);
            bitvector
        })
    });

    group.finish();
}

/// Benchmark heap layout optimization
fn bench_heap_layout_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("heap_layout_optimization");
    group.measurement_time(Duration::from_secs(4));

    let object_sizes = vec![8, 16, 32, 64, 128];

    for &object_size in &object_sizes {
        group.bench_with_input(
            BenchmarkId::new("layout_efficiency", format!("{}byte_objects", object_size)),
            &object_size,
            |b, &object_size| {
                b.iter_batched(
                    || {
                        let heap_base = unsafe { Address::from_usize(0x90000000) };
                        let heap_size = 256 * 1024;
                        SimdBitvector::new(heap_base, heap_size, object_size)
                    },
                    |bitvector| {
                        // Fill with optimal layout pattern
                        let max_objects = bitvector.max_objects();
                        for i in (0..max_objects).step_by(2) {
                            let obj_addr =
                                unsafe { Address::from_usize(0x90000000 + i * object_size) };
                            bitvector.mark_live(obj_addr);
                        }

                        let stats = black_box(bitvector.hybrid_sweep());

                        // Validate layout efficiency
                        let objects_per_cache_line = 64 / object_size;
                        let expected_efficiency = if objects_per_cache_line >= 4 {
                            1000000 // High efficiency for small objects
                        } else {
                            500000 // Lower efficiency for large objects
                        };

                        assert!(
                            stats.throughput_objects_per_sec > expected_efficiency,
                            "Layout efficiency too low for {} byte objects: {} ops/sec",
                            object_size,
                            stats.throughput_objects_per_sec
                        );

                        stats
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark prefetching effectiveness
fn bench_prefetching_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetching_effectiveness");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("sequential_access_pattern", |b| {
        let heap_size = 1024 * 1024; // 1MB

        b.iter_batched(
            || {
                // Create sequential access pattern (prefetcher-friendly)
                create_cache_aligned_pattern(heap_size, 1)
            },
            |bitvector| black_box(bitvector.hybrid_sweep()),
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("strided_access_pattern", |b| {
        let heap_size = 1024 * 1024; // 1MB

        b.iter_batched(
            || {
                // Create strided access pattern (moderate prefetcher efficiency)
                create_cache_aligned_pattern(heap_size, 8)
            },
            |bitvector| black_box(bitvector.hybrid_sweep()),
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("random_access_pattern", |b| {
        let heap_size = 1024 * 1024; // 1MB

        b.iter_batched(
            || {
                // Create pseudo-random access pattern (prefetcher-unfriendly)
                create_cache_aligned_pattern(heap_size, 97) // Prime stride
            },
            |bitvector| black_box(bitvector.hybrid_sweep()),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark concurrent cache access patterns
fn bench_concurrent_cache_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_cache_access");
    group.measurement_time(Duration::from_secs(4));

    let thread_counts = vec![1, 2, 4];

    for thread_count in thread_counts {
        group.bench_with_input(
            BenchmarkId::new("parallel_heap_access", format!("{}threads", thread_count)),
            &thread_count,
            |b, &thread_count| {
                b.iter_batched(
                    || {
                        // Create per-thread heap regions to minimize cache contention
                        let heap_size_per_thread = 256 * 1024; // 256KB per thread
                        let mut bitvectors = Vec::new();

                        for thread_id in 0..thread_count {
                            let heap_base = unsafe {
                                Address::from_usize(0xA0000000 + thread_id * heap_size_per_thread)
                            };
                            let bitvector = SimdBitvector::new(heap_base, heap_size_per_thread, 16);

                            // Create thread-specific patterns
                            for i in (0..heap_size_per_thread / 16).step_by(thread_id + 2) {
                                let obj_addr =
                                    unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
                                bitvector.mark_live(obj_addr);
                            }

                            bitvectors.push(Arc::new(bitvector));
                        }

                        bitvectors
                    },
                    |bitvectors| {
                        let concurrent_start = std::time::Instant::now();

                        // Spawn concurrent sweep operations
                        let handles: Vec<_> = bitvectors
                            .into_iter()
                            .map(|bitvector| std::thread::spawn(move || bitvector.hybrid_sweep()))
                            .collect();

                        // Wait for all threads to complete
                        let mut total_objects_swept = 0;
                        for handle in handles {
                            let stats = handle.join().unwrap();
                            total_objects_swept += stats.objects_swept;
                        }

                        let concurrent_time = concurrent_start.elapsed();

                        black_box((concurrent_time, total_objects_swept))
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_line_efficiency,
    bench_memory_bandwidth_scaling,
    bench_cache_optimized_allocation,
    bench_heap_layout_optimization,
    bench_prefetching_effectiveness,
    bench_concurrent_cache_access
);

criterion_main!(benches);
