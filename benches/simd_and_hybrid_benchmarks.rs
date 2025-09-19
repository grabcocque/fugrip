//! Consolidated SIMD and Hybrid Sweep Performance Benchmarks
//!
//! This benchmark suite consolidates all vectorization, SIMD, and hybrid strategy
//! performance tests into a single comprehensive benchmark that covers:
//! - Pure SIMD sweep performance
//! - Hybrid SIMD+sparse strategy effectiveness
//! - Verse-style optimization comparisons
//! - Adaptive density-based switching
//! - Memory bandwidth utilization

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fugrip::simd_sweep::SimdBitvector;
use mmtk::util::Address;
use std::hint::black_box;
use std::time::Duration;

/// Create test bitvector with specified pattern
fn create_test_bitvector(heap_size: usize, density_percent: usize) -> SimdBitvector {
    let heap_base = unsafe { Address::from_usize(0x50000000) };
    let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

    // Create uniform density pattern
    for i in (0..heap_size / 16).step_by(100) {
        for j in 0..density_percent {
            if i + j < heap_size / 16 {
                let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + (i + j) * 16) };
                bitvector.mark_live(obj_addr);
            }
        }
    }
    bitvector
}

/// Core SIMD sweep performance benchmarks
fn bench_pure_simd_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sweep_pure");
    group.measurement_time(Duration::from_secs(5));

    let heap_sizes = vec![64 * 1024, 256 * 1024, 1024 * 1024];
    let densities = vec![25, 50, 75];

    for &heap_size in &heap_sizes {
        for &density in &densities {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}kb", heap_size / 1024),
                    format!("{}percent", density),
                ),
                &(heap_size, density),
                |b, &(heap_size, density)| {
                    b.iter_batched(
                        || create_test_bitvector(heap_size, density),
                        |bitvector| black_box(bitvector.simd_sweep()),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Hybrid strategy performance across density spectrum
fn bench_hybrid_strategy_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_strategy_effectiveness");
    group.measurement_time(Duration::from_secs(5));

    let heap_size = 512 * 1024; // 512KB for multiple chunks
    let densities = vec![10, 20, 30, 35, 40, 50, 60, 80];

    for density in densities {
        group.bench_with_input(
            BenchmarkId::new("hybrid_adaptive", format!("{}percent", density)),
            &density,
            |b, &density| {
                b.iter_batched(
                    || create_test_bitvector(heap_size, density),
                    |bitvector| {
                        let stats = black_box(bitvector.hybrid_sweep());

                        // Verify strategy selection is correct
                        if density >= 35 {
                            assert!(stats.simd_chunks_processed >= stats.sparse_chunks_processed,
                                "Dense workload should prefer SIMD: density={}%, simd={}, sparse={}",
                                density, stats.simd_chunks_processed, stats.sparse_chunks_processed);
                        } else {
                            assert!(stats.sparse_chunks_processed >= stats.simd_chunks_processed,
                                "Sparse workload should prefer sparse scanning: density={}%, simd={}, sparse={}",
                                density, stats.simd_chunks_processed, stats.sparse_chunks_processed);
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

/// Head-to-head comparison: Hybrid vs Pure SIMD
fn bench_hybrid_vs_pure_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_vs_pure_comparison");
    group.measurement_time(Duration::from_secs(4));

    let test_patterns = vec![
        ("dense_80_percent", 80),
        ("sparse_15_percent", 15),
        ("threshold_35_percent", 35),
        ("mixed_45_percent", 45),
    ];

    let heap_size = 256 * 1024;

    for (pattern_name, density) in test_patterns {
        // Hybrid approach
        group.bench_with_input(
            BenchmarkId::new("hybrid", pattern_name),
            &density,
            |b, &density| {
                b.iter_batched(
                    || create_test_bitvector(heap_size, density),
                    |bitvector| black_box(bitvector.hybrid_sweep()),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Pure SIMD approach
        group.bench_with_input(
            BenchmarkId::new("pure_simd", pattern_name),
            &density,
            |b, &density| {
                b.iter_batched(
                    || create_test_bitvector(heap_size, density),
                    |bitvector| black_box(bitvector.simd_sweep()),
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Verse-style optimization benchmarks
fn bench_verse_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("verse_optimizations");
    group.measurement_time(Duration::from_secs(4));

    let heap_base = unsafe { Address::from_usize(0x60000000) };
    let heap_size = 512 * 1024;

    group.bench_function("verse_style_sparse_patterns", |b| {
        b.iter_batched(
            || {
                let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

                // Create pattern that ensures both dense and sparse chunks
                let total_objects = heap_size / 16;
                let _objects_per_chunk = bitvector.objects_per_chunk();

                // Fill first half of heap with high density (60% - above 35% threshold)
                for i in 0..(total_objects / 2) {
                    if i % 5 < 3 {
                        // 60% density
                        let obj_addr =
                            unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
                        bitvector.mark_live(obj_addr);
                    }
                }

                // Fill second half with low density (20% - below 35% threshold)
                for i in (total_objects / 2)..total_objects {
                    if i % 10 < 2 {
                        // 20% density
                        let obj_addr =
                            unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
                        bitvector.mark_live(obj_addr);
                    }
                }
                bitvector
            },
            |bitvector| {
                let stats = black_box(bitvector.hybrid_sweep());

                // Verify both strategies are utilized for Verse-style patterns
                assert!(
                    stats.simd_chunks_processed > 0,
                    "Should use SIMD for dense clusters"
                );
                assert!(
                    stats.sparse_chunks_processed > 0,
                    "Should use sparse scanning for low-density regions"
                );

                stats
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Large heap scalability benchmarks
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_scalability");
    group.measurement_time(Duration::from_secs(6));

    let heap_sizes = vec![
        1024 * 1024,      // 1MB
        4 * 1024 * 1024,  // 4MB
        16 * 1024 * 1024, // 16MB
    ];

    for &heap_size in &heap_sizes {
        group.bench_with_input(
            BenchmarkId::new(
                "hybrid_large_heap",
                format!("{}mb", heap_size / (1024 * 1024)),
            ),
            &heap_size,
            |b, &heap_size| {
                b.iter_batched(
                    || {
                        let bitvector = create_test_bitvector(heap_size, 40); // 40% density

                        // Verify reasonable performance characteristics
                        assert!(heap_size >= 1024 * 1024, "Test heap should be at least 1MB");
                        bitvector
                    },
                    |bitvector| {
                        let stats = black_box(bitvector.hybrid_sweep());

                        // Validate performance scales reasonably
                        assert!(
                            stats.throughput_objects_per_sec > 100_000,
                            "Throughput too low: {} objects/sec for {} MB heap",
                            stats.throughput_objects_per_sec,
                            heap_size / (1024 * 1024)
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

/// Memory bandwidth utilization benchmarks
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth_utilization");
    group.measurement_time(Duration::from_secs(4));

    let heap_base = unsafe { Address::from_usize(0x70000000) };
    let heap_size = 2 * 1024 * 1024; // 2MB

    group.bench_function("cache_line_aligned_access", |b| {
        b.iter_batched(
            || {
                let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

                // Create cache-line aligned access pattern
                let cache_line_objects = 64 / 16; // 4 objects per cache line

                for i in (0..heap_size / 16).step_by(cache_line_objects * 2) {
                    // Mark one cache line, skip one cache line
                    for j in 0..cache_line_objects {
                        if i + j < heap_size / 16 {
                            let obj_addr =
                                unsafe { Address::from_usize(heap_base.as_usize() + (i + j) * 16) };
                            bitvector.mark_live(obj_addr);
                        }
                    }
                }
                bitvector
            },
            |bitvector| {
                let stats = black_box(bitvector.hybrid_sweep());

                // Calculate and validate memory bandwidth
                let total_memory_scanned = heap_size;
                let time_ns = stats.sweep_time_ns;
                if time_ns > 0 {
                    let bandwidth_gb_per_sec =
                        (total_memory_scanned as f64 * 1e9) / (time_ns as f64 * 1e9);

                    // Should achieve reasonable bandwidth
                    assert!(
                        bandwidth_gb_per_sec > 0.1,
                        "Bandwidth too low: {:.2} GB/s for {} MB heap",
                        bandwidth_gb_per_sec,
                        heap_size / (1024 * 1024)
                    );
                }

                stats
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pure_simd_sweep,
    bench_hybrid_strategy_effectiveness,
    bench_hybrid_vs_pure_comparison,
    bench_verse_optimizations,
    bench_scalability,
    bench_memory_bandwidth
);

criterion_main!(benches);
