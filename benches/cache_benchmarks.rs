//! Comprehensive benchmarks for cache locality optimization in FUGC
//!
//! These benchmarks measure the effectiveness of cache optimization strategies
//! for garbage collection operations, focusing on memory access patterns,
//! cache hit rates, and overall throughput.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mmtk::util::{Address, ObjectReference};
use std::sync::{Arc, Mutex};

use fugrip::cache_optimization::*;
use fugrip::concurrent::{ConcurrentMarkingCoordinator, TricolorMarking};

/// Generate a realistic object graph for benchmarking
fn generate_object_graph(num_objects: usize, locality_factor: f64) -> Vec<ObjectReference> {
    let mut objects: Vec<ObjectReference> = Vec::with_capacity(num_objects);
    let base_addr = 0x10000000;

    for i in 0..num_objects {
        let addr = if i > 0 && fastrand::f64() < locality_factor {
            // Create objects near previous objects for locality
            let prev_addr = objects[i - 1].to_raw_address().as_usize();
            prev_addr + 64 + (fastrand::usize(..256)) // Within ~4 cache lines
        } else {
            // Random placement
            base_addr + (i * 1024) + fastrand::usize(..512)
        };

        if let Some(obj_ref) =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(addr) })
        {
            objects.push(obj_ref);
        }
    }

    objects
}

/// Benchmark cache-aware allocation strategies
fn bench_cache_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_allocation");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("cache_aware", size), size, |b, &size| {
            let allocator = CacheAwareAllocator::new(
                unsafe { Address::from_usize(0x10000000) },
                1024 * 1024, // 1MB
            );

            b.iter(|| {
                allocator.reset();
                for _ in 0..100 {
                    let _ = allocator.allocate_aligned(size, 1);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("naive", size), size, |b, &size| {
            let mut offset = 0;

            b.iter(|| {
                offset = 0;
                for _ in 0..100 {
                    offset += size;
                    // Simulate naive allocation without cache alignment
                    std::hint::black_box(offset);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark locality-aware work stealing
fn bench_work_stealing(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing");

    for num_objects in [100, 1000, 10000].iter() {
        for locality in [0.1, 0.5, 0.9].iter() {
            let objects = generate_object_graph(*num_objects, *locality);

            group.bench_with_input(
                BenchmarkId::new(
                    "locality_aware",
                    format!("{}_{}", num_objects, (locality * 10.0) as i32),
                ),
                &objects,
                |b, objects| {
                    b.iter(|| {
                        let mut stealer = LocalityAwareWorkStealer::new(8);
                        stealer.add_objects(objects.clone());

                        let mut total_processed = 0;
                        while !stealer.get_next_batch(32).is_empty() {
                            total_processed += 32;
                        }
                        std::hint::black_box(total_processed);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "random",
                    format!("{}_{}", num_objects, (locality * 10.0) as i32),
                ),
                &objects,
                |b, objects| {
                    b.iter(|| {
                        let mut remaining = objects.clone();
                        let mut total_processed = 0;

                        while !remaining.is_empty() {
                            let batch_size = std::cmp::min(32, remaining.len());
                            remaining.drain(0..batch_size);
                            total_processed += batch_size;
                        }
                        std::hint::black_box(total_processed);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark cache-optimized marking vs standard marking
fn bench_marking_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("marking_strategies");

    for num_objects in [1000, 5000, 20000].iter() {
        for locality in [0.2, 0.8].iter() {
            let objects = generate_object_graph(*num_objects, *locality);

            group.bench_with_input(
                BenchmarkId::new(
                    "cache_optimized",
                    format!("{}_{}", num_objects, (locality * 10.0) as i32),
                ),
                &objects,
                |b, objects| {
                    let heap_base = unsafe { Address::from_usize(0x10000000) };
                    let tricolor = Arc::new(TricolorMarking::new(heap_base, 64 * 1024 * 1024));
                    let cache_marking = CacheOptimizedMarking::new(tricolor);

                    b.iter(|| {
                        cache_marking.mark_objects_batch(objects);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "standard",
                    format!("{}_{}", num_objects, (locality * 10.0) as i32),
                ),
                &objects,
                |b, objects| {
                    let heap_base = unsafe { Address::from_usize(0x10000000) };
                    let tricolor = Arc::new(TricolorMarking::new(heap_base, 64 * 1024 * 1024));

                    b.iter(|| {
                        for obj in objects {
                            tricolor.set_color(*obj, fugrip::concurrent::ObjectColor::Grey);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory layout optimization effectiveness
fn bench_memory_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");

    let optimizer = MemoryLayoutOptimizer::new();
    let sizes = vec![32, 64, 128, 256, 512];

    group.bench_function("layout_calculation", |b| {
        b.iter(|| {
            let layouts = optimizer.calculate_object_layout(&sizes);
            std::hint::black_box(layouts);
        });
    });

    group.bench_function("metadata_colocation", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let object_addr = unsafe { Address::from_usize(0x10000 + i * 128) };
                let metadata_addr = optimizer.colocate_metadata(object_addr, 16);
                std::hint::black_box(metadata_addr);
            }
        });
    });

    group.finish();
}

/// Benchmark comprehensive GC cycle with cache optimizations
fn bench_gc_cycle_cache_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("gc_cycle_cache");

    for heap_size in [1024 * 1024, 16 * 1024 * 1024].iter() {
        for object_count in [5000, 50000].iter() {
            group.bench_with_input(
                BenchmarkId::new(
                    "optimized",
                    format!("{}MB_{}objs", heap_size / (1024 * 1024), object_count),
                ),
                &(*heap_size, *object_count),
                |b, &(heap_size, object_count)| {
                    b.iter(|| {
                        let objects = generate_object_graph(object_count, 0.7);
                        let heap_base = unsafe { Address::from_usize(0x10000000) };

                        // Simulate GC cycle with cache optimizations
                        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
                        let cache_marking = CacheOptimizedMarking::new(Arc::clone(&tricolor));

                        // Simulate mark phase
                        cache_marking.mark_objects_batch(&objects);

                        // Simulate sweep phase with locality
                        let mut stealer = LocalityAwareWorkStealer::new(4);
                        stealer.add_objects(objects);

                        let mut processed = 0;
                        while !stealer.get_next_batch(64).is_empty() {
                            processed += 64;
                        }

                        std::hint::black_box(processed);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "standard",
                    format!("{}MB_{}objs", heap_size / (1024 * 1024), object_count),
                ),
                &(*heap_size, *object_count),
                |b, &(heap_size, object_count)| {
                    b.iter(|| {
                        let objects = generate_object_graph(object_count, 0.7);
                        let heap_base = unsafe { Address::from_usize(0x10000000) };

                        // Simulate standard GC cycle
                        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

                        // Standard mark phase
                        for obj in &objects {
                            tricolor.set_color(*obj, fugrip::concurrent::ObjectColor::Grey);
                        }

                        // Standard sweep phase
                        for obj in &objects {
                            tricolor.set_color(*obj, fugrip::concurrent::ObjectColor::Black);
                        }

                        std::hint::black_box(objects.len());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark concurrent marking with different thread counts and cache strategies
fn bench_concurrent_marking_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_marking_scalability");

    for thread_count in [1, 2, 4, 8].iter() {
        for cache_optimization in [false, true].iter() {
            group.bench_with_input(
                BenchmarkId::new(
                    if *cache_optimization {
                        "cache_opt"
                    } else {
                        "standard"
                    },
                    format!("{}threads", thread_count),
                ),
                &(*thread_count, *cache_optimization),
                |b, &(thread_count, cache_opt)| {
                    let objects = generate_object_graph(10000, 0.6);

                    b.iter(|| {
                        let heap_base = unsafe { Address::from_usize(0x10000000) };
                        let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
                        let global_roots =
                            Arc::new(Mutex::new(fugrip::roots::GlobalRoots::default()));

                        let coordinator = ConcurrentMarkingCoordinator::new(
                            heap_base,
                            64 * 1024 * 1024,
                            thread_count,
                            thread_registry,
                            global_roots,
                        );

                        if cache_opt {
                            coordinator.mark_objects_cache_optimized(&objects);
                        } else {
                            for obj in &objects {
                                coordinator
                                    .tricolor_marking
                                    .set_color(*obj, fugrip::concurrent::ObjectColor::Grey);
                            }
                        }

                        std::hint::black_box(objects.len());
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_allocation,
    bench_work_stealing,
    bench_marking_strategies,
    bench_memory_layout,
    bench_gc_cycle_cache_impact,
    bench_concurrent_marking_scalability
);

criterion_main!(benches);
