//! Comprehensive FUGC Performance Benchmark Suite
//!
//! This consolidated benchmark suite covers all key performance aspects of FUGC:
//! - Write barrier fast path optimizations
//! - SIMD and hybrid sweep performance
//! - Cache locality optimization
//! - GC coordination and handshake protocols
//! - Memory allocation patterns
//! - Tricolor marking and concurrent collection
//!
//! Consolidates and replaces multiple smaller benchmark files to reduce overlap
//! and provide a comprehensive performance view.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fugrip::concurrent::ObjectColor;
use fugrip::simd_sweep::SimdBitvector;
use fugrip::test_utils::TestFixture;
use fugrip::thread::MutatorThread;
use mmtk::util::{Address, ObjectReference};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

const TEST_HEAP_BASE: usize = 0x10000000;
const TEST_HEAP_SIZE: usize = 16 * 1024 * 1024; // 16MB

/// Write Barrier Performance Benchmarks
fn write_barrier_benchmarks(c: &mut Criterion) {
    let fixture = TestFixture::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, 4);
    let coordinator = &fixture.coordinator;
    let write_barrier = coordinator.write_barrier();
    let heap_base = unsafe { Address::from_usize(TEST_HEAP_BASE) };

    // Create test objects within heap bounds
    let old_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
    let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x2000usize) };
    let mut slot = old_obj;

    let mut group = c.benchmark_group("write_barriers");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    // Benchmark inactive vs active write barriers
    for (name, is_active) in [("inactive", false), ("active", true)] {
        if is_active {
            write_barrier.activate();
        } else {
            write_barrier.deactivate();
        }

        group.bench_function(format!("fast_path_{}", name), |b| {
            b.iter(|| unsafe {
                write_barrier.write_barrier_fast(
                    std::hint::black_box(&mut slot as *mut _),
                    std::hint::black_box(new_obj),
                );
            });
        });

        group.bench_function(format!("bulk_operations_{}", name), |b| {
            let refs = vec![new_obj; 1000];
            b.iter(|| {
                for &obj_ref in std::hint::black_box(&refs) {
                    unsafe {
                        write_barrier.write_barrier_fast(
                            std::hint::black_box(&mut slot as *mut _),
                            std::hint::black_box(obj_ref),
                        );
                    }
                }
            });
        });
    }

    group.finish();
}

/// SIMD Sweep Performance Benchmarks
fn simd_sweep_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sweep");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    // Test different heap sizes and densities
    for &heap_size in &[64 * 1024, 256 * 1024, 1024 * 1024] {
        for &density in &[5, 25, 50, 90] {
            let heap_base = unsafe { Address::from_usize(TEST_HEAP_BASE) };
            let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

            // Create test pattern with specified density
            let total_objects = heap_size / 16;
            let marked_objects = (total_objects * density) / 100;
            for i in 0..marked_objects {
                let offset = (i * total_objects / marked_objects) * 16;
                let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + offset) };
                bitvector.mark_live(obj_addr);
            }

            group.throughput(Throughput::Bytes(heap_size as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "hybrid_sweep",
                    format!("{}KB_{}%", heap_size / 1024, density),
                ),
                &bitvector,
                |b, bv| {
                    b.iter(|| {
                        black_box(bv.hybrid_sweep());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Cache Locality Benchmarks
fn cache_locality_benchmarks(c: &mut Criterion) {
    let fixture = TestFixture::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, 4);
    let coordinator = &fixture.coordinator;
    let tricolor = coordinator.tricolor_marking();

    let mut group = c.benchmark_group("cache_locality");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    // Test different access patterns
    for &stride in &[64, 256, 1024, 4096] {
        let objects = (0..1000)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(
                    TEST_HEAP_BASE + i * stride,
                ))
            })
            .collect::<Vec<_>>();

        group.bench_with_input(
            BenchmarkId::new("sequential_marking", stride),
            &objects,
            |b, objs| {
                b.iter(|| {
                    for &obj in black_box(objs) {
                        tricolor.set_color(obj, ObjectColor::Grey);
                    }
                });
            },
        );

        // Random access pattern
        let mut random_objects = objects.clone();
        fastrand::shuffle(&mut random_objects);

        group.bench_with_input(
            BenchmarkId::new("random_marking", stride),
            &random_objects,
            |b, objs| {
                b.iter(|| {
                    for &obj in black_box(objs) {
                        tricolor.set_color(obj, ObjectColor::Grey);
                    }
                });
            },
        );
    }

    group.finish();
}

/// GC Coordination Benchmarks
fn gc_coordination_benchmarks(c: &mut Criterion) {
    let fixture = TestFixture::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, 4);
    let coordinator = &fixture.coordinator;
    let thread_registry = Arc::clone(fixture.thread_registry());

    let mut group = c.benchmark_group("gc_coordination");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    // Benchmark GC triggering
    group.bench_function("gc_trigger_cycle", |b| {
        b.iter(|| {
            coordinator.trigger_gc();
            black_box(coordinator.wait_until_idle(Duration::from_millis(100)));
        });
    });

    // Benchmark safepoint coordination using Rayon for background thread simulation
    for num_threads in [1, 2, 4, 8] {
        let mutators: Vec<_> = (0..num_threads)
            .map(|i| {
                let mutator = MutatorThread::new(i);
                thread_registry.register(mutator.clone());
                mutator
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("safepoint_coordination", num_threads),
            &num_threads,
            |b, _| {
                b.iter(|| {
                    // Simulate safepoint request across all threads using Rayon
                    use rayon::prelude::*;
                    black_box(mutators.par_iter().for_each(|mutator| {
                        mutator.poll_safepoint();
                    }));
                });
            },
        );

        // Clean up mutators
        for mutator in mutators {
            thread_registry.unregister(mutator.id());
        }
    }

    group.finish();
}

/// Memory Allocation Pattern Benchmarks
fn allocation_benchmarks(c: &mut Criterion) {
    use fugrip::plan::FugcPlanManager;

    let _fixture = TestFixture::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, 4);
    let plan_manager = FugcPlanManager::new();

    let mut group = c.benchmark_group("allocation_patterns");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    // Test different allocation sizes and patterns
    for &size in &[64, 256, 1024, 4096] {
        for &alignment in &[8, 16, 32] {
            group.bench_with_input(
                BenchmarkId::new("alloc_info", format!("{}B_align{}", size, alignment)),
                &(size, alignment),
                |b, &(s, a)| {
                    b.iter(|| {
                        black_box(plan_manager.alloc_info(s, a));
                    });
                },
            );
        }
    }

    // Test allocation under different GC states
    for (name, trigger_gc) in [("idle", false), ("during_gc", true)] {
        if trigger_gc {
            plan_manager.gc();
        }

        group.bench_function(format!("allocation_pressure_{}", name), |b| {
            b.iter(|| {
                for i in 0..100 {
                    let size = 64 + (i % 512);
                    black_box(plan_manager.alloc_info(size, 8));
                }
            });
        });
    }

    group.finish();
}

/// Concurrent Marking Benchmarks
fn concurrent_marking_benchmarks(c: &mut Criterion) {
    let fixture = TestFixture::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, 4);
    let coordinator = &fixture.coordinator;
    let tricolor = coordinator.tricolor_marking();
    let heap_base = unsafe { Address::from_usize(TEST_HEAP_BASE) };

    let mut group = c.benchmark_group("concurrent_marking");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    // Create test object graph
    let objects: Vec<ObjectReference> = (0..1000)
        .map(|i| unsafe {
            ObjectReference::from_raw_address_unchecked(heap_base + (i as usize) * 64)
        })
        .collect();

    // Benchmark different marking scenarios
    for &num_threads in &[1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("parallel_marking", num_threads),
            &objects,
            |b, objs| {
                b.iter(|| {
                    // Use Rayon for parallel marking with automatic work stealing
                    use rayon::prelude::*;
                    black_box(objs.par_iter().for_each(|&obj| {
                        tricolor.set_color(obj, ObjectColor::Grey);
                    }));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    comprehensive_benches,
    write_barrier_benchmarks,
    simd_sweep_benchmarks,
    cache_locality_benchmarks,
    gc_coordination_benchmarks,
    allocation_benchmarks,
    concurrent_marking_benchmarks
);

criterion_main!(comprehensive_benches);
