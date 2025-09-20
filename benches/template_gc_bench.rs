// Real MMTk integration: Comprehensive benchmark suite for FUGC garbage collection performance
// This benchmarks all critical FUGC 8-step protocol phases and concurrent operations

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use fugrip::test_utils::TestFixture;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;
use std::time::Duration;

// Benchmark helper for creating realistic object workloads
struct BenchmarkWorkload {
    objects: Vec<ObjectReference>,
    heap_base: usize,
    heap_size: usize,
}

impl BenchmarkWorkload {
    fn new(object_count: usize) -> Self {
        let heap_base = 0x10000000;
        let heap_size = 64 * 1024 * 1024; // 64MB heap

        let mut objects = Vec::with_capacity(object_count);
        for i in 0..object_count {
            let obj_addr = unsafe { Address::from_usize(heap_base + i * 256) };
            if let Some(obj) = ObjectReference::from_raw_address(obj_addr) {
                objects.push(obj);
            }
        }

        Self {
            objects,
            heap_base,
            heap_size,
        }
    }

    fn objects(&self) -> &[ObjectReference] {
        &self.objects
    }

    fn len(&self) -> usize {
        self.objects.len()
    }
}

// FUGC Phase 1: Idle State and GC Trigger Performance
fn bench_gc_trigger(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_gc_trigger");

    for &thread_count in &[1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("trigger_gc", thread_count),
            &thread_count,
            |b, _| {
                let fixture =
                    TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, thread_count);

                b.iter(|| {
                    // Real MMTk integration: FUGC Step 1 - GC trigger performance
                    // Measures coordinator overhead and trigger latency
                    black_box(fixture.coordinator.trigger_gc());

                    // Wait for completion to isolate trigger performance
                    let _ = black_box(
                        fixture
                            .coordinator
                            .wait_until_idle(Duration::from_millis(100)),
                    );
                });
            },
        );
    }

    group.finish();
}

// FUGC Phase 2: Write Barrier Activation Performance
fn bench_write_barrier_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_write_barrier_activation");

    let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);

    group.bench_function("activate_deactivate", |b| {
        b.iter(|| {
            // Real MMTk integration: FUGC Step 2 - Write barrier activation/deactivation
            // Measures barrier state transition overhead
            let write_barrier = black_box(fixture.coordinator.write_barrier());

            // Activation performance
            black_box(write_barrier.activate());

            // Simulate barrier operations during active phase
            for _ in 0..100 {
                let _ = black_box(write_barrier.is_active());
            }

            // Deactivation performance
            black_box(write_barrier.deactivate());
        });
    });

    group.finish();
}

// FUGC Phase 3: Black Allocation Performance
fn bench_black_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_black_allocation");

    let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
    let workload = BenchmarkWorkload::new(1000);

    group.bench_function("allocate_black_batch", |b| {
        b.iter(|| {
            // Real MMTk integration: FUGC Step 3 - Black allocation performance
            // Measures overhead of allocating objects as black during marking
            let allocator = black_box(fixture.coordinator.black_allocator());

            for &obj in workload.objects() {
                black_box(allocator.allocate_black(obj));
            }
        });
    });

    group.finish();
}

// FUGC Phase 4: Global Root Marking Performance
fn bench_global_root_marking(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_global_root_marking");

    for &root_count in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("mark_global_roots", root_count),
            &root_count,
            |b, _| {
                let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
                let mut root_addrs = Vec::with_capacity(root_count);

                // Create root addresses
                for i in 0..root_count {
                    let root_addr = unsafe { Address::from_usize(0x20000000 + i * 128) };
                    root_addrs.push(root_addr.to_mut_ptr::<u8>());
                }

                b.iter(|| {
                    // Real MMTk integration: FUGC Step 4 - Global root marking performance
                    // Measures performance of marking global roots as grey for concurrent marking
                    {
                        let mut roots = black_box(fixture.global_roots.lock().unwrap());
                        roots.clear();
                        for &root_ptr in &root_addrs {
                            roots.register(root_ptr);
                        }
                    }

                    // Perform root marking (triggers FUGC protocol)
                    black_box(fixture.coordinator.scan_thread_roots_at_safepoint());

                    // Clean up for next iteration
                    {
                        let mut roots = black_box(fixture.global_roots.lock().unwrap());
                        roots.clear();
                    }
                });
            },
        );
    }

    group.finish();
}

// FUGC Phase 5: Stack Scanning Performance (Handshake Protocol)
fn bench_stack_scanning(c: &mut Criterion) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let mut group = c.benchmark_group("f_stack_scanning");

    for &thread_count in &[1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("stack_scan_handshake", thread_count),
            &thread_count,
            |b, _| {
                let fixture = Arc::new(TestFixture::new_with_config(
                    0x10000000,
                    64 * 1024 * 1024,
                    thread_count,
                ));
                let running = Arc::new(AtomicBool::new(true));

                // Spawn background mutator threads that poll safepoints
                let mut handles = Vec::new();
                for i in 0..thread_count {
                    let fixture_clone = Arc::clone(&fixture);
                    let running_clone = Arc::clone(&running);

                    let handle = thread::spawn(move || {
                        if let Some(mutator) = fixture_clone.thread_registry().get(i) {
                            while running_clone.load(Ordering::Relaxed) {
                                // Real MMTk integration: FUGC Step 5 - Stack scanning handshake
                                // Measures handshake protocol performance for cooperative stack scanning
                                black_box(mutator.handler.poll_safepoint());
                                thread::yield_now();
                            }
                        }
                    });
                    handles.push(handle);
                }

                // Give threads time to start and register
                thread::sleep(Duration::from_millis(10));

                b.iter(|| {
                    // Perform handshake-based stack scanning
                    let result = black_box(fixture.coordinator.scan_thread_roots_at_safepoint());
                    assert!(result.is_ok(), "Stack scanning should succeed");
                });

                // Stop threads and wait for cleanup
                running.store(false, Ordering::Relaxed);
                for handle in handles {
                    let _ = handle.join();
                }
            },
        );
    }

    group.finish();
}

// FUGC Phase 6: Parallel Marking Performance
fn bench_parallel_marking(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_parallel_marking");

    for object_count in [1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("parallel_mark_objects", object_count),
            &object_count,
            |b, _| {
                let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
                let workload = BenchmarkWorkload::new(object_count);

                b.iter(|| {
                    // Real MMTk integration: FUGC Phase 6 - Parallel marking performance
                    // Measures work-stealing efficiency and concurrent marking throughput
                    let marked_count = black_box(
                        fixture
                            .coordinator
                            .parallel_coordinator()
                            .parallel_mark(workload.objects().to_vec()),
                    );

                    // Validate that marking occurred
                    assert!(
                        marked_count > 0,
                        "Should mark objects during parallel marking"
                    );

                    // Reset for next iteration
                    fixture.coordinator.parallel_coordinator().reset();
                });
            },
        );
    }

    group.finish();
}

// FUGC Phase 7: Write Barrier Deactivation Performance
fn bench_write_barrier_deactivation(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_write_barrier_deactivation");

    let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);

    group.bench_function("full_barrier_cycle", |b| {
        b.iter(|| {
            // Real MMTk integration: FUGC Phase 7 - Complete barrier cycle performance
            // Measures full activation→operations→deactivation cycle overhead
            let write_barrier = black_box(fixture.coordinator.write_barrier());

            // Start GC cycle (activates barriers)
            black_box(fixture.coordinator.trigger_gc());

            // Simulate concurrent barrier operations
            for i in 0..1000 {
                let obj_addr = unsafe { Address::from_usize(0x10000000 + i * 256) };
                if let Some(obj) = ObjectReference::from_raw_address(obj_addr) {
                    // Simulate barrier check during active GC
                    let _ = black_box(write_barrier.is_active());
                    let _ = black_box(write_barrier.barrier_slow_path(obj));
                }
            }

            // Wait for marking completion and barrier deactivation
            let _ = black_box(
                fixture
                    .coordinator
                    .wait_until_idle(Duration::from_millis(100)),
            );
        });
    });

    group.finish();
}

// FUGC Phase 8: Page-Based Sweep Performance
fn bench_page_sweeping(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_page_sweeping");

    for &page_count in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("page_sweep", page_count),
            &page_count,
            |b, _| {
                let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);

                b.iter(|| {
                    // Real MMTk integration: FUGC Phase 8 - Page-based sweep performance
                    // Measures efficiency of page-level sweeping and color management
                    for page_index in 0..page_count {
                        let color =
                            black_box(fixture.coordinator.page_allocation_color(page_index));

                        // Simulate sweep decision based on color
                        match color {
                            fugrip::fugc_coordinator::AllocationColor::Free => {
                                // Page is free, add to free list
                            }
                            fugrip::fugc_coordinator::AllocationColor::Allocated => {
                                // Page is allocated, check if needs sweeping
                            }
                            fugrip::fugc_coordinator::AllocationColor::Marked => {
                                // Page was marked, keep allocated
                            }
                        }
                    }

                    // Reset page state for next iteration
                    for page_index in 0..page_count {
                        // Simulate page reset after sweep
                        let _ = black_box(fixture.coordinator.page_allocation_color(page_index));
                    }
                });
            },
        );
    }

    group.finish();
}

// Comprehensive FUGC 8-Step Protocol Performance
fn bench_full_fugc_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("f_full_cycle");

    for &object_count in &[1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("complete_fugc_protocol", object_count),
            &object_count,
            |b, _| {
                let fixture = TestFixture::new_with_config(0x10000000, 128 * 1024 * 1024, 8);
                let workload = BenchmarkWorkload::new(object_count);

                b.iter(|| {
                    // Real MMTk integration: Complete FUGC 8-step protocol performance
                    // Measures end-to-end garbage collection cycle performance

                    // Step 1: Trigger GC
                    black_box(fixture.coordinator.trigger_gc());

                    // Step 2: Write barriers should be active automatically

                    // Step 3: Black allocation should be active
                    let allocator = black_box(fixture.coordinator.black_allocator());
                    for &obj in workload.objects().iter().take(100) {
                        black_box(allocator.allocate_black(obj));
                    }

                    // Step 4: Global root marking (triggered by GC trigger)

                    // Step 5: Stack scanning via handshake
                    let _ = black_box(fixture.coordinator.scan_thread_roots_at_safepoint());

                    // Step 6: Parallel marking
                    let marked_count = black_box(
                        fixture
                            .coordinator
                            .parallel_coordinator()
                            .parallel_mark(workload.objects().to_vec()),
                    );

                    // Step 7: Write barrier deactivation (automatic after marking)

                    // Step 8: Page sweeping (automatic after protocol completion)

                    // Wait for cycle completion
                    let completed = black_box(
                        fixture
                            .coordinator
                            .wait_until_idle(Duration::from_millis(500)),
                    );
                    assert!(completed, "FUGC cycle should complete");

                    // Validate cycle effectiveness
                    assert!(marked_count > 0, "Should mark objects during FUGC cycle");

                    // Get cycle statistics
                    let stats = black_box(fixture.coordinator.get_cycle_stats());
                    assert!(stats.total_objects > 0, "Should have processed objects");
                });
            },
        );
    }

    group.finish();
}

// Memory Management Operations Performance
fn bench_memory_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_management");

    let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
    let workload = BenchmarkWorkload::new(1000);

    group.bench_function("finalizer_registration", |b| {
        b.iter(|| {
            // Real MMTk integration: Finalizer registration performance
            // Measures overhead of registering object finalizers
            let memory_manager = black_box(fixture.container.memory_manager());

            for &obj in workload.objects().iter().take(100) {
                let finalizer = Box::new(|| {
                    // Simple finalizer for benchmarking
                    black_box(());
                });
                let _ = black_box(memory_manager.register_finalizer(obj, finalizer));
            }
        });
    });

    group.bench_function("weak_reference_creation", |b| {
        b.iter(|| {
            // Real MMTk integration: Weak reference creation performance
            // Measures overhead of creating weak references
            let memory_manager = black_box(fixture.container.memory_manager());

            for &obj in workload.objects().iter().take(100) {
                let _ = black_box(memory_manager.create_weak_reference(obj));
            }
        });
    });

    group.finish();
}

// Cache Optimization Performance
fn bench_cache_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_optimization");

    let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
    let workload = BenchmarkWorkload::new(5000);

    group.bench_function("cache_optimized_marking", |b| {
        b.iter(|| {
            // Real MMTk integration: Cache-optimized marking performance
            // Measures effectiveness of cache-friendly object marking patterns
            let marked_count = black_box(
                fixture
                    .coordinator
                    .parallel_coordinator()
                    .mark_objects_cache_optimized(workload.objects()),
            );

            assert!(
                marked_count > 0,
                "Should mark objects with cache optimization"
            );
        });
    });

    group.finish();
}

// SIMD Sweeping Performance
fn bench_simd_sweeping(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sweeping");

    for &bitvector_size in &[1024, 4096, 16384] {
        group.bench_with_input(
            BenchmarkId::new("simd_bitvector_sweep", bitvector_size),
            &bitvector_size,
            |b, _| {
                b.iter(|| {
                    // Real MMTk integration: SIMD-optimized sweeping performance
                    // Measures throughput of SIMD-accelerated bitvector operations
                    use fugrip::simd_sweep::SimdBitvector;

                    let bitvector = SimdBitvector::new(
                        unsafe { Address::from_usize(0x30000000) },
                        bitvector_size * 64, // Convert to bytes
                        16,                  // 16-byte alignment
                    );

                    // Mark some objects as live
                    for i in 0..(bitvector_size / 2) {
                        let addr = unsafe { Address::from_usize(0x30000000 + i * 64) };
                        let _ = black_box(bitvector.mark_live(addr));
                    }

                    // Perform SIMD sweep
                    let stats = black_box(bitvector.hybrid_sweep());

                    // Validate sweep results
                    assert!(
                        stats.marked_count > 0,
                        "Should mark objects during SIMD sweep"
                    );
                    assert!(
                        stats.swept_count > 0,
                        "Should sweep objects during SIMD sweep"
                    );
                });
            },
        );
    }

    group.finish();
}

// Handshake Protocol Stress Performance
fn bench_handshake_stress(c: &mut Criterion) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let mut group = c.benchmark_group("handshake_stress");

    for &thread_count in &[4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("handshake_throughput", thread_count),
            &thread_count,
            |b, _| {
                let fixture = Arc::new(TestFixture::new_with_config(
                    0x10000000,
                    64 * 1024 * 1024,
                    thread_count,
                ));
                let running = Arc::new(AtomicBool::new(true));

                // Spawn high-frequency mutator threads
                let mut handles = Vec::new();
                for i in 0..thread_count {
                    let fixture_clone = Arc::clone(&fixture);
                    let running_clone = Arc::clone(&running);

                    let handle = thread::spawn(move || {
                        if let Some(mutator) = fixture_clone.thread_registry().get(i) {
                            while running_clone.load(Ordering::Relaxed) {
                                // High-frequency safepoint polling
                                black_box(mutator.poll_safepoint());

                                // Add some work to simulate real mutator activity
                                if i % 4 == 0 {
                                    let obj_addr =
                                        unsafe { Address::from_usize(0x10000000 + i * 256) };
                                    if let Some(obj) = ObjectReference::from_raw_address(obj_addr) {
                                        let _ =
                                            black_box(mutator.register_stack_root(i as *mut u8));
                                    }
                                }
                            }
                        }
                    });
                    handles.push(handle);
                }

                // Give threads time to start
                thread::sleep(Duration::from_millis(20));

                b.iter(|| {
                    // Stress test handshake protocol under high contention
                    let start = std::time::Instant::now();

                    // Perform handshake with all threads
                    black_box(fixture.coordinator.scan_thread_roots_at_safepoint());
                    // Method returns (), so handshake succeeded if we reach here

                    let duration = start.elapsed();
                    black_box(duration);

                    // Quick validation
                    let stats = black_box(fixture.coordinator.get_cycle_stats());
                    assert!(stats.total_handshakes > 0, "Should perform handshakes");
                });

                // Stop threads
                running.store(false, Ordering::Relaxed);
                for handle in handles {
                    let _ = handle.join();
                }
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gc_trigger,
    bench_write_barrier_activation,
    bench_black_allocation,
    bench_global_root_marking,
    bench_stack_scanning,
    bench_parallel_marking,
    bench_write_barrier_deactivation,
    bench_page_sweeping,
    bench_full_fugc_cycle,
    bench_memory_management,
    bench_cache_optimization,
    bench_simd_sweeping,
    bench_handshake_stress
);
criterion_main!(benches);
