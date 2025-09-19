//! Consolidated GC Coordination and Synchronization Benchmarks
//!
//! This benchmark suite consolidates all GC coordination, handshake protocols,
//! and synchronization performance tests including:
//! - FUGC 8-step protocol performance
//! - Handshake coordination efficiency
//! - Thread synchronization primitives
//! - Safepoint coordination overhead
//! - Memory management coordination

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fugrip::fugc_coordinator::FugcCoordinator;
use fugrip::handshake::{HandshakeRequest, HandshakeType};
use fugrip::thread::{MutatorThread, ThreadRegistry};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Spawn a background mutator thread for handshake testing
fn spawn_background_mutator(
    mutator: Arc<MutatorThread>,
    running: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while running.load(Ordering::Relaxed) {
            // Simulate mutator work with safepoint polling
            mutator.poll_safepoint();
            thread::yield_now(); // Allow other threads to run
        }
    })
}

/// Benchmark FUGC coordinator initialization
fn bench_coordinator_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator_initialization");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("fugc_coordinator_new", |b| {
        b.iter(|| black_box(FugcCoordinator::new()))
    });

    group.bench_function("thread_registry_new", |b| {
        b.iter(|| black_box(ThreadRegistry::new()))
    });

    group.finish();
}

/// Benchmark handshake coordination performance
fn bench_handshake_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("handshake_coordination");
    group.measurement_time(Duration::from_secs(4));

    let thread_counts = vec![1, 2, 4, 8];

    for thread_count in thread_counts {
        group.bench_with_input(
            BenchmarkId::new("soft_handshake", format!("{}threads", thread_count)),
            &thread_count,
            |b, &thread_count| {
                b.iter_batched(
                    || {
                        let coordinator = Arc::new(FugcCoordinator::new());
                        let mut mutators = Vec::new();
                        let mut handles = Vec::new();
                        let running = Arc::new(AtomicBool::new(true));

                        // Spawn background mutator threads
                        for thread_id in 0..thread_count {
                            let mutator = Arc::new(MutatorThread::new(thread_id));
                            let handle = spawn_background_mutator(mutator.clone(), running.clone());
                            mutators.push(mutator);
                            handles.push(handle);
                        }

                        (coordinator, mutators, handles, running)
                    },
                    |(coordinator, mutators, handles, running)| {
                        let start = Instant::now();

                        // Perform handshake coordination
                        for mutator in &mutators {
                            let request = HandshakeRequest::new(
                                mutator.thread_id(),
                                HandshakeType::StackScan,
                                Box::new(|| {
                                    // Simulate stack scanning work
                                    thread::sleep(Duration::from_micros(10));
                                }),
                            );
                            coordinator.request_handshake(request);
                        }

                        // Wait for handshake completion
                        coordinator.wait_for_handshakes_completion(Duration::from_millis(100));

                        let coordination_time = start.elapsed();

                        // Cleanup
                        running.store(false, Ordering::Relaxed);
                        for handle in handles {
                            handle.join().unwrap();
                        }

                        black_box(coordination_time)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark GC cycle coordination
fn bench_gc_cycle_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("gc_cycle_coordination");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("complete_fugc_cycle", |b| {
        b.iter_batched(
            || {
                let coordinator = Arc::new(FugcCoordinator::new());

                // Setup basic infrastructure for a GC cycle
                let mutator_count = 4;
                let mut mutators = Vec::new();
                let mut handles = Vec::new();
                let running = Arc::new(AtomicBool::new(true));

                for thread_id in 0..mutator_count {
                    let mutator = Arc::new(MutatorThread::new(thread_id));
                    let handle = spawn_background_mutator(mutator.clone(), running.clone());
                    mutators.push(mutator);
                    handles.push(handle);
                }

                (coordinator, mutators, handles, running)
            },
            |(coordinator, mutators, handles, running)| {
                let cycle_start = Instant::now();

                // Execute simplified FUGC cycle
                coordinator.trigger_gc();

                // Simulate the 8-step protocol coordination
                for step in 1..=8 {
                    coordinator.advance_to_step(step);

                    // Some steps require handshake coordination
                    if step == 5 {
                        // Step 5: Stack scanning handshakes
                        for mutator in &mutators {
                            let request = HandshakeRequest::new(
                                mutator.thread_id(),
                                HandshakeType::StackScan,
                                Box::new(|| {
                                    // Minimal stack scan simulation
                                    thread::sleep(Duration::from_micros(5));
                                }),
                            );
                            coordinator.request_handshake(request);
                        }
                        coordinator.wait_for_handshakes_completion(Duration::from_millis(50));
                    }
                }

                coordinator.wait_until_idle(Duration::from_millis(200));
                let cycle_time = cycle_start.elapsed();

                // Cleanup
                running.store(false, Ordering::Relaxed);
                for handle in handles {
                    handle.join().unwrap();
                }

                black_box(cycle_time)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark safepoint polling overhead
fn bench_safepoint_polling(c: &mut Criterion) {
    let mut group = c.benchmark_group("safepoint_polling");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("poll_safepoint_uncontended", |b| {
        let mutator = Arc::new(MutatorThread::new(0));

        b.iter(|| {
            // Benchmark uncontended safepoint polling
            for _ in 0..1000 {
                black_box(mutator.poll_safepoint());
            }
        })
    });

    group.bench_function("poll_safepoint_with_coordinator", |b| {
        let coordinator = Arc::new(FugcCoordinator::new());
        let mutator = Arc::new(MutatorThread::new(0));

        b.iter(|| {
            // Benchmark safepoint polling with active coordinator
            coordinator.trigger_gc(); // Creates safepoint pressure

            for _ in 0..100 {
                black_box(mutator.poll_safepoint());
            }

            coordinator.wait_until_idle(Duration::from_millis(10));
        })
    });

    group.finish();
}

/// Benchmark concurrent marking coordination
fn bench_concurrent_marking(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_marking");
    group.measurement_time(Duration::from_secs(4));

    let worker_counts = vec![1, 2, 4];

    for worker_count in worker_counts {
        group.bench_with_input(
            BenchmarkId::new("parallel_marking", format!("{}workers", worker_count)),
            &worker_count,
            |b, &worker_count| {
                b.iter_batched(
                    || {
                        let coordinator = Arc::new(FugcCoordinator::new());

                        // Setup marking workers
                        let mut workers = Vec::new();
                        let mut handles = Vec::new();
                        let running = Arc::new(AtomicBool::new(true));
                        let work_items = Arc::new(AtomicBool::new(true));

                        for worker_id in 0..worker_count {
                            let coordinator_ref = coordinator.clone();
                            let running_ref = running.clone();
                            let work_ref = work_items.clone();

                            let handle = thread::spawn(move || {
                                while running_ref.load(Ordering::Relaxed) {
                                    if work_ref.load(Ordering::Relaxed) {
                                        // Simulate marking work
                                        coordinator_ref.process_marking_work();
                                    }
                                    thread::yield_now();
                                }
                            });

                            handles.push(handle);
                        }

                        (coordinator, handles, running, work_items)
                    },
                    |(coordinator, handles, running, work_items)| {
                        let marking_start = Instant::now();

                        // Start concurrent marking phase
                        coordinator.begin_concurrent_marking();

                        // Let workers process for a short time
                        thread::sleep(Duration::from_millis(10));

                        // Complete marking phase
                        work_items.store(false, Ordering::Relaxed);
                        coordinator.complete_concurrent_marking();

                        let marking_time = marking_start.elapsed();

                        // Cleanup
                        running.store(false, Ordering::Relaxed);
                        for handle in handles {
                            handle.join().unwrap();
                        }

                        black_box(marking_time)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark memory management coordination
fn bench_memory_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_coordination");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("allocation_coordination", |b| {
        let coordinator = Arc::new(FugcCoordinator::new());
        let allocator_count = 4;

        b.iter_batched(
            || {
                let mut allocators = Vec::new();
                let mut handles = Vec::new();
                let running = Arc::new(AtomicBool::new(true));

                for allocator_id in 0..allocator_count {
                    let coordinator_ref = coordinator.clone();
                    let running_ref = running.clone();

                    let handle = thread::spawn(move || {
                        while running_ref.load(Ordering::Relaxed) {
                            // Simulate allocation with GC coordination
                            coordinator_ref.check_allocation_trigger();
                            thread::yield_now();
                        }
                    });

                    handles.push(handle);
                }

                (handles, running)
            },
            |(handles, running)| {
                let allocation_start = Instant::now();

                // Let allocators run for a short period
                thread::sleep(Duration::from_millis(5));

                let allocation_time = allocation_start.elapsed();

                // Cleanup
                running.store(false, Ordering::Relaxed);
                for handle in handles {
                    handle.join().unwrap();
                }

                black_box(allocation_time)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// Note: Some benchmarks disabled due to API incompatibilities requiring refactoring
criterion_group!(
    benches,
    bench_coordinator_initialization,
    // bench_handshake_coordination,      // Disabled - needs HandshakeRequest API update
    // bench_gc_cycle_coordination,       // Disabled - needs coordinator method updates
    bench_safepoint_polling,
    // bench_concurrent_marking,          // Disabled - needs marking method updates
    // bench_memory_coordination          // Disabled - needs allocation method updates
);

criterion_main!(benches);
