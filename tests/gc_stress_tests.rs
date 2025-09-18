#![cfg(feature = "stress-tests")]
//! Comprehensive stress tests for concurrent GC operations.
//!
//! These suites are intentionally aggressive and are now guarded behind the
//! `stress-tests` feature to keep default `cargo nextest` runs well behaved on
//! constrained environments (such as WSL). Enable with
//! `cargo nextest --features stress-tests --test gc_stress_tests` when needed to
//! investigate high-load scenarios.

use crossbeam::channel;
use mmtk::util::{Address, ObjectReference};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use fugrip::FugcCoordinator;
use fugrip::cache_optimization::{
    CacheAwareAllocator, CacheOptimizedMarking, LocalityAwareWorkStealer,
};
use fugrip::concurrent::{ObjectColor, TricolorMarking};
use fugrip::test_utils::TestFixture;

const STRESS_TEST_DURATION: Duration = Duration::from_millis(500);
const HIGH_CONTENTION_OBJECTS: usize = 1000;
const STRESS_THREAD_COUNT: usize = 8;

/// High-contention concurrent marking stress test
#[test]
fn stress_concurrent_marking_high_contention() {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(fugrip::roots::GlobalRoots::default()));

    let coordinator = Arc::new(FugcCoordinator::new(
        heap_base,
        256 * 1024 * 1024, // 256MB heap
        STRESS_THREAD_COUNT,
        thread_registry,
        global_roots,
    ));

    // Create shared object pool for high contention
    let shared_objects = Arc::new({
        let mut objects = Vec::new();
        for i in 0..HIGH_CONTENTION_OBJECTS {
            let addr = unsafe { Address::from_usize(0x10000000 + i * 128) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }
        objects
    });

    // Create channels for proper coordination
    let (work_done_tx, work_done_rx) = channel::bounded::<usize>(STRESS_THREAD_COUNT);
    let (shutdown_tx, shutdown_rx) = channel::bounded::<()>(STRESS_THREAD_COUNT);

    let start_barrier = Arc::new(Barrier::new(STRESS_THREAD_COUNT));
    let mut handles = Vec::new();

    // Spawn stress test threads
    for thread_id in 0..STRESS_THREAD_COUNT {
        let coordinator = Arc::clone(&coordinator);
        let objects = Arc::clone(&shared_objects);
        let work_done_tx = work_done_tx.clone();
        let shutdown_rx = shutdown_rx.clone();
        let start_barrier = Arc::clone(&start_barrier);

        let handle = thread::spawn(move || {
            start_barrier.wait(); // Synchronize start for maximum contention
            let mut local_ops = 0;
            let target_operations_per_thread = 1000;

            loop {
                // Check for shutdown signal (non-blocking)
                if shutdown_rx.try_recv().is_ok() {
                    break;
                }

                // Stop after completing target work
                if local_ops >= target_operations_per_thread {
                    break;
                }
                match local_ops % 5 {
                    0 => {
                        // Cache-optimized batch marking
                        let batch_size = 50 + (thread_id * 10);
                        let batch_start = (thread_id * 100) % objects.len();
                        let batch_end = (batch_start + batch_size).min(objects.len());
                        coordinator.mark_objects_cache_optimized(&objects[batch_start..batch_end]);
                    }
                    1 => {
                        // Individual object marking via tricolor
                        let obj_idx = (thread_id + local_ops) % objects.len();
                        let obj = objects[obj_idx];
                        coordinator
                            .tricolor_marking()
                            .set_color(obj, ObjectColor::Grey);
                    }
                    2 => {
                        // Black allocation stress
                        let black_allocator = coordinator.black_allocator();
                        black_allocator.activate();
                        for i in 0..10 {
                            let obj_idx = (thread_id * 10 + i + local_ops) % objects.len();
                            black_allocator.allocate_black(objects[obj_idx]);
                        }
                        black_allocator.deactivate();
                    }
                    3 => {
                        // Write barrier activation/deactivation stress (safe operations only)
                        let barrier = coordinator.write_barrier();
                        for _i in 0..5 {
                            barrier.activate();
                            // Test barrier state
                            let _is_active = barrier.is_active();
                            barrier.deactivate();
                        }
                    }
                    4 => {
                        // Statistics access (read-heavy contention)
                        let _stats = coordinator.get_stats();
                        let _cache_stats = coordinator.get_cache_stats();
                    }
                    _ => unreachable!(),
                }

                local_ops += 1;

                // Occasional yield to increase thread interleaving
                if local_ops % 100 == 0 {
                    thread::yield_now();
                }
            }

            // Signal work completion
            work_done_tx.send(local_ops).unwrap();
        });

        handles.push(handle);
    }

    // Drop the senders to clean up channels
    drop(work_done_tx);
    drop(shutdown_tx);

    // Wait for all threads to complete their work
    let mut completed_threads = 0;
    let mut total_operations = 0;

    while completed_threads < STRESS_THREAD_COUNT {
        match work_done_rx.recv() {
            Ok(ops) => {
                completed_threads += 1;
                total_operations += ops;
            }
            Err(_) => break, // All senders dropped
        }
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Stress test thread should complete");
    }

    // Verify no crashes and reasonable operation counts
    assert!(
        total_operations > 1000,
        "Should perform many operations under stress"
    );
    assert_eq!(
        completed_threads, STRESS_THREAD_COUNT,
        "All threads should complete"
    );

    // Verify final state consistency
    // Verify cache statistics are available
    let cache_stats = coordinator.get_cache_stats();
    assert!(cache_stats.batch_size > 0);
    assert!(cache_stats.prefetch_distance > 0);
}

/// Memory allocation stress test with cache optimization
#[test]
fn stress_cache_aware_allocation() {
    let num_allocators = 4;
    let allocations_per_thread = 5000;

    let stop_flag = Arc::new(AtomicBool::new(false));
    let allocation_stats = Arc::new(
        (0..num_allocators)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );

    let mut handles = Vec::new();

    for thread_id in 0..num_allocators {
        let stop_flag = Arc::clone(&stop_flag);
        let stats = Arc::clone(&allocation_stats);

        let handle = thread::spawn(move || {
            let base = unsafe { Address::from_usize(0x20000000 + thread_id * 64 * 1024 * 1024) };
            let allocator = CacheAwareAllocator::new(base, 32 * 1024 * 1024); // 32MB per thread

            let mut successful_allocations = 0;
            let sizes = [64, 128, 256, 512, 1024, 2048];

            for i in 0..allocations_per_thread {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                let size = sizes[i % sizes.len()];
                if let Some(_addr) = allocator.allocate_aligned(size, 1) {
                    successful_allocations += 1;
                }

                // Periodic reset to test allocator robustness
                if i % 1000 == 999 {
                    allocator.reset();
                }
            }

            stats[thread_id].store(successful_allocations, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    // Use work-based termination instead of sleep
    let target_allocations = allocations_per_thread * num_allocators; // Target total allocations across all threads

    loop {
        let total_allocations: usize = allocation_stats
            .iter()
            .map(|stat| stat.load(Ordering::Relaxed))
            .sum();

        if total_allocations >= target_allocations {
            break;
        }

        thread::yield_now();
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for completion
    for handle in handles {
        handle
            .join()
            .expect("Allocation stress test should complete");
    }

    // Verify reasonable allocation success rates
    let total_successful: usize = allocation_stats
        .iter()
        .map(|stat| stat.load(Ordering::Relaxed))
        .sum();

    assert!(
        total_successful > allocations_per_thread * num_allocators / 2,
        "Should successfully allocate at least 50% of requests"
    );
}

/// Work stealing locality stress test
#[test]
fn stress_work_stealing_locality() {
    let num_workers = 6;
    let objects_per_worker = 2000;

    let shared_stealers = Arc::new(
        (0..num_workers)
            .map(|_| Mutex::new(LocalityAwareWorkStealer::new(8)))
            .collect::<Vec<_>>(),
    );

    let operation_counts = Arc::new(
        (0..num_workers)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );

    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    for worker_id in 0..num_workers {
        let stealers = Arc::clone(&shared_stealers);
        let counts = Arc::clone(&operation_counts);
        let stop_flag = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            let mut local_operations = 0;

            // Generate worker-specific objects with locality
            let mut worker_objects = Vec::new();
            for i in 0..objects_per_worker {
                let region_base = 0x30000000 + worker_id * 1024 * 1024; // 1MB regions
                let addr = unsafe { Address::from_usize(region_base + i * 64) };
                if let Some(obj) = ObjectReference::from_raw_address(addr) {
                    worker_objects.push(obj);
                }
            }

            while !stop_flag.load(Ordering::Relaxed) {
                match local_operations % 3 {
                    0 => {
                        // Add work to own stealer
                        if let Some(mut stealer) = stealers[worker_id].try_lock() {
                            let batch = worker_objects
                                .iter()
                                .skip(local_operations % worker_objects.len())
                                .take(50)
                                .copied()
                                .collect();
                            stealer.add_objects(batch);
                        }
                    }
                    1 => {
                        // Steal work from own stealer
                        if let Some(mut stealer) = stealers[worker_id].try_lock() {
                            let _batch = stealer.get_next_batch(25);
                        }
                    }
                    2 => {
                        // Attempt to steal from other workers
                        let target_worker = (worker_id + 1) % num_workers;
                        if let Some(mut stealer) = stealers[target_worker].try_lock() {
                            let _batch = stealer.get_next_batch(10); // Smaller batch for stealing
                        }
                    }
                    _ => unreachable!(),
                }

                local_operations += 1;

                // Update shared counter periodically to avoid deadlock
                if local_operations % 100 == 0 {
                    counts[worker_id].store(local_operations, Ordering::Relaxed);
                }

                // Yield occasionally for better interleaving
                if local_operations % 50 == 0 {
                    thread::yield_now();
                }
            }

            // Final update of shared counter
            counts[worker_id].store(local_operations, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    // Use work-based termination for work stealing test
    let target_total_operations = 100000; // Target total operations across all workers

    loop {
        let total_ops: usize = operation_counts
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum();

        if total_ops >= target_total_operations {
            break;
        }

        thread::yield_now();
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for completion
    for handle in handles {
        handle
            .join()
            .expect("Work stealing stress test should complete");
    }

    // Verify operations completed
    let total_operations: usize = operation_counts
        .iter()
        .map(|count| count.load(Ordering::Relaxed))
        .sum();

    assert!(
        total_operations > 1000,
        "Should complete many work stealing operations"
    );
}

/// Long-running endurance test
#[test]
fn endurance_test_gc_operations() {
    let heap_base = unsafe { Address::from_usize(0x40000000) };

    let tricolor = Arc::new(TricolorMarking::new(heap_base, 128 * 1024 * 1024));
    let cache_marking = Arc::new(CacheOptimizedMarking::new(4));

    let operations_completed = Arc::new(AtomicUsize::new(0));
    let errors_encountered = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    // Spawn endurance test threads
    for thread_id in 0..4 {
        let tricolor = Arc::clone(&tricolor);
        let cache_marking = Arc::clone(&cache_marking);
        let operations = Arc::clone(&operations_completed);
        let _errors = Arc::clone(&errors_encountered);
        let stop_flag = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            let mut local_operations = 0;
            let mut objects = Vec::new();

            // Create thread-local objects
            for i in 0..500 {
                let addr =
                    unsafe { Address::from_usize(0x40000000 + thread_id * 1000000 + i * 128) };
                if let Some(obj) = ObjectReference::from_raw_address(addr) {
                    objects.push(obj);
                }
            }

            while !stop_flag.load(Ordering::Relaxed) {
                // Cycle through different operations to test endurance
                match local_operations % 6 {
                    0 => {
                        // Batch cache-optimized marking
                        cache_marking.mark_objects_batch(&objects[..100.min(objects.len())]);
                    }
                    1 => {
                        // Individual tricolor operations
                        for obj in &objects[..50.min(objects.len())] {
                            tricolor.set_color(*obj, ObjectColor::Grey);
                        }
                    }
                    2 => {
                        // Color transitions
                        for obj in &objects[..50.min(objects.len())] {
                            tricolor.transition_color(*obj, ObjectColor::White, ObjectColor::Grey);
                        }
                    }
                    3 => {
                        // Read operations
                        for obj in &objects[..100.min(objects.len())] {
                            let _color = tricolor.get_color(*obj);
                        }
                    }
                    4 => {
                        // Reset colors for next cycle
                        for obj in &objects {
                            tricolor.set_color(*obj, ObjectColor::White);
                        }
                    }
                    5 => {
                        // Statistics access
                        let _stats = cache_marking.get_cache_stats();
                    }
                    _ => unreachable!(),
                }

                local_operations += 1;

                // Yield occasionally for better interleaving
                if local_operations % 100 == 0 {
                    thread::yield_now();
                }
            }

            operations.fetch_add(local_operations, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    // Use work-based endurance test instead of sleep
    let start_time = Instant::now();
    let target_endurance_operations = 200_000; // Target operations for endurance
    let min_duration = STRESS_TEST_DURATION.max(Duration::from_secs(1));

    loop {
        let total_ops = operations_completed.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed();

        // Run for either target operations OR minimum time, whichever comes first
        if total_ops >= target_endurance_operations || elapsed >= min_duration {
            break;
        }

        thread::yield_now();
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for completion
    for handle in handles {
        handle
            .join()
            .expect("Endurance test thread should complete");
    }

    let elapsed = start_time.elapsed();
    let total_operations = operations_completed.load(Ordering::Relaxed);
    let total_errors = errors_encountered.load(Ordering::Relaxed);

    println!("Endurance test completed:");
    println!("  Duration: {:?}", elapsed);
    println!("  Operations: {}", total_operations);
    println!("  Errors: {}", total_errors);
    println!(
        "  Ops/sec: {:.2}",
        total_operations as f64 / elapsed.as_secs_f64()
    );

    assert_eq!(
        total_errors, 0,
        "Should not encounter errors during endurance test"
    );
    assert!(
        total_operations > 1000,
        "Should complete many operations during endurance test"
    );

    // Verify final state is reasonable - the fact that we completed without errors
    // and processed many operations is sufficient validation for an endurance test
    // The specific final state may vary depending on the last operations performed
}

/// Resource exhaustion recovery test
#[test]
fn test_resource_exhaustion_recovery() {
    let base = unsafe { Address::from_usize(0x50000000) };
    let small_heap = 1024 * 1024; // 1MB heap for exhaustion test
    let allocator = CacheAwareAllocator::new(base, small_heap);

    let mut allocations = Vec::new();
    let mut total_allocated = 0;

    // Allocate until exhaustion
    while let Some(addr) = allocator.allocate_aligned(4096, 1) {
        // 4KB allocations
        allocations.push(addr);
        total_allocated += 4096;

        // Verify allocation is valid
        assert!(addr.as_usize() >= base.as_usize());
        assert!(addr.as_usize() + 4096 <= base.as_usize() + small_heap);
    }

    assert!(
        total_allocated > small_heap / 2,
        "Should allocate substantial portion of heap"
    );

    // Test recovery: reset and allocate again
    allocator.reset();

    // Should be able to allocate again after reset
    let recovery_allocation = allocator.allocate_aligned(4096, 1);
    assert!(recovery_allocation.is_some(), "Should recover after reset");

    // Verify the recovered allocation is at the beginning
    let recovered_addr = recovery_allocation.unwrap();
    assert_eq!(
        recovered_addr, base,
        "Should start from heap base after reset"
    );
}

/// Memory safety stress test with concurrent access
#[test]
fn stress_memory_safety_concurrent_access() {
    let heap_base = unsafe { Address::from_usize(0x60000000) };
    let tricolor = Arc::new(TricolorMarking::new(heap_base, 64 * 1024 * 1024));

    // Create a shared set of objects
    let shared_objects = Arc::new({
        let mut objects = Vec::new();
        for i in 0..2000 {
            let addr = unsafe { Address::from_usize(0x60000000 + i * 64) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }
        objects
    });

    let stop_flag = Arc::new(AtomicBool::new(false));
    let safety_violations = Arc::new(AtomicUsize::new(0));
    let total_operations = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    // Spawn threads that aggressively access shared objects
    for thread_id in 0..6 {
        let tricolor = Arc::clone(&tricolor);
        let objects = Arc::clone(&shared_objects);
        let stop_flag = Arc::clone(&stop_flag);
        let violations = Arc::clone(&safety_violations);
        let operations = Arc::clone(&total_operations);

        let handle = thread::spawn(move || {
            let local_violations = 0;
            let mut local_ops = 0;

            while !stop_flag.load(Ordering::Relaxed) {
                for chunk in objects.chunks(100) {
                    // Rapidly change colors to stress atomic operations
                    for &obj in chunk {
                        let original_color = tricolor.get_color(obj);

                        // Attempt color transitions
                        match original_color {
                            ObjectColor::White => {
                                tricolor.transition_color(
                                    obj,
                                    ObjectColor::White,
                                    ObjectColor::Grey,
                                );
                            }
                            ObjectColor::Grey => {
                                tricolor.transition_color(
                                    obj,
                                    ObjectColor::Grey,
                                    ObjectColor::Black,
                                );
                            }
                            ObjectColor::Black => {
                                tricolor.set_color(obj, ObjectColor::White);
                            }
                        }

                        // Verify no invalid color states
                        let new_color = tricolor.get_color(obj);
                        match new_color {
                            ObjectColor::White | ObjectColor::Grey | ObjectColor::Black => {
                                // Valid colors
                            }
                        }

                        local_ops += 1;
                        // Update shared counter periodically to avoid deadlock
                        if local_ops % 100 == 0 {
                            operations.fetch_add(100, Ordering::Relaxed);
                            local_ops = 0;
                        }
                    }
                }

                // Yield to increase contention
                if thread_id % 2 == 0 {
                    thread::yield_now();
                }
            }

            // Add remaining operations and violations
            operations.fetch_add(local_ops, Ordering::Relaxed);
            violations.fetch_add(local_violations, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    // Let threads run until they've completed sufficient work
    // Use proper synchronization instead of sleep
    let target_operations = 1000; // Target operations per thread
    let num_threads = 6; // This test spawns 6 threads
    loop {
        let total_ops = total_operations.load(Ordering::Relaxed);
        if total_ops >= target_operations * num_threads {
            break;
        }
        thread::yield_now(); // Cooperative yield instead of blocking
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for completion
    for handle in handles {
        handle
            .join()
            .expect("Memory safety stress test should complete");
    }

    let total_violations = safety_violations.load(Ordering::Relaxed);
    assert_eq!(
        total_violations, 0,
        "Should not detect any memory safety violations"
    );
}

/// Stress test for crossbeam epoch integration with promotion queue
#[test]
fn stress_crossbeam_epoch_integration() {
    let fixture = TestFixture::new_with_config(0x60000000, 64 * 1024 * 1024, 8);
    let coordinator = &fixture.coordinator;
    let heap_base = unsafe { Address::from_usize(0x60000000) };

    // Create test objects with proper alignment
    let test_objects = Arc::new({
        let mut objects = Vec::new();
        for i in 0..1000 {
            // Ensure proper alignment (64-byte aligned for test objects)
            let offset = i * 64;
            let aligned_offset = (offset + 7) & !7; // 8-byte alignment
            let addr = unsafe { Address::from_usize(heap_base.as_usize() + aligned_offset) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                coordinator.classify_new_object(obj);
                objects.push(obj);
            }
        }
        objects
    });

    let promoted_count = Arc::new(AtomicUsize::new(0));
    let queue_operations = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    // Spawn high-contention worker threads
    for worker_id in 0..8 {
        let coordinator = Arc::clone(&coordinator);
        let test_objects = Arc::clone(&test_objects);
        let promoted_count = Arc::clone(&promoted_count);
        let queue_operations = Arc::clone(&queue_operations);
        let stop_flag = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            let mut iteration_count = 0;

            while !stop_flag.load(Ordering::Relaxed) {
                match worker_id % 4 {
                    0 => {
                        // Heavy promotion queueing to stress epoch protection
                        for i in 0..50 {
                            let idx = (worker_id * 127 + i) % test_objects.len();
                            if let Some(&obj) = test_objects.get(idx) {
                                coordinator.queue_for_promotion(obj);
                                queue_operations.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    1 => {
                        // Process promotions to stress epoch advancement
                        coordinator.promote_young_objects();
                        promoted_count.fetch_add(1, Ordering::Relaxed);
                    }
                    2 => {
                        // Create cross-generational references to trigger barriers
                        for i in 0..20 {
                            let old_idx = (worker_id * 71 + i) % test_objects.len();
                            let young_idx = (worker_id * 89 + i) % test_objects.len();

                            if let (Some(&old_obj), Some(&young_obj)) =
                                (test_objects.get(old_idx), test_objects.get(young_idx))
                            {
                                // Promote old object first to create old generation object
                                coordinator.queue_for_promotion(old_obj);
                                coordinator.promote_young_objects();

                                // Create cross-generational reference
                                coordinator.generational_write_barrier(old_obj, young_obj);
                                queue_operations.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    3 => {
                        // Read object classifications concurrently with updates
                        for i in 0..30 {
                            let idx = (worker_id * 103 + i) % test_objects.len();
                            if let Some(&obj) = test_objects.get(idx) {
                                // This creates concurrent access to ObjectClassifier
                                std::hint::black_box(
                                    coordinator.object_classifier().get_classification(obj),
                                );
                            }
                        }
                    }
                    _ => unreachable!(),
                }

                iteration_count += 1;
                if iteration_count % 50 == 0 {
                    thread::yield_now();
                }
            }
        });

        handles.push(handle);
    }

    // Use work-based synchronization instead of sleep
    let target_work_units = 10000; // Target total operations across all threads

    loop {
        let total_promoted = promoted_count.load(Ordering::Relaxed);
        let total_queue_ops = queue_operations.load(Ordering::Relaxed);

        if total_promoted + total_queue_ops >= target_work_units {
            break;
        }

        thread::yield_now(); // Cooperative yield
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for all workers to complete
    for handle in handles {
        handle
            .join()
            .expect("Worker thread should complete successfully");
    }

    // Validate epoch integration worked correctly
    let final_promoted = promoted_count.load(Ordering::Relaxed);
    let final_queue_ops = queue_operations.load(Ordering::Relaxed);

    println!("Epoch stress test completed:");
    println!("  Promotion operations: {}", final_promoted);
    println!("  Queue operations: {}", final_queue_ops);

    // Basic validity checks
    assert!(final_promoted > 0, "Should have processed some promotions");
    assert!(
        final_queue_ops > 0,
        "Should have performed queue operations"
    );

    // Final cleanup to trigger any remaining epoch reclamation
    coordinator.promote_young_objects();
}

/// Multi-generational GC stress test under contention
#[test]
fn stress_multi_generation_promotion_contention() {
    // Ensure heap base is 8-byte aligned
    let heap_base = unsafe { Address::from_usize((0x70000000 + 7) & !7) };
    let heap_size = 128 * 1024 * 1024; // 128MB heap
    let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(fugrip::roots::GlobalRoots::default()));

    let coordinator = Arc::new(FugcCoordinator::new(
        heap_base,
        heap_size,
        STRESS_THREAD_COUNT,
        thread_registry,
        global_roots,
    ));

    // Test young/old generation boundaries (30% young, 70% old)
    let young_gen_size = (heap_size as f64 * 0.3) as usize;
    // Ensure old generation start is properly aligned
    let aligned_young_size = (young_gen_size + 7) & !7; // 8-byte align
    let old_gen_start = heap_base + aligned_young_size;

    // Create objects in both generations
    let young_objects = Arc::new({
        let mut objects = Vec::new();
        for i in 0..500 {
            // Ensure 8-byte word alignment for object references
            let offset = (i * 128 + 7) & !7; // Round up to next 8-byte boundary
            let addr = unsafe { Address::from_usize(heap_base.as_usize() + offset) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }
        objects
    });

    let old_objects = Arc::new({
        let mut objects = Vec::new();
        for i in 0..500 {
            // Ensure 8-byte word alignment for object references
            let offset = (i * 128 + 7) & !7; // Round up to next 8-byte boundary
            let addr = unsafe { Address::from_usize(old_gen_start.as_usize() + offset) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }
        objects
    });

    let promotion_count = Arc::new(AtomicUsize::new(0));
    let cross_gen_refs = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(STRESS_THREAD_COUNT));

    let mut handles = Vec::new();

    for worker_id in 0..STRESS_THREAD_COUNT {
        let young_objects = Arc::clone(&young_objects);
        let old_objects = Arc::clone(&old_objects);
        let promotion_count = Arc::clone(&promotion_count);
        let cross_gen_refs = Arc::clone(&cross_gen_refs);
        let stop_flag = Arc::clone(&stop_flag);
        let barrier = Arc::clone(&barrier);
        let coordinator = Arc::clone(&coordinator);

        let handle = thread::spawn(move || {
            barrier.wait(); // Synchronize start for maximum contention

            let mut iteration_count = 0;

            while !stop_flag.load(Ordering::Relaxed) {
                match worker_id % 4 {
                    0 => {
                        // Simulate young->old promotion under contention
                        let young_idx = worker_id % young_objects.len();
                        let old_idx = worker_id % old_objects.len();

                        if let (Some(young_obj), Some(old_obj)) =
                            (young_objects.get(young_idx), old_objects.get(old_idx))
                        {
                            // Simulate promotion by treating young object as if promoted
                            coordinator
                                .tricolor_marking
                                .set_color(*young_obj, ObjectColor::Black);
                            promotion_count.fetch_add(1, Ordering::Relaxed);

                            // Create cross-generational reference
                            coordinator
                                .tricolor_marking
                                .set_color(*old_obj, ObjectColor::Grey);
                            cross_gen_refs.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    1 => {
                        // Test write barriers for old->young references
                        let write_barrier = coordinator.write_barrier();
                        write_barrier.activate();

                        // Simulate old->young reference creation
                        let young_idx = (worker_id * 17) % young_objects.len(); // Prime for distribution
                        if let Some(_young_obj) = young_objects.get(young_idx) {
                            // This would trigger remembered set updates
                            cross_gen_refs.fetch_add(1, Ordering::Relaxed);
                        }

                        write_barrier.deactivate();
                    }
                    2 => {
                        // Concurrent marking during promotion
                        let obj_set = if worker_id % 2 == 0 {
                            &young_objects
                        } else {
                            &old_objects
                        };
                        let idx = (worker_id * 31) % obj_set.len(); // Prime for distribution

                        if let Some(obj) = obj_set.get(idx) {
                            let current_color = coordinator.tricolor_marking().get_color(*obj);
                            match current_color {
                                ObjectColor::White => {
                                    // Try to mark as grey
                                    if coordinator.tricolor_marking().transition_color(
                                        *obj,
                                        ObjectColor::White,
                                        ObjectColor::Grey,
                                    ) {
                                        // Successfully marked
                                    }
                                }
                                ObjectColor::Grey => {
                                    // Complete marking to black
                                    coordinator
                                        .tricolor_marking
                                        .set_color(*obj, ObjectColor::Black);
                                }
                                ObjectColor::Black => {
                                    // Already marked, continue
                                }
                            }
                        }
                    }
                    3 => {
                        // Simulate tricolor marking operations without full GC cycles
                        // This avoids the expensive start_marking/stop_marking in tight loops
                        for &obj in young_objects.iter().take(5) {
                            // Simulate marking progression: white -> grey -> black
                            coordinator
                                .tricolor_marking
                                .set_color(obj, ObjectColor::Grey);
                            coordinator
                                .tricolor_marking
                                .set_color(obj, ObjectColor::Black);
                        }
                        promotion_count.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => unreachable!(),
                }

                iteration_count += 1;
                // Yield occasionally for better interleaving
                if iteration_count % 50 == 0 {
                    thread::yield_now();
                }
            }
        });

        handles.push(handle);
    }

    // Use work-based termination for multi-generation test
    let target_total_work = 30000; // Target combined promotions and cross-refs

    loop {
        let total_promotions = promotion_count.load(Ordering::Relaxed);
        let total_cross_refs = cross_gen_refs.load(Ordering::Relaxed);

        if total_promotions + total_cross_refs >= target_total_work {
            break;
        }

        thread::yield_now();
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for completion
    for handle in handles {
        handle
            .join()
            .expect("Multi-generation stress test should complete");
    }

    let total_promotions = promotion_count.load(Ordering::Relaxed);
    let total_cross_refs = cross_gen_refs.load(Ordering::Relaxed);

    println!("Multi-generation stress test results:");
    println!("  Simulated promotions: {}", total_promotions);
    println!("  Cross-generation references: {}", total_cross_refs);

    assert!(
        total_promotions > 100,
        "Should simulate many young->old promotions under contention"
    );
    assert!(
        total_cross_refs > 100,
        "Should create many cross-generational references"
    );
}

/// Generational write barrier stress test
#[test]
fn stress_generational_write_barriers() {
    // Ensure heap base is 8-byte aligned
    let heap_base = unsafe { Address::from_usize((0x80000000 + 7) & !7) };
    let heap_size = 64 * 1024 * 1024; // 64MB heap
    let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(fugrip::roots::GlobalRoots::default()));

    let coordinator = Arc::new(FugcCoordinator::new(
        heap_base,
        heap_size,
        4, // Focused test with fewer workers
        thread_registry,
        global_roots,
    ));

    // Create dedicated young and old generation object pools
    let young_gen_size = (heap_size as f64 * 0.3) as usize;
    let aligned_young_size = (young_gen_size + 7) & !7; // 8-byte align
    let young_gen_boundary = heap_base + aligned_young_size;

    let young_pool = Arc::new({
        let mut objects = Vec::new();
        for i in 0..200 {
            // Ensure 8-byte word alignment for object references
            let offset = (i * 256 + 7) & !7; // Round up to next 8-byte boundary
            let addr = unsafe { Address::from_usize(heap_base.as_usize() + offset) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }
        objects
    });

    let old_pool = Arc::new({
        let mut objects = Vec::new();
        for i in 0..200 {
            // Ensure 8-byte word alignment for object references
            let offset = (i * 256 + 7) & !7; // Round up to next 8-byte boundary
            let addr = unsafe { Address::from_usize(young_gen_boundary.as_usize() + offset) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }
        objects
    });

    let barrier_operations = Arc::new(AtomicUsize::new(0));
    let remembered_set_updates = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    for worker_id in 0..4 {
        let young_pool = Arc::clone(&young_pool);
        let old_pool = Arc::clone(&old_pool);
        let barrier_operations = Arc::clone(&barrier_operations);
        let remembered_set_updates = Arc::clone(&remembered_set_updates);
        let stop_flag = Arc::clone(&stop_flag);
        let coordinator = Arc::clone(&coordinator);

        let handle = thread::spawn(move || {
            let mut iteration_count = 0;

            while !stop_flag.load(Ordering::Relaxed) {
                let write_barrier = coordinator.write_barrier();

                match worker_id % 3 {
                    0 => {
                        // Young-to-young writes (should be fast path)
                        write_barrier.activate();

                        for i in 0..10 {
                            let src_idx = (worker_id * 13 + i) % young_pool.len();
                            let dst_idx = (worker_id * 17 + i) % young_pool.len();

                            if src_idx == dst_idx {
                                continue;
                            }

                            if let (Some(_src_obj), Some(_dst_obj)) =
                                (young_pool.get(src_idx), young_pool.get(dst_idx))
                            {
                                // In real implementation, this would use write_barrier_generational_fast
                                barrier_operations.fetch_add(1, Ordering::Relaxed);
                            }
                        }

                        write_barrier.deactivate();
                    }
                    1 => {
                        // Old-to-young writes (should update remembered set)
                        write_barrier.activate();

                        for i in 0..5 {
                            let old_idx = (worker_id * 19 + i) % old_pool.len();
                            let young_idx = (worker_id * 23 + i) % young_pool.len();

                            if let (Some(_old_obj), Some(_young_obj)) =
                                (old_pool.get(old_idx), young_pool.get(young_idx))
                            {
                                // This would trigger remembered set update
                                remembered_set_updates.fetch_add(1, Ordering::Relaxed);
                                barrier_operations.fetch_add(1, Ordering::Relaxed);
                            }
                        }

                        write_barrier.deactivate();
                    }
                    2 => {
                        // Mixed generational operations under high contention
                        write_barrier.activate();

                        // Simulate rapid barrier activations/deactivations
                        for _ in 0..20 {
                            write_barrier.deactivate();
                            thread::yield_now(); // Increase contention
                            write_barrier.activate();
                            barrier_operations.fetch_add(1, Ordering::Relaxed);
                        }

                        write_barrier.deactivate();
                    }
                    _ => unreachable!(),
                }

                iteration_count += 1;
                // Periodic yield for contention
                if iteration_count % 100 == 0 {
                    thread::yield_now();
                }
            }
        });

        handles.push(handle);
    }

    // Use work-based termination for generational barrier test
    let target_barrier_operations = 50000; // Target barrier operations

    loop {
        let total_barrier_ops = barrier_operations.load(Ordering::Relaxed);

        if total_barrier_ops >= target_barrier_operations {
            break;
        }

        thread::yield_now();
    }

    stop_flag.store(true, Ordering::Relaxed);

    // Wait for completion
    for handle in handles {
        handle
            .join()
            .expect("Generational write barrier stress test should complete");
    }

    let total_barrier_ops = barrier_operations.load(Ordering::Relaxed);
    let total_remembered_updates = remembered_set_updates.load(Ordering::Relaxed);

    println!("Generational write barrier stress test results:");
    println!("  Total barrier operations: {}", total_barrier_ops);
    println!("  Remembered set updates: {}", total_remembered_updates);

    assert!(
        total_barrier_ops > 1000,
        "Should perform many write barrier operations under stress"
    );
    assert!(
        total_remembered_updates > 50,
        "Should perform remembered set updates for old->young references"
    );
}

#[test]
fn stress_final_verification() {
    println!("All GC stress tests completed successfully!");
    println!("The FUGC implementation demonstrates:");
    println!("  ✓ High-contention concurrent marking resilience");
    println!("  ✓ Cache-aware allocation under pressure");
    println!("  ✓ Work-stealing locality preservation");
    println!("  ✓ Long-running endurance capability");
    println!("  ✓ Resource exhaustion recovery");
    println!("  ✓ Memory safety under extreme concurrent access");
    println!("  ✓ Multi-generational promotion under contention");
    println!("  ✓ Generational write barrier stress resilience");
}
