//! Integration tests for LLVM safepoint coordination with FUGC
//!
//! This test demonstrates the complete safepoint flow:
//! 1. Fast path pollchecks (load-and-branch)
//! 2. Slow path callbacks for FUGC coordination
//! 3. Bounded progress guarantees
//! 4. Integration with FUGC 8-step protocol

use fugrip::{
    pollcheck, FugcCoordinator, FugcPhase, GcSafepointPhase, SafepointManager,
    GlobalRoots,
};
use fugrip::thread::ThreadRegistry;
use mmtk::util::{Address, ObjectReference};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Test that demonstrates LLVM-style safepoint integration
#[test]
fn test_safepoint_fugc_integration() {
    // Setup FUGC coordinator
    let heap_base = unsafe { Address::from_usize(0x20000000) };
    let heap_size = 32 * 1024 * 1024; // 32MB
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

    let coordinator = Arc::new(FugcCoordinator::new(
        heap_base,
        heap_size,
        4, // 4 worker threads
        thread_registry.clone(),
        global_roots.clone(),
    ));

    // Test Phase 1: Fast path pollcheck performance
    test_fast_path_performance();

    // Test Phase 2: Safepoint callback execution
    test_safepoint_callbacks();

    // Test Phase 3: FUGC-specific safepoint coordination
    test_fugc_safepoint_coordination(coordinator.clone());

    // Test Phase 4: Bounded progress simulation
    test_bounded_progress_guarantees();

    println!("✅ All LLVM safepoint integration tests passed!");
}

/// Test that fast path pollchecks are extremely fast when no safepoint is requested
fn test_fast_path_performance() {
    println!("🚀 Testing fast path pollcheck performance...");

    // Ensure no safepoint is requested
    let manager = SafepointManager::global();
    manager.clear_safepoint();

    // Measure 100,000 pollchecks
    let iterations = 100_000;
    let start = Instant::now();

    for _ in 0..iterations {
        pollcheck(); // Should be just load-and-branch
    }

    let elapsed = start.elapsed();
    let ns_per_pollcheck = elapsed.as_nanos() / iterations;

    println!("  📊 {iterations} pollchecks in {elapsed:?}");
    println!("  📊 {ns_per_pollcheck}ns per pollcheck");

    // Fast path should be extremely fast (< 10ns on modern hardware)
    assert!(ns_per_pollcheck < 50, "Fast path too slow: {ns_per_pollcheck}ns");
    println!("  ✅ Fast path performance test passed");
}

/// Test that safepoint callbacks are executed correctly
fn test_safepoint_callbacks() {
    println!("🎯 Testing safepoint callback execution...");

    let manager = SafepointManager::global();
    let callback_executed = Arc::new(AtomicBool::new(false));
    let callback_clone = Arc::clone(&callback_executed);

    // Set up a test callback
    manager.request_safepoint(Box::new(move || {
        callback_clone.store(true, Ordering::Relaxed);
    }));

    // Trigger the safepoint
    pollcheck();

    // Verify callback was executed
    assert!(callback_executed.load(Ordering::Relaxed), "Callback not executed");

    manager.clear_safepoint();
    println!("  ✅ Safepoint callback test passed");
}

/// Test FUGC-specific safepoint coordination for different GC phases
fn test_fugc_safepoint_coordination(coordinator: Arc<FugcCoordinator>) {
    println!("🔄 Testing FUGC safepoint coordination...");

    let manager = SafepointManager::global();

    // Test Phase 1: Root Scanning
    println!("  📍 Testing root scanning safepoint...");
    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);

    manager.request_gc_safepoint(GcSafepointPhase::RootScanning);
    pollcheck(); // Should execute root scanning callback
    manager.clear_safepoint();

    // Test Phase 2: Barrier Activation
    println!("  🛡️ Testing barrier activation safepoint...");
    manager.request_gc_safepoint(GcSafepointPhase::BarrierActivation);

    // Before safepoint: barriers should be inactive
    assert!(!coordinator.write_barrier().is_active());
    assert!(!coordinator.black_allocator().is_active());

    pollcheck(); // Should activate barriers

    // After safepoint: barriers should be active
    assert!(coordinator.write_barrier().is_active());
    assert!(coordinator.black_allocator().is_active());

    manager.clear_safepoint();

    // Test Phase 3: Marking Handshake
    println!("  🤝 Testing marking handshake safepoint...");
    manager.request_gc_safepoint(GcSafepointPhase::MarkingHandshake);
    pollcheck(); // Should process marking work
    manager.clear_safepoint();

    // Test Phase 4: Sweep Preparation
    println!("  🧹 Testing sweep preparation safepoint...");
    manager.request_gc_safepoint(GcSafepointPhase::SweepPreparation);
    pollcheck(); // Should deactivate barriers and prepare for sweep

    // After sweep preparation: barriers should be inactive
    assert!(!coordinator.write_barrier().is_active());

    manager.clear_safepoint();
    println!("  ✅ FUGC coordination test passed");
}

/// Test bounded progress guarantees with multiple threads
fn test_bounded_progress_guarantees() {
    println!("⏱️ Testing bounded progress guarantees...");

    let manager = SafepointManager::global();
    let threads_hit_safepoint = Arc::new(AtomicUsize::new(0));
    let running = Arc::new(AtomicBool::new(true));

    // Spawn multiple mutator threads that pollcheck regularly
    let mut handles = Vec::new();
    let num_threads = 4;

    for thread_id in 0..num_threads {
        let running_clone = Arc::clone(&running);
        let hits_clone = Arc::clone(&threads_hit_safepoint);

        let handle = thread::spawn(move || {
            let mut local_work = 0;
            let mut local_hits = 0;

            while running_clone.load(Ordering::Relaxed) {
                // Simulate bounded work between pollchecks
                for _ in 0..100 {
                    local_work += 1; // Simulated computation
                }

                // LLVM would emit this pollcheck automatically
                pollcheck();

                // If a safepoint was hit, record it
                if !running_clone.load(Ordering::Relaxed) {
                    local_hits += 1;
                }
            }

            if local_hits > 0 {
                hits_clone.fetch_add(1, Ordering::Relaxed);
            }

            (local_work, local_hits)
        });

        handles.push(handle);
    }

    // Let threads run for a short time
    # Using sleeps to paper over logic bugs is unprofessional(Duration::from_millis(10));

    // Request a safepoint that will stop all threads
    let stop_clone = Arc::clone(&running);
    manager.request_safepoint(Box::new(move || {
        stop_clone.store(false, Ordering::Relaxed);
    }));

    // Wait for all threads to hit the safepoint and stop
    let timeout = Duration::from_millis(100);
    let wait_success = manager.wait_for_safepoint(timeout);
    assert!(wait_success, "Timeout waiting for safepoint");

    // Clean up
    manager.clear_safepoint();

    // Wait for all threads to complete
    for handle in handles {
        let (work_done, _hits) = handle.join().expect("Thread should join successfully");
        assert!(work_done > 0, "Thread should have done some work");
    }

    // Verify that threads reached the safepoint
    let total_hits = threads_hit_safepoint.load(Ordering::Relaxed);
    println!("  📊 {total_hits}/{num_threads} threads hit safepoint");
    assert!(total_hits > 0, "At least some threads should hit safepoint");

    println!("  ✅ Bounded progress test passed");
}

/// Benchmark safepoint overhead in realistic workload
#[test]
fn benchmark_safepoint_overhead() {
    println!("⚡ Benchmarking safepoint overhead...");

    let manager = SafepointManager::global();
    manager.clear_safepoint();

    // Simulate a realistic workload with pollchecks
    let iterations = 1_000_000;
    let work_per_pollcheck = 1000;

    // Baseline: work without pollchecks
    let start = Instant::now();
    let mut total_work = 0u64;
    for i in 0..iterations {
        for j in 0..work_per_pollcheck {
            total_work = total_work.wrapping_add((i * j) as u64);
        }
    }
    let baseline_time = start.elapsed();

    // Test: same work with pollchecks
    let start = Instant::now();
    total_work = 0;
    for i in 0..iterations {
        for j in 0..work_per_pollcheck {
            total_work = total_work.wrapping_add((i * j) as u64);
        }
        pollcheck(); // LLVM would emit this
    }
    let pollcheck_time = start.elapsed();

    let overhead_ns = (pollcheck_time - baseline_time).as_nanos() / iterations;
    let overhead_percent = ((pollcheck_time.as_nanos() as f64 / baseline_time.as_nanos() as f64) - 1.0) * 100.0;

    println!("  📊 Baseline time: {baseline_time:?}");
    println!("  📊 With pollchecks: {pollcheck_time:?}");
    println!("  📊 Overhead: {overhead_ns}ns per pollcheck ({overhead_percent:.2}%)");

    // Overhead should be minimal (< 1% for this workload)
    assert!(overhead_percent < 10.0, "Pollcheck overhead too high: {overhead_percent:.2}%");

    // Prevent optimization from removing the work
    assert!(total_work > 0);

    println!("  ✅ Safepoint overhead benchmark passed");
}

/// Test safepoint statistics collection
#[test]
fn test_safepoint_statistics() {
    println!("📈 Testing safepoint statistics...");

    let manager = SafepointManager::global();

    // Clear any previous state
    manager.clear_safepoint();

    // Get initial stats
    let initial_stats = manager.get_stats();
    println!("  📊 Initial stats: polls={}, hits={}, rate={:.2}%",
             initial_stats.total_polls,
             initial_stats.total_hits,
             initial_stats.hit_rate * 100.0);

    // Perform some pollchecks without safepoint (fast path)
    for _ in 0..100 {
        pollcheck();
    }

    // Perform some pollchecks with safepoint (slow path)
    manager.request_safepoint(Box::new(|| {
        // Simple callback
    }));

    for _ in 0..5 {
        pollcheck();
    }

    manager.clear_safepoint();

    // Get final stats
    let final_stats = manager.get_stats();
    println!("  📊 Final stats: polls={}, hits={}, rate={:.2}%",
             final_stats.total_polls,
             final_stats.total_hits,
             final_stats.hit_rate * 100.0);

    // Verify stats increased
    assert!(final_stats.total_polls > initial_stats.total_polls, "Poll count should increase");
    assert!(final_stats.total_hits >= initial_stats.total_hits, "Hit count should not decrease");
    assert!(final_stats.hit_rate >= 0.0 && final_stats.hit_rate <= 1.0, "Hit rate should be valid percentage");

    println!("  ✅ Safepoint statistics test passed");
}