use flume::{self};
use fugrip::safepoint::{pollcheck, safepoint_enter, safepoint_exit};
use fugrip::test_utils::TestFixture;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

#[test]
fn safepoint_integration_with_custom_coordinator() {
    // Test the coordinator's barrier activation methods directly
    let fixture = TestFixture::new();
    let coordinator = &fixture.coordinator;

    // Test direct activation
    assert!(!coordinator.write_barrier().is_active());
    assert!(!coordinator.black_allocator().is_active());

    coordinator.activate_barriers_at_safepoint();
    assert!(coordinator.write_barrier().is_active());
    assert!(coordinator.black_allocator().is_active());

    // Reset state
    coordinator.write_barrier().deactivate();
    coordinator.black_allocator().deactivate();
    assert!(!coordinator.write_barrier().is_active());
    assert!(!coordinator.black_allocator().is_active());

    // Test marking handshake method
    coordinator.marking_handshake_at_safepoint();
    // This should not change barrier states but should execute without error

    // Test scan thread roots method
    coordinator.scan_thread_roots_at_safepoint();
    // This should execute without error

    println!("✅ All coordinator safepoint methods work correctly");
}

#[test]
fn safepoint_thread_coordination() {
    let fixture = TestFixture::new();
    let manager = fixture.safepoint_manager();

    // Clear state
    manager.clear_safepoint();

    // Synchronization channels
    let (ready_tx, ready_rx) = flume::bounded(3);
    let (start_tx, start_rx) = flume::bounded(0);
    let (stop_tx, stop_rx) = flume::bounded(0);
    let (done_tx, done_rx) = flume::bounded(3);

    // Register some test threads using scoped rayon tasks
    rayon::scope(|s| {
        for thread_id in 0..3 {
            let ready_tx = ready_tx.clone();
            let start_rx = start_rx.clone();
            let stop_rx = stop_rx.clone();
            let done_tx = done_tx.clone();
            let manager_for_thread = Arc::clone(manager);

            s.spawn(move |_| {
                // Register with the fixture's safepoint manager and cache it
                manager_for_thread.register_and_cache_thread();

                // Signal ready
                ready_tx.send(thread_id).unwrap();

                // Wait for start signal
                start_rx.recv().unwrap();

                // Poll safepoints until stop signal
                loop {
                    if stop_rx.try_recv().is_ok() {
                        break;
                    } else {
                        pollcheck();
                    }
                }

                // Signal completion
                done_tx.send(thread_id).unwrap();
            });
        }
        // The rest of the test runs while the tasks are active

        // Wait for all threads to be ready
        for _ in 0..3 {
            ready_rx.recv().unwrap();
        }

        // Main thread should also use the fixture's manager
        manager.register_and_cache_thread();

        // Request safepoint with execution counter
        let executed = Arc::new(AtomicUsize::new(0));
        let executed_clone = Arc::clone(&executed);
        manager.request_safepoint(Box::new(move || {
            executed_clone.fetch_add(1, Ordering::SeqCst);
        }));

        // Start all threads
        for _ in 0..3 {
            start_tx.send(()).unwrap();
        }

        // Main thread also participates
        for _ in 0..100 {
            pollcheck();
        }

        // Wait for all threads to hit safepoint using proper timeout
        assert!(manager.wait_for_safepoint(Duration::from_secs(5)));

        manager.clear_safepoint();

        // Stop all threads
        for _ in 0..3 {
            stop_tx.send(()).unwrap();
        }

        // Wait for all threads to complete
        for _ in 0..3 {
            done_rx.recv().unwrap();
        }
        // Verify execution count
        assert!(executed.load(Ordering::SeqCst) > 0);
    });
}

#[test]
fn safepoint_soft_handshake() {
    let fixture = TestFixture::new();
    let manager = fixture.safepoint_manager();

    // Synchronization for deterministic test execution
    let (ready_tx, ready_rx) = flume::bounded(2);
    let (stop_tx, stop_rx) = flume::bounded(0);
    let (done_tx, done_rx) = flume::bounded(2);

    // Handshake execution counter
    let handshake_executed = Arc::new(AtomicBool::new(false));
    let handshake_executed_clone = Arc::clone(&handshake_executed);

    // Spawn worker threads that use the fixture's safepoint manager with crossbeam scoped threads
    crossbeam::scope(|s| {
        for thread_id in 0..2 {
            let ready_tx = ready_tx.clone();
            let stop_rx = stop_rx.clone();
            let done_tx = done_tx.clone();
            let manager_for_thread = Arc::clone(manager);

            s.spawn(move |_| {
                // Register with the fixture's safepoint manager and cache it
                manager_for_thread.register_and_cache_thread();

                // Signal ready
                ready_tx.send(thread_id).unwrap();

                // Poll safepoints until stopped
                loop {
                    if stop_rx.try_recv().is_ok() {
                        break;
                    } else {
                        pollcheck();
                    }
                }

                // Signal completion
                done_tx.send(thread_id).unwrap();
            });
        }

        // Wait for all threads to be ready
        for _ in 0..2 {
            ready_rx.recv().unwrap();
        }

        // Main thread should also use the fixture's manager
        manager.register_and_cache_thread();

        // Request soft handshake with simpler notification
        manager.request_soft_handshake(Box::new(move || {
            handshake_executed_clone.store(true, Ordering::SeqCst);
        }));

        // Main thread participates in handshake until completion
        let mut poll_count = 0;
        while !handshake_executed.load(Ordering::SeqCst) && poll_count < 100 {
            pollcheck();
            poll_count += 1;

            if poll_count % 20 == 0 {
                println!(
                    "Poll count: {}, handshake executed: {}",
                    poll_count,
                    handshake_executed.load(Ordering::SeqCst)
                );
            }

            // Small yield to allow other threads to participate
            std::hint::spin_loop();
        }

        // Verify handshake was executed
        assert!(
            handshake_executed.load(Ordering::SeqCst),
            "Handshake should have been executed after {} polls",
            poll_count
        );

        // Stop all threads
        for _ in 0..2 {
            stop_tx.send(()).unwrap();
        }

        // Wait for all threads to complete
        for _ in 0..2 {
            done_rx.recv().unwrap();
        }
    })
    .unwrap();
}

#[test]
fn safepoint_exit_enter() {
    let _fixture = TestFixture::new();

    // Create a channel for proper synchronization instead of sleep
    let (blocking_tx, blocking_rx) = flume::bounded(1);
    let (work_done_tx, work_done_rx) = flume::bounded(1);

    // Use crossbeam scoped threads to simulate the blocking operation without deadlock
    crossbeam::scope(|s| {
        s.spawn(move |_| {
            // Wait for signal to proceed
            blocking_rx.recv().unwrap();

            // Signal work is done
            work_done_tx.send(()).unwrap();
        });

        // Call the exit/enter pair which internally registers the thread
        safepoint_exit();

        // Simulate blocking work using proper channel synchronization
        blocking_tx.send(()).unwrap();
        work_done_rx.recv().unwrap();

        safepoint_enter();

        // Should not panic
        pollcheck();
    })
    .unwrap();
}
