use crossbeam::channel;
use fugrip::di::current_container;
use fugrip::safepoint::{GcSafepointPhase, pollcheck, safepoint_enter, safepoint_exit};
use fugrip::test_utils::TestFixture;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::time::Duration;

#[test]
fn safepoint_integration_with_custom_coordinator() {
    let fixture = TestFixture::new();
    let coordinator = &fixture.coordinator;
    let manager = current_container().safepoint_manager();
    manager.register_thread();

    // Test barrier activation
    let activated = Arc::new(AtomicUsize::new(0));
    let activated_clone = Arc::clone(&activated);
    manager.request_safepoint(Box::new(move || {
        activated_clone.fetch_add(1, Ordering::SeqCst);
    }));
    println!("safepoint stats before: {:?}", manager.get_stats());
    for _ in 0..10 {
        pollcheck();
    }
    println!("safepoint stats after: {:?}", manager.get_stats());
    manager.clear_safepoint();

    // Check if callback was executed
    assert!(activated.load(Ordering::SeqCst) > 0);

    // Test direct activation first
    coordinator.activate_barriers_at_safepoint();
    assert!(coordinator.write_barrier().is_active());

    // Reset for safepoint test
    coordinator.write_barrier().deactivate();
    coordinator.black_allocator().deactivate();
    assert!(!coordinator.write_barrier().is_active());

    // Now test the actual GC safepoint
    manager.request_gc_safepoint(GcSafepointPhase::BarrierActivation);
    pollcheck();
    manager.clear_safepoint();

    // Verify coordinator state was affected
    assert!(coordinator.write_barrier().is_active());
}

#[test]
fn safepoint_thread_coordination() {
    let _fixture = TestFixture::new();
    let manager = current_container().safepoint_manager();

    // Clear state
    manager.clear_safepoint();

    // Register some test threads
    let mut handles = vec![];
    for _ in 0..3 {
        let handle = thread::spawn(move || {
            // Register this thread with the safepoint manager
            current_container().safepoint_manager().register_thread();
            for _ in 0..1000 {
                pollcheck();
            }
        });
        handles.push(handle);
    }

    // Request safepoint and wait
    let executed = Arc::new(AtomicUsize::new(0));
    let executed_clone = Arc::clone(&executed);
    manager.request_safepoint(Box::new(move || {
        executed_clone.fetch_add(1, Ordering::SeqCst);
    }));

    // Main thread also participates
    for _ in 0..100 {
        pollcheck();
    }

    // Wait for all threads to hit safepoint
    assert!(manager.wait_for_safepoint(Duration::from_secs(5)));

    manager.clear_safepoint();

    // Verify execution count
    assert!(executed.load(Ordering::SeqCst) > 0);

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn safepoint_soft_handshake() {
    let _fixture = TestFixture::new();
    let manager = current_container().safepoint_manager();

    // Spawn worker threads that use the dedicated manager instance
    let mut handles = vec![];
    for _ in 0..2 {
        let handle = thread::spawn(move || {
            current_container().safepoint_manager().register_thread();
            for _ in 0..1000 {
                pollcheck();
            }
        });
        handles.push(handle);
    }

    // Request soft handshake on the dedicated manager
    manager.request_soft_handshake(Box::new(|| {}));

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn safepoint_exit_enter() {
    let _fixture = TestFixture::new();

    // Call the exit/enter pair which internally registers the thread
    safepoint_exit();
    // Simulate blocking work using a timed channel wait
    let _ = channel::after(Duration::from_millis(10)).recv();
    safepoint_enter();

    // Should not panic
    pollcheck();
}
