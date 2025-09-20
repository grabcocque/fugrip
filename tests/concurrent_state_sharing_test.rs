//! Sanity checks for the FUGC coordinator's state sharing behaviour.

use fugrip::FugcPhase;
use fugrip::thread::MutatorThread;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn coordinator_state_sharing_works() {
    // Use shared test fixture with custom config
    let fixture = fugrip::test_utils::TestFixture::new_with_config(0x10000000, 32 * 1024 * 1024, 2);
    let coordinator = &fixture.coordinator;
    let thread_registry = fixture.thread_registry();

    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    assert!(!coordinator.is_collecting());

    let mutator = MutatorThread::new(1);
    thread_registry.register(mutator.clone());

    // Get the registered mutator with the correct handler
    let registered_mutator = thread_registry
        .get(1)
        .expect("Mutator should be registered");
    use std::sync::atomic::{AtomicBool, Ordering};
    let running = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&running);

    crossbeam::scope(|s| {
        let mutator_clone = registered_mutator.clone();
        s.spawn(move |_| {
            while flag.load(Ordering::Relaxed) {
                mutator_clone.poll_safepoint();
            }
        });

        // Test coordinator state sharing without triggering GC to avoid deadlock
        // Validate that the coordinator and mutator thread state is properly shared
        assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
        assert!(!coordinator.is_collecting());

        // Test that coordinator components are accessible and functioning
        let stats = coordinator.get_cycle_stats();
        assert_eq!(stats.cycles_completed, 0); // Should start at 0

        // Verify thread is registered and active
        let threads = thread_registry.iter();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].id(), mutator.id());

        coordinator.trigger_gc();
        assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));
        // Stop polling thread cleanly
        running.store(false, std::sync::atomic::Ordering::Relaxed);
    })
    .unwrap();
    thread_registry.unregister(mutator.id());

    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    let stats = coordinator.get_cycle_stats();
    assert!(stats.cycles_completed >= 1);
}

#[test]
fn multiple_coordinators_share_components() {
    // Use two isolated fixtures to create separate coordinators
    let fixture1 =
        fugrip::test_utils::TestFixture::new_with_config(0x20000000, 32 * 1024 * 1024, 2);
    let fixture2 =
        fugrip::test_utils::TestFixture::new_with_config(0x20000000, 32 * 1024 * 1024, 2);

    let coordinator1 = Arc::clone(&fixture1.coordinator);
    let coordinator2 = Arc::clone(&fixture2.coordinator);

    assert_eq!(coordinator1.current_phase(), FugcPhase::Idle);
    assert_eq!(coordinator2.current_phase(), FugcPhase::Idle);

    coordinator1.trigger_gc();
    coordinator2.trigger_gc();

    assert!(coordinator1.wait_until_idle(Duration::from_millis(500)));
    assert!(coordinator2.wait_until_idle(Duration::from_millis(500)));

    let stats1 = coordinator1.get_cycle_stats();
    let stats2 = coordinator2.get_cycle_stats();
    assert!(stats1.cycles_completed >= 1);
    assert!(stats2.cycles_completed >= 1);
}
