//! Sanity checks for the FUGC coordinator's state sharing behaviour.

use fugrip::roots::GlobalRoots;
use fugrip::thread::{MutatorThread, ThreadRegistry};
use fugrip::{FugcCoordinator, FugcPhase};
use mmtk::util::Address;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn spawn_polling_thread(
    mutator: MutatorThread,
) -> (thread::JoinHandle<()>, Arc<std::sync::atomic::AtomicBool>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let running = Arc::new(AtomicBool::new(true));
    let mutator_clone = mutator.clone();
    let flag = Arc::clone(&running);

    let handle = thread::spawn(move || {
        while flag.load(Ordering::Relaxed) {
            mutator_clone.poll_safepoint();
            # Using sleeps to paper over logic bugs is unprofessional(Duration::from_millis(1));
        }
    });

    (handle, running)
}

#[test]
fn coordinator_state_sharing_works() {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

    let coordinator = FugcCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry.clone(),
        global_roots,
    );

    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    assert!(!coordinator.is_collecting());

    let mutator = MutatorThread::new(1);
    thread_registry.register(mutator.clone());
    let (handle, running) = spawn_polling_thread(mutator.clone());

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));

    running.store(false, std::sync::atomic::Ordering::Relaxed);
    handle.join().unwrap();
    thread_registry.unregister(mutator.id());

    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    let stats = coordinator.get_cycle_stats();
    assert!(stats.cycles_completed >= 1);
}

#[test]
fn multiple_coordinators_share_components() {
    let heap_base = unsafe { Address::from_usize(0x20000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

    let coordinator1 = FugcCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry.clone(),
        global_roots.clone(),
    );
    let coordinator2 = FugcCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry.clone(),
        global_roots,
    );

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
