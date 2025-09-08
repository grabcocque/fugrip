use fugrip::collector_phases::*;
use fugrip::{CollectorPhase, GcHeader, GcTrace, SendPtr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

// Test struct for collector phase testing
#[derive(Debug)]
struct TestNode {
    id: usize,
    data: Vec<u8>,
}

unsafe impl GcTrace for TestNode {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // No GC references to trace in this simple struct
    }
}

#[test]
fn test_collector_state_creation() {
    let collector = CollectorState::new();
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Waiting as usize
    );
    assert!(!collector.marking_active.load(Ordering::Acquire));
    assert!(!collector.allocation_color.load(Ordering::Acquire));
}

#[test]
fn test_collector_phase_transitions() {
    let collector = CollectorState::new();

    // Test setting different phases
    collector.set_phase(CollectorPhase::Marking);
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Marking as usize
    );

    collector.set_phase(CollectorPhase::Censusing);
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Censusing as usize
    );

    collector.set_phase(CollectorPhase::Sweeping);
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Sweeping as usize
    );

    collector.set_phase(CollectorPhase::Waiting);
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Waiting as usize
    );
}

#[test]
fn test_collection_request() {
    let collector = CollectorState::new();

    // Initially should be waiting
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Waiting as usize
    );

    // Request collection should change phase to marking
    collector.request_collection();
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Marking as usize
    );

    // Requesting again while already marking should not change phase
    collector.request_collection();
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Marking as usize
    );
}

#[test]
fn test_marking_state() {
    let collector = CollectorState::new();

    // Initially not marking
    assert!(!collector.is_marking());

    // Set marking active
    collector.marking_active.store(true, Ordering::Release);
    assert!(collector.is_marking());

    collector.marking_active.store(false, Ordering::Release);
    assert!(!collector.is_marking());
}

#[test]
fn test_allocation_color() {
    let collector = CollectorState::new();

    // Initially white (false)
    assert!(!collector.allocation_color.load(Ordering::Acquire));

    // Change to black
    collector.allocation_color.store(true, Ordering::Release);
    assert!(collector.allocation_color.load(Ordering::Acquire));

    // Back to white
    collector.allocation_color.store(false, Ordering::Release);
    assert!(!collector.allocation_color.load(Ordering::Acquire));
}

#[test]
fn test_thread_registration_creation() {
    let registration = ThreadRegistration {
        thread_id: thread::current().id(),
        stack_base: 0x1000000,
        stack_bounds: (0x1000000, 0x1100000),
        last_known_sp: AtomicUsize::new(0x1050000),
        local_roots: Vec::new(),
        is_active: AtomicBool::new(true),
    };

    assert!(registration.is_active.load(Ordering::Acquire));
    assert_eq!(registration.stack_base, 0x1000000);
    assert_eq!(registration.stack_bounds, (0x1000000, 0x1100000));
    assert_eq!(
        registration.last_known_sp.load(Ordering::Acquire),
        0x1050000
    );
}

#[test]
fn test_thread_registration_modification() {
    let registration = ThreadRegistration {
        thread_id: thread::current().id(),
        stack_base: 0x1000000,
        stack_bounds: (0x1000000, 0x1100000),
        last_known_sp: AtomicUsize::new(0x1050000),
        local_roots: Vec::new(),
        is_active: AtomicBool::new(true),
    };

    // Modify stack pointer
    registration
        .last_known_sp
        .store(0x1060000, Ordering::Release);
    assert_eq!(
        registration.last_known_sp.load(Ordering::Acquire),
        0x1060000
    );

    // Deactivate thread
    registration.is_active.store(false, Ordering::Release);
    assert!(!registration.is_active.load(Ordering::Acquire));
}

#[test]
fn test_mutator_state_creation() {
    let mutator = MutatorState::new();
    assert!(mutator.local_mark_stack.is_empty());
    assert!(!mutator.is_in_handshake);
    assert!(!mutator.allocating_black);
}

#[test]
fn test_mutator_state_handshake_participation() {
    let mut mutator = MutatorState::new();
    let collector = CollectorState::new();

    // Initially not in handshake
    assert!(!mutator.is_in_handshake);

    // Check handshake when not requested - should do nothing
    mutator.check_handshake(&collector);
    assert!(!mutator.is_in_handshake);

    // Request handshake
    collector.handshake_requested.store(true, Ordering::Release);

    // Check handshake when requested
    mutator.check_handshake(&collector);
    // Should have processed handshake (sets is_in_handshake to false at end)
    assert!(!mutator.is_in_handshake);
}

#[test]
fn test_mutator_allocation_attempt() {
    let mut mutator = MutatorState::new();

    // Try allocation without buffer - should fail
    let result: Option<*mut GcHeader<TestNode>> = mutator.try_allocate();
    assert!(result.is_none());
}

#[test]
fn test_handshake_mechanism() {
    let collector = CollectorState::new();

    // Test handshake request state
    assert!(!collector.handshake_requested.load(Ordering::Acquire));

    collector.handshake_requested.store(true, Ordering::Release);
    assert!(collector.handshake_requested.load(Ordering::Acquire));

    collector
        .handshake_requested
        .store(false, Ordering::Release);
    assert!(!collector.handshake_requested.load(Ordering::Acquire));
}

#[test]
fn test_worker_counting() {
    let collector = CollectorState::new();

    // Initially no workers
    assert_eq!(collector.worker_count.load(Ordering::Acquire), 0);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 0);

    // Simulate worker registration
    collector.worker_count.store(3, Ordering::Release);
    assert_eq!(collector.worker_count.load(Ordering::Acquire), 3);

    // Simulate workers finishing
    collector.workers_finished.store(1, Ordering::Release);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 1);

    collector.workers_finished.store(3, Ordering::Release);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 3);
}

#[test]
fn test_mutator_counting() {
    let collector = CollectorState::new();

    // Initially no active mutators
    assert_eq!(collector.active_mutator_count.load(Ordering::Acquire), 0);
    assert_eq!(
        collector.handshake_acknowledgments.load(Ordering::Acquire),
        0
    );

    // Simulate mutator registration
    collector.active_mutator_count.store(2, Ordering::Release);
    assert_eq!(collector.active_mutator_count.load(Ordering::Acquire), 2);

    // Simulate handshake acknowledgments
    collector
        .handshake_acknowledgments
        .fetch_add(1, Ordering::Release);
    assert_eq!(
        collector.handshake_acknowledgments.load(Ordering::Acquire),
        1
    );

    collector
        .handshake_acknowledgments
        .fetch_add(1, Ordering::Release);
    assert_eq!(
        collector.handshake_acknowledgments.load(Ordering::Acquire),
        2
    );
}

#[test]
fn test_suspension_mechanism() {
    let collector = CollectorState::new();

    // Test suspension state
    assert_eq!(collector.suspend_count.load(Ordering::Acquire), 0);
    assert!(!collector.suspension_requested.load(Ordering::Acquire));

    collector
        .suspension_requested
        .store(true, Ordering::Release);
    assert!(collector.suspension_requested.load(Ordering::Acquire));

    collector.suspend_count.fetch_add(1, Ordering::Release);
    assert_eq!(collector.suspend_count.load(Ordering::Acquire), 1);

    collector
        .suspension_requested
        .store(false, Ordering::Release);
    assert!(!collector.suspension_requested.load(Ordering::Acquire));
}

#[test]
fn test_global_mark_stack() {
    let collector = CollectorState::new();

    // Access global mark stack
    {
        let mut stack = collector.global_mark_stack.lock().unwrap();
        assert!(stack.is_empty());

        // Add some dummy pointers (as usize cast to ptr)
        let dummy_ptr = unsafe { SendPtr::new(0x1000 as *mut GcHeader<()>) };
        stack.push(dummy_ptr);
        assert_eq!(stack.len(), 1);

        stack.clear();
        assert!(stack.is_empty());
    }
}

#[test]
fn test_concurrent_phase_changes() {
    let collector = Arc::new(CollectorState::new());
    let collector_clone = collector.clone();

    let handle = thread::spawn(move || {
        // Change phase from another thread
        collector_clone.set_phase(CollectorPhase::Marking);
        collector_clone.set_phase(CollectorPhase::Censusing);
        collector_clone.set_phase(CollectorPhase::Sweeping);
    });

    handle.join().unwrap();

    // Verify final phase
    assert_eq!(
        collector.phase.load(Ordering::Acquire),
        CollectorPhase::Sweeping as usize
    );
}

#[test]
fn test_registered_threads_access() {
    let collector = CollectorState::new();

    // Access registered threads list
    {
        let threads = collector.registered_threads.lock().unwrap();
        assert!(threads.is_empty());
    }

    // Add a thread registration
    {
        let mut threads = collector.registered_threads.lock().unwrap();
        let registration = ThreadRegistration {
            thread_id: thread::current().id(),
            stack_base: 0x1000000,
            stack_bounds: (0x1000000, 0x1100000),
            last_known_sp: AtomicUsize::new(0x1050000),
            local_roots: Vec::new(),
            is_active: AtomicBool::new(true),
        };
        threads.push(registration);
        assert_eq!(threads.len(), 1);
    }
}
