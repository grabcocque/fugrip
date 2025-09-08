#![cfg(feature = "smoke")]

use fugrip::collector_phases::*;
use fugrip::{CollectorPhase, FreeSingleton, SendPtr};
use std::sync::atomic::Ordering;

// Smoke: parallel marking setup reflects multi-worker configuration
#[test]
fn smoke_parallel_marking_setup() {
    let c = CollectorState::new();

    assert_eq!(c.phase.load(Ordering::Acquire), CollectorPhase::Waiting as usize);
    assert!(!c.marking_active.load(Ordering::Acquire));

    // Begin marking; this should flip allocation color and configure workers
    c.start_marking_phase();

    // Marking may complete quickly; assert configuration side-effects instead
    assert!(c.allocation_color.load(Ordering::Acquire));

    // MarkCoordinator should reflect parallel configuration
    let expected_workers = num_cpus::get();
    assert_eq!(
        c.mark_coordinator.worker_count.load(Ordering::Acquire),
        expected_workers
    );
}

// Smoke: handshake seen by mutator and non-blocking acknowledgement
#[test]
fn smoke_soft_handshake_ack_nonblocking() {
    let c = CollectorState::new();

    // Configure state akin to marking
    c.allocation_color.store(true, Ordering::Release);
    c.active_mutator_count.store(1, Ordering::Release);

    // Request handshake via legacy flag to simulate on-the-fly signal
    c.handshake_requested.store(true, Ordering::Release);

    let mut m = MutatorState::new();
    // Mutator should observe request, capture color, and ack without blocking
    m.check_handshake(&c);

    assert!(!m.is_in_handshake);
    assert!(m.allocating_black);
}

// Smoke: sweeping phase selection (no heap effects asserted here)
#[test]
fn smoke_sweeping_phase_sets_phase() {
    let c = CollectorState::new();
    // Only assert phase transition; avoid invoking sweeping over real heap
    c.set_phase(CollectorPhase::Sweeping);
    assert_eq!(c.phase.load(Ordering::Acquire), CollectorPhase::Sweeping as usize);
}

// Smoke: free singleton is stable and reused (non-moving semantics helper)
#[test]
fn smoke_non_moving_free_singleton_stable_ptr() {
    let a = FreeSingleton::instance();
    let b = FreeSingleton::instance();
    assert_eq!(a, b);
}

// Grey-stack: out-of-work -> soft handshake -> discover new roots -> resume marking -> fixpoint
#[test]
fn smoke_grey_stack_fixpoint_loop() {
    use fugrip::types::{GcHeader, TypeInfo};
    use fugrip::traits::GcTrace;
    use std::sync::atomic::{AtomicBool, AtomicPtr};
    use std::{thread, time::Duration};
    use std::sync::Arc;

    #[derive(Default)]
    struct TestNode { children: Vec<SendPtr<GcHeader<()>>> }
    unsafe impl GcTrace for TestNode {
        unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) { stack.extend(self.children.iter().copied()); }
    }
    fn ti() -> &'static TypeInfo { fugrip::types::type_info::<TestNode>() }
    unsafe fn make(children: Vec<SendPtr<GcHeader<()>>>) -> SendPtr<GcHeader<()>> {
        let h = GcHeader { mark_bit: AtomicBool::new(false), type_info: ti(), forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()), weak_ref_list: AtomicPtr::new(std::ptr::null_mut()), data: TestNode { children } };
        unsafe { SendPtr::new(Box::into_raw(Box::new(h)) as *mut GcHeader<()>) }
    }

    let c = Arc::new(CollectorState::new());

    // Build a small graph that will be revealed by stack scan later: x -> y
    let y = unsafe { make(vec![]) };
    let x = unsafe { make(vec![y]) };

    // Start marking with no global roots; go to out-of-work
    c.set_phase(CollectorPhase::Marking);
    c.marking_active.store(true, Ordering::Release);
    c.trace_to_fixpoint_single_threaded();
    assert!(c.global_mark_stack.lock().unwrap().is_empty());

    // Prepare mock stack root discovery to reveal x on next handshake
    fugrip::collector_phases::smoke_clear_all_roots();
    fugrip::collector_phases::smoke_add_stack_root(x);

    // Helper: run a soft handshake using coordinator fallback (no mutator thread)
    let run_handshake = |c: &Arc<CollectorState>| {
        c.active_mutator_count.store(0, Ordering::Release);
        c.request_handshake_with_actions(vec![HandshakeAction::RequestStackScan]);
    };

    // Soft handshake should reveal x into the global mark stack
    run_handshake(&c);
    assert!(!c.global_mark_stack.lock().unwrap().is_empty());

    // Resume marking to fixpoint; should mark x and y
    c.trace_to_fixpoint_single_threaded();
    unsafe {
        assert!((*x.as_ptr()).mark_bit.load(Ordering::Acquire));
        assert!((*y.as_ptr()).mark_bit.load(Ordering::Acquire));
    }
}

// Placeholder: Dijkstra store barrier semantics (requires barrier hook on writes)
#[test]
#[ignore = "Dijkstra store barrier not wired into this refactor yet"]
fn smoke_dijkstra_store_barrier() {
    // Intentionally left as a placeholder until the store barrier is implemented.
}
