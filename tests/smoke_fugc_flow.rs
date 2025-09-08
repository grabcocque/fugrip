#![cfg(feature = "smoke")]

use fugrip::collector_phases::*;
use fugrip::{CollectorPhase, types::TypeInfo, types::GcHeader, SendPtr};
use fugrip::traits::GcTrace;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;
use std::{thread, time::Duration};

#[derive(Default)]
struct TestNode {
    children: Vec<SendPtr<GcHeader<()>>>,
}

unsafe impl GcTrace for TestNode {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        stack.extend(self.children.iter().copied());
    }
}

fn type_info_for_testnode() -> &'static TypeInfo {
    fugrip::types::type_info::<TestNode>()
}

unsafe fn make_node(children: Vec<SendPtr<GcHeader<()>>>) -> SendPtr<GcHeader<()>> {
    let header = GcHeader {
        mark_bit: AtomicBool::new(false),
        type_info: type_info_for_testnode(),
        forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
        weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
        data: TestNode { children },
    };
    let b = Box::new(header);
    unsafe { SendPtr::new(Box::into_raw(b) as *mut GcHeader<()>) }
}

#[test]
fn smoke_end_to_end_flow() {
    let c = Arc::new(CollectorState::new());

    // Graph: root -> a -> b
    let b = unsafe { make_node(vec![]) };
    let a = unsafe { make_node(vec![b]) };

    // 1) Wait for trigger: simulate by manually starting marking
    assert_eq!(c.phase.load(Ordering::Acquire), CollectorPhase::Waiting as usize);

    // Helper to run a handshake using coordinator-side fallback (no real mutator threads)
    let run_handshake = |c: &Arc<CollectorState>, actions: Vec<HandshakeAction>| {
        c.active_mutator_count.store(0, Ordering::Release);
        c.request_handshake_with_actions(actions);
    };

    // 2) Turn on store barrier, soft handshake (noop)
    c.enable_store_barrier();
    run_handshake(&c, vec![HandshakeAction::Noop]);

    // 3) Turn on black allocation, soft handshake reset caches
    c.allocation_color.store(true, Ordering::Release);
    run_handshake(&c, vec![HandshakeAction::ResetThreadLocalCaches]);

    // Begin marking phase bookkeeping
    c.set_phase(CollectorPhase::Marking);
    c.marking_active.store(true, Ordering::Release);

    // 4) Mark global roots: queue root "a" as a global root
    fugrip::collector_phases::smoke_add_global_root(a);
    c.mark_global_roots();

    // 5) Soft handshake to request stack scan + reset caches
    run_handshake(&c, vec![HandshakeAction::RequestStackScan, HandshakeAction::ResetThreadLocalCaches]);

    // Must have work (from roots + synthetic stack scan)
    assert!(!c.global_mark_stack.lock().unwrap().is_empty());

    // 6) Tracing to fixpoint (single-threaded for smoke)
    c.trace_to_fixpoint_single_threaded();

    // After tracing, a and b should be marked
    unsafe {
        assert!((*a.as_ptr()).mark_bit.load(Ordering::Acquire));
        assert!((*b.as_ptr()).mark_bit.load(Ordering::Acquire));
    }

    // 7) Turn off store barrier, handshake reset caches
    c.disable_store_barrier();
    c.request_handshake_with_actions(vec![HandshakeAction::ResetThreadLocalCaches]);

    // 8) Prepare for sweeping (phase transition only in smoke)
    c.set_phase(CollectorPhase::Sweeping);
    assert_eq!(c.phase.load(Ordering::Acquire), CollectorPhase::Sweeping as usize);

    // 9) Victory: set back to Waiting
    c.set_phase(CollectorPhase::Waiting);
    assert_eq!(c.phase.load(Ordering::Acquire), CollectorPhase::Waiting as usize);
}

#[test]
fn smoke_dijkstra_store_barrier_marks_new_target() {
    let c = CollectorState::new();
    c.set_phase(CollectorPhase::Marking);
    c.marking_active.store(true, Ordering::Release);
    c.enable_store_barrier();

    let child = unsafe { make_node(vec![]) };

    // Simulate a write to a field of a marked object by invoking the post-write barrier
    c.store_barrier_post_write(child);

    unsafe {
        assert!((*child.as_ptr()).mark_bit.load(Ordering::Acquire));
    }
}
