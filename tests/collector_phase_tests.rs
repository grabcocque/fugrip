use fugrip::{CollectorState, MutatorState, SendPtr, CollectorPhase, FreeSingleton};
use std::sync::atomic::Ordering;

#[test]
fn mutator_defaults_and_try_allocate() {
    let mut m = MutatorState::new();
    assert!(m.allocation_buffer.current.is_null());
    assert!(m.try_allocate::<i32>().is_none());
}

#[test]
fn collector_register_and_phase() {
    let c = CollectorState::new();
    assert!(!c.is_marking());
    c.register_mutator_thread();
    assert!(c.get_active_mutator_count() >= 1);
    c.unregister_mutator_thread();

    c.set_phase(CollectorPhase::Marking);
    assert_eq!(c.phase.load(Ordering::Acquire), CollectorPhase::Marking as usize);

    // Register/unregister thread for GC
    let res = c.register_thread_for_gc((0, 1024));
    assert!(res.is_ok());
    c.unregister_thread_from_gc();
}

#[test]
fn steal_and_donate_work() {
    let c = CollectorState::new();

    // Populate global mark stack with a few sentinel pointers
    {
        let mut gs = c.global_mark_stack.lock().unwrap();
        for _ in 0..10 {
            gs.push(unsafe { SendPtr::new(FreeSingleton::instance()) });
        }
    }

    let stolen = c.steal_marking_work();
    assert!(stolen.is_some());

    let mut local = Vec::new();
    for _ in 0..200 {
        local.push(unsafe { SendPtr::new(FreeSingleton::instance()) });
    }

    c.donate_marking_work(&mut local);
    // donation should have reduced the local stack size
    assert!(local.len() < 200);
}
