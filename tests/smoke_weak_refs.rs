#![cfg(feature = "smoke")]

use fugrip::{Gc, Weak, CollectorPhase, types::{GcHeader}, SendPtr};
use fugrip::traits::GcTrace;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

#[derive(Default)]
struct Node {}
unsafe impl GcTrace for Node { unsafe fn trace(&self, _s: &mut Vec<SendPtr<GcHeader<()>>>) {} }

#[test]
fn smoke_weak_upgrade_invalidated_after_census() {
    let target = Gc::new(Node {});
    let weak_gc = Weak::new_simple(&target);

    // Weak can be read and upgrade while target is live
    let wr = weak_gc.read().unwrap();
    assert!(wr.upgrade().is_some());

    // Simulate death of target: clear mark bit and redirect pointer to FreeSingleton
    unsafe {
        let hdr = &*target.as_ptr();
        hdr.mark_bit.store(false, Ordering::Release);
    }

    unsafe {
        let hdr = &*target.as_ptr();
        hdr.forwarding_ptr
            .store(fugrip::types::FreeSingleton::instance(), Ordering::Release);
    }

    // Weak upgrade should now return None (forwarding_ptr points to FreeSingleton)
    let wr2 = weak_gc.read().unwrap();
    assert!(wr2.upgrade().is_none());
}
