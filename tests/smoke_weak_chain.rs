#![cfg(feature = "smoke")]

use fugrip::{Gc, Weak, SendPtr, types::GcHeader};
use fugrip::traits::GcTrace;
use fugrip::memory::CLASSIFIED_ALLOCATOR;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
struct Node {}
unsafe impl GcTrace for Node { unsafe fn trace(&self, _s: &mut Vec<SendPtr<GcHeader<()>>>) {} }

#[test]
fn smoke_weak_chain_invalidation() {
    // Create a target object
    let target = Gc::new(Node {});

    // Allocate several GC-managed Weak nodes linked to the target
    let w1: Gc<Weak<Node>> = CLASSIFIED_ALLOCATOR.allocate_weak(&target);
    let w2: Gc<Weak<Node>> = CLASSIFIED_ALLOCATOR.allocate_weak(&target);
    let w3: Gc<Weak<Node>> = CLASSIFIED_ALLOCATOR.allocate_weak(&target);

    // All should upgrade while target is live
    assert!(w1.read().unwrap().read().unwrap().upgrade().is_some());
    assert!(w2.read().unwrap().read().unwrap().upgrade().is_some());
    assert!(w3.read().unwrap().read().unwrap().upgrade().is_some());

    // Simulate death of target by forwarding to FREE_SINGLETON (sweep effect)
    unsafe {
        let header = &*target.as_ptr();
        header
            .forwarding_ptr
            .store(fugrip::types::FreeSingleton::instance(), Ordering::Release);
    }

    // Now upgrades should fail
    assert!(w1.read().unwrap().read().unwrap().upgrade().is_none());
    assert!(w2.read().unwrap().read().unwrap().upgrade().is_none());
    assert!(w3.read().unwrap().read().unwrap().upgrade().is_none());
}
