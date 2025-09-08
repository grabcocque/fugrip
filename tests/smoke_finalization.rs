#![cfg(feature = "smoke")]

use fugrip::{Finalizable, CollectorPhase};
use fugrip::memory::CLASSIFIED_ALLOCATOR;
use fugrip::collector_phases::CollectorState;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Default)]
struct Resource {
    flag: Arc<AtomicBool>,
}

impl Finalizable for Resource {
    fn finalize(&mut self) {
        self.flag.store(true, Ordering::Release);
    }
}

#[test]
fn smoke_finalizer_reviving_runs_finalize() {
    // Allocate a finalizable object via classified allocator
    let flag = Arc::new(AtomicBool::new(false));
    let res = Resource { flag: flag.clone() };

    let gc_finalizable = CLASSIFIED_ALLOCATOR.allocate_finalizable(res);

    // New object is live; simulate it becoming dead by clearing mark bit
    unsafe {
        let header = &*gc_finalizable.as_ptr();
        header.mark_bit.store(false, Ordering::Release);
    }

    let collector = CollectorState::new();
    collector.reviving_phase();

    // Finalize should have run and set the flag
    assert!(flag.load(Ordering::Acquire));

    // After reviving, phase is Reviving
    assert_eq!(collector.get_phase(), CollectorPhase::Reviving);
}
// Resource contains no GC pointers
unsafe impl fugrip::traits::GcTrace for Resource {
    unsafe fn trace(&self, _stack: &mut Vec<fugrip::SendPtr<fugrip::GcHeader<()>>>) {}
}
