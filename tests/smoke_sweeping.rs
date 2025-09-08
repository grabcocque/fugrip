#![cfg(feature = "smoke")]

use fugrip::{collector::sweep_coordinator::SweepCoordinator, types::{GcHeader, TypeInfo, FreeSingleton}};
use fugrip::{SendPtr, collector_phases::CollectorState};
use fugrip::traits::GcTrace;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

#[derive(Default)]
struct Dummy {}
unsafe impl GcTrace for Dummy { unsafe fn trace(&self, _s: &mut Vec<SendPtr<GcHeader<()>>>) {} }
fn ti() -> &'static TypeInfo { fugrip::types::type_info::<Dummy>() }

unsafe fn make_with_mark(marked: bool) -> *mut GcHeader<()> {
    let h = GcHeader {
        mark_bit: AtomicBool::new(marked),
        type_info: ti(),
        forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
        weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
        data: Dummy {},
    };
    Box::into_raw(Box::new(h)) as *mut GcHeader<()>
}

#[test]
fn smoke_sweeping_list_marks_dead_and_clears_live() {
    let sweeper = SweepCoordinator::new();
    let live = unsafe { make_with_mark(true) };
    let dead = unsafe { make_with_mark(false) };

    sweeper.sweep_headers_list(&[live, dead]);

    unsafe {
        // Live object should be unmarked for the next cycle
        assert_eq!((*live).mark_bit.load(Ordering::Acquire), false);
        // Dead object should forward to free singleton
        let fwd = (*dead).forwarding_ptr.load(Ordering::Acquire);
        assert_eq!(fwd, FreeSingleton::instance());
    }
}

#[test]
fn smoke_allocation_color_depends_on_page_state() {
    let c = CollectorState::new();
    unsafe {
        // Not-yet-swept page: allocation black
        let a = c.smoke_allocate_with_page_state(Dummy {}, false);
        assert_eq!((*a.as_ptr()).mark_bit.load(Ordering::Acquire), true);

        // Already-swept page: allocation white
        let b = c.smoke_allocate_with_page_state(Dummy {}, true);
        assert_eq!((*b.as_ptr()).mark_bit.load(Ordering::Acquire), false);
    }
}
