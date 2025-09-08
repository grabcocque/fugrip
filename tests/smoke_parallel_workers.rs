#![cfg(feature = "smoke")]

use fugrip::{collector_phases::CollectorState, SendPtr, types::{GcHeader, TypeInfo}};
use fugrip::traits::GcTrace;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

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
    SendPtr::new(Box::into_raw(Box::new(header)) as *mut GcHeader<()>)
}

#[test]
fn smoke_parallel_workers_complete_and_reduce_work() {
    use std::sync::Arc;
    let collector = Arc::new(CollectorState::new());

    // Seed global mark stack with many nodes that donate work
    const N: usize = 5000;
    let mut roots = Vec::with_capacity(N);
    for _ in 0..N {
        // Each node has a few FreeSingleton pointers as children (harmless)
        let mut kids = Vec::new();
        for _ in 0..5 {
            kids.push(unsafe { SendPtr::new(fugrip::types::FreeSingleton::instance()) });
        }
        roots.push(unsafe { make_node(kids) });
    }

    {
        let mut global = collector.global_mark_stack.lock().unwrap();
        global.extend(roots);
    }

    // Manually start parallel marking and worker threads without the single-thread tracer
    collector.marking_active.store(true, Ordering::Release);
    collector.enable_store_barrier();
    let worker_count = 4;
    collector.mark_coordinator.start_parallel_marking(worker_count);

    let mut handles = Vec::new();
    for _ in 0..worker_count {
        let c = collector.clone();
        handles.push(std::thread::spawn(move || {
            c.mark_coordinator.run_marking_worker(c.clone());
        }));
    }

    for h in handles { let _ = h.join(); }
    collector.marking_active.store(false, Ordering::Release);
    collector.disable_store_barrier();

    // After convergence, work queue should be empty
    let remaining = collector.mark_coordinator.get_work_queue_size();
    assert_eq!(remaining, 0);

    // And we should have observed some steals or donations
    let steals = collector.mark_coordinator.get_steal_count();
    let donations = collector.mark_coordinator.get_donation_count();
    assert!(steals > 0 || donations > 0, "expected steals or donations, got steals={}, donations={}", steals, donations);
}
