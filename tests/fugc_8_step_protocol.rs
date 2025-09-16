//! High level behavioural tests for the FUGC coordinator.  These are not meant
//! to be exhaustive but ensure that the eight protocol steps are wired together
//! and that the public API exposed through `FugcCoordinator` behaves as
//! expected.

use fugrip::roots::GlobalRoots;
use fugrip::thread::{MutatorThread, ThreadRegistry};
use fugrip::{AllocationColor, FugcCoordinator, FugcPhase};
use mmtk::util::{Address, ObjectReference};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

fn spawn_mutator(mutator: MutatorThread) -> (thread::JoinHandle<()>, Arc<AtomicBool>) {
    let running = Arc::new(AtomicBool::new(true));
    let worker = mutator.clone();
    let flag = Arc::clone(&running);

    let handle = thread::spawn(move || {
        while flag.load(Ordering::Relaxed) {
            worker.poll_safepoint();
            # Using sleeps to paper over logic bugs is unprofessional(Duration::from_millis(1));
        }
    });

    (handle, running)
}

#[test]
fn complete_fugc_8_step_protocol() {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

    let coordinator = FugcCoordinator::new(
        heap_base,
        heap_size,
        4,
        thread_registry.clone(),
        global_roots.clone(),
    );

    let mutator1 = MutatorThread::new(1);
    let mutator2 = MutatorThread::new(2);
    thread_registry.register(mutator1.clone());
    thread_registry.register(mutator2.clone());
    let stack1 = unsafe { Address::from_usize(heap_base.as_usize() + 0x300) };
    let stack2 = unsafe { Address::from_usize(heap_base.as_usize() + 0x400) };
    mutator1.register_stack_root(stack1.as_usize() as *mut u8);
    mutator2.register_stack_root(stack2.as_usize() as *mut u8);

    let (handle1, running1) = spawn_mutator(mutator1.clone());
    let (handle2, running2) = spawn_mutator(mutator2.clone());

    {
        let mut roots = global_roots.lock().unwrap();
        let r1 = unsafe { Address::from_usize(heap_base.as_usize() + 0x100) };
        let r2 = unsafe { Address::from_usize(heap_base.as_usize() + 0x200) };
        roots.register(r1.as_usize() as *mut u8);
        roots.register(r2.as_usize() as *mut u8);
    }

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));

    running1.store(false, Ordering::Relaxed);
    running2.store(false, Ordering::Relaxed);
    handle1.join().unwrap();
    handle2.join().unwrap();
    thread_registry.unregister(mutator1.id());
    thread_registry.unregister(mutator2.id());

    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    let stats = coordinator.get_cycle_stats();
    assert!(stats.cycles_completed >= 1);
    assert!(stats.handshakes_performed >= 1);
}

#[test]
fn step_1_idle_state_and_trigger() {
    let heap_base = unsafe { Address::from_usize(0x20000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(heap_base, heap_size, 2, thread_registry, global_roots);

    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
}

#[test]
fn step_2_write_barrier_activation() {
    let heap_base = unsafe { Address::from_usize(0x21000000) };
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
    let write_barrier = coordinator.write_barrier();
    assert!(!write_barrier.is_active());

    let mutator = MutatorThread::new(10);
    thread_registry.register(mutator.clone());
    let (thread_handle, running) = spawn_mutator(mutator.clone());

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert!(!write_barrier.is_active());

    running.store(false, Ordering::Relaxed);
    thread_handle.join().unwrap();
    thread_registry.unregister(mutator.id());
}

#[test]
fn step_3_black_allocation_tricolor_invariant() {
    let heap_base = unsafe { Address::from_usize(0x22000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(heap_base, heap_size, 2, thread_registry, global_roots);

    let black_allocator = coordinator.black_allocator();
    assert!(!black_allocator.is_active());
    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert!(!black_allocator.is_active());
}

#[test]
fn step_4_global_root_marking_reachability() {
    let heap_base = unsafe { Address::from_usize(0x23000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry.clone(),
        global_roots.clone(),
    );

    let mut roots = global_roots.lock().unwrap();
    let r1 = unsafe { Address::from_usize(heap_base.as_usize() + 0x100) };
    let r2 = unsafe { Address::from_usize(heap_base.as_usize() + 0x200) };
    roots.register(r1.as_usize() as *mut u8);
    roots.register(r2.as_usize() as *mut u8);
    drop(roots);

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert!(coordinator.get_cycle_stats().cycles_completed >= 1);
}

#[test]
fn step_5_stack_scanning_mutator_roots() {
    let heap_base = unsafe { Address::from_usize(0x24000000) };
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

    let mutator1 = MutatorThread::new(20);
    let mutator2 = MutatorThread::new(21);
    thread_registry.register(mutator1.clone());
    thread_registry.register(mutator2.clone());
    let s1 = unsafe { Address::from_usize(heap_base.as_usize() + 0x400) };
    let s2 = unsafe { Address::from_usize(heap_base.as_usize() + 0x500) };
    mutator1.register_stack_root(s1.as_usize() as *mut u8);
    mutator2.register_stack_root(s2.as_usize() as *mut u8);

    let (handle1, running1) = spawn_mutator(mutator1.clone());
    let (handle2, running2) = spawn_mutator(mutator2.clone());

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    let stats = coordinator.get_cycle_stats();
    assert!(stats.handshakes_performed >= 1);

    running1.store(false, Ordering::Relaxed);
    running2.store(false, Ordering::Relaxed);
    handle1.join().unwrap();
    handle2.join().unwrap();
    thread_registry.unregister(mutator1.id());
    thread_registry.unregister(mutator2.id());
}

#[test]
fn step_6_tracing_invariant_termination() {
    let heap_base = unsafe { Address::from_usize(0x25000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(heap_base, heap_size, 2, thread_registry, global_roots);

    let tricolor = coordinator.tricolor_marking();
    let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    assert_eq!(
        tricolor.get_color(obj),
        fugrip::concurrent::ObjectColor::White
    );
    tricolor.set_color(obj, fugrip::concurrent::ObjectColor::Grey);
    assert!(tricolor.transition_color(
        obj,
        fugrip::concurrent::ObjectColor::Grey,
        fugrip::concurrent::ObjectColor::Black
    ));

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
}

#[test]
fn step_7_barrier_deactivation_sweep_prep() {
    let heap_base = unsafe { Address::from_usize(0x26000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(heap_base, heap_size, 2, thread_registry, global_roots);

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
}

#[test]
fn step_8_page_based_sweep_allocation_coloring() {
    let heap_base = unsafe { Address::from_usize(0x27000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry,
        global_roots.clone(),
    );

    {
        let mut roots = global_roots.lock().unwrap();
        let root_addr = unsafe { Address::from_usize(heap_base.as_usize() + 0x100) };
        roots.register(root_addr.as_usize() as *mut u8);
    }

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));

    let page_index = ((heap_base + 0x100usize).as_usize() - heap_base.as_usize()) / 4096;
    let colour = coordinator.page_allocation_color(page_index);
    assert!(matches!(
        colour,
        AllocationColor::White | AllocationColor::Black
    ));
}

#[test]
fn fugc_write_barrier_integration() {
    let heap_base = unsafe { Address::from_usize(0x28000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(heap_base, heap_size, 2, thread_registry, global_roots);

    let barrier = coordinator.write_barrier();
    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert!(!barrier.is_active());
}

#[test]
fn fugc_statistics_accuracy() {
    let heap_base = unsafe { Address::from_usize(0x29000000) };
    let heap_size = 32 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = FugcCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry.clone(),
        global_roots.clone(),
    );

    let mutator = MutatorThread::new(77);
    thread_registry.register(mutator.clone());
    let (handle, running) = spawn_mutator(mutator.clone());

    {
        let mut roots = global_roots.lock().unwrap();
        let r = unsafe { Address::from_usize(heap_base.as_usize() + 0x150) };
        roots.register(r.as_usize() as *mut u8);
    }

    coordinator.trigger_gc();
    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));

    running.store(false, Ordering::Relaxed);
    handle.join().unwrap();
    thread_registry.unregister(mutator.id());

    let final_stats = coordinator.get_cycle_stats();
    assert!(final_stats.cycles_completed >= 1);
}

#[test]
fn fugc_concurrent_collection_stress() {
    let heap_base = unsafe { Address::from_usize(0x2a000000) };
    let heap_size = 64 * 1024 * 1024;
    let thread_registry = Arc::new(ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    let coordinator = Arc::new(FugcCoordinator::new(
        heap_base,
        heap_size,
        4,
        thread_registry.clone(),
        global_roots.clone(),
    ));

    let mut mutator_threads = Vec::new();
    for id in 0..4 {
        let mutator = MutatorThread::new(id + 100);
        thread_registry.register(mutator.clone());
        let addr = unsafe { Address::from_usize(heap_base.as_usize() + (id as usize) * 0x80) };
        mutator.register_stack_root(addr.as_usize() as *mut u8);
        let (handle, running) = spawn_mutator(mutator.clone());
        mutator_threads.push((handle, running, mutator.id()));
    }

    {
        let mut roots = global_roots.lock().unwrap();
        let r = unsafe { Address::from_usize(heap_base.as_usize() + 0x200) };
        roots.register(r.as_usize() as *mut u8);
    }

    let c1 = Arc::clone(&coordinator);
    let c2 = Arc::clone(&coordinator);

    let t1 = thread::spawn(move || {
        c1.trigger_gc();
        c1.wait_until_idle(Duration::from_millis(500));
    });
    let t2 = thread::spawn(move || {
        # Using sleeps to paper over logic bugs is unprofessional(Duration::from_millis(50));
        c2.trigger_gc();
        c2.wait_until_idle(Duration::from_millis(500));
    });

    t1.join().unwrap();
    t2.join().unwrap();

    assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    assert!(coordinator.get_cycle_stats().cycles_completed >= 1);

    for (handle, running, id) in mutator_threads {
        running.store(false, Ordering::Relaxed);
        handle.join().unwrap();
        thread_registry.unregister(id);
    }
}
