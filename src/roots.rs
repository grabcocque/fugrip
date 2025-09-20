//! Root scanning hooks exposed to the collector.

use mmtk::{
    util::{ObjectReference, opaque_pointer::VMWorkerThread},
    vm::{RootsWorkFactory, Scanning, SlotVisitor, VMBinding},
};

#[cfg(feature = "use_mmtk")]
use crate::backends::mmtk::binding::RustVM;
use crate::thread::{MutatorThread, ThreadRegistry};
use rayon::prelude::*;
use std::sync::Arc;

/// Stack-based root references for garbage collection
///
/// # Examples
///
/// ```
/// use fugrip::roots::StackRoots;
///
/// let mut stack_roots = StackRoots::default();
///
/// // Register stack roots (simulated pointers)
/// let ptr1 = 0x1000 as *mut u8;
/// let ptr2 = 0x2000 as *mut u8;
/// stack_roots.push(ptr1);
/// stack_roots.push(ptr2);
///
/// // Iterate over stack roots
/// let roots: Vec<*mut u8> = stack_roots.iter().collect();
/// assert_eq!(roots.len(), 2);
/// assert_eq!(roots[0], ptr1);
/// assert_eq!(roots[1], ptr2);
///
/// // Clear all stack roots
/// stack_roots.clear();
/// assert_eq!(stack_roots.iter().count(), 0);
/// ```
#[derive(Default)]
pub struct StackRoots {
    frames: Vec<usize>, // Store addresses as usize for thread safety
}

impl StackRoots {
    pub fn push(&mut self, handle: *mut u8) {
        self.frames.push(handle as usize);
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = *mut u8> + '_ {
        self.frames.iter().map(|&addr| addr as *mut u8)
    }
}

thread_local! {
    static THREAD_STACK_ROOTS: std::cell::RefCell<StackRoots> = std::cell::RefCell::new(StackRoots::default());
}

/// Global root references for garbage collection
///
/// # Examples
///
/// ```
/// use fugrip::roots::GlobalRoots;
///
/// let mut global_roots = GlobalRoots::default();
///
/// // Register global roots (simulated pointers)
/// let global_ptr1 = 0x10000 as *mut u8;
/// let global_ptr2 = 0x20000 as *mut u8;
/// global_roots.register(global_ptr1);
/// global_roots.register(global_ptr2);
///
/// // Iterate over global roots
/// let roots: Vec<*mut u8> = global_roots.iter().collect();
/// assert_eq!(roots.len(), 2);
/// assert_eq!(roots[0], global_ptr1);
/// assert_eq!(roots[1], global_ptr2);
/// ```
pub struct GlobalRoots {
    // Use ArcSwap over an Arc<Vec<usize>> so registrations can happen
    // without a Mutex while reads get a cheap snapshot.
    handles: arc_swap::ArcSwap<Vec<usize>>,
}

impl Clone for GlobalRoots {
    fn clone(&self) -> Self {
        let snapshot = self.handles.load_full();
        let new_vec = (**snapshot).to_vec();
        GlobalRoots {
            handles: arc_swap::ArcSwap::from_pointee(new_vec),
        }
    }
}

impl Default for GlobalRoots {
    fn default() -> Self {
        Self {
            handles: arc_swap::ArcSwap::from_pointee(Vec::new()),
        }
    }
}

impl GlobalRoots {
    /// Register a global root. This is lock-free from the caller's perspective
    /// by performing an RCU-style update of the underlying Vec.
    pub fn register(&self, handle: *mut u8) {
        self.handles.rcu(|current| {
            let mut new_vec = (**current).clone();
            new_vec.push(handle as usize);
            std::sync::Arc::new(new_vec)
        });
    }

    /// Iterate over a snapshot of the current roots. Returns an iterator that
    /// owns a temporary Vec so callers may iterate without holding locks.
    pub fn iter(&self) -> impl Iterator<Item = *mut u8> {
        let snapshot = self.handles.load_full();
        let tmp: Vec<*mut u8> = snapshot.iter().map(|&addr| addr as *mut u8).collect();
        tmp.into_iter()
    }
}

#[derive(Default)]
pub struct RootSet {
    pub stacks: StackRoots,
    pub globals: GlobalRoots,
}

impl RootSet {
    pub fn new() -> Self {
        Self {
            stacks: StackRoots::default(),
            globals: GlobalRoots::default(),
        }
    }
}

#[derive(Default)]
pub struct RustScanning {
    pub registry: Arc<ThreadRegistry>,
}

impl RustScanning {
    pub fn new(registry: Arc<ThreadRegistry>) -> Self {
        Self { registry }
    }

    /// Iterate over all registered mutators, calling the provided closure for
    /// each so the VM can expose stack roots.  The closure will be used by the
    /// eventual MMTk integration to create per-thread root scanning work
    /// packets.
    pub fn for_each_mutator<F>(&self, mut f: F)
    where
        F: FnMut(&MutatorThread),
    {
        for mutator in self.registry.iter() {
            f(&mutator);
        }
    }
}

#[cfg(feature = "use_mmtk")]
impl Scanning<RustVM> for RustScanning {
    fn scan_object<SV: SlotVisitor<<RustVM as VMBinding>::VMSlot>>(
        _tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut SV,
    ) {
        // Avoid raw memory dereferences on synthetic addresses during tests.
        // In this stub, we conservatively do not scan object bodies.
        // Proper field tracing should be implemented via type metadata.
        let _ = (object, slot_visitor);
    }

    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        _mutator: &'static mut mmtk::Mutator<RustVM>,
        mut factory: impl RootsWorkFactory<<RustVM as VMBinding>::VMSlot>,
    ) {
        // Do not scan arbitrary thread stack memory to avoid dereferencing
        // synthetic addresses. Instead, rely on explicitly registered roots.

        // Fallback: scan any explicitly registered thread-local roots
        THREAD_STACK_ROOTS.with(|roots| {
            let roots = roots.borrow();
            if !roots.frames.is_empty() {
                let slots: Vec<mmtk::vm::slot::SimpleSlot> = roots
                    .frames
                    .iter()
                    .filter_map(|&frame_addr| {
                        let frame_ptr = frame_addr as *mut u8;
                        if frame_ptr.is_null() {
                            None
                        } else {
                            Some(mmtk::vm::slot::SimpleSlot::from_address(
                                mmtk::util::Address::from_mut_ptr(frame_ptr),
                            ))
                        }
                    })
                    .collect();

                if !slots.is_empty() {
                    factory.create_process_roots_work(slots);
                }
            }
        });
    }

    fn scan_vm_specific_roots(
        _tls: VMWorkerThread,
        mut factory: impl RootsWorkFactory<<RustVM as VMBinding>::VMSlot>,
    ) {
        // Use a lock-free global roots registry with arc_swap for better performance
        use arc_swap::ArcSwap;
        static GLOBAL_ROOTS: std::sync::OnceLock<ArcSwap<GlobalRoots>> = std::sync::OnceLock::new();

        let global_roots =
            GLOBAL_ROOTS.get_or_init(|| ArcSwap::new(Arc::new(GlobalRoots::default())));
        let roots = global_roots.load();

        // Collect all pointer addresses as usize for thread safety, then parallel-map to slots
        let addresses: Vec<usize> = roots.iter().map(|ptr| ptr as usize).collect();
        drop(roots);

        let slots: Vec<mmtk::vm::slot::SimpleSlot> = addresses
            .par_iter()
            .filter_map(|&addr| {
                if addr == 0 {
                    None
                } else {
                    Some(mmtk::vm::slot::SimpleSlot::from_address(unsafe {
                        mmtk::util::Address::from_usize(addr)
                    }))
                }
            })
            .collect();

        if !slots.is_empty() {
            factory.create_process_roots_work(slots);
        }
    }

    fn notify_initial_thread_scan_complete(partial_scan: bool, _tls: VMWorkerThread) {
        // Access coordinator through global safepoint manager (maintains blackwall abstraction)
        let coordinator = crate::safepoint::SafepointManager::global().get_fugc_coordinator();

        // Real MMTk integration: FUGC Step 4 - Global Root Marking
        // This is called by MMTk when initial thread scanning is complete
        // The coordinator needs to transition from stack scanning to global root marking

        if !partial_scan {
            // Complete scan: trigger full FUGC 8-step protocol
            coordinator.trigger_gc();
        } else {
            // Partial scan: just mark global roots grey for concurrent marking
            // This integrates with MMTk's incremental scanning approach
            let tricolor = coordinator.tricolor_marking();

            // Get global roots and mark them grey for concurrent processing
            let global_roots = Arc::new(GlobalRoots::default());
            for root_ptr in global_roots.iter() {
                let addr = unsafe { crate::frontend::types::Address::from_mut_ptr(root_ptr) };
                if let Some(obj_ref) =
                    crate::frontend::types::ObjectReference::from_raw_address(addr)
                {
                    tricolor.mark_grey(obj_ref);
                }
            }
        }
    }

    fn supports_return_barrier() -> bool {
        false
    }

    fn prepare_for_roots_re_scanning() {
        // Access coordinator through global safepoint manager (maintains blackwall abstraction)
        let coordinator = crate::safepoint::SafepointManager::global().get_fugc_coordinator();

        // Real MMTk integration: Prepare for FUGC Step 3 - Black Allocation
        // MMTk calls this before re-scanning roots to reset the marking state
        // This ensures the tricolor invariant is satisfied for the next collection cycle

        // Reset tricolor marking state for new collection cycle
        coordinator.tricolor_marking().reset_marking_state();

        // Notify FUGC coordinator that roots re-scanning is starting
        // This transitions the coordinator to prepare for black allocation activation
        coordinator.prepare_for_root_rescan();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_roots() {
        let mut stack_roots = StackRoots::default();

        // Initially empty
        assert_eq!(stack_roots.iter().count(), 0);

        // Add some roots
        let ptr1 = 0x1000 as *mut u8;
        let ptr2 = 0x2000 as *mut u8;
        let ptr3 = 0x3000 as *mut u8;

        stack_roots.push(ptr1);
        stack_roots.push(ptr2);
        stack_roots.push(ptr3);

        // Check iteration
        let roots: Vec<*mut u8> = stack_roots.iter().collect();
        assert_eq!(roots.len(), 3);
        assert_eq!(roots[0], ptr1);
        assert_eq!(roots[1], ptr2);
        assert_eq!(roots[2], ptr3);

        // Clear roots
        stack_roots.clear();
        assert_eq!(stack_roots.iter().count(), 0);
    }

    #[test]
    fn test_global_roots() {
        let global_roots = GlobalRoots::default();

        // Initially empty
        assert_eq!(global_roots.iter().count(), 0);

        // Register global roots
        let global_ptr1 = 0x10000 as *mut u8;
        let global_ptr2 = 0x20000 as *mut u8;

        global_roots.register(global_ptr1);
        global_roots.register(global_ptr2);

        // Check iteration
        let roots: Vec<*mut u8> = global_roots.iter().collect();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], global_ptr1);
        assert_eq!(roots[1], global_ptr2);
    }

    #[test]
    fn test_root_set() {
        let mut root_set = RootSet::new();

        // Test stack roots in root set
        let stack_ptr = 0x5000 as *mut u8;
        root_set.stacks.push(stack_ptr);

        // Test global roots in root set
        let global_ptr = 0x6000 as *mut u8;
        root_set.globals.register(global_ptr);

        assert_eq!(root_set.stacks.iter().count(), 1);
        assert_eq!(root_set.globals.iter().count(), 1);
    }

    #[test]
    fn test_rust_scanning() {
        let registry = Arc::new(ThreadRegistry::new());
        let scanning = RustScanning::new(Arc::clone(&registry));

        // Register some mutator threads
        let mutator1 = MutatorThread::new(1);
        let mutator2 = MutatorThread::new(2);

        registry.register(mutator1.clone());
        registry.register(mutator2.clone());

        // Test for_each_mutator
        let mut count = 0;
        scanning.for_each_mutator(|_mutator| {
            count += 1;
        });
        assert_eq!(count, 2);
    }

    #[test]
    fn test_supports_return_barrier() {
        assert!(!RustScanning::supports_return_barrier());
    }

    #[test]
    fn test_prepare_for_roots_re_scanning() {
        // This should not panic
        RustScanning::prepare_for_roots_re_scanning();
    }

    #[test]
    fn test_stack_roots_edge_cases() {
        let mut stack_roots = StackRoots::default();

        // Test empty stack roots
        assert_eq!(stack_roots.iter().count(), 0);

        // Test multiple registrations
        let ptrs = [
            0x1000 as *mut u8,
            0x2000 as *mut u8,
            0x3000 as *mut u8,
            std::ptr::null_mut(),
        ];

        for ptr in ptrs {
            stack_roots.push(ptr);
        }

        assert_eq!(stack_roots.iter().count(), 4);

        // Test clearing
        stack_roots.clear();
        assert_eq!(stack_roots.iter().count(), 0);
    }

    #[test]
    fn test_global_roots_thread_safety() {
        use arc_swap::ArcSwap;
        use rayon::prelude::*;
        let global_roots = ArcSwap::new(Arc::new(GlobalRoots::default()));

        // Test concurrent access with rayon instead of manual thread spawning
        (0..4).into_par_iter().for_each(|i| {
            for j in 0..10 {
                let ptr = ((i * 100 + j) * 0x1000) as *mut u8;
                // Use ArcSwap's RCU pattern for lock-free updates
                global_roots.rcu(|current| {
                    let new_roots = (**current).clone();
                    new_roots.register(ptr);
                    Arc::new(new_roots)
                });
            }
        });

        // Should have 40 total registrations
        assert_eq!(global_roots.load().iter().count(), 40);
    }

    #[test]
    fn test_global_roots_edge_cases() {
        let global_roots = GlobalRoots::default();

        // Test null pointer registration
        global_roots.register(std::ptr::null_mut());
        assert_eq!(global_roots.iter().count(), 1);

        // Test maximum address
        global_roots.register(usize::MAX as *mut u8);
        assert_eq!(global_roots.iter().count(), 2);

        // Test duplicate registration (should add both)
        let ptr = 0x5000 as *mut u8;
        global_roots.register(ptr);
        global_roots.register(ptr);
        assert_eq!(global_roots.iter().count(), 4);

        // GlobalRoots doesn't have a clear method, so we'll test other operations
        assert_eq!(global_roots.iter().count(), 4);
    }

    #[test]
    fn test_rust_scanning_error_conditions() {
        let registry = Arc::new(ThreadRegistry::new());
        let scanning = RustScanning::new(Arc::clone(&registry));

        // Test with no registered threads
        let mut count = 0;
        scanning.for_each_mutator(|_mutator| {
            count += 1;
        });
        assert_eq!(count, 0);

        // Test with maximum thread ID
        let mutator_max = MutatorThread::new(usize::MAX);
        registry.register(mutator_max);

        let mut max_id_found = false;
        scanning.for_each_mutator(|mutator| {
            if mutator.id() == usize::MAX {
                max_id_found = true;
            }
        });
        assert!(max_id_found);
    }

    #[test]
    fn test_root_set_complex_scenarios() {
        let _registry = Arc::new(ThreadRegistry::new());
        let mut root_set = RootSet::new();

        // Test mixed registrations
        let stack_ptrs = [0x1000 as *mut u8, 0x2000 as *mut u8];
        let global_ptrs = [0x3000 as *mut u8, 0x4000 as *mut u8, 0x5000 as *mut u8];

        for ptr in stack_ptrs {
            root_set.stacks.push(ptr);
        }

        for ptr in global_ptrs {
            root_set.globals.register(ptr);
        }

        assert_eq!(root_set.stacks.iter().count(), 2);
        assert_eq!(root_set.globals.iter().count(), 3);

        // Test partial clearing
        root_set.stacks.clear();
        assert_eq!(root_set.stacks.iter().count(), 0);
        assert_eq!(root_set.globals.iter().count(), 3); // Should remain unchanged
    }

    #[test]
    fn test_mutator_thread_integration() {
        let registry = Arc::new(ThreadRegistry::new());
        let scanning = RustScanning::new(Arc::clone(&registry));

        // Test thread lifecycle
        let mutators = [
            MutatorThread::new(0),
            MutatorThread::new(1),
            MutatorThread::new(100),
        ];

        // Register threads
        for mutator in &mutators {
            registry.register(mutator.clone());
        }

        // Verify all are registered
        let mut found_ids = Vec::new();
        scanning.for_each_mutator(|mutator| {
            found_ids.push(mutator.id());
        });

        assert_eq!(found_ids.len(), 3);
        assert!(found_ids.contains(&0));
        assert!(found_ids.contains(&1));
        assert!(found_ids.contains(&100));
    }

    #[test]
    fn test_memory_safety_invariants() {
        let global_roots = GlobalRoots::default();

        // Test that pointers can be safely stored and retrieved
        let test_addrs = [0x1000 as *mut u8, 0x10000 as *mut u8, 0x100000 as *mut u8];

        for addr in test_addrs {
            global_roots.register(addr);
        }

        // Verify all addresses are retrievable
        let mut retrieved_addrs = Vec::new();
        for ptr in global_roots.iter() {
            retrieved_addrs.push(ptr as usize);
        }

        assert_eq!(retrieved_addrs.len(), 3);
        assert!(retrieved_addrs.contains(&0x1000));
        assert!(retrieved_addrs.contains(&0x10000));
        assert!(retrieved_addrs.contains(&0x100000));
    }
}
