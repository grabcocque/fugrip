//! Root scanning hooks exposed to the collector.

use mmtk::{
    util::{ObjectReference, opaque_pointer::VMWorkerThread},
    vm::{RootsWorkFactory, Scanning, SlotVisitor, VMBinding},
};

use crate::binding::RustVM;
use crate::core::{ObjectModel, RustObjectModel};
use crate::thread::{MutatorThread, ThreadRegistry};
use parking_lot::Mutex;
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
#[derive(Default)]
pub struct GlobalRoots {
    handles: Vec<usize>, // Store addresses as usize for thread safety
}

impl GlobalRoots {
    pub fn register(&mut self, handle: *mut u8) {
        self.handles.push(handle as usize);
    }

    pub fn iter(&self) -> impl Iterator<Item = *mut u8> + '_ {
        self.handles.iter().map(|&addr| addr as *mut u8)
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

impl Scanning<RustVM> for RustScanning {
    fn scan_object<SV: SlotVisitor<<RustVM as VMBinding>::VMSlot>>(
        _tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut SV,
    ) {
        let object_ptr = object.to_raw_address().to_mut_ptr();

        // Try to get type information from the vtable to do proper field tracing
        // For now, fall back to scanning the entire object body
        let header = RustObjectModel::header(object_ptr);

        // Calculate the start of the object body (after the header)
        let body_start =
            unsafe { object_ptr.add(std::mem::size_of::<crate::core::ObjectHeader>()) };

        // For proper field tracing, we would use the vtable to get type information
        // and only visit actual reference fields. For now, we'll scan conservatively.
        let body_size = header.body_size;
        let num_potential_slots = body_size / std::mem::size_of::<*const u8>();

        for i in 0..num_potential_slots {
            let slot_ptr = unsafe { body_start.add(i * std::mem::size_of::<*const u8>()) };
            let potential_ref = unsafe { slot_ptr.cast::<*const u8>().read() };

            // Check if this looks like a valid pointer within the managed heap
            if !potential_ref.is_null() {
                let slot_address = mmtk::util::Address::from_ptr(potential_ref);
                if ObjectReference::from_raw_address(slot_address).is_some() {
                    // This looks like a valid object reference, create a slot for it
                    let vm_slot = mmtk::vm::slot::SimpleSlot::from_address(
                        mmtk::util::Address::from_mut_ptr(slot_ptr),
                    );
                    slot_visitor.visit_slot(vm_slot);
                }
            }
        }
    }

    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        _mutator: &'static mut mmtk::Mutator<RustVM>,
        mut factory: impl RootsWorkFactory<<RustVM as VMBinding>::VMSlot>,
    ) {
        // In a real implementation, this would scan the thread's stack for roots
        // For now, we'll use a thread-local registry of stack roots

        // Get thread-local stack roots
        thread_local! {
            static THREAD_STACK_ROOTS: std::cell::RefCell<StackRoots> = std::cell::RefCell::new(StackRoots::default());
        }

        THREAD_STACK_ROOTS.with(|roots| {
            let roots = roots.borrow();
            let mut slots = Vec::new();

            // Process each registered stack root
            for &frame_addr in &roots.frames {
                let frame_ptr = frame_addr as *mut u8;
                if !frame_ptr.is_null() {
                    let slot = mmtk::vm::slot::SimpleSlot::from_address(
                        mmtk::util::Address::from_mut_ptr(frame_ptr),
                    );
                    slots.push(slot);
                }
            }

            if !slots.is_empty() {
                factory.create_process_roots_work(slots);
            }
        });
    }

    fn scan_vm_specific_roots(
        _tls: VMWorkerThread,
        mut factory: impl RootsWorkFactory<<RustVM as VMBinding>::VMSlot>,
    ) {
        // Use a thread-safe global roots registry
        static GLOBAL_ROOTS: std::sync::OnceLock<Mutex<GlobalRoots>> = std::sync::OnceLock::new();

        let global_roots = GLOBAL_ROOTS.get_or_init(|| Mutex::new(GlobalRoots::default()));
        let roots = global_roots.lock();

        // Process all global root handles
        let slots: Vec<mmtk::vm::slot::SimpleSlot> = roots
            .iter()
            .filter(|&ptr| !ptr.is_null())
            .map(|ptr| {
                mmtk::vm::slot::SimpleSlot::from_address(mmtk::util::Address::from_mut_ptr(ptr))
            })
            .collect();

        if !slots.is_empty() {
            factory.create_process_roots_work(slots);
        }
    }

    fn notify_initial_thread_scan_complete(partial_scan: bool, _tls: VMWorkerThread) {
        let coordinator = {
            let manager = crate::binding::FUGC_PLAN_MANAGER
                .get_or_init(|| parking_lot::Mutex::new(crate::plan::FugcPlanManager::new()))
                .lock();
            Arc::clone(manager.get_fugc_coordinator())
        };

        // FUGC coordinator handles root scanning through its 8-step protocol
        let _ = partial_scan;
        coordinator.trigger_gc();
    }

    fn supports_return_barrier() -> bool {
        false
    }

    fn prepare_for_roots_re_scanning() {
        let coordinator = {
            let manager = crate::binding::FUGC_PLAN_MANAGER
                .get_or_init(|| parking_lot::Mutex::new(crate::plan::FugcPlanManager::new()))
                .lock();
            Arc::clone(manager.get_fugc_coordinator())
        };
        // FUGC coordinator resets are handled internally during collection cycles
        // No explicit reset needed here
        let _ = coordinator; // Avoid unused variable warning
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
        let mut global_roots = GlobalRoots::default();

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
        use std::sync::Mutex;
        let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

        // Test concurrent access
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let roots = Arc::clone(&global_roots);
                std::thread::spawn(move || {
                    for j in 0..10 {
                        let ptr = ((i * 100 + j) * 0x1000) as *mut u8;
                        roots.lock().unwrap().register(ptr);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 40 total registrations
        assert_eq!(global_roots.lock().unwrap().iter().count(), 40);
    }

    #[test]
    fn test_global_roots_edge_cases() {
        let mut global_roots = GlobalRoots::default();

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
        let registry = Arc::new(ThreadRegistry::new());
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
        let mut global_roots = GlobalRoots::default();

        // Test that pointers can be safely stored and retrieved
        let test_addrs = [
            0x1000 as *mut u8,
            0x10000 as *mut u8,
            0x100000 as *mut u8,
        ];

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
