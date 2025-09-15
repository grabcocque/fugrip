//! Root scanning hooks exposed to the collector.

use mmtk::{
    util::{ObjectReference, opaque_pointer::VMWorkerThread},
    vm::{RootsWorkFactory, Scanning, SlotVisitor, VMBinding},
};

use crate::thread::{MutatorThread, ThreadRegistry};
use crate::core::{ObjectModel, RustObjectModel};
use crate::binding::RustVM;

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
    pub registry: ThreadRegistry,
}

impl RustScanning {
    pub fn new(registry: ThreadRegistry) -> Self {
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
        let body_start = unsafe { object_ptr.add(std::mem::size_of::<crate::core::ObjectHeader>()) };

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
                    let vm_slot = mmtk::vm::slot::SimpleSlot::from_address(mmtk::util::Address::from_mut_ptr(slot_ptr));
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
                    let slot = mmtk::vm::slot::SimpleSlot::from_address(mmtk::util::Address::from_mut_ptr(frame_ptr));
                    slots.push(slot);
                }
            }

            if !slots.is_empty() {
                factory.create_process_roots_work(slots);
            }
        });
    }

    fn scan_vm_specific_roots(_tls: VMWorkerThread, mut factory: impl RootsWorkFactory<<RustVM as VMBinding>::VMSlot>) {
        // Use a thread-safe global roots registry
        use std::sync::Mutex;
        static GLOBAL_ROOTS: std::sync::OnceLock<Mutex<GlobalRoots>> = std::sync::OnceLock::new();

        let global_roots = GLOBAL_ROOTS.get_or_init(|| Mutex::new(GlobalRoots::default()));
        let roots = global_roots.lock().unwrap();

        // Process all global root handles
        let slots: Vec<mmtk::vm::slot::SimpleSlot> = roots
            .iter()
            .filter(|&ptr| !ptr.is_null())
            .map(|ptr| mmtk::vm::slot::SimpleSlot::from_address(mmtk::util::Address::from_mut_ptr(ptr)))
            .collect();

        if !slots.is_empty() {
            factory.create_process_roots_work(slots);
        }
    }

    fn notify_initial_thread_scan_complete(_partial_scan: bool, _tls: VMWorkerThread) {
        // No-op stub
    }

    fn supports_return_barrier() -> bool {
        false
    }

    fn prepare_for_roots_re_scanning() {
        // No-op stub
    }
}
