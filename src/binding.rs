//! VM binding skeleton that wires Fugrip into MMTk's trait surface.

use crate::thread::ThreadRegistry;
use crate::core::RustObjectModel;

use once_cell::sync::Lazy;
use std::collections::HashMap;

use mmtk::{
    scheduler::GCWorker,
    util::opaque_pointer::{VMMutatorThread, VMThread, VMWorkerThread},
    util::ObjectReference,
    vm::{ActivePlan, Collection, ReferenceGlue, VMBinding},
    Mutator,
};

/// Marker struct representing the Fugrip VM when interfacing with MMTk.
#[derive(Default, Debug)]
pub struct RustVM;

impl RustVM {
    pub fn registry() -> &'static ThreadRegistry {
        ThreadRegistry::global()
    }
}

impl VMBinding for RustVM {
    type VMSlot = mmtk::vm::slot::SimpleSlot;
    type VMMemorySlice = mmtk::vm::slot::UnimplementedMemorySlice;

    type VMActivePlan = RustActivePlan;
    type VMCollection = RustCollection;
    type VMObjectModel = crate::core::RustObjectModel;
    type VMReferenceGlue = RustReferenceGlue;
    type VMScanning = crate::roots::RustScanning;
}

/// Active plan implementation backing mutator management.
pub struct RustActivePlan;

struct MutatorPtr(*mut Mutator<RustVM>);

unsafe impl Send for MutatorPtr {}
unsafe impl Sync for MutatorPtr {}

static MUTATOR_REGISTRY: Lazy<std::sync::Mutex<HashMap<usize, MutatorPtr>>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

impl RustActivePlan {
    pub fn register_mutator(tls: VMMutatorThread, mutator: &'static mut Mutator<RustVM>) {
        let thread_id = unsafe { std::mem::transmute::<VMMutatorThread, usize>(tls) };
        MUTATOR_REGISTRY
            .lock()
            .unwrap()
            .insert(thread_id, MutatorPtr(mutator as *mut Mutator<RustVM>));
    }

    pub fn unregister_mutator(tls: VMMutatorThread) {
        let thread_id = unsafe { std::mem::transmute::<VMMutatorThread, usize>(tls) };
        MUTATOR_REGISTRY.lock().unwrap().remove(&thread_id);
    }
}

impl ActivePlan<RustVM> for RustActivePlan {
    fn is_mutator(tls: VMThread) -> bool {
        let thread_id = unsafe { std::mem::transmute::<VMThread, usize>(tls) };
        MUTATOR_REGISTRY
            .lock()
            .unwrap()
            .contains_key(&thread_id)
    }

    fn mutator(tls: VMMutatorThread) -> &'static mut Mutator<RustVM> {
        let thread_id = unsafe { std::mem::transmute::<VMMutatorThread, usize>(tls) };
        let registry = MUTATOR_REGISTRY.lock().unwrap();
        let MutatorPtr(ptr) = registry
            .get(&thread_id)
            .expect("mutator not registered for thread");
        unsafe { &mut **ptr }
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut Mutator<RustVM>> + 'a> {
        let pointers: Vec<*mut Mutator<RustVM>> = MUTATOR_REGISTRY
            .lock()
            .unwrap()
            .values()
            .map(|MutatorPtr(ptr)| *ptr)
            .collect();

        Box::new(pointers.into_iter().map(|ptr| unsafe { &mut *ptr }))
    }

    fn number_of_mutators() -> usize {
        MUTATOR_REGISTRY.lock().unwrap().len()
    }

    fn vm_trace_object<Q: mmtk::ObjectQueue>(
        _queue: &mut Q,
        object: ObjectReference,
        _worker: &mut GCWorker<RustVM>,
    ) -> ObjectReference {
        object
    }
}

/// VM-side hooks for coordinating collection.
pub struct RustCollection;

impl Collection<RustVM> for RustCollection {
    fn stop_all_mutators<F>(_: VMWorkerThread, mut mutator_visitor: F)
    where
        F: FnMut(&'static mut Mutator<RustVM>),
    {
        let mutators = ThreadRegistry::global().iter();
        for mutator in &mutators {
            mutator.request_safepoint();
        }
        for mutator in &mutators {
            mutator.poll_safepoint();
        }

        for mutator in RustActivePlan::mutators() {
            mutator_visitor(mutator);
        }
    }

    fn resume_mutators(_: VMWorkerThread) {
        for mutator in ThreadRegistry::global().iter() {
            mutator.clear_safepoint();
        }
    }

    fn block_for_gc(_: VMMutatorThread) {
        std::thread::yield_now();
    }

    fn spawn_gc_thread(tls: VMThread, ctx: mmtk::vm::GCThreadContext<RustVM>) {
        match ctx {
            mmtk::vm::GCThreadContext::Worker(worker) => {
                std::thread::spawn(move || {
                    let worker_tls = VMWorkerThread(tls);
                    let mmtk = worker.mmtk;
                    worker.run(worker_tls, mmtk);
                });
            }
        }
    }
}

/// Weak-reference and finalization glue.
pub struct RustReferenceGlue;

impl ReferenceGlue<RustVM> for RustReferenceGlue {
    type FinalizableType = ObjectReference;

    fn clear_referent(reference: ObjectReference) {
        let object_model = RustObjectModel;
        if let Some(weak_ref_header) = object_model.get_weak_ref_header(reference) {
            unsafe {
                *weak_ref_header = std::ptr::null_mut();
            }
        }
    }

    fn get_referent(object: ObjectReference) -> Option<ObjectReference> {
        let object_model = RustObjectModel;
        if let Some(weak_ref_header) = object_model.get_weak_ref_header(object) {
            let target_ptr = unsafe { *weak_ref_header };
            if target_ptr.is_null() {
                None
            } else {
                let address = mmtk::util::Address::from_mut_ptr(target_ptr);
                ObjectReference::from_raw_address(address)
            }
        } else {
            None
        }
    }

    fn set_referent(reference: ObjectReference, referent: ObjectReference) {
        let object_model = RustObjectModel;
        if let Some(weak_ref_header) = object_model.get_weak_ref_header(reference) {
            unsafe {
                *weak_ref_header = referent.to_raw_address().to_mut_ptr();
            }
        }
    }

    fn enqueue_references(references: &[ObjectReference], _tls: VMWorkerThread) {
        static REFERENCE_QUEUE: Lazy<std::sync::Mutex<Vec<ObjectReference>>> =
            Lazy::new(|| std::sync::Mutex::new(Vec::new()));

        let mut queue = REFERENCE_QUEUE.lock().unwrap();
        for &reference in references {
            queue.push(reference);
        }
    }
}
