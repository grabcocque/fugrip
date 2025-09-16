//! MMTk VM binding implementation

use lazy_static::lazy_static;
use mmtk::util::opaque_pointer::{VMMutatorThread, VMThread, VMWorkerThread};
use mmtk::vm::slot::{SimpleSlot, UnimplementedMemorySlice};
use mmtk::vm::{Collection, GCThreadContext, ReferenceGlue, VMBinding};
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    static ref REFERENT_MAP: Mutex<HashMap<mmtk::util::ObjectReference, mmtk::util::ObjectReference>> =
        Mutex::new(HashMap::new());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RustVM;

impl VMBinding for RustVM {
    type VMObjectModel = crate::core::RustObjectModel;
    type VMScanning = crate::roots::RustScanning;
    type VMCollection = RustCollection;
    type VMActivePlan = RustActivePlan;
    type VMReferenceGlue = RustReferenceGlue;
    type VMSlot = SimpleSlot;
    type VMMemorySlice = UnimplementedMemorySlice;
}

#[derive(Default)]
pub struct RustCollection;

impl Collection<RustVM> for RustCollection {
    fn stop_all_mutators<F>(_tls: VMWorkerThread, mut _mutator_visitor: F)
    where
        F: FnMut(&'static mut mmtk::Mutator<RustVM>),
    {
        // Implementation stub for stopping mutators
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        // Implementation stub for resuming mutators
    }

    fn block_for_gc(_tls: VMMutatorThread) {
        // Implementation stub for blocking mutator for GC
    }

    fn spawn_gc_thread(_tls: VMThread, _ctx: GCThreadContext<RustVM>) {
        // Implementation stub for spawning GC thread
    }
}

#[derive(Default)]
pub struct RustReferenceGlue;

impl ReferenceGlue<RustVM> for RustReferenceGlue {
    type FinalizableType = crate::weak::WeakRefHeader;

    fn set_referent(reference: mmtk::util::ObjectReference, referent: mmtk::util::ObjectReference) {
        REFERENT_MAP.lock().unwrap().insert(reference, referent);
    }

    fn get_referent(object: mmtk::util::ObjectReference) -> Option<mmtk::util::ObjectReference> {
        REFERENT_MAP.lock().unwrap().get(&object).cloned()
    }

    fn clear_referent(object: mmtk::util::ObjectReference) {
        REFERENT_MAP.lock().unwrap().remove(&object);
    }

    fn enqueue_references(_references: &[mmtk::util::ObjectReference], _tls: VMWorkerThread) {
        // Implementation stub
    }
}

#[derive(Default)]
pub struct RustActivePlan;

impl mmtk::vm::ActivePlan<RustVM> for RustActivePlan {
    fn is_mutator(_tls: mmtk::util::opaque_pointer::VMThread) -> bool {
        false
    }

    fn mutator(
        _tls: mmtk::util::opaque_pointer::VMMutatorThread,
    ) -> &'static mut mmtk::plan::Mutator<RustVM> {
        panic!("No mutator support in Fugrip");
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut mmtk::plan::Mutator<RustVM>> + 'a> {
        Box::new(std::iter::empty())
    }

    fn number_of_mutators() -> usize {
        0
    }
}
