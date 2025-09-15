//! MMTk plan implementation for the Fugrip runtime.

use crate::{collector_phases::CollectorPhase, thread::ThreadRegistry};

use mmtk::{
    util::opaque_pointer::VMWorkerThread,
    vm::VMBinding,
};

pub struct RustVMPlan<VM: VMBinding> {
    current_phase: CollectorPhase,
    _phantom: std::marker::PhantomData<VM>,
}

impl<VM: VMBinding> RustVMPlan<VM> {
    pub fn new() -> Self {
        Self {
            current_phase: CollectorPhase::Idle,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn schedule_collection(&mut self) {
        self.current_phase = CollectorPhase::Prepare;
    }

    pub fn current_phase(&self) -> CollectorPhase {
        self.current_phase
    }
}

// For now, we'll delegate to the RustActivePlan implementation
// In a full implementation, RustVMPlan would directly implement Plan trait from MMTk

/// Collection coordination hooks the binding exposes to MMTk.
#[derive(Clone, Debug)]
pub struct RustCollection {
    registry: ThreadRegistry,
}

impl RustCollection {
    pub fn new(registry: ThreadRegistry) -> Self {
        Self { registry }
    }
}

impl Default for RustCollection {
    fn default() -> Self {
        Self::new(ThreadRegistry::new())
    }
}

impl<VM: VMBinding> mmtk::vm::Collection<VM> for RustCollection {
    fn stop_all_mutators<F>(_tls: VMWorkerThread, _mutator_visitor: F)
    where
        F: FnMut(&'static mut mmtk::Mutator<VM>),
    {
        // Stop all mutators implementation
        todo!("Implement stop_all_mutators with mutator visitor");
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        // Resume mutators implementation
        todo!("Implement resume_mutators");
    }

    fn block_for_gc(_tls: mmtk::util::opaque_pointer::VMMutatorThread) {
        // Block for GC implementation
        todo!("Implement block_for_gc");
    }

    fn spawn_gc_thread(
        _tls: mmtk::util::opaque_pointer::VMThread,
        _gc_controller: mmtk::vm::GCThreadContext<VM>,
    ) {
        // Spawn GC thread implementation
        todo!("Implement spawn_gc_thread");
    }
}
