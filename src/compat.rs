//! Compatibility layer that switches between MMTk and native types
//!
//! This module provides type aliases and re-exports that automatically
//! switch between MMTk types and our native replacements based on feature flags.

#[cfg(feature = "use_mmtk")]
pub use mmtk::util::{Address, ObjectReference};

#[cfg(not(feature = "use_mmtk"))]
pub use crate::types::{Address, ObjectReference};

#[cfg(feature = "use_mmtk")]
pub use mmtk::util::constants::*;

#[cfg(not(feature = "use_mmtk"))]
pub use crate::types::constants::*;

// Re-export constants module for easier access
pub mod constants {
    #[cfg(feature = "use_mmtk")]
    pub use mmtk::util::constants::*;

    #[cfg(not(feature = "use_mmtk"))]
    pub use crate::types::constants::*;
}

#[cfg(feature = "use_mmtk")]
pub use mmtk::vm::slot::SimpleSlot;

#[cfg(not(feature = "use_mmtk"))]
pub use crate::types::slot::SimpleSlot;

// Re-export allocation semantics
#[cfg(feature = "use_mmtk")]
pub use mmtk::AllocationSemantics;

#[cfg(not(feature = "use_mmtk"))]
#[derive(Debug, Clone, Copy)]
pub enum AllocationSemantics {
    Default,
    Immortal,
    Los,
}

// Re-export ObjectModel and VM spec types for MMTk compatibility
#[cfg(feature = "use_mmtk")]
pub use mmtk::vm::{
    ObjectModel, VMGlobalLogBitSpec, VMLocalForwardingBitsSpec,
    VMLocalForwardingPointerSpec, VMLocalLOSMarkNurserySpec, VMLocalMarkBitSpec,
};

#[cfg(not(feature = "use_mmtk"))]
pub use crate::types::{
    ObjectModel, VMGlobalLogBitSpec, VMLocalForwardingBitsSpec,
    VMLocalForwardingPointerSpec, VMLocalLOSMarkNurserySpec, VMLocalMarkBitSpec,
};

// Re-export VM binding types if using MMTk
#[cfg(feature = "use_mmtk")]
pub mod vm {
    // Re-export the MMTK struct itself
    pub use mmtk::MMTK;

    pub use mmtk::vm::{
        ActivePlan, Collection, Finalizable, GCThreadContext, ObjectModel, ReferenceGlue,
        RootsWorkFactory, Scanning, SlotVisitor, VMBinding,
    };

    pub mod slot {
        pub use mmtk::vm::slot::*;
    }

    pub mod opaque_pointer {
        pub use mmtk::util::opaque_pointer::*;
    }
}

// Provide stubs if not using MMTk
#[cfg(not(feature = "use_mmtk"))]
pub mod vm {
    use crate::types::{Address, ObjectReference};

    // Stub VMBinding trait
    pub trait VMBinding: 'static + Send + Sync + Sized {
        type VMObjectModel: ObjectModel<Self>;
        type VMCollection: Collection<Self>;
        type VMReferenceGlue: ReferenceGlue<Self>;
    }

    // Stub Collection trait
    pub trait Collection<VM: VMBinding>: 'static + Send + Sync {
        fn stop_all_mutators<F>(_tls: VMWorkerThread, _f: F)
        where
            F: FnOnce(),
        {
        }

        fn resume_mutators(_tls: VMWorkerThread) {}

        fn block_for_gc(_tls: VMMutatorThread) {}

        fn spawn_gc_thread(_tls: VMThread, _ctx: GCThreadContext<VM>) {}

        fn prepare_mutator<T>(_tls: VMMutatorThread, _m: &T) {}

        fn vm_live_bytes() -> usize {
            0
        }
    }

    // Stub ReferenceGlue trait
    pub trait ReferenceGlue<VM: VMBinding>: 'static + Send + Sync {
        fn set_referent(_reference: ObjectReference, _referent: ObjectReference) {}
        fn get_referent(_reference: ObjectReference) -> ObjectReference {
            ObjectReference::NULL
        }
        fn enqueue_references(_references: &[ObjectReference], _tls: VMWorkerThread) {}
    }

    // Stub ObjectModel trait
    pub trait ObjectModel<VM: VMBinding>: 'static + Send + Sync {
        const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec = VMGlobalLogBitSpec { high_bit_offset: 0 };
        const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec =
            VMLocalForwardingPointerSpec {};
        const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec = VMLocalForwardingBitsSpec {};
        const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec = VMLocalMarkBitSpec {};
        const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec = VMLocalLOSMarkNurserySpec {};

        fn copy(
            _from: ObjectReference,
            _semantics: CopySemantics,
            _copy_context: &mut impl CopyContext,
        ) -> ObjectReference {
            ObjectReference::NULL
        }

        fn copy_to(_from: ObjectReference, _to: ObjectReference, _region: Address) -> Address {
            Address::ZERO
        }

        fn get_current_size(_object: ObjectReference) -> usize {
            0
        }

        fn get_size_when_copied(_object: ObjectReference) -> usize {
            0
        }

        fn get_align_when_copied(_object: ObjectReference) -> usize {
            8
        }

        fn get_align_offset_when_copied(_object: ObjectReference) -> usize {
            0
        }

        fn get_reference_when_copied_to(_from: ObjectReference, _to: Address) -> ObjectReference {
            ObjectReference::NULL
        }

        fn get_type_descriptor(_reference: ObjectReference) -> &'static [i8] {
            &[]
        }

        fn object_start_ref(_object: ObjectReference) -> Address {
            Address::ZERO
        }

        fn ref_to_address(_object: ObjectReference) -> Address {
            Address::ZERO
        }

        fn ref_to_object_start(_object: ObjectReference) -> Address {
            Address::ZERO
        }

        fn ref_to_header(_object: ObjectReference) -> Address {
            Address::ZERO
        }

        fn address_to_ref(_addr: Address) -> ObjectReference {
            ObjectReference::NULL
        }

        fn dump_object(_object: ObjectReference) {}
    }

    // Stub ActivePlan trait
    pub trait ActivePlan<VM: VMBinding>: 'static + Send + Sync {
        fn is_mutator(_tls: VMThread) -> bool {
            false
        }
        fn mutator(_tls: VMMutatorThread) -> &'static mut Mutator<VM> {
            panic!("Not implemented")
        }
        fn reset_mutator_iterator() {}
        fn get_next_mutator() -> Option<&'static mut Mutator<VM>> {
            None
        }
        fn number_of_mutators() -> usize {
            0
        }
    }

    // Stub Finalizable trait
    pub trait Finalizable: 'static + Send + Sync {
        fn get_reference(&self) -> ObjectReference;
        fn set_reference(&mut self, _reference: ObjectReference);
        fn keep_alive<VM: VMBinding>(&mut self, _trace: &mut impl ProcessEdgesWork<VM = VM>);
    }

    // Stub GC thread context
    pub enum GCThreadContext<VM: VMBinding> {
        Controller(Box<dyn GCController<VM>>),
        Worker(Box<dyn GCWorker<VM>>),
    }

    pub trait GCController<VM: VMBinding>: 'static + Send + Sync {
        fn run(&mut self, _tls: VMWorkerThread, _mmtk: &'static MMTK<VM>);
    }

    pub trait GCWorker<VM: VMBinding>: 'static + Send + Sync {
        fn run(&mut self, _tls: VMWorkerThread, _mmtk: &'static MMTK<VM>);
    }

    // Stub opaque pointer types
    pub mod opaque_pointer {
        #[derive(Copy, Clone, Debug)]
        pub struct OpaquePointer(pub usize);

        impl OpaquePointer {
            pub fn from_address(addr: crate::types::Address) -> Self {
                OpaquePointer(addr.as_usize())
            }
        }

        #[derive(Copy, Clone, Debug)]
        pub struct VMThread(pub OpaquePointer);

        #[derive(Copy, Clone, Debug)]
        pub struct VMMutatorThread(pub VMThread);

        #[derive(Copy, Clone, Debug)]
        pub struct VMWorkerThread(pub VMThread);
    }

    pub use opaque_pointer::*;

    // Stub types referenced above
    pub struct VMGlobalLogBitSpec {
        pub high_bit_offset: usize,
    }
    pub struct VMLocalForwardingPointerSpec {}
    pub struct VMLocalForwardingBitsSpec {}
    pub struct VMLocalMarkBitSpec {}
    pub struct VMLocalLOSMarkNurserySpec {}

    pub enum CopySemantics {
        DefaultCopy,
        Mature,
        PromoteToMature,
    }

    pub trait CopyContext {}
    pub trait ProcessEdgesWork {
        type VM: VMBinding;
    }

    // Stub MMTK and Mutator
    pub struct MMTK<VM: VMBinding> {
        _phantom: std::marker::PhantomData<VM>,
    }

    pub struct Mutator<VM: VMBinding> {
        _phantom: std::marker::PhantomData<VM>,
    }
}
