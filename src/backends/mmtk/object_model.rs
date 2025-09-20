//! MMTk ObjectModel implementation for RustVM
//!
//! This file contains the MMTk-specific trait implementations that bridge
//! between our frontend object model and MMTk's expectations.

use mmtk::vm::ObjectModel as MMTkObjectModel;
use mmtk::util::{ObjectReference, Address};
use mmtk::util::copy::{CopySemantics, GCWorkerCopyContext};
use mmtk::vm::{VMGlobalLogBitSpec, VMLocalForwardingBitsSpec, VMLocalForwardingPointerSpec,
              VMLocalLOSMarkNurserySpec, VMLocalMarkBitSpec};

use crate::core::{ObjectHeader, ObjectFlags, RustObjectModel};
use super::RustVM;

impl MMTkObjectModel<RustVM> for RustObjectModel {
    const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec = VMGlobalLogBitSpec::side_first();

    const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec =
        VMLocalForwardingPointerSpec::in_header(0);

    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec =
        VMLocalForwardingBitsSpec::in_header(0);

    const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec =
        VMLocalLOSMarkNurserySpec::in_header(0);

    const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec =
        VMLocalMarkBitSpec::in_header(ObjectFlags::MARKED.bits() as usize);

    fn copy(
        from: ObjectReference,
        semantics: CopySemantics,
        copy_context: &mut GCWorkerCopyContext<RustVM>,
    ) -> ObjectReference {
        let bytes = Self::get_current_size(from);
        let dst = copy_context.alloc_copy(from, bytes, 8, 0, semantics);
        let src = from.to_raw_address();

        unsafe {
            std::ptr::copy_nonoverlapping(src.to_ptr::<u8>(), dst.to_mut_ptr::<u8>(), bytes);
        }

        dst
    }

    fn copy_to(from: ObjectReference, to: ObjectReference, region: Address) -> Address {
        let _ = region;
        let bytes = Self::get_current_size(from);
        let src = from.to_raw_address();
        let dst = to.to_raw_address();

        unsafe {
            std::ptr::copy_nonoverlapping(src.to_ptr::<u8>(), dst.to_mut_ptr::<u8>(), bytes);
        }

        dst + bytes
    }

    fn get_reference_when_copied_to(from: ObjectReference, to: Address) -> ObjectReference {
        ObjectReference::from_raw_address(to)
    }

    fn get_current_size(object: ObjectReference) -> usize {
        // Delegate to our frontend object model
        crate::core::RustObjectModel::size(object.to_raw_address().to_mut_ptr())
    }

    fn get_size_when_copied(object: ObjectReference) -> usize {
        Self::get_current_size(object)
    }

    fn get_align_when_copied(_object: ObjectReference) -> usize {
        8
    }

    fn get_align_offset_when_copied(_object: ObjectReference) -> usize {
        0
    }

    fn get_type_descriptor(reference: ObjectReference) -> &'static [i8] {
        // Return a default type descriptor for now
        b"RustObject\0".as_ptr() as *const i8
    }

    fn ref_to_object_start(object: ObjectReference) -> Address {
        object.to_raw_address()
    }

    fn ref_to_header(object: ObjectReference) -> Address {
        object.to_raw_address()
    }


    fn dump_object(object: ObjectReference) {
        println!("Object at {:?}", object.to_raw_address());
        unsafe {
            let header_ptr = object.to_raw_address().to_ptr::<ObjectHeader>();
            if !header_ptr.is_null() {
                let header = std::ptr::read(header_ptr);
                println!("  Header: flags={:?}", header.flags);
            }
        }
    }
}