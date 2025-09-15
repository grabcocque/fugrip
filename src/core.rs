//! Object model definitions for Fugrip VM objects.

use std::{marker::PhantomData, mem::size_of};

use bitflags::bitflags;

use mmtk::{
    util::ObjectReference,
    vm::{
        ObjectModel as MMTkObjectModel, VMGlobalLogBitSpec, VMLocalForwardingBitsSpec,
        VMLocalForwardingPointerSpec, VMLocalLOSMarkNurserySpec, VMLocalMarkBitSpec,
    },
};

use crate::binding::RustVM;

/// Basic header shared by every managed object.  The layout intentionally keeps
/// frequently accessed metadata in the first word to simplify barrier
/// implementations.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ObjectHeader {
    pub flags: ObjectFlags,
    pub layout_id: LayoutId,
    pub body_size: usize,
    pub vtable: *const (),
}

impl Default for ObjectHeader {
    fn default() -> Self {
        Self {
            flags: ObjectFlags::empty(),
            layout_id: LayoutId::default(),
            body_size: 0,
            vtable: std::ptr::null(),
        }
    }
}

bitflags! {
    #[derive(Default, Debug, Clone, Copy)]
    pub struct ObjectFlags: u16 {
        const MARKED = 0b0001;
        const HAS_WEAK_REFS = 0b0010;
        const PINNED = 0b0100;
    }
}

/// Identifier pointing into the VM's layout/descriptor table.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct LayoutId(pub u16);

/// `Gc<T>` is a typed wrapper around MMTk's `ObjectReference`.
pub struct Gc<T> {
    ptr: *mut u8,
    _marker: PhantomData<T>,
}

impl<T> Copy for Gc<T> {}
impl<T> Clone for Gc<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Default for Gc<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Gc<T> {
    pub const fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            _marker: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr.cast()
    }

    pub fn from_raw(ptr: *mut T) -> Self {
        Self {
            ptr: ptr.cast(),
            _marker: PhantomData,
        }
    }

    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    pub fn to_object_reference(&self) -> ObjectReference {
        unsafe {
            ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_mut_ptr(self.ptr))
        }
    }
}

/// Object model hooks consumed by the collector.
pub trait ObjectModel {
    fn header(object: *mut u8) -> ObjectHeader;
    fn size(object: *mut u8) -> usize {
        Self::header(object).body_size + size_of::<ObjectHeader>()
    }
}

/// Trait implemented by GC-aware types to describe how they link to other
/// managed objects.
pub trait Trace {
    fn trace(&self, visitor: &mut dyn FnMut(*mut u8));
}

/// Extended trait for objects that can describe their reference fields for scanning
pub trait Traceable {
    /// Get the offsets of all reference fields in this object
    fn reference_field_offsets() -> &'static [usize];

    /// Trace all references in this object using the provided visitor
    fn trace_references(&self, visitor: &mut dyn FnMut(ObjectReference));
}

impl<T> Trace for Gc<T> {
    fn trace(&self, visitor: &mut dyn FnMut(*mut u8)) {
        if !self.is_null() {
            visitor(self.ptr);
        }
    }
}

/// Concrete object model that we will expose to MMTk once the metadata layout
/// is finalized.
#[derive(Default)]
pub struct RustObjectModel;

impl ObjectModel for RustObjectModel {
    fn header(object: *mut u8) -> ObjectHeader {
        debug_assert!(!object.is_null(), "object header request on null pointer");
        unsafe { object.cast::<ObjectHeader>().read() }
    }
}

impl RustObjectModel {
    pub fn get_weak_ref_header(&self, object: ObjectReference) -> Option<*mut *mut u8> {
        let object_ptr = object.to_raw_address().to_mut_ptr();
        let header = Self::header(object_ptr);

        if header.flags.contains(ObjectFlags::HAS_WEAK_REFS) {
            let header_ptr = object_ptr.cast::<ObjectHeader>();
            let weak_ref_ptr = unsafe { header_ptr.add(1).cast::<*mut u8>() };
            Some(weak_ref_ptr)
        } else {
            None
        }
    }
}

impl MMTkObjectModel<RustVM> for RustObjectModel {
    const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec = VMGlobalLogBitSpec::side_first();

    const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec =
        VMLocalForwardingPointerSpec::in_header(0);

    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec =
        VMLocalForwardingBitsSpec::in_header(0);

    const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec = VMLocalMarkBitSpec::in_header(0);

    const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec =
        VMLocalLOSMarkNurserySpec::in_header(0);

    const OBJECT_REF_OFFSET_LOWER_BOUND: isize = 0;

    fn copy(
        from: ObjectReference,
        semantics: mmtk::util::copy::CopySemantics,
        copy_context: &mut mmtk::util::copy::GCWorkerCopyContext<RustVM>,
    ) -> ObjectReference {
        let object_size = Self::get_current_size(from);
        let align = Self::get_align_when_copied(from);
        let align_offset = Self::get_align_offset_when_copied(from);

        let to_address = copy_context.alloc_copy(from, object_size, align, align_offset, semantics);
        let new_object = Self::get_reference_when_copied_to(from, to_address);

        // Copy the object contents
        unsafe {
            std::ptr::copy_nonoverlapping(
                from.to_raw_address().to_mut_ptr::<u8>(),
                to_address.to_mut_ptr::<u8>(),
                object_size,
            );
        }

        new_object
    }

    fn copy_to(
        from: ObjectReference,
        to: ObjectReference,
        _region: mmtk::util::Address,
    ) -> mmtk::util::Address {
        let object_size = Self::get_current_size(from);

        // Copy the object contents to the specified location
        unsafe {
            std::ptr::copy_nonoverlapping(
                from.to_raw_address().to_mut_ptr::<u8>(),
                to.to_raw_address().to_mut_ptr::<u8>(),
                object_size,
            );
        }

        // Return the address after the copied object
        to.to_raw_address() + object_size
    }

    fn get_current_size(object: ObjectReference) -> usize {
        Self::size(object.to_raw_address().to_mut_ptr())
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

    fn get_reference_when_copied_to(
        _from: ObjectReference,
        to: mmtk::util::Address,
    ) -> ObjectReference {
        unsafe { ObjectReference::from_raw_address_unchecked(to) }
    }

    fn get_type_descriptor(_reference: ObjectReference) -> &'static [i8] {
        unsafe { std::mem::transmute(b"RustVMObject\0" as &[u8]) }
    }

    fn ref_to_object_start(object: ObjectReference) -> mmtk::util::Address {
        object.to_raw_address()
    }

    fn ref_to_header(object: ObjectReference) -> mmtk::util::Address {
        object.to_raw_address()
    }

    fn dump_object(_object: ObjectReference) {
        // Debug object dumping
    }
}
