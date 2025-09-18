//! Object model definitions for Fugrip VM objects.
//!
//! This module provides the core object model for the FUGC garbage collector,
//! including object headers, garbage collected pointers, and tracing infrastructure.
//!
//! # Examples
//!
//! ```
//! use fugrip::core::{ObjectHeader, Gc, ObjectFlags, LayoutId};
//!
//! // Create an object header
//! let header = ObjectHeader {
//!     flags: ObjectFlags::MARKED,
//!     layout_id: LayoutId(1),
//!     body_size: 64,
//!     vtable: std::ptr::null(),
//! };
//!
//! // Create a null Gc pointer
//! let gc_ptr: Gc<u32> = Gc::new();
//! assert!(gc_ptr.is_null());
//! ```

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
///
/// # Examples
///
/// ```
/// use fugrip::core::{ObjectHeader, ObjectFlags, LayoutId};
///
/// // Create a basic object header
/// let header = ObjectHeader {
///     flags: ObjectFlags::MARKED | ObjectFlags::PINNED,
///     layout_id: LayoutId(42),
///     body_size: 128,
///     vtable: std::ptr::null(),
/// };
///
/// assert!(header.flags.contains(ObjectFlags::MARKED));
/// assert!(header.flags.contains(ObjectFlags::PINNED));
/// assert!(!header.flags.contains(ObjectFlags::HAS_WEAK_REFS));
/// assert_eq!(header.body_size, 128);
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
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
    /// Object flags for tracking various object states in the garbage collector.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::core::ObjectFlags;
    ///
    /// // Create flags for a marked, pinned object with weak references
    /// let flags = ObjectFlags::MARKED | ObjectFlags::PINNED | ObjectFlags::HAS_WEAK_REFS;
    ///
    /// assert!(flags.contains(ObjectFlags::MARKED));
    /// assert!(flags.contains(ObjectFlags::PINNED));
    /// assert!(flags.contains(ObjectFlags::HAS_WEAK_REFS));
    ///
    /// // Remove the pinned flag
    /// let unpinned = flags & !ObjectFlags::PINNED;
    /// assert!(!unpinned.contains(ObjectFlags::PINNED));
    /// assert!(unpinned.contains(ObjectFlags::MARKED));
    /// ```
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ObjectFlags: u16 {
        const MARKED = 0b0001;
        const HAS_WEAK_REFS = 0b0010;
        /// PINNED flag for epoch pinning synergy: Objects marked PINNED can be treated as pinned to the current epoch during GC operations,
        /// ensuring safe reclamation. This hybridizes with crossbeam-epoch Guards for invariant enforcement without full pinning overhead.
        const PINNED = 0b0100;
    }
}

/// Identifier pointing into the VM's layout/descriptor table.
///
/// # Examples
///
/// ```
/// use fugrip::core::LayoutId;
///
/// // Create layout IDs for different object types
/// let string_layout = LayoutId(1);
/// let array_layout = LayoutId(2);
/// let default_layout = LayoutId::default();
///
/// assert_eq!(string_layout.0, 1);
/// assert_eq!(array_layout.0, 2);
/// assert_eq!(default_layout.0, 0);
/// assert_ne!(string_layout, array_layout);
/// ```
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct LayoutId(pub u16);

/// `Gc<T>` is a typed wrapper around MMTk's `ObjectReference`.
///
/// # Examples
///
/// ```
/// use fugrip::core::Gc;
/// use mmtk::util::Address;
///
/// // Create a null Gc pointer
/// let null_ptr: Gc<i32> = Gc::new();
/// assert!(null_ptr.is_null());
///
/// // Create from aligned address (simulating heap allocation)
/// let aligned_addr = 0x10000000usize; // Word-aligned address
/// let raw_ptr = aligned_addr as *mut i32;
/// let gc_ptr = Gc::from_raw(raw_ptr);
/// assert!(!gc_ptr.is_null());
/// assert_eq!(gc_ptr.as_ptr(), raw_ptr);
///
/// // Convert to ObjectReference (with proper alignment)
/// let obj_ref = gc_ptr.to_object_reference();
/// // ObjectReference created from aligned pointer
/// assert_eq!(obj_ref.to_raw_address(), unsafe { Address::from_usize(aligned_addr) });
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_flags() {
        let flags = ObjectFlags::MARKED | ObjectFlags::PINNED;
        assert!(flags.contains(ObjectFlags::MARKED));
        assert!(flags.contains(ObjectFlags::PINNED));
        assert!(!flags.contains(ObjectFlags::HAS_WEAK_REFS));

        let default_flags = ObjectFlags::default();
        assert!(!default_flags.contains(ObjectFlags::MARKED));
    }

    #[test]
    fn test_layout_id() {
        let id1 = LayoutId(42);
        let id2 = LayoutId(42);
        let id3 = LayoutId(100);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);

        let default_id = LayoutId::default();
        assert_eq!(default_id, LayoutId(0));
    }

    #[test]
    fn test_object_header() {
        let header = ObjectHeader {
            flags: ObjectFlags::MARKED,
            layout_id: LayoutId(1),
            body_size: 64,
            vtable: std::ptr::null(),
        };

        assert!(header.flags.contains(ObjectFlags::MARKED));
        assert_eq!(header.layout_id, LayoutId(1));
        assert_eq!(header.body_size, 64);
        assert_eq!(header.vtable, std::ptr::null());

        let default_header = ObjectHeader::default();
        assert_eq!(default_header.body_size, 0);
    }

    #[test]
    fn test_gc_creation() {
        let gc1 = Gc::<u32>::new();
        assert!(gc1.is_null());

        let gc2 = Gc::<u32>::default();
        assert!(gc2.is_null());

        let ptr = 0x1000 as *mut u32;
        let gc3 = Gc::from_raw(ptr);
        assert!(!gc3.is_null());
        assert_eq!(gc3.as_ptr(), ptr);
    }

    #[test]
    fn test_gc_trace() {
        let gc_null = Gc::<u32>::new();
        let mut visits = Vec::new();
        gc_null.trace(&mut |ptr| visits.push(ptr));
        assert!(visits.is_empty());

        let gc_valid = Gc::from_raw(0x2000 as *mut u32);
        visits.clear();
        gc_valid.trace(&mut |ptr| visits.push(ptr));
        assert_eq!(visits.len(), 1);
        assert_eq!(visits[0], 0x2000 as *mut u8);
    }

    #[test]
    fn test_gc_to_object_reference() {
        let gc = Gc::from_raw(0x3000 as *mut u32);
        let obj_ref = gc.to_object_reference();
        assert_eq!(obj_ref.to_raw_address().as_usize(), 0x3000);
    }

    #[test]
    fn test_rust_object_model_header() {
        let header = ObjectHeader {
            flags: ObjectFlags::MARKED,
            layout_id: LayoutId(5),
            body_size: 128,
            vtable: std::ptr::null(),
        };

        let mut buffer = [0u8; 256];
        let ptr = buffer.as_mut_ptr();

        unsafe {
            ptr.cast::<ObjectHeader>().write(header);
        }

        let retrieved = RustObjectModel::header(ptr);
        assert_eq!(retrieved.flags, ObjectFlags::MARKED);
        assert_eq!(retrieved.layout_id, LayoutId(5));
        assert_eq!(retrieved.body_size, 128);
    }

    #[test]
    fn test_rust_object_model_size() {
        let header = ObjectHeader {
            flags: ObjectFlags::empty(),
            layout_id: LayoutId(0),
            body_size: 64,
            vtable: std::ptr::null(),
        };

        let mut buffer = [0u8; 256];
        let ptr = buffer.as_mut_ptr();

        unsafe {
            ptr.cast::<ObjectHeader>().write(header);
        }

        let size = RustObjectModel::size(ptr);
        assert_eq!(size, std::mem::size_of::<ObjectHeader>() + 64);
    }

    #[test]
    fn test_rust_object_model_weak_ref() {
        let model = RustObjectModel;

        // Object without weak refs
        let header_no_weak = ObjectHeader {
            flags: ObjectFlags::MARKED,
            layout_id: LayoutId(0),
            body_size: 32,
            vtable: std::ptr::null(),
        };

        let mut buffer1 = [0u8; 256];
        unsafe {
            buffer1
                .as_mut_ptr()
                .cast::<ObjectHeader>()
                .write(header_no_weak);
        }

        let obj_ref1 = unsafe {
            ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_mut_ptr(
                buffer1.as_mut_ptr(),
            ))
        };

        assert!(model.get_weak_ref_header(obj_ref1).is_none());

        // Object with weak refs
        let header_with_weak = ObjectHeader {
            flags: ObjectFlags::HAS_WEAK_REFS,
            layout_id: LayoutId(0),
            body_size: 32,
            vtable: std::ptr::null(),
        };

        let mut buffer2 = [0u8; 256];
        unsafe {
            buffer2
                .as_mut_ptr()
                .cast::<ObjectHeader>()
                .write(header_with_weak);
        }

        let obj_ref2 = unsafe {
            ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_mut_ptr(
                buffer2.as_mut_ptr(),
            ))
        };

        assert!(model.get_weak_ref_header(obj_ref2).is_some());
    }

    #[test]
    fn test_mmtk_object_model_functions() {
        let obj_ref = unsafe {
            ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x4000))
        };

        // Test alignment functions
        assert_eq!(RustObjectModel::get_align_when_copied(obj_ref), 8);
        assert_eq!(RustObjectModel::get_align_offset_when_copied(obj_ref), 0);

        // Test reference functions
        let new_addr = unsafe { mmtk::util::Address::from_usize(0x5000) };
        let new_ref = RustObjectModel::get_reference_when_copied_to(obj_ref, new_addr);
        assert_eq!(new_ref.to_raw_address(), new_addr);

        // Test type descriptor
        let type_desc = RustObjectModel::get_type_descriptor(obj_ref);
        assert!(!type_desc.is_empty());

        // Test address functions
        assert_eq!(
            RustObjectModel::ref_to_object_start(obj_ref),
            obj_ref.to_raw_address()
        );
        assert_eq!(
            RustObjectModel::ref_to_header(obj_ref),
            obj_ref.to_raw_address()
        );

        // Test dump (should not panic)
        RustObjectModel::dump_object(obj_ref);
    }
}
