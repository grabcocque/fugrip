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

use crate::frontend::types::{Address, ObjectReference};

// RustVM should be defined in backend modules, not here

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
/// Aggressively optimized object header using data-oriented design principles.
///
/// **CRITICAL HOT PATH**: Accessed on every allocation, GC operation, and field access.
/// Optimized to fit 4 headers per cache line (16 bytes each instead of 32 bytes).
///
/// # Data-Oriented Optimizations:
/// - 50% size reduction for better cache utilization
/// - Bit-packed flags and metadata for cache efficiency
/// - Field reordering by access frequency
/// - Cache-line friendly alignment
#[repr(C, align(8))]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ObjectHeader {
    /// Hot flags accessed during marking/sweeping (first 8 bytes for cache efficiency)
    pub flags: ObjectFlags, // 2 bytes (was wider before)
    pub layout_id: LayoutId, // 2 bytes (compact type ID)
    pub body_size: u32,      // 4 bytes (sufficient for most objects, was usize)

    /// Less frequently accessed vtable pointer (second 8 bytes)
    pub vtable: *const (), // 8 bytes
}

impl Default for ObjectHeader {
    fn default() -> Self {
        Self {
            flags: ObjectFlags::empty(),
            layout_id: LayoutId::default(),
            body_size: 0u32,
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
/// use crate::frontend::types::Address;
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
/// assert_eq!(obj_ref.to_raw_address(), crate::frontend::types::Address::from_usize(aligned_addr));
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

    pub fn as_raw_ptr(&self) -> *mut T {
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
        unsafe { ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(self.ptr)) }
    }
}

/// Object model hooks consumed by the collector.
pub trait ObjectModel {
    fn header(object: *mut u8) -> ObjectHeader;
    fn size(object: *mut u8) -> usize {
        Self::header(object).body_size as usize + size_of::<ObjectHeader>()
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
    /// Direct access to get_current_size for compatibility
    pub fn get_current_size(object: ObjectReference) -> usize {
        Self::size(object.to_raw_address().to_mut_ptr())
    }

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

    /// Real implementation: Get object size for memory slice integration
    ///
    /// This method provides object size information for the MemorySlice implementation,
    /// enabling proper object boundary detection and memory layout analysis.
    pub fn get_object_size(object_ptr: *mut u8) -> Option<usize> {
        if object_ptr.is_null() {
            return None;
        }

        // Use the existing MMTkObjectModel implementation to get object size
        // This leverages the production-ready size calculation already implemented
        match ObjectReference::from_raw_address(Address::from_mut_ptr(object_ptr)) {
            Some(obj_ref) => {
                // Use the existing get_current_size method from MMTkObjectModel
                Some(Self::get_current_size(obj_ref))
            }
            None => None,
        }
    }
}

// MMTk trait implementations have been moved to src/backends/mmtk/object_model.rs

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
            ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(buffer1.as_mut_ptr()))
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
            ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(buffer2.as_mut_ptr()))
        };

        assert!(model.get_weak_ref_header(obj_ref2).is_some());
    }

    #[test]
    fn test_rust_object_model_get_current_size() {
        let header = ObjectHeader {
            flags: ObjectFlags::empty(),
            layout_id: LayoutId(0),
            body_size: 128,
            vtable: std::ptr::null(),
        };

        let mut buffer = [0u8; 256];
        let ptr = buffer.as_mut_ptr();

        unsafe {
            ptr.cast::<ObjectHeader>().write(header);
        }

        let obj_ref =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(ptr)) };

        let size = RustObjectModel::get_current_size(obj_ref);
        assert_eq!(size, std::mem::size_of::<ObjectHeader>() + 128);
    }

    #[test]
    fn test_rust_object_model_get_object_size() {
        let header = ObjectHeader {
            flags: ObjectFlags::empty(),
            layout_id: LayoutId(0),
            body_size: 256,
            vtable: std::ptr::null(),
        };

        let mut buffer = [0u8; 512];
        let ptr = buffer.as_mut_ptr();

        unsafe {
            ptr.cast::<ObjectHeader>().write(header);
        }

        let size = RustObjectModel::get_object_size(ptr);
        assert!(size.is_some());
        assert_eq!(size.unwrap(), std::mem::size_of::<ObjectHeader>() + 256);

        // Test with null pointer
        let null_size = RustObjectModel::get_object_size(std::ptr::null_mut());
        assert!(null_size.is_none());
    }
}
