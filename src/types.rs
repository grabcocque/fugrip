//! Core types to replace MMTk dependencies
//!
//! These are drop-in replacements for MMTk types that work with jemalloc.

use std::fmt;
use std::hash::Hash;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Drop-in replacement for mmtk::util::Address
/// Represents a memory address in the heap
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct Address(usize);

impl Address {
    pub const ZERO: Address = Address(0);

    pub const fn from_usize(addr: usize) -> Self {
        Address(addr)
    }

    pub const fn as_usize(&self) -> usize {
        self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    pub fn from_ptr<T>(ptr: *const T) -> Self {
        Address(ptr as usize)
    }

    pub fn from_mut_ptr<T>(ptr: *mut T) -> Self {
        Address(ptr as usize)
    }

    pub fn to_ptr<T>(&self) -> *const T {
        self.0 as *const T
    }

    pub fn to_mut_ptr<T>(&self) -> *mut T {
        self.0 as *mut T
    }

    pub fn offset(&self, bytes: isize) -> Address {
        Address((self.0 as isize + bytes) as usize)
    }

    pub fn add(&self, bytes: usize) -> Address {
        Address(self.0 + bytes)
    }

    pub fn sub(&self, bytes: usize) -> Address {
        Address(self.0 - bytes)
    }

    pub fn align_up(&self, align: usize) -> Address {
        let mask = align - 1;
        Address((self.0 + mask) & !mask)
    }

    pub fn align_down(&self, align: usize) -> Address {
        let mask = align - 1;
        Address(self.0 & !mask)
    }

    pub fn is_aligned(&self, align: usize) -> bool {
        self.0 % align == 0
    }

    pub fn is_aligned_to(&self, align: usize) -> bool {
        self.is_aligned(align)
    }

    pub unsafe fn load<T>(&self) -> T {
        unsafe { (self.0 as *const T).read() }
    }
}

impl From<usize> for Address {
    fn from(v: usize) -> Self {
        Address(v)
    }
}

impl From<Address> for usize {
    fn from(a: Address) -> Self {
        a.0
    }
}

impl<T> From<*const T> for Address {
    fn from(p: *const T) -> Self {
        Address(p as usize)
    }
}

impl<T> From<*mut T> for Address {
    fn from(p: *mut T) -> Self {
        Address(p as usize)
    }
}

impl Add<usize> for Address {
    type Output = Address;

    fn add(self, rhs: usize) -> Address {
        Address(self.0 + rhs)
    }
}

impl AddAssign<usize> for Address {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

impl Sub<usize> for Address {
    type Output = Address;

    fn sub(self, rhs: usize) -> Address {
        Address(self.0 - rhs)
    }
}

impl SubAssign<usize> for Address {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

// Subtracting two Addresses yields a usize offset
impl Sub for Address {
    type Output = usize;

    fn sub(self, rhs: Address) -> usize {
        self.0 - rhs.0
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Address({:#x})", self.0)
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

/// Drop-in replacement for mmtk::util::ObjectReference
/// Represents a reference to a GC-managed object
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct ObjectReference(Address);

impl ObjectReference {
    pub const NULL: ObjectReference = ObjectReference(Address::ZERO);

    pub fn from_raw_address(addr: Address) -> Option<Self> {
        // For our stub type, accept any non-zero address
        if addr.is_zero() {
            None
        } else {
            Some(ObjectReference(addr))
        }
    }

    pub unsafe fn from_raw_address_unchecked(addr: Address) -> Self {
        ObjectReference(addr)
    }

    pub fn to_raw_address(&self) -> Address {
        self.0
    }

    pub fn to_address(&self) -> Address {
        self.0
    }

    pub fn is_null(&self) -> bool {
        self.0.is_zero()
    }

    pub fn value(&self) -> usize {
        self.0.as_usize()
    }

    pub fn to_mut_ptr<T>(&self) -> *mut T {
        self.0.to_mut_ptr::<T>()
    }

    pub fn to_ptr<T>(&self) -> *const T {
        self.0.to_ptr::<T>()
    }
}

impl fmt::Debug for ObjectReference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ObjectRef({:#x})", self.0.as_usize())
    }
}

impl fmt::Display for ObjectReference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#x}", self.0.as_usize())
    }
}

impl From<ObjectReference> for Address {
    fn from(o: ObjectReference) -> Self {
        o.0
    }
}

impl From<ObjectReference> for usize {
    fn from(o: ObjectReference) -> Self {
        o.0.as_usize()
    }
}

impl TryFrom<Address> for ObjectReference {
    type Error = ();

    fn try_from(a: Address) -> Result<Self, Self::Error> {
        ObjectReference::from_raw_address(a).ok_or(())
    }
}

// Constants that MMTk code expects
pub mod constants {
    pub const MIN_OBJECT_SIZE: usize = 16;
    pub const BYTES_IN_PAGE: usize = 4096;
    pub const BYTES_IN_WORD: usize = std::mem::size_of::<usize>();
}

// Slot types for compatibility
pub mod slot {
    use super::{Address, ObjectReference};

    #[derive(Debug, Clone, Copy)]
    pub struct SimpleSlot {
        addr: Address,
    }

    impl SimpleSlot {
        pub fn from_address(addr: Address) -> Self {
            SimpleSlot { addr }
        }

        pub fn load(&self) -> ObjectReference {
            unsafe {
                let ptr = self.addr.to_ptr::<ObjectReference>();
                *ptr
            }
        }

        pub fn store(&self, object: ObjectReference) {
            unsafe {
                let ptr = self.addr.to_mut_ptr::<ObjectReference>();
                *ptr = object;
            }
        }
    }
}

// VM specifications for MMTk compatibility when not using MMTk
#[derive(Debug, Clone, Copy)]
pub struct VMGlobalLogBitSpec;

impl VMGlobalLogBitSpec {
    pub const fn side_first() -> Self {
        VMGlobalLogBitSpec
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VMLocalForwardingPointerSpec;

impl VMLocalForwardingPointerSpec {
    pub const fn in_header(_offset: usize) -> Self {
        VMLocalForwardingPointerSpec
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VMLocalForwardingBitsSpec;

impl VMLocalForwardingBitsSpec {
    pub const fn in_header(_offset: usize) -> Self {
        VMLocalForwardingBitsSpec
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VMLocalMarkBitSpec;

impl VMLocalMarkBitSpec {
    pub const fn in_header(_offset: usize) -> Self {
        VMLocalMarkBitSpec
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VMLocalLOSMarkNurserySpec;

impl VMLocalLOSMarkNurserySpec {
    pub const fn in_header(_offset: usize) -> Self {
        VMLocalLOSMarkNurserySpec
    }
}

// Stub ObjectModel trait for compatibility
pub trait ObjectModel<VM> {
    const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec;
    const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec;
    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec;
    const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec;
    const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec;
    const OBJECT_REF_OFFSET_LOWER_BOUND: isize;

    fn copy(
        from: ObjectReference,
        semantics: CopySemantics,
        copy_context: &mut GCWorkerCopyContext<VM>,
    ) -> ObjectReference;

    fn copy_to(from: ObjectReference, to: ObjectReference, region: Address) -> Address;

    fn get_current_size(object: ObjectReference) -> usize;
    fn get_size_when_copied(object: ObjectReference) -> usize;
    fn get_align_when_copied(object: ObjectReference) -> usize;
    fn get_align_offset_when_copied(object: ObjectReference) -> usize;
    fn get_reference_when_copied_to(from: ObjectReference, to: Address) -> ObjectReference;
    fn get_type_descriptor(reference: ObjectReference) -> &'static [i8];
    fn ref_to_object_start(object: ObjectReference) -> Address;
    fn ref_to_header(object: ObjectReference) -> Address;
    fn dump_object(object: ObjectReference);
}

// Stub types for copy semantics
#[derive(Debug, Clone, Copy)]
pub enum CopySemantics {
    Default,
}

// Stub GC worker copy context
pub struct GCWorkerCopyContext<VM> {
    _phantom: std::marker::PhantomData<VM>,
}

impl<VM> GCWorkerCopyContext<VM> {
    pub fn alloc_copy(
        &mut self,
        _original: ObjectReference,
        bytes: usize,
        align: usize,
        _offset: usize,
        _semantics: CopySemantics,
    ) -> Address {
        // Stub implementation - in practice would delegate to allocator
        use std::alloc::{Layout, alloc};
        let layout = Layout::from_size_align(bytes, align).unwrap();
        let ptr = unsafe { alloc(layout) };
        Address::from_usize(ptr as usize)
    }
}
