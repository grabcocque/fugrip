use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8};
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GcError {
    #[error("Access to freed object")]
    AccessToFreedObject,
    #[error("Allocation failed")]
    AllocationFailed,
    #[error("Collection in progress")]
    CollectionInProgress,
}

pub type GcResult<T> = Result<T, GcError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Regular,
    Weak,
    WeakMap,
    ExactPtrTable,
    Finalizable,
}

pub trait GcTrace {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>);
}

pub trait Finalizable {
    fn finalize(&mut self);
}

#[repr(C)]
pub struct GcHeader<T> {
    pub mark_bit: AtomicBool,
    pub type_info: &'static TypeInfo,
    pub forwarding_ptr: AtomicPtr<GcHeader<T>>,
    pub data: T,
}

pub struct TypeInfo {
    pub trace_fn: unsafe fn(*mut GcHeader<()>, &mut Vec<SendPtr<GcHeader<()>>>),
    pub drop_fn: unsafe fn(*mut GcHeader<()>),
    pub redirect_pointers_fn: unsafe fn(*mut GcHeader<()>, *mut GcHeader<()>, *mut GcHeader<()>),
    pub size: usize,
    pub object_type: ObjectType,
}

impl Default for TypeInfo {
    fn default() -> Self {
        TypeInfo {
            trace_fn: |_, _| {},
            drop_fn: |_| {},
            redirect_pointers_fn: |_, _, _| {},
            size: 0,
            object_type: ObjectType::Regular,
        }
    }
}

pub struct Gc<T> {
    pub(crate) ptr: *mut GcHeader<T>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T> Gc<T> {
    pub fn as_ptr(&self) -> *mut GcHeader<T> {
        self.ptr
    }
}

unsafe impl<T: Send> Send for Gc<T> {}
unsafe impl<T: Sync> Sync for Gc<T> {}

impl<T> Clone for Gc<T> {
    fn clone(&self) -> Self {
        Gc {
            ptr: self.ptr,
            _phantom: PhantomData,
        }
    }
}

pub struct Weak<T> {
    pub(crate) target: AtomicPtr<GcHeader<T>>,
    pub(crate) _next_weak: AtomicPtr<Weak<T>>,
}

pub struct GcRef<'a, T> {
    pub(crate) _gc: &'a Gc<T>,
    pub(crate) _phantom: PhantomData<&'a T>,
}

pub struct GcRefMut<'a, T> {
    pub(crate) _gc: &'a Gc<T>,
    pub(crate) _phantom: PhantomData<&'a mut T>,
}

pub struct ReadGuard;
pub struct WriteGuard;

pub fn align_up(addr: *const u8, align: usize) -> *const u8 {
    let addr = addr as usize;
    let remainder = addr % align;
    if remainder == 0 {
        addr as *const u8
    } else {
        (addr + align - remainder) as *const u8
    }
}

impl TypeInfo {
    pub fn for_type<T: GcTrace>() -> &'static TypeInfo {
        // This would normally be cached in a global table
        // For now we leak the memory to get a 'static reference
        Box::leak(Box::new(TypeInfo {
            trace_fn: |obj, stack| unsafe {
                let header = &*(obj as *mut GcHeader<T>);
                header.data.trace(stack);
            },
            drop_fn: |obj| unsafe {
                let header = obj as *mut GcHeader<T>;
                std::ptr::drop_in_place(&mut (*header).data);
            },
            redirect_pointers_fn: |_obj, _dead, _free| {
            },
            size: std::mem::size_of::<GcHeader<T>>(),
            object_type: ObjectType::Regular,
        }))
    }
}

pub fn type_info<T: GcTrace>() -> &'static TypeInfo {
    TypeInfo::for_type::<T>()
}

impl GcTrace for i32 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

impl GcTrace for u32 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

impl GcTrace for i64 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

impl GcTrace for u64 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

impl GcTrace for String {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

impl GcTrace for () {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

impl<T> GcTrace for Gc<T> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        if !self.ptr.is_null() {
            stack.push(SendPtr::new(self.ptr as *mut GcHeader<()>));
        }
    }
}

impl<T> GcTrace for Option<Gc<T>> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        if let Some(gc) = self {
            unsafe { gc.trace(stack); }
        }
    }
}

impl<T> GcTrace for Vec<Gc<T>> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        for gc in self {
            unsafe { gc.trace(stack); }
        }
    }
}

pub struct SegmentBuffer {
    pub current: *mut u8,
    pub end: *mut u8,
    pub segment_id: usize,
}

impl Default for SegmentBuffer {
    fn default() -> Self {
        SegmentBuffer {
            current: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            segment_id: 0,
        }
    }
}

pub struct FinalizableObject<T: Finalizable> {
    pub data: T,
    pub finalize_state: AtomicU8,
}

// Wrapper for raw pointers to make them Send+Sync
#[derive(Clone, Copy)]
pub struct SendPtr<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        SendPtr(ptr)
    }
    
    pub fn as_ptr(&self) -> *mut T {
        self.0
    }
}

pub const FREE_SINGLETON_TYPE_INFO: TypeInfo = TypeInfo {
    trace_fn: |_, _| {},
    drop_fn: |_| {},
    redirect_pointers_fn: |_, _, _| {},
    size: std::mem::size_of::<GcHeader<()>>(),
    object_type: ObjectType::Regular,
};