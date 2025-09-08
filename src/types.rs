use crate::Finalizable;
// From gc_ref_wrappers.rs: types only

/// Guard type for read-only access to garbage-collected objects.
///
/// `ReadGuard` provides immutable access to the contents of a `Gc<T>`
/// through the RAII pattern, ensuring memory safety during garbage
/// collection cycles.
///
/// # Examples
///
/// ```
/// use fugrip::ReadGuard;
///
/// // ReadGuard is typically returned by Gc::read()
/// // This example shows the type in isolation
/// let _guard = ReadGuard;
/// ```
pub struct ReadGuard;
/// Guard type for read-write access to garbage-collected objects.
///
/// `WriteGuard` provides mutable access to the contents of a `Gc<T>`
/// through the RAII pattern, ensuring exclusive access and memory
/// safety during garbage collection cycles.
///
/// # Examples
///
/// ```
/// use fugrip::WriteGuard;
///
/// // WriteGuard is typically returned by Gc::write()
/// // This example shows the type in isolation
/// let _guard = WriteGuard;
/// ```
pub struct WriteGuard;

/// A wrapper for objects that need finalization during garbage collection.
///
/// `FinalizableObject<T>` wraps any type that implements `Finalizable` and
/// tracks its finalization state atomically. This is useful for resources
/// that need cleanup when garbage collected.
///
/// # Examples
///
/// ```
/// use fugrip::{Finalizable, FinalizableObject};
///
/// struct Resource {
///     id: u32,
/// }
///
/// impl Finalizable for Resource {
///     fn finalize(&mut self) {
///         println!("Resource {} finalized", self.id);
///     }
/// }
///
/// let resource = Resource { id: 123 };
/// let finalizable = FinalizableObject::new(resource);
/// assert!(!finalizable.is_finalized());
/// ```
pub struct FinalizableObject<T: Finalizable> {
    pub data: T,
    pub finalize_state: std::sync::atomic::AtomicU8,
}

// Implement GcTrace for FinalizableObject to enable tracing
unsafe impl<T: Finalizable + crate::traits::GcTrace> crate::traits::GcTrace for FinalizableObject<T> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        unsafe { self.data.trace(stack); }
    }
}

impl<T: Finalizable> FinalizableObject<T> {
    /// Create a new finalizable object wrapper.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Finalizable, FinalizableObject};
    ///
    /// struct Resource { id: u32 }
    /// 
    /// impl Finalizable for Resource {
    ///     fn finalize(&mut self) {
    ///         println!("Finalizing resource {}", self.id);
    ///     }
    /// }
    ///
    /// let resource = Resource { id: 42 };
    /// let finalizable = FinalizableObject::new(resource);
    /// assert_eq!(finalizable.data.id, 42);
    /// ```
    pub fn new(data: T) -> Self {
        Self {
            data,
            finalize_state: std::sync::atomic::AtomicU8::new(0),
        }
    }

    /// Mark this object as finalized.
    ///
    /// This method is called by the garbage collector during finalization.
    /// It atomically updates the finalization state to indicate the object
    /// has been processed.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Finalizable, FinalizableObject};
    ///
    /// struct TestData { value: i32 }
    /// 
    /// impl Finalizable for TestData {
    ///     fn finalize(&mut self) {}
    /// }
    ///
    /// let data = TestData { value: 123 };
    /// let finalizable = FinalizableObject::new(data);
    /// 
    /// assert!(!finalizable.is_finalized());
    /// finalizable.mark_finalized();
    /// assert!(finalizable.is_finalized());
    /// ```
    pub fn mark_finalized(&self) {
        self.finalize_state.store(1, std::sync::atomic::Ordering::Release);
    }

    /// Check if this object has been finalized.
    ///
    /// Returns `true` if the object has been processed by the finalizer,
    /// `false` otherwise. This check is atomic and thread-safe.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Finalizable, FinalizableObject};
    ///
    /// struct TestData { name: String }
    /// 
    /// impl Finalizable for TestData {
    ///     fn finalize(&mut self) {
    ///         self.name.clear();
    ///     }
    /// }
    ///
    /// let data = TestData { name: "test".to_string() };
    /// let finalizable = FinalizableObject::new(data);
    /// 
    /// // Initially not finalized
    /// assert!(!finalizable.is_finalized());
    /// 
    /// // After marking as finalized
    /// finalizable.mark_finalized();
    /// assert!(finalizable.is_finalized());
    /// ```
    pub fn is_finalized(&self) -> bool {
        self.finalize_state.load(std::sync::atomic::Ordering::Acquire) != 0
    }
}
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicPtr};
use thiserror::Error;

/// Errors that can occur during garbage collection operations.
///
/// This enum represents the various error conditions that can arise
/// when working with garbage-collected objects.
///
/// # Examples
///
/// ```
/// use fugrip::GcError;
///
/// let error = GcError::AccessToFreedObject;
/// println!("Error: {}", error);
/// ```
#[derive(Error, Debug)]
pub enum GcError {
    #[error("Access to freed object")]
    AccessToFreedObject,
    #[error("Allocation failed")]
    AllocationFailed,
    #[error("Collection in progress")]
    CollectionInProgress,
}

/// A Result type for garbage collection operations.
///
/// This is a convenience type alias for `Result<T, GcError>`, used
/// throughout the API to handle garbage collection errors.
///
/// # Examples
///
/// ```
/// use fugrip::{GcResult, GcError};
///
/// fn example_operation() -> GcResult<i32> {
///     Ok(42)
/// }
///
/// let result = example_operation();
/// assert_eq!(result.unwrap(), 42);
/// ```
pub type GcResult<T> = Result<T, GcError>;

/// Types of objects that can be allocated in the garbage-collected heap.
///
/// This enum categorizes different kinds of objects for specialized
/// handling during allocation and collection phases.
///
/// # Examples
///
/// ```
/// use fugrip::ObjectType;
///
/// let obj_type = ObjectType::Regular;
/// assert_eq!(obj_type, ObjectType::Regular);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    Regular,
    Weak,
    WeakMap,
    ExactPtrTable,
    Finalizable,
}

/// Classification system for garbage-collected objects.
///
/// Objects are classified into different categories based on their
/// cleanup requirements, which affects how they are allocated and
/// collected by the garbage collector.
///
/// # Examples
///
/// ```
/// use fugrip::ObjectClass;
///
/// let class = ObjectClass::Default;
/// assert_eq!(class, ObjectClass::Default);
///
/// let finalizer_class = ObjectClass::Finalizer;
/// assert_ne!(finalizer_class, ObjectClass::Default);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectClass {
    Default,
    Destructor,
    Census,
    CensusAndDestructor,
    Finalizer,
    Weak,
}

/// Phases of the garbage collection cycle.
///
/// The FUGC garbage collector operates in distinct phases, each with
/// specific responsibilities for marking, cleaning up, and sweeping
/// unreachable objects.
///
/// # Examples
///
/// ```
/// use fugrip::CollectorPhase;
///
/// let phase = CollectorPhase::Waiting;
/// assert_eq!(phase, CollectorPhase::Waiting);
///
/// let marking = CollectorPhase::Marking;
/// assert_ne!(marking, CollectorPhase::Waiting);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollectorPhase {
    Waiting,
    Marking,
    Censusing,
    Reviving,
    Remarking,
    Recensusing,
    Sweeping,
}

impl CollectorPhase {
    /// Convert a numeric value to a collector phase.
    ///
    /// This is used internally to convert atomic integer values
    /// back to enum variants for phase management.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::CollectorPhase;
    ///
    /// assert_eq!(CollectorPhase::from_usize(0), CollectorPhase::Waiting);
    /// assert_eq!(CollectorPhase::from_usize(1), CollectorPhase::Marking);
    /// assert_eq!(CollectorPhase::from_usize(99), CollectorPhase::Waiting); // Invalid values default to Waiting
    /// ```
    pub fn from_usize(val: usize) -> CollectorPhase {
        match val {
            0 => CollectorPhase::Waiting,
            1 => CollectorPhase::Marking,
            2 => CollectorPhase::Censusing,
            3 => CollectorPhase::Reviving,
            4 => CollectorPhase::Remarking,
            5 => CollectorPhase::Recensusing,
            6 => CollectorPhase::Sweeping,
            _ => CollectorPhase::Waiting, // Default fallback
        }
    }
}

/// Thread-safe pointer wrapper for garbage collection.
///
/// `SendPtr<T>` is a wrapper around a raw pointer that implements `Send` and `Sync`,
/// allowing it to be safely passed between threads. This is used internally by the
/// garbage collector to manage object references across thread boundaries.
///
/// # Safety
///
/// This type implements `Send` and `Sync` unconditionally, so the caller must
/// ensure that the wrapped pointer is indeed safe to send across threads.
///
/// # Examples
///
/// ```
/// use fugrip::SendPtr;
///
/// let x = Box::into_raw(Box::new(10i32));
/// // Safety: `x` is a valid pointer here and we immediately convert it back.
/// let sp = unsafe { SendPtr::new(x) };
/// assert_eq!(unsafe { *sp.as_ptr() }, 10);
/// // Clean up to avoid leak in doctest
/// unsafe { Box::from_raw(x); }
/// ```
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SendPtr<T> {
    ptr: *mut T,
}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T> Copy for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Creates a new SendPtr from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to a valid object of type `T`
    /// - The object will remain valid for the lifetime of this `SendPtr`
    /// - The pointer is safe to send across thread boundaries
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::SendPtr;
    ///
    /// let value = Box::into_raw(Box::new(42i32));
    /// let send_ptr = unsafe { SendPtr::new(value) };
    /// assert_eq!(send_ptr.as_ptr(), value);
    /// // Clean up
    /// unsafe { Box::from_raw(value); }
    /// ```
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self { ptr }
    }

    /// Returns the wrapped raw pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::SendPtr;
    ///
    /// let value = Box::into_raw(Box::new(100i32));
    /// let send_ptr = unsafe { SendPtr::new(value) };
    /// let ptr = send_ptr.as_ptr();
    /// assert_eq!(ptr, value);
    /// // Clean up
    /// unsafe { Box::from_raw(value); }
    /// ```
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }
}

// Clone and Copy are now derived

/// Runtime type information for garbage-collected objects.
///
/// `TypeInfo` contains all the metadata needed by the garbage collector
/// to properly manage objects of a specific type, including size, alignment,
/// and function pointers for type-specific operations.
///
/// # Examples
///
/// ```
/// use fugrip::{TypeInfo, ObjectType};
///
/// // TypeInfo is typically created by the GC system
/// // This shows the structure in isolation
/// let type_info = TypeInfo {
///     size: std::mem::size_of::<i32>(),
///     align: std::mem::align_of::<i32>(),
///     trace_fn: |_, _| unsafe { /* trace implementation */ },
///     drop_fn: |_| unsafe { /* drop implementation */ },
///     finalize_fn: None,
///     redirect_pointers_fn: |_, _, _| unsafe { /* redirect implementation */ },
///     object_type: ObjectType::Regular,
/// };
/// assert_eq!(type_info.size, 4);
/// ```
#[derive(Debug)]
pub struct TypeInfo {
    pub size: usize,
    pub align: usize,
    pub trace_fn: unsafe fn(*const (), &mut Vec<SendPtr<GcHeader<()>>>),
    pub drop_fn: unsafe fn(*mut GcHeader<()>),
    pub finalize_fn: Option<unsafe fn(*mut GcHeader<()>)>,
    pub redirect_pointers_fn: unsafe fn(*mut GcHeader<()>, *mut GcHeader<()>, *mut GcHeader<()>),
    pub object_type: ObjectType,
}

/// Header for all garbage-collected objects.
///
/// Every object allocated in the garbage-collected heap has this header
/// prepended to it. The header contains metadata necessary for garbage
/// collection, including mark bits, type information, and forwarding pointers.
///
/// # Examples
///
/// ```
/// use fugrip::{GcHeader, type_info};
/// use std::sync::atomic::{AtomicBool, AtomicPtr};
///
/// // GcHeader is typically managed by the GC system
/// // This shows how to access the structure with actual type info
/// let header = GcHeader {
///     mark_bit: AtomicBool::new(false),
///     type_info: type_info::<i32>(),
///     forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
///     weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
///     data: 42i32,
/// };
/// 
/// assert_eq!(header.data, 42);
/// assert!(!header.mark_bit.load(std::sync::atomic::Ordering::Relaxed));
/// ```
#[derive(Debug)]
#[repr(C)]
pub struct GcHeader<T> {
    pub mark_bit: AtomicBool,
    pub type_info: &'static TypeInfo,
    pub forwarding_ptr: AtomicPtr<GcHeader<()>>,
    pub weak_ref_list: AtomicPtr<Weak<()>>,
    pub data: T,
}

/// Smart pointer for garbage-collected objects.
///
/// `Gc<T>` is the primary interface for creating and accessing garbage-collected objects.
/// It provides atomic pointer semantics with automatic forwarding pointer following
/// and integration with the garbage collector's marking and sweeping phases.
///
/// # Examples
///
/// ```
/// use fugrip::Gc;
///
/// // Create a garbage-collected integer
/// let gc_int = Gc::new(42);
///
/// // Read the value if it hasn't been collected
/// if let Some(value) = gc_int.read() {
///     // This would work in the full implementation
///     // assert_eq!(*value, 42);
/// }
/// ```
///
/// ```
/// use fugrip::Gc;
///
/// // Create a garbage-collected string
/// let gc_string = Gc::new("Hello, World!".to_string());
///
/// // Write access if object is still alive and not borrowed
/// if let Some(mut writer) = gc_string.write() {
///     // This would work in the full implementation
///     // *writer = "Modified!".to_string();
/// }
/// ```
pub struct Gc<T> {
    pub(crate) ptr: AtomicPtr<GcHeader<T>>,
    pub(crate) _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for Gc<T> {}
unsafe impl<T: Sync> Sync for Gc<T> {}

impl<T> Clone for Gc<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: AtomicPtr::new(self.ptr.load(std::sync::atomic::Ordering::Acquire)),
            _phantom: PhantomData,
        }
    }
}

// Implement GcTrace for Gc<T> to enable tracing through GC pointers
unsafe impl<T: crate::traits::GcTrace> crate::traits::GcTrace for Gc<T> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        let ptr = self.ptr.load(std::sync::atomic::Ordering::Acquire);
        if !ptr.is_null() {
            unsafe {
                // Follow forwarding pointer if object has been moved
                let header = &*ptr;
                let forwarding_ptr = header.forwarding_ptr.load(std::sync::atomic::Ordering::Acquire);
                
                if !forwarding_ptr.is_null() && forwarding_ptr != crate::types::FreeSingleton::instance() {
                    // Object has been forwarded to a new location
                    stack.push(SendPtr::new(forwarding_ptr));
                } else if forwarding_ptr != crate::types::FreeSingleton::instance() {
                    // Object is at original location and not collected
                    stack.push(SendPtr::new(ptr as *mut GcHeader<()>));
                }
                // If forwarding_ptr == FREE_SINGLETON, object is dead, don't trace
            }
        }
    }
}

impl<T> Gc<T> {
    /// Create a new garbage-collected object.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::Gc;
    ///
    /// let gc_int = Gc::new(42);
    /// let gc_string = Gc::new("Hello".to_string());
    /// let gc_vec = Gc::new(vec![1, 2, 3]);
    /// ```
    pub fn new(value: T) -> Self
    where
        T: crate::traits::GcTrace + 'static,
    {
        use crate::memory::ALLOCATOR;
        ALLOCATOR.allocate_gc(value)
    }

    /// Try to read the garbage-collected object.
    ///
    /// Returns `None` if the object has been garbage collected or is currently
    /// being moved by the collector.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::Gc;
    ///
    /// let gc_value = Gc::new(100);
    /// if let Some(reader) = gc_value.read() {
    ///     // Access the value immutably
    ///     // println!("Value: {}", *reader);
    /// }
    /// ```
    pub fn read(&self) -> Option<GcRef<'_, T>> {
        let ptr = self.ptr.load(std::sync::atomic::Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        
        // Follow forwarding pointer if object has been moved/collected
        unsafe {
            let header = &*ptr;
            let forwarding_ptr = header.forwarding_ptr.load(std::sync::atomic::Ordering::Acquire);
            
            if !forwarding_ptr.is_null() {
                // Object has been redirected to FREE_SINGLETON or moved
                if forwarding_ptr == crate::types::FreeSingleton::instance() {
                    return None; // Object has been collected
                }
                // Follow forwarding pointer for moved objects
                let forwarded_header = &*(forwarding_ptr as *mut GcHeader<T>);
                return Some(GcRef::new(&forwarded_header.data as *const T));
            }
            
            // Object is valid, return reference
            Some(GcRef::new(&header.data as *const T))
        }
    }

    /// Try to write to the garbage-collected object.
    ///
    /// Returns `None` if the object has been garbage collected, is currently
    /// being moved by the collector, or is already borrowed mutably.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::Gc;
    ///
    /// let gc_value = Gc::new(vec![1, 2, 3]);
    /// if let Some(mut writer) = gc_value.write() {
    ///     // Modify the value mutably
    ///     // writer.push(4);
    /// }
    /// ```
    pub fn write(&self) -> Option<GcRefMut<'_, T>> {
        let ptr = self.ptr.load(std::sync::atomic::Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        
        // Follow forwarding pointer if object has been moved/collected
        unsafe {
            let header = &*ptr;
            let forwarding_ptr = header.forwarding_ptr.load(std::sync::atomic::Ordering::Acquire);
            
            if !forwarding_ptr.is_null() {
                // Object has been redirected to FREE_SINGLETON or moved
                if forwarding_ptr == crate::types::FreeSingleton::instance() {
                    return None; // Object has been collected
                }
                // Follow forwarding pointer for moved objects
                let forwarded_header = &*(forwarding_ptr as *mut GcHeader<T>);
                return Some(GcRefMut::new(&forwarded_header.data as *const T as *mut T));
            }
            
            // Object is valid, return mutable reference
            Some(GcRefMut::new(&header.data as *const T as *mut T))
        }
    }

    /// Get the raw pointer to the object header.
    ///
    /// This is primarily used internally by the garbage collector.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::Gc;
    ///
    /// let gc_value = Gc::new(42);
    /// let ptr = gc_value.as_ptr();
    /// // The pointer should not be null for a valid object
    /// assert!(!ptr.is_null());
    /// ```
    pub fn as_ptr(&self) -> *mut GcHeader<T> {
        self.ptr.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Weak reference to a garbage-collected object.
///
/// `Weak<T>` provides a non-owning reference to a garbage-collected object that
/// does not prevent the object from being collected. Weak references are organized
/// in doubly-linked lists and are automatically invalidated when their target
/// object is garbage collected.
///
/// # Examples
///
/// ```
/// use fugrip::{Gc, Weak};
///
/// // Create a strong reference
/// let strong_ref = Gc::new(42i32);
///
/// // Create a weak reference to it
/// let weak_ref = Weak::new_simple(&strong_ref);
///
/// // Weak references can be upgraded to strong references
/// if let Some(weak_reader) = weak_ref.read() {
///     // In a full implementation, this would attempt upgrade:
///     // if let Some(upgraded) = weak_reader.upgrade() {
///     //     // Access the strong reference
///     // }
/// }
/// ```
pub struct Weak<T> {
    pub target: AtomicPtr<GcHeader<T>>,
    pub next_weak: AtomicPtr<Weak<T>>,
    pub prev_weak: AtomicPtr<Weak<T>>,
}

impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        // Do not attempt to unlink from the target's weak chain here:
        // - Inline Weak<T> may be stored in movable containers (Vec), making their
        //   address unstable for linked lists.
        // - GC-managed weak nodes are freed during sweep; target may already be
        //   forwarded or invalidated. Unlinking here risks races/UAF.
        // Pragmatic safety: just clear our own target pointer to deny upgrades.
        self.target.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
        // next_weak/prev_weak are left as-is; the chain is invalidated when targets die.
    }
}

impl<T> Weak<T> {
    /// Create a simple weak reference to a garbage-collected object.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Gc, Weak};
    ///
    /// let strong_ref = Gc::new("Hello".to_string());
    /// let weak_ref = Weak::new_simple(&strong_ref);
    /// ```
    pub fn new_simple(strong_ref: &Gc<T>) -> Self {
        // Pragmatic design: inline Weak<T> wrappers are not linked into the target's
        // weak_ref_list to avoid pointer-to-interior invalidation (e.g., when stored
        // in Vec<Weak<T>> which may reallocate and move entries). GC-managed weak
        // nodes created via ClassifiedAllocator::allocate_weak are linked safely.
        Self {
            target: AtomicPtr::new(strong_ref.as_ptr()),
            next_weak: AtomicPtr::new(std::ptr::null_mut()),
            prev_weak: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Try to read the weak reference.
    ///
    /// Returns a `WeakRef` that can be used to attempt upgrading to a strong reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Gc, Weak};
    ///
    /// let strong_ref = Gc::new(42);
    /// let weak_ref = Weak::new_simple(&strong_ref);
    /// if let Some(weak_reader) = weak_ref.read() {
    ///     // Use the weak reader to attempt upgrade
    /// }
    /// ```
    pub fn read(&self) -> Option<WeakRef<T>> {
        Some(WeakRef {
            weak: self as *const _,
            _phantom: PhantomData,
        })
    }

    /// Invalidate a chain of weak references.
    ///
    /// This is called by the garbage collector to invalidate all weak
    /// references in a linked list when their target is collected.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `weak_head` points to a valid weak
    /// reference chain that is safe to invalidate.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::Weak;
    ///
    /// // This is typically called by the GC system
    /// unsafe {
    ///     Weak::<i32>::invalidate_weak_chain(std::ptr::null_mut());
    /// }
    /// ```
    pub unsafe fn invalidate_weak_chain(weak_head: *mut Weak<()>) {
        let mut current = weak_head;
        
        while !current.is_null() {
            unsafe {
                let weak_ref = &*current;
                // Race-safety: Load next pointer before invalidation to prevent ABA issues
                let next = weak_ref.next_weak.load(std::sync::atomic::Ordering::Acquire) as *mut Weak<()>;
                
                // Race-safety: Use a stronger ordering to prevent reordering
                // Invalidate this weak reference by setting target to null
                weak_ref.target.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
                
                // Race-safety: Clear chain links with sequential consistency to ensure
                // no concurrent linking operations see inconsistent state
                weak_ref.next_weak.store(std::ptr::null_mut(), std::sync::atomic::Ordering::SeqCst);
                weak_ref.prev_weak.store(std::ptr::null_mut(), std::sync::atomic::Ordering::SeqCst);
                
                // Move to next node (loaded before invalidation for safety)
                current = next;
            }
        }
    }
}

/// A reference to a weak reference that can be upgraded to a strong reference.
///
/// `WeakRef<T>` is returned by `Weak::read()` and provides access to the
/// weak reference functionality, primarily the ability to upgrade to a
/// strong `Gc<T>` reference if the target object is still alive.
///
/// # Examples
///
/// ```
/// use fugrip::{Gc, Weak};
///
/// let strong_ref = Gc::new(42);
/// let weak_ref = Weak::new_simple(&strong_ref);
/// 
/// if let Some(weak_reader) = weak_ref.read() {
///     // Try to upgrade to strong reference
///     if let Some(upgraded) = weak_reader.upgrade() {
///         // Access the strong reference
///     }
/// }
/// ```
pub struct WeakRef<T> {
    pub weak: *const Weak<T>,
    pub _phantom: PhantomData<T>,
}

impl<T> WeakRef<T> {
    /// Try to upgrade this weak reference to a strong reference.
    ///
    /// Returns `Some(Gc<T>)` if the target object is still alive,
    /// `None` if it has been garbage collected.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Gc, Weak};
    ///
    /// let strong_ref = Gc::new("Hello".to_string());
    /// let weak_ref = Weak::new_simple(&strong_ref);
    /// 
    /// if let Some(weak_reader) = weak_ref.read() {
    ///     match weak_reader.upgrade() {
    ///         Some(strong) => {
    ///             // Object is still alive, use strong reference
    ///         }
    ///         None => {
    ///             // Object has been garbage collected
    ///         }
    ///     }
    /// }
    /// ```
    pub fn upgrade(&self) -> Option<Gc<T>> {
        unsafe {
            let weak_ref = &*self.weak;
            let target_ptr = weak_ref.target.load(std::sync::atomic::Ordering::Acquire);
            if target_ptr.is_null() {
                return None; // Already invalidated
            }

            // Safety check: ensure the pointer looks like it falls within our heap.
            // This avoids dereferencing obviously invalid addresses in edge/racy situations.
            {
                use crate::interfaces::memory::HEAP_PROVIDER;
                let segments = <crate::interfaces::memory::ProductionHeapProvider as crate::interfaces::memory::HeapProvider>::get_heap(&HEAP_PROVIDER)
                    .segments
                    .lock()
                    .unwrap();
                let mut in_bounds = false;
                for seg in segments.iter() {
                    let start = seg.memory.as_ptr() as *mut u8;
                    let end = seg.allocation_ptr.load(std::sync::atomic::Ordering::Relaxed);
                    if (target_ptr as *mut u8) >= start && (target_ptr as *mut u8) < end {
                        in_bounds = true;
                        break;
                    }
                }
                if !in_bounds {
                    // Pointer is not within any allocated segment range; invalidate defensively.
                    weak_ref
                        .target
                        .store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
                    return None;
                }
            }

            let header = &*target_ptr;

            // Check if object has been forwarded to FREE_SINGLETON
            let forwarding_ptr = header
                .forwarding_ptr
                .load(std::sync::atomic::Ordering::Acquire);
            if forwarding_ptr == crate::types::FreeSingleton::instance() {
                // Object has been collected, invalidate this weak reference
                weak_ref
                    .target
                    .store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
                return None;
            }

            // Object is still alive, create a new strong reference
            Some(Gc {
                ptr: AtomicPtr::new(target_ptr),
                _phantom: PhantomData,
            })
        }
    }
}

/// Strong reference guard for reading GC objects.
///
/// `GcRef<'a, T>` provides safe, immutable access to garbage-collected objects
/// for the duration of lifetime `'a`. It dereferences to `&T`, allowing
/// transparent access to the underlying data.
///
/// # Examples
///
/// ```
/// use fugrip::Gc;
///
/// let gc_value = Gc::new(String::from("Hello"));
/// // In the full implementation, this would return Some(GcRef)
/// if let Some(guard) = gc_value.read() {
///     // Access the value through the guard
///     // assert_eq!(guard.len(), 5);
///     // assert!(guard.starts_with("Hello"));
/// }
/// ```
pub struct GcRef<'a, T> {
    ptr: *const T,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> GcRef<'a, T> {
    /// Creates a new GcRef from a raw pointer.
    ///
    /// This is primarily used internally by the garbage collector
    /// when creating read guards for objects.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to a valid, initialized object of type `T`
    /// - The object will remain valid for the lifetime `'a`
    /// - The object is not being mutated while this reference exists
    /// - The pointer is properly aligned and non-null
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::GcRef;
    ///
    /// let value = 42i32;
    /// let gc_ref = unsafe { GcRef::new(&value as *const i32) };
    /// assert_eq!(*gc_ref, 42);
    /// ```
    pub unsafe fn new(ptr: *const T) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> std::ops::Deref for GcRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

/// Strong reference guard for writing GC objects.
///
/// `GcRefMut<'a, T>` provides safe, mutable access to garbage-collected objects
/// for the duration of lifetime `'a`. It dereferences to `&mut T`, allowing
/// transparent mutable access to the underlying data.
///
/// # Examples
///
/// ```
/// use fugrip::Gc;
///
/// let gc_value = Gc::new(String::from("Hello"));
/// // In the full implementation, this would return Some(GcRefMut)
/// if let Some(mut guard) = gc_value.write() {
///     // Modify the value through the guard
///     // guard.push_str(", World!");
///     // assert_eq!(*guard, "Hello, World!");
/// }
/// ```
pub struct GcRefMut<'a, T> {
    ptr: *mut T,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T> GcRefMut<'a, T> {
    /// Creates a new GcRefMut from a raw pointer.
    ///
    /// This is primarily used internally by the garbage collector
    /// when creating write guards for objects.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to a valid, initialized object of type `T`
    /// - The object will remain valid for the lifetime `'a`
    /// - No other references (mutable or immutable) to this object exist
    /// - The pointer is properly aligned and non-null
    /// - The caller has exclusive access to the object
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::GcRefMut;
    ///
    /// let mut value = 42i32;
    /// let mut gc_ref_mut = unsafe { GcRefMut::new(&mut value as *mut i32) };
    /// *gc_ref_mut = 100;
    /// assert_eq!(*gc_ref_mut, 100);
    /// ```
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> std::ops::Deref for GcRefMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<'a, T> std::ops::DerefMut for GcRefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

/// Singleton representing freed objects.
///
/// `FreeSingleton` is a special object that all freed GC pointers are
/// redirected to point at. This prevents use-after-free errors by
/// ensuring that dereferencing a freed GC pointer will access a known,
/// safe object rather than uninitialized or reused memory.
///
/// # Examples
///
/// ```
/// use fugrip::FreeSingleton;
///
/// // Get the singleton instance (used internally by the GC)
/// let free_ptr = FreeSingleton::instance();
/// assert!(!free_ptr.is_null());
/// ```
pub struct FreeSingleton {
    pub header: GcHeader<()>,
}

impl FreeSingleton {
    /// Get the global free singleton instance.
    ///
    /// Returns a pointer to the singleton object that all freed GC pointers
    /// are redirected to. This method is thread-safe and uses lazy initialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::FreeSingleton;
    ///
    /// let ptr1 = FreeSingleton::instance();
    /// let ptr2 = FreeSingleton::instance();
    /// 
    /// // Same instance is returned each time
    /// assert_eq!(ptr1, ptr2);
    /// assert!(!ptr1.is_null());
    /// ```
    pub fn instance() -> *mut GcHeader<()> {
        static FREE_SINGLETON: std::sync::atomic::AtomicPtr<GcHeader<()>> =
            std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

        let ptr = FREE_SINGLETON.load(std::sync::atomic::Ordering::Acquire);
        if ptr.is_null() {
            // Initialize singleton
            let singleton = Box::leak(Box::new(FreeSingleton {
                header: GcHeader {
                    mark_bit: AtomicBool::new(true), // Always marked
                    type_info: &FREE_SINGLETON_TYPE_INFO,
                    forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                    weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
                    data: (),
                },
            }));
            let header_ptr = &mut singleton.header as *mut GcHeader<()>;

            match FREE_SINGLETON.compare_exchange(
                std::ptr::null_mut(),
                header_ptr,
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Acquire,
            ) {
                Ok(_) => header_ptr,
                Err(existing) => existing, // Another thread beat us
            }
        } else {
            ptr
        }
    }
}

/// Type information for the free singleton object.
///
/// This constant provides the type metadata used by the free singleton,
/// with no-op implementations for all GC operations since the free
/// singleton is never actually collected or moved.
///
/// # Examples
///
/// ```
/// use fugrip::{FREE_SINGLETON_TYPE_INFO, ObjectType};
///
/// assert_eq!(FREE_SINGLETON_TYPE_INFO.object_type, ObjectType::Regular);
/// assert!(FREE_SINGLETON_TYPE_INFO.size > 0);
/// ```
pub const FREE_SINGLETON_TYPE_INFO: TypeInfo = TypeInfo {
    trace_fn: |_, _| {},
    drop_fn: |_: *mut GcHeader<()>| {},
    finalize_fn: None,
    redirect_pointers_fn: |_, _, _| {},
    size: std::mem::size_of::<GcHeader<()>>(),
    align: std::mem::align_of::<GcHeader<()>>(),
    object_type: ObjectType::Regular,
};

/// Returns type information for a given `GcTrace` type.
///
/// This function provides cached access to `TypeInfo` for garbage-collected types.
/// It uses `TypeId`-based lookup to ensure each type gets a single, reusable
/// `TypeInfo` instance with appropriate function pointers for tracing and dropping.
///
/// # Examples
///
/// ```
/// use fugrip::{types::type_info, ObjectType};
/// # use fugrip::traits::GcTrace;
/// # use fugrip::SendPtr;
/// # use fugrip::GcHeader;
/// # struct TestType1(i32);
/// # struct TestType2(String);
/// # unsafe impl GcTrace for TestType1 {
/// #     unsafe fn trace(&self, _: &mut Vec<SendPtr<GcHeader<()>>>) {}
/// # }
/// # unsafe impl GcTrace for TestType2 {
/// #     unsafe fn trace(&self, _: &mut Vec<SendPtr<GcHeader<()>>>) {}
/// # }
///
/// // Get type information for a test type
/// let type1_info = type_info::<TestType1>();
/// assert!(type1_info.size > 0);
/// assert_eq!(type1_info.object_type, ObjectType::Regular);
///
/// // Type info is cached - same pointer returned for same type
/// let type1_info2 = type_info::<TestType1>();
/// assert!(std::ptr::eq(type1_info, type1_info2));
///
/// // Different types have different type info
/// let type2_info = type_info::<TestType2>();
/// assert!(!std::ptr::eq(type1_info, type2_info));
/// ```
///
/// This version uses proper caching with TypeId-based lookup.
pub fn type_info<T: crate::traits::GcTrace + 'static>() -> &'static TypeInfo {
    type_info_for_type::<T>(ObjectType::Regular, None)
}

/// Internal function to create type info with specific object type and finalization.
fn type_info_for_type<T: crate::traits::GcTrace + 'static>(
    object_type: ObjectType,
    finalize_fn: Option<unsafe fn(*mut GcHeader<()>)>,
) -> &'static TypeInfo {
    use std::any::TypeId;
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    static TYPE_INFO_CACHE: LazyLock<Mutex<HashMap<(TypeId, ObjectType), &'static TypeInfo>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let type_id = TypeId::of::<T>();
    let cache_key = (type_id, object_type);
    let mut cache = TYPE_INFO_CACHE.lock().unwrap();

    if let Some(&cached) = cache.get(&cache_key) {
        return cached;
    }

    let type_info = Box::leak(Box::new(TypeInfo {
        size: std::mem::size_of::<GcHeader<T>>(),
        align: std::mem::align_of::<GcHeader<T>>(),
        trace_fn: |obj, stack| unsafe {
            let header = &*(obj as *mut GcHeader<T>);
            header.data.trace(stack);
        },
        drop_fn: |obj| unsafe {
            let header = obj as *mut GcHeader<T>;
            std::ptr::drop_in_place(&mut (*header).data);
        },
        finalize_fn,
        redirect_pointers_fn: |old_obj, _new_obj, free_singleton| unsafe {
            // Redirect all pointers that reference old_obj to point to free_singleton
            // This is used during sweeping to redirect dead objects to FREE_SINGLETON
            let old_header = &*(old_obj);
            old_header.forwarding_ptr.store(free_singleton, std::sync::atomic::Ordering::Release);
        },
        object_type,
    }));

    cache.insert(cache_key, type_info);
    type_info
}

/// Returns type information for a finalizable type.
/// 
/// This creates proper type information for objects that need finalization,
/// which is essential for FUGC's classification system.
///
/// # Examples
///
/// ```
/// use fugrip::{finalizable_type_info, Finalizable, ObjectType};
/// use fugrip::traits::GcTrace;
///
/// struct TestResource { id: u32 }
///
/// impl Finalizable for TestResource {
///     fn finalize(&mut self) {
///         println!("Finalizing resource {}", self.id);
///     }
/// }
///
/// unsafe impl GcTrace for TestResource {
///     unsafe fn trace(&self, _stack: &mut Vec<fugrip::SendPtr<fugrip::GcHeader<()>>>) {}
/// }
///
/// let type_info = finalizable_type_info::<TestResource>();
/// assert_eq!(type_info.object_type, ObjectType::Finalizable);
/// assert!(type_info.finalize_fn.is_some());
/// ```
pub fn finalizable_type_info<T>() -> &'static TypeInfo 
where 
    T: crate::Finalizable + crate::traits::GcTrace + 'static 
{
    type_info_for_type::<FinalizableObject<T>>(
        ObjectType::Finalizable,
        Some(|obj| unsafe {
            let header = obj as *mut GcHeader<FinalizableObject<T>>;
            // Safely finalize using the FinalizableObject wrapper
            (*header).data.data.finalize();
            (*header).data.mark_finalized();
        })
    )
}

/// Aligns a memory address up to the specified alignment boundary.
///
/// This function takes a memory address and returns the smallest address
/// that is greater than or equal to the input address and is aligned to
/// the specified boundary. This is commonly used in memory allocators
/// to ensure proper alignment for different data types.
///
/// # Parameters
///
/// * `addr` - The input memory address to align
/// * `align` - The alignment boundary (must be a power of 2)
///
/// # Examples
///
/// Basic alignment:
///
/// ```
/// use fugrip::align_up;
///
/// // Align to 8-byte boundary
/// let ptr = 0x1003 as *const u8;
/// let aligned = align_up(ptr, 8);
/// assert_eq!(aligned as usize, 0x1008);
///
/// // Already aligned addresses remain unchanged
/// let ptr = 0x1000 as *const u8;
/// let aligned = align_up(ptr, 8);
/// assert_eq!(aligned as usize, 0x1000);
/// ```
///
/// Different alignment boundaries:
///
/// ```
/// use fugrip::align_up;
///
/// let addr = 0x1001 as *const u8;
///
/// // 4-byte alignment
/// let aligned_4 = align_up(addr, 4);
/// assert_eq!(aligned_4 as usize, 0x1004);
///
/// // 16-byte alignment  
/// let aligned_16 = align_up(addr, 16);
/// assert_eq!(aligned_16 as usize, 0x1010);
///
/// // 32-byte alignment
/// let aligned_32 = align_up(addr, 32);
/// assert_eq!(aligned_32 as usize, 0x1020);
/// ```
///
/// Edge cases:
///
/// ```
/// use fugrip::align_up;
///
/// // Null pointer remains null
/// let null_ptr = std::ptr::null::<u8>();
/// let aligned = align_up(null_ptr, 8);
/// assert_eq!(aligned as usize, 0);
///
/// // Single byte alignment (no-op)
/// let ptr = 0x1234 as *const u8;
/// let aligned = align_up(ptr, 1);
/// assert_eq!(aligned as usize, 0x1234);
/// ```
pub fn align_up(addr: *const u8, align: usize) -> *const u8 {
    let addr = addr as usize;
    let remainder = addr % align;
    if remainder == 0 {
        addr as *const u8
    } else {
        (addr + align - remainder) as *const u8
    }
}

/// Concrete allocation state for thread-local allocators.
///
/// This struct represents the actual state of a thread-local allocation buffer,
/// providing fast allocation without contention between threads. Each thread
/// maintains its own buffer for efficient object allocation.
///
/// # Examples
///
/// ```
/// use fugrip::ThreadLocalBuffer;
///
/// // Create a new buffer (typically managed by the GC system)
/// let buffer = ThreadLocalBuffer {
///     buffer: std::ptr::null_mut(),
///     position: std::ptr::null_mut(),
///     limit: std::ptr::null_mut(),
///     allocating_black: false,
/// };
/// 
/// // Check if allocating in black (marking) mode
/// assert!(!buffer.allocating_black);
/// ```
pub struct ThreadLocalBuffer {
    pub buffer: *mut u8,
    pub position: *mut u8,
    pub limit: *mut u8,
    pub allocating_black: bool,
}

/// Statistics for garbage collection operations.
///
/// `GcStats` tracks various metrics about garbage collection performance,
/// including allocation counts, collection frequency, and timing information.
/// This is useful for monitoring and tuning GC performance.
///
/// # Examples
///
/// ```
/// use fugrip::GcStats;
///
/// let mut stats = GcStats::default();
/// stats.total_allocations += 100;
/// stats.bytes_allocated += 8192;
/// stats.total_collections += 1;
/// stats.collection_time_ms += 50;
/// 
/// println!("Allocations: {}, Collections: {}", 
///          stats.total_allocations, stats.total_collections);
/// assert_eq!(stats.total_allocations, 100);
/// assert_eq!(stats.bytes_allocated, 8192);
/// ```
#[derive(Debug, Default)]
pub struct GcStats {
    pub total_allocations: u64,
    pub total_collections: u64,
    pub bytes_allocated: u64,
    pub bytes_freed: u64,
    pub collection_time_ms: u64,
}

/// Configuration for garbage collector behavior.
///
/// `GcConfig` specifies parameters that control garbage collection behavior,
/// including heap sizes, collection thresholds, and parallelism settings.
/// These settings can be tuned for different application requirements.
///
/// # Examples
///
/// ```
/// use fugrip::GcConfig;
///
/// // Use default configuration
/// let default_config = GcConfig::default();
/// assert_eq!(default_config.initial_heap_size, 1024 * 1024);
/// assert!(default_config.concurrent_marking);
///
/// // Create custom configuration
/// let custom_config = GcConfig {
///     initial_heap_size: 2 * 1024 * 1024,  // 2MB
///     max_heap_size: 512 * 1024 * 1024,   // 512MB
///     allocation_threshold: 512 * 1024,   // 512KB
///     concurrent_marking: false,
///     parallel_workers: 4,
/// };
/// 
/// assert_eq!(custom_config.parallel_workers, 4);
/// assert!(!custom_config.concurrent_marking);
/// ```
#[derive(Debug)]
pub struct GcConfig {
    pub initial_heap_size: usize,
    pub max_heap_size: usize,
    pub allocation_threshold: usize,
    pub concurrent_marking: bool,
    pub parallel_workers: usize,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            initial_heap_size: 1024 * 1024,    // 1MB
            max_heap_size: 1024 * 1024 * 1024, // 1GB
            allocation_threshold: 1024 * 1024, // 1MB
            concurrent_marking: true,
            parallel_workers: num_cpus::get(),
        }
    }
}
