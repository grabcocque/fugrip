//! Weak reference infrastructure hooks.

use std::sync::atomic::{AtomicPtr, Ordering};
use crate::{core::{ObjectHeader, Gc, ObjectFlags}};

/// Header for objects that contain weak references
#[derive(Default)]
pub struct WeakRefHeader {
    pub header: ObjectHeader,
    /// Pointer to the weak reference target
    pub weak_target: AtomicPtr<u8>,
}

impl WeakRefHeader {
    pub fn new(layout_id: crate::core::LayoutId, body_size: usize) -> Self {
        let mut header = ObjectHeader::default();
        header.layout_id = layout_id;
        header.body_size = body_size;
        header.flags = ObjectFlags::HAS_WEAK_REFS;

        Self {
            header,
            weak_target: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub fn set_target(&self, target: *mut u8) {
        self.weak_target.store(target, Ordering::SeqCst);
    }

    pub fn get_target(&self) -> *mut u8 {
        self.weak_target.load(Ordering::SeqCst)
    }

    pub fn clear_target(&self) {
        self.weak_target.store(std::ptr::null_mut(), Ordering::SeqCst);
    }
}

/// Weak reference to a GC-managed object
pub struct WeakRef<T> {
    target: AtomicPtr<u8>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> WeakRef<T> {
    /// Create a new weak reference to the given object
    pub fn new(target: Gc<T>) -> Self {
        Self {
            target: AtomicPtr::new(target.as_ptr() as *mut u8),
            _marker: std::marker::PhantomData,
        }
    }

    /// Try to upgrade the weak reference to a strong reference
    pub fn upgrade(&self) -> Option<Gc<T>> {
        let ptr = self.target.load(Ordering::SeqCst);
        if ptr.is_null() {
            None
        } else {
            // In a real implementation, we would check if the object is still alive
            // For now, assume it's valid
            Some(Gc::from_raw(ptr as *mut T))
        }
    }

    /// Check if the weak reference is still valid
    pub fn is_alive(&self) -> bool {
        !self.target.load(Ordering::SeqCst).is_null()
    }

    /// Clear the weak reference
    pub fn clear(&self) {
        self.target.store(std::ptr::null_mut(), Ordering::SeqCst);
    }
}

impl<T> Default for WeakRef<T> {
    fn default() -> Self {
        Self {
            target: AtomicPtr::new(std::ptr::null_mut()),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Clone for WeakRef<T> {
    fn clone(&self) -> Self {
        Self {
            target: AtomicPtr::new(self.target.load(Ordering::SeqCst)),
            _marker: std::marker::PhantomData,
        }
    }
}

/// Weak reference registry for managing weak references during GC
pub struct WeakRefRegistry {
    weak_refs: std::sync::Mutex<Vec<(*mut u8, *mut AtomicPtr<u8>)>>,
}

impl WeakRefRegistry {
    pub fn new() -> Self {
        Self {
            weak_refs: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Register a weak reference for processing during GC
    pub fn register_weak_ref(&self, weak_ref_obj: *mut u8, target_slot: *mut AtomicPtr<u8>) {
        self.weak_refs.lock().unwrap().push((weak_ref_obj, target_slot));
    }

    /// Process all weak references during GC, clearing those whose targets are dead
    pub fn process_weak_refs(&self, is_alive: impl Fn(*mut u8) -> bool) {
        let refs = self.weak_refs.lock().unwrap();
        for (_weak_ref_obj, target_slot) in refs.iter() {
            unsafe {
                if let Some(atomic_ptr) = target_slot.as_ref() {
                    let target = atomic_ptr.load(Ordering::SeqCst);
                    if !target.is_null() && !is_alive(target) {
                        atomic_ptr.store(std::ptr::null_mut(), Ordering::SeqCst);
                    }
                }
            }
        }
    }
}

impl Default for WeakRefRegistry {
    fn default() -> Self {
        Self::new()
    }
}
