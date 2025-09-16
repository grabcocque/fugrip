//! Weak reference infrastructure hooks.

use crate::core::{Gc, ObjectFlags, ObjectHeader};
use mmtk::util::ObjectReference;
use mmtk::vm::Finalizable;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Header for objects that contain weak references
#[derive(Default, Debug)]
pub struct WeakRefHeader {
    pub header: ObjectHeader,
    /// Pointer to the weak reference target
    pub weak_target: AtomicPtr<u8>,
}

impl WeakRefHeader {
    pub fn new(layout_id: crate::core::LayoutId, body_size: usize) -> Self {
        let header = ObjectHeader {
            layout_id,
            body_size,
            flags: ObjectFlags::HAS_WEAK_REFS,
            ..Default::default()
        };

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
        self.weak_target
            .store(std::ptr::null_mut(), Ordering::SeqCst);
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
        self.weak_refs
            .lock()
            .unwrap()
            .push((weak_ref_obj, target_slot));
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

impl Finalizable for WeakRefHeader {
    fn get_reference(&self) -> ObjectReference {
        // For weak references, we return the target if it exists
        let target = self.get_target();
        if target.is_null() {
            // For cleared weak references, return a dummy object reference
            // This should not happen in practice as cleared references shouldn't be finalized
            unsafe {
                ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(
                    0xDEADBEEF,
                ))
            }
        } else {
            // This is a simplification - in a real implementation we'd need to convert
            // the raw pointer to an ObjectReference properly
            ObjectReference::from_raw_address(mmtk::util::Address::from_ptr(target))
                .unwrap_or_else(|| panic!("Invalid object reference"))
        }
    }

    fn set_reference(&mut self, object: ObjectReference) {
        if object.to_raw_address().as_usize() == 0xDEADBEEF {
            self.clear_target();
        } else {
            self.set_target(object.to_raw_address().to_mut_ptr::<u8>());
        }
    }

    fn keep_alive<E: mmtk::scheduler::ProcessEdgesWork>(&mut self, _trace: &mut E) {
        // For weak references, we don't need to keep anything alive
        // The weak reference itself should be traced by the normal object tracing
    }
}

// Make WeakRefHeader Send so it can be used as Finalizable
unsafe impl Send for WeakRefHeader {}
