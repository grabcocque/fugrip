//! Weak reference infrastructure hooks.
//!
//! This module provides weak reference support for the FUGC garbage collector,
//! allowing objects to reference other objects without keeping them alive.
//!
//! # Examples
//!
//! ```
//! use fugrip::weak::{WeakRef, WeakRefRegistry, WeakRefHeader};
//! use fugrip::core::{Gc, LayoutId};
//!
//! // Create a weak reference header
//! let weak_header = WeakRefHeader::new(LayoutId(1), 64);
//! assert!(weak_header.header.flags.contains(fugrip::core::ObjectFlags::HAS_WEAK_REFS));
//!
//! // Create and use weak references
//! let weak_ref: WeakRef<i32> = WeakRef::default();
//! assert!(!weak_ref.is_alive());
//!
//! // Create a registry for managing weak references
//! let registry = WeakRefRegistry::new();
//! ```

use crate::core::{Gc, ObjectFlags, ObjectHeader};
use mmtk::util::ObjectReference;
use mmtk::vm::Finalizable;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Header for objects that contain weak references
///
/// # Examples
///
/// ```
/// use fugrip::weak::WeakRefHeader;
/// use fugrip::core::{LayoutId, ObjectFlags};
///
/// // Create a weak reference header
/// let weak_header = WeakRefHeader::new(LayoutId(42), 128);
/// assert!(weak_header.header.flags.contains(ObjectFlags::HAS_WEAK_REFS));
/// assert_eq!(weak_header.header.body_size, 128);
/// assert!(weak_header.get_target().is_null());
///
/// // Set and get target
/// let target_ptr = 0x12345678 as *mut u8;
/// weak_header.set_target(target_ptr);
/// assert_eq!(weak_header.get_target(), target_ptr);
///
/// // Clear target
/// weak_header.clear_target();
/// assert!(weak_header.get_target().is_null());
/// ```
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
///
/// # Examples
///
/// ```
/// use fugrip::weak::WeakRef;
/// use fugrip::core::Gc;
///
/// // Create a default weak reference
/// let weak_ref: WeakRef<i32> = WeakRef::default();
/// assert!(!weak_ref.is_alive());
/// assert!(weak_ref.upgrade().is_none());
///
/// // Create from a Gc pointer (demonstration only)
/// let gc_ptr: Gc<i32> = Gc::new();
/// let weak_from_gc = WeakRef::new(gc_ptr);
///
/// // Clone weak references
/// let weak_clone = weak_ref.clone();
/// assert_eq!(weak_ref.is_alive(), weak_clone.is_alive());
///
/// // Clear weak reference
/// weak_ref.clear();
/// assert!(!weak_ref.is_alive());
/// ```
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
///
/// # Examples
///
/// ```
/// use fugrip::weak::WeakRefRegistry;
/// use std::sync::atomic::AtomicPtr;
///
/// // Create a registry
/// let registry = WeakRefRegistry::new();
/// let default_registry = WeakRefRegistry::default();
///
/// // Register weak references (demonstration with dummy pointers)
/// let weak_ref_obj = 0x1000 as *mut u8;
/// let target_slot = Box::into_raw(Box::new(AtomicPtr::new(0x2000 as *mut u8)));
///
/// registry.register_weak_ref(weak_ref_obj, target_slot);
///
/// // Process weak references during GC
/// registry.process_weak_refs(|ptr| {
///     // Mock liveness check - always return false to clear all weak refs
///     false
/// });
///
/// // Clean up
/// unsafe { drop(Box::from_raw(target_slot)); }
/// ```
pub struct WeakRefRegistry {
    weak_refs: Mutex<Vec<(*mut u8, *mut AtomicPtr<u8>)>>,
}

impl WeakRefRegistry {
    pub fn new() -> Self {
        Self {
            weak_refs: Mutex::new(Vec::new()),
        }
    }

    /// Register a weak reference for processing during GC
    pub fn register_weak_ref(&self, weak_ref_obj: *mut u8, target_slot: *mut AtomicPtr<u8>) {
        self.weak_refs.lock().push((weak_ref_obj, target_slot));
    }

    /// Process all weak references during GC, clearing those whose targets are dead
    pub fn process_weak_refs(&self, is_alive: impl Fn(*mut u8) -> bool) {
        let refs = self.weak_refs.lock();
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
                    0xDEADBEE8, // Aligned dummy address
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
        if object.to_raw_address().as_usize() == 0xDEADBEF0 {
            // Use aligned address
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Gc, LayoutId};
    use std::sync::atomic::{AtomicPtr, Ordering};

    fn heap_address(offset: usize) -> mmtk::util::Address {
        unsafe { mmtk::util::Address::from_usize(0x1000_0000 + offset) }
    }

    #[test]
    fn weak_ref_header_tracks_target_pointer() {
        let header = WeakRefHeader::new(LayoutId(7), 64);
        assert!(header.get_target().is_null());

        let ptr = 0xDEADusize as *mut u8;
        header.set_target(ptr);
        assert_eq!(header.get_target(), ptr);

        header.clear_target();
        assert!(header.get_target().is_null());
    }

    #[test]
    fn weak_ref_upgrade_and_clear_behavior() {
        let strong = Box::into_raw(Box::new(42_i32));
        let gc = Gc::from_raw(strong);
        let weak = WeakRef::new(gc);

        let upgraded = weak.upgrade().expect("weak reference should upgrade");
        assert_eq!(upgraded.as_ptr(), strong);

        weak.clear();
        assert!(weak.upgrade().is_none());

        // Safety: reclaim the allocation we produced for the test
        unsafe { drop(Box::from_raw(strong)) };
    }

    #[test]
    fn weak_ref_registry_clears_dead_entries() {
        let registry = WeakRefRegistry::new();
        let weak_obj = 0x2000usize as *mut u8;
        let atomic = Box::new(AtomicPtr::new(0x3000usize as *mut u8));
        let raw_slot = Box::into_raw(atomic);

        registry.register_weak_ref(weak_obj, raw_slot);
        registry.process_weak_refs(|_| false); // nothing is alive

        unsafe {
            let slot = &*raw_slot;
            assert!(slot.load(Ordering::SeqCst).is_null());
        }

        // Clean up slot allocation
        unsafe {
            drop(Box::from_raw(raw_slot));
        }
    }

    #[test]
    fn finalizable_contract_maps_object_references() {
        let mut header = WeakRefHeader::new(LayoutId(9), 32);
        let addr = heap_address(0);
        let referent = ObjectReference::from_raw_address(addr).expect("valid reference");

        header.set_target(addr.to_mut_ptr());
        assert_eq!(header.get_reference(), referent);

        // Setting sentinel clears the reference
        let sentinel = unsafe {
            ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0xDEADBEF0))
        };
        header.set_reference(sentinel);
        assert!(header.get_target().is_null());
    }

    #[test]
    fn test_weak_ref_header_creation() {
        let weak_header = WeakRefHeader::new(LayoutId(42), 128);
        assert!(
            weak_header
                .header
                .flags
                .contains(ObjectFlags::HAS_WEAK_REFS)
        );
        assert_eq!(weak_header.header.layout_id, LayoutId(42));
        assert_eq!(weak_header.header.body_size, 128);
        assert!(weak_header.get_target().is_null());
    }

    #[test]
    fn test_weak_ref_header_target_operations() {
        let weak_header = WeakRefHeader::new(LayoutId(1), 64);

        // Initially null
        assert!(weak_header.get_target().is_null());

        // Set target
        let target_ptr = 0x12345678 as *mut u8;
        weak_header.set_target(target_ptr);
        assert_eq!(weak_header.get_target(), target_ptr);

        // Clear target
        weak_header.clear_target();
        assert!(weak_header.get_target().is_null());
    }

    #[test]
    fn test_weak_ref_header_default() {
        let weak_header = WeakRefHeader::default();
        assert!(weak_header.get_target().is_null());
        assert_eq!(weak_header.header.body_size, 0);
    }

    #[test]
    fn test_weak_ref_creation() {
        let weak_ref: WeakRef<i32> = WeakRef::default();
        assert!(!weak_ref.is_alive());
        assert!(weak_ref.upgrade().is_none());
    }

    #[test]
    fn test_weak_ref_from_gc() {
        let gc_ptr: Gc<i32> = Gc::new(); // null Gc
        let weak_ref = WeakRef::new(gc_ptr);
        assert!(!weak_ref.is_alive());

        let gc_ptr_valid = Gc::from_raw(0x1000 as *mut i32);
        let weak_ref_valid = WeakRef::new(gc_ptr_valid);
        assert!(weak_ref_valid.is_alive());
    }

    #[test]
    fn test_weak_ref_upgrade() {
        let weak_ref: WeakRef<i32> = WeakRef::default();
        assert!(weak_ref.upgrade().is_none());

        let gc_ptr = Gc::from_raw(0x2000 as *mut i32);
        let weak_ref_valid = WeakRef::new(gc_ptr);
        let upgraded = weak_ref_valid.upgrade();
        assert!(upgraded.is_some());
        assert_eq!(upgraded.unwrap().as_ptr(), 0x2000 as *mut i32);
    }

    #[test]
    fn test_weak_ref_clear() {
        let gc_ptr = Gc::from_raw(0x3000 as *mut i32);
        let weak_ref = WeakRef::new(gc_ptr);
        assert!(weak_ref.is_alive());

        weak_ref.clear();
        assert!(!weak_ref.is_alive());
        assert!(weak_ref.upgrade().is_none());
    }

    #[test]
    fn test_weak_ref_clone() {
        let gc_ptr = Gc::from_raw(0x4000 as *mut i32);
        let weak_ref = WeakRef::new(gc_ptr);
        let weak_clone = weak_ref.clone();

        assert_eq!(weak_ref.is_alive(), weak_clone.is_alive());

        weak_ref.clear();
        assert!(!weak_ref.is_alive());
        // Clone should still be alive (independent atomic ptr)
        assert!(weak_clone.is_alive());
    }

    #[test]
    fn test_weak_ref_registry_creation() {
        let registry = WeakRefRegistry::new();
        let default_registry = WeakRefRegistry::default();

        // Both should be valid instances
        // We can't easily test internal state, but we can test methods don't panic
        registry.process_weak_refs(|_| true);
        default_registry.process_weak_refs(|_| true);
    }

    #[test]
    fn test_weak_ref_registry_register() {
        let registry = WeakRefRegistry::new();

        let weak_ref_obj = 0x1000 as *mut u8;
        let target_slot = Box::into_raw(Box::new(AtomicPtr::new(0x2000 as *mut u8)));

        registry.register_weak_ref(weak_ref_obj, target_slot);

        // Clean up
        unsafe {
            drop(Box::from_raw(target_slot));
        }
    }

    #[test]
    fn test_weak_ref_registry_process() {
        let registry = WeakRefRegistry::new();

        // Create some weak references
        let target1 = Box::into_raw(Box::new(AtomicPtr::new(0x1000 as *mut u8)));
        let target2 = Box::into_raw(Box::new(AtomicPtr::new(0x2000 as *mut u8)));
        let target3 = Box::into_raw(Box::new(AtomicPtr::new(std::ptr::null_mut())));

        registry.register_weak_ref(0x100 as *mut u8, target1);
        registry.register_weak_ref(0x200 as *mut u8, target2);
        registry.register_weak_ref(0x300 as *mut u8, target3);

        // Process with a mock liveness check (clear all non-null refs)
        registry.process_weak_refs(|ptr| {
            // Mock: only 0x1000 is alive
            ptr == 0x1000 as *mut u8
        });

        // Check results
        unsafe {
            assert_eq!((*target1).load(Ordering::SeqCst), 0x1000 as *mut u8); // Still alive
            assert_eq!((*target2).load(Ordering::SeqCst), std::ptr::null_mut()); // Cleared
            assert_eq!((*target3).load(Ordering::SeqCst), std::ptr::null_mut()); // Was already null

            // Clean up
            drop(Box::from_raw(target1));
            drop(Box::from_raw(target2));
            drop(Box::from_raw(target3));
        }
    }

    #[test]
    fn test_weak_ref_header_finalizable() {
        let mut weak_header = WeakRefHeader::new(LayoutId(1), 64);

        // Test get_reference with null target
        let ref1 = weak_header.get_reference();
        assert_eq!(ref1.to_raw_address().as_usize(), 0xDEADBEE8);

        // Test get_reference with valid target
        weak_header.set_target(0x5000 as *mut u8);
        let ref2 = weak_header.get_reference();
        assert_eq!(ref2.to_raw_address().as_usize(), 0x5000);

        // Test set_reference
        let new_ref = unsafe {
            ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x6000))
        };
        weak_header.set_reference(new_ref);
        assert_eq!(weak_header.get_target(), 0x6000 as *mut u8);

        // Test clearing with special value (aligned)
        let clear_ref = unsafe {
            ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0xDEADBEF0), // Aligned address
            )
        };
        weak_header.set_reference(clear_ref);
        // set_reference checks for 0xDEADBEF0 to clear the target
        assert!(weak_header.get_target().is_null());
    }
}
