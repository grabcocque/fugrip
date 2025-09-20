//! Weak reference implementation with automatic nulling

use crate::frontend::types::ObjectReference;
use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Weak reference implementation with automatic nulling
///
/// This provides Java-style weak references that are automatically nulled
/// when their referenced object is collected.
#[derive(Debug)]
pub struct WeakReference<T> {
    /// The weak pointer to the object
    weak_ref: Weak<T>,
    /// Reference to the object for GC coordination
    object_ref: Option<ObjectReference>,
    /// Flag indicating if the reference has been nulled
    is_nulled: AtomicBool,
    /// Creation timestamp for debugging
    created_at: Instant,
}

impl<T> WeakReference<T> {
    /// Create a new weak reference to an object
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    /// assert!(weak_ref.get().is_some());
    /// ```
    pub fn new(strong_ref: Arc<T>, object_ref: Option<ObjectReference>) -> Self {
        Self {
            weak_ref: Arc::downgrade(&strong_ref),
            object_ref,
            is_nulled: AtomicBool::new(false),
            created_at: Instant::now(),
        }
    }

    /// Get the referenced object if it's still alive
    ///
    /// Returns None if the object has been collected or explicitly nulled.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new("hello");
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    ///
    /// assert_eq!(weak_ref.get().as_deref(), Some(&"hello"));
    /// drop(strong_ref);
    /// assert!(weak_ref.get().is_none());
    /// ```
    pub fn get(&self) -> Option<Arc<T>> {
        if self.is_nulled.load(Ordering::Acquire) {
            return None;
        }
        self.weak_ref.upgrade()
    }

    /// Check if the weak reference is still valid
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    ///
    /// assert!(weak_ref.is_valid());
    /// drop(strong_ref);
    /// assert!(!weak_ref.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        !self.is_nulled.load(Ordering::Acquire) && self.weak_ref.upgrade().is_some()
    }

    /// Explicitly null this weak reference
    ///
    /// This is called by the GC when the referenced object is collected.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    ///
    /// assert!(weak_ref.is_valid());
    /// weak_ref.null();
    /// assert!(!weak_ref.is_valid());
    /// ```
    pub fn null(&self) {
        self.is_nulled.store(true, Ordering::Release);
    }

    /// Get the object reference (if any) for GC coordination
    pub fn object_ref(&self) -> Option<ObjectReference> {
        self.object_ref
    }

    /// Alternative name for object_ref for compatibility
    pub fn object_reference(&self) -> Option<ObjectReference> {
        self.object_ref
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Get the age of this weak reference
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl<T> Clone for WeakReference<T> {
    fn clone(&self) -> Self {
        Self {
            weak_ref: self.weak_ref.clone(),
            object_ref: self.object_ref,
            is_nulled: AtomicBool::new(self.is_nulled.load(Ordering::Acquire)),
            created_at: self.created_at,
        }
    }
}

/// Trait for type-erased weak reference management
pub trait WeakRefTrait: Send + Sync {
    /// Null this weak reference
    fn null(&self);
    /// Check if the reference is still valid
    fn is_valid(&self) -> bool;
    /// Get object reference for GC coordination
    fn object_ref(&self) -> Option<ObjectReference>;
}

impl<T: Send + Sync + 'static> WeakRefTrait for WeakReference<T> {
    fn null(&self) {
        self.null();
    }

    fn is_valid(&self) -> bool {
        self.is_valid()
    }

    fn object_ref(&self) -> Option<ObjectReference> {
        self.object_ref()
    }
}

/// Registry for tracking weak references
///
/// This manages all weak references in the system and provides
/// functionality to null them when their referenced objects are collected.
pub struct WeakRefRegistry {
    /// Map from object reference to weak references pointing to it
    refs_by_object: DashMap<ObjectReference, Vec<Arc<dyn WeakRefTrait>>>,
    /// Total number of weak references registered
    total_registered: AtomicUsize,
    /// Total number of weak references nulled
    total_nulled: AtomicUsize,
    /// Total number of weak references cleaned up
    total_cleaned: AtomicUsize,
}

impl WeakRefRegistry {
    /// Create a new weak reference registry
    pub fn new() -> Self {
        Self {
            refs_by_object: DashMap::new(),
            total_registered: AtomicUsize::new(0),
            total_nulled: AtomicUsize::new(0),
            total_cleaned: AtomicUsize::new(0),
        }
    }

    /// Register a weak reference for an object
    ///
    /// # Arguments
    /// * `object_ref` - The object being weakly referenced
    /// * `weak_ref` - The weak reference to register
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::{WeakRefRegistry, WeakReference};
    /// use crate::frontend::types::ObjectReference;
    /// use std::sync::Arc;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let obj = ObjectReference::from_raw_address(unsafe {
    ///     mmtk::util::Address::from_usize(0x1000)
    /// }).unwrap();
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), Some(obj));
    ///
    /// registry.register(obj, Arc::new(weak_ref));
    /// ```
    pub fn register(&self, object_ref: ObjectReference, weak_ref: Arc<dyn WeakRefTrait>) {
        self.refs_by_object
            .entry(object_ref)
            .or_insert_with(Vec::new)
            .push(weak_ref);
        self.total_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// Null all weak references to a collected object
    ///
    /// This is called by the GC when an object is about to be collected.
    ///
    /// # Arguments
    /// * `object_ref` - The object that was collected
    ///
    /// # Returns
    /// Number of weak references that were nulled
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakRefRegistry;
    /// use crate::frontend::types::ObjectReference;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let obj = ObjectReference::from_raw_address(unsafe {
    ///     mmtk::util::Address::from_usize(0x1000)
    /// }).unwrap();
    ///
    /// let nulled = registry.null_refs_for_object(obj);
    /// println!("Nulled {} weak references", nulled);
    /// ```
    pub fn null_refs_for_object(&self, object_ref: ObjectReference) -> usize {
        if let Some((_, weak_refs)) = self.refs_by_object.remove(&object_ref) {
            let count = weak_refs.len();
            for weak_ref in weak_refs {
                weak_ref.null();
            }
            self.total_nulled.fetch_add(count, Ordering::Relaxed);
            count
        } else {
            0
        }
    }

    /// Clean up invalid weak references
    ///
    /// This removes weak references that are no longer valid from the registry.
    /// Should be called periodically to prevent memory leaks.
    ///
    /// # Returns
    /// Number of invalid references cleaned up
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakRefRegistry;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let cleaned = registry.cleanup_invalid_refs();
    /// println!("Cleaned up {} invalid references", cleaned);
    /// ```
    pub fn cleanup_invalid_refs(&self) -> usize {
        let mut cleaned_count = 0;

        // Collect object refs that need cleanup
        let mut to_remove = Vec::new();
        let mut to_update = Vec::new();

        for mut entry in self.refs_by_object.iter_mut() {
            let object_ref = *entry.key();
            let weak_refs = entry.value_mut();

            // Filter out invalid references
            let original_len = weak_refs.len();
            weak_refs.retain(|weak_ref| weak_ref.is_valid());
            let new_len = weak_refs.len();

            cleaned_count += original_len - new_len;

            if weak_refs.is_empty() {
                to_remove.push(object_ref);
            } else if new_len != original_len {
                to_update.push(object_ref);
            }
        }

        // Update cleaned count
        if cleaned_count > 0 {
            self.total_cleaned
                .fetch_add(cleaned_count, Ordering::Relaxed);
        }

        // Remove empty entries
        for object_ref in to_remove {
            self.refs_by_object.remove(&object_ref);
        }

        cleaned_count
    }

    /// Get statistics about weak references
    ///
    /// # Returns
    /// (total_registered, total_nulled, currently_active)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakRefRegistry;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let (registered, nulled, active) = registry.get_stats();
    /// println!("Registered: {}, Nulled: {}, Active: {}", registered, nulled, active);
    /// ```
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let total_registered = self.total_registered.load(Ordering::Relaxed);
        let total_nulled = self.total_nulled.load(Ordering::Relaxed);
        let currently_active = self.refs_by_object.len();

        (total_registered, total_nulled, currently_active)
    }

    /// Get total number of registered weak references
    pub fn total_count(&self) -> usize {
        self.total_registered.load(Ordering::Relaxed)
    }

    /// Get number of nulled weak references
    pub fn nulled_count(&self) -> usize {
        self.total_nulled.load(Ordering::Relaxed)
    }

    /// Get number of cleaned weak references
    pub fn cleaned_count(&self) -> usize {
        self.total_cleaned.load(Ordering::Relaxed)
    }

    /// Get number of currently active weak references
    pub fn active_count(&self) -> usize {
        self.refs_by_object.len()
    }

    /// Get the number of weak references for a specific object
    ///
    /// # Arguments
    /// * `object_ref` - The object to check
    ///
    /// # Returns
    /// Number of weak references pointing to this object
    pub fn refs_count_for_object(&self, object_ref: ObjectReference) -> usize {
        self.refs_by_object
            .get(&object_ref)
            .map(|refs| refs.len())
            .unwrap_or(0)
    }

    /// Alias for null_refs_for_object for backwards compatibility
    pub fn null_references_to_object(&self, object_ref: ObjectReference) -> usize {
        self.null_refs_for_object(object_ref)
    }

    /// Alias for cleanup_invalid_refs for backwards compatibility
    pub fn cleanup_invalid_references(&self) -> usize {
        self.cleanup_invalid_refs()
    }
}

impl Default for WeakRefRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for weak reference tracking
#[derive(Debug, Clone)]
pub struct WeakRefStats {
    /// Total weak references registered
    pub total_registered: usize,
    /// Total weak references nulled
    pub total_nulled: usize,
    /// Total weak references cleaned up
    pub total_cleaned: usize,
    /// Currently active weak references
    pub currently_active: usize,
}
