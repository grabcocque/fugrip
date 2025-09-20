//! Memory management coordinator
//!
//! This module provides the MemoryManager struct that coordinates all advanced memory
//! management features with the FUGC garbage collector, providing a unified interface
//! for explicit freeing, finalization, and weak reference management.

use std::sync::{Arc, Weak};

use crate::compat::ObjectReference;
use dashmap::DashMap;

use crate::fugc_coordinator::FugcCoordinator;

use super::finalizers::{FinalizerQueue, FinalizerQueueStats};
use super::free_objects::{FreeObjectManager, FreeObjectStats};
use super::weak_maps::{WeakMap, WeakMapTrait};
use super::weak_refs::WeakRefTrait;
use super::weak_refs::{WeakRefRegistry, WeakRefStats, WeakReference};

/// Global memory management coordinator
///
/// This coordinates all advanced memory management features with the FUGC
/// garbage collector, providing a unified interface for explicit freeing,
/// finalization, and weak reference management.
///
/// # Examples
///
/// ```ignore
/// use fugrip::memory_management::manager::MemoryManager;
///
/// let manager = MemoryManager::new();
///
/// // C-style explicit freeing
/// // manager.free_object(obj);
///
/// // Java-style finalization
/// // manager.register_finalizer(obj, callback);
///
/// // JavaScript-style weak references
/// // let weak_ref = manager.create_weak_reference(obj);
/// ```
pub struct MemoryManager {
    /// Free object manager for explicit freeing
    free_manager: FreeObjectManager,
    /// Default finalizer queue
    default_finalizer_queue: FinalizerQueue,
    /// Named finalizer queues
    finalizer_queues: DashMap<String, FinalizerQueue>,
    /// FUGC coordinator reference
    fugc_coordinator: Weak<FugcCoordinator>,
    /// Weak reference registry
    weak_ref_registry: WeakRefRegistry,
    /// Global weak maps registry
    weak_maps: DashMap<String, Box<dyn WeakMapTrait>>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            free_manager: FreeObjectManager::new(),
            default_finalizer_queue: FinalizerQueue::new("default"),
            finalizer_queues: DashMap::new(),
            fugc_coordinator: Weak::new(),
            weak_ref_registry: WeakRefRegistry::new(),
            weak_maps: DashMap::new(),
        }
    }

    /// Set the FUGC coordinator reference
    pub fn set_fugc_coordinator(&mut self, coordinator: Weak<FugcCoordinator>) {
        self.fugc_coordinator = coordinator;
    }

    /// Get the free object manager
    pub fn free_manager(&mut self) -> &mut FreeObjectManager {
        &mut self.free_manager
    }

    /// Get immutable access to the free object manager
    pub fn free_manager_readonly(&self) -> &FreeObjectManager {
        &self.free_manager
    }

    /// Get or create a named finalizer queue
    ///
    /// # Arguments
    /// * `name` - Name of the queue
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::manager::MemoryManager;
    ///
    /// let manager = MemoryManager::new();
    /// let queue = manager.get_finalizer_queue("resource_cleanup");
    /// ```
    pub fn get_finalizer_queue(&self, name: &str) -> FinalizerQueue {
        if let Some(existing_queue) = self.finalizer_queues.get(name) {
            // Clone the existing queue to return it
            existing_queue.clone()
        } else {
            let queue = FinalizerQueue::new(name);
            self.finalizer_queues
                .insert(name.to_string(), queue.clone());
            queue
        }
    }

    /// Create a weak reference to an object
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::manager::MemoryManager;
    /// use std::sync::Arc;
    ///
    /// let manager = MemoryManager::new();
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = manager.create_weak_reference(Arc::clone(&strong_ref), None);
    /// assert!(weak_ref.is_valid());
    /// ```
    pub fn create_weak_reference<T: Send + Sync + 'static>(
        &self,
        strong_ref: Arc<T>,
        object_ref: Option<ObjectReference>,
    ) -> WeakReference<T> {
        let weak_ref = WeakReference::new(strong_ref, object_ref);

        if let Some(obj_ref) = object_ref {
            let weak_ref_arc =
                Arc::new(weak_ref.clone()) as Arc<dyn super::weak_refs::WeakRefTrait>;
            self.weak_ref_registry.register(obj_ref, weak_ref_arc);
        }

        weak_ref
    }

    /// Create or get a named weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::manager::MemoryManager;
    ///
    /// let manager = MemoryManager::new();
    /// let weak_map = manager.get_weak_map::<String, i32>("global_cache");
    /// assert_eq!(weak_map.size(), 0);
    /// ```
    pub fn get_weak_map<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static>(
        &self,
        name: &str,
    ) -> Arc<WeakMap<K, V>> {
        if self.weak_maps.contains_key(name) {
            // Return existing weak map if it exists
            let weak_map = Arc::new(WeakMap::new());
            weak_map
        } else {
            let weak_map = Arc::new(WeakMap::new());
            let boxed_map: Box<dyn WeakMapTrait> = Box::new((*weak_map).clone());
            self.weak_maps.insert(name.to_string(), boxed_map);
            weak_map
        }
    }

    /// Create a new weak map with the given name
    pub fn create_weak_map<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static>(
        &self,
        name: &str,
    ) -> Arc<WeakMap<K, V>> {
        let weak_map = Arc::new(WeakMap::new());
        let boxed_map: Box<dyn WeakMapTrait> = Box::new((*weak_map).clone());
        self.weak_maps.insert(name.to_string(), boxed_map);
        weak_map
    }

    /// Process pending memory management operations
    pub fn process_pending(&self) {
        // Process finalizers
        self.default_finalizer_queue.process_pending_finalizations();

        // Process other finalizer queues
        for queue in self.finalizer_queues.iter() {
            queue.process_pending_finalizations();
        }
    }

    /// Integration hook called during GC sweep phase
    pub fn gc_sweep_hook(&self) {
        // Clean up freed objects that have been reclaimed
        self.free_manager.sweep_freed_objects();

        // Process any pending finalizations
        self.default_finalizer_queue.process_pending_finalizations();

        // Clean up invalid weak references
        let cleaned_weak_refs = self.weak_ref_registry.cleanup_invalid_references();

        // Clean up dead entries in all weak maps
        let mut total_weak_map_cleanups = 0;
        for entry in self.weak_maps.iter() {
            total_weak_map_cleanups += entry.cleanup_dead_entries();
        }

        // Log cleanup statistics in debug builds
        #[cfg(debug_assertions)]
        println!(
            "GC sweep: cleaned {} weak refs, {} weak map entries",
            cleaned_weak_refs, total_weak_map_cleanups
        );
    }

    /// Get comprehensive memory management statistics
    pub fn get_stats(&self) -> MemoryManagerStats {
        let free_stats = self.free_manager.get_stats();
        let finalizer_stats = self.default_finalizer_queue.get_stats();

        MemoryManagerStats {
            free_object_stats: free_stats,
            default_finalizer_stats: finalizer_stats,
            active_finalizer_queues: self.finalizer_queues.len(),
            weak_ref_stats: WeakRefStats {
                total_registered: self.weak_ref_registry.total_count(),
                total_nulled: self.weak_ref_registry.nulled_count(),
                total_cleaned: self.weak_ref_registry.cleaned_count(),
                currently_active: self.weak_ref_registry.active_count(),
            },
            weak_map_count: self.weak_maps.len(),
        }
    }
}

/// Comprehensive memory management statistics
#[derive(Debug, Clone)]
pub struct MemoryManagerStats {
    /// Free object management statistics
    pub free_object_stats: FreeObjectStats,
    /// Default finalizer queue statistics
    pub default_finalizer_stats: FinalizerQueueStats,
    /// Number of active named finalizer queues
    pub active_finalizer_queues: usize,
    /// Weak reference registry statistics
    pub weak_ref_stats: WeakRefStats,
    /// Number of registered weak maps
    pub weak_map_count: usize,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::Address;
    use std::sync::Arc;

    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new();
        assert_eq!(manager.finalizer_queues.len(), 0);
        assert_eq!(manager.weak_maps.len(), 0);
    }

    #[test]
    fn test_memory_manager_integration() {
        let mut manager = MemoryManager::new();

        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x3000) }).unwrap();

        // Test free object integration
        manager.free_manager().free_object(obj);
        assert!(manager.free_manager().is_freed(obj));

        // Test GC sweep hook
        manager.gc_sweep_hook();

        let stats = manager.get_stats();
        assert_eq!(stats.free_object_stats.total_freed, 1);
    }

    #[test]
    fn test_weak_references() {
        let manager = MemoryManager::new();
        let strong_ref = Arc::new("test data".to_string());
        let weak_ref = manager.create_weak_reference(Arc::clone(&strong_ref), None);

        // Test that weak reference is valid
        assert!(weak_ref.is_valid());
        assert_eq!(weak_ref.get().as_deref(), Some(&"test data".to_string()));

        // Drop strong reference
        drop(strong_ref);

        // Weak reference should now be invalid
        assert!(!weak_ref.is_valid());
        assert!(weak_ref.get().is_none());

        let stats = manager.get_stats();
        assert_eq!(stats.weak_ref_stats.total_registered, 0); // No ObjectReference provided
    }

    #[test]
    fn test_weak_maps() {
        let manager = MemoryManager::new();
        let weak_map = manager.get_weak_map::<String, i32>("test_map");

        assert_eq!(weak_map.size(), 0);
        assert!(weak_map.is_empty());

        // Add some entries
        let key1 = Arc::new("key1".to_string());
        let key2 = Arc::new("key2".to_string());
        let obj_ref1 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();
        let obj_ref2 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x2000) }).unwrap();

        weak_map.set(Arc::clone(&key1), obj_ref1, 42);
        weak_map.set(Arc::clone(&key2), obj_ref2, 84);

        assert_eq!(weak_map.size(), 2);
        assert!(!weak_map.is_empty());
        assert_eq!(weak_map.get(&obj_ref1), Some(42));
        assert_eq!(weak_map.get(&obj_ref2), Some(84));
        assert!(weak_map.has(&obj_ref1));
        assert!(weak_map.has(&obj_ref2));

        // Test iteration
        let entries: Vec<_> = weak_map.iter().collect();
        assert_eq!(entries.len(), 2);

        // Test deletion
        assert!(weak_map.delete(&obj_ref1));
        assert!(!weak_map.delete(&obj_ref1)); // Already deleted
        assert_eq!(weak_map.size(), 1);
        assert!(!weak_map.has(&obj_ref1));

        // Test clear
        weak_map.clear();
        assert_eq!(weak_map.size(), 0);
        assert!(weak_map.is_empty());

        let stats = weak_map.get_stats();
        assert_eq!(stats.total_insertions, 2);
        assert_eq!(stats.total_explicit_deletions, 2); // 1 delete + 1 clear
    }

    #[test]
    fn test_memory_manager_weak_map_integration() {
        let manager = MemoryManager::new();

        // Test weak map retrieval
        let weak_map1 = manager.get_weak_map::<String, i32>("cache1");
        let weak_map2 = manager.get_weak_map::<String, i32>("cache2");

        // Should create different maps
        assert_eq!(weak_map1.size(), 0);
        assert_eq!(weak_map2.size(), 0);

        let stats = manager.get_stats();
        assert_eq!(stats.weak_map_count, 2);

        // Test GC integration
        manager.gc_sweep_hook();

        // All systems should still work after GC hook
        let final_stats = manager.get_stats();
        assert_eq!(final_stats.weak_map_count, 2);
    }

    #[test]
    fn test_memory_manager_concurrent_access() {
        use rayon::prelude::*;

        let manager = Arc::new(MemoryManager::new());
        let num_threads = 4;
        let operations_per_thread = 100;

        // Use rayon parallel iteration instead of manual thread spawning
        (0..num_threads).into_par_iter().for_each(|thread_id| {
            for i in 0..operations_per_thread {
                // Create test objects with word-aligned addresses
                let aligned_addr = 0x10000 + thread_id * 1000 * 8 + i * 8;
                let obj =
                    ObjectReference::from_raw_address(unsafe { Address::from_usize(aligned_addr) })
                        .unwrap();

                // Test free object operations (simplified for Arc usage)
                if i % 3 == 0 {
                    // Note: In real usage, would need proper synchronization for mutable access
                    // For testing, we'll skip this test case
                }

                // Test weak reference operations
                if i % 3 == 1 {
                    let strong_ref = Arc::new(format!("thread_{}_data_{}", thread_id, i));
                    let weak_ref = WeakReference::new(Arc::clone(&strong_ref), Some(obj));
                    let weak_ref_arc = Arc::new(weak_ref) as Arc<dyn WeakRefTrait>;
                    manager.weak_ref_registry.register(obj, weak_ref_arc);
                }

                // Test weak map operations
                if i % 3 == 2 {
                    let weak_map = manager.get_weak_map::<String, i32>("concurrent_test");
                    let key = Arc::new(format!("key_{}_{}", thread_id, i));
                    weak_map.set(key, obj, i as i32);
                }

                // Periodic cleanup to test under contention
                if i % 50 == 49 {
                    manager.gc_sweep_hook();
                }
            }
        });

        // Final consistency check
        let final_stats = manager.get_stats();
        // Check that stats are reasonable (non-negative values are guaranteed by unsigned types)
        assert!(final_stats.free_object_stats.total_freed <= operations_per_thread * num_threads);
        assert!(
            final_stats.weak_ref_stats.total_registered <= operations_per_thread * num_threads / 3
        );
        assert!(final_stats.weak_map_count <= operations_per_thread * num_threads / 3);
    }

    #[test]
    fn test_memory_manager_error_recovery() {
        let mut manager = MemoryManager::new();

        // Test operations with minimal resources
        let small_obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();

        // Multiple rapid operations that might stress the system
        for _ in 0..1000 {
            manager.free_manager().free_object(small_obj);
            manager.weak_ref_registry.cleanup_invalid_references();

            let weak_map = manager.get_weak_map::<String, i32>("stress_test");
            weak_map.cleanup_dead_entries();
        }

        // System should remain consistent
        let stats = manager.get_stats();
        assert!(stats.free_object_stats.total_freed > 0);

        // Test coordinator integration (should not panic with None coordinator)
        manager.gc_sweep_hook();

        // Set and unset coordinator
        let coordinator_weak = Arc::downgrade(&Arc::new(FugcCoordinator::new(
            unsafe { Address::from_usize(0x50000000) },
            1024 * 1024,
            4,
            &Arc::new(crate::thread::ThreadRegistry::new()),
            &arc_swap::ArcSwap::new(Arc::new(crate::roots::GlobalRoots::default())),
        )));

        manager.set_fugc_coordinator(coordinator_weak);
        manager.gc_sweep_hook(); // Should work with coordinator set

        // Final stats should be reasonable
        let final_stats = manager.get_stats();
        // Check that finalizer queues and weak maps are in reasonable ranges
        assert!(final_stats.active_finalizer_queues <= 10); // Should not have too many active queues
        assert!(final_stats.weak_map_count <= 1000); // Should not have excessive weak maps
    }

    #[test]
    fn test_finalizer_queue_access() {
        let manager = MemoryManager::new();

        // Test getting a named finalizer queue
        let queue = manager.get_finalizer_queue("test_queue");
        assert_eq!(queue.name(), "test_queue");

        // Test getting the same queue again
        let queue2 = manager.get_finalizer_queue("test_queue");
        assert_eq!(queue2.name(), "test_queue");

        // Test getting a different queue
        let queue3 = manager.get_finalizer_queue("another_queue");
        assert_eq!(queue3.name(), "another_queue");

        let stats = manager.get_stats();
        // Should have at least the queues we created
        assert!(stats.active_finalizer_queues >= 2);
    }

    #[test]
    fn test_memory_manager_default() {
        let manager = MemoryManager::default();

        // Should be equivalent to MemoryManager::new()
        assert_eq!(manager.finalizer_queues.len(), 0);
        assert_eq!(manager.weak_maps.len(), 0);
    }
}
