//! JavaScript-style weak map implementation
//!
//! This provides a WeakMap-like API that allows iteration and counting,
//! unlike JavaScript WeakMaps which are opaque.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::DashMap;

use super::weak_refs::WeakReference;
use crate::compat::{Address, ObjectReference};

/// JavaScript-style weak map implementation
///
/// This provides a WeakMap-like API that allows iteration and counting,
/// unlike JavaScript WeakMaps which are opaque.
#[derive(Debug)]
pub struct WeakMap<K, V> {
    /// The underlying map storage
    map: Arc<DashMap<ObjectReference, (K, V)>>,
    /// Registry for weak key references
    weak_keys: Arc<DashMap<ObjectReference, WeakReference<K>>>,
    /// Total number of entries ever inserted
    total_insertions: Arc<AtomicUsize>,
    /// Total number of entries removed by GC
    total_gc_removals: Arc<AtomicUsize>,
    /// Total number of entries explicitly deleted
    total_explicit_deletions: Arc<AtomicUsize>,
}

impl<K, V> Clone for WeakMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: Arc::clone(&self.map),
            weak_keys: Arc::clone(&self.weak_keys),
            total_insertions: Arc::clone(&self.total_insertions),
            total_gc_removals: Arc::clone(&self.total_gc_removals),
            total_explicit_deletions: Arc::clone(&self.total_explicit_deletions),
        }
    }
}

impl<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static> WeakMap<K, V> {
    /// Create a new weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    ///
    /// let weak_map: WeakMap<String, i32> = WeakMap::new();
    /// assert_eq!(weak_map.size(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            map: Arc::new(DashMap::new()),
            weak_keys: Arc::new(DashMap::new()),
            total_insertions: Arc::new(AtomicUsize::new(0)),
            total_gc_removals: Arc::new(AtomicUsize::new(0)),
            total_explicit_deletions: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Set a key-value pair in the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let mut weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.get(&obj_ref), Some(42));
    /// ```
    pub fn set(&self, key: Arc<K>, key_ref: ObjectReference, value: V) {
        let weak_key = WeakReference::new(key.clone(), Some(key_ref));

        self.weak_keys.insert(key_ref, weak_key);

        let key_owned = Arc::try_unwrap(key).unwrap_or_else(|arc| (*arc).clone());
        self.map.insert(key_ref, (key_owned, value));

        self.total_insertions.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a value from the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.get(&obj_ref), Some(42));
    ///
    /// assert_eq!(weak_map.get(&ObjectReference::from_raw_address(Address::from_usize(0x123)).unwrap()), None);
    /// ```
    pub fn get(&self, key_ref: &ObjectReference) -> Option<V>
    where
        V: Clone,
    {
        self.map.get(key_ref).map(|entry| entry.1.clone())
    }

    /// Check if a key exists in the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// assert!(!weak_map.has(&obj_ref));
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert!(weak_map.has(&obj_ref));
    /// ```
    pub fn has(&self, key_ref: &ObjectReference) -> bool {
        self.map.contains_key(key_ref)
    }

    /// Delete a key-value pair from the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert!(weak_map.delete(&obj_ref));
    /// assert!(!weak_map.has(&obj_ref));
    /// assert!(!weak_map.delete(&obj_ref)); // Already deleted
    /// ```
    pub fn delete(&self, key_ref: &ObjectReference) -> bool {
        let removed_from_map = self.map.remove(key_ref).is_some();
        let removed_from_weak_keys = self.weak_keys.remove(key_ref).is_some();

        if removed_from_map || removed_from_weak_keys {
            self.total_explicit_deletions
                .fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get the current size of the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// assert_eq!(weak_map.size(), 0);
    ///
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.size(), 1);
    /// ```
    pub fn size(&self) -> usize {
        self.map.len()
    }

    /// Check if the weak map is empty
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    ///
    /// let weak_map: WeakMap<String, i32> = WeakMap::new();
    /// assert!(weak_map.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Iterate over all entries in the weak map
    ///
    /// This is a unique feature compared to JavaScript WeakMap.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key1 = Arc::new("key1".to_string());
    /// let key2 = Arc::new("key2".to_string());
    /// let obj_ref1 = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    /// let obj_ref2 = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x100) }).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key1), obj_ref1, 42);
    /// weak_map.set(Arc::clone(&key2), obj_ref2, 84);
    ///
    /// let entries: Vec<_> = weak_map.iter().collect();
    /// assert_eq!(entries.len(), 2);
    /// ```
    pub fn iter(&self) -> WeakMapIterator<K, V> {
        let entries: Vec<_> = self
            .map
            .iter()
            .map(|entry| (*entry.key(), entry.0.clone(), entry.1.clone()))
            .collect();
        WeakMapIterator {
            entries: entries.into_iter(),
        }
    }

    /// Clear all entries from the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    /// use std::sync::Arc;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.size(), 1);
    ///
    /// weak_map.clear();
    /// assert_eq!(weak_map.size(), 0);
    /// ```
    pub fn clear(&self) {
        let cleared_count = self.map.len();
        self.map.clear();
        self.weak_keys.clear();

        self.total_explicit_deletions
            .fetch_add(cleared_count, Ordering::Relaxed);
    }

    /// Remove entries whose keys have been garbage collected
    ///
    /// This is called by the GC to clean up the weak map.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::weak_maps::WeakMap;
    ///
    /// let weak_map: WeakMap<String, i32> = WeakMap::new();
    /// let removed = weak_map.cleanup_dead_entries();
    /// assert_eq!(removed, 0); // No dead entries to clean
    /// ```
    pub fn cleanup_dead_entries(&self) -> usize {
        let mut removed_count = 0;

        // Find keys that are no longer valid
        let dead_keys: Vec<ObjectReference> = self
            .weak_keys
            .iter()
            .filter_map(|entry| {
                if !entry.is_valid() {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();

        // Remove dead entries
        for key_ref in dead_keys {
            if self.map.remove(&key_ref).is_some() {
                removed_count += 1;
            }
            self.weak_keys.remove(&key_ref);
        }

        self.total_gc_removals
            .fetch_add(removed_count, Ordering::Relaxed);
        removed_count
    }

    /// Get statistics for this weak map
    pub fn get_stats(&self) -> WeakMapStats {
        WeakMapStats {
            current_size: self.size(),
            total_insertions: self.total_insertions.load(Ordering::Relaxed),
            total_gc_removals: self.total_gc_removals.load(Ordering::Relaxed),
            total_explicit_deletions: self.total_explicit_deletions.load(Ordering::Relaxed),
        }
    }
}

impl<K, V> Default for WeakMap<K, V>
where
    K: Send + Sync + Clone + 'static,
    V: Send + Sync + Clone + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator for WeakMap entries
pub struct WeakMapIterator<K, V> {
    entries: std::vec::IntoIter<(ObjectReference, K, V)>,
}

impl<K, V> Iterator for WeakMapIterator<K, V> {
    type Item = (ObjectReference, K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next()
    }
}

/// Statistics for weak maps
#[derive(Debug, Clone)]
pub struct WeakMapStats {
    /// Current number of entries
    pub current_size: usize,
    /// Total number of insertions
    pub total_insertions: usize,
    /// Total number of entries removed by GC
    pub total_gc_removals: usize,
    /// Total number of explicit deletions
    pub total_explicit_deletions: usize,
}

/// Trait for type-erased weak map operations
pub trait WeakMapTrait: Send + Sync {
    fn cleanup_dead_entries(&self) -> usize;
    fn size(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static> WeakMapTrait
    for WeakMap<K, V>
{
    fn cleanup_dead_entries(&self) -> usize {
        WeakMap::cleanup_dead_entries(self)
    }

    fn size(&self) -> usize {
        WeakMap::size(self)
    }

    fn is_empty(&self) -> bool {
        WeakMap::is_empty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_weak_map_creation() {
        let weak_map: WeakMap<String, i32> = WeakMap::new();
        assert_eq!(weak_map.size(), 0);
        assert!(weak_map.is_empty());
    }

    #[test]
    fn test_weak_map_set_get() {
        let weak_map = WeakMap::new();
        let key = Arc::new("test_key".to_string());
        let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();

        weak_map.set(Arc::clone(&key), obj_ref, 42);

        assert_eq!(weak_map.size(), 1);
        assert!(weak_map.has(&obj_ref));
        assert_eq!(weak_map.get(&obj_ref), Some(42));
    }

    #[test]
    fn test_weak_map_delete() {
        let weak_map = WeakMap::new();
        let key = Arc::new("test_key".to_string());
        let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();

        weak_map.set(Arc::clone(&key), obj_ref, 42);
        assert!(weak_map.delete(&obj_ref));
        assert!(!weak_map.has(&obj_ref));
        assert!(!weak_map.delete(&obj_ref)); // Already deleted
    }

    #[test]
    fn test_weak_map_clear() {
        let weak_map = WeakMap::new();
        let key1 = Arc::new("key1".to_string());
        let key2 = Arc::new("key2".to_string());
        let obj_ref1 = ObjectReference::from_raw_address(Address::ZERO).unwrap();
        let obj_ref2 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x100) }).unwrap();

        weak_map.set(Arc::clone(&key1), obj_ref1, 42);
        weak_map.set(Arc::clone(&key2), obj_ref2, 84);

        assert_eq!(weak_map.size(), 2);
        weak_map.clear();
        assert_eq!(weak_map.size(), 0);
    }

    #[test]
    fn test_weak_map_iteration() {
        let weak_map = WeakMap::new();
        let key1 = Arc::new("key1".to_string());
        let key2 = Arc::new("key2".to_string());
        let obj_ref1 = ObjectReference::from_raw_address(Address::ZERO).unwrap();
        let obj_ref2 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x100) }).unwrap();

        weak_map.set(Arc::clone(&key1), obj_ref1, 42);
        weak_map.set(Arc::clone(&key2), obj_ref2, 84);

        let entries: Vec<_> = weak_map.iter().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_weak_map_stats() {
        let weak_map = WeakMap::new();
        let key = Arc::new("test_key".to_string());
        let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();

        let stats = weak_map.get_stats();
        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.total_insertions, 0);
        assert_eq!(stats.total_gc_removals, 0);
        assert_eq!(stats.total_explicit_deletions, 0);

        weak_map.set(Arc::clone(&key), obj_ref, 42);
        let stats = weak_map.get_stats();
        assert_eq!(stats.current_size, 1);
        assert_eq!(stats.total_insertions, 1);

        weak_map.delete(&obj_ref);
        let stats = weak_map.get_stats();
        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.total_explicit_deletions, 1);
    }

    #[test]
    fn test_weak_map_default() {
        let weak_map: WeakMap<String, i32> = WeakMap::default();
        assert_eq!(weak_map.size(), 0);
    }

    #[test]
    fn test_weak_map_clone() {
        let weak_map = WeakMap::new();
        let key = Arc::new("test_key".to_string());
        let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();

        weak_map.set(Arc::clone(&key), obj_ref, 42);

        let cloned_map = weak_map.clone();
        assert_eq!(cloned_map.size(), 1);
        assert_eq!(cloned_map.get(&obj_ref), Some(42));
    }
}
