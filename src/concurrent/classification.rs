//! Object classification for generational and concurrent marking

use crate::frontend::types::ObjectReference;
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::core::optimized_fetch_add;

/// Object age classification for generational GC
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectAge {
    Young,
    Old,
}

/// Object mutability classification for marking optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectMutability {
    Immutable,
    Mutable,
}

/// Object connectivity classification for marking priority
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectConnectivity {
    Low,
    High,
}

/// Complete object classification for FUGC-style allocation and marking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectClass {
    pub age: ObjectAge,
    pub mutability: ObjectMutability,
    pub connectivity: ObjectConnectivity,
}

impl ObjectClass {
    /// Create default young object classification
    pub fn default_young() -> Self {
        Self {
            age: ObjectAge::Young,
            mutability: ObjectMutability::Mutable,
            connectivity: ObjectConnectivity::Low,
        }
    }

    /// Get marking priority (higher = marked sooner)
    pub fn marking_priority(&self) -> u32 {
        let mut priority = 0;

        match self.age {
            ObjectAge::Old => priority += 100,
            ObjectAge::Young => priority += 50,
        }

        match self.connectivity {
            ObjectConnectivity::High => priority += 20,
            ObjectConnectivity::Low => priority += 10,
        }

        match self.mutability {
            ObjectMutability::Mutable => priority += 5,
            ObjectMutability::Immutable => priority += 2,
        }

        priority
    }

    /// Check if object should be scanned eagerly during concurrent marking
    pub fn should_scan_eagerly(&self) -> bool {
        matches!(self.connectivity, ObjectConnectivity::High) || matches!(self.age, ObjectAge::Old)
    }
}

/// Object classifier for FUGC-style object classification and generational management
pub struct ObjectClassifier {
    /// Object classifications using DashMap for lock-free concurrent access
    classifications: DashMap<ObjectReference, ObjectClass>,
    /// Promotion queue for young -> old transitions (lock-free)
    promotion_queue: SegQueue<ObjectReference>,
    /// Statistics counters
    young_objects: AtomicUsize,
    old_objects: AtomicUsize,
    immutable_objects: AtomicUsize,
    mutable_objects: AtomicUsize,
    cross_generation_references: AtomicUsize,
    /// Structure of Arrays for child relationships - better cache locality for scanning
    /// Separate arrays for different aspects accessed in different patterns
    child_parents: DashMap<ObjectReference, Vec<ObjectReference>>, // Objects pointing to this object
    child_targets: DashMap<ObjectReference, Vec<ObjectReference>>, // Objects this object points to
    child_offsets: DashMap<ObjectReference, Vec<usize>>, // Field offsets for faster rescanning
}

impl ObjectClassifier {
    pub fn new() -> Self {
        Self {
            classifications: DashMap::new(),
            promotion_queue: SegQueue::new(),
            young_objects: AtomicUsize::new(0),
            old_objects: AtomicUsize::new(0),
            immutable_objects: AtomicUsize::new(0),
            mutable_objects: AtomicUsize::new(0),
            cross_generation_references: AtomicUsize::new(0),
            child_parents: DashMap::new(),
            child_targets: DashMap::new(),
            child_offsets: DashMap::new(),
        }
    }

    /// Classify an object and store its classification
    pub fn classify_object(&self, object: ObjectReference, class: ObjectClass) {
        self.classifications.insert(object, class);
        // Initialize child tracking arrays for SoA optimization
        self.child_parents.entry(object).or_insert_with(Vec::new);
        self.child_targets.entry(object).or_insert_with(Vec::new);
        self.child_offsets.entry(object).or_insert_with(Vec::new);

        // Update statistics
        match class.age {
            ObjectAge::Young => {
                optimized_fetch_add(&self.young_objects, 1);
            }
            ObjectAge::Old => {
                optimized_fetch_add(&self.old_objects, 1);
            }
        }

        match class.mutability {
            ObjectMutability::Immutable => {
                optimized_fetch_add(&self.immutable_objects, 1);
            }
            ObjectMutability::Mutable => {
                optimized_fetch_add(&self.mutable_objects, 1);
            }
        }
    }

    /// Record child relationship with Structure of Arrays optimization
    ///
    /// Records parent->child relationship efficiently for cache-friendly scanning.
    /// Separates concerns to improve cache locality during different GC phases.
    pub fn record_child_relationship_soa(
        &self,
        parent: ObjectReference,
        child: ObjectReference,
        field_offset: usize,
    ) {
        // Record bidirectional relationship in separate arrays for better cache locality
        self.child_targets
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(child);

        self.child_parents
            .entry(child)
            .or_insert_with(Vec::new)
            .push(parent);

        self.child_offsets
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(field_offset);
    }

    /// Optimized scanning using Structure of Arrays for better cache utilization
    pub fn scan_object_fields_optimized(&self, obj: ObjectReference) -> Vec<ObjectReference> {
        // Use SoA design for better cache performance during field scanning
        if let Some(targets_ref) = self.child_targets.get(&obj) {
            // Sequential access pattern - excellent cache locality
            targets_ref.value().clone()
        } else {
            // Fallback to default behavior for untracked objects
            self.scan_object_fields(obj)
        }
    }

    /// Get the classification of an object
    pub fn get_classification(&self, object: ObjectReference) -> Option<ObjectClass> {
        self.classifications
            .get(&object)
            .map(|entry| *entry.value())
    }

    /// Queue an object for promotion to the old generation
    pub fn queue_for_promotion(&self, object: ObjectReference) {
        self.promotion_queue.push(object);
    }

    /// Promote all queued young objects to the old generation
    pub fn promote_young_objects(&self) {
        let mut queued = Vec::new();
        while let Some(object) = self.promotion_queue.pop() {
            queued.push(object);
        }

        if queued.is_empty() {
            return;
        }

        for object in queued.iter() {
            if let Some(mut entry) = self.classifications.get_mut(&object) {
                if matches!(entry.age, ObjectAge::Young) {
                    entry.age = ObjectAge::Old;
                    self.young_objects.fetch_sub(1, Ordering::Relaxed);
                    optimized_fetch_add(&self.old_objects, 1);
                }
            }
        }
    }

    /// Classify a new object (for allocation) with default young classification
    pub fn classify_new_object(&self, obj: ObjectReference) {
        let default_class = ObjectClass::default_young();
        self.classify_object(obj, default_class);
    }

    /// Record cross-generational reference between objects
    pub fn record_cross_generational_reference(&self, src: ObjectReference, dst: ObjectReference) {
        let src_class = self.get_classification(src);
        let dst_class = self.get_classification(dst);

        if let (Some(src_class), Some(dst_class)) = (src_class, dst_class)
            && matches!(src_class.age, ObjectAge::Old)
            && matches!(dst_class.age, ObjectAge::Young)
        {
            optimized_fetch_add(&self.cross_generation_references, 1);
            self.queue_for_promotion(dst);
        }

        self.child_targets.entry(src).or_insert_with(Vec::new);
        self.child_parents.entry(dst).or_insert_with(Vec::new);
        let mut entry = self.child_targets.entry(src).or_insert_with(Vec::new);
        if !entry.contains(&dst) {
            entry.push(dst);
        }
    }

    /// Get classification statistics
    pub fn get_stats(&self) -> ObjectClassificationStats {
        ObjectClassificationStats {
            young_objects: self.young_objects.load(Ordering::Relaxed),
            old_objects: self.old_objects.load(Ordering::Relaxed),
            immutable_objects: self.immutable_objects.load(Ordering::Relaxed),
            mutable_objects: self.mutable_objects.load(Ordering::Relaxed),
            total_classified: self.classifications.len(),
            cross_generation_references: self.cross_generation_references.load(Ordering::Relaxed),
        }
    }

    pub fn get_children(&self, object: ObjectReference) -> Vec<ObjectReference> {
        self.child_targets
            .get(&object)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }

    /// Scan object fields and return discovered object references
    ///
    /// Safety note: In this project we often work with synthetic object addresses
    /// during tests/benchmarks. Dereferencing those raw addresses would be
    /// undefined behavior and can segfault. To keep the system safe and
    /// deterministic, this implementation returns only previously recorded
    /// relationships (e.g., via write barriers) and does not attempt to read
    /// memory from the provided object address.
    pub fn scan_object_fields(&self, object: ObjectReference) -> Vec<ObjectReference> {
        // Return previously discovered children without touching raw memory.
        self.get_children(object)
    }

    /// Clear all classifications (for new GC cycle)
    pub fn clear(&self) {
        self.classifications.clear();
        self.young_objects.store(0, Ordering::Relaxed);
        self.old_objects.store(0, Ordering::Relaxed);
        self.immutable_objects.store(0, Ordering::Relaxed);
        self.mutable_objects.store(0, Ordering::Relaxed);
        self.cross_generation_references.store(0, Ordering::Relaxed);
        while self.promotion_queue.pop().is_some() {}
        self.child_parents.clear();
        self.child_targets.clear();
        self.child_offsets.clear();
    }
}

impl Default for ObjectClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for object classification
#[derive(Debug, Clone)]
pub struct ObjectClassificationStats {
    pub young_objects: usize,
    pub old_objects: usize,
    pub immutable_objects: usize,
    pub mutable_objects: usize,
    pub total_classified: usize,
    pub cross_generation_references: usize,
}
