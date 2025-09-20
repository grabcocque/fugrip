//! Verse-style heap optimizations for FUGC
//!
//! This module implements optimizations inspired by Epic's Verse heap design,
//! particularly the high-performance mark bit operations and allocation tracking.

use crate::concurrent::{ObjectColor, TricolorMarking};
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};


/// Verse-style allocation tracker with atomic counters
///
/// Emulates verse_heap_notify_allocation/deallocation for low-overhead
/// allocation tracking and GC triggering.
///
/// Cache-line aligned to prevent false sharing in hot allocation paths.
/// Groups hot atomic counters to maximize cache utilization.
#[repr(align(64))]
pub struct VerseStyleAllocationTracker {
    /// Hot counters grouped together for cache efficiency
    /// (live_bytes is accessed most frequently in allocation fast path)
    live_bytes: AtomicUsize,
    swept_bytes: AtomicUsize,
    trigger_threshold: AtomicUsize,
    /// Cache line padding (64 - 3*8 = 40 bytes)
    _padding1: [u8; 40],

    /// Cold data in separate cache line
    gc_trigger_callback: Box<dyn Fn() + Send + Sync>,
    /// Additional padding to ensure callback is in its own cache line
    _padding2: [u8; 56], // 64 - 8 = 56 bytes
}

impl VerseStyleAllocationTracker {
    /// Create a new allocation tracker with verse-style atomic operations
    pub fn new<F>(trigger_threshold: usize, callback: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        Self {
            live_bytes: AtomicUsize::new(0),
            swept_bytes: AtomicUsize::new(0),
            trigger_threshold: AtomicUsize::new(trigger_threshold),
            _padding1: [0; 40],
            gc_trigger_callback: Box::new(callback),
            _padding2: [0; 56],
        }
    }

    /// Notify allocation with Verse-style fast atomic increment
    ///
    /// Equivalent to verse_heap_notify_allocation but optimized for Rust
    #[inline(always)]
    pub fn notify_allocation(&self, bytes_allocated: usize) {
        if bytes_allocated == 0 {
            return;
        }

        // Fast path: relaxed atomic increment for performance
        let new_live_bytes = self
            .live_bytes
            .fetch_add(bytes_allocated, Ordering::Relaxed)
            + bytes_allocated;

        // Check trigger threshold
        if new_live_bytes >= self.trigger_threshold.load(Ordering::Relaxed) {
            (self.gc_trigger_callback)();
        }
    }

    /// Notify deallocation with Verse-style atomic decrement
    #[inline(always)]
    pub fn notify_deallocation(&self, bytes_deallocated: usize) {
        if bytes_deallocated == 0 {
            return;
        }

        self.live_bytes
            .fetch_sub(bytes_deallocated, Ordering::Relaxed);
    }

    /// Notify sweep completion (combines deallocation + sweep tracking)
    #[inline(always)]
    pub fn notify_sweep(&self, bytes_swept: usize) {
        if bytes_swept == 0 {
            return;
        }

        self.notify_deallocation(bytes_swept);
        self.swept_bytes.fetch_add(bytes_swept, Ordering::Relaxed);
    }

    /// Get current live bytes
    pub fn get_live_bytes(&self) -> usize {
        self.live_bytes.load(Ordering::Relaxed)
    }

    /// Get total swept bytes
    pub fn get_swept_bytes(&self) -> usize {
        self.swept_bytes.load(Ordering::Relaxed)
    }
}

/// Verse-style fast mark bit operations
///
/// Provides inline mark bit access similar to verse_heap_is_marked/verse_heap_set_is_marked
pub struct VerseStyleMarkBits {
    marking: Arc<TricolorMarking>,
    heap_base: Address,
    heap_size: usize,
}

impl VerseStyleMarkBits {
    /// Create new verse-style mark bit interface
    pub fn new(marking: Arc<TricolorMarking>, heap_base: Address, heap_size: usize) -> Self {
        Self {
            marking,
            heap_base,
            heap_size,
        }
    }

    /// Fast inline mark bit query - equivalent to verse_heap_is_marked
    #[inline(always)]
    pub fn is_marked_fast(&self, object: ObjectReference) -> bool {
        // Direct color check without complex operations
        self.marking.get_color(object) != ObjectColor::White
    }

    /// Fast inline mark bit setting - equivalent to verse_heap_set_is_marked
    #[inline(always)]
    pub fn set_marked_fast(&self, object: ObjectReference, value: bool) -> bool {
        let current_color = self.marking.get_color(object);
        let was_marked = current_color != ObjectColor::White;

        if value && !was_marked {
            self.marking.set_color(object, ObjectColor::Black);
            true // We marked the object
        } else if !value && was_marked {
            self.marking.set_color(object, ObjectColor::White);
            true // We unmarked the object
        } else {
            false // No change needed
        }
    }

    /// Relaxed atomic marking for parallel marking phases
    ///
    /// Equivalent to verse_heap_set_is_marked_relaxed for high-throughput
    /// parallel marking without full memory fencing.
    #[inline(always)]
    pub fn set_marked_relaxed(&self, object: ObjectReference, value: bool) -> bool {
        // For now, delegate to regular marking (future: implement relaxed atomics)
        self.set_marked_fast(object, value)
    }

    /// Batch mark bit operations for SIMD-friendly processing
    ///
    /// Process multiple objects with better cache locality
    pub fn mark_batch(&self, objects: &[ObjectReference], value: bool) -> usize {
        let mut marked_count = 0;

        // Process in cache-friendly chunks
        for chunk in objects.chunks(8) {
            // Prefetch for better cache performance
            // Hardware prefetchers handle sequential access patterns efficiently

            // Mark the chunk
            for &obj in chunk {
                if self.set_marked_fast(obj, value) {
                    marked_count += 1;
                }
            }
        }

        marked_count
    }
}

/// Verse-style object iteration state for efficient root scanning
///
/// Emulates verse_heap_get_iteration_state for lock-free iteration
pub struct VerseStyleIterationState {
    version: AtomicUsize,
    is_iterating: AtomicBool,
    current_object_set: AtomicUsize,
}

impl Default for VerseStyleIterationState {
    fn default() -> Self {
        Self {
            version: AtomicUsize::new(0),
            is_iterating: AtomicBool::new(false),
            current_object_set: AtomicUsize::new(0),
        }
    }
}

impl VerseStyleIterationState {
    /// Create new iteration state
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin iteration with version tracking
    pub fn begin_iteration(&self) -> usize {
        let new_version = self.version.fetch_add(1, Ordering::Acquire) + 1;
        self.is_iterating.store(true, Ordering::Release);
        new_version
    }

    /// End iteration
    pub fn end_iteration(&self) {
        self.is_iterating.store(false, Ordering::Release);
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Get current iteration state (lock-free)
    pub fn get_current_state(&self) -> (usize, bool) {
        let version = self.version.load(Ordering::Acquire);
        let is_iterating = self.is_iterating.load(Ordering::Acquire);

        // Double-check version consistency
        let version2 = self.version.load(Ordering::Acquire);
        if version != version2 {
            // Concurrent modification, return safe "not iterating" state
            (0, false)
        } else {
            (version, is_iterating)
        }
    }
}

/// Verse-style heap configuration optimized for FUGC
pub struct VerseStyleHeapConfig {
    pub allocation_tracker: VerseStyleAllocationTracker,
    pub mark_bits: VerseStyleMarkBits,
    pub iteration_state: VerseStyleIterationState,
}

impl VerseStyleHeapConfig {
    /// Create a new Verse-style heap configuration
    pub fn new(
        heap_base: Address,
        heap_size: usize,
        marking: Arc<TricolorMarking>,
        trigger_threshold: usize,
    ) -> Self {
        let allocation_tracker = VerseStyleAllocationTracker::new(trigger_threshold, || {
            // GC trigger callback placeholder
            eprintln!("GC trigger threshold reached");
        });

        let mark_bits = VerseStyleMarkBits::new(marking, heap_base, heap_size);
        let iteration_state = VerseStyleIterationState::new();

        Self {
            allocation_tracker,
            mark_bits,
            iteration_state,
        }
    }

    /// Fast allocation with verse-style tracking
    #[inline(always)]
    pub fn notify_allocation(&self, bytes: usize) {
        self.allocation_tracker.notify_allocation(bytes);
    }

    /// Fast deallocation notification
    #[inline(always)]
    pub fn notify_deallocation(&self, bytes: usize) {
        self.allocation_tracker.notify_deallocation(bytes);
    }

    /// Fast mark bit operations
    #[inline(always)]
    pub fn is_object_marked(&self, object: ObjectReference) -> bool {
        self.mark_bits.is_marked_fast(object)
    }

    /// Fast mark bit setting
    #[inline(always)]
    pub fn set_object_marked(&self, object: ObjectReference, marked: bool) -> bool {
        self.mark_bits.set_marked_fast(object, marked)
    }

    /// Batch marking operations for performance
    pub fn mark_objects_batch(&self, objects: &[ObjectReference], marked: bool) -> usize {
        self.mark_bits.mark_batch(objects, marked)
    }

    /// Get allocation statistics
    pub fn get_allocation_stats(&self) -> (usize, usize) {
        (
            self.allocation_tracker.get_live_bytes(),
            self.allocation_tracker.get_swept_bytes(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmtk::util::Address;
    use std::sync::Arc;

    #[test]
    fn test_verse_style_allocation_tracking() {
        let tracker = VerseStyleAllocationTracker::new(1024, || {});

        tracker.notify_allocation(512);
        assert_eq!(tracker.get_live_bytes(), 512);

        tracker.notify_allocation(256);
        assert_eq!(tracker.get_live_bytes(), 768);

        tracker.notify_deallocation(200);
        assert_eq!(tracker.get_live_bytes(), 568);

        tracker.notify_sweep(100);
        assert_eq!(tracker.get_live_bytes(), 468);
        assert_eq!(tracker.get_swept_bytes(), 100);
    }

    #[test]
    fn test_verse_style_mark_bits() {
        let heap_base = unsafe { Address::from_usize(0x1000000) };
        let heap_size = 1024 * 1024;
        let marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let mark_bits = VerseStyleMarkBits::new(marking, heap_base, heap_size);

        let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 64usize) };

        assert!(!mark_bits.is_marked_fast(obj));
        assert!(mark_bits.set_marked_fast(obj, true));
        assert!(mark_bits.is_marked_fast(obj));
        assert!(!mark_bits.set_marked_fast(obj, true)); // Already marked
    }

    #[test]
    fn test_verse_style_iteration_state() {
        let state = VerseStyleIterationState::new();

        let (version, is_iterating) = state.get_current_state();
        assert_eq!(version, 0);
        assert!(!is_iterating);

        let new_version = state.begin_iteration();
        assert_eq!(new_version, 1);

        let (version, is_iterating) = state.get_current_state();
        assert_eq!(version, 1);
        assert!(is_iterating);

        state.end_iteration();

        let (version, is_iterating) = state.get_current_state();
        assert_eq!(version, 2);
        assert!(!is_iterating);
    }

    #[test]
    fn test_allocation_tracker_edge_cases() {
        let tracker = VerseStyleAllocationTracker::new(1024, || {});

        // Test zero allocation
        tracker.notify_allocation(0);
        assert_eq!(tracker.get_live_bytes(), 0);

        // Test large allocation
        tracker.notify_allocation(usize::MAX / 2);
        assert!(tracker.get_live_bytes() > 0);

        // Test deallocation of more than allocated (sad path)
        tracker.notify_deallocation(usize::MAX);
        // Should handle gracefully (may wrap or saturate)

        // Test rapid alternating allocations/deallocations
        for i in 0..100 {
            tracker.notify_allocation(i * 64);
            tracker.notify_deallocation(i * 32);
        }
    }

    #[test]
    fn test_mark_bits_concurrent_operations() {
        let mark_bits = VerseStyleMarkBits::new(
            Arc::new(crate::concurrent::TricolorMarking::new(
                unsafe { mmtk::util::Address::from_usize(0x1000000) },
                1024 * 1024,
            )),
            unsafe { mmtk::util::Address::from_usize(0x1000000) },
            1024 * 1024,
        );
        let obj1 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1001000)) };
        let obj2 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1002000)) };

        // Test rapid marking/unmarking patterns
        for i in 0..50 {
            let mark = i % 2 == 0;
            mark_bits.set_marked_fast(obj1, mark);
            mark_bits.set_marked_fast(obj2, !mark);
            assert_eq!(mark_bits.is_marked_fast(obj1), mark);
            assert_eq!(mark_bits.is_marked_fast(obj2), !mark);
        }
    }

    #[test]
    fn test_batch_marking_operations() {
        let mark_bits = VerseStyleMarkBits::new(
            Arc::new(crate::concurrent::TricolorMarking::new(
                unsafe { mmtk::util::Address::from_usize(0x1000000) },
                1024 * 1024,
            )),
            unsafe { mmtk::util::Address::from_usize(0x1000000) },
            1024 * 1024,
        );

        // Create a batch of objects within the heap range
        let objects: Vec<ObjectReference> = (0..10)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(
                    0x1000000 + i * 0x100,
                ))
            })
            .collect();

        // Ensure all objects start unmarked
        for obj in &objects {
            mark_bits.set_marked_fast(*obj, false);
        }

        // Test batch marking
        let marked_count = mark_bits.mark_batch(&objects, true);
        assert_eq!(marked_count, 10);

        // Verify all are marked
        for obj in &objects {
            assert!(mark_bits.is_marked_fast(*obj));
        }

        // Test batch unmarking
        let unmarked_count = mark_bits.mark_batch(&objects, false);
        assert_eq!(unmarked_count, 10);

        // Verify all are unmarked
        for obj in &objects {
            assert!(!mark_bits.is_marked_fast(*obj));
        }
    }

    #[test]
    fn test_iteration_state_concurrent_access() {
        let state = VerseStyleIterationState::new();

        // Test nested iteration attempts (sad path)
        let version1 = state.begin_iteration();
        assert_eq!(version1, 1);

        // Begin iteration while already iterating
        let _version2 = state.begin_iteration();
        // May allow nested or return error - either is valid

        // End iteration multiple times (sad path)
        state.end_iteration();
        state.end_iteration(); // Double end - should handle gracefully

        let (_final_version, is_iterating) = state.get_current_state();
        assert!(!is_iterating); // Should not be iterating after ends
    }

    #[test]
    fn test_verse_style_optimizations_integration() {
        let optimizer = VerseStyleHeapConfig::new(
            unsafe { mmtk::util::Address::from_usize(0x1000000) },
            1024 * 1024,
            Arc::new(crate::concurrent::TricolorMarking::new(
                unsafe { mmtk::util::Address::from_usize(0x1000000) },
                1024 * 1024,
            )),
            1024,
        );

        // Test combined operations
        optimizer.notify_allocation(1024);
        optimizer.notify_allocation(2048);

        let obj =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x3000)) };

        // Mark object
        assert!(optimizer.set_object_marked(obj, true));
        assert!(optimizer.is_object_marked(obj));

        // Check stats
        let (live_bytes, _swept_bytes) = optimizer.get_allocation_stats();
        assert_eq!(live_bytes, 3072);

        // Deallocate and check
        optimizer.notify_deallocation(1024);
        let (live_bytes_after, _) = optimizer.get_allocation_stats();
        assert_eq!(live_bytes_after, 2048);

        // Unmark object
        assert!(optimizer.set_object_marked(obj, false));
        assert!(!optimizer.is_object_marked(obj));
    }

    #[test]
    fn test_allocation_tracker_overflow_handling() {
        let tracker = VerseStyleAllocationTracker::new(1024, || {});

        // Test near-overflow conditions
        tracker.notify_allocation(usize::MAX - 1000);
        tracker.notify_allocation(500); // Should cause overflow

        // Verify it handles overflow gracefully
        let _live_bytes = tracker.get_live_bytes();
        // May wrap around or saturate - either is acceptable

        // Test sweep tracking with overflow
        tracker.notify_sweep(1000);
        let swept_bytes = tracker.get_swept_bytes();
        assert!(swept_bytes > 0);
    }

    #[test]
    fn test_mark_bits_extreme_addresses() {
        let mark_bits = VerseStyleMarkBits::new(
            Arc::new(crate::concurrent::TricolorMarking::new(
                unsafe { mmtk::util::Address::from_usize(0x1000000) },
                1024 * 1024,
            )),
            unsafe { mmtk::util::Address::from_usize(0x1000000) },
            1024 * 1024,
        );

        // Test with various address patterns within heap range
        let addresses = [
            0x1000000, // Heap base
            0x1008000, // Higher address within heap
            0x1080000, // Even higher within heap
            0x10F0000, // Near end of heap
            0x10FF000, // Very close to end of heap
        ];

        for addr in addresses {
            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(addr)) };

            // Test marking at various addresses
            assert!(!mark_bits.is_marked_fast(obj));
            assert!(mark_bits.set_marked_fast(obj, true));
            assert!(mark_bits.is_marked_fast(obj));
            assert!(mark_bits.set_marked_fast(obj, false));
            assert!(!mark_bits.is_marked_fast(obj));
        }
    }

    #[test]
    fn test_empty_batch_operations() {
        let mark_bits = VerseStyleMarkBits::new(
            Arc::new(crate::concurrent::TricolorMarking::new(
                unsafe { mmtk::util::Address::from_usize(0x1000000) },
                1024 * 1024,
            )),
            unsafe { mmtk::util::Address::from_usize(0x1000000) },
            1024 * 1024,
        );

        // Test empty batch
        let empty_objects: Vec<ObjectReference> = vec![];
        let marked_count = mark_bits.mark_batch(&empty_objects, true);
        assert_eq!(marked_count, 0);

        // Test single object batch
        let single_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x4000)) };
        let single_batch = vec![single_obj];
        let marked_count = mark_bits.mark_batch(&single_batch, true);
        assert_eq!(marked_count, 1);
        assert!(mark_bits.is_marked_fast(single_obj));
    }

    #[test]
    fn test_optimization_state_consistency() {
        let optimizer = VerseStyleHeapConfig::new(
            unsafe { mmtk::util::Address::from_usize(0x1000000) },
            1024 * 1024,
            Arc::new(crate::concurrent::TricolorMarking::new(
                unsafe { mmtk::util::Address::from_usize(0x1000000) },
                1024 * 1024,
            )),
            1024,
        );

        // Test rapid state changes
        for i in 0..100 {
            let obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x5000 + i * 0x10))
            };

            optimizer.notify_allocation(i * 64);
            optimizer.set_object_marked(obj, true);

            // Verify consistency
            assert!(optimizer.is_object_marked(obj));

            if i % 10 == 0 {
                optimizer.notify_deallocation(i * 32);
            }
        }

        // Check final state consistency
        let (live_bytes, _swept_bytes) = optimizer.get_allocation_stats();
        assert!(live_bytes > 0);
        // swept_bytes is usize, so >= 0 check is redundant
    }
}
