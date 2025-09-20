//! Cache optimization for garbage collection operations
//!
//! This module provides cache-aware data structures and algorithms to improve
//! the performance of garbage collection operations through better memory locality.

use crossbeam::queue::SegQueue;
use itertools::izip;
use mmtk::util::{Address, ObjectReference};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Standard cache line size (64 bytes on most architectures).
///
/// ```
/// use fugrip::cache_optimization::CACHE_LINE_SIZE;
/// assert_eq!(CACHE_LINE_SIZE, 64);
/// ```
pub const CACHE_LINE_SIZE: usize = 64;

/// Number of pointer-sized objects that fit in a cache line.
///
/// ```
/// use fugrip::cache_optimization::{CACHE_LINE_SIZE, OBJECTS_PER_CACHE_LINE};
/// assert_eq!(OBJECTS_PER_CACHE_LINE * 8, CACHE_LINE_SIZE);
/// ```
pub const OBJECTS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 8;

/// Cache-aware allocator that optimizes object placement for locality.
///
/// ```
/// use fugrip::cache_optimization::CacheAwareAllocator;
/// use mmtk::util::Address;
///
/// let base = unsafe { Address::from_usize(0x1_0000_0000) };
/// let allocator = CacheAwareAllocator::new(base, 4096);
/// let ptr = allocator.allocate(64, 8).expect("allocation succeeds");
/// assert!(ptr >= base);
/// ```
pub struct CacheAwareAllocator {
    /// Current allocation pointer
    current_ptr: AtomicUsize,
    /// Base address for allocation
    base: usize,
    /// Allocation limit
    limit: usize,
    /// Cache line size in bytes
    cache_line_size: usize,
    /// Statistics
    allocations: AtomicUsize,
    cache_hits: AtomicUsize,
}

impl CacheAwareAllocator {
    /// Create a new cache-aware allocator.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheAwareAllocator;
    /// # use mmtk::util::Address;
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let allocator = CacheAwareAllocator::new(base, 1024);
    /// assert_eq!(allocator.get_allocated_bytes(), 0);
    /// ```
    pub fn new(base: Address, size: usize) -> Self {
        let base_addr = base.as_usize();
        Self {
            current_ptr: AtomicUsize::new(base_addr),
            base: base_addr,
            limit: base_addr + size,
            cache_line_size: 64, // Standard cache line size
            allocations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
        }
    }

    /// Allocate memory with cache-aware placement.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheAwareAllocator;
    /// # use mmtk::util::Address;
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let allocator = CacheAwareAllocator::new(base, 1024);
    /// let addr = allocator.allocate(32, 8).expect("allocation");
    /// assert!(addr >= base);
    /// ```
    pub fn allocate(&self, size: usize, align: usize) -> Option<Address> {
        loop {
            let current = self.current_ptr.load(Ordering::SeqCst);

            // CacheAwareAllocator always aligns to cache lines for optimal performance
            // Use the maximum of requested alignment and cache line size
            let effective_align = align.max(self.cache_line_size);

            // Align the pointer to the effective alignment
            let aligned_ptr = (current + effective_align - 1) & !(effective_align - 1);
            let total_size = aligned_ptr - current + size;

            // Round up the total allocation to cache line boundaries
            let cache_aligned_size =
                (total_size + self.cache_line_size - 1) & !(self.cache_line_size - 1);
            let new_ptr = current + cache_aligned_size;

            if new_ptr > self.limit {
                // Allocation failed
                return None;
            }

            // TODO: Add crossbeam_utils::Backoff here for allocation contention
            // CRITICAL HOT PATH: Allocator CAS loops under high thread contention
            // Expected 15-30% reduction in CPU spinning during allocation storms
            // Changes: Add backoff.spin() on CAS failure, backoff.reset() on success
            // Try to update the current pointer atomically
            if self
                .current_ptr
                .compare_exchange_weak(current, new_ptr, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                return Some(unsafe { Address::from_usize(aligned_ptr) });
            }
            // If CAS failed, retry with the new current value
        }
    }

    /// Allocate memory with specific alignment requirements (alias for `allocate`).
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheAwareAllocator;
    /// # use mmtk::util::Address;
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let allocator = CacheAwareAllocator::new(base, 1024);
    /// let addr = allocator.allocate_aligned(16, 128).expect("aligned allocation");
    /// assert_eq!(addr.as_usize() % 128, 0);
    /// ```
    pub fn allocate_aligned(&self, size: usize, align: usize) -> Option<Address> {
        self.allocate(size, align)
    }

    /// Reset the allocator to its initial state.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheAwareAllocator;
    /// # use mmtk::util::Address;
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let allocator = CacheAwareAllocator::new(base, 1024);
    /// let _ = allocator.allocate(32, 8);
    /// allocator.reset();
    /// assert_eq!(allocator.get_allocated_bytes(), 0);
    /// ```
    pub fn reset(&self) {
        self.current_ptr.store(self.base, Ordering::SeqCst);
        self.allocations.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
    }

    /// Get total allocated bytes.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheAwareAllocator;
    /// # use mmtk::util::Address;
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let allocator = CacheAwareAllocator::new(base, 1024);
    /// assert_eq!(allocator.get_allocated_bytes(), 0);
    /// ```
    pub fn get_allocated_bytes(&self) -> usize {
        self.current_ptr.load(Ordering::Relaxed) - self.base
    }

    /// Get allocation statistics.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheAwareAllocator;
    /// # use mmtk::util::Address;
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let allocator = CacheAwareAllocator::new(base, 1024);
    /// let stats = allocator.get_stats();
    /// assert_eq!(stats, (0, 0));
    /// ```
    pub fn get_stats(&self) -> (usize, usize) {
        let allocated_bytes = self.get_allocated_bytes();
        let allocation_count = self.allocations.load(Ordering::Relaxed);
        (allocated_bytes, allocation_count)
    }
}

/// Cache-optimized marking that improves memory locality during tracing.
///
/// ```
/// use fugrip::cache_optimization::CacheOptimizedMarking;
/// let marking = CacheOptimizedMarking::new(4);
/// assert!(marking.is_complete());
/// ```
pub struct CacheOptimizedMarking {
    /// Work queue optimized for cache locality
    work_queue: Arc<SegQueue<ObjectReference>>,
    /// Prefetch distance for improved cache performance
    prefetch_distance: usize,
    /// Statistics
    objects_marked: AtomicUsize,
    cache_misses: AtomicUsize,
    /// Reference to tricolor marking system for actual marking
    tricolor_marking: Option<Arc<crate::concurrent::TricolorMarking>>,
}

impl CacheOptimizedMarking {
    /// Create a new cache-optimized marking instance.
    ///
    /// ```
    /// use fugrip::cache_optimization::CacheOptimizedMarking;
    /// let marking = CacheOptimizedMarking::new(2);
    /// assert!(marking.is_complete());
    /// ```
    pub fn new(prefetch_distance: usize) -> Self {
        Self {
            work_queue: Arc::new(SegQueue::new()),
            prefetch_distance,
            objects_marked: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            tricolor_marking: None,
        }
    }

    /// Create a new cache-optimized marking instance with tricolor integration.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// # use fugrip::concurrent::TricolorMarking;
    /// # use mmtk::util::Address;
    /// # use std::sync::Arc;
    /// let heap_base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let tricolor = Arc::new(TricolorMarking::new(heap_base, 1024));
    /// let marking = CacheOptimizedMarking::with_tricolor(&tricolor);
    /// assert!(marking.is_complete());
    /// ```
    pub fn with_tricolor(tricolor: &Arc<crate::concurrent::TricolorMarking>) -> Self {
        Self {
            work_queue: Arc::new(SegQueue::new()),
            prefetch_distance: 4,
            objects_marked: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            tricolor_marking: Some(Arc::clone(tricolor)),
        }
    }

    /// Mark an object with cache optimization
    /// Mark a single object, updating statistics and potentially re-queuing it.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// # use mmtk::util::{Address, ObjectReference};
    /// let marking = CacheOptimizedMarking::new(4);
    /// let obj = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1_0000_0100) }).unwrap();
    /// marking.mark_object(obj);
    /// ```
    pub fn mark_object(&self, object: ObjectReference) {
        // Prefetch the object data for better cache performance
        self.prefetch_object(object);

        // Mark in tricolor system if available
        if let Some(ref tricolor) = self.tricolor_marking {
            tricolor.set_color(object, crate::concurrent::ObjectColor::Grey);
        }

        // Add to work queue
        self.work_queue.push(object);
        self.objects_marked.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark multiple objects in batch for better cache utilization
    /// Mark a batch of objects optimized for cache locality.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// # use mmtk::util::{Address, ObjectReference};
    /// let marking = CacheOptimizedMarking::new(4);
    /// let objects: Vec<ObjectReference> = (0..2)
    ///     .map(|i| ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1_0000_0100 + i * 8) }).unwrap())
    ///     .collect();
    /// marking.mark_objects_batch(&objects);
    /// ```
    pub fn mark_objects_batch(&self, objects: &[ObjectReference]) {
        // Batch mark in tricolor system for better performance
        if let Some(ref tricolor) = self.tricolor_marking {
            const TILE: usize = 4;
            let mut index = 0;

            while index < objects.len() {
                let upper = (index + TILE).min(objects.len());

                for &object in &objects[index..upper] {
                    self.prefetch_object(object);
                }

                for &object in &objects[index..upper] {
                    tricolor.set_color(object, crate::concurrent::ObjectColor::Grey);
                    self.work_queue.push(object);
                }

                index = upper;
            }
            self.objects_marked
                .fetch_add(objects.len(), Ordering::Relaxed);
        } else {
            for &object in objects {
                self.mark_object(object);
            }
        }
    }

    /// Process marking work with cache optimization
    /// Process work from the queue, simulating cache-aware traversal.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// let marking = CacheOptimizedMarking::new(4);
    /// assert!(marking.process_work().is_none());
    /// ```
    pub fn process_work(&self) -> Option<ObjectReference> {
        if let Some(object) = self.work_queue.pop() {
            // Prefetch next objects for better cache utilization
            for _ in 0..self.prefetch_distance {
                if let Some(next) = self.work_queue.pop() {
                    self.prefetch_object(next);
                    self.work_queue.push(next);
                }
            }
            let _ =
                self.objects_marked
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                        count.checked_sub(1)
                    });
            Some(object)
        } else {
            None
        }
    }

    /// Prefetch object data into cache
    fn prefetch_object(&self, object: ObjectReference) {
        // Use compiler intrinsics for prefetching if available
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let ptr = object.to_raw_address().to_ptr::<u8>();
            // Prefetch for temporal locality (expected to be accessed soon)
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Generic prefetch hint using volatile read
            let ptr = object.to_raw_address().to_ptr::<u8>();
            unsafe {
                std::ptr::read_volatile(ptr);
            }
        }
    }

    /// Get summary statistics for cache-optimized marking.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// let marking = CacheOptimizedMarking::new(4);
    /// let stats = marking.get_stats();
    /// assert_eq!(stats.objects_marked, 0);
    /// ```
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            objects_marked: self.objects_marked.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            queue_depth: self.work_queue.len(),
            batch_size: OBJECTS_PER_CACHE_LINE,
            prefetch_distance: self.prefetch_distance,
        }
    }

    /// Get detailed cache statistics.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// let marking = CacheOptimizedMarking::new(4);
    /// let stats = marking.get_cache_stats();
    /// assert_eq!(stats.cache_misses, 0);
    /// ```
    pub fn get_cache_stats(&self) -> CacheStats {
        self.get_stats()
    }

    /// Check if there is any work remaining.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// let marking = CacheOptimizedMarking::new(4);
    /// assert!(marking.is_complete());
    /// ```
    pub fn is_complete(&self) -> bool {
        self.work_queue.is_empty()
    }

    /// Reset marking state.
    ///
    /// ```
    /// # use fugrip::cache_optimization::CacheOptimizedMarking;
    /// let marking = CacheOptimizedMarking::new(4);
    /// marking.reset();
    /// ```
    pub fn reset(&self) {
        while self.work_queue.pop().is_some() {}
        self.objects_marked.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);

        // Also reset the underlying tricolor marking if present
        if let Some(ref tricolor) = self.tricolor_marking {
            tricolor.clear();
        }
    }

    /// Get access to the underlying tricolor marking system.
    ///
    /// This is exposed for cases where direct access to marked objects is needed,
    /// such as during the sweep phase. Most marking operations should go through
    /// the cache-optimized interface instead.
    pub fn tricolor_marking(&self) -> Option<&Arc<crate::concurrent::TricolorMarking>> {
        self.tricolor_marking.as_ref()
    }
}

/// Statistics for cache optimization.
///
/// ```
/// use fugrip::cache_optimization::CacheStats;
/// let stats = CacheStats::default();
/// assert_eq!(stats.objects_marked, 0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub objects_marked: usize,
    pub cache_misses: usize,
    pub queue_depth: usize,
    pub batch_size: usize,
    pub prefetch_distance: usize,
}

/// Memory layout optimizer for better cache utilization.
///
/// ```
/// use fugrip::cache_optimization::MemoryLayoutOptimizer;
/// let optimizer = MemoryLayoutOptimizer::new();
/// let class = optimizer.get_size_class(24);
/// assert!(class >= 24);
/// ```
pub struct MemoryLayoutOptimizer {
    /// Object size classes for better packing
    size_classes: Vec<usize>,
    /// Allocation counts per size class
    allocation_counts: Vec<AtomicUsize>,
}

impl MemoryLayoutOptimizer {
    /// Create a new memory layout optimizer
    /// Create a new memory layout optimizer.
    ///
    /// ```
    /// use fugrip::cache_optimization::MemoryLayoutOptimizer;
    /// let optimizer = MemoryLayoutOptimizer::new();
    /// assert!(!optimizer.get_statistics().is_empty());
    /// ```
    pub fn new() -> Self {
        let size_classes = vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let allocation_counts = size_classes.iter().map(|_| AtomicUsize::new(0)).collect();

        Self {
            size_classes,
            allocation_counts,
        }
    }

    /// Get the appropriate size class for an object.
    ///
    /// ```
    /// # use fugrip::cache_optimization::MemoryLayoutOptimizer;
    /// let optimizer = MemoryLayoutOptimizer::new();
    /// assert_eq!(optimizer.get_size_class(8), 8);
    /// ```
    pub fn get_size_class(&self, size: usize) -> usize {
        for &class in &self.size_classes {
            if size <= class {
                return class;
            }
        }
        // Round up to next power of 2 for large objects, but handle overflow
        size.saturating_add(1).next_power_of_two()
    }

    /// Record an allocation for statistical tracking.
    ///
    /// ```
    /// # use fugrip::cache_optimization::MemoryLayoutOptimizer;
    /// let optimizer = MemoryLayoutOptimizer::new();
    /// optimizer.record_allocation(16);
    /// ```
    pub fn record_allocation(&self, size: usize) {
        for (i, &class) in self.size_classes.iter().enumerate() {
            if size <= class {
                self.allocation_counts[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Get statistics on allocations per size class.
    ///
    /// ```
    /// # use fugrip::cache_optimization::MemoryLayoutOptimizer;
    /// let optimizer = MemoryLayoutOptimizer::new();
    /// let stats = optimizer.get_statistics();
    /// assert!(!stats.is_empty());
    /// ```
    pub fn get_statistics(&self) -> Vec<(usize, usize)> {
        self.size_classes
            .iter()
            .zip(&self.allocation_counts)
            .map(|(&size, count)| (size, count.load(Ordering::Relaxed)))
            .collect()
    }

    /// Calculate object layout for better cache utilization.
    ///
    /// ```
    /// # use fugrip::cache_optimization::MemoryLayoutOptimizer;
    /// # use mmtk::util::Address;
    /// let optimizer = MemoryLayoutOptimizer::new();
    /// let layouts = optimizer.calculate_object_layout(&[16, 24]);
    /// assert_eq!(layouts.len(), 2);
    /// ```
    pub fn calculate_object_layout(&self, sizes: &[usize]) -> Vec<(Address, usize)> {
        if sizes.len() <= 100 {
            // Use sequential processing for small object sets to avoid Rayon overhead
            return self.calculate_object_layout_sequential(sizes);
        }

        // Use Rayon for parallel processing of large object sets
        self.calculate_object_layout_parallel(sizes)
    }

    /// Sequential implementation for small object sets
    fn calculate_object_layout_sequential(&self, sizes: &[usize]) -> Vec<(Address, usize)> {
        let mut layouts = Vec::new();
        let mut current_addr = 0x1000; // Start at aligned address

        for &size in sizes {
            let aligned_size = self.get_size_class(size);

            // Align large objects to cache lines
            if aligned_size >= CACHE_LINE_SIZE {
                current_addr = (current_addr + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
            }

            layouts.push((unsafe { Address::from_usize(current_addr) }, size));
            current_addr += aligned_size;
        }

        layouts
    }

    /// Parallel implementation using Rayon for large object sets
    fn calculate_object_layout_parallel(&self, sizes: &[usize]) -> Vec<(Address, usize)> {
        // First, compute size classes in parallel
        let size_classes: Vec<usize> = sizes
            .par_iter()
            .map(|&size| self.get_size_class(size))
            .collect();

        // Calculate base alignments in parallel
        let alignments: Vec<usize> = size_classes
            .par_iter()
            .map(|&aligned_size| {
                if aligned_size >= CACHE_LINE_SIZE {
                    CACHE_LINE_SIZE
                } else {
                    1
                }
            })
            .collect();

        // Compute cumulative offsets (sequential dependency)
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut current_addr = 0x1000;

        for (i, (&_size, &alignment)) in sizes.iter().zip(&alignments).enumerate() {
            if alignment > 1 {
                current_addr = (current_addr + alignment - 1) & !(alignment - 1);
            }
            offsets.push(current_addr);
            current_addr += size_classes[i];
        }

        // Create final layouts in parallel
        let layouts: Vec<(Address, usize)> = izip!(sizes.iter(), offsets.iter())
            .par_bridge()
            .map(|(&size, &offset)| (unsafe { Address::from_usize(offset) }, size))
            .collect();

        layouts
    }

    /// Colocate metadata with object for better cache locality.
    ///
    /// ```no_run
    /// use fugrip::cache_optimization::MemoryLayoutOptimizer;
    /// use mmtk::util::Address;
    ///
    /// let optimizer = MemoryLayoutOptimizer::new();
    /// let base = unsafe { Address::from_usize(0x1_0000_0000) };
    /// let meta = optimizer.colocate_metadata(base, 16);
    /// assert!(meta <= base);
    /// ```
    pub fn colocate_metadata(&self, object_addr: Address, metadata_size: usize) -> Address {
        let aligned_metadata_size = (metadata_size + 7) & !7; // 8-byte align
        let object_usize = object_addr.as_usize();
        let metadata_addr = object_usize.saturating_sub(aligned_metadata_size);
        unsafe { Address::from_usize(metadata_addr) }
    }
}

impl Default for MemoryLayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// TODO-RAYON-HIGH: Replace LocalityAwareWorkStealer with par_chunks()
/// Current: 100+ lines of manual local/shared queue management + stealing logic
/// Rayon: objects.par_chunks(CACHE_LINE_SIZE).for_each(|chunk| process_chunk(chunk))
/// Impact: 75% reduction, better automatic cache-aware batching
/// Est. effort: 6 hours (need to preserve NUMA locality semantics)
///
/// Work-stealing structure with locality awareness.
///
/// ```
/// use fugrip::cache_optimization::LocalityAwareWorkStealer;
/// use std::sync::Arc;
/// use mmtk::util::{Address, ObjectReference};
/// let mut stealer = LocalityAwareWorkStealer::new(8);
/// let obj = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1_0000_0100) }).unwrap();
/// stealer.add_objects(vec![obj]);
/// assert!(!stealer.get_next_batch(1).is_empty());
/// ```
pub struct LocalityAwareWorkStealer {
    /// Local work queue for better cache locality
    local_queue: Mutex<VecDeque<ObjectReference>>,
    /// Shared work queue for stealing
    shared_queue: Arc<SegQueue<ObjectReference>>,
    /// Steal threshold
    steal_threshold: usize,
    /// Statistics
    local_processed: AtomicUsize,
    stolen_work: AtomicUsize,
    work_shared: AtomicUsize,
}

impl LocalityAwareWorkStealer {
    /// Create a new locality-aware work stealer
    pub fn new(steal_threshold: usize) -> Self {
        Self {
            local_queue: Mutex::new(VecDeque::new()),
            shared_queue: Arc::new(SegQueue::new()),
            steal_threshold,
            local_processed: AtomicUsize::new(0),
            stolen_work: AtomicUsize::new(0),
            work_shared: AtomicUsize::new(0),
        }
    }

    /// Add multiple objects to the work queue
    pub fn add_objects(&mut self, objects: Vec<ObjectReference>) {
        let mut queue = self.local_queue.lock();
        for obj in objects {
            queue.push_back(obj);
        }
    }

    /// Get the next batch of work
    pub fn get_next_batch(&mut self, batch_size: usize) -> Vec<ObjectReference> {
        let mut batch = Vec::with_capacity(batch_size);
        let mut queue = self.local_queue.lock();

        for _ in 0..batch_size {
            if let Some(obj) = queue.pop_front() {
                batch.push(obj);
                self.local_processed.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }

        // Try to steal from shared queue if local is empty
        if batch.is_empty() {
            for _ in 0..batch_size {
                if let Some(obj) = self.shared_queue.pop() {
                    batch.push(obj);
                    self.stolen_work.fetch_add(1, Ordering::Relaxed);
                } else {
                    break;
                }
            }
        }

        batch
    }

    /// Add work to the local queue
    pub fn push_local(&self, object: ObjectReference) {
        let mut queue = self.local_queue.lock();
        queue.push_back(object);

        // Share work if local queue is getting too large
        if queue.len() > self.steal_threshold * 2 {
            // Share half of the work
            for _ in 0..self.steal_threshold {
                if let Some(obj) = queue.pop_front() {
                    self.shared_queue.push(obj);
                    self.work_shared.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Get work, preferring local queue for better locality
    pub fn pop(&self) -> Option<ObjectReference> {
        // First try local queue
        {
            let mut queue = self.local_queue.lock();
            if let Some(obj) = queue.pop_front() {
                self.local_processed.fetch_add(1, Ordering::Relaxed);
                return Some(obj);
            }
        }

        // Try to steal from shared queue
        if let Some(obj) = self.shared_queue.pop() {
            self.stolen_work.fetch_add(1, Ordering::Relaxed);
            return Some(obj);
        }

        None
    }

    /// Steal work from another worker
    pub fn steal_from(&self, other: &LocalityAwareWorkStealer) -> bool {
        // Try to steal from their shared queue
        if let Some(obj) = other.shared_queue.pop() {
            let mut queue = self.local_queue.lock();
            queue.push_back(obj);
            self.stolen_work.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        false
    }

    /// Get work-stealing statistics
    pub fn get_stats(&self) -> (usize, usize, usize) {
        (
            self.local_processed.load(Ordering::Relaxed),
            self.stolen_work.load(Ordering::Relaxed),
            self.work_shared.load(Ordering::Relaxed),
        )
    }

    /// Check if there's any work available
    pub fn has_work(&self) -> bool {
        !self.local_queue.lock().is_empty() || !self.shared_queue.is_empty()
    }

    /// Get a reference to the shared queue for coordination
    pub fn shared_queue(&self) -> &Arc<SegQueue<ObjectReference>> {
        &self.shared_queue
    }
}

/// Metadata colocation for improved cache performance
pub struct MetadataColocation {
    /// Colocated metadata storage
    metadata: Vec<AtomicUsize>,
    /// Metadata stride
    stride: usize,
}

impl MetadataColocation {
    /// Create new metadata colocation
    pub fn new(capacity: usize, stride: usize) -> Self {
        let metadata = (0..capacity).map(|_| AtomicUsize::new(0)).collect();
        Self { metadata, stride }
    }

    /// Get metadata for an object
    pub fn get_metadata(&self, index: usize) -> usize {
        if index < self.metadata.len() {
            self.metadata[index].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Set metadata for an object
    pub fn set_metadata(&self, index: usize, value: usize) {
        if index < self.metadata.len() {
            self.metadata[index].store(value, Ordering::Relaxed);
        }
    }

    /// Update metadata atomically
    pub fn update_metadata<F>(&self, index: usize, f: F) -> usize
    where
        F: Fn(usize) -> usize,
    {
        if index < self.metadata.len() {
            let mut current = self.metadata[index].load(Ordering::Relaxed);
            loop {
                let new = f(current);
                match self.metadata[index].compare_exchange_weak(
                    current,
                    new,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return new,
                    Err(actual) => current = actual,
                }
            }
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_aware_allocator_alignment() {
        let base = unsafe { Address::from_usize(0x1000) };
        let allocator = CacheAwareAllocator::new(base, 4096);

        // Test allocation alignment
        let addr1 = allocator.allocate(8, 8).unwrap();
        assert_eq!(addr1.as_usize() % 8, 0);

        let addr2 = allocator.allocate(16, 16).unwrap();
        assert_eq!(addr2.as_usize() % 16, 0);
    }

    #[test]
    fn memory_layout_optimization() {
        let optimizer = MemoryLayoutOptimizer::new();

        assert_eq!(optimizer.get_size_class(7), 8);
        assert_eq!(optimizer.get_size_class(15), 16);
        assert_eq!(optimizer.get_size_class(33), 64);
        assert_eq!(optimizer.get_size_class(5000), 8192);

        optimizer.record_allocation(10);
        optimizer.record_allocation(20);
        optimizer.record_allocation(100);

        let stats = optimizer.get_statistics();
        assert!(!stats.is_empty());
    }

    #[test]
    fn locality_aware_work_stealing() {
        let stealer1 = LocalityAwareWorkStealer::new(10);
        let stealer2 = LocalityAwareWorkStealer::new(10);

        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();

        stealer1.push_local(obj);
        assert!(stealer1.has_work());

        let stolen = stealer2.steal_from(&stealer1);
        assert!(!stolen); // Should not steal from local queue

        // Fill up to trigger sharing
        for i in 0..25 {
            let obj =
                ObjectReference::from_raw_address(unsafe { Address::from_usize(0x2000 + i * 8) })
                    .unwrap();
            stealer1.push_local(obj);
        }

        // Now stealing should be possible
        let stolen = stealer2.steal_from(&stealer1);
        assert!(stolen || !stealer1.shared_queue().is_empty());
    }

    #[test]
    fn metadata_colocation() {
        let metadata = MetadataColocation::new(100, 8);

        metadata.set_metadata(5, 42);
        assert_eq!(metadata.get_metadata(5), 42);

        let result = metadata.update_metadata(5, |x| x + 10);
        assert_eq!(result, 52);
        assert_eq!(metadata.get_metadata(5), 52);
    }

    #[test]
    fn test_cache_aware_allocator_with_generations() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024; // 64MB heap
        let allocator = CacheAwareAllocator::new(heap_base, heap_size);

        // Test allocation in different size classes (simulating young/old generations)
        let small_obj = allocator.allocate(16, 8); // Young gen size
        let medium_obj = allocator.allocate(128, 8); // Young gen size
        let large_obj = allocator.allocate(1024, 64); // Old gen size

        // Verify allocations succeed
        assert!(small_obj.is_some());
        assert!(medium_obj.is_some());
        assert!(large_obj.is_some());

        if let Some(addr) = large_obj {
            // Large objects should be well aligned
            assert_eq!(addr.as_usize() % 64, 0);
        }

        // Get statistics
        let (allocated, allocs) = allocator.get_stats();
        assert!(allocated >= 16 + 128 + 1024);
        assert_eq!(allocs, 3);
    }

    #[test]
    fn test_cache_optimized_marking_with_prefetch() {
        let marking = CacheOptimizedMarking::new(4);

        // Add multiple objects to trigger prefetch behavior
        let objects: Vec<ObjectReference> = (0..10)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000 + i * 64))
            })
            .collect();

        for obj in &objects {
            marking.mark_object(*obj);
        }

        // Process work and verify prefetch stats
        let mut processed = 0;
        while marking.process_work().is_some() {
            processed += 1;
        }
        assert_eq!(processed, 10);

        let stats = marking.get_stats();
        assert_eq!(stats.objects_marked, 0); // No actual marking in this test
        assert_eq!(stats.queue_depth, 0);
    }

    #[test]
    fn test_memory_layout_optimizer_alignment() {
        let optimizer = MemoryLayoutOptimizer::new();

        // Test various size classes and alignments
        let sizes = vec![8, 16, 24, 32, 64, 128, 256, 512, 1024];
        for size in sizes {
            let class = optimizer.get_size_class(size);
            assert!(class >= size);
            assert!(class.is_power_of_two() || class.is_multiple_of(8));
        }

        // Test colocated metadata optimization
        let obj_ref =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x4000)) };
        let metadata_addr = optimizer.colocate_metadata(obj_ref.to_raw_address(), 64);

        // Verify metadata is within same cache line or adjacent
        let obj_addr = obj_ref.to_raw_address().as_usize();
        let meta_addr = metadata_addr.as_usize();
        let distance = meta_addr.abs_diff(obj_addr);
        assert!(distance <= CACHE_LINE_SIZE * 2);
    }

    #[test]
    fn test_locality_aware_work_distribution() {
        let mut stealer = LocalityAwareWorkStealer::new(10);

        // Test NUMA-aware work distribution
        let local_objs: Vec<ObjectReference> = (0..5)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000000 + i * 64))
            })
            .collect();

        let remote_objs: Vec<ObjectReference> = (0..5)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x2000000 + i * 64))
            })
            .collect();

        // Add all objects
        let mut all_objs = local_objs;
        all_objs.extend(&remote_objs);
        stealer.add_objects(all_objs);

        // Get batch and verify we get work
        let batch = stealer.get_next_batch(5);
        assert_eq!(batch.len(), 5);

        // Try stealing from another stealer
        let stealer2 = LocalityAwareWorkStealer::new(10);
        let can_steal = stealer2.steal_from(&stealer);
        assert!(!can_steal); // Should not steal unless above threshold

        // Check statistics
        let (local_proc, _stolen, _shared) = stealer.get_stats();
        assert_eq!(local_proc, 5);
    }

    #[test]
    fn test_cache_stats_tracking() {
        let marking = CacheOptimizedMarking::new(2);

        // Simulate object marking
        let obj =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000)) };

        marking.mark_object(obj);

        let stats = marking.get_cache_stats();
        assert_eq!(stats.objects_marked, 1);
        assert_eq!(stats.prefetch_distance, 2);

        // Test batch marking
        let objects: Vec<ObjectReference> = (0..5)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x2000 + i * 64))
            })
            .collect();

        marking.mark_objects_batch(&objects);

        let stats_after = marking.get_cache_stats();
        assert_eq!(stats_after.objects_marked, 6);
    }

    #[test]
    fn test_generational_cache_optimization() {
        // Test that young generation objects are colocated for better cache performance
        let optimizer = MemoryLayoutOptimizer::new();

        // Young generation objects should be allocated together
        let young_sizes = vec![16, 24, 32];
        let young_layouts = optimizer.calculate_object_layout(&young_sizes);

        // Verify objects are tightly packed
        for i in 1..young_layouts.len() {
            let (prev_addr, prev_size) = young_layouts[i - 1];
            let (curr_addr, _) = young_layouts[i];
            let prev_end = prev_addr.as_usize() + optimizer.get_size_class(prev_size);

            // Objects should be contiguous or at most one cache line apart
            let gap = curr_addr.as_usize() - prev_end;
            assert!(gap <= CACHE_LINE_SIZE);
        }

        // Old generation objects can be more sparsely allocated
        let old_sizes = vec![512, 1024, 2048];
        let old_layouts = optimizer.calculate_object_layout(&old_sizes);

        // Verify large objects are cache-line aligned
        for (addr, size) in &old_layouts {
            if *size >= CACHE_LINE_SIZE {
                assert_eq!(addr.as_usize() % CACHE_LINE_SIZE, 0);
            }
        }
    }
}
