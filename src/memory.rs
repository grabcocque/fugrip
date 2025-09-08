use crate::collector_phases::{CollectorState, MUTATOR_STATE};
use crate::traits::*;
use crate::{SendPtr, GcHeader, Weak, ObjectClass, CollectorPhase, Gc, type_info, align_up, TypeInfo, FinalizableObject};
use once_cell::sync::Lazy;
use parking_lot;

/// Memory region abstraction for tracking allocations and free space.
///
/// `MemoryRegion` provides a higher-level interface over memory segments for
/// allocation tracking and free space management. It maintains a free list
/// for efficient memory reuse.
///
/// # Examples
///
/// ```
/// use fugrip::MemoryRegion;
///
/// // Create a memory region from 0x1000 to 0x2000
/// let region = MemoryRegion::new(0x1000, 0x2000);
/// 
/// // Allocate 256 bytes
/// if let Some(ptr) = region.allocate(256) {
///     println!("Allocated at: 0x{:x}", ptr);
///     
///     // Deallocate when done
///     region.deallocate(ptr, 256);
/// }
/// ```
pub struct MemoryRegion {
    pub start: usize,
    pub end: usize,
    pub allocated: AtomicUsize,
    pub free_list: Mutex<Vec<(usize, usize)>>, // (start, size) pairs
}

impl MemoryRegion {
    /// Create a new memory region from start to end address.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::MemoryRegion;
    ///
    /// let region = MemoryRegion::new(0x1000, 0x2000);
    /// assert_eq!(region.start, 0x1000);
    /// assert_eq!(region.end, 0x2000);
    /// ```
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            allocated: AtomicUsize::new(0),
            free_list: Mutex::new(vec![(start, end - start)]),
        }
    }
    
    /// Allocate a block of memory of the specified size.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::MemoryRegion;
    ///
    /// let region = MemoryRegion::new(0x1000, 0x2000);
    /// if let Some(ptr) = region.allocate(256) {
    ///     assert!(ptr >= 0x1000);
    ///     assert!(ptr < 0x2000);
    /// }
    /// ```
    pub fn allocate(&self, size: usize) -> Option<usize> {
        let mut free_list = self.free_list.lock().unwrap();
        
        // Find a suitable free block
        for i in 0..free_list.len() {
            let (block_start, block_size) = free_list[i];
            if block_size >= size {
                // Remove this block and potentially add remainder
                free_list.remove(i);
                if block_size > size {
                    // Add remainder back to free list
                    free_list.push((block_start + size, block_size - size));
                }
                
                self.allocated.fetch_add(size, Ordering::Relaxed);
                return Some(block_start);
            }
        }
        None
    }
    
    /// Deallocate a previously allocated block of memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::MemoryRegion;
    ///
    /// let region = MemoryRegion::new(0x1000, 0x2000);
    /// if let Some(ptr) = region.allocate(256) {
    ///     region.deallocate(ptr, 256);
    ///     // Memory is now available for reallocation
    /// }
    /// ```
    pub fn deallocate(&self, ptr: usize, size: usize) {
        self.allocated.fetch_sub(size, Ordering::Relaxed);
        let mut free_list = self.free_list.lock().unwrap();
        free_list.push((ptr, size));
        // In a real implementation, we'd coalesce adjacent free blocks
    }
}

/// Simple heap abstraction for managing multiple memory regions.
///
/// `Heap` manages a collection of memory regions and tracks total
/// allocation statistics across all regions.
///
/// # Examples
///
/// ```
/// use fugrip::{Heap, MemoryRegion};
///
/// let mut heap = Heap::new();
/// assert_eq!(heap.get_total_capacity(), 0);
/// 
/// // Add a memory region
/// let region = MemoryRegion::new(0x1000, 0x2000);
/// heap.add_region(region);
/// assert_eq!(heap.get_total_capacity(), 0x1000); // 4KB
/// ```
pub struct Heap {
    regions: Vec<MemoryRegion>,
    total_allocated: AtomicUsize,
    total_capacity: AtomicUsize,
}

impl Heap {
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            total_allocated: AtomicUsize::new(0),
            total_capacity: AtomicUsize::new(0),
        }
    }
    
    pub fn add_region(&mut self, region: MemoryRegion) {
        let capacity = region.end - region.start;
        self.total_capacity.fetch_add(capacity, Ordering::Relaxed);
        self.regions.push(region);
    }
    
    pub fn get_total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }
    
    pub fn get_total_capacity(&self) -> usize {
        self.total_capacity.load(Ordering::Relaxed)
    }
}

/// Simple weak reference implementation for testing purposes.
///
/// `WeakReference<T>` provides a basic weak reference that doesn't prevent
/// its target from being deallocated. It includes validity tracking to
/// detect when the target has been freed.
///
/// # Examples
///
/// ```
/// use fugrip::WeakReference;
///
/// let value = Box::into_raw(Box::new(42i32));
/// let weak_ref = WeakReference::new(value);
/// 
/// // Try to upgrade the weak reference
/// if let Some(ptr) = weak_ref.upgrade() {
///     assert_eq!(unsafe { *ptr }, 42);
/// }
/// 
/// // Clean up
/// unsafe { Box::from_raw(value); }
/// ```
pub struct WeakReference<T> {
    pub target: AtomicPtr<T>,
    pub is_valid: AtomicBool,
}

impl<T> WeakReference<T> {
    pub fn new(target: *mut T) -> Self {
        Self {
            target: AtomicPtr::new(target),
            is_valid: AtomicBool::new(true),
        }
    }
    
    pub fn upgrade(&self) -> Option<*mut T> {
        if self.is_valid.load(Ordering::Acquire) {
            let ptr = self.target.load(Ordering::Acquire);
            if !ptr.is_null() {
                return Some(ptr);
            }
        }
        None
    }
    
    pub fn invalidate(&self) {
        self.is_valid.store(false, Ordering::Release);
        self.target.store(std::ptr::null_mut(), Ordering::Release);
    }
}

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

/// High-level garbage-collected allocator.
///
/// `GcAllocator` provides the main allocation interface for the garbage collector.
/// It manages both thread-local fast-path allocation and global slow-path allocation
/// with automatic garbage collection triggering based on allocation thresholds.
///
/// # Examples
///
/// ```
/// use fugrip::memory::GcAllocator;
///
/// let allocator = GcAllocator::new();
/// // Allocation happens automatically through Gc::new()
/// ```
pub struct GcAllocator {
    heap: SegmentedHeap,
    collector: Arc<CollectorState>,
    allocation_threshold: AtomicUsize,
    live_bytes: AtomicUsize,
}

impl Default for GcAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl GcAllocator {
    /// Creates a new garbage collector allocator with default settings.
    ///
    /// The allocator starts with a 1MB allocation threshold that triggers
    /// garbage collection when exceeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::memory::GcAllocator;
    ///
    /// let allocator = GcAllocator::new();
    /// ```
    pub fn new() -> Self {
        GcAllocator {
            heap: SegmentedHeap::new(),
            collector: COLLECTOR.clone(),
            allocation_threshold: AtomicUsize::new(1024 * 1024), // 1MB initial threshold
            live_bytes: AtomicUsize::new(0),
        }
    }

    /// Allocates a garbage-collected object of type `T`.
    ///
    /// This method uses a two-tier allocation strategy:
    /// 1. Fast path: Try thread-local allocation from allocation buffers
    /// 2. Slow path: Fall back to global heap allocation with GC triggering
    ///
    /// # Parameters
    ///
    /// * `value` - The value to store in the garbage-collected heap
    ///
    /// # Returns
    ///
    /// A `Gc<T>` pointer to the allocated object
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{Gc, memory::GcAllocator};
    ///
    /// let allocator = GcAllocator::new();
    /// let gc_value = allocator.allocate_gc(42i32);
    /// // Normally called automatically via Gc::new()
    /// ```
    pub fn allocate_gc<T: GcTrace + 'static>(&self, value: T) -> Gc<T> {
        // Fast path: thread-local allocation
        let ptr = MUTATOR_STATE.with(|state| state.borrow_mut().try_allocate::<T>());

        if let Some(ptr) = ptr {
            let allocating_black = MUTATOR_STATE.with(|state| state.borrow().allocating_black);
            unsafe {
                let header = GcHeader {
                    mark_bit: AtomicBool::new(allocating_black),
                    type_info: type_info::<T>(),
                    forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                    weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
                    data: value,
                };
                std::ptr::write(ptr, header);
                return Gc {
                    ptr: AtomicPtr::new(ptr),
                    _phantom: PhantomData,
                };
            }
        }

        // Slow path: global allocation with potential GC trigger
        self.allocate_slow_path(value)
    }

    fn allocate_slow_path<T: GcTrace + 'static>(&self, value: T) -> Gc<T> {
        // Check if we need to trigger GC
        if self.live_bytes.load(Ordering::Relaxed)
            > self.allocation_threshold.load(Ordering::Relaxed)
        {
            self.collector.request_collection();
        }

        // Delegate to the heap for actual allocation
        self.heap.allocate(value)
    }

    /// Get access to the underlying segmented heap for GC operations.
    /// This is used by the collector during sweeping phases.
    pub fn get_heap(&self) -> &SegmentedHeap {
        &self.heap
    }

    /// Update live bytes count (used during sweeping)
    pub fn add_live_bytes(&self, bytes: usize) {
        self.live_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Reset live bytes count (used at start of collection)
    pub fn reset_live_bytes(&self) {
        self.live_bytes.store(0, Ordering::Relaxed);
    }
}

/// Segmented allocator that manages garbage-collected memory.
///
/// The `SegmentedHeap` divides memory into large segments to reduce contention
/// and improve allocation performance. It uses lock-free allocation within
/// segments and falls back to segment expansion when needed.
///
/// # Examples
///
/// ```
/// use fugrip::memory::SegmentedHeap;
///
/// let heap = SegmentedHeap::new();
/// // Allocation happens automatically through Gc::new()
/// ```
pub struct SegmentedHeap {
    pub segments: Mutex<Vec<Segment>>,
    pub current_segment: AtomicUsize,
    pub collector_state: Arc<CollectorState>,
}

impl Default for SegmentedHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentedHeap {
    /// Creates a new segmented heap with an initial segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::memory::SegmentedHeap;
    ///
    /// let heap = SegmentedHeap::new();
    /// assert_eq!(heap.segment_count(), 1);
    /// ```
    pub fn new() -> Self {
        let initial_segment = Segment::new(0);
        SegmentedHeap {
            segments: Mutex::new(vec![initial_segment]),
            current_segment: AtomicUsize::new(0),
            collector_state: COLLECTOR.clone(),
        }
    }

    /// Adds a new segment to the heap and returns its ID.
    ///
    /// This method is called automatically when existing segments are full.
    /// Each segment provides 1MB of memory for allocation.
    ///
    /// # Returns
    ///
    /// The ID of the newly created segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::memory::SegmentedHeap;
    ///
    /// let heap = SegmentedHeap::new();
    /// let initial_count = heap.segment_count();
    /// let new_id = heap.add_segment();
    /// assert_eq!(heap.segment_count(), initial_count + 1);
    /// assert_eq!(new_id, initial_count);
    /// ```
    pub fn add_segment(&self) -> usize {
        let mut segments = self.segments.lock().unwrap();
        let new_id = segments.len();
        let new_segment = Segment::new(new_id);
        segments.push(new_segment);
        new_id
    }

    /// Returns the current number of segments in the heap.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::memory::SegmentedHeap;
    ///
    /// let heap = SegmentedHeap::new();
    /// assert_eq!(heap.segment_count(), 1); // Initial segment
    ///
    /// heap.add_segment();
    /// assert_eq!(heap.segment_count(), 2);
    /// ```
    pub fn segment_count(&self) -> usize {
        self.segments.lock().unwrap().len()
    }

    pub fn allocate<T: GcTrace + 'static>(&self, value: T) -> Gc<T> {
        let size = std::mem::size_of::<GcHeader<T>>();
        let align = std::mem::align_of::<GcHeader<T>>();

        // Try current segment first
        let current_seg_idx = self
            .current_segment
            .load(std::sync::atomic::Ordering::Relaxed);
        if let Some(ptr) = self.try_allocate_raw_in_segment(current_seg_idx, size, align) {
            return self.initialize_gc_at(ptr, value);
        }

        // Try other segments
        let segment_count = self.segment_count();
        for i in 0..segment_count {
            if i != current_seg_idx
                && let Some(ptr) = self.try_allocate_raw_in_segment(i, size, align)
            {
                // Update current segment hint
                self.current_segment
                    .store(i, std::sync::atomic::Ordering::Relaxed);
                return self.initialize_gc_at(ptr, value);
            }
        }

        // All segments full - add a new segment and try again
        let new_seg_id = self.add_segment();
        if let Some(ptr) = self.try_allocate_raw_in_segment(new_seg_id, size, align) {
            self.current_segment
                .store(new_seg_id, std::sync::atomic::Ordering::Relaxed);
            return self.initialize_gc_at(ptr, value);
        }

        // Even new segment failed - this shouldn't happen
        panic!("Failed to allocate even in new segment")
    }

    fn try_allocate_raw_in_segment(
        &self,
        seg_idx: usize,
        size: usize,
        align: usize,
    ) -> Option<*mut u8> {
        let segments = self.segments.lock().unwrap();
        if seg_idx >= segments.len() {
            return None;
        }

        let segment = &segments[seg_idx];

        loop {
            let current = segment
                .allocation_ptr
                .load(std::sync::atomic::Ordering::Relaxed);
            let aligned = align_up(current, align) as *mut u8;
            let new_ptr = unsafe { aligned.add(size) };

            if new_ptr > segment.end_ptr.load(std::sync::atomic::Ordering::Relaxed) {
                return None; // Segment full
            }

            if segment
                .allocation_ptr
                .compare_exchange_weak(
                    current,
                    new_ptr,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                // Record first allocated address in this segment if not set yet
                let _ = segment.allocated_start.compare_exchange(
                    std::ptr::null_mut(),
                    aligned,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                );
                return Some(aligned);
            }
        }
    }

    fn initialize_gc_at<T: GcTrace + 'static>(&self, ptr: *mut u8, value: T) -> Gc<T> {
        unsafe {
            let header = GcHeader {
                mark_bit: AtomicBool::new(false),
                type_info: type_info::<T>(),
                forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
                data: value,
            };
            std::ptr::write(ptr as *mut GcHeader<T>, header);
            Gc {
                ptr: AtomicPtr::new(ptr as *mut GcHeader<T>),
                _phantom: PhantomData,
            }
        }
    }

    /// Initialize a GC object at a pre-allocated location using a specific TypeInfo.
    /// This is used for specialized classifications like finalizable objects.
    fn initialize_gc_with_type_info<T: GcTrace + 'static>(
        &self,
        ptr: *mut u8,
        info: &'static TypeInfo,
        value: T,
    ) -> Gc<T> {
        unsafe {
            let header = GcHeader {
                mark_bit: AtomicBool::new(false),
                type_info: info,
                forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                weak_ref_list: AtomicPtr::new(std::ptr::null_mut()),
                data: value,
            };
            std::ptr::write(ptr as *mut GcHeader<T>, header);
            Gc {
                ptr: AtomicPtr::new(ptr as *mut GcHeader<T>),
                _phantom: PhantomData,
            }
        }
    }

    /// Allocate a buffer for thread-local allocation
    pub fn allocate_buffer(&self, size: usize) -> Option<BufferInfo> {
        // Try current segment first
        let current_seg_idx = self
            .current_segment
            .load(std::sync::atomic::Ordering::Relaxed);
        
        if let Some(buffer) = self.try_allocate_buffer_in_segment(current_seg_idx, size) {
            return Some(buffer);
        }
        
        // Try other segments
        let segment_count = self.segment_count();
        for i in 0..segment_count {
            if i != current_seg_idx {
                if let Some(buffer) = self.try_allocate_buffer_in_segment(i, size) {
                    self.current_segment.store(i, std::sync::atomic::Ordering::Relaxed);
                    return Some(buffer);
                }
            }
        }
        
        // All segments full - add a new segment and try again
        let new_seg_id = self.add_segment();
        if let Some(buffer) = self.try_allocate_buffer_in_segment(new_seg_id, size) {
            self.current_segment.store(new_seg_id, std::sync::atomic::Ordering::Relaxed);
            return Some(buffer);
        }
        
        None
    }

    fn try_allocate_buffer_in_segment(&self, seg_idx: usize, size: usize) -> Option<BufferInfo> {
        let segments = self.segments.lock().unwrap();
        if seg_idx >= segments.len() {
            return None;
        }
        
        let segment = &segments[seg_idx];
        
        loop {
            let current = segment.allocation_ptr.load(std::sync::atomic::Ordering::Relaxed);
            let aligned_start = crate::align_up(current, 8) as *mut u8;
            let buffer_end = unsafe { aligned_start.add(size) };
            
            if buffer_end > segment.end_ptr.load(std::sync::atomic::Ordering::Relaxed) {
                return None; // Segment doesn't have enough space
            }
            
            if segment
                .allocation_ptr
                .compare_exchange_weak(
                    current,
                    buffer_end,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                // Record first allocated address in this segment if not set yet
                let _ = segment.allocated_start.compare_exchange(
                    std::ptr::null_mut(),
                    aligned_start,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                );
                return Some(BufferInfo {
                    start: aligned_start,
                    end: buffer_end,
                    segment_id: seg_idx,
                });
            }
        }
    }
}

/// Individual memory segment within a segmented heap.
///
/// Each `Segment` represents a contiguous memory region (typically 1MB) used for
/// garbage collection. Segments include their own mark bits for concurrent marking
/// and use atomic allocation pointers for lock-free allocation.
///
/// # Examples
///
/// ```
/// use fugrip::memory::Segment;
///
/// // Create a new segment with ID 0
/// let segment = Segment::new(0);
/// assert_eq!(segment.segment_id, 0);
/// ```
pub struct Segment {
    pub memory: Box<[MaybeUninit<u8>]>,
    pub mark_bits: Box<[AtomicBool]>,
    pub allocation_ptr: AtomicPtr<u8>,
    // First allocated object boundary within this segment (null until first allocation)
    pub allocated_start: AtomicPtr<u8>,
    pub end_ptr: AtomicPtr<u8>, // Changed to AtomicPtr for thread safety
    pub segment_id: usize,
}

unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}

impl Segment {
    pub fn new(id: usize) -> Self {
        const SEGMENT_SIZE: usize = 1024 * 1024; // 1MB segments
        let memory = vec![MaybeUninit::uninit(); SEGMENT_SIZE].into_boxed_slice();

        // Create mark_bits without using vec! macro (AtomicBool doesn't implement Clone)
        let mark_bits_count = SEGMENT_SIZE / 64;
        let mut mark_bits = Vec::with_capacity(mark_bits_count);
        for _ in 0..mark_bits_count {
            mark_bits.push(AtomicBool::new(false));
        }
        let mark_bits = mark_bits.into_boxed_slice();

        let start_ptr = memory.as_ptr() as *mut u8;
        let end_ptr = unsafe { start_ptr.add(SEGMENT_SIZE) } as *const u8;

        Segment {
            memory,
            mark_bits,
            allocation_ptr: AtomicPtr::new(start_ptr),
            allocated_start: AtomicPtr::new(std::ptr::null_mut()),
            end_ptr: AtomicPtr::new(end_ptr as *mut u8),
            segment_id: id,
        }
    }
}

/// Global garbage collection allocator instance.
///
/// This is the main allocator used throughout the application for creating
/// garbage-collected objects. It's initialized lazily on first use.
///
/// # Examples
///
/// ```
/// use fugrip::memory::ALLOCATOR;
///
/// // The allocator is automatically used by Gc::new()
/// // Access to the global allocator for advanced use cases
/// let heap = ALLOCATOR.get_heap();
/// ```
pub static ALLOCATOR: Lazy<GcAllocator> = Lazy::new(GcAllocator::new);

/// Global garbage collector state instance.
///
/// This is the central coordination point for all garbage collection activities.
/// It manages collection phases, thread synchronization, and collection policies.
///
/// # Examples
///
/// ```
/// use fugrip::memory::COLLECTOR;
/// use fugrip::CollectorPhase;
///
/// // Access the global collector state
/// let phase = COLLECTOR.get_phase();
/// assert_eq!(phase, CollectorPhase::Waiting);
/// ```
pub static COLLECTOR: Lazy<Arc<CollectorState>> = Lazy::new(|| Arc::new(CollectorState::new()));


// Note: Weak reference implementation moved to types.rs

// Note: invalidate_weak_chain implementation moved to types.rs

/// Execute the census phase for weak reference cleanup.
///
/// This function initiates the census phase of garbage collection, which is
/// responsible for identifying and cleaning up invalid weak references.
///
/// # Examples
///
/// ```
/// use fugrip::memory::execute_census_phase;
///
/// // Manually trigger census phase (normally called by the collector)
/// execute_census_phase();
/// ```
pub fn execute_census_phase() {
    COLLECTOR.census_phase();
}

// Census phase implementation
impl CollectorState {
    pub fn census_phase(&self) {
        self.phase_manager.set_phase(CollectorPhase::Censusing);
        self.census_weak_references_only();
    }

    /// Execute weak reference census logic without changing phase
    pub fn census_weak_references_only(&self) {

        use crate::memory::CLASSIFIED_ALLOCATOR;
        // ObjectClass is already imported at module level
        let worker_count = num_cpus::get();

        // First, census all weak references themselves
        let weak_object_set = CLASSIFIED_ALLOCATOR.get_weak_object_set();

        // Iterate over all weak references in parallel
        weak_object_set.iterate_parallel(worker_count, |weak_ptr| {
            // Perform census operation on each weak reference
            unsafe {
                let weak_header = &*(weak_ptr.as_ptr() as *mut GcHeader<Weak<()>>);
                let weak_data = &weak_header.data;
                let target = weak_data.target.load(Ordering::Acquire);

                if !target.is_null() && !(*target).mark_bit.load(Ordering::Acquire) {
                    // Target is not marked (dead), redirect weak reference to null
                    weak_data
                        .target
                        .store(std::ptr::null_mut(), Ordering::Release);
                }
            }
        });

        // Second, census objects with Census class (objects that may need weak ref census)
        let census_object_set = CLASSIFIED_ALLOCATOR.get_object_set(ObjectClass::Census);
        census_object_set.iterate_parallel(worker_count, |obj_ptr| {
            // Inline the census logic to avoid borrowing issues
            unsafe {
                let header = &*obj_ptr.as_ptr();

                // If this object is not marked (dead), invalidate all weak references to it
                if !header.mark_bit.load(Ordering::Acquire) {
                    let weak_head = header.weak_ref_list.load(Ordering::Acquire);
                    if !weak_head.is_null() {
                        Weak::<()>::invalidate_weak_chain(weak_head);
                    }
                }
            }
        });

        // Third, census objects with CensusAndDestructor class
        let census_destructor_set =
            CLASSIFIED_ALLOCATOR.get_object_set(ObjectClass::CensusAndDestructor);
        census_destructor_set.iterate_parallel(worker_count, |obj_ptr| {
            // Inline the census logic to avoid borrowing issues
            unsafe {
                let header = &*obj_ptr.as_ptr();

                // If this object is not marked (dead), invalidate all weak references to it
                if !header.mark_bit.load(Ordering::Acquire) {
                    let weak_head = header.weak_ref_list.load(Ordering::Acquire);
                    if !weak_head.is_null() {
                        Weak::<()>::invalidate_weak_chain(weak_head);
                    }
                }
            }
        });
    }

    pub fn census_weak_reference(&self, weak_ptr: *mut GcHeader<()>) {
        unsafe {
            let weak_header = &*(weak_ptr as *mut GcHeader<Weak<()>>);
            let weak_data = &weak_header.data;
            let target = weak_data.target.load(Ordering::Acquire);

            if !target.is_null() && !(*target).mark_bit.load(Ordering::Acquire) {
                // Target is not marked (dead), redirect weak reference to null
                weak_data
                    .target
                    .store(std::ptr::null_mut(), Ordering::Release);
            }
        }
    }

    /// Census objects that may have weak references pointing to them.
    /// This method checks if the object itself is marked. If not,
    /// it invalidates all weak references pointing to it.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - The `obj_ptr` must point to a valid, initialized GcHeader
    /// - The caller must ensure the object is not being accessed by other threads
    /// - This should only be called during the census phase of garbage collection
    /// - The weak reference chain must be valid if it exists
    pub unsafe fn census_object_with_weak_refs(&self, obj_ptr: *mut GcHeader<()>) {
        unsafe {
            let header = &*obj_ptr;

            // If this object is not marked (dead), invalidate all weak references to it
            if !header.mark_bit.load(Ordering::Acquire) {
                let weak_head = header.weak_ref_list.load(Ordering::Acquire);
                if !weak_head.is_null() {
                    Weak::<()>::invalidate_weak_chain(weak_head);
                }
            }
        }
    }
}

/// Buffer for allocations within a specific segment.
///
/// `SegmentBuffer` tracks the current allocation position within a segment
/// and provides fast bump-pointer allocation until the segment is exhausted.
///
/// # Examples
///
/// ```
/// use fugrip::memory::SegmentBuffer;
///
/// let buffer = SegmentBuffer::default();
/// assert!(buffer.current.is_null());
/// assert_eq!(buffer.segment_id, 0);
/// ```
pub struct SegmentBuffer {
    pub current: *mut u8,
    pub end: *mut u8,
    pub segment_id: usize,
}

/// Information about an allocated buffer region.
pub struct BufferInfo {
    pub start: *mut u8,
    pub end: *mut u8,
    pub segment_id: usize,
}

impl Default for SegmentBuffer {
    fn default() -> Self {
        SegmentBuffer {
            current: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            segment_id: 0,
        }
    }
}

/// Object classification system for efficient parallel collection.
///
/// `ClassifiedAllocator` manages separate heaps for different object classes,
/// enabling the garbage collector to handle objects with different lifecycle
/// requirements efficiently (e.g., finalizers, weak references).
///
/// # Examples
///
/// ```
/// use fugrip::{memory::ClassifiedAllocator, ObjectClass};
///
/// // ClassifiedAllocator is typically accessed via the global CLASSIFIED_ALLOCATOR
/// // Direct construction:
/// // let allocator = ClassifiedAllocator::new();
/// ```
pub struct ClassifiedAllocator {
    pub heaps: [SegmentedHeap; 6],   // One per ObjectClass
    pub object_sets: [ObjectSet; 6], // For iteration during collection
}

impl ClassifiedAllocator {
    /// Allocates a garbage-collected object with the specified classification.
    ///
    /// This method allocates memory for the object in the appropriate heap
    /// based on its classification, enabling the garbage collector to handle
    /// different object types efficiently.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{CLASSIFIED_ALLOCATOR, ObjectClass};
    ///
    /// // Allocate a regular object
    /// let regular = CLASSIFIED_ALLOCATOR.allocate_classified(
    ///     42i32,
    ///     ObjectClass::Default
    /// );
    /// // Object allocated successfully with proper GC header
    /// assert!(!regular.as_ptr().is_null());
    ///
    /// // Allocate an object that needs finalization
    /// let finalizable = CLASSIFIED_ALLOCATOR.allocate_classified(
    ///     "cleanup me".to_string(),
    ///     ObjectClass::Finalizer
    /// );
    /// // Object allocated successfully
    /// assert!(!finalizable.as_ptr().is_null());
    /// ```
    pub fn allocate_classified<T: GcTrace + 'static>(&self, value: T, class: ObjectClass) -> Gc<T> {
        let heap_index = class as usize;
        let gc_ptr = self.heaps[heap_index].allocate(value);

        // Register in appropriate object set for collection
        unsafe { self.object_sets[heap_index].register(gc_ptr.as_ptr() as *mut GcHeader<()>) };

        gc_ptr
    }

    /// Allocate a finalizable object with proper classification and type info.
    /// The returned object is wrapped in FinalizableObject<T> and registered in
    /// the Finalizer class for reviving during GC.
    pub fn allocate_finalizable<T>(&self, value: T) -> Gc<FinalizableObject<T>>
    where
        T: crate::Finalizable + GcTrace + 'static,
    {
        let heap_index = ObjectClass::Finalizer as usize;

        // Allocate raw space from the appropriate heap
        let size = std::mem::size_of::<GcHeader<FinalizableObject<T>>>();
        let align = std::mem::align_of::<GcHeader<FinalizableObject<T>>>();

        // Try current segment then others similar to SegmentedHeap::allocate
        let heap = &self.heaps[heap_index];
        let current_seg_idx = heap.current_segment.load(Ordering::Relaxed);
        if let Some(ptr) = heap.try_allocate_raw_in_segment(current_seg_idx, size, align) {
            let info = crate::types::finalizable_type_info::<T>();
            let gc = heap.initialize_gc_with_type_info(ptr, info, FinalizableObject::new(value));
            unsafe { self.object_sets[heap_index].register(gc.as_ptr() as *mut GcHeader<()>) };
            return gc;
        }

        let segment_count = heap.segment_count();
        for i in 0..segment_count {
            if i != current_seg_idx {
                if let Some(ptr) = heap.try_allocate_raw_in_segment(i, size, align) {
                    heap.current_segment.store(i, Ordering::Relaxed);
                    let info = crate::types::finalizable_type_info::<T>();
                    let gc = heap.initialize_gc_with_type_info(ptr, info, FinalizableObject::new(value));
                    unsafe { self.object_sets[heap_index].register(gc.as_ptr() as *mut GcHeader<()>) };
                    return gc;
                }
            }
        }

        // Add a new segment and try again
        let new_seg_id = heap.add_segment();
        if let Some(ptr) = heap.try_allocate_raw_in_segment(new_seg_id, size, align) {
            heap.current_segment.store(new_seg_id, Ordering::Relaxed);
            let info = crate::types::finalizable_type_info::<T>();
            let gc = heap.initialize_gc_with_type_info(ptr, info, FinalizableObject::new(value));
            unsafe { self.object_sets[heap_index].register(gc.as_ptr() as *mut GcHeader<()>) };
            return gc;
        }

        panic!("Failed to allocate finalizable object")
    }

    /// Allocate a GC-managed Weak<T> and link it into the target's weak list.
    /// The weak node is classified under ObjectClass::Weak.
    pub fn allocate_weak<T>(&self, target: &Gc<T>) -> Gc<Weak<T>>
    where
        T: GcTrace + 'static,
    {
        // Allocate the weak node in the Weak class heap
        let weak_node = self.allocate_classified(
            Weak::<T> {
                target: AtomicPtr::new(target.as_ptr()),
                next_weak: AtomicPtr::new(std::ptr::null_mut()),
                prev_weak: AtomicPtr::new(std::ptr::null_mut()),
            },
            ObjectClass::Weak,
        );

        // Link the node into the target's weak chain
        unsafe {
            Self::link_weak_to_target_chain(target, &weak_node);
        }

        weak_node
    }

    /// Link a GC-managed Weak<T> into the target object's weak reference chain.
    unsafe fn link_weak_to_target_chain<T>(target: &Gc<T>, weak_ref: &Gc<Weak<T>>)
    where
        T: GcTrace + 'static,
    {
        let target_header = unsafe { &*target.as_ptr() };
        let weak_ptr_any = weak_ref.as_ptr() as *mut Weak<()>;

        loop {
            // Race-safety: Check forwarding pointer on each loop iteration
            // to handle concurrent sweep/invalidation
            let fwd = target_header
                .forwarding_ptr
                .load(Ordering::Acquire);
            if !fwd.is_null() {
                // Target was forwarded during linking attempt - abort
                return;
            }

            let current_head = target_header.weak_ref_list.load(Ordering::Acquire);

            let weak_data = unsafe { &(*weak_ref.as_ptr()).data };
            weak_data
                .next_weak
                .store(current_head as *mut Weak<T>, Ordering::Relaxed);
            weak_data
                .prev_weak
                .store(std::ptr::null_mut(), Ordering::Relaxed);

            if !current_head.is_null() {
                let prev_head = unsafe { &*current_head };
                // Race-safety: Use SeqCst for prev_weak to ensure visibility
                // to concurrent invalidation operations
                prev_head.prev_weak.store(weak_ptr_any, Ordering::SeqCst);
            }

            match target_header.weak_ref_list.compare_exchange_weak(
                current_head,
                weak_ptr_any,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Race-safety: Final check that target wasn't forwarded
                    // between successful CAS and now
                    let final_fwd = target_header
                        .forwarding_ptr
                        .load(Ordering::Acquire);
                    if !final_fwd.is_null() {
                        // Target was forwarded right after linking - the invalidation
                        // will handle this weak reference, so we're safe
                    }
                    break;
                },
                Err(_) => continue,
            }
        }
    }
}

/// Object sets for efficient iteration during collection phases.
///
/// `ObjectSet` maintains a collection of garbage-collected objects for a specific
/// classification, optimized for iteration during marking and sweeping phases.
/// Uses Vec instead of HashSet for better cache locality during GC traversal.
///
/// # Examples
///
/// ```
/// use fugrip::memory::ObjectSet;
///
/// let object_set = ObjectSet::new();
/// assert_eq!(object_set.len(), 0);
/// ```
pub struct ObjectSet {
    // Changed from HashSet to Vec for GC efficiency - duplicates are rare in GC
    // and Vec provides better cache locality for iteration
    pub objects: parking_lot::RwLock<Vec<SendPtr<GcHeader<()>>>>,
    pub total_bytes: std::sync::Mutex<usize>,
    pub iteration_state: parking_lot::Mutex<IterationState>,
}

/// Iteration state for tracking progress during object set traversal.
///
/// `IterationState` maintains metadata about ongoing iteration over an object
/// set, useful for resumable collection phases and progress tracking.
///
/// # Examples
///
/// ```
/// use fugrip::memory::IterationState;
///
/// // IterationState is typically managed internally by ObjectSet
/// ```
pub struct IterationState {
    _current_index: usize,
    _total_size: usize,
    _is_iterating: bool,
}

impl Default for ObjectSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectSet {
    pub fn new() -> Self {
        ObjectSet {
            objects: parking_lot::RwLock::new(Vec::new()),
            total_bytes: std::sync::Mutex::new(0),
            iteration_state: parking_lot::Mutex::new(IterationState {
                _current_index: 0,
                _total_size: 0,
                _is_iterating: false,
            }),
        }
    }

    /// # Safety
    /// The caller must ensure `ptr` is a valid pointer to an initialized
    /// `GcHeader` that will remain valid for the lifetime of registration.
    pub unsafe fn register(&self, ptr: *mut GcHeader<()>) {
        self.objects.write().push(unsafe { SendPtr::new(ptr) });
    }

    /// Iterate over all objects in parallel, applying the given function to each.
    ///
    /// This method splits the object set across multiple worker threads for
    /// efficient parallel processing during garbage collection phases.
    ///
    /// # Parameters
    ///
    /// * `worker_count` - Number of worker threads to use
    /// * `func` - Function to apply to each object
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{ObjectSet, CLASSIFIED_ALLOCATOR, ObjectClass};
    /// use fugrip::Gc;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use std::sync::Arc;
    ///
    /// // Create some objects
    /// for i in 0..10 {
    ///     CLASSIFIED_ALLOCATOR.allocate_classified(i, ObjectClass::Default);
    /// }
    ///
    /// // Get the object set and count processed objects
    /// let object_set = CLASSIFIED_ALLOCATOR.get_object_set(ObjectClass::Default);
    /// let counter = Arc::new(AtomicUsize::new(0));
    /// let counter_clone = counter.clone();
    ///
    /// // Process objects in parallel
    /// object_set.iterate_parallel(2, move |_obj_ptr| {
    ///     counter_clone.fetch_add(1, Ordering::Relaxed);
    /// });
    ///
    /// assert!(counter.load(Ordering::Relaxed) >= 10);
    /// ```
    pub fn iterate_parallel<F>(&self, worker_count: usize, func: F)
    where
        F: Fn(SendPtr<GcHeader<()>>) + Send + Sync + 'static,
    {
        let objects = self.objects.read();
        let total_objects = objects.len();

        if total_objects == 0 {
            return;
        }

        // objects is already a Vec, so we can slice directly
        let objects_vec = &*objects; // Borrow the Vec

        let chunk_size = total_objects.div_ceil(worker_count);
        let func = std::sync::Arc::new(func);
        let mut handles = Vec::new();

        for worker_id in 0..worker_count {
            let start = worker_id * chunk_size;
            let end = std::cmp::min(start + chunk_size, total_objects);

            if start >= total_objects {
                break;
            }

            let worker_objects: Vec<SendPtr<GcHeader<()>>> =
                objects_vec[start..end].iter().copied().collect();
            let worker_func = func.clone();

            let handle = std::thread::spawn(move || {
                for obj_ptr in worker_objects {
                    worker_func(obj_ptr);
                }
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            handle.join().expect("Worker thread panicked");
        }
    }

    /// Returns the number of objects currently in this set.
    ///
    /// This method provides a snapshot count of registered objects,
    /// useful for monitoring and debugging garbage collection behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{CLASSIFIED_ALLOCATOR, ObjectClass};
    ///
    /// // Initially empty
    /// let object_set = CLASSIFIED_ALLOCATOR.get_object_set(ObjectClass::Default);
    /// let initial_count = object_set.get_object_count();
    ///
    /// // Allocate some objects
    /// CLASSIFIED_ALLOCATOR.allocate_classified(1i32, ObjectClass::Default);
    /// CLASSIFIED_ALLOCATOR.allocate_classified(2i32, ObjectClass::Default);
    ///
    /// // Count should increase
    /// let new_count = object_set.get_object_count();
    /// assert!(new_count >= initial_count + 2);
    /// ```
    pub fn get_object_count(&self) -> usize {
        self.objects.read().len()
    }

    /// Add an object pointer to this set.
    /// This is an alternative to `register` with a more convenient interface.
    pub fn add(&self, ptr: SendPtr<GcHeader<()>>) {
        self.objects.write().push(ptr);
    }

    /// Get the number of objects (alias for get_object_count for compatibility)
    pub fn len(&self) -> usize {
        self.get_object_count()
    }

    /// Check if the object set is empty
    pub fn is_empty(&self) -> bool {
        self.objects.read().is_empty()
    }

    /// Apply a function to each object in the set
    pub fn for_each<F>(&self, func: F)
    where
        F: Fn(SendPtr<GcHeader<()>>),
    {
        let objects = self.objects.read();
        for obj_ptr in objects.iter() {
            func(*obj_ptr);
        }
    }
}

impl Default for ClassifiedAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassifiedAllocator {
    pub fn new() -> Self {
        ClassifiedAllocator {
            heaps: [
                SegmentedHeap::new(), // Default
                SegmentedHeap::new(), // Destructor
                SegmentedHeap::new(), // Census
                SegmentedHeap::new(), // CensusAndDestructor
                SegmentedHeap::new(), // Finalizer
                SegmentedHeap::new(), // Weak
            ],
            object_sets: [
                ObjectSet::new(),
                ObjectSet::new(),
                ObjectSet::new(),
                ObjectSet::new(),
                ObjectSet::new(),
                ObjectSet::new(),
            ],
        }
    }

    /// Get the object set for a specific object class
    pub fn get_object_set(&self, class: ObjectClass) -> &ObjectSet {
        &self.object_sets[class as usize]
    }

    /// Get the weak reference object set specifically for census operations
    pub fn get_weak_object_set(&self) -> &ObjectSet {
        &self.object_sets[ObjectClass::Weak as usize]
    }
}

// Implement AllocatorTrait for ClassifiedAllocator
impl crate::interfaces::AllocatorTrait for ClassifiedAllocator {
    fn allocate_classified<T: GcTrace + 'static>(&self, value: T, class: ObjectClass) -> Gc<T> {
        // Call the inherent method using UFCS to avoid recursion
        ClassifiedAllocator::allocate_classified(self, value, class)
    }

    fn bytes_allocated(&self) -> usize {
        // Approximate by summing segment sizes
        let segments = self.heaps[0].segments.lock().unwrap();
        segments
            .iter()
            .map(|s| {
                let start = s.memory.as_ptr() as usize;
                let end = s.end_ptr.load(std::sync::atomic::Ordering::Relaxed) as usize;
                end - start
            })
            .sum()
    }

    fn object_count(&self) -> usize {
        self.object_sets
            .iter()
            .map(|set| set.get_object_count())
            .sum()
    }
}

/// Global classified allocator instance.
///
/// This is the main allocator for objects that need specific classification
/// (finalizers, weak references, etc.). It manages separate heaps for each
/// object class to optimize garbage collection.
///
/// # Examples
///
/// ```
/// use fugrip::{CLASSIFIED_ALLOCATOR, ObjectClass};
///
/// // Allocate objects with different classifications
/// let regular = CLASSIFIED_ALLOCATOR.allocate_classified(42i32, ObjectClass::Default);
/// let finalizable = CLASSIFIED_ALLOCATOR.allocate_classified(
///     "needs cleanup".to_string(), 
///     ObjectClass::Finalizer
/// );
/// ```
pub static CLASSIFIED_ALLOCATOR: once_cell::sync::Lazy<ClassifiedAllocator> =
    once_cell::sync::Lazy::new(ClassifiedAllocator::new);

/// Global root set for garbage collection.
///
/// This collection holds references to objects that should never be garbage
/// collected. Root objects serve as starting points for garbage collection
/// marking phases and typically include global variables and long-lived data.
///
/// # Examples
///
/// ```
/// use fugrip::{ROOTS, Gc, SendPtr};
///
/// // The ROOTS collection is managed automatically by register_root()
/// // Direct access is typically not needed in user code
/// let root_count = ROOTS.lock().unwrap().len();
/// println!("Current root objects: {}", root_count);
/// ```
pub static ROOTS: once_cell::sync::Lazy<Mutex<Vec<SendPtr<GcHeader<()>>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(Vec::new()));

/// Registers a garbage-collected object as a global root.
///
/// Root objects are never collected by the garbage collector, ensuring they
/// remain accessible throughout the program's lifetime. This is useful for
/// global state and long-lived data structures.
///
/// # Examples
///
/// ```
/// use fugrip::{Gc, register_root, ROOTS};
///
/// // Create a global configuration object
/// let config = Gc::new("app_config".to_string());
///
/// // Check initial root count
/// let initial_count = ROOTS.lock().unwrap().len();
///
/// // Register it as a root so it's never collected
/// register_root(&config);
///
/// // Verify it was added to the roots
/// let new_count = ROOTS.lock().unwrap().len();
/// assert_eq!(new_count, initial_count + 1);
/// ```
pub fn register_root<T>(gc: &Gc<T>)
where
    T: GcTrace + 'static,
{
    ROOTS
        .lock()
        .unwrap()
        .push(unsafe { SendPtr::new(gc.as_ptr() as *mut GcHeader<()>) });
}

/// Scan thread stacks for potential garbage collection roots.
///
/// This function performs conservative stack scanning to identify potential
/// pointers to garbage-collected objects. It delegates to the collector's
/// sophisticated scanning implementation.
///
/// # Examples
///
/// ```
/// use fugrip::{scan_stacks, SendPtr, GcHeader};
///
/// // Stack scanning is typically called automatically by the collector
/// let mut mark_stack = Vec::<SendPtr<GcHeader<()>>>::new();
/// scan_stacks(&mut mark_stack);
/// ```
pub fn scan_stacks(_mark_stack: &mut [SendPtr<GcHeader<()>>]) {
    // Use the sophisticated stack scanning from CollectorState
    // This includes conservative scanning of thread stacks, data segments, and BSS
    COLLECTOR.mark_global_roots();

    // The collector's mark_global_roots() method populates its internal global_mark_stack
    // We could copy those roots to our mark_stack if needed, but typically the collector
    // manages its own marking process through the global_mark_stack
}
