use crate::*;
use rayon::prelude::*;
use std::sync::atomic::Ordering;

/// Coordinates parallel sweeping and pointer redirection.
///
/// This component is responsible for:
/// - Managing parallel segment sweeping operations
/// - Coordinating pointer redirection to free singleton
/// - Handling destructor execution for dead objects
/// - Managing weak reference invalidation
pub struct SweepCoordinator;

impl SweepCoordinator {
    pub fn new() -> Self {
        Self
    }

    /// Smoke-only helper: sweep a provided list of object headers.
    /// This avoids traversing real heap segments and enables safe unit tests.
    #[cfg(feature = "smoke")]
    pub fn sweep_headers_list(&self, headers: &[*mut GcHeader<()>]) {
        let free_singleton = FreeSingleton::instance();
        for &header in headers {
            if header.is_null() { continue; }
            unsafe {
                if self.is_valid_object_header(header) {
                    if !(*header).mark_bit.load(Ordering::Acquire) {
                        // Smoke mode: perform a minimal dead-object processing
                        // without heap-wide pointer redirection to avoid scanning
                        // production segments in tests.
                        let weak_head = (*header).weak_ref_list.load(Ordering::Acquire);
                        if !weak_head.is_null() {
                            Weak::<()>::invalidate_weak_chain(weak_head);
                        }
                        ((*header).type_info.drop_fn)(header);
                        (*header).forwarding_ptr.store(free_singleton, Ordering::Release);
                    } else {
                        // Clear mark bit for next cycle
                        (*header).mark_bit.store(false, Ordering::Release);
                    }
                }
            }
        }
    }

    /// Execute the sweeping phase.
    ///
    /// This phase reclaims memory from unmarked objects, redirects pointers
    /// to the free singleton, and runs destructors.
    pub fn execute_sweeping_phase(
        &self,
        phase_manager: &crate::collector::phase_manager::PhaseManager,
    ) {
        phase_manager.set_phase(CollectorPhase::Sweeping);

        let free_singleton = FreeSingleton::instance();

        // Parallel sweep with redirection
        self.sweep_all_segments_parallel(free_singleton);
    }

    /// Sweep all segments in parallel, freeing unmarked objects.
    pub fn sweep_all_segments_parallel(&self, _free_singleton: *mut GcHeader<()>) {
        use crate::interfaces::memory::HEAP_PROVIDER;

        // Get segments from the global allocator's heap
        let segments = <crate::interfaces::memory::ProductionHeapProvider as crate::interfaces::memory::HeapProvider>::get_heap(&HEAP_PROVIDER).segments.lock().unwrap();

        // Process segments in parallel
        // Note: We get the free_singleton inside each thread to avoid sharing raw pointers
        segments.par_iter().for_each(|segment| {
            let free_singleton = FreeSingleton::instance();
            self.sweep_segment(segment, free_singleton);
        });
    }

    /// Sweep a single segment, processing all objects within it.
    fn sweep_segment(&self, segment: &crate::memory::Segment, free_singleton: *mut GcHeader<()>) {
        // Only sweep the portion of the segment that has actually been allocated into.
        // This avoids probing uninitialized memory which can cause invalid header reads.
        let mut current_ptr = segment.allocated_start.load(Ordering::Relaxed);
        if current_ptr.is_null() {
            return; // Nothing allocated in this segment yet
        }
        let end_ptr = segment.allocation_ptr.load(Ordering::Relaxed);

        while current_ptr < end_ptr {
            let header = current_ptr as *mut GcHeader<()>;

            unsafe {
                // Check if this is a valid object header
                if self.is_valid_object_header(header) {
                    if !(*header).mark_bit.load(Ordering::Acquire) {
                        // Object is dead - process it for collection
                        self.process_dead_object(header, free_singleton);
                    } else {
                        // Clear mark bit for next cycle
                        (*header).mark_bit.store(false, Ordering::Release);
                    }

                    // Advance to next object boundary using recorded size
                    let obj_size = (*header).type_info.size;
                    current_ptr = current_ptr.add(obj_size);
                } else {
                    // Invalid header in allocated range: step conservatively by pointer size
                    current_ptr = current_ptr.add(std::mem::size_of::<*mut u8>());
                }
            }
        }
    }

    /// Process a dead object during sweeping.
    ///
    /// This involves:
    /// 1. Invalidating weak references
    /// 2. Redirecting pointers to free singleton
    /// 3. Running destructor
    /// 4. Setting forwarding pointer
    unsafe fn process_dead_object(
        &self,
        header: *mut GcHeader<()>,
        free_singleton: *mut GcHeader<()>,
    ) {
        unsafe {
            // Step 1: Invalidate all weak references first
            let weak_head = (*header).weak_ref_list.load(Ordering::Acquire);
            if !weak_head.is_null() {
                Weak::<()>::invalidate_weak_chain(weak_head);
            }

            // Step 2: Redirect all pointers to this dead object
            self.redirect_pointers_to_free_singleton(header, free_singleton);

            // Step 3: Run destructor if needed
            ((*header).type_info.drop_fn)(header);

            // Step 4: Mark as free by setting forwarding pointer
            (*header)
                .forwarding_ptr
                .store(free_singleton, Ordering::Release);
        }
    }

    /// Redirects all pointers to a dead object to point to the free singleton instead.
    ///
    /// This is the key innovation of FUGC - instead of moving objects, we redirect
    /// all references to dead objects to point to a special free singleton.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - Both pointers must be valid and properly aligned
    /// - `dead_obj` must point to an object that has been determined to be unreachable
    /// - `free_singleton` must point to the valid free singleton object
    /// - The caller must ensure no other threads are accessing the dead object
    /// - This should only be called during the sweep phase of garbage collection
    pub unsafe fn redirect_pointers_to_free_singleton(
        &self,
        dead_obj: *mut GcHeader<()>,
        free_singleton: *mut GcHeader<()>,
    ) {
        // This is the key innovation - redirect pointers to dead objects
        // In practice, this requires scanning all live objects and updating their pointers
        // FUGC does this efficiently by maintaining pointer maps or using conservative scanning

        self.scan_all_live_objects(|live_obj| unsafe {
            let header = &*live_obj;
            (header.type_info.redirect_pointers_fn)(live_obj, dead_obj, free_singleton);
        });
    }

    /// Scan all live objects in the heap and apply a function to each one.
    /// This is used for pointer redirection - finding all live objects that might
    /// contain pointers to dead objects and updating those pointers.
    pub fn scan_all_live_objects<F>(&self, mut visitor: F)
    where
        F: FnMut(*mut GcHeader<()>),
    {
        use crate::memory::ALLOCATOR;

        // Get segments from the global allocator's heap
        let segments = ALLOCATOR.get_heap().segments.lock().unwrap();

        // Scan each segment for live objects
        for segment in segments.iter() {
            self.scan_segment_for_live_objects(segment, &mut visitor);
        }
    }

    /// Scan a single segment for live objects and apply the visitor function to each.
    fn scan_segment_for_live_objects<F>(&self, segment: &crate::memory::Segment, visitor: &mut F)
    where
        F: FnMut(*mut GcHeader<()>),
    {
        // Scan only the allocated range within the segment to avoid uninitialized memory.
        let mut current_ptr = segment.allocated_start.load(Ordering::Relaxed);
        if current_ptr.is_null() {
            return; // No objects have been allocated in this segment
        }
        let end_ptr = segment.allocation_ptr.load(Ordering::Relaxed);

        while current_ptr < end_ptr {
            let header = current_ptr as *mut GcHeader<()>;

            unsafe {
                // Check if this is a valid object header
                if self.is_valid_object_header(header) {
                    // Check if the object is marked (alive)
                    if (*header).mark_bit.load(Ordering::Acquire) {
                        // This is a live object - apply the visitor function
                        visitor(header);
                    }

                    // Advance to next object boundary using recorded size
                    let obj_size = (*header).type_info.size;
                    current_ptr = current_ptr.add(obj_size);
                } else {
                    // Invalid header in allocated range: step conservatively
                    current_ptr = current_ptr.add(std::mem::size_of::<*mut u8>());
                }
            }
        }
    }

    /// Check if a pointer points to a valid object header.
    /// This prevents crashes from corrupted memory or alignment issues.
    fn is_valid_object_header(&self, header: *mut GcHeader<()>) -> bool {
        // Basic validation checks
        if header.is_null() {
            return false;
        }

        // Check pointer alignment
        if !(header as usize).is_multiple_of(std::mem::align_of::<GcHeader<()>>()) {
            return false;
        }

        unsafe {
            // Check if TypeInfo looks reasonable by checking size field
            let type_info = (*header).type_info;
            let size = type_info.size;

            // Check if size is reasonable (between min object size and max segment size)
            if size < std::mem::size_of::<GcHeader<()>>() || size > 1024 * 1024 {
                return false;
            }
        }

        true
    }

    /// Get statistics about the last sweep operation
    pub fn get_sweep_statistics(&self) -> SweepStatistics {
        // This could be enhanced to track actual statistics during sweeping
        SweepStatistics {
            objects_swept: 0,
            objects_freed: 0,
            bytes_reclaimed: 0,
            weak_refs_invalidated: 0,
        }
    }
}

/// Statistics about a sweep operation
#[derive(Debug, Default)]
pub struct SweepStatistics {
    pub objects_swept: usize,
    pub objects_freed: usize,
    pub bytes_reclaimed: usize,
    pub weak_refs_invalidated: usize,
}

impl Default for SweepCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
