//! Test heap facade for deterministic segment and sweep behavior
#![allow(dead_code, unused_imports, unused_variables)]
//! 
//! This module provides a simplified heap implementation for testing that
//! forces deterministic segment/page behavior and enables sweep-color assertions.

#[cfg(any(test, feature = "smoke"))]
use crate::{
    GcHeader, SendPtr,
    collector::sweep_coordinator::SweepCoordinator,
};
#[cfg(any(test, feature = "smoke"))]
use std::sync::{Arc, Mutex};
#[cfg(any(test, feature = "smoke"))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(any(test, feature = "smoke"))]
use parking_lot::RwLock;

/// A deterministic test heap that provides controlled allocation and sweep behavior.
///
/// This heap forces objects into specific segments and pages for testing,
/// allowing verification of sweep behavior and color assertions.
///
/// # Examples
///
/// ```
/// #[cfg(feature = "smoke")]
/// use fugrip::test_heap::TestHeap;
///
/// #[cfg(feature = "smoke")]
/// {
///     let heap = TestHeap::new();
///     let segment_id = heap.add_test_segment(1024); // 1KB segment
///     
///     // Objects allocated will go to specific segments in deterministic order
///     assert_eq!(heap.segment_count(), 1);
/// }
/// ```
#[cfg(any(test, feature = "smoke"))]
pub struct TestHeap {
    segments: Arc<Mutex<Vec<TestSegment>>>,
    _current_segment: AtomicUsize,
    _collector: Arc<crate::CollectorState>,
    objects: Arc<RwLock<Vec<SendPtr<GcHeader<()>>>>>,
}

/// A test segment with deterministic allocation behavior
#[cfg(any(test, feature = "smoke"))]
pub struct TestSegment {
    pub id: usize,
    pub memory: Box<[u8]>,
    pub allocated: AtomicUsize,
    pub objects: Arc<RwLock<Vec<SendPtr<GcHeader<()>>>>>,
    pub swept: AtomicBool,
}

#[cfg(any(test, feature = "smoke"))]
impl TestSegment {
    /// Create a new test segment with deterministic behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_heap::TestSegment;
    ///
    /// let segment = TestSegment::new(0, 1024);
    /// assert_eq!(segment.id, 0);
    /// assert!(!segment.is_swept());
    /// assert_eq!(segment.get_objects().len(), 0);
    /// ```
    #[cfg(any(test, feature = "smoke"))]
    pub fn new(id: usize, size: usize) -> Self {
        let memory = vec![0u8; size].into_boxed_slice();
        Self {
            id,
            memory,
            allocated: AtomicUsize::new(0),
            objects: Arc::new(RwLock::new(Vec::new())),
            swept: AtomicBool::new(false),
        }
    }

    pub fn add_object(&self, ptr: SendPtr<GcHeader<()>>) {
        self.objects.write().push(ptr);
    }

    pub fn mark_swept(&self) {
        self.swept.store(true, Ordering::Release);
    }

    /// Check if this segment has been swept.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_heap::TestSegment;
    ///
    /// let segment = TestSegment::new(0, 1024);
    /// assert!(!segment.is_swept());
    ///
    /// segment.mark_swept();
    /// assert!(segment.is_swept());
    /// ```
    #[cfg(any(test, feature = "smoke"))]
    pub fn is_swept(&self) -> bool {
        self.swept.load(Ordering::Acquire)
    }

    pub fn get_objects(&self) -> Vec<SendPtr<GcHeader<()>>> {
        self.objects.read().clone()
    }
}

#[cfg(any(test, feature = "smoke"))]
impl TestHeap {
    /// Create a new test heap with deterministic behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_heap::TestHeap;
    ///
    /// let heap = TestHeap::new();
    /// assert_eq!(heap.segment_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            segments: Arc::new(Mutex::new(Vec::new())),
            _current_segment: AtomicUsize::new(0),
            _collector: crate::memory::COLLECTOR.clone(),
            objects: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a test segment with specified size.
    ///
    /// Returns the ID of the newly created segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_heap::TestHeap;
    ///
    /// let heap = TestHeap::new();
    /// let segment_id = heap.add_test_segment(1024);
    /// assert_eq!(segment_id, 0);
    /// assert_eq!(heap.segment_count(), 1);
    /// ```
    pub fn add_test_segment(&self, size: usize) -> usize {
        let mut segments = self.segments.lock().unwrap();
        let id = segments.len();
        let segment = TestSegment::new(id, size);
        segments.push(segment);
        id
    }

    /// Get the current number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.lock().unwrap().len()
    }

    /// Add an object to the specified segment for testing.
    ///
    /// This allows tests to control which segment objects are allocated in,
    /// enabling deterministic sweep testing.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{test_heap::TestHeap, Gc, SendPtr};
    ///
    /// let heap = TestHeap::new();
    /// let segment_id = heap.add_test_segment(1024);
    /// 
    /// let obj = Gc::new(42i32);
    /// let ptr = unsafe { SendPtr::new(obj.as_ptr() as *mut fugrip::GcHeader<()>) };
    /// heap.add_object_to_segment(segment_id, ptr);
    /// 
    /// assert_eq!(heap.get_segment_object_count(segment_id), 1);
    /// ```
    pub fn add_object_to_segment(&self, segment_id: usize, ptr: SendPtr<GcHeader<()>>) {
        let segments = self.segments.lock().unwrap();
        if let Some(segment) = segments.get(segment_id) {
            segment.add_object(ptr);
        }
        
        // Also add to global object list
        self.objects.write().push(ptr);
    }

    /// Get the number of objects in a specific segment.
    pub fn get_segment_object_count(&self, segment_id: usize) -> usize {
        let segments = self.segments.lock().unwrap();
        segments.get(segment_id)
            .map(|s| s.get_objects().len())
            .unwrap_or(0)
    }

    /// Get all objects in a specific segment.
    pub fn get_segment_objects(&self, segment_id: usize) -> Vec<SendPtr<GcHeader<()>>> {
        let segments = self.segments.lock().unwrap();
        segments.get(segment_id)
            .map(|s| s.get_objects())
            .unwrap_or_default()
    }

    /// Perform a deterministic sweep of a specific segment.
    ///
    /// This method allows tests to control exactly when and how segments
    /// are swept, enabling verification of sweep behavior.
    ///
    /// # Returns
    ///
    /// A `SweepResult` containing statistics about the sweep operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{test_heap::TestHeap, CollectorPhase};
    ///
    /// let heap = TestHeap::new();
    /// let segment_id = heap.add_test_segment(1024);
    /// 
    /// // Add some objects and mark them
    /// // ... object setup code ...
    /// 
    /// let result = heap.sweep_segment_deterministic(segment_id);
    /// assert_eq!(result.segment_id, segment_id);
    /// ```
    pub fn sweep_segment_deterministic(&self, segment_id: usize) -> SweepResult {
        let segments = self.segments.lock().unwrap();
        let sweep_coordinator = SweepCoordinator::new();
        
        if let Some(segment) = segments.get(segment_id) {
            let objects = segment.get_objects();
            let mut live_count = 0;
            let mut dead_count = 0;
            let mut bytes_reclaimed = 0;

            // Process each object deterministically
            for &header_ptr in &objects {
                unsafe {
                    let header = &*header_ptr.as_ptr();
                    
                    if header.mark_bit.load(Ordering::Acquire) {
                        // Object is live
                        live_count += 1;
                        // Clear mark bit for next cycle
                        header.mark_bit.store(false, Ordering::Release);
                    } else {
                        // Object is dead - process it
                        dead_count += 1;
                        bytes_reclaimed += header.type_info.size;
                        
                        // Use sweep coordinator to process dead object
                        let free_singleton = crate::FreeSingleton::instance();
                        
                        // Invalidate weak references
                        let weak_head = header.weak_ref_list.load(Ordering::Acquire);
                        if !weak_head.is_null() {
                            crate::Weak::<()>::invalidate_weak_chain(weak_head);
                        }
                        
                        // Run destructor
                        (header.type_info.drop_fn)(header_ptr.as_ptr());
                        
                        // Set forwarding pointer to FREE_SINGLETON
                        header.forwarding_ptr.store(free_singleton, Ordering::Release);
                    }
                }
            }

            // Mark segment as swept
            segment.mark_swept();

            SweepResult {
                segment_id,
                objects_processed: objects.len(),
                live_objects: live_count,
                dead_objects: dead_count,
                bytes_reclaimed,
                swept: true,
            }
        } else {
            SweepResult {
                segment_id,
                objects_processed: 0,
                live_objects: 0,
                dead_objects: 0,
                bytes_reclaimed: 0,
                swept: false,
            }
        }
    }

    /// Sweep all segments deterministically.
    ///
    /// This performs a complete heap sweep in deterministic segment order,
    /// useful for testing complete collection cycles.
    pub fn sweep_all_deterministic(&self) -> Vec<SweepResult> {
        let segment_count = self.segment_count();
        let mut results = Vec::with_capacity(segment_count);
        
        for segment_id in 0..segment_count {
            results.push(self.sweep_segment_deterministic(segment_id));
        }
        
        results
    }

    /// Check if a segment has been swept.
    pub fn is_segment_swept(&self, segment_id: usize) -> bool {
        let segments = self.segments.lock().unwrap();
        segments.get(segment_id)
            .map(|s| s.is_swept())
            .unwrap_or(false)
    }

    /// Reset all segments to unswept state for testing.
    pub fn reset_sweep_state(&self) {
        let segments = self.segments.lock().unwrap();
        for segment in segments.iter() {
            segment.swept.store(false, Ordering::Release);
        }
    }

    /// Get allocation color based on sweep state of the segment.
    ///
    /// In FUGC, allocation color depends on whether the page/segment
    /// has been swept in the current cycle:
    /// - Black allocation: on unswept pages (objects start marked)
    /// - White allocation: on swept pages (objects start unmarked)
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_heap::TestHeap;
    ///
    /// let heap = TestHeap::new();
    /// let segment_id = heap.add_test_segment(1024);
    /// 
    /// // Before sweep: black allocation
    /// assert!(heap.get_allocation_color(segment_id), "Should allocate black on unswept segment");
    /// 
    /// heap.sweep_segment_deterministic(segment_id);
    /// 
    /// // After sweep: white allocation
    /// assert!(!heap.get_allocation_color(segment_id), "Should allocate white on swept segment");
    /// ```
    pub fn get_allocation_color(&self, segment_id: usize) -> bool {
        // Black allocation on unswept pages, white on swept pages
        !self.is_segment_swept(segment_id)
    }

    /// Create a test object with specified allocation color.
    ///
    /// This allows tests to create objects with specific mark states
    /// to simulate allocation on swept vs unswept pages.
    pub fn create_test_object_with_color<T>(&self, value: T, allocate_black: bool) -> crate::Gc<T>
    where
        T: crate::traits::GcTrace + 'static,
    {
        use std::marker::PhantomData;
        
        // Create the object header with specified color
        let header = GcHeader {
            mark_bit: AtomicBool::new(allocate_black),
            type_info: crate::types::type_info::<T>(),
            forwarding_ptr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            weak_ref_list: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            data: value,
        };
        
        let boxed_header = Box::into_raw(Box::new(header));
        
        crate::Gc {
            ptr: std::sync::atomic::AtomicPtr::new(boxed_header),
            _phantom: PhantomData,
        }
    }
}

/// Result of a deterministic sweep operation
#[cfg(any(test, feature = "smoke"))]
#[derive(Debug, Clone)]
pub struct SweepResult {
    pub segment_id: usize,
    pub objects_processed: usize,
    pub live_objects: usize,
    pub dead_objects: usize,
    pub bytes_reclaimed: usize,
    pub swept: bool,
}

#[cfg(any(test, feature = "smoke"))]
impl Default for TestHeap {
    fn default() -> Self {
        Self::new()
    }
}
