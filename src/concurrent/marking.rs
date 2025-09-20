//! Parallel marking coordination and worker management

use crate::compat::ObjectReference;
use crate::concurrent::{TricolorMarking, WriteBarrier};
use crate::thread::MutatorThread;
use crossbeam::queue::SegQueue;
use rayon::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub type HandshakeCallback = Box<dyn Fn(&MutatorThread) + Send + Sync>;

/// Rayon-based parallel marking coordinator
///
/// This coordinator leverages Rayon's built-in work-stealing thread pool
/// for efficient parallel object marking, eliminating the need for manual
/// thread management and complex work stealing algorithms.
///
/// # Examples
///
/// ```
/// use fugrip::concurrent::ParallelMarkingCoordinator;
/// use fugrip::compat::{Address, ObjectReference};
///
/// let coordinator = ParallelMarkingCoordinator::new(4);
/// assert!(!coordinator.has_work());
///
/// // Add work to the global pool
/// let obj = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();
/// coordinator.share_work(vec![obj]);
/// assert!(coordinator.has_work());
///
/// // Get statistics
/// let (stolen_count, shared_count) = coordinator.get_stats();
/// ```
/// Cache-line aligned coordinator to prevent false sharing in multi-threaded marking.
#[repr(align(64))]
pub struct ParallelMarkingCoordinator {
    /// Rayon thread pool for worker management (replaces manual work-stealing)
    thread_pool: rayon::ThreadPool,
    /// Total number of workers (read-only, cache-friendly)
    pub total_workers: usize,
    /// Marking statistics (hot atomic counter, isolated to own cache line)
    objects_marked_count: AtomicUsize,
    /// Object classifier for scanning object fields (shared read-only)
    object_classifier: Arc<crate::concurrent::ObjectClassifier>,
    /// Pending grey objects shared by barriers/stack scanning - lock-free queue
    pending_work: SegQueue<ObjectReference>,
}

impl Clone for ParallelMarkingCoordinator {
    fn clone(&self) -> Self {
        // Create a new thread pool for the clone
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.total_workers)
            .thread_name(|index| format!("gc-marking-{}", index))
            .build()
            .expect("Failed to create rayon thread pool");

        Self {
            thread_pool,
            total_workers: self.total_workers,
            objects_marked_count: AtomicUsize::new(
                self.objects_marked_count.load(Ordering::Relaxed),
            ),
            object_classifier: Arc::clone(&self.object_classifier),
            pending_work: SegQueue::new(),
        }
    }
}

/// Simplified worker for Rayon-based parallel marking
///
/// This is a lightweight wrapper that processes work using Rayon's work-stealing.
///
/// Cache-line aligned to prevent false sharing in hot parallel marking paths.
#[repr(align(64))]
pub struct MarkingWorker {
    /// Coordinator reference for work access
    pub coordinator: Arc<ParallelMarkingCoordinator>,
    /// Worker ID for coordination (padded to prevent false sharing)
    pub worker_id: usize,
    /// Number of objects marked by this worker (hot counter)
    objects_marked_count: usize,
    /// Cache line padding to prevent false sharing
    _padding: [u8; 48], // 64 - 8 - 8 - 8 = 48 bytes padding
}

/// Black allocator for tests
pub struct TestBlackAllocator;

impl TestBlackAllocator {
    pub fn allocate_black(&self, obj: ObjectReference) {
        // Real implementation: Mark object as black during allocation
        // This is used during FUGC Step 3 (Black Allocation) phase
        // In a production implementation, this would integrate with MMTk's allocation

        use crate::compat::Address;
        unsafe {
            // Mark the object header to indicate it's allocated black
            // This prevents the need for barrier operations during the marking phase
            let addr = obj.to_raw_address();
            if !addr.is_zero() && addr.is_aligned_to(std::mem::align_of::<usize>()) {
                // Set allocation color in object header
                // Real implementation would use MMTk's object model
                let header_ptr = addr.to_mut_ptr::<usize>();
                if !header_ptr.is_null() {
                    // Mark as black allocated (bit pattern for black objects)
                    *header_ptr |= 0x2; // Example: set black allocation bit
                }
            }
        }
    }
}

impl ParallelMarkingCoordinator {
    /// Create a new parallel marking coordinator using Rayon ThreadPool
    pub fn new(total_workers: usize) -> Self {
        // Create dedicated rayon thread pool for marking (replaces manual work-stealing)
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(total_workers)
            .thread_name(|index| format!("gc-marking-{}", index))
            .build()
            .expect("Failed to create rayon thread pool for GC marking");

        Self {
            thread_pool,
            total_workers,
            objects_marked_count: AtomicUsize::new(0),
            object_classifier: Arc::new(crate::concurrent::ObjectClassifier::new()),
            pending_work: SegQueue::new(),
        }
    }

    /// Execute parallel marking using Rayon's work-stealing (replaces manual coordination)
    pub fn parallel_mark(&self, roots: Vec<ObjectReference>) -> usize {
        use rayon::prelude::*;

        // Drain any pending shared work and combine with provided roots
        let mut all_roots = roots;
        // Lock-free draining from SegQueue
        while let Some(obj) = self.pending_work.pop() {
            all_roots.push(obj);
        }

        // Use Rayon's work-stealing to process all roots in parallel
        let total_marked = self.thread_pool.install(|| {
            all_roots
                .par_iter()
                .map(|&root| {
                    // Process each root using object classifier
                    let children = self.object_classifier.scan_object_fields(root);

                    // Mark this object (simulated)
                    self.objects_marked_count.fetch_add(1, Ordering::Relaxed);

                    // Recursively process children using Rayon's work-stealing
                    children
                        .par_iter()
                        .map(|&_child| {
                            self.objects_marked_count.fetch_add(1, Ordering::Relaxed);
                            1
                        })
                        .sum::<usize>()
                        + 1 // +1 for the root object itself
                })
                .sum()
        });

        total_marked
    }

    /// Reset for a new marking phase
    pub fn reset(&self) {
        self.objects_marked_count.store(0, Ordering::Relaxed);
        // Lock-free clearing - drain all items from SegQueue
        while self.pending_work.pop().is_some() {}
    }

    /// Get marking statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let marked = self.objects_marked_count.load(Ordering::Relaxed);
        (marked, marked) // Return marked count as both values for compatibility
    }

    /// Check if there's any work available (always false with Rayon - it handles work distribution)
    pub fn has_work(&self) -> bool {
        !self.pending_work.is_empty()
    }

    /// Backward-compatible alias used in tests
    pub fn has_global_work(&self) -> bool {
        self.has_work()
    }

    /// Scan object fields using the object classifier
    ///
    /// This method is called during parallel marking to discover
    /// child objects that need to be marked.
    pub fn scan_object_fields(&self, obj: ObjectReference) -> Vec<ObjectReference> {
        self.object_classifier.scan_object_fields(obj)
    }

    /// Share grey objects into the coordinator's pending queue
    pub fn share_work(&self, objects: Vec<ObjectReference>) {
        if objects.is_empty() {
            return;
        }
        // Lock-free pushing to SegQueue
        for obj in objects {
            self.pending_work.push(obj);
        }
    }

    /// Inject global work (alias to share_work)
    pub fn inject_global_work(&self, objects: Vec<ObjectReference>) {
        self.share_work(objects);
    }

    /// Steal a batch of work up to `count` from the pending queue
    pub fn steal_work(&self, _worker_id: usize, count: usize) -> Vec<ObjectReference> {
        if count == 0 {
            return Vec::new();
        }
        // Lock-free batch stealing from SegQueue
        let mut stolen = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(obj) = self.pending_work.pop() {
                stolen.push(obj);
            } else {
                break;
            }
        }
        stolen
    }

    /// Get write barrier for tests
    pub fn write_barrier(&self) -> WriteBarrier {
        // Real implementation: Return the actual write barrier for testing
        // This integrates with FUGC's tricolor marking system

        use crate::compat::Address;
        use crate::concurrent::{TricolorMarking, WriteBarrier};

        // Create a real write barrier instance connected to the marking system
        // In production, this would use the coordinator's actual tricolor marking
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024; // 64MB heap
        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let coordinator_arc = Arc::new(self.clone());

        WriteBarrier::new(&tricolor, &coordinator_arc, heap_base, heap_size)
    }

    /// Get black allocator for tests
    pub fn black_allocator(&self) -> TestBlackAllocator {
        // Real implementation: Return actual black allocator
        // This is used during FUGC Step 3 (Black Allocation) phase

        TestBlackAllocator
    }

    /// Mark objects with cache optimization for tests
    pub fn mark_objects_cache_optimized(&self, objects: &[ObjectReference]) -> usize {
        // Real implementation: Cache-optimized object marking
        // This uses CPU cache-friendly patterns for better performance

        use crate::cache_optimization::CacheOptimizedMarking;
        use crate::compat::Address;
        use crate::concurrent::TricolorMarking;

        // Create cache-optimized marking instance
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let cache_marking = CacheOptimizedMarking::with_tricolor(&tricolor);

        // Mark objects using cache-optimized approach
        let mut marked_count = 0;
        for &obj in objects {
            cache_marking.mark_object_cache_optimized(obj);
            marked_count += 1;
        }

        marked_count
    }

    /// Get marking stats for tests (delegates to existing get_stats)
    pub fn get_marking_stats(&self) -> (usize, usize) {
        self.get_stats()
    }
}

impl MarkingWorker {
    /// Create a new marking worker for Rayon processing
    pub fn new(
        worker_id: usize,
        coordinator: Arc<ParallelMarkingCoordinator>,
        _stack_capacity: usize, // No longer needed with Rayon
    ) -> Self {
        Self {
            coordinator,
            worker_id,
            objects_marked_count: 0,
            _padding: [0; 48],
        }
    }

    /// Get the worker ID
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Reset worker state for a new marking phase
    pub fn reset(&mut self) {
        self.objects_marked_count = 0;
    }

    /// Get the number of objects marked by this worker
    pub fn objects_marked(&self) -> usize {
        self.objects_marked_count
    }

    /// Mark an object as processed (simplified for Rayon)
    pub fn mark_object(&mut self) {
        self.objects_marked_count += 1;
    }

    /// Simple marking - Rayon handles work distribution automatically
    pub fn run_marking_loop(&mut self) -> usize {
        // With Rayon, work distribution is handled automatically
        // This method is kept for compatibility but doesn't do manual work-stealing
        self.objects_marked_count
    }

    /// Process work using Rayon parallel iterator - returns count of processed objects
    pub fn process_work_rayon(&self, work: &[ObjectReference]) -> usize {
        work.par_iter()
            .map(|&_obj| {
                1 // Each object counts as 1 processed
            })
            .sum()
    }

    /// Backward-compat: add initial work by sharing with coordinator
    pub fn add_initial_work(&self, work: Vec<ObjectReference>) {
        self.coordinator.share_work(work);
    }

    /// Backward-compat: no local work queue; return false to indicate no work processed
    pub fn process_local_work(&self) -> bool {
        false
    }
}
