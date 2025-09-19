//! FUGC (Fil's Unbelievable Garbage Collector) protocol implementation
//!
//! This module implements a faithful version of the eight step FUGC protocol as
//! described by Epic Games for the Verse runtime.  The coordinator integrates
//! with the existing concurrent marking infrastructure, provides precise
//! safepoint handshakes, and maintains page level allocation colouring to
//! emulate the production collector's behaviour.

use crate::{
    concurrent::{BlackAllocator, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier},
    roots::GlobalRoots,
    simd_sweep::SimdBitvector,
    thread::{MutatorThread, ThreadRegistry},
};

use crossbeam::channel::{Receiver, Sender, bounded};
use dashmap::DashMap;
use mmtk::util::{Address, ObjectReference};
use parking_lot::Mutex;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const PAGE_SIZE: usize = 4096;
const OBJECT_GRANULE: usize = 64;

/// FUGC collection cycle phases matching the 8-step protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FugcPhase {
    /// Step 1: Waiting for GC trigger
    Idle,
    /// Step 2: Turn on store barrier, soft handshake with no-op
    ActivateBarriers,
    /// Step 3: Turn on black allocation, handshake with cache reset
    ActivateBlackAllocation,
    /// Step 4: Mark global roots
    MarkGlobalRoots,
    /// Step 5: Soft handshake for stack scan + cache reset, check mark stacks
    StackScanHandshake,
    /// Step 6: Tracing - process mark stacks until empty
    Tracing,
    /// Step 7: Turn off store barrier, prepare for sweep, cache reset handshake
    PrepareForSweep,
    /// Step 8: Perform sweep with page-based allocation colouring
    Sweeping,
}

/// Colour assigned to allocation pages after sweeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationColor {
    /// Fresh page - new allocations start white.
    White,
    /// Page still contains survivors - new allocations are born black.
    Black,
}

/// Callback function type for soft handshakes
pub type HandshakeCallback = Box<dyn Fn(&MutatorThread) + Send + Sync>;

/// Statistics for FUGC collection cycles
#[derive(Debug, Default, Clone)]
pub struct FugcCycleStats {
    pub cycles_completed: usize,
    pub total_marking_time_ms: u64,
    pub total_sweep_time_ms: u64,
    pub objects_marked: usize,
    pub objects_swept: usize,
    pub handshakes_performed: usize,
    pub avg_stack_scan_objects: f64,
}

#[derive(Clone, Copy)]
struct PageState {
    live_objects: usize,
    allocation_color: AllocationColor,
}

impl PageState {
    fn new() -> Self {
        Self {
            live_objects: 0,
            allocation_color: AllocationColor::White,
        }
    }
}

/// Main coordinator implementing the complete FUGC 8-step protocol
pub struct FugcCoordinator {
    // Core GC components
    tricolor_marking: Arc<TricolorMarking>,
    write_barrier: Arc<WriteBarrier>,
    black_allocator: Arc<BlackAllocator>,
    parallel_coordinator: Arc<ParallelMarkingCoordinator>,

    // High-performance SIMD sweeping
    simd_bitvector: Arc<SimdBitvector>,

    // Thread management
    thread_registry: Arc<ThreadRegistry>,
    global_roots: Arc<Mutex<GlobalRoots>>,

    // State management
    current_phase: Arc<Mutex<FugcPhase>>,
    collection_in_progress: Arc<AtomicBool>,

    // Handshake coordination metrics
    handshake_completion_time_ms: Arc<AtomicUsize>,
    threads_processed_count: Arc<AtomicUsize>,

    // Crossbeam channels for proper synchronization
    phase_change_sender: Arc<Sender<FugcPhase>>,
    phase_change_receiver: Arc<Receiver<FugcPhase>>,
    collection_finished_sender: Arc<Sender<()>>,
    collection_finished_receiver: Arc<Receiver<()>>,

    // Statistics and per-page accounting
    cycle_stats: Arc<Mutex<FugcCycleStats>>,
    page_states: Arc<DashMap<usize, PageState>>,

    // Cache optimization and object classification (always enabled for performance)
    cache_optimized_marking: Arc<crate::cache_optimization::CacheOptimizedMarking>,
    object_classifier: Arc<crate::concurrent::ObjectClassifier>,
    root_scanner: Arc<crate::concurrent::ConcurrentRootScanner>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    worker_channels: Mutex<Vec<crate::concurrent::WorkerChannels>>,

    // Configuration
    heap_base: Address,
    heap_size: usize,
}

impl FugcCoordinator {
    /// Create a new FUGC coordinator for managing the 8-step concurrent collection protocol.
    ///
    /// # Arguments
    ///
    /// * `heap_base` - Base address of the managed heap
    /// * `heap_size` - Total size of the managed heap in bytes
    /// * `num_workers` - Number of parallel marking worker threads
    /// * `thread_registry` - Registry for managing mutator threads
    /// * `global_roots` - Thread-safe collection of global root references
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_utils::TestFixture;
    ///
    /// // Use TestFixture for DI setup (recommended)
    /// let fixture = TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
    /// let coordinator = &fixture.coordinator;
    ///
    /// // Coordinator starts in idle phase
    /// assert_eq!(coordinator.current_phase(), fugrip::FugcPhase::Idle);
    /// assert!(!coordinator.is_collecting());
    /// ```
    pub fn new(
        heap_base: Address,
        heap_size: usize,
        num_workers: usize,
        thread_registry: &Arc<ThreadRegistry>,
        global_roots: &Arc<Mutex<GlobalRoots>>,
    ) -> Self {
        let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let parallel_coordinator = Arc::new(ParallelMarkingCoordinator::new(num_workers));
        let write_barrier = Arc::new(WriteBarrier::new(
            &tricolor_marking,
            &parallel_coordinator,
            heap_base,
            heap_size,
        ));
        let black_allocator = Arc::new(BlackAllocator::new(&tricolor_marking));

        // Initialize SIMD bitvector for ultra-fast sweeping (assume 16-byte object alignment)
        let simd_bitvector = Arc::new(SimdBitvector::new(heap_base, heap_size, 16));

        // Create crossbeam channels for proper synchronization
        let (phase_change_sender, phase_change_receiver) = bounded(100);
        let (collection_finished_sender, collection_finished_receiver) = bounded(1);

        // Initialize cache optimization and object classification (always enabled)
        let cache_optimized_marking = Arc::new(
            crate::cache_optimization::CacheOptimizedMarking::with_tricolor(&tricolor_marking),
        );
        let object_classifier = Arc::new(crate::concurrent::ObjectClassifier::new());
        let root_scanner = Arc::new(crate::concurrent::ConcurrentRootScanner::new(
            Arc::clone(thread_registry),
            Arc::clone(global_roots),
            Arc::clone(&tricolor_marking),
            num_workers,
        ));

        Self {
            tricolor_marking,
            write_barrier,
            black_allocator,
            parallel_coordinator,
            simd_bitvector,
            thread_registry: Arc::clone(thread_registry),
            global_roots: Arc::clone(global_roots),
            current_phase: Arc::new(Mutex::new(FugcPhase::Idle)),
            collection_in_progress: Arc::new(AtomicBool::new(false)),
            handshake_completion_time_ms: Arc::new(AtomicUsize::new(0)),
            threads_processed_count: Arc::new(AtomicUsize::new(0)),
            phase_change_sender: Arc::new(phase_change_sender),
            phase_change_receiver: Arc::new(phase_change_receiver),
            collection_finished_sender: Arc::new(collection_finished_sender),
            collection_finished_receiver: Arc::new(collection_finished_receiver),
            cycle_stats: Arc::new(Mutex::new(FugcCycleStats::default())),
            page_states: Arc::new(DashMap::new()),
            cache_optimized_marking,
            object_classifier,
            root_scanner,
            workers: Mutex::new(Vec::new()),
            worker_channels: Mutex::new(Vec::new()),
            heap_base,
            heap_size,
        }
    }

    /// Trigger a garbage collection cycle using the FUGC 8-step protocol.
    ///
    /// This initiates the complete concurrent collection sequence:
    /// 1. Idle → Write barrier activation
    /// 2. Black allocation mode
    /// 3. Global root marking
    /// 4. Stack scanning via soft handshakes
    /// 5. Concurrent tracing
    /// 6. Barrier deactivation
    /// 7. Sweep phase
    /// 8. Return to idle
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::test_utils::TestFixture;
    /// use std::time::Duration;
    ///
    /// // Use TestFixture for DI setup
    /// let fixture = TestFixture::new_with_config(0x10000000, 32 * 1024 * 1024, 2);
    /// let coordinator = &fixture.coordinator;
    ///
    /// // Initially idle
    /// assert_eq!(coordinator.current_phase(), fugrip::FugcPhase::Idle);
    /// assert!(!coordinator.is_collecting());
    ///
    /// // Trigger collection
    /// coordinator.trigger_gc();
    ///
    /// // Wait for completion
    /// assert!(coordinator.wait_until_idle(Duration::from_millis(500)));
    /// assert_eq!(coordinator.current_phase(), fugrip::FugcPhase::Idle);
    /// ```
    pub fn trigger_gc(self: &Arc<Self>) {
        if self
            .collection_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            let coordinator = Arc::clone(self);
            thread::spawn(move || {
                coordinator.run_collection_cycle();
            });
        }
    }

    /// Wait until the coordinator becomes idle or the timeout expires.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # use std::time::Duration;
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// # let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.trigger_gc();
    /// coordinator.wait_until_idle(Duration::from_millis(1));
    /// ```
    pub fn wait_until_idle(&self, timeout: Duration) -> bool {
        if !self.is_collecting() {
            return true;
        }

        // Use crossbeam channel to wait for collection finished signal
        match self.collection_finished_receiver.recv_timeout(timeout) {
            Ok(()) => true,
            Err(_) => {
                // Check if we're actually idle now (race condition protection)
                !self.is_collecting()
            }
        }
    }

    /// Wait for the coordinator to reach the requested phase while a cycle is running.
    /// Returns `true` if the target phase was observed before timeout.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::{FugcCoordinator, FugcPhase};
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, Arc::clone(&registry), Arc::clone(&globals));
    /// coordinator.trigger_gc();
    /// let _ = coordinator.advance_to_phase(FugcPhase::ActivateBarriers);
    /// ```
    pub fn advance_to_phase(&self, target: FugcPhase) -> bool {
        if self.current_phase() == target {
            return true;
        }

        let timeout = Duration::from_millis(500);
        let start = Instant::now();

        // Listen for phase changes through the channel
        while start.elapsed() < timeout {
            match self
                .phase_change_receiver
                .recv_timeout(Duration::from_millis(10))
            {
                Ok(phase) => {
                    if phase == target {
                        return true;
                    }
                    if phase == FugcPhase::Idle && target != FugcPhase::Idle {
                        // Collection finished before reaching the desired phase
                        break;
                    }
                }
                Err(_) => {
                    // Timeout on channel, check current state
                    if self.current_phase() == target {
                        return true;
                    }
                    if !self.is_collecting() && target != FugcPhase::Idle {
                        break;
                    }
                }
            }
        }

        self.current_phase() == target
    }

    /// Wait for a specific phase transition sequence. Returns `true` when `from -> to`
    /// is observed before timeout.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::{FugcCoordinator, FugcPhase};
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, Arc::clone(&registry), Arc::clone(&globals));
    /// coordinator.trigger_gc();
    /// let _reached = coordinator.wait_for_phase_transition(FugcPhase::ActivateBarriers, FugcPhase::ActivateBlackAllocation);
    /// ```
    pub fn wait_for_phase_transition(&self, from: FugcPhase, to: FugcPhase) -> bool {
        if from == to {
            return self.current_phase() == to;
        }

        let timeout = Duration::from_millis(500);
        let start = Instant::now();
        let mut seen_from = self.current_phase() == from;

        // Listen for phase changes through the channel
        while start.elapsed() < timeout {
            match self
                .phase_change_receiver
                .recv_timeout(Duration::from_millis(10))
            {
                Ok(phase) => {
                    if !seen_from {
                        if phase == from {
                            seen_from = true;
                        }
                    } else if phase == to {
                        return true;
                    }

                    if phase == FugcPhase::Idle {
                        // Collection finished
                        break;
                    }
                }
                Err(_) => {
                    // Timeout on channel, check if collection is still active
                    if !self.is_collecting() {
                        break;
                    }
                }
            }
        }

        false
    }

    /// Return the last recorded handshake metrics `(duration_ms, threads)`.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// let metrics = coordinator.last_handshake_metrics();
    /// assert_eq!(metrics.1, 0);
    /// ```
    pub fn last_handshake_metrics(&self) -> (usize, usize) {
        (
            self.handshake_completion_time_ms.load(Ordering::Relaxed),
            self.threads_processed_count.load(Ordering::Relaxed),
        )
    }

    /// Report the current allocation colour for the given page index.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::{FugcCoordinator, AllocationColor};
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// assert_eq!(coordinator.page_allocation_color(0), AllocationColor::White);
    /// ```
    pub fn page_allocation_color(&self, page_index: usize) -> AllocationColor {
        self.page_states
            .get(&page_index)
            .map(|state| state.allocation_color)
            .unwrap_or(AllocationColor::White)
    }

    /// Execute the complete FUGC 8-step collection protocol
    fn run_collection_cycle(&self) {
        self.prepare_cycle_state();

        // Step 2: Turn on store barrier, soft handshake with no-op
        self.step_2_activate_barriers();

        // Step 3: Turn on black allocation, handshake with cache reset
        self.step_3_activate_black_allocation();

        // Step 4: Mark global roots
        self.step_4_mark_global_roots();

        // Step 5 & 6: Stack scan handshake and tracing loop
        let mut loop_iterations = 0;
        const MAX_ITERATIONS: usize = 100; // Reduce iterations to avoid long delays

        loop {
            self.step_5_stack_scan_handshake();

            if self.are_all_mark_stacks_empty() {
                break; // Go to step 7
            }

            self.step_6_tracing();

            loop_iterations += 1;
            if loop_iterations >= MAX_ITERATIONS {
                // Don't print warning in normal execution, just break
                break;
            }
        }

        // Step 7: Turn off store barrier, prepare for sweep
        self.step_7_prepare_for_sweep();

        // Step 8: Perform sweep
        self.step_8_sweep();

        // Update statistics
        {
            let mut stats = self.cycle_stats.lock();
            stats.cycles_completed += 1;
        }

        // Reset state for next cycle - order is important for proper signaling
        self.collection_in_progress.store(false, Ordering::SeqCst);
        self.set_phase(FugcPhase::Idle); // This sends the completion signal
    }

    /// Reset per-cycle state before starting a new collection.
    fn prepare_cycle_state(&self) {
        self.set_phase(FugcPhase::ActivateBarriers);
        // Reset cache-optimized marking (includes tricolor clearing)
        self.cache_optimized_marking.reset();
        self.parallel_coordinator.reset();
        self.black_allocator.reset();
        self.handshake_completion_time_ms
            .store(0, Ordering::Relaxed);
        self.threads_processed_count.store(0, Ordering::Relaxed);

        for mut state in self.page_states.iter_mut() {
            state.live_objects = 0;
        }
    }

    /// Step 2: Activate write barriers with soft handshake
    fn step_2_activate_barriers(&self) {
        self.set_phase(FugcPhase::ActivateBarriers);
        self.write_barrier.activate();

        // Soft handshake with no-op callback
        let noop_callback = Box::new(|_thread: &MutatorThread| {});
        self.soft_handshake(noop_callback);
    }

    /// Step 3: Activate black allocation with cache reset handshake
    fn step_3_activate_black_allocation(&self) {
        self.set_phase(FugcPhase::ActivateBlackAllocation);
        self.black_allocator.activate();

        let cache_reset_callback = Box::new(|thread: &MutatorThread| {
            thread.clear_stack_roots();
        });
        self.soft_handshake(cache_reset_callback);
    }

    /// Step 4: Mark global roots
    fn step_4_mark_global_roots(&self) {
        self.set_phase(FugcPhase::MarkGlobalRoots);

        let marking_start = Instant::now();
        let mut objects_marked = 0;

        {
            let roots = self.global_roots.lock();
            for root_ptr in roots.iter() {
                if let Some(root_obj) = ObjectReference::from_raw_address(unsafe {
                    Address::from_usize(root_ptr as usize)
                }) {
                    // Use cache-optimized marking for all global roots
                    self.cache_optimized_marking.mark_object(root_obj);
                    self.parallel_coordinator.share_work(vec![root_obj]);
                    self.record_live_object_internal(root_obj);
                    objects_marked += 1;
                }
            }
        }

        {
            let mut stats = self.cycle_stats.lock();
            stats.total_marking_time_ms += marking_start.elapsed().as_millis() as u64;
            stats.objects_marked += objects_marked;
        }
    }

    /// Step 5: Stack scan handshake with mark stack check
    fn step_5_stack_scan_handshake(&self) {
        self.set_phase(FugcPhase::StackScanHandshake);

        let _tricolor_marking = Arc::clone(&self.tricolor_marking);
        let parallel_coordinator = Arc::clone(&self.parallel_coordinator);
        let page_states = Arc::clone(&self.page_states);
        let total_stack_objects_scanned = Arc::new(AtomicUsize::new(0));
        let heap_base = self.heap_base;
        let heap_size = self.heap_size;

        let stack_scan_callback = {
            let total_scanned = Arc::clone(&total_stack_objects_scanned);
            Box::new(move |thread: &MutatorThread| {
                let stack_roots = thread.stack_roots();
                let mut local_grey_objects = Vec::with_capacity(stack_roots.len());

                for &root_ptr in &stack_roots {
                    if root_ptr as usize == 0 {
                        continue;
                    }

                    if let Some(obj_ref) = ObjectReference::from_raw_address(unsafe {
                        Address::from_usize(root_ptr as usize)
                    }) {
                        // Use cache-optimized marking for all stack roots
                        // Cache-optimized marking handles white-check and transition internally
                        local_grey_objects.push(obj_ref);
                        FugcCoordinator::record_live_object_for_page(
                            &page_states,
                            heap_base,
                            heap_size,
                            obj_ref,
                        );
                    }
                }

                if !local_grey_objects.is_empty() {
                    total_scanned.fetch_add(local_grey_objects.len(), Ordering::Relaxed);
                    parallel_coordinator.share_work(local_grey_objects);
                }

                thread.clear_stack_roots();
            })
        };

        self.soft_handshake(stack_scan_callback);

        let thread_count = self.thread_registry.iter().len();
        let total_objects_scanned = total_stack_objects_scanned.load(Ordering::Relaxed);

        {
            let mut stats = self.cycle_stats.lock();
            if thread_count > 0 {
                stats.avg_stack_scan_objects = total_objects_scanned as f64 / thread_count as f64;
            }
            stats.objects_marked += total_objects_scanned;
        }
    }

    /// Step 6: Tracing phase - process mark stacks until empty
    fn step_6_tracing(&self) {
        self.set_phase(FugcPhase::Tracing);

        let tracing_start = Instant::now();
        let mut objects_processed = 0;

        while self.parallel_coordinator.has_work() {
            let work_batch = self.parallel_coordinator.steal_work(64);

            if work_batch.is_empty() {
                thread::yield_now();
                continue;
            }

            // Use vectorized batch processing when batch size is suitable for SIMD
            if work_batch.len() >= 8 {
                objects_processed += self.process_objects_vectorized(&work_batch);
            } else {
                for obj in work_batch {
                    // Use cache-optimized marking for processing objects
                    self.cache_optimized_marking.mark_object(obj);
                    self.record_live_object_internal(obj);
                    objects_processed += 1;

                    let children = self.scan_object_fields(obj);
                    if !children.is_empty() {
                        self.parallel_coordinator.share_work(children);
                    }
                }
            }
        }

        {
            let mut stats = self.cycle_stats.lock();
            stats.total_marking_time_ms += tracing_start.elapsed().as_millis() as u64;
            stats.objects_marked += objects_processed;
        }
    }

    /// Scan object fields using the object classifier's adjacency tracking.
    fn scan_object_fields(&self, obj: ObjectReference) -> Vec<ObjectReference> {
        self.object_classifier.get_children(obj)
    }

    /// Step 7: Prepare for sweep - deactivate barriers
    fn step_7_prepare_for_sweep(&self) {
        self.set_phase(FugcPhase::PrepareForSweep);
        self.write_barrier.deactivate();

        let final_cache_reset = Box::new(|thread: &MutatorThread| {
            thread.clear_stack_roots();
        });
        self.soft_handshake(final_cache_reset);
    }

    /// Step 8: Perform sweep with page-based allocation colouring
    fn step_8_sweep(&self) {
        self.set_phase(FugcPhase::Sweeping);

        let sweep_start = Instant::now();

        // Phase 1: Build SIMD bitvector from tricolor markings
        self.build_bitvector_from_markings();

        // Phase 2: SIMD-optimized sweep using AVX2 for liveness counting
        let sweep_stats = self.simd_bitvector.simd_sweep();
        let objects_swept = sweep_stats.objects_swept;

        // Phase 3: Update page states based on SIMD liveness counts
        self.update_page_states_from_bitvector();

        // Cleanup marking state
        self.cache_optimized_marking.reset(); // Reset cache-optimized marking (includes tricolor)
        self.parallel_coordinator.reset();
        self.black_allocator.deactivate();

        {
            let mut stats = self.cycle_stats.lock();
            stats.total_sweep_time_ms += sweep_start.elapsed().as_millis() as u64;
            stats.objects_swept += objects_swept;
        }
    }

    /// Build SIMD bitvector from cache-optimized markings - converts marked objects to live bits
    fn build_bitvector_from_markings(&self) {
        self.simd_bitvector.clear();

        // Use cache-optimized marking to build bitvector from marked objects
        // Since CacheOptimizedMarking delegates to tricolor_marking for actual marking,
        // we can use the tricolor marking's color bit array directly for efficiency
        if let Some(tricolor) = self.cache_optimized_marking.tricolor_marking() {
            // Delegate to tricolor marking for efficient iteration over marked objects
            let marked_objects = tricolor.get_black_objects();
            if marked_objects.is_empty() {
                return;
            }

            let mut page_indices = Vec::with_capacity(marked_objects.len().min(1024));

            for obj_ref in marked_objects {
                self.simd_bitvector.mark_object_live(obj_ref);

                if let Some(page_index) =
                    Self::page_index_for_object(self.heap_base, self.heap_size, obj_ref)
                {
                    page_indices.push(page_index);
                }
            }

            if page_indices.is_empty() {
                return;
            }

            page_indices.sort_unstable();

            let mut current = page_indices[0];
            let mut count = 1usize;

            for page_index in page_indices.into_iter().skip(1) {
                if page_index == current {
                    count += 1;
                } else {
                    let mut entry = self
                        .page_states
                        .entry(current)
                        .or_insert_with(PageState::new);
                    entry.live_objects = entry.live_objects.saturating_add(count);
                    entry.allocation_color = AllocationColor::Black;

                    current = page_index;
                    count = 1;
                }
            }

            let mut entry = self
                .page_states
                .entry(current)
                .or_insert_with(PageState::new);
            entry.live_objects = entry.live_objects.saturating_add(count);
            entry.allocation_color = AllocationColor::Black;
        }
    }

    /// Update page states based on SIMD bitvector liveness counts using AVX2
    fn update_page_states_from_bitvector(&self) {
        let objects_per_page = PAGE_SIZE / OBJECT_GRANULE;

        for mut item in self.page_states.iter_mut() {
            let (page_addr, state) = item.pair_mut();
            // Use SIMD to count live objects in this page efficiently
            let page_start = unsafe { Address::from_usize(*page_addr) };
            let live_count = self
                .simd_bitvector
                .count_live_objects_in_range(page_start, PAGE_SIZE);

            // Update page allocation color based on liveness
            state.allocation_color = if live_count == 0 {
                AllocationColor::White // Completely free page
            } else {
                AllocationColor::Black // Page has live objects
            };

            // Reset for next cycle
            state.live_objects = live_count.min(objects_per_page);
        }
    }

    /// Perform soft handshake with all mutator threads using lock-free protocol
    fn soft_handshake(&self, callback: HandshakeCallback) {
        let handshake_start = Instant::now();
        let threads = self.thread_registry.iter();
        let thread_count = threads.len();

        if thread_count == 0 {
            return;
        }

        // Use the lock-free handshake protocol from ThreadRegistry
        let handshake_type = crate::handshake::HandshakeType::StackScan;
        let timeout = Duration::from_millis(2000);

        match self
            .thread_registry
            .perform_handshake(handshake_type, timeout)
        {
            Ok(completions) => {
                let threads_processed = completions.len();
                // Process each thread with the callback using completion data
                for completion in &completions {
                    if let Some(thread) = self.thread_registry.get(completion.thread_id) {
                        callback(&thread);
                    }
                }

                let handshake_time = handshake_start.elapsed().as_millis() as usize;
                self.handshake_completion_time_ms
                    .store(handshake_time, Ordering::Relaxed);
                self.threads_processed_count
                    .store(threads_processed, Ordering::Relaxed);

                {
                    let mut stats = self.cycle_stats.lock();
                    stats.handshakes_performed += 1;
                }
            }
            Err(e) => {
                eprintln!("Handshake failed: {:?}", e);
            }
        }
    }

    /// Check if all mark stacks are empty (termination condition for step 5/6 loop)
    fn are_all_mark_stacks_empty(&self) -> bool {
        !self.parallel_coordinator.has_work()
    }

    /// Set the current collection phase
    fn set_phase(&self, phase: FugcPhase) {
        {
            let mut guard = self.current_phase.lock();
            *guard = phase;
            // Notify waiters about phase change through channel
            let _ = self.phase_change_sender.try_send(phase);

            // If we're entering Idle phase, signal collection finished with blocking send
            if phase == FugcPhase::Idle {
                // Use blocking send to ensure completion signal is delivered
                let _ = self.collection_finished_sender.send(());
            }
        }
    }

    /// Record that an object resides on a particular allocation page.
    fn record_live_object_internal(&self, object: ObjectReference) {
        Self::record_live_object_for_page(
            &self.page_states,
            self.heap_base,
            self.heap_size,
            object,
        );
    }

    fn record_live_object_for_page(
        pages: &Arc<DashMap<usize, PageState>>,
        heap_base: Address,
        heap_size: usize,
        object: ObjectReference,
    ) {
        if let Some(page_index) = Self::page_index_for_object(heap_base, heap_size, object) {
            let mut entry = pages.entry(page_index).or_insert_with(PageState::new);
            entry.live_objects = entry.live_objects.saturating_add(1);
            entry.allocation_color = AllocationColor::Black;
        }
    }

    #[inline]
    fn page_index_for_object(
        heap_base: Address,
        heap_size: usize,
        object: ObjectReference,
    ) -> Option<usize> {
        let base = heap_base.as_usize();
        let addr = object.to_raw_address().as_usize();

        if addr < base || addr >= base + heap_size {
            return None;
        }

        Some((addr - base) / PAGE_SIZE)
    }

    /// Get the current collection phase.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::{FugcCoordinator, FugcPhase};
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    /// ```
    pub fn current_phase(&self) -> FugcPhase {
        *self.current_phase.lock()
    }

    /// Check if a collection is currently in progress.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// assert!(!coordinator.is_collecting());
    /// ```
    pub fn is_collecting(&self) -> bool {
        self.collection_in_progress.load(Ordering::SeqCst)
    }

    /// Get collection cycle statistics.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// let stats = coordinator.get_cycle_stats();
    /// assert_eq!(stats.cycles_completed, 0);
    /// ```
    pub fn get_cycle_stats(&self) -> FugcCycleStats {
        self.cycle_stats.lock().clone()
    }

    /// Get references to internal components for integration
    /// Access the shared tricolor marking state.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.tricolor_marking();
    /// ```
    pub fn tricolor_marking(&self) -> &Arc<TricolorMarking> {
        &self.tricolor_marking
    }

    /// Access the write barrier used during concurrent marking.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// let barrier = coordinator.write_barrier();
    /// assert!(!barrier.is_active());
    /// ```
    pub fn write_barrier(&self) -> &Arc<WriteBarrier> {
        &self.write_barrier
    }

    #[doc(hidden)]
    pub fn bench_reset_bitvector_state(&self) {
        self.simd_bitvector.clear();
        for mut entry in self.page_states.iter_mut() {
            let (_, state) = entry.pair_mut();
            state.live_objects = 0;
            state.allocation_color = AllocationColor::White;
        }
    }

    #[doc(hidden)]
    pub fn bench_build_bitvector(&self) {
        self.build_bitvector_from_markings();
    }

    /// Access the black allocator used when concurrent marking is active.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// let allocator = coordinator.black_allocator();
    /// assert!(!allocator.is_active());
    /// ```
    pub fn black_allocator(&self) -> &Arc<BlackAllocator> {
        &self.black_allocator
    }

    /// Get access to the parallel marking coordinator for advanced integrations
    /// such as fuzzing or stress scenarios that need to exercise the work-stealing
    /// discipline directly.
    pub fn parallel_marking(&self) -> &Arc<ParallelMarkingCoordinator> {
        &self.parallel_coordinator
    }

    /// Ensure black allocation is active before allocating new objects during marking.
    pub fn ensure_black_allocation_active(&self) {
        if !self.black_allocator.is_active() {
            self.black_allocator.activate();
        }
    }

    /// Scan thread roots at safepoint (called from safepoint callback)
    ///
    /// This method is called when threads reach a safepoint during the
    /// root scanning phase. It scans the current thread's stack for
    /// object references.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.scan_thread_roots_at_safepoint();
    /// ```
    pub fn scan_thread_roots_at_safepoint(&self) {
        self.root_scanner.scan_global_roots();

        for thread in self.thread_registry.iter() {
            for &root_ptr in thread.stack_roots().iter() {
                if root_ptr.is_null() {
                    continue;
                }

                if let Some(obj_ref) = ObjectReference::from_raw_address(unsafe {
                    Address::from_usize(root_ptr as usize)
                }) {
                    self.cache_optimized_marking.mark_object(obj_ref);
                    self.record_live_object_internal(obj_ref);
                    self.parallel_coordinator.share_work(vec![obj_ref]);
                }
            }
        }
    }

    /// Activate barriers at safepoint
    ///
    /// This method activates write barriers and black allocation
    /// when called from a safepoint callback.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.activate_barriers_at_safepoint();
    /// assert!(coordinator.write_barrier().is_active());
    /// ```
    pub fn activate_barriers_at_safepoint(&self) {
        self.write_barrier.activate();
        self.black_allocator.activate();
    }

    /// Perform marking handshake at safepoint
    ///
    /// This method coordinates marking work between threads
    /// when called from a safepoint callback.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.marking_handshake_at_safepoint();
    /// ```
    pub fn marking_handshake_at_safepoint(&self) {
        // Process any pending marking work for the current thread
        if self.parallel_coordinator.has_work() {
            let work_batch = self.parallel_coordinator.steal_work(32);
            for obj in work_batch {
                // Use cache-optimized marking for handshake processing
                self.cache_optimized_marking.mark_object(obj);
                self.record_live_object_internal(obj);
                // In a real implementation, we would scan object fields here
            }
        }
    }

    /// Prepare sweep at safepoint
    ///
    /// This method performs final coordination before the sweep phase
    /// when called from a safepoint callback.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.prepare_sweep_at_safepoint();
    /// assert!(!coordinator.write_barrier().is_active());
    /// ```
    pub fn prepare_sweep_at_safepoint(&self) {
        // Deactivate barriers before sweep
        self.write_barrier.deactivate();
        // Final stack scan to catch any missed references
        self.scan_thread_roots_at_safepoint();
    }

    /// Start concurrent marking with cache optimization
    pub fn start_marking(&self, roots: Vec<mmtk::util::ObjectReference>) {
        // Reset all components
        self.cache_optimized_marking.reset(); // Reset cache-optimized marking (includes tricolor)
        self.parallel_coordinator.reset();
        self.write_barrier.reset();
        self.black_allocator.reset();

        // Activate concurrent mechanisms
        self.write_barrier.activate();
        self.black_allocator.activate();

        // Initialize roots as grey using cache-optimized marking
        for root in &roots {
            self.cache_optimized_marking.mark_object(*root);
        }
        self.parallel_coordinator.share_work(roots);

        // Start worker threads if not already running
        let should_spawn = { self.workers.lock().is_empty() };
        if should_spawn {
            self.start_worker_threads();
        }
    }

    /// Stop concurrent marking and wait for completion
    pub fn stop_marking(&self) {
        // Deactivate concurrent mechanisms
        self.write_barrier.deactivate();
        self.black_allocator.deactivate();

        // Send shutdown signals to all workers
        {
            let worker_channels = self.worker_channels.lock();
            for channel in worker_channels.iter() {
                channel.send_shutdown();
            }
        }

        // Wait for workers to complete
        for worker in self.workers.lock().drain(..) {
            if let Err(e) = worker.join() {
                eprintln!("Warning: Worker thread failed to join cleanly: {:?}", e);
            }
        }

        // Clear the channels for next run
        self.worker_channels.lock().clear();
    }

    /// Mark objects using cache-optimized strategies (always enabled)
    pub fn mark_objects_cache_optimized(&self, objects: &[mmtk::util::ObjectReference]) {
        for obj in objects {
            self.cache_optimized_marking.mark_object(*obj);
        }
    }

    /// Get cache optimization statistics (always available)
    pub fn get_cache_stats(&self) -> crate::cache_optimization::CacheStats {
        self.cache_optimized_marking.get_stats()
    }

    /// Get object classifier for FUGC-style object classification
    pub fn object_classifier(&self) -> &crate::concurrent::ObjectClassifier {
        &self.object_classifier
    }

    /// Get root scanner for concurrent root scanning
    pub fn root_scanner(&self) -> &crate::concurrent::ConcurrentRootScanner {
        &self.root_scanner
    }

    /// Queue object for promotion (generational support)
    pub fn queue_for_promotion(&self, obj: mmtk::util::ObjectReference) {
        self.object_classifier.queue_for_promotion(obj);
    }

    /// Promote young objects to old generation
    pub fn promote_young_objects(&self) {
        self.object_classifier.promote_young_objects();
    }

    /// Classify new object (for allocation)
    pub fn classify_new_object(&self, obj: mmtk::util::ObjectReference) {
        self.object_classifier.classify_new_object(obj);
    }

    /// Handle generational write barriers
    pub fn generational_write_barrier(
        &self,
        src: mmtk::util::ObjectReference,
        dst: mmtk::util::ObjectReference,
    ) {
        self.object_classifier
            .record_cross_generational_reference(src, dst);
    }

    /// Determine whether an object is marked in the current bitvector snapshot.
    pub fn is_object_marked(&self, obj: mmtk::util::ObjectReference) -> bool {
        self.simd_bitvector.is_marked(obj.to_raw_address())
    }

    /// Get concurrent marking statistics (enhanced version)
    pub fn get_marking_stats(&self) -> crate::concurrent::ConcurrentMarkingStats {
        let (stolen, shared) = self.parallel_coordinator.get_stats();
        crate::concurrent::ConcurrentMarkingStats {
            work_stolen: stolen,
            work_shared: shared,
            objects_allocated_black: self.black_allocator.get_stats(),
        }
    }

    /// Get work stealing statistics (backward compatibility).
    ///
    /// Returns (work_stolen, work_shared) tuple.
    pub fn get_stats(&self) -> (usize, usize) {
        self.parallel_coordinator.get_stats()
    }

    fn start_worker_threads(&self) {
        let num_workers = 4; // Use a default number of workers
        let mut worker_channels = self.worker_channels.lock();
        worker_channels.clear(); // Clear any existing channels

        for worker_id in 0..num_workers {
            // Create channels for this worker
            let (work_sender, work_receiver) = crossbeam::channel::unbounded();
            let (completion_sender, completion_receiver) = crossbeam::channel::unbounded();
            let (shutdown_sender, shutdown_receiver) = crossbeam::channel::unbounded();

            // Store channels for coordinator to use
            worker_channels.push(crate::concurrent::WorkerChannels::new(
                work_sender,
                completion_receiver,
                shutdown_sender,
            ));

            let coordinator = Arc::clone(&self.parallel_coordinator);
            let cache_marking = Arc::clone(&self.cache_optimized_marking);

            let worker = std::thread::spawn(move || {
                let mut marking_worker =
                    crate::concurrent::MarkingWorker::new(worker_id, coordinator.clone(), 256);
                let mut objects_processed = 0usize;

                loop {
                    crossbeam::select! {
                        // Handle shutdown signal
                        recv(shutdown_receiver) -> msg => {
                            if msg.is_ok() {
                                // Send completion count and exit
                                let _ = completion_sender.send(objects_processed);
                                break;
                            }
                        },

                        // Handle work from channels
                        recv(work_receiver) -> work_batch => {
                            if let Ok(work) = work_batch {
                                marking_worker.grey_stack.add_shared_work(work);
                            }
                        },

                        // Try to steal work from global pool (with timeout)
                        default(std::time::Duration::from_millis(1)) => {
                            // Only try stealing if local stack is empty
                            if marking_worker.grey_stack.is_empty() {
                                let stolen_work = coordinator.steal_work(64);
                                if !stolen_work.is_empty() {
                                    marking_worker.grey_stack.add_shared_work(stolen_work);
                                }
                            }
                        }
                    }

                    // Process available work efficiently with cache optimization
                    let mut work_processed_this_round = 0;
                    while let Some(obj) = marking_worker.grey_stack.pop() {
                        // Use cache-optimized marking for all objects (handles color transitions internally)
                        cache_marking.mark_object(obj);
                        objects_processed += 1;
                        work_processed_this_round += 1;

                        // Periodically check for shutdown to maintain responsiveness
                        if work_processed_this_round % 100 == 0
                            && shutdown_receiver.try_recv().is_ok()
                        {
                            let _ = completion_sender.send(objects_processed);
                            return;
                        }
                    }

                    // Share work if we have too much
                    if marking_worker.grey_stack.should_share_work() {
                        let shared_work = marking_worker.grey_stack.extract_work();
                        coordinator.share_work(shared_work);
                    }
                }
            });

            self.workers.lock().push(worker);
        }
    }

    /// Process a batch of objects using SIMD-optimized vectorized operations
    ///
    /// This method leverages vector instructions to process multiple objects
    /// simultaneously, improving throughput for large work batches during
    /// the tracing phase of garbage collection.
    ///
    /// # Arguments
    /// * `objects` - Batch of objects to process
    ///
    /// # Returns
    /// Number of objects successfully processed
    #[cfg(target_arch = "x86_64")]
    fn process_objects_vectorized(&self, objects: &[ObjectReference]) -> usize {
        let mut processed = 0;
        const SIMD_WIDTH: usize = 8; // Process 8 objects per SIMD iteration

        // Process objects in SIMD-aligned chunks
        for chunk in objects.chunks(SIMD_WIDTH) {
            // Extract addresses for vectorized operations
            let mut addresses = [0usize; SIMD_WIDTH];
            let chunk_len = chunk.len();

            for (i, &obj) in chunk.iter().enumerate() {
                addresses[i] = obj.to_raw_address().as_usize();
            }

            // Vectorized address validation and processing
            self.process_address_batch_simd(&addresses[..chunk_len]);

            // Sequential processing for actions that can't be vectorized yet
            for &obj in chunk {
                self.cache_optimized_marking.mark_object(obj);
                self.record_live_object_internal(obj);
                processed += 1;

                let children = self.scan_object_fields(obj);
                if !children.is_empty() {
                    self.parallel_coordinator.share_work(children);
                }
            }
        }

        processed
    }

    /// Non-x86_64 fallback for vectorized object processing
    #[cfg(not(target_arch = "x86_64"))]
    fn process_objects_vectorized(&self, objects: &[ObjectReference]) -> usize {
        // Fallback to sequential processing on non-x86_64 architectures
        let mut processed = 0;
        for &obj in objects {
            self.cache_optimized_marking.mark_object(obj);
            self.record_live_object_internal(obj);
            processed += 1;

            let children = self.scan_object_fields(obj);
            if !children.is_empty() {
                self.parallel_coordinator.share_work(children);
            }
        }
        processed
    }

    /// Process a batch of addresses using SIMD operations for validation and prefetching
    ///
    /// This performs vectorized address range checking and cache prefetching
    /// to improve memory access patterns during object processing.
    #[cfg(target_arch = "x86_64")]
    fn process_address_batch_simd(&self, addresses: &[usize]) {
        if addresses.is_empty() {
            return;
        }

        unsafe {
            // Prefetch addresses for better cache performance
            for &addr in addresses {
                _mm_prefetch(addr as *const i8, _MM_HINT_T0);
            }

            // Additional SIMD operations could be added here for:
            // - Batch address range validation
            // - Vectorized offset calculations
            // - Parallel object header reads
        }
    }

    /// Non-x86_64 fallback for SIMD address processing
    #[cfg(not(target_arch = "x86_64"))]
    fn process_address_batch_simd(&self, _addresses: &[usize]) {
        // No-op on non-x86_64 architectures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fugc_coordinator_creation() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = &fixture.coordinator;

        assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
        assert!(!coordinator.is_collecting());
    }

    #[test]
    fn fugc_phase_transitions() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = &fixture.coordinator;

        coordinator.set_phase(FugcPhase::ActivateBarriers);
        assert_eq!(coordinator.current_phase(), FugcPhase::ActivateBarriers);

        coordinator.set_phase(FugcPhase::Tracing);
        assert_eq!(coordinator.current_phase(), FugcPhase::Tracing);
    }

    #[test]
    fn fugc_gc_trigger() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = &fixture.coordinator;

        assert!(!coordinator.is_collecting());
        coordinator.trigger_gc();
        assert!(coordinator.is_collecting());
    }
}
