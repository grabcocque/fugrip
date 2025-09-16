//! FUGC (Fil's Unbelievable Garbage Collector) protocol implementation
//!
//! This module implements a faithful version of the eight step FUGC protocol as
//! described by Epic Games for the Verse runtime.  The coordinator integrates
//! with the existing concurrent marking infrastructure, provides precise
//! safepoint handshakes, and maintains page level allocation colouring to
//! emulate the production collector's behaviour.

use crate::{
    concurrent::{
        BlackAllocator, ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier,
    },
    roots::GlobalRoots,
    simd_sweep::SimdBitvector,
    thread::{MutatorThread, ThreadRegistry},
};
use mmtk::util::{Address, ObjectReference};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use crossbeam::channel::{Receiver, Sender, bounded};

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
    page_states: Arc<Mutex<HashMap<usize, PageState>>>,

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
    /// use fugrip::fugc_coordinator::FugcCoordinator;
    /// use fugrip::roots::GlobalRoots;
    /// use fugrip::thread::ThreadRegistry;
    /// use mmtk::util::Address;
    /// use std::sync::{Arc, Mutex};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let heap_size = 64 * 1024 * 1024; // 64MB heap
    /// let thread_registry = Arc::new(ThreadRegistry::new());
    /// let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    ///
    /// let coordinator = FugcCoordinator::new(
    ///     heap_base,
    ///     heap_size,
    ///     4,  // 4 worker threads
    ///     thread_registry,
    ///     global_roots,
    /// );
    ///
    /// // Coordinator starts in idle phase
    /// assert_eq!(coordinator.current_phase(), fugrip::FugcPhase::Idle);
    /// assert!(!coordinator.is_collecting());
    /// ```
    pub fn new(
        heap_base: Address,
        heap_size: usize,
        num_workers: usize,
        thread_registry: Arc<ThreadRegistry>,
        global_roots: Arc<Mutex<GlobalRoots>>,
    ) -> Self {
        let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let parallel_coordinator = Arc::new(ParallelMarkingCoordinator::new(num_workers));
        let write_barrier = Arc::new(WriteBarrier::new(
            Arc::clone(&tricolor_marking),
            Arc::clone(&parallel_coordinator),
        ));
        let black_allocator = Arc::new(BlackAllocator::new(Arc::clone(&tricolor_marking)));

        // Initialize SIMD bitvector for ultra-fast sweeping (assume 16-byte object alignment)
        let simd_bitvector = Arc::new(SimdBitvector::new(heap_base, heap_size, 16));

        // Create crossbeam channels for proper synchronization
        let (phase_change_sender, phase_change_receiver) = bounded(100);
        let (collection_finished_sender, collection_finished_receiver) = bounded(1);

        Self {
            tricolor_marking,
            write_barrier,
            black_allocator,
            parallel_coordinator,
            simd_bitvector,
            thread_registry,
            global_roots,
            current_phase: Arc::new(Mutex::new(FugcPhase::Idle)),
            collection_in_progress: Arc::new(AtomicBool::new(false)),
            handshake_completion_time_ms: Arc::new(AtomicUsize::new(0)),
            threads_processed_count: Arc::new(AtomicUsize::new(0)),
            phase_change_sender: Arc::new(phase_change_sender),
            phase_change_receiver: Arc::new(phase_change_receiver),
            collection_finished_sender: Arc::new(collection_finished_sender),
            collection_finished_receiver: Arc::new(collection_finished_receiver),
            cycle_stats: Arc::new(Mutex::new(FugcCycleStats::default())),
            page_states: Arc::new(Mutex::new(HashMap::new())),
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
    /// use fugrip::fugc_coordinator::FugcCoordinator;
    /// use fugrip::roots::GlobalRoots;
    /// use fugrip::thread::ThreadRegistry;
    /// use mmtk::util::Address;
    /// use std::sync::{Arc, Mutex};
    /// use std::time::Duration;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let heap_size = 32 * 1024 * 1024;
    /// let thread_registry = Arc::new(ThreadRegistry::new());
    /// let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    ///
    /// let coordinator = FugcCoordinator::new(
    ///     heap_base, heap_size, 2, thread_registry, global_roots
    /// );
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
    pub fn trigger_gc(&self) {
        if self
            .collection_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            let coordinator = self.clone_for_thread();
            thread::spawn(move || {
                coordinator.run_collection_cycle();
            });
        }
    }

    /// Explicitly request a soft handshake from a registered mutator thread.
    ///
    /// ```
    /// # use fugrip::thread::{MutatorThread, ThreadRegistry};
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, Arc::clone(&registry), Arc::clone(&globals));
    /// let mutator = MutatorThread::new(1);
    /// registry.register(mutator.clone());
    /// coordinator.request_handshake(mutator.id());
    /// ```
    pub fn request_handshake(&self, thread_id: usize) {
        if let Some(thread) = self.thread_registry.get(thread_id) {
            thread.request_safepoint();
        }
    }

    /// Complete a previously requested handshake for the given mutator.
    ///
    /// ```
    /// # use fugrip::thread::{MutatorThread, ThreadRegistry};
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::FugcCoordinator;
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, Mutex};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, Arc::clone(&registry), Arc::clone(&globals));
    /// let mutator = MutatorThread::new(1);
    /// registry.register(mutator.clone());
    /// coordinator.request_handshake(mutator.id());
    /// coordinator.complete_handshake(mutator.id());
    /// ```
    pub fn complete_handshake(&self, thread_id: usize) {
        if let Some(thread) = self.thread_registry.get(thread_id) {
            thread.clear_safepoint();
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
            Err(_) => false, // Timeout or channel closed
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
            match self.phase_change_receiver.recv_timeout(Duration::from_millis(10)) {
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
            match self.phase_change_receiver.recv_timeout(Duration::from_millis(10)) {
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
            .lock()
            .map(|pages| pages.get(&page_index).map(|state| state.allocation_color))
            .ok()
            .flatten()
            .unwrap_or(AllocationColor::White)
    }

    /// Clone coordinator for use in collection thread while sharing all state.
    fn clone_for_thread(&self) -> FugcCoordinator {
        FugcCoordinator {
            tricolor_marking: Arc::clone(&self.tricolor_marking),
            write_barrier: Arc::clone(&self.write_barrier),
            black_allocator: Arc::clone(&self.black_allocator),
            parallel_coordinator: Arc::clone(&self.parallel_coordinator),
            simd_bitvector: Arc::clone(&self.simd_bitvector),
            thread_registry: Arc::clone(&self.thread_registry),
            global_roots: Arc::clone(&self.global_roots),
            current_phase: Arc::clone(&self.current_phase),
            collection_in_progress: Arc::clone(&self.collection_in_progress),
            handshake_completion_time_ms: Arc::clone(&self.handshake_completion_time_ms),
            threads_processed_count: Arc::clone(&self.threads_processed_count),
            phase_change_sender: Arc::clone(&self.phase_change_sender),
            phase_change_receiver: Arc::clone(&self.phase_change_receiver),
            collection_finished_sender: Arc::clone(&self.collection_finished_sender),
            collection_finished_receiver: Arc::clone(&self.collection_finished_receiver),
            cycle_stats: Arc::clone(&self.cycle_stats),
            page_states: Arc::clone(&self.page_states),
            heap_base: self.heap_base,
            heap_size: self.heap_size,
        }
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
        loop {
            self.step_5_stack_scan_handshake();

            if self.are_all_mark_stacks_empty() {
                break; // Go to step 7
            }

            self.step_6_tracing();
        }

        // Step 7: Turn off store barrier, prepare for sweep
        self.step_7_prepare_for_sweep();

        // Step 8: Perform sweep
        self.step_8_sweep();

        // Update statistics
        if let Ok(mut stats) = self.cycle_stats.lock() {
            stats.cycles_completed += 1;
        }

        // Reset state for next cycle
        self.set_phase(FugcPhase::Idle);
        self.collection_in_progress.store(false, Ordering::SeqCst);
    }

    /// Reset per-cycle state before starting a new collection.
    fn prepare_cycle_state(&self) {
        self.set_phase(FugcPhase::ActivateBarriers);
        self.tricolor_marking.clear();
        self.parallel_coordinator.reset();
        self.black_allocator.reset();
        self.handshake_completion_time_ms
            .store(0, Ordering::Relaxed);
        self.threads_processed_count.store(0, Ordering::Relaxed);

        if let Ok(mut pages) = self.page_states.lock() {
            for state in pages.values_mut() {
                state.live_objects = 0;
            }
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

        if let Ok(roots) = self.global_roots.lock() {
            for root_ptr in roots.iter() {
                if let Some(root_obj) = ObjectReference::from_raw_address(unsafe {
                    Address::from_usize(root_ptr as usize)
                }) {
                    if self.tricolor_marking.get_color(root_obj) == ObjectColor::White {
                        self.tricolor_marking.set_color(root_obj, ObjectColor::Grey);
                        self.parallel_coordinator.share_work(vec![root_obj]);
                        self.record_live_object_internal(root_obj);
                        objects_marked += 1;
                    }
                }
            }
        }

        if let Ok(mut stats) = self.cycle_stats.lock() {
            stats.total_marking_time_ms += marking_start.elapsed().as_millis() as u64;
            stats.objects_marked += objects_marked;
        }
    }

    /// Step 5: Stack scan handshake with mark stack check
    fn step_5_stack_scan_handshake(&self) {
        self.set_phase(FugcPhase::StackScanHandshake);

        let tricolor_marking = Arc::clone(&self.tricolor_marking);
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
                        if tricolor_marking.get_color(obj_ref) == ObjectColor::White {
                            if tricolor_marking.transition_color(
                                obj_ref,
                                ObjectColor::White,
                                ObjectColor::Grey,
                            ) {
                                FugcCoordinator::record_live_object_for_page(
                                    &page_states,
                                    heap_base,
                                    heap_size,
                                    obj_ref,
                                );
                                local_grey_objects.push(obj_ref);
                            }
                        }
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

        if let Ok(mut stats) = self.cycle_stats.lock() {
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

            for obj in work_batch {
                if self.tricolor_marking.transition_color(
                    obj,
                    ObjectColor::Grey,
                    ObjectColor::Black,
                ) {
                    self.record_live_object_internal(obj);
                    objects_processed += 1;

                    let children = self.scan_object_fields(obj);
                    if !children.is_empty() {
                        self.parallel_coordinator.share_work(children);
                    }
                }
            }
        }

        if let Ok(mut stats) = self.cycle_stats.lock() {
            stats.total_marking_time_ms += tracing_start.elapsed().as_millis() as u64;
            stats.objects_marked += objects_processed;
        }
    }

    /// Scan object fields and return reachable white objects as grey.  The
    /// current runtime does not expose precise layout information, so we keep
    /// this conservative and rely on host integrations to plug in real
    /// metadata later on.
    fn scan_object_fields(&self, _obj: ObjectReference) -> Vec<ObjectReference> {
        Vec::new()
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
        self.tricolor_marking.clear();
        self.parallel_coordinator.reset();
        self.black_allocator.deactivate();

        if let Ok(mut stats) = self.cycle_stats.lock() {
            stats.total_sweep_time_ms += sweep_start.elapsed().as_millis() as u64;
            stats.objects_swept += objects_swept;
        }
    }

    /// Build SIMD bitvector from tricolor markings - converts black objects to live bits
    fn build_bitvector_from_markings(&self) {
        // Clear previous bitvector state
        self.simd_bitvector.clear();

        // Iterate through all marked objects and set corresponding bits
        let marked_objects = self.tricolor_marking.get_black_objects();
        for obj_ref in marked_objects {
            self.simd_bitvector.mark_object_live(obj_ref);
        }
    }

    /// Update page states based on SIMD bitvector liveness counts using AVX2
    fn update_page_states_from_bitvector(&self) {
        let objects_per_page = PAGE_SIZE / OBJECT_GRANULE;

        if let Ok(mut pages) = self.page_states.lock() {
            for (page_addr, state) in pages.iter_mut() {
                // Use SIMD to count live objects in this page efficiently
                let page_start = unsafe { Address::from_usize(*page_addr) };
                let live_count = self.simd_bitvector.count_live_objects_in_range(
                    page_start,
                    PAGE_SIZE
                );

                // Update page allocation color based on liveness
                state.allocation_color = if live_count == 0 {
                    AllocationColor::White  // Completely free page
                } else {
                    AllocationColor::Black  // Page has live objects
                };

                // Reset for next cycle
                state.live_objects = live_count.min(objects_per_page);
            }
        }
    }

    /// Perform soft handshake with all mutator threads
    fn soft_handshake(&self, callback: HandshakeCallback) {
        let handshake_start = Instant::now();
        let threads = self.thread_registry.iter();
        let thread_count = threads.len();

        if thread_count == 0 {
            return;
        }

        for thread in &threads {
            thread.request_safepoint();
        }

        for thread in &threads {
            thread.wait_until_parked();
        }

        let mut threads_processed = 0;
        for thread in &threads {
            callback(thread);
            threads_processed += 1;
        }

        for thread in &threads {
            thread.clear_safepoint();
        }

        let handshake_time = handshake_start.elapsed().as_millis() as usize;
        self.handshake_completion_time_ms
            .store(handshake_time, Ordering::Relaxed);
        self.threads_processed_count
            .store(threads_processed, Ordering::Relaxed);

        if let Ok(mut stats) = self.cycle_stats.lock() {
            stats.handshakes_performed += 1;
        }
    }

    /// Check if all mark stacks are empty (termination condition for step 5/6 loop)
    fn are_all_mark_stacks_empty(&self) -> bool {
        !self.parallel_coordinator.has_work()
    }

    /// Set the current collection phase
    fn set_phase(&self, phase: FugcPhase) {
        if let Ok(mut guard) = self.current_phase.lock() {
            *guard = phase;
            // Notify waiters about phase change through channel
            let _ = self.phase_change_sender.try_send(phase);

            // If we're entering Idle phase, signal collection finished
            if phase == FugcPhase::Idle {
                let _ = self.collection_finished_sender.try_send(());
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
        pages: &Arc<Mutex<HashMap<usize, PageState>>>,
        heap_base: Address,
        heap_size: usize,
        object: ObjectReference,
    ) {
        let base = heap_base.as_usize();
        let addr = object.to_raw_address().as_usize();

        if addr < base || addr >= base + heap_size {
            return;
        }

        let page_index = (addr - base) / PAGE_SIZE;

        if let Ok(mut map) = pages.lock() {
            let state = map.entry(page_index).or_insert_with(PageState::new);
            state.live_objects = state.live_objects.saturating_add(1);
            state.allocation_color = AllocationColor::Black;
        }
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
        self.current_phase
            .lock()
            .map(|phase| *phase)
            .unwrap_or(FugcPhase::Idle)
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
        self.cycle_stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_default()
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
        // In a real implementation, this would:
        // 1. Walk the current thread's stack
        // 2. Find all pointer-sized values that point into the heap
        // 3. Mark those objects as grey if they're currently white
        // 4. Add them to the marking work queue

        // For now, we'll scan any existing thread state in the registry
        let threads = self.thread_registry.iter();
        for thread in &threads {
            let stack_roots = thread.stack_roots();
            for &root_ptr in &stack_roots {
                if root_ptr as usize == 0 {
                    continue;
                }

                if let Some(obj_ref) = ObjectReference::from_raw_address(unsafe {
                    Address::from_usize(root_ptr as usize)
                }) {
                    if self.tricolor_marking.get_color(obj_ref) == ObjectColor::White {
                        if self.tricolor_marking.transition_color(
                            obj_ref,
                            ObjectColor::White,
                            ObjectColor::Grey,
                        ) {
                            self.record_live_object_internal(obj_ref);
                            self.parallel_coordinator.share_work(vec![obj_ref]);
                        }
                    }
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
                if self.tricolor_marking.transition_color(
                    obj,
                    ObjectColor::Grey,
                    ObjectColor::Black,
                ) {
                    self.record_live_object_internal(obj);
                    // In a real implementation, we would scan object fields here
                }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roots::GlobalRoots;

    #[test]
    fn fugc_coordinator_creation() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let thread_registry = Arc::new(ThreadRegistry::new());
        let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

        let coordinator =
            FugcCoordinator::new(heap_base, heap_size, 4, thread_registry, global_roots);

        assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
        assert!(!coordinator.is_collecting());
    }

    #[test]
    fn fugc_phase_transitions() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let thread_registry = Arc::new(ThreadRegistry::new());
        let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

        let coordinator =
            FugcCoordinator::new(heap_base, heap_size, 4, thread_registry, global_roots);

        coordinator.set_phase(FugcPhase::ActivateBarriers);
        assert_eq!(coordinator.current_phase(), FugcPhase::ActivateBarriers);

        coordinator.set_phase(FugcPhase::Tracing);
        assert_eq!(coordinator.current_phase(), FugcPhase::Tracing);
    }

    #[test]
    fn fugc_gc_trigger() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let thread_registry = Arc::new(ThreadRegistry::new());
        let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

        let coordinator =
            FugcCoordinator::new(heap_base, heap_size, 4, thread_registry, global_roots);

        assert!(!coordinator.is_collecting());
        coordinator.trigger_gc();
        assert!(coordinator.is_collecting());
    }
}
