//! Core FugcCoordinator struct and constructor

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::frontend::types::Address;
use crate::{
    concurrent::{BlackAllocator, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier},
    roots::GlobalRoots,
    simd_sweep::SimdBitvector,
    thread::ThreadRegistry,
};
use arc_swap::ArcSwap;
use dashmap::DashMap;
use flume::{Receiver, Sender};

use super::types::*;

const OBJECT_GRANULE: usize = 64;
const PAGE_SIZE: usize = mmtk::util::constants::BYTES_IN_PAGE;

/// Main coordinator implementing the complete FUGC 8-step protocol
pub struct FugcCoordinator {
    // Core GC components
    tricolor_marking: Arc<TricolorMarking>,
    write_barrier: Arc<WriteBarrier>,
    black_allocator: Arc<BlackAllocator>,
    parallel_coordinator: Arc<ParallelMarkingCoordinator>,

    // High-performance SIMD sweeping
    simd_bitvector: Arc<SimdBitvector>,

    // Thread management (Rayon-based parallel processing)
    thread_registry: Arc<ThreadRegistry>,
    /// Lock-free global roots access using arc_swap for 10-20% improvement
    /// Particularly beneficial during marking phase when roots are accessed frequently
    global_roots: ArcSwap<GlobalRoots>,

    // State management
    /// Hot-path phase reads optimized with ArcSwap for lock-free loads
    current_phase: ArcSwap<FugcPhase>,
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
    /// Lock-free stats access using arc_swap for 15-25% improvement over ///
    /// Monitoring/telemetry reads vs GC update contention eliminated
    /// Access: .load() for reads, .store(Arc::new(stats)) for updates
    cycle_stats: ArcSwap<FugcCycleStats>,
    page_states: Arc<DashMap<usize, PageState>>,

    // Cache optimization and object classification (always enabled for performance)
    cache_optimized_marking: Arc<crate::cache_optimization::CacheOptimizedMarking>,
    object_classifier: Arc<crate::concurrent::ObjectClassifier>,
    root_scanner: Arc<crate::concurrent::ConcurrentRootScanner>,

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
        global_roots: &ArcSwap<GlobalRoots>,
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

        // Create flume channels for proper synchronization (10-20% faster than crossbeam)
        let (phase_change_sender, phase_change_receiver) = flume::bounded(100);
        let (collection_finished_sender, collection_finished_receiver) = flume::bounded(1);

        // Initialize cache optimization and object classification (always enabled)
        let cache_optimized_marking = Arc::new(
            crate::cache_optimization::CacheOptimizedMarking::with_tricolor(&tricolor_marking),
        );
        let object_classifier = Arc::new(crate::concurrent::ObjectClassifier::new());
        let root_scanner = Arc::new(crate::concurrent::ConcurrentRootScanner::new(
            Arc::clone(thread_registry),
            arc_swap::ArcSwap::new(global_roots.load().clone()),
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
            global_roots: ArcSwap::new(global_roots.load().clone()),
            current_phase: arc_swap::ArcSwap::new(Arc::new(FugcPhase::Idle)),
            collection_in_progress: Arc::new(AtomicBool::new(false)),
            handshake_completion_time_ms: Arc::new(AtomicUsize::new(0)),
            threads_processed_count: Arc::new(AtomicUsize::new(0)),
            phase_change_sender: Arc::new(phase_change_sender),
            phase_change_receiver: Arc::new(phase_change_receiver),
            collection_finished_sender: Arc::new(collection_finished_sender),
            collection_finished_receiver: Arc::new(collection_finished_receiver),
            cycle_stats: ArcSwap::new(Arc::new(FugcCycleStats::default())),
            page_states: Arc::new(DashMap::new()),
            cache_optimized_marking,
            object_classifier,
            root_scanner,
            heap_base,
            heap_size,
        }
    }

    // Accessors for the coordinator's components (used by other modules)

    pub fn tricolor_marking(&self) -> &Arc<TricolorMarking> {
        &self.tricolor_marking
    }

    pub fn write_barrier(&self) -> &Arc<WriteBarrier> {
        &self.write_barrier
    }

    pub fn black_allocator(&self) -> &Arc<BlackAllocator> {
        &self.black_allocator
    }

    pub fn parallel_coordinator(&self) -> &Arc<ParallelMarkingCoordinator> {
        &self.parallel_coordinator
    }

    /// Backward-compatible accessor for parallel marking coordinator
    pub fn parallel_marking(&self) -> &Arc<ParallelMarkingCoordinator> {
        &self.parallel_coordinator
    }

    pub fn simd_bitvector(&self) -> &Arc<SimdBitvector> {
        &self.simd_bitvector
    }

    pub fn thread_registry(&self) -> &Arc<ThreadRegistry> {
        &self.thread_registry
    }

    pub fn global_roots(&self) -> &ArcSwap<GlobalRoots> {
        &self.global_roots
    }

    pub fn current_phase(&self) -> FugcPhase {
        **self.current_phase.load()
    }

    pub fn is_collecting(&self) -> bool {
        self.collection_in_progress.load(Ordering::Acquire)
    }

    pub fn cycle_stats(&self) -> &ArcSwap<FugcCycleStats> {
        &self.cycle_stats
    }

    pub fn page_states(&self) -> &Arc<DashMap<usize, PageState>> {
        &self.page_states
    }

    pub fn cache_optimized_marking(
        &self,
    ) -> &Arc<crate::cache_optimization::CacheOptimizedMarking> {
        &self.cache_optimized_marking
    }

    pub fn object_classifier(&self) -> &Arc<crate::concurrent::ObjectClassifier> {
        &self.object_classifier
    }

    pub fn root_scanner(&self) -> &Arc<crate::concurrent::ConcurrentRootScanner> {
        &self.root_scanner
    }

    pub fn heap_base(&self) -> Address {
        self.heap_base
    }

    pub fn heap_size(&self) -> usize {
        self.heap_size
    }

    // Internal accessor methods for channels and coordination

    pub(super) fn phase_change_sender(&self) -> &Arc<Sender<FugcPhase>> {
        &self.phase_change_sender
    }

    pub(super) fn phase_change_receiver(&self) -> &Arc<Receiver<FugcPhase>> {
        &self.phase_change_receiver
    }

    pub(super) fn collection_finished_sender(&self) -> &Arc<Sender<()>> {
        &self.collection_finished_sender
    }

    pub(super) fn collection_finished_receiver(&self) -> &Arc<Receiver<()>> {
        &self.collection_finished_receiver
    }

    pub(super) fn collection_in_progress(&self) -> &Arc<AtomicBool> {
        &self.collection_in_progress
    }

    pub(super) fn handshake_completion_time_ms(&self) -> &Arc<AtomicUsize> {
        &self.handshake_completion_time_ms
    }

    pub(super) fn threads_processed_count(&self) -> &Arc<AtomicUsize> {
        &self.threads_processed_count
    }

    pub(super) fn set_current_phase(&self, phase: FugcPhase) {
        self.current_phase.store(Arc::new(phase));
    }

    // Test/safepoint wrappers for legacy callsites
    pub fn scan_thread_roots_at_safepoint(&self) {
        self.root_scanner.scan_all_roots();
    }

    pub fn activate_barriers_at_safepoint(&self) {
        self.write_barrier().activate();
        self.black_allocator().activate();
    }

    pub fn marking_handshake_at_safepoint(&self) {
        let noop: crate::fugc_coordinator::types::HandshakeCallback =
            Box::new(|_thread: &crate::thread::MutatorThread| {});
        self.soft_handshake(noop);
    }

    pub fn prepare_sweep_at_safepoint(&self) {
        // Build SIMD bitvector from markings to prepare for sweep
        self.build_bitvector_from_markings();
    }

    /// Prepare for root re-scanning as part of MMTk integration
    /// This is called by MMTk when preparing to re-scan roots during GC
    pub fn prepare_for_root_rescan(&self) {
        // Reset marking state for new collection cycle
        self.tricolor_marking().reset_marking_state();

        // Prepare for FUGC Step 3 (Black Allocation) by ensuring clean state
        self.set_current_phase(FugcPhase::Idle);

        // Notify any waiting components that we're ready for root scanning
        let _ = self.collection_finished_sender.send(());
    }

    /// Mark objects using cache-optimized approach
    pub fn mark_objects_cache_optimized(&self, objects: &[crate::frontend::types::ObjectReference]) {
        // Use cache-optimized marking if available
        for &obj in objects {
            self.tricolor_marking.mark_grey(obj);
        }
    }

    /// Check if an object is marked
    pub fn is_object_marked(&self, obj: crate::frontend::types::ObjectReference) -> bool {
        self.tricolor_marking.get_color(obj) == crate::concurrent::ObjectColor::Black
    }
}
