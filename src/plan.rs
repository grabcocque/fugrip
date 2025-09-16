//! FUGC plan implementation that integrates with MMTk's MarkSweep plan and provides
//! FUGC-specific concurrent marking and optimized write barriers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use mmtk::MMTK;
use mmtk::util::ObjectReference;

use crate::binding::RustVM;
use crate::concurrent::WriteBarrier;
use crate::fugc_coordinator::FugcCoordinator;

/// FUGC plan manager that coordinates MMTk MarkSweep plan with FUGC concurrent features.
/// This follows MMTk's architecture by using the public API and integrating at the VM binding level.
///
/// # Examples
///
/// ```
/// use fugrip::plan::FugcPlanManager;
///
/// // Create a new FUGC plan manager
/// let mut plan_manager = FugcPlanManager::new();
///
/// // Check initial configuration
/// assert!(plan_manager.is_concurrent_collection_enabled());
///
/// // Configure concurrent collection
/// plan_manager.set_concurrent_collection(false);
/// assert!(!plan_manager.is_concurrent_collection_enabled());
///
/// // Get statistics (will have default values before MMTk initialization)
/// let stats = plan_manager.get_fugc_stats();
/// assert_eq!(stats.work_stolen, 0);
/// assert_eq!(stats.work_shared, 0);
/// ```
pub struct FugcPlanManager {
    /// The MMTk instance configured with MarkSweep plan
    mmtk: Option<&'static MMTK<RustVM>>,

    /// FUGC 8-step protocol coordinator (primary coordinator)
    fugc_coordinator: Arc<FugcCoordinator>,

    /// Flag to enable/disable concurrent collection features
    concurrent_collection_enabled: AtomicBool,
}

impl FugcPlanManager {
    /// Create a new FUGC plan manager. The MMTk instance will be set later during initialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    ///
    /// let plan_manager = FugcPlanManager::new();
    /// assert!(plan_manager.is_concurrent_collection_enabled());
    ///
    /// // Plan manager is ready but not yet initialized with MMTk
    /// // Will panic if mmtk() is called before initialize()
    /// ```
    pub fn new() -> Self {
        // Initialize FUGC coordinator with basic configuration
        let thread_registry = Arc::new(crate::thread::ThreadRegistry::new());
        let global_roots = Arc::new(Mutex::new(crate::roots::GlobalRoots::default()));

        // Use reasonable defaults for heap configuration
        let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) }; // 256MB base
        let heap_size = 128 * 1024 * 1024; // 128MB heap
        let num_workers = 4; // 4 GC workers

        let fugc_coordinator = Arc::new(FugcCoordinator::new(
            heap_base,
            heap_size,
            num_workers,
            thread_registry,
            global_roots,
        ));

        Self {
            mmtk: None,
            fugc_coordinator,
            concurrent_collection_enabled: AtomicBool::new(true),
        }
    }

    /// Initialize the FUGC plan manager with an MMTk instance.
    /// This should be called once during VM initialization.
    pub fn initialize(&mut self, mmtk: &'static MMTK<RustVM>) {
        self.mmtk = Some(mmtk);
    }

    /// Get the MMTk instance. Panics if not initialized.
    pub fn mmtk(&self) -> &'static MMTK<RustVM> {
        self.mmtk.expect("FUGC plan manager not initialized")
    }

    /// Get access to the optimized write barrier for VM integration
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    ///
    /// let plan_manager = FugcPlanManager::new();
    /// let write_barrier = plan_manager.get_write_barrier();
    ///
    /// // Write barrier can be used for FUGC-optimized pointer updates
    /// assert!(!write_barrier.is_active()); // Initially inactive
    /// ```
    pub fn get_write_barrier(&self) -> &WriteBarrier {
        self.fugc_coordinator.write_barrier()
    }

    /// Enable or disable concurrent collection features
    pub fn set_concurrent_collection(&self, enabled: bool) {
        self.concurrent_collection_enabled
            .store(enabled, Ordering::Relaxed);
    }

    /// Check if concurrent collection is enabled
    pub fn is_concurrent_collection_enabled(&self) -> bool {
        self.concurrent_collection_enabled.load(Ordering::Relaxed)
    }

    /// Get the FUGC coordinator for advanced operations
    pub fn get_fugc_coordinator(&self) -> &Arc<FugcCoordinator> {
        &self.fugc_coordinator
    }

    /// Start concurrent marking with given roots
    pub fn start_concurrent_marking(&self, roots: Vec<ObjectReference>) {
        if self.is_concurrent_collection_enabled() {
            // Use FUGC coordinator for concurrent marking
            self.fugc_coordinator.trigger_gc();
            let _ = roots; // FUGC uses global roots internally
        }
    }

    /// Finish concurrent marking and wait for completion
    pub fn finish_concurrent_marking(&self) {
        if self.is_concurrent_collection_enabled() {
            // Wait for FUGC collection to complete
            use std::time::Duration;
            self.fugc_coordinator
                .wait_until_idle(Duration::from_millis(5000));
        }
    }

    /// Allocate object using MMTk with FUGC optimizations
    /// Note: This is a simplified API. In practice, allocation goes through MMTk mutators.
    pub fn alloc_info(&self, size: usize, align: usize) -> (usize, usize) {
        // Return size and alignment info for FUGC-optimized allocation
        // In practice, this would coordinate with MMTk's allocation sites

        let aligned_size = (size + align - 1) & !(align - 1);

        // If concurrent collection is enabled, we would mark newly allocated objects as black
        if self.is_concurrent_collection_enabled() {
            // In a real implementation, this would coordinate with the black allocator
        }

        (aligned_size, align)
    }

    /// Post allocation hook for FUGC-specific processing
    pub fn post_alloc(&self, obj: ObjectReference, bytes: usize) {
        // In a real implementation, this would call MMTk's post_alloc via the mutator
        // For now, this is a placeholder showing the FUGC-specific processing

        let _ = obj;
        let _ = bytes;

        // Additional FUGC-specific post-allocation processing
        if self.is_concurrent_collection_enabled() {
            self.fugc_coordinator.black_allocator().allocate_black(obj);
        }
    }

    /// Handle write barrier - core FUGC optimization
    pub fn handle_write_barrier(
        &self,
        src: ObjectReference,
        slot: mmtk::util::Address,
        target: ObjectReference,
    ) {
        // In a real implementation, this would call MMTk's write barrier via the mutator
        // For now, this demonstrates the FUGC write barrier integration

        let _ = src;
        let _ = slot;
        let _ = target;

        // FUGC-specific optimized write barrier
        if self.is_concurrent_collection_enabled() {
            // Convert the slot address to a mutable pointer for the write barrier
            let slot_ptr = slot.to_ptr::<ObjectReference>() as *mut ObjectReference;
            unsafe {
                self.get_write_barrier().write_barrier(slot_ptr, target);
            }
        }
    }

    /// Get statistics from both MMTk and FUGC extensions
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    ///
    /// let plan_manager = FugcPlanManager::new();
    /// let stats = plan_manager.get_fugc_stats();
    ///
    /// // Statistics provide insight into FUGC performance
    /// assert!(stats.concurrent_collection_enabled);
    /// assert_eq!(stats.work_stolen, 0); // No work stealing yet
    /// assert_eq!(stats.work_shared, 0); // No work sharing yet
    /// assert_eq!(stats.objects_allocated_black, 0); // No black allocations yet
    /// ```
    pub fn get_fugc_stats(&self) -> FugcStats {
        // Get stats from FUGC coordinator components
        let black_allocator_stats = self.fugc_coordinator.black_allocator().get_stats();
        let cycle_stats = self.fugc_coordinator.get_cycle_stats();

        // Note: In a real implementation, we would get actual MMTk stats
        let (total_bytes, used_bytes) = if let Some(mmtk) = self.mmtk {
            // MMTk plans have get_total_pages and other methods, but not get_total_bytes
            // We'll use placeholder values for demonstration
            let total_pages = mmtk.get_plan().get_total_pages();
            let reserved_pages = mmtk.get_plan().get_reserved_pages();
            (total_pages * 4096, reserved_pages * 4096) // Assume 4KB pages
        } else {
            (0, 0)
        };

        FugcStats {
            concurrent_collection_enabled: self.is_concurrent_collection_enabled(),
            work_stolen: 0, // FUGC uses different work coordination
            work_shared: cycle_stats.handshakes_performed, // Use handshakes as work sharing metric
            objects_allocated_black: black_allocator_stats,
            total_bytes,
            used_bytes,
        }
    }

    /// Trigger garbage collection with FUGC optimizations.
    ///
    /// Initiates the complete FUGC 8-step protocol through the integrated coordinator.
    /// This method only triggers collection if MMTk has been initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    /// use std::time::Duration;
    ///
    /// let plan_manager = FugcPlanManager::new();
    ///
    /// // Initially not collecting
    /// assert!(!plan_manager.is_fugc_collecting());
    ///
    /// // Trigger collection (uses coordinator directly since MMTk not initialized)
    /// plan_manager.gc();
    ///
    /// // Monitor collection progress
    /// let coordinator = plan_manager.get_fugc_coordinator();
    /// coordinator.wait_until_idle(Duration::from_millis(500));
    /// ```
    pub fn gc(&self) {
        if let Some(_mmtk) = self.mmtk {
            // Always use FUGC 8-step protocol collection
            self.fugc_coordinator.trigger_gc();

            // Note: In a real implementation, this would coordinate with MMTk's collection API
            // MMTk handles the actual sweeping and memory reclamation
        }
    }

    /// Get access to the FUGC coordinator for advanced operations.
    ///
    /// Provides direct access to the underlying FUGC coordinator for fine-grained
    /// control over collection phases, handshakes, and component access.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    ///
    /// let plan_manager = FugcPlanManager::new();
    /// let coordinator = plan_manager.fugc_coordinator();
    ///
    /// // Access coordinator components directly
    /// let write_barrier = coordinator.write_barrier();
    /// let tricolor_marking = coordinator.tricolor_marking();
    /// let black_allocator = coordinator.black_allocator();
    ///
    /// // Check current state
    /// assert_eq!(coordinator.current_phase(), fugrip::FugcPhase::Idle);
    /// ```
    pub fn fugc_coordinator(&self) -> &Arc<FugcCoordinator> {
        &self.fugc_coordinator
    }

    /// Check the current phase of FUGC collection.
    ///
    /// Returns the current step in the FUGC 8-step protocol, providing
    /// visibility into collection progress.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    ///
    /// let plan_manager = FugcPlanManager::new();
    ///
    /// // Initially idle
    /// assert_eq!(plan_manager.fugc_phase(), fugrip::FugcPhase::Idle);
    ///
    /// // After triggering collection, phase will advance
    /// plan_manager.get_fugc_coordinator().trigger_gc();
    /// // Phase will be one of the active collection phases
    /// ```
    pub fn fugc_phase(&self) -> crate::fugc_coordinator::FugcPhase {
        self.fugc_coordinator.current_phase()
    }

    /// Check if FUGC collection is currently in progress.
    ///
    /// Returns true if the FUGC 8-step protocol is actively running,
    /// false if the coordinator is idle.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    /// use std::time::Duration;
    ///
    /// let plan_manager = FugcPlanManager::new();
    ///
    /// // Initially not collecting
    /// assert!(!plan_manager.is_fugc_collecting());
    ///
    /// // After triggering collection
    /// plan_manager.get_fugc_coordinator().trigger_gc();
    /// assert!(plan_manager.is_fugc_collecting());
    ///
    /// // Wait for completion
    /// plan_manager.get_fugc_coordinator().wait_until_idle(Duration::from_millis(500));
    /// assert!(!plan_manager.is_fugc_collecting());
    /// ```
    pub fn is_fugc_collecting(&self) -> bool {
        self.fugc_coordinator.is_collecting()
    }

    /// Get FUGC cycle statistics.
    ///
    /// Returns detailed performance metrics from the FUGC coordinator,
    /// including cycle counts, timing information, and handshake statistics.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::plan::FugcPlanManager;
    /// use std::time::Duration;
    ///
    /// let plan_manager = FugcPlanManager::new();
    ///
    /// // Initial stats
    /// let initial_stats = plan_manager.get_fugc_cycle_stats();
    /// assert_eq!(initial_stats.cycles_completed, 0);
    ///
    /// // Trigger and complete a collection
    /// let coordinator = plan_manager.get_fugc_coordinator();
    /// coordinator.trigger_gc();
    /// coordinator.wait_until_idle(Duration::from_millis(500));
    ///
    /// // Updated stats
    /// let final_stats = plan_manager.get_fugc_cycle_stats();
    /// assert!(final_stats.cycles_completed >= 1);
    /// ```
    pub fn get_fugc_cycle_stats(&self) -> crate::fugc_coordinator::FugcCycleStats {
        self.fugc_coordinator.get_cycle_stats()
    }
}

/// Statistics for FUGC plan operations
///
/// # Examples
///
/// ```
/// use fugrip::plan::{FugcPlanManager, FugcStats};
///
/// let plan_manager = FugcPlanManager::new();
/// let stats = plan_manager.get_fugc_stats();
///
/// // Check concurrent collection status
/// assert!(stats.concurrent_collection_enabled);
///
/// // Work stealing and sharing statistics
/// println!("Work stolen: {}", stats.work_stolen);
/// println!("Work shared: {}", stats.work_shared);
///
/// // Memory statistics
/// println!("Total bytes: {}", stats.total_bytes);
/// println!("Used bytes: {}", stats.used_bytes);
///
/// // Black allocation during concurrent marking
/// println!("Objects allocated black: {}", stats.objects_allocated_black);
/// ```
#[derive(Debug, Clone)]
pub struct FugcStats {
    pub concurrent_collection_enabled: bool,
    pub work_stolen: usize,
    pub work_shared: usize,
    pub objects_allocated_black: usize,
    pub total_bytes: usize,
    pub used_bytes: usize,
}

impl Default for FugcPlanManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure MMTk to use the optimal plan for FUGC characteristics.
///
/// **MarkSweep vs Immix Choice:**
///
/// **MarkSweep** is the better choice for FUGC because:
/// 1. **Non-moving**: Like FUGC, MarkSweep never moves objects, which eliminates
///    the need for evacuation barriers and simplifies concurrent marking.
/// 2. **Simple concurrent marking**: Objects stay in place during concurrent marking,
///    making it easier to implement FUGC's incremental stack scanning.
/// 3. **Predictable performance**: No evacuation pauses or compaction overhead.
/// 4. **Better for concurrent collection**: Concurrent marking is simpler when
///    objects don't move during the process.
///
/// **Immix** has some benefits (better cache locality, partial compaction) but:
/// 1. **Moving**: Immix can move objects during defragmentation, which complicates
///    concurrent collection and requires more sophisticated barriers.
/// 2. **Complex concurrent marking**: Object movement during concurrent phases
///    requires additional synchronization and forwarding logic.
/// 3. **FUGC compatibility**: FUGC's design assumes non-moving collection.
///
/// Therefore, we configure MMTk to use **MarkSweep** as the base plan.
///
/// # Examples
///
/// ```
/// use fugrip::plan::create_fugc_mmtk_options;
///
/// let options = create_fugc_mmtk_options();
///
/// // Options are configured for FUGC characteristics
/// // Thread count is optimized for concurrent collection
/// // Stress factor is set for incremental collection behavior
/// ```
pub fn create_fugc_mmtk_options() -> mmtk::util::options::Options {
    let mut options = mmtk::util::options::Options::default();

    // Use MarkSweep plan - optimal for FUGC's non-moving concurrent design
    if !options
        .plan
        .set(mmtk::util::options::PlanSelector::MarkSweep)
    {
        panic!("Failed to set MarkSweep plan");
    }

    // Enable concurrent collection features
    let thread_count = std::cmp::max(1, num_cpus::get() / 2); // Reserve half CPUs for mutators
    if !options.threads.set(thread_count) {
        panic!("Failed to set thread count");
    }

    // Configure for FUGC-style incremental collection
    if !options.stress_factor.set(4096) {
        panic!("Failed to set stress factor");
    }

    options
}
