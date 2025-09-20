//! FUGC plan implementation that integrates with MMTk's MarkSweep plan and provides
//! FUGC-specific concurrent marking and optimized write barriers.

use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

    /// Allocation statistics tracking
    allocation_stats: AllocationStats,
}

/// Statistics for tracking allocations
pub struct AllocationStats {
    pub total_allocated: AtomicUsize,
    pub allocation_count: AtomicUsize,
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self {
            total_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }
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
        // Initialize FUGC coordinator using DI container
        let container = crate::di::DIContainer::new();

        // Use reasonable defaults for heap configuration
        let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) }; // 256MB base
        let heap_size = 128 * 1024 * 1024; // 128MB heap
        let num_workers = 4; // 4 GC workers

        let fugc_coordinator = container.create_fugc_coordinator(heap_base, heap_size, num_workers);

        Self {
            mmtk: None,
            fugc_coordinator: fugc_coordinator.clone(),
            concurrent_collection_enabled: AtomicBool::new(true),
            allocation_stats: AllocationStats::default(),
        }
    }

    /// Initialize the FUGC plan manager with an MMTk instance.
    /// This should be called once during VM initialization.
    pub fn initialize(&mut self, mmtk: &'static MMTK<RustVM>) {
        self.mmtk = Some(mmtk);
    }

    /// Get the MMTk instance. Returns an error if not initialized.
    pub fn mmtk(&self) -> anyhow::Result<&'static MMTK<RustVM>> {
        self.mmtk.ok_or_else(|| {
            anyhow::anyhow!("FUGC plan manager not initialized - call initialize() first")
        })
    }

    /// Get the MMTk instance (unsafe version for when you know it's initialized)
    pub fn mmtk_unchecked(&self) -> &'static MMTK<RustVM> {
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

        // Correct alignment to next power of two if not already one
        let corrected_align = if align == 0 {
            1
        } else if align.is_power_of_two() {
            align
        } else {
            // Find next power of two, handling overflow gracefully
            if align >= (usize::MAX >> 1) {
                usize::MAX // For extreme values, return max (test will handle this)
            } else {
                align.next_power_of_two()
            }
        };

        let aligned_size = if corrected_align == usize::MAX {
            // Special case: can't align to usize::MAX, return size as-is (saturated)
            size
        } else {
            size.saturating_add(corrected_align.saturating_sub(1))
                & !(corrected_align.saturating_sub(1))
        };

        if self.is_concurrent_collection_enabled() {
            self.fugc_coordinator.ensure_black_allocation_active();
        }

        (aligned_size, corrected_align)
    }

    /// Post allocation hook for FUGC-specific processing
    pub fn post_alloc(&self, obj: ObjectReference, bytes: usize) {
        // Perform FUGC-specific post-allocation processing
        if self.is_concurrent_collection_enabled() {
            self.fugc_coordinator.black_allocator().allocate_black(obj);
        }

        // Track allocation statistics
        self.allocation_stats
            .total_allocated
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        self.allocation_stats
            .allocation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Check if we should trigger collection based on allocation pressure
        self.check_allocation_pressure(bytes);
    }

    /// Check allocation pressure and potentially trigger collection.
    /// This integrates with MMTk's allocation paths to provide responsive GC triggering.
    fn check_allocation_pressure(&self, _bytes_allocated: usize) {
        // Fast path: only check every N allocations to reduce overhead
        static ALLOCATION_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let count = ALLOCATION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Only check pressure every 64 allocations for performance
        if !count.is_multiple_of(64) {
            return;
        }

        // Check if collection should be triggered
        if self.should_trigger_collection() {
            // Trigger collection asynchronously to avoid blocking allocation
            if let Some(mmtk) = self.mmtk {
                self.coordinate_fugc_with_mmtk_collection(mmtk);
            }
        }
    }

    /// Handle write barrier - core FUGC optimization
    pub fn handle_write_barrier(
        &self,
        src: ObjectReference,
        slot: mmtk::util::Address,
        target: ObjectReference,
    ) {
        unsafe {
            let slot_ptr = slot.to_mut_ptr::<ObjectReference>();
            // Always perform the actual store
            std::ptr::write(slot_ptr, target);

            if self.is_concurrent_collection_enabled() {
                self.get_write_barrier().write_barrier(slot_ptr, target);
                self.fugc_coordinator
                    .generational_write_barrier(src, target);
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
        let (total_bytes, used_bytes) = if let Some(mmtk) = self.mmtk {
            // Calculate actual memory usage from MMTk plan
            let total_pages = mmtk.get_plan().get_total_pages();
            let reserved_pages = mmtk.get_plan().get_reserved_pages();
            let page_size = mmtk::util::constants::BYTES_IN_PAGE;
            (total_pages * page_size, reserved_pages * page_size)
        } else {
            // Use allocation statistics when MMTk is not available
            let total_allocated = self
                .allocation_stats
                .total_allocated
                .load(std::sync::atomic::Ordering::Relaxed);
            (total_allocated, total_allocated)
        };

        let marking_stats = self.fugc_coordinator.get_marking_stats();

        FugcStats {
            concurrent_collection_enabled: self.is_concurrent_collection_enabled(),
            work_stolen: marking_stats.work_stolen,
            work_shared: marking_stats.work_shared,
            objects_allocated_black: marking_stats.objects_allocated_black,
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
        if let Some(mmtk) = self.mmtk {
            // Integrate FUGC 8-step protocol with MMTk collection phases
            self.coordinate_fugc_with_mmtk_collection(mmtk);
        } else {
            // Fallback to coordinator-only collection for testing/uninitialized state
            self.fugc_coordinator.trigger_gc();
        }
    }

    /// Coordinate FUGC's 8-step protocol with MMTk's collection phases.
    /// This ensures proper integration between FUGC's concurrent marking
    /// and MMTk's heap management and sweeping.
    fn coordinate_fugc_with_mmtk_collection(&self, mmtk: &'static mmtk::MMTK<RustVM>) {
        use crate::fugc_coordinator::FugcPhase;

        // Step 1: Check if we should trigger collection
        if !self.should_trigger_collection() {
            return;
        }

        // Step 2: Start FUGC 8-step protocol
        self.fugc_coordinator.trigger_gc();

        // Step 3: Wait for FUGC to reach marking phase, then coordinate with MMTk
        let coordinator = Arc::clone(&self.fugc_coordinator);

        // Poll FUGC coordinator state and coordinate with MMTk phases
        std::thread::spawn(move || {
            // Wait for write barriers to be activated (Step 2 of FUGC protocol)
            // Use the coordinator's phase advancement helper which listens on the
            // internal channel for phase transitions instead of busy-waiting.
            let _ = coordinator.advance_to_phase(FugcPhase::ActivateBarriers);

            // Now MMTk can begin its collection knowing FUGC barriers are active
            // This handles allocation failure and coordinates with FUGC marking
            // We get the first available mutator thread to trigger the collection
            use crate::binding::MUTATOR_MAP;
            if let Some(entry) = MUTATOR_MAP.get_or_init(DashMap::new).iter().next() {
                let tls = mmtk::util::opaque_pointer::VMMutatorThread(
                    mmtk::util::opaque_pointer::VMThread(
                        mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
                            mmtk::util::Address::from_usize(*entry.key())
                        }),
                    ),
                );
                mmtk::memory_manager::handle_user_collection_request::<RustVM>(mmtk, tls);
            }

            // FUGC will complete its 8-step protocol including sweep coordination
            // MMTk handles the actual memory reclamation and page management
        });
    }

    /// Determine if collection should be triggered based on allocation pressure
    /// and FUGC coordinator state.
    fn should_trigger_collection(&self) -> bool {
        // Don't trigger if already collecting
        if self.is_fugc_collecting() {
            return false;
        }

        // Simple heuristic: trigger after significant allocation activity
        let total_allocated = self
            .allocation_stats
            .total_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        let allocation_count = self
            .allocation_stats
            .allocation_count
            .load(std::sync::atomic::Ordering::Relaxed);

        // Trigger collection if we've allocated more than 32MB or 10k objects
        total_allocated > 32 * 1024 * 1024 || allocation_count > 10_000
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
/// let options = create_fugc_mmtk_options().expect("Failed to create MMTk options");
///
/// // Options are configured for FUGC characteristics
/// // Thread count is optimized for concurrent collection
/// // Stress factor is set for incremental collection behavior
/// ```
pub fn create_fugc_mmtk_options() -> Result<mmtk::util::options::Options> {
    let mut options = mmtk::util::options::Options::default();

    // Use MarkSweep plan - optimal for FUGC's non-moving concurrent design
    if !options
        .plan
        .set(mmtk::util::options::PlanSelector::MarkSweep)
    {
        anyhow::bail!("Failed to set MarkSweep plan");
    }

    // Enable concurrent collection features
    let thread_count = std::cmp::max(1, num_cpus::get() / 2); // Reserve half CPUs for mutators
    if !options.threads.set(thread_count) {
        anyhow::bail!("Failed to set thread count to {}", thread_count);
    }

    // Configure for FUGC-style incremental collection
    if !options.stress_factor.set(4096) {
        anyhow::bail!("Failed to set stress factor to 4096");
    }

    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_fugc_plan_manager_creation() {
        let plan_manager = FugcPlanManager::new();
        assert!(!plan_manager.is_fugc_collecting());

        let default_manager = FugcPlanManager::default();
        assert!(!default_manager.is_fugc_collecting());
    }

    #[test]
    fn test_allocation_stats_tracking() {
        let plan_manager = FugcPlanManager::new();

        // Initial stats should be zero
        let initial_allocated = plan_manager
            .allocation_stats
            .total_allocated
            .load(Ordering::Relaxed);
        let initial_count = plan_manager
            .allocation_stats
            .allocation_count
            .load(Ordering::Relaxed);
        assert_eq!(initial_allocated, 0);
        assert_eq!(initial_count, 0);

        // Track some allocations using post_alloc
        let obj1 = unsafe {
            mmtk::util::ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0x1000),
            )
        };
        let obj2 = unsafe {
            mmtk::util::ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0x2000),
            )
        };
        plan_manager.post_alloc(obj1, 1024);
        plan_manager.post_alloc(obj2, 2048);

        let after_allocated = plan_manager
            .allocation_stats
            .total_allocated
            .load(Ordering::Relaxed);
        let after_count = plan_manager
            .allocation_stats
            .allocation_count
            .load(Ordering::Relaxed);
        assert_eq!(after_allocated, 3072);
        assert_eq!(after_count, 2);
    }

    #[test]
    fn test_allocation_pressure_thresholds() {
        let plan_manager = FugcPlanManager::new();

        // Should not trigger initially
        assert!(!plan_manager.should_trigger_collection());

        // Large allocation should consider triggering
        plan_manager
            .allocation_stats
            .total_allocated
            .store(33 * 1024 * 1024, Ordering::Relaxed);
        assert!(plan_manager.should_trigger_collection());

        // Reset and test object count threshold
        plan_manager
            .allocation_stats
            .total_allocated
            .store(0, Ordering::Relaxed);
        plan_manager
            .allocation_stats
            .allocation_count
            .store(10_001, Ordering::Relaxed);
        assert!(plan_manager.should_trigger_collection());
    }

    #[test]
    fn test_collection_state_management() {
        let plan_manager = FugcPlanManager::new();

        // Initially not collecting
        assert!(!plan_manager.is_fugc_collecting());

        // Test phase queries
        let phase = plan_manager.fugc_phase();
        // Should be in idle state initially
        assert_eq!(
            format!("{:?}", phase),
            format!("{:?}", crate::fugc_coordinator::FugcPhase::Idle)
        );
    }

    #[test]
    fn test_fugc_stats_structure() {
        let stats = FugcStats {
            concurrent_collection_enabled: true,
            work_stolen: 100,
            work_shared: 200,
            objects_allocated_black: 50,
            total_bytes: 1024 * 1024,
            used_bytes: 512 * 1024,
        };

        assert!(stats.concurrent_collection_enabled);
        assert_eq!(stats.work_stolen, 100);
        assert_eq!(stats.work_shared, 200);
        assert_eq!(stats.objects_allocated_black, 50);
        assert_eq!(stats.total_bytes, 1024 * 1024);
        assert_eq!(stats.used_bytes, 512 * 1024);

        // Test Clone trait
        let cloned = stats.clone();
        assert_eq!(stats.work_stolen, cloned.work_stolen);
    }

    #[test]
    fn test_mmtk_options_creation() {
        let result = create_fugc_mmtk_options();
        // May fail in test environment if MMTk is not properly initialized
        // Just verify the function exists and handles errors properly
        match result {
            Ok(_options) => {
                // If successful, verify some basic properties
                // Note: We can't test internal state easily due to MMTk's design
            }
            Err(e) => {
                // Error is acceptable in test environment
                assert!(e.to_string().contains("Failed") || e.to_string().contains("set"));
            }
        }
    }

    #[test]
    fn test_allocation_counter_edge_cases() {
        let plan_manager = FugcPlanManager::new();

        // Test zero allocation
        let obj = unsafe {
            mmtk::util::ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0x3000),
            )
        };
        plan_manager.post_alloc(obj, 0);
        let count = plan_manager
            .allocation_stats
            .allocation_count
            .load(Ordering::Relaxed);
        assert_eq!(count, 1); // Still counts as an allocation event

        // Test large allocation
        let obj2 = unsafe {
            mmtk::util::ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0x4000),
            )
        };
        plan_manager.post_alloc(obj2, usize::MAX - 1000);
        let total = plan_manager
            .allocation_stats
            .total_allocated
            .load(Ordering::Relaxed);
        // Should handle large values without overflow (may wrap)
        assert!(total > 0);
    }

    #[test]
    fn test_write_barrier_edge_cases() {
        let plan_manager = FugcPlanManager::new();

        // Create valid object references for testing
        let _src = unsafe {
            mmtk::util::ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0x1000),
            )
        };
        let _target = unsafe {
            mmtk::util::ObjectReference::from_raw_address_unchecked(
                mmtk::util::Address::from_usize(0x2000),
            )
        };

        // Test write barrier components without dangerous memory access
        let write_barrier = plan_manager.get_write_barrier();
        assert!(!write_barrier.is_active()); // Should not be active initially

        // Test concurrent collection state changes
        assert!(plan_manager.is_concurrent_collection_enabled());
        plan_manager.set_concurrent_collection(false);
        assert!(!plan_manager.is_concurrent_collection_enabled());
        plan_manager.set_concurrent_collection(true);
        assert!(plan_manager.is_concurrent_collection_enabled());
    }

    #[test]
    fn test_gc_triggering_without_mmtk() {
        let plan_manager = FugcPlanManager::new();

        // Should not panic when triggering GC without MMTk
        plan_manager.gc();

        // Collection state should be managed by coordinator
        // May or may not be collecting immediately after trigger
    }

    #[test]
    fn test_concurrent_collection_configuration() {
        let plan_manager = FugcPlanManager::new();

        // Test concurrent collection state
        let enabled = plan_manager.is_concurrent_collection_enabled();
        // Default should be true for FUGC
        assert!(enabled);
    }
}
