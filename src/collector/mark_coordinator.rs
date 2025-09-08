use crate::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Coordinates parallel marking workers and manages marking state.
/// 
/// This component is responsible for:
/// - Managing the global mark stack shared between workers
/// - Coordinating work stealing and donation between parallel markers
/// - Tracking worker lifecycle and completion
/// - Managing marking phase activation state
pub struct MarkCoordinator {
    /// Global mark stack shared between all marking workers
    pub global_mark_stack: Mutex<Vec<SendPtr<GcHeader<()>>>>,
    
    /// Number of active marking workers
    pub worker_count: AtomicUsize,
    
    /// Number of workers that have finished marking
    pub workers_finished: AtomicUsize,
    
    /// Whether marking phase is currently active
    pub marking_active: AtomicBool,
    
    /// Current allocation color (true = black, false = white)
    pub allocation_color: AtomicBool,

    /// Count of steal operations performed (for diagnostics/testing)
    pub steal_count: AtomicUsize,

    /// Count of donation operations performed (for diagnostics/testing)
    pub donation_count: AtomicUsize,
}

impl MarkCoordinator {
    /// Create a new mark coordinator for parallel marking.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::collector::mark_coordinator::MarkCoordinator;
    ///
    /// let coordinator = MarkCoordinator::new();
    /// assert!(!coordinator.is_marking_active());
    /// assert_eq!(coordinator.get_work_queue_size(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            global_mark_stack: Mutex::new(Vec::new()),
            worker_count: AtomicUsize::new(0),
            workers_finished: AtomicUsize::new(0),
            marking_active: AtomicBool::new(false),
            allocation_color: AtomicBool::new(false),
            steal_count: AtomicUsize::new(0),
            donation_count: AtomicUsize::new(0),
        }
    }

    /// Start parallel marking with the specified number of workers.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::collector::mark_coordinator::MarkCoordinator;
    ///
    /// let coordinator = MarkCoordinator::new();
    /// coordinator.start_parallel_marking(4);
    /// assert!(coordinator.is_marking_active());
    /// assert!(coordinator.get_allocation_color()); // Black allocation during marking
    /// ```
    pub fn start_parallel_marking(&self, worker_count: usize) {
        self.marking_active.store(true, Ordering::Release);
        self.worker_count.store(worker_count, Ordering::Release);
        self.workers_finished.store(0, Ordering::Release);
        
        // Switch to black allocation during marking
        self.allocation_color.store(true, Ordering::Release);
    }

    /// Stop the marking phase
    /// Check if all workers are idle (no active work stealing/processing).
    /// 
    /// This is used for quiescence detection in marking sessions.
    pub fn all_workers_idle(&self) -> bool {
        let active = self.worker_count.load(Ordering::Acquire);
        let finished = self.workers_finished.load(Ordering::Acquire);
        active == 0 || finished >= active
    }
    
    pub fn stop_marking(&self) {
        self.marking_active.store(false, Ordering::Release);
    }

    /// Check if marking is currently active.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::collector::mark_coordinator::MarkCoordinator;
    ///
    /// let coordinator = MarkCoordinator::new();
    /// assert!(!coordinator.is_marking_active());
    ///
    /// coordinator.start_parallel_marking(2);
    /// assert!(coordinator.is_marking_active());
    ///
    /// coordinator.stop_marking();
    /// assert!(!coordinator.is_marking_active());
    /// ```
    pub fn is_marking_active(&self) -> bool {
        self.marking_active.load(Ordering::Acquire)
    }

    /// Get current allocation color (true = black, false = white).
    ///
    /// In FUGC, allocation color indicates whether newly allocated objects
    /// start marked (black) or unmarked (white) based on the collection phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::collector::mark_coordinator::MarkCoordinator;
    ///
    /// let coordinator = MarkCoordinator::new();
    /// assert!(!coordinator.get_allocation_color()); // White by default
    ///
    /// coordinator.start_parallel_marking(1);
    /// assert!(coordinator.get_allocation_color()); // Black during marking
    /// ```
    pub fn get_allocation_color(&self) -> bool {
        self.allocation_color.load(Ordering::Acquire)
    }

    /// Set allocation color for new objects
    pub fn set_allocation_color(&self, black: bool) {
        self.allocation_color.store(black, Ordering::Release);
    }

    /// Steal work from the global mark stack.
    /// Returns a batch of work items to process locally.
    pub fn steal_work(&self) -> Option<Vec<SendPtr<GcHeader<()>>>> {
        const STEAL_BATCH_SIZE: usize = 32;

        let mut global_stack = self.global_mark_stack.lock().unwrap();
        if global_stack.is_empty() {
            return None;
        }

        // Steal up to STEAL_BATCH_SIZE items, or half the remaining work
        let steal_count = (global_stack.len() / 2).clamp(1, STEAL_BATCH_SIZE);

        let stolen_work = global_stack.drain(..steal_count).collect();
        self.steal_count.fetch_add(1, Ordering::Relaxed);
        Some(stolen_work)
    }

    /// Donate work from local stack back to the global stack.
    /// This helps balance work across all worker threads.
    pub fn donate_work(&self, local_stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        const DONATION_BATCH_SIZE: usize = 100;

        if local_stack.len() <= DONATION_BATCH_SIZE {
            return; // Not enough work to donate
        }

        // Donate half of the local work to the global stack
        let donate_count = local_stack.len() / 2;
        let mut donated_work = local_stack.drain(..donate_count).collect::<Vec<_>>();

        let mut global_stack = self.global_mark_stack.lock().unwrap();
        global_stack.append(&mut donated_work);
        self.donation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Add root objects to the mark stack to start marking
    pub fn add_roots(&self, roots: Vec<SendPtr<GcHeader<()>>>) {
        let mut global_stack = self.global_mark_stack.lock().unwrap();
        global_stack.extend(roots);
    }

    /// Check if marking is complete (no work remaining and all workers finished)
    pub fn is_marking_complete(&self) -> bool {
        let global_stack = self.global_mark_stack.lock().unwrap();
        let workers_finished = self.workers_finished.load(Ordering::Acquire);
        let total_workers = self.worker_count.load(Ordering::Acquire);
        
        global_stack.is_empty() && workers_finished >= total_workers
    }

    /// Signal that a worker has finished marking
    pub fn worker_finished(&self) {
        self.workers_finished.fetch_add(1, Ordering::Release);
    }

    /// Reset coordinator state for a new marking cycle
    pub fn reset_for_new_cycle(&self) {
        let mut global_stack = self.global_mark_stack.lock().unwrap();
        global_stack.clear();
        
        self.worker_count.store(0, Ordering::Release);
        self.workers_finished.store(0, Ordering::Release);
        self.marking_active.store(false, Ordering::Release);
        self.allocation_color.store(false, Ordering::Release); // Switch back to white
        self.steal_count.store(0, Ordering::Release);
        self.donation_count.store(0, Ordering::Release);
    }

    /// Get the current size of the global mark stack (for monitoring)
    pub fn get_work_queue_size(&self) -> usize {
        self.global_mark_stack.lock().unwrap().len()
    }

    /// Get number of steal operations observed
    pub fn get_steal_count(&self) -> usize {
        self.steal_count.load(Ordering::Acquire)
    }

    /// Get number of donation operations observed
    pub fn get_donation_count(&self) -> usize {
        self.donation_count.load(Ordering::Acquire)
    }

    /// Execute the marking worker loop
    pub fn run_marking_worker(&self, collector: Arc<crate::collector_phases::CollectorState>) {
        let mut local_stack: Vec<SendPtr<GcHeader<()>>> = Vec::new();

        while self.is_marking_active() {
            // Check for suspension request
            if collector.suspension_manager.is_suspension_requested() {
                // Acknowledge suspension and wait for resume
                collector.suspension_manager.worker_suspended();
                continue;
            }

            // Try to get work from global stack
            if local_stack.is_empty() {
                if let Some(work) = self.steal_work() {
                    local_stack.extend(work);
                } else {
                    // No work available, yield and try again
                    std::thread::yield_now();
                    continue;
                }
            }

            // Process local work
            while let Some(header_ptr) = local_stack.pop() {
                unsafe {
                    let header = &*header_ptr.as_ptr();
                    if !header.mark_bit.load(Ordering::Acquire) {
                        header.mark_bit.store(true, Ordering::Release);

                        // Trace outgoing pointers using type info
                        let obj_ptr = header_ptr.as_ptr() as *const ();
                        (header.type_info.trace_fn)(obj_ptr, &mut local_stack);
                    }
                }

                // Donate work back if stack gets too large
                if local_stack.len() > 1000 {
                    self.donate_work(&mut local_stack);
                }
            }
        }

        // Donate remaining work before finishing
        if !local_stack.is_empty() {
            self.donate_work(&mut local_stack);
        }

        // Signal that this worker has finished
        self.worker_finished();
    }
}

impl Default for MarkCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
