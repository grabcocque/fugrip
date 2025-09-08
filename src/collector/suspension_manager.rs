use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// Suspension management for fork safety - separated for better testing
pub struct SuspensionManager {
    pub suspend_count: AtomicUsize,
    pub suspension_requested: AtomicBool,
    pub suspended: Condvar,
    pub active_worker_count: AtomicUsize,
    pub suspended_worker_count: AtomicUsize,
}

impl Default for SuspensionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SuspensionManager {
    pub fn new() -> Self {
        Self {
            suspend_count: AtomicUsize::new(0),
            suspension_requested: AtomicBool::new(false),
            suspended: Condvar::new(),
            active_worker_count: AtomicUsize::new(0),
            suspended_worker_count: AtomicUsize::new(0),
        }
    }

    pub fn register_worker_thread(&self) {
        self.active_worker_count.fetch_add(1, Ordering::Release);
    }

    pub fn unregister_worker_thread(&self) {
        self.active_worker_count.fetch_sub(1, Ordering::Release);
    }

    pub fn worker_acknowledge_suspension(&self) {
        let prev_suspended = self.suspended_worker_count.fetch_add(1, Ordering::Release);
        let active_workers = self.active_worker_count.load(Ordering::Acquire);

        // If this is the last worker to suspend, notify the collector
        if prev_suspended + 1 >= active_workers {
            self.suspended.notify_all();
        }
    }

    pub fn worker_acknowledge_resumption(&self) {
        self.suspended_worker_count.fetch_sub(1, Ordering::Release);
    }

    // Fork safety methods from suspend_for_fork.rs
    pub fn suspend_for_fork(&self) {
        let suspend_count = self.suspend_count.fetch_add(1, Ordering::AcqRel);

        if suspend_count == 0 {
            // First suspension - actually suspend the collector
            self.request_suspension();

            // Wait for all collector threads and workers to stop
            self.wait_for_suspension();
        }
    }

    pub fn resume_after_fork(&self) {
        let suspend_count = self.suspend_count.fetch_sub(1, Ordering::AcqRel);

        if suspend_count == 1 {
            // Last resume - restart the collector
            self.resume_collection();
        }
    }

    /// Request suspension of all GC activities.
    /// This signals all collector threads to suspend their operations.
    pub fn request_suspension(&self) {
        // Signal suspension request
        self.suspension_requested.store(true, Ordering::Release);
    }

    /// Wait for all GC activities to be suspended.
    /// This blocks until all collector threads have acknowledged suspension.
    pub fn wait_for_suspension(&self) {
        // Reset suspended worker count
        self.suspended_worker_count.store(0, Ordering::Release);

        // Wait for all active workers to acknowledge suspension
        let active_workers = self.active_worker_count.load(Ordering::Acquire);

        if active_workers > 0 {
            let guard = Mutex::new(());
            let guard = guard.lock().unwrap();

            // Wait with timeout for all workers to suspend
            let _result = self.suspended.wait_timeout_while(
                guard,
                std::time::Duration::from_millis(1000), // Increased timeout
                |_| self.suspended_worker_count.load(Ordering::Acquire) < active_workers,
            );

            // If timeout occurred, log a warning but continue
            // In production, we might want to be more aggressive
            let suspended_count = self.suspended_worker_count.load(Ordering::Acquire);
            if suspended_count < active_workers {
                eprintln!(
                    "Warning: Only {}/{} worker threads acknowledged suspension",
                    suspended_count, active_workers
                );
            }
        }
    }

    /// Resume collection after suspension.
    /// This allows GC activities to restart.
    pub fn resume_collection(&self) {
        // Clear the suspension request
        self.suspension_requested.store(false, Ordering::Release);

        // Notify any threads waiting for resume
        self.suspended.notify_all();
    }

    /// Check if suspension is currently requested.
    /// Worker threads should check this periodically and suspend if needed.
    pub fn is_suspension_requested(&self) -> bool {
        self.suspension_requested.load(Ordering::Acquire)
    }

    /// Called by worker threads to acknowledge suspension and wait for resume.
    pub fn worker_suspended(&self) {
        // Acknowledge suspension
        self.worker_acknowledge_suspension();

        // Wait for suspension to be cleared
        let guard = Mutex::new(());
        let guard = guard.lock().unwrap();
        let _result = self
            .suspended
            .wait_while(guard, |_| self.suspension_requested.load(Ordering::Acquire));

        // Acknowledge resumption
        self.worker_acknowledge_resumption();
    }
}
