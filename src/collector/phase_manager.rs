use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::CollectorPhase;

pub struct PhaseManager {
    phase: AtomicUsize, // CollectorPhase as usize
    phase_changed: Condvar,
    phase_change_mutex: Mutex<()>,
}

impl PhaseManager {
    pub fn new(initial_phase: CollectorPhase) -> Self {
        PhaseManager {
            phase: AtomicUsize::new(initial_phase as usize),
            phase_changed: Condvar::new(),
            phase_change_mutex: Mutex::new(()),
        }
    }

    pub fn request_collection(&self) {
        // Simple trigger - set phase to Marking if we're currently Waiting
        let result = self.phase.compare_exchange(
            CollectorPhase::Waiting as usize,
            CollectorPhase::Marking as usize,
            Ordering::Release,
            Ordering::Relaxed,
        );

        // If phase change succeeded, notify collector threads
        if result.is_ok() {
            let _guard = self.phase_change_mutex.lock().unwrap();
            self.phase_changed.notify_all();
        }
    }

    pub fn set_phase(&self, new_phase: CollectorPhase) {
        self.phase.store(new_phase as usize, Ordering::Release);
        let _guard = self.phase_change_mutex.lock().unwrap();
        self.phase_changed.notify_all();
    }

    pub fn wait_for_phase_change(&self, expected_phase: CollectorPhase) {
        let guard = self.phase_change_mutex.lock().unwrap();
        let _result = self.phase_changed.wait_while(guard, |_| {
            self.phase.load(Ordering::Acquire) == expected_phase as usize
        });
    }

    pub fn current_phase(&self) -> CollectorPhase {
        // Helper to get the current phase as enum
        CollectorPhase::from_usize(self.phase.load(Ordering::Acquire))
    }
}
