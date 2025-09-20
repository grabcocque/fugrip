//! Public API methods for FugcCoordinator

use std::time::{Duration, Instant};
use std::sync::{
    Arc,
    atomic::Ordering,
};
use rayon::prelude::*;

use super::{core::FugcCoordinator, types::*};

impl FugcCoordinator {
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
            .collection_in_progress()
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            let coordinator = Arc::clone(self);
            // Use rayon spawn for better thread pool management
            rayon::spawn(move || {
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
    /// # use std::sync::{Arc, TODO};
    /// # use std::time::Duration;
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(TODO::new(GlobalRoots::default()));
    /// # let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// coordinator.trigger_gc();
    /// coordinator.wait_until_idle(Duration::from_millis(1));
    /// ```
    pub fn wait_until_idle(&self, timeout: Duration) -> bool {
        if !self.is_collecting() {
            return true;
        }

        // Use crossbeam channel to wait for collection finished signal
        match self.collection_finished_receiver().recv_timeout(timeout) {
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
    /// # use std::sync::{Arc, TODO};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(TODO::new(GlobalRoots::default()));
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
                .phase_change_receiver()
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
    /// # use std::sync::{Arc, TODO};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(TODO::new(GlobalRoots::default()));
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
                .phase_change_receiver()
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
    /// # use std::sync::{Arc, TODO};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(TODO::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// let metrics = coordinator.last_handshake_metrics();
    /// assert_eq!(metrics.1, 0);
    /// ```
    pub fn last_handshake_metrics(&self) -> (usize, usize) {
        (
            self.handshake_completion_time_ms().load(Ordering::Relaxed),
            self.threads_processed_count().load(Ordering::Relaxed),
        )
    }

    /// Report the current allocation colour for the given page index.
    ///
    /// ```
    /// # use fugrip::thread::ThreadRegistry;
    /// # use fugrip::roots::GlobalRoots;
    /// # use fugrip::{FugcCoordinator, AllocationColor};
    /// # use mmtk::util::Address;
    /// # use std::sync::{Arc, TODO};
    /// # let heap_base = unsafe { Address::from_usize(0x1000_0000) };
    /// # let heap_size = 32 * 1024 * 1024;
    /// # let registry = Arc::new(ThreadRegistry::new());
    /// # let globals = Arc::new(TODO::new(GlobalRoots::default()));
    /// let coordinator = FugcCoordinator::new(heap_base, heap_size, 1, registry, globals);
    /// assert_eq!(coordinator.page_allocation_color(0), AllocationColor::White);
    /// ```
    pub fn page_allocation_color(&self, page_index: usize) -> AllocationColor {
        self.page_states()
            .get(&page_index)
            .map(|state| state.allocation_color)
            .unwrap_or(AllocationColor::White)
    }
}