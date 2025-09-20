//! Helper methods for FugcCoordinator

use std::time::{Duration, Instant};
use std::sync::{
    Arc,
    atomic::Ordering,
};
use dashmap::DashMap;
use mmtk::util::{Address, ObjectReference};

use super::{core::FugcCoordinator, types::*};
use crate::thread::MutatorThread;

const PAGE_SIZE: usize = mmtk::util::constants::BYTES_IN_PAGE;

impl FugcCoordinator {
    /// Perform soft handshake with all mutator threads using lock-free protocol
    pub(super) fn soft_handshake(&self, callback: HandshakeCallback) {
        let handshake_start = Instant::now();
        let threads = self.thread_registry().iter();
        let thread_count = threads.len();

        if thread_count == 0 {
            return;
        }

        // Use the lock-free handshake protocol from ThreadRegistry
        let handshake_type = crate::handshake::HandshakeType::StackScan;
        let timeout = Duration::from_millis(2000);

        match self
            .thread_registry()
            .perform_handshake(handshake_type, timeout)
        {
            Ok(completions) => {
                let threads_processed = completions.len();
                // Process each thread with the callback using completion data
                for completion in &completions {
                    if let Some(thread) = self.thread_registry().get(completion.thread_id) {
                        callback(&thread);
                    }
                }

                let handshake_time = handshake_start.elapsed().as_millis() as usize;
                self.handshake_completion_time_ms()
                    .store(handshake_time, Ordering::Relaxed);
                self.threads_processed_count()
                    .store(threads_processed, Ordering::Relaxed);

                self.cycle_stats().rcu(|stats| {
                    let mut new_stats = (**stats).clone();
                    new_stats.handshakes_performed += 1;
                    Arc::new(new_stats)
                });
            }
            Err(e) => {
                eprintln!("Handshake failed: {:?}", e);
            }
        }
    }

    /// Check if all mark stacks are empty (termination condition for step 5/6 loop)
    pub(super) fn are_all_mark_stacks_empty(&self) -> bool {
        !self.parallel_coordinator().has_work()
    }

    /// Record that an object resides on a particular allocation page.
    pub(super) fn record_live_object_internal(&self, object: ObjectReference) {
        Self::record_live_object_for_page(
            self.page_states(),
            self.heap_base(),
            self.heap_size(),
            object,
        );
    }

    pub(super) fn record_live_object_for_page(
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
    pub(super) fn page_index_for_object(
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

    /// Get collection cycle statistics.
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
    /// let stats = coordinator.get_cycle_stats();
    /// assert_eq!(stats.cycles_completed, 0);
    /// ```
    pub fn get_cycle_stats(&self) -> FugcCycleStats {
        (**self.cycle_stats().load()).clone()
    }

    #[doc(hidden)]
    pub fn bench_reset_bitvector_state(&self) {
        self.simd_bitvector().clear();
    }

    // Helper for benchmark setup - avoids changing public API
    #[doc(hidden)]
    pub fn bench_mark_objects(&self, objects: &[ObjectReference]) {
        for &obj in objects {
            self.cache_optimized_marking().mark_object(obj);
        }
    }

    #[doc(hidden)]
    pub fn bench_sweep_phase(&self) -> crate::simd_sweep::SweepStatistics {
        self.simd_bitvector().simd_sweep()
    }
}