//! FUGC 8-step protocol implementation

use std::time::Instant;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use rayon::prelude::*;
use mmtk::util::{Address, ObjectReference};

use super::{core::FugcCoordinator, types::*};
use crate::thread::MutatorThread;

const OBJECT_GRANULE: usize = 64;
const PAGE_SIZE: usize = mmtk::util::constants::BYTES_IN_PAGE;

impl FugcCoordinator {
    /// Execute the complete FUGC 8-step collection protocol
    pub(super) fn run_collection_cycle(&self) {
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

        // Update statistics atomically using RCU
        self.cycle_stats().rcu(|stats| {
            let mut new_stats = (**stats).clone();
            new_stats.cycles_completed += 1;
            Arc::new(new_stats)
        });

        // Reset state for next cycle - order is important for proper signaling
        self.collection_in_progress().store(false, Ordering::SeqCst);
        self.set_phase(FugcPhase::Idle); // This sends the completion signal
    }

    /// Reset per-cycle state before starting a new collection.
    fn prepare_cycle_state(&self) {
        self.set_phase(FugcPhase::ActivateBarriers);
        // Reset cache-optimized marking (includes tricolor clearing)
        self.cache_optimized_marking().reset();
        self.parallel_coordinator().reset();
        self.black_allocator().reset();
        self.handshake_completion_time_ms()
            .store(0, Ordering::Relaxed);
        self.threads_processed_count().store(0, Ordering::Relaxed);

        for mut state in self.page_states().iter_mut() {
            state.live_objects = 0;
        }
    }

    /// Step 2: Activate write barriers with soft handshake
    fn step_2_activate_barriers(&self) {
        self.set_phase(FugcPhase::ActivateBarriers);
        self.write_barrier().activate();

        let noop_callback = Box::new(|_thread: &MutatorThread| {});
        self.soft_handshake(noop_callback);
    }

    /// Step 3: Activate black allocation with cache reset handshake
    fn step_3_activate_black_allocation(&self) {
        self.set_phase(FugcPhase::ActivateBlackAllocation);
        self.black_allocator().activate();

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
            let roots = self.global_roots().load();
            // Convert to Vec to enable parallel processing
            let root_addresses: Vec<usize> =
                roots.iter().map(|root_ptr| root_ptr as usize).collect();

            // Process global roots in parallel chunks for better cache locality
            let processed_count: usize = root_addresses
                .par_chunks(16) // Smaller chunks for global roots (typically fewer than stack roots)
                .map(|chunk| {
                    let mut local_count = 0;
                    for &root_addr in chunk {
                        if let Some(root_obj) = ObjectReference::from_raw_address(unsafe {
                            Address::from_usize(root_addr)
                        }) {
                            // Use cache-optimized marking for all global roots
                            self.cache_optimized_marking().mark_object(root_obj);
                            self.parallel_coordinator().share_work(vec![root_obj]);
                            self.record_live_object_internal(root_obj);
                            local_count += 1;
                        }
                    }
                    local_count
                })
                .sum();
            objects_marked += processed_count;
        }

        self.cycle_stats().rcu(|stats| {
            let mut new_stats = (**stats).clone();
            new_stats.total_marking_time_ms += marking_start.elapsed().as_millis() as u64;
            new_stats.objects_marked += objects_marked;
            Arc::new(new_stats)
        });
    }

    /// Step 5: Stack scan using rayon scoped threads (simplified from handshake protocol)
    fn step_5_stack_scan_handshake(&self) {
        self.set_phase(FugcPhase::StackScanHandshake);

        let _tricolor_marking = Arc::clone(self.tricolor_marking());
        let parallel_coordinator = Arc::clone(self.parallel_coordinator());
        let page_states = Arc::clone(self.page_states());
        let total_stack_objects_scanned = Arc::new(AtomicUsize::new(0));
        let heap_base = self.heap_base();
        let heap_size = self.heap_size();

        // Use rayon scoped threads instead of complex handshake protocol
        let threads: Vec<_> = self.thread_registry().iter().into_iter().collect();
        let thread_count = threads.len();

        if thread_count > 0 {
            // Rayon scope eliminates the need for handshake coordination
            rayon::scope(|s| {
                for thread in &threads {
                    let total_scanned = Arc::clone(&total_stack_objects_scanned);
                    let parallel_coordinator = Arc::clone(&parallel_coordinator);
                    let page_states = Arc::clone(&page_states);

                    s.spawn(move |_| {
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
                            parallel_coordinator.inject_global_work(local_grey_objects);
                        }

                        thread.clear_stack_roots();
                    });
                }
            });
        }

        let total_objects_scanned = total_stack_objects_scanned.load(Ordering::Relaxed);

        self.cycle_stats().rcu(|stats| {
            let mut new_stats = (**stats).clone();
            if thread_count > 0 {
                new_stats.avg_stack_scan_objects =
                    total_objects_scanned as f64 / thread_count as f64;
            }
            new_stats.objects_marked += total_objects_scanned;
            Arc::new(new_stats)
        });
    }

    /// Step 6: Tracing phase using rayon parallel execution (simplified from manual coordination)
    fn step_6_tracing(&self) {
        self.set_phase(FugcPhase::Tracing);

        let tracing_start = Instant::now();

        // Use rayon parallel execution instead of manual work stealing
        let objects_processed = self.parallel_coordinator().parallel_mark(vec![]);

        self.cycle_stats().rcu(|stats| {
            let mut new_stats = (**stats).clone();
            new_stats.total_marking_time_ms += tracing_start.elapsed().as_millis() as u64;
            new_stats.objects_marked += objects_processed;
            Arc::new(new_stats)
        });
    }

    /// Scan object fields using the object classifier's adjacency tracking.
    fn scan_object_fields(&self, obj: ObjectReference) -> Vec<ObjectReference> {
        self.object_classifier().get_children(obj)
    }

    /// Step 7: Prepare for sweep - deactivate barriers
    fn step_7_prepare_for_sweep(&self) {
        self.set_phase(FugcPhase::PrepareForSweep);
        self.write_barrier().deactivate();

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
        let sweep_stats = self.simd_bitvector().simd_sweep();
        let objects_swept = sweep_stats.objects_swept;

        // Phase 3: Update page states based on SIMD liveness counts
        self.update_page_states_from_bitvector();

        // Cleanup marking state
        self.cache_optimized_marking().reset(); // Reset cache-optimized marking (includes tricolor)
        self.parallel_coordinator().reset();
        self.black_allocator().deactivate();

        self.cycle_stats().rcu(|stats| {
            let mut new_stats = (**stats).clone();
            new_stats.total_sweep_time_ms += sweep_start.elapsed().as_millis() as u64;
            new_stats.objects_swept += objects_swept;
            Arc::new(new_stats)
        });
    }

    /// Build SIMD bitvector from cache-optimized markings - converts marked objects to live bits
    fn build_bitvector_from_markings(&self) {
        self.simd_bitvector().clear();

        // Use cache-optimized marking to build bitvector from marked objects
        // Since CacheOptimizedMarking delegates to tricolor_marking for actual marking,
        // we can use the tricolor marking's color bit array directly for efficiency
        if let Some(tricolor) = self.cache_optimized_marking().tricolor_marking() {
            // Delegate to tricolor marking for efficient iteration over marked objects
            let marked_objects = tricolor.get_black_objects();
            if marked_objects.is_empty() {
                return;
            }

            let mut page_indices = Vec::with_capacity(marked_objects.len().min(1024));

            for obj_ref in marked_objects {
                self.simd_bitvector().mark_object_live(obj_ref);

                if let Some(page_index) =
                    Self::page_index_for_object(self.heap_base(), self.heap_size(), obj_ref)
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
                        .page_states()
                        .entry(current)
                        .or_insert_with(PageState::new);
                    entry.live_objects = entry.live_objects.saturating_add(count);
                    entry.allocation_color = AllocationColor::Black;

                    current = page_index;
                    count = 1;
                }
            }

            let mut entry = self
                .page_states()
                .entry(current)
                .or_insert_with(PageState::new);
            entry.live_objects = entry.live_objects.saturating_add(count);
            entry.allocation_color = AllocationColor::Black;
        }
    }

    /// Update page states based on SIMD bitvector liveness counts using AVX2
    fn update_page_states_from_bitvector(&self) {
        let objects_per_page = PAGE_SIZE / OBJECT_GRANULE;

        for mut item in self.page_states().iter_mut() {
            let (page_index, state) = item.pair_mut();
            // Compute the page's start address from index
            let page_start = unsafe {
                Address::from_usize(self.heap_base().as_usize() + (*page_index) * PAGE_SIZE)
            };
            let live_count = self
                .simd_bitvector()
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

    // Helper method to send phase change notifications
    pub(super) fn set_phase(&self, phase: FugcPhase) {
        self.set_current_phase(phase);
        let _ = self.phase_change_sender().send(phase);
    }
}