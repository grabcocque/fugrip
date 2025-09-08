use crate::*;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Coordinates finalizable object lifecycle and reviving phase.
/// 
/// This component is responsible for:
/// - Managing finalizable objects during collection cycles
/// - Implementing the reviving phase for unmarked finalizable objects
/// - Coordinating finalizer execution and object revival
/// - Managing the remarking phase after object revival
pub struct FinalizerCoordinator {
    /// Objects that have been revived and need remarking
    revival_stack: Mutex<Vec<SendPtr<GcHeader<()>>>>,
}

impl FinalizerCoordinator {
    pub fn new() -> Self {
        Self {
            revival_stack: Mutex::new(Vec::new()),
        }
    }

    /// Execute the reviving phase.
    /// 
    /// This phase finds unmarked objects with finalizers and revives them,
    /// giving them a chance to run their finalizers before being collected.
    pub fn execute_reviving_phase(&self, phase_manager: &crate::collector::phase_manager::PhaseManager) {
        phase_manager.set_phase(CollectorPhase::Reviving);

        // Get finalizer set from the classified allocator
        use crate::memory::CLASSIFIED_ALLOCATOR;
        let finalizer_set = CLASSIFIED_ALLOCATOR.get_object_set(ObjectClass::Finalizer);
        
        // Find unmarked objects with finalizers, run their finalize_fn, and revive them
        self.iterate_and_finalize_objects(finalizer_set);
    }

    /// Execute the remarking phase.
    /// 
    /// This phase marks any objects reachable from revived finalizable objects.
    /// This runs without handshakes since revived objects are quarantined.
    pub fn execute_remarking_phase(&self, phase_manager: &crate::collector::phase_manager::PhaseManager) {
        phase_manager.set_phase(CollectorPhase::Remarking);

        // Move revived objects to the global mark stack for processing
        // We'll use the existing global collector infrastructure for remarking
        {
            let mut revival_stack = self.revival_stack.lock().unwrap();
            if !revival_stack.is_empty() {
                use crate::memory::COLLECTOR;
                let mut global_stack = COLLECTOR.global_mark_stack.lock().unwrap();
                global_stack.extend(revival_stack.drain(..));
            }
        }

        // In this refactor stage, we skip spinning explicit remarking workers.
        // Phase is set and revival roots were moved to the global stack; marking
        // will be processed by the normal marking machinery when active.
    }

    /// Iterate over finalizer objects to find unmarked ones, run finalize_fn, and revive.
    fn iterate_and_finalize_objects(&self, finalizer_set: &crate::memory::ObjectSet) {
        let revival_stack_shared = Arc::new(Mutex::new(Vec::<SendPtr<GcHeader<()>>>::new()));
        let worker_count = num_cpus::get();

        // Use the ObjectSet's parallel iteration method
        let stack_clone = revival_stack_shared.clone();
        finalizer_set.iterate_parallel(worker_count, move |obj_ptr| {
            unsafe {
                let header = &*obj_ptr.as_ptr();

                // If object is unmarked (dead), run finalize and then revive it
                if !header.mark_bit.load(Ordering::Acquire) {
                    // Run finalize if provided by type info
                    if let Some(finalize) = header.type_info.finalize_fn {
                        finalize(obj_ptr.as_ptr());
                    }
                    // Revive object by marking it so it can be remarked
                    header.mark_bit.store(true, Ordering::Release);

                    // Add to the shared revival stack using SendPtr for thread safety
                    let mut stack = stack_clone.lock().unwrap();
                    stack.push(obj_ptr);
                }
            }
        });

        // Merge results back to the main revival stack
        let shared_stack = Arc::try_unwrap(revival_stack_shared)
            .unwrap_or_else(|_| panic!("Multiple references to revival_stack_shared"))
            .into_inner()
            .unwrap();
        
        let mut revival_stack = self.revival_stack.lock().unwrap();
        revival_stack.extend(shared_stack);
    }

    /// Get the number of objects currently awaiting remarking
    pub fn get_revival_queue_size(&self) -> usize {
        self.revival_stack.lock().unwrap().len()
    }

    /// Clear the revival stack (used between collection cycles)
    pub fn clear_revival_stack(&self) {
        self.revival_stack.lock().unwrap().clear();
    }

    /// Check if there are any objects awaiting remarking
    pub fn has_revived_objects(&self) -> bool {
        !self.revival_stack.lock().unwrap().is_empty()
    }

    /// Execute the census phase - handling weak reference cleanup
    pub fn execute_census_phase(&self, phase_manager: &crate::collector::phase_manager::PhaseManager) {
        // Set the phase to censusing 
        phase_manager.set_phase(CollectorPhase::Censusing);
        
        // Use the existing weak reference cleanup logic from memory.rs
        // but don't call execute_census_phase since it would set the phase again
        use crate::memory::COLLECTOR;
        let collector = &*COLLECTOR;
        
        // Call the census logic directly without changing phase
        collector.census_weak_references_only();
    }
}

impl Default for FinalizerCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
