use crate::core::*;
use crate::collector_phase::{CollectorPhase, CollectorState};
use std::sync::atomic::Ordering;

// FinalizableObject is now defined in core.rs

impl CollectorState {
    pub fn reviving_phase(&self) {
        self.phase
            .store(CollectorPhase::Reviving as usize, Ordering::Release);

        // TODO: Get finalizer set from allocator
        // let finalizer_set = &ALLOCATOR.object_sets[ObjectClass::Finalizer as usize];
        let revival_stack = Vec::new();

        // Find unmarked objects with finalizers
        // TODO: Implement parallel iteration
        /*self.iterate_objects_parallel(finalizer_set, |obj_ptr| {
            unsafe {
                let header = &*obj_ptr;
                if !header.mark_bit.load(Ordering::Acquire) {
                    // Object needs finalization - revive it
                    header.mark_bit.store(true, Ordering::Release);
                    revival_stack.push(obj_ptr);

                    // Mark it for finalization
                    if let Some(finalizable) = header.type_info.as_finalizable(obj_ptr) {
                        finalizable.set_needs_finalization();
                    }
                }
            }
        });*/

        // Push revived objects to mark stack for remarking phase
        let revival_stack_wrapped: Vec<SendPtr<GcHeader<()>>> = revival_stack.into_iter()
            .map(|ptr| SendPtr::new(ptr))
            .collect();
        self.global_mark_stack.lock().unwrap().extend(revival_stack_wrapped);
    }

    pub fn remarking_phase(&self) {
        // Mark any objects reachable from revived finalizable objects
        // This runs without handshakes since revived objects are quarantined
        self.phase
            .store(CollectorPhase::Remarking as usize, Ordering::Release);

        while !self.global_mark_stack.lock().unwrap().is_empty() {
            let worker_count = num_cpus::get();
            for _ in 0..worker_count {
                let _state_clone = std::sync::Arc::new(self);
                // TODO: Fix this - need Arc for self
                // std::thread::spawn(move || state_clone.marking_worker());
            }
            // Wait for workers to finish
        }
    }
}
