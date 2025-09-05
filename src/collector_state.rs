use crate::core::*;
use crate::collector_phase::{CollectorPhase, CollectorState};
use std::sync::atomic::Ordering;

impl CollectorState {
    pub fn start_marking_phase(&self) {
        self.phase
            .store(CollectorPhase::Marking as usize, Ordering::Release);
        self.marking_active.store(true, Ordering::Release);

        // Soft handshake to stop allocators and switch to black allocation
        // TODO: Implement handshake
        // self.request_handshake();
        self.allocation_color.store(true, Ordering::Release); // Switch to black

        // Start parallel markers
        let worker_count = num_cpus::get();
        for _ in 0..worker_count {
            // TODO: Need Arc for self to spawn threads
            // std::thread::spawn(|| self.marking_worker());
        }

        // Mark roots
        // TODO: Implement root marking
        // self.mark_global_roots();
    }

    pub fn marking_worker(&self) {
        let mut local_stack: Vec<SendPtr<GcHeader<()>>> = Vec::new();

        while self.marking_active.load(Ordering::Acquire) {
            // Try to get work from global stack
            if local_stack.is_empty() {
                // TODO: Implement work stealing
                /*if let Some(work) = self.steal_marking_work() {
                    local_stack.extend(work);
                } else {
                    // Wait for more work or termination
                    std::thread::yield_now();
                    continue;
                }*/
                break; // Temporary
            }

            // Process local work
            while let Some(header_ptr) = local_stack.pop() {
                unsafe {
                    let header = &*header_ptr.as_ptr();
                    if !header.mark_bit.load(Ordering::Acquire) {
                        header.mark_bit.store(true, Ordering::Release);

                        // Trace outgoing pointers using type info
                        (header.type_info.trace_fn)(header_ptr.as_ptr(), &mut local_stack);
                    }
                }

                // Donate work back if stack gets too large
                if local_stack.len() > 1000 {
                    // TODO: Implement work donation
                    // self.donate_marking_work(&mut local_stack);
                }
            }
        }

        // Donate remaining work
        // TODO: Implement work donation
        // self.donate_marking_work(&mut local_stack);
    }
}
