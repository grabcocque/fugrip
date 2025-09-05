use crate::core::*;
use crate::collector_phase::CollectorState;
use crate::segmented_heap::COLLECTOR;
use std::sync::atomic::Ordering;

impl CollectorState {
    pub fn suspend_for_fork(&self) {
        let suspend_count = self.suspend_count.fetch_add(1, Ordering::AcqRel);

        if suspend_count == 0 {
            // First suspension - actually suspend the collector
            // TODO: Implement suspension request
            // self.request_suspension();

            // Wait for all collector threads and workers to stop
            // TODO: Implement proper suspension synchronization
            // let guard = self.suspension_lock.lock().unwrap();
            // let _guard = self
            //     .suspended
            //     .wait_while(guard, |&mut suspended| !suspended)
            //     .unwrap();
        }
    }

    pub fn resume_after_fork(&self) {
        let suspend_count = self.suspend_count.fetch_sub(1, Ordering::AcqRel);

        if suspend_count == 1 {
            // Last resume - restart the collector
            // TODO: Implement collection resumption
            // self.resume_collection();
        }
    }
}

// Allocator trait for weak references
pub trait AllocatorTrait {
    fn allocate_classified<T: GcTrace>(&self, value: T, class: crate::object_class::ObjectClass) -> Gc<T>;
}

// Safe fork wrapper
pub fn gc_safe_fork() -> Result<libc::pid_t, std::io::Error> {
    COLLECTOR.suspend_for_fork();

    let result = unsafe { libc::fork() };

    COLLECTOR.resume_after_fork();

    if result == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(result)
    }
}
