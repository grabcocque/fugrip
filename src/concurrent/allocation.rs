//! Black allocation manager for concurrent marking

use crate::compat::ObjectReference;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use super::{core::optimized_fetch_add, tricolor::TricolorMarking};
use crate::concurrent::ObjectColor;

/// Black allocation manager for concurrent marking
pub struct BlackAllocator {
    /// Tricolor marking state
    pub tricolor_marking: Arc<TricolorMarking>,
    /// Flag indicating if black allocation is active
    black_allocation_active: AtomicBool,
    /// Statistics
    objects_allocated_black: AtomicUsize,
}

impl BlackAllocator {
    pub fn new(tricolor_marking: &Arc<TricolorMarking>) -> Self {
        Self {
            tricolor_marking: Arc::clone(tricolor_marking),
            black_allocation_active: AtomicBool::new(false),
            objects_allocated_black: AtomicUsize::new(0),
        }
    }

    /// Activate black allocation during marking
    pub fn activate(&self) {
        self.black_allocation_active.store(true, Ordering::SeqCst);
    }

    /// Deactivate black allocation after marking
    pub fn deactivate(&self) {
        self.black_allocation_active.store(false, Ordering::SeqCst);
    }

    /// Check if black allocation is active
    pub fn is_active(&self) -> bool {
        self.black_allocation_active.load(Ordering::SeqCst)
    }

    /// Mark a newly allocated object as black during concurrent marking
    /// This prevents the object from being collected in the current cycle
    pub fn allocate_black(&self, object: ObjectReference) {
        if !self.is_active() {
            // Black allocation not active, object starts white
            return;
        }

        // Mark the newly allocated object as black
        self.tricolor_marking.set_color(object, ObjectColor::Black);
        optimized_fetch_add(&self.objects_allocated_black, 1);
    }

    /// Get statistics for black allocation
    pub fn get_stats(&self) -> usize {
        self.objects_allocated_black.load(Ordering::Relaxed)
    }

    /// Reset for a new marking phase
    pub fn reset(&self) {
        self.black_allocation_active.store(false, Ordering::SeqCst);
        self.objects_allocated_black.store(0, Ordering::Relaxed);
    }
}
