use crate::core::*;
use crate::segmented_heap::SegmentedHeap;
use parking_lot::{RwLock, Mutex};

#[derive(Debug, Clone, Copy)]
pub enum ObjectClass {
    Default,
    Destructor,          // Objects with destructors
    Census,              // Objects needing weak reference census
    CensusAndDestructor, // Both census and destructor
    Finalizer,           // Objects with finalizers
    Weak,                // Weak references themselves
}

pub struct ClassifiedAllocator {
    pub heaps: [SegmentedHeap; 6],   // One per ObjectClass
    pub object_sets: [ObjectSet; 6], // For iteration during collection
}

impl ClassifiedAllocator {
    pub fn allocate_classified<T: GcTrace>(&self, value: T, class: ObjectClass) -> Gc<T> {
        let heap_index = class as usize;
        let gc_ptr = self.heaps[heap_index].allocate(value);

        // Register in appropriate object set for collection
        self.object_sets[heap_index].register(gc_ptr.as_ptr() as *mut GcHeader<()>);

        gc_ptr
    }
}

// Object sets for efficient iteration during collection phases
pub struct ObjectSet {
    pub objects: RwLock<Vec<*mut GcHeader<()>>>,
    pub iteration_state: Mutex<IterationState>,
}

pub struct IterationState {
    _current_index: usize,
    _total_size: usize,
    _is_iterating: bool,
}

impl ObjectSet {
    pub fn register(&self, ptr: *mut GcHeader<()>) {
        self.objects.write().push(ptr);
    }
}
