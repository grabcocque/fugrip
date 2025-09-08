use fugrip::interfaces::*;
use fugrip::{Gc, GcHeader, GcTrace, ObjectClass, SendPtr};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TestData(i32);

unsafe impl GcTrace for TestData {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

struct MockAllocator {
    bytes_allocated: AtomicUsize,
    object_count: AtomicUsize,
}

impl MockAllocator {
    fn new() -> Self {
        Self {
            bytes_allocated: AtomicUsize::new(0),
            object_count: AtomicUsize::new(0),
        }
    }
}

impl AllocatorTrait for MockAllocator {
    fn allocate_classified<T: GcTrace + 'static>(&self, value: T, _class: ObjectClass) -> Gc<T> {
        self.bytes_allocated
            .fetch_add(std::mem::size_of::<T>(), Ordering::SeqCst);
        self.object_count.fetch_add(1, Ordering::SeqCst);
        unsafe { std::mem::transmute(Box::into_raw(Box::new(value))) }
    }

    fn bytes_allocated(&self) -> usize {
        self.bytes_allocated.load(Ordering::SeqCst)
    }

    fn object_count(&self) -> usize {
        self.object_count.load(Ordering::SeqCst)
    }
}

#[test]
fn test_allocator_trait_minimal() {
    let allocator = MockAllocator::new();
    let _ = allocator.allocate_classified(TestData(1), ObjectClass::Default);
    assert_eq!(allocator.object_count(), 1);
    assert!(allocator.bytes_allocated() >= std::mem::size_of::<TestData>());
}

#[test]
fn test_heap_provider_access() {
    // Access the production heap provider and verify basic property
    let heap = HEAP_PROVIDER.get_heap();
    assert!(heap.segment_count() >= 1);
}

#[test]
fn test_threading_provider_hooks() {
    // Exercise the threading provider API without panicking
    THREADING_PROVIDER.register_mutator_thread();
    let count = THREADING_PROVIDER.get_active_mutator_count();
    assert!(count >= 1);
    THREADING_PROVIDER.unregister_mutator_thread();
}

