use fugrip::align_up;
use fugrip::{ObjectClass, memory::{CLASSIFIED_ALLOCATOR, SegmentedHeap}};

#[test]
fn segmented_heap_basic_ops() {
    let heap = SegmentedHeap::new();
    assert_eq!(heap.segment_count(), 1);
    let new_id = heap.add_segment();
    assert!(heap.segment_count() >= 2);
    assert_eq!(new_id, 1);

    // allocate a few items to exercise allocate path
    let a = heap.allocate(10i32);
    let b = heap.allocate(20i32);
    assert_eq!(*a.read().unwrap(), 10);
    assert_eq!(*b.read().unwrap(), 20);
}

#[test]
fn classified_allocator_and_object_set() {
    // allocate via global classified allocator
    let one = CLASSIFIED_ALLOCATOR.allocate_classified(123i32, ObjectClass::Default);
    assert_eq!(*one.read().unwrap(), 123);

    // object set should report at least one object in default class
    let set = CLASSIFIED_ALLOCATOR.get_object_set(ObjectClass::Default);
    assert!(set.get_object_count() >= 1);

    // iterate_parallel should run without panic
    set.iterate_parallel(2, |_ptr| {});
}

#[test]
fn types_align_up() {
    let ptr = 100usize as *const u8;
    let aligned = align_up(ptr, 8);
    assert_eq!((aligned as usize) % 8, 0);
}
