//! Basic integration tests for concurrent marking infrastructure

use fugrip::concurrent::{
    BlackAllocator, MarkingWorker, ObjectColor, ParallelMarkingCoordinator, TricolorMarking,
    WriteBarrier,
};
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;
use std::thread;

#[test]
fn write_barrier_integration_test() {
    let heap_base = unsafe { Address::from_usize(0x100000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(2));
    let barrier = WriteBarrier::new(&marking, &coordinator, heap_base, 0x100000);

    // Create test objects
    let obj1 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
    let obj2 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x2000usize) };

    // Set initial color
    marking.set_color(obj1, ObjectColor::White);

    // Activate barrier and test
    barrier.activate();

    let mut slot = obj1;
    unsafe { barrier.write_barrier(&mut slot as *mut ObjectReference, obj2) };

    // Verify write barrier shaded the old value
    assert_eq!(marking.get_color(obj1), ObjectColor::Grey);
    assert_eq!(slot, obj2);

    barrier.deactivate();
}

#[test]
fn black_allocator_integration_test() {
    let heap_base = unsafe { Address::from_usize(0x200000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let allocator = BlackAllocator::new(&marking);

    let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };

    // Test without activation
    allocator.allocate_black(obj);
    assert_eq!(
        allocator.tricolor_marking.get_color(obj),
        ObjectColor::White
    );

    // Test with activation
    allocator.activate();
    allocator.allocate_black(obj);
    assert_eq!(
        allocator.tricolor_marking.get_color(obj),
        ObjectColor::Black
    );
    assert_eq!(allocator.get_stats(), 1);

    allocator.deactivate();
}

#[test]
fn tricolor_marking_concurrent_test() {
    let heap_base = unsafe { Address::from_usize(0x300000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));

    let num_threads = 4;
    let objects_per_thread = 25;

    crossbeam::scope(|s| {
        for thread_id in 0..num_threads {
            let marking = Arc::clone(&marking);

            s.spawn(move |_| {
                for i in 0..objects_per_thread {
                    let offset = (thread_id * objects_per_thread + i) * 0x100usize;
                    let obj =
                        unsafe { ObjectReference::from_raw_address_unchecked(heap_base + offset) };

                    // Test color transitions
                    marking.set_color(obj, ObjectColor::White);
                    assert_eq!(marking.get_color(obj), ObjectColor::White);

                    let success =
                        marking.transition_color(obj, ObjectColor::White, ObjectColor::Grey);
                    assert!(success);
                    assert_eq!(marking.get_color(obj), ObjectColor::Grey);

                    let success =
                        marking.transition_color(obj, ObjectColor::Grey, ObjectColor::Black);
                    assert!(success);
                    assert_eq!(marking.get_color(obj), ObjectColor::Black);
                }
            });
        }
    })
    .unwrap();
}

#[test]
fn parallel_marking_work_distribution() {
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(3));
    let heap_base = unsafe { Address::from_usize(0x400000) };

    // Create initial work
    let work: Vec<ObjectReference> = (0..100)
        .map(|i| unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) })
        .collect();

    coordinator.share_work(work);
    assert!(coordinator.has_work());

    let num_workers = 3;
    let results: Vec<usize> = crossbeam::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..num_workers {
            let coordinator = Arc::clone(&coordinator);
            handles.push(s.spawn(move |_| {
                let mut total_work = 0;

                for _ in 0..10 {
                    let stolen = coordinator.steal_work(0, 5);
                    total_work += stolen.len();

                    if !stolen.is_empty() {
                        // Share some work back
                        if stolen.len() > 2 {
                            coordinator.share_work(stolen[stolen.len() / 2..].to_vec());
                        }
                    }
                }

                total_work
            }));
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
    .unwrap();
    let total_processed: usize = results.iter().sum();

    assert!(total_processed > 0);

    let (stolen_count, _shared_count) = coordinator.get_stats();
    assert!(stolen_count > 0);
}

#[test]
fn marking_worker_functionality() {
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
    let mut worker = MarkingWorker::new(0, coordinator, 50);

    // Test basic operations
    assert_eq!(worker.objects_marked(), 0);
    // Note: grey_stack is no longer exposed - Rayon handles work distribution internally

    // Add work (now a no-op since Rayon handles work distribution)
    let heap_base = unsafe { Address::from_usize(0x500000) };
    let objects: Vec<ObjectReference> = (0..5)
        .map(|i| unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) })
        .collect();

    worker.add_initial_work(objects);
    // Note: can't test grey_stack.len() anymore since it's handled by Rayon internally

    // Test reset
    worker.reset();
    assert_eq!(worker.objects_marked(), 0);
}

#[test]
fn write_barrier_bulk_operations() {
    let heap_base = unsafe { Address::from_usize(0x600000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
    let barrier = WriteBarrier::new(&marking, &coordinator, heap_base, 0x100000);

    barrier.activate();

    // Test with a smaller number to avoid memory issues
    let num_updates = 5;
    let mut slots = Vec::with_capacity(num_updates);

    // Initialize slots
    for i in 0..num_updates {
        let old_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) };
        barrier
            .tricolor_marking
            .set_color(old_obj, ObjectColor::White);
        slots.push(old_obj);
    }

    // Create updates array
    let mut updates = Vec::with_capacity(num_updates);
    for i in 0..num_updates {
        let new_obj = unsafe {
            ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize + 0x50usize)
        };
        updates.push((unsafe { slots.as_mut_ptr().add(i) }, new_obj));
    }

    // Perform bulk operation
    barrier.write_barrier_bulk(&updates);

    // Verify all old objects were shaded
    for i in 0..num_updates {
        let old_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) };
        let color = barrier.tricolor_marking.get_color(old_obj);
        assert!(color == ObjectColor::Grey || color == ObjectColor::Black);
    }

    barrier.deactivate();
}
