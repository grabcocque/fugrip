//! Integration tests for concurrent marking infrastructure

use fugrip::concurrent::{
    WriteBarrier, BlackAllocator, TricolorMarking,
    ParallelMarkingCoordinator, ConcurrentMarkingCoordinator, ObjectColor,
};
use mmtk::util::{ObjectReference, Address};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn concurrent_marking_full_workflow() {
    let heap_base = unsafe { Address::from_usize(0x100000) };
    let heap_size = 0x100000;

    // Create required dependencies
    let thread_registry = std::sync::Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = std::sync::Arc::new(fugrip::roots::GlobalRoots::default());

    let mut coordinator = ConcurrentMarkingCoordinator::new(
        heap_base,
        heap_size,
        2,
        thread_registry,
        global_roots,
    );

    // Create a realistic object graph
    let root1 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
    let root2 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x2000usize) };
    let child1 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x3000usize) };
    let child2 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x4000usize) };
    let grandchild = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x5000usize) };

    // Start concurrent marking
    coordinator.start_marking(vec![root1, root2]);

    // Verify that write barrier and black allocator are active
    assert!(coordinator.write_barrier().is_active());
    assert!(coordinator.black_allocator().is_active());

    // Simulate mutator activity with write barrier operations
    let barrier = coordinator.write_barrier();

    // Create slots for testing write barrier
    let mut slot1 = child1;
    let mut slot2 = child2;

    // Set initial colors for testing
    let tricolor = &coordinator.tricolor_marking;
    tricolor.set_color(child1, ObjectColor::White);
    tricolor.set_color(child2, ObjectColor::White);
    tricolor.set_color(grandchild, ObjectColor::White);

    // Perform write barrier operations (should shade old values)
    barrier.write_barrier(&mut slot1 as *mut ObjectReference, grandchild);
    barrier.write_barrier(&mut slot2 as *mut ObjectReference, ObjectReference::from_raw_address(Address::ZERO).unwrap_or(root1));

    // Verify write barrier shaded the old values
    assert_eq!(tricolor.get_color(child1), ObjectColor::Grey);
    assert_eq!(tricolor.get_color(child2), ObjectColor::Grey);

    // Test black allocation during marking
    let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x6000usize) };
    coordinator.black_allocator().allocate_black(new_obj);
    assert_eq!(tricolor.get_color(new_obj), ObjectColor::Black);

    // Allow some processing time
    thread::sleep(Duration::from_millis(50));

    // Get statistics
    let stats = coordinator.get_stats();
    assert!(stats.objects_allocated_black >= 1);

    // Stop marking
    coordinator.stop_marking();
    assert!(!coordinator.write_barrier().is_active());
    assert!(!coordinator.black_allocator().is_active());
}

#[test]
fn write_barrier_concurrent_stress_test() {
    let heap_base = unsafe { Address::from_usize(0x200000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(4));
    let barrier = Arc::new(WriteBarrier::new(Arc::clone(&marking), coordinator));

    barrier.activate();

    let num_threads = 4;
    let operations_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let barrier = Arc::clone(&barrier);
            let marking = Arc::clone(&marking);

            thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let offset = (thread_id * operations_per_thread + i) * 0x100usize;
                    let old_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + offset) };
                    let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + offset + 0x50usize) };

                    // Set old object to white
                    marking.set_color(old_obj, ObjectColor::White);

                    // Create a slot and perform write barrier
                    let mut slot = old_obj;
                    barrier.write_barrier(&mut slot as *mut ObjectReference, new_obj);

                    // Verify the write barrier worked
                    assert_eq!(slot, new_obj);
                    // Old object should be shaded to grey
                    let color = marking.get_color(old_obj);
                    assert!(color == ObjectColor::Grey || color == ObjectColor::Black);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    barrier.deactivate();
}

#[test]
fn black_allocator_concurrent_stress_test() {
    let heap_base = unsafe { Address::from_usize(0x300000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let allocator = Arc::new(BlackAllocator::new(marking));

    allocator.activate();

    let num_threads = 3;
    let allocations_per_thread = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let allocator = Arc::clone(&allocator);

            thread::spawn(move || {
                for i in 0..allocations_per_thread {
                    let offset = (thread_id * allocations_per_thread + i) * 0x100usize;
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + offset) };

                    allocator.allocate_black(obj);

                    // Verify the object was marked black
                    assert_eq!(allocator.tricolor_marking.get_color(obj), ObjectColor::Black);
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify total allocation count
    let expected_total = num_threads * allocations_per_thread;
    assert_eq!(allocator.get_stats(), expected_total);

    allocator.deactivate();
}

#[test]
fn tricolor_marking_atomic_operations() {
    let heap_base = unsafe { Address::from_usize(0x400000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));

    let num_threads = 6;
    let objects_per_thread = 30;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let marking = Arc::clone(&marking);

            thread::spawn(move || {
                for i in 0..objects_per_thread {
                    let offset = (thread_id * objects_per_thread + i) * 0x100usize;
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + offset) };

                    // Test atomic color transitions
                    marking.set_color(obj, ObjectColor::White);
                    assert_eq!(marking.get_color(obj), ObjectColor::White);

                    // Transition to grey
                    let success = marking.transition_color(obj, ObjectColor::White, ObjectColor::Grey);
                    assert!(success);
                    assert_eq!(marking.get_color(obj), ObjectColor::Grey);

                    // Transition to black
                    let success = marking.transition_color(obj, ObjectColor::Grey, ObjectColor::Black);
                    assert!(success);
                    assert_eq!(marking.get_color(obj), ObjectColor::Black);

                    // Invalid transition should fail
                    let success = marking.transition_color(obj, ObjectColor::White, ObjectColor::Black);
                    assert!(!success);
                    assert_eq!(marking.get_color(obj), ObjectColor::Black); // Should remain black
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn parallel_marking_work_stealing_simulation() {
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(3));
    let heap_base = unsafe { Address::from_usize(0x500000) };

    // Create a large amount of work
    let initial_work: Vec<ObjectReference> = (0..300)
        .map(|i| unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) })
        .collect();

    coordinator.share_work(initial_work);
    assert!(coordinator.has_work());

    let num_workers = 3;
    let handles: Vec<_> = (0..num_workers)
        .map(|_worker_id| {
            let coordinator = Arc::clone(&coordinator);

            thread::spawn(move || {
                let mut total_stolen = 0;

                // Each worker tries to steal work multiple times
                for _ in 0..20 {
                    let stolen = coordinator.steal_work(10);
                    total_stolen += stolen.len();

                    if !stolen.is_empty() {
                        // Simulate processing some work
                        thread::sleep(Duration::from_millis(1));

                        // Share some work back if we got a lot
                        if stolen.len() > 5 {
                            let to_share = stolen[stolen.len()/2..].to_vec();
                            coordinator.share_work(to_share);
                        }
                    }

                    thread::sleep(Duration::from_millis(2));
                }

                total_stolen
            })
        })
        .collect();

    let results: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let total_processed: usize = results.iter().sum();

    // Verify that work was distributed and processed
    assert!(total_processed > 0);
    println!("Total work items processed: {}", total_processed);

    let (stolen_count, shared_count) = coordinator.get_stats();
    assert!(stolen_count > 0);
    assert!(shared_count > 0);
    println!("Work stealing events: {}, sharing events: {}", stolen_count, shared_count);
}

#[test]
fn write_barrier_bulk_operations_performance() {
    let heap_base = unsafe { Address::from_usize(0x600000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
    let barrier = WriteBarrier::new(Arc::clone(&marking), coordinator);

    barrier.activate();

    // Create a large number of updates for bulk operation
    let num_updates = 1000;
    let mut objects = Vec::new();
    let mut slots = Vec::with_capacity(num_updates);
    let mut updates = Vec::with_capacity(num_updates);

    // Initialize slots first
    for i in 0..num_updates {
        let old_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) };
        marking.set_color(old_obj, ObjectColor::White);
        slots.push(old_obj);
        objects.push(old_obj);
    }

    // Create updates array after slots is fully populated
    for i in 0..num_updates {
        let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize + 0x50usize) };
        updates.push((
            unsafe { slots.as_mut_ptr().add(i) } as *mut ObjectReference,
            new_obj,
        ));
    }

    // Perform bulk write barrier operation
    let start = std::time::Instant::now();
    barrier.write_barrier_bulk(&updates);
    let duration = start.elapsed();

    println!("Bulk write barrier for {} updates took: {:?}", num_updates, duration);

    // Verify all old objects were shaded
    for (i, &old_obj) in objects.iter().enumerate() {
        let color = marking.get_color(old_obj);
        assert!(
            color == ObjectColor::Grey || color == ObjectColor::Black,
            "Object {} was not shaded, color: {:?}",
            i,
            color
        );
    }

    barrier.deactivate();
}

#[test]
fn concurrent_marking_termination_detection() {
    let heap_base = unsafe { Address::from_usize(0x700000) };

    // Create required dependencies
    let thread_registry = std::sync::Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = std::sync::Arc::new(fugrip::roots::GlobalRoots::default());

    let mut coordinator = ConcurrentMarkingCoordinator::new(
        heap_base,
        0x100000,
        2,
        thread_registry,
        global_roots,
    );

    // Start with minimal work to ensure quick termination
    let root = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
    coordinator.start_marking(vec![root]);

    // Allow some processing time
    thread::sleep(Duration::from_millis(100));

    // Stop marking - should terminate cleanly
    let start = std::time::Instant::now();
    coordinator.stop_marking();
    let stop_duration = start.elapsed();

    println!("Concurrent marking termination took: {:?}", stop_duration);

    // Should terminate reasonably quickly (less than 1 second)
    assert!(stop_duration < Duration::from_secs(1));

    // Verify systems are deactivated
    assert!(!coordinator.write_barrier().is_active());
    assert!(!coordinator.black_allocator().is_active());
}



#[test]
fn write_barrier_array_operations() {
    let heap_base = unsafe { Address::from_usize(0x900000) };
    let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
    let barrier = WriteBarrier::new(marking, coordinator);

    barrier.activate();

    // Create an array of object references
    let array_size = 20;
    let mut object_array: Vec<ObjectReference> = (0..array_size)
        .map(|i| unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) })
        .collect();

    // Set all objects to white initially
    for &obj in &object_array {
        barrier.tricolor_marking.set_color(obj, ObjectColor::White);
    }

    // Update array elements using array write barrier
    for i in 0..array_size {
        let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + (i + array_size) * 0x100usize) };

        barrier.array_write_barrier(
            object_array.as_mut_ptr() as *mut u8,
            i,
            std::mem::size_of::<ObjectReference>(),
            new_obj,
        );
    }

    // Verify all original objects were shaded
    for i in 0..array_size {
        let original_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 0x100usize) };
        let color = barrier.tricolor_marking.get_color(original_obj);
        assert!(
            color == ObjectColor::Grey || color == ObjectColor::Black,
            "Array element {} was not shaded",
            i
        );
    }

    barrier.deactivate();
}