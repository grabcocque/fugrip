use fugrip::concurrent::{
    BlackAllocator, ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier,
};
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

#[test]
fn reproduce_concurrent_stress_artifact() {
    // The minimized artifact that caused the crash
    let data: &[u8] = &[33, 98, 98, 98, 10, 120, 120, 124, 0, 0, 0, 0, 0, 0, 0, 0];

    println!("Reproducing artifact with data: {:?}", data);

    // Set up concurrent GC infrastructure (same as fuzz target)
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB
    let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(2));
    let write_barrier = WriteBarrier::new(tricolor_marking.clone(), coordinator.clone());
    let black_allocator = BlackAllocator::new(tricolor_marking.clone());

    // Parse the input data same way as fuzz target
    let num_objects = (data[0] as usize % 16) + 4; // 4-19 objects
    println!("num_objects: {}", num_objects);

    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Set initial colors
    for (i, &obj) in objects.iter().enumerate() {
        let color_byte = data.get(i + 1).unwrap_or(&0);
        let color = match color_byte % 3 {
            0 => ObjectColor::White,
            1 => ObjectColor::Grey,
            _ => ObjectColor::Black,
        };
        println!(
            "Setting object {} to color {:?} (byte: {})",
            i, color, color_byte
        );
        tricolor_marking.set_color(obj, color);
    }

    // Parse stress operations
    let mut data_idx = num_objects + 1;
    let num_stress_ops = (data.get(data_idx).unwrap_or(&0) % 32) + 8; // 8-39 operations
    data_idx += 1;

    println!("num_stress_ops: {}, data_idx: {}", num_stress_ops, data_idx);

    // Simulate concurrent stress scenarios
    for op_idx in 0..num_stress_ops {
        if data_idx + 2 >= data.len() {
            println!("Breaking at op_idx: {}, data_idx: {}", op_idx, data_idx);
            break;
        }

        let stress_scenario = data[data_idx] % 6;
        let obj1_idx = data[data_idx + 1] as usize % objects.len();
        let obj2_idx = *data.get(data_idx + 2).unwrap_or(&0) as usize % objects.len();
        data_idx += 3;

        println!(
            "Op {}: scenario={}, obj1_idx={}, obj2_idx={}",
            op_idx, stress_scenario, obj1_idx, obj2_idx
        );

        match stress_scenario {
            0 => {
                // Concurrent marking stress: rapid color transitions
                let obj = objects[obj1_idx];
                let current_color = tricolor_marking.get_color(obj);
                let next_color = match current_color {
                    ObjectColor::White => ObjectColor::Grey,
                    ObjectColor::Grey => ObjectColor::Black,
                    ObjectColor::Black => ObjectColor::White,
                };
                println!("  Transitioning {:?} -> {:?}", current_color, next_color);
                let _success = tricolor_marking.transition_color(obj, current_color, next_color);
            }
            1 => {
                // Write barrier stress: activate/deactivate rapidly
                println!("  Write barrier operation");
                if op_idx % 2 == 0 {
                    write_barrier.activate();
                    println!("    Activated write barrier");
                } else {
                    write_barrier.deactivate();
                    println!("    Deactivated write barrier");
                }

                // Perform write operation
                let mut slot = objects[obj1_idx];
                let new_value = objects[obj2_idx];
                println!("    Writing slot[{}] = obj[{}]", obj1_idx, obj2_idx);
                unsafe {
                    write_barrier.write_barrier(&mut slot as *mut ObjectReference, new_value);
                }
            }
            2 => {
                // Black allocator stress: allocation during marking
                println!("  Black allocator operation");
                if op_idx % 3 == 0 {
                    black_allocator.activate();
                    println!("    Activated black allocator");
                } else if op_idx % 3 == 1 {
                    black_allocator.deactivate();
                    println!("    Deactivated black allocator");
                }

                println!("    Allocating black: obj[{}]", obj1_idx);
                black_allocator.allocate_black(objects[obj1_idx]);
            }
            3 => {
                // Work coordination stress: rapid sharing/stealing
                println!("  Work coordination");
                let work_batch = vec![objects[obj1_idx], objects[obj2_idx]];
                coordinator.share_work(work_batch);

                let _stolen = coordinator.steal_work(1);
            }
            4 => {
                // Bulk operation stress
                println!("  Bulk operations");
                if data_idx + 3 < data.len() {
                    let bulk_size = (data[data_idx] % 4) + 2; // 2-5 operations
                    let mut bulk_updates = Vec::new();

                    for i in 0..bulk_size {
                        let src_idx = (obj1_idx + i as usize) % objects.len();
                        let dst_idx = (obj2_idx + i as usize) % objects.len();
                        let mut slot = objects[src_idx];
                        bulk_updates.push((&mut slot as *mut ObjectReference, objects[dst_idx]));
                    }

                    write_barrier.write_barrier_bulk(&bulk_updates);
                    data_idx += 1;
                }
            }
            5 => {
                // Mixed state verification stress
                println!("  Mixed state verification");
                let obj = objects[obj1_idx];
                let _color_before = tricolor_marking.get_color(obj);

                // Perform multiple operations in sequence
                black_allocator.allocate_black(obj);
                let color_after_alloc = tricolor_marking.get_color(obj);

                // Verify consistency
                if black_allocator.is_active() {
                    assert_eq!(color_after_alloc, ObjectColor::Black);
                }

                let _transition_success =
                    tricolor_marking.transition_color(obj, color_after_alloc, ObjectColor::Grey);
            }
            _ => unreachable!(),
        }
    }

    // Final consistency checks
    write_barrier.deactivate();
    black_allocator.deactivate();

    // Verify all objects have valid colors
    for &obj in &objects {
        let color = tricolor_marking.get_color(obj);
        assert!(matches!(
            color,
            ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
        ));
    }

    println!("Test completed successfully");
}
