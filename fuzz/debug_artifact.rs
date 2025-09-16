use fugrip::concurrent::{
    BlackAllocator, ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier,
};
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

fn main() {
    // The minimized artifact that caused the crash
    let data: &[u8] = &[33, 98, 98, 98, 10, 120, 120, 124, 0, 0, 0, 0, 0, 0, 0, 0];

    println!("Reproducing artifact with data: {:?}", data);
    println!("Data analysis:");
    println!(
        "  data[0] = {} -> num_objects = {}",
        data[0],
        (data[0] as usize % 16) + 4
    );
    for (i, &byte) in data.iter().enumerate().take(10) {
        println!("  data[{}] = {} (0x{:02x})", i, byte, byte);
    }

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
    println!("After objects setup, data_idx = {}", data_idx);

    if data_idx >= data.len() {
        println!("ERROR: data_idx {} >= data.len() {}", data_idx, data.len());
        return;
    }

    let num_stress_ops = (data.get(data_idx).unwrap_or(&0) % 32) + 8; // 8-39 operations
    data_idx += 1;

    println!("num_stress_ops: {}, data_idx: {}", num_stress_ops, data_idx);

    // Simulate stress operations step by step with detailed logging
    for op_idx in 0..num_stress_ops {
        println!("\n=== Operation {} ===", op_idx);
        if data_idx + 2 >= data.len() {
            println!(
                "Breaking at op_idx: {}, data_idx: {} (would exceed data.len() = {})",
                op_idx,
                data_idx,
                data.len()
            );
            break;
        }

        let stress_scenario = data[data_idx] % 6;
        let obj1_idx = data[data_idx + 1] as usize % objects.len();
        let obj2_idx = *data.get(data_idx + 2).unwrap_or(&0) as usize % objects.len();

        println!(
            "Raw bytes: [{}, {}, {}]",
            data[data_idx],
            data[data_idx + 1],
            data.get(data_idx + 2).unwrap_or(&0)
        );
        println!(
            "Parsed: scenario={}, obj1_idx={}, obj2_idx={}",
            stress_scenario, obj1_idx, obj2_idx
        );

        data_idx += 3;

        match stress_scenario {
            0 => {
                println!("  [SCENARIO 0] Concurrent marking stress");
                let obj = objects[obj1_idx];
                let current_color = tricolor_marking.get_color(obj);
                let next_color = match current_color {
                    ObjectColor::White => ObjectColor::Grey,
                    ObjectColor::Grey => ObjectColor::Black,
                    ObjectColor::Black => ObjectColor::White,
                };
                println!(
                    "    Transitioning obj[{}]: {:?} -> {:?}",
                    obj1_idx, current_color, next_color
                );
                let success = tricolor_marking.transition_color(obj, current_color, next_color);
                println!("    Transition success: {}", success);
            }
            1 => {
                println!("  [SCENARIO 1] Write barrier stress");
                if op_idx % 2 == 0 {
                    write_barrier.activate();
                    println!("    Activated write barrier");
                } else {
                    write_barrier.deactivate();
                    println!("    Deactivated write barrier");
                }

                let mut slot = objects[obj1_idx];
                let new_value = objects[obj2_idx];
                println!("    Writing: slot[{}] = obj[{}]", obj1_idx, obj2_idx);
                println!("    Barrier active: {}", write_barrier.is_active());

                unsafe {
                    write_barrier.write_barrier(&mut slot as *mut ObjectReference, new_value);
                }
                println!("    Write completed successfully");
            }
            2 => {
                println!("  [SCENARIO 2] Black allocator stress");
                if op_idx % 3 == 0 {
                    black_allocator.activate();
                    println!("    Activated black allocator");
                } else if op_idx % 3 == 1 {
                    black_allocator.deactivate();
                    println!("    Deactivated black allocator");
                }

                let obj = objects[obj1_idx];
                let color_before = tricolor_marking.get_color(obj);
                println!(
                    "    Allocating black: obj[{}] (was {:?})",
                    obj1_idx, color_before
                );
                println!("    Allocator active: {}", black_allocator.is_active());

                black_allocator.allocate_black(obj);

                let color_after = tricolor_marking.get_color(obj);
                println!("    Color after allocation: {:?}", color_after);
            }
            3 => {
                println!("  [SCENARIO 3] Work coordination stress");
                let work_batch = vec![objects[obj1_idx], objects[obj2_idx]];
                println!("    Sharing work batch with {} objects", work_batch.len());
                coordinator.share_work(work_batch);

                let stolen = coordinator.steal_work(1);
                println!("    Stolen {} objects", stolen.len());
            }
            4 => {
                println!("  [SCENARIO 4] Bulk operation stress");
                if data_idx < data.len() {
                    let bulk_size = (data[data_idx] % 4) + 2;
                    println!("    Bulk size: {}", bulk_size);
                    let mut bulk_updates = Vec::new();

                    for i in 0..bulk_size {
                        let src_idx = (obj1_idx + i as usize) % objects.len();
                        let dst_idx = (obj2_idx + i as usize) % objects.len();
                        let mut slot = objects[src_idx];
                        bulk_updates.push((&mut slot as *mut ObjectReference, objects[dst_idx]));
                    }

                    println!("    Performing {} bulk updates", bulk_updates.len());
                    write_barrier.write_barrier_bulk(&bulk_updates);
                    data_idx += 1;
                    println!("    Bulk operations completed");
                } else {
                    println!("    Skipping bulk ops (no data)");
                }
            }
            5 => {
                println!("  [SCENARIO 5] Mixed state verification stress");
                let obj = objects[obj1_idx];
                let color_before = tricolor_marking.get_color(obj);
                println!("    Object color before: {:?}", color_before);

                black_allocator.allocate_black(obj);
                let color_after_alloc = tricolor_marking.get_color(obj);
                println!("    Color after black allocation: {:?}", color_after_alloc);

                if black_allocator.is_active() {
                    if color_after_alloc != ObjectColor::Black {
                        println!(
                            "    ERROR: Expected black color but got {:?}",
                            color_after_alloc
                        );
                        panic!("Color assertion failed!");
                    }
                }

                let transition_success =
                    tricolor_marking.transition_color(obj, color_after_alloc, ObjectColor::Grey);
                println!("    Transition to grey success: {}", transition_success);
            }
            _ => unreachable!(),
        }
    }

    println!("\n=== Final cleanup ===");
    write_barrier.deactivate();
    black_allocator.deactivate();

    // Verify all objects have valid colors
    for (i, &obj) in objects.iter().enumerate() {
        let color = tricolor_marking.get_color(obj);
        println!("Final color of obj[{}]: {:?}", i, color);
        assert!(matches!(
            color,
            ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
        ));
    }

    println!("Test completed successfully");
}
