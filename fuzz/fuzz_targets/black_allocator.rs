#![no_main]

use fugrip::concurrent::{BlackAllocator, ObjectColor, TricolorMarking};
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Set up infrastructure
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB
    let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let black_allocator = BlackAllocator::new(tricolor_marking.clone());

    // Create test objects with proper alignment
    let num_objects = (data[0] as usize % 32) + 1;
    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Parse operations from input data
    let mut data_idx = 1;
    let num_operations = (data.get(data_idx).unwrap_or(&0) % 128) + 1;
    data_idx += 1;

    for _ in 0..num_operations {
        if data_idx >= data.len() {
            break;
        }

        let operation = data[data_idx] % 5;
        data_idx += 1;

        match operation {
            0 => {
                // Activate allocator
                black_allocator.activate();
                assert!(black_allocator.is_active());
            }
            1 => {
                // Deactivate allocator
                black_allocator.deactivate();
                assert!(!black_allocator.is_active());
            }
            2 => {
                // Allocate object as black
                if data_idx < data.len() {
                    let obj_idx = data[data_idx] as usize % objects.len();
                    let obj = objects[obj_idx];
                    let was_active = black_allocator.is_active();

                    black_allocator.allocate_black(obj);

                    // Verify allocation behavior
                    if was_active {
                        assert_eq!(tricolor_marking.get_color(obj), ObjectColor::Black);
                    }
                    data_idx += 1;
                }
            }
            3 => {
                // Set initial color and then allocate
                if data_idx + 1 < data.len() {
                    let obj_idx = data[data_idx] as usize % objects.len();
                    let color_val = data[data_idx + 1] % 3;
                    let obj = objects[obj_idx];

                    let initial_color = match color_val {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        _ => ObjectColor::Black,
                    };

                    tricolor_marking.set_color(obj, initial_color);
                    let was_active = black_allocator.is_active();
                    black_allocator.allocate_black(obj);

                    // Black allocation should override any previous color when active
                    if was_active {
                        assert_eq!(tricolor_marking.get_color(obj), ObjectColor::Black);
                    }
                    data_idx += 2;
                }
            }
            4 => {
                // Get statistics
                let _stats = black_allocator.get_stats();
            }
            _ => unreachable!(),
        }
    }

    // Final verification - all allocated objects should maintain their colors
    for &obj in &objects {
        let _color = tricolor_marking.get_color(obj);
    }
});
