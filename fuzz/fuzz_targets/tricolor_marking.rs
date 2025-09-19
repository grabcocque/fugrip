#![no_main]

use fugrip::concurrent::ObjectColor;
use fugrip::test_utils::TestFixture;
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Set up tricolor marking using TestFixture
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB
    let fixture = TestFixture::new_with_config(heap_base.as_usize(), heap_size, 2);
    let coordinator = Arc::clone(&fixture.coordinator);
    let marking = Arc::clone(coordinator.tricolor_marking());

    // Create test objects with proper alignment
    let num_objects = (data[0] as usize % 64) + 1; // 1-64 objects
    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Perform operations based on input data
    let mut data_idx = 1;
    let num_operations = (data.get(data_idx).unwrap_or(&0) % 128) + 1;
    data_idx += 1;

    for _ in 0..num_operations {
        if data_idx + 2 >= data.len() {
            break;
        }

        let operation = data[data_idx] % 4;
        let obj_idx = data[data_idx + 1] as usize % objects.len();
        let obj = objects[obj_idx];
        data_idx += 2;

        match operation {
            0 => {
                // Set color
                let color = match data.get(data_idx) {
                    Some(&val) => match val % 3 {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        _ => ObjectColor::Black,
                    },
                    None => ObjectColor::White,
                };
                marking.set_color(obj, color);
                data_idx += 1;
            }
            1 => {
                // Get color
                let _color = marking.get_color(obj);
            }
            2 => {
                // Transition color
                if data_idx + 1 < data.len() {
                    let from_color = match data[data_idx] % 3 {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        _ => ObjectColor::Black,
                    };
                    let to_color = match data[data_idx + 1] % 3 {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        _ => ObjectColor::Black,
                    };
                    let _success = marking.transition_color(obj, from_color, to_color);
                    data_idx += 2;
                }
            }
            3 => {
                // Transition white to grey (shade operation)
                let _success = marking.transition_color(obj, ObjectColor::White, ObjectColor::Grey);
            }
            _ => unreachable!(),
        }
    }
});
