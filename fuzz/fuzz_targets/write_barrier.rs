#![no_main]

use fugrip::test_utils::TestFixture;
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB

    let fixture = TestFixture::new_with_config(heap_base.as_usize(), heap_size, 2);
    let coordinator = Arc::clone(&fixture.coordinator);
    let barrier = Arc::clone(coordinator.write_barrier());

    let num_operations = (data[0] as usize % 32) + 1; // 1-32 operations
    let should_activate = data[1] & 1 == 1;

    if should_activate {
        barrier.activate();
    } else {
        barrier.deactivate();
    }

    // Create some test objects with proper alignment
    let objects: Vec<ObjectReference> = (0..16)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Set initial colors through the coordinator so state stays consistent
    for (i, &obj) in objects.iter().enumerate() {
        let color = match data.get(i + 2) {
            Some(&val) => match val % 3 {
                0 => fugrip::concurrent::ObjectColor::White,
                1 => fugrip::concurrent::ObjectColor::Grey,
                _ => fugrip::concurrent::ObjectColor::Black,
            },
            None => fugrip::concurrent::ObjectColor::White,
        };
        coordinator.tricolor_marking().set_color(obj, color);
    }

    let mut data_idx = 18;
    for _ in 0..num_operations {
        if data_idx + 2 >= data.len() {
            break;
        }

        let slot_idx = data[data_idx] as usize % objects.len();
        let value_idx = data[data_idx + 1] as usize % objects.len();
        data_idx += 2;

        let mut slot = objects[slot_idx];
        let new_value = objects[value_idx];

        unsafe {
            barrier.write_barrier(&mut slot as *mut ObjectReference, new_value);
        }
    }

    barrier.deactivate();
});
