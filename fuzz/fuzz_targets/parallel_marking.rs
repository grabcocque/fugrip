#![no_main]

use fugrip::concurrent::ParallelMarkingCoordinator;
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let num_workers = ((data[0] as usize % 8) + 1).min(4); // 1-4 workers
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(num_workers));

    // Create test objects
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let num_objects = (data[1] as usize % 128) + 1;
    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Parse operations from input data
    let mut data_idx = 2;
    let num_operations = (data.get(data_idx).unwrap_or(&0) % 64) + 1;
    data_idx += 1;

    for _ in 0..num_operations {
        if data_idx >= data.len() {
            break;
        }

        let operation = data[data_idx] % 3;
        data_idx += 1;

        match operation {
            0 => {
                // Share work
                if data_idx < data.len() {
                    let work_size = (data[data_idx] as usize % 32) + 1;
                    data_idx += 1;

                    let work: Vec<ObjectReference> = objects
                        .iter()
                        .take(work_size.min(objects.len()))
                        .copied()
                        .collect();

                    coordinator.share_work(work);
                }
            }
            1 => {
                // Steal work
                if data_idx < data.len() {
                    let steal_size = (data[data_idx] as usize % 16) + 1;
                    data_idx += 1;

                    let _stolen = coordinator.steal_work(steal_size);
                }
            }
            2 => {
                // Check if work is available
                let _has_work = coordinator.has_work();
            }
            _ => unreachable!(),
        }
    }

    // Get final statistics
    let _stats = coordinator.get_stats();
});
