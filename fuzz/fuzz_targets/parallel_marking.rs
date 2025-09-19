#![no_main]

use fugrip::concurrent::ParallelMarkingCoordinator;
use fugrip::test_utils::TestFixture;
use fugrip::thread::MutatorThread;
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB

    let worker_count = ((data[0] as usize % 8) + 1).min(4);
    let mutator_count = ((data[1] as usize % 6) + 1).max(1);

    let fixture = TestFixture::new_with_config(heap_base.as_usize(), heap_size, worker_count);
    let parallel: Arc<ParallelMarkingCoordinator> =
        Arc::clone(fixture.coordinator.parallel_marking());
    let thread_registry = Arc::clone(fixture.thread_registry());

    for id in 0..mutator_count {
        let mutator = MutatorThread::new(id);
        thread_registry.register(mutator);
    }

    // Create test objects
    let num_objects = (data[2] as usize % 128) + 1;
    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Parse operations from input data
    let mut data_idx = 3;
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
                // Share work batches drawn from fuzz data
                if data_idx < data.len() {
                    let work_size = (data[data_idx] as usize % 32) + 1;
                    data_idx += 1;

                    let work: Vec<ObjectReference> = objects
                        .iter()
                        .cycle()
                        .take(work_size.min(objects.len()))
                        .copied()
                        .collect();

                    parallel.share_work(work);
                }
            }
            1 => {
                // Steal work with varying batch sizes
                if data_idx < data.len() {
                    let steal_size = (data[data_idx] as usize % 16) + 1;
                    data_idx += 1;
                    let _stolen = parallel.steal_work(0, steal_size);
                }
            }
            2 => {
                // Observe the work availability surface
                let _has_work = parallel.has_work();
            }
            _ => unreachable!(),
        }
    }

    // Capture final statistics for sanity checking in the harness
    let _stats = parallel.get_stats();
});
