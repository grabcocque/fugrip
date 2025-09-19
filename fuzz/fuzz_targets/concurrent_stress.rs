#![no_main]

use fugrip::concurrent::{ObjectColor, ParallelMarkingCoordinator};
use fugrip::test_utils::TestFixture;
use fugrip::thread::MutatorThread;
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;
use std::time::Duration;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB

    let worker_hint = ((data[0] as usize % 8) + 1).min(4);
    let mutator_count = ((data[1] as usize % 6) + 1).max(1);

    let fixture = TestFixture::new_with_config(heap_base.as_usize(), heap_size, worker_hint);
    let coordinator = Arc::clone(&fixture.coordinator);
    let write_barrier = Arc::clone(coordinator.write_barrier());
    let black_allocator = Arc::clone(coordinator.black_allocator());
    let parallel: Arc<ParallelMarkingCoordinator> = Arc::clone(coordinator.parallel_marking());
    let thread_registry = Arc::clone(fixture.thread_registry());

    // Register synthetic mutator threads so the handshake discipline is active
    let mut mutators = Vec::new();
    for id in 0..mutator_count {
        let mutator = MutatorThread::new(id);
        thread_registry.register(mutator.clone());
        mutators.push(mutator);
    }

    // Create test objects
    let num_objects = (data[2] as usize % 16) + 4; // 4-19 objects
    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize); // 64-byte aligned
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    // Set initial colors
    for (i, &obj) in objects.iter().enumerate() {
        let color_byte = data.get(i + 3).copied().unwrap_or_default();
        let color = match color_byte % 3 {
            0 => ObjectColor::White,
            1 => ObjectColor::Grey,
            _ => ObjectColor::Black,
        };
        coordinator.tricolor_marking().set_color(obj, color);
    }

    let mut data_idx = num_objects + 3;
    let num_stress_ops = (data.get(data_idx).unwrap_or(&0) % 32) + 8; // 8-39 operations
    data_idx += 1;

    for op_idx in 0..num_stress_ops {
        if data_idx + 2 >= data.len() {
            break;
        }

        let stress_scenario = data[data_idx] % 7;
        let obj1_idx = data[data_idx + 1] as usize % objects.len();
        let obj2_idx = data[data_idx + 2] as usize % objects.len();
        data_idx += 3;

        match stress_scenario {
            0 => {
                // Concurrent marking stress: rapid color transitions
                let obj = objects[obj1_idx];
                let current_color = coordinator.tricolor_marking().get_color(obj);
                let next_color = match current_color {
                    ObjectColor::White => ObjectColor::Grey,
                    ObjectColor::Grey => ObjectColor::Black,
                    ObjectColor::Black => ObjectColor::White,
                };
                let _ =
                    coordinator
                        .tricolor_marking()
                        .transition_color(obj, current_color, next_color);
            }
            1 => {
                // Write barrier stress: activate/deactivate rapidly
                if op_idx % 2 == 0 {
                    write_barrier.activate();
                } else {
                    write_barrier.deactivate();
                }

                let mut slot = objects[obj1_idx];
                let new_value = objects[obj2_idx];
                unsafe {
                    write_barrier.write_barrier(&mut slot as *mut ObjectReference, new_value);
                }
            }
            2 => {
                // Black allocator stress: allocation during different states
                match op_idx % 3 {
                    0 => black_allocator.activate(),
                    1 => black_allocator.deactivate(),
                    _ => {}
                }
                black_allocator.allocate_black(objects[obj1_idx]);
            }
            3 => {
                // Work coordination stress: rapid sharing/stealing
                let work_batch = vec![objects[obj1_idx], objects[obj2_idx]];
                parallel.share_work(work_batch);
                let _ = parallel.steal_work(0, 1);
            }
            4 => {
                // Individual write barrier operations with varied colors
                if data_idx < data.len() {
                    let num_writes = (data[data_idx] % 4) + 1; // 1-4 operations
                    data_idx += 1;

                    for i in 0..num_writes {
                        let src_idx = (obj1_idx + i as usize) % objects.len();
                        let dst_idx = (obj2_idx + i as usize) % objects.len();
                        let mut slot = objects[src_idx];
                        unsafe {
                            write_barrier
                                .write_barrier(&mut slot as *mut ObjectReference, objects[dst_idx]);
                        }
                    }
                }
            }
            5 => {
                // Mixed state verification stress
                let obj = objects[obj1_idx];
                black_allocator.allocate_black(obj);
                let color_after = coordinator.tricolor_marking().get_color(obj);
                if black_allocator.is_active() {
                    assert_eq!(color_after, ObjectColor::Black);
                }

                let _ = coordinator.tricolor_marking().transition_color(
                    obj,
                    color_after,
                    ObjectColor::Grey,
                );
            }
            6 => {
                // Exercise the safepoint/handshake discipline
                let timeout_ms = data.get(data_idx).copied().unwrap_or(1) as u64 % 8;
                data_idx = data_idx.saturating_add(1);

                coordinator.activate_barriers_at_safepoint();
                coordinator.black_allocator().activate();
                coordinator.trigger_gc();
                let _ = coordinator.wait_until_idle(Duration::from_millis(timeout_ms + 1));

                for mutator in &mutators {
                    mutator.poll_safepoint();
                }

                let _metrics = coordinator.last_handshake_metrics();
            }
            _ => unreachable!(),
        }
    }

    write_barrier.deactivate();
    black_allocator.deactivate();

    for mutator in &mutators {
        mutator.poll_safepoint();
    }
});
