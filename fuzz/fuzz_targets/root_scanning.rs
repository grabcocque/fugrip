#![no_main]

use fugrip::roots::{GlobalRoots, StackRoots};
use libfuzzer_sys::fuzz_target;
use mmtk::util::Address;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let heap_base = unsafe { Address::from_usize(0x10000000) };

    // Test both stack and global root management
    let test_stack_roots = data[0] & 1 == 1;
    let num_operations = (data[1] as usize % 64) + 1;

    if test_stack_roots {
        // Test stack roots
        let mut stack_roots = StackRoots::default();
        let mut data_idx = 2;

        for _ in 0..num_operations {
            if data_idx + 7 >= data.len() {
                break;
            }

            let operation = data[data_idx] % 4;
            data_idx += 1;

            match operation {
                0 => {
                    // Push root (create valid pointer)
                    let offset = u32::from_le_bytes([
                        data[data_idx],
                        data[data_idx + 1],
                        data[data_idx + 2],
                        data[data_idx + 3],
                    ]) as usize
                        % (1024 * 1024); // Within 1MB

                    let root_addr = heap_base + (offset & !7usize); // 8-byte align
                    stack_roots.push(root_addr.to_mut_ptr());
                    data_idx += 4;
                }
                1 => {
                    // Clear to simulate pop behavior
                    if stack_roots.iter().count() > 0 {
                        let all_roots: Vec<_> = stack_roots.iter().collect();
                        stack_roots.clear();
                        // Re-add all but last
                        for &root in &all_roots[..all_roots.len().saturating_sub(1)] {
                            stack_roots.push(root);
                        }
                    }
                }
                2 => {
                    // Clear all roots
                    stack_roots.clear();
                }
                3 => {
                    // Iterate through roots
                    let count = stack_roots.iter().count();
                    assert!(count <= 10000); // Reasonable upper bound
                }
                _ => unreachable!(),
            }
        }

        // Final verification
        let final_count = stack_roots.iter().count();
        assert!(final_count <= 10000);
    } else {
        // Test global roots
        let mut global_roots = GlobalRoots::default();
        let mut data_idx = 2;

        for _ in 0..num_operations {
            if data_idx + 7 >= data.len() {
                break;
            }

            let operation = data[data_idx] % 4;
            data_idx += 1;

            match operation {
                0 => {
                    // Register global root
                    let offset = u32::from_le_bytes([
                        data[data_idx],
                        data[data_idx + 1],
                        data[data_idx + 2],
                        data[data_idx + 3],
                    ]) as usize
                        % (1024 * 1024); // Within 1MB

                    let root_addr = heap_base + (offset & !7usize); // 8-byte align
                    global_roots.register(root_addr.to_mut_ptr());
                    data_idx += 4;
                }
                1 => {
                    // Simulate unregister by clearing and re-adding all but last
                    if global_roots.iter().count() > 0 {
                        let all_roots: Vec<_> = global_roots.iter().collect();
                        // GlobalRoots doesn't have clear method, recreate
                        let _ = &global_roots; // Use the existing one
                        // Re-add all but last
                        for &root in &all_roots[..all_roots.len().saturating_sub(1)] {
                            global_roots.register(root);
                        }
                    }
                }
                2 => {
                    // Clear all globals (create new instance)
                    global_roots = GlobalRoots::default();
                }
                3 => {
                    // Iterate through roots
                    let count = global_roots.iter().count();
                    assert!(count <= 10000); // Reasonable upper bound
                }
                _ => unreachable!(),
            }
        }

        // Final verification
        let final_count = global_roots.iter().count();
        assert!(final_count <= 10000);
    }
});
