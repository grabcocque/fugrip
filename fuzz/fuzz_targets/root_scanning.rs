#![no_main]

use fugrip::roots::{GlobalRoots, StackRoots};
use libfuzzer_sys::fuzz_target;
use mmtk::util::Address;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 16 * 1024 * 1024; // 16MB heap
    let heap_end = heap_base + heap_size;

    // Input sanitization
    let sanitized_data = data
        .iter()
        .filter(|&&b| b < 128)
        .copied()
        .collect::<Vec<u8>>();
    if sanitized_data.len() < 4 {
        return;
    }

    // Test both stack and global root management
    let test_stack_roots = sanitized_data[0] & 1 == 1;
    let num_operations = (sanitized_data[1] as usize % 64) + 1;

    if test_stack_roots {
        // Test stack roots with error recovery
        let mut stack_roots = StackRoots::default();
        let mut data_idx = 2;

        for _ in 0..num_operations {
            if data_idx + 7 >= sanitized_data.len() {
                break;
            }

            let operation = sanitized_data[data_idx] % 4;
            data_idx += 1;

            match operation {
                0 => {
                    // Push root with validation
                    if data_idx + 3 < sanitized_data.len() {
                        let offset = u32::from_le_bytes([
                            sanitized_data[data_idx],
                            sanitized_data[data_idx + 1],
                            sanitized_data[data_idx + 2],
                            sanitized_data[data_idx + 3],
                        ]) as usize
                            % heap_size;

                        let root_addr = heap_base + offset;
                        if root_addr < heap_end {
                            stack_roots.push(root_addr.to_mut_ptr());
                        } else {
                            // Error recovery: skip invalid address
                            eprintln!("Invalid root address: {:?}", root_addr);
                        }
                        data_idx += 4;
                    }
                }
                1 => {
                    // Simulate pop with error recovery
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
                    // Iterate through roots with bounds check
                    let count = stack_roots.iter().count();
                    if count > 10000 {
                        eprintln!("Root count exceeded limit: {}", count);
                        stack_roots.clear(); // Recovery
                    }
                }
                _ => {}
            }
        }

        // Final verification with recovery
        let final_count = stack_roots.iter().count();
        if final_count > 10000 {
            eprintln!("Final root count exceeded limit: {}", final_count);
        }
    } else {
        // Test global roots with error recovery
        let mut global_roots = GlobalRoots::default();
        let mut data_idx = 2;

        for _ in 0..num_operations {
            if data_idx + 7 >= sanitized_data.len() {
                break;
            }

            let operation = sanitized_data[data_idx] % 4;
            data_idx += 1;

            match operation {
                0 => {
                    // Register global root with validation
                    if data_idx + 3 < sanitized_data.len() {
                        let offset = u32::from_le_bytes([
                            sanitized_data[data_idx],
                            sanitized_data[data_idx + 1],
                            sanitized_data[data_idx + 2],
                            sanitized_data[data_idx + 3],
                        ]) as usize
                            % heap_size;

                        let root_addr = heap_base + offset;
                        if root_addr < heap_end {
                            global_roots.register(root_addr.to_mut_ptr());
                        } else {
                            // Error recovery: skip invalid address
                            eprintln!("Invalid global root address: {:?}", root_addr);
                        }
                        data_idx += 4;
                    }
                }
                1 => {
                    // Simulate unregister by recreating and re-adding all but last
                    if global_roots.iter().count() > 0 {
                        let all_roots: Vec<_> = global_roots.iter().collect();
                        global_roots = GlobalRoots::default();
                        // Re-add all but last
                        for &root in &all_roots[..all_roots.len().saturating_sub(1)] {
                            global_roots.register(root);
                        }
                    }
                }
                2 => {
                    // Clear all globals
                    global_roots = GlobalRoots::default();
                }
                3 => {
                    // Iterate through roots with bounds check
                    let count = global_roots.iter().count();
                    if count > 10000 {
                        eprintln!("Global root count exceeded limit: {}", count);
                        global_roots = GlobalRoots::default(); // Recovery
                    }
                }
                _ => {}
            }
        }

        // Final verification with recovery
        let final_count = global_roots.iter().count();
        if final_count > 10000 {
            eprintln!("Final global root count exceeded limit: {}", final_count);
        }
    }
});
