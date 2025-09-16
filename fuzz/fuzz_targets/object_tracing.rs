#![no_main]

use fugrip::core::{Gc, Trace};
use libfuzzer_sys::fuzz_target;
use mmtk::util::Address;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let heap_base = unsafe { Address::from_usize(0x10000000) };

    // Create test objects for tracing
    let num_objects = (data[0] as usize % 16) + 1;
    let mut traced_pointers = Vec::new();

    // Test Gc<T> tracing
    for i in 0..num_objects {
        if i * 8 + 8 > data.len() {
            break;
        }

        // Extract pointer value from input data
        let ptr_bytes = &data[i * 8..(i * 8) + 8];
        let ptr_value = u64::from_le_bytes([
            ptr_bytes[0],
            ptr_bytes[1],
            ptr_bytes[2],
            ptr_bytes[3],
            ptr_bytes[4],
            ptr_bytes[5],
            ptr_bytes[6],
            ptr_bytes[7],
        ]);

        // Create aligned pointer within reasonable bounds
        let offset = (ptr_value as usize % (1024 * 1024)) & !7usize; // 8-byte aligned, within 1MB
        let obj_addr = heap_base + offset;

        // Create Gc handle safely
        let gc_handle = Gc::<u32>::from_raw(obj_addr.to_mut_ptr() as *mut u32);

        // Test tracing
        let mut tracer = |ptr: *mut u8| {
            traced_pointers.push(ptr);
        };

        gc_handle.trace(&mut tracer);

        // Verify trace behavior
        assert_eq!(traced_pointers.len(), i + 1);
        assert_eq!(traced_pointers[i], obj_addr.to_mut_ptr());
    }

    // Test with different trace patterns
    let remaining_data = &data[num_objects * 8..];
    if !remaining_data.is_empty() {
        let trace_pattern = remaining_data[0] % 4;

        match trace_pattern {
            0 => {
                // Test null pointer tracing
                let null_gc = Gc::<u32>::from_raw(std::ptr::null_mut());
                let mut null_traced = Vec::new();
                let mut null_tracer = |ptr: *mut u8| null_traced.push(ptr);
                null_gc.trace(&mut null_tracer);
                assert_eq!(null_traced.len(), 1);
                assert!(null_traced[0].is_null());
            }
            1 => {
                // Test multiple Gc objects tracing in sequence
                traced_pointers.clear();
                let mut batch_tracer = |ptr: *mut u8| traced_pointers.push(ptr);

                for i in 0..3.min(num_objects) {
                    let offset = (i * 128) & !7usize; // Different offsets, 8-byte aligned
                    let addr = heap_base + offset;
                    let gc = Gc::<u64>::from_raw(addr.to_mut_ptr() as *mut u64);
                    gc.trace(&mut batch_tracer);
                }

                assert_eq!(traced_pointers.len(), 3.min(num_objects));
            }
            2 => {
                // Test tracer with side effects
                let mut trace_count = 0usize;
                let mut counting_tracer = |_ptr: *mut u8| {
                    trace_count += 1;
                };

                // Trace several objects
                for i in 0..5.min(num_objects) {
                    let offset = (i * 64) & !7usize;
                    let addr = heap_base + offset;
                    let gc = Gc::<i32>::from_raw(addr.to_mut_ptr() as *mut i32);
                    gc.trace(&mut counting_tracer);
                }

                assert_eq!(trace_count, 5.min(num_objects));
            }
            3 => {
                // Test boundary addresses
                let boundary_addrs = [
                    heap_base,                    // Start of heap
                    heap_base + 4096usize,        // Page boundary
                    heap_base + (64 * 1024usize), // 64KB boundary
                ];

                traced_pointers.clear();
                let mut boundary_tracer = |ptr: *mut u8| traced_pointers.push(ptr);

                for &addr in &boundary_addrs {
                    let gc = Gc::<u8>::from_raw(addr.to_mut_ptr());
                    gc.trace(&mut boundary_tracer);
                }

                assert_eq!(traced_pointers.len(), boundary_addrs.len());
            }
            _ => unreachable!(),
        }
    }

    // Final verification: all traced pointers should be within expected bounds
    for &ptr in &traced_pointers {
        if !ptr.is_null() {
            let addr = ptr as usize;
            assert!(addr >= heap_base.as_usize());
            assert!(addr < heap_base.as_usize() + 2 * 1024 * 1024); // Within 2MB
            assert_eq!(addr % 8, 0); // Should be 8-byte aligned
        }
    }
});
