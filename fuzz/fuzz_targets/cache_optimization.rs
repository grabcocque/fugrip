#![no_main]

use fugrip::cache_optimization::{CacheAwareAllocator, LocalityAwareWorkStealer};
use libfuzzer_sys::fuzz_target;
use mmtk::util::{Address, ObjectReference};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 0x1000000; // 16MB

    // Test CacheAwareAllocator
    if data[0] & 1 == 0 {
        let allocator = CacheAwareAllocator::new(base, heap_size);

        // Parse allocation requests from input data
        let mut data_idx = 1;
        let num_allocations = (data.get(data_idx).unwrap_or(&0) % 32) + 1;
        data_idx += 1;

        for _ in 0..num_allocations {
            if data_idx + 1 >= data.len() {
                break;
            }

            // Generate allocation size (8 bytes to 4KB)
            let size_byte = data[data_idx];
            let size = ((size_byte as usize % 512) + 1) * 8; // 8 bytes to 4KB, 8-byte aligned
            data_idx += 1;

            // Attempt allocation
            if let Some(addr) = allocator.allocate_aligned(size, 8) {
                // Verify allocation is within bounds
                assert!(addr.as_usize() >= base.as_usize());
                assert!(addr.as_usize() + size <= base.as_usize() + heap_size);

                // Verify cache line alignment
                assert_eq!(addr.as_usize() % 64, 0);
            }
        }
    } else {
        // Test LocalityAwareWorkStealer
        let batch_size = ((data[1] as usize % 16) + 1) * 4; // 4 to 64, multiple of 4
        let mut stealer = LocalityAwareWorkStealer::new(batch_size);

        // Create test objects
        let num_objects = (data[2] as usize % 128) + 1;
        let objects: Vec<ObjectReference> = (0..num_objects)
            .map(|i| unsafe {
                let addr = base + (i * 64usize); // 64-byte aligned
                ObjectReference::from_raw_address_unchecked(addr)
            })
            .collect();

        // Add objects based on input pattern
        let mut data_idx = 3;
        for &obj in &objects {
            if data_idx >= data.len() {
                break;
            }

            // Add object if bit is set
            if data[data_idx] & 1 == 1 {
                stealer.add_objects(vec![obj]);
            }
            data_idx += 1;
        }

        // Test work stealing operations
        let num_steals = data.get(data_idx).unwrap_or(&0) % 16;
        for _ in 0..num_steals {
            let batch = stealer.get_next_batch(batch_size);
            // Verify batch size is reasonable
            assert!(batch.len() <= batch_size);
        }
    }
});
