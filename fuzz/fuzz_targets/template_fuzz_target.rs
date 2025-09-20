// Template fuzz target for `fugrip` to be used with `cargo-fuzz`.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Real MMTk integration: Fuzz testing for FUGC garbage collection operations
    // This exercises the core FUGC 8-step protocol with fuzz-generated inputs
    use fugrip::test_utils::TestFixture;
    use mmtk::util::{Address, ObjectReference};
    use std::sync::Arc;

    // Use data length to determine test scenario
    let test_scenario = data.len() % 8;

    // Create a test fixture with MMTk integration
    let fixture = TestFixture::new_with_config(
        0x10000000, // 256MB heap base
        64 * 1024 * 1024, // 64MB heap size
        4, // 4 workers
    );

    match test_scenario {
        0 => {
            // Fuzz test: Object allocation and marking
            if data.len() >= 8 {
                let obj_addr = unsafe { Address::from_usize(0x10000000 + u64::from_be_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize) };
                if let Some(obj) = ObjectReference::from_raw_address(obj_addr) {
                    // Test object classification and marking
                    let classification = fixture.coordinator.object_classifier().get_classification(obj);
                    fixture.coordinator.tricolor_marking().set_color(obj, classification.into());
                }
            }
        }
        1 => {
            // Fuzz test: Write barrier operations
            if data.len() >= 16 {
                let src_addr = unsafe { Address::from_usize(0x10000000 + u64::from_be_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize) };
                let dst_addr = unsafe { Address::from_usize(0x10000000 + u64::from_be_bytes(data[8..16].try_into().unwrap_or([0; 8])) as usize) };

                if let (Some(src_obj), Some(dst_obj)) = (ObjectReference::from_raw_address(src_addr), ObjectReference::from_raw_address(dst_addr)) {
                    // Test write barrier activation
                    let write_barrier = fixture.coordinator.write_barrier();
                    write_barrier.activate();

                    // Test generational write barrier
                    fixture.coordinator.generational_write_barrier(src_obj, dst_obj);

                    write_barrier.deactivate();
                }
            }
        }
        2 => {
            // Fuzz test: Root scanning operations
            if data.len() >= 4 {
                let root_count = u32::from_be_bytes(data[0..4].try_into().unwrap_or([0; 4])) as usize % 100;

                // Test root registration and scanning
                for i in 0..root_count {
                    let root_addr = unsafe { Address::from_usize(0x10000000 + i * 128) };
                    fixture.global_roots.register(root_addr.to_mut_ptr::<u8>());
                }

                // Perform root scanning
                fixture.coordinator.scan_thread_roots_at_safepoint();
            }
        }
        3 => {
            // Fuzz test: Stack scanning via handshake protocol
            if data.len() >= 8 {
                let thread_id = u64::from_be_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize % 16;

                // Test handshake protocol for stack scanning (FUGC Step 5)
                if let Some(mutator) = fixture.thread_registry.get(thread_id) {
                    mutator.handler.poll_safepoint();
                }
            }
        }
        4 => {
            // Fuzz test: Parallel marking coordination
            if data.len() >= 4 {
                let object_count = u32::from_be_bytes(data[0..4].try_into().unwrap_or([0; 4])) as usize % 50;

                // Create test objects for parallel marking
                let mut objects = Vec::new();
                for i in 0..object_count {
                    let addr = unsafe { Address::from_usize(0x10000000 + i * 256) };
                    if let Some(obj) = ObjectReference::from_raw_address(addr) {
                        objects.push(obj);
                    }
                }

                // Test parallel marking (FUGC Step 6)
                fixture.coordinator.parallel_coordinator().parallel_mark(objects);
            }
        }
        5 => {
            // Fuzz test: Cache optimization operations
            if data.len() >= 8 {
                let batch_size = u64::from_be_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize % 25;
                let prefetch_distance = u64::from_be_bytes(data[8..16].try_into().unwrap_or([0; 8])) as usize % 10;

                // Test cache-optimized allocation
                let allocator = fugrip::cache_optimization::CacheAwareAllocator::new(
                    unsafe { Address::from_usize(0x20000000) },
                    1024 * 1024, // 1MB
                );

                for i in 0..batch_size {
                    let size = 64 + (i * prefetch_distance) % 256;
                    let _ = allocator.allocate_aligned(size, 8);
                }
            }
        }
        6 => {
            // Fuzz test: SIMD sweeping operations
            if data.len() >= 4 {
                let pattern = u32::from_be_bytes(data[0..4].try_into().unwrap_or([0; 4]));

                // Test SIMD bitvector operations
                use fugrip::simd_sweep::SimdBitvector;
                let bitvector = SimdBitvector::new(
                    unsafe { Address::from_usize(0x30000000) },
                    1024 * 1024, // 1MB
                    16, // 16-byte alignment
                );

                // Create marking pattern based on fuzz input
                for i in 0..100 {
                    let addr = unsafe { Address::from_usize(0x30000000 + i * 64) };
                    if (pattern + i as u32) % 3 != 0 {
                        let _ = bitvector.mark_live(addr);
                    }
                }

                // Test hybrid sweep
                let _stats = bitvector.hybrid_sweep();
            }
        }
        _ => {
            // Fuzz test: Memory management operations
            if data.len() >= 8 {
                let operation = u64::from_be_bytes(data[0..8].try_into().unwrap_or([0; 8])) % 3;

                match operation {
                    0 => {
                        // Test finalization
                        if let Some(obj) = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x40000000) }) {
                            fixture.memory_manager().register_finalizer(obj, Box::new(|| {}));
                        }
                    }
                    1 => {
                        // Test weak references
                        if let Some(obj) = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x40000100) }) {
                            let _weak_ref = fixture.memory_manager().create_weak_reference(obj);
                        }
                    }
                    2 => {
                        // Test object freeing
                        if let Some(obj) = ObjectReference::from_raw_address(unsafe { Address::from_usize(0x40000200) }) {
                            fixture.memory_manager().free_object(obj);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Always perform cleanup to ensure test isolation
    fixture.coordinator.parallel_coordinator().reset();
});
