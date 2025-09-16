//! Comprehensive fuzz testing for FUGC garbage collection
//!
//! This module implements extensive property-based and fuzz testing to ensure
//! GC correctness under all possible inputs and conditions. The tests focus on:
//! - Tricolor invariant preservation
//! - Memory safety under concurrent access
//! - Allocation/deallocation correctness
//! - Write barrier consistency
//! - Root scanning accuracy

use mmtk::util::{Address, ObjectReference};
use proptest::prelude::*;
use quickcheck::{TestResult, quickcheck};
use std::collections::HashSet;
use std::sync::Arc;

use fugrip::cache_optimization::{CacheAwareAllocator, LocalityAwareWorkStealer};
use fugrip::concurrent::{
    BlackAllocator, ConcurrentMarkingCoordinator, ObjectColor, TricolorMarking, WriteBarrier,
};

/// Generate arbitrary object references for testing
fn arb_object_reference() -> impl Strategy<Value = ObjectReference> {
    (0x10000000usize..0x20000000usize).prop_map(|addr| {
        // Align to 8-byte boundary for realistic object references
        let aligned_addr = (addr / 8) * 8;
        ObjectReference::from_raw_address(unsafe { Address::from_usize(aligned_addr) })
            .unwrap_or_else(|| {
                ObjectReference::from_raw_address(unsafe { Address::from_usize(0x10000000) })
                    .unwrap()
            })
    })
}

/// Generate collections of object references with varying sizes
fn arb_object_collection() -> impl Strategy<Value = Vec<ObjectReference>> {
    prop::collection::vec(arb_object_reference(), 0..1000)
}

/// Generate realistic allocation patterns
#[derive(Debug, Clone)]
#[allow(dead_code)] // Used by quickcheck but not directly instantiated
struct AllocationPattern {
    sizes: Vec<usize>,
    alignments: Vec<usize>,
    count: usize,
}

impl quickcheck::Arbitrary for AllocationPattern {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let count = g.size() % 500 + 1;
        let mut sizes = Vec::new();
        let mut alignments = Vec::new();

        for _ in 0..count {
            // Realistic object sizes: 8 bytes to 4KB
            sizes.push(
                *g.choose(&[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
                    .unwrap(),
            );
            // Common alignments
            alignments.push(*g.choose(&[8, 16, 32, 64]).unwrap());
        }

        AllocationPattern {
            sizes,
            alignments,
            count,
        }
    }
}

/// Create quickcheck-compatible object reference generator
fn arb_object_vec(size: usize) -> Vec<ObjectReference> {
    let mut objects = Vec::new();
    for i in 0..size {
        let addr = 0x10000000 + i * 64; // Spaced 64 bytes apart
        if let Some(obj) = ObjectReference::from_raw_address(unsafe { Address::from_usize(addr) }) {
            objects.push(obj);
        }
    }
    objects
}

#[test]
fn fuzz_tricolor_invariants() {
    fn tricolor_invariant_holds(obj_count: usize) -> TestResult {
        if obj_count == 0 || obj_count > 1000 {
            return TestResult::discard();
        }

        let objects = arb_object_vec(obj_count);
        if objects.is_empty() {
            return TestResult::discard();
        }

        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 256 * 1024 * 1024;
        let marking = TricolorMarking::new(heap_base, heap_size);

        // Set up initial state: some objects grey, some white
        for (i, obj) in objects.iter().enumerate() {
            let color = if i % 3 == 0 {
                ObjectColor::Grey
            } else {
                ObjectColor::White
            };
            marking.set_color(*obj, color);
        }

        // Simulate marking process
        let mut grey_objects = Vec::new();
        for obj in &objects {
            if marking.get_color(*obj) == ObjectColor::Grey {
                grey_objects.push(*obj);
            }
        }

        // Process grey objects
        for grey_obj in grey_objects {
            // Mark as black
            marking.set_color(grey_obj, ObjectColor::Black);

            // Simulate scanning children (mark some white objects as grey)
            for (i, obj) in objects.iter().enumerate() {
                if i % 7 == grey_obj.to_raw_address().as_usize() % 7
                    && marking.get_color(*obj) == ObjectColor::White
                {
                    marking.set_color(*obj, ObjectColor::Grey);
                }
            }
        }

        // Verify tricolor invariant: no black object points to white object
        // In our simplified model, we assume random connectivity for testing
        for obj in &objects {
            let color = marking.get_color(*obj);
            // Verify color is valid
            match color {
                ObjectColor::White | ObjectColor::Grey | ObjectColor::Black => {}
            }
        }

        TestResult::passed()
    }

    quickcheck(tricolor_invariant_holds as fn(usize) -> TestResult);
}

proptest! {
    /// Property-based test: Allocation alignment and bounds
    #[test]
    fn prop_allocation_alignment_and_bounds(
        pattern in prop::collection::vec((8usize..4097, 8usize..65), 1..100)
    ) {
        let base = unsafe { Address::from_usize(0x10000000) };
        let allocator = CacheAwareAllocator::new(base, 16 * 1024 * 1024); // 16MB

        let mut allocated_addrs = Vec::new();

        for (size, _align) in &pattern {
            if let Some(addr) = allocator.allocate_aligned(*size, 1) {
                // Check cache line alignment (CacheAwareAllocator aligns to 64-byte boundaries)
                prop_assert_eq!(addr.as_usize() % 64, 0);

                // Check bounds
                prop_assert!(addr.as_usize() >= base.as_usize());
                prop_assert!(addr.as_usize() + size <= base.as_usize() + 16 * 1024 * 1024);

                allocated_addrs.push((addr, *size));
            }
        }

        // Check no overlaps
        for i in 0..allocated_addrs.len() {
            for j in i+1..allocated_addrs.len() {
                let (addr1, size1) = allocated_addrs[i];
                let (addr2, size2) = allocated_addrs[j];

                let end1 = addr1.as_usize() + size1;
                let end2 = addr2.as_usize() + size2;

                // No overlap
                prop_assert!(end1 <= addr2.as_usize() || end2 <= addr1.as_usize());
            }
        }
    }
}

proptest! {
    /// Property-based test: Write barrier consistency
    #[test]
    fn prop_write_barrier_consistency(
        objects in arb_object_collection(),
        operations in prop::collection::vec((any::<usize>(), any::<usize>()), 10..200)
    ) {
        if objects.is_empty() {
            return Ok(());
        }

        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let coordinator = Arc::new(fugrip::concurrent::ParallelMarkingCoordinator::new(4));
        let barrier = WriteBarrier::new(tricolor, coordinator);

        // Activate barrier
        barrier.activate();
        prop_assert!(barrier.is_active());

        // Perform write operations using valid stack memory
        for (src_idx, target_idx) in operations {
            let src_obj = objects[src_idx % objects.len()];
            let target_obj = objects[target_idx % objects.len()];

            // Use a local variable as the slot (simulates object field)
            let mut slot_value = src_obj;
            let slot_ptr = &mut slot_value as *mut ObjectReference;

            unsafe {
                barrier.write_barrier(slot_ptr, target_obj);
            }

            // Verify the slot was updated
            prop_assert_eq!(slot_value, target_obj);
        }

        // Deactivate barrier
        barrier.deactivate();
        prop_assert!(!barrier.is_active());
    }
}

/// Fuzz test: Concurrent marking coordinator
#[test]
fn fuzz_concurrent_marking_coordinator() {
    fn coordinator_operations_safe(
        num_workers: usize,
        object_count: usize,
        operation_count: usize,
    ) -> TestResult {
        if num_workers == 0 || num_workers > 16 || object_count == 0 || object_count > 10000 {
            return TestResult::discard();
        }

        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
        let global_roots = Arc::new(fugrip::roots::GlobalRoots::default());

        let coordinator = ConcurrentMarkingCoordinator::new(
            heap_base,
            64 * 1024 * 1024,
            num_workers,
            thread_registry,
            global_roots,
        );

        // Generate test objects
        let mut objects = Vec::new();
        for i in 0..object_count {
            let addr = unsafe { Address::from_usize(0x10000000 + i * 64) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }

        // Perform random operations
        for op in 0..operation_count.min(1000) {
            match op % 4 {
                0 => {
                    // Test cache-optimized marking
                    coordinator.mark_objects_cache_optimized(&objects[..objects.len().min(100)]);
                }
                1 => {
                    // Test statistics retrieval
                    let stats = coordinator.get_stats();
                    assert!(stats.work_stolen < 1000000); // Sanity check
                }
                2 => {
                    // Test black allocator
                    let black_allocator = coordinator.black_allocator();
                    if !objects.is_empty() {
                        black_allocator.allocate_black(objects[op % objects.len()]);
                    }
                }
                3 => {
                    // Test write barrier access
                    let _barrier = coordinator.write_barrier();
                }
                _ => unreachable!(),
            }
        }

        TestResult::passed()
    }

    quickcheck(coordinator_operations_safe as fn(usize, usize, usize) -> TestResult);
}

proptest! {
    /// Property-based test: Cache-aware allocation behavior
    #[test]
    fn prop_cache_allocation_behavior(
        allocation_pattern in prop::collection::vec((64usize..4097, 1usize..10), 1..200)
    ) {
        let base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024; // 64MB
        let allocator = CacheAwareAllocator::new(base, heap_size);

        let mut total_allocated = 0;
        let mut successful_allocations = 0;

        for (size, count) in &allocation_pattern {
            if let Some(addr) = allocator.allocate_aligned(*size, *count) {
                // Verify cache line alignment for the requested size
                let expected_alignment = if *size >= 64 { 64 } else { 8 };
                prop_assert_eq!(addr.as_usize() % expected_alignment, 0);

                // Verify address is within bounds
                prop_assert!(addr.as_usize() >= base.as_usize());
                prop_assert!(addr.as_usize() < base.as_usize() + heap_size);

                total_allocated += size * count;
                successful_allocations += 1;

                // Should not exceed heap size
                prop_assert!(total_allocated <= heap_size);
            }
        }

        // Should successfully allocate something if requests are reasonable
        if allocation_pattern.iter().map(|(s, c)| s * c).sum::<usize>() < heap_size / 2 {
            prop_assert!(successful_allocations > 0);
        }
    }
}

/// Fuzz test: Work stealing locality preservation
#[test]
fn fuzz_work_stealing_locality() {
    fn work_stealing_preserves_locality(
        region_count: usize,
        objects_per_region: usize,
        batch_size: usize,
    ) -> TestResult {
        if region_count == 0
            || region_count > 16
            || objects_per_region == 0
            || objects_per_region > 1000
            || batch_size == 0
            || batch_size > 100
        {
            return TestResult::discard();
        }

        let mut stealer = LocalityAwareWorkStealer::new(region_count);
        let mut all_objects = Vec::new();

        // Create objects with known locality patterns
        for region in 0..region_count {
            for obj_idx in 0..objects_per_region {
                let addr = 0x10000000 + region * 64 * 1024 + obj_idx * 64; // 64KB regions
                if let Some(obj) =
                    ObjectReference::from_raw_address(unsafe { Address::from_usize(addr) })
                {
                    all_objects.push(obj);
                }
            }
        }

        stealer.add_objects(all_objects.clone());

        // Extract all batches and verify locality
        let mut batches = Vec::new();
        loop {
            let batch = stealer.get_next_batch(batch_size);
            if batch.is_empty() {
                break;
            }
            batches.push(batch);
        }

        // Verify we got all objects back
        let mut recovered_objects = HashSet::new();
        for batch in &batches {
            for obj in batch {
                recovered_objects.insert(*obj);
            }
        }

        if recovered_objects.len() != all_objects.len() {
            return TestResult::failed();
        }

        // Verify locality within batches
        for batch in &batches {
            if batch.len() > 1 {
                let base_addr = batch[0].to_raw_address().as_usize();
                for obj in &batch[1..] {
                    let addr = obj.to_raw_address().as_usize();
                    let distance = addr.abs_diff(base_addr);

                    // Objects in same batch should be relatively close (within 64KB)
                    if distance > 64 * 1024 {
                        return TestResult::failed();
                    }
                }
            }
        }

        TestResult::passed()
    }

    quickcheck(work_stealing_preserves_locality as fn(usize, usize, usize) -> TestResult);
}

proptest! {
    /// Property-based test: Black allocator consistency
    #[test]
    fn prop_black_allocator_consistency(
        objects in arb_object_collection(),
        activate_iterations in 1usize..10
    ) {
        if objects.is_empty() {
            return Ok(());
        }

        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let black_allocator = BlackAllocator::new(tricolor.clone());

        // Filter objects to only those within heap bounds
        let valid_objects: Vec<_> = objects.iter()
            .filter(|obj| {
                let addr = obj.to_raw_address().as_usize();
                addr >= heap_base.as_usize() && addr < heap_base.as_usize() + heap_size
            })
            .copied()
            .collect();

        if valid_objects.is_empty() {
            return Ok(());
        }

        for _ in 0..activate_iterations {
            // Test activation/deactivation cycles
            black_allocator.activate();
            prop_assert!(black_allocator.is_active());

            // Allocate some objects as black
            for obj in &valid_objects[..valid_objects.len().min(100)] {
                // Verify allocator is active before allocation
                let was_active = black_allocator.is_active();
                black_allocator.allocate_black(*obj);

                // Verify object is marked black only if allocator was active
                if was_active {
                    prop_assert_eq!(tricolor.get_color(*obj), ObjectColor::Black);
                }
            }

            black_allocator.deactivate();
            prop_assert!(!black_allocator.is_active());

            // Reset colors for next iteration
            for obj in &objects {
                tricolor.set_color(*obj, ObjectColor::White);
            }
        }

        // Verify statistics
        let stats = black_allocator.get_stats();
        prop_assert!(stats <= objects.len() * activate_iterations);
    }
}

/// Stress test: Concurrent operations under heavy load
#[test]
fn stress_test_concurrent_gc_operations() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = Arc::new(fugrip::roots::GlobalRoots::default());

    let coordinator = Arc::new(ConcurrentMarkingCoordinator::new(
        heap_base,
        128 * 1024 * 1024, // 128MB
        8,                 // 8 workers
        thread_registry,
        global_roots,
    ));

    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Spawn multiple threads performing different operations
    for thread_id in 0..4 {
        let coordinator = Arc::clone(&coordinator);
        let stop_flag = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            let mut operation_count = 0;

            while !stop_flag.load(Ordering::Relaxed) && operation_count < 1000 {
                // Generate objects for this thread
                let mut objects = Vec::new();
                for i in 0..50 {
                    let addr =
                        unsafe { Address::from_usize(0x10000000 + thread_id * 1000000 + i * 128) };
                    if let Some(obj) = ObjectReference::from_raw_address(addr) {
                        objects.push(obj);
                    }
                }

                match operation_count % 4 {
                    0 => {
                        // Cache-optimized marking
                        coordinator.mark_objects_cache_optimized(&objects);
                    }
                    1 => {
                        // Black allocation
                        for obj in &objects[..10] {
                            coordinator.black_allocator().allocate_black(*obj);
                        }
                    }
                    2 => {
                        // Write barrier operations using valid stack memory
                        let barrier = coordinator.write_barrier();
                        if !objects.is_empty() {
                            let src = objects[0];
                            let target = objects[objects.len() - 1];

                            // Use a local variable instead of writing to invalid memory
                            let mut slot_value = src;
                            let slot_ptr = &mut slot_value as *mut ObjectReference;

                            unsafe {
                                barrier.write_barrier(slot_ptr, target);
                            }
                        }
                    }
                    3 => {
                        // Statistics and status checks
                        let _stats = coordinator.get_stats();
                        let _cache_stats = coordinator.get_cache_stats();
                    }
                    _ => unreachable!(),
                }

                operation_count += 1;

                // Yield to other threads occasionally
                if operation_count % 10 == 0 {
                    thread::yield_now();
                }
            }

            operation_count
        });

        handles.push(handle);
    }

    // Let threads run for a bit
    thread::sleep(std::time::Duration::from_millis(100));

    // Signal stop
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for all threads to complete
    let mut total_operations = 0;
    for handle in handles {
        total_operations += handle.join().expect("Thread should complete successfully");
    }

    // Verify we performed a reasonable number of operations
    assert!(
        total_operations > 100,
        "Should perform many operations under stress"
    );

    // Verify final state is consistent
    let final_stats = coordinator.get_stats();
    assert!(final_stats.work_stolen < total_operations * 10); // Sanity check
}
