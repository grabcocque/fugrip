//! Cache locality performance tests for FUGC garbage collection
//!
//! These tests validate cache optimization strategies and measure their
//! effectiveness in real-world scenarios.

use mmtk::util::{Address, ObjectReference};
use std::sync::{Arc, Mutex};

use fugrip::cache_optimization::*;
use fugrip::concurrent::{ConcurrentMarkingCoordinator, ObjectColor, TricolorMarking};

/// Test cache-friendly allocation patterns
#[test]
fn test_cache_friendly_allocation_patterns() {
    let base = unsafe { Address::from_usize(0x10000000) };
    let allocator = CacheAwareAllocator::new(base, 1024 * 1024);

    let mut allocated_addrs = Vec::new();

    // Allocate multiple objects and verify cache line alignment
    for _ in 0..100 {
        if let Some(addr) = allocator.allocate_aligned(64, 1) {
            allocated_addrs.push(addr);
        }
    }

    // Verify all allocations are cache line aligned
    for addr in &allocated_addrs {
        assert_eq!(
            addr.as_usize() % CACHE_LINE_SIZE,
            0,
            "Address {:x} is not cache line aligned",
            addr.as_usize()
        );
    }

    // Verify sequential allocations are properly spaced
    for i in 1..allocated_addrs.len() {
        let prev = allocated_addrs[i - 1].as_usize();
        let curr = allocated_addrs[i].as_usize();
        assert_eq!(
            curr - prev,
            CACHE_LINE_SIZE,
            "Sequential allocations not properly spaced"
        );
    }
}

/// Test locality-aware work distribution
#[test]
fn test_locality_aware_work_distribution() {
    let mut stealer = LocalityAwareWorkStealer::new(4);

    // Create objects with varying locality patterns
    let mut objects = Vec::new();

    // Group 1: High locality (same region)
    for i in 0..10 {
        let addr = unsafe { Address::from_usize(0x10000 + i * 64) };
        if let Some(obj) = ObjectReference::from_raw_address(addr) {
            objects.push(obj);
        }
    }

    // Group 2: Different region
    for i in 0..10 {
        let addr = unsafe { Address::from_usize(0x20000 + i * 64) };
        if let Some(obj) = ObjectReference::from_raw_address(addr) {
            objects.push(obj);
        }
    }

    stealer.add_objects(objects);

    // Verify that batches maintain locality
    let mut batches = Vec::new();
    loop {
        let batch = stealer.get_next_batch(5);
        if batch.is_empty() {
            break;
        }
        batches.push(batch);
    }

    assert!(
        batches.len() >= 2,
        "Should create multiple batches for different regions"
    );

    // Verify objects in each batch have better locality
    for batch in &batches {
        if batch.len() > 1 {
            let base_addr = batch[0].to_raw_address().as_usize();
            for obj in &batch[1..] {
                let addr = obj.to_raw_address().as_usize();
                let distance = addr.abs_diff(base_addr);
                assert!(
                    distance < 64 * 1024,
                    "Objects in batch should have good locality"
                );
            }
        }
    }
}

/// Test cache-optimized marking effectiveness
#[test]
fn test_cache_optimized_marking_effectiveness() {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let cache_marking = CacheOptimizedMarking::with_tricolor(Arc::clone(&tricolor));

    // Create objects with different locality patterns
    let mut high_locality_objects = Vec::new();
    let mut low_locality_objects = Vec::new();

    // High locality: objects close together
    for i in 0..100 {
        let addr = unsafe { Address::from_usize(0x10000000 + i * 64) };
        if let Some(obj) = ObjectReference::from_raw_address(addr) {
            high_locality_objects.push(obj);
        }
    }

    // Low locality: objects spread out
    for i in 0..100 {
        let addr = unsafe { Address::from_usize(0x10000000 + i * 4096) };
        if let Some(obj) = ObjectReference::from_raw_address(addr) {
            low_locality_objects.push(obj);
        }
    }

    // Test batch marking with high locality
    cache_marking.mark_objects_batch(&high_locality_objects);

    // Verify all objects were marked
    for obj in &high_locality_objects {
        let color = tricolor.get_color(*obj);
        assert_ne!(color, ObjectColor::White, "Object should be marked");
    }

    // Reset colors
    for obj in &high_locality_objects {
        tricolor.set_color(*obj, ObjectColor::White);
    }

    // Test with low locality
    cache_marking.mark_objects_batch(&low_locality_objects);

    for obj in &low_locality_objects {
        let color = tricolor.get_color(*obj);
        assert_ne!(color, ObjectColor::White, "Object should be marked");
    }

    // Verify cache stats are available
    let stats = cache_marking.get_cache_stats();
    assert_eq!(stats.batch_size, OBJECTS_PER_CACHE_LINE);
    assert!(stats.prefetch_distance > 0);
}

/// Test memory layout optimization
#[test]
fn test_memory_layout_optimization() {
    let optimizer = MemoryLayoutOptimizer::new();

    // Test object layout calculation
    let sizes = vec![32, 64, 128, 256];
    let layouts = optimizer.calculate_object_layout(&sizes);

    assert_eq!(layouts.len(), sizes.len());

    // Verify proper alignment for large objects
    for (i, (addr, size)) in layouts.iter().enumerate() {
        assert_eq!(*size, sizes[i]);

        if *size >= CACHE_LINE_SIZE {
            assert_eq!(
                addr.as_usize() % CACHE_LINE_SIZE,
                0,
                "Large object should be cache line aligned"
            );
        }
    }

    // Test metadata colocation
    let object_addr = unsafe { Address::from_usize(0x2000) };
    let metadata_addr = optimizer.colocate_metadata(object_addr, 24);

    // Metadata should be placed before the object
    assert!(metadata_addr.as_usize() < object_addr.as_usize());

    // Should be properly aligned
    let offset = object_addr.as_usize() - metadata_addr.as_usize();
    assert_eq!(offset % 8, 0, "Metadata should be 8-byte aligned");
}

/// Test cache performance under concurrent access
#[test]
fn test_concurrent_cache_access() {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = Arc::new(Mutex::new(fugrip::roots::GlobalRoots::default()));

    let coordinator = ConcurrentMarkingCoordinator::new(
        heap_base,
        64 * 1024 * 1024,
        4, // 4 workers
        thread_registry,
        global_roots,
    );

    // Create objects for concurrent processing
    let mut objects = Vec::new();
    for i in 0..1000 {
        let addr = unsafe { Address::from_usize(0x10000000 + i * 128) };
        if let Some(obj) = ObjectReference::from_raw_address(addr) {
            objects.push(obj);
        }
    }

    // Test cache-optimized concurrent marking
    coordinator.mark_objects_cache_optimized(&objects);

    // Verify objects were processed
    for obj in &objects {
        let color = coordinator.tricolor_marking.get_color(*obj);
        // Objects should be either grey (in queue) or black (processed)
        assert_ne!(color, ObjectColor::White, "Object should be marked");
    }

    // Verify cache stats are available
    if let Some(cache_stats) = coordinator.get_cache_stats() {
        assert!(cache_stats.batch_size > 0);
        assert!(cache_stats.prefetch_distance > 0);
    }
}

/// Performance comparison test
#[test]
fn test_cache_optimization_performance_comparison() {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

    // Create a large set of objects with realistic locality patterns
    let mut objects = Vec::new();
    for i in 0..5000 {
        // 70% of objects have good locality, 30% are scattered
        let addr = if i % 10 < 7 {
            // Good locality: within 16KB regions
            let region = i / 100;
            let offset = (i % 100) * 64;
            0x10000000 + region * 16384 + offset
        } else {
            // Poor locality: scattered
            0x10000000 + i * 4096
        };

        if let Some(obj) = ObjectReference::from_raw_address(unsafe { Address::from_usize(addr) }) {
            objects.push(obj);
        }
    }

    // Measure cache-optimized marking
    let cache_marking = CacheOptimizedMarking::with_tricolor(Arc::clone(&tricolor));
    let start = std::time::Instant::now();
    cache_marking.mark_objects_batch(&objects);
    let cache_optimized_time = start.elapsed();

    // Reset colors
    for obj in &objects {
        tricolor.set_color(*obj, ObjectColor::White);
    }

    // Measure standard marking
    let start = std::time::Instant::now();
    for obj in &objects {
        tricolor.set_color(*obj, ObjectColor::Grey);
    }
    let standard_time = start.elapsed();

    println!("Cache-optimized marking: {:?}", cache_optimized_time);
    println!("Standard marking: {:?}", standard_time);

    // Cache-optimized should generally be competitive
    // Note: For small workloads, standard marking may be faster due to lower overhead
    // For larger workloads with good locality, cache optimization should provide benefits
    // This test verifies that cache optimization doesn't cause catastrophic slowdown
    assert!(
        cache_optimized_time < standard_time * 6,
        "Cache-optimized marking should not be more than 6x slower (was {:?} vs {:?})",
        cache_optimized_time,
        standard_time
    );
}

/// Test cache-aware allocation under memory pressure
#[test]
fn test_cache_allocation_under_pressure() {
    let base = unsafe { Address::from_usize(0x10000000) };
    let small_heap = 64 * 1024; // 64KB heap
    let allocator = CacheAwareAllocator::new(base, small_heap);

    let mut allocations = Vec::new();

    // Allocate until exhaustion
    while let Some(addr) = allocator.allocate_aligned(CACHE_LINE_SIZE, 1) {
        allocations.push(addr);
        // Verify cache line alignment even under pressure
        assert_eq!(addr.as_usize() % CACHE_LINE_SIZE, 0);
    }

    // Should have allocated a reasonable number of cache lines
    let expected_allocations = small_heap / CACHE_LINE_SIZE;
    assert!(
        allocations.len() >= expected_allocations - 1,
        "Should allocate most of the available space"
    );

    // Verify no double allocation
    for i in 0..allocations.len() {
        for j in i + 1..allocations.len() {
            assert_ne!(allocations[i], allocations[j], "No duplicate allocations");
        }
    }

    // Test reset functionality
    allocator.reset();
    assert_eq!(allocator.get_allocated_bytes(), 0);

    // Should be able to allocate again
    let new_alloc = allocator.allocate_aligned(CACHE_LINE_SIZE, 1);
    assert!(new_alloc.is_some());
}
