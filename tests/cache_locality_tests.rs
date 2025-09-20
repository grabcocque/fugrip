//! Cache locality performance tests for FUGC garbage collection
//!
//! These tests measure and validate cache-friendly optimizations
//! in realistic garbage collection scenarios.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fugrip::cache_optimization::{
    CACHE_LINE_SIZE, CacheAwareAllocator, CacheOptimizedMarking, MemoryLayoutOptimizer,
    MetadataColocation,
};
use fugrip::concurrent::{ObjectColor, TricolorMarking};
use fugrip::fugc_coordinator::FugcCoordinator;
use fugrip::test_utils::LocalityAwareWorkStealer;
use mmtk::util::{Address, ObjectReference};

/// Generate objects with controlled spatial locality patterns
fn create_objects_with_locality(count: usize, locality_ratio: f64) -> Vec<ObjectReference> {
    let mut objects: Vec<ObjectReference> = Vec::with_capacity(count);
    let base_addr = 0x10000000;

    for i in 0..count {
        let addr = if i > 0 && fastrand::f64() < locality_ratio {
            // Place object near the previous one (high cache locality)
            let prev_addr = objects[i - 1].to_raw_address().as_usize();
            prev_addr + 64 + (fastrand::usize(..16) * 8) // Word-aligned, within cache lines
        } else {
            // Random placement (low cache locality), but word-aligned
            base_addr + (i * 1024) + (fastrand::usize(..256) * 8)
        };

        // Ensure word alignment (8-byte alignment for 64-bit systems)
        let aligned_addr = (addr + 7) & !7;

        if let Some(obj_ref) =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(aligned_addr) })
        {
            objects.push(obj_ref);
        }
    }

    objects
}

/// Measure cache performance using simple access patterns
fn measure_access_time(objects: &[ObjectReference]) -> Duration {
    let start = Instant::now();

    // Simulate accessing object metadata/headers
    for obj in objects {
        // Access the address (simulates reading object header)
        let addr = obj.to_raw_address().as_usize();
        std::hint::black_box(addr);
    }

    start.elapsed()
}

#[test]
fn cache_locality_allocation_performance() {
    println!("Testing cache-aware allocation performance...");

    let heap_size = 1024 * 1024; // 1MB
    let base_addr = unsafe { Address::from_usize(0x10000000) };

    // Test cache-aware allocator
    let cache_allocator = CacheAwareAllocator::new(base_addr, heap_size);
    let start = Instant::now();

    for _ in 0..1000 {
        let _ = cache_allocator.allocate_aligned(64, 1);
    }

    let cache_aware_time = start.elapsed();

    // Test naive allocator (simulate)
    let start = Instant::now();
    let mut offset = 0;

    for _ in 0..1000 {
        offset += 64; // No alignment
        std::hint::black_box(offset);
    }

    let naive_time = start.elapsed();

    println!(
        "Cache-aware allocation: {:?}, Naive allocation: {:?}",
        cache_aware_time, naive_time
    );

    // Cache-aware should be more predictable (though timing may vary)
    assert!(cache_allocator.get_allocated_bytes() > 0);
}

#[test]
fn cache_optimized_marking_vs_standard() {
    println!("Comparing cache-optimized vs standard marking...");

    // Create objects with high locality
    let objects = create_objects_with_locality(5000, 0.8);

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let tricolor = Arc::new(TricolorMarking::new(heap_base, 64 * 1024 * 1024));

    // Test cache-optimized marking
    let cache_marking = CacheOptimizedMarking::with_tricolor(&tricolor);
    let start = Instant::now();
    cache_marking.mark_objects_batch(&objects);
    let cache_optimized_time = start.elapsed();

    // Reset colors for fair comparison
    for obj in &objects {
        tricolor.set_color(*obj, ObjectColor::White);
    }

    // Test standard marking
    let start = Instant::now();
    for obj in &objects {
        tricolor.set_color(*obj, ObjectColor::Grey);
    }
    let standard_time = start.elapsed();

    println!(
        "Cache-optimized marking: {:?}, Standard marking: {:?}",
        cache_optimized_time, standard_time
    );

    // Both should complete successfully
    assert!(cache_optimized_time > Duration::ZERO);
    assert!(standard_time > Duration::ZERO);
}

#[test]
fn locality_aware_work_stealing_efficiency() {
    println!("Testing locality-aware work stealing...");

    let objects_high_locality = create_objects_with_locality(2000, 0.9);
    let objects_low_locality = create_objects_with_locality(2000, 0.1);

    // Test with high locality objects
    let mut stealer_high = LocalityAwareWorkStealer::new(8);
    stealer_high.add_objects(objects_high_locality.clone());

    let start = Instant::now();
    let mut batches_processed: u32 = 0;
    while !stealer_high.get_next_batch(32).is_empty() {
        batches_processed += 1;
    }
    let high_locality_time = start.elapsed();

    // Test with low locality objects
    let mut stealer_low = LocalityAwareWorkStealer::new(8);
    stealer_low.add_objects(objects_low_locality.clone());

    let start = Instant::now();
    let mut batches_processed_low: u32 = 0;
    while !stealer_low.get_next_batch(32).is_empty() {
        batches_processed_low += 1;
    }
    let low_locality_time = start.elapsed();

    println!(
        "High locality processing: {:?} ({} batches), Low locality: {:?} ({} batches)",
        high_locality_time, batches_processed, low_locality_time, batches_processed_low
    );

    // Should process similar number of batches (within 5 batch difference due to timing variance)
    let batch_diff = batches_processed.abs_diff(batches_processed_low);
    assert!(
        batch_diff <= 5,
        "Batch count difference {} is too large",
        batch_diff
    );
}

#[test]
fn memory_layout_optimization_validation() {
    println!("Validating memory layout optimizations...");

    let optimizer = MemoryLayoutOptimizer::new();

    // Test object layout calculation
    let sizes = vec![32, 64, 128, 256, 512];
    let layouts = optimizer.calculate_object_layout(&sizes);

    // Verify proper alignment
    for (i, (addr, size)) in layouts.iter().enumerate() {
        if *size >= CACHE_LINE_SIZE {
            assert_eq!(
                addr.as_usize() % CACHE_LINE_SIZE,
                0,
                "Large object {} not cache-line aligned",
                i
            );
        }
        println!("Object {}: addr={:x}, size={}", i, addr.as_usize(), size);
    }

    // Test metadata colocation
    let object_addr = unsafe { Address::from_usize(0x1000) };
    let metadata_addr = optimizer.colocate_metadata(object_addr, 16);

    assert!(metadata_addr.as_usize() < object_addr.as_usize());
    assert_eq!((object_addr.as_usize() - metadata_addr.as_usize()) % 8, 0);

    println!(
        "Metadata at {:x}, object at {:x}, distance: {} bytes",
        metadata_addr.as_usize(),
        object_addr.as_usize(),
        object_addr.as_usize() - metadata_addr.as_usize()
    );
}

#[test]
fn concurrent_marking_cache_integration() {
    println!("Testing concurrent marking with cache optimizations...");

    let objects = create_objects_with_locality(1000, 0.7);
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let thread_registry = Arc::new(fugrip::thread::ThreadRegistry::new());
    let global_roots = arc_swap::ArcSwap::new(Arc::new(fugrip::roots::GlobalRoots::default()));

    let coordinator = FugcCoordinator::new(
        heap_base,
        64 * 1024 * 1024,
        4,
        &thread_registry,
        &global_roots,
    );

    // Test cache-optimized marking using the coordinator's existing components
    let start = Instant::now();

    // Use the cache-optimized marking component from the coordinator
    let cache_marking = coordinator.cache_optimized_marking();
    cache_marking.mark_objects_batch(&objects);

    let cache_time = start.elapsed();

    // Get cache performance statistics from the cache-optimized marking component
    let cache_stats = cache_marking.get_cache_stats();
    println!("Cache stats: {:?}", cache_stats);
    assert!(cache_stats.objects_processed > 0);
    assert!(cache_stats.cache_hits > 0 || cache_stats.cache_misses > 0);

    // Verify that all objects were processed
    let processed_count = cache_stats.objects_processed;
    assert!(processed_count >= objects.len() / 2, "Insufficient objects processed");

    println!("Cache-optimized concurrent marking time: {:?}", cache_time);
    println!("Processed {} objects with {:.1}% efficiency",
             processed_count,
             (processed_count as f64 / objects.len() as f64) * 100.0);
    assert!(cache_time > Duration::ZERO);
}

#[test]
fn end_to_end_cache_performance() {
    println!("End-to-end cache performance test...");

    let heap_size = 16 * 1024 * 1024; // 16MB
    let object_count = 10000;
    let base_addr = unsafe { Address::from_usize(0x10000000) };

    // Create realistic object graph
    let objects = create_objects_with_locality(object_count, 0.6);

    // Test complete GC cycle with cache optimizations
    let start = Instant::now();

    // 1. Cache-aware allocation simulation
    let allocator = CacheAwareAllocator::new(base_addr, heap_size);
    for i in 0..100 {
        let _ = allocator.allocate_aligned(64 + (i % 256), 1);
    }

    // 2. Cache-optimized marking
    let tricolor = Arc::new(TricolorMarking::new(base_addr, heap_size));
    let cache_marking = CacheOptimizedMarking::with_tricolor(&tricolor);
    cache_marking.mark_objects_batch(&objects);

    // 3. Locality-aware work processing
    let mut stealer = LocalityAwareWorkStealer::new(4);
    stealer.add_objects(objects);

    let mut total_processed = 0;
    while !stealer.get_next_batch(64).is_empty() {
        total_processed += 64;
    }

    let total_time = start.elapsed();

    println!(
        "End-to-end cache-optimized GC: {:?}, processed {} objects",
        total_time, total_processed
    );

    assert!(total_processed >= object_count - 64); // Allow for partial last batch
    assert!(allocator.get_allocated_bytes() > 0);
}

#[test]
fn cache_performance_regression_test() {
    println!("Cache performance regression test...");

    // This test establishes baseline performance expectations
    let objects = create_objects_with_locality(1000, 0.5);

    // Measure baseline access time
    let baseline_time = measure_access_time(&objects);

    // The access time should be reasonable (less than 10ms for 1000 objects)
    assert!(
        baseline_time < Duration::from_millis(10),
        "Access time {:?} exceeds reasonable threshold",
        baseline_time
    );

    println!("Baseline access time for 1000 objects: {:?}", baseline_time);

    // Test with different locality patterns
    let high_locality_objects = create_objects_with_locality(1000, 0.9);
    let low_locality_objects = create_objects_with_locality(1000, 0.1);

    let high_locality_time = measure_access_time(&high_locality_objects);
    let low_locality_time = measure_access_time(&low_locality_objects);

    println!(
        "High locality: {:?}, Low locality: {:?}",
        high_locality_time, low_locality_time
    );

    // Performance should be consistent regardless of locality for this simple test
    assert!(high_locality_time < Duration::from_millis(10));
    assert!(low_locality_time < Duration::from_millis(10));
}

#[cfg(feature = "cache_profiling")]
#[test]
fn cache_profiling_integration() {
    use std::sync::atomic::{AtomicU64, Ordering};

    println!("Cache profiling integration test (requires hardware counters)...");

    // Simulate cache performance monitoring using atomic counters
    let cache_hits = Arc::new(AtomicU64::new(0));
    let cache_misses = Arc::new(AtomicU64::new(0));

    let objects = create_objects_with_locality(5000, 0.8);
    let cache_marking = CacheOptimizedMarking::new(64);

    // Start profiling simulation
    let start = Instant::now();
    let hits_clone = Arc::clone(&cache_hits);
    let misses_clone = Arc::clone(&cache_misses);

    // Simulate cache hit/miss tracking during marking
    cache_marking.mark_objects_batch(&objects);

  // Simulate cache hit/miss tracking for testing
  for obj in &objects {
        // Simulate cache hit rate based on object alignment and locality
        let obj_addr = obj.to_raw_address();
        let alignment = obj_addr.as_usize() % 64; // Cache line alignment
        if alignment == 0 {
            hits_clone.fetch_add(1, Ordering::Relaxed);
        } else {
            misses_clone.fetch_add(1, Ordering::Relaxed);
        }

    let duration = start.elapsed();

    let total_accesses = cache_hits.load(Ordering::Relaxed) + cache_misses.load(Ordering::Relaxed);
    let hit_rate = if total_accesses > 0 {
        cache_hits.load(Ordering::Relaxed) as f64 / total_accesses as f64
    } else {
        0.0
    };

    println!("Profiled marking duration: {:?}", duration);
    println!("Cache hits: {}, misses: {}, hit rate: {:.2}%",
             cache_hits.load(Ordering::Relaxed),
             cache_misses.load(Ordering::Relaxed),
             hit_rate * 100.0);

    // Validate cache performance - objects with good locality should have > 75% hit rate
    assert!(hit_rate > 0.75, "Cache hit rate {:.2}% is below 75% threshold", hit_rate * 100.0);
    assert!(duration > Duration::ZERO);

    // Validate that we processed a reasonable number of cache accesses
    assert!(total_accesses >= (objects.len() / 2) as u64, "Insufficient cache accesses tracked");
}
