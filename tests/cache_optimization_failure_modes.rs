//! Failure mode and boundary tests for cache_optimization.rs
//!
//! This module tests failure conditions, edge cases, and boundary scenarios
//! for cache optimization components.

use fugrip::cache_optimization::{
    CacheAwareAllocator, CacheOptimizedMarking,
    CACHE_LINE_SIZE, OBJECTS_PER_CACHE_LINE
};
use fugrip::concurrent::TricolorMarking;
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[cfg(test)]
mod failure_mode_tests {

    /// Test failure modes and edge cases in CacheAwareAllocator
    mod cache_aware_allocator_tests {
        use super::*;

        #[test]
        fn test_allocator_out_of_memory() {
            // Test allocation failure when memory is exhausted
            let base = unsafe { Address::from_usize(0x10000) };
            let small_size = 128; // Very small allocator
            let allocator = CacheAwareAllocator::new(base, small_size);

            // First allocation should succeed
            let result1 = allocator.allocate(64, 8);
            assert!(result1.is_some());

            // Second allocation should fail due to insufficient space
            let result2 = allocator.allocate(128, 8);
            assert!(result2.is_none());
        }

        #[test]
        fn test_allocator_zero_size() {
            // Test allocation with zero size
            let base = unsafe { Address::from_usize(0x20000) };
            let allocator = CacheAwareAllocator::new(base, 1024);

            let result = allocator.allocate(0, 8);
            // Should handle zero-size allocation gracefully
            assert!(result.is_some());
        }

        #[test]
        fn test_allocator_zero_alignment() {
            // Test allocation with zero alignment
            let base = unsafe { Address::from_usize(0x30000) };
            let allocator = CacheAwareAllocator::new(base, 1024);

            let result = allocator.allocate(64, 0);
            // Should handle zero alignment gracefully (likely using cache line alignment)
            assert!(result.is_some());
        }

        #[test]
        fn test_allocator_extreme_alignment() {
            // Test allocation with very large alignment
            let base = unsafe { Address::from_usize(0x40000) };
            let allocator = CacheAwareAllocator::new(base, 8192);

            // Large alignment that's still reasonable
            let result1 = allocator.allocate(64, 4096);
            assert!(result1.is_some());

            // Extremely large alignment that might fail
            let result2 = allocator.allocate(64, 8192);
            // May succeed or fail depending on implementation
        }

        #[test]
        fn test_allocator_concurrent_exhaustion() {
            // Test concurrent allocation leading to exhaustion
            let base = unsafe { Address::from_usize(0x50000) };
            let allocator = Arc::new(CacheAwareAllocator::new(base, 1024));
            let mut handles = vec![];

            for i in 0..10 {
                let allocator_clone = Arc::clone(&allocator);
                let handle = thread::spawn(move || {
                    // Each thread tries to allocate
                    let result = allocator_clone.allocate(100, 8);
                    (i, result.is_some())
                });
                handles.push(handle);
            }

            let mut successes = 0;
            for handle in handles {
                let (_, success) = handle.join().unwrap();
                if success {
                    successes += 1;
                }
            }

            // Some allocations should succeed, but not all due to size limit
            assert!(successes > 0);
            assert!(successes < 10);
        }

        #[test]
        fn test_allocator_boundary_sizes() {
            // Test allocation at cache line boundaries
            let base = unsafe { Address::from_usize(0x60000) };
            let allocator = CacheAwareAllocator::new(base, 4096);

            let test_sizes = vec![
                1,                    // Minimum
                CACHE_LINE_SIZE - 1,  // Just under cache line
                CACHE_LINE_SIZE,      // Exactly cache line
                CACHE_LINE_SIZE + 1,  // Just over cache line
                CACHE_LINE_SIZE * 2,  // Multiple cache lines
            ];

            for size in test_sizes {
                let result = allocator.allocate(size, 8);
                assert!(result.is_some(), "Allocation of size {} failed", size);
            }
        }

        #[test]
        fn test_allocator_overflow_protection() {
            // Test allocation with sizes that could cause overflow
            let base = unsafe { Address::from_usize(0x70000) };
            let allocator = CacheAwareAllocator::new(base, usize::MAX / 2);

            // Large but reasonable size
            let result1 = allocator.allocate(1024 * 1024, 64);
            assert!(result1.is_some());

            // Very large size that should be rejected
            let result2 = allocator.allocate(usize::MAX / 4, 64);
            assert!(result2.is_none());
        }

        #[test]
        fn test_allocator_reset_consistency() {
            // Test that reset properly restores allocator state
            let base = unsafe { Address::from_usize(0x80000) };
            let allocator = CacheAwareAllocator::new(base, 1024);

            // Make some allocations
            let _result1 = allocator.allocate(64, 8);
            let _result2 = allocator.allocate(128, 16);

            let allocated_before = allocator.get_allocated_bytes();
            assert!(allocated_before > 0);

            // Reset allocator
            allocator.reset();

            let allocated_after = allocator.get_allocated_bytes();
            assert_eq!(allocated_after, 0);

            // Should be able to allocate again
            let result3 = allocator.allocate(256, 8);
            assert!(result3.is_some());
        }

        #[test]
        fn test_allocator_statistics_consistency() {
            // Test that statistics remain consistent
            let base = unsafe { Address::from_usize(0x90000) };
            let allocator = CacheAwareAllocator::new(base, 1024);

            let (initial_bytes, initial_count) = allocator.get_stats();
            assert_eq!(initial_bytes, 0);
            assert_eq!(initial_count, 0);

            // Make allocations and verify stats update
            for i in 1..=5 {
                let result = allocator.allocate(32, 8);
                assert!(result.is_some());

                let (bytes, count) = allocator.get_stats();
                assert!(bytes > 0);
                assert_eq!(count, i);
            }
        }
    }

    /// Test failure modes and edge cases in CacheOptimizedMarking
    mod cache_optimized_marking_tests {
        use super::*;

        #[test]
        fn test_marking_without_tricolor() {
            // Test marking without tricolor backing
            let marking = CacheOptimizedMarking::new(4);

            // Should handle marking without tricolor backend
            let obj = ObjectReference::from_raw_address(
                unsafe { Address::from_usize(0xa0000) }
            ).unwrap();

            marking.mark_object(obj);
            assert!(!marking.is_complete()); // Should have work in queue
        }

        #[test]
        fn test_marking_empty_batch() {
            // Test marking with empty object batch
            let marking = CacheOptimizedMarking::new(4);
            let empty_batch: Vec<ObjectReference> = vec![];

            marking.mark_objects_batch(&empty_batch);
            assert!(marking.is_complete()); // Should remain complete
        }

        #[test]
        fn test_marking_large_batch() {
            // Test marking with very large object batch
            let marking = CacheOptimizedMarking::new(4);
            let large_batch: Vec<ObjectReference> = (0..10000)
                .map(|i| ObjectReference::from_raw_address(
                    unsafe { Address::from_usize(0xb0000 + i * 8) }
                ).unwrap())
                .collect();

            marking.mark_objects_batch(&large_batch);
            assert!(!marking.is_complete()); // Should have lots of work
        }

        #[test]
        fn test_marking_concurrent_operations() {
            // Test concurrent marking operations
            let marking = Arc::new(CacheOptimizedMarking::new(4));
            let mut handles = vec![];

            for thread_id in 0..5 {
                let marking_clone = Arc::clone(&marking);
                let handle = thread::spawn(move || {
                    for i in 0..100 {
                        let obj = ObjectReference::from_raw_address(
                            unsafe { Address::from_usize(0xc0000 + thread_id * 1000 + i * 8) }
                        ).unwrap();
                        marking_clone.mark_object(obj);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 500); // 5 threads * 100 objects
        }

        #[test]
        fn test_marking_with_tricolor_integration() {
            // Test marking with tricolor integration
            let heap_base = unsafe { Address::from_usize(0xd0000) };
            let heap_size = 4096;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
            let marking = CacheOptimizedMarking::with_tricolor(&tricolor);

            let obj = ObjectReference::from_raw_address(
                unsafe { Address::from_usize(0xd0100) }
            ).unwrap();

            marking.mark_object(obj);

            // Should have integrated with tricolor system
            let color = tricolor.get_color(obj);
            assert_eq!(color, fugrip::concurrent::ObjectColor::Grey);
        }

        #[test]
        fn test_marking_reset_with_tricolor() {
            // Test reset operation with tricolor integration
            let heap_base = unsafe { Address::from_usize(0xe0000) };
            let heap_size = 4096;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
            let marking = CacheOptimizedMarking::with_tricolor(&tricolor);

            // Mark some objects
            for i in 0..10 {
                let obj = ObjectReference::from_raw_address(
                    unsafe { Address::from_usize(0xe0100 + i * 8) }
                ).unwrap();
                marking.mark_object(obj);
            }

            let stats_before = marking.get_stats();
            assert!(stats_before.objects_marked > 0);

            // Reset should clear everything
            marking.reset();

            let stats_after = marking.get_stats();
            assert_eq!(stats_after.objects_marked, 0);
            assert!(marking.is_complete());
        }

        #[test]
        fn test_marking_zero_prefetch_distance() {
            // Test marking with zero prefetch distance
            let marking = CacheOptimizedMarking::new(0);

            let obj = ObjectReference::from_raw_address(
                unsafe { Address::from_usize(0xf0000) }
            ).unwrap();

            marking.mark_object(obj);
            // Should handle zero prefetch distance gracefully
        }

        #[test]
        fn test_marking_extreme_prefetch_distance() {
            // Test marking with very large prefetch distance
            let marking = CacheOptimizedMarking::new(usize::MAX);

            let obj = ObjectReference::from_raw_address(
                unsafe { Address::from_usize(0x100000) }
            ).unwrap();

            marking.mark_object(obj);
            // Should handle extreme prefetch distance gracefully
        }
    }

    /// Test failure modes and edge cases in MemoryLayoutOptimizer
    mod memory_layout_optimizer_tests {
        use super::*;

        #[test]
        fn test_optimizer_zero_size() {
            // Test size class determination for zero size
            let optimizer = MemoryLayoutOptimizer::new();
            let size_class = optimizer.get_size_class(0);

            // Should return smallest size class (8 bytes)
            assert_eq!(size_class, 8);
        }

        #[test]
        fn test_optimizer_very_large_size() {
            // Test size class determination for very large size
            let optimizer = MemoryLayoutOptimizer::new();
            let large_size = usize::MAX / 2;
            let size_class = optimizer.get_size_class(large_size);

            // Should handle large sizes without overflow
            assert!(size_class >= large_size);
            assert!(size_class.is_power_of_two());
        }

        #[test]
        fn test_optimizer_power_of_two_boundary() {
            // Test size class determination at power-of-two boundaries
            let optimizer = MemoryLayoutOptimizer::new();

            let test_sizes = vec![
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
            ];

            for size in test_sizes {
                let size_class = optimizer.get_size_class(size);
                assert!(size_class >= size);

                // For sizes within predefined classes, should match exactly or next class
                if size <= 4096 {
                    assert!(size_class <= size * 2);
                }
            }
        }

        #[test]
        fn test_optimizer_allocation_recording() {
            // Test allocation recording functionality
            let optimizer = MemoryLayoutOptimizer::new();

            // Record various allocations
            for size in [8, 16, 32, 64, 128] {
                for _ in 0..10 {
                    optimizer.record_allocation(size);
                }
            }

            let stats = optimizer.get_statistics();
            assert!(!stats.is_empty());

            // Should have recorded allocations for multiple size classes
            let total_allocations: usize = stats.values().sum();
            assert_eq!(total_allocations, 50); // 5 sizes * 10 allocations each
        }

        #[test]
        fn test_optimizer_concurrent_recording() {
            // Test concurrent allocation recording
            let optimizer = Arc::new(MemoryLayoutOptimizer::new());
            let mut handles = vec![];

            for thread_id in 0..5 {
                let optimizer_clone = Arc::clone(&optimizer);
                let handle = thread::spawn(move || {
                    for i in 0..100 {
                        let size = 8 + (i % 8) * 8; // Sizes 8, 16, 24, ...
                        optimizer_clone.record_allocation(size);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            let stats = optimizer.get_statistics();
            let total_allocations: usize = stats.values().sum();
            assert_eq!(total_allocations, 500); // 5 threads * 100 allocations
        }

        #[test]
        fn test_optimizer_size_class_consistency() {
            // Test that size class determination is consistent
            let optimizer = MemoryLayoutOptimizer::new();

            for size in 1..=4096 {
                let size_class1 = optimizer.get_size_class(size);
                let size_class2 = optimizer.get_size_class(size);

                assert_eq!(size_class1, size_class2, "Size class should be consistent for size {}", size);
                assert!(size_class1 >= size, "Size class {} should be >= size {}", size_class1, size);
            }
        }
    }

    /// Test failure modes and edge cases in LocalityAwareWorkStealer
    mod locality_aware_work_stealer_tests {
        use super::*;

        #[test]
        fn test_work_stealer_empty_operations() {
            // Test work stealer operations when empty
            let stealer = LocalityAwareWorkStealer::new(4);

            // Should handle empty state gracefully
            let batch = stealer.get_next_batch(10);
            assert!(batch.is_empty());

            assert!(!stealer.has_work());
        }

        #[test]
        fn test_work_stealer_zero_batch_size() {
            // Test work stealer with zero batch size
            let stealer = LocalityAwareWorkStealer::new(4);

            // Add some work
            for i in 0..10 {
                let obj = ObjectReference::from_raw_address(
                    unsafe { Address::from_usize(0x110000 + i * 8) }
                ).unwrap();
                stealer.add_work(obj);
            }

            // Request zero-sized batch
            let batch = stealer.get_next_batch(0);
            assert!(batch.is_empty());
        }

        #[test]
        fn test_work_stealer_oversized_batch() {
            // Test work stealer with batch size larger than available work
            let stealer = LocalityAwareWorkStealer::new(4);

            // Add small amount of work
            for i in 0..5 {
                let obj = ObjectReference::from_raw_address(
                    unsafe { Address::from_usize(0x120000 + i * 8) }
                ).unwrap();
                stealer.add_work(obj);
            }

            // Request larger batch than available
            let batch = stealer.get_next_batch(20);
            assert!(batch.len() <= 5); // Should return at most what's available
        }

        #[test]
        fn test_work_stealer_concurrent_access() {
            // Test concurrent access to work stealer
            let stealer = Arc::new(LocalityAwareWorkStealer::new(8));
            let mut handles = vec![];

            // Spawn producers
            for thread_id in 0..3 {
                let stealer_clone = Arc::clone(&stealer);
                let handle = thread::spawn(move || {
                    for i in 0..100 {
                        let obj = ObjectReference::from_raw_address(
                            unsafe { Address::from_usize(0x130000 + thread_id * 1000 + i * 8) }
                        ).unwrap();
                        stealer_clone.add_work(obj);
                    }
                });
                handles.push(handle);
            }

            // Spawn consumers
            for _ in 0..2 {
                let stealer_clone = Arc::clone(&stealer);
                let handle = thread::spawn(move || {
                    let mut total_stolen = 0;
                    while total_stolen < 50 {
                        let batch = stealer_clone.get_next_batch(10);
                        total_stolen += batch.len();
                        if batch.is_empty() {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn test_work_stealer_threshold_behavior() {
            // Test behavior at different threshold values
            let thresholds = vec![0, 1, 4, 16, 64];

            for threshold in thresholds {
                let stealer = LocalityAwareWorkStealer::new(threshold);

                // Add work up to threshold
                for i in 0..threshold + 10 {
                    let obj = ObjectReference::from_raw_address(
                        unsafe { Address::from_usize(0x140000 + i * 8) }
                    ).unwrap();
                    stealer.add_work(obj);
                }

                assert!(stealer.has_work());

                // Should be able to steal work
                let batch = stealer.get_next_batch(5);
                if threshold > 0 {
                    assert!(!batch.is_empty());
                }
            }
        }
    }

    /// Test failure modes and edge cases in MetadataColocation
    mod metadata_colocation_tests {
        use super::*;

        #[test]
        fn test_metadata_colocation_zero_capacity() {
            // Test metadata colocation with zero capacity
            let metadata = MetadataColocation::new(0, 8);

            // Should handle zero capacity gracefully
            let result = metadata.allocate_slot();
            assert!(result.is_none());
        }

        #[test]
        fn test_metadata_colocation_zero_stride() {
            // Test metadata colocation with zero stride
            let metadata = MetadataColocation::new(10, 0);

            // Should handle zero stride gracefully (may use default)
            let result = metadata.allocate_slot();
            // Implementation-specific behavior
        }

        #[test]
        fn test_metadata_colocation_exhaustion() {
            // Test metadata colocation when slots are exhausted
            let capacity = 5;
            let metadata = MetadataColocation::new(capacity, 8);

            // Allocate all slots
            let mut slots = vec![];
            for _ in 0..capacity {
                let slot = metadata.allocate_slot();
                assert!(slot.is_some());
                slots.push(slot.unwrap());
            }

            // Next allocation should fail
            let result = metadata.allocate_slot();
            assert!(result.is_none());
        }

        #[test]
        fn test_metadata_colocation_concurrent_allocation() {
            // Test concurrent slot allocation
            let metadata = Arc::new(MetadataColocation::new(100, 8));
            let mut handles = vec![];

            for _ in 0..10 {
                let metadata_clone = Arc::clone(&metadata);
                let handle = thread::spawn(move || {
                    let mut allocated = 0;
                    for _ in 0..15 {
                        if metadata_clone.allocate_slot().is_some() {
                            allocated += 1;
                        }
                    }
                    allocated
                });
                handles.push(handle);
            }

            let mut total_allocated = 0;
            for handle in handles {
                total_allocated += handle.join().unwrap();
            }

            // Should have allocated up to capacity
            assert!(total_allocated <= 100);
            assert!(total_allocated > 0);
        }
    }

    /// Test edge cases in cache optimization constants and utilities
    mod constants_and_utilities_tests {
        use super::*;

        #[test]
        fn test_cache_line_size_constants() {
            // Test cache line size constants
            assert_eq!(CACHE_LINE_SIZE, 64);
            assert_eq!(OBJECTS_PER_CACHE_LINE, 8); // 64 / 8 = 8 pointer-sized objects
            assert_eq!(OBJECTS_PER_CACHE_LINE * 8, CACHE_LINE_SIZE);
        }

        #[test]
        fn test_cache_stats_default() {
            // Test default cache stats
            let stats = CacheStats::default();
            assert_eq!(stats.objects_marked, 0);
            assert_eq!(stats.cache_misses, 0);
            assert_eq!(stats.queue_depth, 0);
            assert_eq!(stats.batch_size, 0);
            assert_eq!(stats.prefetch_distance, 0);
        }

        #[test]
        fn test_cache_stats_consistency() {
            // Test cache stats field consistency
            let mut stats = CacheStats::default();
            stats.objects_marked = 100;
            stats.cache_misses = 10;
            stats.queue_depth = 5;

            // Create another instance and verify independence
            let stats2 = CacheStats::default();
            assert_eq!(stats2.objects_marked, 0);
            assert_ne!(stats.objects_marked, stats2.objects_marked);
        }
    }
}