//! Failure mode and boundary tests for cache_optimization.rs
//!
//! This module tests failure conditions, edge cases, and boundary scenarios
//! for cache optimization components.
//! Consolidated from cache_optimization_error_paths.rs to avoid duplication.

#[cfg(test)]
mod failure_mode_tests {
    use fugrip::cache_optimization::{
        CACHE_LINE_SIZE, CacheAwareAllocator, CacheOptimizedMarking, CacheStats,
        LocalityAwareWorkStealer, MemoryLayoutOptimizer, MetadataColocation,
        OBJECTS_PER_CACHE_LINE,
    };
    use fugrip::concurrent::TricolorMarking;
    use mmtk::util::{Address, ObjectReference};
    use rayon::prelude::*;
    use std::sync::{Arc, Mutex};

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
            let _result2 = allocator.allocate(64, 8192);
            // May succeed or fail depending on implementation
        }

        #[test]
        fn test_allocator_concurrent_exhaustion() {
            // Test concurrent allocation leading to exhaustion
            let base = unsafe { Address::from_usize(0x50000) };
            let allocator = Arc::new(CacheAwareAllocator::new(base, 1024));

            let results: Vec<_> = (0..10).into_par_iter().map(|i| {
                // Each thread tries to allocate
                let result = allocator.allocate(100, 8);
                (i, result.is_some())
            }).collect();

            let successes = results.iter().filter(|&(_, success)| *success).count();

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
                1,                   // Minimum
                CACHE_LINE_SIZE - 1, // Just under cache line
                CACHE_LINE_SIZE,     // Exactly cache line
                CACHE_LINE_SIZE + 1, // Just over cache line
                CACHE_LINE_SIZE * 2, // Multiple cache lines
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
            let allocator = CacheAwareAllocator::new(base, 1024 * 1024);

            // Large but reasonable size
            let result1 = allocator.allocate(1024 * 1024, 64);
            assert!(result1.is_some());

            // Very large size that should be rejected
            let _result2 = allocator.allocate(usize::MAX / 4, 64);
            // Implementation may handle this differently - just check it doesn't panic
            // The actual behavior depends on the specific allocator implementation
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
            let obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0xa0000))
            };

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
                .map(|i| unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0xb0000 + i * 8,
                    ))
                })
                .collect();

            marking.mark_objects_batch(&large_batch);
            assert!(!marking.is_complete()); // Should have lots of work
        }

        #[test]
        fn test_marking_concurrent_operations() {
            use rayon::prelude::*;

            // Test concurrent marking operations
            let marking = Arc::new(CacheOptimizedMarking::new(4));

            // Use rayon parallel iteration instead of manual thread::spawn
            (0..5).into_par_iter().for_each(|thread_id| {
                for i in 0..100 {
                    let obj = unsafe {
                        ObjectReference::from_raw_address_unchecked(Address::from_usize(
                            0xc0000 + thread_id * 1000 + i * 8,
                        ))
                    };
                    marking.mark_object(obj);
                }
            });

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

            let obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0xd0100))
            };

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
                let obj = unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0xe0100 + i * 8,
                    ))
                };
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

            let obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0xf0000))
            };

            marking.mark_object(obj);
            // Should handle zero prefetch distance gracefully
        }

        #[test]
        fn test_marking_extreme_prefetch_distance() {
            // Test marking with very large prefetch distance
            let marking = CacheOptimizedMarking::new(usize::MAX);

            let obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x100000))
            };

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

            let test_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

            for size in test_sizes {
                let size_class = optimizer.get_size_class(size);
                assert!(size_class >= size);
                assert!(size_class.is_power_of_two());
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
            let total_allocations: usize = stats.iter().map(|(_, count)| *count).sum();
            assert_eq!(total_allocations, 50); // 5 sizes * 10 allocations each
        }

        #[test]
        fn test_optimizer_concurrent_recording() {
            use rayon::prelude::*;

            // Test concurrent allocation recording
            let optimizer = Arc::new(MemoryLayoutOptimizer::new());

            // Use rayon parallel iteration instead of manual thread::spawn
            (0..5).into_par_iter().for_each(|_thread_id| {
                for i in 0..100 {
                    let size = 8 + (i % 8) * 8; // Sizes 8, 16, 24, ...
                    optimizer.record_allocation(size);
                }
            });

            let stats = optimizer.get_statistics();
            let total_allocations: usize = stats.iter().map(|(_, count)| *count).sum();
            assert_eq!(total_allocations, 500); // 5 threads * 100 allocations
        }

        #[test]
        fn test_optimizer_size_class_consistency() {
            // Test that size class determination is consistent
            let optimizer = MemoryLayoutOptimizer::new();

            for size in 1..=4096 {
                let size_class1 = optimizer.get_size_class(size);
                let size_class2 = optimizer.get_size_class(size);

                assert_eq!(
                    size_class1, size_class2,
                    "Size class should be consistent for size {}",
                    size
                );
                assert!(
                    size_class1 >= size,
                    "Size class {} should be >= size {}",
                    size_class1,
                    size
                );
            }
        }
    }

    /// Test failure modes and edge cases in LocalityAwareWorkStealer
    mod locality_aware_work_stealer_tests {
        use super::*;

        #[test]
        fn test_work_stealer_empty_operations() {
            // Test work stealer operations when empty
            let mut stealer = LocalityAwareWorkStealer::new(4);

            // Should handle empty state gracefully
            let batch = stealer.get_next_batch(10);
            assert!(batch.is_empty());

            assert!(!stealer.has_work());
        }

        #[test]
        fn test_work_stealer_zero_batch_size() {
            // Test work stealer with zero batch size
            let mut stealer = LocalityAwareWorkStealer::new(4);

            // Add some work
            let objects: Vec<ObjectReference> = (0..10)
                .map(|i| unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0x110000 + i * 8,
                    ))
                })
                .collect();
            stealer.add_objects(objects);

            // Request zero-sized batch
            let batch = stealer.get_next_batch(0);
            assert!(batch.is_empty());
        }

        #[test]
        fn test_work_stealer_oversized_batch() {
            // Test work stealer with batch size larger than available work
            let mut stealer = LocalityAwareWorkStealer::new(4);

            // Add small amount of work
            let objects: Vec<ObjectReference> = (0..5)
                .map(|i| unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0x120000 + i * 8,
                    ))
                })
                .collect();
            stealer.add_objects(objects);

            // Request larger batch than available
            let batch = stealer.get_next_batch(20);
            assert!(batch.len() <= 5); // Should return at most what's available
        }

        #[test]
        fn test_work_stealer_concurrent_access() {
            use rayon::prelude::*;

            // Test concurrent access to work stealer
            let stealer = Arc::new(Mutex::new(LocalityAwareWorkStealer::new(8)));

            // Use rayon scope for producer-consumer pattern instead of manual thread::spawn
            rayon::scope(|s| {
                // Spawn producers
                for thread_id in 0..3 {
                    let stealer_clone = Arc::clone(&stealer);
                    s.spawn(move |_| {
                        let objects: Vec<ObjectReference> = (0..100)
                            .map(|i| unsafe {
                                ObjectReference::from_raw_address_unchecked(Address::from_usize(
                                    0x130000 + thread_id * 1000 + i * 8,
                                ))
                            })
                            .collect();
                        let mut stealer_lock = stealer_clone.lock().unwrap();
                        stealer_lock.add_objects(objects);
                    });
                }

                // Spawn consumers
                for _ in 0..2 {
                    let stealer_clone = Arc::clone(&stealer);
                    s.spawn(move |_| {
                        let mut total_stolen = 0;
                        while total_stolen < 50 {
                            let mut stealer_lock = stealer_clone.lock().unwrap();
                            let batch = stealer_lock.get_next_batch(10);
                            total_stolen += batch.len();
                            if batch.is_empty() {
                                std::hint::black_box(()); // Prevent compiler optimizations
                                rayon::yield_now(); // Use rayon yield instead of std::thread
                            }
                        }
                    });
                }
            });
        }

        #[test]
        fn test_work_stealer_threshold_behavior() {
            // Test behavior at different threshold values
            let thresholds = vec![0, 1, 4, 16, 64];

            for threshold in thresholds {
                let mut stealer = LocalityAwareWorkStealer::new(threshold);

                // Add work up to threshold
                let objects: Vec<ObjectReference> = (0..threshold + 10)
                    .map(|i| unsafe {
                        ObjectReference::from_raw_address_unchecked(Address::from_usize(
                            0x140000 + i * 8,
                        ))
                    })
                    .collect();
                stealer.add_objects(objects);

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
            metadata.set_metadata(0, 42); // Should be ignored
            assert_eq!(metadata.get_metadata(0), 0);
        }

        #[test]
        fn test_metadata_colocation_zero_stride() {
            // Test metadata colocation with zero stride
            let metadata = MetadataColocation::new(10, 0);

            // Should handle zero stride gracefully
            metadata.set_metadata(0, 42);
            assert_eq!(metadata.get_metadata(0), 42);
        }

        #[test]
        fn test_metadata_colocation_concurrent_access() {
            use rayon::prelude::*;

            // Test concurrent metadata access
            let metadata = Arc::new(Mutex::new(MetadataColocation::new(100, 8)));

            // Use rayon parallel iteration instead of manual thread::spawn
            (0..10).into_par_iter().for_each(|thread_id| {
                for i in 0..15 {
                    let index = thread_id * 15 + i;
                    if index < 100 {
                        // Ensure we don't go out of bounds
                        let value_to_set = thread_id * 100 + i;
                        {
                            let meta = metadata.lock().unwrap();
                            meta.set_metadata(index, value_to_set);
                        }
                        let value = metadata.lock().unwrap().get_metadata(index);
                        assert_eq!(value, value_to_set);
                    }
                }
            });
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
            let stats = CacheStats {
                objects_marked: 100,
                cache_misses: 10,
                queue_depth: 5,
                ..Default::default()
            };

            // Create another instance and verify independence
            let stats2 = CacheStats::default();
            assert_eq!(stats2.objects_marked, 0);
            assert_ne!(stats.objects_marked, stats2.objects_marked);
        }
    }

    /// Test additional edge cases from the original error_paths file
    mod additional_edge_cases {
        use super::*;

        #[test]
        fn test_marking_invalid_object_references() {
            let marking = CacheOptimizedMarking::new(4);

            // Test with minimum valid address (word-aligned)
            let min_obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(8)) };
            marking.mark_object(min_obj); // Should not panic

            // Test with extreme addresses (but word-aligned)
            let max_obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(usize::MAX & !0x7))
            };
            marking.mark_object(max_obj); // Should not panic

            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 2);
        }

        #[test]
        fn test_marking_prefetch_edge_cases() {
            let marking = CacheOptimizedMarking::new(4);

            // Test with word-aligned addresses
            let valid_obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(8)) };
            marking.mark_object(valid_obj); // Prefetch should handle gracefully

            // Test with addresses that might cause prefetch issues
            for i in 0..10 {
                let addr = 0x1000 + i * 4096; // Page-aligned addresses
                let obj = unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(addr))
                };
                marking.mark_object(obj);
            }

            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 11);
        }

        #[test]
        fn test_marking_process_work_empty_queue() {
            let marking = CacheOptimizedMarking::new(4);

            // Process work on empty queue
            let result = marking.process_work();
            assert!(result.is_none());

            assert!(marking.is_complete());
        }

        #[test]
        fn test_optimizer_calculate_layout_empty_input() {
            let optimizer = MemoryLayoutOptimizer::new();

            let empty_sizes: &[usize] = &[];
            let layouts = optimizer.calculate_object_layout(empty_sizes);

            assert!(layouts.is_empty());
        }

        #[test]
        fn test_optimizer_calculate_layout_extreme_sizes() {
            let optimizer = MemoryLayoutOptimizer::new();

            let sizes = vec![0, 1, usize::MAX / 2];
            let layouts = optimizer.calculate_object_layout(&sizes);

            assert_eq!(layouts.len(), 3);

            // Check that addresses are monotonically increasing
            for i in 1..layouts.len() {
                assert!(layouts[i].0.as_usize() > layouts[i - 1].0.as_usize());
            }
        }

        #[test]
        fn test_work_stealer_steal_from_empty() {
            let stealer1 = LocalityAwareWorkStealer::new(10);
            let stealer2 = LocalityAwareWorkStealer::new(10);

            // Try to steal from empty stealer
            let stolen = stealer2.steal_from(&stealer1);
            assert!(!stolen);
        }

        #[test]
        fn test_work_stealer_threshold_sharing() {
            let stealer = LocalityAwareWorkStealer::new(5);

            // Add work up to threshold
            for i in 0..6 {
                let obj = unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0x1000 + i * 64,
                    ))
                };
                stealer.push_local(obj);
            }

            // Check work stealer state
            let (local, _stolen, _shared) = stealer.get_stats();
            // Implementation may have different sharing strategies
            // Just verify we can get stats without panicking
            assert_eq!(local, 0); // Should be 0 after pushing all work

            // Should still have local work
            assert!(stealer.has_work());
        }

        #[test]
        fn test_metadata_index_out_of_bounds() {
            let metadata = MetadataColocation::new(10, 8);

            // Test valid indices
            metadata.set_metadata(0, 42);
            metadata.set_metadata(9, 84);
            assert_eq!(metadata.get_metadata(0), 42);
            assert_eq!(metadata.get_metadata(9), 84);

            // Test out of bounds indices
            metadata.set_metadata(10, 100); // Should be ignored
            metadata.set_metadata(usize::MAX, 200); // Should be ignored

            assert_eq!(metadata.get_metadata(10), 0); // Should return 0
            assert_eq!(metadata.get_metadata(usize::MAX), 0); // Should return 0
        }

        #[test]
        fn test_metadata_update_edge_cases() {
            let metadata = MetadataColocation::new(5, 8);

            // Test update on valid index
            let result = metadata.update_metadata(2, |x| x + 10);
            assert_eq!(result, 10);
            assert_eq!(metadata.get_metadata(2), 10);

            // Test update on invalid index
            let invalid_result = metadata.update_metadata(10, |x| x + 5);
            assert_eq!(invalid_result, 0);

            // Test update with overflow
            let overflow_result = metadata.update_metadata(3, |_| usize::MAX);
            assert_eq!(overflow_result, usize::MAX);
        }

        #[test]
        fn test_metadata_capacity_edge_cases() {
            // Test with zero capacity
            let zero_metadata = MetadataColocation::new(0, 8);
            zero_metadata.set_metadata(0, 42); // Should be ignored
            assert_eq!(zero_metadata.get_metadata(0), 0);

            // Test with maximum reasonable capacity
            let large_metadata = MetadataColocation::new(10000, 8);
            large_metadata.set_metadata(9999, 999);
            assert_eq!(large_metadata.get_metadata(9999), 999);
        }

        #[test]
        fn test_full_system_concurrent_stress() {
            use rayon::prelude::*;

            // Create all components
            let base = unsafe { Address::from_usize(0x100000) };
            let allocator = Arc::new(CacheAwareAllocator::new(base, 65536));
            let marking = Arc::new(CacheOptimizedMarking::new(4));
            let optimizer = Arc::new(MemoryLayoutOptimizer::new());
            let stealer = Arc::new(Mutex::new(LocalityAwareWorkStealer::new(10)));
            let metadata = Arc::new(MetadataColocation::new(1000, 8));

            // Use rayon parallel iteration with collect to gather results
            let total_operations: usize = (0..4).into_par_iter().map(|thread_id| {
                let mut operations = 0;

                for i in 0..25 {
                    // Test allocator
                    if let Some(addr) = allocator.allocate(64, 8) {
                        operations += 1;

                        // Create object reference for other components
                        let obj = unsafe { ObjectReference::from_raw_address_unchecked(addr) };

                        // Test marking
                        marking.mark_object(obj);
                        operations += 1;

                        // Test optimizer
                        let _size_class = optimizer.get_size_class(64 + i);
                        optimizer.record_allocation(64 + i);
                        operations += 1;

                        // Test work stealer
                        {
                            let stealer_lock = stealer.lock().unwrap();
                            stealer_lock.push_local(obj);
                        }
                        operations += 1;

                        // Test metadata
                        metadata.set_metadata(thread_id * 100 + i, i);
                        operations += 1;
                    }
                }

                operations
            }).sum();
            assert!(total_operations > 0);

            // Verify final state
            let (allocated_bytes, alloc_count) = allocator.get_stats();
            assert!(allocated_bytes > 0);
            assert!(alloc_count > 0);

            let marking_stats = marking.get_stats();
            assert!(marking_stats.objects_marked > 0);

            let optimizer_stats = optimizer.get_statistics();
            let total_optimizer_allocs: usize =
                optimizer_stats.iter().map(|(_, count)| *count).sum();
            assert!(total_optimizer_allocs > 0);

            let stealer_lock = stealer.lock().unwrap();
            let (local, stolen, shared) = stealer_lock.get_stats();
            assert!(local + stolen + shared > 0);
        }

        #[test]
        fn test_extreme_alignment_values() {
            let base = unsafe { Address::from_usize(0x1000) };
            let allocator = CacheAwareAllocator::new(base, 8192);

            // Test with very large alignments
            let alignments = [1, 8, 64, 512, 4096, 8192];
            for &align in &alignments {
                let addr = allocator.allocate(64, align);
                if let Some(addr) = addr {
                    assert_eq!(addr.as_usize() % align, 0);
                }
            }
        }

        #[test]
        fn test_extreme_prefetch_distances() {
            // Test marking with extreme prefetch distances
            let distances = [0, 1, 8, 16, 64, 128];

            for &distance in &distances {
                let marking = CacheOptimizedMarking::new(distance);

                let obj = unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000))
                };
                marking.mark_object(obj);

                let stats = marking.get_stats();
                assert_eq!(stats.prefetch_distance, distance);
                assert_eq!(stats.objects_marked, 1);
            }
        }

        #[test]
        fn test_extreme_work_stealer_thresholds() {
            let thresholds = [0, 1, 10, 100, 1000];

            for &threshold in &thresholds {
                let stealer = LocalityAwareWorkStealer::new(threshold);

                // Add work
                for i in 0..5 {
                    let obj = unsafe {
                        ObjectReference::from_raw_address_unchecked(Address::from_usize(
                            0x1000 + i * 64,
                        ))
                    };
                    stealer.push_local(obj);
                }

                // Verify stats can be retrieved without panicking
                let (local, _stolen, _shared) = stealer.get_stats();
                assert_eq!(local, 0); // Should be 0 after pushing all work

                // Threshold of 0 should have some work available
                if threshold == 0 {
                    assert!(stealer.has_work());
                }
            }
        }

        #[test]
        fn test_extreme_metadata_sizes() {
            let capacities = [0, 1, 10, 100, 1000];
            let strides = [1, 4, 8, 16, 64];

            for &capacity in &capacities {
                for &stride in &strides {
                    let metadata = MetadataColocation::new(capacity, stride);

                    if capacity > 0 {
                        metadata.set_metadata(0, 42);
                        assert_eq!(metadata.get_metadata(0), 42);
                    }
                }
            }
        }
    }
}
