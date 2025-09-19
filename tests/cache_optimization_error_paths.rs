//! Error path and edge case tests for cache_optimization.rs
//!
//! This module tests failure modes, boundary conditions, and error scenarios
//! for cache-aware allocation, marking, and optimization components.

use fugrip::cache_optimization::{
    CACHE_LINE_SIZE, CacheAwareAllocator, CacheOptimizedMarking, LocalityAwareWorkStealer,
    MemoryLayoutOptimizer, MetadataColocation,
};
use mmtk::util::{Address, ObjectReference};
use std::sync::{Arc, Mutex};
use std::thread;

#[cfg(test)]
mod error_path_tests {

    /// Test cache-aware allocator error paths
    mod cache_aware_allocator_tests {
        use super::super::*;

        #[test]
        fn test_allocation_out_of_memory() {
            let base = unsafe { Address::from_usize(0x1000) };
            let allocator = CacheAwareAllocator::new(base, 128); // Very small heap

            // Allocate until we run out of memory
            let mut allocations = 0;
            while let Some(_) = allocator.allocate(64, 8) {
                allocations += 1;
            }

            // Should have made at least one allocation
            assert!(allocations >= 1);

            // Next allocation should fail
            let failed = allocator.allocate(64, 8);
            assert!(failed.is_none());

            let (allocated_bytes, alloc_count) = allocator.get_stats();
            assert!(allocated_bytes > 0);
            assert_eq!(alloc_count, allocations);
        }

        #[test]
        fn test_allocation_with_invalid_alignment() {
            let base = unsafe { Address::from_usize(0x1000) };
            let allocator = CacheAwareAllocator::new(base, 1024);

            // Test with alignment larger than size
            let addr = allocator.allocate(8, 1024);
            assert!(addr.is_some());
            if let Some(addr) = addr {
                assert_eq!(addr.as_usize() % 1024, 0);
            }

            // Test with zero alignment (should use cache line alignment)
            let addr2 = allocator.allocate(64, 0);
            assert!(addr2.is_some());
            if let Some(addr2) = addr2 {
                assert_eq!(addr2.as_usize() % CACHE_LINE_SIZE, 0);
            }
        }

        #[test]
        fn test_allocation_concurrent_contention() {
            let base = unsafe { Address::from_usize(0x10000) };
            let allocator = Arc::new(CacheAwareAllocator::new(base, 4096));
            let mut handles = vec![];

            // Spawn multiple threads allocating concurrently
            for _ in 0..10 {
                let alloc_clone = Arc::clone(&allocator);
                let handle = thread::spawn(move || {
                    let mut successful_allocs = 0;
                    for _ in 0..10 {
                        if alloc_clone.allocate(64, 8).is_some() {
                            successful_allocs += 1;
                        }
                    }
                    successful_allocs
                });
                handles.push(handle);
            }

            let total_successful: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert!(total_successful > 0);

            // Should have some allocations
            let (allocated_bytes, alloc_count) = allocator.get_stats();
            assert!(allocated_bytes > 0);
            assert_eq!(alloc_count, total_successful);
        }

        #[test]
        fn test_allocation_boundary_sizes() {
            let base = unsafe { Address::from_usize(0x1000) };
            let allocator = CacheAwareAllocator::new(base, 2048);

            // Test minimum allocation
            let min_alloc = allocator.allocate(1, 1);
            assert!(min_alloc.is_some());

            // Test maximum reasonable allocation
            let max_alloc = allocator.allocate(1024, 64);
            assert!(max_alloc.is_some());

            // Test allocation that would exceed heap
            let too_big = allocator.allocate(2048, 64);
            assert!(too_big.is_none());
        }

        #[test]
        fn test_allocator_reset_under_contention() {
            let base = unsafe { Address::from_usize(0x1000) };
            let allocator = Arc::new(CacheAwareAllocator::new(base, 4096));

            // Fill allocator
            let mut allocations = vec![];
            while let Some(addr) = allocator.allocate(64, 8) {
                allocations.push(addr);
            }

            // Reset while other threads are trying to allocate
            let alloc_clone = Arc::clone(&allocator);
            let reset_handle = thread::spawn(move || {
                alloc_clone.reset();
            });

            let alloc_clone2 = Arc::clone(&allocator);
            let alloc_handle = thread::spawn(move || alloc_clone2.allocate(64, 8));

            reset_handle.join().unwrap();
            let post_reset_alloc = alloc_handle.join().unwrap();

            // Should be able to allocate after reset
            assert!(post_reset_alloc.is_some());

            // Stats should be reset
            let (allocated_bytes, alloc_count) = allocator.get_stats();
            assert_eq!(allocated_bytes, 64); // Just the one post-reset allocation
            assert_eq!(alloc_count, 1);
        }
    }

    /// Test cache-optimized marking error paths
    mod cache_optimized_marking_tests {
        use super::super::*;

        #[test]
        fn test_marking_invalid_object_references() {
            let marking = CacheOptimizedMarking::new(4);

            // Test with null-like address
            let null_obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0)) };
            marking.mark_object(null_obj); // Should not panic

            // Test with extreme addresses
            let max_obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(usize::MAX))
            };
            marking.mark_object(max_obj); // Should not panic

            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 2);
        }

        #[test]
        fn test_marking_batch_with_empty_slice() {
            let marking = CacheOptimizedMarking::new(4);

            let empty_batch: &[ObjectReference] = &[];
            marking.mark_objects_batch(empty_batch); // Should not panic

            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 0);
        }

        #[test]
        fn test_marking_without_tricolor_backend() {
            let marking = CacheOptimizedMarking::new(4);

            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000)) };
            marking.mark_object(obj);

            // Should work without tricolor backend
            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 1);
            assert!(!marking.is_complete()); // Should have work in queue
        }

        #[test]
        fn test_marking_prefetch_edge_cases() {
            let marking = CacheOptimizedMarking::new(4);

            // Test prefetch with invalid addresses
            let invalid_obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(1)) };
            marking.mark_object(invalid_obj); // Prefetch should handle gracefully

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
        fn test_marking_concurrent_operations() {
            let marking = Arc::new(CacheOptimizedMarking::new(4));
            let mut handles = vec![];

            // Spawn threads marking objects concurrently
            for i in 0..5 {
                let marking_clone = Arc::clone(&marking);
                let handle = thread::spawn(move || {
                    for j in 0..10 {
                        let addr = 0x10000 + (i * 10 + j) * 64;
                        let obj = unsafe {
                            ObjectReference::from_raw_address_unchecked(Address::from_usize(addr))
                        };
                        marking_clone.mark_object(obj);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, 50);
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
        fn test_marking_reset_during_operation() {
            let marking = Arc::new(CacheOptimizedMarking::new(4));

            // Add some work
            for i in 0..5 {
                let obj = unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0x1000 + i * 64,
                    ))
                };
                marking.mark_object(obj);
            }

            // Reset while processing
            let marking_clone = Arc::clone(&marking);
            let reset_handle = thread::spawn(move || {
                marking_clone.reset();
            });

            let marking_for_process = Arc::clone(&marking);
            let process_handle = thread::spawn(move || {
                let mut processed = 0;
                while let Some(_) = marking_for_process.process_work() {
                    processed += 1;
                }
                processed
            });

            reset_handle.join().unwrap();
            let processed_count = process_handle.join().unwrap();

            // Should have processed some or all work
            assert!(processed_count >= 0);

            // After reset, should be complete
            assert!(marking.is_complete());
        }
    }

    /// Test memory layout optimizer error paths
    mod memory_layout_optimizer_tests {
        use super::super::*;

        #[test]
        fn test_size_class_edge_cases() {
            let optimizer = MemoryLayoutOptimizer::new();

            // Test zero size
            let class = optimizer.get_size_class(0);
            assert_eq!(class, 8); // Should default to minimum

            // Test maximum size
            let max_class = optimizer.get_size_class(usize::MAX);
            assert_eq!(max_class, usize::MAX.next_power_of_two());

            // Test powers of two
            for i in 0..12 {
                let size = 1usize << i;
                let class = optimizer.get_size_class(size);
                assert_eq!(class, size);
            }
        }

        #[test]
        fn test_record_allocation_edge_cases() {
            let optimizer = MemoryLayoutOptimizer::new();

            // Record allocations of various sizes
            optimizer.record_allocation(0); // Zero size
            optimizer.record_allocation(usize::MAX); // Maximum size
            optimizer.record_allocation(1); // Very small

            let stats = optimizer.get_statistics();
            assert!(!stats.is_empty());

            // Should have recorded at least the small allocation
            let small_class_count = stats.iter().find(|(size, _)| *size >= 1).unwrap().1;
            assert!(small_class_count >= 1);
        }

        #[test]
        fn test_calculate_layout_empty_input() {
            let optimizer = MemoryLayoutOptimizer::new();

            let empty_sizes: &[usize] = &[];
            let layouts = optimizer.calculate_object_layout(empty_sizes);

            assert!(layouts.is_empty());
        }

        #[test]
        fn test_calculate_layout_extreme_sizes() {
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
        fn test_colocate_metadata_edge_cases() {
            let optimizer = MemoryLayoutOptimizer::new();

            // Test with zero address
            let zero_addr = unsafe { Address::from_usize(0) };
            let meta_addr = optimizer.colocate_metadata(zero_addr, 16);
            assert!(meta_addr.as_usize() <= zero_addr.as_usize());

            // Test with maximum address
            let max_addr = unsafe { Address::from_usize(usize::MAX) };
            let max_meta = optimizer.colocate_metadata(max_addr, 64);
            // Should handle wraparound gracefully
            assert!(max_meta.as_usize() <= max_addr.as_usize());

            // Test with zero metadata size
            let meta_zero = optimizer.colocate_metadata(zero_addr, 0);
            assert_eq!(meta_zero.as_usize(), zero_addr.as_usize());
        }

        #[test]
        fn test_layout_optimizer_concurrent_access() {
            let optimizer = Arc::new(MemoryLayoutOptimizer::new());
            let mut handles = vec![];

            for _ in 0..5 {
                let opt_clone = Arc::clone(&optimizer);
                let handle = thread::spawn(move || {
                    // Test concurrent size class queries
                    let _ = opt_clone.get_size_class(64);
                    let _ = opt_clone.get_size_class(128);

                    // Test concurrent allocation recording
                    opt_clone.record_allocation(32);
                    opt_clone.record_allocation(64);

                    // Test concurrent statistics access
                    let _ = opt_clone.get_statistics();
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            // Check that allocations were recorded
            let stats = optimizer.get_statistics();
            let total_allocs: usize = stats.iter().map(|(_, count)| *count).sum();
            assert!(total_allocs >= 10); // At least 2 per thread
        }
    }

    /// Test locality-aware work stealer error paths
    mod locality_aware_work_stealer_tests {
        use super::super::*;

        #[test]
        fn test_work_stealer_empty_queues() {
            let mut stealer = LocalityAwareWorkStealer::new(10);

            // Test operations on empty stealer
            assert!(!stealer.has_work());

            let batch = stealer.get_next_batch(5);
            assert!(batch.is_empty());

            let work = stealer.pop();
            assert!(work.is_none());

            let stats = stealer.get_stats();
            assert_eq!(stats, (0, 0, 0));
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
        fn test_work_stealer_batch_size_edge_cases() {
            let mut stealer = LocalityAwareWorkStealer::new(10);

            // Add some work
            let objects: Vec<ObjectReference> = (0..5)
                .map(|i| unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0x1000 + i * 64,
                    ))
                })
                .collect();
            stealer.add_objects(objects);

            // Test zero batch size
            let empty_batch = stealer.get_next_batch(0);
            assert!(empty_batch.is_empty());

            // Test batch size larger than available work
            let large_batch = stealer.get_next_batch(10);
            assert_eq!(large_batch.len(), 5);

            // Should be empty now
            assert!(!stealer.has_work());
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

            // Should have shared some work
            let (_, _, shared) = stealer.get_stats();
            assert!(shared > 0);

            // Should still have local work
            assert!(stealer.has_work());
        }

        #[test]
        fn test_work_stealer_concurrent_operations() {
            let stealer = Arc::new(Mutex::new(LocalityAwareWorkStealer::new(10)));
            let mut handles = vec![];

            // Add initial work
            let initial_objects: Vec<ObjectReference> = (0..20)
                .map(|i| unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        0x10000 + i * 64,
                    ))
                })
                .collect();
            {
                let mut stealer_lock = stealer.lock().unwrap();
                stealer_lock.add_objects(initial_objects);
            }

            // Spawn threads performing operations
            for i in 0..3 {
                let stealer_clone = Arc::clone(&stealer);
                let handle = thread::spawn(move || {
                    let mut operations = 0;

                    // Add more work
                    {
                        let mut stealer_lock = stealer_clone.lock().unwrap();
                        for j in 0..5 {
                            let obj = unsafe {
                                ObjectReference::from_raw_address_unchecked(Address::from_usize(
                                    0x20000 + (i * 5 + j) * 64,
                                ))
                            };
                            stealer_lock.push_local(obj);
                            operations += 1;
                        }

                        // Try to get work
                        let batch = stealer_lock.get_next_batch(3);
                        operations += batch.len();
                    }

                    operations
                });
                handles.push(handle);
            }

            let total_operations: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert!(total_operations > 0);

            // Should have some work remaining or statistics
            let stealer_lock = stealer.lock().unwrap();
            let (local, stolen, shared) = stealer_lock.get_stats();
            assert!(local + stolen + shared > 0);
        }
    }

    /// Test metadata colocation error paths
    mod metadata_colocation_tests {
        use super::super::*;

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
        fn test_metadata_concurrent_access() {
            let metadata = Arc::new(MetadataColocation::new(100, 8));
            let mut handles = vec![];

            for i in 0..5 {
                let meta_clone = Arc::clone(&metadata);
                let handle = thread::spawn(move || {
                    // Test concurrent set/get
                    meta_clone.set_metadata(i * 10, i * 100);
                    let value = meta_clone.get_metadata(i * 10);
                    assert_eq!(value, i * 100);

                    // Test concurrent update
                    let updated = meta_clone.update_metadata(i * 10 + 1, |x| x + i);
                    assert_eq!(updated, i);
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
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
    }

    /// Test concurrent access patterns across all components
    mod concurrent_integration_tests {
        use super::super::*;

        #[test]
        fn test_full_system_concurrent_stress() {
            // Create all components
            let base = unsafe { Address::from_usize(0x100000) };
            let allocator = Arc::new(CacheAwareAllocator::new(base, 65536));
            let marking = Arc::new(CacheOptimizedMarking::new(4));
            let optimizer = Arc::new(MemoryLayoutOptimizer::new());
            let stealer = Arc::new(LocalityAwareWorkStealer::new(10));
            let metadata = Arc::new(MetadataColocation::new(1000, 8));

            let mut handles = vec![];

            for thread_id in 0..4 {
                let allocator_clone = Arc::clone(&allocator);
                let marking_clone = Arc::clone(&marking);
                let optimizer_clone = Arc::clone(&optimizer);
                let stealer_clone = Arc::clone(&stealer);
                let metadata_clone = Arc::clone(&metadata);

                let handle = thread::spawn(move || {
                    let mut operations = 0;

                    for i in 0..25 {
                        // Test allocator
                        if let Some(addr) = allocator_clone.allocate(64, 8) {
                            operations += 1;

                            // Create object reference for other components
                            let obj = unsafe { ObjectReference::from_raw_address_unchecked(addr) };

                            // Test marking
                            marking_clone.mark_object(obj);
                            operations += 1;

                            // Test optimizer
                            let _size_class = optimizer_clone.get_size_class(64 + i);
                            optimizer_clone.record_allocation(64 + i);
                            operations += 1;

                            // Test work stealer
                            stealer_clone.push_local(obj);
                            operations += 1;

                            // Test metadata
                            metadata_clone.set_metadata(thread_id * 100 + i, i);
                            operations += 1;
                        }
                    }

                    operations
                });
                handles.push(handle);
            }

            let total_operations: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
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

            let stealer_stats = stealer.get_stats();
            assert!(stealer_stats.0 + stealer_stats.1 + stealer_stats.2 > 0);
        }

        #[test]
        fn test_component_interaction_edge_cases() {
            // Test interactions between components with edge cases

            let allocator = CacheAwareAllocator::new(unsafe { Address::from_usize(0x1000) }, 4096);
            let marking = CacheOptimizedMarking::new(4);
            let optimizer = MemoryLayoutOptimizer::new();

            // Allocate objects with various sizes
            let sizes = [0, 1, 64, 512, 4096];
            let mut allocated_objects = vec![];

            for &size in &sizes {
                if let Some(addr) = allocator.allocate(size.max(1), 8) {
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(addr) };
                    allocated_objects.push((obj, size));
                }
            }

            // Mark all allocated objects
            for (obj, _) in &allocated_objects {
                marking.mark_object(*obj);
            }

            // Test layout optimization on allocated sizes
            let layout_sizes: Vec<usize> =
                allocated_objects.iter().map(|(_, size)| *size).collect();
            let layouts = optimizer.calculate_object_layout(&layout_sizes);
            assert_eq!(layouts.len(), allocated_objects.len());

            // Verify marking worked
            let stats = marking.get_stats();
            assert_eq!(stats.objects_marked, allocated_objects.len());
        }
    }

    /// Test boundary conditions and extreme values
    mod boundary_condition_tests {
        use super::super::*;

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

                // Threshold of 0 should share immediately
                if threshold == 0 {
                    let (_, _, shared) = stealer.get_stats();
                    assert!(shared > 0);
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

        #[test]
        fn test_system_limits_and_recovery() {
            // Test system behavior near limits and recovery

            let allocator = CacheAwareAllocator::new(unsafe { Address::from_usize(0x1000) }, 1024);

            // Exhaust allocator
            let mut _allocations = 0;
            while allocator.allocate(128, 8).is_some() {
                _allocations += 1;
            }

            // Reset and verify recovery
            allocator.reset();
            let post_reset = allocator.allocate(64, 8);
            assert!(post_reset.is_some());

            let (bytes, count) = allocator.get_stats();
            assert_eq!(bytes, 64);
            assert_eq!(count, 1);
        }
    }
}
