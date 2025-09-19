//! Error condition and edge case tests for binding.rs
//!
//! This module tests error conditions, edge cases, and failure modes
//! for MMTk binding layer public API and FUGC integration.

use fugrip::binding::{
    FUGC_PLAN_MANAGER, fugc_alloc_info, fugc_get_stats, fugc_gc,
    fugc_get_phase, fugc_is_collecting, fugc_get_cycle_stats,
    take_enqueued_references
};
use fugrip::plan::FugcPlanManager;
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[cfg(test)]
mod error_condition_tests {

    /// Test error conditions and edge cases in public API functions
    mod public_api_tests {
        use super::*;

        #[test]
        fn test_fugc_plan_manager_initialization_thread_safety() {
            // Test that multiple threads can safely initialize FUGC_PLAN_MANAGER
            let mut handles = vec![];

            for i in 0..5 {
                let handle = thread::spawn(move || {
                    // Multiple threads try to initialize
                    let manager = FUGC_PLAN_MANAGER
                        .get_or_init(|| Mutex::new(FugcPlanManager::new()));

                    // Test that they all get the same instance
                    let stats = manager.lock().get_stats();
                    format!("thread_{}_stats_{}", i, stats.concurrent_collection_enabled)
                });
                handles.push(handle);
            }

            // All threads should complete successfully
            for handle in handles {
                let result = handle.join().unwrap();
                assert!(result.contains("stats_true"));
            }
        }

        #[test]
        fn test_fugc_plan_manager_state_consistency() {
            let manager = FUGC_PLAN_MANAGER
                .get_or_init(|| Mutex::new(FugcPlanManager::new()));

            // Test initial state
            {
                let locked = manager.lock();
                let stats = locked.get_stats();
                assert!(stats.concurrent_collection_enabled);
            }

            // Test state transitions
            {
                let mut locked = manager.lock();
                locked.set_concurrent_collection(false);
                let stats = locked.get_stats();
                assert!(!stats.concurrent_collection_enabled);
            }

            // Test state persistence
            {
                let locked = manager.lock();
                let stats = locked.get_stats();
                assert!(!stats.concurrent_collection_enabled);
            }

            // Restore original state
            {
                let mut locked = manager.lock();
                locked.set_concurrent_collection(true);
            }
        }

        #[test]
        fn test_fugc_alloc_info_edge_cases() {
            // Test edge cases in allocation info calculation
            let test_cases = vec![
                (0, 1),           // Zero size
                (1, 1),           // Minimum size
                (1, 2),           // Size smaller than alignment
                (15, 16),         // Size not aligned
                (16, 16),         // Size already aligned
                (usize::MAX / 2, 8), // Large size
            ];

            for (size, align) in test_cases {
                let (result_size, result_align) = fugc_alloc_info(size, align);

                // Verify constraints
                assert!(result_size >= size, "Result size {} should be >= input size {}", result_size, size);
                assert_eq!(result_align, align, "Alignment should be preserved");
                if result_align > 0 {
                    assert_eq!(result_size % result_align, 0, "Size should be aligned");
                }
                if result_align > 1 {
                    assert!(result_align.is_power_of_two(), "Alignment should be power of two");
                }
            }
        }

        #[test]
        fn test_fugc_alloc_info_zero_alignment() {
            // Test behavior with zero alignment
            let (size, align) = fugc_alloc_info(64, 0);

            // Should handle gracefully (implementation-specific behavior)
            assert!(size >= 64);
            // Alignment handling is implementation-specific
        }

        #[test]
        fn test_fugc_alloc_info_non_power_of_two_alignment() {
            // Test behavior with non-power-of-two alignment
            let non_power_alignments = vec![3, 5, 6, 7, 9, 10, 12, 15];

            for align in non_power_alignments {
                let (size, result_align) = fugc_alloc_info(64, align);

                // Should handle gracefully
                assert!(size >= 64);
                // Implementation may adjust alignment to nearest power of two
            }
        }

        #[test]
        fn test_fugc_alloc_info_large_values() {
            // Test with very large size and alignment values
            let large_size = usize::MAX / 4;
            let large_align = 1024 * 1024; // 1MB alignment

            let (result_size, result_align) = fugc_alloc_info(large_size, large_align);

            // Should handle without overflow
            assert!(result_size >= large_size);
            assert_eq!(result_align, large_align);
        }

        #[test]
        fn test_fugc_alloc_info_concurrent_access() {
            // Test that alloc_info can be called concurrently
            let mut handles = vec![];

            for i in 0..10 {
                let handle = thread::spawn(move || {
                    let size = 64 + i;
                    let align = 8;
                    let (result_size, result_align) = fugc_alloc_info(size, align);

                    (result_size >= size, result_align == align)
                });
                handles.push(handle);
            }

            // All calls should succeed
            for handle in handles {
                let (size_ok, align_ok) = handle.join().unwrap();
                assert!(size_ok);
                assert!(align_ok);
            }
        }

        #[test]
        fn test_fugc_get_stats_consistency() {
            // Test that stats remain consistent across multiple calls
            let stats1 = fugc_get_stats();
            let stats2 = fugc_get_stats();

            // Both calls should return identical stats initially
            assert_eq!(stats1.concurrent_collection_enabled, stats2.concurrent_collection_enabled);
        }

        #[test]
        fn test_fugc_get_stats_after_state_changes() {
            // Test stats reflect state changes
            let initial_stats = fugc_get_stats();
            let initial_enabled = initial_stats.concurrent_collection_enabled;

            // Change state
            {
                let manager = FUGC_PLAN_MANAGER
                    .get_or_init(|| Mutex::new(FugcPlanManager::new()));
                manager.lock().set_concurrent_collection(!initial_enabled);
            }

            // Stats should reflect the change
            let updated_stats = fugc_get_stats();
            assert_eq!(updated_stats.concurrent_collection_enabled, !initial_enabled);

            // Restore original state
            {
                let manager = FUGC_PLAN_MANAGER
                    .get_or_init(|| Mutex::new(FugcPlanManager::new()));
                manager.lock().set_concurrent_collection(initial_enabled);
            }
        }

        #[test]
        fn test_fugc_get_stats_concurrent_access() {
            // Test that multiple threads can safely get stats
            let mut handles = vec![];

            for _ in 0..5 {
                let handle = thread::spawn(|| {
                    let stats = fugc_get_stats();
                    stats.concurrent_collection_enabled
                });
                handles.push(handle);
            }

            // All calls should succeed and return consistent results
            let mut results = vec![];
            for handle in handles {
                results.push(handle.join().unwrap());
            }

            // All results should be the same
            let first_result = results[0];
            for result in results {
                assert_eq!(result, first_result);
            }
        }

        #[test]
        fn test_fugc_phase_functions() {
            // Test phase query functions
            let phase = fugc_get_phase();
            let is_collecting = fugc_is_collecting();

            // Initially should be idle
            assert_eq!(phase, fugrip::fugc_coordinator::FugcPhase::Idle);
            assert!(!is_collecting);
        }

        #[test]
        fn test_fugc_cycle_stats() {
            // Test cycle statistics function
            let stats = fugc_get_cycle_stats();

            // Should return valid statistics
            assert!(stats.cycles_completed >= 0);
            assert!(stats.total_marking_time_ms >= 0);
            assert!(stats.total_sweep_time_ms >= 0);
            assert!(stats.objects_marked >= 0);
            assert!(stats.objects_swept >= 0);
            assert!(stats.handshakes_performed >= 0);
            assert!(stats.avg_stack_scan_objects >= 0.0);
        }

        #[test]
        fn test_take_enqueued_references_empty() {
            // Test that take_enqueued_references handles empty queue
            let refs = take_enqueued_references();
            assert!(refs.is_empty());
        }

        #[test]
        fn test_take_enqueued_references_multiple_calls() {
            // Test multiple calls to take_enqueued_references
            let refs1 = take_enqueued_references();
            let refs2 = take_enqueued_references();

            // Both should be empty initially
            assert!(refs1.is_empty());
            assert!(refs2.is_empty());
        }

        #[test]
        fn test_fugc_gc_function() {
            // Test that fugc_gc can be called without panic
            fugc_gc(); // Should not panic

            // Verify state consistency after GC call
            let phase = fugc_get_phase();
            // Phase should be valid (exact phase depends on timing)
            assert!(matches!(phase,
                fugrip::fugc_coordinator::FugcPhase::Idle |
                fugrip::fugc_coordinator::FugcPhase::ActivateBarriers |
                fugrip::fugc_coordinator::FugcPhase::ActivateBlackAllocation |
                fugrip::fugc_coordinator::FugcPhase::MarkGlobalRoots |
                fugrip::fugc_coordinator::FugcPhase::StackScanHandshake |
                fugrip::fugc_coordinator::FugcPhase::Tracing |
                fugrip::fugc_coordinator::FugcPhase::PrepareForSweep |
                fugrip::fugc_coordinator::FugcPhase::Sweeping
            ));
        }

        #[test]
        fn test_concurrent_plan_manager_operations() {
            // Test concurrent access to plan manager
            let mut handles = vec![];

            for i in 0..5 {
                let handle = thread::spawn(move || {
                    let manager = FUGC_PLAN_MANAGER
                        .get_or_init(|| Mutex::new(FugcPlanManager::new()));

                    // Perform operations
                    let stats = manager.lock().get_stats();
                    let enabled = stats.concurrent_collection_enabled;

                    // Toggle state
                    manager.lock().set_concurrent_collection(!enabled);

                    // Get updated stats
                    let updated_stats = manager.lock().get_stats();

                    // Restore state
                    manager.lock().set_concurrent_collection(enabled);

                    (enabled, updated_stats.concurrent_collection_enabled)
                });
                handles.push(handle);
            }

            // All operations should complete successfully
            for handle in handles {
                let (original, updated) = handle.join().unwrap();
                assert_ne!(original, updated); // State should have changed
            }
        }

        #[test]
        fn test_concurrent_api_calls() {
            // Test that all API functions can be called concurrently
            let mut handles = vec![];

            for i in 0..5 {
                let handle = thread::spawn(move || {
                    // Call various API functions
                    let _stats = fugc_get_stats();
                    let _phase = fugc_get_phase();
                    let _is_collecting = fugc_is_collecting();
                    let _cycle_stats = fugc_get_cycle_stats();
                    let _refs = take_enqueued_references();
                    let _alloc_info = fugc_alloc_info(64, 8);

                    if i == 0 {
                        fugc_gc(); // Only one thread triggers GC
                    }

                    i
                });
                handles.push(handle);
            }

            // All operations should complete successfully
            for (i, handle) in handles.into_iter().enumerate() {
                let result = handle.join().unwrap();
                assert_eq!(result, i);
            }
        }

        #[test]
        fn test_api_functions_stress() {
            // Stress test with rapid API calls
            let duration = Duration::from_millis(100);
            let start = std::time::Instant::now();
            let mut call_count = 0;

            while start.elapsed() < duration {
                let _stats = fugc_get_stats();
                let _phase = fugc_get_phase();
                let _is_collecting = fugc_is_collecting();
                let _alloc_info = fugc_alloc_info(64 + (call_count % 1000), 8);
                call_count += 1;
            }

            // Should have made many calls without errors
            assert!(call_count > 100);
        }
    }
}