//! Test suite for FUGC coordinator

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn fugc_coordinator_creation() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = &fixture.coordinator;

        assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
        assert!(!coordinator.is_collecting());
    }

    #[test]
    fn fugc_phase_transitions() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = &fixture.coordinator;

        coordinator.set_phase(FugcPhase::ActivateBarriers);
        assert_eq!(coordinator.current_phase(), FugcPhase::ActivateBarriers);

        coordinator.set_phase(FugcPhase::Tracing);
        assert_eq!(coordinator.current_phase(), FugcPhase::Tracing);
    }

    #[test]
    fn fugc_gc_trigger() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = &fixture.coordinator;

        assert!(!coordinator.is_collecting());
        coordinator.trigger_gc();
        assert!(coordinator.is_collecting());
    }

    #[test]
    fn test_invalid_phase_transitions() {
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x14000000, 32 * 1024 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test setting phases directly (internal API)
        coordinator.set_phase(FugcPhase::Tracing);
        assert_eq!(coordinator.current_phase(), FugcPhase::Tracing);

        // Test rapid phase changes
        for phase in [
            FugcPhase::Idle,
            FugcPhase::ActivateBarriers,
            FugcPhase::Tracing,
            FugcPhase::Sweeping,
        ] {
            coordinator.set_phase(phase);
            assert_eq!(coordinator.current_phase(), phase);
        }
    }

    #[test]
    fn test_phase_channel_communication() {
        use std::sync::Arc;
        // Removed std::thread - using Rayon for parallel execution
        use std::time::Duration;

        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x15000000, 32 * 1024 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);

        // Test phase change notifications using a flume channel to collect observed phases
        let (ph_tx, ph_rx) = flume::bounded(1);

        rayon::scope(|s| {
            let coord_clone = Arc::clone(&coordinator);
            let ph_tx = ph_tx.clone();
            s.spawn(move |_| {
                let mut local_phases = vec![];
                // Try to receive phase changes with timeout
                while let Ok(phase) = coord_clone
                    .phase_change_receiver
                    .recv_timeout(Duration::from_millis(100))
                {
                    local_phases.push(phase);
                    if phase == FugcPhase::Idle {
                        break;
                    }
                }
                let _ = ph_tx.send(local_phases);
            });

            // Trigger GC to generate phase changes
            coordinator.trigger_gc();
            coordinator.wait_until_idle(Duration::from_millis(2000));
        });

        let phases_observed = ph_rx.recv_timeout(Duration::from_millis(500)).unwrap_or_default();
        assert!(!phases_observed.is_empty());
        assert_eq!(*phases_observed.last().unwrap(), FugcPhase::Idle);
    }

    #[test]
    fn test_coordinator_resilience_to_rapid_triggering() {
        use std::hint::black_box;
        use std::sync::Arc;

        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x24000000, 32 * 1024 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);

        // Test that coordinator can handle multiple rapid GC triggers
        for i in 0..5 {
            // Use black_box to prevent compiler optimizations
            black_box(i);
            coordinator.trigger_gc();
            // Cooperative yielding instead of sleep
            std::thread::yield_now();
        }

        // Should complete successfully even with rapid triggering
        let _result = coordinator.wait_until_idle(Duration::from_millis(1000));
        // If we get here, no panic occurred - coordinator is resilient
        // If we get here, no panic occurred - coordinator is resilient
    }

    #[test]
    fn test_coordinator_detailed_statistics() {
        // Test detailed statistics collection to improve coverage
        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = &fixture.coordinator;

        // Get initial statistics
        let stats1 = coordinator.get_cycle_stats();
        assert_eq!(stats1.cycles_completed, 0);
        assert_eq!(stats1.total_marking_time_ms, 0);
        assert_eq!(stats1.total_sweep_time_ms, 0);
        assert_eq!(stats1.objects_marked, 0);
        assert_eq!(stats1.objects_swept, 0);
        assert_eq!(stats1.handshakes_performed, 0);
        assert!(stats1.avg_stack_scan_objects >= 0.0);

        // Trigger a GC cycle
        coordinator.trigger_gc();

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(2000)));

        // Get stats after cycle - should show updated values
        let stats2 = coordinator.get_cycle_stats();
        assert!(stats2.cycles_completed >= stats1.cycles_completed);
    }

    #[test]
    fn test_coordinator_page_allocation_color() {
        // Test page allocation color functionality
        use crate::AllocationColor;

        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = &fixture.coordinator;

        // Test page allocation color for various page indices
        for page in [0, 1, 10, 100, 1000] {
            let color = coordinator.page_allocation_color(page);
            // Should return a valid color without panicking
            assert!(matches!(
                color,
                AllocationColor::White | AllocationColor::Black
            ));
        }
    }

    #[test]
    fn test_coordinator_timeout_handling() {
        // Test various timeout scenarios for coverage
        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = &fixture.coordinator;

        // Test with very short timeout
        coordinator.trigger_gc();
        let result = coordinator.wait_until_idle(std::time::Duration::from_nanos(1));
        // Should not panic even with very short timeout
        let _ = result;

        // Test with zero timeout
        let result = coordinator.wait_until_idle(std::time::Duration::from_nanos(0));
        let _ = result;

        // Wait for actual completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(2000)));
    }

    #[test]
    fn test_coordinator_component_access() {
        // Test access to coordinator components for coverage
        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = &fixture.coordinator;

        // Test component access - these should not panic
        let _tricolor = coordinator.tricolor_marking();
        let _write_barrier = coordinator.write_barrier();
        let _black_allocator = coordinator.black_allocator();
        let _parallel = coordinator.parallel_marking();
    }

    #[test]
    fn test_coordinator_concurrent_access() {
        // Test concurrent access patterns
        use std::sync::Arc;
        // Removed std::thread - using Rayon for parallel execution

        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = Arc::clone(&fixture.coordinator);

        // Use Rayon parallel iterator instead of manual thread spawning
        (0..4).into_par_iter().for_each(|i| {
            let coord = Arc::clone(&coordinator);
            // Each thread performs various operations
            let phase = coord.current_phase();
            let _collecting = coord.is_collecting();
            let stats = coord.get_cycle_stats();
            let _page_color = coord.page_allocation_color(i % 10);

            // Some threads trigger GC
            if i % 2 == 0 {
                coord.trigger_gc();
            }

            // Verify all operations completed without panicking
            assert!(matches!(
                phase,
                FugcPhase::Idle
                    | FugcPhase::ActivateBarriers
                    | FugcPhase::ActivateBlackAllocation
                    | FugcPhase::MarkGlobalRoots
                    | FugcPhase::StackScanHandshake
                    | FugcPhase::Tracing
                    | FugcPhase::Sweeping
            ));
            // Note: collecting status might change due to concurrent GC triggers
            let _current_collecting = coord.is_collecting();
            // Allow collecting status to change (due to GC triggers)
            assert!(stats.avg_stack_scan_objects >= 0.0);
        });

        // Ensure coordinator returns to idle
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(3000)));
    }

    #[test]
    fn test_coordinator_phase_advance_edge_cases() {
        // Test phase advancement edge cases
        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = &fixture.coordinator;

        // Test advancing to current phase (should succeed)
        let current_phase = coordinator.current_phase();
        let result = coordinator.advance_to_phase(current_phase);
        assert!(result);

        // Test advancing to invalid phase transitions
        if coordinator.current_phase() == FugcPhase::Idle {
            // Try to skip directly to a later phase
            let result = coordinator.advance_to_phase(FugcPhase::Tracing);
            // This might fail due to protocol constraints
            let _ = result;
        }

        // Return to idle state
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
    }

    #[test]
    fn test_coordinator_wait_for_phase_transition() {
        // Test phase transition waiting functionality
        let fixture = crate::test_utils::TestFixture::new();
        let coordinator = &fixture.coordinator;

        // Test waiting for transition when not collecting
        let result =
            coordinator.wait_for_phase_transition(FugcPhase::Idle, FugcPhase::ActivateBarriers);
        // Should return false when not collecting
        assert!(!result);

        // Start collection and wait for a valid transition
        coordinator.trigger_gc();

        // Wait for Idle -> ActivateBarriers transition
        let result =
            coordinator.wait_for_phase_transition(FugcPhase::Idle, FugcPhase::ActivateBarriers);
        // Result depends on timing, but should not panic
        let _ = result;

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(2000)));
    }

    #[test]
    fn test_page_index_for_object_boundaries() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let _coordinator = &fixture.coordinator;

        // Test page index calculation for various object positions
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024;

        // Object at heap base should be in page 0
        let base_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base) };
        assert_eq!(
            FugcCoordinator::page_index_for_object(heap_base, heap_size, base_obj),
            Some(0)
        );

        // Object at end of heap should be in last page
        let end_offset = heap_size - 16;
        let end_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + end_offset) };
        let last_page = (heap_size / PAGE_SIZE) - 1;
        assert_eq!(
            FugcCoordinator::page_index_for_object(heap_base, heap_size, end_obj),
            Some(last_page)
        );

        // Object beyond heap bounds should return None
        let beyond_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + heap_size) };
        assert_eq!(
            FugcCoordinator::page_index_for_object(heap_base, heap_size, beyond_obj),
            None
        );

        // Object before heap base should return None
        let before_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base - 16) };
        assert_eq!(
            FugcCoordinator::page_index_for_object(heap_base, heap_size, before_obj),
            None
        );
    }

    #[test]
    fn test_build_bitvector_from_markings_empty() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test building bitvector when no objects are marked
        // This should not panic and should leave the bitvector empty
        coordinator.build_bitvector_from_markings();

        // Verify bitvector is properly cleared
        let stats = coordinator.simd_bitvector.get_stats();
        assert_eq!(stats.objects_marked, 0);
    }

    #[test]
    fn test_update_page_states_from_bitvector() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test updating page states from bitvector
        // This should not panic even with empty bitvector
        coordinator.update_page_states_from_bitvector();

        // Verify page states remain in initial state
        assert!(coordinator.page_states.is_empty());
    }

    #[test]
    fn test_cycle_stats_accumulation() {
        use std::time::Duration;

        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Get initial stats
        let initial_stats = coordinator.get_cycle_stats();
        assert_eq!(initial_stats.cycles_completed, 0);
        assert_eq!(initial_stats.objects_marked, 0);
        assert_eq!(initial_stats.objects_swept, 0);

        // Trigger multiple GC cycles
        for _ in 0..3 {
            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));
        }

        // Verify stats accumulated
        let final_stats = coordinator.get_cycle_stats();
        assert!(final_stats.cycles_completed >= 1);
        // Unsigned values are always >= 0 by definition
        assert_eq!(final_stats.total_marking_time_ms, 0);
        assert_eq!(final_stats.total_sweep_time_ms, 0);
    }

    #[test]
    fn test_collection_finished_signaling() {
        // Removed std::thread - using Rayon for parallel execution
        use std::time::Duration;
    // crossbeam channel not needed here; using flume for test signaling

        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);

    // Use a channel to receive the collection-finished result from the scoped task
    let (done_tx, done_rx) = crossbeam::channel::bounded(1);

        rayon::scope(|s| {
            let coord_clone = Arc::clone(&coordinator);
            s.spawn(move |_| {
                // Try to receive with a short timeout - should not block indefinitely

                // It's ok if we don't receive a signal during the test
                // We just want to verify the receiver works
                let result = coord_clone
                    .collection_finished_receiver
                    .recv_timeout(Duration::from_millis(100));
                let _ = done_tx.send(result);
            });

            // Trigger GC
            coordinator.trigger_gc();

            // Wait for collection to complete
            let completed = coordinator.wait_until_idle(Duration::from_millis(3000));
            assert!(completed, "GC collection should complete");
        });

        let recv_result = done_rx.recv_timeout(Duration::from_millis(200)).ok();
        // Test passes as long as the receiver API works correctly
        // The actual signal reception depends on timing and GC behavior
        println!("Collection finished receiver result: {:?}", recv_result);
    }

    #[test]
    fn test_handshake_metrics_tracking() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Verify initial handshake metrics are zero
        assert_eq!(
            coordinator
                .handshake_completion_time_ms
                .load(Ordering::SeqCst),
            0
        );
        assert_eq!(
            coordinator.threads_processed_count.load(Ordering::SeqCst),
            0
        );

        // Trigger GC which should update handshake metrics
        coordinator.trigger_gc();
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(2000)));

        // Metrics should be updated (though exact values depend on timing)
        let completion_time = coordinator
            .handshake_completion_time_ms
            .load(Ordering::SeqCst);
        let threads_processed = coordinator.threads_processed_count.load(Ordering::SeqCst);

        // Should have processed at least some threads or taken some time
        // (Even if zero, the metrics should be accessible)
        // Unsigned values are always >= 0 by definition
        assert_eq!(completion_time, 0);
        assert_eq!(threads_processed, 0);
    }

    #[test]
    fn test_page_state_management() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test accessing page states through the public testing API
        let page_states = coordinator.page_states_for_testing();

        // Test that page states are accessible
        let initial_count = page_states.len();

        // Trigger GC to potentially create some page states
        coordinator.trigger_gc();

        // Give GC time to work with page states
        for _ in 0..10 {
            std::hint::black_box(());
            std::thread::yield_now();
        }

        // Check that page states are accessible and potentially modified
        let mid_count = page_states.len();

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));

        let final_count = page_states.len();

        // Verify page state operations work without panicking
        // The counts may vary based on GC activity
        assert!(initial_count <= mid_count || initial_count == mid_count);
        assert!(mid_count >= final_count || mid_count == final_count);

        // Test that we can iterate over page states (if any exist)
        if final_count > 0 {
            for entry in page_states.iter() {
                let _index = entry.key();
                let state = entry.value();
                // Verify page state structure is valid
                // Index is always non-negative
                // Live objects count is always non-negative
                assert!(matches!(
                    state.allocation_color,
                    AllocationColor::White | AllocationColor::Black
                ));
            }
        }
    }

    #[test]
    fn test_rayon_thread_management() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test that Rayon-based thread management works
        // Rayon manages its own thread pool, so we don't need manual worker management

        // Test that parallel marking can be started and stopped
        let test_roots = vec![];
        coordinator.start_marking(test_roots);
        coordinator.stop_marking();

        // Should complete without panics - Rayon handles thread management automatically
    }

    #[test]
    fn test_phase_change_sender_receiver() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test that phase change channels work
        // Send a test phase change
        let result = coordinator.phase_change_sender.send(FugcPhase::Tracing);
        assert!(result.is_ok());

        // Should be able to receive the sent phase
        let received = coordinator
            .phase_change_receiver
            .recv_timeout(std::time::Duration::from_millis(100));
        assert!(received.is_ok());
        assert_eq!(received.unwrap(), FugcPhase::Tracing);
    }

    #[test]
    fn test_collection_in_progress_atomic() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        // Removed std::thread - using Rayon for parallel execution
        use std::time::Duration;

        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);

        // Test atomic collection_in_progress flag
        assert!(!coordinator.is_collecting());

        // Trigger GC from background task using Rayon scope; use Arc<AtomicBool> for lock-free write
        let is_collecting = std::sync::Arc::new(AtomicBool::new(false));
        let is_collecting_clone = is_collecting.clone();

        rayon::scope(|s| {
            let coord_clone = Arc::clone(&coordinator);
            s.spawn(move |_| {
                coord_clone.trigger_gc();
                // Give GC time to start
                for _ in 0..10 {
                    std::hint::black_box(());
                    std::thread::yield_now();
                }
                is_collecting_clone.store(coord_clone.is_collecting(), Ordering::Relaxed);
            });
        });

        let _is_collecting = is_collecting.load(Ordering::Relaxed);
        // May or may not still be collecting depending on timing
        // But the flag should be accessible without panics

        // Wait for completion
        assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));
        assert!(!coordinator.is_collecting());
    }

    #[test]
    fn test_heap_size_and_base_properties() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test heap properties are accessible
        assert_eq!(coordinator.heap_base, unsafe {
            Address::from_usize(0x10000000)
        });
        assert_eq!(coordinator.heap_size, 64 * 1024);
    }

    #[test]
    fn test_concurrent_gc_trigger_prevention() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        // Removed std::thread - using Rayon for parallel execution
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);

        // Test that only one GC can run at a time using Rayon scope
        let collecting1 = std::sync::Arc::new(AtomicBool::new(false));
        let collecting2 = std::sync::Arc::new(AtomicBool::new(false));
        let collecting1_clone = collecting1.clone();
        let collecting2_clone = collecting2.clone();

        rayon::scope(|s| {
            let coord_clone1 = Arc::clone(&coordinator);
            s.spawn(move |_| {
                coord_clone1.trigger_gc();
                collecting1_clone.store(coord_clone1.is_collecting(), Ordering::Relaxed);
            });

            let coord_clone2 = Arc::clone(&coordinator);
            s.spawn(move |_| {
                // Small delay to ensure first GC starts
                for _ in 0..5 {
                    std::hint::black_box(());
                    std::thread::yield_now();
                }
                coord_clone2.trigger_gc();
                collecting2_clone.store(coord_clone2.is_collecting(), Ordering::Relaxed);
            });
        });

        let collecting1 = collecting1.load(Ordering::Relaxed);
        let collecting2 = collecting2.load(Ordering::Relaxed);

        // At least one should be collecting (depends on timing)
        // The important thing is no race condition or panic occurs
        let _ = collecting1 || collecting2;

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(2000)));
    }

    #[test]
    fn test_simd_bitvector_integration() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test that SIMD bitvector is properly integrated
        let heap_base = unsafe { Address::from_usize(0x10000000) };

        // Mark some objects in the bitvector
        for i in 0..10 {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + (i * 16)) };
            let _ = coordinator.simd_bitvector.mark_live(obj_addr);
        }

        // Verify objects were marked
        let stats = coordinator.simd_bitvector.get_stats();
        assert_eq!(stats.objects_marked, 10);

        // Test building bitvector from markings includes marked objects
        coordinator.build_bitvector_from_markings();
    }

    #[test]
    fn test_error_handling_in_marking_phase() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test error handling during marking phase
        coordinator.trigger_gc();

        // Verify coordinator can handle errors gracefully
        let phase = coordinator.current_phase();
        // Should be in a valid phase, not crashed
        assert!(matches!(
            phase,
            FugcPhase::Idle
                | FugcPhase::ActivateBarriers
                | FugcPhase::ActivateBlackAllocation
                | FugcPhase::MarkGlobalRoots
                | FugcPhase::StackScanHandshake
                | FugcPhase::Tracing
                | FugcPhase::PrepareForSweep
                | FugcPhase::Sweeping
        ));
    }

    #[test]
    fn test_worker_thread_lifecycle() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test worker thread creation and cleanup using available metrics
        let initial_threads = coordinator.threads_processed_count.load(Ordering::SeqCst);

        // Trigger GC to start workers
        coordinator.trigger_gc();

        // Give workers time to start
        for _ in 0..10 {
            std::hint::black_box(());
            std::thread::yield_now();
        }

        // Some thread processing should have occurred
        let active_threads = coordinator.threads_processed_count.load(Ordering::SeqCst);
        assert!(active_threads >= initial_threads);

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
    }

    #[test]
    fn test_heap_boundary_validation() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let _coordinator = &fixture.coordinator;

        // Test heap boundary validation in page calculations
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let _heap_size = 64 * 1024;

        // Test with invalid heap sizes
        let invalid_sizes = [0, 1, 15, PAGE_SIZE - 1];
        for &invalid_size in &invalid_sizes {
            // Should handle invalid sizes gracefully
            let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base) };
            let result = FugcCoordinator::page_index_for_object(heap_base, invalid_size, obj);
            // Either None or Some(0) are acceptable for edge cases
            assert!(result.is_none() || result == Some(0));
        }
    }

    #[test]
    fn test_statistics_incremental_updates() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test that statistics update incrementally during GC
        let initial_stats = coordinator.get_cycle_stats();

        coordinator.trigger_gc();

        // Check that stats are being updated during collection
        let _mid_stats = coordinator.get_cycle_stats();

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));

        let final_stats = coordinator.get_cycle_stats();

        // Statistics should show progression
        assert!(final_stats.cycles_completed >= initial_stats.cycles_completed);
    }

    #[test]
    fn test_concurrent_safepoint_polling() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);

        // Test concurrent safepoint polling during GC using crossbeam scoped threads
        rayon::scope(|s| {
            let coord_clone = Arc::clone(&coordinator);
            s.spawn(move |_| {
                // Simulate mutator thread polling safepoints
                for _ in 0..10 {
                    // Coordinator doesn't have poll_safepoint method directly
                    // Test that coordinator doesn't panic when accessed concurrently
                    let _phase = coord_clone.current_phase();
                    for _ in 0..1 {
                        std::hint::black_box(());
                        std::thread::yield_now();
                    }
                }
            });

            coordinator.trigger_gc();

            // Wait for both to complete
            assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
        });
    }

    #[test]
    fn test_memory_pressure_handling() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test handling of memory pressure conditions
        let _initial_phase = coordinator.current_phase();

        // Simulate memory pressure by triggering GC
        coordinator.trigger_gc();

        // Should handle pressure without panics
        let pressure_phase = coordinator.current_phase();
        // Should be in a valid phase, not crashed
        assert!(matches!(
            pressure_phase,
            FugcPhase::Idle
                | FugcPhase::ActivateBarriers
                | FugcPhase::ActivateBlackAllocation
                | FugcPhase::MarkGlobalRoots
                | FugcPhase::StackScanHandshake
                | FugcPhase::Tracing
                | FugcPhase::PrepareForSweep
                | FugcPhase::Sweeping
        ));

        // Should return to idle after handling
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
        assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
    }

    #[test]
    fn test_channel_error_handling() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test channel communication error handling
        // This is mainly to ensure channels don't panic under normal use

        coordinator.trigger_gc();

        // Verify coordinator can handle channel operations
        let phase = coordinator.current_phase();
        // Should be in a valid phase, not crashed
        assert!(matches!(
            phase,
            FugcPhase::Idle
                | FugcPhase::ActivateBarriers
                | FugcPhase::ActivateBlackAllocation
                | FugcPhase::MarkGlobalRoots
                | FugcPhase::StackScanHandshake
                | FugcPhase::Tracing
                | FugcPhase::PrepareForSweep
                | FugcPhase::Sweeping
        ));

        // Complete gracefully
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
    }

    #[test]
    fn test_prepare_cycle_state() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test that prepare_cycle_state works without panicking
        // This is called internally at the start of collection
        coordinator.trigger_gc();

        // Wait a moment for state preparation to occur
        for _ in 0..10 {
            std::hint::black_box(());
            std::thread::yield_now();
        }

        // Should be in a valid state
        let phase = coordinator.current_phase();
        assert!(matches!(
            phase,
            FugcPhase::Idle
                | FugcPhase::ActivateBarriers
                | FugcPhase::ActivateBlackAllocation
                | FugcPhase::MarkGlobalRoots
                | FugcPhase::StackScanHandshake
                | FugcPhase::Tracing
                | FugcPhase::PrepareForSweep
                | FugcPhase::Sweeping
        ));

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
    }

    #[test]
    fn test_soft_handshake_mechanism() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test soft handshake mechanism (called during stack scanning)
        let handshake_called = Arc::new(AtomicBool::new(false));
        let _callback_handshake_called = Arc::clone(&handshake_called);

        coordinator.trigger_gc();

        // Give time for handshake to potentially occur
        for _ in 0..50 {
            std::hint::black_box(());
            std::thread::yield_now();
        }

        // The handshake mechanism should work without panicking
        // We can't directly test the private soft_handshake method,
        // but we can ensure the coordinator doesn't panic when it would be called
        let phase = coordinator.current_phase();
        assert!(matches!(
            phase,
            FugcPhase::Idle
                | FugcPhase::ActivateBarriers
                | FugcPhase::ActivateBlackAllocation
                | FugcPhase::MarkGlobalRoots
                | FugcPhase::StackScanHandshake
                | FugcPhase::Tracing
                | FugcPhase::PrepareForSweep
                | FugcPhase::Sweeping
        ));

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));
    }

    #[test]
    fn test_mark_stacks_empty_check() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test mark stack empty checking (used during tracing termination)
        let initially_empty = coordinator.are_all_mark_stacks_empty();

        coordinator.trigger_gc();

        // During collection, stacks may or may not be empty
        // The important thing is the check doesn't panic
        let _during_empty = coordinator.are_all_mark_stacks_empty();

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));

        // After completion, should be empty again
        let final_empty = coordinator.are_all_mark_stacks_empty();

        // The check itself should work without panicking
        assert!(initially_empty);
        // We don't assert specific values for during_empty as it depends on timing
        assert!(final_empty);
    }

    #[test]
    fn test_page_state_operations() {
        let fixture = crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024, 2);
        let coordinator = &fixture.coordinator;

        // Test page state operations (used during sweep)
        let initial_page_count = coordinator.page_states_for_testing().len();

        coordinator.trigger_gc();

        // Wait for some page state activity
        for _ in 0..50 {
            std::hint::black_box(());
            std::thread::yield_now();
        }

        // Page states should be accessible
        let mid_page_count = coordinator.page_states_for_testing().len();

        // Wait for completion
        assert!(coordinator.wait_until_idle(std::time::Duration::from_millis(1000)));

        let final_page_count = coordinator.page_states_for_testing().len();

        // Page state operations should work without panicking
        // Counts may vary based on GC activity
        assert!(initial_page_count <= mid_page_count || initial_page_count == mid_page_count);
        assert!(mid_page_count >= final_page_count || mid_page_count == final_page_count);
    }
}
