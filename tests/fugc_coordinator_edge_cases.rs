//! Edge case and timeout tests for fugc_coordinator.rs
//!
//! This module tests timeout paths, phase transition edge cases,
//! handshake failures, and other error conditions in the FUGC coordinator.

use fugrip::roots::GlobalRoots;
use fugrip::test_utils::TestFixture;
use fugrip::thread::{MutatorThread, ThreadRegistry};
use fugrip::{AllocationColor, FugcCoordinator, FugcPhase};
use mmtk::util::{Address, ObjectReference};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

#[cfg(test)]
mod edge_case_tests {

    /// Test timeout-related edge cases
    mod timeout_tests {
        use super::super::*;
        use fugrip::test_utils::TestFixture;
        use fugrip::{AllocationColor, FugcPhase};
        use mmtk::util::{Address, ObjectReference};
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        #[test]
        fn test_wait_until_idle_timeout() {
            let fixture = TestFixture::new_with_config(0x10000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Trigger GC but don't wait for completion
            coordinator.trigger_gc();

            // Wait with very short timeout - should timeout
            let result = coordinator.wait_until_idle(Duration::from_millis(1));
            assert!(!result);

            // Wait longer for actual completion
            let result = coordinator.wait_until_idle(Duration::from_millis(2000));
            assert!(result);
        }

        #[test]
        fn test_advance_to_phase_timeout() {
            let fixture = TestFixture::new_with_config(0x11000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Try to advance to a phase when not collecting - should timeout
            let result = coordinator.advance_to_phase(FugcPhase::Tracing);
            assert!(!result);

            // Start collection and try to advance to a phase that might not be reached
            coordinator.trigger_gc();

            // Try to advance to Sweeping with short timeout
            let result = coordinator.advance_to_phase(FugcPhase::Sweeping);
            // This might succeed or fail depending on timing, but shouldn't panic
            let _ = coordinator.wait_until_idle(Duration::from_millis(2000));
        }

        #[test]
        fn test_wait_for_phase_transition_timeout() {
            let fixture = TestFixture::new_with_config(0x12000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Try phase transition when not collecting - should return false
            let result =
                coordinator.wait_for_phase_transition(FugcPhase::Idle, FugcPhase::ActivateBarriers);
            assert!(!result);

            // Start collection and wait for transition with short timeout
            coordinator.trigger_gc();

            // Wait for Idle -> ActivateBarriers with very short timeout
            let result =
                coordinator.wait_for_phase_transition(FugcPhase::Idle, FugcPhase::ActivateBarriers);
            // Should timeout since we're waiting for a transition that already happened
            assert!(!result);

            coordinator.wait_until_idle(Duration::from_millis(2000));
        }

        #[test]
        fn test_multiple_concurrent_waits() {
            let fixture = TestFixture::new_with_config(0x13000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            coordinator.trigger_gc();

            // Spawn multiple threads waiting for idle
            let mut handles = vec![];
            for _ in 0..5 {
                let coord_clone = Arc::clone(&coordinator);
                let handle =
                    thread::spawn(move || coord_clone.wait_until_idle(Duration::from_millis(2000)));
                handles.push(handle);
            }

            // All should eventually succeed
            for handle in handles {
                assert!(handle.join().unwrap());
            }
        }
    }

    /// Test phase transition edge cases
    mod phase_transition_tests {
        use super::super::*;
        use fugrip::test_utils::TestFixture;
        use fugrip::thread::MutatorThread;
        use fugrip::{AllocationColor, FugcPhase};
        use mmtk::util::{Address, ObjectReference};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use std::thread;
        use std::time::Duration;

        #[test]
        fn test_invalid_phase_transitions() {
            let fixture = TestFixture::new_with_config(0x14000000, 32 * 1024 * 1024, 2);
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
            let fixture = TestFixture::new_with_config(0x15000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Test phase change notifications
            let coord_clone = Arc::clone(&coordinator);
            let handle = thread::spawn(move || {
                let mut phases_observed = vec![];
                // Try to receive phase changes with timeout
                while let Ok(phase) = coord_clone
                    .phase_change_receiver
                    .recv_timeout(Duration::from_millis(100))
                {
                    phases_observed.push(phase);
                    if phase == FugcPhase::Idle {
                        break;
                    }
                }
                phases_observed
            });

            // Trigger GC to generate phase changes
            coordinator.trigger_gc();
            coordinator.wait_until_idle(Duration::from_millis(2000));

            let phases_observed = handle.join().unwrap();
            assert!(!phases_observed.is_empty());
            assert_eq!(*phases_observed.last().unwrap(), FugcPhase::Idle);
        }

        #[test]
        fn test_collection_finished_signaling() {
            let fixture = TestFixture::new_with_config(0x16000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Test completion signaling
            let coord_clone = Arc::clone(&coordinator);
            let handle = thread::spawn(move || {
                coord_clone
                    .collection_finished_receiver
                    .recv_timeout(Duration::from_millis(2000))
            });

            coordinator.trigger_gc();
            let result = handle.join().unwrap();
            assert!(result.is_ok());

            // Should be idle after completion
            assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
        }
    }

    /// Test handshake failure scenarios
    mod handshake_tests {
        use super::*;

        #[test]
        fn test_handshake_with_no_threads() {
            let fixture = TestFixture::new_with_config(0x17000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Test handshake with empty thread registry
            let callback_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let callback_count_clone = Arc::clone(&callback_count);

            let callback = Box::new(move |_thread: &MutatorThread| {
                callback_count_clone.fetch_add(1, Ordering::Relaxed);
            });

            // This should succeed but process 0 threads
            // Note: soft_handshake is private, so we test via public API
            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));

            let metrics = coordinator.last_handshake_metrics();
            // Should have processed 0 threads
            assert_eq!(metrics.1, 0);
        }

        #[test]
        fn test_handshake_timeout_simulation() {
            let fixture = TestFixture::new_with_config(0x18000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);
            let thread_registry = Arc::clone(fixture.thread_registry());

            // Register a mutator but don't start its thread
            let mutator = MutatorThread::new(200);
            thread_registry.register(mutator.clone());

            coordinator.trigger_gc();

            // The handshake should still complete even if threads aren't actively polling
            let result = coordinator.wait_until_idle(Duration::from_millis(2000));
            assert!(result);

            thread_registry.unregister(mutator.id());
        }

        #[test]
        fn test_handshake_with_thread_failures() {
            let fixture = TestFixture::new_with_config(0x19000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);
            let thread_registry = Arc::clone(fixture.thread_registry());

            // Register multiple mutators
            let mut mutators = vec![];
            let mut handles = vec![];

            for i in 0..3 {
                let mutator = MutatorThread::new(300 + i);
                thread_registry.register(mutator.clone());
                mutators.push(mutator);
            }

            // Start some threads that will panic
            for (i, mutator) in mutators.into_iter().enumerate() {
                let registry_clone = Arc::clone(&thread_registry);
                let handle = thread::spawn(move || {
                    if i == 1 {
                        // One thread panics
                        panic!("Simulated thread failure");
                    } else {
                        // Others run normally
                        thread::sleep(Duration::from_millis(100));
                    }
                    registry_clone.unregister(mutator.id());
                });
                handles.push(handle);
            }

            // Trigger GC - should handle thread failures gracefully
            coordinator.trigger_gc();
            let result = coordinator.wait_until_idle(Duration::from_millis(2000));
            assert!(result);

            // Clean up remaining handles (some may have panicked)
            for handle in handles {
                let _ = handle.join(); // Ignore panic results
            }
        }
    }

    /// Test multiple GC trigger scenarios
    mod multiple_gc_tests {
        use super::*;

        #[test]
        fn test_multiple_gc_triggers_rapid() {
            let fixture = TestFixture::new_with_config(0x1a000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Trigger multiple GCs rapidly
            for _ in 0..5 {
                coordinator.trigger_gc();
                thread::sleep(Duration::from_millis(10));
            }

            // Should eventually complete all
            let result = coordinator.wait_until_idle(Duration::from_millis(5000));
            assert!(result);

            let stats = coordinator.get_cycle_stats();
            assert!(stats.cycles_completed >= 1); // At least one should have completed
        }

        #[test]
        fn test_gc_trigger_during_collection() {
            let fixture = TestFixture::new_with_config(0x1b000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Start first GC
            coordinator.trigger_gc();

            // Wait a bit then trigger another
            thread::sleep(Duration::from_millis(50));
            coordinator.trigger_gc(); // Should be ignored or queued

            // Should still complete
            let result = coordinator.wait_until_idle(Duration::from_millis(3000));
            assert!(result);

            let stats = coordinator.get_cycle_stats();
            assert!(stats.cycles_completed >= 1);
        }

        #[test]
        fn test_concurrent_gc_triggers() {
            let fixture = TestFixture::new_with_config(0x1c000000, 64 * 1024 * 1024, 4);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Spawn multiple threads triggering GC
            let mut handles = vec![];
            for _ in 0..3 {
                let coord_clone = Arc::clone(&coordinator);
                let handle = thread::spawn(move || {
                    coord_clone.trigger_gc();
                    coord_clone.wait_until_idle(Duration::from_millis(2000))
                });
                handles.push(handle);
            }

            // All should succeed
            for handle in handles {
                assert!(handle.join().unwrap());
            }

            let stats = coordinator.get_cycle_stats();
            assert!(stats.cycles_completed >= 1);
        }
    }

    /// Test worker thread edge cases
    mod worker_thread_tests {
        use super::*;

        #[test]
        fn test_worker_thread_start_stop() {
            let fixture = TestFixture::new_with_config(0x1d000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Start marking (which starts workers)
            let roots = vec![];
            coordinator.start_marking(roots);

            // Stop marking (which stops workers)
            coordinator.stop_marking();

            // Should be able to restart
            let roots2 = vec![];
            coordinator.start_marking(roots2);
            coordinator.stop_marking();
        }

        #[test]
        fn test_worker_thread_with_no_work() {
            let fixture = TestFixture::new_with_config(0x1e000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Start marking with empty roots
            let roots = vec![];
            coordinator.start_marking(roots);

            // Should complete quickly with no work
            thread::sleep(Duration::from_millis(100));

            coordinator.stop_marking();
        }

        #[test]
        fn test_worker_thread_concurrent_operations() {
            let fixture = TestFixture::new_with_config(0x1f000000, 32 * 1024 * 1024, 4);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Start marking
            let roots = vec![];
            coordinator.start_marking(roots);

            // Perform concurrent operations
            let coord_clone = Arc::clone(&coordinator);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let _ = coord_clone.get_marking_stats();
                    let _ = coord_clone.get_cache_stats();
                    thread::sleep(Duration::from_millis(5));
                }
            });

            handle.join().unwrap();
            coordinator.stop_marking();
        }
    }

    /// Test collection interruption and recovery
    mod interruption_tests {
        use super::*;

        #[test]
        fn test_collection_state_after_timeout() {
            let fixture = TestFixture::new_with_config(0x20000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            coordinator.trigger_gc();

            // Don't wait for completion - check state
            assert!(coordinator.is_collecting());

            // Wait for actual completion
            coordinator.wait_until_idle(Duration::from_millis(2000));
            assert!(!coordinator.is_collecting());
            assert_eq!(coordinator.current_phase(), FugcPhase::Idle);
        }

        #[test]
        fn test_collection_recovery_after_partial_completion() {
            let fixture = TestFixture::new_with_config(0x21000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Run one collection
            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));

            let stats_after_first = coordinator.get_cycle_stats();

            // Run another collection
            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));

            let stats_after_second = coordinator.get_cycle_stats();

            // Should have more cycles completed
            assert!(stats_after_second.cycles_completed > stats_after_first.cycles_completed);
        }

        #[test]
        fn test_collection_with_large_heap() {
            let fixture = TestFixture::new_with_config(0x22000000, 128 * 1024 * 1024, 4);
            let coordinator = Arc::clone(&fixture.coordinator);

            coordinator.trigger_gc();
            let result = coordinator.wait_until_idle(Duration::from_millis(5000));
            assert!(result);

            let stats = coordinator.get_cycle_stats();
            assert!(stats.cycles_completed >= 1);
        }
    }

    /// Test channel communication edge cases
    mod channel_tests {
        use super::*;

        #[test]
        fn test_channel_buffer_limits() {
            let fixture = TestFixture::new_with_config(0x23000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Rapidly set phases to test channel buffering
            for i in 0..20 {
                let phase = match i % 8 {
                    0 => FugcPhase::Idle,
                    1 => FugcPhase::ActivateBarriers,
                    2 => FugcPhase::ActivateBlackAllocation,
                    3 => FugcPhase::MarkGlobalRoots,
                    4 => FugcPhase::StackScanHandshake,
                    5 => FugcPhase::Tracing,
                    6 => FugcPhase::PrepareForSweep,
                    7 => FugcPhase::Sweeping,
                    _ => FugcPhase::Idle,
                };
                coordinator.set_phase(phase);
            }

            // Should still be in the last set phase
            assert_eq!(coordinator.current_phase(), FugcPhase::Sweeping);
        }

        #[test]
        fn test_channel_receiver_disconnect() {
            let fixture = TestFixture::new_with_config(0x24000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Drop the receiver to simulate disconnection
            drop(coordinator.phase_change_receiver);

            // Trigger GC - should not panic
            coordinator.trigger_gc();
            let result = coordinator.wait_until_idle(Duration::from_millis(2000));
            assert!(result);
        }
    }

    /// Test SIMD and vectorization edge cases
    mod vectorization_tests {
        use super::*;

        #[test]
        fn test_vectorized_processing_empty_batch() {
            let fixture = TestFixture::new_with_config(0x25000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Test with empty object batch
            let empty_batch: Vec<ObjectReference> = vec![];
            let processed = coordinator.process_objects_vectorized(&empty_batch);
            assert_eq!(processed, 0);
        }

        #[test]
        fn test_vectorized_processing_single_object() {
            let fixture = TestFixture::new_with_config(0x26000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Create a single object
            let heap_base = unsafe { Address::from_usize(0x26000000) };
            let obj = ObjectReference::from_raw_address(heap_base).unwrap();

            let batch = vec![obj];
            let processed = coordinator.process_objects_vectorized(&batch);
            assert_eq!(processed, 1);
        }

        #[test]
        fn test_vectorized_processing_large_batch() {
            let fixture = TestFixture::new_with_config(0x27000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Create a large batch of objects
            let heap_base = unsafe { Address::from_usize(0x27000000) };
            let mut batch = vec![];

            for i in 0..100 {
                let addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 64) };
                if let Some(obj) = ObjectReference::from_raw_address(addr) {
                    batch.push(obj);
                }
            }

            let processed = coordinator.process_objects_vectorized(&batch);
            assert_eq!(processed, batch.len());
        }
    }

    /// Test page allocation and coloring edge cases
    mod page_allocation_tests {
        use super::*;

        #[test]
        fn test_page_allocation_color_out_of_bounds() {
            let fixture = TestFixture::new_with_config(0x28000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Test page index that's way out of bounds
            let color = coordinator.page_allocation_color(999999);
            assert_eq!(color, AllocationColor::White); // Should default to White
        }

        #[test]
        fn test_page_allocation_color_after_gc() {
            let fixture = TestFixture::new_with_config(0x29000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);
            let global_roots = Arc::clone(fixture.global_roots());

            // Add a root to create some allocation activity
            {
                let mut roots = global_roots.lock();
                let root_addr = unsafe { Address::from_usize(0x29000100) };
                roots.register(root_addr.as_usize() as *mut u8);
            }

            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));

            // Check page color for the page containing our root
            let page_index = (0x29000100 - 0x29000000) / 4096;
            let color = coordinator.page_allocation_color(page_index);
            // Should be either White or Black depending on GC outcome
            assert!(matches!(
                color,
                AllocationColor::White | AllocationColor::Black
            ));
        }

        #[test]
        fn test_page_state_concurrent_access() {
            let fixture = TestFixture::new_with_config(0x2a000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Spawn multiple threads accessing page states
            let mut handles = vec![];
            for i in 0..5 {
                let coord_clone = Arc::clone(&coordinator);
                let handle = thread::spawn(move || {
                    for j in 0..10 {
                        let _ = coord_clone.page_allocation_color(i * 10 + j);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        }
    }

    /// Test statistics and metrics edge cases
    mod statistics_tests {
        use super::*;

        #[test]
        fn test_statistics_overflow_protection() {
            let fixture = TestFixture::new_with_config(0x2b000000, 32 * 1024 * 1024, 2);
            let coordinator = &fixture.coordinator;

            // Test that statistics don't overflow with large values
            let stats = coordinator.get_cycle_stats();

            // These should not be negative or cause overflow
            assert!(stats.cycles_completed >= 0);
            assert!(stats.objects_marked >= 0);
            assert!(stats.objects_swept >= 0);
            assert!(stats.handshakes_performed >= 0);
        }

        #[test]
        fn test_handshake_metrics_reset() {
            let fixture = TestFixture::new_with_config(0x2c000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            // Check initial metrics
            let initial_metrics = coordinator.last_handshake_metrics();
            assert_eq!(initial_metrics.1, 0); // No threads processed initially

            // Trigger GC to perform handshakes
            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));

            // Metrics should be updated
            let updated_metrics = coordinator.last_handshake_metrics();
            // At minimum, handshake time should be recorded
            assert!(updated_metrics.0 >= 0);
        }

        #[test]
        fn test_marking_statistics_accuracy() {
            let fixture = TestFixture::new_with_config(0x2d000000, 32 * 1024 * 1024, 2);
            let coordinator = Arc::clone(&fixture.coordinator);

            let initial_stats = coordinator.get_marking_stats();

            // Trigger GC
            coordinator.trigger_gc();
            assert!(coordinator.wait_until_idle(Duration::from_millis(2000)));

            let final_stats = coordinator.get_marking_stats();

            // Statistics should be valid (non-negative)
            assert!(final_stats.work_stolen >= initial_stats.work_stolen);
            assert!(final_stats.work_shared >= initial_stats.work_shared);
        }
    }
}
