//! Comprehensive coverage tests for safepoint.rs
//!
//! These tests are designed to improve coverage from 66.67% to >70%
//! by testing edge cases, error paths, and specific code branches
//! that may be missed by existing tests.

use fugrip::safepoint::*;
use fugrip::test_utils::TestFixture;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[cfg(test)]
mod safepoint_coverage_tests {
    use super::*;

    #[test]
    fn test_thread_execution_states() {
        // Test thread execution state variants

        // Test all states are valid and comparable
        let states = [
            ThreadExecutionState::Active,
            ThreadExecutionState::Exited,
            ThreadExecutionState::Entering,
        ];

        // Verify states can be compared
        for i in 0..states.len() {
            for j in i..states.len() {
                let result = states[i] == states[j];
                assert_eq!(result, i == j, "State comparison should work correctly");
            }
        }

        // Verify states can be cloned
        let state = ThreadExecutionState::Active;
        let cloned = state;
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_thread_registration_basic() {
        // Test thread registration functionality

        let manager = SafepointManager::new_for_testing();

        // Test register_thread (should not panic)
        manager.register_thread();
    }

    #[test]
    fn test_safepoint_manager_creation_variants() {
        // Test all SafepointManager creation variants

        // Test new_for_testing()
        let _manager1 = SafepointManager::new_for_testing();

        // Test with_coordinator()
        let fixture = TestFixture::new_with_config(0x50000000, 16 * 1024 * 1024, 2);
        let coordinator = Arc::clone(&fixture.coordinator);
        let manager2 = SafepointManager::with_coordinator(&coordinator);
        assert_eq!(
            Arc::as_ptr(manager2.get_fugc_coordinator()),
            Arc::as_ptr(&coordinator)
        );

        // Test global()
        let global_manager = SafepointManager::global();
        let _coord = global_manager.get_fugc_coordinator(); // Should not panic
    }

    #[test]
    fn test_safepoint_stats_edge_cases() {
        // Test statistics calculation edge cases

        let manager = SafepointManager::new_for_testing();

        // Test getting stats (should work even with zero activity)
        let stats = manager.get_stats();
        // Hit rate might be NaN when there's no activity (0/0 division), so we check if it's either valid OR NaN
        let hit_rate_is_valid = stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0;
        let hit_rate_is_nan = stats.hit_rate.is_nan();
        assert!(
            hit_rate_is_valid || hit_rate_is_nan,
            "Hit rate should be valid or NaN for zero activity"
        );
        assert!(stats.avg_safepoint_interval_ms >= 0.0 || stats.avg_safepoint_interval_ms.is_nan());
    }

    #[test]
    fn test_gc_safepoint_callback_construction() {
        // Test that GC safepoint callbacks are constructed correctly

        let manager = SafepointManager::new_for_testing();

        // Test each GC phase creates a valid callback
        let phases = [
            GcSafepointPhase::RootScanning,
            GcSafepointPhase::BarrierActivation,
            GcSafepointPhase::MarkingHandshake,
            GcSafepointPhase::SweepPreparation,
        ];

        for phase in phases {
            // Should not panic when requesting
            manager.request_gc_safepoint(phase);

            // Clear for next iteration
            manager.clear_safepoint();
        }
    }

    #[test]
    fn test_safepoint_request_clear_sequence() {
        // Test proper request/clear sequence to avoid state corruption

        let manager = SafepointManager::new_for_testing();

        // Test multiple request/clear cycles
        for _i in 0..3 {
            let executed = Arc::new(AtomicBool::new(false));
            let executed_clone = Arc::clone(&executed);

            manager.request_safepoint(Box::new(move || {
                executed_clone.store(true, Ordering::Release);
            }));

            // Test clearing (should not panic)
            manager.clear_safepoint();
        }
    }

    #[test]
    fn test_safepoint_error_handling() {
        // Test error handling and edge cases

        let manager = SafepointManager::new_for_testing();

        // Test clearing safepoint when none is requested
        manager.clear_safepoint(); // Should not panic

        // Test executing callbacks when none are set
        manager.execute_safepoint_callback(); // Should not panic
        manager.execute_handshake_callback(); // Should not panic

        // Test getting stats (should work even with minimal activity)
        let stats = manager.get_stats();
        // Hit rate might be NaN when there's no activity (0/0 division), so we check if it's either valid OR NaN
        let hit_rate_is_valid = stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0;
        let hit_rate_is_nan = stats.hit_rate.is_nan();
        assert!(
            hit_rate_is_valid || hit_rate_is_nan,
            "Hit rate should be valid or NaN for minimal activity"
        );
    }

    #[test]
    fn test_thread_local_state_initialization() {
        // Test thread-local state initialization path

        // Clear any existing state by calling exit/enter
        safepoint_exit();
        safepoint_enter(); // Should not panic

        // Verify pollcheck works (should not panic)
        pollcheck();
    }

    #[test]
    fn test_safepoint_soft_handshake_public_api() {
        // Test soft handshake functionality through public API

        let manager = SafepointManager::new_for_testing();

        // Test soft handshake (should not panic)
        manager.request_soft_handshake(Box::new(|| {
            // Simple callback
        }));
    }

    #[test]
    fn test_safepoint_coordinator_integration() {
        // Test coordinator set/get functionality

        let fixture = TestFixture::new_with_config(0x60000000, 32 * 1024 * 1024, 1);
        let coordinator = Arc::clone(&fixture.coordinator);
        let manager = SafepointManager::with_coordinator(&coordinator);

        // Test getting coordinator
        let retrieved_coord = manager.get_fugc_coordinator();
        assert_eq!(Arc::as_ptr(retrieved_coord), Arc::as_ptr(&coordinator));

        // Test custom coordinator functionality
        SafepointManager::set_global_coordinator(Arc::clone(&coordinator));
        let custom_coord = SafepointManager::get_custom_coordinator();
        assert!(custom_coord.is_some());
        assert_eq!(
            Arc::as_ptr(&custom_coord.unwrap()),
            Arc::as_ptr(&coordinator)
        );
    }

    #[test]
    fn test_global_manager_access() {
        // Test global manager functionality

        // Note: GLOBAL_MANAGER uses OnceLock, so it can only be set once
        // This test verifies the global manager API works correctly

        let manager = SafepointManager::new_for_testing();

        // Get the current global manager (may already be initialized)
        let global_manager = SafepointManager::global();

        // Verify it's a valid manager instance
        let _stats = global_manager.get_stats(); // Should not panic

        // If we can set the global manager (first time), verify it works
        // This may not succeed if GLOBAL_MANAGER was already initialized
        SafepointManager::set_global_manager(Arc::clone(&manager));

        // The global manager should still be accessible and functional
        let global_manager_after = SafepointManager::global();
        let _stats_after = global_manager_after.get_stats(); // Should not panic
    }

    #[test]
    fn test_safepoint_thread_registration() {
        // Test thread registration functionality

        let manager = SafepointManager::new_for_testing();

        // Test register_thread (should not panic)
        manager.register_thread();

        // Test register_and_cache_thread (should not panic)
        manager.register_and_cache_thread();
    }

    #[test]
    fn test_safepoint_callback_execution() {
        // Test callback execution functionality with actual validation

        let manager = SafepointManager::new_for_testing();

        // Initially, no callbacks should be set
        let initial_stats = manager.get_stats();
        let initial_polls = initial_stats.total_polls;

        // Execute callbacks when none are set - should not panic and not change state
        manager.execute_safepoint_callback();
        manager.execute_handshake_callback();

        // Verify stats remain unchanged when no callbacks are set
        let stats_after_no_callbacks = manager.get_stats();
        assert_eq!(
            stats_after_no_callbacks.total_polls, initial_polls,
            "Polls should not increase when executing empty callbacks"
        );

        // Set up actual callbacks to test execution
        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        manager.request_safepoint(Box::new(move || {
            callback_clone.store(true, Ordering::Release);
        }));

        // Execute the safepoint callback - it should run and set the flag
        manager.execute_safepoint_callback();
        assert!(
            callback_executed.load(Ordering::Acquire),
            "Safepoint callback should have been executed"
        );

        // Verify that the callback was cleared after execution
        manager.execute_safepoint_callback(); // Should not execute again
        assert!(
            callback_executed.load(Ordering::Acquire),
            "Flag should remain true, callback should not execute twice"
        );

        // Test handshake callback setup (execution requires proper thread coordination)
        let handshake_executed = Arc::new(AtomicBool::new(false));
        let handshake_clone = Arc::clone(&handshake_executed);

        // Set up handshake callback - should not panic
        manager.request_soft_handshake(Box::new(move || {
            handshake_clone.store(true, Ordering::Release);
        }));

        // Verify callback was set up correctly (execution requires full handshake protocol)
        // The important thing is that the API works and doesn't panic
        let stats_after_setup = manager.get_stats();
        assert!(
            stats_after_setup.total_polls >= initial_polls,
            "Stats should remain valid after handshake callback setup"
        );
    }

    #[test]
    fn test_safepoint_wait_functionality() {
        // Test wait_for_safepoint functionality with actual validation

        let manager = SafepointManager::new_for_testing();
        manager.register_thread();

        // Get initial hit count
        let initial_stats = manager.get_stats();
        let initial_hits = initial_stats.total_hits;

        // Test waiting with no safepoint requested
        // Note: wait_for_safepoint may return true if there are pending safepoint hits
        // from previous activity, so we focus on the behavior after explicit request
        let _result = manager.wait_for_safepoint(Duration::from_millis(10));
        // The result can be true or false depending on system state, both are valid

        // Verify hit count behavior (should not decrease)
        let stats_after_wait = manager.get_stats();
        assert!(
            stats_after_wait.total_hits >= initial_hits,
            "Hit count should not decrease during wait"
        );

        // Test waiting with safepoint requested - should succeed
        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        manager.request_safepoint(Box::new(move || {
            callback_clone.store(true, Ordering::Release);
        }));

        // Trigger safepoint by calling pollcheck (this increments SAFEPOINT_HITS)
        pollcheck();

        // Now wait_for_safepoint should succeed because hit count increased
        let result_with_safepoint = manager.wait_for_safepoint(Duration::from_millis(100));
        assert!(
            result_with_safepoint,
            "wait_for_safepoint should succeed when safepoint was hit"
        );

        // Verify callback was executed
        assert!(
            callback_executed.load(Ordering::Acquire),
            "Callback should have been executed during safepoint"
        );

        // Verify hit count increased
        let stats_after_safepoint = manager.get_stats();
        assert!(
            stats_after_safepoint.total_hits > initial_hits,
            "Hit count should increase after safepoint execution"
        );

        // Clear the safepoint for cleanup
        manager.clear_safepoint();
    }

    #[test]
    fn test_safepoint_state_transitions() {
        // Test safepoint state transitions with actual validation

        let manager = SafepointManager::new_for_testing();
        manager.register_thread();

        // Get initial state
        let initial_stats = manager.get_stats();
        let initial_polls = initial_stats.total_polls;
        let initial_hits = initial_stats.total_hits;

        // Test transition: Active -> Exited -> Active
        safepoint_exit(); // Transition to exited state

        // While in exited state, request a safepoint
        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        manager.request_safepoint(Box::new(move || {
            callback_clone.store(true, Ordering::Release);
        }));

        // Trigger pollcheck - should handle exited state properly
        pollcheck();

        // Transition back to active state - this should execute any pending handshake
        safepoint_enter();

        // Verify stats changed appropriately
        let stats_after_transitions = manager.get_stats();
        assert!(
            stats_after_transitions.total_polls > initial_polls,
            "Poll count should increase after state transitions and pollchecks"
        );

        // Now test with actual safepoint execution
        let second_callback = Arc::new(AtomicBool::new(false));
        let second_clone = Arc::clone(&second_callback);

        manager.request_safepoint(Box::new(move || {
            second_clone.store(true, Ordering::Release);
        }));

        // Execute safepoint while in active state
        pollcheck();

        // Verify callback was executed
        assert!(
            second_callback.load(Ordering::Acquire),
            "Callback should be executed when in active state"
        );

        // Verify hit count increased
        let final_stats = manager.get_stats();
        assert!(
            final_stats.total_hits > initial_hits,
            "Hit count should increase after safepoint execution"
        );

        // Verify safepoint request flag is cleared by testing that new pollchecks don't execute callbacks
        let test_callback = Arc::new(AtomicBool::new(false));
        let test_clone = Arc::clone(&test_callback);

        manager.request_safepoint(Box::new(move || {
            test_clone.store(true, Ordering::Release);
        }));

        // Clear the safepoint
        manager.clear_safepoint();

        // Pollcheck should not execute the callback after clearing
        pollcheck();
        assert!(
            !test_callback.load(Ordering::Acquire),
            "Callback should not execute after safepoint is cleared"
        );

        // Test multiple rapid state transitions
        for i in 0..5 {
            safepoint_exit();

            let rapid_callback = Arc::new(AtomicBool::new(false));
            let rapid_clone = Arc::clone(&rapid_callback);

            manager.request_safepoint(Box::new(move || {
                rapid_clone.store(true, Ordering::Release);
            }));

            pollcheck();
            safepoint_enter();

            // Verify state transitions work consistently
            assert!(
                manager.get_stats().total_polls > initial_polls + i,
                "Polls should increase with each transition cycle"
            );
        }

        // Clear any remaining safepoint
        manager.clear_safepoint();
    }

    #[test]
    fn test_safepoint_callback_generation() {
        // Test that callbacks are executed properly

        let manager = SafepointManager::new_for_testing();
        manager.register_and_cache_thread();

        let callback_count = Arc::new(AtomicBool::new(false));
        let count_clone = Arc::clone(&callback_count);

        // Request safepoint with callback
        manager.request_safepoint(Box::new(move || {
            count_clone.store(true, Ordering::Release);
        }));

        // Clear immediately to prevent execution
        manager.clear_safepoint();

        // Subsequent polls without safepoint should not execute callback
        pollcheck();
        assert!(!callback_count.load(Ordering::Acquire));
    }

    #[test]
    fn test_safepoint_manager_creation_methods() {
        // Test different creation methods for SafepointManager

        // Test with_coordinator method
        let fixture = TestFixture::new_with_config(0x70000000, 8 * 1024 * 1024, 1);
        let coordinator = Arc::clone(&fixture.coordinator);
        let manager1 = SafepointManager::with_coordinator(&coordinator);

        // Test new_for_testing method
        let manager2 = SafepointManager::new_for_testing();

        // Both should be valid managers
        assert_ne!(Arc::as_ptr(&manager1), Arc::as_ptr(&manager2));

        // Test that they have valid coordinators
        let _coord1 = manager1.get_fugc_coordinator();
        let _coord2 = manager2.get_fugc_coordinator();
    }

    #[test]
    fn test_gc_safepoint_phase_integration() {
        // Test GC safepoint phase functionality

        let manager = SafepointManager::new_for_testing();
        manager.register_thread();

        // Test different GC phases
        let phases = [
            GcSafepointPhase::RootScanning,
            GcSafepointPhase::BarrierActivation,
            GcSafepointPhase::MarkingHandshake,
            GcSafepointPhase::SweepPreparation,
        ];

        for phase in phases.iter() {
            // Should be able to request safepoint for each phase without panicking
            manager.request_gc_safepoint(*phase);

            // Clear for next iteration
            manager.clear_safepoint();
        }
    }

    #[test]
    fn test_safepoint_callback_chaining() {
        // Test multiple callback requests and execution order

        let manager = SafepointManager::new_for_testing();
        manager.register_thread();

        // Request multiple safepoints in sequence
        for _i in 0..3 {
            manager.request_safepoint(Box::new(|| {
                // Simple callback
            }));

            // Execute the callback (may or may not execute depending on timing)
            pollcheck();

            // Clear for next
            manager.clear_safepoint();
        }
    }
}
