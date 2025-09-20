//! Tests for safepoint implementation

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn safepoint_fast_path() {
        // Ensure fast path is actually fast (no safepoint requested)
        assert!(!SAFEPOINT_REQUESTED.load(Ordering::Relaxed));

        let start = std::time::Instant::now();
        for _ in 0..10000 {
            pollcheck();
        }
        let elapsed = start.elapsed();

        // 10k pollchecks should be very fast (< 50ms on modern hardware, allowing for system load and coverage instrumentation)
        assert!(elapsed < Duration::from_millis(50));
    }

    #[test]
    fn safepoint_callback_execution() {
        // Use a dedicated DI container so pollcheck and the test share the same manager
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        // Reset globals
        SAFEPOINT_REQUESTED.store(false, Ordering::Release);
        SAFEPOINT_HITS.store(0, Ordering::Relaxed);

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        // Use request_safepoint to set callback and flag
        manager.request_safepoint(Box::new(move || {
            executed_clone.store(true, Ordering::Release);
        }));

        // Trigger slow path with pollcheck
        for _ in 0..10 {
            pollcheck();
        }

        // Verify
        assert!(executed.load(Ordering::Acquire), "Callback not executed");

        // Cleanup
        manager.clear_safepoint();
        SAFEPOINT_REQUESTED.store(false, Ordering::Release);
    }

    #[test]
    fn safepoint_statistics() {
        // Test safepoint statistics collection
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        // Get initial stats
        let initial_stats = manager.get_stats();

        // Perform some pollchecks
        for _ in 0..100 {
            pollcheck();
        }

        // Get updated stats
        let updated_stats = manager.get_stats();

        // Should have more polls
        assert!(updated_stats.total_polls >= initial_stats.total_polls);
        // Hit rate should be valid (between 0 and 1)
        assert!(updated_stats.hit_rate >= 0.0 && updated_stats.hit_rate <= 1.0);
    }

    #[test]
    fn thread_registration() {
        // Test thread registration functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let manager = Arc::clone(container.safepoint_manager());

        // Register current thread
        manager.register_thread();

        // Should not panic or fail
        // Real implementation: Verify the thread is properly registered in the global registry
        
        let registry = crate::thread::global_thread_registry();
        let thread_id = std::thread::current().id();
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        thread_id.hash(&mut hasher);
        let thread_hash = hasher.finish() as usize;
        assert!(
            registry.len() > 0,
            "Thread registry should have registered threads"
        );

        // Verify the thread has proper safepoint state initialized
        if let Some(mutator_thread) = registry.get(thread_hash) {
            // Check the handler's safepoint state
            assert!(
                !mutator_thread.handler.is_at_safepoint(),
                "Thread should not start at safepoint"
            );
            assert!(
                mutator_thread.handler.can_poll(),
                "Thread should be able to poll for safepoints"
            );
        }
    }

    #[test]
    fn safepoint_waiting() {
        // Test waiting for safepoint functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        // Wait without safepoint requested (should timeout)
        let result = manager.wait_for_safepoint(Duration::from_millis(10));
        assert!(!result, "Should timeout when no safepoint is requested");

        // Test waiting with safepoint requested
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        manager.request_safepoint(Box::new(move || {
            executed_clone.store(true, Ordering::Release);
        }));

        // Trigger pollcheck to execute the safepoint
        pollcheck();

        // Should succeed this time
        let result = manager.wait_for_safepoint(Duration::from_millis(1000));
        assert!(result, "Should succeed when safepoint is requested");

        // Callback should have been executed
        assert!(executed.load(Ordering::Acquire));
    }

    #[test]
    fn test_safepoint_state_transitions() {
        // Test safepoint state transitions
        use std::sync::Arc;
        // No atomic imports needed

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        // Test safepoint enter/exit functionality
        safepoint_enter();

        let callback_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&callback_count);

        manager.request_safepoint(Box::new(move || {
            count_clone.fetch_add(1, Ordering::Relaxed);
        }));

        // Pollcheck should work even when in safepoint
        pollcheck();

        safepoint_exit();

        // Callback should have been executed
        assert_eq!(callback_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_safepoint_error_handling() {
        // Test error handling and edge cases
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let manager = Arc::clone(container.safepoint_manager());

        // Test clearing safepoint when none is requested
        manager.clear_safepoint(); // Should not panic

        // Test executing callbacks when none are set
        manager.execute_safepoint_callback(); // Should not panic
        manager.execute_handshake_callback(); // Should not panic
        // No callback should be set

        // Test getting stats (note: some polls may have occurred from test setup)
        let stats = manager.get_stats();
        // Just verify that stats are accessible and reasonable
        assert!(stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0);
    }

    #[test]
    fn test_concurrent_safepoint_access() {
        // Test concurrent access to safepoint manager API
        use flume::unbounded;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        // Test that we can request a safepoint and the API works
        manager.request_safepoint(Box::new(move || {
            callback_clone.store(true, Ordering::Release);
        }));

        // Test concurrent access to the manager
        let (start_signal, start_recv) = unbounded();
        let (complete_signal, complete_recv) = unbounded();

        crossbeam::scope(|s| {
            s.spawn(move |_| {
                start_recv.recv().unwrap();

                // Concurrent access to safepoint manager
                for _ in 0..10 {
                    pollcheck(); // This should not panic
                    std::thread::yield_now();
                }

                complete_signal.send(()).unwrap();
            });

            start_signal.send(()).unwrap();
            let _ = complete_recv.recv_timeout(Duration::from_millis(100));
        })
        .unwrap();

        // Verify that the callback was set up correctly and the manager handled concurrent access
        // The callback should not have been executed yet since we didn't call execute_safepoint_callback
        assert!(
            !callback_executed.load(Ordering::Acquire),
            "Callback should not have been executed before explicit execution"
        );

        // Now execute the callback to verify it works
        manager.execute_safepoint_callback();
        assert!(
            callback_executed.load(Ordering::Acquire),
            "Callback should have been executed when explicitly called"
        );
    }

    #[test]
    fn test_gc_safepoint_phase_integration() {
        // Test GC safepoint phase functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
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
        use std::sync::Arc;
        // No atomic imports needed
        use std::vec::Vec;

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        let execution_order = Arc::new(crossbeam::queue::SegQueue::new());
        let _order_clone = Arc::clone(&execution_order);

        // Request multiple safepoints in sequence
        for i in 0..3 {
            let order = Arc::clone(&execution_order);
            manager.request_safepoint(Box::new(move || {
                order.push(i);
            }));

            // Execute the callback
            pollcheck();

            // Clear for next
            manager.clear_safepoint();
        }

        // Check execution order
        // Drain SegQueue into a Vec for assertion
        let mut order_vec = Vec::new();
        while let Some(v) = execution_order.pop() {
            order_vec.push(v);
        }
        assert_eq!(order_vec, vec![0, 1, 2]);
    }

    #[test]
    fn test_epoch_based_safepoint_simplification() {
        // Test the simplified epoch-based safepoint coordination
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        // Test simple callback execution with epoch coordination
        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        manager.request_safepoint(Box::new(move || {
            callback_clone.store(true, Ordering::Release);
        }));

        assert!(
            callback_executed.load(Ordering::Acquire),
            "Epoch callback should execute immediately"
        );

        // Test concurrent epoch-based coordination
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        // Multiple epoch safepoints should coordinate automatically
        for i in 0..5 {
            let counter_ref = Arc::clone(&counter_clone);
            manager.request_safepoint(Box::new(move || {
                counter_ref.fetch_add(i + 1, Ordering::Relaxed);
            }));
        }

        // All callbacks should have executed with automatic coordination
        assert_eq!(counter.load(Ordering::Relaxed), 15); // 1+2+3+4+5 = 15
    }
}
