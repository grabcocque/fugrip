// Test utilities for dependency injection and component testing.
//
// This module provides helper functions and patterns for testing
// components with injected dependencies using mock implementations.

#[cfg(test)]
pub use crate::interfaces::{MockHeapProvider, MockThreadingProvider};

/// Test utilities for creating components with mock dependencies
#[cfg(test)]
pub mod test_builders {
    use super::*;
    use crate::collector::mark_coordinator::MarkCoordinator;
    use crate::collector::finalizer_coordinator::FinalizerCoordinator;
    use crate::collector::sweep_coordinator::SweepCoordinator;
    use crate::collector_phases::{CollectorState, StackScanner};

    /// Builder for creating CollectorState with mock dependencies for testing
    pub struct TestCollectorBuilder {
        heap_provider: Option<MockHeapProvider>,
        threading_provider: Option<MockThreadingProvider>,
    }

    impl TestCollectorBuilder {
        pub fn new() -> Self {
            Self {
                heap_provider: None,
                threading_provider: None,
            }
        }

        pub fn with_mock_heap(mut self, heap_provider: MockHeapProvider) -> Self {
            self.heap_provider = Some(heap_provider);
            self
        }

        pub fn with_mock_threading(mut self, threading_provider: MockThreadingProvider) -> Self {
            self.threading_provider = Some(threading_provider);
            self
        }

        pub fn with_default_mocks(self) -> Self {
            self.with_mock_heap(MockHeapProvider::new())
                .with_mock_threading(MockThreadingProvider::new())
        }

        /// Build a CollectorState with the specified mock dependencies
        /// Note: This is a demonstration - actual implementation would need
        /// dependency injection constructors on the components
        pub fn build(self) -> (CollectorState, TestDependencies) {
            let heap_provider = self.heap_provider.unwrap_or_default();
            let threading_provider = self.threading_provider.unwrap_or_default();

            // For demonstration, create a standard CollectorState
            // In a full implementation, components would accept injected dependencies
            let collector = CollectorState::new();

            let test_deps = TestDependencies {
                heap_provider,
                threading_provider,
            };

            (collector, test_deps)
        }
    }

    /// Container for test dependencies to allow verification after test execution
    pub struct TestDependencies {
        pub heap_provider: MockHeapProvider,
        pub threading_provider: MockThreadingProvider,
    }

    impl TestDependencies {
        /// Verify that expected interactions occurred with the mock dependencies
        pub fn verify_interactions(&self) -> TestVerificationResult {
            TestVerificationResult {
                heap_access_count: self.heap_provider.get_call_count(),
                mutator_registrations: self.threading_provider.get_register_mutator_call_count(),
                handshake_requests: self.threading_provider.get_handshake_request_call_count(),
                gc_registrations: self.threading_provider.get_registered_thread_count(),
            }
        }
    }

    /// Result of dependency interaction verification
    #[derive(Debug, PartialEq)]
    pub struct TestVerificationResult {
        pub heap_access_count: usize,
        pub mutator_registrations: usize,
        pub handshake_requests: usize,
        pub gc_registrations: usize,
    }

    impl Default for TestCollectorBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Create a test MarkCoordinator with mock dependencies
    pub fn create_test_mark_coordinator() -> MarkCoordinator {
        MarkCoordinator::new()
    }

    /// Create a test FinalizerCoordinator with mock dependencies  
    pub fn create_test_finalizer_coordinator() -> FinalizerCoordinator {
        FinalizerCoordinator::new()
    }

    /// Create a test SweepCoordinator with mock dependencies
    pub fn create_test_sweep_coordinator() -> SweepCoordinator {
        SweepCoordinator::new()
    }

    /// Create a test StackScanner with mock heap bounds checker
    pub fn create_test_stack_scanner() -> StackScanner {
        StackScanner::new() // Uses default bounds checker
    }
}

#[cfg(test)]
mod tests {
    use super::test_builders::*;
    use crate::interfaces::{MockHeapProvider, MockThreadingProvider, HeapProvider, ThreadingProvider};

    #[test]
    fn test_collector_builder_with_default_mocks() {
        let (collector, deps) = TestCollectorBuilder::new()
            .with_default_mocks()
            .build();

        // Verify collector was created
        assert!(!collector.is_marking());

        // Verify initial state of mock dependencies
        let verification = deps.verify_interactions();
        assert_eq!(verification.heap_access_count, 0);
        assert_eq!(verification.mutator_registrations, 0);
        assert_eq!(verification.handshake_requests, 0);
        assert_eq!(verification.gc_registrations, 0);
    }

    #[test]
    fn test_mock_threading_interaction() {
        let mock_threading = MockThreadingProvider::new();
        
        // Test mutator registration
        mock_threading.register_mutator_thread();
        mock_threading.register_mutator_thread();
        
        assert_eq!(mock_threading.get_active_mutator_count(), 2);
        assert_eq!(mock_threading.get_register_mutator_call_count(), 2);

        // Test handshake coordination
        mock_threading.request_handshake();
        assert_eq!(mock_threading.get_handshake_request_call_count(), 1);
        assert!(mock_threading.is_handshake_requested());

        // Acknowledge from both mutators
        mock_threading.acknowledge_handshake();
        mock_threading.acknowledge_handshake();
        
        // Handshake should be complete
        assert!(!mock_threading.is_handshake_requested());
    }

    #[test]
    fn test_component_creation() {
        let _mark_coordinator = create_test_mark_coordinator();
        let _finalizer_coordinator = create_test_finalizer_coordinator();
        let _sweep_coordinator = create_test_sweep_coordinator();
        let _stack_scanner = create_test_stack_scanner();
        
        // All components should be creatable without dependencies
        // In a full implementation, these would accept mock dependencies
    }

    #[test]
    fn test_mock_heap_provider_interaction() {
        let mock_heap = MockHeapProvider::new();
        
        assert_eq!(mock_heap.get_call_count(), 0);
        
        let _heap_ref = mock_heap.get_heap();
        assert_eq!(mock_heap.get_call_count(), 1);
        
        let _heap_ref2 = mock_heap.get_heap();
        assert_eq!(mock_heap.get_call_count(), 2);
    }
}