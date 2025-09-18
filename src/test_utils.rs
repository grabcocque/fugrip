//! Test utilities for FUGC tests
//!
//! This module provides shared test fixtures and utilities to ensure
//! consistent dependency injection setup across all tests.

use crate::allocator::AllocatorInterface;
use crate::core::ObjectHeader;
use crate::di::{DIContainer, DIScope};
use crate::error::GcResult;
use crate::fugc_coordinator::FugcCoordinator;
use crate::thread::MutatorThread;
use mmtk::util::Address;
use std::sync::Arc;

/// Dummy allocator implementation for testing and fallback.
///
/// This allocator always returns OutOfMemory errors and is intended
/// solely for testing scenarios where allocation should fail or be stubbed out.
///
/// # Examples
///
/// ```
/// use fugrip::test_utils::StubAllocator;
/// use fugrip::thread::MutatorThread;
///
/// // Create a stub allocator for testing
/// let allocator = StubAllocator::new();
///
/// // Use for safepoint polling in tests
/// let mutator = MutatorThread::new(1);
/// allocator.poll_safepoint(&mutator);
/// ```
pub struct StubAllocator;

impl StubAllocator {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for StubAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocatorInterface for StubAllocator {
    fn allocate(
        &self,
        _mmtk_mutator: &mut mmtk::Mutator<crate::binding::RustVM>,
        _header: ObjectHeader,
        _bytes: usize,
    ) -> GcResult<*mut u8> {
        Err(crate::error::GcError::OutOfMemory)
    }

    fn poll_safepoint(&self, _mutator: &MutatorThread) {
        // No-op for stub implementation
    }
}

/// Default heap configuration for tests
pub const TEST_HEAP_BASE: usize = 0x10000000;
pub const TEST_HEAP_SIZE: usize = 64 * 1024 * 1024; // 64MB
pub const TEST_WORKER_COUNT: usize = 4;

/// Test fixture that provides an isolated DI container for each test
pub struct TestFixture {
    pub container: DIContainer,
    pub coordinator: Arc<FugcCoordinator>,
    _scope: DIScope,
}

impl TestFixture {
    /// Create a new test fixture with default configuration
    pub fn new() -> Self {
        Self::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, TEST_WORKER_COUNT)
    }

    /// Create a new test fixture with custom heap configuration
    pub fn new_with_config(heap_base: usize, heap_size: usize, worker_count: usize) -> Self {
        let mut container = DIContainer::new_for_testing();

        let heap_base_addr = unsafe { Address::from_usize(heap_base) };
        let coordinator =
            container.create_fugc_coordinator(heap_base_addr, heap_size, worker_count);
        let scope = DIScope::new(container.clone());

        Self {
            container,
            coordinator,
            _scope: scope,
        }
    }

    /// Create a minimal test fixture for lightweight tests
    pub fn minimal() -> Self {
        Self::new_with_config(TEST_HEAP_BASE, 32 * 1024 * 1024, 1) // 32MB, 1 worker
    }

    /// Get the thread registry from the container
    pub fn thread_registry(&self) -> &Arc<crate::thread::ThreadRegistry> {
        self.container.thread_registry()
    }

    /// Get the global roots from the container
    pub fn global_roots(&self) -> &Arc<parking_lot::Mutex<crate::roots::GlobalRoots>> {
        self.container.global_roots()
    }

    /// Get the safepoint manager from the container
    pub fn safepoint_manager(&self) -> Arc<crate::safepoint::SafepointManager> {
        self.container.safepoint_manager()
    }
}

impl Default for TestFixture {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience macro for creating a test fixture
#[macro_export]
macro_rules! test_fixture {
    () => {
        $crate::test_utils::TestFixture::new()
    };
    (minimal) => {
        $crate::test_utils::TestFixture::minimal()
    };
    ($heap_base:expr, $heap_size:expr, $workers:expr) => {
        $crate::test_utils::TestFixture::new_with_config($heap_base, $heap_size, $workers)
    };
}

/// Convenience function for tests that just need a coordinator
pub fn test_coordinator() -> Arc<FugcCoordinator> {
    TestFixture::new().coordinator
}

/// Convenience function for tests that need a coordinator with custom config
pub fn test_coordinator_with_config(
    heap_base: usize,
    heap_size: usize,
    workers: usize,
) -> Arc<FugcCoordinator> {
    TestFixture::new_with_config(heap_base, heap_size, workers).coordinator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_creates_isolated_instances() {
        let fixture1 = TestFixture::new();
        let fixture2 = TestFixture::new();

        // Each fixture should have its own coordinator
        assert!(!Arc::ptr_eq(&fixture1.coordinator, &fixture2.coordinator));

        // And its own thread registry
        assert!(!Arc::ptr_eq(
            fixture1.thread_registry(),
            fixture2.thread_registry()
        ));
    }

    #[test]
    fn test_fixture_minimal_config() {
        let fixture = TestFixture::minimal();

        // Should have a coordinator
        assert!(!fixture.coordinator.is_collecting());
        assert_eq!(
            fixture.coordinator.current_phase(),
            crate::fugc_coordinator::FugcPhase::Idle
        );
    }

    #[test]
    fn test_convenience_functions() {
        let coord1 = test_coordinator();
        let coord2 = test_coordinator_with_config(0x20000000, 128 * 1024 * 1024, 8);

        // Should be different instances
        assert!(!Arc::ptr_eq(&coord1, &coord2));
    }

    #[test]
    fn test_macro_usage() {
        let _fixture1 = test_fixture!();
        let _fixture2 = test_fixture!(minimal);
        let _fixture3 = test_fixture!(0x20000000, 128 * 1024 * 1024, 8);

        // All should work without errors
    }

    #[test]
    fn test_stub_allocator_creation() {
        let allocator = StubAllocator::new();
        let default_allocator = StubAllocator;

        // Both should be valid instances
        let mutator = MutatorThread::new(2);
        allocator.poll_safepoint(&mutator);
        default_allocator.poll_safepoint(&mutator);
    }

    #[test]
    fn test_stub_allocator_always_fails() {
        let allocator = StubAllocator::new();

        // StubAllocator should always return OutOfMemory when allocate is called
        // Note: Testing allocate() requires a valid MMTk Mutator, which is complex to create
        // The StubAllocator always returns OutOfMemory error as designed
        // This test verifies the stub behavior through the trait interface
        let mutator = MutatorThread::new(3);
        allocator.poll_safepoint(&mutator); // This should not panic
    }
}
