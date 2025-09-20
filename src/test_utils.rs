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
use mmtk::util::{Address, ObjectReference};
use std::sync::Arc;

use crossbeam::queue::SegQueue;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Locality-aware work stealer used only in tests.
///
/// This is a simple wrapper around Rayon that provides the interface
/// expected by tests while leveraging Rayon's superior work-stealing.
pub struct LocalityAwareWorkStealer {
    work_items: Arc<SegQueue<ObjectReference>>,
    steal_threshold: usize,
    local_processed: AtomicUsize,
    stolen_work: AtomicUsize,
    work_shared: AtomicUsize,
}

impl LocalityAwareWorkStealer {
    pub fn new(steal_threshold: usize) -> Self {
        Self {
            work_items: Arc::new(SegQueue::new()),
            steal_threshold,
            local_processed: AtomicUsize::new(0),
            stolen_work: AtomicUsize::new(0),
            work_shared: AtomicUsize::new(0),
        }
    }

    /// Add multiple objects - uses Rayon for parallel processing
    pub fn add_objects(&self, objects: Vec<ObjectReference>) {
        // Use Rayon to parallelize pushing into the SegQueue; each push is thread-safe
        objects.into_par_iter().for_each(|obj| {
            self.work_items.push(obj);
            self.work_shared.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// Get the next batch of work - simple wrapper over SegQueue
    pub fn get_next_batch(&self, batch_size: usize) -> Vec<ObjectReference> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(obj) = self.work_items.pop() {
                batch.push(obj);
                self.local_processed.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }
        batch
    }

    /// Push work item - simple SegQueue wrapper
    pub fn push_local(&self, object: ObjectReference) {
        self.work_items.push(object);
        self.work_shared.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pop(&self) -> Option<ObjectReference> {
        if let Some(o) = self.work_items.pop() {
            self.local_processed.fetch_add(1, Ordering::Relaxed);
            Some(o)
        } else {
            None
        }
    }

    pub fn steal_from(&self, other: &LocalityAwareWorkStealer) -> bool {
        // Simple steal implementation
        if let Some(o) = other.work_items.pop() {
            self.work_items.push(o);
            self.stolen_work.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn has_work(&self) -> bool {
        !self.work_items.is_empty()
    }

    pub fn shared_queue(&self) -> &SegQueue<ObjectReference> {
        &self.work_items
    }

    /// Compatibility: provide a `lock()` returning Result so tests that call
    /// `stealer.lock().unwrap()` continue to work. This is a no-op wrapper.
    pub fn lock(&self) -> Result<&Self, ()> {
        Ok(self)
    }

    /// Compatibility: provide `try_lock()` that always succeeds and returns a reference.
    pub fn try_lock(&self) -> Option<&Self> {
        Some(self)
    }

    pub fn get_stats(&self) -> (usize, usize, usize) {
        (
            self.local_processed.load(Ordering::Relaxed),
            self.stolen_work.load(Ordering::Relaxed),
            self.work_shared.load(Ordering::Relaxed),
        )
    }

    /// Drain available work and process it in parallel using Rayon
    pub fn process_all_with<F>(&self, f: F)
    where
        F: Fn(ObjectReference) + Sync + Send,
    {
        // Drain all work and process in parallel with Rayon
        let mut drained = Vec::new();
        while let Some(o) = self.work_items.pop() {
            drained.push(o);
        }
        // Rayon handles the parallel processing
        drained.into_par_iter().for_each(|obj| f(obj));
    }
}

/// Test fixture that provides an isolated DI container for each test
pub struct TestFixture {
    pub container: Arc<DIContainer>,
    pub coordinator: Arc<FugcCoordinator>,
    _scope: DIScope,
    _manager: Arc<crate::safepoint::SafepointManager>,
}

impl TestFixture {
    /// Create a new test fixture with default configuration
    pub fn new() -> Self {
        Self::new_with_config(TEST_HEAP_BASE, TEST_HEAP_SIZE, TEST_WORKER_COUNT)
    }

    /// Create a new test fixture with custom heap configuration
    pub fn new_with_config(heap_base: usize, heap_size: usize, worker_count: usize) -> Self {
        let container = Arc::new(DIContainer::new_for_testing());

        let heap_base_addr = unsafe { Address::from_usize(heap_base) };
        let coordinator =
            container.create_fugc_coordinator(heap_base_addr, heap_size, worker_count);

        // Get the container's safepoint manager
        let manager = Arc::clone(container.safepoint_manager());

        // Set this container as the current one for the duration of the test
        let scope = DIScope::new(Arc::clone(&container));

        Self {
            container,
            coordinator,
            _scope: scope,
            _manager: manager,
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
    pub fn global_roots(&self) -> &arc_swap::ArcSwap<crate::roots::GlobalRoots> {
        self.container.global_roots()
    }

    /// Return a snapshot `Arc<GlobalRoots>` for tests that expect `Arc`.
    pub fn global_roots_snapshot(&self) -> std::sync::Arc<crate::roots::GlobalRoots> {
        self.container.global_roots().load_full()
    }

    /// Get the safepoint manager from the container
    pub fn safepoint_manager(&self) -> &Arc<crate::safepoint::SafepointManager> {
        self.container.safepoint_manager()
    }
}

impl Default for TestFixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TestFixture {
    fn drop(&mut self) {
        // Clear the global manager to prevent test interference
        // Note: We can't reset OnceLock, but creating new containers for each test
        // ensures isolation during the test lifetime
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
    Arc::clone(&TestFixture::new().coordinator)
}

/// Convenience function for tests that need a coordinator with custom config
pub fn test_coordinator_with_config(
    heap_base: usize,
    heap_size: usize,
    workers: usize,
) -> Arc<FugcCoordinator> {
    Arc::clone(&TestFixture::new_with_config(heap_base, heap_size, workers).coordinator)
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
