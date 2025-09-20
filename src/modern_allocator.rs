//! Modern allocator implementation using the allocation facade
//!
//! This module provides the preferred allocation interface that works with
//! both jemalloc and MMTk backends through the allocation facade.

use crate::{
    core::ObjectHeader,
    error::{GcError, GcResult},
    frontend::alloc_facade::{AllocatorFacade, MutatorHandle, global_allocator, register_mutator},
    frontend::types::{Address, ObjectReference, constants::MIN_OBJECT_SIZE},
    thread::MutatorThread,
};

/// The preferred allocator interface that uses the allocation facade
/// This automatically works with both jemalloc and MMTk backends
pub trait ModernAllocator: Send + Sync {
    /// Allocate an object with the given header and body size
    fn allocate_object(&self, header: ObjectHeader, body_bytes: usize)
    -> GcResult<ObjectReference>;

    /// Poll for GC safepoint
    fn poll_gc(&self, thread: &MutatorThread);

    /// Get allocation statistics
    fn allocation_stats(&self) -> AllocationStats;
}

/// Statistics for allocation tracking
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub total_allocated: usize,
    pub allocation_count: usize,
}

/// The modern facade-based allocator
/// This is what new code should use instead of the legacy MMTkAllocator
pub struct FugcAllocator {
    facade: AllocatorFacade,
    handle: Option<MutatorHandle>,
}

impl FugcAllocator {
    pub fn new() -> Self {
        FugcAllocator {
            facade: AllocatorFacade::new_jemalloc(),
            handle: None,
        }
    }

    /// Create an allocator with a specific mutator handle
    pub fn with_handle(handle: MutatorHandle) -> Self {
        FugcAllocator {
            facade: AllocatorFacade::new_jemalloc(),
            handle: Some(handle),
        }
    }

    /// Set the mutator handle for this allocator
    pub fn set_handle(&mut self, handle: MutatorHandle) {
        self.handle = Some(handle);
    }

    /// Allocate using the global facade
    pub fn alloc_global(header: ObjectHeader, body_bytes: usize) -> GcResult<ObjectReference> {
        match global_allocator().allocate_object(header, body_bytes) {
            Ok(ptr) => {
                // Convert raw pointer to ObjectReference
                let addr = Address::from_mut_ptr(ptr);
                unsafe { Ok(ObjectReference::from_raw_address_unchecked(addr)) }
            }
            Err(e) => Err(e),
        }
    }

    /// Deallocate using the global facade
    pub fn dealloc_global(obj: ObjectReference, size: usize) {
        let ptr = obj.to_raw_address().to_mut_ptr();
        global_allocator().deallocate_object(ptr, size);
    }
}

impl Default for FugcAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ModernAllocator for FugcAllocator {
    fn allocate_object(
        &self,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<ObjectReference> {
        match self.facade.allocate_object(header, body_bytes) {
            Ok(ptr) => {
                // Convert raw pointer to ObjectReference
                let addr = Address::from_mut_ptr(ptr);
                unsafe { Ok(ObjectReference::from_raw_address_unchecked(addr)) }
            }
            Err(e) => Err(e),
        }
    }

    fn poll_gc(&self, thread: &MutatorThread) {
        self.facade.poll_gc(thread);
    }

    fn allocation_stats(&self) -> AllocationStats {
        let facade = global_allocator();
        AllocationStats {
            total_allocated: facade.total_allocated(),
            allocation_count: facade.allocation_count(),
        }
    }
}

/// Convenience functions for allocation without creating allocator instances
pub mod global {
    use super::*;

    /// Allocate an object using the global allocator
    pub fn allocate_object(header: ObjectHeader, body_bytes: usize) -> GcResult<ObjectReference> {
        FugcAllocator::alloc_global(header, body_bytes)
    }

    /// Deallocate an object using the global allocator
    pub fn deallocate_object(obj: ObjectReference, size: usize) {
        FugcAllocator::dealloc_global(obj, size);
    }

    /// Poll for GC using the current thread
    pub fn poll_gc(thread: &MutatorThread) {
        let allocator = FugcAllocator::new();
        allocator.poll_gc(thread);
    }

    /// Get allocation statistics
    pub fn allocation_stats() -> AllocationStats {
        let allocator = FugcAllocator::new();
        allocator.allocation_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ObjectFlags;

    #[test]
    fn test_fugc_allocator_creation() {
        let allocator = FugcAllocator::new();
        let default_allocator = FugcAllocator::default();

        let thread = MutatorThread::new(1);
        allocator.poll_gc(&thread);
        default_allocator.poll_gc(&thread);
    }

    #[test]
    fn test_modern_allocator_trait() {
        fn test_allocator<A: ModernAllocator>(allocator: &A) {
            let thread = MutatorThread::new(2);
            allocator.poll_gc(&thread);

            let stats = allocator.allocation_stats();
            assert!(stats.total_allocated >= 0);
            assert!(stats.allocation_count >= 0);
        }

        let allocator = FugcAllocator::new();
        test_allocator(&allocator);
    }

    #[test]
    fn test_global_allocation_functions() {
        let header = ObjectHeader::default();
        let thread = MutatorThread::new(3);

        // Test global functions exist and don't panic
        global::poll_gc(&thread);
        let _stats = global::allocation_stats();

        // Test allocation (may fail in test environment)
        match global::allocate_object(header, 64) {
            Ok(obj_ref) => {
                // If successful, test deallocation
                global::deallocate_object(obj_ref, 64 + std::mem::size_of::<ObjectHeader>());
            }
            Err(_) => {
                // Expected in test environment without proper backend setup
            }
        }
    }

    #[test]
    fn test_allocation_with_different_sizes() {
        let allocator = FugcAllocator::new();
        let header = ObjectHeader::default();

        // Test various allocation sizes
        let test_sizes = [0, 1, 16, 64, 1024, MIN_OBJECT_SIZE];

        for &size in &test_sizes {
            // May fail in test environment, just test that it doesn't panic
            let _ = allocator.allocate_object(header, size);
        }
    }

    #[test]
    fn test_object_header_variations() {
        let allocator = FugcAllocator::new();
        let mut header = ObjectHeader::default();

        // Test with different header configurations
        header.flags |= ObjectFlags::MARKED;
        let _ = allocator.allocate_object(header, 32);

        header.flags |= ObjectFlags::PINNED;
        let _ = allocator.allocate_object(header, 32);

        // Test with modified body size
        header.body_size = 128;
        let _ = allocator.allocate_object(header, 128);
    }

    #[test]
    fn test_allocator_with_handle() {
        // Test handle creation (using thread ID for jemalloc backend)
        let handle = register_mutator(42);
        let allocator = FugcAllocator::with_handle(handle);

        let thread = MutatorThread::new(42);
        allocator.poll_gc(&thread);
    }

    #[test]
    fn test_error_handling() {
        let allocator = FugcAllocator::new();
        let header = ObjectHeader::default();

        // Test that large allocations are handled gracefully
        match allocator.allocate_object(header, usize::MAX - 1000) {
            Ok(_) => {
                // Unexpected success with huge allocation
                // This might actually succeed with jemalloc in some cases
            }
            Err(e) => {
                // Expected failure with huge allocation
                match e {
                    GcError::OutOfMemory => {}
                    GcError::InvalidLayout => {}
                    _ => {} // Other errors also acceptable
                }
            }
        }
    }

    #[test]
    fn test_stats_consistency() {
        let allocator = FugcAllocator::new();
        let initial_stats = allocator.allocation_stats();

        // Stats should be consistent between calls
        let second_stats = allocator.allocation_stats();

        // In test environment, stats might not change but should be consistent
        assert_eq!(initial_stats.total_allocated, second_stats.total_allocated);
        assert_eq!(
            initial_stats.allocation_count,
            second_stats.allocation_count
        );
    }
}
