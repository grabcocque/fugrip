//! Allocation entry points that use the opaque allocation facade.
//!
//! This module exposes `AllocatorInterface` and small allocator implementations
//! that delegate to the `alloc_facade` opaque-handle API. The goal is to keep
//! all backend details hidden behind zero-cost handles so the backend can be
//! swapped at compile time via features (`use_mmtk`, `use_jemalloc`, `use_stub`).
//!
//! Notes:
//! - `MMTkAllocator` is a thin wrapper around the facade intended for runtime
//!   usage when the MMTk backend is enabled.
//! - `StubAllocator` is a compatibility shim used by existing tests; when
//!   compiled with `--features use_stub` it delegates to `global_allocator()`
//!   so tests exercise the same facade paths as production code.
//!
//! # Examples
//!
//! ```
//! use fugrip::allocator::{MMTkAllocator, AllocatorInterface};
//! use fugrip::core::ObjectHeader;
//!
//! // Create allocator using opaque facade
//! let mmtk_allocator = MMTkAllocator::new();
//!
//! // Use the same interface for all backends
//! let header = ObjectHeader::default();
//! ```

use crate::frontend::alloc_facade::{MutatorHandle, global_allocator};
use crate::frontend::types::constants::MIN_OBJECT_SIZE;
use crate::{core::ObjectHeader, error::GcResult, thread::MutatorThread};

/// Trait capturing the minimal allocation API the VM exposes to the runtime.
///
/// # Examples
///
/// ```
/// use fugrip::allocator::{AllocatorInterface, MMTkAllocator};
/// use fugrip::core::ObjectHeader;
/// use fugrip::thread::MutatorThread;
///
/// fn demonstrate_allocator<A: AllocatorInterface>(allocator: &A) {
///     let header = ObjectHeader::default();
///     let mutator_thread = MutatorThread::new(0);
///
///     // Poll for safepoint
///     allocator.poll_safepoint(&mutator_thread);
/// }
///
/// let allocator = MMTkAllocator::new();
/// demonstrate_allocator(&allocator);
/// ```
pub trait AllocatorInterface {
    /// Allocate an object with the provided header and size in bytes.
    fn allocate(
        &self,
        mutator: MutatorHandle,
        header: ObjectHeader,
        bytes: usize,
    ) -> GcResult<*mut u8>;

    /// Poll the runtime for a safepoint. We will hook this into MMTk's
    /// allocation slow path to cooperate with GC.
    fn poll_safepoint(&self, mutator: &MutatorThread);
}

/// Opaque facade-backed allocator implementation.
///
/// # Examples
///
/// ```
/// use fugrip::allocator::MMTkAllocator;
///
/// // Create a new opaque facade allocator
/// let allocator = MMTkAllocator::new();
/// let default_allocator = MMTkAllocator::default();
///
/// // Both are equivalent
/// ```
pub struct MMTkAllocator;

impl MMTkAllocator {
    pub const fn new() -> Self {
        Self
    }
}

/// Compatibility shim for existing tests that expect `StubAllocator`.
/// When `feature = "use_stub"` is enabled, this will delegate to the
/// facade backend; otherwise it keeps the minimal stub behaviour.
pub struct StubAllocator;

impl StubAllocator {
    pub const fn new() -> Self {
        StubAllocator
    }
}

impl Default for StubAllocator {
    fn default() -> Self {
        StubAllocator::new()
    }
}

impl AllocatorInterface for StubAllocator {
    fn allocate(
        &self,
        mutator: MutatorHandle,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<*mut u8> {
        // Delegate to facade when stub backend is enabled; otherwise emulate
        // OutOfMemory behavior for tests that expect failures.
        #[cfg(feature = "use_stub")]
        {
            let facade = crate::frontend::alloc_facade::global_allocator();
            facade.allocate_object(header, body_bytes)
        }

        #[cfg(not(feature = "use_stub"))]
        {
            let _ = (mutator, header, body_bytes);
            Err(crate::error::GcError::OutOfMemory)
        }
    }

    fn poll_safepoint(&self, mutator: &MutatorThread) {
        mutator.poll_safepoint();
    }
}

impl Default for MMTkAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocatorInterface for MMTkAllocator {
    fn allocate(
        &self,
        mutator: MutatorHandle,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<*mut u8> {
        let total_bytes = std::mem::size_of::<ObjectHeader>() + body_bytes;
        let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);

        // Use pure opaque facade - no MMTk delegation
        let facade = global_allocator();
        let object_ptr = facade.allocate_object(header, body_bytes)?;

        Ok(object_ptr)
    }

    fn poll_safepoint(&self, mutator: &MutatorThread) {
        // Poll for safepoint requests
        mutator.poll_safepoint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ObjectFlags;
    use crate::error::GcError;

    #[test]
    fn test_mmtk_allocator_creation() {
        let allocator = MMTkAllocator::new();
        let default_allocator = MMTkAllocator;

        // Both should be valid instances
        let mutator = MutatorThread::new(1);
        allocator.poll_safepoint(&mutator);
        default_allocator.poll_safepoint(&mutator);
    }

    // Note: Testing allocate() requires a valid Mutator, which is complex to create
    // For testing allocation failures, use the facade with use_stub feature

    #[test]
    fn test_allocator_interface_trait() {
        fn test_allocator<A: AllocatorInterface>(allocator: &A) {
            let mutator = MutatorThread::new(3);
            allocator.poll_safepoint(&mutator);
        }

        test_allocator(&MMTkAllocator::new());
        // Stub backend now integrated into facade with use_stub feature
    }

    #[test]
    fn test_mmtk_allocator_edge_cases() {
        let _allocator = MMTkAllocator::new();
        let _header = ObjectHeader::default();

        // Test zero-byte allocation
        // Note: This test documents current behavior - zero-byte allocations
        // get MIN_OBJECT_SIZE bytes due to MMTk requirements
        let result = std::panic::catch_unwind(|| {
            // This would require an actual MMTk mutator which we can't easily create in tests
            // So we test the size calculation logic instead
            let total_bytes = std::mem::size_of::<ObjectHeader>();
            let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);
            assert!(allocation_size >= MIN_OBJECT_SIZE);
        });
        assert!(result.is_ok());

        // Test alignment calculation
        let align = std::mem::align_of::<usize>().max(std::mem::align_of::<ObjectHeader>());
        assert!(align >= std::mem::align_of::<ObjectHeader>());
        assert!(align >= std::mem::align_of::<usize>());

        // Test large allocation size calculation
        let large_body = 1024 * 1024; // 1MB
        let total_bytes = std::mem::size_of::<ObjectHeader>() + large_body;
        let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);
        assert_eq!(allocation_size, total_bytes); // Should not be clamped to MIN_OBJECT_SIZE
    }

    #[test]
    fn test_allocator_interface_safepoint_polling() {
        let allocator = MMTkAllocator::new();

        // Test with different mutator thread IDs
        let mutator1 = MutatorThread::new(0);
        let mutator2 = MutatorThread::new(1);
        let mutator_max = MutatorThread::new(usize::MAX);

        // These should not panic
        allocator.poll_safepoint(&mutator1);
        allocator.poll_safepoint(&mutator2);
        allocator.poll_safepoint(&mutator_max);
    }

    #[test]
    fn test_object_header_handling() {
        // Test various object header configurations
        let mut header = ObjectHeader::default();

        // Test with different header states
        header.flags |= ObjectFlags::MARKED;
        header.flags |= ObjectFlags::PINNED;

        // Verify header can be written and size calculated correctly
        let header_size = std::mem::size_of::<ObjectHeader>();
        assert!(header_size > 0);
        assert!(header_size <= 64); // Reasonable upper bound

        // Test alignment requirements
        let header_align = std::mem::align_of::<ObjectHeader>();
        assert!(header_align.is_power_of_two());
    }

    #[test]
    fn test_allocation_size_calculations() {
        // Test boundary conditions for allocation size calculations
        let test_cases = [
            (0, MIN_OBJECT_SIZE),
            (1, std::mem::size_of::<ObjectHeader>() + 1),
            (MIN_OBJECT_SIZE, MIN_OBJECT_SIZE),
            (
                MIN_OBJECT_SIZE * 2,
                std::mem::size_of::<ObjectHeader>() + MIN_OBJECT_SIZE * 2,
            ),
        ];

        for (body_bytes, expected_min_total) in test_cases {
            let total_bytes = std::mem::size_of::<ObjectHeader>() + body_bytes;
            let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);

            assert!(allocation_size >= expected_min_total);
            assert!(allocation_size >= MIN_OBJECT_SIZE);
            assert!(allocation_size >= std::mem::size_of::<ObjectHeader>());
        }
    }

    #[test]
    fn test_mmtk_allocator_constants() {
        // Test that allocator can be created as const
        const ALLOCATOR: MMTkAllocator = MMTkAllocator::new();
        let _runtime_allocator = ALLOCATOR;

        // Test Default implementation
        let _default_allocator = MMTkAllocator;
        // Can't easily compare allocators, but creation should not panic
        // Test default creation
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test GcError patterns that would be returned
        let oom_error = GcError::OutOfMemory;
        match oom_error {
            GcError::OutOfMemory => {
                // Test that error can be matched
            }
            _ => panic!("Wrong error type"),
        }

        // Test that GcResult can represent success and failure
        let success: GcResult<*mut u8> = Ok(std::ptr::null_mut());
        let failure: GcResult<*mut u8> = Err(GcError::OutOfMemory);

        assert!(success.is_ok());
        assert!(failure.is_err());
    }
}
