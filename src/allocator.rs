//! Allocation entry points that delegate to MMTk.
//!
//! This module provides the allocation interface for the FUGC garbage collector,
//! supporting both MMTk-backed allocation and stub implementations for testing.
//!
//! # Examples
//!
//! ```
//! use fugrip::allocator::{MMTkAllocator, StubAllocator, AllocatorInterface};
//! use fugrip::core::ObjectHeader;
//!
//! // Create allocators
//! let mmtk_allocator = MMTkAllocator::new();
//! let stub_allocator = StubAllocator::new();
//!
//! // Both implement the same interface
//! let header = ObjectHeader::default();
//! ```

use crate::{
    binding::fugc_post_alloc,
    core::ObjectHeader,
    error::{GcError, GcResult},
    thread::MutatorThread,
};
use mmtk::{
    plan::AllocationSemantics,
    util::{ObjectReference, constants::MIN_OBJECT_SIZE},
};

/// Trait capturing the minimal allocation API the VM exposes to the runtime.
///
/// # Examples
///
/// ```
/// use fugrip::allocator::{AllocatorInterface, StubAllocator};
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
/// let allocator = StubAllocator::new();
/// demonstrate_allocator(&allocator);
/// ```
pub trait AllocatorInterface {
    /// Allocate an object with the provided header and size in bytes.
    fn allocate(
        &self,
        mmtk_mutator: &mut mmtk::Mutator<crate::binding::RustVM>,
        header: ObjectHeader,
        bytes: usize,
    ) -> GcResult<*mut u8>;

    /// Poll the runtime for a safepoint. We will hook this into MMTk's
    /// allocation slow path to cooperate with GC.
    fn poll_safepoint(&self, mutator: &MutatorThread);
}

/// MMTk-backed allocator implementation.
///
/// # Examples
///
/// ```
/// use fugrip::allocator::MMTkAllocator;
///
/// // Create a new MMTk allocator
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

impl Default for MMTkAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocatorInterface for MMTkAllocator {
    fn allocate(
        &self,
        mmtk_mutator: &mut mmtk::Mutator<crate::binding::RustVM>,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<*mut u8> {
        let total_bytes = std::mem::size_of::<ObjectHeader>() + body_bytes;
        let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);
        let align = std::mem::align_of::<usize>().max(std::mem::align_of::<ObjectHeader>());

        let address = mmtk::memory_manager::alloc(
            mmtk_mutator,
            allocation_size,
            align,
            0,
            AllocationSemantics::Default,
        );

        if address.is_zero() {
            return Err(GcError::OutOfMemory);
        }

        let object_ptr = address.to_mut_ptr::<u8>();

        unsafe {
            // Write object header and clear the body so callers observe deterministic state.
            std::ptr::write(object_ptr.cast::<ObjectHeader>(), header);
            if body_bytes > 0 {
                std::ptr::write_bytes(
                    object_ptr.add(std::mem::size_of::<ObjectHeader>()),
                    0,
                    body_bytes,
                );
            }
        }

        let object_ref = unsafe { ObjectReference::from_raw_address_unchecked(address) };
        mmtk::memory_manager::post_alloc(
            mmtk_mutator,
            object_ref,
            allocation_size,
            AllocationSemantics::Default,
        );
        fugc_post_alloc(object_ref, body_bytes);

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
    // The StubAllocator always returns OutOfMemory error as designed

    #[test]
    fn test_allocator_interface_trait() {
        fn test_allocator<A: AllocatorInterface>(allocator: &A) {
            let mutator = MutatorThread::new(3);
            allocator.poll_safepoint(&mutator);
        }

        test_allocator(&MMTkAllocator::new());
        // StubAllocator moved to test_utils.rs for testing purposes
    }
}
