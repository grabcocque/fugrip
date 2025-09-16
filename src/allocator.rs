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

use crate::{core::ObjectHeader, error::GcResult, thread::MutatorThread};

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
        _mmtk_mutator: &mut mmtk::Mutator<crate::binding::RustVM>,
        _header: ObjectHeader,
        _body_bytes: usize,
    ) -> GcResult<*mut u8> {
        // TODO: Implement actual MMTk allocation
        // For now, return an error to indicate allocation failure
        Err(crate::error::GcError::OutOfMemory)
    }

    fn poll_safepoint(&self, mutator: &MutatorThread) {
        // Poll for safepoint requests
        mutator.poll_safepoint();
    }
}

/// Dummy allocator implementation for testing and fallback.
///
/// # Examples
///
/// ```
/// use fugrip::allocator::{StubAllocator, AllocatorInterface};
/// use fugrip::core::ObjectHeader;
/// use fugrip::thread::MutatorThread;
///
/// // Create a stub allocator for testing
/// let allocator = StubAllocator::new();
/// let default_allocator = StubAllocator::default();
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
