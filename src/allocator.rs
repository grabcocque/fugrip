//! Allocation entry points that delegate to MMTk.

use crate::{core::ObjectHeader, error::GcResult, thread::MutatorThread};

/// Trait capturing the minimal allocation API the VM exposes to the runtime.
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
