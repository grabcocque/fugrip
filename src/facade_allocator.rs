//! Allocator implementation that uses the allocation facade
//!
//! This replaces the direct MMTk allocator with one that goes through
//! our abstraction layer, allowing us to swap backends.

use crate::alloc_facade::global_allocator;
use crate::compat::{Address, ObjectReference};
use crate::core::ObjectHeader;
use crate::error::{GcError, GcResult};
use crate::thread::MutatorThread;

/// Allocator that delegates to the global allocation facade
pub struct FacadeAllocator {
    // No state needed - uses global facade
}

impl FacadeAllocator {
    pub fn new() -> Self {
        FacadeAllocator {}
    }

    /// Allocate an object through the facade
    pub fn allocate(&self, header: ObjectHeader, body_bytes: usize) -> GcResult<ObjectReference> {
        let total_bytes = std::mem::size_of::<ObjectHeader>() + body_bytes;
        let facade = global_allocator();

        // Use the facade to allocate
        facade.allocate_object(header, body_bytes)
    }

    /// Allocate raw memory through the facade
    pub fn allocate_raw(&self, size: usize, align: usize) -> GcResult<Address> {
        // For now, we need to go through the object allocation API
        // In a real implementation, we'd add raw allocation to the facade
        let header = ObjectHeader::default();
        let obj_ref = self.allocate(header, size - std::mem::size_of::<ObjectHeader>())?;
        Ok(obj_ref.to_address())
    }

    /// Poll for safepoint
    pub fn poll_safepoint(&self, mutator: &MutatorThread) {
        mutator.poll_safepoint();
    }
}

impl Default for FacadeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait that provides a clean allocation interface
/// This is what the rest of the codebase should use
pub trait CleanAllocator: Send + Sync {
    /// Allocate an object with the given header and size
    fn allocate_object(&self, header: ObjectHeader, body_bytes: usize)
    -> GcResult<ObjectReference>;

    /// Deallocate an object (for sweep phase)
    fn deallocate_object(&self, obj: ObjectReference, size: usize);

    /// Poll for GC safepoint
    fn poll_gc(&self, thread: &MutatorThread);
}

impl CleanAllocator for FacadeAllocator {
    fn allocate_object(
        &self,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<ObjectReference> {
        self.allocate(header, body_bytes)
    }

    fn deallocate_object(&self, obj: ObjectReference, size: usize) {
        global_allocator().deallocate_object(obj, size);
    }

    fn poll_gc(&self, thread: &MutatorThread) {
        self.poll_safepoint(thread);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facade_allocator_creation() {
        let allocator = FacadeAllocator::new();
        let thread = MutatorThread::new(1);
        allocator.poll_safepoint(&thread);
    }

    #[test]
    fn test_clean_allocator_trait() {
        fn use_allocator<A: CleanAllocator>(allocator: &A) {
            let thread = MutatorThread::new(2);
            allocator.poll_gc(&thread);
        }

        let allocator = FacadeAllocator::new();
        use_allocator(&allocator);
    }
}
