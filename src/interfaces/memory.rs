// Memory-specific interfaces and lightweight production adapters.
//
// This module provides a small HeapProvider trait and a zero-sized
// ProductionHeapProvider that delegates to the existing global allocator.
// The goal is to route direct ALLOCATOR usage through this stable
// interface during the staged refactor.

use crate::memory;
use crate::memory::SegmentedHeap;
use crate::traits::GcTrace;
use crate::{Gc, ObjectClass};

/// Trait providing access to heap internals required by the collector.
/// Implementations should avoid exposing mutable access beyond what's
/// necessary for the collector's read-only inspection needs.
pub trait HeapProvider: Send + Sync + 'static {
    /// Get a reference to the segmented heap.
    fn get_heap(&self) -> &SegmentedHeap;
}

/// Production implementation that delegates to the global allocator.
#[derive(Debug, Clone, Copy)]
pub struct ProductionHeapProvider;

impl HeapProvider for ProductionHeapProvider {
    fn get_heap(&self) -> &SegmentedHeap {
        // Delegate to the existing global allocator in memory.rs
        memory::ALLOCATOR.get_heap()
    }
}

/// Interface for garbage-collected memory allocation.
///
/// This trait defines the contract for allocators that can create
/// garbage-collected objects with different classifications.
pub trait AllocatorTrait {
    /// Allocate a garbage-collected object with the specified classification.
    fn allocate_classified<T: GcTrace + 'static>(&self, value: T, class: ObjectClass) -> Gc<T>;
    
    /// Get the total bytes allocated.
    fn bytes_allocated(&self) -> usize;
    
    /// Get the total object count.
    fn object_count(&self) -> usize;
}

/// Global heap provider instance used by collector phases.
pub static HEAP_PROVIDER: ProductionHeapProvider = ProductionHeapProvider;
