// Interface modules for dependency injection and testability.
//
// This module provides abstract interfaces for core system dependencies,
// along with both production implementations and mocks for testing.

pub mod memory;
pub mod threading;

#[cfg(test)]
pub mod mocks;

#[cfg(test)]
pub mod test_utils;

// Re-export commonly used traits and types
pub use memory::{AllocatorTrait, HeapProvider, ProductionHeapProvider, HEAP_PROVIDER};
pub use threading::{ThreadingProvider, ProductionThreadingProvider, THREADING_PROVIDER};

#[cfg(test)]
pub use mocks::{MockHeapProvider, MockThreadingProvider};

#[cfg(test)]
pub use test_utils::test_builders;