//! Frontend opaque handle interface
//!
//! This module contains all code that is "in front of the blackwall".
//! Everything here uses opaque handles and zero-cost abstractions only.
//! NO backend-specific types are allowed here.

pub mod alloc_facade;
pub mod allocator;
pub mod types;

// Re-export key frontend types
pub use alloc_facade::{AllocatorFacade, MutatorHandle, PlanHandle, global_allocator};
pub use allocator::{AllocatorInterface, MMTkAllocator, StubAllocator};
pub use types::{Address, ObjectReference};