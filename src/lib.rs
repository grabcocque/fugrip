#![allow(dead_code)]

//! Core library entry point for the Fugrip runtime.
//!
//! The project is in the middle of a redesign that pivots toward an
//! MMTk-backed garbage collector.  The modules declared here provide the
//! high-level scaffolding we will flesh out in subsequent milestones.

pub mod alloc_facade;
pub mod allocator;
pub mod binding;
pub mod cache_optimization;
pub mod collector_phases;
mod compat;
pub mod concurrent;
pub mod core;
pub mod di;
pub mod error;
pub mod facade_allocator;
pub mod fugc_coordinator;
pub mod handshake;
pub mod hot_counters;
pub mod memory_management;
pub mod modern_allocator;
pub mod opaque_handles;
pub mod plan;
pub mod pollcheck_macros;
pub mod roots;
pub mod safepoint;
pub mod simd_sweep;
pub mod test_utils;
pub mod thread;
pub mod types;
pub mod verse_style_optimizations;
pub mod weak;
pub mod zero_cost_allocator;

pub use alloc_facade::{AllocatorFacade, global_allocator};
pub use allocator::{AllocatorInterface, MMTkAllocator};
pub use error::{GcError, GcResult};
pub use facade_allocator::{CleanAllocator, FacadeAllocator};
pub use fugc_coordinator::{AllocationColor, FugcCoordinator, FugcCycleStats, FugcPhase};
pub use memory_management::{
    FinalizerQueue, FinalizerQueueStats, FreeObjectManager, MemoryManager, MemoryManagerStats,
    WeakMap, WeakReference,
};
pub use modern_allocator::{FugcAllocator, ModernAllocator};
pub use opaque_handles::{
    AllocatorStats, BackendType, MutatorId, ObjectId, OpaqueAllocator, PlanId,
};
pub use plan::FugcPlanManager;
pub use roots::{GlobalRoots, StackRoots};
pub use safepoint::{
    GcSafepointPhase, SafepointManager, SafepointStats, ThreadExecutionState, pollcheck,
    safepoint_enter, safepoint_exit,
};
pub use test_utils::StubAllocator;
pub use zero_cost_allocator::ZeroCostAllocator;

// Optional: use jemalloc as the global allocator when enabled
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL_JEMALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
