#![allow(dead_code)]

//! Core library entry point for the Fugrip runtime.
//!
//! The project is in the middle of a redesign that pivots toward an
//! MMTk-backed garbage collector.  The modules declared here provide the
//! high-level scaffolding we will flesh out in subsequent milestones.

pub mod allocator;
pub mod binding;
pub mod cache_optimization;
pub mod collector_phases;
pub mod concurrent;
pub mod core;
pub mod di;
pub mod error;
pub mod fugc_coordinator;
pub mod handshake;
pub mod memory_management;
pub mod plan;
pub mod pollcheck_macros;
pub mod roots;
pub mod safepoint;
pub mod simd_sweep;
pub mod test_utils;
pub mod thread;
pub mod weak;

pub use allocator::{AllocatorInterface, MMTkAllocator};
pub use binding::RustVM;
pub use error::{GcError, GcResult};
pub use fugc_coordinator::{AllocationColor, FugcCoordinator, FugcCycleStats, FugcPhase};
pub use memory_management::{
    FinalizerQueue, FinalizerQueueStats, FreeObjectManager, MemoryManager, MemoryManagerStats,
    WeakMap, WeakMapStats, WeakReference,
};
pub use plan::FugcPlanManager;
pub use roots::{GlobalRoots, StackRoots};
pub use safepoint::{
    GcSafepointPhase, SafepointManager, SafepointStats, ThreadExecutionState, pollcheck,
    safepoint_enter, safepoint_exit,
};
pub use test_utils::StubAllocator;
