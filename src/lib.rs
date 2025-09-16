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
pub mod error;
pub mod plan;
pub mod roots;
pub mod thread;
pub mod weak;

pub use allocator::{AllocatorInterface, MMTkAllocator};
pub use binding::RustVM;
pub use error::{GcError, GcResult};
pub use plan::FugcPlanManager;
pub use roots::{GlobalRoots, StackRoots};
pub use weak::{WeakRef, WeakRefHeader, WeakRefRegistry};
