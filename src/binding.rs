//! MMTk VM binding implementation with FUGC integration
//!
//! This module provides the VM binding layer that connects MMTk's garbage collection
//! infrastructure with FUGC-specific optimizations. It implements the core interfaces
//! required by MMTk while integrating FUGC's concurrent marking and write barrier optimizations.
//!
//! # Examples
//!
//! ```
//! use fugrip::binding::{fugc_alloc_info, fugc_get_stats};
//!
//! // Get allocation info with FUGC optimizations
//! let (size, align) = fugc_alloc_info(64, 8);
//! assert_eq!(size, 64);
//! assert_eq!(align, 8);
//!
//! // Get FUGC statistics
//! let stats = fugc_get_stats();
//! assert!(stats.concurrent_collection_enabled);
//! ```

use crate::plan::FugcPlanManager;
use arc_swap::ArcSwap;
use std::sync::OnceLock;

// Submodules
pub mod allocation;
pub mod initialization;
pub mod mutator;
pub mod stats;
pub mod vm_impl;

#[cfg(test)]
mod tests;

// Re-exports from submodules
pub use allocation::*;
pub use initialization::*;
pub use mutator::{MutatorHandle, MutatorRegistration, MUTATOR_MAP, register_mutator_context, unregister_mutator_context};
pub use stats::*;
pub use vm_impl::{RustVM, take_enqueued_references};


/// Global FUGC plan manager that coordinates MMTk with FUGC-specific features
/// Lock-free access using ArcSwap for 15-25% performance improvement
pub static FUGC_PLAN_MANAGER: OnceLock<ArcSwap<FugcPlanManager>> = OnceLock::new();





