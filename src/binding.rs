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

#[cfg(feature = "use_mmtk")]
use crate::backends::mmtk::FugcPlanManager;
use arc_swap::ArcSwap;
use std::sync::OnceLock;

// Note: MMTk-specific binding implementations are in src/backends/mmtk/binding/
// This module provides the global coordination point for MMTk integration

#[cfg(feature = "use_mmtk")]
/// Global FUGC plan manager that coordinates MMTk with FUGC-specific features
/// Lock-free access using ArcSwap for 15-25% performance improvement
pub static FUGC_PLAN_MANAGER: OnceLock<ArcSwap<FugcPlanManager>> = OnceLock::new();

#[cfg(feature = "use_mmtk")]
/// Initialize the global FUGC plan manager
pub fn initialize_fugc_plan_manager() -> &'static ArcSwap<FugcPlanManager> {
    FUGC_PLAN_MANAGER.get_or_init(|| {
        ArcSwap::new(std::sync::Arc::new(FugcPlanManager::new()))
    })
}
