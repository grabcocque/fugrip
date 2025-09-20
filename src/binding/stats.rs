//! Statistics and state APIs for FUGC monitoring

use crate::{fugc_coordinator::FugcCycleStats, fugc_coordinator::FugcPhase, plan::FugcPlanManager};
use arc_swap::ArcSwap;
use std::sync::Arc;

use super::FUGC_PLAN_MANAGER;

/// Trigger garbage collection with FUGC optimizations using MMTk's GC trigger API.
/// This integrates with MMTk's allocation failure handling and FUGC's 8-step protocol.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::fugc_gc;
///
/// // Trigger garbage collection
/// fugc_gc();
/// ```
pub fn fugc_gc() {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .gc();
}

/// Get FUGC statistics
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_get_stats;
///
/// // Get current FUGC statistics
/// let stats = fugc_get_stats();
///
/// // Check if concurrent collection is enabled
/// assert!(stats.concurrent_collection_enabled);
///
/// // View work-stealing statistics
/// println!("Work stolen: {}", stats.work_stolen);
/// println!("Work shared: {}", stats.work_shared);
///
/// // View memory statistics
/// println!("Total bytes: {}", stats.total_bytes);
/// println!("Used bytes: {}", stats.used_bytes);
///
/// // View concurrent allocation statistics
/// println!("Objects allocated black: {}", stats.objects_allocated_black);
/// ```
pub fn fugc_get_stats() -> crate::plan::FugcStats {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .get_fugc_stats()
}

/// Get the current FUGC collection phase
pub fn fugc_get_phase() -> FugcPhase {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .fugc_phase()
}

/// Check if FUGC collection is currently in progress
pub fn fugc_is_collecting() -> bool {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .is_fugc_collecting()
}

/// Get FUGC cycle statistics
pub fn fugc_get_cycle_stats() -> FugcCycleStats {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .get_fugc_cycle_stats()
}
