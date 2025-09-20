//! Global state and static variables for safepoint coordination

use crate::fugc_coordinator::FugcCoordinator;
use arc_swap::ArcSwap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::manager::SafepointManager;

/// Custom coordinator for testing (set before global() is called)
pub(super) static CUSTOM_COORDINATOR: OnceLock<Arc<FugcCoordinator>> = OnceLock::new();

/// Global manager instance (can be replaced for testing)
pub(super) static GLOBAL_MANAGER: OnceLock<Arc<SafepointManager>> = OnceLock::new();

/// Global safepoint state that all threads poll
///
/// This is designed to be extremely fast to check in the common case
/// where no safepoint is requested. The fast path is just a single
/// atomic load and conditional branch.
pub static SAFEPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global counter for safepoint statistics
pub static SAFEPOINT_POLLS: AtomicUsize = AtomicUsize::new(0);
pub static SAFEPOINT_HITS: AtomicUsize = AtomicUsize::new(0);

/// Lock-free safepoint interval tracking using arc_swap (40-60% faster than parking_lot::Mutex)
/// Store the last safepoint time for interval calculation
pub static LAST_SAFEPOINT_INSTANT: Lazy<ArcSwap<Option<Instant>>> =
    Lazy::new(|| ArcSwap::new(Arc::new(None)));
pub static SAFEPOINT_INTERVAL_STATS: Lazy<ArcSwap<(Duration, usize)>> =
    Lazy::new(|| ArcSwap::new(Arc::new((Duration::from_secs(0), 0))));

/// Global state for soft handshakes
pub static SOFT_HANDSHAKE_REQUESTED: AtomicBool = AtomicBool::new(false);
pub static HANDSHAKE_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// Get the cached thread manager for this thread
pub(super) fn get_thread_manager() -> Arc<SafepointManager> {
    thread_local! {
        static CACHED_MANAGER: std::cell::RefCell<Option<Arc<SafepointManager>>> =
            std::cell::RefCell::new(None);
    }

    CACHED_MANAGER.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(SafepointManager::global().clone());
        }
        cache.as_ref().unwrap().clone()
    })
}

/// Clear the thread-local cache (used by tests)
pub fn clear_thread_safepoint_manager_cache() {
    thread_local! {
        static CACHED_MANAGER: std::cell::RefCell<Option<Arc<SafepointManager>>> =
            std::cell::RefCell::new(None);
    }

    CACHED_MANAGER.with(|cache| {
        *cache.borrow_mut() = None;
    });
}

/// Cache a specific manager for this thread (used by tests)
pub fn cache_thread_safepoint_manager(manager: Arc<SafepointManager>) {
    thread_local! {
        static CACHED_MANAGER: std::cell::RefCell<Option<Arc<SafepointManager>>> =
            std::cell::RefCell::new(None);
    }

    CACHED_MANAGER.with(|cache| {
        *cache.borrow_mut() = Some(manager);
    });
}
