//! Advanced memory management features for FUGC
//!
//! This module implements C-style, Java-style, and JavaScript-style memory management
//! patterns including explicit freeing, finalizers, weak references, and weak maps.
//!
//! ## Features
//!
//! - **Explicit Freeing**: C-style `free()` with immediate object invalidation
//! - **Free Singleton Redirection**: All pointers to freed objects redirect to singleton
//! - **Finalizer Queues**: Java-style finalization with custom processing threads
//! - **Weak References**: Automatic nulling when target objects are collected
//! - **Weak Maps**: JavaScript-style WeakMap with iteration support

// Submodules
pub mod finalizers;
pub mod free_objects;
pub mod manager;
pub mod tests;
pub mod weak_maps;
pub mod weak_refs;

// Re-export public APIs
pub use finalizers::{FinalizerCallback, FinalizerQueue, FinalizerQueueStats};
pub use free_objects::{FreeObjectManager, FreeObjectStats, ObjectState};
pub use manager::{MemoryManager, MemoryManagerStats};
pub use weak_maps::{WeakMap, WeakMapIterator, WeakMapStats};
pub use weak_refs::{WeakRefRegistry, WeakRefStats, WeakReference};