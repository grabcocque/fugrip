//! FUGC (Fil's Unbelievable Garbage Collector) protocol implementation
//!
//! This module implements a faithful version of the eight step FUGC protocol as
//! described by Epic Games for the Verse runtime.  The coordinator integrates
//! with the existing concurrent marking infrastructure, provides precise
//! safepoint handshakes, and maintains page level allocation colouring to
//! emulate the production collector's behaviour.

// Submodules
pub mod api;
pub mod core;
pub mod helpers;
pub mod protocol;
pub mod types;

#[cfg(test)]
mod tests;

// Re-exports
pub use core::FugcCoordinator;
pub use types::*;
