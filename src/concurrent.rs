//! Concurrent marking infrastructure for FUGC-style garbage collection
//!
//! This module provides the complete concurrent marking infrastructure including:
//! - Tricolor marking state management
//! - Parallel marking coordination using Rayon
//! - Write barriers with generational optimization
//! - Black allocation for concurrent marking
//! - Object classification and generational management
//! - Root scanning and worker coordination

pub mod allocation;
pub mod barriers;
pub mod classification;
pub mod coordination;
pub mod core;
pub mod marking;
pub mod tricolor;

#[cfg(test)]
pub mod tests;

// Re-export all the core types and functions
pub use allocation::*;
pub use barriers::*;
pub use classification::*;
pub use coordination::*;
pub use core::*;
pub use marking::*;
pub use tricolor::*;
