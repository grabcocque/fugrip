//! MMTk backend implementation
//!
//! This module contains all MMTk-specific code that implements the
//! actual garbage collection functionality. All code here is "behind
//! the blackwall" and uses MMTk types directly.

pub mod binding;
pub mod object_model;
pub mod plan;

// Re-export key MMTk binding types for internal use
pub use binding::RustVM;
pub use plan::{FugcPlanManager, create_fugc_mmtk_options};