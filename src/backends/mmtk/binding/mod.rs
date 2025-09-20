//! MMTk VM binding implementation modules
//!
//! This module contains MMTk-specific binding code that implements the VM interface.
//! All code here can use MMTk types directly since it's behind the blackwall.

pub mod allocation;
pub mod initialization;
pub mod mutator;
pub mod mutator_helpers;
pub mod stats;
pub mod tests;
pub mod vm_impl;

// Re-export key binding types for internal backend use
pub use vm_impl::RustVM;
pub use mutator::{MUTATOR_MAP, mutator_thread_key};

// Note: FUGC_PLAN_MANAGER is in the root binding module (src/binding.rs)
// Backend files should import it directly as: use crate::binding::FUGC_PLAN_MANAGER;