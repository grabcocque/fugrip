//! Backend implementations
//!
//! This module contains all backend-specific code that is "behind the blackwall".
//! Each backend can use its own native types and APIs without restriction.

#[cfg(feature = "use_mmtk")]
pub mod mmtk;

#[cfg(feature = "use_jemalloc")]
pub mod jemalloc;

#[cfg(feature = "use_stub")]
pub mod stub;