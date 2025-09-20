//! LLVM-style safepoint implementation for FUGC
//!
//! This module provides bounded-progress safepoints with fast load-and-branch
//! pollchecks and slow path callbacks. The compiler emits pollchecks frequently
//! enough to guarantee that only a bounded amount of progress can occur before
//! a safepoint is reached.
//!
//! ## Architecture
//!
//! - **Fast Path**: Simple load-and-branch that checks a global safepoint flag
//! - **Slow Path**: Callback mechanism that performs FUGC-specific work
//! - **Bounded Progress**: Compiler guarantees pollchecks occur regularly
//! - **Low Overhead**: Fast path designed for minimal performance impact
//!
//! ## Usage
//!
//! ```ignore
//! use fugrip::safepoint::{SafepointManager, pollcheck};
//!
//! // In generated code or hot loops
//! pollcheck(); // Fast load-and-branch, rarely taken
//!
//! // When GC needs to coordinate
//! let manager = SafepointManager::global();
//! manager.request_safepoint(|| {
//!     // FUGC work happens here
//! });
//! ```

pub mod events;
pub mod globals;
pub mod manager;
pub mod phases;
pub mod state;

#[cfg(test)]
pub mod tests;

// Re-export all the core types and functions
pub use events::*;
pub use globals::{
    HANDSHAKE_GENERATION, SAFEPOINT_HITS, SAFEPOINT_POLLS, SAFEPOINT_REQUESTED,
    SOFT_HANDSHAKE_REQUESTED, cache_thread_safepoint_manager, clear_thread_safepoint_manager_cache,
};
pub use manager::{HandshakeCallback, SafepointCallback, SafepointManager, SafepointStats};
pub use phases::*;
pub use state::{ThreadExecutionState, pollcheck, safepoint_enter, safepoint_exit};
