//! Error types for the Fugrip runtime.

use std::fmt;

/// Errors that can occur during GC operations
#[derive(Debug, Clone)]
pub enum GcError {
    /// Allocation failed due to out of memory
    OutOfMemory,
    /// Invalid object reference
    InvalidReference,
    /// Thread registry operation failed
    ThreadError(String),
    /// MMTk operation failed
    MmtkError(String),
}

impl fmt::Display for GcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GcError::OutOfMemory => write!(f, "Out of memory"),
            GcError::InvalidReference => write!(f, "Invalid object reference"),
            GcError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
            GcError::MmtkError(msg) => write!(f, "MMTk error: {}", msg),
        }
    }
}

impl std::error::Error for GcError {}

/// Result type for GC operations
pub type GcResult<T> = Result<T, GcError>;
