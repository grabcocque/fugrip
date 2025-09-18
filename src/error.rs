//! Error types for the Fugrip runtime.

use std::fmt;

/// Errors that can occur during GC operations
///
/// # Examples
///
/// ```
/// use fugrip::error::{GcError, GcResult};
///
/// // Create different types of GC errors
/// let oom_error = GcError::OutOfMemory;
/// let invalid_ref_error = GcError::InvalidReference;
/// let thread_error = GcError::ThreadError("Failed to register thread".to_string());
/// let mmtk_error = GcError::MmtkError("Plan initialization failed".to_string());
///
/// // Use GcResult for error handling
/// let success: GcResult<u32> = Ok(42);
/// let failure: GcResult<u32> = Err(GcError::OutOfMemory);
///
/// assert!(success.is_ok());
/// assert!(failure.is_err());
///
/// // Error display
/// assert_eq!(oom_error.to_string(), "Out of memory");
/// assert_eq!(invalid_ref_error.to_string(), "Invalid object reference");
/// ```
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
    /// GC phase transition failed
    PhaseTransitionFailed,
    /// Emergency stop requested
    EmergencyStop,
}

impl fmt::Display for GcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GcError::OutOfMemory => write!(f, "Out of memory"),
            GcError::InvalidReference => write!(f, "Invalid object reference"),
            GcError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
            GcError::MmtkError(msg) => write!(f, "MMTk error: {}", msg),
            GcError::PhaseTransitionFailed => write!(f, "GC phase transition failed"),
            GcError::EmergencyStop => write!(f, "Emergency stop requested"),
        }
    }
}

impl std::error::Error for GcError {}

/// Result type for GC operations
pub type GcResult<T> = Result<T, GcError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats_readable_messages() {
        let cases = [
            (GcError::OutOfMemory, "Out of memory"),
            (GcError::InvalidReference, "Invalid object reference"),
            (
                GcError::ThreadError("register failed".into()),
                "Thread error: register failed",
            ),
            (
                GcError::MmtkError("plan init".into()),
                "MMTk error: plan init",
            ),
            (GcError::PhaseTransitionFailed, "GC phase transition failed"),
            (GcError::EmergencyStop, "Emergency stop requested"),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn gc_result_alias_behaves_like_result() {
        fn take_result(value: GcResult<usize>) -> usize {
            value.unwrap_or_default()
        }

        assert_eq!(take_result(Ok(42)), 42);
        assert_eq!(take_result(Err(GcError::OutOfMemory)), 0);
    }
}
