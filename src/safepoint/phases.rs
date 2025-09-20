//! FUGC-specific safepoint phases and integration

/// FUGC-specific safepoint phases
///
/// Different phases of garbage collection require different
/// safepoint coordination strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcSafepointPhase {
    /// Root scanning phase - scan thread stacks and globals
    RootScanning,
    /// Barrier activation - enable write barriers for concurrent marking
    BarrierActivation,
    /// Marking handshake - coordinate marking between threads
    MarkingHandshake,
    /// Sweep preparation - prepare for sweep phase
    SweepPreparation,
}

// Safepoint integration methods are implemented directly in fugc_coordinator.rs
