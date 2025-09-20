//! Event system for safepoint and GC coordination

/// Unified event type for flume event bus consolidation
/// Replaces multiple crossbeam channels across the project for 10-20% performance improvement
#[derive(Debug, Clone)]
pub enum GcEvent {
    /// Safepoint hit notification
    SafepointHit,
    /// Work notification for memory management finalizers
    WorkNotification,
    /// Phase change notification from FUGC coordinator
    PhaseChange(crate::fugc_coordinator::FugcPhase),
    /// Collection finished notification
    CollectionFinished,
    /// Handshake completion notification with thread ID and sequence
    HandshakeCompletion { thread_id: usize, sequence: u64 },
    /// Handshake release notification with sequence
    HandshakeRelease(u64),
    /// Thread registration event
    ThreadRegistered(std::thread::ThreadId),
    /// Handshake request for specific thread
    HandshakeRequest {
        thread_id: usize,
        request: HandshakeRequest,
    },
    /// Work assignment for parallel marking
    WorkAssignment(Vec<mmtk::util::ObjectReference>),
    /// Thread coordination signals
    ThreadStart(usize),
    /// Thread done signal
    ThreadDone(usize),
    /// Generic coordination signal
    CoordinationSignal,
}

/// Handshake request data for thread coordination
#[derive(Debug, Clone)]
pub struct HandshakeRequest {
    pub sequence: u64,
    pub callback: Option<String>, // Simplified callback representation for Clone
}
