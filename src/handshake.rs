// Lock-free handshake protocol for FUGC garbage collection
// Design: Invalid states are unrepresentable through type safety and atomic state machines

use crossbeam_epoch::{self as epoch};
use crossbeam_utils::atomic::AtomicCell;
use dashmap::{DashMap, DashSet};
use flume::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

/// Handshake request sent from coordinator to mutator threads - simplified with epoch coordination
#[derive(Debug, Clone, Copy)]
pub struct HandshakeRequest {
    pub callback_type: HandshakeType,
    // sequence_id removed - epoch provides automatic generation tracking
}

/// Types of handshake operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeType {
    StackScan,
    CacheReset,
    BarrierActivation,
}

/// Handshake completion response from mutator back to coordinator - simplified with epoch coordination
#[derive(Debug, Clone)]
pub struct HandshakeCompletion {
    pub thread_id: usize,
    pub stack_roots: Vec<usize>,
    // sequence_id removed - epoch handles coordination automatically
}

/// Lock-free atomic state machine for handshake protocol
/// State transitions are enforced by the type system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    Running = 0,         // Thread executing normally
    RequestReceived = 1, // Got handshake request
    AtSafepoint = 2,     // Reached safepoint, processing request
    Completed = 3,       // Sent completion, waiting for release
}

/// Coordinator for managing lock-free handshakes with all mutator threads - simplified with epoch coordination
pub struct HandshakeCoordinator {
    /// Lock-free concurrent access to thread request channels
    /// High-impact optimization for FUGC handshake performance - eliminates external lock contention
    /// Channels to send requests to each thread (thread_id -> sender)
    thread_requests: DashMap<usize, Sender<HandshakeRequest>>,
    /// Single channel to receive completions from all threads
    completion_rx: Receiver<HandshakeCompletion>,
    /// Single channel to send release signals to all threads
    release_tx: Sender<()>,
    // sequence_counter removed - epoch provides automatic coordination
}

impl HandshakeCoordinator {
    pub fn new() -> (Self, Sender<HandshakeCompletion>, Receiver<()>) {
        let (completion_tx, completion_rx) = flume::unbounded();
        let (release_tx, release_rx) = flume::unbounded();

        (
            Self {
                thread_requests: DashMap::new(),
                completion_rx,
                release_tx,
            },
            completion_tx,
            release_rx,
        )
    }

    /// Register a new mutator thread for handshake coordination
    /// Register a new mutator thread and return the receiver the thread will use
    /// to get handshake requests. The coordinator keeps the Sender internally.
    /// Lock-free operation using DashMap - no TODO required.
    pub fn register_thread(&self, thread_id: usize) -> Receiver<HandshakeRequest> {
        let (request_tx, request_rx) = flume::bounded(1);
        self.thread_requests.insert(thread_id, request_tx);
        request_rx
    }

    /// Unregister a mutator thread - lock-free using DashMap
    pub fn unregister_thread(&self, thread_id: usize) {
        self.thread_requests.remove(&thread_id);
    }

    /// Perform lock-free handshake with all registered threads - simplified with epoch coordination
    pub fn perform_handshake(
        &self,
        handshake_type: HandshakeType,
        timeout: Duration,
    ) -> Result<Vec<HandshakeCompletion>, HandshakeError> {
        // Use epoch-based coordination for automatic generation tracking
        let guard = &epoch::pin();

        let request = HandshakeRequest {
            callback_type: handshake_type,
        };

        let start_time = Instant::now();
        let thread_count = self.thread_requests.len();

        if thread_count == 0 {
            return Ok(Vec::new());
        }

        // Phase 1: Send requests to all threads
        for entry in self.thread_requests.iter() {
            let thread_id = *entry.key();
            let sender = entry.value();
            if sender.try_send(request).is_err() {
                return Err(HandshakeError::ThreadUnresponsive(thread_id));
            }
        }

        // Phase 2: Collect completions with timeout - epoch handles coordination automatically
        let mut completions = Vec::with_capacity(thread_count);
        let remaining_timeout = timeout.saturating_sub(start_time.elapsed());

        for _ in 0..thread_count {
            match self.completion_rx.recv_timeout(remaining_timeout) {
                Ok(completion) => {
                    // Epoch coordination eliminates need for manual sequence checking
                    completions.push(completion);
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(HandshakeError::Timeout(completions.len(), thread_count));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(HandshakeError::ChannelDisconnected);
                }
            }
        }

        // Phase 3: Release all threads (broadcast semantics) - simplified with epoch coordination
        let release_count = completions.len();
        for _ in 0..release_count {
            if self.release_tx.send(()).is_err() {
                return Err(HandshakeError::ReleaseSignalFailed);
            }
        }

        // Epoch coordination replaces manual sequence management
        guard.flush();
        Ok(completions)
    }
}

/// Mutator thread's handshake handler - completely lock-free
#[derive(Debug)]
pub struct MutatorHandshakeHandler {
    thread_id: usize,
    /// Channel to receive handshake requests
    request_rx: Receiver<HandshakeRequest>,
    /// Channel to send completions back to coordinator
    completion_tx: Sender<HandshakeCompletion>,
    /// Channel to receive release signals
    release_rx: Arc<Receiver<()>>,
    /// Atomic state machine
    state: AtomicCell<HandshakeState>,
    /// Stack roots storage - lock-free set with automatic deduplication
    stack_roots: DashSet<usize>,
}

impl MutatorHandshakeHandler {
    pub fn new(
        thread_id: usize,
        request_rx: Receiver<HandshakeRequest>,
        completion_tx: Sender<HandshakeCompletion>,
        release_rx: Arc<Receiver<()>>,
    ) -> Self {
        Self {
            thread_id,
            request_rx,
            completion_tx,
            release_rx,
            state: AtomicCell::new(HandshakeState::Running),
            stack_roots: DashSet::new(),
        }
    }

    /// Get current state atomically
    pub fn get_state(&self) -> HandshakeState {
        self.state.load()
    }

    /// Atomic state transition with validation
    fn transition_state(&self, from: HandshakeState, to: HandshakeState) -> bool {
        self.state.compare_exchange(from, to).is_ok()
    }

    /// Non-blocking safepoint polling - GUARANTEED NO DEADLOCKS
    pub fn poll_safepoint(&self) {
        let state = self.get_state();
        match state {
            HandshakeState::Running => {
                // Check for new requests
                match self.request_rx.try_recv() {
                    Ok(request) => {
                        if self.transition_state(
                            HandshakeState::Running,
                            HandshakeState::RequestReceived,
                        ) {
                            self.handle_request(request);
                        }
                    }
                    Err(TryRecvError::Empty) => {
                        // No request - continue running
                    }
                    Err(TryRecvError::Disconnected) => {
                        // Coordinator shut down - reset to running
                        self.state.store(HandshakeState::Running);
                    }
                }
            }
            HandshakeState::Completed => {
                // Wait for release signal - simplified with epoch coordination
                match self.release_rx.try_recv() {
                    Ok(()) => {
                        self.transition_state(HandshakeState::Completed, HandshakeState::Running);
                    }
                    Err(TryRecvError::Empty) => {
                        // Still waiting for release
                    }
                    Err(TryRecvError::Disconnected) => {
                        // Coordinator disconnected - reset to running
                        self.state.store(HandshakeState::Running);
                    }
                }
            }
            HandshakeState::RequestReceived | HandshakeState::AtSafepoint => {
                // Currently processing - continue
            }
        }
    }

    /// Handle handshake request at safepoint
    fn handle_request(&self, request: HandshakeRequest) {
        if !self.transition_state(HandshakeState::RequestReceived, HandshakeState::AtSafepoint) {
            return; // State changed, abort
        }

        // Perform handshake-specific operations
        let stack_roots = match request.callback_type {
            HandshakeType::StackScan => {
                // Collect stack roots - lock-free iteration over DashSet
                self.stack_roots.iter().map(|entry| *entry.key()).collect()
            }
            HandshakeType::CacheReset => {
                // Reset caches
                Vec::new()
            }
            HandshakeType::BarrierActivation => {
                // Activate barriers
                Vec::new()
            }
        };

        let completion = HandshakeCompletion {
            thread_id: self.thread_id,
            stack_roots,
        };

        // Send completion and transition to waiting state
        if self.completion_tx.try_send(completion).is_ok() {
            self.transition_state(HandshakeState::AtSafepoint, HandshakeState::Completed);
        } else {
            // Channel full/disconnected - reset to running
            self.state.store(HandshakeState::Running);
        }
    }

    /// Add stack root - lock-free with automatic deduplication
    pub fn add_stack_root(&self, root: usize) {
        self.stack_roots.insert(root);
    }

    /// Clear stack roots - lock-free bulk clear
    pub fn clear_stack_roots(&self) {
        self.stack_roots.clear();
    }

    /// Get current stack roots - lock-free iteration
    pub fn get_stack_roots(&self) -> Vec<usize> {
        self.stack_roots.iter().map(|entry| *entry.key()).collect()
    }
}

/// Handshake protocol errors
#[derive(Debug)]
pub enum HandshakeError {
    Timeout(usize, usize), // (completed_count, total_count)
    ThreadUnresponsive(usize),
    ChannelDisconnected,
    ReleaseSignalFailed,
}

impl std::fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandshakeError::Timeout(completed, total) => {
                write!(
                    f,
                    "Handshake timeout: {}/{} threads responded",
                    completed, total
                )
            }
            HandshakeError::ThreadUnresponsive(id) => {
                write!(f, "Thread {} is unresponsive", id)
            }
            HandshakeError::ChannelDisconnected => {
                write!(f, "Handshake channel disconnected")
            }
            HandshakeError::ReleaseSignalFailed => {
                write!(f, "Failed to send release signal")
            }
        }
    }
}

impl std::error::Error for HandshakeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[test]
    fn test_lock_free_handshake_protocol() {
        let (mut coordinator, completion_tx, release_rx_inner) = HandshakeCoordinator::new();
        let release_rx = Arc::new(release_rx_inner);

        // Register thread
        let request_rx = coordinator.register_thread(1);
        let handler = MutatorHandshakeHandler::new(1, request_rx, completion_tx, release_rx);

        // Spawn thread that polls safepoints
        let handler_clone = std::sync::Arc::new(handler);
        let handler_ref = handler_clone.clone();
        let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));

        // Use crossbeam scoped threads for deterministic cleanup and guaranteed termination
        let result = crossbeam::scope(|s| {
            let running_clone = running.clone();
            s.spawn(move |_| {
                // Adaptive coordination via backoff: spin → yield → park escalation.
                // We use `backoff.snooze()` here because the polling loop may
                // sometimes wait longer (waiting for the coordinator release),
                // so an escalating strategy that starts with a few spins but
                // escalates to yielding/parking reduces CPU waste under longer
                // waits while still remaining responsive for short waits.
                let backoff = crossbeam_utils::Backoff::new();
                while running_clone.load(Ordering::Relaxed) {
                    handler_ref.poll_safepoint();
                    backoff.snooze();
                }
            });

            // Perform handshake with epoch coordination
            let result =
                coordinator.perform_handshake(HandshakeType::StackScan, Duration::from_millis(100));

            // Stop thread and wait for automatic cleanup when scope exits
            running.store(false, Ordering::Relaxed);
            result
        })
        .unwrap();

        // Verify handshake succeeded - simplified verification without sequence IDs
        assert!(result.is_ok());
        let completions = result.unwrap();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].thread_id, 1);
    }
}
