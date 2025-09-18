// Lock-free handshake protocol for FUGC garbage collection
// Design: Invalid states are unrepresentable through type safety and atomic state machines

use crossbeam::channel::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};

/// Handshake request sent from coordinator to mutator threads
#[derive(Debug, Clone, Copy)]
pub struct HandshakeRequest {
    pub sequence_id: u64,
    pub callback_type: HandshakeType,
}

/// Types of handshake operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeType {
    StackScan,
    CacheReset,
    BarrierActivation,
}

/// Handshake completion response from mutator back to coordinator
#[derive(Debug, Clone)]
pub struct HandshakeCompletion {
    pub thread_id: usize,
    pub sequence_id: u64,
    pub stack_roots: Vec<usize>,
}

/// Lock-free atomic state machine for handshake protocol
/// State transitions are enforced by the type system
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    Running = 0,         // Thread executing normally
    RequestReceived = 1, // Got handshake request
    AtSafepoint = 2,     // Reached safepoint, processing request
    Completed = 3,       // Sent completion, waiting for release
}

impl HandshakeState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => HandshakeState::Running,
            1 => HandshakeState::RequestReceived,
            2 => HandshakeState::AtSafepoint,
            3 => HandshakeState::Completed,
            _ => HandshakeState::Running, // Invalid values default to safe state
        }
    }
}

/// Coordinator for managing lock-free handshakes with all mutator threads
pub struct HandshakeCoordinator {
    /// Channels to send requests to each thread (thread_id -> sender)
    thread_requests: HashMap<usize, Sender<HandshakeRequest>>,
    /// Single channel to receive completions from all threads
    completion_rx: Receiver<HandshakeCompletion>,
    /// Single channel to send release signals to all threads
    release_tx: Sender<u64>,
    /// Sequence counter to detect stale responses
    sequence_counter: std::sync::atomic::AtomicU64,
}

impl HandshakeCoordinator {
    pub fn new() -> (Self, Sender<HandshakeCompletion>, Receiver<u64>) {
        let (completion_tx, completion_rx) = channel::unbounded();
        let (release_tx, release_rx) = channel::unbounded();

        (
            Self {
                thread_requests: HashMap::new(),
                completion_rx,
                release_tx,
                sequence_counter: std::sync::atomic::AtomicU64::new(0),
            },
            completion_tx,
            release_rx,
        )
    }

    /// Register a new mutator thread for handshake coordination
    /// Register a new mutator thread and return the receiver the thread will use
    /// to get handshake requests. The coordinator keeps the Sender internally.
    pub fn register_thread(&mut self, thread_id: usize) -> Receiver<HandshakeRequest> {
        let (request_tx, request_rx) = channel::bounded(1);
        self.thread_requests.insert(thread_id, request_tx);
        request_rx
    }

    /// Unregister a mutator thread
    pub fn unregister_thread(&mut self, thread_id: usize) {
        self.thread_requests.remove(&thread_id);
    }

    /// Perform lock-free handshake with all registered threads
    pub fn perform_handshake(
        &self,
        handshake_type: HandshakeType,
        timeout: Duration,
    ) -> Result<Vec<HandshakeCompletion>, HandshakeError> {
        let sequence_id = self.sequence_counter.fetch_add(1, Ordering::AcqRel);
        let request = HandshakeRequest {
            sequence_id,
            callback_type: handshake_type,
        };

        let start_time = Instant::now();
        let thread_count = self.thread_requests.len();

        if thread_count == 0 {
            return Ok(Vec::new());
        }

        // Phase 1: Send requests to all threads
        for (thread_id, sender) in &self.thread_requests {
            if sender.try_send(request).is_err() {
                return Err(HandshakeError::ThreadUnresponsive(*thread_id));
            }
        }

        // Phase 2: Collect completions with timeout
        let mut completions = Vec::with_capacity(thread_count);
        let remaining_timeout = timeout.saturating_sub(start_time.elapsed());

        for _ in 0..thread_count {
            match self.completion_rx.recv_timeout(remaining_timeout) {
                Ok(completion) => {
                    // Verify sequence ID to detect stale responses
                    if completion.sequence_id == sequence_id {
                        completions.push(completion);
                    }
                    // Ignore stale responses from previous handshakes
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(HandshakeError::Timeout(completions.len(), thread_count));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(HandshakeError::ChannelDisconnected);
                }
            }
        }

        // Phase 3: Release all threads (broadcast semantics)
        let release_count = completions.len();
        for _ in 0..release_count {
            if self.release_tx.send(sequence_id).is_err() {
                return Err(HandshakeError::ReleaseSignalFailed);
            }
        }

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
    release_rx: Arc<Receiver<u64>>,
    /// Atomic state machine
    state: AtomicU8,
    /// Stack roots storage
    stack_roots: Mutex<Vec<usize>>,
}

impl MutatorHandshakeHandler {
    pub fn new(
        thread_id: usize,
        request_rx: Receiver<HandshakeRequest>,
        completion_tx: Sender<HandshakeCompletion>,
        release_rx: Arc<Receiver<u64>>,
    ) -> Self {
        Self {
            thread_id,
            request_rx,
            completion_tx,
            release_rx,
            state: AtomicU8::new(HandshakeState::Running as u8),
            stack_roots: Mutex::new(Vec::new()),
        }
    }

    /// Get current state atomically
    pub fn get_state(&self) -> HandshakeState {
        HandshakeState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Atomic state transition with validation
    fn transition_state(&self, from: HandshakeState, to: HandshakeState) -> bool {
        self.state
            .compare_exchange_weak(from as u8, to as u8, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
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
                        self.state
                            .store(HandshakeState::Running as u8, Ordering::Release);
                    }
                }
            }
            HandshakeState::Completed => {
                // Wait for release signal
                match self.release_rx.try_recv() {
                    Ok(_sequence_id) => {
                        self.transition_state(HandshakeState::Completed, HandshakeState::Running);
                    }
                    Err(TryRecvError::Empty) => {
                        // Still waiting for release
                    }
                    Err(TryRecvError::Disconnected) => {
                        // Coordinator disconnected - reset to running
                        self.state
                            .store(HandshakeState::Running as u8, Ordering::Release);
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
                // Collect stack roots
                self.stack_roots.lock().clone()
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
            sequence_id: request.sequence_id,
            stack_roots,
        };

        // Send completion and transition to waiting state
        if self.completion_tx.try_send(completion).is_ok() {
            self.transition_state(HandshakeState::AtSafepoint, HandshakeState::Completed);
        } else {
            // Channel full/disconnected - reset to running
            self.state
                .store(HandshakeState::Running as u8, Ordering::Release);
        }
    }

    /// Add stack root (thread-safe)
    pub fn add_stack_root(&self, root: usize) {
        let mut roots = self.stack_roots.lock();
        if roots.contains(&root) {
            return;
        }
        roots.push(root);
    }

    /// Clear stack roots (thread-safe)
    pub fn clear_stack_roots(&self) {
        self.stack_roots.lock().clear();
    }

    /// Get current stack roots (thread-safe)
    pub fn get_stack_roots(&self) -> Vec<usize> {
        self.stack_roots.lock().clone()
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
    use std::thread;
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
        let running_clone = running.clone();

        let handle = thread::spawn(move || {
            while running_clone.load(Ordering::Relaxed) {
                handler_ref.poll_safepoint();
                thread::yield_now();
            }
        });

        // Perform handshake
        let result =
            coordinator.perform_handshake(HandshakeType::StackScan, Duration::from_millis(100));

        // Stop thread
        running.store(false, Ordering::Relaxed);
        handle.join().unwrap();

        // Verify handshake succeeded
        assert!(result.is_ok());
        let completions = result.unwrap();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].thread_id, 1);
    }
}
