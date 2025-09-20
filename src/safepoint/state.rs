//! Thread state management and pollcheck implementation

use std::sync::atomic::{AtomicU8, Ordering};
use std::thread::ThreadId;
use std::time::Instant;

use super::globals::{SAFEPOINT_HITS, SAFEPOINT_POLLS, SAFEPOINT_REQUESTED, get_thread_manager};

/// Thread execution state for safepoint coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadExecutionState {
    /// Thread is running normal code
    Running,
    /// Thread is at a safepoint
    AtSafepoint,
    /// Thread is blocked in system call or I/O
    Blocked,
}

/// Per-thread safepoint state
#[derive(Debug)]
pub(super) struct ThreadSafepointState {
    /// Current execution state
    state: AtomicU8,
    /// Thread ID for debugging
    thread_id: ThreadId,
    /// Last safepoint time for interval tracking
    last_safepoint: Option<Instant>,
    /// Number of safepoints this thread has hit
    local_hits: usize,
}

impl ThreadSafepointState {
    pub(super) fn new(thread_id: ThreadId) -> Self {
        Self {
            state: AtomicU8::new(ThreadExecutionState::Running as u8),
            thread_id,
            last_safepoint: None,
            local_hits: 0,
        }
    }

    pub(super) fn get_state(&self) -> ThreadExecutionState {
        match self.state.load(Ordering::Acquire) {
            0 => ThreadExecutionState::Running,
            1 => ThreadExecutionState::AtSafepoint,
            2 => ThreadExecutionState::Blocked,
            _ => ThreadExecutionState::Running, // Default
        }
    }

    pub(super) fn set_state(&self, state: ThreadExecutionState) {
        self.state.store(state as u8, Ordering::Release);
    }

    pub(super) fn record_safepoint_hit(&mut self) {
        self.last_safepoint = Some(Instant::now());
        self.local_hits += 1;
    }

    pub(super) fn local_hits(&self) -> usize {
        self.local_hits
    }
}

/// Fast pollcheck implementation
///
/// This is the primary entry point for safepoint checks. It's designed
/// to be extremely fast in the common case where no safepoint is requested.
///
/// # Examples
///
/// ```ignore
/// use fugrip::safepoint::pollcheck;
///
/// loop {
///     // Do some work
///     pollcheck(); // Fast check, rarely taken
/// }
/// ```
pub fn pollcheck() {
    // Increment poll counter for statistics
    SAFEPOINT_POLLS.fetch_add(1, Ordering::Relaxed);

    // Fast path: no safepoint requested (this should be the common case)
    if unlikely(!SAFEPOINT_REQUESTED.load(Ordering::Acquire)) {
        return;
    }

    // Slow path: safepoint is requested
    safepoint_slow_path();
}

/// Branch prediction hint for unlikely conditions
#[inline(always)]
fn unlikely(condition: bool) -> bool {
    #[cold]
    fn cold() {}
    if !condition {
        cold()
    }
    condition
}

/// Thread-local safepoint state
thread_local! {
    static THREAD_STATE: std::cell::RefCell<Option<ThreadSafepointState>> =
        std::cell::RefCell::new(None);
}

/// Exit safepoint state (called when leaving a safepoint)
///
/// This transitions the thread from `AtSafepoint` back to `Running` state.
/// It's typically called automatically by the safepoint infrastructure.
///
/// # Examples
///
/// ```ignore
/// use fugrip::safepoint::{safepoint_enter, safepoint_exit};
///
/// safepoint_enter();
/// // ... perform safepoint work ...
/// safepoint_exit();
/// ```
pub fn safepoint_exit() {
    THREAD_STATE.with(|state| {
        let state = state.borrow_mut();
        if let Some(ref thread_state) = *state {
            thread_state.set_state(ThreadExecutionState::Running);
        }
    });
}

/// Enter safepoint state (called when entering a safepoint)
///
/// This transitions the thread from `Running` to `AtSafepoint` state.
/// It's typically called automatically by the safepoint infrastructure.
///
/// # Examples
///
/// ```ignore
/// use fugrip::safepoint::{safepoint_enter, safepoint_exit};
///
/// safepoint_enter();
/// // ... perform safepoint work ...
/// safepoint_exit();
/// ```
pub fn safepoint_enter() {
    THREAD_STATE.with(|state| {
        let mut state = state.borrow_mut();
        initialize_thread_state(&mut state);
        if let Some(ref thread_state) = *state {
            thread_state.set_state(ThreadExecutionState::AtSafepoint);
        }
    });
}

/// Initialize thread state if not already done
fn initialize_thread_state(state_ref: &mut Option<ThreadSafepointState>) {
    if state_ref.is_none() {
        let thread_id = std::thread::current().id();
        *state_ref = Some(ThreadSafepointState::new(thread_id));

        // Register this thread with the global manager
        let manager = get_thread_manager();
        manager.register_thread();
    }
}

/// Slow path for safepoint processing
///
/// This function is called when a safepoint is actually requested.
/// It executes the safepoint callback and updates statistics.
#[cold]
fn safepoint_slow_path() {
    // Update hit statistics
    SAFEPOINT_HITS.fetch_add(1, Ordering::Relaxed);

    // Get the thread manager and execute safepoint callback
    let manager = get_thread_manager();

    // Enter safepoint state
    THREAD_STATE.with(|state| {
        let mut state = state.borrow_mut();
        initialize_thread_state(&mut state);

        if let Some(ref mut thread_state) = *state {
            thread_state.set_state(ThreadExecutionState::AtSafepoint);
            thread_state.record_safepoint_hit();
        }
    });

    // Execute the safepoint callback
    manager.execute_safepoint_callback();

    // Execute handshake callback if needed
    manager.execute_handshake_callback();

    // Send safepoint hit notification via event bus
    if let Ok(()) = manager
        .event_bus_sender()
        .send(super::events::GcEvent::SafepointHit)
    {
        // Event sent successfully
    }

    // Exit safepoint state
    THREAD_STATE.with(|state| {
        let state = state.borrow();
        if let Some(ref thread_state) = *state {
            thread_state.set_state(ThreadExecutionState::Running);
        }
    });
}
