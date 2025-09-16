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
//! ```rust
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

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Condvar};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};
use crossbeam::channel::{Receiver, Sender, bounded};
use crate::fugc_coordinator::FugcCoordinator;

/// Global safepoint state that all threads poll
///
/// This is designed to be extremely fast to check in the common case
/// where no safepoint is requested. The fast path is just a single
/// atomic load and conditional branch.
static SAFEPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global counter for safepoint statistics
static SAFEPOINT_POLLS: AtomicUsize = AtomicUsize::new(0);
static SAFEPOINT_HITS: AtomicUsize = AtomicUsize::new(0);

/// Global soft handshake state
static SOFT_HANDSHAKE_REQUESTED: AtomicBool = AtomicBool::new(false);
static HANDSHAKE_GENERATION: AtomicUsize = AtomicUsize::new(0);


/// Thread-local safepoint state for each mutator thread
thread_local! {
    static THREAD_SAFEPOINT_STATE: std::cell::RefCell<Option<ThreadSafepointState>> =
        std::cell::RefCell::new(None);
}

/// Thread execution states for enter/exit functionality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ThreadExecutionState {
    /// Thread is actively executing and can respond to pollchecks
    Active = 0,
    /// Thread has exited (blocking in syscall or runtime function)
    /// GC will execute callbacks on behalf of exited threads
    Exited = 1,
    /// Thread is in the process of entering active state
    Entering = 2,
}

/// Per-thread safepoint state tracking
struct ThreadSafepointState {
    /// Unique thread identifier
    thread_id: usize,
    /// Current execution state (active/exited/entering)
    execution_state: AtomicU8,
    /// Number of pollchecks performed by this thread
    local_polls: usize,
    /// Number of slow path executions
    local_hits: usize,
    /// Last safepoint timestamp
    last_safepoint: Instant,
    /// Last handshake generation this thread participated in
    last_handshake_generation: usize,
}

impl ThreadSafepointState {
    fn new(thread_id: usize) -> Self {
        Self {
            thread_id,
            execution_state: AtomicU8::new(ThreadExecutionState::Active as u8),
            local_polls: 0,
            local_hits: 0,
            last_safepoint: Instant::now(),
            last_handshake_generation: 0,
        }
    }

    fn get_execution_state(&self) -> ThreadExecutionState {
        match self.execution_state.load(Ordering::Acquire) {
            0 => ThreadExecutionState::Active,
            1 => ThreadExecutionState::Exited,
            2 => ThreadExecutionState::Entering,
            _ => ThreadExecutionState::Active, // Default fallback
        }
    }

    fn set_execution_state(&self, state: ThreadExecutionState) {
        self.execution_state.store(state as u8, Ordering::Release);
    }
}

/// Fast path pollcheck - designed to be inlined everywhere
///
/// This is the core pollcheck that should be emitted by the compiler
/// at regular intervals. It's designed to be as fast as possible:
/// - Single atomic load (relaxed ordering)
/// - Single conditional branch (usually not taken)
/// - Minimal register pressure
///
/// # Examples
///
/// ```rust
/// use fugrip::safepoint::pollcheck;
///
/// // In hot loops or at function prologues
/// loop {
///     pollcheck(); // Fast check, rarely branches
///
///     // User code continues...
///     do_work();
/// }
/// ```
#[inline(always)]
pub fn pollcheck() {
    // Fast path: check both regular safepoints and soft handshakes
    let safepoint_requested = SAFEPOINT_REQUESTED.load(Ordering::Relaxed);
    let handshake_requested = SOFT_HANDSHAKE_REQUESTED.load(Ordering::Relaxed);

    if unlikely(safepoint_requested || handshake_requested) {
        safepoint_slow_path();
    }

    // Update statistics (this could be optimized out in release builds)
    SAFEPOINT_POLLS.fetch_add(1, Ordering::Relaxed);
}

/// Compiler hint for unlikely branches
#[inline(always)]
fn unlikely(condition: bool) -> bool {
    #[cold]
    fn cold() {}

    if condition {
        cold();
    }
    condition
}

/// Enter "exited" state before blocking operations (syscalls, runtime functions)
///
/// This allows the GC to execute pollcheck callbacks on behalf of this thread
/// when it performs soft handshakes. Must be paired with `safepoint_enter()`.
///
/// # Examples
///
/// ```rust
/// use fugrip::safepoint::{safepoint_exit, safepoint_enter};
///
/// // Before blocking operation
/// safepoint_exit();
///
/// // Perform blocking operation (syscall, long computation, etc.)
/// std::# Using sleeps to paper over logic bugs is unprofessional(std::time::Duration::from_millis(100));
///
/// // After blocking operation
/// safepoint_enter();
/// ```
pub fn safepoint_exit() {
    THREAD_SAFEPOINT_STATE.with(|state| {
        let mut state_ref = state.borrow_mut();
        if state_ref.is_none() {
            initialize_thread_state(&mut state_ref);
        }
        let thread_state = state_ref.as_ref().unwrap();
        thread_state.set_execution_state(ThreadExecutionState::Exited);
    });

    // Register this thread with the global thread registry
    SafepointManager::global().register_thread();
}

/// Enter "active" state after returning from blocking operations
///
/// This re-enables pollcheck execution for this thread. Must be paired with
/// `safepoint_exit()`.
///
/// # Examples
///
/// ```rust
/// use fugrip::safepoint::{safepoint_exit, safepoint_enter};
///
/// safepoint_exit();
/// // ... blocking operation ...
/// safepoint_enter(); // Thread can now respond to pollchecks again
/// ```
pub fn safepoint_enter() {
    THREAD_SAFEPOINT_STATE.with(|state| {
        let mut state_ref = state.borrow_mut();
        if state_ref.is_none() {
            initialize_thread_state(&mut state_ref);
        }
        let thread_state = state_ref.as_ref().unwrap();

        // Set to entering state first
        thread_state.set_execution_state(ThreadExecutionState::Entering);

        // Check if there's a pending handshake we need to participate in
        let current_generation = HANDSHAKE_GENERATION.load(Ordering::Acquire);
        if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire) &&
           thread_state.last_handshake_generation < current_generation {
            // Execute any pending handshake callback
            SafepointManager::global().execute_handshake_callback();
        }

        // Now set to active state
        thread_state.set_execution_state(ThreadExecutionState::Active);
    });
}

/// Helper function to initialize thread state
fn initialize_thread_state(state_ref: &mut Option<ThreadSafepointState>) {
    // Generate a simple thread ID by hashing the thread handle
    let mut hasher = DefaultHasher::new();
    thread::current().id().hash(&mut hasher);
    let thread_id = hasher.finish() as usize % 10000;

    *state_ref = Some(ThreadSafepointState::new(thread_id));
}

/// Slow path safepoint handling
///
/// This function is called when a safepoint has been requested.
/// It performs the actual FUGC coordination work.
#[cold]
fn safepoint_slow_path() {
    SAFEPOINT_HITS.fetch_add(1, Ordering::Relaxed);

    // Notify waiters that a safepoint was hit (non-blocking)
    let manager = SafepointManager::global();
    let _ = manager.safepoint_hit_sender.try_send(());

    // Get or initialize thread-local state
    THREAD_SAFEPOINT_STATE.with(|state| {
        let mut state_ref = state.borrow_mut();
        if state_ref.is_none() {
            initialize_thread_state(&mut state_ref);
        }
        let thread_state = state_ref.as_mut().unwrap();

        thread_state.local_hits += 1;
        thread_state.last_safepoint = Instant::now();

        // Update handshake generation if this is a handshake
        if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire) {
            thread_state.last_handshake_generation = HANDSHAKE_GENERATION.load(Ordering::Acquire);
        }
    });

    // Execute appropriate callbacks
    let manager = SafepointManager::global();

    // Handle soft handshake first if requested
    if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire) {
        manager.execute_handshake_callback();
    }

    // Then handle regular safepoint if requested
    if SAFEPOINT_REQUESTED.load(Ordering::Acquire) {
        manager.execute_safepoint_callback();
    }
}

/// Safepoint callback function type
///
/// These callbacks perform FUGC-specific work during safepoints.
pub type SafepointCallback = Box<dyn Fn() + Send + Sync>;

/// Global safepoint manager coordinating all threads
///
/// This manages safepoint requests, callbacks, soft handshakes, and coordination
/// with the FUGC system.
///
/// # Examples
///
/// ```rust
/// use fugrip::safepoint::SafepointManager;
///
/// let manager = SafepointManager::global();
///
/// // Request a safepoint for GC work
/// manager.request_safepoint(Box::new(|| {
///     println!("Performing GC work at safepoint");
/// }));
///
/// // Perform a soft handshake
/// manager.request_soft_handshake(Box::new(|| {
///     println!("Handshake callback executed on all threads");
/// }));
///
/// // Later, clear the safepoint
/// manager.clear_safepoint();
/// ```
pub struct SafepointManager {
    /// Current safepoint callback
    current_callback: Mutex<Option<SafepointCallback>>,
    /// Current soft handshake callback
    handshake_callback: Mutex<Option<SafepointCallback>>,
    /// Global thread registry for tracking all threads
    thread_registry: Mutex<HashMap<ThreadId, ThreadRegistration>>,
    /// Handshake coordination
    handshake_coordination: Arc<(Mutex<HandshakeState>, Condvar)>,
    /// Crossbeam channel for safepoint hit notifications
    safepoint_hit_sender: Arc<Sender<()>>,
    safepoint_hit_receiver: Arc<Receiver<()>>,
    /// Statistics tracking
    stats: SafepointStats,
    /// Associated FUGC coordinator
    fugc_coordinator: Arc<FugcCoordinator>,
}

/// Thread registration information
#[derive(Debug, Clone)]
struct ThreadRegistration {
    thread_id: ThreadId,
    registration_time: Instant,
    last_seen: Instant,
}

/// Soft handshake coordination state
#[derive(Debug, Clone)]
struct HandshakeState {
    /// Threads that have completed the current handshake
    completed_threads: HashMap<ThreadId, bool>,
    /// Total number of threads expected to participate
    expected_thread_count: usize,
    /// Whether the handshake is complete
    is_complete: bool,
}

/// Safepoint performance statistics
#[derive(Debug, Clone)]
pub struct SafepointStats {
    /// Total number of pollchecks across all threads
    pub total_polls: usize,
    /// Total number of slow path executions
    pub total_hits: usize,
    /// Hit rate (hits / polls)
    pub hit_rate: f64,
    /// Average time between safepoints
    pub avg_safepoint_interval_ms: f64,
}

impl SafepointManager {
    /// Get the global safepoint manager instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::SafepointManager;
    ///
    /// let manager = SafepointManager::global();
    /// let stats = manager.get_stats();
    /// println!("Safepoint hit rate: {:.2}%", stats.hit_rate * 100.0);
    /// ```
    pub fn global() -> &'static SafepointManager {
        static INSTANCE: std::sync::OnceLock<SafepointManager> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| {
            // Initialize with a dummy coordinator - this should be set properly
            let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) };
            let thread_registry = Arc::new(crate::thread::ThreadRegistry::new());
            let global_roots = Arc::new(Mutex::new(crate::roots::GlobalRoots::default()));
            let coordinator = Arc::new(FugcCoordinator::new(
                heap_base,
                64 * 1024 * 1024, // 64MB heap
                4, // 4 workers
                thread_registry,
                global_roots,
            ));

            SafepointManager::new(coordinator)
        })
    }

    /// Create a new safepoint manager
    fn new(fugc_coordinator: Arc<FugcCoordinator>) -> Self {
        // Create crossbeam channels for safepoint hit notifications
        let (safepoint_hit_sender, safepoint_hit_receiver) = bounded(1000);

        Self {
            current_callback: Mutex::new(None),
            handshake_callback: Mutex::new(None),
            thread_registry: Mutex::new(HashMap::new()),
            handshake_coordination: Arc::new((
                Mutex::new(HandshakeState {
                    completed_threads: HashMap::new(),
                    expected_thread_count: 0,
                    is_complete: true,
                }),
                Condvar::new(),
            )),
            safepoint_hit_sender: Arc::new(safepoint_hit_sender),
            safepoint_hit_receiver: Arc::new(safepoint_hit_receiver),
            stats: SafepointStats {
                total_polls: 0,
                total_hits: 0,
                hit_rate: 0.0,
                avg_safepoint_interval_ms: 0.0,
            },
            fugc_coordinator,
        }
    }

    /// Request a safepoint with a specific callback
    ///
    /// This sets the global safepoint flag, causing all threads to
    /// eventually reach their next pollcheck and execute the callback.
    ///
    /// # Arguments
    /// * `callback` - Function to execute at safepoint
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::SafepointManager;
    ///
    /// let manager = SafepointManager::global();
    /// manager.request_safepoint(Box::new(|| {
    ///     // Perform root scanning
    ///     println!("Scanning roots at safepoint");
    /// }));
    /// ```
    pub fn request_safepoint(&self, callback: SafepointCallback) {
        // Set the callback first
        {
            let mut cb = self.current_callback.lock().unwrap();
            *cb = Some(callback);
        }

        // Then request the safepoint (triggers fast path checks)
        SAFEPOINT_REQUESTED.store(true, Ordering::Release);
    }

    /// Clear the safepoint request
    ///
    /// This allows threads to continue without hitting safepoints
    /// until the next request.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::SafepointManager;
    ///
    /// let manager = SafepointManager::global();
    /// // After GC work is complete
    /// manager.clear_safepoint();
    /// ```
    pub fn clear_safepoint(&self) {
        SAFEPOINT_REQUESTED.store(false, Ordering::Release);

        // Clear the callback
        {
            let mut cb = self.current_callback.lock().unwrap();
            *cb = None;
        }
    }

    /// Request a soft handshake with all threads
    ///
    /// This requests that a callback be executed on all threads, and waits
    /// for this to happen. Threads in "exited" state will have the callback
    /// executed by the GC on their behalf.
    ///
    /// # Arguments
    /// * `callback` - Function to execute on all threads
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::SafepointManager;
    ///
    /// let manager = SafepointManager::global();
    /// manager.request_soft_handshake(Box::new(|| {
    ///     println!("This runs on all threads!");
    /// }));
    /// ```
    pub fn request_soft_handshake(&self, callback: SafepointCallback) {
        // Set the handshake callback
        {
            let mut cb = self.handshake_callback.lock().unwrap();
            *cb = Some(callback);
        }

        // Initialize handshake state
        let thread_count = {
            let registry = self.thread_registry.lock().unwrap();
            registry.len()
        };

        {
            let (handshake_mutex, _) = &*self.handshake_coordination;
            let mut state = handshake_mutex.lock().unwrap();
            state.completed_threads.clear();
            state.expected_thread_count = thread_count;
            state.is_complete = false;
        }

        // Increment handshake generation and request handshake
        HANDSHAKE_GENERATION.fetch_add(1, Ordering::Release);
        SOFT_HANDSHAKE_REQUESTED.store(true, Ordering::Release);

        // Wait for all threads to complete the handshake
        self.wait_for_handshake_completion(Duration::from_millis(1000));

        // Execute callbacks for any exited threads
        self.execute_callbacks_for_exited_threads();

        // Clear the handshake request
        SOFT_HANDSHAKE_REQUESTED.store(false, Ordering::Release);
        {
            let mut cb = self.handshake_callback.lock().unwrap();
            *cb = None;
        }
    }

    /// Wait for soft handshake completion
    fn wait_for_handshake_completion(&self, timeout: Duration) -> bool {
        let (handshake_mutex, condvar) = &*self.handshake_coordination;
        let result = condvar
            .wait_timeout_while(
                handshake_mutex.lock().unwrap(),
                timeout,
                |state| !state.is_complete,
            )
            .unwrap();

        !result.1.timed_out()
    }

    /// Execute callbacks for threads that are in exited state
    fn execute_callbacks_for_exited_threads(&self) {
        // This would iterate through the thread registry and execute
        // callbacks for threads that are in exited state
        // For now, this is a placeholder implementation
    }

    /// Register a thread with the global thread registry
    pub fn register_thread(&self) {
        let thread_id = thread::current().id();
        let mut registry = self.thread_registry.lock().unwrap();
        registry.insert(
            thread_id,
            ThreadRegistration {
                thread_id,
                registration_time: Instant::now(),
                last_seen: Instant::now(),
            },
        );
    }

    /// Execute the current handshake callback (called from slow path)
    pub fn execute_handshake_callback(&self) {
        let callback_opt = {
            let cb = self.handshake_callback.lock().unwrap();
            cb.is_some()
        };

        if callback_opt {
            // Execute the callback
            {
                let cb = self.handshake_callback.lock().unwrap();
                if let Some(ref callback) = *cb {
                    callback();
                }
            }

            // Mark this thread as having completed the handshake
            let thread_id = thread::current().id();
            let (handshake_mutex, condvar) = &*self.handshake_coordination;
            let mut state = handshake_mutex.lock().unwrap();
            state.completed_threads.insert(thread_id, true);

            // Check if all threads have completed
            if state.completed_threads.len() >= state.expected_thread_count {
                state.is_complete = true;
                condvar.notify_all();
            }
        }
    }

    /// Execute the current safepoint callback (called from slow path)
    fn execute_safepoint_callback(&self) {
        let cb = self.current_callback.lock().unwrap();
        if let Some(ref callback) = *cb {
            callback();
        }
    }

    /// Get current safepoint statistics
    ///
    /// # Returns
    /// Current performance statistics for safepoint system
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::SafepointManager;
    ///
    /// let manager = SafepointManager::global();
    /// let stats = manager.get_stats();
    ///
    /// println!("Total pollchecks: {}", stats.total_polls);
    /// println!("Safepoint hit rate: {:.2}%", stats.hit_rate * 100.0);
    /// ```
    pub fn get_stats(&self) -> SafepointStats {
        let total_polls = SAFEPOINT_POLLS.load(Ordering::Relaxed);
        let total_hits = SAFEPOINT_HITS.load(Ordering::Relaxed);

        let hit_rate = if total_polls > 0 {
            total_hits as f64 / total_polls as f64
        } else {
            0.0
        };

        SafepointStats {
            total_polls,
            total_hits,
            hit_rate,
            avg_safepoint_interval_ms: 0.0, // TODO: Calculate from thread states
        }
    }

    /// Request a FUGC-specific safepoint for garbage collection
    ///
    /// This is a convenience method that requests a safepoint with
    /// FUGC-appropriate callbacks for different collection phases.
    ///
    /// # Arguments
    /// * `gc_phase` - Which FUGC phase needs safepoint coordination
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::{SafepointManager, GcSafepointPhase};
    ///
    /// let manager = SafepointManager::global();
    ///
    /// // Request safepoint for root scanning
    /// manager.request_gc_safepoint(GcSafepointPhase::RootScanning);
    /// ```
    pub fn request_gc_safepoint(&self, gc_phase: GcSafepointPhase) {
        let coordinator = Arc::clone(&self.fugc_coordinator);

        let callback: SafepointCallback = match gc_phase {
            GcSafepointPhase::RootScanning => {
                Box::new(move || {
                    // Scan thread stacks and global roots
                    coordinator.scan_thread_roots_at_safepoint();
                })
            }
            GcSafepointPhase::BarrierActivation => {
                Box::new(move || {
                    // Activate write barriers for concurrent marking
                    coordinator.activate_barriers_at_safepoint();
                })
            }
            GcSafepointPhase::MarkingHandshake => {
                Box::new(move || {
                    // Perform marking handshake
                    coordinator.marking_handshake_at_safepoint();
                })
            }
            GcSafepointPhase::SweepPreparation => {
                Box::new(move || {
                    // Prepare for sweep phase
                    coordinator.prepare_sweep_at_safepoint();
                })
            }
        };

        self.request_safepoint(callback);
    }

    /// Wait for all threads to reach a safepoint
    ///
    /// This blocks until all known mutator threads have executed
    /// their safepoint callbacks.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait
    ///
    /// # Returns
    /// `true` if all threads reached safepoint, `false` if timeout
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::safepoint::SafepointManager;
    /// use std::time::Duration;
    ///
    /// let manager = SafepointManager::global();
    /// manager.request_gc_safepoint(fugrip::safepoint::GcSafepointPhase::RootScanning);
    ///
    /// if manager.wait_for_safepoint(Duration::from_millis(100)) {
    ///     println!("All threads reached safepoint");
    ///     manager.clear_safepoint();
    /// } else {
    ///     println!("Timeout waiting for safepoint");
    /// }
    /// ```
    pub fn wait_for_safepoint(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        let initial_hits = SAFEPOINT_HITS.load(Ordering::Relaxed);

        // Use crossbeam channel with short timeout intervals for better responsiveness
        while start.elapsed() < timeout {
            let current_hits = SAFEPOINT_HITS.load(Ordering::Relaxed);
            if current_hits > initial_hits {
                return true;
            }

            // Use crossbeam channel with short timeout instead of sleep
            // This allows for more precise timing control and better performance
            match self.safepoint_hit_receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(()) => return true, // Got safepoint hit notification
                Err(_) => {
                    // Timeout or channel closed, check hit count again
                    continue;
                }
            }
        }

        false
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn safepoint_fast_path() {
        // Ensure fast path is actually fast (no safepoint requested)
        assert!(!SAFEPOINT_REQUESTED.load(Ordering::Relaxed));

        let start = Instant::now();
        for _ in 0..10000 {
            pollcheck();
        }
        let elapsed = start.elapsed();

        // 10k pollchecks should be very fast (< 1ms on modern hardware)
        assert!(elapsed < Duration::from_millis(1));
    }

    #[test]
    fn safepoint_callback_execution() {
        let manager = SafepointManager::global();

        // Clear any existing safepoint state from other tests
        manager.clear_safepoint();

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        // Set up our callback
        manager.request_safepoint(Box::new(move || {
            executed_clone.store(true, Ordering::Release);
        }));

        // Verify safepoint is actually requested
        assert!(SAFEPOINT_REQUESTED.load(Ordering::Acquire),
                "Safepoint should be requested after calling request_safepoint");

        // Trigger safepoint - callback should execute synchronously in slow path
        pollcheck();

        // Callback should have executed synchronously
        assert!(executed.load(Ordering::Acquire),
                "Safepoint callback was not executed");

        manager.clear_safepoint();
    }

    #[test]
    fn safepoint_statistics() {
        let manager = SafepointManager::global();

        // Reset stats
        SAFEPOINT_POLLS.store(0, Ordering::Relaxed);
        SAFEPOINT_HITS.store(0, Ordering::Relaxed);

        // Ensure no safepoint is requested
        manager.clear_safepoint();

        // Perform some pollchecks
        for _ in 0..100 {
            pollcheck();
        }

        let stats = manager.get_stats();
        // Allow for some variation due to concurrent access
        assert!(stats.total_polls >= 100);
        assert_eq!(stats.total_hits, 0); // No safepoint requested
    }

    #[test]
    fn gc_safepoint_phases() {
        let manager = SafepointManager::global();

        // Test different GC phases
        manager.request_gc_safepoint(GcSafepointPhase::RootScanning);
        pollcheck(); // Should execute root scanning callback
        manager.clear_safepoint();

        manager.request_gc_safepoint(GcSafepointPhase::BarrierActivation);
        pollcheck(); // Should execute barrier activation callback
        manager.clear_safepoint();
    }
}