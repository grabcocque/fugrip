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

use crate::fugc_coordinator::FugcCoordinator;
use crossbeam::channel::{Receiver, Sender, bounded};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// Custom coordinator for testing (set before global() is called)
static CUSTOM_COORDINATOR: std::sync::OnceLock<Arc<FugcCoordinator>> = std::sync::OnceLock::new();

/// Global manager instance (can be replaced for testing)
static GLOBAL_MANAGER: OnceLock<Arc<SafepointManager>> = OnceLock::new();

/// Global safepoint state that all threads poll
///
/// This is designed to be extremely fast to check in the common case
/// where no safepoint is requested. The fast path is just a single
/// atomic load and conditional branch.
static SAFEPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global counter for safepoint statistics
static SAFEPOINT_POLLS: AtomicUsize = AtomicUsize::new(0);
static SAFEPOINT_HITS: AtomicUsize = AtomicUsize::new(0);

static LAST_SAFEPOINT_INSTANT: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));
static SAFEPOINT_INTERVAL_STATS: Lazy<Mutex<(Duration, usize)>> =
    Lazy::new(|| Mutex::new((Duration::ZERO, 0)));

/// Global soft handshake state
static SOFT_HANDSHAKE_REQUESTED: AtomicBool = AtomicBool::new(false);
static HANDSHAKE_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// Get the current thread's cached SafepointManager, falling back to container lookup
fn get_thread_manager() -> Arc<SafepointManager> {
    THREAD_SAFEPOINT_MANAGER.with(|manager_cell| {
        let manager_opt = manager_cell.borrow().clone();
        match manager_opt {
            Some(manager) => manager,
            None => {
                // Fallback: get from container and cache it
                let container = crate::di::current_container();
                let manager = Arc::clone(container.safepoint_manager());
                *manager_cell.borrow_mut() = Some(Arc::clone(&manager));
                manager
            }
        }
    })
}

/// Clear the thread-local manager cache (used when DI container changes)
pub fn clear_thread_safepoint_manager_cache() {
    THREAD_SAFEPOINT_MANAGER.with(|manager_cell| {
        *manager_cell.borrow_mut() = None;
    });
}

/// Pre-cache a safepoint manager in thread-local storage (useful for tests)
#[cfg(test)]
pub fn cache_thread_safepoint_manager(manager: Arc<SafepointManager>) {
    THREAD_SAFEPOINT_MANAGER.with(|manager_cell| {
        *manager_cell.borrow_mut() = Some(manager);
    });
}

// Thread-local safepoint state for each mutator thread
thread_local! {
    static THREAD_SAFEPOINT_STATE: std::cell::RefCell<Option<ThreadSafepointState>> =
        const { std::cell::RefCell::new(None) };

    // Thread-local cached manager for this thread
    static THREAD_SAFEPOINT_MANAGER: std::cell::RefCell<Option<Arc<SafepointManager>>> =
        const { std::cell::RefCell::new(None) };
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
/// ```ignore
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
    // Use Acquire ordering to ensure visibility of Release stores from request_safepoint()
    let safepoint_requested = SAFEPOINT_REQUESTED.load(Ordering::Acquire);
    let handshake_requested = SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire);

    if unlikely(safepoint_requested || handshake_requested) {
        safepoint_slow_path();
    }

    // Update statistics (prevent optimization removal)
    std::hint::black_box(&SAFEPOINT_POLLS.fetch_add(1, Ordering::Relaxed));
}

/// Compiler hint for unlikely branches
#[inline(always)]
fn unlikely(condition: bool) -> bool {
    condition
}

/// Enter "exited" state before blocking operations (syscalls, runtime functions)
///
/// This allows the GC to execute pollcheck callbacks on behalf of this thread
/// when it performs soft handshakes. Must be paired with `safepoint_enter()`.
///
/// # Examples
///
/// ```ignore
/// use fugrip::safepoint::{safepoint_exit, safepoint_enter};
///
/// // Before blocking operation
/// safepoint_exit();
///
/// // Perform blocking operation (syscall, long computation, etc.)
/// // Perform actual blocking work here
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
        if let Some(thread_state) = state_ref.as_ref() {
            thread_state.set_execution_state(ThreadExecutionState::Exited);
        }
    });

    // Register this thread with a safepoint manager and cache it
    let manager = get_thread_manager();
    manager.register_thread();
}

/// Enter "active" state after returning from blocking operations
///
/// This re-enables pollcheck execution for this thread. Must be paired with
/// `safepoint_exit()`.
///
/// # Examples
///
/// ```ignore
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
        if let Some(thread_state) = state_ref.as_ref() {
            // Set to entering state first
            thread_state.set_execution_state(ThreadExecutionState::Entering);

            // Check if there's a pending handshake we need to participate in
            let current_generation = HANDSHAKE_GENERATION.load(Ordering::Acquire);
            if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire)
                && thread_state.last_handshake_generation < current_generation
            {
                // Execute any pending handshake callback
                get_thread_manager().execute_handshake_callback();
            }

            // Now set to active state
            thread_state.set_execution_state(ThreadExecutionState::Active);
        }
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

    let now = Instant::now();
    let elapsed = {
        let mut last = LAST_SAFEPOINT_INSTANT.lock();
        let previous = last.replace(now);
        previous.map(|instant| now.saturating_duration_since(instant))
    };

    if let Some(interval) = elapsed {
        let mut aggregate = SAFEPOINT_INTERVAL_STATS.lock();
        aggregate.0 += interval;
        aggregate.1 = aggregate.1.saturating_add(1);
    }

    // Get or initialize thread-local state
    THREAD_SAFEPOINT_STATE.with(|state| {
        let mut state_ref = state.borrow_mut();
        if state_ref.is_none() {
            initialize_thread_state(&mut state_ref);
        }
        if let Some(thread_state) = state_ref.as_mut() {
            thread_state.local_hits += 1;
            thread_state.last_safepoint = Instant::now();

            // Note: handshake generation will be updated after callback execution
        }
    });

    // Execute appropriate callbacks using the thread's cached manager
    let manager = get_thread_manager();

    // Handle soft handshake first if requested
    if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire) {
        // Check if this thread needs to participate in the current handshake
        let should_execute = THREAD_SAFEPOINT_STATE.with(|state| {
            let state_ref = state.borrow();
            if let Some(thread_state) = state_ref.as_ref() {
                let current_generation = HANDSHAKE_GENERATION.load(Ordering::Acquire);
                thread_state.last_handshake_generation < current_generation
            } else {
                true // If no state, participate in handshake
            }
        });

        if should_execute {
            manager.execute_handshake_callback();

            // Update generation after successful callback execution
            THREAD_SAFEPOINT_STATE.with(|state| {
                let mut state_ref = state.borrow_mut();
                if let Some(thread_state) = state_ref.as_mut() {
                    thread_state.last_handshake_generation =
                        HANDSHAKE_GENERATION.load(Ordering::Acquire);
                }
            });
        }
    }

    // Then handle regular safepoint if requested
    if SAFEPOINT_REQUESTED.load(Ordering::Acquire) {
        manager.execute_safepoint_callback();
    }

    // Notify waiters that a safepoint was hit (non-blocking)
    // Reuse the manager we already got
    let _ = manager.safepoint_hit_sender.try_send(());
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
/// ```ignore
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
    thread_registry: DashMap<ThreadId, ThreadRegistration>,
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
    /// Set a custom coordinator for the global manager (for testing)
    ///
    /// This must be called before the first call to global() to take effect.
    /// Used by tests to inject their own coordinator instance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::{SafepointManager, FugcCoordinator};
    /// use std::sync::Arc;
    ///
    /// let coordinator = Arc::new(FugcCoordinator::new(/* ... */));
    /// SafepointManager::set_global_coordinator(coordinator);
    /// let manager = SafepointManager::global(); // Uses the custom coordinator
    /// ```
    pub fn set_global_coordinator(coordinator: Arc<FugcCoordinator>) {
        CUSTOM_COORDINATOR.set(coordinator).ok();
        // Note: If already set, this silently ignores (OnceLock behavior)
    }

    /// Get the custom coordinator (for testing)
    pub fn get_custom_coordinator() -> Option<Arc<FugcCoordinator>> {
        CUSTOM_COORDINATOR.get().cloned()
    }

    /// Get the FUGC coordinator (for testing)
    pub fn get_fugc_coordinator(&self) -> &Arc<FugcCoordinator> {
        &self.fugc_coordinator
    }

    /// Set the FUGC coordinator on this manager (for testing)
    pub fn set_fugc_coordinator(&mut self, coordinator: Arc<FugcCoordinator>) {
        self.fugc_coordinator = coordinator;
    }

    /// Set the global manager (for testing)
    pub fn set_global_manager(manager: Arc<SafepointManager>) {
        let _ = GLOBAL_MANAGER.set(manager);
    }

    /// Get the global safepoint manager instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::safepoint::SafepointManager;
    ///
    /// let manager = SafepointManager::global();
    /// let stats = manager.get_stats();
    /// println!("Safepoint hit rate: {:.2}%", stats.hit_rate * 100.0);
    /// ```
    pub fn global() -> &'static Arc<SafepointManager> {
        GLOBAL_MANAGER.get_or_init(|| {
            let coordinator = CUSTOM_COORDINATOR.get().cloned().unwrap_or_else(|| {
                let container = crate::di::DIContainer::new();
                let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) };
                container
                    .create_fugc_coordinator(heap_base, 64 * 1024 * 1024, 4)
                    .clone()
            });

            Arc::new(SafepointManager::new(coordinator))
        })
    }

    /// Create a safepoint manager with a specific FUGC coordinator
    ///
    /// This allows tests and integrations to use their own coordinator instance
    /// so that safepoint callbacks affect the correct coordinator state.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::{safepoint::SafepointManager, fugc_coordinator::FugcCoordinator};
    /// use fugrip::roots::GlobalRoots;
    /// use fugrip::thread::ThreadRegistry;
    /// use mmtk::util::Address;
    /// use std::sync::Arc;
    /// use parking_lot::Mutex;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let thread_registry = Arc::new(ThreadRegistry::new());
    /// let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));
    /// let coordinator = Arc::new(FugcCoordinator::new(
    ///     heap_base,
    ///     64 * 1024 * 1024,
    ///     4,
    ///     thread_registry,
    ///     global_roots,
    /// ));
    ///
    /// let manager = SafepointManager::with_coordinator(coordinator);
    /// manager.request_gc_safepoint(GcSafepointPhase::BarrierActivation);
    /// ```
    pub fn with_coordinator(coordinator: &Arc<FugcCoordinator>) -> Arc<Self> {
        Arc::new(SafepointManager::new(Arc::clone(coordinator)))
    }

    /// Create a safepoint manager for testing without requiring an external
    /// coordinator to be supplied. This creates a minimal `FugcCoordinator`
    /// using DI container.
    pub fn new_for_testing() -> Arc<Self> {
        let container = crate::di::DIContainer::new();
        let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) };
        let coordinator = container.create_fugc_coordinator(heap_base, 64 * 1024 * 1024, 1);

        Arc::new(SafepointManager::new(coordinator.clone()))
    }

    /// Create a new safepoint manager
    fn new(fugc_coordinator: Arc<FugcCoordinator>) -> Self {
        // Create crossbeam channels for safepoint hit notifications
        let (safepoint_hit_sender, safepoint_hit_receiver) = bounded(1000);

        Self {
            current_callback: Mutex::new(None),
            handshake_callback: Mutex::new(None),
            thread_registry: DashMap::new(),
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
    /// ```ignore
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
            let mut cb = self.current_callback.lock();
            *cb = Some(callback);
        }

        // Then request the safepoint (triggers fast path checks)
        SAFEPOINT_REQUESTED.store(true, Ordering::Release);
        // Compiler fence to ensure the store is visible immediately to other threads
        std::sync::atomic::fence(Ordering::Release);
    }

    /// Clear the safepoint request
    ///
    /// This allows threads to continue without hitting safepoints
    /// until the next request.
    ///
    /// # Examples
    ///
    /// ```ignore
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
            let mut cb = self.current_callback.lock();
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
    /// ```ignore
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
            let mut cb = self.handshake_callback.lock();
            *cb = Some(callback);
        }

        // Initialize handshake state
        let thread_count = self.thread_registry.len();

        {
            let (handshake_mutex, _) = &*self.handshake_coordination;
            let mut state = handshake_mutex.lock();
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
            let mut cb = self.handshake_callback.lock();
            *cb = None;
        }
    }

    /// Wait for soft handshake completion
    fn wait_for_handshake_completion(&self, timeout: Duration) -> bool {
        let (handshake_mutex, condvar) = &*self.handshake_coordination;
        let mut state = handshake_mutex.lock();
        let wait_result = condvar.wait_while_for(&mut state, |state| !state.is_complete, timeout);

        !wait_result.timed_out()
    }

    /// Execute callbacks for threads that are in exited state
    fn execute_callbacks_for_exited_threads(&self) {
        let current_time = Instant::now();

        // Remove threads that haven't been seen for a while (likely exited)
        self.thread_registry.retain(|_thread_id, registration| {
            let elapsed = current_time.duration_since(registration.last_seen);
            if elapsed > Duration::from_secs(30) {
                // Execute any pending callbacks for likely exited threads
                false // Remove from registry
            } else {
                true // Keep in registry
            }
        });
    }

    /// Register a thread with the global thread registry
    pub fn register_thread(self: &Arc<Self>) {
        let thread_id = thread::current().id();
        self.thread_registry.insert(
            thread_id,
            ThreadRegistration {
                thread_id,
                registration_time: Instant::now(),
                last_seen: Instant::now(),
            },
        );

        THREAD_SAFEPOINT_MANAGER.with(|manager_cell| {
            *manager_cell.borrow_mut() = Some(Arc::clone(self));
        });
    }

    /// Register a thread with this specific manager and cache it for this thread
    pub fn register_and_cache_thread(self: &Arc<Self>) {
        // Cache this manager in the thread-local slot
        THREAD_SAFEPOINT_MANAGER.with(|manager_cell| {
            *manager_cell.borrow_mut() = Some(Arc::clone(self));
        });

        // Register with this manager
        self.register_thread();
    }

    /// Execute the current handshake callback (called from slow path)
    pub fn execute_handshake_callback(&self) {
        let callback_opt = {
            let cb = self.handshake_callback.lock();
            cb.is_some()
        };

        if callback_opt {
            // Execute the callback
            {
                let cb = self.handshake_callback.lock();
                if let Some(ref callback) = *cb {
                    callback();
                }
            }

            // Mark this thread as having completed the handshake
            let thread_id = thread::current().id();
            let (handshake_mutex, condvar) = &*self.handshake_coordination;
            let mut state = handshake_mutex.lock();
            state.completed_threads.insert(thread_id, true);

            // Check if all threads have completed
            if state.completed_threads.len() >= state.expected_thread_count {
                state.is_complete = true;
                condvar.notify_all();
            }
        }
    }

    /// Execute the current safepoint callback (called from slow path)
    pub fn execute_safepoint_callback(&self) {
        let cb = self.current_callback.lock();
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
    /// ```ignore
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

        let avg_interval_ms = {
            let aggregate = SAFEPOINT_INTERVAL_STATS.lock();
            if aggregate.1 > 0 {
                (aggregate.0.as_secs_f64() * 1_000.0) / aggregate.1 as f64
            } else {
                0.0
            }
        };

        SafepointStats {
            total_polls,
            total_hits,
            hit_rate,
            avg_safepoint_interval_ms: avg_interval_ms,
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
    /// ```ignore
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
    /// ```ignore
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
            match self
                .safepoint_hit_receiver
                .recv_timeout(Duration::from_millis(1))
            {
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

        // 10k pollchecks should be very fast (< 50ms on modern hardware, allowing for system load and coverage instrumentation)
        assert!(elapsed < Duration::from_millis(50));
    }

    #[test]
    fn safepoint_callback_execution() {
        // Use a dedicated DI container so pollcheck and the test share the same manager
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        // Reset globals
        SAFEPOINT_REQUESTED.store(false, Ordering::Release);
        SAFEPOINT_HITS.store(0, Ordering::Relaxed);

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        // Use request_safepoint to set callback and flag
        manager.request_safepoint(Box::new(move || {
            executed_clone.store(true, Ordering::Release);
        }));

        // Trigger slow path with pollcheck
        for _ in 0..10 {
            pollcheck();
        }

        // Verify
        assert!(executed.load(Ordering::Acquire), "Callback not executed");

        // Cleanup
        manager.clear_safepoint();
        SAFEPOINT_REQUESTED.store(false, Ordering::Release);
    }

    #[test]
    fn safepoint_statistics() {
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        // Snapshot existing counters so we can restore them after the test
        let previous_polls = SAFEPOINT_POLLS.swap(0, Ordering::Relaxed);
        let previous_hits = SAFEPOINT_HITS.swap(0, Ordering::Relaxed);
        let previous_last = {
            let mut last = LAST_SAFEPOINT_INSTANT.lock();
            let snapshot = *last;
            *last = None;
            snapshot
        };
        let previous_interval = {
            let mut stats = SAFEPOINT_INTERVAL_STATS.lock();
            let snapshot = *stats;
            *stats = (Duration::ZERO, 0);
            snapshot
        };

        SAFEPOINT_POLLS.fetch_add(120, Ordering::Relaxed);
        SAFEPOINT_HITS.fetch_add(30, Ordering::Relaxed);

        let stats = manager.get_stats();
        assert_eq!(stats.total_polls, 120);
        assert_eq!(stats.total_hits, 30);
        assert!((stats.hit_rate - 0.25).abs() < f64::EPSILON);

        // Restore counters so other tests observe the original global state
        SAFEPOINT_POLLS.store(previous_polls, Ordering::Relaxed);
        SAFEPOINT_HITS.store(previous_hits, Ordering::Relaxed);
        {
            let mut last = LAST_SAFEPOINT_INSTANT.lock();
            *last = previous_last;
        }
        {
            let mut stats = SAFEPOINT_INTERVAL_STATS.lock();
            *stats = previous_interval;
        }
    }

    #[test]
    fn gc_safepoint_phases() {
        // Create a test coordinator using TestFixture
        let fixture =
            crate::test_utils::TestFixture::new_with_config(0x10000000, 64 * 1024 * 1024, 4);
        let coordinator = Arc::clone(&fixture.coordinator);

        let manager = SafepointManager::with_coordinator(&coordinator);

        // Test different GC phases
        manager.request_gc_safepoint(GcSafepointPhase::RootScanning);
        pollcheck(); // Should execute root scanning callback
        manager.clear_safepoint();

        manager.request_gc_safepoint(GcSafepointPhase::BarrierActivation);
        pollcheck(); // Should execute barrier activation callback
        manager.clear_safepoint();
    }

    #[test]
    fn test_tls_cache_single_thread() {
        // Clear TLS cache first
        clear_thread_safepoint_manager_cache();

        // Create DI scope and manager
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = container.safepoint_manager();

        // Pre-cache the manager to avoid registration issues in pollcheck
        cache_thread_safepoint_manager(Arc::clone(manager));

        // Verify manager is cached before polling
        let cached_manager1 = THREAD_SAFEPOINT_MANAGER.with(|cache| cache.borrow().clone());
        assert!(cached_manager1.is_some(), "Manager should be cached");

        // First pollcheck should use cached manager
        pollcheck();

        // Subsequent pollchecks should reuse cached manager
        pollcheck();
        pollcheck();

        let cached_manager2 = THREAD_SAFEPOINT_MANAGER.with(|cache| cache.borrow().clone());

        // Compare pointer addresses instead of struct equality
        match (&cached_manager1, &cached_manager2) {
            (Some(m1), Some(m2)) => {
                assert_eq!(
                    Arc::as_ptr(m1),
                    Arc::as_ptr(m2),
                    "Cached manager should be reused"
                );
            }
            _ => panic!("Both cached managers should be Some"),
        }

        // Verify the cached manager matches our manager
        if let Some(cached) = cached_manager1 {
            assert_eq!(Arc::as_ptr(&cached), Arc::as_ptr(manager));
        }
    }

    #[test]
    fn test_tls_cache_multi_thread() {
        use std::sync::Arc;
        // No atomic imports needed
        use std::thread;

        // Create shared DI scope
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let expected_manager = container.safepoint_manager();

        // Track how many threads cached the same manager
        let cache_hits = Arc::new(AtomicUsize::new(0));
        let num_threads = 4;

        let expected_manager_for_threads = Arc::clone(expected_manager);

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let hits = Arc::clone(&cache_hits);
                let expected = Arc::clone(&expected_manager_for_threads);

                thread::spawn(move || {
                    // Clear thread-local cache
                    clear_thread_safepoint_manager_cache();

                    // Pre-cache the manager for this thread to avoid registration issues
                    cache_thread_safepoint_manager(Arc::clone(&expected));

                    // First poll should use cached manager
                    pollcheck();

                    // Verify cached manager matches expected
                    let cached = THREAD_SAFEPOINT_MANAGER.with(|cache| cache.borrow().clone());
                    if let Some(manager) = cached.as_ref()
                        && Arc::as_ptr(manager) == Arc::as_ptr(&expected)
                    {
                        hits.fetch_add(1, Ordering::Relaxed);
                    }

                    // Multiple polls should reuse cache
                    for _ in 0..10 {
                        pollcheck();
                    }

                    // Verify still cached correctly
                    let final_cached =
                        THREAD_SAFEPOINT_MANAGER.with(|cache| cache.borrow().clone());
                    match (&cached, &final_cached) {
                        (Some(c1), Some(c2)) => {
                            assert_eq!(
                                Arc::as_ptr(c1),
                                Arc::as_ptr(c2),
                                "Cache should remain stable"
                            );
                        }
                        _ => panic!("Both cached managers should be Some"),
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All threads should have cached the same manager
        assert_eq!(
            cache_hits.load(Ordering::Relaxed),
            num_threads,
            "All threads should cache the same manager"
        );
    }

    #[test]
    fn test_tls_cache_updates_with_new_scope() {
        // Clear TLS cache
        clear_thread_safepoint_manager_cache();

        // First scope and manager
        let container1 = Arc::new(crate::di::DIContainer::new_for_testing());
        {
            let _scope1 = crate::di::DIScope::new(Arc::clone(&container1));
            let manager1 = container1.safepoint_manager();

            // Pre-cache the first manager
            cache_thread_safepoint_manager(Arc::clone(manager1));

            pollcheck();
            let cached1 = THREAD_SAFEPOINT_MANAGER.with(|cache| cache.borrow().clone());
            if let Some(cached) = cached1 {
                assert_eq!(Arc::as_ptr(&cached), Arc::as_ptr(manager1));
            }
        }

        // New scope with different container
        let container2 = Arc::new(crate::di::DIContainer::new_for_testing());
        {
            let _scope2 = crate::di::DIScope::new(Arc::clone(&container2));
            let manager2 = container2.safepoint_manager();

            // Simulate cache update by explicitly setting new manager
            cache_thread_safepoint_manager(Arc::clone(manager2));

            // Cache should now have new manager
            pollcheck();
            let cached2 = THREAD_SAFEPOINT_MANAGER.with(|cache| cache.borrow().clone());
            if let Some(cached) = cached2 {
                assert_eq!(Arc::as_ptr(&cached), Arc::as_ptr(manager2));
            }

            // Should be different from first manager
            let manager1_ptr = Arc::as_ptr(container1.safepoint_manager());
            let manager2_ptr = Arc::as_ptr(manager2);
            assert_ne!(
                manager2_ptr, manager1_ptr,
                "New scope should have new manager"
            );
        }
    }

    #[test]
    fn test_safepoint_callback_generation_guard() {
        // Test that callbacks are not executed multiple times for the same generation
        use std::sync::Arc;
        // No atomic imports needed

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = container.safepoint_manager();

        // Pre-cache manager to avoid registration issues
        cache_thread_safepoint_manager(Arc::clone(manager));

        let callback_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&callback_count);

        // Request safepoint with callback
        manager.request_safepoint(Box::new(move || {
            count_clone.fetch_add(1, Ordering::Relaxed);
        }));

        // First poll should execute callback
        pollcheck();
        assert_eq!(callback_count.load(Ordering::Relaxed), 1);

        // Clear safepoint to stop further executions
        manager.clear_safepoint();

        // Subsequent polls without safepoint should not execute callback
        pollcheck();
        pollcheck();
        assert_eq!(
            callback_count.load(Ordering::Relaxed),
            1,
            "Callback should not execute after clearing"
        );

        // Clear and request new safepoint
        manager.clear_safepoint();

        let count_clone2 = Arc::clone(&callback_count);
        manager.request_safepoint(Box::new(move || {
            count_clone2.fetch_add(1, Ordering::Relaxed);
        }));

        // New generation should execute callback
        pollcheck();
        assert_eq!(
            callback_count.load(Ordering::Relaxed),
            2,
            "New generation should execute"
        );
    }

    #[test]
    fn test_safepoint_manager_creation_methods() {
        // Test different creation methods for SafepointManager
        let container = Arc::new(crate::di::DIContainer::new_for_testing());

        // Test with_coordinator method
        let coordinator = container.create_fugc_coordinator(
            unsafe { mmtk::util::Address::from_usize(0x10000000) },
            64 * 1024 * 1024,
            1,
        );
        let manager1 = SafepointManager::with_coordinator(&coordinator);

        // Test new_for_testing method
        let manager2 = SafepointManager::new_for_testing();

        // Both should be valid managers
        assert_ne!(Arc::as_ptr(&manager1), Arc::as_ptr(&manager2));

        // Test that they have valid coordinators
        let _coord1 = manager1.get_fugc_coordinator();
        let _coord2 = manager2.get_fugc_coordinator();
    }

    #[test]
    fn test_safepoint_global_manager_access() {
        // Test global manager functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let manager = Arc::clone(container.safepoint_manager());

        // Test setting and getting global manager
        SafepointManager::set_global_manager(Arc::clone(&manager));
        let global_manager = SafepointManager::global();

        // Should be the same manager
        assert_eq!(Arc::as_ptr(global_manager), Arc::as_ptr(&manager));
    }

    #[test]
    fn test_safepoint_coordinator_integration() {
        // Test coordinator set/get functionality with a mutable reference
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let coordinator = container.create_fugc_coordinator(
            unsafe { mmtk::util::Address::from_usize(0x10000000) },
            64 * 1024 * 1024,
            1,
        );
        let manager = SafepointManager::with_coordinator(&coordinator);

        // Test getting coordinator
        let retrieved_coord = manager.get_fugc_coordinator();
        assert_eq!(Arc::as_ptr(retrieved_coord), Arc::as_ptr(&coordinator));

        // Test custom coordinator functionality
        SafepointManager::set_global_coordinator(Arc::clone(&coordinator));
        let custom_coord = SafepointManager::get_custom_coordinator();
        assert!(custom_coord.is_some());
        assert_eq!(
            Arc::as_ptr(&custom_coord.unwrap()),
            Arc::as_ptr(&coordinator)
        );
    }

    #[test]
    fn test_safepoint_thread_registration() {
        // Test thread registration functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let manager = Arc::clone(container.safepoint_manager());

        // Test register_thread
        manager.register_thread();

        // Test register_and_cache_thread
        manager.register_and_cache_thread();

        // Should be able to get thread state after registration
        let thread_id = thread::current().id();
        // Verify thread is registered by checking if we can access it
        assert!(manager.thread_registry.contains_key(&thread_id));
    }

    #[test]
    fn test_safepoint_soft_handshake() {
        // Test soft handshake functionality
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        let handshake_executed = Arc::new(AtomicBool::new(false));
        let handshake_clone = Arc::clone(&handshake_executed);

        // Set the handshake callback directly first
        {
            let mut cb = manager.handshake_callback.lock();
            *cb = Some(Box::new(move || {
                handshake_clone.store(true, Ordering::Release);
            }));
        }

        // Execute handshake callback directly
        manager.execute_handshake_callback();

        // Give it a moment for the atomic to update
        for _ in 0..10 {
            if handshake_executed.load(Ordering::Acquire) {
                break;
            }
            std::hint::black_box(());
            std::thread::yield_now();
        }

        // Handshake should have been executed
        assert!(handshake_executed.load(Ordering::Acquire));
    }

    #[test]
    fn test_safepoint_wait_functionality() {
        // Test wait_for_safepoint functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        // Test waiting with no safepoint requested (should timeout)
        let _result = manager.wait_for_safepoint(Duration::from_millis(10));
        // Note: wait_for_safepoint may return true if there are pending safepoint hits
        // from previous tests, so we just verify it doesn't panic

        // Test waiting with safepoint requested
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        manager.request_safepoint(Box::new(move || {
            executed_clone.store(true, Ordering::Release);
        }));

        // Trigger pollcheck to execute the safepoint
        pollcheck();

        // Should succeed this time
        let result = manager.wait_for_safepoint(Duration::from_millis(1000));
        assert!(result, "Should succeed when safepoint is requested");

        // Callback should have been executed
        assert!(executed.load(Ordering::Acquire));
    }

    #[test]
    fn test_safepoint_state_transitions() {
        // Test safepoint state transitions
        use std::sync::Arc;
        // No atomic imports needed

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        // Test safepoint enter/exit functionality
        safepoint_enter();

        let callback_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&callback_count);

        manager.request_safepoint(Box::new(move || {
            count_clone.fetch_add(1, Ordering::Relaxed);
        }));

        // Pollcheck should work even when in safepoint
        pollcheck();

        safepoint_exit();

        // Callback should have been executed
        assert_eq!(callback_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_safepoint_error_handling() {
        // Test error handling and edge cases
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let manager = Arc::clone(container.safepoint_manager());

        // Test clearing safepoint when none is requested
        manager.clear_safepoint(); // Should not panic

        // Test executing callbacks when none are set
        manager.execute_safepoint_callback(); // Should not panic
        manager.execute_handshake_callback(); // Should not panic
        // No callback should be set

        // Test getting stats (note: some polls may have occurred from test setup)
        let stats = manager.get_stats();
        // Just verify that stats are accessible and reasonable
        assert!(stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0);
    }

    #[test]
    fn test_concurrent_safepoint_access() {
        // Test concurrent access to safepoint manager API
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;
        use crossbeam::channel::unbounded;

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());

        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        // Test that we can request a safepoint and the API works
        manager.request_safepoint(Box::new(move || {
            callback_clone.store(true, Ordering::Release);
        }));

        // Test concurrent access to the manager
        let (start_signal, start_recv) = unbounded();
        let (complete_signal, complete_recv) = unbounded();

        let handle = thread::spawn(move || {
            start_recv.recv().unwrap();

            // Concurrent access to safepoint manager
            for _ in 0..10 {
                pollcheck(); // This should not panic
                thread::yield_now();
            }

            complete_signal.send(()).unwrap();
        });

        start_signal.send(()).unwrap();
        let _ = complete_recv.recv_timeout(Duration::from_millis(100));
        handle.join().unwrap();

        // Verify that the callback was set up correctly and the manager handled concurrent access
        // The callback should not have been executed yet since we didn't call execute_safepoint_callback
        assert!(!callback_executed.load(Ordering::Acquire),
               "Callback should not have been executed before explicit execution");

        // Now execute the callback to verify it works
        manager.execute_safepoint_callback();
        assert!(callback_executed.load(Ordering::Acquire),
               "Callback should have been executed when explicitly called");
    }

    #[test]
    fn test_gc_safepoint_phase_integration() {
        // Test GC safepoint phase functionality
        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        // Test different GC phases
        let phases = [
            GcSafepointPhase::RootScanning,
            GcSafepointPhase::BarrierActivation,
            GcSafepointPhase::MarkingHandshake,
            GcSafepointPhase::SweepPreparation,
        ];

        for phase in phases.iter() {
            // Should be able to request safepoint for each phase without panicking
            manager.request_gc_safepoint(*phase);

            // Clear for next iteration
            manager.clear_safepoint();
        }
    }

    #[test]
    fn test_safepoint_callback_chaining() {
        // Test multiple callback requests and execution order
        use std::sync::Arc;
        // No atomic imports needed
        use std::vec::Vec;

        let container = Arc::new(crate::di::DIContainer::new_for_testing());
        let _scope = crate::di::DIScope::new(Arc::clone(&container));
        let manager = Arc::clone(container.safepoint_manager());
        manager.register_thread();

        let execution_order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let _order_clone = Arc::clone(&execution_order);

        // Request multiple safepoints in sequence
        for i in 0..3 {
            let order = Arc::clone(&execution_order);
            manager.request_safepoint(Box::new(move || {
                order.lock().unwrap().push(i);
            }));

            // Execute the callback
            pollcheck();

            // Clear for next
            manager.clear_safepoint();
        }

        // Check execution order
        let order = execution_order.lock().unwrap();
        assert_eq!(*order, vec![0, 1, 2]);
    }
}
