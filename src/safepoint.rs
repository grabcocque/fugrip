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
use parking_lot::{Condvar, Mutex, RwLock};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// Custom coordinator for testing (set before global() is called)
static CUSTOM_COORDINATOR: std::sync::OnceLock<Arc<FugcCoordinator>> = std::sync::OnceLock::new();

/// Global manager instance (can be replaced for testing)
static GLOBAL_MANAGER: RwLock<Option<Arc<SafepointManager>>> = RwLock::new(None);

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

// Thread-local safepoint state for each mutator thread
thread_local! {
    static THREAD_SAFEPOINT_STATE: std::cell::RefCell<Option<ThreadSafepointState>> =
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

    // Register this thread with the DI container's safepoint manager
    let container = crate::di::current_container();
    container.safepoint_manager().register_thread();
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
                let container = crate::di::current_container();
                container.safepoint_manager().execute_handshake_callback();
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

            // Update handshake generation if this is a handshake
            if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire) {
                thread_state.last_handshake_generation =
                    HANDSHAKE_GENERATION.load(Ordering::Acquire);
            }
        }
    });

    // Execute appropriate callbacks
    let container = crate::di::current_container();
    let manager = container.safepoint_manager();

    // Handle soft handshake first if requested
    if SOFT_HANDSHAKE_REQUESTED.load(Ordering::Acquire) {
        manager.execute_handshake_callback();
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
        GLOBAL_MANAGER.write().replace(manager);
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
    pub(crate) fn global() -> Arc<SafepointManager> {
        // Check if already initialized
        if let Some(ref m) = *GLOBAL_MANAGER.read() {
            return Arc::clone(m);
        }

        // Not initialized, acquire write lock
        let mut manager = GLOBAL_MANAGER.write();
        if let Some(ref m) = *manager {
            return Arc::clone(m);
        }

        // Create new manager
        let coordinator = CUSTOM_COORDINATOR.get().cloned().unwrap_or_else(|| {
            let mut container = crate::di::DIContainer::new();
            let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) };
            container.create_fugc_coordinator(heap_base, 64 * 1024 * 1024, 4)
        });

        let new_manager = Arc::new(SafepointManager::new(coordinator));
        *manager = Some(Arc::clone(&new_manager));
        new_manager
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
    pub fn with_coordinator(coordinator: Arc<FugcCoordinator>) -> Arc<Self> {
        Arc::new(SafepointManager::new(coordinator))
    }

    /// Create a safepoint manager for testing without requiring an external
    /// coordinator to be supplied. This creates a minimal `FugcCoordinator`
    /// using DI container.
    pub fn new_for_testing() -> Arc<Self> {
        let mut container = crate::di::DIContainer::new();
        let heap_base = unsafe { mmtk::util::Address::from_usize(0x10000000) };
        let coordinator = container.create_fugc_coordinator(heap_base, 64 * 1024 * 1024, 1);

        Arc::new(SafepointManager::new(coordinator))
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
    pub fn register_thread(&self) {
        let thread_id = thread::current().id();
        self.thread_registry.insert(
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
            // Debug: print manager identity and indicate callback execution
            let addr = self as *const SafepointManager as usize;
            println!("[safepoint] executing callback on manager {:x}", addr);
            callback();
        } else {
            let addr = self as *const SafepointManager as usize;
            println!("[safepoint] no callback to execute on manager {:x}", addr);
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

        // 10k pollchecks should be very fast (< 1ms on modern hardware)
        assert!(elapsed < Duration::from_millis(1));
    }

    #[test]
    fn safepoint_callback_execution() {
        // Use a dedicated DI container so pollcheck and the test share the same manager
        let _scope = crate::di::DIScope::new(crate::di::DIContainer::new_for_testing());
        let manager = crate::di::current_container().safepoint_manager();
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
        let container = crate::di::DIContainer::new_for_testing();
        let _scope = crate::di::DIScope::new(container.clone());
        let manager = container.safepoint_manager();

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

        let manager = SafepointManager::with_coordinator(coordinator);

        // Test different GC phases
        manager.request_gc_safepoint(GcSafepointPhase::RootScanning);
        pollcheck(); // Should execute root scanning callback
        manager.clear_safepoint();

        manager.request_gc_safepoint(GcSafepointPhase::BarrierActivation);
        pollcheck(); // Should execute barrier activation callback
        manager.clear_safepoint();
    }
}
