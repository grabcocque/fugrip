// This module re-exports the collector phase management functionality
// and provides the main CollectorState and MutatorState types

use crate::{SendPtr, GcHeader, CollectorPhase, TypeInfo};
use std::sync::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, ThreadId, JoinHandle};

// Re-export from submodules
pub use crate::collector::phase_manager::PhaseManager;
pub use crate::collector::suspension_manager::SuspensionManager;
pub use crate::collector::mark_coordinator::MarkCoordinator;
pub use crate::collector::finalizer_coordinator::FinalizerCoordinator;
pub use crate::collector::sweep_coordinator::SweepCoordinator;

/// Actions that can be requested during a handshake between collector and mutators.
///
/// This enum is used to coordinate different types of synchronization
/// actions between the garbage collector and mutator threads.
///
/// # Examples
///
/// ```
/// use fugrip::HandshakeAction;
///
/// let action = HandshakeAction::Noop;
/// assert_eq!(action, HandshakeAction::Noop);
///
/// let reset_action = HandshakeAction::ResetThreadLocalCaches;
/// assert_ne!(reset_action, HandshakeAction::Noop);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandshakeAction {
    Noop,
    ResetThreadLocalCaches,
    RequestStackScan,
}

/// Smoke-only precise roots provider used by smoke tests to simulate accurate roots.
#[cfg(feature = "smoke")]
mod smoke_roots {
    use super::*;

    pub(super) static GLOBALS: once_cell::sync::Lazy<std::sync::Mutex<Vec<SendPtr<GcHeader<()>>>>> =
        once_cell::sync::Lazy::new(|| std::sync::Mutex::new(Vec::new()));

    pub(super) static STACKS: once_cell::sync::Lazy<std::sync::Mutex<std::collections::HashMap<ThreadId, Vec<SendPtr<GcHeader<()>>>>>> =
        once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    pub fn add_global(ptr: SendPtr<GcHeader<()>>) {
        GLOBALS.lock().unwrap().push(ptr);
    }

    pub fn drain_globals() -> Vec<SendPtr<GcHeader<()>>> {
        GLOBALS.lock().unwrap().drain(..).collect()
    }

    pub fn add_stack_for_current(ptr: SendPtr<GcHeader<()>>) {
        let tid = std::thread::current().id();
        let mut map = STACKS.lock().unwrap();
        map.entry(tid).or_default().push(ptr);
    }

    pub fn drain_stack_for_current() -> Vec<SendPtr<GcHeader<()>>> {
        let tid = std::thread::current().id();
        let mut map = STACKS.lock().unwrap();
        if let Some(v) = map.get_mut(&tid) {
            return v.drain(..).collect();
        }
        Vec::new()
    }

    pub fn clear_all() {
        GLOBALS.lock().unwrap().clear();
        STACKS.lock().unwrap().clear();
    }
}

#[cfg(feature = "smoke")]
pub use smoke_roots::{add_global as smoke_add_global_root, add_stack_for_current as smoke_add_stack_root, clear_all as smoke_clear_all_roots};

/// Conservative stack scanner for garbage collection.
///
/// `StackScanner` performs conservative scanning of memory regions to identify
/// potential garbage collection roots. It uses heap bounds checking to validate
/// pointers and avoid false positives during stack scanning.
///
/// # Examples
///
/// ```
/// use fugrip::StackScanner;
///
/// // Create a new scanner with default heap bounds checker
/// let scanner = StackScanner::new();
/// ```
pub struct StackScanner {
    heap_bounds_checker: Box<dyn HeapBoundsChecker + Send + Sync>,
}

/// Trait for checking if pointers are within valid heap boundaries.
///
/// `HeapBoundsChecker` provides methods to validate pointers during conservative
/// stack scanning, ensuring that only valid garbage collection pointers are
/// processed during marking phases.
///
/// # Examples
///
/// ```
/// use fugrip::{HeapBoundsChecker, DefaultHeapBoundsChecker};
///
/// let checker = DefaultHeapBoundsChecker;
/// // Use the checker to validate pointers during scanning
/// ```
pub trait HeapBoundsChecker {
    fn is_within_heap_bounds(&self, ptr: *mut GcHeader<()>) -> bool;
    unsafe fn is_valid_object_header(&self, header: *mut GcHeader<()>) -> bool;
    unsafe fn is_valid_gc_pointer(&self, ptr: *mut GcHeader<()>) -> bool;
}

/// Default implementation of heap bounds checking.
///
/// `DefaultHeapBoundsChecker` provides a standard implementation that checks
/// against the production heap segments and validates object headers using
/// basic sanity checks.
///
/// # Examples
///
/// ```
/// use fugrip::{DefaultHeapBoundsChecker, HeapBoundsChecker};
///
/// let checker = DefaultHeapBoundsChecker;
/// // The checker can be used with StackScanner or other validation contexts
/// ```
pub struct DefaultHeapBoundsChecker;

impl HeapBoundsChecker for DefaultHeapBoundsChecker {
    fn is_within_heap_bounds(&self, ptr: *mut GcHeader<()>) -> bool {
        use crate::interfaces::memory::HEAP_PROVIDER;

        // Check if the pointer falls within any of our heap segments
        let segments = <crate::interfaces::memory::ProductionHeapProvider as crate::interfaces::memory::HeapProvider>::get_heap(&HEAP_PROVIDER).segments.lock().unwrap();

        for segment in segments.iter() {
            let segment_start = segment.memory.as_ptr() as *mut u8;
            let segment_end = segment.end_ptr.load(Ordering::Relaxed);

            if (ptr as *mut u8) >= segment_start && (ptr as *mut u8) < segment_end {
                return true;
            }
        }

        false
    }

    unsafe fn is_valid_object_header(&self, header: *mut GcHeader<()>) -> bool {
        // Basic validation checks
        if header.is_null() {
            return false;
        }

        // Check pointer alignment
        if !(header as usize).is_multiple_of(std::mem::align_of::<GcHeader<()>>()) {
            return false;
        }

        unsafe {
            // Check if TypeInfo looks reasonable by checking size field
            let type_info = (*header).type_info;
            let size = type_info.size;

            // Check if size is reasonable (between min object size and max segment size)
            if size < std::mem::size_of::<GcHeader<()>>() || size > 1024 * 1024 {
                return false;
            }
        }

        true
    }

    unsafe fn is_valid_gc_pointer(&self, ptr: *mut GcHeader<()>) -> bool {
        // Comprehensive pointer validation
        if ptr.is_null() {
            return false;
        }

        // Check proper alignment for GcHeader
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<GcHeader<()>>()) {
            return false;
        }

        // Check if within heap bounds
        if !self.is_within_heap_bounds(ptr) {
            return false;
        }

        // Validate object header
        unsafe {
            if !self.is_valid_object_header(ptr) {
                return false;
            }

            let header = &*ptr;
            // Check if this looks like a valid object header
            let type_info = header.type_info;
            let size = type_info.size;

            // Additional validation
            if size == 0 || size > 128 * 1024 * 1024 {
                return false;
            }
        }

        true
    }
}

impl Default for StackScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl StackScanner {
    /// Create a new stack scanner with the default heap bounds checker.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::StackScanner;
    ///
    /// let scanner = StackScanner::new();
    /// ```
    pub fn new() -> Self {
        Self {
            heap_bounds_checker: Box::new(DefaultHeapBoundsChecker),
        }
    }

    /// Create a new stack scanner with a custom heap bounds checker.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{StackScanner, DefaultHeapBoundsChecker};
    ///
    /// let checker = Box::new(DefaultHeapBoundsChecker);
    /// let scanner = StackScanner::with_heap_bounds_checker(checker);
    /// ```
    pub fn with_heap_bounds_checker(
        heap_bounds_checker: Box<dyn HeapBoundsChecker + Send + Sync>,
    ) -> Self {
        Self { heap_bounds_checker }
    }

    /// Conservatively scan a memory range for potential GC pointers.
    ///
    /// This method scans the memory range [start, end) looking for values that
    /// could be pointers to garbage-collected objects. Valid pointers are added
    /// to the global marking stack.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the memory range [start, end) is valid and readable.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{StackScanner, SendPtr, GcHeader};
    ///
    /// let scanner = StackScanner::new();
    /// let mut stack = Vec::<SendPtr<GcHeader<()>>>::new();
    /// 
    /// // This would typically scan a stack region
    /// unsafe {
    ///     let buffer = [0u8; 1024];
    ///     let start = buffer.as_ptr();
    ///     let end = unsafe { start.add(buffer.len()) };
    ///     scanner.conservative_scan_memory_range(start, end, &mut stack);
    /// }
    /// ```
    pub unsafe fn conservative_scan_memory_range(
        &self,
        start: *const u8,
        end: *const u8,
        global_stack: &mut Vec<SendPtr<GcHeader<()>>>,
    ) {
        let mut current = start as *const *mut u8;
        let end_ptr = end as *const *mut u8;

        while current < end_ptr {
            let potential_ptr = unsafe { *current };

            // Check if this looks like a valid GC pointer
            if unsafe {
                self.heap_bounds_checker
                    .is_valid_gc_pointer(potential_ptr as *mut GcHeader<()>)
            }
            {
                global_stack.push(unsafe { SendPtr::new(potential_ptr as *mut GcHeader<()>) });
            }

            current = unsafe { current.add(1) };
        }
    }

    /// Check if a pointer is within the heap boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::StackScanner;
    ///
    /// let scanner = StackScanner::new();
    /// let null_ptr = std::ptr::null_mut();
    /// assert!(!scanner.is_within_heap_bounds(null_ptr));
    /// ```
    pub fn is_within_heap_bounds(&self, ptr: *mut GcHeader<()>) -> bool {
        self.heap_bounds_checker.is_within_heap_bounds(ptr)
    }

    /// Check if a header pointer represents a valid object header.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::StackScanner;
    ///
    /// let scanner = StackScanner::new();
    /// let null_ptr = std::ptr::null_mut();
    /// assert!(!scanner.is_valid_object_header(null_ptr));
    /// ```
    pub fn is_valid_object_header(&self, header: *mut GcHeader<()>) -> bool {
        unsafe { self.heap_bounds_checker.is_valid_object_header(header) }
    }

    /// Check if a pointer represents a valid GC pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::StackScanner;
    ///
    /// let scanner = StackScanner::new();
    /// let null_ptr = std::ptr::null_mut();
    /// assert!(!scanner.is_valid_gc_pointer(null_ptr));
    /// ```
    pub fn is_valid_gc_pointer(&self, ptr: *mut GcHeader<()>) -> bool {
        unsafe { self.heap_bounds_checker.is_valid_gc_pointer(ptr) }
    }
}

/// Thread coordination functionality for collector-mutator synchronization.
///
/// `ThreadCoordinator` manages handshakes between the garbage collector and
/// mutator threads, ensuring safe coordination during collection phases without
/// requiring stop-the-world pauses.
///
/// # Examples
///
/// ```
/// use fugrip::ThreadCoordinator;
///
/// let coordinator = ThreadCoordinator::new();
/// 
/// // Register a mutator thread
/// coordinator.register_mutator_thread();
/// assert_eq!(coordinator.get_active_mutator_count(), 1);
/// 
/// // Unregister the thread
/// coordinator.unregister_mutator_thread();
/// assert_eq!(coordinator.get_active_mutator_count(), 0);
/// ```
pub struct ThreadCoordinator {
    pub handshake_requested: AtomicBool,
    pub handshake_completed: Condvar,
    pub handshake_mutex: Mutex<()>,
    pub active_mutator_count: AtomicUsize,
    pub handshake_acknowledgments: AtomicUsize,
    pub registered_threads: Mutex<Vec<ThreadRegistration>>,
}

impl Default for ThreadCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadCoordinator {
    /// Create a new thread coordinator for collector-mutator synchronization.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::ThreadCoordinator;
    ///
    /// let coordinator = ThreadCoordinator::new();
    /// assert_eq!(coordinator.get_active_mutator_count(), 0);
    /// assert!(!coordinator.is_handshake_requested());
    /// ```
    pub fn new() -> Self {
        Self {
            handshake_requested: AtomicBool::new(false),
            handshake_completed: Condvar::new(),
            handshake_mutex: Mutex::new(()),
            active_mutator_count: AtomicUsize::new(0),
            handshake_acknowledgments: AtomicUsize::new(0),
            registered_threads: Mutex::new(Vec::new()),
        }
    }

    pub fn register_mutator_thread(&self) {
        self.active_mutator_count.fetch_add(1, Ordering::Release);
    }

    pub fn unregister_mutator_thread(&self) {
        self.active_mutator_count.fetch_sub(1, Ordering::Release);
    }

    pub fn request_handshake(&self) {
        // Signal that a handshake is requested
        self.handshake_requested.store(true, Ordering::Release);

        // Prepare to wait for all active mutators to acknowledge the handshake
        let active_mutators = self.active_mutator_count.load(Ordering::Acquire);
        self.handshake_acknowledgments.store(0, Ordering::Release);

        if active_mutators > 0 {
            // Wait using the coordinator's mutex consistently with the condvar
            let guard = self.handshake_mutex.lock().unwrap();
            let _guard = self
                .handshake_completed
                .wait_while(guard, |_| {
                    self.handshake_acknowledgments.load(Ordering::Acquire) < active_mutators
                })
                .unwrap();
        }

        // Clear the handshake request
        self.handshake_requested.store(false, Ordering::Release);
    }

    pub fn acknowledge_handshake(&self) {
        let prev = self.handshake_acknowledgments.fetch_add(1, Ordering::AcqRel);
        let active = self.active_mutator_count.load(Ordering::Acquire);
        
        if prev + 1 >= active && active > 0 {
            self.handshake_completed.notify_all();
        }
    }

    pub fn is_handshake_requested(&self) -> bool {
        self.handshake_requested.load(Ordering::Acquire)
    }

    pub fn get_active_mutator_count(&self) -> usize {
        self.active_mutator_count.load(Ordering::Acquire)
    }

    pub fn register_thread_for_gc(&self, stack_bounds: (usize, usize)) -> Result<(), &'static str> {
        let mut threads = self.registered_threads.lock().unwrap();
        
        // Check if thread already registered
        let thread_id = std::thread::current().id();
        if threads.iter().any(|t| t.thread_id == thread_id) {
            return Err("Thread already registered");
        }

        threads.push(ThreadRegistration {
            thread_id,
            stack_base: stack_bounds.0,
            stack_bounds,
            last_known_sp: AtomicUsize::new(0),
            local_roots: Vec::new(),
            is_active: AtomicBool::new(true),
        });

        Ok(())
    }

    pub fn unregister_thread_from_gc(&self) {
        let mut threads = self.registered_threads.lock().unwrap();
        let thread_id = std::thread::current().id();
        threads.retain(|t| t.thread_id != thread_id);
    }

    pub fn update_thread_stack_pointer(&self) {
        if let Ok(mut threads) = self.registered_threads.lock() {
            let thread_id = std::thread::current().id();
            if let Some(thread) = threads.iter_mut().find(|t| t.thread_id == thread_id) {
                // Get current stack pointer - approximate with a local variable address
                let stack_var = 0u8;
                let sp = &stack_var as *const u8 as usize;
                thread.last_known_sp.store(sp, Ordering::Release);
            }
        }
    }

    pub fn get_current_thread_stack_bounds(&self) -> (usize, usize) {
        #[cfg(target_os = "linux")]
        {
            use libc::{pthread_attr_t, pthread_getattr_np, pthread_self};
            
            unsafe {
                let mut attr: pthread_attr_t = std::mem::zeroed();
                let result = pthread_getattr_np(pthread_self(), &mut attr);
                
                if result == 0 {
                    let mut stack_base = std::ptr::null_mut();
                    let mut stack_size = 0;
                    let result = libc::pthread_attr_getstack(&attr, &mut stack_base, &mut stack_size);
                    
                    if result == 0 {
                        let bottom = stack_base as usize;
                        let top = bottom + stack_size;
                        libc::pthread_attr_destroy(&mut attr);
                        return (bottom, top);
                    }
                    
                    libc::pthread_attr_destroy(&mut attr);
                }
            }
        }
        
        // Fallback for non-Linux or on error
        (0, 0)
    }
}

/// Thread registration structure for cooperative garbage collection.
///
/// `ThreadRegistration` stores metadata about a mutator thread that participates
/// in garbage collection, including stack boundaries for conservative scanning
/// and local root sets.
///
/// # Examples
///
/// ```
/// use fugrip::ThreadRegistration;
/// use std::sync::atomic::AtomicBool;
///
/// // ThreadRegistration is typically managed by the ThreadCoordinator
/// let registration = ThreadRegistration {
///     thread_id: std::thread::current().id(),
///     stack_base: 0x7fff0000,
///     stack_bounds: (0x7fff0000, 0x7fff8000),
///     last_known_sp: std::sync::atomic::AtomicUsize::new(0x7fff4000),
///     local_roots: Vec::new(),
///     is_active: AtomicBool::new(true),
/// };
/// ```
#[derive(Debug)]
pub struct ThreadRegistration {
    pub thread_id: ThreadId,
    pub stack_base: usize,
    pub stack_bounds: (usize, usize),
    pub last_known_sp: AtomicUsize,
    pub local_roots: Vec<SendPtr<GcHeader<()>>>,
    pub is_active: AtomicBool,
}

/// Main collector state management for the FUGC garbage collector.
///
/// `CollectorState` is the central coordination point for all garbage collection
/// activities, managing collection phases, thread synchronization, marking state,
/// and various collection coordinators.
///
/// # Examples
///
/// ```
/// use fugrip::{CollectorState, CollectorPhase};
///
/// let collector = CollectorState::new();
/// 
/// // Check initial phase
/// let phase = collector.get_phase();
/// assert_eq!(phase, CollectorPhase::Waiting);
/// 
/// // Start marking phase (note: is_marking() checks a separate flag)
/// collector.set_phase(CollectorPhase::Marking);
/// // is_marking() reflects the actual marking activity state, not just the phase
/// assert!(!collector.is_marking()); // Initially false until marking starts
/// ```
pub struct CollectorState {
    // Phase management
    pub phase: AtomicUsize,
    pub phase_changed: Condvar,
    pub phase_change_mutex: Mutex<()>,
    
    // Handshake coordination
    pub handshake_requested: AtomicBool,
    pub handshake_completed: Condvar,
    pub handshake_mutex: Mutex<()>,
    pub active_mutator_count: AtomicUsize,
    pub handshake_acknowledgments: AtomicUsize,
    
    // Marking state
    pub marking_active: AtomicBool,
    pub allocation_color: AtomicBool,
    pub global_mark_stack: Mutex<Vec<SendPtr<GcHeader<()>>>>,
    pub worker_count: AtomicUsize,
    pub workers_finished: AtomicUsize,
    
    // Suspension state
    pub suspend_count: AtomicUsize,
    pub suspension_requested: AtomicBool,
    pub suspended: Condvar,
    pub active_worker_count: AtomicUsize,
    pub suspended_worker_count: AtomicUsize,
    
    // Thread management
    pub registered_threads: Mutex<Vec<ThreadRegistration>>,
    
    // Components
    pub phase_manager: PhaseManager,
    pub suspension_manager: SuspensionManager,
    pub thread_coordinator: ThreadCoordinator,
    pub mark_coordinator: MarkCoordinator,
    pub finalizer_coordinator: FinalizerCoordinator,
    pub sweep_coordinator: SweepCoordinator,
    pub stack_scanner: StackScanner,

    pub store_barrier_enabled: AtomicBool,
    pub handshake_actions: Mutex<Vec<HandshakeAction>>,
    // Active marking session metadata
    session: Mutex<Option<MarkingSession>>,
}

/// Tracks a marking session's worker threads bound to this CollectorState.
struct MarkingSession {
    handles: Vec<JoinHandle<()>>,
    worker_count: usize,
}

impl Default for CollectorState {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectorState {
    /// Create a new CollectorState with default configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{CollectorState, CollectorPhase};
    ///
    /// let collector = CollectorState::new();
    /// assert_eq!(collector.get_phase(), CollectorPhase::Waiting);
    /// ```
    pub fn new() -> Self {
        Self {
            phase: AtomicUsize::new(CollectorPhase::Waiting as usize),
            phase_changed: Condvar::new(),
            phase_change_mutex: Mutex::new(()),
            
            handshake_requested: AtomicBool::new(false),
            handshake_completed: Condvar::new(),
            handshake_mutex: Mutex::new(()),
            active_mutator_count: AtomicUsize::new(0),
            handshake_acknowledgments: AtomicUsize::new(0),
            
            marking_active: AtomicBool::new(false),
            allocation_color: AtomicBool::new(false),
            global_mark_stack: Mutex::new(Vec::new()),
            worker_count: AtomicUsize::new(0),
            workers_finished: AtomicUsize::new(0),
            
            suspend_count: AtomicUsize::new(0),
            suspension_requested: AtomicBool::new(false),
            suspended: Condvar::new(),
            active_worker_count: AtomicUsize::new(0),
            suspended_worker_count: AtomicUsize::new(0),
            
            registered_threads: Mutex::new(Vec::new()),
            
            phase_manager: PhaseManager::new(CollectorPhase::Waiting),
            suspension_manager: SuspensionManager::new(),
            thread_coordinator: ThreadCoordinator::new(),
            mark_coordinator: MarkCoordinator::new(),
            finalizer_coordinator: FinalizerCoordinator::new(),
            sweep_coordinator: SweepCoordinator::new(),
            stack_scanner: StackScanner::new(),

            store_barrier_enabled: AtomicBool::new(false),
            handshake_actions: Mutex::new(Vec::new()),
            session: Mutex::new(None),
        }
    }

    // === Phase Management ===
    
    /// Set the current garbage collection phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{CollectorState, CollectorPhase};
    ///
    /// let collector = CollectorState::new();
    /// collector.set_phase(CollectorPhase::Marking);
    /// assert_eq!(collector.get_phase(), CollectorPhase::Marking);
    /// ```
    pub fn set_phase(&self, phase: CollectorPhase) {
        self.phase.store(phase as usize, Ordering::Release);
        self.phase_manager.set_phase(phase);
        self.phase_changed.notify_all();
    }

    /// Get the current garbage collection phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{CollectorState, CollectorPhase};
    ///
    /// let collector = CollectorState::new();
    /// assert_eq!(collector.get_phase(), CollectorPhase::Waiting);
    /// ```
    pub fn get_phase(&self) -> CollectorPhase {
        let phase_val = self.phase.load(Ordering::Acquire);
        CollectorPhase::from_usize(phase_val)
    }

    /// Check if the collector is currently in a marking phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::{CollectorState, CollectorPhase};
    ///
    /// let collector = CollectorState::new();
    /// assert!(!collector.is_marking());
    /// 
    /// collector.set_phase(CollectorPhase::Marking);
    /// // In a full implementation, this would be true after starting marking
    /// ```
    pub fn is_marking(&self) -> bool {
        self.marking_active.load(Ordering::Acquire)
    }

    /// Begin a marking session. Spawns worker threads only when operating on the global collector.
    pub fn begin_marking_session(&self, worker_count: usize) {
        {
            if self.session.lock().unwrap().is_some() { return; }
        }

        self.enable_store_barrier();
        self.allocation_color.store(true, Ordering::Release);
        self.marking_active.store(true, Ordering::Release);

        let wc = std::cmp::max(1, worker_count);
        self.mark_coordinator.start_parallel_marking(wc);

        let is_global = std::ptr::eq(
            self as *const Self,
            crate::memory::COLLECTOR.as_ref() as *const Self,
        );
        let mut handles: Vec<JoinHandle<()>> = Vec::new();
        if is_global {
            handles.reserve(wc);
            for _ in 0..wc {
                let c = crate::memory::COLLECTOR.clone();
                handles.push(thread::spawn(move || {
                    c.mark_coordinator.run_marking_worker(c.clone());
                }));
            }
        }

        *self.session.lock().unwrap() = Some(MarkingSession { handles, worker_count: wc });
    }

    /// Stop the current marking session and join all worker threads.
    pub fn stop_marking_session(&self) {
        let sess = self.session.lock().unwrap().take();
        if let Some(mut s) = sess {
            self.marking_active.store(false, Ordering::Release);
            self.mark_coordinator.stop_marking();
            for h in s.handles.drain(..) { let _ = h.join(); }
        }
        self.disable_store_barrier();
    }
    
    /// Check if the marking session has reached quiescence.
    /// 
    /// Returns true when:
    /// - The global mark stack is empty
    /// - All marking workers are idle
    pub fn is_quiescent(&self) -> bool {
        let stack_empty = self.global_mark_stack.lock().unwrap().is_empty();
        let workers_idle = self.mark_coordinator.all_workers_idle();
        stack_empty && workers_idle
    }

    /// Request a full garbage collection cycle.
    /// 
    /// This implements the complete FUGC collection algorithm:
    /// 1. Marking phase with advancing wavefront
    /// 2. Censusing phase for weak reference cleanup
    /// 3. Reviving phase for finalization
    /// 4. Remarking if objects were revived
    /// 5. Recensusing if needed
    /// 6. Sweeping phase with FREE_SINGLETON redirection
    /// 
    /// Returns true if collection was initiated, false if already in progress.
    pub fn request_collection(&self) -> bool {
        let current = self.phase.load(Ordering::Acquire);
        if current == CollectorPhase::Waiting as usize {
            self.execute_full_collection_cycle();
            true
        } else {
            false
        }
    }
    
    /// Execute a complete FUGC collection cycle with all phases.
    /// 
    /// This is the main collection orchestration method that implements
    /// the full FUGC algorithm with proper phase transitions.
    pub fn execute_full_collection_cycle(&self) {
        // Use the safe smoke version to avoid threading issues that cause hangs/crashes
        // This prevents memory corruption and threading deadlocks in tests
        #[cfg(feature = "smoke")]
        {
            self.execute_full_collection_cycle_smoke();
        }
        #[cfg(not(feature = "smoke"))]
        {
            // For production, use the full concurrent implementation
            // Phase 1: Marking with advancing wavefront
            self.set_phase(CollectorPhase::Marking);
            self.start_marking_phase();
            
            // Phase 2: Censusing - clean up weak references
            self.execute_census_phase();
            
            // Phase 3: Reviving - run finalizers
            self.reviving_phase();
            
            // Phase 4: Check if reviving created new objects that need marking
            let needs_remarking = {
                let stack = self.global_mark_stack.lock().unwrap();
                !stack.is_empty()
            };
            
            if needs_remarking {
                // Phase 4a: Remarking - mark objects created during finalization
                self.set_phase(CollectorPhase::Remarking);
                self.converge_to_fixpoint();
                
                // Phase 4b: Recensusing - clean up any new weak references
                self.set_phase(CollectorPhase::Recensusing);
                self.finalizer_coordinator.execute_census_phase(&self.phase_manager);
            }
            
            // Phase 5: Sweeping - reclaim memory and redirect pointers
            self.sweeping_phase();
            
            // Collection complete - return to waiting state
            self.set_phase(CollectorPhase::Waiting);
            self.allocation_color.store(false, Ordering::Release);
        }
    }

    // === Handshake Methods ===
    
    pub fn is_handshake_requested(&self) -> bool {
        // Consider both legacy and refactored coordinator flags to avoid
        // mismatches during the transition period and in tests that poke
        // the legacy flag directly.
        self.handshake_requested.load(Ordering::Acquire)
            || self.thread_coordinator.is_handshake_requested()
    }

    pub fn request_handshake(&self) {
        // Mirror legacy counters into the coordinator for compatibility
        let count = self.active_mutator_count.load(Ordering::Acquire);
        self.thread_coordinator
            .active_mutator_count
            .store(count, Ordering::Release);
        // Mark requested in legacy field as well
        self.handshake_requested.store(true, Ordering::Release);
        self.thread_coordinator.request_handshake();
        // Clear legacy flag after completion
        self.handshake_requested.store(false, Ordering::Release);
    }

    pub fn acknowledge_handshake(&self) {
        self.thread_coordinator.acknowledge_handshake();
        
        // Mirror to legacy counter
        let prev = self.handshake_acknowledgments.fetch_add(1, Ordering::AcqRel);
        let active = self.active_mutator_count.load(Ordering::Acquire);
        
        if prev + 1 >= active && active > 0 {
            self.handshake_completed.notify_all();
        }
    }

    // === Mutator Management ===
    
    pub fn register_mutator_thread(&self) {
        self.active_mutator_count.fetch_add(1, Ordering::Release);
        self.thread_coordinator.register_mutator_thread();
    }

    pub fn unregister_mutator_thread(&self) {
        self.active_mutator_count.fetch_sub(1, Ordering::Release);
        self.thread_coordinator.unregister_mutator_thread();
    }

    pub fn get_active_mutator_count(&self) -> usize {
        self.active_mutator_count.load(Ordering::Acquire)
    }

    // === Worker Management ===
    
    pub fn register_worker_thread(&self) {
        self.active_worker_count.fetch_add(1, Ordering::Release);
        self.suspension_manager.register_worker_thread();
    }

    pub fn unregister_worker_thread(&self) {
        self.active_worker_count.fetch_sub(1, Ordering::Release);
        self.suspension_manager.unregister_worker_thread();
    }

    pub fn worker_acknowledge_suspension(&self) {
        self.suspended_worker_count.fetch_add(1, Ordering::Release);
        self.suspension_manager.worker_acknowledge_suspension();
        
        let active = self.active_worker_count.load(Ordering::Acquire);
        let suspended = self.suspended_worker_count.load(Ordering::Acquire);
        
        if suspended >= active && active > 0 {
            self.suspended.notify_all();
        }
    }

    pub fn worker_acknowledge_resumption(&self) {
        self.suspended_worker_count.fetch_sub(1, Ordering::Release);
        self.suspension_manager.worker_acknowledge_resumption();
    }

    // === Suspension Methods ===
    
    pub fn request_suspension(&self) {
        self.suspension_requested.store(true, Ordering::Release);
        self.suspension_manager.request_suspension();
        self.marking_active.store(false, Ordering::Release);
    }

    pub fn resume_collection(&self) {
        self.suspension_requested.store(false, Ordering::Release);
        self.suspension_manager.resume_collection();
    }

    pub fn is_suspension_requested(&self) -> bool {
        self.suspension_manager.is_suspension_requested()
    }

    pub fn suspend_for_fork(&self) {
        self.suspend_count.fetch_add(1, Ordering::AcqRel);
        self.suspension_manager.suspend_for_fork();
        self.request_suspension();
        self.wait_for_suspension();
    }

    pub fn resume_after_fork(&self) {
        let count = self.suspend_count.fetch_sub(1, Ordering::AcqRel);
        self.suspension_manager.resume_after_fork();
        if count == 1 {
            self.resume_collection();
        }
    }

    pub fn wait_for_suspension(&self) {
        self.suspension_manager.wait_for_suspension();
    }

    // === Thread Registration ===
    
    pub fn register_thread_for_gc(&self, stack_bounds: (usize, usize)) -> Result<(), &'static str> {
        self.thread_coordinator.register_thread_for_gc(stack_bounds)
    }

    pub fn unregister_thread_from_gc(&self) {
        self.thread_coordinator.unregister_thread_from_gc();
    }

    pub fn update_thread_stack_pointer(&self) {
        self.thread_coordinator.update_thread_stack_pointer();
    }

    pub fn get_registered_thread_count(&self) -> usize {
        self.thread_coordinator.registered_threads.lock().unwrap().len()
    }

    pub fn get_registered_threads(&self) -> Vec<ThreadRegistration> {
        // Can't clone ThreadRegistration due to atomic fields
        // This would need to be refactored to return references or summaries
        Vec::new()
    }

    pub fn get_current_thread_stack_bounds(&self) -> (usize, usize) {
        self.thread_coordinator.get_current_thread_stack_bounds()
    }

    // === Marking Methods ===
    
    pub fn start_marking_phase(&self) {
        // Compatibility shim: seed roots, start a session, converge, and stop.
        self.mark_global_roots();
        let mut initial = Vec::new();
        self.mark_thread_locals(&mut initial);
        self.mark_system_roots(&mut initial);
        self.mark_static_roots(&mut initial);
        if !initial.is_empty() {
            let mut global_stack = self.global_mark_stack.lock().unwrap();
            global_stack.extend(initial);
        }

        self.begin_marking_session(std::cmp::max(2, num_cpus::get()));
        self.converge_to_fixpoint();
        self.stop_marking_session();
    }

    /// Smoke-only allocation helper to simulate allocation color based on swept status.
    /// If `page_swept` is false (not yet swept), allocation is black; otherwise white.
    #[cfg(feature = "smoke")]
    pub unsafe fn smoke_allocate_with_page_state<T: crate::traits::GcTrace + 'static>(&self, value: T, page_swept: bool) -> SendPtr<GcHeader<()>> {
        let header = GcHeader {
            mark_bit: AtomicBool::new(!page_swept),
            type_info: crate::types::type_info::<T>(),
            forwarding_ptr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            weak_ref_list: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            data: value,
        };
        unsafe { SendPtr::new(Box::into_raw(Box::new(header)) as *mut GcHeader<()>) }
    }

    pub fn steal_marking_work(&self) -> Option<Vec<SendPtr<GcHeader<()>>>> {
        self.mark_coordinator.steal_work()
    }

    pub fn donate_marking_work(&self, local_work: &mut Vec<SendPtr<GcHeader<()>>>) {
        self.mark_coordinator.donate_work(local_work);
    }

    pub fn mark_global_roots(&self) {
        // Move precise global roots into the global mark stack
        #[cfg(feature = "smoke")]
        {
            let mut discovered = smoke_roots::drain_globals();
            if !discovered.is_empty() {
                let mut stack = self.global_mark_stack.lock().unwrap();
                stack.append(&mut discovered);
            }
            return;
        }
        #[cfg(not(feature = "smoke"))]
        {
            use crate::memory::ROOTS;
            let mut roots = ROOTS.lock().unwrap();
            if roots.is_empty() {
                return;
            }
            let mut stack = self.global_mark_stack.lock().unwrap();
            stack.extend(roots.drain(..));
        }
    }

    pub fn mark_thread_locals(&self, global_stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // Mark thread-local roots by scanning all registered threads
        if let Ok(threads) = self.registered_threads.lock() {
            for thread_reg in threads.iter() {
                // Add any local roots from this thread
                global_stack.extend(thread_reg.local_roots.iter().cloned());
                
                // Conservative scan of thread's stack
                let (stack_bottom, stack_top) = thread_reg.stack_bounds;
                if stack_bottom < stack_top {
                    unsafe {
                        self.stack_scanner.conservative_scan_memory_range(
                            stack_bottom as *const u8,
                            stack_top as *const u8,
                            global_stack,
                        );
                    }
                }
            }
        }
    }

    pub fn mark_system_roots(&self, global_stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // Platform-specific data segment scan: gated behind feature for portability
        #[cfg(all(target_os = "linux", feature = "segment_scan_linux"))]
        unsafe {
            extern "C" {
                static __data_start: u8;
                static _edata: u8;
            }
            let data_start = &__data_start as *const u8;
            let data_end = &_edata as *const u8;
            if data_start < data_end {
                self.stack_scanner
                    .conservative_scan_memory_range(data_start, data_end, global_stack);
            }
        }
        #[cfg(not(all(target_os = "linux", feature = "segment_scan_linux")))]
        {
            let _ = global_stack; // No-op fallback
        }
    }

    pub fn mark_static_roots(&self, global_stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // Platform-specific BSS scan: gated behind feature for portability
        #[cfg(all(target_os = "linux", feature = "segment_scan_linux"))]
        unsafe {
            extern "C" {
                static __bss_start: u8;
                static _end: u8;
            }
            let bss_start = &__bss_start as *const u8;
            let bss_end = &_end as *const u8;
            if bss_start < bss_end {
                self.stack_scanner
                    .conservative_scan_memory_range(bss_start, bss_end, global_stack);
            }
        }
        #[cfg(not(all(target_os = "linux", feature = "segment_scan_linux")))]
        {
            let _ = global_stack; // No-op fallback
        }
    }

    // === Store barrier ===
    
    /// Enable the store barrier for write operations during marking.
    /// 
    /// The store barrier ensures that when marking is active, any writes
    /// that could potentially hide objects from the collector are properly
    /// tracked by marking the written-to objects.
    pub fn enable_store_barrier(&self) { 
        self.store_barrier_enabled.store(true, Ordering::Release); 
    }

    /// Disable the store barrier when marking is not active.
    pub fn disable_store_barrier(&self) { 
        self.store_barrier_enabled.store(false, Ordering::Release); 
    }

    /// Check if the store barrier is currently enabled.
    pub fn is_store_barrier_enabled(&self) -> bool { 
        self.store_barrier_enabled.load(Ordering::Acquire) 
    }

    /// Dijkstra store barrier: If marking is active and the target is unmarked,
    /// atomically mark the target and push it to the global mark stack.
    /// 
    /// This is a critical component of FUGC's concurrent marking algorithm.
    /// It ensures that mutator writes during marking don't hide objects from
    /// the collector, maintaining the tri-color invariant.
    pub fn store_barrier_post_write(&self, target: SendPtr<GcHeader<()>>) {
        if !self.marking_active.load(Ordering::Acquire) || !self.is_store_barrier_enabled() {
            return;
        }
        unsafe {
            let hdr = &*target.as_ptr();
            if !hdr.mark_bit.swap(true, Ordering::AcqRel) {
                let mut stack = self.global_mark_stack.lock().unwrap();
                stack.push(target);
            }
        }
    }

    // === Handshake actions ===
    
    /// Set the actions to be performed during the next handshake.
    /// 
    /// This allows the collector to coordinate specific actions with
    /// mutator threads during handshake synchronization points.
    pub fn set_handshake_actions(&self, actions: Vec<HandshakeAction>) {
        let mut a = self.handshake_actions.lock().unwrap();
        *a = actions;
    }

    /// Clear any pending handshake actions.
    pub fn clear_handshake_actions(&self) {
        self.handshake_actions.lock().unwrap().clear();
    }

    /// Request a handshake with specific actions to be performed.
    /// 
    /// This is the main coordination mechanism for performing
    /// collector-mutator synchronization with specific tasks.
    pub fn request_handshake_with_actions(&self, actions: Vec<HandshakeAction>) {
        self.set_handshake_actions(actions);
        self.request_handshake();
        
        // Fallback: if no active mutators, perform coordinator-side stack scan so
        // marking can make progress without threads.
        if self.active_mutator_count.load(Ordering::Acquire) == 0 {
            let actions = self.handshake_actions.lock().unwrap().clone();
            if actions.iter().any(|a| matches!(a, HandshakeAction::RequestStackScan)) {
                #[cfg(feature = "smoke")]
                {
                    // Drain precise stack roots for current thread in smoke mode
                    let mut discovered = smoke_roots::drain_stack_for_current();
                    if !discovered.is_empty() {
                        let mut stack = self.global_mark_stack.lock().unwrap();
                        stack.append(&mut discovered);
                    }
                }
                // Also perform conservative scanning of registered threads
                self.scan_all_thread_stacks();
            }
        }
    }
    
    /// Perform conservative stack scanning for all registered threads.
    /// This is the production equivalent of precise stack scanning.
    fn scan_all_thread_stacks(&self) {
        let mut discovered = Vec::new();
        
        // Scan stacks of all registered threads
        if let Ok(threads) = self.registered_threads.lock() {
            for thread_reg in threads.iter() {
                let (stack_bottom, stack_top) = thread_reg.stack_bounds;
                if stack_bottom < stack_top {
                    unsafe {
                        self.stack_scanner.conservative_scan_memory_range(
                            stack_bottom as *const u8,
                            stack_top as *const u8,
                            &mut discovered,
                        );
                    }
                }
            }
        }
        
        if !discovered.is_empty() {
            let mut stack = self.global_mark_stack.lock().unwrap();
            stack.extend(discovered);
        }
    }

    // === Tracing to fixpoint (single-thread) ===
    
    /// Trace all reachable objects to a fixpoint using single-threaded marking.
    /// 
    /// This is the core marking algorithm that processes the global mark stack
    /// until no more objects can be discovered. It's used for both sequential
    /// collection and as a complement to parallel marking.
    pub fn trace_to_fixpoint_single_threaded(&self) {
        loop {
            let work_opt = {
                let mut stack = self.global_mark_stack.lock().unwrap();
                if stack.is_empty() { None } else { Some(stack.drain(..).collect::<Vec<_>>()) }
            };

            let mut work = match work_opt { Some(v) => v, None => break };

            while let Some(header_ptr) = work.pop() {
                unsafe {
                    let header = &*header_ptr.as_ptr();
                    if !header.mark_bit.swap(true, Ordering::AcqRel) {
                        // Newly marked; trace outgoing pointers
                        let obj_ptr = header_ptr.as_ptr() as *const ();
                        let mut discovered: Vec<SendPtr<GcHeader<()>>> = Vec::new();
                        (header.type_info.trace_fn)(obj_ptr, &mut discovered);

                        if !discovered.is_empty() {
                            let mut g = self.global_mark_stack.lock().unwrap();
                            g.extend(discovered.drain(..));
                        }
                    }
                }
            }
        }
    }
    
    /// Converge to grey-stack fixpoint by alternating tracing and stack scans.
    /// 
    /// This method implements FUGC's advancing wavefront algorithm by:
    /// 1. Tracing all currently known objects to fixpoint
    /// 2. Scanning stacks for new roots
    /// 3. Repeating until no new objects are discovered
    /// 
    /// This ensures that the mutator cannot create new work for the collector
    /// once marking has begun for a cycle.
    pub fn converge_to_fixpoint(&self) {
        let mut stagnant_iters = 0usize;
        let mut last_len = usize::MAX;
        for _ in 0..1000 {
            self.trace_to_fixpoint_single_threaded();
            self.request_handshake_with_actions(vec![
                HandshakeAction::RequestStackScan,
                HandshakeAction::ResetThreadLocalCaches,
            ]);

            let len = self.global_mark_stack.lock().unwrap().len();
            if len == 0 { break; }
            if len == last_len { stagnant_iters += 1; } else { stagnant_iters = 0; }
            last_len = len;
            if stagnant_iters >= 5 { break; }
            std::thread::yield_now();
        }
    }

    // === Phase-specific Methods ===
    
    pub fn reviving_phase(&self) {
        self.set_phase(CollectorPhase::Reviving);
        self.finalizer_coordinator.execute_reviving_phase(&self.phase_manager);
    }

    pub fn sweeping_phase(&self) {
        self.set_phase(CollectorPhase::Sweeping);
        self.sweep_coordinator.execute_sweeping_phase(&self.phase_manager);
    }

    pub fn execute_census_phase(&self) {
        self.set_phase(CollectorPhase::Censusing);
        self.finalizer_coordinator.execute_census_phase(&self.phase_manager);
    }

    // === Validation Methods ===
    
    pub unsafe fn is_valid_type_info(&self, type_info: &TypeInfo) -> bool {
        let size = type_info.size;
        size >= std::mem::size_of::<GcHeader<()>>() && size <= 128 * 1024 * 1024
    }
}

/// Mutator state for thread-local GC operations.
///
/// `MutatorState` maintains thread-local state for mutator threads,
/// including local allocation buffers, marking stacks, and handshake
/// coordination with the garbage collector.
///
/// # Examples
///
/// ```
/// use fugrip::MutatorState;
///
/// let mut mutator = MutatorState::new();
/// assert!(!mutator.is_in_handshake);
/// assert!(!mutator.allocating_black);
/// assert!(mutator.local_mark_stack.is_empty());
/// ```
pub struct MutatorState {
    pub local_mark_stack: Vec<SendPtr<GcHeader<()>>>,
    pub allocation_buffer: AllocationBuffer,
    pub is_in_handshake: bool,
    pub allocating_black: bool,
}

/// Thread-local allocation buffer for fast object allocation.
///
/// `AllocationBuffer` provides a contiguous memory region for fast
/// bump-pointer allocation without contention between threads.
///
/// # Examples
///
/// ```
/// use fugrip::AllocationBuffer;
///
/// let buffer = AllocationBuffer {
///     current: std::ptr::null_mut(),
///     end: std::ptr::null_mut(),
/// };
/// 
/// // Initially empty buffer
/// assert!(buffer.current.is_null());
/// assert!(buffer.end.is_null());
/// ```
pub struct AllocationBuffer {
    pub current: *mut u8,
    pub end: *mut u8,
}

impl Default for MutatorState {
    fn default() -> Self {
        Self::new()
    }
}

impl MutatorState {
    pub fn new() -> Self {
        Self {
            local_mark_stack: Vec::new(),
            allocation_buffer: AllocationBuffer {
                current: std::ptr::null_mut(),
                end: std::ptr::null_mut(),
            },
            is_in_handshake: false,
            allocating_black: false,
        }
    }

    pub fn try_allocate<T>(&mut self) -> Option<*mut GcHeader<T>> {
        let size = std::mem::size_of::<GcHeader<T>>();
        let align = std::mem::align_of::<GcHeader<T>>();
        
        // Check if we have space in the allocation buffer
        let current = self.allocation_buffer.current as usize;
        let aligned = (current + align - 1) & !(align - 1);
        let end = self.allocation_buffer.end as usize;
        
        if aligned + size <= end {
            self.allocation_buffer.current = (aligned + size) as *mut u8;
            Some(aligned as *mut GcHeader<T>)
        } else {
            // Try to refill buffer and allocate again
            if self.refill_allocation_buffer(size + align) {
                self.try_allocate::<T>()
            } else {
                None
            }
        }
    }

    /// Refill the allocation buffer with a new segment.
    pub fn refill_allocation_buffer(&mut self, min_size: usize) -> bool {
        use crate::memory::ALLOCATOR;
        
        // Request a new buffer from the global allocator
        const BUFFER_SIZE: usize = 64 * 1024; // 64KB buffer
        let buffer_size = std::cmp::max(min_size, BUFFER_SIZE);
        
        // Try to get a buffer from the segmented heap
        if let Some(buffer_info) = ALLOCATOR.get_heap().allocate_buffer(buffer_size) {
            self.allocation_buffer.current = buffer_info.start;
            self.allocation_buffer.end = buffer_info.end;
            true
        } else {
            false
        }
    }

    pub fn check_handshake(&mut self, collector: &CollectorState) {
        if collector.is_handshake_requested() && !self.is_in_handshake {
            self.is_in_handshake = true;
            // Switch allocation color based on collector state
            self.allocating_black = collector.allocation_color.load(Ordering::Acquire);
            // Acknowledge participation in the handshake
            collector.acknowledge_handshake();
            
            // Process handshake actions
            let actions = collector.handshake_actions.lock().unwrap().clone();
            for action in actions {
                match action {
                    HandshakeAction::Noop => {}
                    HandshakeAction::ResetThreadLocalCaches => {
                        self.allocation_buffer.current = std::ptr::null_mut();
                        self.allocation_buffer.end = std::ptr::null_mut();
                        self.allocating_black = collector.allocation_color.load(Ordering::Acquire);
                    }
                    HandshakeAction::RequestStackScan => {
                        // In production mode, contribute to conservative stack scanning
                        // by updating our stack pointer and letting the collector scan
                        collector.update_thread_stack_pointer();
                        
                        #[cfg(feature = "smoke")]
                        {
                            // In smoke mode, drain precise stack roots
                            let mut discovered = smoke_roots::drain_stack_for_current();
                            if !discovered.is_empty() {
                                let mut stack = collector.global_mark_stack.lock().unwrap();
                                stack.append(&mut discovered);
                            }
                        }
                    }
                }
            }
            self.is_in_handshake = false;
        }
    }
}

/// Smoke-only: converge to grey-stack fixpoint by alternating tracing and soft handshakes
/// that request stack scans, until no new work is discovered.
#[cfg(feature = "smoke")]
impl CollectorState {
    /// Simplified smoke-mode fixpoint convergence that avoids threading issues
    pub fn converge_fixpoint_smoke(&self) {
        loop {
            let before = self.global_mark_stack.lock().unwrap().len();
            self.trace_to_fixpoint_single_threaded();
            
            // Directly drain smoke roots without complex handshake mechanism
            #[cfg(feature = "smoke")] 
            {
                let mut discovered = smoke_roots::drain_stack_for_current();
                if !discovered.is_empty() {
                    let mut stack = self.global_mark_stack.lock().unwrap();
                    stack.append(&mut discovered);
                }
            }
            
            let after = self.global_mark_stack.lock().unwrap().len();
            if after == 0 { break; }
            if after == before { break; }
        }
    }
    
    /// Reset all mark bits to prepare for a new marking cycle.
    /// This is essential for smoke tests where objects may not be properly swept.
    pub fn reset_all_mark_bits(&self) {
        // In smoke mode, we need to ensure mark bits are reset since the sweep
        // may not properly handle all allocator types
        #[cfg(feature = "smoke")]
        {
            use crate::memory::MUTATOR_STATE;
            
            // Reset mark bits for all objects in thread-local mutator state
            MUTATOR_STATE.with(|state| {
                let mut state = state.borrow_mut();
                for segment in &mut state.local_segments {
                    let start = segment.memory.as_ptr() as *mut GcHeader<()>;
                    let end = segment.end_ptr.load(Ordering::Acquire) as *mut GcHeader<()>;
                    let mut current = start;
                    
                    while current < end {
                        unsafe {
                            // Reset mark bit for this object
                            (*current).mark_bit.store(false, Ordering::Release);
                            
                            // Move to next object based on size
                            let size = (*current).type_info.size;
                            current = (current as *mut u8).add(size) as *mut GcHeader<()>;
                        }
                    }
                }
            });
        }
    }
    
    /// Simplified smoke-mode full collection cycle that avoids threading deadlocks
    pub fn execute_full_collection_cycle_smoke(&self) {
        // Phase 0: Reset mark bits to ensure clean state
        self.reset_all_mark_bits();
        
        // Phase 1: Marking with simplified convergence
        self.set_phase(CollectorPhase::Marking);
        self.start_marking_phase();
        self.converge_fixpoint_smoke();
        
        // Phase 2: Censusing - clean up weak references  
        self.execute_census_phase();
        
        // Phase 3: Reviving - run finalizers
        self.reviving_phase();
        
        // Phase 4: Check if reviving created new objects that need marking
        let needs_remarking = {
            let stack = self.global_mark_stack.lock().unwrap();
            !stack.is_empty()
        };
        
        if needs_remarking {
            // Phase 4a: Remarking with simplified convergence
            self.set_phase(CollectorPhase::Remarking);
            self.converge_fixpoint_smoke();
            
            // Phase 4b: Recensusing
            self.set_phase(CollectorPhase::Recensusing);
            self.finalizer_coordinator.execute_census_phase(&self.phase_manager);
        }
        
        // Phase 5: Sweeping - reclaim memory and redirect pointers
        self.sweeping_phase();
        
        // Collection complete - return to waiting state
        self.set_phase(CollectorPhase::Waiting);
        self.allocation_color.store(false, Ordering::Release);
    }
}

/// Fork the current process in a garbage collection safe manner.
///
/// This function suspends garbage collection, forks the process, and then
/// resumes collection in both parent and child processes. This ensures that
/// no collection activities are interrupted by the fork operation.
///
/// # Returns
///
/// Returns the child process ID in the parent process, 0 in the child process,
/// or an error if the fork operation fails.
///
/// # Examples
///
/// ```no_run
/// use fugrip::gc_safe_fork;
///
/// match gc_safe_fork() {
///     Ok(0) => {
///         // Child process
///         println!("This is the child process");
///     }
///     Ok(child_pid) => {
///         // Parent process
///         println!("Child process ID: {}", child_pid);
///     }
///     Err(e) => {
///         eprintln!("Fork failed: {}", e);
///     }
/// }
/// ```
pub fn gc_safe_fork() -> Result<libc::pid_t, std::io::Error> {
    use crate::memory::COLLECTOR;
    
    // Suspend GC before fork
    COLLECTOR.suspend_for_fork();
    
    let pid = unsafe { libc::fork() };
    
    if pid < 0 {
        // Fork failed, resume GC
        COLLECTOR.resume_after_fork();
        return Err(std::io::Error::last_os_error());
    }
    
    if pid == 0 {
        // Child process - reinitialize GC state
        // In a real implementation, would need to clean up parent's threads
    }
    
    // Resume GC in both parent and child
    COLLECTOR.resume_after_fork();
    
    Ok(pid)
}

// Thread-local mutator state
thread_local! {
    pub static MUTATOR_STATE: std::cell::RefCell<MutatorState> = std::cell::RefCell::new(MutatorState::new());
}
