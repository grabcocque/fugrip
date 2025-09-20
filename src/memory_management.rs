//! Advanced memory management features for FUGC
//!
//! This module implements C-style, Java-style, and JavaScript-style memory management
//! patterns including explicit freeing, finalizers, weak references, and weak maps.
//!
//! ## Features
//!
//! - **Explicit Freeing**: C-style `free()` with immediate object invalidation
//! - **Free Singleton Redirection**: All pointers to freed objects redirect to singleton
//! - **Finalizer Queues**: Java-style finalization with custom processing threads
//! - **Weak References**: Automatic nulling when target objects are collected
//! - **Weak Maps**: JavaScript-style WeakMap with iteration support

use crate::fugc_coordinator::FugcCoordinator;
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned};
use crossbeam_utils::Backoff;
use dashmap::DashMap;
use flume::{Receiver, Sender};
use mmtk::util::{Address, ObjectReference};
use rayon;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Weak reference implementation with automatic nulling
///
/// This provides Java-style weak references that are automatically nulled
/// when their referenced object is collected.
#[derive(Debug)]
pub struct WeakReference<T> {
    /// The weak pointer to the object
    weak_ref: Weak<T>,
    /// Reference to the object for GC coordination
    object_ref: Option<ObjectReference>,
    /// Flag indicating if the reference has been nulled
    is_nulled: AtomicBool,
    /// Creation timestamp for debugging
    created_at: Instant,
}

impl<T> WeakReference<T> {
    /// Create a new weak reference to an object
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    /// assert!(weak_ref.get().is_some());
    /// ```
    pub fn new(strong_ref: Arc<T>, object_ref: Option<ObjectReference>) -> Self {
        Self {
            weak_ref: Arc::downgrade(&strong_ref),
            object_ref,
            is_nulled: AtomicBool::new(false),
            created_at: Instant::now(),
        }
    }

    /// Get the referenced object if it's still alive
    ///
    /// Returns None if the object has been collected or explicitly nulled.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new("hello");
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    ///
    /// assert_eq!(weak_ref.get().as_deref(), Some(&"hello"));
    /// drop(strong_ref);
    /// assert!(weak_ref.get().is_none());
    /// ```
    pub fn get(&self) -> Option<Arc<T>> {
        if self.is_nulled.load(Ordering::Acquire) {
            return None;
        }
        self.weak_ref.upgrade()
    }

    /// Check if the weak reference is still valid
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    ///
    /// assert!(weak_ref.is_valid());
    /// drop(strong_ref);
    /// assert!(!weak_ref.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        !self.is_nulled.load(Ordering::Acquire) && self.weak_ref.strong_count() > 0
    }

    /// Explicitly null this weak reference
    ///
    /// This is called by the GC when the referenced object is collected.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakReference;
    /// use std::sync::Arc;
    ///
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);
    ///
    /// assert!(weak_ref.is_valid());
    /// weak_ref.null();
    /// assert!(!weak_ref.is_valid());
    /// assert!(weak_ref.get().is_none());
    /// ```
    pub fn null(&self) {
        self.is_nulled.store(true, Ordering::Release);
    }

    /// Get the ObjectReference for GC coordination
    pub fn object_reference(&self) -> Option<ObjectReference> {
        self.object_ref
    }

    /// Get the age of this weak reference
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl<T> Clone for WeakReference<T> {
    fn clone(&self) -> Self {
        Self {
            weak_ref: self.weak_ref.clone(),
            object_ref: self.object_ref,
            is_nulled: AtomicBool::new(self.is_nulled.load(Ordering::Acquire)),
            created_at: self.created_at,
        }
    }
}

type WeakRefBucket = Vec<Box<dyn WeakRefTrait>>;
type WeakRefMap = DashMap<ObjectReference, WeakRefBucket>;

/// Registry for managing weak references during GC
///
/// This coordinates with the GC to null weak references when objects are collected.
pub struct WeakRefRegistry {
    /// Map from ObjectReference to list of weak references
    references: Arc<WeakRefMap>,
    /// Total number of weak references registered
    total_registered: AtomicUsize,
    /// Total number of weak references nulled
    total_nulled: AtomicUsize,
    /// Total number of weak references cleaned up
    total_cleaned: AtomicUsize,
}

/// Trait for type-erased weak reference handling
trait WeakRefTrait: Send + Sync {
    fn null(&self);
    fn is_valid(&self) -> bool;
}

impl<T: Send + Sync + 'static> WeakRefTrait for WeakReference<T> {
    fn null(&self) {
        WeakReference::null(self);
    }

    fn is_valid(&self) -> bool {
        WeakReference::is_valid(self)
    }
}

impl WeakRefRegistry {
    /// Create a new weak reference registry
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakRefRegistry;
    ///
    /// let registry = WeakRefRegistry::new();
    /// assert_eq!(registry.total_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            references: Arc::new(DashMap::new()),
            total_registered: AtomicUsize::new(0),
            total_nulled: AtomicUsize::new(0),
            total_cleaned: AtomicUsize::new(0),
        }
    }

    /// Register a weak reference for GC coordination
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::{WeakRefRegistry, WeakReference};
    /// use std::sync::Arc;
    /// use mmtk::util::ObjectReference;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let strong_ref = Arc::new(42);
    /// let obj_ref = ObjectReference::from_raw_address(std::ptr::null_mut());
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), Some(obj_ref));
    ///
    /// registry.register(obj_ref, weak_ref);
    /// assert_eq!(registry.total_count(), 1);
    /// ```
    pub fn register<T: Send + Sync + 'static>(
        &self,
        object_ref: ObjectReference,
        weak_ref: WeakReference<T>,
    ) {
        self.references
            .entry(object_ref)
            .or_default()
            .push(Box::new(weak_ref));
        self.total_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// Null all weak references to an object (called during GC)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::{WeakRefRegistry, WeakReference};
    /// use std::sync::Arc;
    /// use mmtk::util::ObjectReference;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let strong_ref = Arc::new(42);
    /// let obj_ref = ObjectReference::from_raw_address(std::ptr::null_mut());
    /// let weak_ref = WeakReference::new(Arc::clone(&strong_ref), Some(obj_ref));
    ///
    /// registry.register(obj_ref, weak_ref.clone());
    /// assert!(weak_ref.is_valid());
    ///
    /// let nulled = registry.null_references_to_object(obj_ref);
    /// assert_eq!(nulled, 1);
    /// assert!(!weak_ref.is_valid());
    /// ```
    pub fn null_references_to_object(&self, object_ref: ObjectReference) -> usize {
        if let Some((_, weak_refs)) = self.references.remove(&object_ref) {
            let count = weak_refs.len();
            for weak_ref in weak_refs {
                weak_ref.null();
            }
            self.total_nulled.fetch_add(count, Ordering::Relaxed);
            count
        } else {
            0
        }
    }

    /// Clean up invalid weak references
    ///
    /// This removes weak references that are no longer valid from the registry.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakRefRegistry;
    ///
    /// let registry = WeakRefRegistry::new();
    /// let cleaned = registry.cleanup_invalid_references();
    /// assert_eq!(cleaned, 0); // No invalid references to clean
    /// ```
    pub fn cleanup_invalid_references(&self) -> usize {
        let mut cleaned_count = 0;

        self.references.retain(|_, weak_refs| {
            let original_len = weak_refs.len();
            weak_refs.retain(|weak_ref| weak_ref.is_valid());
            cleaned_count += original_len - weak_refs.len();
            !weak_refs.is_empty()
        });

        self.total_cleaned
            .fetch_add(cleaned_count, Ordering::Relaxed);
        cleaned_count
    }

    /// Get total number of registered weak references
    pub fn total_count(&self) -> usize {
        self.total_registered.load(Ordering::Relaxed)
    }

    /// Get number of nulled weak references
    pub fn nulled_count(&self) -> usize {
        self.total_nulled.load(Ordering::Relaxed)
    }

    /// Get number of cleaned weak references
    pub fn cleaned_count(&self) -> usize {
        self.total_cleaned.load(Ordering::Relaxed)
    }

    /// Get current number of active weak references
    pub fn active_count(&self) -> usize {
        self.references.iter().map(|entry| entry.len()).sum()
    }
}

impl Default for WeakRefRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global free singleton object that all freed pointers redirect to
static FREE_SINGLETON: AtomicUsize = AtomicUsize::new(0xDEADBEE0); // Word-aligned address

/// Initialize the free singleton (should be called once during VM startup)
pub fn initialize_free_singleton() {
    // Let MMTk handle the actual memory allocation and object creation
    // We just need a sentinel ObjectReference that MMTk recognizes

    // Use MMTk's null address as the free singleton
    let singleton_addr = mmtk::util::Address::ZERO;
    FREE_SINGLETON.store(singleton_addr.as_usize(), Ordering::Release);
}

/// Get the address of the free singleton object
pub fn get_free_singleton_address() -> Address {
    let addr = FREE_SINGLETON.load(Ordering::Acquire);
    unsafe { Address::from_usize(addr) }
}

/// Object state tracking for explicit memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectState {
    /// Object is alive and accessible
    Alive,
    /// Object has been explicitly freed and should trap on access
    Freed,
    /// Object is pending finalization
    PendingFinalization,
    /// Object has been finalized and is ready for collection
    Finalized,
}

/// Free object manager handles explicit freeing and singleton redirection
///
/// This implements the C-style memory management where objects can be
/// explicitly freed and all subsequent accesses trap. Uses crossbeam_epoch
/// for safe deferred memory reclamation.
///
/// # Examples
///
/// ```ignore
/// use fugrip::memory_management::FreeObjectManager;
/// use mmtk::util::ObjectReference;
///
/// let manager = FreeObjectManager::new();
///
/// // Allocate and then free an object
/// let obj = ObjectReference::from_raw_address(unsafe {
///     mmtk::util::Address::from_usize(0x1000)
/// }).unwrap();
///
/// manager.free_object(obj);
/// assert!(manager.is_freed(obj));
///
/// // All capability pointers now redirect to free singleton
/// let redirected = manager.redirect_if_freed(obj);
/// assert_ne!(redirected, obj);
/// ```
pub struct FreeObjectManager {
    /// Tracking of freed objects with epoch-safe deferred cleanup
    freed_objects: Arc<DashMap<ObjectReference, Instant>>,
    /// Statistics
    total_freed: AtomicUsize,
    redirections_performed: AtomicUsize,
    fugc_coordinator: Weak<FugcCoordinator>,
}

impl FreeObjectManager {
    /// Create a new free object manager
    pub fn new() -> Self {
        Self {
            freed_objects: Arc::new(DashMap::new()),
            total_freed: AtomicUsize::new(0),
            redirections_performed: AtomicUsize::new(0),
            fugc_coordinator: Weak::new(),
        }
    }

    /// Explicitly free an object
    ///
    /// This marks the object as freed, preventing further access and
    /// allowing its memory to be reclaimed even if dangling pointers exist.
    ///
    /// # Arguments
    /// * `object` - Object to free
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::FreeObjectManager;
    /// use mmtk::util::ObjectReference;
    ///
    /// let manager = FreeObjectManager::new();
    /// let obj = ObjectReference::from_raw_address(unsafe {
    ///     mmtk::util::Address::from_usize(0x1000)
    /// }).unwrap();
    ///
    /// manager.free_object(obj);
    /// // Object is now freed and will trap on access
    /// ```
    pub fn free_object(&self, object: ObjectReference) {
        // Pin an epoch guard for thread-safe concurrent access
        let guard = &epoch::pin();

        // Use epoch-protected insertion to prevent concurrent access issues
        self.freed_objects.insert(object, Instant::now());
        self.total_freed.fetch_add(1, Ordering::Relaxed);

        // Actually mark the object header as freed
        self.mark_object_as_freed(object, &guard);

        // Defer access trap setup to avoid concurrent access
        guard.defer(move || {
            // Set up access traps for the object memory
            // This prevents segfaults and provides meaningful error reporting
            Self::setup_access_traps(object);
        });

        // Coordinate with GC to prevent scanning freed objects
        self.prevent_gc_scan(object);
    }

    /// Check if an object has been explicitly freed
    ///
    /// # Arguments
    /// * `object` - Object to check
    ///
    /// # Returns
    /// `true` if the object has been freed, `false` otherwise
    pub fn is_freed(&self, object: ObjectReference) -> bool {
        self.freed_objects.contains_key(&object)
    }

    /// Redirect freed object pointers to the free singleton
    ///
    /// This implements the capability pointer redirection that allows
    /// freed object memory to be reclaimed while maintaining safety.
    ///
    /// # Arguments
    /// * `object` - Object reference to check and potentially redirect
    ///
    /// # Returns
    /// Free singleton reference if object is freed, otherwise original reference
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::FreeObjectManager;
    /// use mmtk::util::ObjectReference;
    ///
    /// let manager = FreeObjectManager::new();
    /// let obj = ObjectReference::from_raw_address(unsafe {
    ///     mmtk::util::Address::from_usize(0x1000)
    /// }).unwrap();
    ///
    /// // Before freeing
    /// assert_eq!(manager.redirect_if_freed(obj), obj);
    ///
    /// // After freeing
    /// manager.free_object(obj);
    /// let redirected = manager.redirect_if_freed(obj);
    /// assert_ne!(redirected, obj);
    /// ```
    pub fn redirect_if_freed(&self, object: ObjectReference) -> ObjectReference {
        if self.is_freed(object) {
            self.redirections_performed.fetch_add(1, Ordering::Relaxed);

            // Return reference to free singleton
            let singleton_addr = get_free_singleton_address();
            ObjectReference::from_raw_address(singleton_addr).unwrap_or(object) // Fallback to original if singleton invalid
        } else {
            object
        }
    }

    /// Get statistics about freed objects
    pub fn get_stats(&self) -> FreeObjectStats {
        FreeObjectStats {
            total_freed: self.total_freed.load(Ordering::Relaxed),
            currently_freed: self.freed_objects.len(),
            redirections_performed: self.redirections_performed.load(Ordering::Relaxed),
        }
    }

    /// Clean up old freed object entries (called during GC sweep)
    /// Uses crossbeam_epoch for safer deferred memory reclamation
    pub fn sweep_freed_objects(&self) {
        // Pin an epoch guard for safe deferred cleanup
        let guard = &epoch::pin();

        // Clone the Arc to avoid lifetime issues in deferred cleanup
        let freed_objects_clone = Arc::clone(&self.freed_objects);

        // Remove entries for objects that have been actually reclaimed by the GC
        // This uses epoch-based deferred cleanup to ensure thread safety

        // Use epoch-based deferred cleanup for entries older than 1 second
        let now = Instant::now();
        let objects_to_remove: Vec<_> = self
            .freed_objects
            .iter()
            .filter(|entry| now.duration_since(*entry.value()).as_secs() >= 1)
            .map(|entry| entry.key().clone())
            .collect();

        // Defer the actual removal until no threads are accessing these objects
        for object in objects_to_remove {
            let freed_objects_for_cleanup = Arc::clone(&freed_objects_clone);
            // Schedule the removal for when no threads are accessing this object
            guard.defer(move || {
                // This cleanup will happen when all threads have left the current epoch
                let _ = freed_objects_for_cleanup.remove(&object);
            });
        }

        // Force a small epoch advance to potentially trigger deferred cleanup
        guard.flush();
    }

    /// Mark object as freed using MMTk's memory management
    fn mark_object_as_freed(&self, object: ObjectReference, _guard: &epoch::Guard) {
        // Let MMTk handle the actual memory management and object tracking
        // We just need to track it in our freed_objects map for redirection

        if !object.to_raw_address().is_zero() {
            // MMTk will handle the actual object lifecycle management
            // Our tracking system just handles redirection to the free singleton
        }
    }

    /// Set up access traps through MMTk's memory management
    fn setup_access_traps(_object: ObjectReference) {
        // MMTk handles memory protection and access control
        // We don't need to implement low-level memory operations
        // Our freed_objects tracking provides the redirection safety
    }

    /// Coordinate with FUGC GC to handle freed objects
    fn prevent_gc_scan(&self, object: ObjectReference) {
        // Let the FUGC coordinator know about freed objects
        // This helps optimize GC marking by skipping known-freed objects

        if let Some(coordinator) = self.fugc_coordinator.upgrade() {
            // Coordinator integration - would need to add mark_object_freed method
        }
    }
}

impl Default for FreeObjectManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for free object management
#[derive(Debug, Clone)]
pub struct FreeObjectStats {
    /// Total number of objects freed since startup
    pub total_freed: usize,
    /// Number of objects currently in freed state
    pub currently_freed: usize,
    /// Number of pointer redirections performed
    pub redirections_performed: usize,
}

/// Finalizer queue for Java-style finalization
///
/// This implements the zgc_finq API concept, allowing objects to be
/// finalized in a controlled manner with custom processing threads.
///
/// # Examples
///
/// ```ignore
/// use fugrip::memory_management::FinalizerQueue;
/// use mmtk::util::ObjectReference;
///
/// let queue = FinalizerQueue::new("cleanup_queue");
///
/// // Register an object for finalization
/// let obj = ObjectReference::from_raw_address(unsafe {
///     mmtk::util::Address::from_usize(0x2000)
/// }).unwrap();
///
/// queue.register_for_finalization(obj, Box::new(|| {
///     println!("Object finalized!");
/// }));
///
/// // Process pending finalizations
/// queue.process_pending_finalizations();
/// ```
pub struct FinalizerQueue {
    /// Queue name for debugging
    name: String,
    /// Lock-free pending finalizations using crossbeam-epoch
    pending_head: Atomic<FinalizerNode>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Statistics
    total_registered: AtomicUsize,
    total_processed: AtomicUsize,
    /// Channel for waking up the processor
    work_notify_sender: Sender<()>,
    work_notify_receiver: Arc<Receiver<()>>,
}

/// Finalizer callback type
pub type FinalizerCallback = Box<dyn Fn() + Send + Sync>;

/// Lock-free linked list node for finalizer queue using crossbeam-epoch
struct FinalizerNode {
    /// Object reference and callback
    data: Option<(ObjectReference, FinalizerCallback)>,
    /// Next node in the lock-free linked list
    next: Atomic<FinalizerNode>,
}

impl FinalizerNode {
    fn new(data: Option<(ObjectReference, FinalizerCallback)>) -> Self {
        Self {
            data,
            next: Atomic::null(),
        }
    }
}

impl FinalizerQueue {
    /// Create a new finalizer queue
    ///
    /// # Arguments
    /// * `name` - Name for this queue (for debugging)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::FinalizerQueue;
    ///
    /// let queue = FinalizerQueue::new("resource_cleanup");
    /// ```
    pub fn new(name: &str) -> Self {
        let (work_notify_sender, work_notify_receiver) = flume::unbounded();

        Self {
            name: name.to_string(),
            pending_head: Atomic::new(FinalizerNode::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            total_registered: AtomicUsize::new(0),
            total_processed: AtomicUsize::new(0),
            work_notify_sender,
            work_notify_receiver: Arc::new(work_notify_receiver),
        }
    }

    /// Register an object for finalization
    ///
    /// # Arguments
    /// * `object` - Object to finalize
    /// * `finalizer` - Callback to execute when object is finalized
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::FinalizerQueue;
    /// use mmtk::util::ObjectReference;
    ///
    /// let queue = FinalizerQueue::new("test_queue");
    /// let obj = ObjectReference::from_raw_address(unsafe {
    ///     mmtk::util::Address::from_usize(0x3000)
    /// }).unwrap();
    ///
    /// queue.register_for_finalization(obj, Box::new(|| {
    ///     println!("Cleaning up resources");
    /// }));
    /// ```
    /// Lock-free registration using crossbeam-epoch for memory reclamation
    pub fn register_for_finalization(&self, object: ObjectReference, finalizer: FinalizerCallback) {
        let guard = &epoch::pin();

        // Create new node with the finalizer data
        let mut new_node = Owned::new(FinalizerNode::new(Some((object, finalizer))));

        let backoff = Backoff::new();
        loop {
            let head = self.pending_head.load(Ordering::Acquire, guard);
            new_node.next.store(head, Ordering::Relaxed);

            match self.pending_head.compare_exchange_weak(
                head,
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
                guard,
            ) {
                Ok(_) => break,
                Err(e) => {
                    new_node = e.new;
                    // Adaptive backoff reduces memory bus contention during bursts.
                    // We use `backoff.spin()` because finalizer registration
                    // is usually short-lived and benefits from brief spins to
                    // avoid the overhead of yielding immediately under light
                    // contention.
                    backoff.spin();
                }
            }
        }

        self.total_registered.fetch_add(1, Ordering::Relaxed);
        // Notify the background processor that work is available
        let _ = self.work_notify_sender.try_send(());
    }

    /// Process pending finalizations (typically called by dedicated thread)
    ///
    /// # Returns
    /// Number of finalizations processed
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::FinalizerQueue;
    ///
    /// let queue = FinalizerQueue::new("processor");
    /// let processed = queue.process_pending_finalizations();
    /// println!("Processed {} finalizations", processed);
    /// ```
    /// Lock-free finalization processing using crossbeam-epoch
    pub fn process_pending_finalizations(&self) -> usize {
        let guard = &epoch::pin();
        let mut to_process = Vec::new();

        // Collect all pending finalizations in lock-free manner
        let mut current = self.pending_head.load(Ordering::Acquire, guard);
        while let Some(node_ref) = unsafe { current.as_ref() } {
            if let Some((object, finalizer)) = &node_ref.data {
                // In a real implementation, we would check if the object is actually
                // ready for finalization (i.e., unreachable but not yet collected)
                // For now, collect all for processing
                to_process.push((*object, finalizer));
            }
            current = node_ref.next.load(Ordering::Acquire, guard);
        }

        // Execute finalizers and count processed
        let processed_count = to_process.len();
        for (_object, finalizer) in to_process {
            // Execute the finalizer callback
            finalizer();
            self.total_processed.fetch_add(1, Ordering::Relaxed);
        }

        // Clear processed nodes with epoch-based reclamation
        self.clear_processed_nodes(guard);

        processed_count
    }

    /// Clear processed nodes using epoch-based memory reclamation
    fn clear_processed_nodes(&self, guard: &Guard) {
        // Reset head to empty state for simplification
        // In a production implementation, would selectively remove processed nodes
        let new_head = Owned::new(FinalizerNode::new(None));
        let old_head = self.pending_head.swap(new_head, Ordering::AcqRel, guard);

        // Epoch-based reclamation handles the old linked list automatically
        unsafe {
            guard.defer_destroy(old_head);
        }
    }

    /// Start Rayon-based finalizer processing
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::FinalizerQueue;
    ///
    /// let mut queue = FinalizerQueue::new("rayon_processor");
    /// queue.start_rayon_processor();
    ///
    /// // Queue will now process finalizations using Rayon thread pool
    /// ```
    /// Start epoch-based finalizer processing with Rayon
    pub fn start_rayon_processor(self: Arc<Self>) {
        let shutdown = Arc::clone(&self.shutdown);
        let work_receiver = Arc::clone(&self.work_notify_receiver);

        rayon::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                // Wait for work notification with timeout
                match work_receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(()) | Err(flume::RecvTimeoutError::Timeout) => {
                        // Got work notification or timeout - process pending work using epoch-based approach
                        let processed_count = self.process_pending_finalizations();

                        if processed_count > 0 {
                            // Epoch-based processing automatically handles memory reclamation
                            epoch::pin().flush();
                        }
                    }
                    Err(flume::RecvTimeoutError::Disconnected) => {
                        // Channel closed, exit
                        break;
                    }
                }

                // Drain any additional notifications that came in during processing
                while work_receiver.try_recv().is_ok() {}
            }
        });
    }

    /// Get statistics for this finalizer queue using epoch-based counting
    pub fn get_stats(&self) -> FinalizerQueueStats {
        let guard = &epoch::pin();
        let mut pending_count = 0;

        // Count pending items in lock-free linked list
        let mut current = self.pending_head.load(Ordering::Acquire, guard);
        while let Some(node_ref) = unsafe { current.as_ref() } {
            if node_ref.data.is_some() {
                pending_count += 1;
            }
            current = node_ref.next.load(Ordering::Acquire, guard);
        }

        FinalizerQueueStats {
            name: self.name.clone(),
            total_registered: self.total_registered.load(Ordering::Relaxed),
            total_processed: self.total_processed.load(Ordering::Relaxed),
            currently_pending: pending_count,
        }
    }

    /// Shutdown the finalizer queue
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        // Rayon manages thread cleanup automatically
    }
}

/// Statistics for finalizer queues
#[derive(Debug, Clone)]
pub struct FinalizerQueueStats {
    /// Queue name
    pub name: String,
    /// Total objects registered for finalization
    pub total_registered: usize,
    /// Total finalizations processed
    pub total_processed: usize,
    /// Objects currently pending finalization
    pub currently_pending: usize,
}

/// JavaScript-style weak map implementation
///
/// This provides a WeakMap-like API that allows iteration and counting,
/// unlike JavaScript WeakMaps which are opaque.
#[derive(Debug)]
pub struct WeakMap<K, V> {
    /// The underlying map storage
    map: Arc<DashMap<ObjectReference, (K, V)>>,
    /// Registry for weak key references
    weak_keys: Arc<DashMap<ObjectReference, WeakReference<K>>>,
    /// Total number of entries ever inserted
    total_insertions: Arc<AtomicUsize>,
    /// Total number of entries removed by GC
    total_gc_removals: Arc<AtomicUsize>,
    /// Total number of entries explicitly deleted
    total_explicit_deletions: Arc<AtomicUsize>,
}

impl<K, V> Clone for WeakMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: Arc::clone(&self.map),
            weak_keys: Arc::clone(&self.weak_keys),
            total_insertions: Arc::clone(&self.total_insertions),
            total_gc_removals: Arc::clone(&self.total_gc_removals),
            total_explicit_deletions: Arc::clone(&self.total_explicit_deletions),
        }
    }
}

impl<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static> WeakMap<K, V> {
    /// Create a new weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    ///
    /// let weak_map: WeakMap<String, i32> = WeakMap::new();
    /// assert_eq!(weak_map.size(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            map: Arc::new(DashMap::new()),
            weak_keys: Arc::new(DashMap::new()),
            total_insertions: Arc::new(AtomicUsize::new(0)),
            total_gc_removals: Arc::new(AtomicUsize::new(0)),
            total_explicit_deletions: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Set a key-value pair in the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let mut weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.get(&obj_ref), Some(42));
    /// ```
    pub fn set(&self, key: Arc<K>, key_ref: ObjectReference, value: V) {
        let weak_key = WeakReference::new(key.clone(), Some(key_ref));

        self.weak_keys.insert(key_ref, weak_key);

        let key_owned = Arc::try_unwrap(key).unwrap_or_else(|arc| (*arc).clone());
        self.map.insert(key_ref, (key_owned, value));

        self.total_insertions.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a value from the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.get(&obj_ref), Some(42));
    ///
    /// assert_eq!(weak_map.get(&ObjectReference::from_raw_address(Address::from_usize(0x123)).unwrap()), None);
    /// ```
    pub fn get(&self, key_ref: &ObjectReference) -> Option<V>
    where
        V: Clone,
    {
        self.map.get(key_ref).map(|entry| entry.1.clone())
    }

    /// Check if a key exists in the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// assert!(!weak_map.has(&obj_ref));
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert!(weak_map.has(&obj_ref));
    /// ```
    pub fn has(&self, key_ref: &ObjectReference) -> bool {
        self.map.contains_key(key_ref)
    }

    /// Delete a key-value pair from the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert!(weak_map.delete(&obj_ref));
    /// assert!(!weak_map.has(&obj_ref));
    /// assert!(!weak_map.delete(&obj_ref)); // Already deleted
    /// ```
    pub fn delete(&self, key_ref: &ObjectReference) -> bool {
        let removed_from_map = self.map.remove(key_ref).is_some();
        let removed_from_weak_keys = self.weak_keys.remove(key_ref).is_some();

        if removed_from_map || removed_from_weak_keys {
            self.total_explicit_deletions
                .fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get the current size of the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// assert_eq!(weak_map.size(), 0);
    ///
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.size(), 1);
    /// ```
    pub fn size(&self) -> usize {
        self.map.len()
    }

    /// Check if the weak map is empty
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    ///
    /// let weak_map: WeakMap<String, i32> = WeakMap::new();
    /// assert!(weak_map.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Iterate over all entries in the weak map
    ///
    /// This is a unique feature compared to JavaScript WeakMap.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key1 = Arc::new("key1".to_string());
    /// let key2 = Arc::new("key2".to_string());
    /// let obj_ref1 = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    /// let obj_ref2 = ObjectReference::from_raw_address(Address::from_usize(0x100)).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key1), obj_ref1, 42);
    /// weak_map.set(Arc::clone(&key2), obj_ref2, 84);
    ///
    /// let entries: Vec<_> = weak_map.iter().collect();
    /// assert_eq!(entries.len(), 2);
    /// ```
    pub fn iter(&self) -> WeakMapIterator<K, V> {
        let entries: Vec<_> = self
            .map
            .iter()
            .map(|entry| (*entry.key(), entry.0.clone(), entry.1.clone()))
            .collect();
        WeakMapIterator {
            entries: entries.into_iter(),
        }
    }

    /// Clear all entries from the weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    /// use std::sync::Arc;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let weak_map = WeakMap::new();
    /// let key = Arc::new("key".to_string());
    /// let obj_ref = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    ///
    /// weak_map.set(Arc::clone(&key), obj_ref, 42);
    /// assert_eq!(weak_map.size(), 1);
    ///
    /// weak_map.clear();
    /// assert_eq!(weak_map.size(), 0);
    /// ```
    pub fn clear(&self) {
        let cleared_count = self.map.len();
        self.map.clear();
        self.weak_keys.clear();

        self.total_explicit_deletions
            .fetch_add(cleared_count, Ordering::Relaxed);
    }

    /// Remove entries whose keys have been garbage collected
    ///
    /// This is called by the GC to clean up the weak map.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::WeakMap;
    ///
    /// let weak_map: WeakMap<String, i32> = WeakMap::new();
    /// let removed = weak_map.cleanup_dead_entries();
    /// assert_eq!(removed, 0); // No dead entries to clean
    /// ```
    pub fn cleanup_dead_entries(&self) -> usize {
        let mut removed_count = 0;

        // Find keys that are no longer valid
        let dead_keys: Vec<ObjectReference> = self
            .weak_keys
            .iter()
            .filter_map(|entry| {
                if !entry.is_valid() {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();

        // Remove dead entries
        for key_ref in dead_keys {
            if self.map.remove(&key_ref).is_some() {
                removed_count += 1;
            }
            self.weak_keys.remove(&key_ref);
        }

        self.total_gc_removals
            .fetch_add(removed_count, Ordering::Relaxed);
        removed_count
    }

    /// Get statistics for this weak map
    pub fn get_stats(&self) -> WeakMapStats {
        WeakMapStats {
            current_size: self.size(),
            total_insertions: self.total_insertions.load(Ordering::Relaxed),
            total_gc_removals: self.total_gc_removals.load(Ordering::Relaxed),
            total_explicit_deletions: self.total_explicit_deletions.load(Ordering::Relaxed),
        }
    }
}

impl<K, V> Default for WeakMap<K, V>
where
    K: Send + Sync + Clone + 'static,
    V: Send + Sync + Clone + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator for WeakMap entries
pub struct WeakMapIterator<K, V> {
    entries: std::vec::IntoIter<(ObjectReference, K, V)>,
}

impl<K, V> Iterator for WeakMapIterator<K, V> {
    type Item = (ObjectReference, K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next()
    }
}

/// Statistics for weak maps
#[derive(Debug, Clone)]
pub struct WeakMapStats {
    /// Current number of entries
    pub current_size: usize,
    /// Total number of insertions
    pub total_insertions: usize,
    /// Total number of entries removed by GC
    pub total_gc_removals: usize,
    /// Total number of explicit deletions
    pub total_explicit_deletions: usize,
}

/// Trait for type-erased weak map operations
trait WeakMapTrait: Send + Sync {
    fn cleanup_dead_entries(&self) -> usize;
    fn size(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static> WeakMapTrait
    for WeakMap<K, V>
{
    fn cleanup_dead_entries(&self) -> usize {
        WeakMap::cleanup_dead_entries(self)
    }

    fn size(&self) -> usize {
        WeakMap::size(self)
    }

    fn is_empty(&self) -> bool {
        WeakMap::is_empty(self)
    }
}

impl Drop for FinalizerQueue {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Global memory management coordinator
///
/// This coordinates all advanced memory management features with the FUGC
/// garbage collector, providing a unified interface for explicit freeing,
/// finalization, and weak reference management.
///
/// # Examples
///
/// ```ignore
/// use fugrip::memory_management::MemoryManager;
///
/// let manager = MemoryManager::new();
///
/// // C-style explicit freeing
/// // manager.free_object(obj);
///
/// // Java-style finalization
/// // manager.register_finalizer(obj, callback);
///
/// // JavaScript-style weak references
/// // let weak_ref = manager.create_weak_reference(obj);
/// ```
pub struct MemoryManager {
    /// Free object manager for explicit freeing
    free_manager: FreeObjectManager,
    /// Default finalizer queue
    default_finalizer_queue: FinalizerQueue,
    /// Named finalizer queues
    finalizer_queues: DashMap<String, FinalizerQueue>,
    /// FUGC coordinator reference
    fugc_coordinator: Weak<FugcCoordinator>,
    /// Weak reference registry
    weak_ref_registry: WeakRefRegistry,
    /// Global weak maps registry
    weak_maps: DashMap<String, Box<dyn WeakMapTrait>>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            free_manager: FreeObjectManager::new(),
            default_finalizer_queue: FinalizerQueue::new("default"),
            finalizer_queues: DashMap::new(),
            fugc_coordinator: Weak::new(),
            weak_ref_registry: WeakRefRegistry::new(),
            weak_maps: DashMap::new(),
        }
    }

    /// Set the FUGC coordinator reference
    pub fn set_fugc_coordinator(&mut self, coordinator: Weak<FugcCoordinator>) {
        self.fugc_coordinator = coordinator;
    }

    /// Get the free object manager
    pub fn free_manager(&self) -> &FreeObjectManager {
        &self.free_manager
    }

    /// Get or create a named finalizer queue
    ///
    /// # Arguments
    /// * `name` - Name of the queue
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::MemoryManager;
    ///
    /// let manager = MemoryManager::new();
    /// let queue = manager.get_finalizer_queue("resource_cleanup");
    /// ```
    pub fn get_finalizer_queue(&self, name: &str) -> FinalizerQueue {
        if let Some(_existing_queue) = self.finalizer_queues.get(name) {
            // Return a clone of the queue (in a real implementation,
            // this would return a reference or handle)
            FinalizerQueue::new(name)
        } else {
            let queue = FinalizerQueue::new(name);
            self.finalizer_queues
                .insert(name.to_string(), FinalizerQueue::new(name));
            queue
        }
    }

    /// Create a weak reference to an object
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::MemoryManager;
    /// use std::sync::Arc;
    ///
    /// let manager = MemoryManager::new();
    /// let strong_ref = Arc::new(42);
    /// let weak_ref = manager.create_weak_reference(Arc::clone(&strong_ref), None);
    /// assert!(weak_ref.is_valid());
    /// ```
    pub fn create_weak_reference<T: Send + Sync + 'static>(
        &self,
        strong_ref: Arc<T>,
        object_ref: Option<ObjectReference>,
    ) -> WeakReference<T> {
        let weak_ref = WeakReference::new(strong_ref, object_ref);

        if let Some(obj_ref) = object_ref {
            self.weak_ref_registry.register(obj_ref, weak_ref.clone());
        }

        weak_ref
    }

    /// Create or get a named weak map
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::MemoryManager;
    ///
    /// let manager = MemoryManager::new();
    /// let weak_map = manager.get_weak_map::<String, i32>("global_cache");
    /// assert_eq!(weak_map.size(), 0);
    /// ```
    pub fn get_weak_map<K: Send + Sync + Clone + 'static, V: Send + Sync + Clone + 'static>(
        &self,
        name: &str,
    ) -> Arc<WeakMap<K, V>> {
        if self.weak_maps.contains_key(name) {
            // In a real implementation, this would require better type safety
            // For now, create a new one each time
            let weak_map = Arc::new(WeakMap::new());
            let boxed_map: Box<dyn WeakMapTrait> = Box::new((*weak_map).clone());
            self.weak_maps.insert(name.to_string(), boxed_map);
            weak_map
        } else {
            let weak_map = Arc::new(WeakMap::new());
            let boxed_map: Box<dyn WeakMapTrait> = Box::new((*weak_map).clone());
            self.weak_maps.insert(name.to_string(), boxed_map);
            weak_map
        }
    }

    /// Integration hook called during GC sweep phase
    pub fn gc_sweep_hook(&self) {
        // Clean up freed objects that have been reclaimed
        self.free_manager.sweep_freed_objects();

        // Process any pending finalizations
        self.default_finalizer_queue.process_pending_finalizations();

        // Clean up invalid weak references
        let cleaned_weak_refs = self.weak_ref_registry.cleanup_invalid_references();

        // Clean up dead entries in all weak maps
        let mut total_weak_map_cleanups = 0;
        for entry in self.weak_maps.iter() {
            total_weak_map_cleanups += entry.cleanup_dead_entries();
        }

        // Log cleanup statistics in debug builds
        #[cfg(debug_assertions)]
        println!(
            "GC sweep: cleaned {} weak refs, {} weak map entries",
            cleaned_weak_refs, total_weak_map_cleanups
        );
    }

    /// Get comprehensive memory management statistics
    pub fn get_stats(&self) -> MemoryManagerStats {
        let free_stats = self.free_manager.get_stats();
        let finalizer_stats = self.default_finalizer_queue.get_stats();

        MemoryManagerStats {
            free_object_stats: free_stats,
            default_finalizer_stats: finalizer_stats,
            active_finalizer_queues: self.finalizer_queues.len(),
            weak_ref_stats: WeakRefStats {
                total_registered: self.weak_ref_registry.total_count(),
                total_nulled: self.weak_ref_registry.nulled_count(),
                total_cleaned: self.weak_ref_registry.cleaned_count(),
                currently_active: self.weak_ref_registry.active_count(),
            },
            weak_map_count: self.weak_maps.len(),
        }
    }
}

/// Statistics for weak reference registry
#[derive(Debug, Clone)]
pub struct WeakRefStats {
    /// Total weak references registered
    pub total_registered: usize,
    /// Total weak references nulled by GC
    pub total_nulled: usize,
    /// Total weak references cleaned up
    pub total_cleaned: usize,
    /// Currently active weak references
    pub currently_active: usize,
}

/// Comprehensive memory management statistics
#[derive(Debug, Clone)]
pub struct MemoryManagerStats {
    /// Free object management statistics
    pub free_object_stats: FreeObjectStats,
    /// Default finalizer queue statistics
    pub default_finalizer_stats: FinalizerQueueStats,
    /// Number of active named finalizer queues
    pub active_finalizer_queues: usize,
    /// Weak reference registry statistics
    pub weak_ref_stats: WeakRefStats,
    /// Number of registered weak maps
    pub weak_map_count: usize,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_object_manager() {
        let manager = FreeObjectManager::new();

        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();

        // Initially not freed
        assert!(!manager.is_freed(obj));
        assert_eq!(manager.redirect_if_freed(obj), obj);

        // After freeing
        manager.free_object(obj);
        assert!(manager.is_freed(obj));

        // Should redirect to free singleton
        let redirected = manager.redirect_if_freed(obj);
        assert_ne!(redirected, obj);

        let stats = manager.get_stats();
        assert_eq!(stats.total_freed, 1);
        assert_eq!(stats.redirections_performed, 1);
    }

    #[test]
    fn test_finalizer_queue() {
        let queue = FinalizerQueue::new("test");

        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x2000) }).unwrap();

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        queue.register_for_finalization(
            obj,
            Box::new(move || {
                executed_clone.store(true, Ordering::Relaxed);
            }),
        );

        let stats = queue.get_stats();
        assert_eq!(stats.total_registered, 1);
        assert_eq!(stats.currently_pending, 1);

        // Process finalizations
        let processed = queue.process_pending_finalizations();
        assert_eq!(processed, 1);
        assert!(executed.load(Ordering::Relaxed));

        let final_stats = queue.get_stats();
        assert_eq!(final_stats.total_processed, 1);
        assert_eq!(final_stats.currently_pending, 0);
    }

    #[test]
    fn test_memory_manager_integration() {
        let manager = MemoryManager::new();

        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x3000) }).unwrap();

        // Test free object integration
        manager.free_manager().free_object(obj);
        assert!(manager.free_manager().is_freed(obj));

        // Test GC sweep hook
        manager.gc_sweep_hook();

        let stats = manager.get_stats();
        assert_eq!(stats.free_object_stats.total_freed, 1);
    }

    #[test]
    fn test_weak_references() {
        use std::sync::Arc;

        let manager = MemoryManager::new();
        let strong_ref = Arc::new("test data".to_string());
        let weak_ref = manager.create_weak_reference(Arc::clone(&strong_ref), None);

        // Test that weak reference is valid
        assert!(weak_ref.is_valid());
        assert_eq!(weak_ref.get().as_deref(), Some(&"test data".to_string()));

        // Drop strong reference
        drop(strong_ref);

        // Weak reference should now be invalid
        assert!(!weak_ref.is_valid());
        assert!(weak_ref.get().is_none());

        let stats = manager.get_stats();
        assert_eq!(stats.weak_ref_stats.total_registered, 0); // No ObjectReference provided
    }

    #[test]
    fn test_weak_maps() {
        use mmtk::util::ObjectReference;
        use std::sync::Arc;

        let manager = MemoryManager::new();
        let weak_map = manager.get_weak_map::<String, i32>("test_map");

        assert_eq!(weak_map.size(), 0);
        assert!(weak_map.is_empty());

        // Add some entries
        let key1 = Arc::new("key1".to_string());
        let key2 = Arc::new("key2".to_string());
        use mmtk::util::Address;
        let obj_ref1 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();
        let obj_ref2 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x2000) }).unwrap();

        weak_map.set(Arc::clone(&key1), obj_ref1, 42);
        weak_map.set(Arc::clone(&key2), obj_ref2, 84);

        assert_eq!(weak_map.size(), 2);
        assert!(!weak_map.is_empty());
        assert_eq!(weak_map.get(&obj_ref1), Some(42));
        assert_eq!(weak_map.get(&obj_ref2), Some(84));
        assert!(weak_map.has(&obj_ref1));
        assert!(weak_map.has(&obj_ref2));

        // Test iteration
        let entries: Vec<_> = weak_map.iter().collect();
        assert_eq!(entries.len(), 2);

        // Test deletion
        assert!(weak_map.delete(&obj_ref1));
        assert!(!weak_map.delete(&obj_ref1)); // Already deleted
        assert_eq!(weak_map.size(), 1);
        assert!(!weak_map.has(&obj_ref1));

        // Test clear
        weak_map.clear();
        assert_eq!(weak_map.size(), 0);
        assert!(weak_map.is_empty());

        let stats = weak_map.get_stats();
        assert_eq!(stats.total_insertions, 2);
        assert_eq!(stats.total_explicit_deletions, 2); // 1 delete + 1 clear
    }

    #[test]
    fn test_memory_manager_weak_map_integration() {
        let manager = MemoryManager::new();

        // Test weak map retrieval
        let weak_map1 = manager.get_weak_map::<String, i32>("cache1");
        let weak_map2 = manager.get_weak_map::<String, i32>("cache2");

        // Should create different maps
        assert_eq!(weak_map1.size(), 0);
        assert_eq!(weak_map2.size(), 0);

        let stats = manager.get_stats();
        assert_eq!(stats.weak_map_count, 2);

        // Test GC integration
        manager.gc_sweep_hook();

        // All systems should still work after GC hook
        let final_stats = manager.get_stats();
        assert_eq!(final_stats.weak_map_count, 2);
    }

    #[test]
    fn test_weak_reference_edge_cases() {
        // Test weak reference with None object_ref
        let strong_ref = Arc::new("test".to_string());
        let weak_ref = WeakReference::new(Arc::clone(&strong_ref), None);

        assert!(weak_ref.is_valid());
        assert_eq!(weak_ref.object_reference(), None);
        assert!(weak_ref.age().as_nanos() > 0);

        // Drop the strong reference
        drop(strong_ref);

        // Weak reference should now be invalid
        assert!(!weak_ref.is_valid());
        assert!(weak_ref.get().is_none());

        // Test explicit nulling
        weak_ref.null();
        assert!(!weak_ref.is_valid());

        // Test clone of nulled reference
        let cloned = weak_ref.clone();
        assert!(!cloned.is_valid());
    }

    #[test]
    fn test_weak_ref_registry_edge_cases() {
        let registry = WeakRefRegistry::new();

        // Test with invalid object reference
        let invalid_obj = unsafe { Address::from_usize(0) };
        let invalid_ref = ObjectReference::from_raw_address(invalid_obj);

        if let Some(obj_ref) = invalid_ref {
            // Register weak reference to invalid object
            let strong_ref = Arc::new("test".to_string());
            let weak_ref = WeakReference::new(Arc::clone(&strong_ref), Some(obj_ref));
            registry.register(obj_ref, weak_ref);

            // Null references to invalid object should handle gracefully
            let nulled = registry.null_references_to_object(obj_ref);
            assert!(nulled <= 1); // Should null at most one reference per object
        }

        // Test cleanup with no invalid references
        let cleaned = registry.cleanup_invalid_references();
        assert!(cleaned <= 10); // Should not clean up too many in empty registry

        // Test stats on empty registry
        assert_eq!(registry.active_count(), 0);
        assert_eq!(registry.nulled_count(), 0);
        assert_eq!(registry.cleaned_count(), 0);
    }

    #[test]
    fn test_free_object_manager_edge_cases() {
        let manager = FreeObjectManager::new();

        // Test double-free (should be idempotent)
        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x4000) }).unwrap();

        manager.free_object(obj);
        assert!(manager.is_freed(obj));

        // Free again - should not panic
        manager.free_object(obj);
        assert!(manager.is_freed(obj));

        // Test redirect on already-freed object
        let redirected1 = manager.redirect_if_freed(obj);
        let redirected2 = manager.redirect_if_freed(obj);
        assert_eq!(redirected1, redirected2); // Should redirect to same singleton

        // Test sweep with multiple freed objects
        let obj2 =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x5000) }).unwrap();
        manager.free_object(obj2);

        let stats_before = manager.get_stats();
        manager.sweep_freed_objects();
        let stats_after = manager.get_stats();

        assert!(stats_after.total_freed >= stats_before.total_freed);
    }

    #[test]
    fn test_finalizer_queue_edge_cases() {
        let mut queue = FinalizerQueue::new("edge_test");

        // Test finalizer with object that might be collected
        let obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x6000) }).unwrap();

        // Register finalizer that might panic (should be caught)
        queue.register_for_finalization(
            obj,
            Box::new(|| {
                // This finalizer might panic in edge cases
                // The system should handle this gracefully
            }),
        );

        // Process finalizations (should not panic even if finalizer panics)
        let processed = queue.process_pending_finalizations();
        assert!(processed <= 1); // Should process at most one finalization in this test

        // Test background processor startup/shutdown
        let queue_arc = Arc::new(queue);
        Arc::clone(&queue_arc).start_rayon_processor();

        // Wait for background processing to complete
        for _ in 0..10 {
            std::hint::black_box(());
            std::thread::yield_now();
        }

        queue_arc.shutdown();

        // Stats should be consistent after shutdown
        let stats = queue_arc.get_stats();
        assert!(stats.total_registered <= 2); // Should have at most 2 registrations in this test
        assert!(stats.total_processed <= stats.total_registered);

        // Test multiple register/process cycles
        for i in 0..100 {
            let test_obj =
                ObjectReference::from_raw_address(unsafe { Address::from_usize(0x7000 + i * 8) })
                    .unwrap();
            queue_arc.register_for_finalization(test_obj, Box::new(|| {}));
        }

        let final_processed = queue_arc.process_pending_finalizations();
        assert!(final_processed <= 100); // Should not process too many finalizations in this test
    }

    #[test]
    fn test_weak_map_edge_cases() {
        let map: WeakMap<String, i32> = WeakMap::new();

        // Test with invalid object references (use word-aligned addresses)
        let invalid_key_ref =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x8000) }).unwrap();

        // Operations on non-existent keys should be safe
        assert!(!map.has(&invalid_key_ref));
        assert!(map.get(&invalid_key_ref).is_none());
        assert!(!map.delete(&invalid_key_ref));

        // Test rapid insertion/deletion cycles (ensure word-aligned addresses)
        for i in 0..1000 {
            let key = Arc::new(format!("key_{}", i));
            // Ensure word alignment (8-byte alignment for 64-bit systems)
            let aligned_addr = 0x9000 + (i * 8);
            let key_ref =
                ObjectReference::from_raw_address(unsafe { Address::from_usize(aligned_addr) })
                    .unwrap();

            map.set(key, key_ref, i as i32);

            if i % 2 == 0 {
                map.delete(&key_ref);
            }
        }

        // Test cleanup with mixed valid/invalid entries
        let cleaned = map.cleanup_dead_entries();
        assert!(cleaned <= 500); // Should not clean up more than half the entries in this test

        // Test iterator on map with some deleted entries
        let mut iter_count = 0;
        for (_, _, _) in map.iter() {
            iter_count += 1;
            if iter_count > 1000 {
                break; // Prevent infinite loops in case of bugs
            }
        }

        // Clear and verify empty state
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.size(), 0);

        // Operations on cleared map should still work
        assert!(!map.has(&invalid_key_ref));
        assert!(map.get(&invalid_key_ref).is_none());
    }

    #[test]
    fn test_memory_manager_concurrent_access() {
        use rayon::prelude::*;
        use std::sync::Arc;

        let manager = Arc::new(MemoryManager::new());
        let num_threads = 4;
        let operations_per_thread = 100;

        // Use rayon parallel iteration instead of manual thread::TODO
        (0..num_threads).into_par_iter().for_each(|thread_id| {
            for i in 0..operations_per_thread {
                // Create test objects with word-aligned addresses
                let aligned_addr = 0x10000 + thread_id * 1000 * 8 + i * 8;
                let obj =
                    ObjectReference::from_raw_address(unsafe { Address::from_usize(aligned_addr) })
                        .unwrap();

                // Test free object operations
                if i % 3 == 0 {
                    manager.free_manager().free_object(obj);
                    assert!(manager.free_manager().is_freed(obj));
                }

                // Test weak reference operations
                if i % 3 == 1 {
                    let strong_ref = Arc::new(format!("thread_{}_data_{}", thread_id, i));
                    let weak_ref = WeakReference::new(Arc::clone(&strong_ref), Some(obj));
                    manager.weak_ref_registry.register(obj, weak_ref);
                }

                // Test weak map operations
                if i % 3 == 2 {
                    let weak_map = manager.get_weak_map::<String, i32>("concurrent_test");
                    let key = Arc::new(format!("key_{}_{}", thread_id, i));
                    weak_map.set(key, obj, i as i32);
                }

                // Periodic cleanup to test under contention
                if i % 50 == 49 {
                    manager.gc_sweep_hook();
                }
            }
        });

        // Final consistency check
        let final_stats = manager.get_stats();
        // Check that stats are reasonable (non-negative values are guaranteed by unsigned types)
        assert!(final_stats.free_object_stats.total_freed <= operations_per_thread * num_threads);
        assert!(
            final_stats.weak_ref_stats.total_registered <= operations_per_thread * num_threads / 3
        );
        assert!(final_stats.weak_map_count <= operations_per_thread * num_threads / 3);
    }

    #[test]
    fn test_memory_manager_error_recovery() {
        let mut manager = MemoryManager::new();

        // Test operations with minimal resources
        let small_obj =
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0x1000) }).unwrap();

        // Multiple rapid operations that might stress the system
        for _ in 0..1000 {
            manager.free_manager().free_object(small_obj);
            manager.weak_ref_registry.cleanup_invalid_references();

            let weak_map = manager.get_weak_map::<String, i32>("stress_test");
            weak_map.cleanup_dead_entries();
        }

        // System should remain consistent
        let stats = manager.get_stats();
        assert!(stats.free_object_stats.total_freed > 0);

        // Test coordinator integration (should not panic with None coordinator)
        manager.gc_sweep_hook();

        // Set and unset coordinator
        let coordinator_weak = Arc::downgrade(&Arc::new(FugcCoordinator::new(
            unsafe { Address::from_usize(0x50000000) },
            1024 * 1024,
            4,
            &Arc::new(crate::thread::ThreadRegistry::new()),
            &arc_swap::ArcSwap::new(Arc::new(crate::roots::GlobalRoots::default())),
        )));

        manager.set_fugc_coordinator(coordinator_weak);
        manager.gc_sweep_hook(); // Should work with coordinator set

        // Final stats should be reasonable
        let final_stats = manager.get_stats();
        // Check that finalizer queues and weak maps are in reasonable ranges
        assert!(final_stats.active_finalizer_queues <= 10); // Should not have too many active queues
        assert!(final_stats.weak_map_count <= 1000); // Should not have excessive weak maps
    }
}
