//! FUGC - A Rust Implementation of Epic Games' Unbelievable Garbage Collector
//!
//! This crate provides a concurrent, non-moving garbage collector inspired by
//! Epic Games' FUGC (Fil's Unbelievable Garbage Collector) from the Verse
//! programming language. It implements advanced GC features like free singleton
//! redirection and sophisticated weak reference handling while maintaining
//! compatibility with Rust's borrow checker.
//!
//! ## Key Features
//!
//! - **Advancing Wavefront Collection**: Mutator cannot create new work for the
//!   collector once marking has begun for a cycle
//! - **Non-moving Design**: Ensures pointer stability for FFI compatibility
//! - **Free Singleton Redirection**: Atomically redirects dead object pointers
//!   to prevent use-after-free
//! - **Concurrent Collection**: Background collection with minimal pause times
//! - **Safepoint Infrastructure**: Cooperative handshake mechanism for thread
//!   coordination
//! - **Fork Safety**: Built-in support for process forking
//!
//! ## Quick Start
//!
//! ```rust
//! use fugrip::Gc;
//!
//! // Create garbage-collected objects
//! let gc_string = Gc::new("Hello, FUGC!".to_string());
//! let gc_number = Gc::new(42i32);
//!
//! // Read from GC objects
//! if let Some(string_ref) = gc_string.read() {
//!     println!("String: {}", *string_ref);
//! }
//!
//! if let Some(number_ref) = gc_number.read() {
//!     println!("Number: {}", *number_ref);
//! }
//! ```
//!
//! ## Architecture Overview
//!
//! The garbage collector consists of several key components:
//!
//! - **Core Smart Pointers**: [`Gc<T>`] for managed objects with atomic
//!   redirection to [`FREE_SINGLETON`] for collected objects
//! - **Memory Management**: Segmented heap with thread-local allocation buffers
//! - **Collection Phases**: Concurrent marking, weak reference censusing,
//!   finalizer reviving, and sweeping
//! - **Thread Coordination**: [`CollectorState`] manages GC phases and worker
//!   threads, [`MutatorState`] handles thread-local operations
//! - **Object Classification**: Different object types allocated in distinct
//!   heaps for efficient collection
//!
//! ## Safety and Threading
//!
//! FUGC uses a safepoint-based approach to ensure thread safety:
//!
//! ```rust
//! use fugrip::{Gc, memory::COLLECTOR};
//! use std::thread;
//!
//! // Threads automatically participate in safepoint protocol
//! let handle = thread::spawn(|| {
//!     COLLECTOR.register_mutator_thread();
//!     
//!     let obj = Gc::new(vec![1, 2, 3, 4, 5]);
//!     
//!     // Safepoint checks happen automatically during GC operations
//!     if let Some(data) = obj.read() {
//!         println!("Data: {:?}", *data);
//!     }
//!     
//!     COLLECTOR.unregister_mutator_thread();
//! });
//!
//! handle.join().unwrap();
//! ```

// Crate module declarations
pub mod collector;
pub mod collector_phases;
pub mod interfaces;
pub mod memory;
pub mod traits;
pub mod types;

#[cfg(any(test, feature = "smoke"))]
pub mod test_heap;

// Re-export the public API surface from submodules
pub use collector_phases::{
    AllocationBuffer, CollectorState, DefaultHeapBoundsChecker, HandshakeAction, HeapBoundsChecker,
    MutatorState, StackScanner, ThreadCoordinator, ThreadRegistration, gc_safe_fork,
};

#[cfg(feature = "smoke")]
pub use collector_phases::{smoke_add_global_root, smoke_add_stack_root, smoke_clear_all_roots};
pub use traits::*;
pub use types::{
    CollectorPhase, FREE_SINGLETON_TYPE_INFO, FinalizableObject, FreeSingleton, Gc, GcConfig,
    GcError, GcHeader, GcRef, GcRefMut, GcResult, GcStats, ObjectClass, ObjectType, ReadGuard,
    SendPtr, ThreadLocalBuffer, TypeInfo, Weak, WeakRef, WriteGuard, align_up,
    finalizable_type_info, type_info,
};

pub use memory::{
    ALLOCATOR, CLASSIFIED_ALLOCATOR, COLLECTOR, ClassifiedAllocator, Heap, MemoryRegion, ObjectSet,
    ROOTS, Segment, SegmentBuffer, SegmentedHeap, WeakReference, execute_census_phase,
    register_root, scan_stacks,
};

/// Trait for objects that need custom cleanup when collected by the garbage collector.
///
/// Types that implement `Finalizable` can perform custom cleanup operations
/// when the garbage collector determines they are no longer reachable. This
/// is similar to destructors but runs during the GC finalization phase.
///
/// # Examples
///
/// Basic finalizable object:
///
/// ```rust
/// use fugrip::Finalizable;
/// use std::sync::atomic::{AtomicBool, Ordering};
/// use std::sync::Arc;
///
/// struct FileHandle {
///     name: String,
///     is_finalized: Arc<AtomicBool>,
/// }
///
/// impl Finalizable for FileHandle {
///     fn finalize(&mut self) {
///         println!("Finalizing file handle: {}", self.name);
///         self.is_finalized.store(true, Ordering::Release);
///         // Perform cleanup like closing file descriptors
///     }
/// }
///
/// let finalized_flag = Arc::new(AtomicBool::new(false));
/// let mut file_handle = FileHandle {
///     name: "example.txt".to_string(),
///     is_finalized: finalized_flag.clone(),
/// };
///
/// // Manually trigger finalization for demonstration
/// file_handle.finalize();
/// assert!(finalized_flag.load(Ordering::Acquire));
/// ```
///
/// Using with FinalizableObject wrapper:
///
/// ```rust
/// use fugrip::{Finalizable, FinalizableObject};
///
/// struct Resource {
///     id: u32,
/// }
///
/// impl Finalizable for Resource {
///     fn finalize(&mut self) {
///         println!("Resource {} is being finalized", self.id);
///         // Cleanup resource
///     }
/// }
///
/// let resource = Resource { id: 42 };
/// let finalizable = FinalizableObject::new(resource);
///
/// // The resource will be finalized when collected
/// assert!(!finalizable.is_finalized());
/// ```
pub trait Finalizable {
    /// Called by the garbage collector when the object is being finalized.
    ///
    /// This method is invoked during the reviving phase of garbage collection
    /// for objects that are determined to be unreachable. Implementations
    /// should perform any necessary cleanup operations.
    ///
    /// # Safety
    ///
    /// This method is called by the GC and should not:
    /// - Create new GC references that could resurrect the object
    /// - Perform operations that could panic
    /// - Access other objects that might also be in the finalization process
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugrip::Finalizable;
    ///
    /// struct NetworkConnection {
    ///     socket_fd: i32,
    /// }
    ///
    /// impl Finalizable for NetworkConnection {
    ///     fn finalize(&mut self) {
    ///         // Close the network socket
    ///         if self.socket_fd >= 0 {
    ///             // In real code, would call close(self.socket_fd)
    ///             println!("Closing socket {}", self.socket_fd);
    ///             self.socket_fd = -1;
    ///         }
    ///     }
    /// }
    /// ```
    fn finalize(&mut self);
}
// No local tests; rely on crate-wide tests
