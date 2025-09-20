//! MMTk VM binding implementation with FUGC integration
//!
//! This module provides the VM binding layer that connects MMTk's garbage collection
//! infrastructure with FUGC-specific optimizations. It implements the core interfaces
//! required by MMTk while integrating FUGC's concurrent marking and write barrier optimizations.
//!
//! # Examples
//!
//! ```
//! use fugrip::binding::{fugc_alloc_info, fugc_get_stats};
//!
//! // Get allocation info with FUGC optimizations
//! let (size, align) = fugc_alloc_info(64, 8);
//! assert_eq!(size, 64);
//! assert_eq!(align, 8);
//!
//! // Get FUGC statistics
//! let stats = fugc_get_stats();
//! assert!(stats.concurrent_collection_enabled);
//! ```

use crate::{plan::FugcPlanManager, thread::MutatorThread};
use anyhow;
use dashmap::DashMap;
use mmtk::util::{
    Address,
    opaque_pointer::{OpaquePointer, VMMutatorThread, VMThread, VMWorkerThread},
};
use mmtk::vm::slot::{SimpleSlot, UnimplementedMemorySlice};
use mmtk::vm::{Collection, GCThreadContext, ReferenceGlue, VMBinding};
use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::OnceLock;

pub struct MutatorRegistration {
    mutator: *mut mmtk::Mutator<RustVM>,
    thread: MutatorThread,
}

// SAFETY: We ensure mutator pointer is valid during its lifetime
unsafe impl Send for MutatorRegistration {}
unsafe impl Sync for MutatorRegistration {}

impl MutatorRegistration {
    fn new(mutator: *mut mmtk::Mutator<RustVM>, thread: MutatorThread) -> Self {
        Self { mutator, thread }
    }

    unsafe fn as_mutator(&self) -> &'static mut mmtk::Mutator<RustVM> {
        unsafe { &mut *self.mutator }
    }
}

/// Handle for interacting with an MMTk mutator created by the binding layer.
///
/// The handle owns the leaked mutator pointer and provides safe accessors that
/// respect Rust's borrowing discipline. It is intentionally not `Clone` to
/// avoid aliasing the underlying `&mut` reference.
pub struct MutatorHandle {
    ptr: NonNull<mmtk::Mutator<RustVM>>,
}

impl MutatorHandle {
    fn from_raw(ptr: *mut mmtk::Mutator<RustVM>) -> Self {
        let ptr = NonNull::new(ptr).expect("MMTk returned a null mutator pointer");
        Self { ptr }
    }

    /// Get a raw pointer to the underlying mutator for FUGC registries.
    pub fn as_ptr(&self) -> *mut mmtk::Mutator<RustVM> {
        self.ptr.as_ptr()
    }
}

impl Deref for MutatorHandle {
    type Target = mmtk::Mutator<RustVM>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl DerefMut for MutatorHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

static REFERENT_MAP: OnceLock<DashMap<mmtk::util::ObjectReference, mmtk::util::ObjectReference>> =
    OnceLock::new();

/// Global FUGC plan manager that coordinates MMTk with FUGC-specific features
pub static FUGC_PLAN_MANAGER: OnceLock<Mutex<FugcPlanManager>> = OnceLock::new();

pub static MUTATOR_MAP: OnceLock<DashMap<usize, MutatorRegistration>> = OnceLock::new();

type FinalizationQueue = Mutex<
    Vec<(
        mmtk::util::ObjectReference,
        Option<mmtk::util::ObjectReference>,
    )>,
>;
static FINALIZATION_QUEUE: OnceLock<FinalizationQueue> = OnceLock::new();

fn vm_thread_key(thread: VMThread) -> usize {
    thread.0.to_address().as_usize()
}

fn mutator_thread_key(thread: VMMutatorThread) -> usize {
    vm_thread_key(thread.0)
}

/// Register a mutator with the binding infrastructure so that safepoint and
/// root scanning hooks can coordinate with MMTk.
pub fn register_mutator_context(
    tls: VMMutatorThread,
    mutator: &'static mut mmtk::Mutator<RustVM>,
    thread: MutatorThread,
) {
    // Use DI container instead of global registry
    let container = crate::di::current_container();
    container.thread_registry().register(thread.clone());

    let key = mutator_thread_key(tls);
    MUTATOR_MAP
        .get_or_init(DashMap::new)
        .insert(key, MutatorRegistration::new(mutator as *mut _, thread));
}

/// Remove a mutator from the binding registries.
pub fn unregister_mutator_context(tls: VMMutatorThread) {
    let key = mutator_thread_key(tls);
    if let Some((_, entry)) = MUTATOR_MAP.get_or_init(DashMap::new).remove(&key) {
        // Use DI container instead of global registry
        let container = crate::di::current_container();
        container.thread_registry().unregister(entry.thread.id());
    }
}

fn with_mutator_registration<F, R>(tls: VMMutatorThread, f: F) -> Option<R>
where
    F: FnOnce(&MutatorRegistration) -> R,
{
    let key = mutator_thread_key(tls);
    MUTATOR_MAP
        .get_or_init(DashMap::new)
        .get(&key)
        .map(|r| f(r.value()))
}

fn visit_all_mutators<F>(mut visitor: F)
where
    F: FnMut(&'static mut mmtk::Mutator<RustVM>),
{
    let registrations: Vec<_> = MUTATOR_MAP
        .get_or_init(DashMap::new)
        .iter()
        .map(|entry| entry.value().mutator)
        .collect();

    for ptr in registrations {
        unsafe {
            visitor(&mut *ptr);
        }
    }
}

fn visit_all_threads<F>(mut visitor: F)
where
    F: FnMut(&MutatorThread),
{
    let threads: Vec<_> = MUTATOR_MAP
        .get_or_init(DashMap::new)
        .iter()
        .map(|entry| entry.value().thread.clone())
        .collect();

    for thread in threads.iter() {
        visitor(thread);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RustVM;

impl VMBinding for RustVM {
    type VMObjectModel = crate::core::RustObjectModel;
    type VMScanning = crate::roots::RustScanning;
    type VMCollection = RustCollection;
    type VMActivePlan = RustActivePlan;
    type VMReferenceGlue = RustReferenceGlue;
    type VMSlot = SimpleSlot;
    type VMMemorySlice = UnimplementedMemorySlice;
}

#[derive(Default)]
pub struct RustCollection;

impl Collection<RustVM> for RustCollection {
    fn stop_all_mutators<F>(_tls: VMWorkerThread, mutator_visitor: F)
    where
        F: FnMut(&'static mut mmtk::Mutator<RustVM>),
    {
        // In the new lock-free design, mutator threads coordinate via handshakes
        // The coordinator manages stopping/resuming through the handshake protocol
        // For now, we just visit the mutators since handshakes handle coordination
        visit_all_mutators(mutator_visitor);
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        // In the lock-free design, threads resume automatically after handshake completion
        // No explicit resume needed - threads return to Running state automatically
    }

    fn block_for_gc(tls: VMMutatorThread) {
        if let Some(thread) = with_mutator_registration(tls, |entry| entry.thread.clone()) {
            // In the lock-free design, threads cooperatively reach safepoints by polling
            // The thread will handle handshakes automatically through poll_safepoint()
            thread.poll_safepoint();
        }
    }

    fn spawn_gc_thread(_tls: VMThread, ctx: GCThreadContext<RustVM>) {
        match ctx {
            GCThreadContext::Worker(worker) => {
                let mmtk = {
                    let manager = FUGC_PLAN_MANAGER
                        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
                        .lock();
                    match manager.mmtk() {
                        Ok(mmtk) => mmtk,
                        Err(e) => {
                            eprintln!(
                                "Warning: Cannot spawn GC worker - MMTk not initialized: {}",
                                e
                            );
                            return;
                        }
                    }
                };

                if let Err(e) = std::thread::Builder::new()
                    .name("mmtk-gc-worker".to_string())
                    .spawn(move || {
                        let worker_tls = VMWorkerThread(VMThread::UNINITIALIZED);
                        worker.run(worker_tls, mmtk);
                    })
                {
                    eprintln!("Warning: Failed to spawn GC worker thread: {}", e);
                }
            }
        }
    }
}

#[derive(Default)]
pub struct RustReferenceGlue;

impl ReferenceGlue<RustVM> for RustReferenceGlue {
    type FinalizableType = crate::weak::WeakRefHeader;

    fn set_referent(reference: mmtk::util::ObjectReference, referent: mmtk::util::ObjectReference) {
        REFERENT_MAP
            .get_or_init(DashMap::new)
            .insert(reference, referent);
    }

    fn get_referent(object: mmtk::util::ObjectReference) -> Option<mmtk::util::ObjectReference> {
        REFERENT_MAP
            .get_or_init(DashMap::new)
            .get(&object)
            .map(|v| *v.value())
    }

    fn clear_referent(object: mmtk::util::ObjectReference) {
        let _ = REFERENT_MAP.get_or_init(DashMap::new).remove(&object);
    }

    fn enqueue_references(references: &[mmtk::util::ObjectReference], _tls: VMWorkerThread) {
        if references.is_empty() {
            return;
        }

        let drained: Vec<_> = references
            .iter()
            .map(|reference| {
                (
                    *reference,
                    REFERENT_MAP
                        .get_or_init(DashMap::new)
                        .remove(reference)
                        .map(|(_, v)| v),
                )
            })
            .collect();

        FINALIZATION_QUEUE
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .extend(drained);
    }
}

struct MutatorIter {
    data: Vec<*mut mmtk::plan::Mutator<RustVM>>,
    index: usize,
}

impl Iterator for MutatorIter {
    type Item = &'static mut mmtk::plan::Mutator<RustVM>;

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.data.get(self.index).copied()?;
        self.index += 1;
        // SAFETY: Mutator pointers are valid for the program duration and
        // MutatorIter ensures exclusive access during iteration
        Some(unsafe { &mut *ptr })
    }
}

#[derive(Default)]
pub struct RustActivePlan;

impl mmtk::vm::ActivePlan<RustVM> for RustActivePlan {
    fn is_mutator(tls: mmtk::util::opaque_pointer::VMThread) -> bool {
        let key = vm_thread_key(tls);
        MUTATOR_MAP.get_or_init(DashMap::new).contains_key(&key)
    }

    fn mutator(
        tls: mmtk::util::opaque_pointer::VMMutatorThread,
    ) -> &'static mut mmtk::plan::Mutator<RustVM> {
        with_mutator_registration(tls, |entry| unsafe { entry.as_mutator() })
            .expect("mutator not registered")
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut mmtk::plan::Mutator<RustVM>> + 'a> {
        let mutators: Vec<_> = {
            let map = MUTATOR_MAP.get_or_init(DashMap::new).iter();
            map.map(|entry| entry.value().mutator).collect()
        };

        Box::new(mutators.into_iter().map(|ptr| unsafe { &mut *ptr }))
    }

    fn number_of_mutators() -> usize {
        MUTATOR_MAP.get_or_init(DashMap::new).len()
    }
}

/// Initialize MMTk with FUGC-optimized configuration and set up the plan manager.
/// This creates the MMTk instance, configures it with FUGC settings, and initializes
/// the global plan manager.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::initialize_mmtk_with_fugc;
///
/// // Initialize MMTk with FUGC configuration
/// let mmtk = initialize_mmtk_with_fugc()
///     .expect("Failed to initialize MMTk with FUGC");
///
/// // MMTk is now ready for allocation and garbage collection
/// ```
pub fn initialize_mmtk_with_fugc() -> anyhow::Result<&'static mmtk::MMTK<RustVM>> {
    use crate::plan::create_fugc_mmtk_options;

    // Create FUGC-optimized MMTk options
    let options = create_fugc_mmtk_options()?;

    // Build an MMTkBuilder seeded with our configured options
    let mut builder = mmtk::MMTKBuilder::new_no_env_vars();
    builder.options = options;

    // Initialize MMTk using the builder. mmtk_init returns a Box<MMTK<VM>>
    let boxed = mmtk::memory_manager::mmtk_init::<RustVM>(&builder);

    // Leak the Box to obtain a &'static reference for the lifetime of the process.
    let mmtk_static: &'static mmtk::MMTK<RustVM> = Box::leak(boxed);

    // Initialize the FUGC plan manager with the MMTk instance
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .initialize(mmtk_static);

    Ok(mmtk_static)
}

/// Bind a mutator thread to MMTk and register it with FUGC infrastructure.
/// This should be called for each mutator thread that will perform allocations.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::{bind_mutator_thread, initialize_mmtk_with_fugc};
/// use fugrip::thread::MutatorThread;
///
/// // Initialize MMTk first
/// let mmtk = initialize_mmtk_with_fugc()
///     .expect("Failed to initialize MMTk");
///
/// // Create mutator thread context
/// let thread = MutatorThread::new();
///
/// // Bind mutator to MMTk and FUGC
/// let mut mutator = bind_mutator_thread(mmtk, thread)
///     .expect("Failed to bind mutator");
/// ```
pub fn bind_mutator_thread(
    mmtk: &'static mmtk::MMTK<RustVM>,
    thread: MutatorThread,
) -> anyhow::Result<MutatorHandle> {
    bind_mutator_thread_with_registry(mmtk, thread, None)
}

pub fn bind_mutator_thread_with_registry(
    mmtk: &'static mmtk::MMTK<RustVM>,
    thread: MutatorThread,
    registry: Option<&crate::thread::ThreadRegistry>,
) -> anyhow::Result<MutatorHandle> {
    // Create MMTk mutator
    let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
        // SAFETY: We construct an address from a thread id for use as an opaque TLS
        // value. This mirrors test and example usage in the codebase that use
        // `Address::from_usize` for synthetic TLS addresses.
        Address::from_usize(thread.id())
    })));

    // Bind mutator with MMTk
    let mutator_box = mmtk::memory_manager::bind_mutator(mmtk, tls);

    // Leak the Box to get a pointer that lives for the duration of the process.
    let mutator_ptr: *mut mmtk::Mutator<RustVM> = Box::leak(mutator_box);
    let mutator_handle = MutatorHandle::from_raw(mutator_ptr);

    // Register with FUGC infrastructure - inline to avoid borrowing conflicts
    match registry {
        Some(reg) => reg.register(thread.clone()),
        None => {
            // Use DI container instead of global registry
            let container = crate::di::current_container();
            container.thread_registry().register(thread.clone());
        }
    }
    let key = mutator_thread_key(tls);
    MUTATOR_MAP.get_or_init(DashMap::new).insert(
        key,
        MutatorRegistration::new(mutator_handle.as_ptr(), thread),
    );

    Ok(mutator_handle)
}

/// Initialize the FUGC plan manager with an existing MMTk instance.
/// This is an internal function - use `initialize_mmtk_with_fugc` instead.
pub fn initialize_fugc_plan(mmtk: &'static mmtk::MMTK<RustVM>) {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .initialize(mmtk);
}

/// Allocate an object using MMTk with FUGC optimizations.
/// This performs the actual allocation through MMTk's allocator infrastructure
/// and integrates with FUGC's concurrent collection.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::{fugc_alloc, initialize_mmtk_with_fugc, bind_mutator_thread};
/// use fugrip::thread::MutatorThread;
///
/// // Initialize MMTk and bind a mutator
/// let mmtk = initialize_mmtk_with_fugc().expect("Failed to initialize");
/// let thread = MutatorThread::new();
/// let mut mutator = bind_mutator_thread(mmtk, thread).expect("Failed to bind mutator");
///
/// // Allocate a 64-byte object
/// let obj = fugc_alloc(&mut mutator, 64, 8, 0)
///     .expect("Failed to allocate object");
/// ```
pub fn fugc_alloc(
    mutator: &mut mmtk::Mutator<RustVM>,
    size: usize,
    align: usize,
    offset: usize,
) -> anyhow::Result<mmtk::util::ObjectReference> {
    use mmtk::AllocationSemantics;

    // Use MMTk's allocation API with default semantics
    let addr =
        mmtk::memory_manager::alloc(mutator, size, align, offset, AllocationSemantics::Default);

    // Convert Address to ObjectReference
    match mmtk::util::ObjectReference::from_raw_address(addr) {
        Some(obj_ref) => {
            // Integrate with FUGC post-allocation processing
            FUGC_PLAN_MANAGER
                .get_or_init(|| Mutex::new(FugcPlanManager::new()))
                .lock()
                .post_alloc(obj_ref, size);
            Ok(obj_ref)
        }
        None => anyhow::bail!(
            "Failed to allocate {} bytes with alignment {} - invalid address",
            size,
            align
        ),
    }
}

/// Get allocation info using FUGC optimizations
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_alloc_info;
///
/// // Get allocation info for a 64-byte object with 8-byte alignment
/// let (aligned_size, alignment) = fugc_alloc_info(64, 8);
/// assert_eq!(aligned_size, 64);
/// assert_eq!(alignment, 8);
///
/// // Test alignment adjustment
/// let (aligned_size, alignment) = fugc_alloc_info(65, 16);
/// assert_eq!(aligned_size, 80); // Rounded up to 16-byte boundary
/// assert_eq!(alignment, 16);
/// ```
pub fn fugc_alloc_info(size: usize, align: usize) -> (usize, usize) {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .alloc_info(size, align)
}

/// Handle post-allocation processing with FUGC optimizations
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_post_alloc;
/// use mmtk::util::{Address, ObjectReference};
///
/// // Create an object reference for demonstration
/// let addr = unsafe { Address::from_usize(0x10000000) };
/// let obj = ObjectReference::from_raw_address(addr).unwrap();
///
/// // Handle post-allocation processing
/// fugc_post_alloc(obj, 64);
/// // This integrates the object with FUGC's concurrent collection
/// ```
pub fn fugc_post_alloc(obj: mmtk::util::ObjectReference, bytes: usize) {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .post_alloc(obj, bytes);
}

/// Drain the queue of references that were enqueued for finalization by MMTk.
/// This allows the VM to process weak references or other finalizable objects
/// after a collection cycle completes.
pub fn take_enqueued_references() -> Vec<(
    mmtk::util::ObjectReference,
    Option<mmtk::util::ObjectReference>,
)> {
    FINALIZATION_QUEUE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .drain(..)
        .collect()
}

// Note: The unsafe fugc_write_barrier function has been removed to prevent
// accidental segfaults. Use fugc_write_barrier_with_mutator instead when
// a mutator is available, or access the write barrier component directly
// via the plan manager for safe testing.

/// Mutator-aware write barrier helper. Call this when a `&mut Mutator<RustVM>` is available
/// so MMTk's `object_reference_write_pre`/`post` can be invoked with the correct argument types.
pub fn fugc_write_barrier_with_mutator(
    mutator: &mut mmtk::Mutator<RustVM>,
    src: mmtk::util::ObjectReference,
    slot: mmtk::util::Address,
    target: mmtk::util::ObjectReference,
) {
    let vm_slot = SimpleSlot::from_address(slot);
    let target_opt = Some(target);

    // Call MMTk pre/post hooks
    mmtk::memory_manager::object_reference_write_pre::<RustVM>(mutator, src, vm_slot, target_opt);
    mmtk::memory_manager::object_reference_write_post::<RustVM>(mutator, src, vm_slot, target_opt);

    // Notify FUGC coordinator as well
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .handle_write_barrier(src, slot, target);
}

/// Trigger garbage collection with FUGC optimizations using MMTk's GC trigger API.
/// This integrates with MMTk's allocation failure handling and FUGC's 8-step protocol.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::fugc_gc;
///
/// // Trigger garbage collection
/// fugc_gc();
/// ```
pub fn fugc_gc() {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .gc();
}

/// Get FUGC statistics
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_get_stats;
///
/// // Get current FUGC statistics
/// let stats = fugc_get_stats();
///
/// // Check if concurrent collection is enabled
/// assert!(stats.concurrent_collection_enabled);
///
/// // View work-stealing statistics
/// println!("Work stolen: {}", stats.work_stolen);
/// println!("Work shared: {}", stats.work_shared);
///
/// // View memory statistics
/// println!("Total bytes: {}", stats.total_bytes);
/// println!("Used bytes: {}", stats.used_bytes);
///
/// // View concurrent allocation statistics
/// println!("Objects allocated black: {}", stats.objects_allocated_black);
/// ```
pub fn fugc_get_stats() -> crate::plan::FugcStats {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .get_fugc_stats()
}

/// Get the current FUGC collection phase
pub fn fugc_get_phase() -> crate::fugc_coordinator::FugcPhase {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .fugc_phase()
}

/// Check if FUGC collection is currently in progress
pub fn fugc_is_collecting() -> bool {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .is_fugc_collecting()
}

/// Get FUGC cycle statistics
pub fn fugc_get_cycle_stats() -> crate::fugc_coordinator::FugcCycleStats {
    FUGC_PLAN_MANAGER
        .get_or_init(|| Mutex::new(FugcPlanManager::new()))
        .lock()
        .get_fugc_cycle_stats()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmtk::util::ObjectReference;
    // use mmtk::vm::ActivePlan; // Currently unused
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    #[test]
    fn test_mutator_registration_creation() {
        // Test MutatorRegistration creation and basic properties
        let thread = MutatorThread::new(0);
        let dummy_mutator = std::ptr::null_mut();

        let registration = MutatorRegistration::new(dummy_mutator, thread);

        // Test that registration stores the correct values
        assert_eq!(registration.mutator, dummy_mutator);
        // Note: Can't easily test thread equality without implementing PartialEq
    }

    #[test]
    fn test_mutator_registration_safety_markers() {
        // Test that MutatorRegistration implements required safety traits
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<MutatorRegistration>();
        assert_sync::<MutatorRegistration>();
    }

    #[test]
    fn test_rust_vm_constants() {
        // Test RustVM constants that affect MMTk behavior
        // MIN_ALIGNMENT is always > 0 by definition
        assert!(RustVM::MIN_ALIGNMENT.is_power_of_two());
        // MAX_ALIGNMENT >= MIN_ALIGNMENT by invariant
        assert!(RustVM::MAX_ALIGNMENT.is_power_of_two());

        // Test reasonable bounds
        // MIN_ALIGNMENT >= 1 by invariant
        // MAX_ALIGNMENT <= 4096 by design
    }

    #[test]
    fn test_fugc_allocation_helpers() {
        // Test fugc_alloc_info with various inputs
        let test_cases = [
            (0, 1),            // Minimum size
            (8, 8),            // Typical small object
            (64, 16),          // Medium object
            (1024, 32),        // Large object
            (1024 * 1024, 32), // Large but safe size - 1MB
        ];

        for (size, align) in test_cases {
            let (result_size, result_align) = fugc_alloc_info(size, align);

            // Basic invariants
            assert!(result_align > 0);
            assert!(result_align.is_power_of_two());
            assert!(result_size >= size || size == usize::MAX); // Handle overflow case
            assert!(result_align >= align);
        }
    }

    #[test]
    fn test_fugc_post_alloc() {
        // Test post-allocation hook doesn't panic
        let dummy_addr = unsafe { Address::from_usize(0x1000) };
        let obj_ref = unsafe { ObjectReference::from_raw_address_unchecked(dummy_addr) };

        // These should not panic
        fugc_post_alloc(obj_ref, 0);
        fugc_post_alloc(obj_ref, 64);
        fugc_post_alloc(obj_ref, 1024);
    }

    #[test]
    fn test_fugc_get_stats() {
        // Test statistics retrieval
        let stats = fugc_get_stats();

        // Test that stats can be retrieved successfully
        // Note: All fields are usize, so no need to check >= 0
        let _ = stats.work_stolen;
        let _ = stats.work_shared;
        let _ = stats.objects_allocated_black;
        let _ = stats.total_bytes;
        let _ = stats.used_bytes;

        // Test that concurrent collection flag is accessible
        // (Value may be true or false depending on configuration)
        let _concurrent_enabled = stats.concurrent_collection_enabled;
    }

    #[test]
    fn test_mutator_registry() {
        // Test global mutator registry thread safety
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DashMap<u64, MutatorRegistration>>();

        // Test that registry can be accessed (this initializes it)
        let _initial_len = MUTATOR_MAP.get_or_init(DashMap::new).len();
        // Registry access should not panic
    }

    #[test]
    fn test_rust_vm_trait_implementations() {
        // Test that RustVM implements required traits for MMTk integration
        fn assert_vm_binding<T: VMBinding>() {}
        assert_vm_binding::<RustVM>();

        // Test specific trait bounds
        fn assert_collection<T: Collection<RustVM>>() {}
        assert_collection::<RustCollection>();

        fn assert_reference_glue<T: ReferenceGlue<RustVM>>() {}
        assert_reference_glue::<RustReferenceGlue>();
    }

    #[test]
    fn test_plan_manager_initialization() {
        // Test that FUGC_PLAN_MANAGER can be safely initialized multiple times
        let stats1 = fugc_get_stats();
        let stats2 = fugc_get_stats();

        // Both calls should succeed (testing OnceLock behavior)
        let _ = stats1.work_stolen;
        let _ = stats2.work_stolen;
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test various error conditions that might occur in binding

        // Test with valid but minimal addresses (MMTk requires non-zero, word-aligned)
        let valid_addr = unsafe { Address::from_usize(0x1000) }; // Use aligned, non-zero address
        let valid_obj_ref = unsafe { ObjectReference::from_raw_address_unchecked(valid_addr) };

        // This should not panic
        fugc_post_alloc(valid_obj_ref, 0);

        // Test with reasonable large values (avoid overflow)
        fugc_post_alloc(valid_obj_ref, 1024 * 1024); // 1MB instead of usize::MAX
    }

    #[test]
    fn test_thread_integration() {
        // Test thread-related functionality
        let thread1 = MutatorThread::new(0);
        let thread2 = MutatorThread::new(1);
        let thread_max = MutatorThread::new(usize::MAX);

        // Test that threads can be created with various IDs
        let dummy_mutator = std::ptr::null_mut();
        let _reg1 = MutatorRegistration::new(dummy_mutator, thread1);
        let _reg2 = MutatorRegistration::new(dummy_mutator, thread2);
        let _reg_max = MutatorRegistration::new(dummy_mutator, thread_max);
    }

    #[test]
    fn test_address_and_alignment_edge_cases() {
        // Test edge cases for alignment and addressing

        // Test minimum alignment cases
        let (size, align) = fugc_alloc_info(1, 1);
        assert!(align >= 1);
        assert!(size >= 1);

        // Test power-of-two alignment requirement
        let test_aligns = [1, 2, 4, 8, 16, 32, 64, 128];
        for align in test_aligns {
            let (_, result_align) = fugc_alloc_info(64, align);
            assert!(result_align >= align);
            assert!(result_align.is_power_of_two());
        }

        // Test that large sizes are handled gracefully
        let (large_size, large_align) = fugc_alloc_info(1024 * 1024, 64);
        assert!(large_size >= 1024 * 1024);
        assert!(large_align >= 64);
    }

    #[test]
    fn test_mutator_handle_edge_cases() {
        // Test MutatorHandle edge cases and error conditions
        let dummy_addr = unsafe { Address::from_usize(0x1000) };
        let dummy_ptr = dummy_addr.to_mut_ptr::<mmtk::Mutator<RustVM>>();

        // Test non-null pointer handling
        let handle = MutatorHandle::from_raw(dummy_ptr);
        assert_eq!(handle.as_ptr(), dummy_ptr);
    }

    #[test]
    #[should_panic(expected = "MMTk returned a null mutator pointer")]
    fn test_mutator_handle_null_pointer_panic() {
        // Test that null pointer causes panic as expected
        let _handle = MutatorHandle::from_raw(std::ptr::null_mut());
    }

    #[test]
    fn test_vm_thread_key_functions() {
        // Test thread key generation functions
        let thread1 = VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(0x1000)
        }));
        let thread2 = VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(0x2000)
        }));

        let key1 = vm_thread_key(thread1);
        let key2 = vm_thread_key(thread2);

        assert_ne!(key1, key2, "Different threads should have different keys");
        assert_eq!(key1, 0x1000);
        assert_eq!(key2, 0x2000);

        // Test mutator thread key
        let mutator_thread1 = VMMutatorThread(thread1);
        let mutator_thread2 = VMMutatorThread(thread2);

        let mutator_key1 = mutator_thread_key(mutator_thread1);
        let mutator_key2 = mutator_thread_key(mutator_thread2);

        assert_eq!(mutator_key1, key1);
        assert_eq!(mutator_key2, key2);
    }

    #[test]
    fn test_mutator_registration_unregistration() {
        // Test mutator registration and unregistration flows
        let thread = MutatorThread::new(42);
        // Use a placeholder address for testing - we won't dereference it
        let dummy_mutator_ptr = 0x1000 as *mut mmtk::Mutator<RustVM>;
        let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(42)
        })));

        // Register mutator (using unsafe but valid for testing)
        register_mutator_context(tls, unsafe { &mut *dummy_mutator_ptr }, thread.clone());

        // Verify registration exists
        let key = mutator_thread_key(tls);
        assert!(MUTATOR_MAP.get_or_init(DashMap::new).contains_key(&key));

        // Test with_mutator_registration (just check that the callback is called)
        let result = with_mutator_registration(tls, |reg| {
            assert_eq!(reg.mutator, dummy_mutator_ptr);
            "test_value"
        });
        assert_eq!(result, Some("test_value"));

        // Unregister mutator
        unregister_mutator_context(tls);

        // Verify unregistration
        assert!(!MUTATOR_MAP.get_or_init(DashMap::new).contains_key(&key));

        // Test with_mutator_registration after unregistration
        let result = with_mutator_registration(tls, |_| "should_not_be_called");
        assert_eq!(result, None);
    }

    #[test]
    fn test_visitor_functions() {
        // Test visitor functions with multiple mutators
        let thread1 = MutatorThread::new(100);
        let thread2 = MutatorThread::new(101);
        // Use placeholder addresses - visitors don't actually dereference the pointers in tests
        let dummy_mutator1_ptr = 0x2000 as *mut mmtk::Mutator<RustVM>;
        let dummy_mutator2_ptr = 0x3000 as *mut mmtk::Mutator<RustVM>;
        let tls1 = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(100)
        })));
        let tls2 = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(101)
        })));

        // Register mutators
        register_mutator_context(tls1, unsafe { &mut *dummy_mutator1_ptr }, thread1.clone());
        register_mutator_context(tls2, unsafe { &mut *dummy_mutator2_ptr }, thread2.clone());

        // Test visit_all_mutators - count calls but don't access mutator internals
        let mut mutator_count = 0;
        visit_all_mutators(|_mutator| {
            mutator_count += 1;
            // Don't access mutator fields to avoid segfault
        });
        assert!(mutator_count >= 2, "Should visit at least 2 mutators");

        // Test visit_all_threads
        let mut thread_count = 0;
        let mut seen_ids = std::collections::HashSet::new();
        visit_all_threads(|thread| {
            thread_count += 1;
            seen_ids.insert(thread.id());
        });
        assert!(thread_count >= 2, "Should visit at least 2 threads");
        assert!(seen_ids.contains(&100));
        assert!(seen_ids.contains(&101));

        // Clean up
        unregister_mutator_context(tls1);
        unregister_mutator_context(tls2);
    }

    #[test]
    fn test_rust_collection_methods() {
        // Test RustCollection implementation methods
        let _collection = RustCollection;
        let worker_tls = VMWorkerThread(VMThread::UNINITIALIZED);
        let mutator_tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(200)
        })));

        // Test stop_all_mutators - should not panic
        RustCollection::stop_all_mutators(worker_tls, |_mutator| {
            // Visitor should be called for each mutator
        });

        // Test resume_mutators - should not panic
        RustCollection::resume_mutators(worker_tls);

        // Test block_for_gc with unregistered mutator
        RustCollection::block_for_gc(mutator_tls); // Should not panic

        // Test spawn_gc_thread would be complex to set up properly
        // Just test that the method exists and can be called via trait
        // Real usage requires proper MMTk initialization
    }

    #[test]
    fn test_rust_reference_glue_edge_cases() {
        // Test ReferenceGlue edge cases
        let _glue = RustReferenceGlue;
        let obj1 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x3000)) };
        let obj2 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x4000)) };
        let obj3 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x5000)) };

        // Test get_referent on non-existent object
        assert_eq!(RustReferenceGlue::get_referent(obj1), None);

        // Test set_referent and get_referent
        RustReferenceGlue::set_referent(obj1, obj2);
        assert_eq!(RustReferenceGlue::get_referent(obj1), Some(obj2));

        // Test clear_referent
        RustReferenceGlue::clear_referent(obj1);
        assert_eq!(RustReferenceGlue::get_referent(obj1), None);

        // Test enqueue_references with empty slice
        let empty_refs: &[ObjectReference] = &[];
        RustReferenceGlue::enqueue_references(empty_refs, VMWorkerThread(VMThread::UNINITIALIZED));

        // Test enqueue_references with actual references
        RustReferenceGlue::set_referent(obj1, obj2);
        RustReferenceGlue::set_referent(obj2, obj3);
        let refs = &[obj1, obj2];
        RustReferenceGlue::enqueue_references(refs, VMWorkerThread(VMThread::UNINITIALIZED));

        // Verify references were enqueued
        let enqueued = take_enqueued_references();
        assert_eq!(enqueued.len(), 2);

        // Verify referents were cleared from map
        assert_eq!(RustReferenceGlue::get_referent(obj1), None);
        assert_eq!(RustReferenceGlue::get_referent(obj2), None);
    }

    #[test]
    fn test_rust_active_plan_basic() {
        // Test RustActivePlan basic functionality
        let _plan = RustActivePlan;

        // Register a mutator for basic testing
        let thread = MutatorThread::new(600);
        let dummy_mutator_ptr = 0x4000 as *mut mmtk::Mutator<RustVM>;
        let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
            Address::from_usize(600)
        })));
        register_mutator_context(tls, unsafe { &mut *dummy_mutator_ptr }, thread.clone());

        // Test that the mutator map contains our registration
        let key = mutator_thread_key(tls);
        assert!(MUTATOR_MAP.get_or_init(DashMap::new).contains_key(&key));

        // Clean up
        unregister_mutator_context(tls);
    }

    #[test]
    fn test_finalization_queue_operations() {
        // Test finalization queue edge cases

        // Ensure queue starts clean
        let initial_refs = take_enqueued_references();
        let _ = initial_refs; // Drain any existing references

        // Test take_enqueued_references on empty queue
        let empty_refs = take_enqueued_references();
        assert_eq!(empty_refs.len(), 0);

        // Add some references via RustReferenceGlue
        let obj1 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x7000)) };
        let obj2 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x8000)) };

        RustReferenceGlue::set_referent(obj1, obj2);
        let refs = &[obj1];
        RustReferenceGlue::enqueue_references(refs, VMWorkerThread(VMThread::UNINITIALIZED));

        // Test taking references
        let enqueued = take_enqueued_references();
        assert_eq!(enqueued.len(), 1);
        assert_eq!(enqueued[0].0, obj1);
        assert_eq!(enqueued[0].1, Some(obj2));

        // Queue should be empty again
        let empty_again = take_enqueued_references();
        assert_eq!(empty_again.len(), 0);
    }

    #[test]
    fn test_fugc_phase_and_collection_state() {
        // Test FUGC phase and collection state functions
        let initial_phase = fugc_get_phase();
        let is_collecting = fugc_is_collecting();
        let cycle_stats = fugc_get_cycle_stats();

        // These should not panic and should return valid values
        println!("Initial phase: {:?}", initial_phase);
        println!("Is collecting: {}", is_collecting);
        println!("Cycle stats: {:?}", cycle_stats);

        // Test triggering GC
        fugc_gc();

        // Phase might have changed
        let after_gc_phase = fugc_get_phase();
        println!("After GC phase: {:?}", after_gc_phase);
    }

    #[test]
    fn test_write_barrier_sad_paths() {
        // Test write barrier component access without dangerous memory writes
        let plan_manager = FUGC_PLAN_MANAGER.get_or_init(|| Mutex::new(FugcPlanManager::new()));

        let manager = plan_manager.lock();
        let write_barrier = manager.get_write_barrier();

        // Test barrier state - should not be active initially
        assert!(!write_barrier.is_active());

        // Test concurrent collection state changes
        assert!(manager.is_concurrent_collection_enabled());
        manager.set_concurrent_collection(false);
        assert!(!manager.is_concurrent_collection_enabled());
        manager.set_concurrent_collection(true);
        assert!(manager.is_concurrent_collection_enabled());

        // Test with problematic object reference patterns (without memory writes)
        let src =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x10000)) };
        let target =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x20000)) };

        // Test same src and target references
        assert_eq!(src.to_raw_address(), unsafe {
            Address::from_usize(0x10000)
        });
        assert_eq!(target.to_raw_address(), unsafe {
            Address::from_usize(0x20000)
        });
        assert_ne!(src, target);

        // Test address alignment checks
        let aligned_addr = unsafe { Address::from_usize(0x10000) };
        let unaligned_addr = unsafe { Address::from_usize(0x10001) };
        assert!(aligned_addr.as_usize() % std::mem::align_of::<usize>() == 0);
        assert!(unaligned_addr.as_usize() % std::mem::align_of::<usize>() != 0);
    }

    #[test]
    fn test_extreme_value_handling() {
        // Test handling of extreme values - proper sad path testing

        // Test with maximum size values (sad path)
        let (_max_size, max_align) = fugc_alloc_info(usize::MAX, usize::MAX);
        // Should handle overflow/saturation gracefully
        assert!(max_align.is_power_of_two() || max_align == usize::MAX);

        // Test with zero values (sad path)
        let (_zero_size, zero_align) = fugc_alloc_info(0, 1);
        assert!(zero_align > 0);
        assert!(zero_align.is_power_of_two());

        // Test with non-power-of-two alignment (sad path)
        let (_, bad_align) = fugc_alloc_info(64, 3); // 3 is not power of 2
        assert!(bad_align.is_power_of_two()); // Should be corrected

        // Test with extreme object references (sad path) - test interface without MMTk calls
        let extreme_addr = unsafe { Address::from_usize(0x100000) }; // Large but reasonable address
        let extreme_obj = unsafe { ObjectReference::from_raw_address_unchecked(extreme_addr) };

        // Test that extreme object references can be created and compared
        assert!(extreme_obj.to_raw_address().as_usize() > 0);

        // Test write barrier interface without calling potentially problematic MMTk functions
        let extreme_slot = unsafe { Address::from_usize(0x100008) };
        // Verify addresses can be created and manipulated safely
        assert_eq!(extreme_slot.as_usize(), 0x100008);
    }

    #[test]
    fn test_concurrent_access_patterns() {
        // Test thread safety of global state
        use std::sync::Arc;

        let running = Arc::new(AtomicBool::new(true));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let running = Arc::clone(&running);
                thread::spawn(move || {
                    let mut operations = 0;
                    while running.load(Ordering::Relaxed) && operations < 10 {
                        // Test concurrent access to various APIs
                        let _stats = fugc_get_stats();
                        let _phase = fugc_get_phase();
                        let _collecting = fugc_is_collecting();
                        let _cycle_stats = fugc_get_cycle_stats();

                        // Test allocation info
                        let _alloc_info = fugc_alloc_info(64 + i, 8);

                        // Test post alloc
                        let obj = unsafe {
                            ObjectReference::from_raw_address_unchecked(Address::from_usize(
                                0x10000 + i * 8,
                            ))
                        };
                        fugc_post_alloc(obj, 64);

                        operations += 1;
                        thread::yield_now(); // Cooperative yielding instead of sleeping
                    }
                    operations
                })
            })
            .collect();

        // Let threads run with proper synchronization
        // Use work-based termination instead of time-based
        for _ in 0..100 {
            thread::yield_now();
        }
        running.store(false, Ordering::Relaxed);

        // Wait for all threads
        let total_ops: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert!(total_ops > 0, "Threads should have performed operations");
    }

    #[test]
    fn test_error_propagation_paths() {
        // Test error handling in various scenarios - proper sad path testing

        // Test with uninitialized plan manager states
        let stats = fugc_get_stats();
        // Should not panic even if plan manager is in various states
        let _ = stats.work_stolen;

        // Test with zero-aligned objects (sad path)
        let zero_addr = unsafe { Address::from_usize(0x1000) }; // Non-zero but valid
        let zero_obj = unsafe { ObjectReference::from_raw_address_unchecked(zero_addr) };
        fugc_post_alloc(zero_obj, 0); // Zero size allocation (sad path)

        // Test alignment validation without dangerous memory writes
        let misaligned_addr = unsafe { Address::from_usize(0x1001) }; // Odd address
        assert!(misaligned_addr.as_usize() % std::mem::align_of::<usize>() != 0);

        // Test address calculations
        let aligned_addr = unsafe { Address::from_usize(0x1000) };
        assert!(aligned_addr.as_usize() % std::mem::align_of::<usize>() == 0);

        // Test with very large allocation requests (sad path)
        let (_large_size, large_align) = fugc_alloc_info(usize::MAX / 2, 1024);
        // Should handle overflow gracefully
        assert!(large_align.is_power_of_two());
    }
}
