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
use lazy_static::lazy_static;
use mmtk::util::{
    Address,
    opaque_pointer::{OpaquePointer, VMMutatorThread, VMThread, VMWorkerThread},
};
use mmtk::vm::slot::{SimpleSlot, UnimplementedMemorySlice};
use mmtk::vm::{Collection, GCThreadContext, ReferenceGlue, VMBinding};
use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

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

lazy_static! {
    static ref REFERENT_MAP: DashMap<mmtk::util::ObjectReference, mmtk::util::ObjectReference> =
        DashMap::new();

    /// Global FUGC plan manager that coordinates MMTk with FUGC-specific features
    pub static ref FUGC_PLAN_MANAGER: Mutex<FugcPlanManager> = Mutex::new(FugcPlanManager::new());

    pub static ref MUTATOR_MAP: DashMap<usize, MutatorRegistration> = DashMap::new();

    static ref FINALIZATION_QUEUE: Mutex<Vec<(mmtk::util::ObjectReference, Option<mmtk::util::ObjectReference>)>> =
        Mutex::new(Vec::new());
}

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
    MUTATOR_MAP.insert(key, MutatorRegistration::new(mutator as *mut _, thread));
}

/// Remove a mutator from the binding registries.
pub fn unregister_mutator_context(tls: VMMutatorThread) {
    let key = mutator_thread_key(tls);
    if let Some((_, entry)) = MUTATOR_MAP.remove(&key) {
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
    MUTATOR_MAP.get(&key).map(|r| f(r.value()))
}

fn visit_all_mutators<F>(mut visitor: F)
where
    F: FnMut(&'static mut mmtk::Mutator<RustVM>),
{
    let registrations: Vec<_> = MUTATOR_MAP
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
                    let manager = FUGC_PLAN_MANAGER.lock();
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
        REFERENT_MAP.insert(reference, referent);
    }

    fn get_referent(object: mmtk::util::ObjectReference) -> Option<mmtk::util::ObjectReference> {
        REFERENT_MAP.get(&object).map(|v| *v.value())
    }

    fn clear_referent(object: mmtk::util::ObjectReference) {
        let _ = REFERENT_MAP.remove(&object);
    }

    fn enqueue_references(references: &[mmtk::util::ObjectReference], _tls: VMWorkerThread) {
        if references.is_empty() {
            return;
        }

        let drained: Vec<_> = references
            .iter()
            .map(|reference| (*reference, REFERENT_MAP.remove(reference).map(|(_, v)| v)))
            .collect();

        FINALIZATION_QUEUE.lock().extend(drained);
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
        MUTATOR_MAP.contains_key(&key)
    }

    fn mutator(
        tls: mmtk::util::opaque_pointer::VMMutatorThread,
    ) -> &'static mut mmtk::plan::Mutator<RustVM> {
        with_mutator_registration(tls, |entry| unsafe { entry.as_mutator() })
            .expect("mutator not registered")
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut mmtk::plan::Mutator<RustVM>> + 'a> {
        let mutators: Vec<_> = {
            let map = MUTATOR_MAP.iter();
            map.map(|entry| entry.value().mutator).collect()
        };

        Box::new(mutators.into_iter().map(|ptr| unsafe { &mut *ptr }))
    }

    fn number_of_mutators() -> usize {
        MUTATOR_MAP.len()
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
    FUGC_PLAN_MANAGER.lock().initialize(mmtk_static);

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
    MUTATOR_MAP.insert(
        key,
        MutatorRegistration::new(mutator_handle.as_ptr(), thread),
    );

    Ok(mutator_handle)
}

/// Initialize the FUGC plan manager with an existing MMTk instance.
/// This is an internal function - use `initialize_mmtk_with_fugc` instead.
pub fn initialize_fugc_plan(mmtk: &'static mmtk::MMTK<RustVM>) {
    FUGC_PLAN_MANAGER.lock().initialize(mmtk);
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
            FUGC_PLAN_MANAGER.lock().post_alloc(obj_ref, size);
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
    FUGC_PLAN_MANAGER.lock().alloc_info(size, align)
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
    FUGC_PLAN_MANAGER.lock().post_alloc(obj, bytes);
}

/// Drain the queue of references that were enqueued for finalization by MMTk.
/// This allows the VM to process weak references or other finalizable objects
/// after a collection cycle completes.
pub fn take_enqueued_references() -> Vec<(
    mmtk::util::ObjectReference,
    Option<mmtk::util::ObjectReference>,
)> {
    FINALIZATION_QUEUE.lock().drain(..).collect()
}

/// Handle write barrier with FUGC optimizations using MMTk's memory manager API.
/// This should be called on every pointer store to maintain concurrent marking invariants.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::fugc_write_barrier;
/// use mmtk::util::{Address, ObjectReference};
///
/// // Create object references for demonstration
/// let src_addr = unsafe { Address::from_usize(0x10000000) };
/// let target_addr = unsafe { Address::from_usize(0x10001000) };
/// let slot_addr = unsafe { Address::from_usize(0x10000008) };
///
/// let src = ObjectReference::from_raw_address(src_addr).unwrap();
/// let target = ObjectReference::from_raw_address(target_addr).unwrap();
///
/// // Handle write barrier for pointer update
/// fugc_write_barrier(src, slot_addr, target);
/// // This ensures concurrent marking invariants are maintained
/// ```
pub fn fugc_write_barrier(
    src: mmtk::util::ObjectReference,
    slot: mmtk::util::Address,
    target: mmtk::util::ObjectReference,
) {
    // If the caller doesn't have a mutator reference, just notify the FUGC coordinator.
    // Call sites that have a mutator should use `fugc_write_barrier_with_mutator` to
    // invoke MMTk's pre/post hooks with the correct mutator argument.
    FUGC_PLAN_MANAGER
        .lock()
        .handle_write_barrier(src, slot, target);
}

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
    FUGC_PLAN_MANAGER.lock().gc();
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
    FUGC_PLAN_MANAGER.lock().get_fugc_stats()
}

/// Get the current FUGC collection phase
pub fn fugc_get_phase() -> crate::fugc_coordinator::FugcPhase {
    FUGC_PLAN_MANAGER.lock().fugc_phase()
}

/// Check if FUGC collection is currently in progress
pub fn fugc_is_collecting() -> bool {
    FUGC_PLAN_MANAGER.lock().is_fugc_collecting()
}

/// Get FUGC cycle statistics
pub fn fugc_get_cycle_stats() -> crate::fugc_coordinator::FugcCycleStats {
    FUGC_PLAN_MANAGER.lock().get_fugc_cycle_stats()
}
