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

use crate::{
    plan::FugcPlanManager,
    thread::{MutatorThread, ThreadRegistry},
};
use lazy_static::lazy_static;
use mmtk::util::opaque_pointer::{VMMutatorThread, VMThread, VMWorkerThread};
use mmtk::vm::slot::{SimpleSlot, UnimplementedMemorySlice};
use mmtk::vm::{Collection, GCThreadContext, ReferenceGlue, VMBinding};
use std::collections::HashMap;
use std::sync::Mutex;

struct MutatorRegistration {
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

lazy_static! {
    static ref REFERENT_MAP: Mutex<HashMap<mmtk::util::ObjectReference, mmtk::util::ObjectReference>> =
        Mutex::new(HashMap::new());

    /// Global FUGC plan manager that coordinates MMTk with FUGC-specific features
    pub static ref FUGC_PLAN_MANAGER: Mutex<FugcPlanManager> = Mutex::new(FugcPlanManager::new());

    static ref MUTATOR_MAP: Mutex<HashMap<usize, MutatorRegistration>> =
        Mutex::new(HashMap::new());

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
    ThreadRegistry::global().register(thread.clone());

    let key = mutator_thread_key(tls);
    MUTATOR_MAP
        .lock()
        .unwrap()
        .insert(key, MutatorRegistration::new(mutator as *mut _, thread));
}

/// Remove a mutator from the binding registries.
pub fn unregister_mutator_context(tls: VMMutatorThread) {
    let key = mutator_thread_key(tls);
    if let Some(entry) = MUTATOR_MAP.lock().unwrap().remove(&key) {
        ThreadRegistry::global().unregister(entry.thread.id());
    }
}

fn with_mutator_registration<F, R>(tls: VMMutatorThread, f: F) -> Option<R>
where
    F: FnOnce(&MutatorRegistration) -> R,
{
    let key = mutator_thread_key(tls);
    MUTATOR_MAP.lock().unwrap().get(&key).map(f)
}

fn visit_all_mutators<F>(mut visitor: F)
where
    F: FnMut(&'static mut mmtk::Mutator<RustVM>),
{
    let registrations: Vec<_> = {
        let map = MUTATOR_MAP.lock().unwrap();
        map.values().map(|entry| entry.mutator).collect()
    };

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
    let threads: Vec<_> = {
        let map = MUTATOR_MAP.lock().unwrap();
        map.values().map(|entry| entry.thread.clone()).collect()
    };

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
    fn stop_all_mutators<F>(_tls: VMWorkerThread, mut mutator_visitor: F)
    where
        F: FnMut(&'static mut mmtk::Mutator<RustVM>),
    {
        visit_all_threads(|thread| thread.request_safepoint());
        visit_all_threads(|thread| thread.wait_until_parked());
        visit_all_mutators(|mutator| mutator_visitor(mutator));
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        visit_all_threads(|thread| thread.clear_safepoint());
    }

    fn block_for_gc(tls: VMMutatorThread) {
        if let Some(thread) = with_mutator_registration(tls, |entry| entry.thread.clone()) {
            thread.request_safepoint();
            thread.wait_until_parked();
        }
    }

    fn spawn_gc_thread(_tls: VMThread, ctx: GCThreadContext<RustVM>) {
        match ctx {
            GCThreadContext::Worker(worker) => {
                let mmtk = {
                    let manager = FUGC_PLAN_MANAGER.lock().unwrap();
                    manager.mmtk()
                };

                std::thread::Builder::new()
                    .name("mmtk-gc-worker".to_string())
                    .spawn(move || {
                        let worker_tls = VMWorkerThread(VMThread::UNINITIALIZED);
                        worker.run(worker_tls, mmtk);
                    })
                    .expect("failed to spawn GC worker thread");
            }
        }
    }
}

#[derive(Default)]
pub struct RustReferenceGlue;

impl ReferenceGlue<RustVM> for RustReferenceGlue {
    type FinalizableType = crate::weak::WeakRefHeader;

    fn set_referent(reference: mmtk::util::ObjectReference, referent: mmtk::util::ObjectReference) {
        REFERENT_MAP.lock().unwrap().insert(reference, referent);
    }

    fn get_referent(object: mmtk::util::ObjectReference) -> Option<mmtk::util::ObjectReference> {
        REFERENT_MAP.lock().unwrap().get(&object).cloned()
    }

    fn clear_referent(object: mmtk::util::ObjectReference) {
        REFERENT_MAP.lock().unwrap().remove(&object);
    }

    fn enqueue_references(references: &[mmtk::util::ObjectReference], _tls: VMWorkerThread) {
        if references.is_empty() {
            return;
        }

        let drained: Vec<_> = {
            let mut referents = REFERENT_MAP.lock().unwrap();
            references
                .iter()
                .map(|reference| (*reference, referents.remove(reference)))
                .collect()
        };

        FINALIZATION_QUEUE.lock().unwrap().extend(drained);
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
        MUTATOR_MAP.lock().unwrap().contains_key(&key)
    }

    fn mutator(
        tls: mmtk::util::opaque_pointer::VMMutatorThread,
    ) -> &'static mut mmtk::plan::Mutator<RustVM> {
        with_mutator_registration(tls, |entry| unsafe { entry.as_mutator() })
            .expect("mutator not registered")
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut mmtk::plan::Mutator<RustVM>> + 'a> {
        let mutators: Vec<_> = {
            let map = MUTATOR_MAP.lock().unwrap();
            map.values().map(|entry| entry.mutator).collect()
        };

        Box::new(mutators.into_iter().map(|ptr| unsafe { &mut *ptr }))
    }

    fn number_of_mutators() -> usize {
        MUTATOR_MAP.lock().unwrap().len()
    }
}

/// Initialize the FUGC plan manager with an MMTk instance.
/// This should be called after MMTk initialization.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::{initialize_fugc_plan, RustVM};
/// use fugrip::plan::create_fugc_mmtk_options;
/// use mmtk::MMTK;
/// use std::sync::Arc;
///
/// // This example shows the intended initialization pattern
/// // (actual MMTk initialization is more complex and requires VM integration)
///
/// // Get FUGC-optimized options
/// let options = Arc::new(create_fugc_mmtk_options());
///
/// // Note: Actual MMTK initialization would use mmtk::memory_manager::bind_mutator
/// // and other MMTk APIs. This is a simplified example showing the configuration.
/// println!("FUGC plan configured with optimized settings");
///
/// // In a real implementation, initialize_fugc_plan would be called with the
/// // properly initialized MMTk instance
/// ```
pub fn initialize_fugc_plan(mmtk: &'static mmtk::MMTK<RustVM>) {
    FUGC_PLAN_MANAGER.lock().unwrap().initialize(mmtk);
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
    FUGC_PLAN_MANAGER.lock().unwrap().alloc_info(size, align)
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
    FUGC_PLAN_MANAGER.lock().unwrap().post_alloc(obj, bytes);
}

/// Drain the queue of references that were enqueued for finalization by MMTk.
/// This allows the VM to process weak references or other finalizable objects
/// after a collection cycle completes.
pub fn take_enqueued_references() -> Vec<(
    mmtk::util::ObjectReference,
    Option<mmtk::util::ObjectReference>,
)> {
    FINALIZATION_QUEUE.lock().unwrap().drain(..).collect()
}

/// Handle write barrier with FUGC optimizations
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
    FUGC_PLAN_MANAGER
        .lock()
        .unwrap()
        .handle_write_barrier(src, slot, target);
}

/// Trigger garbage collection with FUGC optimizations
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_gc;
///
/// // Trigger garbage collection
/// fugc_gc();
/// // This initiates concurrent marking and collection with FUGC optimizations
/// ```
pub fn fugc_gc() {
    FUGC_PLAN_MANAGER.lock().unwrap().gc();
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
    FUGC_PLAN_MANAGER.lock().unwrap().get_fugc_stats()
}

/// Get the current FUGC collection phase
pub fn fugc_get_phase() -> crate::fugc_coordinator::FugcPhase {
    FUGC_PLAN_MANAGER.lock().unwrap().fugc_phase()
}

/// Check if FUGC collection is currently in progress
pub fn fugc_is_collecting() -> bool {
    FUGC_PLAN_MANAGER.lock().unwrap().is_fugc_collecting()
}

/// Get FUGC cycle statistics
pub fn fugc_get_cycle_stats() -> crate::fugc_coordinator::FugcCycleStats {
    FUGC_PLAN_MANAGER.lock().unwrap().get_fugc_cycle_stats()
}
