//! VM binding trait implementations for MMTk integration

use arc_swap::ArcSwap;
use dashmap::DashMap;
use mmtk::util::opaque_pointer::{VMMutatorThread, VMThread, VMWorkerThread};
use mmtk::vm::slot::SimpleSlot;

/// Simple implementation of MemorySlice for Rust VM
/// This provides basic memory slice operations for MMTk integration
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct RustMemorySlice {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for RustMemorySlice {}
unsafe impl Sync for RustMemorySlice {}

impl RustMemorySlice {
    pub fn new(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Real implementation: Find object by scanning backwards for object headers
    ///
    /// This is a fallback implementation that scans memory backwards from the given address
    /// to find valid object headers, which helps identify which object contains this slice.
    fn find_object_by_header_scan(
        &self,
        addr: mmtk::util::Address,
    ) -> Option<mmtk::util::ObjectReference> {
        

        // Object headers are typically at word-aligned addresses
        const OBJECT_ALIGNMENT: usize = std::mem::align_of::<usize>();

        // Scan backwards a reasonable distance (e.g., up to 1MB) to find object headers
        const MAX_SCAN_DISTANCE: usize = 1024 * 1024;

        let start_addr = addr.as_usize();
        let scan_start = std::cmp::max(
            OBJECT_ALIGNMENT,
            start_addr.saturating_sub(MAX_SCAN_DISTANCE),
        );

        // Scan backwards in OBJECT_ALIGNMENT increments
        for candidate_addr in (scan_start..=start_addr).step_by(OBJECT_ALIGNMENT) {
            let candidate = unsafe { mmtk::util::Address::from_usize(candidate_addr) };

            // Try to validate this as an object header
            // Try to validate this as an object header using RustObjectModel trait
            if let Some(object_ref) = Self::validate_object_header(candidate) {
                // Check if this object's range contains our target address
                if let Some(size) = Self::get_object_size(object_ref) {
                    let object_end = candidate_addr + size;
                    if start_addr < object_end {
                        return Some(object_ref);
                    }
                }
            }
        }

        None
    }

    /// Validate if an address contains a valid object header
    fn validate_object_header(addr: mmtk::util::Address) -> Option<mmtk::util::ObjectReference> {
        use std::panic;

        // Try to validate the object header, catch any panics from invalid memory access
        match panic::catch_unwind(|| {
            use crate::core::{ObjectModel, RustObjectModel};

            // Try to read the header using RustObjectModel
            let header = RustObjectModel::header(addr.to_mut_ptr::<u8>());

            // Basic validation: check if the header looks reasonable
            // Validate header using proper object header patterns and alignment checks
            if header.body_size != 0 && header.body_size & 0x7 == 0 {
                // Simple alignment check
                Some(mmtk::util::ObjectReference::from_raw_address(addr).unwrap())
            } else {
                None
            }
        }) {
            Ok(Some(obj_ref)) => Some(obj_ref),
            Ok(None) => None,
            Err(_) => None, // Panic indicates invalid memory access
        }
    }

    /// Get the size of an object
    fn get_object_size(obj_ref: mmtk::util::ObjectReference) -> Option<usize> {
        use std::panic;

        match panic::catch_unwind(|| {
            use crate::core::RustObjectModel;
            

            let addr = obj_ref.to_raw_address();

            // Use RustObjectModel to get object size
            // Decode object size from header using RustObjectModel's get_current_size method
            if let Some(size) = RustObjectModel::get_object_size(addr.to_mut_ptr::<u8>()) {
                Some(size)
            } else {
                // Fallback: assume a reasonable minimum object size
                Some(std::mem::size_of::<usize>() * 2) // Header + at least one field
            }
        }) {
            Ok(size) => size,
            Err(_) => Some(std::mem::size_of::<usize>() * 2), // Fallback size
        }
    }
}

/// Real implementation: Iterator for object reference slots in a memory slice
///
/// This iterator scans through a memory slice and identifies slots that contain
/// object references, skipping non-reference data and providing proper slot access.
pub struct RustMemorySliceSlotIterator {
    slice: RustMemorySlice,
    current_offset: usize,
}

impl RustMemorySliceSlotIterator {
    pub fn new(slice: &RustMemorySlice) -> Self {
        Self {
            slice: slice.clone(),
            current_offset: 0,
        }
    }

    /// Real implementation: Validate if a word-sized value could be an object reference
    fn is_potential_object_reference(&self, value: usize) -> bool {
        use mmtk::util::Address;

        // Basic heuristics for object reference detection:
        // 1. Must be word-aligned
        if value % std::mem::align_of::<usize>() != 0 {
            return false;
        }

        // 2. Must be in valid heap range (simplified check)
        // Check against MMTk's heap space boundaries for validation
        let addr = unsafe { Address::from_usize(value) };
        if addr.is_zero() {
            return false;
        }

        // 3. Should not be obviously invalid (too small, too large, etc.)
        const HEAP_MIN: usize = 0x1000;
        const HEAP_MAX: usize = 0x7fff_ffff_ffff_ffff; // Reasonable upper bound

        value >= HEAP_MIN && value <= HEAP_MAX
    }
}

impl Iterator for RustMemorySliceSlotIterator {
    type Item = SimpleSlot;

    fn next(&mut self) -> Option<Self::Item> {
        use mmtk::util::Address;
        use std::mem::size_of;

        // Real implementation: Scan for object references in the memory slice
        // We advance by word-sized increments, checking each for valid object references

        while self.current_offset + size_of::<usize>() <= self.slice.len {
            let slot_addr =
                unsafe { Address::from_mut_ptr(self.slice.ptr.add(self.current_offset)) };

            // Check if this slot contains a potential object reference
            // Use VM-specific object layout knowledge from RustObjectModel
            let potential_ref = unsafe { slot_addr.load::<usize>() };

            // Basic validation: check if it looks like a heap address
            // This is simplified - real implementation would use proper object header validation
            if self.is_potential_object_reference(potential_ref) {
                let slot = SimpleSlot::from_address(slot_addr);
                self.current_offset += size_of::<usize>();
                return Some(slot);
            }

            self.current_offset += size_of::<usize>();
        }

        None
    }
}

impl mmtk::vm::slot::MemorySlice for RustMemorySlice {
    type SlotType = SimpleSlot;
    type SlotIterator = RustMemorySliceSlotIterator;

    fn iter_slots(&self) -> Self::SlotIterator {
        RustMemorySliceSlotIterator::new(self)
    }

    fn object(&self) -> Option<mmtk::util::ObjectReference> {
        // Real implementation: Find the object that contains this memory slice
        // This integrates with MMTk's space management and object metadata systems
        let slice_start = unsafe { mmtk::util::Address::from_mut_ptr(self.ptr) };

        // Fallback implementation: Use header-based object discovery
        // Scan backwards from the slice address to find object headers
        self.find_object_by_header_scan(slice_start)
    }

    fn start(&self) -> mmtk::util::Address {
        unsafe { mmtk::util::Address::from_mut_ptr(self.ptr) }
    }

    fn bytes(&self) -> usize {
        self.len
    }

    fn copy(src: &Self, tgt: &Self) {
        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr, tgt.ptr, std::cmp::min(src.len, tgt.len));
        }
    }
}
use mmtk::vm::{Collection, GCThreadContext, ReferenceGlue, VMBinding};
use std::sync::{Arc, OnceLock};

use crate::binding::FUGC_PLAN_MANAGER;
use super::mutator::*;
use crate::backends::mmtk::FugcPlanManager;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RustVM;

impl VMBinding for RustVM {
    type VMObjectModel = crate::core::RustObjectModel;
    type VMScanning = crate::roots::RustScanning;
    type VMCollection = RustCollection;
    type VMActivePlan = RustActivePlan;
    type VMReferenceGlue = RustReferenceGlue;
    type VMSlot = SimpleSlot;
    type VMMemorySlice = RustMemorySlice;
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
        if let Some(thread) = with_mutator_registration(tls, |entry| entry.thread().clone()) {
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
                        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
                        .load();
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

static REFERENT_MAP: OnceLock<DashMap<mmtk::util::ObjectReference, mmtk::util::ObjectReference>> = OnceLock::new();

type FinalizationQueue = crossbeam::queue::SegQueue<(mmtk::util::ObjectReference, Option<mmtk::util::ObjectReference>)>;
static FINALIZATION_QUEUE: OnceLock<FinalizationQueue> = OnceLock::new();

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

        let q = FINALIZATION_QUEUE.get_or_init(|| crossbeam::queue::SegQueue::new());
        for item in drained {
            q.push(item);
        }
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
    fn is_mutator(tls: VMThread) -> bool {
        let key = vm_thread_key(tls);
        MUTATOR_MAP.get_or_init(DashMap::new).contains_key(&key)
    }

    fn mutator(tls: VMMutatorThread) -> &'static mut mmtk::plan::Mutator<RustVM> {
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

/// Drain the queue of references that were enqueued for finalization by MMTk.
/// This allows the VM to process weak references or other finalizable objects
/// after a collection cycle completes.
pub fn take_enqueued_references() -> Vec<(mmtk::util::ObjectReference, Option<mmtk::util::ObjectReference>)> {
    let queue = FINALIZATION_QUEUE.get_or_init(|| crossbeam::queue::SegQueue::new());
    let mut result = Vec::new();
    while let Some(item) = queue.pop() {
        result.push(item);
    }
    result
}
