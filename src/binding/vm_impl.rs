//! VM binding trait implementations for MMTk integration

use arc_swap::ArcSwap;
use dashmap::DashMap;
use mmtk::util::{
    opaque_pointer::{VMMutatorThread, VMThread, VMWorkerThread},
    ObjectReference,
};
use mmtk::vm::slot::{SimpleSlot, UnimplementedMemorySlice};
use mmtk::vm::{Collection, GCThreadContext, ReferenceGlue, VMBinding};
use std::sync::{Arc, OnceLock};

use super::{mutator::*, FUGC_PLAN_MANAGER};
use crate::plan::FugcPlanManager;

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

static REFERENT_MAP: OnceLock<DashMap<ObjectReference, ObjectReference>> = OnceLock::new();

type FinalizationQueue = crossbeam::queue::SegQueue<(
    ObjectReference,
    Option<ObjectReference>,
)>;
static FINALIZATION_QUEUE: OnceLock<FinalizationQueue> = OnceLock::new();

#[derive(Default)]
pub struct RustReferenceGlue;

impl ReferenceGlue<RustVM> for RustReferenceGlue {
    type FinalizableType = crate::weak::WeakRefHeader;

    fn set_referent(reference: ObjectReference, referent: ObjectReference) {
        REFERENT_MAP
            .get_or_init(DashMap::new)
            .insert(reference, referent);
    }

    fn get_referent(object: ObjectReference) -> Option<ObjectReference> {
        REFERENT_MAP
            .get_or_init(DashMap::new)
            .get(&object)
            .map(|v| *v.value())
    }

    fn clear_referent(object: ObjectReference) {
        let _ = REFERENT_MAP.get_or_init(DashMap::new).remove(&object);
    }

    fn enqueue_references(references: &[ObjectReference], _tls: VMWorkerThread) {
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

    fn mutator(
        tls: VMMutatorThread,
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

/// Drain the queue of references that were enqueued for finalization by MMTk.
/// This allows the VM to process weak references or other finalizable objects
/// after a collection cycle completes.
pub fn take_enqueued_references() -> Vec<(
    ObjectReference,
    Option<ObjectReference>,
)> {
    let queue = FINALIZATION_QUEUE.get_or_init(|| crossbeam::queue::SegQueue::new());
    let mut result = Vec::new();
    while let Some(item) = queue.pop() {
        result.push(item);
    }
    result
}