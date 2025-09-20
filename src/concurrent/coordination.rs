//! Concurrent root scanning and worker coordination

use crate::compat::{Address, ObjectReference};
use arc_swap::ArcSwap;
use flume::{Receiver, Sender};
use std::sync::{Arc, atomic::AtomicUsize};

use super::{core::optimized_fetch_add, tricolor::TricolorMarking};
use crate::concurrent::ObjectColor;

/// Concurrent root scanner for parallel root enumeration during marking
pub struct ConcurrentRootScanner {
    /// Thread registry for accessing mutator threads
    thread_registry: Arc<crate::thread::ThreadRegistry>,
    /// Global roots manager
    global_roots: ArcSwap<crate::roots::GlobalRoots>,
    /// Shared tricolor marking state to update root colors
    marking: Arc<TricolorMarking>,
    /// Number of worker threads for root scanning
    num_workers: usize,
    /// Statistics
    roots_scanned: AtomicUsize,
}

impl ConcurrentRootScanner {
    pub fn new(
        thread_registry: Arc<crate::thread::ThreadRegistry>,
        global_roots: ArcSwap<crate::roots::GlobalRoots>,
        marking: Arc<TricolorMarking>,
        num_workers: usize,
    ) -> Self {
        Self {
            thread_registry,
            global_roots,
            marking,
            num_workers,
            roots_scanned: AtomicUsize::new(0),
        }
    }

    pub fn scan_global_roots(&self) {
        let roots = self.global_roots.load();
        let mut scanned = 0;
        for root_ptr in roots.iter() {
            if let Some(root_obj) = ObjectReference::from_raw_address(unsafe {
                mmtk::util::Address::from_usize(root_ptr as usize)
            }) && self.marking.get_color(root_obj) == ObjectColor::White
            {
                self.marking.set_color(root_obj, ObjectColor::Grey);
                scanned += 1;
            }
        }
        optimized_fetch_add(&self.roots_scanned, scanned);
    }

    pub fn scan_thread_roots(&self) {
        let mut scanned = 0;
        for mutator in self.thread_registry.iter() {
            for &root_ptr in mutator.stack_roots().iter() {
                if root_ptr.is_null() {
                    continue;
                }

                if let Some(root_obj) = ObjectReference::from_raw_address(unsafe {
                    Address::from_usize(root_ptr as usize)
                }) && self.marking.get_color(root_obj) == ObjectColor::White
                {
                    self.marking.set_color(root_obj, ObjectColor::Grey);
                    scanned += 1;
                }
            }
        }
        optimized_fetch_add(&self.roots_scanned, scanned);
    }

    pub fn scan_all_roots(&self) {
        self.scan_global_roots();
        self.scan_thread_roots();
    }

    pub fn start_concurrent_scanning(&self) {
        // Start background root scanning if needed
        // For now, this is a no-op since global roots are scanned synchronously
    }
}

/// Worker coordination channels
pub struct WorkerChannels {
    /// Channel for sending work to this specific worker
    work_sender: Sender<Vec<ObjectReference>>,
    /// Channel for receiving completion signals from this worker
    completion_receiver: Receiver<usize>,
    /// Channel for sending shutdown signal to this worker
    shutdown_sender: Sender<()>,
}

impl WorkerChannels {
    pub fn new(
        work_sender: Sender<Vec<ObjectReference>>,
        completion_receiver: Receiver<usize>,
        shutdown_sender: Sender<()>,
    ) -> Self {
        Self {
            work_sender,
            completion_receiver,
            shutdown_sender,
        }
    }

    pub fn work_sender(&self) -> &Sender<Vec<ObjectReference>> {
        &self.work_sender
    }

    pub fn completion_receiver(&self) -> &Receiver<usize> {
        &self.completion_receiver
    }

    pub fn send_shutdown(&self) {
        let _ = self.shutdown_sender.send(());
    }

    /// Send work to this worker
    pub fn send_work(
        &self,
        work: Vec<ObjectReference>,
    ) -> Result<(), flume::SendError<Vec<ObjectReference>>> {
        self.work_sender.send(work)
    }
}

/// Statistics for concurrent marking
#[derive(Debug, Clone)]
pub struct ConcurrentMarkingStats {
    pub work_stolen: usize,
    pub work_shared: usize,
    pub objects_allocated_black: usize,
}
