//! Finalizer queue and callback processing
//!
//! This module provides Java-style finalization with epoch-based memory management,
//! allowing objects to register cleanup callbacks that run after they are garbage collected.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crate::compat::{Address, ObjectReference};
use flume::{Receiver, Sender};
use rayon;

/// A finalizer callback that will be invoked when an object is garbage collected
pub type FinalizerCallback = Box<dyn FnOnce() + Send + Sync + 'static>;

/// A node in the finalizer linked list
struct FinalizerNode {
    /// The finalizer callback to execute
    callback: FinalizerCallback,
}

impl FinalizerNode {
    /// Create a new finalizer node
    fn new(callback: FinalizerCallback) -> Self {
        Self { callback }
    }

    /// Execute the finalizer and consume the node
    fn execute(mut self) {
        (self.callback)();
    }
}

impl std::fmt::Debug for FinalizerNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FinalizerNode")
    }
}

/// Thread-safe finalizer queue for garbage collected objects
#[derive(Debug)]
pub struct FinalizerQueue {
    /// Name for debugging and statistics
    name: String,
    /// Pending finalizers
    pending_finalizers: parking_lot::Mutex<Vec<FinalizerNode>>,
    /// Total objects registered for finalization
    total_registered: AtomicUsize,
    /// Total finalizations processed
    total_processed: AtomicUsize,
    /// Shutdown flag for processing thread
    shutdown: Arc<AtomicBool>,
    /// Work notification channel
    work_notify_sender: Sender<()>,
    work_notify_receiver: Arc<Receiver<()>>,
}

impl FinalizerQueue {
    /// Get the name of this finalizer queue
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a new finalizer queue with the given name
    pub fn new(name: &str) -> Self {
        let (work_notify_sender, work_notify_receiver) = flume::bounded(1);

        Self {
            name: name.to_string(),
            pending_finalizers: parking_lot::Mutex::new(Vec::new()),
            total_registered: AtomicUsize::new(0),
            total_processed: AtomicUsize::new(0),
            shutdown: Arc::new(AtomicBool::new(false)),
            work_notify_sender,
            work_notify_receiver: Arc::new(work_notify_receiver),
        }
    }

    /// Register a finalizer for an object
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fugrip::memory_management::finalizers::FinalizerQueue;
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let queue = FinalizerQueue::new("test_queue");
    /// let obj = ObjectReference::from_raw_address(Address::ZERO).unwrap();
    /// queue.register_for_finalization(obj, Box::new(|| {
    ///     println!("Object finalized!");
    /// }));
    /// ```
    pub fn register_for_finalization(
        &self,
        _object: ObjectReference,
        finalizer: FinalizerCallback,
    ) {
        let node = FinalizerNode::new(finalizer);
        let mut pending = self.pending_finalizers.lock();
        pending.push(node);
        drop(pending);

        self.total_registered.fetch_add(1, Ordering::Relaxed);

        // Notify processing thread about new work
        let _ = self.work_notify_sender.try_send(());
    }

    /// Register a finalizer callback directly (legacy method)
    ///
    /// This method is kept for backwards compatibility but doesn't track objects.
    pub fn register_finalizer(&self, callback: FinalizerCallback) {
        let node = FinalizerNode::new(callback);
        let mut pending = self.pending_finalizers.lock();
        pending.push(node);
        drop(pending);

        self.total_registered.fetch_add(1, Ordering::Relaxed);

        // Notify processing thread about new work
        let _ = self.work_notify_sender.try_send(());
    }

    /// Process pending finalizations
    ///
    /// Returns the number of finalizers processed
    pub fn process_pending_finalizations(&self) -> usize {
        let mut pending = self.pending_finalizers.lock();
        let to_process = std::mem::take(&mut *pending);
        drop(pending);

        let processed_count = to_process.len();
        for node in to_process {
            node.execute();
        }

        if processed_count > 0 {
            self.total_processed
                .fetch_add(processed_count, Ordering::Relaxed);
        }

        processed_count
    }

    /// Start finalizer processing with Rayon
    pub fn start_rayon_processor(self: Arc<Self>) {
        let shutdown = Arc::clone(&self.shutdown);
        let work_receiver = self.work_notify_receiver.clone();

        rayon::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                // Wait for work notification with timeout
                match work_receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(()) | Err(flume::RecvTimeoutError::Timeout) => {
                        // Got work notification or timeout - process pending work
                        let processed_count = self.process_pending_finalizations();

                        // If we processed something, yield to let other threads run
                        if processed_count > 0 {
                            std::thread::yield_now();
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

    /// Get statistics for this finalizer queue
    pub fn get_stats(&self) -> FinalizerQueueStats {
        let pending = self.pending_finalizers.lock();
        let currently_pending = pending.len();
        drop(pending);

        FinalizerQueueStats {
            name: self.name.clone(),
            total_registered: self.total_registered.load(Ordering::Relaxed),
            total_processed: self.total_processed.load(Ordering::Relaxed),
            currently_pending,
        }
    }

    /// Shutdown the finalizer queue
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        // Rayon manages thread cleanup automatically
    }
}

impl Clone for FinalizerQueue {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            pending_finalizers: parking_lot::Mutex::new(Vec::new()),
            total_registered: AtomicUsize::new(self.total_registered.load(Ordering::Relaxed)),
            total_processed: AtomicUsize::new(self.total_processed.load(Ordering::Relaxed)),
            shutdown: Arc::clone(&self.shutdown),
            work_notify_sender: self.work_notify_sender.clone(),
            work_notify_receiver: self.work_notify_receiver.clone(),
        }
    }
}

impl Drop for FinalizerQueue {
    fn drop(&mut self) {
        self.shutdown();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_finalizer_queue_creation() {
        let queue = FinalizerQueue::new("test_queue");
        assert_eq!(queue.name, "test_queue");
    }

    #[test]
    fn test_finalizer_registration() {
        let queue = FinalizerQueue::new("test_queue");
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);

        queue.register_for_finalization(
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0) }).unwrap(),
            Box::new(move || {
                called_clone.store(true, Ordering::Relaxed);
            }),
        );

        assert_eq!(queue.total_registered.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_finalizer_processing() {
        let queue = FinalizerQueue::new("test_queue");
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);

        queue.register_for_finalization(
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0) }).unwrap(),
            Box::new(move || {
                called_clone.store(true, Ordering::Relaxed);
            }),
        );

        let processed = queue.process_pending_finalizations();
        assert_eq!(processed, 1);
        assert!(called.load(Ordering::Relaxed));
        assert_eq!(queue.total_processed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_finalizer_stats() {
        let queue = FinalizerQueue::new("test_queue");

        // Initially empty
        let stats = queue.get_stats();
        assert_eq!(stats.total_registered, 0);
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.currently_pending, 0);

        // Register some finalizers
        queue.register_for_finalization(
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0) }).unwrap(),
            Box::new(|| {}),
        );
        queue.register_for_finalization(
            ObjectReference::from_raw_address(unsafe { Address::from_usize(0) }).unwrap(),
            Box::new(|| {}),
        );

        let stats = queue.get_stats();
        assert_eq!(stats.total_registered, 2);
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.currently_pending, 2);
    }

    #[test]
    fn test_empty_queue_processing() {
        let queue = FinalizerQueue::new("test_queue");
        let processed = queue.process_pending_finalizations();
        assert_eq!(processed, 0);
    }
}
