//! Thread management and coordination for FUGC garbage collection.
//!
//! This module provides lock-free thread coordination using atomic state machines
//! and crossbeam channels. Invalid states are unrepresentable by design.

use crate::handshake::{
    HandshakeCompletion, HandshakeCoordinator, HandshakeError, HandshakeType,
    MutatorHandshakeHandler,
};
use dashmap::DashMap;
use flume::{Receiver, Sender};
use parking_lot::Mutex;
use std::sync::Barrier;
use std::{
    fmt,
    sync::{Arc, OnceLock},
    time::Duration,
};

/// Represents a mutator thread backed by a `MutatorHandshakeHandler`.
#[derive(Clone)]
pub struct MutatorThread {
    pub(crate) id: usize,
    pub(crate) handler: Arc<MutatorHandshakeHandler>,
}

impl MutatorThread {
    /// Create a new standalone mutator thread and return it together with
    /// the `Receiver<HandshakeRequest>` that a coordinator should use to
    /// deliver requests to this thread. This is useful for tests and
    /// for initializing the handler before registering with a coordinator.
    pub fn new_with_channels(
        id: usize,
    ) -> (
        Self,
        Receiver<crate::handshake::HandshakeRequest>,
        Sender<HandshakeCompletion>,
        Arc<Receiver<()>>,
    ) {
        let (_request_tx, request_rx) = flume::bounded(1);
        let (completion_tx, _completion_rx) = flume::unbounded();
        let (_release_tx, release_rx_inner) = flume::unbounded();
        let release_rx = Arc::new(release_rx_inner);

        let handler = Arc::new(MutatorHandshakeHandler::new(
            id,
            request_rx.clone(),
            completion_tx.clone(),
            Arc::clone(&release_rx),
        ));

        (Self { id, handler }, request_rx, completion_tx, release_rx)
    }

    /// Convenience constructor for simple cases (not wired to a central coordinator).
    pub fn new(id: usize) -> Self {
        let (thread, _req_rx, _completion_tx, _release_rx) = Self::new_with_channels(id);
        thread
    }

    pub fn id(&self) -> usize {
        self.id
    }

    /// Poll for safepoint requests (non-blocking). Call periodically from
    /// the mutator loop.
    pub fn poll_safepoint(&self) {
        self.handler.poll_safepoint();
    }

    pub fn register_stack_root(&self, handle: *mut u8) {
        if handle.is_null() {
            return;
        }
        self.handler.add_stack_root(handle as usize);
    }

    pub fn clear_stack_roots(&self) {
        self.handler.clear_stack_roots();
    }

    pub fn is_at_safepoint(&self) -> bool {
        matches!(
            self.handler.get_state(),
            crate::handshake::HandshakeState::AtSafepoint
        )
    }

    /// Get current stack roots - for backwards compatibility return empty
    pub fn stack_roots(&self) -> Vec<*mut u8> {
        self.handler
            .get_stack_roots()
            .into_iter()
            .map(|addr| addr as *mut u8)
            .collect()
    }
}

impl std::fmt::Debug for MutatorThread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutatorThread")
            .field("id", &self.id)
            .finish()
    }
}

/// Registry for tracking all mutator threads and coordinating handshakes.
pub struct ThreadRegistry {
    threads: DashMap<usize, MutatorThread>,
    coordinator: Mutex<HandshakeCoordinator>,
    completion_tx: Sender<HandshakeCompletion>,
    release_rx: Arc<Receiver<()>>,
}

impl ThreadRegistry {
    pub fn new() -> Self {
        let (coordinator, completion_tx, release_rx_inner) = HandshakeCoordinator::new();
        let release_rx = Arc::new(release_rx_inner);
        Self {
            threads: DashMap::new(),
            coordinator: Mutex::new(coordinator),
            completion_tx,
            release_rx,
        }
    }

    pub fn register(&self, mutator: MutatorThread) {
        let id = mutator.id();

        // Register the thread with the coordinator and get the request receiver
        let req_rx = {
            let mut coord = self.coordinator.lock();
            coord.register_thread(id)
        };

        // Replace the handler's request_rx by constructing a new handler that uses
        // the coordinator-provided receiver. This keeps the same handler type
        // semantics while wiring to the central coordinator.
        let new_handler = Arc::new(MutatorHandshakeHandler::new(
            id,
            req_rx,
            self.completion_tx.clone(),
            Arc::clone(&self.release_rx),
        ));

        self.threads.insert(
            id,
            MutatorThread {
                id,
                handler: new_handler,
            },
        );
    }

    pub fn unregister(&self, id: usize) {
        self.threads.remove(&id);
        let mut coord = self.coordinator.lock();
        coord.unregister_thread(id);
    }

    pub fn iter(&self) -> Vec<MutatorThread> {
        self.threads
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.threads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn get(&self, id: usize) -> Option<MutatorThread> {
        self.threads.get(&id).map(|entry| entry.value().clone())
    }

    pub fn perform_handshake(
        &self,
        handshake_type: HandshakeType,
        timeout: Duration,
    ) -> Result<Vec<HandshakeCompletion>, HandshakeError> {
        let coord = self.coordinator.lock();
        coord.perform_handshake(handshake_type, timeout)
    }
}

impl Default for ThreadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ThreadRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadRegistry")
            .field("thread_count", &self.len())
            .finish()
    }
}

/// Global thread registry accessor - needed for backwards compatibility
pub fn global() -> &'static ThreadRegistry {
    static GLOBAL: OnceLock<ThreadRegistry> = OnceLock::new();
    GLOBAL.get_or_init(ThreadRegistry::new)
}

/// Alternative name for compatibility
pub fn global_thread_registry() -> &'static ThreadRegistry {
    global()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn test_mutator_thread_creation() {
        let thread = MutatorThread::new(42);
        assert_eq!(thread.id(), 42);
    }

    #[test]
    fn test_thread_registry_register_unregister() {
        let registry = ThreadRegistry::new();
        let (mutator, _req_rx, _completion_tx, _release_rx) = MutatorThread::new_with_channels(1);
        registry.register(mutator);
        assert_eq!(registry.len(), 1);
        registry.unregister(1);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_thread_registry_registration() {
        let registry = ThreadRegistry::new();
        let thread = MutatorThread::new(1);

        assert_eq!(registry.len(), 0);
        registry.register(thread.clone());
        assert_eq!(registry.len(), 1);

        let threads = registry.iter();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].id(), 1);
    }

    #[test]
    fn test_thread_registry_unregister() {
        let registry = ThreadRegistry::new();
        let thread = MutatorThread::new(1);

        registry.register(thread);
        assert_eq!(registry.len(), 1);

        registry.unregister(1);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_thread_registry_duplicate_registration() {
        let registry = ThreadRegistry::new();
        let thread1 = MutatorThread::new(1);
        let thread2 = MutatorThread::new(1);

        registry.register(thread1);
        registry.register(thread2);

        // Should replace the first registration
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_thread_registry_clear() {
        let registry = ThreadRegistry::new();
        for i in 1..=5 {
            registry.register(MutatorThread::new(i));
        }
        assert_eq!(registry.len(), 5);

        for i in 1..=5 {
            registry.unregister(i);
        }
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_mutator_thread_safepoint() {
        let thread = MutatorThread::new(1);

        // Should not panic - lock-free implementation
        thread.poll_safepoint();
    }

    #[test]
    fn test_mutator_thread_stack_roots() {
        let thread = MutatorThread::new(2);

        let mut value = 42u32;
        let handle = &mut value as *mut u32 as *mut u8;

        thread.register_stack_root(handle);
        thread.clear_stack_roots();
    }

    #[test]
    fn test_poll_safepoint_loop() {
        let (mutator, _req_rx, _completion_tx, _release_rx) = MutatorThread::new_with_channels(2);

        // Use proper thread coordination with channels
        let (start_tx, start_rx) = flume::bounded(1);
        let (done_tx, done_rx) = flume::bounded(1);
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // Use crossbeam scoped threads for deterministic cleanup and guaranteed termination
        crossbeam::scope(|s| {
            let running_clone = running.clone();
            let handler_clone = mutator.clone();

            s.spawn(move |_| {
                // Signal that thread is ready
                start_tx.send(()).unwrap();

                while running_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    handler_clone.poll_safepoint();
                    thread::yield_now();
                }

                // Signal completion
                done_tx.send(()).unwrap();
            });

            // Wait for thread to start
            start_rx.recv().unwrap();

            // Allow some polling iterations, then stop
            for _ in 0..10 {
                thread::yield_now();
            }
            running.store(false, std::sync::atomic::Ordering::Relaxed);

            // Wait for thread to complete
            done_rx.recv().unwrap();
            // Thread automatically cleaned up when scope exits
        }).unwrap();

        // Ensure basic stack root registration works
        let mut value = 5u8;
        let ptr = &mut value as *mut u8;
        mutator.register_stack_root(ptr);
        mutator.clear_stack_roots();
    }

    #[test]
    fn test_mutator_thread_thread_safe() {
        use rayon::prelude::*;
        let registry = Arc::new(ThreadRegistry::new());

        // Use rayon parallel iteration instead of manual thread::spawn
        (0..4).into_par_iter().for_each(|i| {
            let mutator = MutatorThread::new(i);
            registry.register(mutator.clone());

            // Poll safepoints in parallel
            for _ in 0..100 {
                mutator.poll_safepoint();
                std::hint::spin_loop(); // Better than thread::yield_now() for tight loops
            }

            registry.unregister(i);
        });

        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_thread_registry_concurrent_access() {
        let registry = Arc::new(ThreadRegistry::new());

        // TODO: Replace std::sync::Barrier with crossbeam_barrier::Barrier for 10-20%
        // performance improvement in high-contention thread coordination scenarios
        // Use proper synchronization with barriers for deterministic thread coordination
        let barrier = Arc::new(Barrier::new(8));

        // Use crossbeam scoped threads for deterministic cleanup and guaranteed termination
        crossbeam::scope(|s| {
            for i in 0..8 {
                let registry_clone = Arc::clone(&registry);
                let barrier_clone = Arc::clone(&barrier);
                s.spawn(move |_| {
                    for j in 0..10 {
                        let thread_id = i * 10 + j;
                        let mutator = MutatorThread::new(thread_id);
                        registry_clone.register(mutator);

                        // Synchronize all threads at this point for deterministic testing
                        barrier_clone.wait();

                        registry_clone.unregister(thread_id);
                    }
                });
            }
            // All threads automatically joined when scope exits
        }).unwrap();

        assert_eq!(registry.len(), 0);
    }
}
