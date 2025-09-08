// Threading interfaces and production adapters.
//
// This module provides a ThreadingProvider trait that exposes the
// operations needed by the collector for thread registration, handshake
// coordination and worker suspension helpers. A ProductionThreadingProvider
// delegates to the existing runtime structures (global COLLECTOR fields)
// to preserve existing behavior while allowing an interface to be used by
// the rest of the codebase.
use crate::collector_phases::ThreadRegistration;
use crate::memory;

/// Threading provider trait - encapsulates thread lifecycle and coordination.
pub trait ThreadingProvider: Send + Sync + 'static {
    fn register_mutator_thread(&self);
    fn unregister_mutator_thread(&self);

    fn get_active_mutator_count(&self) -> usize;

    fn request_handshake(&self);
    fn is_handshake_requested(&self) -> bool;
    fn acknowledge_handshake(&self);

    fn register_thread_for_gc(&self, stack_bounds: (usize, usize)) -> Result<(), &'static str>;
    fn unregister_thread_from_gc(&self);
    fn update_thread_stack_pointer(&self);
    fn get_current_thread_stack_bounds(&self) -> (usize, usize);

    fn worker_suspended(&self);

    /// Iterate over registered thread registrations holding the lock for the duration
    /// of the callback. This provides a safe abstraction for callers that need to
    /// examine registered thread data without reaching into the ThreadCoordinator.
    fn for_each_registered_thread<F>(&self, f: F)
    where
        F: FnMut(&ThreadRegistration);
}

/// Production implementation delegating to global COLLECTOR internals.
#[derive(Debug, Clone, Copy)]
pub struct ProductionThreadingProvider;

impl ThreadingProvider for ProductionThreadingProvider {
    fn register_mutator_thread(&self) {
        memory::COLLECTOR.register_mutator_thread();
    }

    fn unregister_mutator_thread(&self) {
        memory::COLLECTOR.unregister_mutator_thread();
    }

    fn get_active_mutator_count(&self) -> usize {
        memory::COLLECTOR
            .thread_coordinator
            .get_active_mutator_count()
    }

    fn request_handshake(&self) {
        memory::COLLECTOR.request_handshake();
    }

    fn is_handshake_requested(&self) -> bool {
        memory::COLLECTOR.is_handshake_requested()
    }

    fn acknowledge_handshake(&self) {
        memory::COLLECTOR.acknowledge_handshake();
    }

    fn register_thread_for_gc(&self, stack_bounds: (usize, usize)) -> Result<(), &'static str> {
        memory::COLLECTOR.register_thread_for_gc(stack_bounds)
    }

    fn unregister_thread_from_gc(&self) {
        memory::COLLECTOR.unregister_thread_from_gc();
    }

    fn update_thread_stack_pointer(&self) {
        memory::COLLECTOR.update_thread_stack_pointer();
    }

    fn get_current_thread_stack_bounds(&self) -> (usize, usize) {
        memory::COLLECTOR.get_current_thread_stack_bounds()
    }

    fn worker_suspended(&self) {
        memory::COLLECTOR.suspension_manager.worker_suspended();
    }

    fn for_each_registered_thread<F>(&self, mut f: F)
    where
        F: FnMut(&ThreadRegistration),
    {
        if let Ok(threads) = memory::COLLECTOR
            .thread_coordinator
            .registered_threads
            .lock()
        {
            for reg in threads.iter() {
                f(reg);
            }
        }
    }
}

/// Global provider instance used by collector call sites.
pub static THREADING_PROVIDER: ProductionThreadingProvider = ProductionThreadingProvider;
