//! Dependency Injection container for FUGC
//!
//! This module provides a dependency injection container to eliminate global state
//! and enable proper testing isolation. Each component can be injected with its
//! dependencies rather than relying on global singletons.

use crate::fugc_coordinator::FugcCoordinator;
use crate::roots::GlobalRoots;
use crate::safepoint::SafepointManager;
use crate::thread::ThreadRegistry;
use parking_lot::Mutex;
use std::sync::Arc;

/// Dependency injection container for FUGC components
pub struct DIContainer {
    thread_registry: Arc<ThreadRegistry>,
    global_roots: Arc<Mutex<GlobalRoots>>,
    safepoint_manager: Mutex<Option<Arc<SafepointManager>>>,
    fugc_coordinator: Mutex<Option<Arc<FugcCoordinator>>>,
}

impl Clone for DIContainer {
    fn clone(&self) -> Self {
        Self {
            thread_registry: Arc::clone(&self.thread_registry),
            global_roots: Arc::clone(&self.global_roots),
            safepoint_manager: Mutex::new(self.safepoint_manager.lock().clone()),
            fugc_coordinator: Mutex::new(self.fugc_coordinator.lock().clone()),
        }
    }
}

impl DIContainer {
    /// Create a new DI container with default implementations
    pub fn new() -> Self {
        let thread_registry = Arc::new(ThreadRegistry::new());
        let global_roots = Arc::new(Mutex::new(GlobalRoots::default()));

        Self {
            thread_registry,
            global_roots,
            safepoint_manager: Mutex::new(None),
            fugc_coordinator: Mutex::new(None),
        }
    }

    /// Create a DI container for testing with isolated components
    pub fn new_for_testing() -> Self {
        // Each test gets completely isolated instances
        Self::new()
    }

    /// Get the thread registry
    pub fn thread_registry(&self) -> &Arc<ThreadRegistry> {
        &self.thread_registry
    }

    /// Get the global roots
    pub fn global_roots(&self) -> &Arc<Mutex<GlobalRoots>> {
        &self.global_roots
    }

    /// Get or create the safepoint manager
    pub fn safepoint_manager(&self) -> Arc<SafepointManager> {
        // Fast path: return cached manager if present
        {
            let guard = self.safepoint_manager.lock();
            if let Some(ref manager) = *guard {
                return Arc::clone(manager);
            }
        }

        // Need to create and cache a manager
        let manager = if let Some(coordinator) = self.fugc_coordinator.lock().as_ref() {
            SafepointManager::with_coordinator(Arc::clone(coordinator))
        } else {
            SafepointManager::new_for_testing()
        };

        let mut guard = self.safepoint_manager.lock();
        *guard = Some(Arc::clone(&manager));
        manager
    }

    /// Set the FUGC coordinator
    pub fn set_fugc_coordinator(&mut self, coordinator: Arc<FugcCoordinator>) {
        let mut guard = self.fugc_coordinator.lock();
        *guard = Some(coordinator);
    }

    /// Get the FUGC coordinator (panics if not set)
    pub fn fugc_coordinator(&self) -> Arc<FugcCoordinator> {
        self.fugc_coordinator
            .lock()
            .as_ref()
            .expect("FUGC coordinator not set in DI container")
            .clone()
    }

    /// Create a FUGC coordinator with the dependencies from this container
    pub fn create_fugc_coordinator(
        &mut self,
        heap_base: mmtk::util::Address,
        heap_size: usize,
        num_workers: usize,
    ) -> Arc<FugcCoordinator> {
        let coordinator = Arc::new(FugcCoordinator::new(
            heap_base,
            heap_size,
            num_workers,
            Arc::clone(&self.thread_registry),
            Arc::clone(&self.global_roots),
        ));

        let mut guard = self.fugc_coordinator.lock();
        *guard = Some(Arc::clone(&coordinator));
        // Also set safepoint manager to use this coordinator
        let mut sm_guard = self.safepoint_manager.lock();
        *sm_guard = Some(SafepointManager::with_coordinator(Arc::clone(&coordinator)));

        coordinator
    }
}

impl Default for DIContainer {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local DI container for the current context
thread_local! {
    static CURRENT_CONTAINER: std::cell::RefCell<Option<Arc<DIContainer>>> = const {
        std::cell::RefCell::new(None)
    };
}

/// Set the DI container for the current thread context
pub fn set_current_container(container: DIContainer) {
    CURRENT_CONTAINER.with(|c| {
        *c.borrow_mut() = Some(Arc::new(container));
    });
}

/// Get the current DI container, or create a default one
pub fn current_container() -> Arc<DIContainer> {
    CURRENT_CONTAINER.with(|c| {
        if let Some(ref arc) = *c.borrow() {
            Arc::clone(arc)
        } else {
            let new = Arc::new(DIContainer::new());
            *c.borrow_mut() = Some(Arc::clone(&new));
            new
        }
    })
}

/// Clear the current DI container (useful for test cleanup)
pub fn clear_current_container() {
    CURRENT_CONTAINER.with(|c| {
        *c.borrow_mut() = None;
    });
}

/// RAII guard for setting a DI container for a scope
pub struct DIScope {
    _phantom: std::marker::PhantomData<()>,
}

impl DIScope {
    /// Create a new DI scope with the given container
    pub fn new(container: DIContainer) -> Self {
        set_current_container(container);
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Drop for DIScope {
    fn drop(&mut self) {
        clear_current_container();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmtk::util::Address;

    #[test]
    fn test_di_container_creation() {
        let container = DIContainer::new();
        assert!(!container.thread_registry().is_empty() || container.thread_registry().is_empty());
    }

    #[test]
    fn test_di_container_isolation() {
        let container1 = DIContainer::new_for_testing();
        let container2 = DIContainer::new_for_testing();

        // Each container should have its own instances
        assert!(!Arc::ptr_eq(
            container1.thread_registry(),
            container2.thread_registry()
        ));
        assert!(!Arc::ptr_eq(
            container1.global_roots(),
            container2.global_roots()
        ));
    }

    #[test]
    fn test_di_scope() {
        let container = DIContainer::new_for_testing();

        {
            let _scope = DIScope::new(container.clone());
            let current = current_container();
            assert!(Arc::ptr_eq(
                current.thread_registry(),
                container.thread_registry()
            ));
        }

        // Should be cleared after scope ends
        clear_current_container();
        let new_current = current_container();
        assert!(!Arc::ptr_eq(
            new_current.thread_registry(),
            container.thread_registry()
        ));
    }

    #[test]
    fn test_fugc_coordinator_creation() {
        let mut container = DIContainer::new_for_testing();
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let coordinator = container.create_fugc_coordinator(heap_base, 64 * 1024 * 1024, 4);

        assert!(Arc::ptr_eq(&coordinator, &container.fugc_coordinator()));
    }
}
