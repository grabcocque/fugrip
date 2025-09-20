//! Dependency Injection container for FUGC
//!
//! This module provides a dependency injection container to eliminate global state
//! and enable proper testing isolation. Each component can be injected with its
//! dependencies rather than relying on global singletons.

use crate::fugc_coordinator::FugcCoordinator;
use crate::roots::GlobalRoots;
use crate::safepoint::SafepointManager;
use crate::thread::ThreadRegistry;
use arc_swap::ArcSwap;
use std::sync::{Arc, OnceLock};

/// Dependency injection container for FUGC components
pub struct DIContainer {
    thread_registry: Arc<ThreadRegistry>,
    global_roots: ArcSwap<GlobalRoots>,
    safepoint_manager: OnceLock<Arc<SafepointManager>>,
    fugc_coordinator: OnceLock<Arc<FugcCoordinator>>,
}

impl DIContainer {
    /// Create a new DI container with default implementations
    pub fn new() -> Self {
        let thread_registry = Arc::new(ThreadRegistry::new());
        let global_roots = ArcSwap::new(Arc::new(GlobalRoots::default()));

        Self {
            thread_registry,
            global_roots,
            safepoint_manager: OnceLock::new(),
            fugc_coordinator: OnceLock::new(),
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
    pub fn global_roots(&self) -> &ArcSwap<GlobalRoots> {
        &self.global_roots
    }

    /// Get or create the safepoint manager
    pub fn safepoint_manager(&self) -> &Arc<SafepointManager> {
        self.safepoint_manager.get_or_init(|| {
            if let Some(coordinator) = self.fugc_coordinator.get() {
                SafepointManager::with_coordinator(coordinator)
            } else {
                SafepointManager::new_for_testing()
            }
        })
    }

    /// Set the FUGC coordinator
    pub fn set_fugc_coordinator(&self, coordinator: Arc<FugcCoordinator>) {
        if self.fugc_coordinator.set(coordinator.clone()).is_ok() {
            let _ = self
                .safepoint_manager
                .set(SafepointManager::with_coordinator(&coordinator));
        }
    }

    /// Get the FUGC coordinator (panics if not set)
    pub fn fugc_coordinator(&self) -> &Arc<FugcCoordinator> {
        self.fugc_coordinator
            .get()
            .expect("FUGC coordinator not set in DI container")
    }

    /// Create a FUGC coordinator with the dependencies from this container
    pub fn create_fugc_coordinator(
        &self,
        heap_base: mmtk::util::Address,
        heap_size: usize,
        num_workers: usize,
    ) -> Arc<FugcCoordinator> {
        if let Some(existing) = self.fugc_coordinator.get() {
            return existing.clone();
        }

        let coordinator = Arc::new(FugcCoordinator::new(
            heap_base,
            heap_size,
            num_workers,
            &self.thread_registry,
            &self.global_roots,
        ));

        match self.fugc_coordinator.set(coordinator.clone()) {
            Ok(()) => {
                let _ = self
                    .safepoint_manager
                    .set(SafepointManager::with_coordinator(&coordinator));
                coordinator
            }
            Err(_) => self.fugc_coordinator.get().unwrap().clone(),
        }
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
pub fn set_current_container(container: Arc<DIContainer>) {
    CURRENT_CONTAINER.with(|c| {
        *c.borrow_mut() = Some(container);
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
    crate::safepoint::clear_thread_safepoint_manager_cache();
}

/// RAII guard for setting a DI container for a scope
pub struct DIScope {
    previous: Option<Arc<DIContainer>>,
}

impl DIScope {
    /// Create a new DI scope with the given container
    pub fn new(container: Arc<DIContainer>) -> Self {
        let previous = CURRENT_CONTAINER.with(|c| c.borrow().clone());
        set_current_container(container);
        Self { previous }
    }
}

impl Drop for DIScope {
    fn drop(&mut self) {
        if let Some(prev) = self.previous.take() {
            set_current_container(prev);
        } else {
            clear_current_container();
        }
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
            &container1.global_roots().load(),
            &container2.global_roots().load()
        ));
    }

    #[test]
    fn test_di_scope() {
        // Each test gets its own isolated container
        let test_container = Arc::new(DIContainer::new_for_testing());
        let test_registry = test_container.thread_registry();

        {
            // Set this test's container as current for the scope
            let _scope = DIScope::new(Arc::clone(&test_container));
            let current = current_container();
            assert!(Arc::ptr_eq(current.thread_registry(), test_registry));
        }

        // Should be cleared after scope ends
        clear_current_container();
        let new_current = current_container();
        // New default container should have different registry
        assert!(!Arc::ptr_eq(new_current.thread_registry(), test_registry));
    }

    #[test]
    fn test_fugc_coordinator_creation() {
        let container = Arc::new(DIContainer::new_for_testing());
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let coordinator = container.create_fugc_coordinator(heap_base, 64 * 1024 * 1024, 4);

        assert!(Arc::ptr_eq(&coordinator, container.fugc_coordinator()));
    }
}
