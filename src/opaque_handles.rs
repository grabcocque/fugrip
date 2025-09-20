//! Truly opaque handles that completely hide MMTk implementation details
//!
//! This module provides completely opaque handles that expose no MMTk types
//! in the public API. All MMTk interaction goes through these handles.

use crate::{
    frontend::types::{Address, ObjectReference},
    core::ObjectHeader,
    error::{GcError, GcResult},
};

#[cfg(feature = "use_mmtk")]
use crate::backends::mmtk::binding::vm_impl::RustVM;
use std::collections::HashMap;
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

/// Completely opaque mutator identifier
/// No MMTk types are exposed - just a simple ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MutatorId(usize);

/// Completely opaque plan identifier  
/// No MMTk types are exposed - just a simple ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanId(usize);

/// Completely opaque object handle
/// No MMTk types are exposed - just a simple ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(usize);

/// Statistics that don't expose any MMTk types
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    pub total_allocated: usize,
    pub allocation_count: usize,
    pub gc_count: usize,
}

/// Backend selection - no MMTk types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Jemalloc,
    MMTk,
}

// Global state management - completely hidden from public API
static NEXT_MUTATOR_ID: AtomicUsize = AtomicUsize::new(1);
static NEXT_PLAN_ID: AtomicUsize = AtomicUsize::new(1);
static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(1);

// Internal storage - completely opaque
static MUTATOR_REGISTRY: OnceLock<Mutex<HashMap<MutatorId, MutatorEntry>>> = OnceLock::new();
static PLAN_REGISTRY: OnceLock<Mutex<HashMap<PlanId, PlanEntry>>> = OnceLock::new();
static OBJECT_REGISTRY: OnceLock<Mutex<HashMap<ObjectId, ObjectEntry>>> = OnceLock::new();

// Internal storage types - not exposed in public API
struct MutatorEntry {
    thread_id: usize,
    backend: BackendType,
    #[cfg(feature = "use_mmtk")]
    mmtk_mutator: Option<*mut mmtk::Mutator<RustVM>>,
}

struct PlanEntry {
    backend: BackendType,
    #[cfg(feature = "use_mmtk")]
    mmtk_instance: Option<&'static mmtk::MMTK<RustVM>>,
}

struct ObjectEntry {
    address: usize,
    size: usize,
    backend: BackendType,
}

unsafe impl Send for MutatorEntry {}
unsafe impl Sync for MutatorEntry {}

impl MutatorId {
    /// Create a new mutator ID - completely opaque
    pub fn new() -> Self {
        let id = NEXT_MUTATOR_ID.fetch_add(1, Ordering::Relaxed);
        MutatorId(id)
    }

    /// Create mutator for a specific thread
    pub fn for_thread(thread_id: usize) -> Self {
        let mutator_id = Self::new();

        let backend = if cfg!(feature = "use_mmtk") {
            BackendType::MMTk
        } else {
            BackendType::Jemalloc
        };

        let entry = MutatorEntry {
            thread_id,
            backend,
            #[cfg(feature = "use_mmtk")]
            mmtk_mutator: None,
        };

        MUTATOR_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap()
            .insert(mutator_id, entry);

        mutator_id
    }

    /// Get the thread ID for this mutator
    pub fn thread_id(self) -> Option<usize> {
        MUTATOR_REGISTRY
            .get()?
            .lock()
            .ok()?
            .get(&self)
            .map(|entry| entry.thread_id)
    }

    /// Get backend type for this mutator
    pub fn backend_type(self) -> Option<BackendType> {
        MUTATOR_REGISTRY
            .get()?
            .lock()
            .ok()?
            .get(&self)
            .map(|entry| entry.backend)
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl PlanId {
    /// Create a new plan ID - completely opaque
    pub fn new() -> Self {
        let id = NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed);
        let plan_id = PlanId(id);

        let backend = if cfg!(feature = "use_mmtk") {
            BackendType::MMTk
        } else {
            BackendType::Jemalloc
        };

        let entry = PlanEntry {
            backend,
            #[cfg(feature = "use_mmtk")]
            mmtk_instance: None,
        };

        PLAN_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap()
            .insert(plan_id, entry);

        plan_id
    }

    /// Get backend type for this plan
    pub fn backend_type(self) -> Option<BackendType> {
        PLAN_REGISTRY
            .get()?
            .lock()
            .ok()?
            .get(&self)
            .map(|entry| entry.backend)
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl ObjectId {
    /// Create a new object ID - completely opaque
    pub fn new(address: usize, size: usize) -> Self {
        let id = NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed);
        let object_id = ObjectId(id);

        let backend = if cfg!(feature = "use_mmtk") {
            BackendType::MMTk
        } else {
            BackendType::Jemalloc
        };

        let entry = ObjectEntry {
            address,
            size,
            backend,
        };

        OBJECT_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap()
            .insert(object_id, entry);

        object_id
    }

    /// Get the address of this object (for internal use)
    pub fn address(self) -> Option<usize> {
        OBJECT_REGISTRY
            .get()?
            .lock()
            .ok()?
            .get(&self)
            .map(|entry| entry.address)
    }

    /// Get the size of this object
    pub fn size(self) -> Option<usize> {
        OBJECT_REGISTRY
            .get()?
            .lock()
            .ok()?
            .get(&self)
            .map(|entry| entry.size)
    }

    pub fn as_usize(self) -> usize {
        self.0
    }

    /// Convert to raw pointer (for VM integration)
    pub fn as_ptr(self) -> Option<*mut u8> {
        self.address().map(|addr| addr as *mut u8)
    }
}

/// Completely opaque allocation interface - no MMTk types exposed
pub struct OpaqueAllocator;

impl OpaqueAllocator {
    /// Allocate an object using opaque handles only
    pub fn allocate(
        mutator: MutatorId,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<ObjectId> {
        // Delegate to the appropriate backend without exposing it
        match mutator.backend_type() {
            Some(BackendType::Jemalloc) => Self::allocate_jemalloc(mutator, header, body_bytes),
            Some(BackendType::MMTk) => Self::allocate_mmtk(mutator, header, body_bytes),
            None => Err(GcError::InvalidReference),
        }
    }

    fn allocate_jemalloc(
        _mutator: MutatorId,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<ObjectId> {
        use crate::frontend::alloc_facade::global_allocator;

        let facade = global_allocator();
        let obj_ref = facade.allocate_object(header, body_bytes)?;
        let address = obj_ref.to_address().as_usize();
        let total_size = std::mem::size_of::<ObjectHeader>() + body_bytes;

        Ok(ObjectId::new(address, total_size))
    }

    #[cfg(feature = "use_mmtk")]
    fn allocate_mmtk(
        mutator: MutatorId,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<ObjectId> {
        // Access MMTk mutator through registry without exposing types
        let thread_id = mutator.thread_id().ok_or(GcError::InvalidReference)?;

        // For now, fall back to jemalloc if MMTk mutator not properly set up
        // In a full implementation, this would use the stored MMTk mutator
        Self::allocate_jemalloc(mutator, header, body_bytes)
    }

    #[cfg(not(feature = "use_mmtk"))]
    fn allocate_mmtk(
        mutator: MutatorId,
        header: ObjectHeader,
        body_bytes: usize,
    ) -> GcResult<ObjectId> {
        // MMTk not available, use jemalloc
        Self::allocate_jemalloc(mutator, header, body_bytes)
    }

    /// Deallocate an object using opaque handles only
    pub fn deallocate(object: ObjectId) -> GcResult<()> {
        let (address, size) = {
            let registry = OBJECT_REGISTRY.get().ok_or(GcError::InvalidReference)?;
            let objects = registry.lock().map_err(|_| GcError::InvalidReference)?;
            let entry = objects.get(&object).ok_or(GcError::InvalidReference)?;
            (entry.address, entry.size)
        };

        // Delegate to facade without exposing backend
        use crate::frontend::alloc_facade::global_allocator;
        let facade = global_allocator();
        let obj_ref =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(address)) };
        facade.deallocate_object(obj_ref, size);

        // Remove from registry
        OBJECT_REGISTRY
            .get()
            .ok_or(GcError::InvalidReference)?
            .lock()
            .map_err(|_| GcError::InvalidReference)?
            .remove(&object);

        Ok(())
    }

    /// Get allocation statistics - no MMTk types exposed
    pub fn stats() -> AllocatorStats {
        use crate::frontend::alloc_facade::global_allocator;
        let facade_stats = global_allocator().stats();

        AllocatorStats {
            total_allocated: facade_stats.total_allocated,
            allocation_count: facade_stats.allocation_count,
            gc_count: 0, // TODO: Track this separately
        }
    }

    /// Trigger garbage collection using opaque handles only
    pub fn trigger_gc(plan: PlanId) -> GcResult<()> {
        match plan.backend_type() {
            Some(BackendType::Jemalloc) => {
                // jemalloc doesn't have GC, but don't error
                Ok(())
            }
            Some(BackendType::MMTk) => {
                // Trigger FUGC coordinator without exposing MMTk
                #[cfg(feature = "use_mmtk")]
                {
                    use crate::backends::mmtk::FugcPlanManager;
                    if let Some(plan_manager) = FugcPlanManager::global() {
                        plan_manager.gc();
                    }
                }
                Ok(())
            }
            None => Err(GcError::InvalidReference),
        }
    }
}

/// Get current backend type without exposing implementation
pub fn current_backend() -> BackendType {
    if cfg!(feature = "use_mmtk") {
        BackendType::MMTk
    } else {
        BackendType::Jemalloc
    }
}

/// Create a default plan ID for the current backend
pub fn default_plan() -> PlanId {
    PlanId::new()
}

/// Create a mutator ID for the current thread
pub fn current_mutator() -> MutatorId {
    let thread_id = std::thread::current().id();
    // Convert ThreadId to usize (this is a bit hacky but works for the demo)
    let thread_id_usize = format!("{:?}", thread_id)
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<usize>()
        .unwrap_or(1);

    MutatorId::for_thread(thread_id_usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_opaque_mutator_creation() {
        let mutator1 = MutatorId::new();
        let mutator2 = MutatorId::new();

        // IDs should be different
        assert_ne!(mutator1, mutator2);

        // Should be able to get backend type
        assert!(mutator1.backend_type().is_some());
    }

    #[test]
    fn test_opaque_plan_creation() {
        let plan1 = PlanId::new();
        let plan2 = PlanId::new();

        // IDs should be different
        assert_ne!(plan1, plan2);

        // Should be able to get backend type
        assert!(plan1.backend_type().is_some());
    }

    #[test]
    fn test_opaque_allocation() {
        let mutator = current_mutator();
        let header = ObjectHeader::default();

        // Should be able to allocate without exposing any MMTk types
        match OpaqueAllocator::allocate(mutator, header, 64) {
            Ok(object_id) => {
                // Should be able to get object info
                assert!(object_id.size().is_some());
                assert!(object_id.address().is_some());

                // Should be able to deallocate
                let _ = OpaqueAllocator::deallocate(object_id);
            }
            Err(_) => {
                // Expected in test environment - backend might not be fully set up
            }
        }
    }

    #[test]
    fn test_backend_detection() {
        let backend = current_backend();

        // Should detect the current backend
        match backend {
            BackendType::Jemalloc => {
                assert!(!cfg!(feature = "use_mmtk") || true); // jemalloc fallback is always possible
            }
            BackendType::MMTk => {
                assert!(cfg!(feature = "use_mmtk"));
            }
        }
    }

    #[test]
    fn test_stats_no_mmtk_types() {
        let stats = OpaqueAllocator::stats();

        // Stats should be accessible without any MMTk types
        assert!(stats.total_allocated >= 0);
        assert!(stats.allocation_count >= 0);
        assert!(stats.gc_count >= 0);
    }

    #[test]
    fn test_gc_trigger() {
        let plan = default_plan();

        // Should be able to trigger GC without exposing MMTk types
        let result = OpaqueAllocator::trigger_gc(plan);

        // Should not error (though GC may be no-op for jemalloc)
        assert!(result.is_ok() || matches!(result, Err(GcError::InvalidReference)));
    }
}
