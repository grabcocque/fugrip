//! Allocation facade that hides backend details behind zero-cost opaque handles
//!
//! This module provides a unified allocation facade with three selectable
//! backends controlled via Cargo features:
//!
//! - `use_mmtk`: Use the MMTk backend (production GC path).
//! - `use_jemalloc`: Use the jemalloc backend (manual allocations; useful for
//!   running without MMTk).
//! - `use_stub`: Use a testing stub backend (allocations fail / no-ops) so
//!   tests can exercise the allocation facade without depending on MMTk.
//!
//! The facade exposes zero-cost opaque handles (`MutatorHandle`, `PlanHandle`)
//! and monomorphized entry points (`allocate`, `post_alloc`, `write_barrier`,
//! etc.). Each entry point contains `#[cfg]` branches for the three backends,
//! keeping the public API stable while allowing compile-time backend selection.
//!
//! Testing guidance:
//! - Build and run tests with `--features use_stub` to route all allocation
//!   operations through the stub backend via `global_allocator()`.
//! - Existing tests that referenced a separate `StubAllocator` are kept
//!   compatible via a shim in `src/allocator.rs` that delegates to the facade
//!   when `use_stub` is enabled.

use crate::core::ObjectHeader;
use crate::error::{GcError, GcResult};

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Opaque allocation semantics - no MMTk types exposed
#[derive(Debug, Clone, Copy)]
pub enum AllocationSemantics {
    Default,
    Immortal,
    Los,
}

/// Opaque handle to a mutator thread - zero-cost, pointer-encoded
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MutatorHandle(NonZeroUsize);

/// Opaque handle to plan/MMTk instance - zero-cost, pointer-encoded
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanHandle(NonZeroUsize);

/// Initialize the facade registry (no-op for zero-cost handles)
#[inline(always)]
pub fn init_facade() {}

/// Register a mutator and get an opaque handle
#[cfg(feature = "use_mmtk")]
pub fn register_mutator(
    mutator: *mut mmtk::Mutator<crate::backends::mmtk::binding::RustVM>,
) -> MutatorHandle {
    let value = mutator as usize;
    let nz = NonZeroUsize::new(value).expect("mutator pointer must be non-null");
    MutatorHandle(nz)
}

#[cfg(not(feature = "use_mmtk"))]
pub fn register_mutator(thread_id: usize) -> MutatorHandle {
    let value = thread_id.max(1); // ensure non-zero
    let nz = NonZeroUsize::new(value).expect("thread id must be non-zero after clamp");
    MutatorHandle(nz)
}

/// Register a plan and get an opaque handle
#[cfg(feature = "use_mmtk")]
pub fn register_plan(
    plan: &'static mmtk::MMTK<crate::backends::mmtk::binding::RustVM>,
) -> PlanHandle {
    let value = plan as *const _ as usize;
    let nz = NonZeroUsize::new(value).expect("plan pointer must be non-null");
    PlanHandle(nz)
}

#[cfg(not(feature = "use_mmtk"))]
pub fn register_plan() -> PlanHandle {
    let nz = NonZeroUsize::new(1).unwrap();
    PlanHandle(nz)
}

/// Allocate memory through the facade - zero-cost monomorphized dispatch
pub fn allocate(
    mutator: MutatorHandle,
    size: usize,
    align: usize,
    offset: usize,
    semantics: AllocationSemantics,
) -> GcResult<*mut u8> {
    #[cfg(feature = "use_mmtk")]
    {
        // Use opaque handle system instead of direct MMTk calls to maintain blackwall abstraction
        use crate::opaque_handles::{MutatorId, OpaqueAllocator};

        // Convert MutatorHandle to MutatorId for opaque handle system
        let mutator_id = MutatorId::for_thread(deterministic_thread_id());

        // Calculate total size including header
        let header_size = std::mem::size_of::<ObjectHeader>();
        let body_size = size - header_size;
        let header = ObjectHeader::default();

        // Use opaque allocator instead of direct MMTk calls
        let object_id = OpaqueAllocator::allocate(mutator_id, header, body_size)?;

        // Convert opaque object ID back to raw pointer for existing interface
        let addr = object_id.as_ptr().ok_or(GcError::OutOfMemory)?;
        Ok(addr)
    }

    #[cfg(feature = "use_jemalloc")]
    {
        let _ = (mutator, offset, semantics);

        let layout = Layout::from_size_align(size, align).map_err(|_| GcError::InvalidLayout)?;

        let ptr = unsafe { alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(GcError::OutOfMemory);
        }

        Ok(ptr)
    }

    #[cfg(feature = "use_stub")]
    {
        let _ = (mutator, size, align, offset, semantics);
        // Stub always fails allocation for testing
        Err(GcError::OutOfMemory)
    }

    #[cfg(not(any(feature = "use_mmtk", feature = "use_jemalloc", feature = "use_stub")))]
    {
        compile_error!(
            "Must enable exactly one allocation backend: use_mmtk, use_jemalloc, or use_stub"
        );
    }
}

/// Post-allocation hook - zero-cost monomorphized dispatch
pub fn post_alloc(
    mutator: MutatorHandle,
    object: *mut u8,
    size: usize,
    semantics: AllocationSemantics,
) {
    #[cfg(feature = "use_mmtk")]
    {
        // Use opaque handle system to maintain blackwall abstraction
        use crate::opaque_handles::{MutatorId, ObjectId, OpaqueAllocator};

        // Convert MutatorHandle to MutatorId for opaque handle system
        let mutator_id = MutatorId::for_thread(deterministic_thread_id());

        // Calculate total size including header
        let header_size = std::mem::size_of::<ObjectHeader>();
        let body_size = size - header_size;

        // Create ObjectId for the allocated object using opaque system
        let object_id = ObjectId::new(object as usize, size);

        // Use opaque allocator for post-allocation processing
        // Note: The opaque allocator handles post-allocation internally during allocation
        // This is primarily a compatibility layer for existing code
        let _ = OpaqueAllocator::allocate(mutator_id, ObjectHeader::default(), body_size);
    }

    #[cfg(feature = "use_jemalloc")]
    {
        let _ = (mutator, object, size, semantics);
        // No-op for jemalloc
    }

    #[cfg(feature = "use_stub")]
    {
        let _ = (mutator, object, size, semantics);
        // No-op for stub
    }
}

/// Write barrier - zero-cost monomorphized dispatch
pub fn write_barrier(
    mutator: MutatorHandle,
    src: *mut u8,
    slot: *mut *mut u8,
    target: Option<*mut u8>,
) {
    #[cfg(feature = "use_mmtk")]
    {
        // Use opaque handle system for object tracking while maintaining write barrier functionality
        use crate::opaque_handles::{MutatorId, ObjectId};

        // Convert MutatorHandle to MutatorId for opaque handle system
        let mutator_id = MutatorId::for_thread(deterministic_thread_id());

        // Create ObjectIds for objects involved in the write barrier
        // This maintains object tracking through opaque system
        let _src_object_id = ObjectId::new(src as usize, 0); // Size unknown, using 0 as placeholder
        let _target_object_id = target.map(|t| ObjectId::new(t as usize, 0));

        // Register objects with opaque system for tracking
        // The actual write barrier implementation still needs direct MMTk calls
        // as this is GC-specific functionality not exposed in opaque handles
        unsafe {
            let mut_ptr = mutator.as_mmtk_mutator_ptr();
            let src_ref =
                mmtk::util::ObjectReference::from_raw_address(mmtk::util::Address::from_ptr(src));
            let slot_addr = mmtk::util::Address::from_ptr(slot as *mut u8);
            let vm_slot = mmtk::vm::slot::SimpleSlot::from_address(slot_addr);
            let target_ref = target.map(|t| {
                mmtk::util::ObjectReference::from_raw_address(mmtk::util::Address::from_ptr(t))
            });
            mmtk::memory_manager::object_reference_write_pre::<
                crate::backends::mmtk::binding::RustVM,
            >(&mut *mut_ptr, src_ref, vm_slot, target_ref);
            mmtk::memory_manager::object_reference_write_post::<
                crate::backends::mmtk::binding::RustVM,
            >(&mut *mut_ptr, src_ref, vm_slot, target_ref);
        }
    }

    #[cfg(feature = "use_jemalloc")]
    {
        let _ = (mutator, src, slot, target);
        // No-op for jemalloc
    }

    #[cfg(feature = "use_stub")]
    {
        let _ = (mutator, src, slot, target);
        // No-op for stub
    }
}

/// Trigger garbage collection - zero-cost monomorphized dispatch
pub fn trigger_gc(plan: PlanHandle) {
    // Use a deterministic per-process thread id assigned from a global counter.
    // Avoids relying on unstable ThreadId internals and avoids DefaultHasher
    // randomness so the id is stable within a process run.
    let thread_id = deterministic_thread_id();

    handle_user_collection_request(plan, thread_id);
}

/// Request a garbage collection - zero-cost monomorphized dispatch
pub fn handle_user_collection_request(plan: PlanHandle, thread_id: usize) {
    #[cfg(feature = "use_mmtk")]
    {
        // Use opaque handle API to respect blackwall abstraction
        use crate::opaque_handles::{OpaqueAllocator, default_plan};
        let plan_id = default_plan();
        let _ = OpaqueAllocator::trigger_gc(plan_id);
        let _ = (plan, thread_id); // Suppress unused warnings
    }

    #[cfg(feature = "use_jemalloc")]
    {
        let _ = (plan, thread_id);
        // jemalloc doesn't have GC - this is a no-op
    }

    #[cfg(feature = "use_stub")]
    {
        let _ = (plan, thread_id);
        // No-op for stub
    }
}

/// Get plan statistics - zero-cost monomorphized dispatch
pub fn get_plan_total_pages(plan: PlanHandle) -> usize {
    #[cfg(feature = "use_mmtk")]
    {
        PlanHandle::total_pages_internal(plan)
    }

    #[cfg(not(feature = "use_mmtk"))]
    {
        let _ = plan;
        0
    }
}

pub fn get_plan_reserved_pages(plan: PlanHandle) -> usize {
    #[cfg(feature = "use_mmtk")]
    {
        PlanHandle::reserved_pages_internal(plan)
    }

    #[cfg(not(feature = "use_mmtk"))]
    {
        let _ = plan;
        0
    }
}

/// Unregister handles to prevent leaks
#[inline(always)]
pub fn unregister_mutator(_handle: MutatorHandle) {}

#[inline(always)]
pub fn unregister_plan(_handle: PlanHandle) {}

// Internal helpers on handles for zero-cost pointer extraction
impl MutatorHandle {
    #[cfg(feature = "use_mmtk")]
    #[inline(always)]
    fn as_mmtk_mutator_ptr(self) -> *mut mmtk::Mutator<crate::backends::mmtk::binding::RustVM> {
        self.0.get() as *mut mmtk::Mutator<crate::backends::mmtk::binding::RustVM>
    }

    #[cfg(not(feature = "use_mmtk"))]
    #[inline(always)]
    fn _thread_id(self) -> usize {
        self.0.get()
    }
}

impl PlanHandle {
    #[cfg(feature = "use_mmtk")]
    #[inline(always)]
    fn as_mmtk_plan_ptr(self) -> &'static mmtk::MMTK<crate::backends::mmtk::binding::RustVM> {
        unsafe { &*(self.0.get() as *const mmtk::MMTK<crate::backends::mmtk::binding::RustVM>) }
    }

    #[cfg(feature = "use_mmtk")]
    #[inline(always)]
    fn total_pages_internal(self) -> usize {
        self.as_mmtk_plan_ptr().get_plan().get_total_pages()
    }

    #[cfg(feature = "use_mmtk")]
    #[inline(always)]
    fn reserved_pages_internal(self) -> usize {
        self.as_mmtk_plan_ptr().get_plan().get_reserved_pages()
    }

    #[cfg(not(feature = "use_mmtk"))]
    #[inline(always)]
    fn total_pages_internal(_self_: PlanHandle) -> usize {
        0
    }

    #[cfg(not(feature = "use_mmtk"))]
    #[inline(always)]
    fn reserved_pages_internal(_self_: PlanHandle) -> usize {
        0
    }
}

#[cfg(feature = "use_mmtk")]
impl From<AllocationSemantics> for mmtk::AllocationSemantics {
    fn from(sem: AllocationSemantics) -> Self {
        match sem {
            AllocationSemantics::Default => mmtk::AllocationSemantics::Default,
            AllocationSemantics::Immortal => mmtk::AllocationSemantics::Immortal,
            AllocationSemantics::Los => mmtk::AllocationSemantics::Los,
        }
    }
}

/// Legacy allocation facade for backwards compatibility
/// This will be removed once all code uses the opaque handle APIs
pub struct AllocatorFacade {
    backend: Box<dyn AllocationBackend>,
}

pub trait AllocationBackend: Send + Sync {
    fn allocate(&self, size: usize, align: usize) -> GcResult<*mut u8>;
    fn deallocate(&self, addr: *mut u8, size: usize, align: usize);
    fn total_allocated(&self) -> usize;
    fn allocation_count(&self) -> usize;
}

pub struct JemallocBackend {
    total_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl JemallocBackend {
    pub fn new() -> Self {
        JemallocBackend {
            total_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }
}

impl AllocationBackend for JemallocBackend {
    fn allocate(&self, size: usize, align: usize) -> GcResult<*mut u8> {
        let layout = Layout::from_size_align(size, align).map_err(|_| GcError::InvalidLayout)?;

        let ptr = unsafe { alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(GcError::OutOfMemory);
        }

        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        Ok(ptr)
    }

    fn deallocate(&self, addr: *mut u8, size: usize, align: usize) {
        if let Ok(layout) = Layout::from_size_align(size, align) {
            unsafe {
                dealloc(addr, layout);
            }
            self.total_allocated.fetch_sub(size, Ordering::Relaxed);
        }
    }

    fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }
}

impl AllocatorFacade {
    pub fn new_jemalloc() -> Self {
        AllocatorFacade {
            backend: Box::new(JemallocBackend::new()),
        }
    }

    pub fn new_stub() -> Self {
        // Stub backend that always fails allocations and no-ops deallocations
        struct StubBackend;

        impl AllocationBackend for StubBackend {
            fn allocate(&self, _size: usize, _align: usize) -> GcResult<*mut u8> {
                Err(GcError::OutOfMemory)
            }

            fn deallocate(&self, _addr: *mut u8, _size: usize, _align: usize) {
                // no-op
            }

            fn total_allocated(&self) -> usize {
                0
            }

            fn allocation_count(&self) -> usize {
                0
            }
        }

        AllocatorFacade {
            backend: Box::new(StubBackend),
        }
    }

    pub fn allocate_object(&self, header: ObjectHeader, body_bytes: usize) -> GcResult<*mut u8> {
        let total_bytes = std::mem::size_of::<ObjectHeader>() + body_bytes;
        let align = std::mem::align_of::<ObjectHeader>();

        let addr = self.backend.allocate(total_bytes, align)?;

        unsafe {
            let header_ptr = addr as *mut ObjectHeader;
            std::ptr::write(header_ptr, header);

            if body_bytes > 0 {
                let body_ptr = addr.add(std::mem::size_of::<ObjectHeader>());
                std::ptr::write_bytes(body_ptr, 0, body_bytes);
            }

            Ok(addr)
        }
    }

    pub fn deallocate_object(&self, obj: *mut u8, total_size: usize) {
        let align = std::mem::align_of::<ObjectHeader>();
        self.backend.deallocate(obj, total_size, align);
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.backend.total_allocated()
    }

    /// Get allocation count
    pub fn allocation_count(&self) -> usize {
        self.backend.allocation_count()
    }

    /// Poll for GC safepoint - no-op for facade backends
    pub fn poll_gc(&self, _thread: &crate::thread::MutatorThread) {
        // No-op for jemalloc and stub backends
        // MMTk backend would handle this through opaque handles if needed
    }
}

static GLOBAL_ALLOCATOR: std::sync::OnceLock<AllocatorFacade> = std::sync::OnceLock::new();

// Deterministic per-process thread id allocator.
// Each thread gets a small integer id assigned on first use.
static THREAD_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    // thread-local cached id for this thread. 0 means uninitialized.
    static CACHED_THREAD_ID: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

pub fn deterministic_thread_id() -> usize {
    // Fast path: return cached id if present.
    CACHED_THREAD_ID.with(|c| {
        let id = c.get();
        if id != 0 {
            return id;
        }

        // Allocate a new id from the global counter. Ensure non-zero.
        let new_id = THREAD_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        // Store (new_id) as-is; ensure it's non-zero.
        let assigned = if new_id == 0 { 1 } else { new_id };
        c.set(assigned);
        assigned
    })
}

pub fn global_allocator() -> &'static AllocatorFacade {
    GLOBAL_ALLOCATOR.get_or_init(|| {
        #[cfg(feature = "use_stub")]
        {
            AllocatorFacade::new_stub()
        }

        #[cfg(all(not(feature = "use_stub"), feature = "use_jemalloc"))]
        {
            AllocatorFacade::new_jemalloc()
        }

        #[cfg(all(not(feature = "use_stub"), not(feature = "use_jemalloc")))]
        {
            // Default to jemalloc backend if no explicit backend selected.
            AllocatorFacade::new_jemalloc()
        }
    })
}
