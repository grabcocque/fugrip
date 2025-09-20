//! Allocation facade that completely hides MMTk implementation details
//!
//! This provides zero-cost opaque handles with no vtables - all dispatch
//! happens at compile time through monomorphization.
//!
//! ALL MMTk interactions MUST go through this facade.

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
pub fn register_mutator(mutator: *mut mmtk::Mutator<crate::binding::RustVM>) -> MutatorHandle {
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
pub fn register_plan(plan: &'static mmtk::MMTK<crate::binding::RustVM>) -> PlanHandle {
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
        let addr = unsafe {
            let ptr = mutator.as_mmtk_mutator_ptr();
            mmtk::memory_manager::alloc(&mut *ptr, size, align, offset, semantics.into())
        };

        if addr.is_zero() {
            return Err(GcError::OutOfMemory);
        }
        Ok(addr.to_mut_ptr::<u8>())
    }

    #[cfg(not(feature = "use_mmtk"))]
    {
        let _ = (mutator, offset, semantics);

        let layout = Layout::from_size_align(size, align).map_err(|_| GcError::InvalidLayout)?;

        let ptr = unsafe { alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(GcError::OutOfMemory);
        }

        Ok(ptr)
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
        unsafe {
            let ptr = mutator.as_mmtk_mutator_ptr();
            let obj_ref = mmtk::util::ObjectReference::from_raw_address(
                mmtk::util::Address::from_ptr(object),
            );
            mmtk::memory_manager::post_alloc(&mut *ptr, obj_ref, size, semantics.into());
        }
    }

    #[cfg(not(feature = "use_mmtk"))]
    {
        let _ = (mutator, object, size, semantics);
        // No-op for jemalloc
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
        unsafe {
            let mut_ptr = mutator.as_mmtk_mutator_ptr();
            let src_ref =
                mmtk::util::ObjectReference::from_raw_address(mmtk::util::Address::from_ptr(src));
            let slot_addr = mmtk::util::Address::from_ptr(slot as *mut u8);
            let vm_slot = mmtk::vm::slot::SimpleSlot::from_address(slot_addr);
            let target_ref = target.map(|t| {
                mmtk::util::ObjectReference::from_raw_address(mmtk::util::Address::from_ptr(t))
            });
            mmtk::memory_manager::object_reference_write_pre::<crate::binding::RustVM>(
                &mut *mut_ptr,
                src_ref,
                vm_slot,
                target_ref,
            );
            mmtk::memory_manager::object_reference_write_post::<crate::binding::RustVM>(
                &mut *mut_ptr,
                src_ref,
                vm_slot,
                target_ref,
            );
        }
    }

    #[cfg(not(feature = "use_mmtk"))]
    {
        let _ = (mutator, src, slot, target);
        // No-op for jemalloc
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
        let tls = mmtk::util::opaque_pointer::VMThread::from_usize(thread_id);
        unsafe {
            mmtk::memory_manager::handle_user_collection_request::<crate::binding::RustVM>(
                plan.as_mmtk_plan_ptr(),
                tls,
            );
        }
    }

    #[cfg(not(feature = "use_mmtk"))]
    {
        let _ = (plan, thread_id);
        // Trigger manual GC if supported
        if let Some(plan_manager) = crate::plan::FugcPlanManager::global() {
            plan_manager.gc();
        }
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
    fn as_mmtk_mutator_ptr(self) -> *mut mmtk::Mutator<crate::binding::RustVM> {
        self.0.get() as *mut mmtk::Mutator<crate::binding::RustVM>
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
    fn as_mmtk_plan_ptr(self) -> &'static mmtk::MMTK<crate::binding::RustVM> {
        unsafe { &*(self.0.get() as *const mmtk::MMTK<crate::binding::RustVM>) }
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
    GLOBAL_ALLOCATOR.get_or_init(|| AllocatorFacade::new_jemalloc())
}
