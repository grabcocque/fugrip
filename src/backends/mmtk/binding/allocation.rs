//! Allocation APIs with FUGC optimizations

use mmtk::util::{Address, ObjectReference};
use mmtk::vm::slot::SimpleSlot;
use crate::backends::mmtk::FugcPlanManager;
use anyhow;
use arc_swap::ArcSwap;
use std::sync::Arc;

use crate::binding::FUGC_PLAN_MANAGER;
use super::vm_impl::RustVM;

/// Allocate an object using MMTk with FUGC optimizations.
/// This performs the actual allocation through MMTk's allocator infrastructure
/// and integrates with FUGC's concurrent collection.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::{fugc_alloc, initialize_mmtk_with_fugc, bind_mutator_thread};
/// use fugrip::thread::MutatorThread;
///
/// // Initialize MMTk and bind a mutator
/// let mmtk = initialize_mmtk_with_fugc().expect("Failed to initialize");
/// let thread = MutatorThread::new();
/// let mut mutator = bind_mutator_thread(mmtk, thread).expect("Failed to bind mutator");
///
/// // Allocate a 64-byte object
/// let obj = fugc_alloc(&mut mutator, 64, 8, 0)
///     .expect("Failed to allocate object");
/// ```
pub fn fugc_alloc(
    mutator: &mut mmtk::Mutator<RustVM>,
    size: usize,
    align: usize,
    offset: usize,
) -> anyhow::Result<ObjectReference> {
    use mmtk::AllocationSemantics;

    // Use MMTk's allocation API with default semantics
    let addr =
        mmtk::memory_manager::alloc(mutator, size, align, offset, AllocationSemantics::Default);

    // Convert Address to ObjectReference
    match ObjectReference::from_raw_address(addr) {
        Some(obj_ref) => {
            // Integrate with FUGC post-allocation processing
            FUGC_PLAN_MANAGER
                .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
                .load()
                .post_alloc(obj_ref, size);
            Ok(obj_ref)
        }
        None => anyhow::bail!(
            "Failed to allocate {} bytes with alignment {} - invalid address",
            size,
            align
        ),
    }
}

/// Get allocation info using FUGC optimizations
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_alloc_info;
///
/// // Get allocation info for a 64-byte object with 8-byte alignment
/// let (aligned_size, alignment) = fugc_alloc_info(64, 8);
/// assert_eq!(aligned_size, 64);
/// assert_eq!(alignment, 8);
///
/// // Test alignment adjustment
/// let (aligned_size, alignment) = fugc_alloc_info(65, 16);
/// assert_eq!(aligned_size, 80); // Rounded up to 16-byte boundary
/// assert_eq!(alignment, 16);
/// ```
pub fn fugc_alloc_info(size: usize, align: usize) -> (usize, usize) {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .alloc_info(size, align)
}

/// Handle post-allocation processing with FUGC optimizations
///
/// # Examples
///
/// ```
/// use fugrip::binding::fugc_post_alloc;
/// use crate::frontend::types::{Address, ObjectReference};
///
/// // Create an object reference for demonstration
/// let addr = unsafe { Address::from_usize(0x10000000) };
/// let obj = ObjectReference::from_raw_address(addr).unwrap();
///
/// // Handle post-allocation processing
/// fugc_post_alloc(obj, 64);
/// // This integrates the object with FUGC's concurrent collection
/// ```
pub fn fugc_post_alloc(obj: ObjectReference, bytes: usize) {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .post_alloc(obj, bytes);
}

// Note: The unsafe fugc_write_barrier function has been removed to prevent
// accidental segfaults. Use fugc_write_barrier_with_mutator instead when
// a mutator is available, or access the write barrier component directly
// via the plan manager for safe testing.

/// Mutator-aware write barrier helper. Call this when a `&mut Mutator<RustVM>` is available
/// so MMTk's `object_reference_write_pre`/`post` can be invoked with the correct argument types.
pub fn fugc_write_barrier_with_mutator(
    mutator: &mut mmtk::Mutator<RustVM>,
    src: ObjectReference,
    slot: Address,
    target: ObjectReference,
) {
    let vm_slot = SimpleSlot::from_address(slot);
    let target_opt = Some(target);

    // Call MMTk pre/post hooks
    mmtk::memory_manager::object_reference_write_pre::<RustVM>(mutator, src, vm_slot, target_opt);
    mmtk::memory_manager::object_reference_write_post::<RustVM>(mutator, src, vm_slot, target_opt);

    // Notify FUGC coordinator as well
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .handle_write_barrier(src, slot, target);
}
