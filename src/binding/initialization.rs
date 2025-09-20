//! MMTk initialization and configuration with FUGC optimizations

use crate::compat::{
    Address,
    vm::opaque_pointer::{OpaquePointer, VMMutatorThread, VMThread},
};
use crate::{plan::FugcPlanManager, thread::MutatorThread};
use anyhow;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use std::sync::Arc;

use super::{FUGC_PLAN_MANAGER, mutator::*, vm_impl::RustVM};

/// Initialize MMTk with FUGC-optimized configuration and set up the plan manager.
/// This creates the MMTk instance, configures it with FUGC settings, and initializes
/// the global plan manager.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::initialize_mmtk_with_fugc;
///
/// // Initialize MMTk with FUGC configuration
/// let mmtk = initialize_mmtk_with_fugc()
///     .expect("Failed to initialize MMTk with FUGC");
///
/// // MMTk is now ready for allocation and garbage collection
/// ```
pub fn initialize_mmtk_with_fugc() -> anyhow::Result<&'static mmtk::MMTK<RustVM>> {
    use crate::plan::create_fugc_mmtk_options;

    // Create FUGC-optimized MMTk options
    let options = create_fugc_mmtk_options()?;

    // Build an MMTkBuilder seeded with our configured options
    let mut builder = mmtk::MMTKBuilder::new_no_env_vars();
    builder.options = options;

    // Initialize MMTk using the builder. mmtk_init returns a Box<MMTK<VM>>
    let boxed = mmtk::memory_manager::mmtk_init::<RustVM>(&builder);

    // Leak the Box to obtain a &'static reference for the lifetime of the process.
    let mmtk_static: &'static mmtk::MMTK<RustVM> = Box::leak(boxed);

    // Initialize the FUGC plan manager with the MMTk instance
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .initialize(mmtk_static);

    Ok(mmtk_static)
}

/// Bind a mutator thread to MMTk and register it with FUGC infrastructure.
/// This should be called for each mutator thread that will perform allocations.
///
/// # Examples
///
/// ```no_run
/// use fugrip::binding::{bind_mutator_thread, initialize_mmtk_with_fugc};
/// use fugrip::thread::MutatorThread;
///
/// // Initialize MMTk first
/// let mmtk = initialize_mmtk_with_fugc()
///     .expect("Failed to initialize MMTk");
///
/// // Create mutator thread context
/// let thread = MutatorThread::new();
///
/// // Bind mutator to MMTk and FUGC
/// let mut mutator = bind_mutator_thread(mmtk, thread)
///     .expect("Failed to bind mutator");
/// ```
pub fn bind_mutator_thread(
    mmtk: &'static mmtk::MMTK<RustVM>,
    thread: MutatorThread,
) -> anyhow::Result<MutatorHandle> {
    bind_mutator_thread_with_registry(mmtk, thread, None)
}

pub fn bind_mutator_thread_with_registry(
    mmtk: &'static mmtk::MMTK<RustVM>,
    thread: MutatorThread,
    registry: Option<&crate::thread::ThreadRegistry>,
) -> anyhow::Result<MutatorHandle> {
    // Create MMTk mutator
    let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
        // SAFETY: We construct an address from a thread id for use as an opaque TLS
        // value. This mirrors test and example usage in the codebase that use
        // `Address::from_usize` for synthetic TLS addresses.
        Address::from_usize(thread.id())
    })));

    // Bind mutator with MMTk
    let mutator_box = mmtk::memory_manager::bind_mutator(mmtk, tls);

    // Leak the Box to get a pointer that lives for the duration of the process.
    let mutator_ptr: *mut mmtk::Mutator<RustVM> = Box::leak(mutator_box);
    let mutator_handle = MutatorHandle::from_raw(mutator_ptr);

    // Register with FUGC infrastructure - inline to avoid borrowing conflicts
    match registry {
        Some(reg) => reg.register(thread.clone()),
        None => {
            // Use DI container instead of global registry
            let container = crate::di::current_container();
            container.thread_registry().register(thread.clone());
        }
    }
    let key = mutator_thread_key(tls);
    MUTATOR_MAP.get_or_init(DashMap::new).insert(
        key,
        MutatorRegistration::new(mutator_handle.as_ptr(), thread),
    );

    Ok(mutator_handle)
}

/// Initialize the FUGC plan manager with an existing MMTk instance.
/// This is an internal function - use `initialize_mmtk_with_fugc` instead.
pub fn initialize_fugc_plan(mmtk: &'static mmtk::MMTK<RustVM>) {
    FUGC_PLAN_MANAGER
        .get_or_init(|| ArcSwap::new(Arc::new(FugcPlanManager::new())))
        .load()
        .initialize(mmtk);
}
