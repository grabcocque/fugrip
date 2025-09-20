//! Mutator registration and management for MMTk VM binding

use crate::compat::vm::opaque_pointer::{VMMutatorThread, VMThread};
use crate::thread::MutatorThread;
use dashmap::DashMap;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::OnceLock;

use super::vm_impl::RustVM;

pub struct MutatorRegistration {
    pub mutator: *mut mmtk::Mutator<RustVM>,
    thread: MutatorThread,
}

// SAFETY: We ensure mutator pointer is valid during its lifetime
unsafe impl Send for MutatorRegistration {}
unsafe impl Sync for MutatorRegistration {}

impl MutatorRegistration {
    pub fn new(mutator: *mut mmtk::Mutator<RustVM>, thread: MutatorThread) -> Self {
        Self { mutator, thread }
    }

    pub unsafe fn as_mutator(&self) -> &'static mut mmtk::Mutator<RustVM> {
        unsafe { &mut *self.mutator }
    }

    pub fn thread(&self) -> &MutatorThread {
        &self.thread
    }
}

/// Handle for interacting with an MMTk mutator created by the binding layer.
///
/// The handle owns the leaked mutator pointer and provides safe accessors that
/// respect Rust's borrowing discipline. It is intentionally not `Clone` to
/// avoid aliasing the underlying `&mut` reference.
pub struct MutatorHandle {
    ptr: NonNull<mmtk::Mutator<RustVM>>,
}

impl MutatorHandle {
    pub fn from_raw(ptr: *mut mmtk::Mutator<RustVM>) -> Self {
        let ptr = NonNull::new(ptr).expect("MMTk returned a null mutator pointer");
        Self { ptr }
    }

    /// Get a raw pointer to the underlying mutator for FUGC registries.
    pub fn as_ptr(&self) -> *mut mmtk::Mutator<RustVM> {
        self.ptr.as_ptr()
    }
}

impl Deref for MutatorHandle {
    type Target = mmtk::Mutator<RustVM>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl DerefMut for MutatorHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

pub static MUTATOR_MAP: OnceLock<DashMap<usize, MutatorRegistration>> = OnceLock::new();

pub fn vm_thread_key(thread: VMThread) -> usize {
    thread.0.to_address().as_usize()
}

pub fn mutator_thread_key(thread: VMMutatorThread) -> usize {
    vm_thread_key(thread.0)
}

/// Register a mutator with the binding infrastructure so that safepoint and
/// root scanning hooks can coordinate with MMTk.
pub fn register_mutator_context(
    tls: VMMutatorThread,
    mutator: &'static mut mmtk::Mutator<RustVM>,
    thread: MutatorThread,
) {
    // Use DI container instead of global registry
    let container = crate::di::current_container();
    container.thread_registry().register(thread.clone());

    let key = mutator_thread_key(tls);
    MUTATOR_MAP
        .get_or_init(DashMap::new)
        .insert(key, MutatorRegistration::new(mutator as *mut _, thread));
}

/// Remove a mutator from the binding registries.
pub fn unregister_mutator_context(tls: VMMutatorThread) {
    let key = mutator_thread_key(tls);
    if let Some((_, entry)) = MUTATOR_MAP.get_or_init(DashMap::new).remove(&key) {
        // Use DI container instead of global registry
        let container = crate::di::current_container();
        container.thread_registry().unregister(entry.thread.id());
    }
}

pub fn with_mutator_registration<F, R>(tls: VMMutatorThread, f: F) -> Option<R>
where
    F: FnOnce(&MutatorRegistration) -> R,
{
    let key = mutator_thread_key(tls);
    MUTATOR_MAP
        .get_or_init(DashMap::new)
        .get(&key)
        .map(|r| f(r.value()))
}

pub fn visit_all_mutators<F>(mut visitor: F)
where
    F: FnMut(&'static mut mmtk::Mutator<RustVM>),
{
    let registrations: Vec<_> = MUTATOR_MAP
        .get_or_init(DashMap::new)
        .iter()
        .map(|entry| entry.value().mutator)
        .collect();

    for ptr in registrations {
        unsafe {
            visitor(&mut *ptr);
        }
    }
}

pub fn visit_all_threads<F>(mut visitor: F)
where
    F: FnMut(&MutatorThread),
{
    let threads: Vec<_> = MUTATOR_MAP
        .get_or_init(DashMap::new)
        .iter()
        .map(|entry| entry.value().thread.clone())
        .collect();

    for thread in threads.iter() {
        visitor(thread);
    }
}
