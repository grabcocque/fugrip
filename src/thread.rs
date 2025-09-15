//! Thread registry and mutator state used by the runtime.

use std::{
    fmt,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

#[derive(Default, Debug)]
struct MutatorInner {
    id: usize,
    safepoint_requested: AtomicBool,
    parked: AtomicBool,
    lock: Mutex<()>,
    cv: Condvar,
}

impl MutatorInner {
    fn new(id: usize) -> Self {
        Self {
            id,
            safepoint_requested: AtomicBool::new(false),
            parked: AtomicBool::new(false),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    fn request_safepoint(&self) {
        self.safepoint_requested.store(true, Ordering::SeqCst);
    }

    fn clear_safepoint(&self) {
        self.safepoint_requested.store(false, Ordering::SeqCst);
        self.cv.notify_all();
    }

    fn wait_until_parked(&self) {
        while !self.parked.load(Ordering::SeqCst) {
            thread::yield_now();
        }
    }
}

pub struct MutatorThread {
    inner: Arc<MutatorInner>,
}

impl MutatorThread {
    pub fn new(id: usize) -> Self {
        Self {
            inner: Arc::new(MutatorInner::new(id)),
        }
    }

    fn from_inner(inner: Arc<MutatorInner>) -> Self {
        Self { inner }
    }

    pub fn id(&self) -> usize {
        self.inner.id
    }

    pub fn poll_safepoint(&self) {
        if self.inner.safepoint_requested.load(Ordering::SeqCst) {
            let mut guard = self.inner.lock.lock().unwrap();
            self.inner.parked.store(true, Ordering::SeqCst);
            while self.inner.safepoint_requested.load(Ordering::SeqCst) {
                guard = self.inner.cv.wait(guard).unwrap();
            }
            self.inner.parked.store(false, Ordering::SeqCst);
        }
    }

    pub(crate) fn request_safepoint(&self) {
        self.inner.request_safepoint();
    }

    pub(crate) fn clear_safepoint(&self) {
        self.inner.clear_safepoint();
    }

    pub(crate) fn wait_until_parked(&self) {
        self.inner.wait_until_parked();
    }
}

impl Clone for MutatorThread {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Default for MutatorThread {
    fn default() -> Self {
        Self::new(0)
    }
}

impl fmt::Debug for MutatorThread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MutatorThread").field(&self.id()).finish()
    }
}

#[derive(Clone, Default, Debug)]
pub struct ThreadRegistry {
    mutators: Arc<Mutex<Vec<Arc<MutatorInner>>>>,
}

impl ThreadRegistry {
    pub fn new() -> Self {
        Self {
            mutators: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Global registry accessor used by the VM binding.
    pub fn global() -> &'static ThreadRegistry {
        static GLOBAL: OnceLock<ThreadRegistry> = OnceLock::new();
        GLOBAL.get_or_init(ThreadRegistry::new)
    }

    pub fn register(&self, thread: MutatorThread) {
        self.mutators.lock().unwrap().push(thread.inner.clone());
    }

    pub fn iter(&self) -> Vec<MutatorThread> {
        self.mutators
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .map(MutatorThread::from_inner)
            .collect()
    }
}
