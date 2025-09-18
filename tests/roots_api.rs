use fugrip::{
    roots::RustScanning,
    thread::{MutatorThread, ThreadRegistry},
};
use std::sync::Arc;

#[test]
fn for_each_mutator_invokes_callback() {
    let registry = Arc::new(ThreadRegistry::new());
    registry.register(MutatorThread::new(1));
    registry.register(MutatorThread::new(2));

    let scanning = RustScanning::new(Arc::clone(&registry));
    let mut collected = Vec::new();
    scanning.for_each_mutator(|mutator| collected.push(mutator.id()));
    collected.sort_unstable();
    assert_eq!(collected, vec![1, 2]);
}
