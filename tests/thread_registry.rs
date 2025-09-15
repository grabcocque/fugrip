use fugrip::thread::{MutatorThread, ThreadRegistry};

#[test]
fn registers_and_enumerates_mutators() {
    let registry = ThreadRegistry::new();
    registry.register(MutatorThread::new(1));
    registry.register(MutatorThread::new(2));

    let mut ids: Vec<_> = registry.iter().into_iter().map(|m| m.id()).collect();
    ids.sort_unstable();

    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn safepoint_poll_is_noop_when_not_requested() {
    let mutator = MutatorThread::new(42);
    // Should return immediately when no safepoint is requested.
    mutator.poll_safepoint();
}
