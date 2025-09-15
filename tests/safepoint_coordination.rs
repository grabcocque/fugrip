use fugrip::thread::{MutatorThread, ThreadRegistry};
use std::sync::Arc;
use std::thread;

#[test]
fn safepoint_request_and_clear() {
    let mutator = MutatorThread::new(1);

    // Initially no safepoint should be requested
    mutator.poll_safepoint(); // Should return immediately

    // We can't test requesting safepoints from outside the thread,
    // but we can test the polling interface
    let _mutator = mutator;
}

#[test]
fn thread_registry_safepoint_coordination() {
    let registry = ThreadRegistry::new();
    let mutator1 = MutatorThread::new(1);
    let mutator2 = MutatorThread::new(2);

    registry.register(mutator1.clone());
    registry.register(mutator2.clone());

    // Test that we can iterate and call methods on all mutators
    for mutator in registry.iter() {
        mutator.poll_safepoint();
    }

    // Verify we have the right number of mutators
    assert_eq!(registry.iter().len(), 2);
}

#[test]
fn mutator_thread_operations() {
    let mutator = MutatorThread::new(42);

    // Test ID retrieval
    assert_eq!(mutator.id(), 42);

    // Test cloning
    let mutator2 = mutator.clone();
    assert_eq!(mutator2.id(), 42);

    // Test that they're different instances but refer to same inner state
    assert_eq!(mutator.id(), mutator2.id());
}

#[test]
fn safepoint_state_transitions() {
    let mutator = MutatorThread::new(1);

    // Test multiple poll calls work
    for _ in 0..5 {
        mutator.poll_safepoint();
    }

    // Test that the mutator can be used after polling
    let _id = mutator.id();
}

#[test]
fn thread_registry_thread_safety() {
    let registry = Arc::new(ThreadRegistry::new());
    let mut handles = vec![];

    // Spawn multiple threads that register mutators
    for i in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let mutator = MutatorThread::new(i);
            registry_clone.register(mutator);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all mutators were registered
    let mutators = registry.iter();
    assert_eq!(mutators.len(), 5);

    // Verify IDs are unique
    let mut ids: Vec<_> = mutators.into_iter().map(|m| m.id()).collect();
    ids.sort();
    assert_eq!(ids, vec![0, 1, 2, 3, 4]);
}