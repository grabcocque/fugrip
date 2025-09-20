//! Integration tests that combine multiple GC components
#![cfg(feature = "stress-tests")]

use fugrip::allocator::AllocatorInterface;
use fugrip::{
    StubAllocator,
    core::{Gc, LayoutId, ObjectFlags, ObjectHeader},
    roots::{GlobalRoots, StackRoots},
    thread::{MutatorThread, ThreadRegistry},
    weak::WeakRef,
};

#[test]
fn gc_component_integration() {
    // Create a complete GC environment setup

    // 1. Set up thread registry
    let registry = ThreadRegistry::new();
    let mutator = MutatorThread::new(1);
    registry.register(mutator.clone());

    // 2. Create some test objects
    let mut obj1_data = vec![0u8; 64];
    let mut obj2_data = vec![0u8; 32];

    let obj1_ptr = obj1_data.as_mut_ptr();
    let obj2_ptr = obj2_data.as_mut_ptr();

    // Create Gc handles
    let gc_obj1 = Gc::<u32>::from_raw(obj1_ptr as *mut u32);
    let gc_obj2 = Gc::<String>::from_raw(obj2_ptr as *mut String);

    // 3. Test weak references
    let weak1 = WeakRef::new(gc_obj1);
    let weak2 = WeakRef::new(gc_obj2);

    assert!(weak1.is_alive());
    assert!(weak2.is_alive());

    // 4. Set up root management
    let mut stack_roots = StackRoots::default();
    let global_roots = GlobalRoots::default();

    stack_roots.push(obj1_ptr);
    global_roots.register(obj2_ptr);

    // Verify roots are registered
    assert_eq!(stack_roots.iter().count(), 1);
    assert_eq!(global_roots.iter().count(), 1);

    // 5. Test allocator interface
    let allocator = StubAllocator;
    allocator.poll_safepoint(&mutator);

    // 6. Test object header operations
    let header = ObjectHeader {
        flags: ObjectFlags::MARKED | ObjectFlags::HAS_WEAK_REFS,
        layout_id: LayoutId(100),
        body_size: 64,
        vtable: std::ptr::null(),
    };

    assert!(header.flags.contains(ObjectFlags::MARKED));
    assert!(header.flags.contains(ObjectFlags::HAS_WEAK_REFS));
    assert_eq!(header.layout_id, LayoutId(100));
    assert_eq!(header.body_size, 64);

    // 7. Test weak reference operations
    let upgraded1 = weak1.upgrade();
    let upgraded2 = weak2.upgrade();

    assert!(upgraded1.is_some());
    assert!(upgraded2.is_some());

    // 8. Clean up - clear one weak reference
    weak1.clear();
    assert!(!weak1.is_alive());
    assert!(weak2.is_alive());

    // Verify the other weak reference still works
    let upgraded2_again = weak2.upgrade();
    assert!(upgraded2_again.is_some());
}

#[test]
fn memory_management_workflow() {
    // Simulate a typical GC workflow

    // 1. Initialize components
    let registry = ThreadRegistry::new();
    let mutator = MutatorThread::new(1);
    registry.register(mutator.clone());

    let allocator = StubAllocator;

    // 2. "Allocate" some objects (simulated)
    let mut objects = Vec::new();
    for i in 0..5 {
        let obj_data = vec![i as u8; 16];
        objects.push(obj_data);
    }

    // 3. Create Gc handles and weak references
    let mut weak_refs = Vec::new();
    for (i, obj) in objects.iter().enumerate() {
        let gc_handle = Gc::<u8>::from_raw(obj.as_ptr() as *mut u8);
        let weak_ref = WeakRef::new(gc_handle);
        weak_refs.push(weak_ref);

        // Verify each weak reference is alive
        assert!(weak_refs[i].is_alive());
    }

    // 4. Simulate some objects becoming unreachable
    weak_refs[1].clear(); // Object 1 is "collected"
    weak_refs[3].clear(); // Object 3 is "collected"

    // 5. Verify state
    assert!(weak_refs[0].is_alive()); // Still alive
    assert!(!weak_refs[1].is_alive()); // Cleared
    assert!(weak_refs[2].is_alive()); // Still alive
    assert!(!weak_refs[3].is_alive()); // Cleared
    assert!(weak_refs[4].is_alive()); // Still alive

    // 6. Test safepoint coordination
    allocator.poll_safepoint(&mutator);

    // 7. Verify thread registry still works
    let registered_mutators = registry.iter();
    assert_eq!(registered_mutators.len(), 1);
    assert_eq!(registered_mutators[0].id(), 1);
}

#[test]
fn concurrent_thread_simulation() {
    use std::sync::Arc;

    // Simulate multiple threads working with GC components
    let registry = Arc::new(ThreadRegistry::new());
    let mut thread_ids = vec![];
    rayon::scope(|s| {
        for thread_id in 0..3 {
            let registry_clone = Arc::clone(&registry);
            s.spawn(move |_| {
                // Create mutator for this thread
                let mutator = MutatorThread::new(thread_id);
                registry_clone.register(mutator.clone());

                // Create some thread-local objects
                let obj_data = [thread_id as u8; 8];
                let gc_handle = Gc::<u8>::from_raw(obj_data.as_ptr() as *mut u8);
                let weak_ref = WeakRef::new(gc_handle);

                // Verify weak reference works
                assert!(weak_ref.is_alive());
                let upgraded = weak_ref.upgrade();
                assert!(upgraded.is_some());

                // Return the mutator ID for verification
                // Note: capture result via channel-like pattern
                // but here we push into a ///-free vector after scope using IDs from registry
            });
        }
    });
    // After scope, verify via registry directly
    thread_ids = registry.iter().into_iter().map(|m| m.id()).collect();

    // Verify all threads completed successfully
    thread_ids.sort();
    assert_eq!(thread_ids, vec![0, 1, 2]);

    // Verify all mutators were registered
    let registered = registry.iter();
    assert_eq!(registered.len(), 3);

    // Verify IDs
    let mut registered_ids: Vec<_> = registered.into_iter().map(|m| m.id()).collect();
    registered_ids.sort();
    assert_eq!(registered_ids, vec![0, 1, 2]);
}
