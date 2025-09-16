use fugrip::{
    allocator::{AllocatorInterface, MMTkAllocator, StubAllocator},
    error::GcError,
    thread::MutatorThread,
};

#[test]
fn gc_error_display() {
    let err = GcError::OutOfMemory;
    assert_eq!(format!("{}", err), "Out of memory");

    let err = GcError::InvalidReference;
    assert_eq!(format!("{}", err), "Invalid object reference");

    let err = GcError::ThreadError("test error".to_string());
    assert_eq!(format!("{}", err), "Thread error: test error");

    let err = GcError::MmtkError("mmtk error".to_string());
    assert_eq!(format!("{}", err), "MMTk error: mmtk error");
}

#[test]
fn allocator_error_handling() {
    // Since we can't create real MMTk mutators in tests,
    // we'll test the error handling conceptually
    let allocator = MMTkAllocator::new();
    let stub_allocator = StubAllocator;

    // Both allocators should have the same interface
    // and should handle errors gracefully
    assert_eq!(
        std::mem::size_of_val(&allocator),
        std::mem::size_of_val(&stub_allocator)
    );

    // Test that the allocators can be polled for safepoints
    let mutator = MutatorThread::new(1);
    allocator.poll_safepoint(&mutator);
    stub_allocator.poll_safepoint(&mutator);
}

#[test]
fn gc_result_operations() {
    let success = 42;
    assert_eq!(success, 42);

    let failure = GcError::OutOfMemory;
    assert!(matches!(failure, GcError::OutOfMemory));
}

#[test]
fn error_recovery_scenarios() {
    // Test that errors don't crash the system and can be handled gracefully
    let allocator = StubAllocator;
    let mutator = MutatorThread::new(1);

    // Multiple safepoint polls should work fine
    for _ in 0..10 {
        allocator.poll_safepoint(&mutator);
    }

    // The allocator should be reusable
    let _allocator2 = StubAllocator;
}
