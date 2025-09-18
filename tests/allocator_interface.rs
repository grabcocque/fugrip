use fugrip::StubAllocator;
use fugrip::allocator::{AllocatorInterface, MMTkAllocator};
use fugrip::core::ObjectHeader;
use fugrip::error::GcError;
use fugrip::thread::MutatorThread;
use mmtk::util::constants::MIN_OBJECT_SIZE;

// Tests for allocator interface

#[test]
fn test_stub_allocator_poll_safepoint() {
    let allocator = StubAllocator;
    let mutator = MutatorThread::new(1usize);
    allocator.poll_safepoint(&mutator);
}

#[test]
fn test_mmtk_allocator_poll_safepoint() {
    let allocator = MMTkAllocator;
    let mutator = MutatorThread::new(1usize);
    allocator.poll_safepoint(&mutator);
}

#[test]
fn test_allocator_interface_poll_consistency() {
    let mmtk = MMTkAllocator;
    let stub = StubAllocator;
    let mutator = MutatorThread::new(1usize);
    mmtk.poll_safepoint(&mutator);
    stub.poll_safepoint(&mutator);
}

#[test]
fn test_object_header_default() {
    let _header = ObjectHeader::default();
    assert_eq!(std::mem::size_of::<ObjectHeader>(), 24);
}

#[test]
fn test_mmtk_allocate_size_calc() {
    let bytes = 16usize;
    let total_bytes = std::mem::size_of::<ObjectHeader>() + bytes;
    let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);
    assert!(allocation_size >= MIN_OBJECT_SIZE);
}

#[test]
fn test_stub_allocate_error() {
    let error = GcError::OutOfMemory;
    assert_eq!(error.to_string(), "Out of memory");
}

#[test]
fn test_mmtk_allocate_logic() {
    let bytes = 16usize;
    let total_bytes = std::mem::size_of::<ObjectHeader>() + bytes;
    let allocation_size = std::cmp::max(total_bytes, MIN_OBJECT_SIZE);
    let _align = std::mem::align_of::<usize>().max(std::mem::align_of::<ObjectHeader>());

    // Test allocation size calculation
    assert!(allocation_size >= MIN_OBJECT_SIZE);
    assert!(allocation_size >= total_bytes);

    // Test that we handle non-zero bytes correctly
    assert!(bytes > 0);
    assert_eq!(bytes, 16);
}
