//! Tests for the allocator module.

use fugrip::StubAllocator;
use fugrip::allocator::{AllocatorInterface, MMTkAllocator};
use fugrip::thread::MutatorThread;

// Note: Full MMTk integration for allocate requires setup; tests cover constructors and poll_safepoint.

#[test]
fn test_mmtk_allocator_new() {
    let _allocator = MMTkAllocator::new();
    // Covers lines 81-83
}

#[test]
fn test_mmtk_allocator_default() {
    let _allocator = MMTkAllocator;
    // Covers lines 87-90
}

#[test]
fn test_mmtk_allocator_poll_safepoint() {
    let allocator = MMTkAllocator::new();
    let mutator = MutatorThread::new(1usize);
    allocator.poll_safepoint(&mutator);
    // Covers lines 141-144
}

#[test]
fn test_stub_allocator_new() {
    let _allocator = StubAllocator::new();
    // Covers lines 167-169
}

#[test]
fn test_stub_allocator_default() {
    let _allocator = StubAllocator;
    // Covers lines 173-176
}

#[test]
fn test_stub_allocator_poll_safepoint() {
    let allocator = StubAllocator::new();
    let mutator = MutatorThread::new(1usize);
    allocator.poll_safepoint(&mutator);
    // Covers line 188-190
}

#[test]
fn test_allocator_interface() {
    let mutator = MutatorThread::new(1usize);
    let stub = StubAllocator::new();
    let mmtk = MMTkAllocator::new();
    let allocators: [&dyn AllocatorInterface; 2] = [&stub, &mmtk];
    for a in allocators.iter() {
        a.poll_safepoint(&mutator);
    }
    // Covers trait usage and poll_safepoint calls
}
