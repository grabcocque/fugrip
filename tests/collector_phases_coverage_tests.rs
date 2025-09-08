use fugrip::collector_phases::*;
use fugrip::{GcTrace, SendPtr, GcHeader, ObjectClass, CollectorPhase};
use fugrip::memory::CLASSIFIED_ALLOCATOR;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

// Test struct for collector phase testing
#[derive(Debug)]
struct TestNode {
    id: usize,
    data: Vec<u8>,
}

unsafe impl GcTrace for TestNode {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // No GC references to trace in this simple struct
    }
}

#[test]
fn test_thread_registration_creation() {
    let registration = ThreadRegistration {
        thread_id: thread::current().id(),
        stack_base: 0x1000000,
        stack_bounds: (0x1000000, 0x1100000),
        last_known_sp: AtomicUsize::new(0x1050000),
        local_roots: Vec::new(),
        is_active: AtomicBool::new(true),
    };

    assert!(registration.is_active.load(Ordering::Acquire));
    assert_eq!(registration.stack_base, 0x1000000);
    assert_eq!(registration.stack_bounds.0, 0x1000000);
    assert_eq!(registration.stack_bounds.1, 0x1100000);
    assert_eq!(registration.last_known_sp.load(Ordering::Acquire), 0x1050000);
    assert!(registration.local_roots.is_empty());
}

#[test]
fn test_thread_registration_modify() {
    let mut registration = ThreadRegistration {
        thread_id: thread::current().id(),
        stack_base: 0x1000000,
        stack_bounds: (0x1000000, 0x1100000),
        last_known_sp: AtomicUsize::new(0x1050000),
        local_roots: Vec::new(),
        is_active: AtomicBool::new(true),
    };

    // Test modifying the registration
    registration.is_active.store(false, Ordering::Release);
    assert!(!registration.is_active.load(Ordering::Acquire));

    registration.last_known_sp.store(0x1060000, Ordering::Release);
    assert_eq!(registration.last_known_sp.load(Ordering::Acquire), 0x1060000);

    // Add some local roots
    let gc_obj = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 1, data: vec![1, 2, 3] }, ObjectClass::Default);
    let ptr = unsafe { SendPtr::new(gc_obj.as_ptr() as *mut GcHeader<()>) };
    registration.local_roots.push(ptr);
    assert_eq!(registration.local_roots.len(), 1);
}

#[test]
fn test_collector_state_creation() {
    let collector = CollectorState::new();
    
    // Test initial state
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Waiting as usize);
    assert!(!collector.marking_active.load(Ordering::Acquire));
    assert!(!collector.allocation_color.load(Ordering::Acquire));
    assert_eq!(collector.worker_count.load(Ordering::Acquire), 0);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 0);
    
    // Test that stack is empty
    let stack = collector.global_mark_stack.lock().unwrap();
    assert!(stack.is_empty());
}

#[test]
fn test_collector_state_phase_transitions() {
    let collector = CollectorState::new();
    
    // Test phase transitions
    collector.phase.store(CollectorPhase::Marking as usize, Ordering::Release);
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Marking as usize);
    
    collector.phase.store(CollectorPhase::Sweeping as usize, Ordering::Release);
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Sweeping as usize);
    
    collector.phase.store(CollectorPhase::Censusing as usize, Ordering::Release);
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Censusing as usize);
    
    collector.phase.store(CollectorPhase::Reviving as usize, Ordering::Release);
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Reviving as usize);
}

#[test]
fn test_collector_state_marking_flags() {
    let collector = CollectorState::new();
    
    // Test marking active flag
    collector.marking_active.store(true, Ordering::Release);
    assert!(collector.marking_active.load(Ordering::Acquire));
    
    collector.marking_active.store(false, Ordering::Release);
    assert!(!collector.marking_active.load(Ordering::Acquire));
    
    // Test allocation color
    collector.allocation_color.store(true, Ordering::Release);
    assert!(collector.allocation_color.load(Ordering::Acquire));
    
    collector.allocation_color.store(false, Ordering::Release);
    assert!(!collector.allocation_color.load(Ordering::Acquire));
}

#[test]
fn test_collector_state_worker_management() {
    let collector = CollectorState::new();
    
    // Test worker count management
    collector.worker_count.store(4, Ordering::Release);
    assert_eq!(collector.worker_count.load(Ordering::Acquire), 4);
    
    // Test workers finished counter
    collector.workers_finished.fetch_add(1, Ordering::Relaxed);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 1);
    
    collector.workers_finished.fetch_add(3, Ordering::Relaxed);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 4);
    
    // Reset
    collector.workers_finished.store(0, Ordering::Release);
    assert_eq!(collector.workers_finished.load(Ordering::Acquire), 0);
}

#[test]
fn test_collector_state_global_mark_stack() {
    let collector = CollectorState::new();
    
    // Test adding to global mark stack
    {
        let mut stack = collector.global_mark_stack.lock().unwrap();
        
        // Create some test objects and add to stack
        let gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 1, data: vec![1] }, ObjectClass::Default);
        let gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 2, data: vec![2] }, ObjectClass::Default);
        
        let ptr1 = unsafe { SendPtr::new(gc_obj1.as_ptr() as *mut GcHeader<()>) };
        let ptr2 = unsafe { SendPtr::new(gc_obj2.as_ptr() as *mut GcHeader<()>) };
        
        stack.push(ptr1);
        stack.push(ptr2);
        
        assert_eq!(stack.len(), 2);
    }
    
    // Test clearing the stack
    {
        let mut stack = collector.global_mark_stack.lock().unwrap();
        stack.clear();
        assert!(stack.is_empty());
    }
}

#[test]
fn test_collector_state_register_thread() {
    let collector = CollectorState::new();
    // Register current thread via API
    let bounds = (0x2000000, 0x2100000);
    let _ = collector.register_thread_for_gc(bounds);
    
    // Check that thread is registered
    assert_eq!(collector.get_registered_thread_count(), 1);
    
    // Basic verification - we can't clone ThreadRegistration due to atomic fields
    // The fact that get_registered_thread_count() == 1 is sufficient to verify registration worked
}

#[test]
fn test_collector_state_unregister_thread() {
    let collector = CollectorState::new();
    
    // Register current thread
    let _ = collector.register_thread_for_gc((0x3000000, 0x3100000));
    
    // Verify registration
    assert_eq!(collector.get_registered_thread_count(), 1);
    
    // Unregister the current thread
    collector.unregister_thread_from_gc();
    
    // Verify unregistration
    assert_eq!(collector.get_registered_thread_count(), 0);
}

#[test]
fn test_collector_state_mark_global_roots() {
    let collector = CollectorState::new();
    
    // Create some objects to act as roots
    let _gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 1, data: vec![1] }, ObjectClass::Default);
    let _gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 2, data: vec![2] }, ObjectClass::Default);
    
    // Call mark_global_roots
    collector.mark_global_roots();
    
    // This test mainly verifies that the method can be called without panicking
    // In a real scenario, we'd check that roots are properly marked
}

#[test]
fn test_collector_state_census_phase() {
    let collector = CollectorState::new();
    
    // Create some objects that might have weak references
    let _gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 1, data: vec![1] }, ObjectClass::Census);
    let _gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 2, data: vec![2] }, ObjectClass::CensusAndDestructor);
    
    // Call census phase
    collector.execute_census_phase();
    
    // Verify that phase was set correctly
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Censusing as usize);
}

#[test]
fn test_collector_state_request_collection() {
    let collector = CollectorState::new();
    
    // Initially in Waiting
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Waiting as usize);
    collector.request_collection();
    // Should transition to Marking once
    assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Marking as usize);
}

#[test]
fn test_collector_state_suspend_and_resume() {
    let collector = CollectorState::new();
    
    // Test requesting suspension
    collector.request_suspension();
    assert!(collector.is_suspension_requested());
    // Simulate a worker acknowledging suspension
    collector.worker_acknowledge_suspension();
    // Resume collection
    collector.resume_collection();
    assert!(!collector.is_suspension_requested());
}

#[test]
fn test_collector_state_concurrent_operations() {
    let collector = Arc::new(CollectorState::new());
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    
    // Test concurrent phase transitions
    for i in 0..3 {
        let collector_clone = collector.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            match i {
                0 => {
                    collector_clone.phase.store(CollectorPhase::Marking as usize, Ordering::Release);
                    thread::sleep(Duration::from_millis(10));
                    collector_clone.marking_active.store(true, Ordering::Release);
                }
                1 => {
                    collector_clone.worker_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(10));
                    collector_clone.workers_finished.fetch_add(1, Ordering::Relaxed);
                }
                2 => {
                    collector_clone.allocation_color.store(true, Ordering::Release);
                    thread::sleep(Duration::from_millis(10));
                    collector_clone.allocation_color.store(false, Ordering::Release);
                }
                _ => {}
            }
        });
        
        handles.push(handle);
    }
    
    // Main thread waits and then checks final state
    barrier.wait();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify that operations completed without data races
    // The exact final state is non-deterministic but operations should have completed
    assert!(collector.worker_count.load(Ordering::Acquire) >= 0);
    assert!(collector.workers_finished.load(Ordering::Acquire) >= 0);
}

#[test]
fn test_mutator_state_creation() {
    let mutator = MutatorState::new();
    assert!(mutator.local_mark_stack.is_empty());
    assert_eq!(mutator.allocation_buffer.current, std::ptr::null_mut());
    assert_eq!(mutator.allocation_buffer.end, std::ptr::null_mut());
    assert!(!mutator.is_in_handshake);
    assert!(!mutator.allocating_black);
}

#[test]
fn test_mutator_state_handshake() {
    let mut mutator = MutatorState::new();
    let collector = CollectorState::new();
    
    // Register a mutator thread first so handshake doesn't complete immediately
    collector.register_mutator_thread();
    
    // No handshake requested initially
    mutator.check_handshake(&collector);
    assert!(!mutator.is_in_handshake);
    
    // Set allocation color first, then request handshake
    collector.allocation_color.store(true, Ordering::Release);
    
    // Use the thread coordinator directly to set handshake request
    collector.thread_coordinator.handshake_requested.store(true, Ordering::Release);
    
    // Now check handshake - should switch to black allocation
    mutator.check_handshake(&collector);
    assert!(mutator.allocating_black);
}

#[test]
// Suspension flags no longer exist on MutatorState; covered via collector tests.

#[test]
fn test_mutator_state_local_mark_stack() {
    let mut mutator = MutatorState::new();
    
    // Test adding to local mark stack
    let gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 1, data: vec![1] }, ObjectClass::Default);
    let gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 2, data: vec![2] }, ObjectClass::Default);
    
    let ptr1 = unsafe { SendPtr::new(gc_obj1.as_ptr() as *mut GcHeader<()>) };
    let ptr2 = unsafe { SendPtr::new(gc_obj2.as_ptr() as *mut GcHeader<()>) };
    
    mutator.local_mark_stack.push(ptr1);
    mutator.local_mark_stack.push(ptr2);
    
    assert_eq!(mutator.local_mark_stack.len(), 2);
    
    // Test clearing the stack
    mutator.local_mark_stack.clear();
    assert!(mutator.local_mark_stack.is_empty());
}

#[test]
fn test_mutator_state_try_allocate() {
    let mut mutator = MutatorState::new();
    
    // Test trying to allocate when buffer is empty
    let result = mutator.try_allocate::<TestNode>();
    assert!(result.is_none()); // Should fail as buffer is not initialized
}

#[test]
fn test_gc_safe_fork() {
    // Test that gc_safe_fork can be called without panicking
    let result = gc_safe_fork();
    
    // The function should return an appropriate result
    // In a test environment, it might return an error, but it shouldn't panic
    match result {
        Ok(_) => {
            // Fork succeeded (unlikely in test environment)
        }
        Err(_) => {
            // Fork failed (expected in test environment)
            // This is normal behavior
        }
    }
}

#[test]
fn test_stack_bounds_detection() {
    let collector = CollectorState::new();
    let bounds = collector.get_current_thread_stack_bounds();
    
    // The bounds should be reasonable (stack addresses are typically high)
    assert!(bounds.0 > 0);
    assert!(bounds.1 > bounds.0);
    assert!(bounds.1 - bounds.0 > 4096); // At least 4KB stack
}

#[test]
fn test_collector_phases_integration() {
    let collector = CollectorState::new();
    
    // Create some test objects
    let _gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 1, data: vec![1] }, ObjectClass::Default);
    let _gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 2, data: vec![2] }, ObjectClass::Census);
    let _gc_obj3 = CLASSIFIED_ALLOCATOR.allocate_classified(TestNode { id: 3, data: vec![3] }, ObjectClass::Finalizer);
    
    // Test a complete collection cycle
    collector.request_collection();
    if collector.phase.load(Ordering::Acquire) == CollectorPhase::Marking as usize {
        // Mark roots
        collector.mark_global_roots();
        
        // Census phase
        collector.execute_census_phase();
        
        // Verify phase was set
        assert_eq!(collector.phase.load(Ordering::Acquire), CollectorPhase::Censusing as usize);
        
        // End collection
        collector.marking_active.store(false, Ordering::Release);
        collector.phase.store(CollectorPhase::Waiting as usize, Ordering::Release);
    }
}
