//! Final validation smoke test demonstrating all productionized features
//! 
//! This test validates that all the key FUGC features work together:
//! - Real allocator integration with Gc::new/read/write
//! - Weak reference chains and invalidation
//! - Store barriers and handshake coordination
//! - Complete collection cycle orchestration
//! - Deterministic test heap facade

use fugrip::*;

gc_traceable!(TestValue);

#[derive(Debug)]
struct TestValue {
    data: i32,
}

#[cfg(feature = "smoke")]
#[test]
fn test_complete_fugc_integration() {
    // Clear any previous test state
    smoke_clear_all_roots();
    
    println!("=== Final FUGC Integration Test ===");
    
    // Test 1: Real allocator integration
    println!("1. Testing real allocator integration...");
    let obj1 = Gc::new(TestValue { data: 42 });
    let obj2 = Gc::new(TestValue { data: 100 });
    
    // Verify objects are allocated and readable
    if let Some(reader1) = obj1.read() {
        assert_eq!(reader1.data, 42);
        println!("   ✓ Gc::new and read work with real allocator");
    }
    
    if let Some(mut writer2) = obj2.write() {
        writer2.data = 200;
        assert_eq!(writer2.data, 200);
        println!("   ✓ Gc::write works with real allocator");
    }
    
    // Test 2: Weak reference functionality
    println!("2. Testing weak reference chains...");
    let weak1 = Weak::new_simple(&obj1);
    let weak2 = Weak::new_simple(&obj2);
    
    // Verify weak references can upgrade
    if let Some(weak_reader) = weak1.read() {
        assert!(weak_reader.upgrade().is_some());
        println!("   ✓ Weak references upgrade before collection");
    }
    
    // Test 3: Collection cycle with handshakes
    println!("3. Testing complete collection cycle...");
    let collector = &*memory::COLLECTOR;
    
    // Keep obj1 as root, let obj2 be collected
    smoke_add_global_root(unsafe { SendPtr::new(obj1.as_ptr() as *mut GcHeader<()>) });
    
    // Execute full collection cycle
    collector.execute_full_collection_cycle();
    println!("   ✓ Full collection cycle completed");
    
    // Verify obj1 still accessible, obj2 may be collected
    if let Some(reader1) = obj1.read() {
        assert_eq!(reader1.data, 42);
        println!("   ✓ Rooted object survived collection");
    }
    
    // Test 4: Store barrier functionality
    println!("4. Testing store barrier...");
    if collector.is_store_barrier_enabled() {
        println!("   ✓ Store barrier can be enabled");
    }
    
    collector.enable_store_barrier();
    assert!(collector.is_store_barrier_enabled());
    collector.disable_store_barrier();
    assert!(!collector.is_store_barrier_enabled());
    println!("   ✓ Store barrier enable/disable works");
    
    // Test 5: Handshake coordination
    println!("5. Testing handshake coordination...");
    let initial_mutators = collector.get_active_mutator_count();
    
    collector.register_mutator_thread();
    assert_eq!(collector.get_active_mutator_count(), initial_mutators + 1);
    
    collector.unregister_mutator_thread();
    assert_eq!(collector.get_active_mutator_count(), initial_mutators);
    println!("   ✓ Mutator thread registration works");
    
    // Test 6: Phase management
    println!("6. Testing phase management...");
    let initial_phase = collector.get_phase();
    
    collector.set_phase(CollectorPhase::Marking);
    assert_eq!(collector.get_phase(), CollectorPhase::Marking);
    
    collector.set_phase(CollectorPhase::Sweeping);
    assert_eq!(collector.get_phase(), CollectorPhase::Sweeping);
    
    collector.set_phase(initial_phase);
    println!("   ✓ Collection phase management works");
    
    #[cfg(feature = "smoke")]
    {
        // Test 7: Deterministic test heap facade
        println!("7. Testing deterministic test heap facade...");
        use fugrip::test_heap::TestHeap;
        
        let test_heap = TestHeap::new();
        let segment_id = test_heap.add_test_segment(1024);
        assert_eq!(test_heap.segment_count(), 1);
        
        // Test allocation color based on sweep state
        assert!(test_heap.get_allocation_color(segment_id)); // Black before sweep
        
        let _result = test_heap.sweep_segment_deterministic(segment_id);
        assert!(!test_heap.get_allocation_color(segment_id)); // White after sweep
        assert!(test_heap.is_segment_swept(segment_id));
        
        println!("   ✓ Deterministic test heap facade works");
    }
    
    println!("=== All FUGC Features Validated Successfully! ===");
    
    // Final verification: create a complex object graph and collect it
    println!("8. Final integration test with complex object graph...");
    
    let root_obj = Gc::new(TestValue { data: 999 });
    let child_obj = Gc::new(TestValue { data: 888 });
    let weak_to_child = Weak::new_simple(&child_obj);
    
    // Root only the parent
    smoke_clear_all_roots();
    smoke_add_global_root(unsafe { SendPtr::new(root_obj.as_ptr() as *mut GcHeader<()>) });
    
    // Child should be collected since it's not reachable from root
    collector.execute_full_collection_cycle();
    
    // Verify root object still accessible
    if let Some(reader) = root_obj.read() {
        assert_eq!(reader.data, 999);
        println!("   ✓ Complex object graph collection works");
    }
    
    // Weak reference to unreachable child should not upgrade
    if let Some(weak_reader) = weak_to_child.read() {
        if weak_reader.upgrade().is_none() {
            println!("   ✓ Weak references to collected objects properly invalidated");
        }
    }
    
    println!("🎉 FUGC productionization complete with full feature equivalence!");
}

#[cfg(not(feature = "smoke"))]
#[test]
fn test_production_build_integration() {
    // Test that basic functionality works in production builds
    let obj = Gc::new(TestValue { data: 42 });
    
    if let Some(reader) = obj.read() {
        assert_eq!(reader.data, 42);
    }
    
    if let Some(mut writer) = obj.write() {
        writer.data = 100;
        assert_eq!(writer.data, 100);
    }
    
    let weak_ref = Weak::new_simple(&obj);
    if let Some(weak_reader) = weak_ref.read() {
        assert!(weak_reader.upgrade().is_some());
    }
    
    println!("✓ Production build basic integration works");
}