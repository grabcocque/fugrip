use fugrip::{Gc, GcTrace, SendPtr, GcHeader, ObjectClass, Weak};
use fugrip::memory::*;
use fugrip::interfaces::AllocatorTrait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

// Test struct for memory operations
#[derive(Debug, Clone)]
struct TestData {
    id: usize,
    data: Vec<u8>,
    value: String,
}

unsafe impl GcTrace for TestData {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // No GC references to trace in this simple struct
    }
}

// Test struct with GC references
struct LinkedNode {
    id: usize,
    next: Option<Gc<LinkedNode>>,
}

unsafe impl GcTrace for LinkedNode {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        if let Some(ref next) = self.next {
            // In a real implementation, this would add the next node to the stack
            // For testing, we'll just verify the trace method is called
            unsafe {
                let ptr = next.as_ptr() as *mut GcHeader<()>;
                stack.push(SendPtr::new(ptr));
            }
        }
    }
}

#[test]
fn test_segmented_heap_new() {
    let heap = SegmentedHeap::new();
    
    // Test initial state
    assert_eq!(heap.segment_count(), 1); // Should start with one segment
    
    // Test that we can allocate on the new heap
    let test_obj = TestData {
        id: 1,
        data: vec![1, 2, 3, 4],
        value: "test".to_string(),
    };
    
    let gc_obj = heap.allocate(test_obj);
    assert!(!gc_obj.as_ptr().is_null());
}

#[test]
fn test_segmented_heap_add_segment() {
    let heap = SegmentedHeap::new();
    let initial_count = heap.segment_count();
    
    // Add a new segment
    let new_id = heap.add_segment();
    assert_eq!(new_id, 1); // Should be the second segment (0-indexed)
    
    // Check that segment count increased
    assert!(heap.segment_count() > initial_count);
}

#[test]
fn test_segmented_heap_multiple_allocations() {
    let heap = SegmentedHeap::new();
    
    // Allocate many objects to test heap growth
    let mut allocations = Vec::new();
    for i in 0..100 {
        let test_obj = TestData {
            id: i,
            data: vec![i as u8; 100], // 100 bytes each
            value: format!("test_{}", i),
        };
        
        let gc_obj = heap.allocate(test_obj);
        assert!(!gc_obj.as_ptr().is_null());
        allocations.push(gc_obj);
    }
    
    // Verify all allocations are valid
    assert_eq!(allocations.len(), 100);
    
    // Verify data integrity
    for (i, gc_obj) in allocations.iter().enumerate() {
        let obj_ref = gc_obj.read().unwrap();
        assert_eq!(obj_ref.id, i);
        assert_eq!(obj_ref.value, format!("test_{}", i));
    }
}

#[test]
fn test_gc_allocator_new_and_default() {
    let allocator1 = GcAllocator::new();
    let allocator2 = GcAllocator::default();
    
    // Both should be able to allocate objects
    let obj1 = allocator1.allocate_gc(TestData {
        id: 1,
        data: vec![1, 2, 3],
        value: "obj1".to_string(),
    });
    
    let obj2 = allocator2.allocate_gc(TestData {
        id: 2,
        data: vec![4, 5, 6],
        value: "obj2".to_string(),
    });
    
    assert!(!obj1.as_ptr().is_null());
    assert!(!obj2.as_ptr().is_null());
    
    // Verify data
    assert_eq!(obj1.read().unwrap().id, 1);
    assert_eq!(obj2.read().unwrap().id, 2);
}

#[test]
fn test_gc_allocator_stress_allocation() {
    let allocator = GcAllocator::new();
    
    // Stress test with many allocations
    for i in 0..1000 {
        let obj = allocator.allocate_gc(TestData {
            id: i,
            data: vec![i as u8; 50],
            value: format!("stress_{}", i),
        });
        
        assert!(!obj.as_ptr().is_null());
        
        // Verify data integrity for some objects
        if i % 100 == 0 {
            let obj_ref = obj.read().unwrap();
            assert_eq!(obj_ref.id, i);
            assert_eq!(obj_ref.value, format!("stress_{}", i));
        }
    }
}

#[test]
fn test_classified_allocator_all_classes() {
    let allocator = &*CLASSIFIED_ALLOCATOR;
    
    // Test allocation for each object class
    let classes = [
        ObjectClass::Default,
        ObjectClass::Destructor,
        ObjectClass::Census,
        ObjectClass::CensusAndDestructor,
        ObjectClass::Finalizer,
        ObjectClass::Weak,
    ];
    
    for (i, &class) in classes.iter().enumerate() {
        let obj = allocator.allocate_classified(
            TestData {
                id: i,
                data: vec![i as u8; 20],
                value: format!("class_{:?}", class),
            },
            class,
        );
        
        assert!(!obj.as_ptr().is_null());
        
        // Verify data
        let obj_ref = obj.read().unwrap();
        assert_eq!(obj_ref.id, i);
        assert_eq!(obj_ref.value, format!("class_{:?}", class));
        
        // Verify object is registered in correct object set
        let object_set = allocator.get_object_set(class);
        assert!(object_set.get_object_count() > 0);
    }
}

#[test]
fn test_classified_allocator_statistics() {
    let allocator = &*CLASSIFIED_ALLOCATOR;
    
    let initial_bytes = allocator.bytes_allocated();
    let initial_count = allocator.object_count();
    
    // Allocate some objects
    for i in 0..10 {
        let _obj = allocator.allocate_classified(
            TestData {
                id: i,
                data: vec![i as u8; 100], // 100 bytes each
                value: format!("stats_{}", i),
            },
            ObjectClass::Default,
        );
    }
    
    let final_bytes = allocator.bytes_allocated();
    let final_count = allocator.object_count();
    
    // Should have more objects and bytes
    assert!(final_count >= initial_count + 10);
    assert!(final_bytes >= initial_bytes); // Bytes might not increase linearly due to alignment
}

#[test]
fn test_object_set_operations() {
    let object_set = ObjectSet::new();
    
    // Test initial state
    assert_eq!(object_set.len(), 0);
    assert_eq!(object_set.get_object_count(), 0);
    assert!(object_set.is_empty());
    
    // Add some objects
    let gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(
        TestData { id: 1, data: vec![1], value: "1".to_string() },
        ObjectClass::Default
    );
    let gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(
        TestData { id: 2, data: vec![2], value: "2".to_string() },
        ObjectClass::Default
    );
    
    let ptr1 = unsafe { SendPtr::new(gc_obj1.as_ptr() as *mut GcHeader<()>) };
    let ptr2 = unsafe { SendPtr::new(gc_obj2.as_ptr() as *mut GcHeader<()>) };
    
    object_set.add(ptr1);
    object_set.add(ptr2);
    
    // Test state after adding
    assert_eq!(object_set.len(), 2);
    assert_eq!(object_set.get_object_count(), 2);
    assert!(!object_set.is_empty());
    
    // Test iteration
    let counter = AtomicUsize::new(0);
    object_set.for_each(|_ptr| {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(counter.load(Ordering::Relaxed), 2);
}

#[test]
fn test_object_set_parallel_iteration() {
    let object_set = ObjectSet::new();
    
    // Add many objects
    for i in 0..100 {
        let gc_obj = CLASSIFIED_ALLOCATOR.allocate_classified(
            TestData { id: i, data: vec![i as u8], value: format!("{}", i) },
            ObjectClass::Default
        );
        let ptr = unsafe { SendPtr::new(gc_obj.as_ptr() as *mut GcHeader<()>) };
        object_set.add(ptr);
    }
    
    // Test parallel iteration
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    
    object_set.iterate_parallel(4, move |_ptr| {
        counter_clone.fetch_add(1, Ordering::Relaxed);
    });
    
    // Should have processed all objects
    assert_eq!(counter.load(Ordering::Relaxed), 100);
}

#[test]
fn test_object_set_empty_parallel_iteration() {
    let object_set = ObjectSet::new();
    
    // Test parallel iteration on empty set
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    
    object_set.iterate_parallel(4, move |_ptr| {
        counter_clone.fetch_add(1, Ordering::Relaxed);
    });
    
    // Should have processed no objects
    assert_eq!(counter.load(Ordering::Relaxed), 0);
}

#[test]
fn test_weak_reference_creation() {
    let strong_ref = Gc::new(TestData {
        id: 42,
        data: vec![1, 2, 3, 4],
        value: "weak_test".to_string(),
    });
    
    // Test simple weak reference creation
    let weak_ref = Weak::new_simple(&strong_ref);
    
    // Test upgrade
    if let Some(weak_guard) = weak_ref.read() {
        if let Some(upgraded) = weak_guard.upgrade() {
            let data = upgraded.read().unwrap();
            assert_eq!(data.id, 42);
            assert_eq!(data.value, "weak_test");
        }
    }
}

#[test]
fn test_weak_reference_invalidation() {
    let weak_ref = {
        let temp_strong = Gc::new(TestData {
            id: 99,
            data: vec![9, 9, 9],
            value: "temporary".to_string(),
        });
        Weak::new_simple(&temp_strong)
    }; // temp_strong goes out of scope
    
    // In a real GC, the weak reference might become invalid
    // For now, just test that we can attempt to upgrade
    if let Some(weak_guard) = weak_ref.read() {
        let _upgrade_result = weak_guard.upgrade();
        // The result could be Some or None depending on GC timing
    }
}

#[test]
fn test_gc_pointer_read_write() {
    let gc_obj = Gc::new(TestData {
        id: 100,
        data: vec![10, 20, 30],
        value: "read_write_test".to_string(),
    });
    
    // Test read
    {
        let obj_ref = gc_obj.read().unwrap();
        assert_eq!(obj_ref.id, 100);
        assert_eq!(obj_ref.value, "read_write_test");
        assert_eq!(obj_ref.data, vec![10, 20, 30]);
    }
    
    // Test write
    {
        let mut obj_ref = gc_obj.write().unwrap();
        obj_ref.id = 200;
        obj_ref.value = "modified".to_string();
        obj_ref.data.push(40);
    }
    
    // Verify modifications
    {
        let obj_ref = gc_obj.read().unwrap();
        assert_eq!(obj_ref.id, 200);
        assert_eq!(obj_ref.value, "modified");
        assert_eq!(obj_ref.data, vec![10, 20, 30, 40]);
    }
}

#[test]
fn test_gc_pointer_concurrent_access() {
    let gc_obj = Arc::new(Gc::new(TestData {
        id: 0,
        data: vec![],
        value: "concurrent".to_string(),
    }));
    
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    
    // Test concurrent reads
    for i in 0..3 {
        let gc_clone = gc_obj.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            // Each thread tries to read the object
            if let Some(obj_ref) = gc_clone.read() {
                assert_eq!(obj_ref.value, "concurrent");
                
                // Some threads try to write
                if i == 0 {
                    drop(obj_ref);
                    if let Some(mut obj_ref) = gc_clone.write() {
                        obj_ref.id += 1;
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    barrier.wait();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    if let Some(obj_ref) = gc_obj.read() {
        assert_eq!(obj_ref.value, "concurrent");
        // ID might have been incremented by concurrent writes
        assert!(obj_ref.id < 100); // Reasonable upper bound
    }
}

#[test]
fn test_linked_list_tracing() {
    // Create a linked list of nodes
    let node3 = Gc::new(LinkedNode { id: 3, next: None });
    let node2 = Gc::new(LinkedNode { id: 2, next: Some(node3.clone()) });
    let node1 = Gc::new(LinkedNode { id: 1, next: Some(node2.clone()) });
    
    // Test that tracing works
    let mut trace_stack = Vec::new();
    unsafe {
        if let Some(node_ref) = node1.read() {
            node_ref.trace(&mut trace_stack);
        }
    }
    
    // Should have traced at least one reference
    assert!(!trace_stack.is_empty());
}

#[test]
fn test_allocator_trait_implementation() {
    let allocator = &*CLASSIFIED_ALLOCATOR;
    
    // Test the AllocatorTrait interface
    let initial_bytes = allocator.bytes_allocated();
    let initial_count = allocator.object_count();
    
    // Allocate using the trait interface
    let obj = allocator.allocate_classified(
        TestData {
            id: 999,
            data: vec![9; 50],
            value: "trait_test".to_string(),
        },
        ObjectClass::Default,
    );
    
    assert!(!obj.as_ptr().is_null());
    
    let final_bytes = allocator.bytes_allocated();
    let final_count = allocator.object_count();
    
    // Statistics should reflect the allocation
    assert!(final_count > initial_count);
    assert!(final_bytes >= initial_bytes);
    
    // Verify object data
    let obj_ref = obj.read().unwrap();
    assert_eq!(obj_ref.id, 999);
    assert_eq!(obj_ref.value, "trait_test");
}

#[test]
fn test_root_registration() {
    // Test root registration functionality
    let root_obj = Gc::new(TestData {
        id: 1000,
        data: vec![1, 0, 0, 0],
        value: "root_object".to_string(),
    });
    
    // Register as root
    register_root(&root_obj);
    
    // Verify the object is accessible
    let obj_ref = root_obj.read().unwrap();
    assert_eq!(obj_ref.id, 1000);
    assert_eq!(obj_ref.value, "root_object");
}

#[test]
fn test_stack_scanning() {
    // Test stack scanning functionality
    let mut mark_stack = Vec::new();
    
    // Add some fake objects to scan
    let gc_obj1 = CLASSIFIED_ALLOCATOR.allocate_classified(
        TestData { id: 1, data: vec![1], value: "scan1".to_string() },
        ObjectClass::Default
    );
    let gc_obj2 = CLASSIFIED_ALLOCATOR.allocate_classified(
        TestData { id: 2, data: vec![2], value: "scan2".to_string() },
        ObjectClass::Default
    );
    
    let ptr1 = unsafe { SendPtr::new(gc_obj1.as_ptr() as *mut GcHeader<()>) };
    let ptr2 = unsafe { SendPtr::new(gc_obj2.as_ptr() as *mut GcHeader<()>) };
    
    mark_stack.push(ptr1);
    mark_stack.push(ptr2);
    
    // Call stack scanning
    scan_stacks(&mut mark_stack);
    
    // Function should complete without panicking
    // In a real implementation, it would populate the mark stack with roots
}

#[test]
fn test_large_object_allocation() {
    let allocator = GcAllocator::new();
    
    // Test allocating a large object
    let large_data = vec![42u8; 1024 * 1024]; // 1MB
    let large_obj = allocator.allocate_gc(TestData {
        id: 2000,
        data: large_data.clone(),
        value: "large_object".to_string(),
    });
    
    assert!(!large_obj.as_ptr().is_null());
    
    // Verify data integrity
    let obj_ref = large_obj.read().unwrap();
    assert_eq!(obj_ref.id, 2000);
    assert_eq!(obj_ref.data.len(), 1024 * 1024);
    assert_eq!(obj_ref.value, "large_object");
    assert_eq!(obj_ref.data, large_data);
}

#[test]
fn test_zero_sized_allocation() {
    let allocator = GcAllocator::new();
    
    // Test allocating zero-sized types
    let unit_obj = allocator.allocate_gc(());
    assert!(!unit_obj.as_ptr().is_null());
    
    // Should be able to read the unit value
    let _unit_ref = unit_obj.read().unwrap();
}

#[test]
fn test_memory_alignment() {
    let allocator = GcAllocator::new();
    
    // Test various aligned types
    struct Aligned16 {
        _data: [u64; 2], // 16-byte aligned
    }
    
    unsafe impl GcTrace for Aligned16 {
        unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
    }
    
    let aligned_obj = allocator.allocate_gc(Aligned16 { _data: [1, 2] });
    assert!(!aligned_obj.as_ptr().is_null());
    
    // Check alignment
    let ptr = aligned_obj.as_ptr() as usize;
    assert_eq!(ptr % std::mem::align_of::<GcHeader<Aligned16>>(), 0);
}

#[test]
fn test_concurrent_allocation() {
    let allocator = Arc::new(GcAllocator::new());
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = Vec::new();
    
    // Test concurrent allocations
    for thread_id in 0..4 {
        let allocator_clone = allocator.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            // Each thread allocates multiple objects
            for i in 0..50 {
                let obj = allocator_clone.allocate_gc(TestData {
                    id: thread_id * 1000 + i,
                    data: vec![thread_id as u8; 10],
                    value: format!("thread_{}_obj_{}", thread_id, i),
                });
                
                assert!(!obj.as_ptr().is_null());
                
                // Verify data integrity
                let obj_ref = obj.read().unwrap();
                assert_eq!(obj_ref.id, thread_id * 1000 + i);
                assert_eq!(obj_ref.value, format!("thread_{}_obj_{}", thread_id, i));
            }
        });
        
        handles.push(handle);
    }
    
    barrier.wait();
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_object_set_concurrent_access() {
    let object_set = Arc::new(ObjectSet::new());
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = Vec::new();
    
    // Test concurrent object set operations
    for thread_id in 0..4 {
        let set_clone = object_set.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            // Each thread adds objects to the set
            for i in 0..25 {
                let gc_obj = CLASSIFIED_ALLOCATOR.allocate_classified(
                    TestData {
                        id: thread_id * 100 + i,
                        data: vec![thread_id as u8, i as u8],
                        value: format!("concurrent_{}_{}", thread_id, i),
                    },
                    ObjectClass::Default,
                );
                
                let ptr = unsafe { SendPtr::new(gc_obj.as_ptr() as *mut GcHeader<()>) };
                set_clone.add(ptr);
            }
        });
        
        handles.push(handle);
    }
    
    barrier.wait();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let final_count = object_set.len();
    assert_eq!(final_count, 4 * 25); // 4 threads * 25 objects each
}
