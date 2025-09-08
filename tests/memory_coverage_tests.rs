use fugrip::{GcTrace, SendPtr, GcHeader, ObjectClass};
use fugrip::memory::*;
use fugrip::interfaces::AllocatorTrait;
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

// Test struct for memory operations
#[derive(Debug)]
struct TestItem {
    id: usize,
    data: Vec<u8>,
}

unsafe impl GcTrace for TestItem {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // No GC references to trace in this simple struct
    }
}

#[test]
fn test_gc_allocator_creation() {
    let allocator = GcAllocator::new();
    
    // Test default creation
    let allocator2 = GcAllocator::default();
    
    // Both should have similar initial state (can't directly compare but can test they work)
    let obj1 = allocator.allocate_gc(TestItem { id: 1, data: vec![1, 2, 3] });
    let obj2 = allocator2.allocate_gc(TestItem { id: 2, data: vec![4, 5, 6] });
    
    assert!(!obj1.as_ptr().is_null());
    assert!(!obj2.as_ptr().is_null());
}

#[test]
fn test_gc_allocator_fast_path() {
    let allocator = GcAllocator::new();
    
    // Test multiple small allocations (likely using fast path)
    for i in 0..50 {
        let obj = allocator.allocate_gc(TestItem { 
            id: i, 
            data: vec![i as u8; 10] 
        });
        assert!(!obj.as_ptr().is_null());
        
        // Verify data integrity
        unsafe {
            let header = &*obj.as_ptr();
            assert_eq!(header.data.id, i);
            assert_eq!(header.data.data.len(), 10);
        }
    }
}

#[test]
fn test_gc_allocator_slow_path() {
    let allocator = GcAllocator::new();
    
    // Test large allocations (likely triggering slow path)
    for i in 0..10 {
        let large_data = vec![i as u8; 1024 * 16]; // 16KB per object
        let obj = allocator.allocate_gc(TestItem { 
            id: i, 
            data: large_data 
        });
        assert!(!obj.as_ptr().is_null());
        
        // Verify data integrity
        unsafe {
            let header = &*obj.as_ptr();
            assert_eq!(header.data.id, i);
            assert_eq!(header.data.data.len(), 1024 * 16);
        }
    }
}

#[test]
fn test_segmented_heap_creation() {
    let heap = SegmentedHeap::new();
    
    // Test basic allocation functionality
    for i in 0..100 {
        let obj = TestItem { id: i, data: vec![i as u8; 100] };
        let gc_obj = heap.allocate(obj);
        assert!(!gc_obj.as_ptr().is_null());
    }
}

#[test]
fn test_classified_allocator_creation() {
    let allocator = ClassifiedAllocator::new();
    
    // Test allocation in different classes
    let classes = [
        ObjectClass::Default,
        ObjectClass::Destructor,
        ObjectClass::Census,
        ObjectClass::CensusAndDestructor,
        ObjectClass::Finalizer,
        ObjectClass::Weak,
    ];
    
    for &class in &classes {
        let obj = TestItem { id: class as usize, data: vec![1, 2, 3] };
        let gc_obj = allocator.allocate_classified(obj, class);
        assert!(!gc_obj.as_ptr().is_null());
        
        // Verify the object was allocated correctly
        unsafe {
            let header = &*gc_obj.as_ptr();
            assert_eq!(header.data.id, class as usize);
        }
    }
}

#[test]
fn test_classified_allocator_byte_counting() {
    let allocator = ClassifiedAllocator::new();
    
    let initial_bytes = allocator.bytes_allocated();
    let initial_count = allocator.object_count();
    
    // Allocate some objects
    for i in 0..10 {
        let obj = TestItem { id: i, data: vec![i as u8; i * 10] };
        let _gc_obj = allocator.allocate_classified(obj, ObjectClass::Default);
    }
    
    let final_bytes = allocator.bytes_allocated();
    let final_count = allocator.object_count();
    
    // Should have more bytes and objects allocated
    assert!(final_bytes >= initial_bytes);
    assert!(final_count >= initial_count + 10);
}

#[test]
fn test_object_set_functionality() {
    let object_set = ObjectSet::new();
    
    // Initially should be in default state
    let initial_count = object_set.len();
    
    // Add some objects (using dummy pointers for testing)
    for i in 1..=10 {
        let ptr = (i * 16) as *mut GcHeader<()>; // Simulate aligned pointers
        if !ptr.is_null() {
            object_set.add(unsafe { SendPtr::new(ptr) });
        }
    }
    
    let final_count = object_set.len();
    assert!(final_count >= initial_count);
    
    // Test iteration
    let visited = AtomicUsize::new(0);
    object_set.for_each(|_ptr| {
        visited.fetch_add(1, Ordering::Relaxed);
    });
    
    assert_eq!(visited.load(Ordering::Relaxed), final_count);
}

#[test]
fn test_concurrent_gc_allocation() {
    let allocator = Arc::new(GcAllocator::new());
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    
    for thread_id in 0..3 {
        let allocator_clone = Arc::clone(&allocator);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start
            
            // Each thread allocates objects
            for i in 0..20 {
                let obj = TestItem {
                    id: thread_id * 1000 + i,
                    data: vec![(thread_id * i) as u8; 50],
                };
                let gc_obj = allocator_clone.allocate_gc(obj);
                assert!(!gc_obj.as_ptr().is_null());
                
                // Verify data integrity
                unsafe {
                    let header = &*gc_obj.as_ptr();
                    assert_eq!(header.data.id, thread_id * 1000 + i);
                }
            }
        });
        
        handles.push(handle);
    }
    
    barrier.wait(); // Start all threads
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_classified_allocation() {
    let allocator = Arc::new(ClassifiedAllocator::new());
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    
    for thread_id in 0..3 {
        let allocator_clone = Arc::clone(&allocator);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start
            
            let class = match thread_id % 3 {
                0 => ObjectClass::Default,
                1 => ObjectClass::Finalizer,
                _ => ObjectClass::Weak,
            };
            
            // Each thread allocates objects
            for i in 0..15 {
                let obj = TestItem {
                    id: thread_id * 1000 + i,
                    data: vec![(thread_id * i) as u8; 30],
                };
                let gc_obj = allocator_clone.allocate_classified(obj, class);
                assert!(!gc_obj.as_ptr().is_null());
            }
        });
        
        handles.push(handle);
    }
    
    barrier.wait(); // Start all threads
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let total_bytes = allocator.bytes_allocated();
    let total_objects = allocator.object_count();
    
    assert!(total_bytes > 0);
    assert!(total_objects >= 45); // 3 threads * 15 objects each
}

#[test]
fn test_object_set_concurrent_operations() {
    let object_set = Arc::new(ObjectSet::new());
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    
    for thread_id in 0..3 {
        let set_clone = Arc::clone(&object_set);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start
            
            // Each thread adds different objects
            for i in 0..10 {
                let ptr = ((thread_id * 100 + i) * 16) as *mut GcHeader<()>;
                if !ptr.is_null() {
                    set_clone.add(unsafe { SendPtr::new(ptr) });
                }
            }
        });
        
        handles.push(handle);
    }
    
    barrier.wait(); // Start all threads
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let final_count = object_set.len();
    assert!(final_count > 0);
    
    // Test iteration after concurrent modifications
    let visited = AtomicUsize::new(0);
    object_set.for_each(|_ptr| {
        visited.fetch_add(1, Ordering::Relaxed);
    });
    
    assert_eq!(visited.load(Ordering::Relaxed), final_count);
}

#[test]
fn test_allocation_stress() {
    let allocator = GcAllocator::new();
    
    // Stress test with various object sizes
    for round in 0..5 {
        for i in 0..100 {
            let size = (i % 20 + 1) * 50; // Variable sizes
            let obj = TestItem {
                id: round * 100 + i,
                data: vec![i as u8; size],
            };
            let gc_obj = allocator.allocate_gc(obj);
            assert!(!gc_obj.as_ptr().is_null());
        }
    }
}

#[test]
fn test_large_object_allocation() {
    let allocator = GcAllocator::new();
    
    // Test very large object allocation
    let large_obj = TestItem {
        id: 999999,
        data: vec![0xAB; 1024 * 1024], // 1MB object
    };
    
    let gc_obj = allocator.allocate_gc(large_obj);
    assert!(!gc_obj.as_ptr().is_null());
    
    // Verify data integrity
    unsafe {
        let header = &*gc_obj.as_ptr();
        assert_eq!(header.data.id, 999999);
        assert_eq!(header.data.data.len(), 1024 * 1024);
        assert_eq!(header.data.data[0], 0xAB);
        assert_eq!(header.data.data[1024 * 1024 - 1], 0xAB);
    }
}

#[test]
fn test_zero_sized_allocation() {
    let allocator = GcAllocator::new();
    
    // Test zero-sized data
    let empty_obj = TestItem {
        id: 0,
        data: vec![], // Empty vector
    };
    
    let gc_obj = allocator.allocate_gc(empty_obj);
    assert!(!gc_obj.as_ptr().is_null());
    
    // Verify data integrity
    unsafe {
        let header = &*gc_obj.as_ptr();
        assert_eq!(header.data.id, 0);
        assert_eq!(header.data.data.len(), 0);
    }
}

#[test]
fn test_mixed_allocation_patterns() {
    let gc_allocator = GcAllocator::new();
    let classified_allocator = ClassifiedAllocator::new();
    
    // Mix allocations between different allocators
    for i in 0..50 {
        if i % 2 == 0 {
            let obj = TestItem { id: i, data: vec![i as u8; 20] };
            let _gc_obj = gc_allocator.allocate_gc(obj);
        } else {
            let obj = TestItem { id: i, data: vec![i as u8; 25] };
            let class = if i % 4 == 1 { ObjectClass::Default } else { ObjectClass::Finalizer };
            let _gc_obj = classified_allocator.allocate_classified(obj, class);
        }
    }
    
    // Verify classified allocator stats
    let bytes = classified_allocator.bytes_allocated();
    let count = classified_allocator.object_count();
    
    assert!(bytes > 0);
    assert!(count >= 25); // Half of 50 objects went to classified allocator
}

#[test]
fn test_object_alignment() {
    let allocator = GcAllocator::new();
    
    // Test that allocated objects have proper alignment
    for i in 0..20 {
        let obj = TestItem { id: i, data: vec![i as u8; i + 1] };
        let gc_obj = allocator.allocate_gc(obj);
        
        let ptr = gc_obj.as_ptr() as usize;
        // GcHeader should be properly aligned
        assert_eq!(ptr % std::mem::align_of::<GcHeader<TestItem>>(), 0);
    }
}