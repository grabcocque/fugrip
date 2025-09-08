use fugrip::{SendPtr, GcHeader};
use fugrip::traits::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// Test struct for comprehensive testing
struct ComprehensiveTestStruct {
    id: usize,
    data: Vec<u8>,
    connections: Vec<*const ()>,
    active: bool,
}

unsafe impl GcTrace for ComprehensiveTestStruct {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // Would trace connections in real implementation
    }
}

impl GcFinalize for ComprehensiveTestStruct {
    fn finalize(&mut self) {
        self.active = false;
        self.data.clear();
    }
}

impl GcDrop for ComprehensiveTestStruct {
    fn gc_drop(&mut self) {
        self.connections.clear();
    }
}

impl GcCensus for ComprehensiveTestStruct {
    fn census(&self) -> bool {
        self.data.len() > 1024
    }
}

impl GcRevive for ComprehensiveTestStruct {
    fn should_revive(&self) -> bool {
        self.active && self.id % 2 == 0
    }
}

impl GcStats for ComprehensiveTestStruct {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len() + (self.connections.len() * 8)
    }
    
    fn child_count(&self) -> usize {
        self.connections.len()
    }
    
    fn custom_stats(&self) -> std::collections::HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("id".to_string(), self.id as u64);
        stats.insert("data_size".to_string(), self.data.len() as u64);
        stats
    }
}

impl GcVisitable for ComprehensiveTestStruct {
    fn accept_visitor<V: GcVisitor>(&self, visitor: &mut V) {
        let self_ptr = self as *const Self as *const GcHeader<()>;
        visitor.visit_object(self_ptr, self.size_bytes());
        
        for &connection in &self.connections {
            visitor.visit_reference(self_ptr, connection as *const GcHeader<()>);
        }
    }
}

struct TestMarker {
    marked_objects: Arc<AtomicUsize>,
}

impl GcMarker for TestMarker {
    unsafe fn mark_object(&self, _ptr: *mut GcHeader<()>) {
        self.marked_objects.fetch_add(1, Ordering::Relaxed);
    }
    
    unsafe fn is_marked(&self, _ptr: *mut GcHeader<()>) -> bool {
        // For testing, alternate between marked/unmarked
        let count = self.marked_objects.load(Ordering::Relaxed);
        count % 2 == 1
    }
}

impl GcMark for ComprehensiveTestStruct {
    unsafe fn mark(&self, marker: &dyn GcMarker) {
        // Mark self
        let self_ptr = self as *const Self as *mut GcHeader<()>;
        unsafe { marker.mark_object(self_ptr) };
        
        // Mark connections
        for &connection in &self.connections {
            if !connection.is_null() {
                unsafe { marker.mark_object(connection as *mut GcHeader<()>) };
            }
        }
    }
}

#[test]
fn test_comprehensive_gc_trace() {
    let obj = ComprehensiveTestStruct {
        id: 123,
        data: vec![1, 2, 3, 4, 5],
        connections: vec![0x1000 as *const (), 0x2000 as *const ()],
        active: true,
    };
    
    let mut stack = Vec::new();
    unsafe { obj.trace(&mut stack); }
    
    // Stack operations should not panic
    assert_eq!(obj.id, 123);
}

#[test]
fn test_comprehensive_finalization() {
    let mut obj = ComprehensiveTestStruct {
        id: 456,
        data: vec![10, 20, 30],
        connections: vec![0x3000 as *const ()],
        active: true,
    };
    
    assert!(obj.active);
    assert!(!obj.data.is_empty());
    
    obj.finalize();
    
    assert!(!obj.active);
    assert!(obj.data.is_empty());
}

#[test]
fn test_comprehensive_gc_drop() {
    let mut obj = ComprehensiveTestStruct {
        id: 789,
        data: vec![40, 50],
        connections: vec![0x4000 as *const (), 0x5000 as *const ()],
        active: true,
    };
    
    assert_eq!(obj.connections.len(), 2);
    
    obj.gc_drop();
    
    assert!(obj.connections.is_empty());
}

#[test]
fn test_comprehensive_census() {
    let small_obj = ComprehensiveTestStruct {
        id: 1,
        data: vec![1; 100], // Small data
        connections: vec![],
        active: true,
    };
    
    let large_obj = ComprehensiveTestStruct {
        id: 2,
        data: vec![1; 2048], // Large data
        connections: vec![],
        active: true,
    };
    
    assert!(!small_obj.census()); // Should not need cleanup
    assert!(large_obj.census());  // Should need cleanup
}

#[test]
fn test_comprehensive_revive() {
    let even_active_obj = ComprehensiveTestStruct {
        id: 4, // Even
        data: vec![],
        connections: vec![],
        active: true,
    };
    
    let odd_active_obj = ComprehensiveTestStruct {
        id: 3, // Odd
        data: vec![],
        connections: vec![],
        active: true,
    };
    
    let even_inactive_obj = ComprehensiveTestStruct {
        id: 6, // Even
        data: vec![],
        connections: vec![],
        active: false,
    };
    
    assert!(even_active_obj.should_revive()); // Even and active
    assert!(!odd_active_obj.should_revive()); // Odd
    assert!(!even_inactive_obj.should_revive()); // Inactive
}

#[test]
fn test_comprehensive_stats() {
    let obj = ComprehensiveTestStruct {
        id: 999,
        data: vec![1; 256],
        connections: vec![0x1000 as *const (), 0x2000 as *const (), 0x3000 as *const ()],
        active: true,
    };
    
    let size = obj.size_bytes();
    assert!(size > 256); // Should include struct size + data + connections
    
    assert_eq!(obj.child_count(), 3);
    
    let custom_stats = obj.custom_stats();
    assert_eq!(custom_stats.get("id"), Some(&999));
    assert_eq!(custom_stats.get("data_size"), Some(&256));
}

#[test]
fn test_comprehensive_visitor_pattern() {
    struct TestVisitor {
        objects_visited: usize,
        references_visited: usize,
    }
    
    impl GcVisitor for TestVisitor {
        fn visit_object(&mut self, _ptr: *const GcHeader<()>, _size: usize) {
            self.objects_visited += 1;
        }
        
        fn visit_reference(&mut self, _from: *const GcHeader<()>, _to: *const GcHeader<()>) {
            self.references_visited += 1;
        }
    }
    
    let obj = ComprehensiveTestStruct {
        id: 111,
        data: vec![1, 2, 3],
        connections: vec![0x1000 as *const (), 0x2000 as *const ()],
        active: true,
    };
    
    let mut visitor = TestVisitor {
        objects_visited: 0,
        references_visited: 0,
    };
    
    obj.accept_visitor(&mut visitor);
    
    assert_eq!(visitor.objects_visited, 1);
    assert_eq!(visitor.references_visited, 2);
}

#[test]
fn test_comprehensive_marking() {
    let marker = TestMarker {
        marked_objects: Arc::new(AtomicUsize::new(0)),
    };
    
    let obj = ComprehensiveTestStruct {
        id: 222,
        data: vec![1, 2, 3, 4],
        connections: vec![0x1000 as *const (), 0x2000 as *const (), 0x3000 as *const ()],
        active: true,
    };
    
    unsafe { obj.mark(&marker); }
    
    // Should have marked self + 3 connections = 4 objects
    assert_eq!(marker.marked_objects.load(Ordering::Relaxed), 4);
}

#[test]
fn test_marker_is_marked_alternating() {
    let marker = TestMarker {
        marked_objects: Arc::new(AtomicUsize::new(1)), // Start with 1 for odd
    };
    
    let dummy_ptr = 0x1000 as *mut GcHeader<()>;
    
    // Should be marked (count is odd)
    assert!(unsafe { marker.is_marked(dummy_ptr) });
    
    // Add one more mark
    unsafe { marker.mark_object(dummy_ptr) };
    
    // Should now be unmarked (count is even)
    assert!(!unsafe { marker.is_marked(dummy_ptr) });
}

#[test]
fn test_gc_serialize_context_trait() {
    struct MockSerializeContext {
        visited: std::collections::HashSet<*const GcHeader<()>>,
        next_id: u64,
    }
    
    impl GcSerializeContext for MockSerializeContext {
        fn is_visited(&self, ptr: *const GcHeader<()>) -> bool {
            self.visited.contains(&ptr)
        }
        
        fn mark_visited(&mut self, ptr: *const GcHeader<()>) {
            self.visited.insert(ptr);
        }
        
        fn get_object_id(&self, ptr: *const GcHeader<()>) -> u64 {
            ptr as u64
        }
    }
    
    let mut context = MockSerializeContext {
        visited: std::collections::HashSet::new(),
        next_id: 0,
    };
    
    let ptr1 = 0x1000 as *const GcHeader<()>;
    let ptr2 = 0x2000 as *const GcHeader<()>;
    
    // Initially not visited
    assert!(!context.is_visited(ptr1));
    assert!(!context.is_visited(ptr2));
    
    // Mark as visited
    context.mark_visited(ptr1);
    assert!(context.is_visited(ptr1));
    assert!(!context.is_visited(ptr2));
    
    // Test object IDs
    assert_eq!(context.get_object_id(ptr1), 0x1000);
    assert_eq!(context.get_object_id(ptr2), 0x2000);
}

#[test]
fn test_gc_deserialize_context_trait() {
    struct MockDeserializeContext {
        objects: std::collections::HashMap<u64, *mut GcHeader<()>>,
    }
    
    impl GcDeserializeContext for MockDeserializeContext {
        fn register_object(&mut self, id: u64, ptr: *mut GcHeader<()>) {
            self.objects.insert(id, ptr);
        }
        
        fn get_object(&self, id: u64) -> Option<*mut GcHeader<()>> {
            self.objects.get(&id).copied()
        }
    }
    
    let mut context = MockDeserializeContext {
        objects: std::collections::HashMap::new(),
    };
    
    let ptr1 = 0x1000 as *mut GcHeader<()>;
    let ptr2 = 0x2000 as *mut GcHeader<()>;
    
    // Initially no objects
    assert!(context.get_object(1).is_none());
    assert!(context.get_object(2).is_none());
    
    // Register objects
    context.register_object(1, ptr1);
    context.register_object(2, ptr2);
    
    // Should now be retrievable
    assert_eq!(context.get_object(1), Some(ptr1));
    assert_eq!(context.get_object(2), Some(ptr2));
    assert!(context.get_object(3).is_none()); // Non-existent
}

#[test]
fn test_concurrent_marker_operations() {
    let marker = Arc::new(TestMarker {
        marked_objects: Arc::new(AtomicUsize::new(0)),
    });
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads that do marking
    for thread_id in 0..4 {
        let marker_clone = Arc::clone(&marker);
        
        let handle = thread::spawn(move || {
            for i in 0..10 {
                let ptr = ((thread_id * 100 + i) * 8) as *mut GcHeader<()>;
                unsafe {
                    marker_clone.mark_object(ptr);
                }
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Should have marked 4 * 10 = 40 objects
    assert_eq!(marker.marked_objects.load(Ordering::Relaxed), 40);
}

#[test]
fn test_stress_gc_trace_implementations() {
    // Test all primitive implementations under stress
    let mut stack = Vec::new();
    
    for _ in 0..1000 {
        unsafe {
            123i32.trace(&mut stack);
            456usize.trace(&mut stack);
            "test".to_string().trace(&mut stack);
            ().trace(&mut stack);
            true.trace(&mut stack);
            'x'.trace(&mut stack);
            42.0f64.trace(&mut stack);
        }
        
        stack.clear(); // Clear for next iteration
    }
}

#[test]
fn test_container_trace_stress() {
    let mut stack = Vec::new();
    
    // Test Vec tracing
    let test_vec: Vec<i32> = (0..100).collect();
    unsafe { test_vec.trace(&mut stack); }
    
    // Test Option tracing
    let some_value = Some(42i32);
    let none_value: Option<i32> = None;
    unsafe { 
        some_value.trace(&mut stack);
        none_value.trace(&mut stack);
    }
    
    // Test Box tracing
    let boxed_value = Box::new(123i32);
    unsafe { boxed_value.trace(&mut stack); }
    
    // Test tuple tracing
    let tuple1 = (1i32,);
    let tuple2 = (1i32, 2i32);
    let tuple3 = (1i32, 2i32, 3i32);
    unsafe {
        tuple1.trace(&mut stack);
        tuple2.trace(&mut stack);
        tuple3.trace(&mut stack);
    }
}

#[test]
fn test_object_lifecycle_simulation() {
    let mut obj = ComprehensiveTestStruct {
        id: 1000,
        data: vec![1; 2048],
        connections: vec![0x1000 as *const (), 0x2000 as *const ()],
        active: true,
    };
    
    // Phase 1: Creation and initial state
    assert!(obj.active);
    assert_eq!(obj.child_count(), 2);
    
    // Phase 2: Census check (should need cleanup due to large data)
    assert!(obj.census());
    
    // Phase 3: Should revive (even ID and active)
    assert!(obj.should_revive());
    
    // Phase 4: Finalization
    obj.finalize();
    assert!(!obj.active);
    assert!(obj.data.is_empty());
    
    // Phase 5: Should not revive anymore (inactive)
    assert!(!obj.should_revive());
    
    // Phase 6: Final cleanup
    obj.gc_drop();
    assert!(obj.connections.is_empty());
}

#[test]
fn test_all_numeric_type_traces() {
    let mut stack = Vec::new();
    
    unsafe {
        // Test all numeric types
        (42u8).trace(&mut stack);
        (42i8).trace(&mut stack);
        (42u16).trace(&mut stack);
        (42i16).trace(&mut stack);
        (42u32).trace(&mut stack);
        (42i32).trace(&mut stack);
        (42u64).trace(&mut stack);
        (42i64).trace(&mut stack);
        (42usize).trace(&mut stack);
        (42isize).trace(&mut stack);
        (42.0f32).trace(&mut stack);
        (42.0f64).trace(&mut stack);
    }
    
    // All traces should complete without issues
    assert!(stack.is_empty()); // Primitives don't add to stack
}
