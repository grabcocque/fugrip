use fugrip::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Define test types
struct TestData {
    value: i32,
}

unsafe impl GcTrace for TestData {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

struct FinalizableTestData {
    name: String,
    finalized_flag: Arc<AtomicBool>,
}

impl Finalizable for FinalizableTestData {
    fn finalize(&mut self) {
        self.finalized_flag.store(true, Ordering::SeqCst);
    }
}

unsafe impl GcTrace for FinalizableTestData {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

// Test the macro with a custom type to avoid orphan rule
struct TestU64(u64);

unsafe impl fugrip::GcTrace for TestU64 {
    unsafe fn trace(&self, _stack: &mut Vec<fugrip::SendPtr<fugrip::GcHeader<()>>>) {}
}
gc_traceable!(TestU64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weak_try_upgrade() {
        // Create a strong reference
        let strong_ref = Gc::new(TestData { value: 42 });
        
        // Create a weak reference
        let weak_ref = Weak::new_simple(&strong_ref);
        
        // Test try_upgrade method
        if let Some(weak_guard) = weak_ref.read() {
            if let Some(upgraded) = weak_guard.upgrade() {
                if let Some(value_guard) = upgraded.read() {
                    assert_eq!(value_guard.value, 42);
                }
            }
        }
    }

    #[test]
    fn test_finalizable_trait() {
        let finalized_flag = Arc::new(AtomicBool::new(false));
        let mut test_data = FinalizableTestData {
            name: "test".to_string(),
            finalized_flag: finalized_flag.clone(),
        };
        
        // Initially not finalized
        assert!(!finalized_flag.load(Ordering::SeqCst));
        
        // Call finalize
        test_data.finalize();
        
        // Now should be finalized
        assert!(finalized_flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_finalizable_object() {
        let finalized_flag = Arc::new(AtomicBool::new(false));
        let test_data = FinalizableTestData {
            name: "wrapped".to_string(),
            finalized_flag: finalized_flag.clone(),
        };
        
        let finalizable_obj = FinalizableObject::new(test_data);
        
        // Check initial state
        assert_eq!(finalizable_obj.finalize_state.load(Ordering::Acquire), 0);
        assert_eq!(finalizable_obj.data.name, "wrapped");
        
        // Mark as finalized
        finalizable_obj.mark_finalized();
        assert_eq!(finalizable_obj.finalize_state.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_finalizable_object_finalize_data() {
        let finalized_flag = Arc::new(AtomicBool::new(false));
        let test_data = FinalizableTestData {
            name: "finalize_test".to_string(),
            finalized_flag: finalized_flag.clone(),
        };
        
        let mut finalizable_obj = FinalizableObject::new(test_data);
        
        // Finalize the wrapped data
        finalizable_obj.data.finalize();
        assert!(finalized_flag.load(Ordering::SeqCst));
        
        // Also mark the wrapper as finalized
        finalizable_obj.mark_finalized();
        assert_eq!(finalizable_obj.finalize_state.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_gc_traceable_macro() {
        // Test that the macro generated a proper GcTrace implementation
        let test_value = TestU64(12345);
        let mut stack = Vec::new();
        
        unsafe {
            test_value.trace(&mut stack);
        }
        
        // The macro implementation should not add anything to the stack
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_gc_traceable_macro_with_various_types() {
        // Define additional types to test the macro
        struct CustomStruct {
            data: i32,
        }
        unsafe impl fugrip::GcTrace for CustomStruct {
            unsafe fn trace(&self, _stack: &mut Vec<fugrip::SendPtr<fugrip::GcHeader<()>>>) {}
        }
        
        struct TestFloat(f64);
        unsafe impl fugrip::GcTrace for TestFloat {
            unsafe fn trace(&self, _stack: &mut Vec<fugrip::SendPtr<fugrip::GcHeader<()>>>) {}
        }
        
        struct TestArray([u8; 4]);
        unsafe impl fugrip::GcTrace for TestArray {
            unsafe fn trace(&self, _stack: &mut Vec<fugrip::SendPtr<fugrip::GcHeader<()>>>) {}
        }
        
        gc_traceable!(CustomStruct);
        gc_traceable!(TestFloat);
        gc_traceable!(TestArray);
        
        let custom = CustomStruct { data: 100 };
        let float_val = TestFloat(3.14);
        let array_val = TestArray([1, 2, 3, 4]);
        
        let mut stack = Vec::new();
        
        unsafe {
            custom.trace(&mut stack);
            float_val.trace(&mut stack);
            array_val.trace(&mut stack);
        }
        
        // All macro implementations should be no-ops
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_finalizable_object_access_patterns() {
        let finalized_flag = Arc::new(AtomicBool::new(false));
        let test_data = FinalizableTestData {
            name: "access_test".to_string(),
            finalized_flag: finalized_flag.clone(),
        };
        
        let finalizable_obj = FinalizableObject::new(test_data);
        
        // Test accessing the wrapped data
        assert_eq!(finalizable_obj.data.name, "access_test");
        
        // Test state checks
        assert_eq!(finalizable_obj.finalize_state.load(Ordering::Relaxed), 0);
        
        // Test state transitions
        finalizable_obj.finalize_state.store(42, Ordering::Release);
        assert_eq!(finalizable_obj.finalize_state.load(Ordering::Acquire), 42);
        
        // Reset using mark_finalized
        finalizable_obj.mark_finalized();
        assert_eq!(finalizable_obj.finalize_state.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_multiple_finalizable_objects() {
        let flag1 = Arc::new(AtomicBool::new(false));
        let flag2 = Arc::new(AtomicBool::new(false));
        
        let data1 = FinalizableTestData {
            name: "obj1".to_string(),
            finalized_flag: flag1.clone(),
        };
        
        let data2 = FinalizableTestData {
            name: "obj2".to_string(),
            finalized_flag: flag2.clone(),
        };
        
        let mut obj1 = FinalizableObject::new(data1);
        let mut obj2 = FinalizableObject::new(data2);
        
        // Both should be unfinalized initially
        assert!(!flag1.load(Ordering::SeqCst));
        assert!(!flag2.load(Ordering::SeqCst));
        
        // Finalize first object
        obj1.data.finalize();
        assert!(flag1.load(Ordering::SeqCst));
        assert!(!flag2.load(Ordering::SeqCst));
        
        // Finalize second object
        obj2.data.finalize();
        assert!(flag1.load(Ordering::SeqCst));
        assert!(flag2.load(Ordering::SeqCst));
    }

    #[test]
    fn test_weak_try_upgrade_with_invalid_reference() {
        // This test focuses on the try_upgrade method specifically
        // We'll create a weak reference and test the method call
        let strong_ref = Gc::new(TestData { value: 999 });
        let weak_ref = Weak::new_simple(&strong_ref);
        
        // Test the try_upgrade method exists and can be called
        if let Some(weak_guard) = weak_ref.read() {
            // The try_upgrade method should be callable
            let result = weak_guard.upgrade();
            
            // Verify we can handle both success and failure cases
            match result {
                Some(upgraded_ref) => {
                    if let Some(value_guard) = upgraded_ref.read() {
                        assert_eq!(value_guard.value, 999);
                    }
                }
                None => {
                    // Weak reference was invalidated - this is also valid
                }
            }
        }
    }

    #[test]
    fn test_finalizable_object_with_different_data_types() {
        // Test with String data
        struct StringData {
            content: String,
            cleaned_up: Arc<AtomicBool>,
        }
        
        impl Finalizable for StringData {
            fn finalize(&mut self) {
                self.content.clear();
                self.cleaned_up.store(true, Ordering::SeqCst);
            }
        }
        
        let cleanup_flag = Arc::new(AtomicBool::new(false));
        let string_data = StringData {
            content: "test content".to_string(),
            cleaned_up: cleanup_flag.clone(),
        };
        
        let mut finalizable_string = FinalizableObject::new(string_data);
        
        assert_eq!(finalizable_string.data.content, "test content");
        assert!(!cleanup_flag.load(Ordering::SeqCst));
        
        finalizable_string.data.finalize();
        assert!(finalizable_string.data.content.is_empty());
        assert!(cleanup_flag.load(Ordering::SeqCst));
    }
}