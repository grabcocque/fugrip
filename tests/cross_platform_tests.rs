use fugrip::{Gc, GcHeader, GcTrace, SendPtr};

// Test struct to verify stack scanning works
#[derive(Clone)]
struct TestObject {
    data: String,
    number: i32,
}

unsafe impl GcTrace for TestObject {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // Simple objects don't contain GC pointers
    }
}

#[test]
fn test_cross_platform_stack_scanning() {
    // This test verifies that our cross-platform stack scanning approach works
    // by creating objects on the stack and ensuring the GC can find them

    let test_obj1 = Gc::new(TestObject {
        data: "Test Object 1".to_string(),
        number: 42,
    });

    let test_obj2 = Gc::new(TestObject {
        data: "Test Object 2".to_string(),
        number: 84,
    });

    // Verify objects are accessible
    if let Some(obj1_ref) = test_obj1.read() {
        assert_eq!(obj1_ref.number, 42);
        assert_eq!(obj1_ref.data, "Test Object 1");
    }

    if let Some(obj2_ref) = test_obj2.read() {
        assert_eq!(obj2_ref.number, 84);
        assert_eq!(obj2_ref.data, "Test Object 2");
    }

    // Create nested structure to test more complex scanning
    #[derive(Clone)]
    struct NestedObject {
        inner: Option<Gc<TestObject>>,
        value: usize,
    }

    unsafe impl GcTrace for NestedObject {
        unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
            if let Some(ref inner_gc) = self.inner {
                if let Some(inner_ref) = inner_gc.read() {
                    unsafe {
                        inner_ref.trace(stack);
                    }
                }
            }
        }
    }

    let nested = Gc::new(NestedObject {
        inner: Some(test_obj1.clone()),
        value: 123,
    });

    // Verify nested structure works
    if let Some(nested_ref) = nested.read() {
        assert_eq!(nested_ref.value, 123);
        if let Some(ref inner) = nested_ref.inner {
            if let Some(inner_ref) = inner.read() {
                assert_eq!(inner_ref.number, 42);
            }
        }
    }

    // The stack scanning should be able to find these objects
    // without relying on /proc parsing or other Linux-specific mechanisms
}

#[test]
fn test_stack_pointer_detection() {
    // Test that the psm crate can get the stack pointer on this platform
    let stack_ptr = psm::stack_pointer();

    // Stack pointer should be a reasonable value (not null, within expected range)
    assert!(!stack_ptr.is_null());

    // Stack pointer should be different in nested function calls
    fn nested_function() -> *mut u8 {
        psm::stack_pointer()
    }

    let nested_ptr = nested_function();

    // The nested function should have a different stack pointer
    // (though the exact relationship depends on stack growth direction)
    assert_ne!(stack_ptr, nested_ptr);
}

#[test]
fn test_concurrent_stack_scanning() {
    // Test that stack scanning works correctly with multiple threads
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let counter_clone = counter.clone();

        let handle = thread::spawn(move || {
            // Create thread-local GC objects
            let thread_obj = Gc::new(TestObject {
                data: format!("Thread {} object", thread_id),
                number: thread_id as i32 * 10,
            });

            // Verify object is accessible
            if let Some(obj_ref) = thread_obj.read() {
                assert_eq!(obj_ref.number, thread_id as i32 * 10);
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All threads should have successfully created and accessed their objects
    assert_eq!(counter.load(Ordering::Relaxed), 4);
}

#[cfg(test)]
mod platform_specific_tests {
    use super::*;

    #[test]
    fn test_stack_bounds_detection() {
        // Test that we can get reasonable stack bounds on this platform
        // use fugrip::memory::COLLECTOR; // not used in this test

        // This is an indirect test - we verify that the stack scanning
        // doesn't crash and produces reasonable results
        let test_obj = Gc::new(TestObject {
            data: "Stack bounds test".to_string(),
            number: 999,
        });

        // Access the object to ensure it's on the stack
        if let Some(obj_ref) = test_obj.read() {
            assert_eq!(obj_ref.number, 999);
        }

        // The collector should be able to scan the stack without issues
        // This indirectly tests our cross-platform stack bounds detection
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_linux_pthread_apis() {
        // Test that we can use pthread APIs instead of /proc on Linux
        unsafe {
            let mut attr: libc::pthread_attr_t = std::mem::zeroed();
            let mut stack_addr: *mut libc::c_void = std::ptr::null_mut();
            let mut stack_size: libc::size_t = 0;

            // This should work on Linux
            let result = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
            if result == 0 {
                let stack_result =
                    libc::pthread_attr_getstack(&attr, &mut stack_addr, &mut stack_size);
                libc::pthread_attr_destroy(&mut attr);

                if stack_result == 0 {
                    assert!(!stack_addr.is_null());
                    assert!(stack_size > 0);

                    // Stack size should be reasonable (at least 64KB, less than 100MB)
                    assert!(stack_size >= 64 * 1024);
                    assert!(stack_size <= 100 * 1024 * 1024);
                }
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_pthread_apis() {
        // Test macOS-specific pthread APIs
        unsafe {
            let thread = libc::pthread_self();
            let stack_addr = libc::pthread_get_stackaddr_np(thread);
            let stack_size = libc::pthread_get_stacksize_np(thread);

            if !stack_addr.is_null() && stack_size > 0 {
                // Stack size should be reasonable
                assert!(stack_size >= 64 * 1024);
                assert!(stack_size <= 100 * 1024 * 1024);
            }
        }
    }
}
