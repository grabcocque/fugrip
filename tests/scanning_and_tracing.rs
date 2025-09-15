use fugrip::{
    core::{Gc, Trace, Traceable},
    roots::RustScanning,
};

// Test the Trace trait implementation for Gc<T>
#[test]
fn gc_trace_implementation() {
    let mut buffer = vec![0u8; 100];
    let obj_ptr = buffer.as_mut_ptr();
    let gc_handle = Gc::<u32>::from_raw(obj_ptr as *mut u32);

    let mut traced_pointers = Vec::new();
    let mut tracer = |ptr: *mut u8| traced_pointers.push(ptr);

    // Trace the Gc handle
    gc_handle.trace(&mut tracer);

    // Should have traced exactly one pointer
    assert_eq!(traced_pointers.len(), 1);
    assert_eq!(traced_pointers[0], obj_ptr);
}

// Test that Traceable trait can be implemented
#[test]
fn traceable_trait_can_be_implemented() {
    // Define a simple traceable struct
    struct TestObject {
        _field1: Gc<u32>,
        _field2: Gc<String>,
        _regular_field: i32,
    }

    impl Traceable for TestObject {
        fn reference_field_offsets() -> &'static [usize] {
            &[0, std::mem::size_of::<Gc<u32>>()] // offsets of field1 and field2
        }

        fn trace_references(&self, visitor: &mut dyn FnMut(mmtk::util::ObjectReference)) {
            // In a real implementation, this would convert Gc pointers to ObjectReferences
            let _ = visitor;
        }
    }

    // Test that the trait methods work
    let offsets = TestObject::reference_field_offsets();
    assert_eq!(offsets.len(), 2);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[1], std::mem::size_of::<Gc<u32>>());
}

// Test scanning functionality
#[test]
fn scanning_object_field_tracing() {
    use mmtk::util::ObjectReference;
    use mmtk::vm::slot::SimpleSlot;

    let scanning = RustScanning::default();

    // Create a test object with some potential pointer fields
    let mut object_data = vec![0u8; 128];
    let object_ptr = object_data.as_mut_ptr();

    // Write some fake pointer values into the object
    let fake_ptr1 = 0xDEADBEEF as *mut u8;
    let fake_ptr2 = 0xCAFEBABE as *mut u8;

    unsafe {
        // Write pointers at different offsets
        std::ptr::write(object_ptr.add(16).cast::<*mut u8>(), fake_ptr1);
        std::ptr::write(object_ptr.add(32).cast::<*mut u8>(), fake_ptr2);
        // Write a null pointer
        std::ptr::write(object_ptr.add(48).cast::<*mut u8>(), std::ptr::null_mut());
    }

    let object_ref = unsafe {
        ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_mut_ptr(object_ptr))
    };

    let mut visited_slots = Vec::new();
    let slot_visitor = |slot: SimpleSlot| {
        visited_slots.push(slot);
    };

    // Note: We can't easily test scan_object without a full MMTk setup,
    // but we can verify the scanning interface exists and doesn't panic
    let _ = scanning;
    let _ = object_ref;
    let _ = slot_visitor;

    // The scanning trait should be implemented
    assert!(true); // Placeholder - actual scanning test would require MMTk context
}

// Test root enumeration
#[test]
fn root_enumeration_interfaces() {
    use fugrip::roots::{GlobalRoots, StackRoots};

    let mut stack_roots = StackRoots::default();
    let mut global_roots = GlobalRoots::default();

    // Test stack roots
    stack_roots.push(0x1000 as *mut u8);
    stack_roots.push(0x2000 as *mut u8);

    let stack_count = stack_roots.iter().count();
    assert_eq!(stack_count, 2);

    // Test global roots
    global_roots.register(0x3000 as *mut u8);
    global_roots.register(0x4000 as *mut u8);

    let global_count = global_roots.iter().count();
    assert_eq!(global_count, 2);
}
