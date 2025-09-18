use fugrip::{
    core::{Gc, LayoutId, ObjectFlags},
    weak::{WeakRef, WeakRefHeader, WeakRefRegistry},
};
use mmtk::vm::Finalizable;

#[test]
fn weak_ref_creation_and_upgrade() {
    // Create a mock object
    let mut buffer = vec![0u8; 100];
    let obj_ptr = buffer.as_mut_ptr();

    // Create a Gc pointing to our mock object
    let gc_handle = Gc::<u32>::from_raw(obj_ptr as *mut u32);

    // Create a weak reference
    let weak_ref = WeakRef::new(gc_handle);

    // Should be able to upgrade initially
    let upgraded = weak_ref.upgrade();
    assert!(upgraded.is_some());
    assert_eq!(upgraded.unwrap().as_ptr(), obj_ptr as *mut u32);

    // Should be alive
    assert!(weak_ref.is_alive());
}

#[test]
fn weak_ref_clear() {
    let mut buffer = vec![0u8; 100];
    let obj_ptr = buffer.as_mut_ptr();
    let gc_handle = Gc::<u32>::from_raw(obj_ptr as *mut u32);

    let weak_ref = WeakRef::new(gc_handle);
    assert!(weak_ref.is_alive());

    // Clear the weak reference
    weak_ref.clear();
    assert!(!weak_ref.is_alive());

    // Upgrade should fail
    let upgraded = weak_ref.upgrade();
    assert!(upgraded.is_none());
}

#[test]
fn weak_ref_header_operations() {
    let header = WeakRefHeader::new(LayoutId(42), 64);

    assert!(header.header.flags.contains(ObjectFlags::HAS_WEAK_REFS));
    assert_eq!(header.header.layout_id, LayoutId(42));
    assert_eq!(header.header.body_size, 64);

    // Test target operations
    let target_ptr = 0x1000 as *mut u8;
    header.set_target(target_ptr);
    assert_eq!(header.get_target(), target_ptr);

    header.clear_target();
    assert_eq!(header.get_target(), std::ptr::null_mut());
}

#[test]
fn weak_ref_registry_operations() {
    let registry = WeakRefRegistry::new();

    // Create some mock atomic pointers
    use std::sync::atomic::{AtomicPtr, Ordering};
    let atomic1 = AtomicPtr::new(0x1000 as *mut u8);
    let atomic2 = AtomicPtr::new(0x2000 as *mut u8);

    // Register weak references
    registry.register_weak_ref(0x100 as *mut u8, &atomic1 as *const _ as *mut _);
    registry.register_weak_ref(0x200 as *mut u8, &atomic2 as *const _ as *mut _);

    // Test processing with all objects alive
    let is_alive = |ptr| ptr == 0x1000 as *mut u8 || ptr == 0x2000 as *mut u8;
    registry.process_weak_refs(is_alive);

    // Both should still be alive
    assert_eq!(atomic1.load(Ordering::SeqCst), 0x1000 as *mut u8);
    assert_eq!(atomic2.load(Ordering::SeqCst), 0x2000 as *mut u8);

    // Test processing with one object dead
    let is_alive_partial = |ptr| ptr == 0x1000 as *mut u8; // Only first object alive
    registry.process_weak_refs(is_alive_partial);

    // First should still be alive, second should be cleared
    assert_eq!(atomic1.load(Ordering::SeqCst), 0x1000 as *mut u8);
    assert_eq!(atomic2.load(Ordering::SeqCst), std::ptr::null_mut());
}

#[test]
fn weak_ref_clone() {
    let mut buffer = vec![0u8; 100];
    let obj_ptr = buffer.as_mut_ptr();
    let gc_handle = Gc::<u32>::from_raw(obj_ptr as *mut u32);

    let weak_ref1 = WeakRef::new(gc_handle);
    let weak_ref2 = weak_ref1.clone();

    // Both should be alive
    assert!(weak_ref1.is_alive());
    assert!(weak_ref2.is_alive());

    // Clearing one shouldn't affect the other
    weak_ref1.clear();
    assert!(!weak_ref1.is_alive());
    assert!(weak_ref2.is_alive());
}

#[test]
fn weak_ref_default() {
    let weak_ref = WeakRef::<u32>::default();
    assert!(!weak_ref.is_alive());
    assert!(weak_ref.upgrade().is_none());
}

#[test]
fn weak_ref_header_finalizable_get_reference() {
    use mmtk::util::Address;
    use mmtk::util::ObjectReference;

    let mut header = WeakRefHeader::new(LayoutId(1), 64);

    // Test with no target (should return dummy)
    let ref_obj = Finalizable::get_reference(&header);
    let dummy_addr = unsafe { Address::from_usize(0xDEADBEE8) }; // Match source
    assert_eq!(ref_obj.to_raw_address(), dummy_addr);

    // Set a valid target
    let valid_addr = unsafe { Address::from_usize(0x10000000) };
    let valid_obj = ObjectReference::from_raw_address(valid_addr).unwrap();
    header.set_reference(valid_obj);

    // Now get_reference should return the valid object
    let got_ref = Finalizable::get_reference(&header);
    assert_eq!(got_ref, valid_obj);
}

#[test]
fn weak_ref_header_finalizable_set_reference() {
    use mmtk::util::{Address, ObjectReference};

    let mut header = WeakRefHeader::new(LayoutId(1), 64);

    // Set valid reference
    let valid_addr = unsafe { Address::from_usize(0x10000000) };
    let valid_obj = ObjectReference::from_raw_address(valid_addr).unwrap();
    Finalizable::set_reference(&mut header, valid_obj);

    // Verify target set
    let target = header.get_target();
    assert_eq!(target as usize, valid_addr.as_usize());

    // Set dummy (clear)
    let dummy =
        ObjectReference::from_raw_address(unsafe { Address::from_usize(0xDEADBEE8) }).unwrap(); // Match source dummy
    Finalizable::set_reference(&mut header, dummy);

    // Should be set to dummy target
    let dummy_ptr = unsafe { Address::from_usize(0xDEADBEE8).to_mut_ptr::<u8>() };
    assert_eq!(header.get_target(), dummy_ptr);
}
