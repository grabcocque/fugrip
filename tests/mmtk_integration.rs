//! Tests for MMTk integration functionality

use fugrip::{
    allocator::{AllocatorInterface, MMTkAllocator},
    binding::RustVM,
    binding::vm_impl::{RustActivePlan, RustReferenceGlue},
    core::{LayoutId, ObjectFlags, ObjectHeader, ObjectModel, RustObjectModel},
    roots::{GlobalRoots, RustScanning, StackRoots},
    thread::{MutatorThread, ThreadRegistry},
    weak::{WeakRef, WeakRefHeader, WeakRefRegistry},
};

use mmtk::{
    util::ObjectReference,
    vm::{ActivePlan, ObjectModel as MMTkObjectModel, ReferenceGlue, VMBinding},
};

#[test]
fn vm_binding_traits_implemented() {
    // Test that RustVM implements VMBinding
    fn check_vm_binding<T: VMBinding>() {}
    check_vm_binding::<RustVM>();
}

#[test]
fn active_plan_mutator_registry() {
    // Test mutator registration and lookup
    let thread_id = 42usize;
    let vm_thread =
        unsafe { std::mem::transmute::<usize, mmtk::util::opaque_pointer::VMThread>(thread_id) };

    // Initially no mutator should be registered
    assert!(!RustActivePlan::is_mutator(vm_thread));
    assert_eq!(RustActivePlan::number_of_mutators(), 0);
}

#[test]
fn object_model_header_operations() {
    let header = ObjectHeader {
        flags: ObjectFlags::MARKED,
        layout_id: LayoutId(123),
        body_size: 64,
        vtable: std::ptr::null(),
    };

    // Allocate memory for a test object
    let test_object = Box::leak(Box::new([0u8; 128]));
    let object_ptr = test_object.as_mut_ptr();

    // Write header to the object
    unsafe {
        std::ptr::write(object_ptr.cast::<ObjectHeader>(), header);
    }

    // Test header retrieval
    let retrieved_header = RustObjectModel::header(object_ptr);
    assert!(retrieved_header.flags.contains(ObjectFlags::MARKED));
    assert_eq!(retrieved_header.layout_id, LayoutId(123));
    assert_eq!(retrieved_header.body_size, 64);

    // Test size calculation
    let size = RustObjectModel::size(object_ptr);
    assert_eq!(size, std::mem::size_of::<ObjectHeader>() + 64);
}

#[test]
fn weak_reference_operations() {
    use fugrip::core::Gc;

    // Create a mock target object
    let target_data = Box::leak(Box::new(42i32));
    let target = Gc::from_raw(target_data);

    // Create weak reference
    let weak_ref = WeakRef::new(target);

    // Test initial state
    assert!(weak_ref.is_alive());

    // Test upgrade
    let upgraded = weak_ref.upgrade();
    assert!(upgraded.is_some());

    // Test clearing
    weak_ref.clear();
    assert!(!weak_ref.is_alive());
    assert!(weak_ref.upgrade().is_none());
}

#[test]
fn weak_ref_header_operations() {
    let weak_header = WeakRefHeader::new(LayoutId(456), 32);

    // Test initial state
    assert!(
        weak_header
            .header
            .flags
            .contains(ObjectFlags::HAS_WEAK_REFS)
    );
    assert_eq!(weak_header.header.layout_id, LayoutId(456));
    assert_eq!(weak_header.header.body_size, 32);
    assert!(weak_header.get_target().is_null());

    // Test target setting and retrieval
    let target_ptr = 0x1234_5678 as *mut u8;
    weak_header.set_target(target_ptr);
    assert_eq!(weak_header.get_target(), target_ptr);

    // Test clearing
    weak_header.clear_target();
    assert!(weak_header.get_target().is_null());
}

#[test]
fn weak_ref_registry_operations() {
    let registry = WeakRefRegistry::new();

    // Create test objects
    let weak_ref_obj = Box::leak(Box::new([0u8; 64])).as_mut_ptr();
    let target_obj = Box::leak(Box::new([0u8; 32])).as_mut_ptr();

    let target_slot = Box::leak(Box::new(std::sync::atomic::AtomicPtr::new(target_obj)));

    // Register weak reference
    registry.register_weak_ref(weak_ref_obj, target_slot);

    // Test processing with alive target
    registry.process_weak_refs(|ptr| ptr == target_obj);
    assert_eq!(
        target_slot.load(std::sync::atomic::Ordering::SeqCst),
        target_obj
    );

    // Test processing with dead target
    registry.process_weak_refs(|_| false);
    assert!(
        target_slot
            .load(std::sync::atomic::Ordering::SeqCst)
            .is_null()
    );
}

#[test]
fn reference_glue_operations() {
    use mmtk::util::Address;

    // Create a test object with weak reference header
    let weak_header = WeakRefHeader::new(LayoutId(789), 16);
    let object_ptr = Box::leak(Box::new(weak_header)) as *mut WeakRefHeader as *mut u8;
    let object_ref =
        unsafe { ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(object_ptr)) };

    // Test referent operations
    assert!(RustReferenceGlue::get_referent(object_ref).is_none());

    let target_obj = Box::leak(Box::new([0u8; 16])).as_mut_ptr();
    let target_ref =
        unsafe { ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(target_obj)) };

    RustReferenceGlue::set_referent(object_ref, target_ref);
    let retrieved = RustReferenceGlue::get_referent(object_ref);
    assert!(retrieved.is_some());
    assert_eq!(
        retrieved.unwrap().to_raw_address(),
        target_ref.to_raw_address()
    );

    // Test clearing
    RustReferenceGlue::clear_referent(object_ref);
    assert!(RustReferenceGlue::get_referent(object_ref).is_none());
}

#[test]
fn thread_registry_operations() {
    let registry = ThreadRegistry::new();
    let mutator = MutatorThread::new(1);

    registry.register(mutator.clone());

    // Test that we can iterate over registered mutators
    let mutators = registry.iter();
    assert_eq!(mutators.len(), 1);
    assert_eq!(mutators[0].id(), 1);
}

#[test]
fn safepoint_operations() {
    let mutator = MutatorThread::new(1);

    // Test basic safepoint polling (should not block when no safepoint requested)
    mutator.poll_safepoint();
}

#[test]
fn scanning_trait_exists() {
    // Just verify that RustScanning can be instantiated
    let _scanning = RustScanning::default();
}

#[test]
fn stack_roots_management() {
    let mut stack_roots = StackRoots::default();

    let handle1 = 0x1000 as *mut u8;
    let handle2 = 0x2000 as *mut u8;

    stack_roots.push(handle1);
    stack_roots.push(handle2);

    let handles: Vec<*mut u8> = stack_roots.iter().collect();
    assert_eq!(handles.len(), 2);
    assert!(handles.contains(&handle1));
    assert!(handles.contains(&handle2));

    stack_roots.clear();
    assert_eq!(stack_roots.iter().count(), 0);
}

#[test]
fn global_roots_management() {
    let mut global_roots = GlobalRoots::default();

    let handle1 = 0x3000 as *mut u8;
    let handle2 = 0x4000 as *mut u8;

    global_roots.register(handle1);
    global_roots.register(handle2);

    let handles: Vec<*mut u8> = global_roots.iter().collect();
    assert_eq!(handles.len(), 2);
    assert!(handles.contains(&handle1));
    assert!(handles.contains(&handle2));
}

#[test]
fn mmtk_allocator_interface() {
    let allocator = MMTkAllocator::new();
    let mutator = MutatorThread::new(1);

    // Test allocator interface exists
    allocator.poll_safepoint(&mutator);

    // Note: We can't easily test actual allocation without a full MMTk setup
    // but we can verify the interface is implemented correctly
}

#[test]
fn object_model_copy_operations() {
    use mmtk::util::Address;

    // Create source object
    let mut src_data = vec![0u8; 128];
    let header = ObjectHeader {
        flags: ObjectFlags::MARKED,
        layout_id: LayoutId(200),
        body_size: 64,
        vtable: std::ptr::null(),
    };

    unsafe {
        std::ptr::write(src_data.as_mut_ptr().cast::<ObjectHeader>(), header);
        // Write some test data in the body
        for i in 0..64 {
            src_data[std::mem::size_of::<ObjectHeader>() + i] = (i % 256) as u8;
        }
    }

    let src_ref = unsafe {
        ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(src_data.as_mut_ptr()))
    };

    // Test get_current_size
    let size = <RustObjectModel as MMTkObjectModel<RustVM>>::get_current_size(src_ref);
    assert_eq!(size, std::mem::size_of::<ObjectHeader>() + 64);

    // Test get_size_when_copied
    let copy_size = <RustObjectModel as MMTkObjectModel<RustVM>>::get_size_when_copied(src_ref);
    assert_eq!(copy_size, size);

    // Test alignment functions
    let align = <RustObjectModel as MMTkObjectModel<RustVM>>::get_align_when_copied(src_ref);
    assert_eq!(align, 8);

    let align_offset =
        <RustObjectModel as MMTkObjectModel<RustVM>>::get_align_offset_when_copied(src_ref);
    assert_eq!(align_offset, 0);
}
