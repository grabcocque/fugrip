//! Integration tests for the `core` module primitives.

use fugrip::core::{Gc, LayoutId, ObjectFlags, ObjectHeader, ObjectModel, RustObjectModel, Trace};
use mmtk::util::{Address, ObjectReference};
use mmtk::vm::ObjectModel as _;

fn make_object(body_size: usize) -> (Vec<u8>, ObjectReference) {
    let total_size = body_size + std::mem::size_of::<ObjectHeader>();
    let mut buffer = vec![0u8; total_size];
    let ptr = buffer.as_mut_ptr();
    let header = ObjectHeader {
        flags: ObjectFlags::MARKED,
        layout_id: LayoutId(7),
        body_size: body_size.try_into().unwrap(),
        vtable: std::ptr::null(),
    };

    unsafe {
        ptr.cast::<ObjectHeader>().write(header);
    }

    let reference =
        unsafe { ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(ptr)) };

    (buffer, reference)
}

#[test]
fn object_header_defaults_are_zeroed() {
    let header = ObjectHeader::default();
    assert_eq!(header.flags, ObjectFlags::empty());
    assert_eq!(header.layout_id, LayoutId::default());
    assert_eq!(header.body_size, 0);
    assert!(header.vtable.is_null());
}

#[test]
fn object_flags_composition_behaves_like_bitflags() {
    let flags = ObjectFlags::MARKED | ObjectFlags::PINNED;
    assert!(flags.contains(ObjectFlags::MARKED));
    assert!(flags.contains(ObjectFlags::PINNED));
    assert!(!flags.contains(ObjectFlags::HAS_WEAK_REFS));

    let cleared = flags & !ObjectFlags::PINNED;
    assert!(cleared.contains(ObjectFlags::MARKED));
    assert!(!cleared.contains(ObjectFlags::PINNED));
}

#[test]
fn gc_pointer_roundtrip_and_trace() {
    let value = Box::into_raw(Box::new(123_u32));
    let gc = Gc::from_raw(value);
    assert!(!gc.is_null());
    assert_eq!(gc.as_ptr(), value);

    let mut visited = Vec::new();
    gc.trace(&mut |ptr| visited.push(ptr));
    assert_eq!(visited, vec![value.cast::<u8>()]);

    unsafe { drop(Box::from_raw(value)) };
}

#[test]
fn gc_null_pointer_traces_to_nothing() {
    let gc: Gc<u32> = Gc::new();
    assert!(gc.is_null());

    let mut visited = Vec::new();
    gc.trace(&mut |ptr| visited.push(ptr));
    assert!(visited.is_empty());
}

#[test]
fn rust_object_model_reads_header_and_size() {
    let (buffer, reference) = make_object(48);
    let ptr = buffer.as_ptr() as *mut u8;

    let header = RustObjectModel::header(ptr);
    assert_eq!(header.body_size, 48);
    assert_eq!(header.layout_id, LayoutId(7));

    let size = RustObjectModel::size(ptr);
    assert_eq!(size, 48 + std::mem::size_of::<ObjectHeader>());

    // current size delegates to size
    assert_eq!(RustObjectModel::get_current_size(reference), size);
}

#[test]
fn object_reference_conversion_matches_gc_pointer() {
    let value = Box::into_raw(Box::new(321_u64));
    let gc = Gc::from_raw(value);
    let obj_ref = gc.to_object_reference();
    assert_eq!(
        obj_ref.to_raw_address(),
        Address::from_mut_ptr(value.cast::<u8>())
    );
    unsafe { drop(Box::from_raw(value)) };
}
