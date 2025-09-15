use fugrip::core::{Gc, LayoutId, ObjectFlags, ObjectHeader, ObjectModel, RustObjectModel};

#[test]
fn object_header_defaults() {
    let header = ObjectHeader::default();
    assert_eq!(header.flags.bits(), ObjectFlags::empty().bits());
    assert_eq!(header.body_size, 0);
    assert_eq!(header.vtable, std::ptr::null());
}

#[test]
fn gc_wrapper_round_trip() {
    let raw = 0x1000usize as *mut u8;
    let handle = Gc::<u32>::from_raw(raw.cast());
    assert_eq!(handle.as_ptr(), raw.cast());
}

#[test]
fn rust_object_model_reads_header() {
    let header = ObjectHeader {
        flags: ObjectFlags::MARKED,
        body_size: 16,
        vtable: std::ptr::null(),
        layout_id: LayoutId(7),
    };
    let mut buffer = vec![0u8; std::mem::size_of::<ObjectHeader>() + 16];
    unsafe {
        std::ptr::write(buffer.as_mut_ptr().cast(), header);
    }
    let object_start = buffer.as_mut_ptr();
    let loaded = RustObjectModel::header(object_start);
    assert_eq!(loaded.flags.bits(), ObjectFlags::MARKED.bits());
    assert_eq!(
        RustObjectModel::size(object_start),
        std::mem::size_of::<ObjectHeader>() + 16
    );
}
