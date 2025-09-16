use fugrip::{
    allocator::{AllocatorInterface, MMTkAllocator},
    binding::RustVM,
    core::ObjectHeader,
    error::GcResult,
    thread::MutatorThread,
};

#[test]
fn mmtk_allocator_interface() {
    let allocator = MMTkAllocator::new();
    let mutator = MutatorThread::new(99);

    // Test that the allocator can be polled for safepoints
    allocator.poll_safepoint(&mutator);

    // Note: We can't test actual allocation without a full MMTk runtime setup
    // But we can verify the interface works and doesn't panic
}

#[test]
fn allocator_interface_can_be_implemented() {
    struct CountingAllocator(std::sync::atomic::AtomicUsize);

    impl AllocatorInterface for CountingAllocator {
        fn allocate(
            &self,
            _mmtk_mutator: &mut mmtk::Mutator<RustVM>,
            _header: ObjectHeader,
            bytes: usize,
        ) -> GcResult<*mut u8> {
            self.0.fetch_add(bytes, std::sync::atomic::Ordering::SeqCst);
            Ok(std::ptr::null_mut())
        }

        fn poll_safepoint(&self, mutator: &MutatorThread) {
            mutator.poll_safepoint();
        }
    }

    let allocator = CountingAllocator(std::sync::atomic::AtomicUsize::new(0));
    let mutator = MutatorThread::new(1);
    allocator.poll_safepoint(&mutator);

    // Note: Can't easily test allocate without a real MMTk mutator
    assert_eq!(allocator.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
fn object_header_creation() {
    use fugrip::core::{LayoutId, ObjectFlags};

    let header = ObjectHeader {
        flags: ObjectFlags::MARKED | ObjectFlags::HAS_WEAK_REFS,
        layout_id: LayoutId(42),
        body_size: 128,
        vtable: std::ptr::null(),
    };

    assert!(header.flags.contains(ObjectFlags::MARKED));
    assert!(header.flags.contains(ObjectFlags::HAS_WEAK_REFS));
    assert!(!header.flags.contains(ObjectFlags::PINNED));
    assert_eq!(header.layout_id, LayoutId(42));
    assert_eq!(header.body_size, 128);
}
