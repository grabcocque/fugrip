use fugrip::memory::*;
use fugrip::{GcHeader, GcTrace, SendPtr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// Test struct for memory testing
#[derive(Debug)]
struct TestNode {
    _id: usize,
    _data: Vec<u8>,
}

unsafe impl GcTrace for TestNode {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // No GC references to trace in this simple struct
    }
}

#[test]
fn test_object_set_creation() {
    let set = ObjectSet::new();
    assert!(set.is_empty());
    assert_eq!(*set.total_bytes.lock().unwrap(), 0);
}

#[test]
fn test_object_set_basic_operations() {
    let set = ObjectSet::new();

    // Add some objects
    let ptr1 = unsafe { SendPtr::new(0x1000 as *mut GcHeader<()>) };
    let ptr2 = unsafe { SendPtr::new(0x2000 as *mut GcHeader<()>) };

    set.objects.write().insert(ptr1.clone());
    assert_eq!(set.len(), 1);
    assert!(set.objects.read().contains(&ptr1));

    set.objects.write().insert(ptr2.clone());
    assert_eq!(set.len(), 2);
    assert!(set.objects.read().contains(&ptr2));

    // Remove object
    set.objects.write().remove(&ptr1);
    assert_eq!(set.len(), 1);
    assert!(!set.objects.read().contains(&ptr1));
    assert!(set.objects.read().contains(&ptr2));
}

#[test]
fn test_object_set_byte_tracking() {
    let set = ObjectSet::new();

    // Initially zero bytes
    assert_eq!(*set.total_bytes.lock().unwrap(), 0);

    // Modify byte count
    *set.total_bytes.lock().unwrap() = 1024;
    assert_eq!(*set.total_bytes.lock().unwrap(), 1024);

    *set.total_bytes.lock().unwrap() += 512;
    assert_eq!(*set.total_bytes.lock().unwrap(), 1536);

    *set.total_bytes.lock().unwrap() = 0;
    assert_eq!(*set.total_bytes.lock().unwrap(), 0);
}

#[test]
fn test_memory_region_creation() {
    let region = MemoryRegion {
        start: 0x10000000,
        end: 0x20000000,
        allocated: AtomicUsize::new(0),
        free_list: std::sync::Mutex::new(Vec::new()),
    };

    assert_eq!(region.start, 0x10000000);
    assert_eq!(region.end, 0x20000000);
    assert_eq!(region.allocated.load(Ordering::Acquire), 0);
}

#[test]
fn test_memory_region_allocation_tracking() {
    let region = MemoryRegion {
        start: 0x10000000,
        end: 0x20000000,
        allocated: AtomicUsize::new(0),
        free_list: std::sync::Mutex::new(Vec::new()),
    };

    // Track allocations
    region.allocated.store(1024, Ordering::Release);
    assert_eq!(region.allocated.load(Ordering::Acquire), 1024);

    region.allocated.fetch_add(512, Ordering::Release);
    assert_eq!(region.allocated.load(Ordering::Acquire), 1536);

    region.allocated.fetch_sub(256, Ordering::Release);
    assert_eq!(region.allocated.load(Ordering::Acquire), 1280);
}

#[test]
fn test_memory_region_free_list() {
    let region = MemoryRegion {
        start: 0x10000000,
        end: 0x20000000,
        allocated: AtomicUsize::new(0),
        free_list: std::sync::Mutex::new(Vec::new()),
    };

    // Access free list
    {
        let mut free_list = region.free_list.lock().unwrap();
        assert!(free_list.is_empty());

        // Add free blocks
        free_list.push((0x10001000, 1024));
        free_list.push((0x10002000, 2048));
        assert_eq!(free_list.len(), 2);

        // Check entries
        assert_eq!(free_list[0], (0x10001000, 1024));
        assert_eq!(free_list[1], (0x10002000, 2048));

        free_list.clear();
        assert!(free_list.is_empty());
    }
}

#[test]
fn test_heap_creation() {
    let heap = Heap::new();
    assert_eq!(heap.get_total_allocated(), 0);
    assert_eq!(heap.get_total_capacity(), 0);
}

#[test]
fn test_heap_statistics() {
    let mut heap = Heap::new();

    // Test adding regions and capacity tracking
    let region1 = MemoryRegion::new(0x10000000, 0x10001000); // 4KB
    let region2 = MemoryRegion::new(0x20000000, 0x20002000); // 8KB
    
    heap.add_region(region1);
    assert_eq!(heap.get_total_capacity(), 4096);
    
    heap.add_region(region2);
    assert_eq!(heap.get_total_capacity(), 12288); // 4KB + 8KB
}

#[test]
fn test_heap_region_management() {
    let mut heap = Heap::new();

    // Initially empty
    assert_eq!(heap.get_total_capacity(), 0);
    assert_eq!(heap.get_total_allocated(), 0);

    // Add a region
    let region = MemoryRegion::new(0x30000000, 0x30004000); // 16KB
    heap.add_region(region);
    assert_eq!(heap.get_total_capacity(), 16384);
}

#[test]
fn test_segment_buffer_creation() {
    let buffer = SegmentBuffer::default();
    assert!(buffer.current.is_null());
    assert!(buffer.end.is_null());
}

#[test]
fn test_segment_buffer_operations() {
    let mut buffer = SegmentBuffer::default();

    // Initially null pointers
    assert!(buffer.current.is_null());
    assert!(buffer.end.is_null());

    // Set buffer pointers
    buffer.current = 0x1000 as *mut u8;
    buffer.end = 0x2000 as *mut u8;

    assert_eq!(buffer.current as usize, 0x1000);
    assert_eq!(buffer.end as usize, 0x2000);

    // Reset buffer
    buffer.current = std::ptr::null_mut();
    buffer.end = std::ptr::null_mut();

    assert!(buffer.current.is_null());
    assert!(buffer.end.is_null());
}

#[test]
fn test_weak_reference_creation() {
    let target_ptr = 0x1000 as *mut i32;
    let weak_ref = WeakReference::new(target_ptr);

    assert_eq!(weak_ref.target.load(Ordering::Acquire), target_ptr);
    assert!(weak_ref.is_valid.load(Ordering::Acquire));
}

#[test]
fn test_weak_reference_operations() {
    let weak_ref = WeakReference::new(0x1000 as *mut i32);

    // Test upgrade when valid
    assert!(weak_ref.upgrade().is_some());
    assert_eq!(weak_ref.upgrade().unwrap(), 0x1000 as *mut i32);

    // Change target
    weak_ref.target.store(0x2000 as *mut i32, Ordering::Release);
    assert_eq!(weak_ref.target.load(Ordering::Acquire), 0x2000 as *mut i32);

    // Invalidate
    weak_ref.invalidate();
    assert!(!weak_ref.is_valid.load(Ordering::Acquire));
    assert!(weak_ref.upgrade().is_none());
    assert!(weak_ref.target.load(Ordering::Acquire).is_null());
}


#[test]
fn test_concurrent_heap_statistics() {
    let heap = Arc::new(std::sync::Mutex::new(Heap::new()));
    let heap_clone = heap.clone();

    let handle = thread::spawn(move || {
        let mut heap_guard = heap_clone.lock().unwrap();
        let region = MemoryRegion::new(0x40000000, 0x40001000); // 4KB
        heap_guard.add_region(region);
    });

    handle.join().unwrap();

    let heap_guard = heap.lock().unwrap();
    assert_eq!(heap_guard.get_total_capacity(), 4096);
}

#[test]
fn test_memory_region_concurrent_allocation() {
    let region = Arc::new(MemoryRegion {
        start: 0x10000000,
        end: 0x20000000,
        allocated: AtomicUsize::new(0),
        free_list: std::sync::Mutex::new(Vec::new()),
    });

    let region_clone = region.clone();

    let handle = thread::spawn(move || {
        region_clone.allocated.fetch_add(2048, Ordering::Release);
        let mut free_list = region_clone.free_list.lock().unwrap();
        free_list.push((0x10003000, 4096));
    });

    handle.join().unwrap();

    assert_eq!(region.allocated.load(Ordering::Acquire), 2048);
    let free_list = region.free_list.lock().unwrap();
    assert_eq!(free_list.len(), 1);
    assert_eq!(free_list[0], (0x10003000, 4096));
}

#[test]
fn test_weak_reference_concurrent_access() {
    let weak_ref = Arc::new(WeakReference::new(0x1000 as *mut i32));

    let weak_ref_clone = weak_ref.clone();

    let handle = thread::spawn(move || {
        weak_ref_clone.target.store(0x4000 as *mut i32, Ordering::Release);
        weak_ref_clone.invalidate();
    });

    handle.join().unwrap();

    // After invalidation, target should be null and upgrade should fail
    assert!(weak_ref.target.load(Ordering::Acquire).is_null());
    assert!(!weak_ref.is_valid.load(Ordering::Acquire));
    assert!(weak_ref.upgrade().is_none());
}
