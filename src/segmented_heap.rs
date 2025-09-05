use crate::core::*;
use crate::collector_phase::CollectorState;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize};
use std::sync::Arc;
use once_cell::sync::Lazy;

// Segmented allocator that plays nice with borrow checker
pub struct SegmentedHeap {
    pub segments: Vec<Segment>,
    pub current_segment: AtomicUsize,
    pub collector_state: Arc<CollectorState>,
}

impl SegmentedHeap {
    pub fn new() -> Self {
        let initial_segment = Segment::new(0);
        SegmentedHeap {
            segments: vec![initial_segment],
            current_segment: AtomicUsize::new(0),
            collector_state: COLLECTOR.clone(),
        }
    }
    
    pub fn allocate<T: GcTrace>(&self, _value: T) -> Gc<T> {
        unimplemented!("SegmentedHeap::allocate")
    }
}

pub struct Segment {
    pub memory: Box<[MaybeUninit<u8>]>,
    pub mark_bits: Box<[AtomicBool]>,
    pub allocation_ptr: AtomicPtr<u8>,
    pub end_ptr: AtomicPtr<u8>, // Changed to AtomicPtr for thread safety
    pub segment_id: usize,
}

unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}

impl Segment {
    pub fn new(id: usize) -> Self {
        const SEGMENT_SIZE: usize = 1024 * 1024; // 1MB segments
        let memory = vec![MaybeUninit::uninit(); SEGMENT_SIZE].into_boxed_slice();
        
        // Create mark_bits without using vec! macro (AtomicBool doesn't implement Clone)
        let mark_bits_count = SEGMENT_SIZE / 64;
        let mut mark_bits = Vec::with_capacity(mark_bits_count);
        for _ in 0..mark_bits_count {
            mark_bits.push(AtomicBool::new(false));
        }
        let mark_bits = mark_bits.into_boxed_slice();
        
        let start_ptr = memory.as_ptr() as *mut u8;
        let end_ptr = unsafe { start_ptr.add(SEGMENT_SIZE) } as *const u8;
        
        Segment {
            memory,
            mark_bits,
            allocation_ptr: AtomicPtr::new(start_ptr),
            end_ptr: AtomicPtr::new(end_ptr as *mut u8),
            segment_id: id,
        }
    }
}

// Global references to allocator and collector
pub static ALLOCATOR: Lazy<GcAllocator> = Lazy::new(|| {
    GcAllocator::new()
});

pub static COLLECTOR: Lazy<Arc<CollectorState>> = Lazy::new(|| {
    Arc::new(CollectorState::new())
});

use crate::gc_allocator::GcAllocator;

// Safe interface that prevents direct mutation during GC
impl<T: GcTrace> Gc<T> {
    pub fn new(value: T) -> Self {
        ALLOCATOR.allocate_gc(value)
    }
}

impl<'a, T> GcRef<'a, T> {
    pub fn new(gc: &'a Gc<T>) -> Self {
        GcRef {
            _gc: gc,
            _phantom: PhantomData,
        }
    }
    
    pub fn new_from_ptr(_ptr: *mut GcHeader<T>) -> Self {
        unimplemented!("new_from_ptr")
    }
}

impl<'a, T> GcRefMut<'a, T> {
    pub fn new(gc: &'a Gc<T>) -> Self {
        GcRefMut {
            _gc: gc,
            _phantom: PhantomData,
        }
    }
}
