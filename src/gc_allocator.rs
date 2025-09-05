use crate::core::*;
use crate::collector_phase::{CollectorState, MUTATOR_STATE};
use crate::segmented_heap::{SegmentedHeap, COLLECTOR};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::marker::PhantomData;

pub struct GcAllocator {
    heap: SegmentedHeap,
    collector: Arc<CollectorState>,
    allocation_threshold: AtomicUsize,
    live_bytes: AtomicUsize,
}

impl GcAllocator {
    pub fn new() -> Self {
        GcAllocator {
            heap: SegmentedHeap::new(),
            collector: COLLECTOR.clone(),
            allocation_threshold: AtomicUsize::new(1024 * 1024), // 1MB initial threshold
            live_bytes: AtomicUsize::new(0),
        }
    }
    
    pub fn allocate_gc<T: GcTrace>(&self, value: T) -> Gc<T> {
        // Fast path: thread-local allocation
        let ptr = MUTATOR_STATE.with(|state| {
            state.borrow_mut().try_allocate::<T>()
        });
        
        if let Some(ptr) = ptr {
            let allocating_black = MUTATOR_STATE.with(|state| state.borrow().allocating_black);
            unsafe {
                let header = GcHeader {
                    mark_bit: AtomicBool::new(allocating_black),
                    type_info: type_info::<T>(),
                    forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                    data: value,
                };
                std::ptr::write(ptr, header);
                return Gc {
                    ptr,
                    _phantom: PhantomData,
                };
            }
        }

        // Slow path: global allocation with potential GC trigger
        self.allocate_slow_path(value)
    }

    fn allocate_slow_path<T: GcTrace>(&self, value: T) -> Gc<T> {
        // Check if we need to trigger GC
        if self.live_bytes.load(Ordering::Relaxed)
            > self.allocation_threshold.load(Ordering::Relaxed)
        {
            self.collector.request_collection();
        }

        // Allocate from global segments
        let segment_id = self.heap.current_segment.load(Ordering::Relaxed);
        let segment = &self.heap.segments[segment_id];

        // Try allocation with CAS
        let size = std::mem::size_of::<GcHeader<T>>();
        let align = std::mem::align_of::<GcHeader<T>>();

        loop {
            let current = segment.allocation_ptr.load(Ordering::Relaxed);
            let aligned = align_up(current, align) as *mut u8;
            let new_ptr = unsafe { aligned.add(size) };

            if new_ptr > segment.end_ptr.load(Ordering::Relaxed) {
                // Need new segment
                // TODO: allocate new segment
                unimplemented!("Need to allocate new segment");
            }

            if segment
                .allocation_ptr
                .compare_exchange_weak(current, new_ptr as *mut u8, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                unsafe {
                    let header = GcHeader {
                        mark_bit: AtomicBool::new(
                            self.collector.allocation_color.load(Ordering::Acquire),
                        ),
                        type_info: type_info::<T>(),
                        forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                        data: value,
                    };
                    std::ptr::write(aligned as *mut GcHeader<T>, header);
                    return Gc {
                        ptr: aligned as *mut GcHeader<T>,
                        _phantom: PhantomData,
                    };
                }
            }
        }
    }
}
