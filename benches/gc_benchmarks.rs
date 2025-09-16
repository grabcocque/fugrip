//! Performance benchmarks for write barrier fast path optimizations

use criterion::{Criterion, criterion_group, criterion_main};
use fugrip::concurrent::{
    BlackAllocator, ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier,
};
use mmtk::util::{Address, ObjectReference};
use std::hint::black_box;
use std::sync::Arc;

fn write_barrier_benchmarks(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x1000000) };
    let heap_size = 0x1000000; // 16MB heap
    let marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(4));
    let barrier = WriteBarrier::new(Arc::clone(&marking), coordinator);

    // Create test objects
    let old_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
    let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x2000usize) };
    let mut slot = old_obj;

    // Set up initial state
    marking.set_color(old_obj, ObjectColor::White);

    let mut group = c.benchmark_group("write_barrier");

    // Benchmark inactive barrier (fast path)
    group.bench_function("inactive_barrier", |b| {
        barrier.deactivate(); // Ensure barrier is inactive
        b.iter(|| unsafe {
            barrier.write_barrier(
                black_box(&mut slot as *mut ObjectReference),
                black_box(new_obj),
            );
        });
    });

    // Benchmark active barrier (slow path)
    group.bench_function("active_barrier", |b| {
        barrier.activate(); // Ensure barrier is active
        b.iter(|| unsafe {
            barrier.write_barrier(
                black_box(&mut slot as *mut ObjectReference),
                black_box(new_obj),
            );
        });
    });

    // Benchmark optimized fast variant
    group.bench_function("write_barrier_fast", |b| {
        barrier.activate();
        b.iter(|| unsafe {
            barrier.write_barrier_fast(
                black_box(&mut slot as *mut ObjectReference),
                black_box(new_obj),
            );
        });
    });

    group.finish();
}

fn tricolor_marking_benchmarks(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x2000000) };
    let heap_size = 0x1000000;
    let marking = TricolorMarking::new(heap_base, heap_size);

    // Create test objects
    let objects: Vec<ObjectReference> = (0..100) // Reduced from 1000 to avoid memory issues
        .map(|i| unsafe {
            ObjectReference::from_raw_address_unchecked(heap_base + i * 0x1000usize)
        })
        .collect();

    let mut group = c.benchmark_group("tricolor_marking");

    group.bench_function("set_color_white", |b| {
        b.iter(|| {
            for &obj in &objects {
                marking.set_color(black_box(obj), black_box(ObjectColor::White));
            }
        });
    });

    group.bench_function("get_color", |b| {
        b.iter(|| {
            for &obj in &objects {
                black_box(marking.get_color(black_box(obj)));
            }
        });
    });

    group.finish();
}

fn black_allocation_benchmarks(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x4000000) };
    let heap_size = 0x1000000;
    let marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let black_allocator = BlackAllocator::new(Arc::clone(&marking));

    let mut group = c.benchmark_group("black_allocation");

    group.bench_function("is_black_allocation_active", |b| {
        b.iter(|| {
            black_box(black_allocator.is_active());
        });
    });

    group.bench_function("allocate_black_object", |b| {
        b.iter(|| {
            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
            black_allocator.allocate_black(obj);
            black_box(());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    write_barrier_benchmarks,
    tricolor_marking_benchmarks,
    black_allocation_benchmarks
);
criterion_main!(benches);
