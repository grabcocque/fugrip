//! Performance benchmarks for write barrier fast path optimizations

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fugrip::concurrent::{
    BlackAllocator, ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier,
};
use mmtk::util::{Address, ObjectReference};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;

fn write_barrier_benchmarks(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x1000000) };
    debug_assert!(
        heap_base.as_usize() > 0,
        "Invalid heap base in write barrier benchmark"
    );
    let heap_size = 0x1000000; // 16MB heap
    let marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let coordinator = Arc::new(ParallelMarkingCoordinator::new(4));
    let barrier = WriteBarrier::new(Arc::clone(&marking), coordinator, heap_base, heap_size);

    // Create test objects
    let old_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
    debug_assert!(
        old_obj.to_raw_address().as_usize() > heap_base.as_usize(),
        "Invalid old_obj address"
    );
    let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x2000usize) };
    debug_assert!(
        new_obj.to_raw_address().as_usize() > heap_base.as_usize(),
        "Invalid new_obj address"
    );
    let mut slot = old_obj;

    // Set up initial state
    marking.set_color(old_obj, ObjectColor::White);

    let mut group = c.benchmark_group("write_barrier");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    // Vary barrier states
    for barrier_active in [false, true].iter() {
        let id = if *barrier_active {
            "active"
        } else {
            "inactive"
        };
        group.bench_function(format!("{}_barrier", id), |b| {
            if *barrier_active {
                barrier.activate();
            } else {
                barrier.deactivate();
            }
            b.iter(|| unsafe {
                barrier.write_barrier(
                    black_box(&mut slot as *mut ObjectReference),
                    black_box(new_obj),
                );
            });
        });

        group.bench_function(format!("{}_barrier_fast", id), |b| {
            if *barrier_active {
                barrier.activate();
            } else {
                barrier.deactivate();
            }
            b.iter(|| unsafe {
                barrier.write_barrier_fast(
                    black_box(&mut slot as *mut ObjectReference),
                    black_box(new_obj),
                );
            });
        });
    }

    // Benchmark optimized fast variant with different object sizes
    for size in [64usize, 256usize, 1024usize].iter() {
        let test_obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + *size) };
        debug_assert!(
            test_obj.to_raw_address().as_usize() > heap_base.as_usize(),
            "Invalid test_obj address"
        );
        group.bench_with_input(
            BenchmarkId::new("write_barrier_fast_size", size),
            &test_obj,
            |b, obj| {
                barrier.activate();
                b.iter(|| unsafe {
                    barrier.write_barrier_fast(
                        black_box(&mut slot as *mut ObjectReference),
                        black_box(*obj),
                    );
                });
            },
        );
    }

    group.finish();
}

fn tricolor_marking_benchmarks(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x2000000) };
    debug_assert!(
        heap_base.as_usize() > 0,
        "Invalid heap base in tricolor marking"
    );
    let heap_size = 0x1000000;
    let marking = TricolorMarking::new(heap_base, heap_size);

    let mut group = c.benchmark_group("tricolor_marking");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    for num_objects in [100, 1000, 10000].iter() {
        let objects: Vec<ObjectReference> = (0..*num_objects)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(heap_base + i * 0x1000usize)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("set_color_white", num_objects),
            &objects,
            |b, objects| {
                b.iter(|| {
                    for &obj in objects.iter() {
                        marking.set_color(black_box(obj), black_box(ObjectColor::White));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("get_color", num_objects),
            &objects,
            |b, objects| {
                b.iter(|| {
                    for &obj in objects.iter() {
                        black_box(marking.get_color(black_box(obj)));
                    }
                });
            },
        );
    }

    group.finish();
}

fn black_allocation_benchmarks(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x4000000) };
    debug_assert!(
        heap_base.as_usize() > 0,
        "Invalid heap base in black allocation"
    );
    let heap_size = 0x1000000;
    let marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let black_allocator = BlackAllocator::new(Arc::clone(&marking));

    let mut group = c.benchmark_group("black_allocation");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    // Test with different allocation offsets
    for offset in [0x1000usize, 0x2000usize, 0x4000usize].iter() {
        let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + *offset) };
        debug_assert!(
            obj.to_raw_address().as_usize() > heap_base.as_usize(),
            "Invalid obj address in allocation"
        );
        group.bench_with_input(
            BenchmarkId::new("allocate_black_object", offset),
            &obj,
            |b, obj| {
                b.iter(|| {
                    black_allocator.allocate_black(*obj);
                    black_box(());
                });
            },
        );
    }

    // Vary allocation states
    for active in [false, true].iter() {
        if *active {
            black_allocator.activate();
        } else {
            black_allocator.deactivate();
        }

        group.bench_with_input(BenchmarkId::new("is_active", active), &(), |b, ()| {
            b.iter(|| {
                black_box(black_allocator.is_active());
            });
        });
    }

    group.finish();
}

fn replay_fuzz_operations(
    data: &[u8],
    black_allocator: &BlackAllocator,
    tricolor_marking: &Arc<TricolorMarking>,
    objects: &[ObjectReference],
) {
    if data.len() < 8 {
        return;
    }

    let mut data_idx = 1;
    let num_operations = (data.get(1).unwrap_or(&0) % 128) + 1;
    debug_assert!(
        num_operations > 0 && num_operations <= 128,
        "Invalid number of operations"
    );
    data_idx += 1;

    for _ in 0..num_operations {
        if data_idx >= data.len() {
            break;
        }

        let operation = data[data_idx] % 5;
        debug_assert!(operation < 5, "Invalid operation code: {}", operation);
        data_idx += 1;

        match operation {
            0 => {
                black_allocator.activate();
            }
            1 => {
                black_allocator.deactivate();
            }
            2 => {
                if data_idx < data.len() && !objects.is_empty() {
                    let obj_idx = data[data_idx] as usize % objects.len();
                    debug_assert!(obj_idx < objects.len(), "Object index out of bounds");
                    let obj = objects[obj_idx];
                    black_allocator.allocate_black(obj);
                    data_idx += 1;
                }
            }
            3 => {
                if data_idx + 1 < data.len() && !objects.is_empty() {
                    let obj_idx = data[data_idx] as usize % objects.len();
                    debug_assert!(obj_idx < objects.len(), "Object index out of bounds");
                    let color_val = data[data_idx + 1] % 3;
                    let obj = objects[obj_idx];

                    let initial_color = match color_val {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        _ => ObjectColor::Black,
                    };

                    tricolor_marking.set_color(obj, initial_color);
                    black_allocator.allocate_black(obj);
                    data_idx += 2;
                }
            }
            4 => {
                let _stats = black_allocator.get_stats();
            }
            _ => {}
        }
    }
}

fn bench_fuzz_corpus(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuzz_corpus_black_allocator");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    debug_assert!(heap_base.as_usize() > 0, "Invalid heap base in fuzz corpus");
    let heap_size = 0x1000000;
    let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
    let black_allocator = BlackAllocator::new(tricolor_marking.clone());

    let num_objects = 32;
    let objects: Vec<ObjectReference> = (0..num_objects)
        .map(|i| unsafe {
            let addr = heap_base + (i * 64usize);
            debug_assert!(
                addr.as_usize() < heap_base.as_usize() + heap_size,
                "Object address out of heap bounds"
            );
            ObjectReference::from_raw_address_unchecked(addr)
        })
        .collect();

    let corpus_dir = Path::new("fuzz/corpus/black_allocator");
    if let Ok(entries) = fs::read_dir(corpus_dir) {
        let mut corpus_files: Vec<_> = entries.filter_map(|entry| entry.ok()).take(20).collect();
        corpus_files.sort_by_key(|a| a.path());

        for entry in corpus_files {
            if let (Ok(path), Ok(data)) = (entry.path().canonicalize(), fs::read(entry.path())) {
                let filename = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                group.bench_with_input(
                    BenchmarkId::new("replay_operations", filename),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            black_allocator.deactivate();
                            for &obj in &objects {
                                tricolor_marking.set_color(obj, ObjectColor::White);
                            }
                            replay_fuzz_operations(
                                data,
                                &black_allocator,
                                &tricolor_marking,
                                &objects,
                            );
                            black_box(&black_allocator);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

fn bitvector_build_benchmark(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x6000000) };
    debug_assert!(
        heap_base.as_usize() > 0,
        "Invalid heap base in bitvector benchmark"
    );
    let heap_size = 64 * 1024 * 1024;
    let fixture =
        fugrip::test_utils::TestFixture::new_with_config(heap_base.as_usize(), heap_size, 4);
    let coordinator = &fixture.coordinator;

    let objects: Vec<ObjectReference> = (0..20_000)
        .map(|i| unsafe { ObjectReference::from_raw_address_unchecked(heap_base + i * 64usize) })
        .collect();

    {
        let marking = coordinator.tricolor_marking();
        for &obj in &objects {
            marking.set_color(obj, ObjectColor::Black);
        }
    }

    c.bench_function("build_bitvector_from_markings", |b| {
        b.iter(|| {
            coordinator.bench_reset_bitvector_state();
            coordinator.bench_build_bitvector();
        });
    });
}

criterion_group!(
    benches,
    write_barrier_benchmarks,
    tricolor_marking_benchmarks,
    black_allocation_benchmarks,
    bitvector_build_benchmark,
    bench_fuzz_corpus,
    bench_root_scanning,
    bench_safepoint_coordination
);
fn bench_root_scanning(c: &mut Criterion) {
    let mut group = c.benchmark_group("root_scanning");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    let heap_base = unsafe { Address::from_usize(0x5000000) };
    debug_assert!(
        heap_base.as_usize() > 0,
        "Invalid heap base in root scanning"
    );
    let heap_size = 0x1000000;
    let marking = TricolorMarking::new(heap_base, heap_size);

    // Simulate roots (stack, globals, etc.)
    for num_roots in [100, 1000, 10000].iter() {
        let roots: Vec<ObjectReference> = (0..*num_roots)
            .map(|i| unsafe {
                let addr = heap_base + i * 0x1000usize;
                ObjectReference::from_raw_address_unchecked(addr)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("scan_roots", num_roots),
            &roots,
            |b, roots| {
                b.iter(|| {
                    for &root in roots {
                        marking.set_color(black_box(root), black_box(ObjectColor::Grey));
                    }
                    black_box(roots.len());
                });
            },
        );

        // Simulate root tracing with object graph
        let objects: Vec<ObjectReference> = (0..num_roots * 10)
            .map(|i| unsafe {
                let addr = heap_base + (*num_roots * 0x1000usize) + i * 64usize;
                ObjectReference::from_raw_address_unchecked(addr)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("trace_from_roots", num_roots),
            &(&roots, &objects),
            |b, (roots, objects)| {
                b.iter(|| {
                    for &_root in *roots {
                        // Simulate tracing from root to connected objects
                        for obj in (*objects).iter().take(5) {
                            // Assume 5 objects per root
                            marking.set_color(black_box(*obj), black_box(ObjectColor::Grey));
                        }
                    }
                    black_box((**objects).len());
                });
            },
        );
    }

    group.finish();
}

fn bench_safepoint_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("safepoint_coordination");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    let heap_base = unsafe { Address::from_usize(0x6000000) };
    debug_assert!(
        heap_base.as_usize() > 0,
        "Invalid heap base in safepoint benchmark"
    );
    let heap_size = 0x1000000;
    let _marking = Arc::new(TricolorMarking::new(heap_base, heap_size));

    for num_threads in [1, 4, 8, 16].iter() {
        // Simulate safepoint rendezvous
        group.bench_with_input(
            BenchmarkId::new("rendezvous", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    // Simulate threads polling safepoint
                    let mut active_threads = 0;
                    for _ in 0..num_threads {
                        // Simulate thread handshake
                        active_threads += 1;
                        std::hint::black_box(active_threads);
                    }
                    // Simulate coordinator waiting for all threads
                    std::hint::black_box(num_threads);
                });
            },
        );

        // Simulate safepoint polling overhead
        group.bench_with_input(
            BenchmarkId::new("polling_overhead", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    for _ in 0..1000 {
                        // 1000 polling cycles
                        // Simulate mutator thread polling
                        if fastrand::bool() {
                            // 50% chance of safepoint
                            std::hint::black_box("safepoint_hit");
                        }
                    }
                    std::hint::black_box(num_threads);
                });
            },
        );
    }

    group.finish();
}

criterion_main!(benches);
