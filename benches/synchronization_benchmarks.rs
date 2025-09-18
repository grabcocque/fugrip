use criterion::{Criterion, criterion_group, criterion_main};
use fugrip::thread::MutatorThread;
use mmtk::util::Address;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

fn bench_phase_reading_contention(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let fixture =
        fugrip::test_utils::TestFixture::new_with_config(heap_base.as_usize(), heap_size, 4);
    let coordinator = &fixture.coordinator;

    c.bench_function("phase_reading_contention", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let c = Arc::clone(coordinator);
                    thread::spawn(move || {
                        for _ in 0..1000 {
                            std::hint::black_box(c.current_phase());
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
}

fn bench_phase_transitions(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let fixture =
        fugrip::test_utils::TestFixture::new_with_config(heap_base.as_usize(), heap_size, 4);
    let coordinator = &fixture.coordinator;

    c.bench_function("phase_transitions", |b| {
        b.iter(|| {
            coordinator.trigger_gc();
            let _ = coordinator.wait_until_idle(Duration::from_millis(200));
            std::hint::black_box(coordinator.current_phase());
        });
    });
}

fn bench_handshake_coordination(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let fixture =
        fugrip::test_utils::TestFixture::new_with_config(heap_base.as_usize(), heap_size, 4);
    let coordinator = &fixture.coordinator;
    let thread_registry = fixture.thread_registry();

    let running = Arc::new(AtomicBool::new(true));
    let mut handles = Vec::new();

    for id in 0..4 {
        let mutator = MutatorThread::new(id);
        thread_registry.register(mutator.clone());
        let flag = Arc::clone(&running);
        let coord = Arc::clone(coordinator);
        handles.push(thread::spawn(move || {
            while flag.load(Ordering::Relaxed) {
                mutator.poll_safepoint();
                // Simulate some GC work
                std::hint::black_box(coord.current_phase());
                thread::yield_now();
            }
        }));
    }

    c.bench_function("handshake_coordination", |b| {
        b.iter(|| {
            let coord = Arc::clone(coordinator);
            coord.trigger_gc();
            let _ = coord.wait_until_idle(Duration::from_millis(200));
            std::hint::black_box(coord.last_handshake_metrics());
        });
    });

    running.store(false, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("mutator thread should exit cleanly");
    }
    for id in 0..4 {
        thread_registry.unregister(id);
    }
}

fn bench_statistics_updates(c: &mut Criterion) {
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 64 * 1024 * 1024;
    let fixture =
        fugrip::test_utils::TestFixture::new_with_config(heap_base.as_usize(), heap_size, 4);
    let coordinator = &fixture.coordinator;

    c.bench_function("statistics_updates", |b| {
        b.iter(|| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let c = Arc::clone(coordinator);
                    thread::spawn(move || {
                        for _ in 0..100 {
                            std::hint::black_box(c.get_cycle_stats());
                        }
                    })
                })
                .collect();

            for worker in workers {
                worker.join().unwrap();
            }
        });
    });
}

criterion_group!(
    benches,
    bench_phase_reading_contention,
    bench_phase_transitions,
    bench_handshake_coordination,
    bench_statistics_updates,
);
criterion_main!(benches);
