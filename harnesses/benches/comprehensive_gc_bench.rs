use criterion::{Criterion, criterion_group, criterion_main};

use fugrip_harnesses::{run_mixed_allocation_workload, run_parallel_queue_work};

fn bench_parallel_queue(c: &mut Criterion) {
    c.bench_function("parallel_queue_100k_8", |b| {
        b.iter(|| run_parallel_queue_work(100_000, 8))
    });
}

fn bench_mixed_alloc(c: &mut Criterion) {
    c.bench_function("mixed_alloc_50k", |b| {
        b.iter(|| run_mixed_allocation_workload(50_000))
    });
}

criterion_group!(benches, bench_parallel_queue, bench_mixed_alloc);
criterion_main!(benches);
