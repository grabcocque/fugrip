use crossbeam_queue::SegQueue;
use rayon::prelude::*;
use std::sync::Arc;

pub fn run_parallel_queue_work(items: usize, threads: usize) {
    let queue = Arc::new(SegQueue::new());
    for i in 0..items {
        queue.push(i);
    }

    let q = queue.clone();
    (0..threads).into_par_iter().for_each(|_| {
        while let Some(_v) = q.pop() {
            // simulate light processing
        }
    });
}

pub fn run_mixed_allocation_workload(items: usize) {
    // Placeholder: simulate allocations and deallocations using simple vectors
    let mut vecs = Vec::new();
    for i in 0..items {
        if i % 3 == 0 {
            vecs.push(vec![0u8; 64]);
        } else if i % 7 == 0 {
            vecs.pop();
        }
    }
}
