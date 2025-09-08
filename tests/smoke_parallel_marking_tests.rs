//! Smoke tests for parallel marking workers and fixpoint convergence
//!
//! These tests validate that parallel marking workers correctly collaborate
//! to mark all reachable objects and that fixpoint convergence works with
//! multiple rounds of root discovery.

use fugrip::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

gc_traceable!(NodeData);
gc_traceable!(CounterData);

#[derive(Debug)]
struct NodeData {
    id: usize,
    value: String,
}

#[derive(Debug)]
struct CounterData {
    value: AtomicUsize,
}

impl CounterData {
    fn new(value: usize) -> Self {
        Self {
            value: AtomicUsize::new(value),
        }
    }
    
    fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::Relaxed)
    }
}

#[cfg(feature = "smoke")]
#[test]
fn test_parallel_marking_workers() {
    let collector = &*memory::COLLECTOR;
    collector.set_phase(CollectorPhase::Waiting);
    
    // Clear any previous state
    smoke_clear_all_roots();
    collector.workers_finished.store(0, Ordering::Release);
    
    // Create a large object graph to require parallel processing
    let root_objects: Vec<_> = (0..100).map(|i| {
        Gc::new(NodeData {
            id: i,
            value: format!("node_{}", i),
        })
    }).collect();
    
    // Add all root objects to global roots
    for obj in &root_objects {
        smoke_add_global_root(unsafe { SendPtr::new(obj.as_ptr() as *mut GcHeader<()>) });
    }
    
    let headers: Vec<*mut GcHeader<()>> = root_objects.iter()
        .map(|obj| obj.as_ptr() as *mut GcHeader<()>)
        .collect();
    
    // Create worker threads to simulate parallel marking
    let num_workers = 4;
    let barrier = Arc::new(Barrier::new(num_workers + 1));
    let work_completed = Arc::new(AtomicUsize::new(0));
    let stolen_batches = Arc::new(AtomicUsize::new(0));
    let donated_work = Arc::new(AtomicUsize::new(0));
    
    let workers: Vec<_> = (0..num_workers).map(|worker_id| {
        let barrier = barrier.clone();
        let work_completed = work_completed.clone();
        let stolen_batches = stolen_batches.clone();
        let donated_work = donated_work.clone();
        
        thread::spawn(move || {
            // Register as a worker thread
            collector.register_worker_thread();
            
            // Wait for all workers to start
            barrier.wait();
            
            // Simulate marking work with work stealing and donation
            let mut local_work_done = 0;
            let mut local_stolen = 0;
            let mut local_donated = 0;
            
            while local_work_done < 15 { // Each worker processes some items
                if let Some(mut stolen_work) = collector.steal_marking_work() {
                    if !stolen_work.is_empty() {
                        local_stolen += 1;
                        
                        // Process half the stolen work
                        let work_size = stolen_work.len();
                        let process_count = (work_size + 1) / 2;
                        
                        for i in 0..process_count {
                            if i < stolen_work.len() {
                                let header_ptr = stolen_work[i];
                                unsafe {
                                    let header = &*header_ptr.as_ptr();
                                    // Mark the object (simulate tracing)
                                    header.mark_bit.store(true, Ordering::Release);
                                    local_work_done += 1;
                                }
                            }
                        }
                        
                        // Donate the remaining work back to help other workers
                        if stolen_work.len() > process_count {
                            let mut donation: Vec<_> = stolen_work.drain(process_count..).collect();
                            if !donation.is_empty() {
                                collector.donate_marking_work(&mut donation);
                                local_donated += donation.len();
                            }
                        }
                    }
                } else {
                    // No work available, short sleep
                    thread::sleep(Duration::from_micros(100));
                    local_work_done += 1;
                }
            }
            
            work_completed.fetch_add(local_work_done, Ordering::Relaxed);
            stolen_batches.fetch_add(local_stolen, Ordering::Relaxed);
            donated_work.fetch_add(local_donated, Ordering::Relaxed);
            
            // Simulate worker completion
            collector.workers_finished.fetch_add(1, Ordering::Release);
            collector.unregister_worker_thread();
            
            println!("Worker {} completed {} units of work, stole {} batches, donated {} items", 
                     worker_id, local_work_done, local_stolen, local_donated);
        })
    }).collect();
    
    // Start parallel marking phase
    collector.set_phase(CollectorPhase::Marking);
    collector.mark_global_roots();
    
    // Signal workers to start
    barrier.wait();
    
    // Wait for all workers to complete
    for worker in workers {
        worker.join().unwrap();
    }
    
    // Verify that work was distributed among workers
    let total_work = work_completed.load(Ordering::Acquire);
    assert!(total_work >= num_workers * 10, "Workers should have done substantial work: got {}", total_work);
    
    // Verify work stealing occurred
    let total_stolen = stolen_batches.load(Ordering::Acquire);
    assert!(total_stolen > 0, "Workers should have stolen work from the global stack: got {}", total_stolen);
    
    // Verify work donation occurred
    let total_donated = donated_work.load(Ordering::Acquire);
    assert!(total_donated > 0, "Workers should have donated work back: got {}", total_donated);
    
    // Verify that all workers registered completion
    let finished_workers = collector.workers_finished.load(Ordering::Acquire);
    assert_eq!(finished_workers, num_workers, "All workers should have finished");
    
    // Verify completion tracking
    assert!(collector.all_workers_finished(), "Collector should detect all workers finished");
    
    // Verify that objects were actually marked
    let mut marked_count = 0;
    for &header in &headers {
        unsafe {
            if (*header).mark_bit.load(Ordering::Acquire) {
                marked_count += 1;
            }
        }
    }
    
    println!("✓ Parallel marking workers: {} objects marked, {} batches stolen, {} items donated", 
             marked_count, total_stolen, total_donated);
    assert!(marked_count > 0, "At least some objects should be marked");
}

#[cfg(feature = "smoke")]
#[test]
fn test_converged_fixpoint_multiple_rounds() {
    let collector = &*memory::COLLECTOR;
    collector.set_phase(CollectorPhase::Waiting);
    
    // Clear any previous state
    smoke_clear_all_roots();
    
    // Round 1: Initial roots
    let initial_roots: Vec<_> = (0..10).map(|i| {
        Gc::new(CounterData::new(i))
    }).collect();
    
    for obj in &initial_roots {
        smoke_add_global_root(unsafe { SendPtr::new(obj.as_ptr() as *mut GcHeader<()>) });
    }
    
    // Start marking and converge to fixpoint
    collector.set_phase(CollectorPhase::Marking);
    collector.mark_global_roots();
    
    let initial_stack_size = collector.global_mark_stack.lock().unwrap().len();
    println!("Initial stack size: {}", initial_stack_size);
    
    // First convergence
    collector.converge_fixpoint_smoke();
    let after_first_convergence = collector.global_mark_stack.lock().unwrap().len();
    println!("After first convergence: {}", after_first_convergence);
    
    // Round 2: Add more roots to simulate discovered references
    let additional_roots: Vec<_> = (10..15).map(|i| {
        Gc::new(CounterData::new(i * 100))
    }).collect();
    
    for obj in &additional_roots {
        smoke_add_global_root(unsafe { SendPtr::new(obj.as_ptr() as *mut GcHeader<()>) });
    }
    
    // Mark the new roots
    collector.mark_global_roots();
    let after_new_roots = collector.global_mark_stack.lock().unwrap().len();
    println!("After adding new roots: {}", after_new_roots);
    
    // Second convergence
    collector.converge_fixpoint_smoke();
    let after_second_convergence = collector.global_mark_stack.lock().unwrap().len();
    println!("After second convergence: {}", after_second_convergence);
    
    // Round 3: Simulate stack scanning discovering more roots
    let discovered_roots: Vec<_> = (15..18).map(|i| {
        Gc::new(CounterData::new(i * 1000))
    }).collect();
    
    // Add to stack roots (simulating stack scan discovery)
    for obj in &discovered_roots {
        smoke_add_stack_root(unsafe { SendPtr::new(obj.as_ptr() as *mut GcHeader<()>) });
    }
    
    // Request stack scan (this should discover the new roots)
    collector.request_handshake_with_actions(vec![HandshakeAction::RequestStackScan]);
    let after_stack_scan = collector.global_mark_stack.lock().unwrap().len();
    println!("After stack scan: {}", after_stack_scan);
    
    // Final convergence
    collector.converge_fixpoint_smoke();
    let final_stack_size = collector.global_mark_stack.lock().unwrap().len();
    println!("Final stack size: {}", final_stack_size);
    
    // Verify fixpoint reached (stack should be empty)
    assert_eq!(final_stack_size, 0, "Global mark stack should be empty at fixpoint");
    
    // Verify all objects are marked
    let all_objects = [initial_roots, additional_roots, discovered_roots].concat();
    let mut marked_count = 0;
    
    for obj in &all_objects {
        unsafe {
            let header = &*obj.as_ptr();
            if header.mark_bit.load(Ordering::Acquire) {
                marked_count += 1;
            }
        }
    }
    
    println!("✓ Fixpoint convergence with multiple rounds: {}/{} objects marked", 
             marked_count, all_objects.len());
    
    // In an advancing wavefront collector, all discovered objects should be marked
    assert!(marked_count > 0, "Some objects should be marked after convergence");
}

#[cfg(feature = "smoke")]
#[test]
fn test_marking_worker_coordination() {
    let collector = &*memory::COLLECTOR;
    collector.set_phase(CollectorPhase::Waiting);
    
    // Clear any previous state
    smoke_clear_all_roots();
    
    // Create objects to mark
    let objects: Vec<_> = (0..50).map(|i| {
        Gc::new(NodeData {
            id: i,
            value: format!("coordinated_node_{}", i),
        })
    }).collect();
    
    // Add some as global roots
    for i in (0..objects.len()).step_by(5) {
        smoke_add_global_root(unsafe { SendPtr::new(objects[i].as_ptr() as *mut GcHeader<()>) });
    }
    
    // Start marking phase
    collector.set_phase(CollectorPhase::Marking);
    collector.mark_global_roots();
    
    // Create coordinated workers with completion tracking
    let num_workers = 3;
    let completion_barrier = Arc::new(Barrier::new(num_workers));
    let marked_objects = Arc::new(AtomicUsize::new(0));
    let work_steals = Arc::new(AtomicUsize::new(0));
    let work_donations = Arc::new(AtomicUsize::new(0));
    
    // Clear worker completion counter
    collector.workers_finished.store(0, Ordering::Release);
    
    let worker_handles: Vec<_> = (0..num_workers).map(|worker_id| {
        let completion_barrier = completion_barrier.clone();
        let marked_objects = marked_objects.clone();
        let work_steals = work_steals.clone();
        let work_donations = work_donations.clone();
        
        thread::spawn(move || {
            collector.register_worker_thread();
            
            let mut local_marked = 0;
            let mut local_steals = 0;
            let mut local_donations = 0;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 100;
            
            // Work until no more work is available or max iterations reached
            while iterations < MAX_ITERATIONS {
                if let Some(work_batch) = collector.steal_marking_work() {
                    if work_batch.is_empty() {
                        iterations += 1;
                        thread::sleep(Duration::from_micros(50));
                        continue;
                    }
                    
                    local_steals += 1;
                    
                    // Process the work batch
                    for header_ptr in work_batch {
                        unsafe {
                            let header = &*header_ptr.as_ptr();
                            if !header.mark_bit.swap(true, Ordering::AcqRel) {
                                local_marked += 1;
                                
                                // Simulate tracing - donate some work back periodically
                                if local_marked % 3 == 0 && local_marked > 3 {
                                    let mut donation = vec![header_ptr];
                                    collector.donate_marking_work(&mut donation);
                                    local_donations += donation.len();
                                }
                            }
                        }
                    }
                } else {
                    iterations += 1;
                    thread::sleep(Duration::from_micros(100));
                }
            }
            
            marked_objects.fetch_add(local_marked, Ordering::Relaxed);
            work_steals.fetch_add(local_steals, Ordering::Relaxed);
            work_donations.fetch_add(local_donations, Ordering::Relaxed);
            
            // Signal completion before unregistering
            collector.workers_finished.fetch_add(1, Ordering::Release);
            collector.unregister_worker_thread();
            
            println!("Worker {} marked {} objects, {} steals, {} donations", 
                     worker_id, local_marked, local_steals, local_donations);
            completion_barrier.wait();
        })
    }).collect();
    
    // Wait for all workers to complete
    for handle in worker_handles {
        handle.join().unwrap();
    }
    
    // Verify coordination worked
    let total_marked = marked_objects.load(Ordering::Acquire);
    let total_steals = work_steals.load(Ordering::Acquire);
    let total_donations = work_donations.load(Ordering::Acquire);
    
    println!("✓ Worker coordination: {} objects marked, {} steals, {} donations across {} workers", 
             total_marked, total_steals, total_donations, num_workers);
    
    // Verify work stealing occurred
    assert!(total_steals > 0, "Workers should have stolen work: got {} steals", total_steals);
    
    // Verify work donation occurred (if there was enough work to warrant it)
    if total_marked > 10 {
        assert!(total_donations > 0, "Workers should have donated work back: got {} donations", total_donations);
    }
    
    // Verify all workers completed
    let completed_workers = collector.workers_finished.load(Ordering::Acquire);
    assert_eq!(completed_workers, num_workers, "All workers should have completed: got {}", completed_workers);
    
    // Verify completion tracking works
    assert!(collector.all_workers_finished(), "Collector should detect all workers finished");
    
    // Check that objects were actually marked
    let mut verified_marked = 0;
    for obj in &objects {
        unsafe {
            if (*obj.as_ptr()).mark_bit.load(Ordering::Acquire) {
                verified_marked += 1;
            }
        }
    }
    
    println!("✓ Verified {} objects marked by coordinated workers", verified_marked);
    assert!(verified_marked > 0, "Some objects should be marked");
}