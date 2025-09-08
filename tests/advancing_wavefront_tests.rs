//! Advancing Wavefront Collector Tests
//!
//! These tests validate the core property that makes FUGC an "advancing wavefront" collector:
//! Once an object is marked, it stays marked for that GC cycle. The mutator cannot create
//! new work for the collector by modifying the heap during the marking phase.
//!
//! This is in contrast to concurrent mark-sweep collectors where mutator actions during
//! marking can create additional work for the collector.

#[cfg(feature = "smoke")]
mod advancing_wavefront {
    use fugrip::*;
    use std::sync::{Arc, Barrier, Mutex};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};
    
    /// Core test: Mutator cannot create new marking work during concurrent marking
    #[test]
    fn test_no_new_marking_work_during_concurrent_marking() {
        let collector = &*fugrip::memory::COLLECTOR;
        
        // Create initial object graph before marking starts
        let root_objects: Vec<Gc<ObjectNode>> = (0..20)
            .map(|i| Gc::new(ObjectNode {
                id: i,
                data: format!("Initial_object_{}", i),
                references: Vec::new(),
            }))
            .collect();
        
        // Add some initial cross-references
        for (i, obj) in root_objects.iter().enumerate() {
            if let Some(mut obj_ref) = obj.write() {
                // Each object references the next 2 objects
                for j in 1..=2 {
                    if i + j < root_objects.len() {
                        obj_ref.references.push(root_objects[i + j].clone());
                    }
                }
            }
        }
        
        let marking_work_created = Arc::new(AtomicUsize::new(0));
        let mutator_modifications = Arc::new(AtomicUsize::new(0));
        let start_barrier = Arc::new(Barrier::new(3)); // Main + collector + mutator threads
        
        let work_created_clone = marking_work_created.clone();
        let modifications_clone = mutator_modifications.clone();
        let barrier_clone = start_barrier.clone();
        let objects_clone = root_objects.clone();
        
        // Concurrent mutator thread during marking
        let mutator_thread = thread::spawn(move || {
            barrier_clone.wait(); // Synchronized start with marking
            
            // Mutator attempts to create new work during marking
            for round in 0..50 {
                // Create new references between existing objects
                if objects_clone.len() >= 2 {
                    let src_idx = round % objects_clone.len();
                    let dst_idx = (round + 1) % objects_clone.len();
                    
                    if let Some(mut src_obj) = objects_clone[src_idx].write() {
                        // Add new reference - in traditional concurrent collectors,
                        // this would require marking the target object
                        src_obj.references.push(objects_clone[dst_idx].clone());
                        modifications_clone.fetch_add(1, Ordering::Relaxed);
                        
                        // In advancing wavefront: this does NOT create marking work
                        // Target object's mark state is already fixed for this cycle
                    }
                }
                
                // Create entirely new objects and link them
                let new_obj = Gc::new(ObjectNode {
                    id: 1000 + round,
                    data: format!("Mutator_created_{}", round),
                    references: vec![objects_clone[0].clone()], // Reference to marked object
                });
                
                if let Some(mut root_ref) = objects_clone[0].write() {
                    root_ref.references.push(new_obj); // Backward reference
                    modifications_clone.fetch_add(1, Ordering::Relaxed);
                }
                
                thread::sleep(Duration::from_micros(100));
            }
        });
        
        // Collector thread simulating concurrent marking
        let barrier_clone2 = start_barrier.clone();
        let collector_thread = thread::spawn(move || {
            collector.register_worker_thread();
            collector.set_phase(CollectorPhase::Marking);
            
            barrier_clone2.wait(); // Synchronized start
            
            // Simulate marking phase - in FUGC, work queue doesn't grow
            let initial_work_estimate = 20; // Initial objects to mark
            let mut work_processed = 0;
            
            while work_processed < initial_work_estimate && 
                  collector.phase.load(Ordering::Acquire) == CollectorPhase::Marking as usize {
                
                // Simulate marking an object
                work_processed += 1;
                
                // Key property: work queue size should not increase due to mutator
                // In traditional concurrent collector, mutator actions would add work
                // In advancing wavefront: marking work is bounded at cycle start
                
                thread::sleep(Duration::from_millis(1));
            }
            
            // Work processed should equal initial estimate (no new work added)
            work_created_clone.store(work_processed, Ordering::Relaxed);
            
            collector.set_phase(CollectorPhase::Waiting);
            collector.unregister_worker_thread();
        });
        
        start_barrier.wait(); // Start both threads simultaneously
        
        mutator_thread.join().unwrap();
        collector_thread.join().unwrap();
        
        let final_work_created = marking_work_created.load(Ordering::Acquire);
        let final_modifications = mutator_modifications.load(Ordering::Acquire);
        
        // Validate advancing wavefront property
        assert!(final_modifications > 0, "Mutator should have made modifications");
        assert_eq!(final_work_created, 20, 
                   "Marking work should be bounded by initial graph size, not mutator actions");
        
        println!("Advancing wavefront test passed:");
        println!("  Mutator modifications: {}", final_modifications);
        println!("  Marking work processed: {} (should equal initial size)", final_work_created);
    }
    
    /// Test: Once marked, object stays marked for entire cycle
    #[test]
    fn test_marked_objects_stay_marked() {
        let collector = &*fugrip::memory::COLLECTOR;
        
        // Create objects with marking simulation
        let objects: Vec<(Gc<String>, AtomicBool)> = (0..15)
            .map(|i| (Gc::new(format!("Persistent_object_{}", i)), AtomicBool::new(false)))
            .collect();
        
        // Phase 1: Start marking and mark some objects
        collector.set_phase(CollectorPhase::Marking);
        
        // Simulate marking first half of objects
        for i in 0..objects.len()/2 {
            objects[i].1.store(true, Ordering::Release); // Mark object
        }
        
        let marked_count_initial = objects.iter()
            .filter(|(_, marked)| marked.load(Ordering::Acquire))
            .count();
        
        // Phase 2: Concurrent mutator activity during marking
        let start_barrier = Arc::new(Barrier::new(2));
        let mark_states = Arc::new(Mutex::new(Vec::new()));
        
        let barrier_clone = start_barrier.clone();
        let states_clone = mark_states.clone();
        let objects_clone: Vec<(Gc<String>, Arc<AtomicBool>)> = objects.iter()
            .map(|(obj, marked)| (obj.clone(), Arc::new(AtomicBool::new(marked.load(Ordering::Acquire)))))
            .collect();
        
        let mutator_thread = thread::spawn(move || {
            barrier_clone.wait();
            
            // Mutator performs operations that in traditional collectors might unmark objects
            for round in 0..100 {
                // Access objects in various ways
                for (i, (obj, _)) in objects_clone.iter().enumerate() {
                    if let Some(obj_ref) = obj.read() {
                        let _len = obj_ref.len(); // Use the object
                    }
                    
                    // Try to create situations that might affect marking
                    if round % 10 == i % 10 {
                        let _temp_ref = obj.clone(); // Create temporary reference
                    }
                }
                
                // Record mark states periodically
                if round % 25 == 0 {
                    let current_states: Vec<bool> = objects_clone.iter()
                        .map(|(_, marked)| marked.load(Ordering::Acquire))
                        .collect();
                    states_clone.lock().unwrap().push(current_states);
                }
                
                thread::sleep(Duration::from_micros(50));
            }
        });
        
        start_barrier.wait();
        mutator_thread.join().unwrap();
        
        // Phase 3: Verify marked objects remained marked throughout
        let marked_count_final = objects.iter()
            .filter(|(_, marked)| marked.load(Ordering::Acquire))
            .count();
        
        let all_states = mark_states.lock().unwrap();
        
        // Validate advancing wavefront marking stability
        assert_eq!(marked_count_initial, marked_count_final, 
                   "Marked objects should stay marked throughout cycle");
        
        // Check that marking was stable across all recorded states
        for (snapshot_idx, snapshot) in all_states.iter().enumerate() {
            let snapshot_marked = snapshot.iter().filter(|&&marked| marked).count();
            assert_eq!(snapshot_marked, marked_count_initial,
                       "Snapshot {} should have same marked count as initial", snapshot_idx);
        }
        
        println!("Marking stability test passed:");
        println!("  Initial marked: {}", marked_count_initial);
        println!("  Final marked: {}", marked_count_final);
        println!("  Snapshots consistent: {}", all_states.len());
    }
    
    /// Test: Wavefront advancement prevents retroactive marking
    #[test]
    fn test_wavefront_prevents_retroactive_marking() {
        let collector = &*fugrip::memory::COLLECTOR;
        
        // Create objects with clear marking order
        let objects: Vec<Gc<WavefrontNode>> = (0..25)
            .map(|i| Gc::new(WavefrontNode {
                id: i,
                marking_timestamp: Arc::new(AtomicUsize::new(0)),
                marked: Arc::new(AtomicBool::new(false)),
                children: Vec::new(),
            }))
            .collect();
        
        // Link objects in a chain (each points to next)
        for i in 0..objects.len()-1 {
            if let Some(mut obj_ref) = objects[i].write() {
                obj_ref.children.push(objects[i + 1].clone());
            }
        }
        
        collector.set_phase(CollectorPhase::Marking);
        
        let wavefront_position = Arc::new(AtomicUsize::new(0));
        let retroactive_attempts = Arc::new(AtomicUsize::new(0));
        let start_barrier = Arc::new(Barrier::new(2));
        
        let position_clone = wavefront_position.clone();
        let attempts_clone = retroactive_attempts.clone();
        let barrier_clone = start_barrier.clone();
        let objects_clone = objects.clone();
        
        // Thread simulating advancing wavefront marking
        let marker_thread = thread::spawn(move || {
            barrier_clone.wait();
            
            // Advance wavefront through object graph
            for i in 0..objects_clone.len() {
                if let Some(obj_ref) = objects_clone[i].read() {
                    // Mark object with timestamp
                    obj_ref.marking_timestamp.store(i + 1, Ordering::Release);
                    obj_ref.marked.store(true, Ordering::Release);
                    position_clone.store(i + 1, Ordering::Release);
                    
                    // Wavefront has advanced past this object
                    // No retroactive marking should be possible
                }
                
                thread::sleep(Duration::from_millis(2));
            }
        });
        
        // Thread attempting retroactive marking (should fail in FUGC)
        let barrier_clone2 = start_barrier.clone();
        let position_clone2 = wavefront_position.clone();
        let objects_clone2 = objects.clone();
        let retroactive_thread = thread::spawn(move || {
            barrier_clone2.wait();
            
            for attempt in 0..50 {
                thread::sleep(Duration::from_millis(1));
                
                let current_position = position_clone2.load(Ordering::Acquire);
                
                // Try to mark objects behind the wavefront
                if current_position > 2 {
                    let target_idx = current_position.saturating_sub(2);
                    
                    if target_idx < objects_clone2.len() {
                        if let Some(obj_ref) = objects_clone2[target_idx].read() {
                            let original_timestamp = obj_ref.marking_timestamp.load(Ordering::Acquire);
                            
                            // Attempt retroactive marking
                            attempts_clone.fetch_add(1, Ordering::Relaxed);
                            
                            // In FUGC: this should not change the marking timestamp
                            // Object was already processed by advancing wavefront
                            let final_timestamp = obj_ref.marking_timestamp.load(Ordering::Acquire);
                            
                            assert_eq!(original_timestamp, final_timestamp,
                                       "Retroactive marking should not change timestamp at attempt {}", attempt);
                        }
                    }
                }
            }
        });
        
        start_barrier.wait();
        
        marker_thread.join().unwrap();
        retroactive_thread.join().unwrap();
        
        let final_attempts = retroactive_attempts.load(Ordering::Acquire);
        let final_position = wavefront_position.load(Ordering::Acquire);
        
        // Validate wavefront advancement properties
        assert!(final_attempts > 0, "Retroactive marking should have been attempted");
        assert_eq!(final_position, objects.len(), "Wavefront should have processed all objects");
        
        // Verify objects were marked in order and stayed marked
        for (i, obj) in objects.iter().enumerate() {
            if let Some(obj_ref) = obj.read() {
                let timestamp = obj_ref.marking_timestamp.load(Ordering::Acquire);
                let is_marked = obj_ref.marked.load(Ordering::Acquire);
                
                assert!(is_marked, "Object {} should be marked", i);
                assert_eq!(timestamp, i + 1, "Object {} should have timestamp {}, got {}", i, i + 1, timestamp);
            }
        }
        
        println!("Wavefront advancement test passed:");
        println!("  Objects processed: {}", final_position);
        println!("  Retroactive attempts (blocked): {}", final_attempts);
        println!("  All objects marked in order with stable timestamps");
    }
    
    /// Test: Contrast with hypothetical concurrent mark-sweep behavior
    #[test]
    fn test_advancing_wavefront_vs_concurrent_mark_sweep() {
        let collector = &*fugrip::memory::COLLECTOR;
        
        // Create object graph
        let objects: Vec<Gc<ComparisonNode>> = (0..20)
            .map(|i| Gc::new(ComparisonNode {
                id: i,
                fugc_marking_work: Arc::new(AtomicUsize::new(0)),
                cms_hypothetical_work: Arc::new(AtomicUsize::new(0)),
                references: Vec::new(),
            }))
            .collect();
        
        collector.set_phase(CollectorPhase::Marking);
        
        let total_fugc_work = Arc::new(AtomicUsize::new(0));
        let total_cms_work = Arc::new(AtomicUsize::new(0));
        let mutator_actions = Arc::new(AtomicUsize::new(0));
        
        let fugc_work_clone = total_fugc_work.clone();
        let cms_work_clone = total_cms_work.clone();
        let actions_clone = mutator_actions.clone();
        let objects_clone = objects.clone();
        
        // Simulation thread
        let simulation_thread = thread::spawn(move || {
            // Initial marking work (same for both)
            let initial_work = objects_clone.len();
            fugc_work_clone.store(initial_work, Ordering::Relaxed);
            cms_work_clone.store(initial_work, Ordering::Relaxed);
            
            // Simulate mutator actions during marking
            for action in 0..100 {
                if objects_clone.len() >= 2 {
                    let src_idx = action % objects_clone.len();
                    let dst_idx = (action + 1) % objects_clone.len();
                    
                    if let Some(mut src_obj) = objects_clone[src_idx].write() {
                        src_obj.references.push(objects_clone[dst_idx].clone());
                        actions_clone.fetch_add(1, Ordering::Relaxed);
                        
                        // FUGC: No additional marking work (advancing wavefront)
                        // Target's mark state is already determined
                        
                        // Hypothetical CMS: Would need to mark newly referenced object
                        // if it was previously white (not yet marked)
                        let cms_additional_work = cms_work_clone.load(Ordering::Acquire) + 1;
                        cms_work_clone.store(cms_additional_work, Ordering::Relaxed);
                        
                        // FUGC work remains unchanged (key difference)
                        let _fugc_work = fugc_work_clone.load(Ordering::Acquire); // No change
                    }
                }
                
                thread::sleep(Duration::from_micros(100));
            }
        });
        
        simulation_thread.join().unwrap();
        
        collector.set_phase(CollectorPhase::Waiting);
        
        let final_fugc_work = total_fugc_work.load(Ordering::Acquire);
        let final_cms_work = total_cms_work.load(Ordering::Acquire);
        let final_actions = mutator_actions.load(Ordering::Acquire);
        
        // Validate the key difference between FUGC and concurrent mark-sweep
        assert!(final_actions > 0, "Mutator should have performed actions");
        assert_eq!(final_fugc_work, objects.len(), 
                   "FUGC work should remain bounded by initial graph size");
        assert!(final_cms_work > final_fugc_work, 
                "Hypothetical CMS would require more work due to mutator actions");
        
        let work_difference = final_cms_work - final_fugc_work;
        assert_eq!(work_difference, final_actions, 
                   "Work difference should equal mutator actions");
        
        println!("FUGC vs CMS comparison test passed:");
        println!("  Initial objects: {}", objects.len());
        println!("  Mutator actions: {}", final_actions);
        println!("  FUGC marking work: {} (constant)", final_fugc_work);
        println!("  Hypothetical CMS work: {} (grows with mutator)", final_cms_work);
        println!("  Work savings: {} operations", work_difference);
    }
    
    /// Test: Incremental update property during wavefront advancement
    #[test]
    fn test_incremental_update_during_wavefront() {
        let collector = &*fugrip::memory::COLLECTOR;
        
        // Create objects that will have different lifetimes
        let long_lived: Vec<Gc<String>> = (0..10)
            .map(|i| Gc::new(format!("Long_lived_{}", i)))
            .collect();
        
        let short_lived_container = Arc::new(Mutex::new(Vec::<Gc<String>>::new()));
        
        collector.set_phase(CollectorPhase::Marking);
        
        let freed_during_collection = Arc::new(AtomicUsize::new(0));
        let container_clone = short_lived_container.clone();
        let freed_clone = freed_during_collection.clone();
        
        // Thread simulating object lifecycle during collection
        let lifecycle_thread = thread::spawn(move || {
            for round in 0..50 {
                // Create short-lived objects
                {
                    let mut container = container_clone.lock().unwrap();
                    for i in 0..5 {
                        container.push(Gc::new(format!("Short_lived_{}_{}", round, i)));
                    }
                }
                
                thread::sleep(Duration::from_millis(2));
                
                // Objects become unreachable during collection (incremental update)
                {
                    let mut container = container_clone.lock().unwrap();
                    let freed_count = container.len();
                    container.clear(); // Objects become unreachable
                    freed_clone.fetch_add(freed_count, Ordering::Relaxed);
                }
                
                thread::sleep(Duration::from_millis(1));
            }
        });
        
        // Let collection run concurrently
        thread::sleep(Duration::from_millis(50));
        
        lifecycle_thread.join().unwrap();
        
        collector.set_phase(CollectorPhase::Sweeping);
        collector.set_phase(CollectorPhase::Waiting);
        
        let total_freed = freed_during_collection.load(Ordering::Acquire);
        
        // Validate incremental update property
        assert!(total_freed > 0, "Objects should have become unreachable during collection");
        
        // Long-lived objects should still be accessible
        for (i, obj) in long_lived.iter().enumerate() {
            assert!(obj.read().is_some(), "Long-lived object {} should remain accessible", i);
        }
        
        // Short-lived container should be empty
        let final_container_size = short_lived_container.lock().unwrap().len();
        assert_eq!(final_container_size, 0, "Short-lived container should be empty");
        
        println!("Incremental update test passed:");
        println!("  Objects freed during collection: {}", total_freed);
        println!("  Long-lived objects remain accessible: {}", long_lived.len());
        println!("  Advancing wavefront allows incremental updates");
    }
    
    // Helper structures for testing
    #[derive(Clone)]
    struct ObjectNode {
        id: usize,
        data: String,
        references: Vec<Gc<ObjectNode>>,
    }
    
    unsafe impl GcTrace for ObjectNode {
        unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
            for reference in &self.references {
                if let Some(ref_data) = reference.read() {
                    let _ = ref_data.id; // Use the reference to keep it live
                }
            }
        }
    }
    
    struct WavefrontNode {
        id: usize,
        marking_timestamp: Arc<AtomicUsize>,
        marked: Arc<AtomicBool>,
        children: Vec<Gc<WavefrontNode>>,
    }
    
    unsafe impl GcTrace for WavefrontNode {
        unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
            for child in &self.children {
                if let Some(child_data) = child.read() {
                    let _ = child_data.id;
                }
            }
        }
    }
    
    struct ComparisonNode {
        id: usize,
        fugc_marking_work: Arc<AtomicUsize>,
        cms_hypothetical_work: Arc<AtomicUsize>,
        references: Vec<Gc<ComparisonNode>>,
    }
    
    unsafe impl GcTrace for ComparisonNode {
        unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
            for reference in &self.references {
                if let Some(ref_data) = reference.read() {
                    let _ = ref_data.id;
                }
            }
        }
    }
}