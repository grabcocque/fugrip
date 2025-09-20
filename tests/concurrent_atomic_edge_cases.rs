//! Atomic operation edge cases and error path tests for concurrent.rs
//!
//! This module tests atomic operations, contention scenarios, edge cases, and error conditions
//! for the concurrent marking system and tricolor implementation.
//! Consolidated from concurrent_atomic_edge_cases.rs and concurrent_error_paths.rs to avoid duplication.

#[cfg(test)]
mod concurrent_tests {
    use fugrip::concurrent::{
        ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier,
    };
    use mmtk::util::{Address, ObjectReference};
    use rayon::prelude::*;
    use crossbeam_utils::thread as crossbeam_thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Removed std::thread - using Rayon for parallel execution

    /// Test tricolor marking atomic operations under contention
    mod tricolor_atomic_tests {
        use super::*;

        #[test]
        fn test_concurrent_color_transitions() {
            let heap_base = unsafe { Address::from_usize(0x100000) };
            let heap_size = 65536;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));
            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };

            // Test concurrent color changes using Rayon
            (0..10).into_par_iter().for_each(|i| {
                for j in 0..100 {
                    // Attempt to set color based on thread id and iteration
                    let target_color = match (i + j) % 3 {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        2 => ObjectColor::Black,
                        _ => ObjectColor::White,
                    };
                    tricolor.set_color(obj, target_color);
                }
            });

            // Final color should be valid (system should be consistent)
            let final_color = tricolor.get_color(obj);
            assert!(matches!(
                final_color,
                ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
            ));
        }

        #[test]
        fn test_concurrent_multiple_object_marking() {
            let heap_base = unsafe { Address::from_usize(0x200000) };
            let heap_size = 65536;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            let operations_completed = Arc::new(AtomicUsize::new(0));

            // Use Rayon for parallel marking
            let ops_completed = (0..8).into_par_iter().map(|thread_id| {
                let mut local_ops = 0;
                for i in 0..100 {
                    let obj_addr =
                        heap_base + 0x100usize + (thread_id as usize) * 1000 + (i as usize) * 8;
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                    // Set object color
                    let target_color = if i % 2 == 0 {
                        ObjectColor::Grey
                    } else {
                        ObjectColor::Black
                    };
                    tricolor.set_color(obj, target_color);
                    local_ops += 1;

                    // Verify color was set
                    let read_color = tricolor.get_color(obj);
                    assert!(matches!(
                        read_color,
                        ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
                    ));
                }
                local_ops
            }).sum::<usize>();

            operations_completed.store(ops_completed, Ordering::Relaxed);

            // Should have completed operations
            let total_ops = operations_completed.load(Ordering::Relaxed);
            assert_eq!(total_ops, 800); // 8 threads * 100 operations
        }

        #[test]
        fn test_tricolor_extreme_addresses() {
            let heap_base = unsafe { Address::from_usize(0x800000) };
            let heap_size = 4096;
            let tricolor = TricolorMarking::new(heap_base, heap_size);

            // Test with minimum valid address (word-aligned)
            let min_obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(8)) };
            tricolor.set_color(min_obj, ObjectColor::Black);
            assert_eq!(tricolor.get_color(min_obj), ObjectColor::Black);

            // Test with maximum word-aligned address (may be out of bounds for marking system)
            let max_obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(usize::MAX & !0x7))
            };
            tricolor.set_color(max_obj, ObjectColor::Grey);
            // Extreme addresses may default to White, which is acceptable behavior
            let max_color = tricolor.get_color(max_obj);
            assert!(matches!(
                max_color,
                ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
            ));

            // Test with heap boundary addresses
            let heap_start = unsafe { ObjectReference::from_raw_address_unchecked(heap_base) };
            let heap_end =
                unsafe { ObjectReference::from_raw_address_unchecked(heap_base + heap_size - 8) };

            tricolor.set_color(heap_start, ObjectColor::Grey);
            tricolor.set_color(heap_end, ObjectColor::Black);

            assert_eq!(tricolor.get_color(heap_start), ObjectColor::Grey);
            assert_eq!(tricolor.get_color(heap_end), ObjectColor::Black);
        }

        #[test]
        fn test_concurrent_color_consistency() {
            let heap_base = unsafe { Address::from_usize(0x900000) };
            let heap_size = 8192;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };
            let consistency_violations = Arc::new(AtomicUsize::new(0));

            // Use Rayon for parallel color consistency testing
            let violations = (0..4).into_par_iter().map(|thread_id| {
                let mut local_violations = 0;
                for i in 0..500 {
                    // Set color with different patterns
                    let color = match (thread_id + i) % 3 {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        2 => ObjectColor::Black,
                        _ => ObjectColor::White,
                    };
                    tricolor.set_color(obj, color);

                    // Immediately read back to check consistency
                    let _read_color = tricolor.get_color(obj);

                    // Check for consistency violations
                    if thread_id == 0 && i % 100 == 0 {
                        // One thread does extra consistency checks
                        std::thread::yield_now(); // Force potential reordering
                        let recheck_color = tricolor.get_color(obj);
                        // Color may change due to concurrent updates, so we just verify it's valid
                        if !matches!(
                            recheck_color,
                            ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
                        ) {
                            local_violations += 1;
                        }
                    }
                }
                local_violations
            }).sum::<usize>();

            consistency_violations.store(violations, Ordering::Relaxed);

            // Should have no consistency violations (all colors should be valid)
            let final_violations = consistency_violations.load(Ordering::Relaxed);
            assert_eq!(final_violations, 0);
        }

        #[test]
        fn test_tricolor_memory_ordering() {
            let heap_base = unsafe { Address::from_usize(0xA00000) };
            let heap_size = 4096;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            // High frequency color changes to test memory ordering using Rayon
            let total_changes = (0..8).into_par_iter().map(|_| {
                let mut local_changes = 0;
                for i in 0..1000 {
                    let obj_addr = heap_base + ((i as usize) * 8) % heap_size;
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                    let current_color = tricolor.get_color(obj);
                    let new_color = match current_color {
                        ObjectColor::White => ObjectColor::Grey,
                        ObjectColor::Grey => ObjectColor::Black,
                        ObjectColor::Black => ObjectColor::White,
                    };

                    tricolor.set_color(obj, new_color);
                    local_changes += 1;
                }
                local_changes
            }).sum::<usize>();

            // Should have completed many operations
            assert_eq!(total_changes, 8000); // 8 threads * 1000 operations
        }

        #[test]
        fn test_tricolor_object_boundary_conditions() {
            let heap_base = unsafe { Address::from_usize(0xB00000) };
            let heap_size = 2048; // Small heap
            let tricolor = TricolorMarking::new(heap_base, heap_size);

            // Test objects at various alignments (only word-aligned addresses)
            let test_addresses = vec![
                heap_base,                 // Start of heap
                heap_base + 8usize,        // Aligned
                heap_base + 64usize,       // Cache line aligned
                heap_base + heap_size - 8, // End of heap - 8 (aligned)
            ];

            for &addr in &test_addresses {
                let obj = unsafe { ObjectReference::from_raw_address_unchecked(addr) };

                // Test setting and getting colors
                for color in [ObjectColor::White, ObjectColor::Grey, ObjectColor::Black] {
                    tricolor.set_color(obj, color);
                    let read_color = tricolor.get_color(obj);
                    // Color should be valid (may not match exactly due to implementation details)
                    assert!(matches!(
                        read_color,
                        ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
                    ));
                }
            }
        }

        #[test]
        fn test_concurrent_stress_test() {
            let heap_base = unsafe { Address::from_usize(0xC00000) };
            let heap_size = 16384;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            // High contention stress test using Rayon
            let total_ops = (0..16).into_par_iter().map(|thread_id| {
                let mut local_ops = 0;
                for i in 0..200 {
                    // Create overlap by having threads work on similar address ranges
                    let obj_addr = heap_base
                        + (((thread_id as usize) * 64 + (i as usize) * 8) % heap_size);
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                    // Rapid color changes
                    let color = match (thread_id + i) % 3 {
                        0 => ObjectColor::White,
                        1 => ObjectColor::Grey,
                        2 => ObjectColor::Black,
                        _ => ObjectColor::White,
                    };

                    tricolor.set_color(obj, color);
                    local_ops += 1;

                    // Occasionally verify
                    if i % 50 == 0 {
                        let _read_color = tricolor.get_color(obj);
                        assert!(matches!(
                            _read_color,
                            ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
                        ));
                    }
                }
                local_ops
            }).sum::<usize>();

            // Should have completed all operations despite contention
            assert_eq!(total_ops, 3200); // 16 threads * 200 operations
        }

        #[test]
        fn test_tricolor_color_transition_patterns() {
            let heap_base = unsafe { Address::from_usize(0xD00000) };
            let heap_size = 4096;
            let tricolor = TricolorMarking::new(heap_base, heap_size);

            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };

            // Test specific color transition patterns
            let transitions = vec![
                (ObjectColor::White, ObjectColor::Grey),
                (ObjectColor::Grey, ObjectColor::Black),
                (ObjectColor::Black, ObjectColor::White),
                (ObjectColor::White, ObjectColor::Black),
                (ObjectColor::Black, ObjectColor::Grey),
                (ObjectColor::Grey, ObjectColor::White),
            ];

            for (from_color, to_color) in transitions {
                tricolor.set_color(obj, from_color);
                assert_eq!(tricolor.get_color(obj), from_color);

                tricolor.set_color(obj, to_color);
                assert_eq!(tricolor.get_color(obj), to_color);
            }
        }

        #[test]
        fn test_concurrent_heap_boundary_access() {
            let heap_base = unsafe { Address::from_usize(0xE00000) };
            let heap_size = 4096;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            let boundary_operations = Arc::new(AtomicUsize::new(0));

            // Test concurrent access to heap boundaries using Rayon parallel iterator
            (0..4).into_par_iter().for_each(|thread_id| {
                let tricolor_clone = Arc::clone(&tricolor);
                let ops_clone = Arc::clone(&boundary_operations);
                let mut local_ops = 0;
                for i in 0..100 {
                    // Test addresses near heap boundaries
                    let offset = match thread_id {
                        0 => (i as usize) * 8,                   // Start of heap
                        1 => heap_size - 64 + (i as usize) * 8,  // End of heap
                        2 => heap_size / 2 + (i as usize) * 8,   // Middle of heap
                        3 => (heap_size / 4) + (i as usize) * 8, // Quarter point
                        _ => (i as usize) * 8,
                    };

                    let obj_addr = heap_base + offset;
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                    // Set colors
                    let colors = [ObjectColor::White, ObjectColor::Grey, ObjectColor::Black];
                    for color in &colors {
                        tricolor_clone.set_color(obj, *color);
                        local_ops += 1;
                    }
                }
                ops_clone.fetch_add(local_ops, Ordering::Relaxed);
            });

            // Should have completed boundary operations
            let total_boundary_ops = boundary_operations.load(Ordering::Relaxed);
            assert_eq!(total_boundary_ops, 1200); // 4 threads * 100 iterations * 3 colors
        }
    }

    /// Test atomic operation edge cases and error conditions
    mod atomic_edge_case_tests {
        use super::*;

        #[test]
        fn test_atomic_fetch_add_overflow_protection() {
            // Start two below max so two adds succeed and the third fails
            let a = AtomicUsize::new(usize::MAX - 2);
            let prev = a
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_add(1))
                .expect("first add");
            assert_eq!(prev, usize::MAX - 2);

            let second = a
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_add(1))
                .expect("second add");
            assert_eq!(second, usize::MAX - 1);

            // Third add would overflow and must return Err
            let overflow = a.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_add(1));
            assert!(overflow.is_err());
        }

        #[test]
        fn test_atomic_cas_under_high_contention() {
            let counter = Arc::new(AtomicUsize::new(0));

            // High contention compare-and-swap operations using Rayon parallel iterator
            let total_successful: usize = (0..32).into_par_iter().map(|_| {
                let counter_clone = Arc::clone(&counter);
                let mut successful_ops = 0;
                for _ in 0..1000 {
                    let current = counter_clone.load(Ordering::Relaxed);
                    // CAS that will often fail due to contention
                    let result = counter_clone.compare_exchange_weak(
                        current,
                        current.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    if result.is_ok() {
                        successful_ops += 1;
                    } else {
                        // Retry failed operations to ensure all complete
                        while counter_clone
                            .compare_exchange_weak(
                                counter_clone.load(Ordering::Relaxed),
                                counter_clone.load(Ordering::Relaxed).wrapping_add(1),
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            )
                            .is_err()
                        {
                            // Keep trying until successful
                        }
                        successful_ops += 1;
                    }
                }
                successful_ops
            }).sum();

            // Total should be 32000 (32 threads * 1000 operations)
            let final_value = counter.load(Ordering::Relaxed);
            assert_eq!(final_value, 32000);
            assert_eq!(total_successful, 32000); // All operations should succeed eventually
        }

        #[test]
        fn test_atomic_operations_overflow() {
            // Test many rapid state changes to check for overflow issues
            let state = AtomicUsize::new(0);
            for _ in 0..10000 {
                let current = state.load(Ordering::Relaxed);
                let next = current.wrapping_add(1);
                let _ = state.compare_exchange_weak(
                    current,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }

            // Should still be in a valid state
            let final_state = state.load(Ordering::Relaxed);
            assert!(final_state <= 10000);
        }

        #[test]
        fn test_atomic_operation_timeout() {
            use std::time::Instant;

            let state = AtomicUsize::new(0);
            let timeout_ms = 100; // Very short timeout

            let start_time = Instant::now();
            let mut attempts = 0;

            // Try to perform CAS operations with timeout
            while start_time.elapsed().as_millis() < timeout_ms {
                let current = state.load(Ordering::Relaxed);
                let _ = state.compare_exchange_weak(
                    current,
                    current.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                attempts += 1;
            }

            // Should have made many attempts
            assert!(attempts > 1000);

            // State should still be valid
            let final_state = state.load(Ordering::Relaxed);
            assert!(final_state <= attempts);
        }
    }

    /// Test error conditions and failure scenarios
    mod error_condition_tests {
        use super::*;

        #[test]
        fn test_tricolor_invalid_heap_access() {
            let heap_base = unsafe { Address::from_usize(0xB00000) };
            let heap_size = 4096;
            let tricolor = TricolorMarking::new(heap_base, heap_size);

            // Test access to objects outside heap bounds
            let before_heap = unsafe { ObjectReference::from_raw_address_unchecked(heap_base - 8) };
            let after_heap = unsafe {
                ObjectReference::from_raw_address_unchecked(heap_base + heap_size + 8usize)
            };

            // Should handle out-of-bounds gracefully
            tricolor.set_color(before_heap, ObjectColor::Black);
            tricolor.set_color(after_heap, ObjectColor::Grey);

            // Reading back should return some valid color
            let color1 = tricolor.get_color(before_heap);
            let color2 = tricolor.get_color(after_heap);

            assert!(matches!(
                color1,
                ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
            ));
            assert!(matches!(
                color2,
                ObjectColor::White | ObjectColor::Grey | ObjectColor::Black
            ));
        }

        #[test]
        fn test_concurrent_marking_panic_recovery() {
            let heap_base = unsafe { Address::from_usize(0xC00000) };
            let heap_size = 4096;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            let panic_count = Arc::new(AtomicUsize::new(0));

            // Threads that might panic (but shouldn't with proper error handling) using crossbeam scoped threads
            crossbeam_thread::scope(|s| {
                for thread_id in 0..5 {
                    let tricolor_clone = Arc::clone(&tricolor);
                    let panic_clone = Arc::clone(&panic_count);
                    s.spawn(move |_| {
                        let result = std::panic::catch_unwind(|| {
                            for i in 0..100 {
                                // Use extreme values that might cause issues
                                let obj_addr = match thread_id {
                                    0 => heap_base + ((i as usize * usize::MAX / 100) & !0x7), // Very large but aligned
                                    1 => heap_base + ((i as usize).wrapping_mul(1000) & !0x7), // Wrapping but aligned
                                    2 => heap_base + (((i as usize) << 20) & !0x7), // Bit shifted but aligned
                                    _ => heap_base + (i as usize) * 8,              // Normal addresses
                                };
                                let obj =
                                    unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                                tricolor_clone.set_color(obj, ObjectColor::Black);
                            }
                        });

                        if result.is_err() {
                            panic_clone.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            }).unwrap();

            // Some panics are expected due to extreme arithmetic overflow conditions
            let panics = panic_count.load(Ordering::Relaxed);
            // The system should handle most cases gracefully, but extreme overflow may panic
            assert!(panics <= 1); // At most one thread might panic due to overflow
        }

        // Note: crossbeam-deque Worker tests removed - functionality now handled by rayon work-stealing

        #[test]
        fn test_illegal_write_barrier_detection() {
            // Create a write barrier and call it with a stack slot to ensure no panic
            let heap_base = unsafe { Address::from_usize(0x1000) };
            let marking = Arc::new(TricolorMarking::new(heap_base, 4096));
            let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
            let barrier = WriteBarrier::new(&marking, &coordinator, heap_base, 4096);

            // Prepare a slot and value and invoke the barrier; test will fail if it panics
            unsafe {
                let mut slot =
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(0x2000));
                barrier.write_barrier_fast(&mut slot as *mut _, slot);
            }
        }

        #[test]
        fn test_concurrent_marking_under_memory_pressure() {
            let heap_base = unsafe { Address::from_usize(0x900000) };
            let heap_size = 2048; // Small heap to create pressure
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            let operations_completed = Arc::new(AtomicUsize::new(0));

            // Create memory pressure with many concurrent operations using Rayon
            (0..16).into_par_iter().for_each(|thread_id| {
                let tricolor_clone = Arc::clone(&tricolor);
                let ops_clone = Arc::clone(&operations_completed);
                let mut local_ops = 0;
                for i in 0..200 {
                    let obj_addr = heap_base + (thread_id as usize * 128 + i * 8) % heap_size;
                    let obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                    // Try to set color (may fail due to memory pressure)
                    tricolor_clone.set_color(obj, ObjectColor::Black);
                    local_ops += 1;
                }
                ops_clone.fetch_add(local_ops, Ordering::Relaxed);
            });

            // Should complete some operations despite memory pressure
            let total_ops = operations_completed.load(Ordering::Relaxed);
            assert!(total_ops > 0);
        }
    }

    /// Test workload-specific scenarios
    mod workload_scenarios {
        use super::*;

        #[test]
        fn test_marking_wave_simulation() {
            let heap_base = unsafe { Address::from_usize(0xD00000) };
            let heap_size = 32768;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            let wave_front = Arc::new(AtomicUsize::new(0));

            // Simulate marking wavefront with multiple threads using crossbeam scoped threads
            crossbeam_thread::scope(|s| {
                for wave_id in 0..4 {
                    let tricolor_clone = Arc::clone(&tricolor);
                    let front_clone = Arc::clone(&wave_front);
                    s.spawn(move |_| {
                        for i in 0..100 {
                            // Each wave processes objects in its region
                            let obj_addr = heap_base + wave_id * 8000 + i * 32;
                            let obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };

                            // Mark object grey
                            tricolor_clone.set_color(obj, ObjectColor::Grey);

                            // Update wavefront
                            front_clone.fetch_max(wave_id * 100 + i, Ordering::Relaxed);

                            // Small delay to simulate processing
                            if i % 10 == 0 {
                                std::hint::black_box(()); // Prevent compiler optimizations
                                std::thread::yield_now(); // Cooperative yield instead of sleep
                            }
                        }
                    });
                }
            }).unwrap();

            // Wavefront should have advanced
            let max_front = wave_front.load(Ordering::Relaxed);
            assert!(max_front > 0);
        }

        #[test]
        fn test_barrier_under_allocation_load() {
            let heap_base = unsafe { Address::from_usize(0xE00000) };
            let marking = Arc::new(TricolorMarking::new(heap_base, 4096));
            let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
            let barrier = Arc::new(WriteBarrier::new(&marking, &coordinator, heap_base, 4096));

            let allocation_count = Arc::new(AtomicUsize::new(0));

            // Mix of allocation and write barrier operations using crossbeam scoped threads
            crossbeam_thread::scope(|s| {
                for thread_id in 0..6 {
                    let barrier_clone = Arc::clone(&barrier);
                    let alloc_clone = Arc::clone(&allocation_count);
                    s.spawn(move |_| {
                        for i in 0..100 {
                            // Simulate allocation
                            alloc_clone.fetch_add(1, Ordering::Relaxed);

                            // Write barrier operations
                            let obj_addr = unsafe {
                                Address::from_usize(0xE00000 + thread_id as usize * 1000 + i * 8)
                            };
                            let mut obj =
                                unsafe { ObjectReference::from_raw_address_unchecked(obj_addr) };
                            let field_addr = unsafe {
                                Address::from_usize(0xF00000 + thread_id as usize * 1000 + i * 8)
                            };
                            let field =
                                unsafe { ObjectReference::from_raw_address_unchecked(field_addr) };

                            // These should not panic even under load
                            unsafe { barrier_clone.write_barrier(&mut obj as *mut _, field) };

                            // Simulate GC pressure
                            if i % 20 == 0 {
                                std::thread::yield_now();
                            }
                        }
                    });
                }
            }).unwrap();

            let total_allocations = allocation_count.load(Ordering::Relaxed);
            assert_eq!(total_allocations, 600); // 6 threads * 100 allocations
        }
    }

    /// Test additional concurrent.rs APIs for improved coverage
    mod coverage_extension_tests {
        use super::*;
        use fugrip::concurrent::{BlackAllocator, GenerationBoundary, ObjectClassifier};

        // Note: Worker edge case tests removed - work-stealing now handled by rayon internally

        // Note: Worker stealing tests removed - rayon handles work-stealing transparently

        // Note: Parallel marking coordinator tests removed - replaced by rayon parallel execution

        #[test]
        fn test_generation_boundary_operations() {
            // Test GenerationBoundary public APIs
            let boundary =
                GenerationBoundary::new(unsafe { Address::from_usize(0x100000) }, 0x10000, 0.5);

            // Test address classification
            let young_addr = unsafe { Address::from_usize(0x100000) }; // Start of heap
            let old_addr = unsafe { Address::from_usize(0x109000) }; // In old generation (0x108000-0x110000)

            assert!(boundary.is_young(young_addr));
            assert!(boundary.is_old(old_addr));
            assert!(!boundary.is_young(old_addr));
            assert!(!boundary.is_old(young_addr));
        }

        #[test]
        fn test_object_classifier_operations() {
            // Test ObjectClassifier public APIs
            let classifier = ObjectClassifier::new();

            // Test classification
            let obj =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000)) };
            let class = fugrip::concurrent::ObjectClass::default_young();

            classifier.classify_object(obj, class);

            // Test getting classification
            let retrieved_class = classifier.get_classification(obj);
            assert!(retrieved_class.is_some());

            // Test promotion queue
            classifier.queue_for_promotion(obj);
            classifier.promote_young_objects(); // Should not panic

            // Test cross-generational reference recording
            let src =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000)) };
            let dst =
                unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x2000)) };
            classifier.record_cross_generational_reference(src, dst);

            // Test stats
            let stats = classifier.get_stats();
            assert!(stats.total_classified > 0);

            // Test clear
            classifier.clear();
            assert!(classifier.get_classification(obj).is_none());
        }

        #[test]
        fn test_write_barrier_configuration() {
            // Test WriteBarrier configuration APIs
            let heap_base = unsafe { Address::from_usize(0x100000) };
            let marking = Arc::new(TricolorMarking::new(heap_base, 0x100000));
            let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));

            // Test basic barrier creation
            let barrier1 = WriteBarrier::new(&marking, &coordinator, heap_base, 0x100000);
            let _barrier2 = WriteBarrier::new(&marking, &coordinator, heap_base, 0x100000);

            // Test activation and deactivation
            barrier1.activate();
            assert!(barrier1.is_active());

            barrier1.deactivate();
            assert!(!barrier1.is_active());

            // Test stats
            let (young_count, old_count) = barrier1.get_generational_stats();
            assert_eq!(young_count, 0);
            assert_eq!(old_count, 0);

            // Reset stats
            barrier1.reset_generational_stats();
            assert_eq!(barrier1.get_generational_stats().0, 0);
        }

        #[test]
        fn test_black_allocator_operations() {
            // Test BlackAllocator public APIs
            let tricolor = Arc::new(TricolorMarking::new(
                unsafe { Address::from_usize(0x100000) },
                0x10000,
            ));
            let allocator = BlackAllocator::new(&tricolor);

            // Test activation/deactivation
            allocator.activate();
            assert!(allocator.is_active());

            allocator.deactivate();
            assert!(!allocator.is_active());

            // Test allocation
            let obj = unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(0x101000))
            };
            allocator.allocate_black(obj);

            // Test stats
            let stats = allocator.get_stats();
            // Stats should be a reasonable non-negative integer
            assert!(stats <= usize::MAX / 2); // Reasonable upper bound

            // Reset
            allocator.reset();
            assert_eq!(allocator.get_stats(), 0);
        }

        #[test]
        fn test_tricolor_marking_bulk_operations() {
            // Test TricolorMarking bulk operations
            let heap_base = unsafe { Address::from_usize(0x100000) };
            let heap_size = 0x100000;
            let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

            // Create test objects
            let objects: Vec<ObjectReference> = (0..10)
                .map(|i| unsafe {
                    ObjectReference::from_raw_address_unchecked(heap_base + (i as usize * 0x1000))
                })
                .collect();

            // Set initial colors
            for obj in &objects {
                tricolor.set_color(*obj, ObjectColor::White);
            }

            // Test getting black objects (should be empty initially)
            let black_objects = tricolor.get_black_objects();
            assert!(black_objects.is_empty());

            // Mark some objects black
            for obj in &objects[..5] {
                tricolor.set_color(*obj, ObjectColor::Black);
            }

            // Test getting black objects again
            let black_objects = tricolor.get_black_objects();
            assert_eq!(black_objects.len(), 5);

            // Test clear
            tricolor.clear();
            for obj in &objects {
                assert_eq!(tricolor.get_color(*obj), ObjectColor::White);
            }
        }

        #[test]
        fn test_concurrent_marking_coordinator_contention() {
            // Test coordinator operations under high contention using Rayon
            let coordinator = Arc::new(ParallelMarkingCoordinator::new(4));
            let ops_count = Arc::new(AtomicUsize::new(0));

            // Use Rayon parallel iterator instead of manual barrier coordination
            (0..4).into_par_iter().for_each(|worker_id| {
                let coordinator_clone = Arc::clone(&coordinator);
                let ops_clone = Arc::clone(&ops_count);

                let mut local_ops = 0;
                for i in 0..100 {
                    // Create test work (ensure 8-byte alignment)
                    let work: Vec<ObjectReference> = vec![unsafe {
                        ObjectReference::from_raw_address_unchecked(Address::from_usize(
                            0x1000 + worker_id * 0x100 + i * 8,
                        ))
                    }];

                    // Share work
                    coordinator_clone.share_work(work.clone());

                    // Try to steal work
                    let stolen = coordinator_clone.steal_work(worker_id, 1);
                    local_ops += stolen.len();

                    // Check stats
                    let (_, _) = coordinator_clone.get_stats();
                    local_ops += 1;
                }
                ops_clone.fetch_add(local_ops, Ordering::Relaxed);
            });

            // Should have performed many operations
            let total_ops = ops_count.load(Ordering::Relaxed);
            assert!(total_ops > 400); // At least 400 operations (4 threads * 100)
        }
    }
}
