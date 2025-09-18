//! Performance tests for write barrier fast path optimizations

use std::sync::Arc;
use std::time::Instant;

use fugrip::concurrent::{ObjectColor, ParallelMarkingCoordinator, TricolorMarking, WriteBarrier};
use mmtk::util::{Address, ObjectReference};

/// Benchmark configuration
const BENCHMARK_ITERATIONS: usize = 1_000_000;
const WARMUP_ITERATIONS: usize = 100_000;

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test setup for write barrier benchmarks
    fn setup_write_barrier() -> (WriteBarrier, Vec<ObjectReference>) {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let heap_size = 0x100000; // 1MB heap
        let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
        let barrier = WriteBarrier::new(tricolor_marking, coordinator, heap_base, heap_size);

        // Create test objects
        let objects: Vec<ObjectReference> = (0..1000)
            .map(|i| unsafe {
                ObjectReference::from_raw_address_unchecked(heap_base + (i * 8usize))
            })
            .collect();

        (barrier, objects)
    }

    /// Benchmark inactive write barrier (common case)
    #[test]
    fn bench_write_barrier_inactive() {
        let (barrier, objects) = setup_write_barrier();

        // Ensure barrier is inactive (should be by default)
        assert!(!barrier.is_active());

        let mut slots: Vec<ObjectReference> = vec![objects[0]; BENCHMARK_ITERATIONS];
        let new_value = objects[1];

        // Warmup
        for i in 0..WARMUP_ITERATIONS {
            let slot_ptr = &mut slots[i % 100] as *mut ObjectReference;
            unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        }

        // Benchmark fast path
        let start = Instant::now();
        for slot in slots.iter_mut().take(BENCHMARK_ITERATIONS) {
            let slot_ptr = slot as *mut ObjectReference;
            unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        }
        let fast_duration = start.elapsed();

        // Benchmark standard interface
        let start = Instant::now();
        for slot in slots.iter_mut().take(BENCHMARK_ITERATIONS) {
            let slot_ptr = slot as *mut ObjectReference;
            unsafe { barrier.write_barrier(slot_ptr, new_value) };
        }
        let standard_duration = start.elapsed();

        println!("Inactive barrier performance:");
        println!(
            "  Fast path: {:?} ({:.2} ns/op)",
            fast_duration,
            fast_duration.as_nanos() as f64 / BENCHMARK_ITERATIONS as f64
        );
        println!(
            "  Standard:  {:?} ({:.2} ns/op)",
            standard_duration,
            standard_duration.as_nanos() as f64 / BENCHMARK_ITERATIONS as f64
        );

        // Fast path should be at least as fast as standard
        // Debug builds have different optimization characteristics
        let margin = if cfg!(debug_assertions) { 200 } else { 110 };
        assert!(fast_duration <= standard_duration * margin / 100);
    }

    /// Benchmark active write barrier (slow path)
    #[test]
    fn bench_write_barrier_active() {
        let (barrier, objects) = setup_write_barrier();

        // Activate the barrier
        barrier.activate();
        assert!(barrier.is_active());

        // Set up some white objects for the barrier to process
        for obj in &objects[0..10] {
            barrier.tricolor_marking.set_color(*obj, ObjectColor::White);
        }

        let mut slots: Vec<ObjectReference> = vec![objects[0]; BENCHMARK_ITERATIONS];
        let new_value = objects[1];

        // Warmup
        for i in 0..WARMUP_ITERATIONS {
            let slot_ptr = &mut slots[i % 100] as *mut ObjectReference;
            unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        }

        // Benchmark fast path (which delegates to slow path when active)
        let start = Instant::now();
        for slot in slots.iter_mut().take(BENCHMARK_ITERATIONS) {
            let slot_ptr = slot as *mut ObjectReference;
            unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        }
        let active_duration = start.elapsed();

        println!("Active barrier performance:");
        println!(
            "  With slow path: {:?} ({:.2} ns/op)",
            active_duration,
            active_duration.as_nanos() as f64 / BENCHMARK_ITERATIONS as f64
        );

        // Active barriers are expected to be slower, but should still be reasonable
        assert!(active_duration.as_millis() < 5000); // Should complete within 5 seconds
    }

    /// Benchmark bulk write barrier operations
    #[test]
    fn bench_bulk_write_barrier() {
        let (barrier, objects) = setup_write_barrier();

        // Test with different batch sizes
        for batch_size in [1, 2, 3, 8, 16, 64] {
            let updates: Vec<_> = (0..batch_size)
                .map(|i| {
                    let mut slot = objects[i % objects.len()];
                    let slot_ptr = &mut slot as *mut ObjectReference;
                    let new_value = objects[(i + 1) % objects.len()];
                    (slot_ptr, new_value)
                })
                .collect();

            let iterations = BENCHMARK_ITERATIONS / batch_size.max(1);

            // Warmup
            for _ in 0..iterations.min(WARMUP_ITERATIONS) {
                unsafe { barrier.write_barrier_bulk_fast(&updates) };
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..iterations {
                unsafe { barrier.write_barrier_bulk_fast(&updates) };
            }
            let duration = start.elapsed();

            println!(
                "Bulk barrier (batch size {}): {:?} ({:.2} ns/update)",
                batch_size,
                duration,
                duration.as_nanos() as f64 / (iterations * batch_size) as f64
            );
        }
    }

    /// Benchmark array write barrier operations
    #[test]
    fn bench_array_write_barrier() {
        let (barrier, objects) = setup_write_barrier();

        let mut array: Vec<ObjectReference> = vec![objects[0]; 1000];
        let new_value = objects[1];
        let element_size = std::mem::size_of::<ObjectReference>();

        // Warmup
        for i in 0..WARMUP_ITERATIONS {
            let index = i % array.len();
            unsafe {
                barrier.write_barrier_array_fast(
                    array.as_mut_ptr() as *mut u8,
                    index,
                    element_size,
                    new_value,
                );
            }
        }

        // Benchmark array barrier
        let start = Instant::now();
        for i in 0..BENCHMARK_ITERATIONS {
            let index = i % array.len();
            unsafe {
                barrier.write_barrier_array_fast(
                    array.as_mut_ptr() as *mut u8,
                    index,
                    element_size,
                    new_value,
                );
            }
        }
        let array_duration = start.elapsed();

        // Compare with regular barrier
        let start = Instant::now();
        for i in 0..BENCHMARK_ITERATIONS {
            let index = i % array.len();
            let slot_ptr = &mut array[index] as *mut ObjectReference;
            unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        }
        let regular_duration = start.elapsed();

        println!("Array barrier performance:");
        println!(
            "  Array-specific: {:?} ({:.2} ns/op)",
            array_duration,
            array_duration.as_nanos() as f64 / BENCHMARK_ITERATIONS as f64
        );
        println!(
            "  Regular:        {:?} ({:.2} ns/op)",
            regular_duration,
            regular_duration.as_nanos() as f64 / BENCHMARK_ITERATIONS as f64
        );

        // Array barrier should be competitive with regular barrier
        assert!(array_duration <= regular_duration * 140 / 100); // Allow 40% margin
    }

    /// Test that fast path optimizations don't break correctness
    #[test]
    fn test_fast_path_correctness() {
        let (barrier, objects) = setup_write_barrier();

        // Test inactive barrier correctness
        let mut slot = objects[0];
        let slot_ptr = &mut slot as *mut ObjectReference;
        let new_value = objects[1];

        unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        assert_eq!(slot, new_value);

        // Test active barrier correctness
        barrier.activate();
        barrier
            .tricolor_marking
            .set_color(objects[0], ObjectColor::White);

        let mut slot = objects[0];
        let slot_ptr = &mut slot as *mut ObjectReference;
        let new_value = objects[2];

        unsafe { barrier.write_barrier_fast(slot_ptr, new_value) };
        assert_eq!(slot, new_value);

        // The old value should have been shaded to grey
        assert_eq!(
            barrier.tricolor_marking.get_color(objects[0]),
            ObjectColor::Grey
        );
    }

    /// Stress test with many concurrent write barrier operations
    #[test]
    fn stress_test_write_barriers() {
        let (barrier, objects) = setup_write_barrier();

        // Large number of operations to stress-test the implementation
        const STRESS_ITERATIONS: usize = 10_000_000;
        let mut slots: Vec<ObjectReference> = vec![objects[0]; 1000];

        let start = Instant::now();
        for i in 0..STRESS_ITERATIONS {
            let slot_index = i % slots.len();
            let obj_index = (i / 1000) % objects.len();
            let slot_ptr = &mut slots[slot_index] as *mut ObjectReference;
            unsafe { barrier.write_barrier_fast(slot_ptr, objects[obj_index]) };
        }
        let stress_duration = start.elapsed();

        println!(
            "Stress test ({} operations): {:?} ({:.2} ns/op)",
            STRESS_ITERATIONS,
            stress_duration,
            stress_duration.as_nanos() as f64 / STRESS_ITERATIONS as f64
        );

        // Should complete in reasonable time
        assert!(stress_duration.as_secs() < 10);
    }
}
