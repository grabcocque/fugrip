//! Validation tests for benchmark infrastructure
//!
//! Tests the microbenchmark helpers and density generators to ensure
//! they produce intended patterns for reliable benchmarking.

use fugrip::simd_sweep::SimdBitvector;
use mmtk::util::Address;

#[test]
fn test_dense_pattern_generation() {
    // Test that we can generate high-density patterns reliably
    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

    // Generate dense pattern (80% density)
    let objects_per_chunk = 4096; // 64KB / 16 bytes
    let target_marked = (objects_per_chunk as f64 * 0.8) as usize;

    for i in 0..target_marked {
        let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        bitvector.mark_live(obj);
    }

    // Verify density
    let marked_count = bitvector.get_chunk_population(0);
    let density = marked_count as f64 / objects_per_chunk as f64;

    assert!(
        density >= 0.75,
        "Dense pattern should have >75% density, got {:.2}%",
        density * 100.0
    );
    assert_eq!(
        marked_count, target_marked,
        "Should mark exact target count"
    );
}

#[test]
fn test_sparse_pattern_generation() {
    // Test that we can generate low-density patterns reliably
    let heap_base = unsafe { Address::from_usize(0x20000000) };
    let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

    // Generate sparse pattern (5% density)
    let objects_per_chunk = 4096;
    let target_marked = (objects_per_chunk as f64 * 0.05) as usize;

    // Mark objects spaced far apart
    for i in 0..target_marked {
        let spacing = objects_per_chunk / target_marked;
        let offset = i * spacing * 16;
        let obj = unsafe { Address::from_usize(heap_base.as_usize() + offset) };
        bitvector.mark_live(obj);
    }

    // Verify sparsity
    let marked_count = bitvector.get_chunk_population(0);
    let density = marked_count as f64 / objects_per_chunk as f64;

    assert!(
        density <= 0.10,
        "Sparse pattern should have <10% density, got {:.2}%",
        density * 100.0
    );
    assert_eq!(
        marked_count, target_marked,
        "Should mark exact target count"
    );
}

#[test]
fn test_mixed_density_pattern() {
    // Test generating patterns with different densities per chunk
    let heap_base = unsafe { Address::from_usize(0x30000000) };
    let bitvector = SimdBitvector::new(heap_base, 256 * 1024, 16); // 4 chunks

    let objects_per_chunk = 4096;

    // Chunk 0: Very dense (90%)
    let dense_count = (objects_per_chunk as f64 * 0.9) as usize;
    for i in 0..dense_count {
        let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        bitvector.mark_live(obj);
    }

    // Chunk 1: Medium density (50%)
    let medium_count = objects_per_chunk / 2;
    let chunk1_base = heap_base.as_usize() + 64 * 1024;
    for i in 0..medium_count {
        let obj = unsafe { Address::from_usize(chunk1_base + i * 32) }; // Every other object
        bitvector.mark_live(obj);
    }

    // Chunk 2: Sparse (5%)
    let sparse_count = objects_per_chunk / 20;
    let chunk2_base = heap_base.as_usize() + 128 * 1024;
    for i in 0..sparse_count {
        let obj = unsafe { Address::from_usize(chunk2_base + i * 320) }; // Widely spaced
        bitvector.mark_live(obj);
    }

    // Chunk 3: Empty (0%)
    // Don't mark anything

    // Verify each chunk has expected density
    let chunk0_density = bitvector.get_chunk_population(0) as f64 / objects_per_chunk as f64;
    let chunk1_density = bitvector.get_chunk_population(1) as f64 / objects_per_chunk as f64;
    let chunk2_density = bitvector.get_chunk_population(2) as f64 / objects_per_chunk as f64;
    let chunk3_density = bitvector.get_chunk_population(3) as f64 / objects_per_chunk as f64;

    assert!(
        chunk0_density >= 0.85,
        "Chunk 0 should be dense: {:.2}%",
        chunk0_density * 100.0
    );
    assert!(
        (0.40..=0.60).contains(&chunk1_density),
        "Chunk 1 should be medium: {:.2}%",
        chunk1_density * 100.0
    );
    assert!(
        chunk2_density <= 0.10,
        "Chunk 2 should be sparse: {:.2}%",
        chunk2_density * 100.0
    );
    assert_eq!(chunk3_density, 0.0, "Chunk 3 should be empty");
}

#[test]
fn test_hybrid_strategy_selection() {
    // Test that hybrid sweep correctly chooses strategies based on density
    let heap_base = unsafe { Address::from_usize(0x40000000) };
    let bitvector = SimdBitvector::new(heap_base, 128 * 1024, 16); // 2 chunks

    // Make first chunk dense (85% density)
    let objects_per_chunk = 4096;
    let dense_count = (objects_per_chunk as f64 * 0.85) as usize;
    for i in 0..dense_count {
        let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        bitvector.mark_live(obj);
    }

    // Make second chunk sparse (3% density)
    let sparse_count = objects_per_chunk / 33;
    let chunk1_base = heap_base.as_usize() + 64 * 1024;
    for i in 0..sparse_count {
        let obj = unsafe { Address::from_usize(chunk1_base + i * 512) };
        bitvector.mark_live(obj);
    }

    // Perform hybrid sweep
    let stats = bitvector.hybrid_sweep();

    // Verify both strategies were used
    assert!(
        stats.simd_chunks_processed > 0,
        "Should use SIMD for dense chunk"
    );
    assert!(
        stats.sparse_chunks_processed > 0,
        "Should use sparse for sparse chunk"
    );

    // Calculate expected dead objects: total capacity minus marked objects
    let total_capacity = 2 * objects_per_chunk; // 2 chunks
    let total_marked = dense_count + sparse_count;
    let expected_dead = total_capacity - total_marked;
    assert_eq!(
        stats.objects_swept, expected_dead,
        "Should sweep all unmarked (dead) objects"
    );

    // Verify strategy selection was optimal
    assert_eq!(
        stats.simd_chunks_processed, 1,
        "Should process exactly 1 chunk with SIMD"
    );
    assert_eq!(
        stats.sparse_chunks_processed, 1,
        "Should process exactly 1 chunk with sparse"
    );
}

#[test]
fn test_benchmark_pattern_consistency() {
    // Test that patterns are reproducible for benchmarking
    let heap_base = unsafe { Address::from_usize(0x50000000) };

    // Generate same pattern twice
    let bitvector1 = SimdBitvector::new(heap_base, 64 * 1024, 16);
    let bitvector2 = SimdBitvector::new(heap_base, 64 * 1024, 16);

    // Mark same objects in both
    for i in (0..1000).step_by(3) {
        let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        bitvector1.mark_live(obj);
        bitvector2.mark_live(obj);
    }

    // Both should have identical stats
    let stats1 = bitvector1.get_stats();
    let stats2 = bitvector2.get_stats();

    assert_eq!(stats1.objects_marked, stats2.objects_marked);
    assert_eq!(
        bitvector1.get_chunk_population(0),
        bitvector2.get_chunk_population(0)
    );

    // Both should have identical sweep performance
    let sweep1 = bitvector1.hybrid_sweep();
    let sweep2 = bitvector2.hybrid_sweep();

    assert_eq!(sweep1.objects_swept, sweep2.objects_swept);
    assert_eq!(sweep1.simd_chunks_processed, sweep2.simd_chunks_processed);
    assert_eq!(
        sweep1.sparse_chunks_processed,
        sweep2.sparse_chunks_processed
    );
}

#[test]
fn test_benchmark_scaling_validation() {
    // Test that benchmark patterns scale correctly with heap size
    let heap_base = unsafe { Address::from_usize(0x60000000) };

    // Test different heap sizes
    let sizes = [64 * 1024, 128 * 1024, 256 * 1024];
    let target_density = 0.3; // 30% density

    for &size in &sizes {
        let bitvector = SimdBitvector::new(heap_base, size, 16);
        let max_objects = size / 16;
        let target_marked = (max_objects as f64 * target_density) as usize;

        // Mark objects to achieve target density
        for i in 0..target_marked {
            let spacing = max_objects / target_marked;
            let offset = (i * spacing).min(max_objects - 1) * 16;
            let obj = unsafe { Address::from_usize(heap_base.as_usize() + offset) };
            bitvector.mark_live(obj);
        }

        let stats = bitvector.get_stats();
        let actual_density = stats.objects_marked as f64 / max_objects as f64;

        assert!(
            (actual_density - target_density).abs() < 0.05,
            "Density should be close to target for size {}: got {:.2}%, expected {:.2}%",
            size,
            actual_density * 100.0,
            target_density * 100.0
        );
    }
}

#[test]
#[cfg(feature = "performance-tests")]
fn test_performance_measurement_validity() {
    // Test that performance measurements are reasonable and consistent
    let heap_base = unsafe { Address::from_usize(0x70000000) };
    let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

    // Mark a known number of objects
    let marked_count = 1000;
    for i in 0..marked_count {
        let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        bitvector.mark_live(obj);
    }

    // Perform multiple sweeps and verify consistency
    let mut sweep_times = Vec::new();
    let mut objects_swept = Vec::new();

    for _i in 0..5 {
        // Create a fresh bitvector for each sweep to ensure consistent test conditions
        let fresh_bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

        // Mark the same pattern of objects
        for j in 0..marked_count {
            let obj = unsafe { Address::from_usize(heap_base.as_usize() + j * 16) };
            fresh_bitvector.mark_live(obj);
        }

        let stats = fresh_bitvector.hybrid_sweep();
        sweep_times.push(stats.sweep_time_ns);
        objects_swept.push(stats.objects_swept);
    }

    // All sweeps should process dead objects consistently
    // Since we mark the same objects each time, the number of dead objects should be consistent
    let first_sweep_count = objects_swept[0];
    assert!(
        objects_swept
            .iter()
            .all(|&count| count == first_sweep_count),
        "All sweeps should process the same number of dead objects consistently"
    );

    // Performance should be reasonable (not zero, not extremely high)
    assert!(
        sweep_times.iter().all(|&time| time > 0),
        "Sweep times should be greater than zero"
    );

    assert!(
        sweep_times.iter().all(|&time| time < 1_000_000_000), // < 1 second
        "Sweep times should be reasonable (< 1s)"
    );

    // Calculate coefficient of variation to check consistency
    let mean_time = sweep_times.iter().sum::<u64>() as f64 / sweep_times.len() as f64;
    let variance = sweep_times
        .iter()
        .map(|&time| {
            let diff = time as f64 - mean_time;
            diff * diff
        })
        .sum::<f64>()
        / sweep_times.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean_time;

    assert!(
        cv < 1.0,
        "Performance measurements should be reasonably consistent (CV < 100%), got {:.2}%",
        cv * 100.0
    );
}
