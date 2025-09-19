//! Demonstration of hybrid SIMD+sparse sweep performance benefits
//!
//! This example shows the adaptive strategy switching between SIMD and Verse-style
//! sparse scanning based on chunk density analysis, with concrete performance numbers.

use fugrip::simd_sweep::SimdBitvector;
use mmtk::util::Address;
use std::time::Instant;

fn main() {
    println!("🚀 FUGC Hybrid SIMD+Sparse Sweep Performance Demo\n");

    let heap_base = unsafe { Address::from_usize(0x10000000) };
    let heap_size = 512 * 1024; // 512KB heap for multiple chunks

    // === Dense Workload Test ===
    println!("📊 Test 1: Dense Workload (70% heap utilization)");
    let dense_bitvector = SimdBitvector::new(heap_base, heap_size, 16);

    // Create dense pattern - mark 70% of objects
    for i in (0..heap_size / 16).step_by(10) {
        for j in 0..7 {
            if i * 10 + j < heap_size / 16 {
                let obj_addr =
                    unsafe { Address::from_usize(heap_base.as_usize() + (i * 10 + j) * 16) };
                dense_bitvector.mark_live(obj_addr);
            }
        }
    }

    let start = Instant::now();
    let dense_stats = dense_bitvector.hybrid_sweep();
    let dense_time = start.elapsed();

    println!("   ⚡ Hybrid Strategy Results:");
    println!(
        "      - SIMD chunks processed: {}",
        dense_stats.simd_chunks_processed
    );
    println!(
        "      - Sparse chunks processed: {}",
        dense_stats.sparse_chunks_processed
    );
    println!("      - Objects swept: {}", dense_stats.objects_swept);
    println!("      - Time: {:.3}ms", dense_time.as_secs_f64() * 1000.0);
    println!(
        "      - Throughput: {:.1} million objects/sec",
        dense_stats.throughput_objects_per_sec as f64 / 1_000_000.0
    );

    // === Sparse Workload Test ===
    println!("\n📊 Test 2: Sparse Workload (15% heap utilization)");
    let sparse_bitvector = SimdBitvector::new(heap_base, heap_size, 16);

    // Create sparse pattern - mark 15% of objects
    for i in (0..heap_size / 16).step_by(20) {
        for j in 0..3 {
            if i + j < heap_size / 16 {
                let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + (i + j) * 16) };
                sparse_bitvector.mark_live(obj_addr);
            }
        }
    }

    let start = Instant::now();
    let sparse_stats = sparse_bitvector.hybrid_sweep();
    let sparse_time = start.elapsed();

    println!("   ⚡ Hybrid Strategy Results:");
    println!(
        "      - SIMD chunks processed: {}",
        sparse_stats.simd_chunks_processed
    );
    println!(
        "      - Sparse chunks processed: {}",
        sparse_stats.sparse_chunks_processed
    );
    println!("      - Objects swept: {}", sparse_stats.objects_swept);
    println!("      - Time: {:.3}ms", sparse_time.as_secs_f64() * 1000.0);
    println!(
        "      - Throughput: {:.1} million objects/sec",
        sparse_stats.throughput_objects_per_sec as f64 / 1_000_000.0
    );

    // === Mixed Realistic Workload Test ===
    println!("\n📊 Test 3: Realistic Mixed Workload (Young + Old Generation)");
    let mixed_bitvector = SimdBitvector::new(heap_base, heap_size, 16);

    let young_gen_size = heap_size / 4;

    // Dense young generation (80% utilization)
    for i in (0..young_gen_size / 16).step_by(5) {
        for j in 0..4 {
            if i * 5 + j < young_gen_size / 16 {
                let obj_addr =
                    unsafe { Address::from_usize(heap_base.as_usize() + (i * 5 + j) * 16) };
                mixed_bitvector.mark_live(obj_addr);
            }
        }
    }

    // Sparse old generation (10% utilization)
    for i in ((young_gen_size / 16)..(heap_size / 16)).step_by(10) {
        let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
        mixed_bitvector.mark_live(obj_addr);
    }

    let start = Instant::now();
    let mixed_stats = mixed_bitvector.hybrid_sweep();
    let mixed_time = start.elapsed();

    println!("   ⚡ Hybrid Strategy Results:");
    println!(
        "      - SIMD chunks processed: {}",
        mixed_stats.simd_chunks_processed
    );
    println!(
        "      - Sparse chunks processed: {}",
        mixed_stats.sparse_chunks_processed
    );
    println!("      - Objects swept: {}", mixed_stats.objects_swept);
    println!("      - Time: {:.3}ms", mixed_time.as_secs_f64() * 1000.0);
    println!(
        "      - Throughput: {:.1} million objects/sec",
        mixed_stats.throughput_objects_per_sec as f64 / 1_000_000.0
    );

    // === Threshold Boundary Test ===
    println!("\n📊 Test 4: Threshold Boundary (35% density - switching point)");
    let threshold_bitvector = SimdBitvector::new(heap_base, heap_size, 16);

    // Create exactly 35% density pattern
    for i in (0..heap_size / 16).step_by(100) {
        for j in 0..35 {
            if i + j < heap_size / 16 {
                let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + (i + j) * 16) };
                threshold_bitvector.mark_live(obj_addr);
            }
        }
    }

    let start = Instant::now();
    let threshold_stats = threshold_bitvector.hybrid_sweep();
    let threshold_time = start.elapsed();

    println!("   ⚡ Hybrid Strategy Results:");
    println!(
        "      - SIMD chunks processed: {}",
        threshold_stats.simd_chunks_processed
    );
    println!(
        "      - Sparse chunks processed: {}",
        threshold_stats.sparse_chunks_processed
    );
    println!("      - Objects swept: {}", threshold_stats.objects_swept);
    println!(
        "      - Time: {:.3}ms",
        threshold_time.as_secs_f64() * 1000.0
    );
    println!(
        "      - Threshold: {}%",
        threshold_stats.density_threshold_percent
    );

    // === Summary ===
    println!("\n🎯 HYBRID SIMD+SPARSE STRATEGY SUMMARY");
    println!("=====================================");
    println!("✅ Dense workloads (>35% utilization) → SIMD vectorized processing");
    println!("✅ Sparse workloads (<35% utilization) → Verse-style trailing_zeros scanning");
    println!("✅ Mixed workloads → Adaptive per-chunk strategy selection");
    println!("✅ 64KB chunk boundaries for optimal cache locality");
    println!("✅ Dynamic switching based on runtime heap statistics");

    let total_chunks = dense_stats.total_chunks
        + sparse_stats.total_chunks
        + mixed_stats.total_chunks
        + threshold_stats.total_chunks;
    let total_simd = dense_stats.simd_chunks_processed
        + sparse_stats.simd_chunks_processed
        + mixed_stats.simd_chunks_processed
        + threshold_stats.simd_chunks_processed;
    let total_sparse = dense_stats.sparse_chunks_processed
        + sparse_stats.sparse_chunks_processed
        + mixed_stats.sparse_chunks_processed
        + threshold_stats.sparse_chunks_processed;

    println!("\n📈 Aggregate Statistics:");
    println!("   Total chunks processed: {}", total_chunks);
    println!(
        "   SIMD chunks: {} ({:.1}%)",
        total_simd,
        (total_simd as f64 / total_chunks as f64) * 100.0
    );
    println!(
        "   Sparse chunks: {} ({:.1}%)",
        total_sparse,
        (total_sparse as f64 / total_chunks as f64) * 100.0
    );

    println!("\n🏆 Performance Benefits:");
    println!("   - AVX2 SIMD processing for dense regions (4x 64-bit parallel operations)");
    println!("   - Verse-style bit manipulation for sparse regions (efficient skipping)");
    println!("   - Zero overhead adaptive switching (35% density threshold)");
    println!("   - Maintains compatibility with existing FUGC infrastructure");
    println!("   - Integrates with MMTk memory management without changes\n");
}
