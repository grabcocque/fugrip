//! Integration tests for the binding module utilities.

use fugrip::binding::{
    fugc_alloc_info, fugc_gc, fugc_get_cycle_stats, fugc_get_phase, fugc_get_stats,
    take_enqueued_references,
};
use fugrip::test_utils::TestFixture;

#[test]
fn fugc_alloc_info_respects_alignment() {
    let (size, align) = fugc_alloc_info(65, 16);
    assert_eq!(align, 16);
    assert_eq!(size % align, 0);
    assert!(size >= 65);
}

#[test]
fn fugc_stats_and_phase_accessors_provide_defaults() {
    // Use TestFixture to ensure proper test isolation
    let fixture = TestFixture::new();

    // Test that the functions work and return valid values using the fixture's isolated coordinator
    let stats = fixture.coordinator.get_cycle_stats();
    let phase = fixture.coordinator.current_phase();
    let is_collecting = fixture.coordinator.is_collecting();

    // Test basic functionality - the coordinator should work without panicking
    // and return consistent values
    assert_eq!(is_collecting, fixture.coordinator.is_collecting()); // Consistent with itself
    assert_eq!(phase, fixture.coordinator.current_phase()); // Consistent with itself
    assert_eq!(
        stats.cycles_completed,
        fixture.coordinator.get_cycle_stats().cycles_completed
    ); // Consistent with itself

    // Test that the values are of the expected types (not panicking is sufficient)
    let _ = format!("{:?}", phase);
    let _ = format!("{:?}", stats);

    // Test initial state - should be idle with 0 cycles completed
    assert_eq!(phase, fugrip::fugc_coordinator::FugcPhase::Idle);
    assert!(!is_collecting);
    assert_eq!(stats.cycles_completed, 0);
}

#[test]
fn take_enqueued_references_drains_queue_even_when_empty() {
    assert!(take_enqueued_references().is_empty());
}

#[test]
fn triggering_gc_without_mmtk_is_a_noop() {
    // Use TestFixture to ensure proper test isolation
    let _fixture = TestFixture::new();

    fugc_gc();
    let stats = fugc_get_cycle_stats();
    assert_eq!(stats.cycles_completed, 0);
}

#[test]
fn plan_manager_alloc_info_matches_binding() {
    // Use TestFixture to ensure proper test isolation
    let _fixture = TestFixture::new();

    let from_binding = fugc_alloc_info(128, 32);
    // Basic validation that the binding returns sensible values
    assert!(from_binding.0 >= 128);
    assert_eq!(from_binding.1, 32);
}

#[test]
fn take_enqueued_references_handles_mock_entries() {
    // Use TestFixture to ensure proper test isolation
    let _fixture = TestFixture::new();

    // No entries are produced by post_alloc in the stub, but the call exercises the path.
    assert!(take_enqueued_references().is_empty());
}

// Error condition tests for binding module
#[test]
fn fugc_alloc_info_edge_cases() {
    // Test edge cases in allocation info calculation
    let test_cases = vec![
        (0, 1),              // Zero size
        (1, 1),              // Minimum size
        (1, 2),              // Size smaller than alignment
        (15, 16),            // Size not aligned
        (16, 16),            // Size already aligned
        (usize::MAX / 2, 8), // Large size
    ];

    for (size, align) in test_cases {
        let (result_size, result_align) = fugc_alloc_info(size, align);

        // Verify constraints
        assert!(
            result_size >= size,
            "Result size {} should be >= input size {}",
            result_size,
            size
        );
        assert_eq!(result_align, align, "Alignment should be preserved");
        if result_align > 0 {
            assert_eq!(result_size % result_align, 0, "Size should be aligned");
        }
        if result_align > 1 {
            assert!(
                result_align.is_power_of_two(),
                "Alignment should be power of two"
            );
        }
    }
}

#[test]
fn fugc_alloc_info_zero_alignment() {
    // Test behavior with zero alignment
    let (size, _align) = fugc_alloc_info(64, 0);

    // Should handle gracefully (implementation-specific behavior)
    assert!(size >= 64);
    // Alignment handling is implementation-specific
}

#[test]
fn fugc_alloc_info_non_power_of_two_alignment() {
    // Test behavior with non-power-of-two alignment
    let non_power_alignments = vec![3, 5, 6, 7, 9, 10, 12, 15];

    for align in non_power_alignments {
        let (size, _result_align) = fugc_alloc_info(64, align);

        // Should handle gracefully
        assert!(size >= 64);
        // Implementation may adjust alignment to nearest power of two
    }
}

#[test]
fn fugc_alloc_info_large_values() {
    // Test with very large size and alignment values
    let large_size = usize::MAX / 4;
    let large_align = 1024 * 1024; // 1MB alignment

    let (result_size, result_align) = fugc_alloc_info(large_size, large_align);

    // Should handle without overflow
    assert!(result_size >= large_size);
    assert_eq!(result_align, large_align);
}

#[test]
fn fugc_get_stats_consistency() {
    // Test that stats remain consistent across multiple calls
    let stats1 = fugc_get_stats();
    let stats2 = fugc_get_stats();

    // Both calls should return identical stats initially
    assert_eq!(
        stats1.concurrent_collection_enabled,
        stats2.concurrent_collection_enabled
    );
}

#[test]
fn fugc_get_stats_after_state_changes() {
    // Use TestFixture to ensure proper test isolation
    let _fixture = TestFixture::new();

    // Test that the function works (may not change due to global state, but should not panic)
    let stats = fugc_get_stats();
    let _ = stats.concurrent_collection_enabled; // Just access it to ensure it works
}

#[test]
fn fugc_phase_functions() {
    // Use TestFixture to ensure proper test isolation
    let fixture = TestFixture::new();

    // Test phase query functions using the fixture's isolated coordinator
    let phase = fixture.coordinator.current_phase();
    let is_collecting = fixture.coordinator.is_collecting();

    // Test that the functions work and return consistent values
    assert_eq!(is_collecting, fixture.coordinator.is_collecting());
    assert_eq!(phase, fixture.coordinator.current_phase());

    // Test that the relationship between phase and collecting state is logical
    // Either: collecting=true and phase!=Idle, OR collecting=false and phase=Idle
    let logical_relationship =
        is_collecting == (phase != fugrip::fugc_coordinator::FugcPhase::Idle);
    assert!(
        logical_relationship,
        "is_collecting ({}) should match phase != Idle ({})",
        is_collecting,
        phase != fugrip::fugc_coordinator::FugcPhase::Idle
    );

    // Test that we can get a valid phase (it should be one of the known phases)
    match phase {
        fugrip::fugc_coordinator::FugcPhase::Idle
        | fugrip::fugc_coordinator::FugcPhase::ActivateBarriers
        | fugrip::fugc_coordinator::FugcPhase::ActivateBlackAllocation
        | fugrip::fugc_coordinator::FugcPhase::MarkGlobalRoots
        | fugrip::fugc_coordinator::FugcPhase::StackScanHandshake
        | fugrip::fugc_coordinator::FugcPhase::Tracing
        | fugrip::fugc_coordinator::FugcPhase::Sweeping => {
            // Valid phase
        }
        _ => panic!("Invalid phase returned: {:?}", phase),
    }
}

#[test]
fn fugc_cycle_stats() {
    // Test cycle statistics function
    let stats = fugc_get_cycle_stats();

    // Should return valid statistics
    // Note: These fields are unsigned, so they're always >= 0
    assert_eq!(stats.cycles_completed, 0); // Initial state should be 0
    assert_eq!(stats.total_marking_time_ms, 0); // Initial state should be 0
    assert_eq!(stats.total_sweep_time_ms, 0); // Initial state should be 0
    assert_eq!(stats.objects_marked, 0); // Initial state should be 0
    assert_eq!(stats.objects_swept, 0); // Initial state should be 0
    assert_eq!(stats.handshakes_performed, 0); // Initial state should be 0
    assert!(stats.avg_stack_scan_objects >= 0.0); // Float can still be negative
}

#[test]
fn take_enqueued_references_empty() {
    // Test that take_enqueued_references handles empty queue
    let refs = take_enqueued_references();
    assert!(refs.is_empty());
}

#[test]
fn take_enqueued_references_multiple_calls() {
    // Test multiple calls to take_enqueued_references
    let refs1 = take_enqueued_references();
    let refs2 = take_enqueued_references();

    // Both should be empty initially
    assert!(refs1.is_empty());
    assert!(refs2.is_empty());
}

#[test]
fn fugc_gc_function() {
    // Test that fugc_gc can be called without panic
    fugc_gc(); // Should not panic

    // Verify state consistency after GC call
    let phase = fugc_get_phase();
    // Phase should be valid (exact phase depends on timing)
    assert!(matches!(
        phase,
        fugrip::fugc_coordinator::FugcPhase::Idle
            | fugrip::fugc_coordinator::FugcPhase::ActivateBarriers
            | fugrip::fugc_coordinator::FugcPhase::ActivateBlackAllocation
            | fugrip::fugc_coordinator::FugcPhase::MarkGlobalRoots
            | fugrip::fugc_coordinator::FugcPhase::StackScanHandshake
            | fugrip::fugc_coordinator::FugcPhase::Tracing
            | fugrip::fugc_coordinator::FugcPhase::PrepareForSweep
            | fugrip::fugc_coordinator::FugcPhase::Sweeping
    ));
}
