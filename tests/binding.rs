//! Integration tests for the binding module utilities.

use fugrip::binding::{
    FUGC_PLAN_MANAGER, fugc_alloc_info, fugc_gc, fugc_get_cycle_stats, fugc_get_phase,
    fugc_get_stats, fugc_is_collecting, take_enqueued_references,
};
use fugrip::fugc_coordinator::FugcPhase;
use fugrip::plan::FugcPlanManager;
use mmtk::util::Address;

#[test]
fn fugc_alloc_info_respects_alignment() {
    let (size, align) = fugc_alloc_info(65, 16);
    assert_eq!(align, 16);
    assert_eq!(size % align, 0);
    assert!(size >= 65);
}

#[test]
fn fugc_stats_and_phase_accessors_provide_defaults() {
    let stats = fugc_get_stats();
    let phase = fugc_get_phase();

    assert_eq!(phase, FugcPhase::Idle);
    assert!(stats.concurrent_collection_enabled);
    assert!(!fugc_is_collecting());

    // Ensure we can toggle concurrent collection and observe the change.
    {
        let manager = FUGC_PLAN_MANAGER.lock();
        manager.set_concurrent_collection(false);
    }
    assert!(!fugc_get_stats().concurrent_collection_enabled);

    {
        let manager = FUGC_PLAN_MANAGER.lock();
        manager.set_concurrent_collection(true);
    }
}

#[test]
fn take_enqueued_references_drains_queue_even_when_empty() {
    assert!(take_enqueued_references().is_empty());
}

#[test]
fn triggering_gc_without_mmtk_is_a_noop() {
    // Ensure the global manager is in a clean state to start.
    {
        let mut manager = FUGC_PLAN_MANAGER.lock();
        *manager = FugcPlanManager::new();
    }

    fugc_gc();
    let stats = fugc_get_cycle_stats();
    assert_eq!(stats.cycles_completed, 0);
}

#[test]
fn plan_manager_alloc_info_matches_binding() {
    let manager = FUGC_PLAN_MANAGER.lock();
    let expected = manager.alloc_info(128, 32);
    drop(manager);

    let from_binding = fugc_alloc_info(128, 32);
    assert_eq!(expected, from_binding);
}

#[test]
fn take_enqueued_references_handles_mock_entries() {
    // Simulate a finalization entry by pushing directly through the plan manager.
    let fake_addr = unsafe { Address::from_usize(0x2000_0000) };
    let fake_obj = mmtk::util::ObjectReference::from_raw_address(fake_addr).unwrap();

    {
        let manager = FUGC_PLAN_MANAGER.lock();
        manager.post_alloc(fake_obj, 0);
    }

    // No entries are produced by post_alloc in the stub, but the call exercises the path.
    assert!(take_enqueued_references().is_empty());
}
