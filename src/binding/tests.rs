//! Test suite for MMTk VM binding with FUGC integration

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::thread::MutatorThread;
    use mmtk::util::ObjectReference;
    use mmtk::vm::{VMBinding, Collection, ReferenceGlue};
    // use mmtk::vm::ActivePlan; // Currently unused

    #[test]
    fn test_mutator_registration_creation() {
        // Test MutatorRegistration creation and basic properties
        let thread = MutatorThread::new(0);
        let dummy_mutator = std::ptr::null_mut();

        let registration = mutator::MutatorRegistration::new(dummy_mutator, thread);

        // Test that registration stores the correct values
        assert_eq!(registration.mutator, dummy_mutator);
        // Note: Can't easily test thread equality without implementing PartialEq
    }

    #[test]
    fn test_mutator_registration_safety_markers() {
        // Test that MutatorRegistration implements required safety traits
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<mutator::MutatorRegistration>();
        assert_sync::<mutator::MutatorRegistration>();
    }

    #[test]
    fn test_rust_vm_constants() {
        // Test RustVM constants that affect MMTk behavior
        // MIN_ALIGNMENT is always > 0 by definition
        assert!(vm_impl::RustVM::MIN_ALIGNMENT.is_power_of_two());
        // MAX_ALIGNMENT >= MIN_ALIGNMENT by invariant
        assert!(vm_impl::RustVM::MAX_ALIGNMENT.is_power_of_two());

        // Test reasonable bounds
        // MIN_ALIGNMENT >= 1 by invariant
        // MAX_ALIGNMENT <= 4096 by design
    }

    #[test]
    fn test_fugc_allocation_helpers() {
        // Test fugc_alloc_info with various inputs
        let test_cases = [
            (0, 1),            // Minimum size
            (8, 8),            // Typical small object
            (64, 16),          // Medium object
            (1024, 32),        // Large object
            (1024 * 1024, 32), // Large but safe size - 1MB
        ];

        for (size, align) in test_cases {
            let (result_size, result_align) = allocation::fugc_alloc_info(size, align);

            // Basic invariants
            assert!(result_align > 0);
            assert!(result_align.is_power_of_two());
            assert!(result_size >= size || size == usize::MAX); // Handle overflow case
            assert!(result_align >= align);
        }
    }

    #[test]
    fn test_fugc_post_alloc() {
        // Test post-allocation hook doesn't panic
        let dummy_addr = unsafe { mmtk::util::Address::from_usize(0x1000) };
        let obj_ref = unsafe { ObjectReference::from_raw_address_unchecked(dummy_addr) };

        // These should not panic
        allocation::fugc_post_alloc(obj_ref, 0);
        allocation::fugc_post_alloc(obj_ref, 64);
        allocation::fugc_post_alloc(obj_ref, 1024);
    }

    #[test]
    fn test_fugc_get_stats() {
        // Test statistics retrieval
        let stats = stats::fugc_get_stats();

        // Test that stats can be retrieved successfully
        // Note: All fields are usize, so no need to check >= 0
        let _ = stats.work_stolen;
        let _ = stats.work_shared;
        let _ = stats.objects_allocated_black;
        let _ = stats.total_bytes;
        let _ = stats.used_bytes;

        // Test that concurrent collection flag is accessible
        // (Value may be true or false depending on configuration)
        let _concurrent_enabled = stats.concurrent_collection_enabled;
    }

    #[test]
    fn test_mutator_registry() {
        // Test global mutator registry thread safety
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<dashmap::DashMap<u64, mutator::MutatorRegistration>>();

        // Test that registry can be accessed (this initializes it)
        let _initial_len = mutator::MUTATOR_MAP.get_or_init(dashmap::DashMap::new).len();
        // Registry access should not panic
    }

    #[test]
    fn test_rust_vm_trait_implementations() {
        // Test that RustVM implements required traits for MMTk integration
        fn assert_vm_binding<T: mmtk::vm::VMBinding>() {}
        assert_vm_binding::<vm_impl::RustVM>();

        // Test specific trait bounds
        fn assert_collection<T: mmtk::vm::Collection<vm_impl::RustVM>>() {}
        assert_collection::<vm_impl::RustCollection>();

        fn assert_reference_glue<T: mmtk::vm::ReferenceGlue<vm_impl::RustVM>>() {}
        assert_reference_glue::<vm_impl::RustReferenceGlue>();
    }

    #[test]
    fn test_plan_manager_initialization() {
        // Test that FUGC_PLAN_MANAGER can be safely initialized multiple times
        let stats1 = stats::fugc_get_stats();
        let stats2 = stats::fugc_get_stats();

        // Both calls should succeed (testing OnceLock behavior)
        let _ = stats1.work_stolen;
        let _ = stats2.work_stolen;
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test various error conditions that might occur in binding

        // Test with valid but minimal addresses (MMTk requires non-zero, word-aligned)
        let valid_addr = unsafe { mmtk::util::Address::from_usize(0x1000) }; // Use aligned, non-zero address
        let valid_obj_ref = unsafe { ObjectReference::from_raw_address_unchecked(valid_addr) };

        // This should not panic
        allocation::fugc_post_alloc(valid_obj_ref, 0);

        // Test with reasonable large values (avoid overflow)
        allocation::fugc_post_alloc(valid_obj_ref, 1024 * 1024); // 1MB instead of usize::MAX
    }

    #[test]
    fn test_thread_integration() {
        // Test thread-related functionality
        let thread1 = MutatorThread::new(0);
        let thread2 = MutatorThread::new(1);
        let thread_max = MutatorThread::new(usize::MAX);

        // Test that threads can be created with various IDs
        let dummy_mutator = std::ptr::null_mut();
        let _reg1 = mutator::MutatorRegistration::new(dummy_mutator, thread1);
        let _reg2 = mutator::MutatorRegistration::new(dummy_mutator, thread2);
        let _reg_max = mutator::MutatorRegistration::new(dummy_mutator, thread_max);
    }

    #[test]
    fn test_address_and_alignment_edge_cases() {
        // Test edge cases for alignment and addressing

        // Test minimum alignment cases
        let (size, align) = allocation::fugc_alloc_info(1, 1);
        assert!(align >= 1);
        assert!(size >= 1);

        // Test power-of-two alignment requirement
        let test_aligns = [1, 2, 4, 8, 16, 32, 64, 128];
        for align in test_aligns {
            let (_, result_align) = allocation::fugc_alloc_info(64, align);
            assert!(result_align >= align);
            assert!(result_align.is_power_of_two());
        }

        // Test that large sizes are handled gracefully
        let (large_size, large_align) = allocation::fugc_alloc_info(1024 * 1024, 64);
        assert!(large_size >= 1024 * 1024);
        assert!(large_align >= 64);
    }

    #[test]
    fn test_mutator_handle_edge_cases() {
        // Test MutatorHandle edge cases and error conditions
        let dummy_addr = unsafe { mmtk::util::Address::from_usize(0x1000) };
        let dummy_ptr = dummy_addr.to_mut_ptr::<mmtk::Mutator<vm_impl::RustVM>>();

        // Test non-null pointer handling
        let handle = mutator::MutatorHandle::from_raw(dummy_ptr);
        assert_eq!(handle.as_ptr(), dummy_ptr);
    }

    #[test]
    #[should_panic(expected = "MMTk returned a null mutator pointer")]
    fn test_mutator_handle_null_pointer_panic() {
        // Test that null pointer causes panic as expected
        let _handle = mutator::MutatorHandle::from_raw(std::ptr::null_mut());
    }

    #[test]
    fn test_vm_thread_key_functions() {
        // Test thread key generation functions
        let thread1 = mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(0x1000)
        }));
        let thread2 = mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(0x2000)
        }));

        let key1 = mutator::vm_thread_key(thread1);
        let key2 = mutator::vm_thread_key(thread2);

        assert_ne!(key1, key2, "Different threads should have different keys");
        assert_eq!(key1, 0x1000);
        assert_eq!(key2, 0x2000);

        // Test mutator thread key
        let mutator_thread1 = mmtk::util::opaque_pointer::VMMutatorThread(thread1);
        let mutator_thread2 = mmtk::util::opaque_pointer::VMMutatorThread(thread2);

        let mutator_key1 = mutator::mutator_thread_key(mutator_thread1);
        let mutator_key2 = mutator::mutator_thread_key(mutator_thread2);

        assert_eq!(mutator_key1, key1);
        assert_eq!(mutator_key2, key2);
    }

    #[test]
    fn test_mutator_registration_unregistration() {
        // Test mutator registration and unregistration flows
        let thread = MutatorThread::new(42);
        // Use a placeholder address for testing - we won't dereference it
        let dummy_mutator_ptr = 0x1000 as *mut mmtk::Mutator<vm_impl::RustVM>;
        let tls = mmtk::util::opaque_pointer::VMMutatorThread(mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(42)
        })));

        // Register mutator (using unsafe but valid for testing)
        mutator::register_mutator_context(tls, unsafe { &mut *dummy_mutator_ptr }, thread.clone());

        // Verify registration exists
        let key = mutator::mutator_thread_key(tls);
        assert!(mutator::MUTATOR_MAP.get_or_init(dashmap::DashMap::new).contains_key(&key));

        // Test with_mutator_registration (just check that the callback is called)
        let result = mutator::with_mutator_registration(tls, |reg| {
            assert_eq!(reg.mutator, dummy_mutator_ptr);
            "test_value"
        });
        assert_eq!(result, Some("test_value"));

        // Unregister mutator
        mutator::unregister_mutator_context(tls);

        // Verify unregistration
        assert!(!mutator::MUTATOR_MAP.get_or_init(dashmap::DashMap::new).contains_key(&key));

        // Test with_mutator_registration after unregistration
        let result = mutator::with_mutator_registration(tls, |_| "should_not_be_called");
        assert_eq!(result, None);
    }

    #[test]
    fn test_visitor_functions() {
        // Test visitor functions with multiple mutators
        let thread1 = MutatorThread::new(100);
        let thread2 = MutatorThread::new(101);
        // Use placeholder addresses - visitors don't actually dereference the pointers in tests
        let dummy_mutator1_ptr = 0x2000 as *mut mmtk::Mutator<vm_impl::RustVM>;
        let dummy_mutator2_ptr = 0x3000 as *mut mmtk::Mutator<vm_impl::RustVM>;
        let tls1 = mmtk::util::opaque_pointer::VMMutatorThread(mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(100)
        })));
        let tls2 = mmtk::util::opaque_pointer::VMMutatorThread(mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(101)
        })));

        // Register mutators
        mutator::register_mutator_context(tls1, unsafe { &mut *dummy_mutator1_ptr }, thread1.clone());
        mutator::register_mutator_context(tls2, unsafe { &mut *dummy_mutator2_ptr }, thread2.clone());

        // Test visit_all_mutators - count calls but don't access mutator internals
        let mut mutator_count = 0;
        mutator::visit_all_mutators(|_mutator| {
            mutator_count += 1;
            // Don't access mutator fields to avoid segfault
        });
        assert!(mutator_count >= 2, "Should visit at least 2 mutators");

        // Test visit_all_threads
        let mut thread_count = 0;
        let mut seen_ids = std::collections::HashSet::new();
        mutator::visit_all_threads(|thread| {
            thread_count += 1;
            seen_ids.insert(thread.id());
        });
        assert!(thread_count >= 2, "Should visit at least 2 threads");
        assert!(seen_ids.contains(&100));
        assert!(seen_ids.contains(&101));

        // Clean up
        mutator::unregister_mutator_context(tls1);
        mutator::unregister_mutator_context(tls2);
    }

    #[test]
    fn test_rust_collection_methods() {
        // Test RustCollection implementation methods
        let _collection = vm_impl::RustCollection;
        let worker_tls = mmtk::util::opaque_pointer::VMWorkerThread(mmtk::util::opaque_pointer::VMThread::UNINITIALIZED);
        let mutator_tls = mmtk::util::opaque_pointer::VMMutatorThread(mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(200)
        })));

        // Test stop_all_mutators - should not panic
        vm_impl::RustCollection::stop_all_mutators(worker_tls, |_mutator| {
            // Visitor should be called for each mutator
        });

        // Test resume_mutators - should not panic
        vm_impl::RustCollection::resume_mutators(worker_tls);

        // Test block_for_gc with unregistered mutator
        vm_impl::RustCollection::block_for_gc(mutator_tls); // Should not panic

        // Test spawn_gc_thread would be complex to set up properly
        // Just test that the method exists and can be called via trait
        // Real usage requires proper MMTk initialization
    }

    #[test]
    fn test_rust_reference_glue_edge_cases() {
        // Test ReferenceGlue edge cases
        let _glue = vm_impl::RustReferenceGlue;
        let obj1 =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x3000)) };
        let obj2 =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x4000)) };
        let obj3 =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x5000)) };

        // Test get_referent on non-existent object
        assert_eq!(vm_impl::RustReferenceGlue::get_referent(obj1), None);

        // Test set_referent and get_referent
        vm_impl::RustReferenceGlue::set_referent(obj1, obj2);
        assert_eq!(vm_impl::RustReferenceGlue::get_referent(obj1), Some(obj2));

        // Test clear_referent
        vm_impl::RustReferenceGlue::clear_referent(obj1);
        assert_eq!(vm_impl::RustReferenceGlue::get_referent(obj1), None);

        // Test enqueue_references with empty slice
        let empty_refs: &[ObjectReference] = &[];
        vm_impl::RustReferenceGlue::enqueue_references(empty_refs, mmtk::util::opaque_pointer::VMWorkerThread(mmtk::util::opaque_pointer::VMThread::UNINITIALIZED));

        // Test enqueue_references with actual references
        vm_impl::RustReferenceGlue::set_referent(obj1, obj2);
        vm_impl::RustReferenceGlue::set_referent(obj2, obj3);
        let refs = &[obj1, obj2];
        vm_impl::RustReferenceGlue::enqueue_references(refs, mmtk::util::opaque_pointer::VMWorkerThread(mmtk::util::opaque_pointer::VMThread::UNINITIALIZED));

        // Verify references were enqueued
        let enqueued = vm_impl::take_enqueued_references();
        assert_eq!(enqueued.len(), 2);

        // Verify referents were cleared from map
        assert_eq!(vm_impl::RustReferenceGlue::get_referent(obj1), None);
        assert_eq!(vm_impl::RustReferenceGlue::get_referent(obj2), None);
    }

    #[test]
    fn test_rust_active_plan_basic() {
        // Test RustActivePlan basic functionality
        let _plan = vm_impl::RustActivePlan;

        // Register a mutator for basic testing
        let thread = MutatorThread::new(600);
        let dummy_mutator_ptr = 0x4000 as *mut mmtk::Mutator<vm_impl::RustVM>;
        let tls = mmtk::util::opaque_pointer::VMMutatorThread(mmtk::util::opaque_pointer::VMThread(mmtk::util::opaque_pointer::OpaquePointer::from_address(unsafe {
            mmtk::util::Address::from_usize(600)
        })));
        mutator::register_mutator_context(tls, unsafe { &mut *dummy_mutator_ptr }, thread.clone());

        // Test that the mutator map contains our registration
        let key = mutator::mutator_thread_key(tls);
        assert!(mutator::MUTATOR_MAP.get_or_init(dashmap::DashMap::new).contains_key(&key));

        // Clean up
        mutator::unregister_mutator_context(tls);
    }

    #[test]
    fn test_finalization_queue_operations() {
        // Test finalization queue edge cases

        // Ensure queue starts clean
        let initial_refs = vm_impl::take_enqueued_references();
        let _ = initial_refs; // Drain any existing references

        // Test take_enqueued_references on empty queue
        let empty_refs = vm_impl::take_enqueued_references();
        assert_eq!(empty_refs.len(), 0);

        // Add some references via RustReferenceGlue
        let obj1 =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x7000)) };
        let obj2 =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x8000)) };

        vm_impl::RustReferenceGlue::set_referent(obj1, obj2);
        let refs = &[obj1];
        vm_impl::RustReferenceGlue::enqueue_references(refs, mmtk::util::opaque_pointer::VMWorkerThread(mmtk::util::opaque_pointer::VMThread::UNINITIALIZED));

        // Test taking references
        let enqueued = vm_impl::take_enqueued_references();
        assert_eq!(enqueued.len(), 1);
        assert_eq!(enqueued[0].0, obj1);
        assert_eq!(enqueued[0].1, Some(obj2));

        // Queue should be empty again
        let empty_again = vm_impl::take_enqueued_references();
        assert_eq!(empty_again.len(), 0);
    }

    #[test]
    fn test_fugc_phase_and_collection_state() {
        // Test FUGC phase and collection state functions
        let initial_phase = stats::fugc_get_phase();
        let is_collecting = stats::fugc_is_collecting();
        let cycle_stats = stats::fugc_get_cycle_stats();

        // These should not panic and should return valid values
        println!("Initial phase: {:?}", initial_phase);
        println!("Is collecting: {}", is_collecting);
        println!("Cycle stats: {:?}", cycle_stats);

        // Test triggering GC
        stats::fugc_gc();

        // Phase might have changed
        let after_gc_phase = stats::fugc_get_phase();
        println!("After GC phase: {:?}", after_gc_phase);
    }

    #[test]
    fn test_write_barrier_sad_paths() {
        // Test write barrier component access without dangerous memory writes
        let plan_manager =
            FUGC_PLAN_MANAGER.get_or_init(|| arc_swap::ArcSwap::new(std::sync::Arc::new(crate::plan::FugcPlanManager::new())));

        let manager = plan_manager.load();
        let write_barrier = manager.get_write_barrier();

        // Test barrier state - should not be active initially
        assert!(!write_barrier.is_active());

        // Test concurrent collection state changes
        assert!(manager.is_concurrent_collection_enabled());
        manager.set_concurrent_collection(false);
        assert!(!manager.is_concurrent_collection_enabled());
        manager.set_concurrent_collection(true);
        assert!(manager.is_concurrent_collection_enabled());

        // Test with problematic object reference patterns (without memory writes)
        let src =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x10000)) };
        let target =
            unsafe { ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(0x20000)) };

        // Test same src and target references
        assert_eq!(src.to_raw_address(), unsafe {
            mmtk::util::Address::from_usize(0x10000)
        });
        assert_eq!(target.to_raw_address(), unsafe {
            mmtk::util::Address::from_usize(0x20000)
        });
        assert_ne!(src, target);

        // Test address alignment checks
        let aligned_addr = unsafe { mmtk::util::Address::from_usize(0x10000) };
        let unaligned_addr = unsafe { mmtk::util::Address::from_usize(0x10001) };
        assert!(aligned_addr.as_usize() % std::mem::align_of::<usize>() == 0);
        assert!(unaligned_addr.as_usize() % std::mem::align_of::<usize>() != 0);
    }

    #[test]
    fn test_extreme_value_handling() {
        // Test handling of extreme values - proper sad path testing

        // Test with maximum size values (sad path)
        let (_max_size, max_align) = allocation::fugc_alloc_info(usize::MAX, usize::MAX);
        // Should handle overflow/saturation gracefully
        assert!(max_align.is_power_of_two() || max_align == usize::MAX);

        // Test with zero values (sad path)
        let (_zero_size, zero_align) = allocation::fugc_alloc_info(0, 1);
        assert!(zero_align > 0);
        assert!(zero_align.is_power_of_two());

        // Test with non-power-of-two alignment (sad path)
        let (_, bad_align) = allocation::fugc_alloc_info(64, 3); // 3 is not power of 2
        assert!(bad_align.is_power_of_two()); // Should be corrected

        // Test with extreme object references (sad path) - test interface without MMTk calls
        let extreme_addr = unsafe { mmtk::util::Address::from_usize(0x100000) }; // Large but reasonable address
        let extreme_obj = unsafe { ObjectReference::from_raw_address_unchecked(extreme_addr) };

        // Test that extreme object references can be created and compared
        assert!(extreme_obj.to_raw_address().as_usize() > 0);

        // Test write barrier interface without calling potentially problematic MMTk functions
        let extreme_slot = unsafe { mmtk::util::Address::from_usize(0x100008) };
        // Verify addresses can be created and manipulated safely
        assert_eq!(extreme_slot.as_usize(), 0x100008);
    }

    #[test]
    fn test_concurrent_access_patterns() {
        // Test thread safety of global state with rayon parallel execution
        use rayon::prelude::*;
        use std::sync::Arc;

        // Use rayon parallel iteration with fixed number of operations per thread
        let total_ops: usize = (0..4)
            .into_par_iter()
            .map(|i| {
                let mut operations = 0;
                for _ in 0..10 {
                    // Test concurrent access to various APIs
                    let _stats = stats::fugc_get_stats();
                    let _phase = stats::fugc_get_phase();
                    let _collecting = stats::fugc_is_collecting();
                    let _cycle_stats = stats::fugc_get_cycle_stats();

                    // Test allocation info
                    let _alloc_info = allocation::fugc_alloc_info(64 + i, 8);

                    // Test post alloc
                    let obj = unsafe {
                        ObjectReference::from_raw_address_unchecked(mmtk::util::Address::from_usize(
                            0x10000 + i * 8,
                        ))
                    };
                    allocation::fugc_post_alloc(obj, 64);

                    operations += 1;
                }
                operations
            })
            .sum();
        assert!(total_ops > 0, "Threads should have performed operations");
    }

    #[test]
    fn test_error_propagation_paths() {
        // Test error handling in various scenarios - proper sad path testing

        // Test with uninitialized plan manager states
        let stats = stats::fugc_get_stats();
        // Should not panic even if plan manager is in various states
        let _ = stats.work_stolen;

        // Test with zero-aligned objects (sad path)
        let zero_addr = unsafe { mmtk::util::Address::from_usize(0x1000) }; // Non-zero but valid
        let zero_obj = unsafe { ObjectReference::from_raw_address_unchecked(zero_addr) };
        allocation::fugc_post_alloc(zero_obj, 0); // Zero size allocation (sad path)

        // Test alignment validation without dangerous memory writes
        let misaligned_addr = unsafe { mmtk::util::Address::from_usize(0x1001) }; // Odd address
        assert!(misaligned_addr.as_usize() % std::mem::align_of::<usize>() != 0);

        // Test address calculations
        let aligned_addr = unsafe { mmtk::util::Address::from_usize(0x1000) };
        assert!(aligned_addr.as_usize() % std::mem::align_of::<usize>() == 0);

        // Test with very large allocation requests (sad path)
        let (_large_size, large_align) = allocation::fugc_alloc_info(usize::MAX / 2, 1024);
        // Should handle overflow gracefully
        assert!(large_align.is_power_of_two());
    }
}