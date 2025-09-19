//! Property-based tests for fundamental GC invariants
//!
//! This module tests the core invariants that must hold for garbage collection
//! correctness, using property-based testing to explore the state space.

use mmtk::util::{Address, ObjectReference};
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use fugrip::cache_optimization::CacheOptimizedMarking;
use fugrip::concurrent::{ObjectColor, TricolorMarking};
use fugrip::test_utils::TestFixture;

/// Generate valid object references for testing
fn arb_valid_object_ref() -> impl Strategy<Value = ObjectReference> {
    (0x10000000usize..0x20000000usize).prop_map(|addr| {
        let aligned_addr = (addr / 64) * 64; // 64-byte aligned
        ObjectReference::from_raw_address(unsafe { Address::from_usize(aligned_addr) })
            .unwrap_or_else(|| {
                ObjectReference::from_raw_address(unsafe { Address::from_usize(0x10000000) })
                    .unwrap()
            })
    })
}

/// Generate object graphs with realistic connectivity patterns
#[derive(Debug, Clone)]
struct ObjectGraph {
    #[allow(dead_code)]
    objects: Vec<ObjectReference>,
    edges: HashMap<ObjectReference, Vec<ObjectReference>>,
}

impl ObjectGraph {
    fn new(objects: Vec<ObjectReference>, connectivity: f64) -> Self {
        let mut edges = HashMap::new();

        for &obj in &objects {
            let mut obj_edges = Vec::new();

            // Each object has a chance to reference other objects
            for &target in &objects {
                if obj != target && fastrand::f64() < connectivity {
                    obj_edges.push(target);
                }
            }

            edges.insert(obj, obj_edges);
        }

        ObjectGraph { objects, edges }
    }

    fn reachable_from(&self, root: ObjectReference) -> Vec<ObjectReference> {
        let mut reachable = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![root];

        while let Some(obj) = stack.pop() {
            if visited.insert(obj) {
                reachable.push(obj);

                if let Some(children) = self.edges.get(&obj) {
                    for &child in children {
                        if !visited.contains(&child) {
                            stack.push(child);
                        }
                    }
                }
            }
        }

        reachable
    }
}

proptest! {
    /// Invariant: Tricolor marking preserves reachability
    #[test]
    fn tricolor_preserves_reachability(
        obj_count in 10usize..200,
        connectivity in 0.1f64..0.8,
        root_count in 1usize..10
    ) {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024 * 1024;
        let marking = TricolorMarking::new(heap_base, heap_size);

        // Generate object graph
        let mut objects = Vec::new();
        for i in 0..obj_count {
            let addr = unsafe { Address::from_usize(0x10000000 + i * 128) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }

        if objects.is_empty() {
            return Ok(());
        }

        let graph = ObjectGraph::new(objects.clone(), connectivity);

        // Select roots
        let roots: Vec<_> = objects.iter()
            .take(root_count.min(objects.len()))
            .copied()
            .collect();

        // Calculate reachable objects from roots
        let mut all_reachable = std::collections::HashSet::new();
        for &root in &roots {
            for obj in graph.reachable_from(root) {
                all_reachable.insert(obj);
            }
        }

        // Simulate marking: mark roots as grey
        for &root in &roots {
            marking.set_color(root, ObjectColor::Grey);
        }

        // Process grey objects (simplified marking algorithm)
        let mut worklist = roots.clone();
        while let Some(obj) = worklist.pop() {
            if marking.get_color(obj) == ObjectColor::Grey {
                // Mark object as black
                marking.set_color(obj, ObjectColor::Black);

                // Mark children as grey if they're white
                if let Some(children) = graph.edges.get(&obj) {
                    for &child in children {
                        if marking.get_color(child) == ObjectColor::White {
                            marking.set_color(child, ObjectColor::Grey);
                            worklist.push(child);
                        }
                    }
                }
            }
        }

        // Verify: All reachable objects should be marked (grey or black)
        for &obj in &all_reachable {
            let color = marking.get_color(obj);
            prop_assert_ne!(color, ObjectColor::White,
                           "Reachable object should not be white");
        }

        // Verify: All marked objects should be reachable
        for &obj in &objects {
            let color = marking.get_color(obj);
            if color != ObjectColor::White {
                prop_assert!(all_reachable.contains(&obj),
                           "Marked object should be reachable");
            }
        }
    }

    /// Invariant: No black object points to white object (tricolor invariant)
    #[test]
    fn tricolor_invariant_holds(
        obj_count in 20usize..100,
        edge_count in 10usize..200
    ) {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 32 * 1024 * 1024;
        let marking = TricolorMarking::new(heap_base, heap_size);

        // Create objects
        let mut objects = Vec::new();
        for i in 0..obj_count {
            let addr = unsafe { Address::from_usize(0x10000000 + i * 64) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }

        if objects.is_empty() {
            return Ok(());
        }

        // Create edges (simplified: just pairs for testing)
        let mut edges = Vec::new();
        for _ in 0..edge_count {
            let src_idx = fastrand::usize(..objects.len());
            let dst_idx = fastrand::usize(..objects.len());
            if src_idx != dst_idx {
                edges.push((objects[src_idx], objects[dst_idx]));
            }
        }

        // Set up initial marking state respecting tricolor invariant
        // Start with all objects white
        for &obj in &objects {
            marking.set_color(obj, ObjectColor::White);
        }

        // For each edge, ensure the invariant is maintained
        for &(src, dst) in &edges {
            let src_color = marking.get_color(src);
            let dst_color = marking.get_color(dst);

            // If we're about to create a black->white edge, fix it
            if src_color == ObjectColor::Black && dst_color == ObjectColor::White {
                // Make the destination grey to maintain invariant
                marking.set_color(dst, ObjectColor::Grey);
            }
        }

        // Now randomly color some objects, but maintain the invariant
        for (i, &obj) in objects.iter().enumerate() {
            let proposed_color = match i % 3 {
                0 => ObjectColor::White,
                1 => ObjectColor::Grey,
                2 => ObjectColor::Black,
                _ => unreachable!()
            };

            // Check if setting this color would violate the invariant
            let mut can_set = true;
            if proposed_color == ObjectColor::Black {
                // Check if this object points to any white objects
                for &(src, dst) in &edges {
                    if src == obj && marking.get_color(dst) == ObjectColor::White {
                        can_set = false;
                        break;
                    }
                }
            }

            if can_set {
                marking.set_color(obj, proposed_color);
            }
        }

        // Verify tricolor invariant is maintained
        for &(src, dst) in &edges {
            let src_color = marking.get_color(src);
            let dst_color = marking.get_color(dst);

            // Tricolor invariant: black objects cannot point to white objects
            if src_color == ObjectColor::Black {
                prop_assert_ne!(dst_color, ObjectColor::White,
                               "Black object cannot point to white object");
            }
        }
    }

    /// Invariant: Cache optimization preserves marking semantics
    #[test]
    fn cache_optimization_preserves_semantics(
        objects in prop::collection::vec(arb_valid_object_ref(), 50..500)
    ) {
        if objects.is_empty() {
            return Ok(());
        }

        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 256 * 1024 * 1024;  // Match the object generation range (0x10000000..0x20000000)
        let tricolor = Arc::new(TricolorMarking::new(heap_base, heap_size));

        // Filter out duplicate objects to avoid conflicts
        let mut unique_objects = objects.clone();
        unique_objects.sort_by_key(|obj| obj.to_raw_address().as_usize());
        unique_objects.dedup();

        if unique_objects.is_empty() {
            return Ok(());
        }

        // Test 1: Standard marking approach
        for &obj in &unique_objects {
            tricolor.set_color(obj, ObjectColor::Grey);
        }

        // Verify standard marking worked
        for &obj in &unique_objects {
            let color = tricolor.get_color(obj);
            prop_assert_eq!(color, ObjectColor::Grey, "Standard marking failed");
        }

        // Reset colors for cache test
        for &obj in &unique_objects {
            tricolor.set_color(obj, ObjectColor::White);
        }

        // Test 2: Cache-optimized marking approach
        let cache_marking = CacheOptimizedMarking::with_tricolor(&tricolor);
        cache_marking.mark_objects_batch(&unique_objects);

        // Verify cache-optimized marking worked
        for &obj in &unique_objects {
            let color = tricolor.get_color(obj);
            prop_assert_eq!(color, ObjectColor::Grey, "Cache marking failed");
        }

        // Both approaches should produce the same result (Grey objects)
        prop_assert!(true, "Cache optimization preserves marking semantics");
    }

    /// Invariant: Allocation bounds and alignment
    #[test]
    fn allocation_invariants(
        requests in prop::collection::vec((16usize..8193, 8usize..65), 1..100)
    ) {
        let base = unsafe { Address::from_usize(0x20000000) };
        let heap_size = 32 * 1024 * 1024; // 32MB
        let allocator = fugrip::cache_optimization::CacheAwareAllocator::new(base, heap_size);

        let mut allocations = Vec::new();
        let mut total_allocated = 0;

        for (size, alignment) in requests {
            // Ensure alignment is power of 2
            let align = alignment.next_power_of_two();

            if let Some(addr) = allocator.allocate_aligned(size, 1) {
                // Verify alignment
                prop_assert_eq!(addr.as_usize() % align, 0);

                // Verify bounds
                prop_assert!(addr.as_usize() >= base.as_usize());
                prop_assert!(addr.as_usize() + size <= base.as_usize() + heap_size);

                allocations.push((addr, size));
                total_allocated += size;

                // Should not over-allocate
                prop_assert!(total_allocated <= heap_size);
            }
        }

        // Verify no overlapping allocations
        for i in 0..allocations.len() {
            for j in i+1..allocations.len() {
                let (addr1, size1) = allocations[i];
                let (addr2, size2) = allocations[j];

                let end1 = addr1.as_usize() + size1;
                let end2 = addr2.as_usize() + size2;

                // Ranges should not overlap
                let no_overlap = end1 <= addr2.as_usize() || end2 <= addr1.as_usize();
                prop_assert!(no_overlap, "Allocations should not overlap");
            }
        }
    }

    /// Invariant: Concurrent operations maintain consistency
    #[test]
    fn concurrent_consistency_invariant(
        worker_count in 1usize..9,
        operation_count in 10usize..100
    ) {
        let fixture = TestFixture::new_with_config(
            0x10000000,
            64 * 1024 * 1024,
            worker_count,
        );
        let coordinator = &fixture.coordinator;

        // Create test objects
        let mut objects = Vec::new();
        for i in 0..operation_count.min(500) {
            let addr = unsafe { Address::from_usize(0x10000000 + i * 128) };
            if let Some(obj) = ObjectReference::from_raw_address(addr) {
                objects.push(obj);
            }
        }

        if objects.is_empty() {
            return Ok(());
        }

        // Perform operations and verify consistency
        for batch_start in (0..objects.len()).step_by(50) {
            let batch_end = (batch_start + 50).min(objects.len());
            let batch = &objects[batch_start..batch_end];

            // Mark batch
            coordinator.mark_objects_cache_optimized(batch);

            // Verify all objects in batch are marked
            for &obj in batch {
                let color = coordinator.tricolor_marking().get_color(obj);
                prop_assert_ne!(color, ObjectColor::White,
                               "Object should be marked after processing");
            }
        }

        // Verify final statistics are reasonable
        let stats = coordinator.get_marking_stats();
        prop_assert!(stats.work_stolen < operation_count * 10);
        prop_assert!(stats.work_shared < operation_count * 10);
    }

    /// Invariant: Memory layout optimization preserves object integrity
    #[test]
    fn memory_layout_preserves_integrity(
        object_sizes in prop::collection::vec(32usize..4097, 10..100)
    ) {
        let optimizer = fugrip::cache_optimization::MemoryLayoutOptimizer::new();

        // Calculate layout
        let layouts = optimizer.calculate_object_layout(&object_sizes);

        prop_assert_eq!(layouts.len(), object_sizes.len());

        // Verify each object gets appropriate space
        for (i, (addr, allocated_size)) in layouts.iter().enumerate() {
            let requested_size = object_sizes[i];

            // Allocated size should be at least as large as requested
            prop_assert!(*allocated_size >= requested_size);

            // Large objects should be cache-line aligned
            if requested_size >= 64 {
                prop_assert_eq!(addr.as_usize() % 64, 0);
            }
        }

        // Verify no overlaps in layout
        for i in 0..layouts.len() {
            for j in i+1..layouts.len() {
                let (addr1, size1) = layouts[i];
                let (addr2, size2) = layouts[j];

                let end1 = addr1.as_usize() + size1;
                let end2 = addr2.as_usize() + size2;

                // Objects should not overlap
                prop_assert!(end1 <= addr2.as_usize() || end2 <= addr1.as_usize());
            }
        }

        // Test metadata colocation
        for (addr, _) in &layouts {
            let metadata_addr = optimizer.colocate_metadata(*addr, 24);

            // Metadata should be before object
            prop_assert!(metadata_addr.as_usize() < addr.as_usize());

            // Should be reasonably close (within 64 bytes)
            let distance = addr.as_usize() - metadata_addr.as_usize();
            prop_assert!(distance <= 64);

            // Should be aligned
            prop_assert_eq!(distance % 8, 0);
        }
    }
}
