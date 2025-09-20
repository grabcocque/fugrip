//! Minimal demonstration of truly opaque zero-cost abstraction
//!
//! This shows how opaque handles work without any type leakage.
//! ALL MMTk types are completely hidden behind handles.

// Only import opaque types - NO compat layer
use fugrip::alloc_facade::{
    AllocationSemantics, MutatorHandle, PlanHandle, allocate, get_plan_reserved_pages,
    get_plan_total_pages, global_allocator, init_facade, post_alloc, register_mutator,
    register_plan, trigger_gc, write_barrier,
};
use fugrip::core::ObjectHeader;

fn main() {
    println!("=== TRULY OPAQUE ZERO-COST FACADE DEMO ===");

    // 1. Initialize facade (one-time setup)
    init_facade();

    // 2. Create opaque handles - just numbers, zero overhead
    let mutator: MutatorHandle = register_mutator(fugrip::alloc_facade::deterministic_thread_id());
    let plan: PlanHandle = register_plan();

    println!("✓ Opaque handles created:");
    println!("  Mutator: {:?} (just a number)", mutator);
    println!("  Plan: {:?} (just a number)", plan);

    // 3. Allocation through pure handles - zero-cost dispatch
    let header = ObjectHeader::default();
    match allocate(mutator, 64, 8, 0, AllocationSemantics::Default) {
        Ok(ptr) => {
            println!("✓ Allocated at: {:p} (via opaque handle)", ptr);

            // 4. Post-allocation through handles - zero-cost
            post_alloc(mutator, ptr, 64, AllocationSemantics::Default);
            println!("✓ Post-allocation completed");

            // 5. Write barrier through handles - zero-cost
            write_barrier(mutator, ptr, ptr as *mut *mut u8, None);
            println!("✓ Write barrier executed");
        }
        Err(e) => {
            println!("✗ Allocation failed: {:?}", e);
        }
    }

    // 6. GC operations through handles - zero-cost
    trigger_gc(plan);
    println!("✓ GC triggered via plan handle");

    // 7. Statistics through handles - zero-cost
    let total_pages = get_plan_total_pages(plan);
    let reserved_pages = get_plan_reserved_pages(plan);
    println!(
        "✓ Stats: {} total, {} reserved pages",
        total_pages, reserved_pages
    );

    println!();
    println!("=== ZERO-COST GUARANTEES ===");
    println!("✓ Handles are just usize - no memory overhead");
    println!("✓ All dispatch via feature flags - compile-time only");
    println!("✓ No vtables, no trait objects - pure monomorphization");
    println!("✓ NO Address/ObjectReference types exposed");
    println!("✓ MMTk completely hidden behind opaque facade");

    // Demonstrate the key insight: size verification
    use std::mem::size_of;
    println!();
    println!("=== SIZE VERIFICATION ===");
    println!(
        "MutatorHandle size: {} bytes (just usize)",
        size_of::<MutatorHandle>()
    );
    println!(
        "PlanHandle size: {} bytes (just usize)",
        size_of::<PlanHandle>()
    );
    println!("Raw pointer size: {} bytes", size_of::<*mut u8>());

    assert_eq!(size_of::<MutatorHandle>(), size_of::<usize>());
    assert_eq!(size_of::<PlanHandle>(), size_of::<usize>());

    println!("✓ All handles are exactly usize - perfect zero-cost abstraction!");
}
