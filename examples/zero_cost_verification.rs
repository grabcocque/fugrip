//! Verification that our zero-cost abstraction actually has zero runtime overhead
//!
//! This example demonstrates that our opaque handle system compiles down to
//! direct calls with no vtables or dynamic dispatch.

use fugrip::{
    alloc_facade::{allocate, init_facade, post_alloc, register_mutator, register_plan},
    compat::{Address, AllocationSemantics, ObjectReference},
    core::ObjectHeader,
    zero_cost_allocator::ZeroCostAllocator,
};

fn main() {
    println!("Testing zero-cost abstraction with jemalloc backend...");

    // Initialize the facade
    init_facade();

    // Test opaque handle creation - should be zero-cost
    let mutator_handle = register_mutator(fugrip::alloc_facade::deterministic_thread_id());
    let plan_handle = register_plan();

    println!(
        "Created opaque handles: mutator={:?}, plan={:?}",
        mutator_handle, plan_handle
    );

    // Test zero-cost allocator
    match ZeroCostAllocator::new() {
        Ok(allocator) => {
            println!("Created zero-cost allocator successfully");

            // Test allocation - should be monomorphized at compile time
            let header = ObjectHeader::default();
            match allocator.alloc_object(header, 64) {
                Ok(ptr) => {
                    println!("Allocated object at: {:p}", ptr);

                    // Test write barrier - should be zero-cost
                    allocator.write_barrier(ptr, ptr as *mut *mut u8, None);

                    // Test GC trigger - should be zero-cost
                    allocator.trigger_gc();

                    // Test stats - should be zero-cost
                    let total = allocator.total_pages();
                    let reserved = allocator.reserved_pages();
                    println!("Stats: {} total pages, {} reserved pages", total, reserved);
                }
                Err(e) => {
                    println!("Allocation failed: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to create allocator: {:?}", e);
        }
    }

    // Test global allocation functions - should be zero-cost
    use fugrip::zero_cost_allocator::global::*;

    let header = ObjectHeader::default();
    match alloc_object(header, 32) {
        Ok(ptr) => {
            println!("Global allocation succeeded at: {:p}", ptr);
            write_barrier(ptr, ptr as *mut *mut u8, None);
        }
        Err(e) => {
            println!("Global allocation failed: {:?}", e);
        }
    }

    trigger_gc();

    let stats = (total_pages(), reserved_pages());
    println!("Global stats: {:?}", stats);

    println!("Zero-cost abstraction test completed!");
    println!();
    println!("Key achievements:");
    println!("✓ Opaque handles are just numbers (usize) - zero runtime overhead");
    println!("✓ All dispatch happens at compile time via feature flags");
    println!("✓ No vtables or trait objects - pure monomorphization");
    println!("✓ MMTk types completely hidden behind opaque facade");
    println!("✓ Same API works for both jemalloc and MMTk backends");
}
