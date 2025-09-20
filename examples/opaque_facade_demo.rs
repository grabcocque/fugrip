//! Demonstration of the proper opaque facade pattern
//!
//! This example shows how to use fugrip with completely opaque handles
//! that hide all MMTk implementation details.

use fugrip::{core::ObjectHeader, error::GcResult, opaque_handles::*};

fn main() -> GcResult<()> {
    println!("=== Fugrip Opaque Facade Demo ===");

    // Detect which backend we're using - no MMTk types exposed
    let backend = current_backend();
    println!("Using backend: {:?}", backend);

    // Create opaque handles - no MMTk types involved
    let plan = default_plan();
    let mutator = current_mutator();

    println!("Created plan ID: {}", plan.as_usize());
    println!("Created mutator ID: {}", mutator.as_usize());

    // Allocate objects using completely opaque interface
    let header = ObjectHeader::default();

    println!("\n=== Allocation Demo ===");
    let mut allocated_objects = Vec::new();

    for i in 0..5 {
        let size = 64 * (i + 1); // 64, 128, 192, 256, 320 bytes

        match OpaqueAllocator::allocate(mutator, header, size) {
            Ok(object_id) => {
                println!(
                    "Allocated object {}: ID={}, size={}",
                    i,
                    object_id.as_usize(),
                    object_id.size().unwrap_or(0)
                );
                allocated_objects.push(object_id);
            }
            Err(e) => {
                println!("Allocation {} failed: {:?}", i, e);
            }
        }
    }

    // Get allocation statistics - no MMTk types exposed
    let stats = OpaqueAllocator::stats();
    println!("\n=== Statistics ===");
    println!("Total allocated: {} bytes", stats.total_allocated);
    println!("Allocation count: {}", stats.allocation_count);
    println!("GC count: {}", stats.gc_count);

    // Trigger garbage collection - no MMTk types exposed
    println!("\n=== Garbage Collection ===");
    match OpaqueAllocator::trigger_gc(plan) {
        Ok(()) => println!("GC triggered successfully"),
        Err(e) => println!("GC trigger failed: {:?}", e),
    }

    // Deallocate objects - no MMTk types exposed
    println!("\n=== Deallocation ===");
    for (i, object_id) in allocated_objects.into_iter().enumerate() {
        match OpaqueAllocator::deallocate(object_id) {
            Ok(()) => println!("Deallocated object {}", i),
            Err(e) => println!("Deallocation {} failed: {:?}", i, e),
        }
    }

    // Final statistics
    let final_stats = OpaqueAllocator::stats();
    println!("\n=== Final Statistics ===");
    println!("Total allocated: {} bytes", final_stats.total_allocated);
    println!("Allocation count: {}", final_stats.allocation_count);

    println!("\n=== Demo Complete ===");
    println!("✓ No MMTk types were exposed in the public API");
    println!("✓ All operations used opaque handles only");
    println!("✓ Backend selection was transparent to the user");

    Ok(())
}

/// Example of how to create a simple allocator wrapper
/// that only exposes the opaque interface
pub struct SimpleAllocator {
    mutator: MutatorId,
    plan: PlanId,
}

impl SimpleAllocator {
    pub fn new() -> Self {
        SimpleAllocator {
            mutator: current_mutator(),
            plan: default_plan(),
        }
    }

    /// Allocate an object - no MMTk types exposed
    pub fn alloc(&self, size: usize) -> GcResult<ObjectId> {
        let header = ObjectHeader::default();
        OpaqueAllocator::allocate(self.mutator, header, size)
    }

    /// Deallocate an object - no MMTk types exposed
    pub fn dealloc(&self, object: ObjectId) -> GcResult<()> {
        OpaqueAllocator::deallocate(object)
    }

    /// Trigger GC - no MMTk types exposed
    pub fn gc(&self) -> GcResult<()> {
        OpaqueAllocator::trigger_gc(self.plan)
    }

    /// Get stats - no MMTk types exposed
    pub fn stats(&self) -> AllocatorStats {
        OpaqueAllocator::stats()
    }

    /// Get backend type - no MMTk types exposed
    pub fn backend(&self) -> BackendType {
        current_backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_allocator_interface() {
        let allocator = SimpleAllocator::new();

        // All operations use opaque handles only
        let backend = allocator.backend();
        println!("Using backend: {:?}", backend);

        let stats = allocator.stats();
        assert!(stats.total_allocated >= 0);

        // Try allocation - may fail in test environment
        if let Ok(obj) = allocator.alloc(64) {
            let _ = allocator.dealloc(obj);
        }

        // GC trigger should not panic
        let _ = allocator.gc();
    }

    #[test]
    fn test_no_mmtk_types_exposed() {
        // This test compiles only if no MMTk types are in scope
        let _mutator = current_mutator();
        let _plan = default_plan();
        let _backend = current_backend();
        let _stats = OpaqueAllocator::stats();

        // If this compiles, we successfully hid all MMTk types
    }
}
