//! Helper functions for creating and managing opaque mutator handles
//!
//! This module provides functions to create MutatorContext handles from
//! various sources, abstracting away the differences between MMTk and jemalloc.

use crate::{
    alloc_facade::MutatorContext,
    thread::MutatorThread,
    error::{GcError, GcResult},
};

/// Create a mutator context from a thread
/// This works for both MMTk and jemalloc backends
pub fn create_mutator_context(thread: &MutatorThread) -> MutatorContext {
    #[cfg(feature = "use_mmtk")]
    {
        // Try to get an existing MMTk mutator for this thread
        if let Some(mutator_ptr) = get_mmtk_mutator_for_thread(thread.id()) {
            MutatorContext::from_mmtk_mutator(mutator_ptr)
        } else {
            // Fall back to thread ID for jemalloc mode
            MutatorContext::from_thread_id(thread.id())
        }
    }
    #[cfg(not(feature = "use_mmtk"))]
    {
        MutatorContext::from_thread_id(thread.id())
    }
}

/// Create a mutator context for a new thread with MMTk integration
/// This initializes a proper MMTk mutator if the MMTk backend is available
pub fn create_and_bind_mutator(
    thread: MutatorThread,
) -> GcResult<MutatorContext> {
    #[cfg(feature = "use_mmtk")]
    {
        use crate::plan::FugcPlanManager;
        
        // Try to get MMTk instance and create proper mutator
        if let Some(plan_manager) = FugcPlanManager::global() {
            if let Ok(mmtk) = plan_manager.mmtk() {
                // Create and bind MMTk mutator
                match super::initialization::bind_mutator_thread(mmtk, thread) {
                    Ok(mutator_handle) => {
                        return Ok(MutatorContext::from_mmtk_mutator(mutator_handle.as_ptr()));
                    }
                    Err(_) => {
                        // Fall through to jemalloc mode
                    }
                }
            }
        }
    }
    
    // Fall back to jemalloc mode (always works)
    Ok(MutatorContext::from_thread_id(thread.id()))
}

/// Get an existing MMTk mutator pointer for a thread ID
#[cfg(feature = "use_mmtk")]
fn get_mmtk_mutator_for_thread(thread_id: usize) -> Option<*mut mmtk::Mutator<super::RustVM>> {
    use super::{MUTATOR_MAP, mutator_thread_key};
    use crate::compat::vm::opaque_pointer::{VMMutatorThread, VMThread, OpaquePointer};
    use crate::compat::Address;
    
    let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
        Address::from_usize(thread_id)
    })));
    
    let key = mutator_thread_key(tls);
    MUTATOR_MAP.get_or_init(|| dashmap::DashMap::new())
        .get(&key)
        .map(|entry| entry.mutator)
}

/// Helper function to check if MMTk is available and initialized
pub fn is_mmtk_available() -> bool {
    #[cfg(feature = "use_mmtk")]
    {
        use crate::plan::FugcPlanManager;
        FugcPlanManager::global()
            .and_then(|pm| pm.mmtk().ok())
            .is_some()
    }
    #[cfg(not(feature = "use_mmtk"))]
    {
        false
    }
}

/// Convenience function: create mutator context for current thread
pub fn current_thread_mutator_context() -> MutatorContext {
    let thread = MutatorThread::current();
    create_mutator_context(&thread)
}

/// Example usage patterns
#[cfg(test)]
mod examples {
    use super::*;
    use crate::core::ObjectHeader;
    
    /// Example: Allocate using modern interface
    #[test]
    fn example_allocation_with_context() {
        let thread = MutatorThread::new(1);
        let context = create_mutator_context(&thread);
        
        // Use the context with allocators that need it
        let header = ObjectHeader::default();
        
        // This would work with either backend:
        // let obj = some_allocator.allocate_with_context(header, 64, Some(context));
    }
    
    /// Example: Create and bind for new thread
    #[test] 
    fn example_thread_setup() {
        let thread = MutatorThread::new(2);
        
        match create_and_bind_mutator(thread) {
            Ok(context) => {
                // Context is ready for allocation
                println!("Mutator context created successfully");
            }
            Err(e) => {
                println!("Failed to create mutator context: {}", e);
            }
        }
    }
    
    /// Example: Check backend availability
    #[test]
    fn example_backend_detection() {
        if is_mmtk_available() {
            println!("Using MMTk backend");
        } else {
            println!("Using jemalloc backend");
        }
    }
}
