// Mock implementations for dependency injection testing.
//
// This module provides mock implementations of all interface traits
// to enable comprehensive unit testing without real dependencies.

use crate::collector_phases::ThreadRegistration;
use crate::memory::SegmentedHeap;
use crate::interfaces::memory::HeapProvider;
use crate::interfaces::threading::ThreadingProvider;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::collections::HashMap;
use std::thread::ThreadId;

/// Mock heap provider for testing heap-dependent functionality
#[derive(Clone)]
pub struct MockHeapProvider {
    pub heap: Arc<SegmentedHeap>,
    pub get_heap_call_count: Arc<AtomicUsize>,
}

impl MockHeapProvider {
    pub fn new() -> Self {
        Self {
            heap: Arc::new(SegmentedHeap::new()),
            get_heap_call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn get_call_count(&self) -> usize {
        self.get_heap_call_count.load(Ordering::Acquire)
    }
}

impl HeapProvider for MockHeapProvider {
    fn get_heap(&self) -> &SegmentedHeap {
        self.get_heap_call_count.fetch_add(1, Ordering::Release);
        &self.heap
    }
}

/// Mock threading provider for testing thread coordination functionality
#[derive(Debug)]
pub struct MockThreadingProvider {
    pub mutator_count: Arc<AtomicUsize>,
    pub handshake_requested: Arc<AtomicBool>,
    pub handshake_acknowledgments: Arc<AtomicUsize>,
    pub registered_threads: Arc<Mutex<HashMap<ThreadId, ThreadRegistration>>>,
    
    // Call tracking for verification
    pub register_mutator_calls: Arc<AtomicUsize>,
    pub unregister_mutator_calls: Arc<AtomicUsize>,
    pub handshake_request_calls: Arc<AtomicUsize>,
    pub handshake_acknowledge_calls: Arc<AtomicUsize>,
    pub register_gc_calls: Arc<AtomicUsize>,
    pub unregister_gc_calls: Arc<AtomicUsize>,
}

impl MockThreadingProvider {
    pub fn new() -> Self {
        Self {
            mutator_count: Arc::new(AtomicUsize::new(0)),
            handshake_requested: Arc::new(AtomicBool::new(false)),
            handshake_acknowledgments: Arc::new(AtomicUsize::new(0)),
            registered_threads: Arc::new(Mutex::new(HashMap::new())),
            register_mutator_calls: Arc::new(AtomicUsize::new(0)),
            unregister_mutator_calls: Arc::new(AtomicUsize::new(0)),
            handshake_request_calls: Arc::new(AtomicUsize::new(0)),
            handshake_acknowledge_calls: Arc::new(AtomicUsize::new(0)),
            register_gc_calls: Arc::new(AtomicUsize::new(0)),
            unregister_gc_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    // Helper methods for test verification
    pub fn get_mutator_count(&self) -> usize {
        self.mutator_count.load(Ordering::Acquire)
    }

    pub fn get_register_mutator_call_count(&self) -> usize {
        self.register_mutator_calls.load(Ordering::Acquire)
    }

    pub fn get_handshake_request_call_count(&self) -> usize {
        self.handshake_request_calls.load(Ordering::Acquire)
    }

    pub fn get_registered_thread_count(&self) -> usize {
        self.registered_threads.lock().unwrap().len()
    }

    pub fn set_handshake_requested(&self, requested: bool) {
        self.handshake_requested.store(requested, Ordering::Release);
    }
}

impl ThreadingProvider for MockThreadingProvider {
    fn register_mutator_thread(&self) {
        self.register_mutator_calls.fetch_add(1, Ordering::Release);
        self.mutator_count.fetch_add(1, Ordering::Release);
    }

    fn unregister_mutator_thread(&self) {
        self.unregister_mutator_calls.fetch_add(1, Ordering::Release);
        self.mutator_count.fetch_sub(1, Ordering::Release);
    }

    fn get_active_mutator_count(&self) -> usize {
        self.mutator_count.load(Ordering::Acquire)
    }

    fn request_handshake(&self) {
        self.handshake_request_calls.fetch_add(1, Ordering::Release);
        self.handshake_requested.store(true, Ordering::Release);
        
        // Simulate immediate completion for simple mock behavior
        let active_mutators = self.mutator_count.load(Ordering::Acquire);
        if self.handshake_acknowledgments.load(Ordering::Acquire) >= active_mutators {
            self.handshake_requested.store(false, Ordering::Release);
        }
    }

    fn is_handshake_requested(&self) -> bool {
        self.handshake_requested.load(Ordering::Acquire)
    }

    fn acknowledge_handshake(&self) {
        self.handshake_acknowledge_calls.fetch_add(1, Ordering::Release);
        let prev_acks = self.handshake_acknowledgments.fetch_add(1, Ordering::Release);
        let active_mutators = self.mutator_count.load(Ordering::Acquire);

        // Complete handshake if this is the last acknowledgment
        if prev_acks + 1 >= active_mutators {
            self.handshake_requested.store(false, Ordering::Release);
        }
    }

    fn register_thread_for_gc(&self, stack_bounds: (usize, usize)) -> Result<(), &'static str> {
        self.register_gc_calls.fetch_add(1, Ordering::Release);
        
        let current_thread_id = std::thread::current().id();
        let current_sp = 0x1000000; // Mock stack pointer
        
        let registration = ThreadRegistration {
            thread_id: current_thread_id,
            stack_base: stack_bounds.1,
            stack_bounds,
            last_known_sp: AtomicUsize::new(current_sp),
            local_roots: Vec::new(),
            is_active: AtomicBool::new(true),
        };

        let mut threads = self.registered_threads.lock().unwrap();
        
        if threads.contains_key(&current_thread_id) {
            return Err("Thread already registered");
        }
        
        threads.insert(current_thread_id, registration);
        Ok(())
    }

    fn unregister_thread_from_gc(&self) {
        self.unregister_gc_calls.fetch_add(1, Ordering::Release);
        
        let current_thread_id = std::thread::current().id();
        let mut threads = self.registered_threads.lock().unwrap();
        threads.remove(&current_thread_id);
    }

    fn update_thread_stack_pointer(&self) {
        let current_thread_id = std::thread::current().id();
        let current_sp = 0x1000500; // Mock updated stack pointer
        
        if let Ok(mut threads) = self.registered_threads.lock() {
            if let Some(registration) = threads.get_mut(&current_thread_id) {
                registration.last_known_sp.store(current_sp, Ordering::Release);
            }
        }
    }

    fn get_current_thread_stack_bounds(&self) -> (usize, usize) {
        // Return mock stack bounds
        (0x1000000, 0x1100000) // 1MB mock stack
    }

    fn worker_suspended(&self) {
        // Mock implementation - in real implementation this would 
        // acknowledge suspension and wait for resume
    }

    fn for_each_registered_thread<F>(&self, mut f: F)
    where
        F: FnMut(&ThreadRegistration),
    {
        if let Ok(threads) = self.registered_threads.lock() {
            for registration in threads.values() {
                f(registration);
            }
        }
    }
}

impl Clone for MockThreadingProvider {
    fn clone(&self) -> Self {
        Self {
            mutator_count: self.mutator_count.clone(),
            handshake_requested: self.handshake_requested.clone(),
            handshake_acknowledgments: self.handshake_acknowledgments.clone(),
            registered_threads: self.registered_threads.clone(),
            register_mutator_calls: self.register_mutator_calls.clone(),
            unregister_mutator_calls: self.unregister_mutator_calls.clone(),
            handshake_request_calls: self.handshake_request_calls.clone(),
            handshake_acknowledge_calls: self.handshake_acknowledge_calls.clone(),
            register_gc_calls: self.register_gc_calls.clone(),
            unregister_gc_calls: self.unregister_gc_calls.clone(),
        }
    }
}

impl Default for MockHeapProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MockThreadingProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_heap_provider() {
        let mock = MockHeapProvider::new();
        
        // Test that calls are tracked
        assert_eq!(mock.get_call_count(), 0);
        
        let _heap = mock.get_heap();
        assert_eq!(mock.get_call_count(), 1);
        
        let _heap2 = mock.get_heap();
        assert_eq!(mock.get_call_count(), 2);
    }
    
    #[test]
    fn test_mock_threading_provider() {
        let mock = MockThreadingProvider::new();
        
        // Test mutator registration
        assert_eq!(mock.get_mutator_count(), 0);
        assert_eq!(mock.get_register_mutator_call_count(), 0);
        
        mock.register_mutator_thread();
        assert_eq!(mock.get_mutator_count(), 1);
        assert_eq!(mock.get_register_mutator_call_count(), 1);
        
        mock.unregister_mutator_thread();
        assert_eq!(mock.get_mutator_count(), 0);
        
        // Test handshake coordination
        assert!(!mock.is_handshake_requested());
        
        mock.request_handshake();
        assert_eq!(mock.get_handshake_request_call_count(), 1);
        
        // With no mutators, handshake should complete immediately
        assert!(!mock.is_handshake_requested());
    }
    
    #[test]
    fn test_mock_threading_gc_registration() {
        let mock = MockThreadingProvider::new();
        
        assert_eq!(mock.get_registered_thread_count(), 0);
        
        let result = mock.register_thread_for_gc((0x1000000, 0x1100000));
        assert!(result.is_ok());
        assert_eq!(mock.get_registered_thread_count(), 1);
        
        // Test double registration fails
        let result2 = mock.register_thread_for_gc((0x1000000, 0x1100000));
        assert!(result2.is_err());
        assert_eq!(mock.get_registered_thread_count(), 1);
        
        mock.unregister_thread_from_gc();
        assert_eq!(mock.get_registered_thread_count(), 0);
    }
}