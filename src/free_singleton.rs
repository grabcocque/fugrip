use crate::core::*;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

// Free singleton - the key innovation for definitive freeing
static FREE_SINGLETON: AtomicPtr<GcHeader<()>> = AtomicPtr::new(std::ptr::null_mut());

pub struct FreeSingleton {
    header: GcHeader<()>,
}

impl FreeSingleton {
    pub fn instance() -> *mut GcHeader<()> {
        let ptr = FREE_SINGLETON.load(Ordering::Acquire);
        if ptr.is_null() {
            // Initialize singleton
            let singleton = Box::leak(Box::new(FreeSingleton {
                header: GcHeader {
                    mark_bit: AtomicBool::new(true), // Always marked
                    type_info: &FREE_SINGLETON_TYPE_INFO,
                    forwarding_ptr: AtomicPtr::new(std::ptr::null_mut()),
                    data: (),
                },
            }));
            let header_ptr = &mut singleton.header as *mut GcHeader<()>;

            match FREE_SINGLETON.compare_exchange(
                std::ptr::null_mut(),
                header_ptr,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => header_ptr,
                Err(existing) => existing, // Another thread beat us
            }
        } else {
            ptr
        }
    }
}

// Enhanced Gc pointer with redirection support
impl<T> Gc<T> {
    pub fn read(&self) -> GcResult<GcRef<'_, T>> {
        // Check if redirected to free singleton
        let current_ptr = self.ptr;
        let free_singleton = FreeSingleton::instance();

        if current_ptr as *mut GcHeader<()> == free_singleton {
            return Err(GcError::AccessToFreedObject);
        }

        // Check for forwarding (in case of compaction in future)
        unsafe {
            let forwarding = (*current_ptr).forwarding_ptr.load(Ordering::Acquire);
            if !forwarding.is_null() {
                // TODO: In a real implementation, we'd need to handle forwarding pointers
                // For now, just use the forwarded pointer without updating self
                return Ok(GcRef::new_from_ptr(forwarding));
            }
        }

        Ok(GcRef::new(self))
    }
    
    pub fn write(&self) -> Option<GcRefMut<'_, T>> {
        use crate::segmented_heap::COLLECTOR;
        if COLLECTOR.is_marking() {
            None // Prevent mutation during marking
        } else {
            Some(GcRefMut::new(self))
        }
    }
}
