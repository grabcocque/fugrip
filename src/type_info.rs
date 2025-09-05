use crate::core::*;
use std::sync::Mutex;
use once_cell::sync::Lazy;

// Macro to generate type info
#[macro_export]
macro_rules! gc_traceable {
    ($t:ty) => {
        impl $crate::GcTrace for $t {
            unsafe fn trace(&self, stack: &mut Vec<$crate::SendPtr<$crate::GcHeader<()>>>) {
                // Use reflection or manual implementation to trace Gc<T> fields
            }
        }
    };
}

// Root set management
pub static ROOTS: Lazy<Mutex<Vec<SendPtr<GcHeader<()>>>>> = Lazy::new(|| {
    Mutex::new(Vec::new())
});

pub fn register_root<T>(gc: &Gc<T>) {
    ROOTS.lock().unwrap().push(SendPtr::new(gc.as_ptr() as *mut GcHeader<()>));
}

// Stack scanning (requires cooperation with Rust runtime)
pub fn scan_stacks(_mark_stack: &mut Vec<SendPtr<GcHeader<()>>>) {
    // This would need integration with Rust's stack scanning
    // or require explicit root registration
    // TODO: Implement stack scanning
}
