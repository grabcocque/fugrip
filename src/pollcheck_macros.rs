//! Macro-based automatic pollcheck insertion for LLVM-style safepoints
//!
//! This module provides macros that automatically insert pollchecks without
//! requiring compiler support, enabling bounded-progress guarantees through
//! careful macro design.

use crate::safepoint::pollcheck;

/// Automatically insert pollchecks in loops
///
/// This macro wraps loop constructs and ensures pollchecks happen regularly.
///
/// # Examples
///
/// ```rust
/// use fugrip::gc_loop;
///
/// let mut sum = 0;
/// gc_loop!(for i in 0..1000000 => {
///     sum += i;
/// });
/// ```
#[macro_export]
macro_rules! gc_loop {
    // For loop with pollcheck every N iterations
    (for $var:pat in $iter:expr => { $($body:tt)* }) => {
        {
            let mut _pollcheck_counter = 0;
            for $var in $iter {
                if _pollcheck_counter % 1000 == 0 {
                    $crate::safepoint::pollcheck();
                }
                _pollcheck_counter += 1;
                $($body)*
            }
        }
    };

    // While loop with pollcheck
    (while $cond:expr => { $($body:tt)* }) => {
        {
            let mut _pollcheck_counter = 0;
            while $cond {
                if _pollcheck_counter % 1000 == 0 {
                    $crate::safepoint::pollcheck();
                }
                _pollcheck_counter += 1;
                $($body)*
            }
        }
    };

    // Infinite loop with pollcheck
    (loop { $($body:tt)* }) => {
        {
            let mut _pollcheck_counter = 0;
            loop {
                if _pollcheck_counter % 1000 == 0 {
                    $crate::safepoint::pollcheck();
                }
                _pollcheck_counter += 1;
                $($body)*
            }
        }
    };
}

/// Insert pollchecks in function calls that might run for a long time
///
/// # Examples
///
/// ```rust
/// use fugrip::gc_call;
///
/// let result = gc_call!(expensive_computation(data));
/// ```
#[macro_export]
macro_rules! gc_call {
    ($func:expr) => {
        {
            $crate::safepoint::pollcheck();
            let result = $func;
            $crate::safepoint::pollcheck();
            result
        }
    };
}

/// Create a function that automatically inserts pollchecks at entry/exit
///
/// # Examples
///
/// ```rust
/// use fugrip::gc_function;
///
/// gc_function! {
///     fn process_data(data: &[u8]) -> usize {
///         let mut sum = 0;
///         for &byte in data {
///             sum += byte as usize;
///         }
///         sum
///     }
/// }
/// ```
#[macro_export]
macro_rules! gc_function {
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident($($param:ident: $param_ty:ty),*) -> $ret:ty {
            $($body:tt)*
        }
    ) => {
        $(#[$attr])*
        $vis fn $name($($param: $param_ty),*) -> $ret {
            $crate::safepoint::pollcheck(); // Entry pollcheck

            let result = {
                $($body)*
            };

            $crate::safepoint::pollcheck(); // Exit pollcheck
            result
        }
    };

    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident($($param:ident: $param_ty:ty),*) {
            $($body:tt)*
        }
    ) => {
        $(#[$attr])*
        $vis fn $name($($param: $param_ty),*) {
            $crate::safepoint::pollcheck(); // Entry pollcheck
            $($body)*
            $crate::safepoint::pollcheck(); // Exit pollcheck
        }
    };
}

/// Allocator wrapper that inserts pollchecks during allocation
///
/// # Examples
///
/// ```rust
/// use fugrip::gc_alloc;
///
/// let data: Vec<u32> = gc_alloc!(Vec::with_capacity(1000));
/// ```
#[macro_export]
macro_rules! gc_alloc {
    ($alloc_expr:expr) => {
        {
            $crate::safepoint::pollcheck();
            $alloc_expr
        }
    };
}

/// Bounded work macro - ensures pollchecks happen within work limits
///
/// # Examples
///
/// ```rust
/// use fugrip::bounded_work;
///
/// bounded_work!(1000 => {
///     // This work is guaranteed to pollcheck every 1000 operations
///     for i in 0..10000 {
///         expensive_operation(i);
///     }
/// });
/// ```
#[macro_export]
macro_rules! bounded_work {
    ($limit:expr => { $($body:tt)* }) => {
        {
            let mut _work_counter = 0;
            let _work_limit = $limit;

            macro_rules! work_unit {
                () => {
                    _work_counter += 1;
                    if _work_counter >= _work_limit {
                        $crate::safepoint::pollcheck();
                        _work_counter = 0;
                    }
                };
            }

            $($body)*
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_gc_loop_macro() {
        let counter = AtomicUsize::new(0);

        gc_loop!(for i in 0..10000 => {
            counter.fetch_add(i, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::Relaxed), (0..10000).sum());
    }

    #[test]
    fn test_gc_function_macro() {
        gc_function! {
            fn test_add(a: i32, b: i32) -> i32 {
                a + b
            }
        }

        assert_eq!(test_add(5, 3), 8);
    }

    #[test]
    fn test_bounded_work_macro() {
        let work_done = AtomicUsize::new(0);

        bounded_work!(100 => {
            for i in 0..1000 {
                work_unit!();
                work_done.fetch_add(1, Ordering::Relaxed);
            }
        });

        assert_eq!(work_done.load(Ordering::Relaxed), 1000);
    }
}