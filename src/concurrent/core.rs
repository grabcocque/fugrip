//! Core concurrent marking types and utilities

use std::sync::atomic::{AtomicUsize, Ordering};

/// Branch prediction hints for performance-critical code paths
#[inline(always)]
pub(crate) fn likely(b: bool) -> bool {
    #[cold]
    fn cold() {}
    if !b {
        cold()
    }
    b
}

#[inline(always)]
pub(crate) fn unlikely(b: bool) -> bool {
    #[cold]
    fn cold() {}
    if b {
        cold()
    }
    b
}

/// Optimized fetch-and-add operation for statistics counters
///
/// This provides a more efficient alternative to repeated fetch_add operations
/// by using fetch_add with Relaxed ordering for non-critical statistics.
///
/// # Arguments
/// * `counter` - The atomic counter to increment
/// * `value` - The value to add (typically 1 for simple counting)
///
/// # Examples
/// ```rust,ignore
/// use std::sync::atomic::AtomicUsize;
/// use fugrip::concurrent::optimized_fetch_add;
///
/// let counter = AtomicUsize::new(0);
/// optimized_fetch_add(&counter, 1); // Increment by 1
/// ```
#[inline(always)]
pub fn optimized_fetch_add(counter: &AtomicUsize, value: usize) {
    // Use Relaxed ordering for statistics counters since we don't need
    // strict synchronization for non-critical metrics
    counter.fetch_add(value, Ordering::Relaxed);
}

/// Optimized fetch-and-add operation that returns the previous value
///
/// This is useful when you need both the old and new values, such as
/// when implementing work stealing algorithms or threshold checks.
///
/// # Arguments
/// * `counter` - The atomic counter to increment
/// * `value` - The value to add
///
/// # Returns
/// The value of the counter before the increment
///
/// # Examples
/// ```rust,ignore
/// use std::sync::atomic::AtomicUsize;
/// use fugrip::concurrent::optimized_fetch_add_return_prev;
///
/// let counter = AtomicUsize::new(10);
/// let prev = optimized_fetch_add_return_prev(&counter, 5);
/// assert_eq!(prev, 10); // Previous value was 10
/// assert_eq!(counter.load(Ordering::Relaxed), 15); // Now 15
/// ```
#[inline(always)]
pub fn optimized_fetch_add_return_prev(counter: &AtomicUsize, value: usize) -> usize {
    counter.fetch_add(value, Ordering::Relaxed)
}

/// Color states for tricolor marking algorithm used in concurrent garbage collection.
/// This implements Dijkstra's tricolor invariant for safe concurrent marking.
///
/// # Examples
///
/// ```
/// use fugrip::concurrent::ObjectColor;
///
/// // Objects start as white (unmarked)
/// let initial_color = ObjectColor::White;
/// assert_eq!(initial_color, ObjectColor::White);
///
/// // During marking, objects become grey (marked but not scanned)
/// let marked_color = ObjectColor::Grey;
/// assert_ne!(marked_color, ObjectColor::White);
///
/// // After scanning children, objects become black (fully processed)
/// let scanned_color = ObjectColor::Black;
/// assert_ne!(scanned_color, ObjectColor::Grey);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectColor {
    /// White objects are unmarked and candidates for collection
    White,
    /// Grey objects are marked but their children haven't been scanned yet
    Grey,
    /// Black objects are fully marked with all children scanned
    Black,
}
