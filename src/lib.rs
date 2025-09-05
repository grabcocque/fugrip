pub mod core;
pub mod collector_phase;
pub mod collector_state;
pub mod finalizable;
pub mod free_singleton;
pub mod gc_allocator;
pub mod object_class;
pub mod segmented_heap;
pub mod suspend_for_fork;
pub mod sweeping_phase;
pub mod type_info;
pub mod weak;

pub use core::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
