//! FUGC coordinator types and data structures

use crate::thread::MutatorThread;
use std::sync::Arc;

/// FUGC collection cycle phases matching the 8-step protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FugcPhase {
    /// Step 1: Waiting for GC trigger
    Idle,
    /// Step 2: Turn on store barrier, soft handshake with no-op
    ActivateBarriers,
    /// Step 3: Turn on black allocation, handshake with cache reset
    ActivateBlackAllocation,
    /// Step 4: Mark global roots
    MarkGlobalRoots,
    /// Step 5: Soft handshake for stack scan + cache reset, check mark stacks
    StackScanHandshake,
    /// Step 6: Tracing - process mark stacks until empty
    Tracing,
    /// Step 7: Turn off store barrier, prepare for sweep, cache reset handshake
    PrepareForSweep,
    /// Step 8: Perform sweep with page-based allocation colouring
    Sweeping,
}

/// Colour assigned to allocation pages after sweeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationColor {
    /// Fresh page - new allocations start white.
    White,
    /// Page still contains survivors - new allocations are born black.
    Black,
}

/// Callback function type for soft handshakes
pub type HandshakeCallback = Box<dyn Fn(&MutatorThread) + Send + Sync>;

/// Statistics for FUGC collection cycles
#[derive(Debug, Default, Clone)]
pub struct FugcCycleStats {
    pub cycles_completed: usize,
    pub total_marking_time_ms: u64,
    pub total_sweep_time_ms: u64,
    pub objects_marked: usize,
    pub objects_swept: usize,
    pub handshakes_performed: usize,
    pub avg_stack_scan_objects: f64,
}

#[derive(Clone, Copy)]
pub struct PageState {
    pub live_objects: usize,
    pub allocation_color: AllocationColor,
}

impl PageState {
    pub fn new() -> Self {
        Self {
            live_objects: 0,
            allocation_color: AllocationColor::White,
        }
    }
}