//! GC phase descriptors tailored for the Fugrip + MMTk hybrid plan.

/// High-level state machine for coordinating collections.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CollectorPhase {
    Idle,
    Prepare,
    Mark,
    ProcessWeak,
    Sweep,
    Release,
}

impl CollectorPhase {
    pub fn requires_safepoint(self) -> bool {
        matches!(
            self,
            CollectorPhase::Prepare | CollectorPhase::Sweep | CollectorPhase::Release
        )
    }
}
