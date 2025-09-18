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

#[cfg(test)]
mod tests {
    use super::CollectorPhase;

    #[test]
    fn safepoint_requirements_match_specification() {
        let safepoint_phases = [
            CollectorPhase::Prepare,
            CollectorPhase::Sweep,
            CollectorPhase::Release,
        ];

        for phase in safepoint_phases.iter().copied() {
            assert!(
                phase.requires_safepoint(),
                "{phase:?} should require safepoint"
            );
        }

        for phase in [
            CollectorPhase::Idle,
            CollectorPhase::Mark,
            CollectorPhase::ProcessWeak,
        ] {
            assert!(
                !phase.requires_safepoint(),
                "{phase:?} should not require safepoint"
            );
        }
    }
}
