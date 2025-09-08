# CollectorState decomposition design

## Context

Referenced code locations:

- [`src/lib.rs`](src/lib.rs:1)
- [`src/collector/mod.rs`](src/collector/mod.rs:1)
- [`src/collector/phase_manager.rs`](src/collector/phase_manager.rs:1)
- [`src/collector/suspension_manager.rs`](src/collector/suspension_manager.rs:1)
- [`src/collector_phases.rs`](src/collector_phases.rs:1)
- [`src/memory.rs`](src/memory.rs:1)
- [`src/types.rs`](src/types.rs:1)
- [`src/interfaces.rs`](src/interfaces.rs:1)

Note: This is a design-only document. Do not modify code in this task.

## Executive summary

The CollectorState decomposition extracts responsibilities currently bundled in `CollectorState` into cohesive components to improve testability, reduce coupling, and enable safe, incremental refactors. The decomposition isolates lifecycle/phase handling, memory bookkeeping, thread coordination, stack scanning, and external interactions into separate modules so each can be developed, tested, and iterated independently.

The migration is planned to preserve the current public API during the transition by introducing a minimal stable façade (CollectorStateCore) and compatibility shims. This allows progressive replacement of internals without breaking tests or consumers.

## Component list

The new components below are suggested modules with public interfaces. Each entry begins with a file link followed by a one-sentence summary and the public API sketch.

1. CollectorStateCore — [`src/collector/state_core.rs`](src/collector/state_core.rs:1)  
   Responsibility: Core minimal state and lifecycle hooks that remain the stable façade used by callers.  
   Interface sketch:
   [`src/collector/state_core.rs`](src/collector/state_core.rs:1)
   pub struct CollectorStateCore {
   pub phase_manager: crate::collector::phase_manager::PhaseManager,
   pub memory_manager: crate::collector::memory_manager::MemoryManager,
   pub thread_coordinator: crate::collector::thread_coordinator::ThreadCoordinator,
   pub suspension_manager: crate::collector::suspension_manager::SuspensionManager,
   pub phase_controller: crate::collector::phase_controller::PhaseController,
   }
   impl CollectorStateCore {
   pub fn new(config: &crate::GcConfig, deps: Box<dyn crate::collector::external_deps::ExternalDeps + Send + Sync>) -> Self;
   pub fn request_collection(&self);
   pub fn current_phase(&self) -> crate::CollectorPhase;
   pub fn is_handshake_requested(&self) -> bool;
   pub fn acknowledge_handshake(&self);
   pub fn register_thread_for_gc(&self, bounds: (usize, usize)) -> Result<(), &'static str>;
   pub fn unregister_thread_from_gc(&self);
   }
   Notes: CollectorStateCore owns the concrete components and provides the stable public API. It should be Send + Sync and use interior mutability (Mutex/Atomic) for shared state. The public API intentionally mirrors the legacy CollectorState façade to minimize callers' changes.

2. MemoryManager — [`src/collector/memory_manager.rs`](src/collector/memory_manager.rs:1)  
   Responsibility: Allocation bookkeeping, segment management, and object metadata helpers.  
   Interface sketch:
   [`src/collector/memory_manager.rs`](src/collector/memory_manager.rs:1)
   pub struct MemoryManager { /_ internal fields (allocator handle, segments, stats) _/ }
   impl MemoryManager {
   pub fn new(config: &crate::GcConfig) -> Self;
   pub fn allocate<T>(&self) -> crate::GcResult<*mut crate::GcHeader<T>>;
   pub fn get_free_singleton(&self) -> *mut crate::GcHeader<()>;
   pub fn get_heap_segments(&self) -> Vec<crate::memory::Segment>;
   pub fn scan_segment_for_live_objects<F>(&self, segment: &crate::memory::Segment, visitor: &mut F) where F: FnMut(*mut crate::GcHeader<()>) + Send;
   pub fn sweep_all_segments_parallel(&self, free_singleton: *mut crate::GcHeader<()>);
   }
   Notes: MemoryManager owns allocator state; concurrency: Send + Sync; internal mutability via Mutex for segment lists and atomics for stats. Exposes thin helpers used by PhaseController and marking workers.

3. ThreadCoordinator — [`src/collector/thread_coordinator.rs`](src/collector/thread_coordinator.rs:1)  
   Responsibility: Thread registration, handshake coordination, and mutator bookkeeping.  
   Interface sketch:
   [`src/collector/thread_coordinator.rs`](src/collector/thread_coordinator.rs:1)
   pub struct ThreadCoordinator { /_ handshake flags, thread registry _/ }
   impl ThreadCoordinator {
   pub fn new() -> Self;
   pub fn register*mutator_thread(&self);
   pub fn unregister_mutator_thread(&self);
   pub fn request_handshake(&self);
   pub fn is_handshake_requested(&self) -> bool;
   pub fn acknowledge_handshake(&self);
   pub fn register_thread_for_gc(&self, stack_bounds: (usize, usize)) -> Result<(), &'static str>;
   pub fn unregister_thread_from_gc(&self);
   pub fn update_thread_stack_pointer(&self);
   pub fn get_current_thread_stack_bounds(&self) -> (usize, usize);
   }
   Notes: ThreadCoordinator uses atomics and Condvar for wait/notify; Send + Sync. Thread registration stored under a Mutex<Vec<*>>. Ownership: ThreadCoordinator is owned by CollectorStateCore and holds no references with non-'static lifetimes.

4. StackScanner — [`src/collector/stack_scanner.rs`](src/collector/stack_scanner.rs:1)  
   Responsibility: Encapsulate stack and memory conservative scanning logic using heap validation helpers.  
   Interface sketch:
   [`src/collector/stack_scanner.rs`](src/collector/stack_scanner.rs:1)
   pub struct StackScanner {
   pub fn new() -> Self;
   pub fn with_heap_checker(heap_checker: Box<dyn crate::collector_phases::HeapBoundsChecker + Send + Sync>) -> Self;
   pub unsafe fn conservative_scan_memory_range(&self, start: *const u8, end: *const u8, global_stack: &mut Vec<crate::SendPtr<crate::GcHeader<()>>>);
   pub fn is_within_heap_bounds(&self, ptr: *mut crate::GcHeader<()>) -> bool;
   pub fn is_valid_gc_pointer(&self, ptr: *mut crate::GcHeader<()>) -> bool;
   }
   Notes: StackScanner holds a boxed trait object for heap checks. It is Send + Sync if heap_checker is Send + Sync; methods that inspect memory are unsafe and documented.

5. PhaseController — [`src/collector/phase_controller.rs`](src/collector/phase_controller.rs:1)  
   Responsibility: Orchestrates GC phases and coordinates worker lifecycles and phase transitions.  
   Interface sketch:
   [`src/collector/phase_controller.rs`](src/collector/phase_controller.rs:1)
   pub struct PhaseController { /_ references to phase manager, thread coordinator, memory manager _/ }
   impl PhaseController {
   pub fn new(core: &CollectorStateCore) -> Self;
   pub fn start_collection(&self);
   pub fn run_marking(&self);
   pub fn run_sweeping(&self);
   pub fn request_phase_transition(&self, phase: crate::CollectorPhase);
   pub fn current_phase(&self) -> crate::CollectorPhase;
   }
   Notes: PhaseController coordinates short-lived worker threads, calls into MemoryManager and StackScanner, and updates PhaseManager. It is Send + Sync and owns only 'static references or Arc clones passed from CollectorStateCore.

6. ExternalDeps trait module — [`src/collector/external_deps.rs`](src/collector/external_deps.rs:1)  
   Responsibility: Abstract external interactions (psm, logging, OS stack queries) to reduce coupling and ease testing.  
   Interface sketch:
   [`src/collector/external_deps.rs`](src/collector/external_deps.rs:1)
   pub trait ExternalDeps {
   fn now(&self) -> std::time::Instant;
   fn stack_pointer(&self) -> \*const u8;
   fn log_warn(&self, msg: &str);
   fn log_info(&self, msg: &str);
   fn spawn_worker<F>(&self, name: &str, f: F) where F: FnOnce() + Send + 'static;
   }
   Notes: ExternalDeps is a thin trait; implementations live in a small module and are provided to CollectorStateCore at construction time. This trait avoids direct psm/OS calls scattered across modules.

## Interaction design (call-flow sequences)

Below are prose sequence diagrams for core flows. Each numbered step shows the responsible component and key methods invoked.

Normal allocation (mutator fast path)

1. Mutator thread calls its thread-local allocation helper (existing MutatorState::try_allocate) — responsibility: mutator-local code.
2. If the thread-local buffer has space, allocation succeeds locally; no collector interaction.
3. If buffer exhausted, mutator calls CollectorStateCore::memory_manager.allocate::<T>().
4. MemoryManager performs allocator bookkeeping, updates stats, and returns a pointer or an allocation error (GcError::AllocationFailed).
5. If allocation crossed a configured threshold, MemoryManager or the mutator signals PhaseController::start_collection() via CollectorStateCore::request_collection().

Begin GC phase (request -> marking start)

1. External or internal trigger calls CollectorStateCore::request_collection() (or PhaseController::start_collection()).
2. CollectorStateCore forwards to PhaseManager (PhaseManager::request_collection()) to set phase to Marking.
3. PhaseController notices the phase change (or is the caller) and invokes ThreadCoordinator::request_handshake() to coordinate mutators.
4. ThreadCoordinator sets handshake_requested and waits for acknowledgments using Condvar.
5. Mutator threads periodically poll ThreadCoordinator::is_handshake_requested() and when observed call ThreadCoordinator::acknowledge_handshake().
6. After acknowledgments complete, PhaseController spawns marking workers via ExternalDeps::spawn_worker and calls PhaseController::run_marking().

Mark (parallel marking)

1. PhaseController-run workers pop initial roots (from globals and static roots) via MemoryManager helpers and push to a shared mark stack.
2. Workers repeatedly pop items and scan their outgoing references, validating pointers with StackScanner::is_valid_gc_pointer.
3. Valid live objects are pushed to the global mark stack; marking uses atomic mark bits and MemoryManager helper methods.
4. Workers coordinate via atomic counters to implement work-stealing/donation; PhaseController monitors workers_finished to know completion.

Scan stacks (precise and conservative)

1. PhaseController asks ThreadCoordinator for registered threads and their stack bounds.
2. For precise stacks, mutator thread registrations expose local roots; PhaseController may request the mutator to scan its local_roots.
3. For conservative scanning, PhaseController uses StackScanner::conservative_scan_memory_range for specified address ranges and accumulates candidate pointers on the global mark stack.

Suspend/resume threads (suspension flow)

1. When a suspension is needed (e.g., for fork), PhaseController calls SuspensionManager::request_suspension().
2. Worker and mutator threads periodically check SuspensionManager::is_suspension_requested(); when true they call SuspensionManager::worker_suspended() which blocks until resume.
3. SuspensionManager::wait_for_suspension() waits until suspended_worker_count >= active_worker_count, with a timeout and diagnostic logging.
4. On resume, SuspensionManager::resume_collection() clears the flag and notifies waiting threads (Condvar notify_all).

Sweep

1. Once marking/reviving phases finish, PhaseController calls MemoryManager::sweep_all_segments_parallel(free_singleton).
2. MemoryManager spawns sweep workers (ExternalDeps::spawn_worker) to walk segments and free unreachable objects, using allocator internals and redirect_pointers helpers.
3. On completion PhaseController sets PhaseManager::set_phase(CollectorPhase::Waiting) and updates stats.

## Migration plan — prioritized, incremental steps with minimal code changes per step

Overview: perform migration in small steps that create new modules, add internal shims in CollectorState, and progressively move callers to the new CollectorStateCore facade. Each step keeps public API stable by adding forwarding/shim methods.

Step 1 — Create component modules and interfaces (non-invasive)

- Files created:
  - `src/collector/state_core.rs`
  - `src/collector/memory_manager.rs`
  - `src/collector/thread_coordinator.rs`
  - `src/collector/stack_scanner.rs`
  - `src/collector/phase_controller.rs`
  - `src/collector/external_deps.rs`
- Files edited:
  - `src/collector_phases.rs` (add thin construction adapters that instantiate new components but do not change public API)
- API changes: none exported; CollectorState keeps existing public fields. New modules only add new types.
- Call-sites to update: none externally; internal CollectorState will be updated to construct the new components (zero external callsite edits).
- Change budget: modify 1 module (`src/collector_phases.rs`) internally; create 6 new files.
- Verification checklist:
  - cargo build
  - cargo test --tests
- Risk/effort: small. Mitigation: do not remove or change any existing public fields; implement CollectorState construction that populates both legacy fields and new components.

Step 2 — Move ThreadCoordinator and SuspensionManager usage into new modules

- Files created/edited:
  - created: `src/collector/thread_coordinator.rs` (port the ThreadCoordinator implementation)
  - edited: `src/collector_phases.rs` to delegate thread-related logic to ThreadCoordinator
- API changes: new module exposes ThreadCoordinator with identical methods used today.
- Call-sites to update (≤5):
  - In `src/collector_phases.rs` update methods: `request_handshake`, `acknowledge_handshake`, `register_thread_for_gc`, `unregister_thread_from_gc`, `is_handshake_requested` to call `self.thread_coordinator.*` (approx 3-5 spots).
- Change budget: change only `src/collector_phases.rs` internals to forward calls.
- Verification checklist:
  - cargo build
  - cargo test for handshake/suspension: `tests/collector_phase_tests.rs`, `tests/collector_phases_simple_tests.rs`
- Risk/effort: small/medium. Mitigation: keep legacy public fields and provide forwarding implementations until tests are green.

Step 3 — Introduce MemoryManager and move allocation bookkeeping

- Files created/edited:
  - created: `src/collector/memory_manager.rs`
  - edited: `src/memory.rs` (small delegating helpers) and `src/collector_phases.rs` (mutator allocation paths)
- API changes: MemoryManager public API added. CollectorState will forward allocation requests to MemoryManager; no change in public signatures of MutatorState.try_allocate visible externally.
- Call-sites to update (3-5):
  - `src/collector_phases.rs::MutatorState::try_allocate` -> delegate to `self.core.memory_manager.allocate::<T>()` (1)
  - CollectorState sweeping/scan entrypoints -> call MemoryManager helpers (2-4)
- Change budget: one module's public API addition, up to 5 callsite updates.
- Verification checklist:
  - cargo build
  - run memory tests: `tests/memory_simple_tests.rs`, `tests/memory_coverage_tests.rs`
- Risk/effort: medium. Mitigation: keep existing allocation helpers and add delegations; run memory-focused tests.

Step 4 — Extract StackScanner and PhaseController; wire PhaseController as orchestrator

- Files created/edited:
  - created: `src/collector/stack_scanner.rs`, `src/collector/phase_controller.rs`
  - edited: `src/collector_phases.rs` to delegate scanning and phase orchestration to PhaseController
- API changes: PhaseController public API added. CollectorState methods like `start_marking_phase` will internally call `self.phase_controller.start_collection()` while preserving the original public method signatures temporarily.
- Call-sites to update (≤5):
  - `CollectorState::start_marking_phase` -> delegate to `PhaseController::start_collection` (1)
  - marking/sweep internal helpers -> delegate to PhaseController or StackScanner (2-4)
- Change budget: one new module public API; update up to 5 callsites in `src/collector_phases.rs`.
- Verification checklist:
  - cargo build
  - run marking and sweep tests: `tests/collector_phases_comprehensive_tests.rs`, `tests/collector_phases_expanded_tests.rs`
- Risk/effort: medium. Mitigation: keep CollectorState forwarding methods and run the full suite for coverage.

Step 5 — Remove duplicated legacy fields and switch callers to CollectorStateCore facade

- Files created/edited:
  - edited: `src/lib.rs` (optionally re-export CollectorStateCore), `src/collector_phases.rs` (remove legacy exposed public fields once compatibility is verified)
- API changes: public API change (remove legacy fields) but provide a compatibility module `collector::compat` containing shim accessors that map old names to new APIs for a transitional release.
- Call-sites to update (≤5):
  - Tests and internal modules referencing CollectorState.{phase, suspend_count, ...} -> update to use accessors (up to 5 files).
- Change budget: one module's public API change (CollectorState/CollectorStateCore) and a small number of callsite edits.
- Verification checklist:
  - cargo build
  - cargo test (full suite)
- Risk/effort: medium/large. Mitigation: publish compatibility shims for at least one release cycle and use CI to catch regressions.

## Concrete list of source locations that will require edits during migration

- `src/collector_phases.rs` — CollectorState impl (new(), start_marking_phase, marking_worker, sweeping_phase, scan and sweep helpers).
- `src/collector/phase_manager.rs` — integration points (PhaseManager already exists; ensure usage is routed via PhaseController).
- `src/collector/suspension_manager.rs` — leave as-is but update call-sites in CollectorState to call into this module.
- `src/memory.rs` — small delegations to MemoryManager or to adapt internals.
- `src/lib.rs` — re-exports and potential public API re-shaping to export CollectorStateCore in later steps.
- Tests and test helpers:
  - `tests/collector_phase_tests.rs`
  - `tests/collector_phases_simple_tests.rs`
  - `tests/collector_phases_comprehensive_tests.rs`
  - `tests/memory_simple_tests.rs`
  - Any other test referencing CollectorState's legacy public fields

## Backwards-compatibility considerations

- During migration maintain all current public fields and method signatures on `CollectorState` and make them thin wrappers that delegate to the new components. This ensures existing tests and consumers continue to function.
- After successful migration and test coverage, introduce `collector::compat` module that maps deprecated names to new accessors and mark legacy fields/methods as #[deprecated] for at least one release.
- When removing legacy fields, provide clear migration notes and compiler-time deprecation warnings guiding users to new accessor methods on CollectorStateCore/compat.

## Notes on testing strategy during migration

- After each step run `cargo build` and targeted test subsets as described in each step's verification checklist.
- Add unit tests for each new component as it becomes non-trivial:
  - MemoryManager: allocation, get_free_singleton, scanning a synthetic segment, sweep invariants.
  - ThreadCoordinator: handshake lifecycle, register/unregister, concurrency tests with mock threads.
  - StackScanner: conservative_scan_memory_range against a synthetic memory buffer containing valid and invalid pointers.
  - PhaseController: simple orchestration smoke tests (start_collection -> run_marking -> run_sweeping) using injected ExternalDeps that records worker spawns.
- Add an integration smoke test that performs a simple allocation, requests collection, waits for completion, and asserts heap invariants remain valid.

## Recommended tiny helper traits (names and exact signatures)

- [`src/collector/external_deps.rs`](src/collector/external_deps.rs:1)
  pub trait ExternalDeps {
  fn now(&self) -> std::time::Instant;
  fn stack_pointer(&self) -> \*const u8;
  fn log_warn(&self, msg: &str);
  fn log_info(&self, msg: &str);
  fn spawn_worker<F>(&self, name: &str, f: F) where F: FnOnce() + Send + 'static;
  }

- [`src/collector/memory_manager.rs`](src/collector/memory_manager.rs:1)
  pub trait HeapView {
  fn get_segments(&self) -> Vec<crate::memory::Segment>;
  fn find_segment_for_ptr(&self, ptr: \*mut crate::GcHeader<()>) -> Option<crate::memory::Segment>;
  }

## Constraints (repeated)

- Do not propose code edits in this task.
- Do not propose a full runtime redesign—focus strictly on decomposing responsibilities currently in CollectorState.
- Public behavior and external API must be preserved; any proposed API changes must be accompanied by a migration shim plan.
- Keep proposed module and type names consistent with existing project naming conventions.

## Appendix: example minimal shim signatures (for migration)

- [`src/collector/state_core.rs`](src/collector/state_core.rs:1)
  impl CollectorState {
  pub fn request_collection(&self) { self.phase_manager.request_collection(); }
  pub fn is_handshake_requested(&self) -> bool { self.thread_coordinator.is_handshake_requested() }
  }

End of design document.
