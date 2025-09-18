# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust implementation of FUGC (Fil's Unbelievable Garbage Collector), a parallel concurrent on-the-fly grey-stack Dijkstra accurate non-moving garbage collector originally designed for the Verse programming language. The project implements a lock-free handshake protocol and integrates with MMTk (Memory Management Toolkit) for production-ready garbage collection.

## Build & Development Commands

```bash
# Build the project
cargo build

# Run all tests using nextest (preferred test runner)
cargo nextest run
# OR use cargo aliases:
cargo nextest-all

# Run specific test categories
cargo nextest run --features smoke      # Lightweight GC semantics validation
cargo nextest run --features stress-tests  # Expensive stress tests
cargo nextest-smoke    # Alias for smoke tests
cargo nextest-stress   # Alias for stress tests

# Run a single test
cargo nextest run test_name

# Run without fail-fast (see all test results)
cargo nextest run --no-fail-fast

# Check compilation without building
cargo check

# Format code
cargo fmt

# Run lints
cargo clippy

# Run specific benchmarks (disabled by default)
cargo bench --bench gc_benchmarks
cargo bench --bench cache_benchmarks
```

## Current Implementation Status

**✅ Core Infrastructure Complete:**

- Lock-free handshake protocol using crossbeam channels and atomic state machines
- Thread registry with deadlock-free coordination
- MMTk VM binding layer (RustVM)
- Safepoint infrastructure with pollcheck macros
- SIMD-optimized sweeping algorithms

**🔄 FUGC 8-Step Protocol:**

- Steps 1-8 implemented but integration layer needs work
- Lock-free handshake eliminates deadlocks (252/259 tests pass)
- Some coordinator integration issues remain (7 failing tests)

**📊 Test Health: 97.3% pass rate (252/259 tests)**

## Architecture

The project implements a FUGC-inspired garbage collector for a Rust VM using MMTk as the foundation.

### MMTk Integration Strategy

**MMTk Plan Composition:**

```rust
// Custom plan combining FUGC concepts with MMTk infrastructure
pub struct RustVMPlan<VM: VMBinding> {
    common: CommonPlan<VM>,
    mark_compact: MarkCompact<VM>, // Non-moving as per FUGC design
    barrier: DijkstraBarrier<VM>,
    // FUGC-style generational if needed later
}
```

**VM Binding Layer Design:**

- **Object Model**: Define how VM objects map to MMTk's expectations
- **Root Scanning**: Integrate with thread registry for stack/global roots
- **Allocation Sites**: Hook MMTk allocators into VM allocation points

### Hybrid Memory Management

**MMTk + jemalloc Strategy:**

- MMTk handles GC heap (managed objects)
- jemalloc for VM infrastructure (bytecode, JIT code, metadata)
- Simpler FFI boundary than libpas

**Alternative libpas Integration** (for advanced workloads):

- MMTk delegates to libpas for large object spaces
- Better for mixed workloads with varying allocation patterns

### Safepoint Integration

```rust
// At bytecode dispatch/loop headers
fn execute_bytecode() {
    loop {
        if unlikely(safepoint_requested()) {
            vm_safepoint(); // Includes GC poll
        }
        // Execute instruction
    }
}

// At allocation sites (handled by MMTk)
fn allocate<T>() -> Gc<T> {
    // MMTk handles safepoint polling internally
    mmtk_alloc(size_of::<T>())
}
```

### Gc<T> API Design for VM

```rust
// VM-specific wrapper over MMTk's ObjectReference
pub struct Gc<T> {
    ptr: ObjectReference, // MMTk's object reference
    _phantom: PhantomData<T>,
}

// Barrier integration (MMTk handles the heavy lifting)
impl<T> Gc<T> {
    pub fn write(&self, field: &mut Gc<U>, value: Gc<U>) {
        // MMTk's write barrier
        mmtk::memory_manager::object_reference_write(
            self.ptr, field as *mut _ as Address, value.ptr
        );
    }
}
```

### Lock-Free Handshake Protocol (Core Innovation)

**Critical Achievement:** The project implements a deadlock-impossible handshake protocol using:

- **Atomic State Machines**: 4-state protocol (Running → RequestReceived → AtSafepoint → Completed)
- **Crossbeam Channels**: Bounded request channels, unbounded completion channels
- **Type Safety**: Invalid states are unrepresentable by design
- **No Locks**: Uses `AtomicU8` and `compare_exchange_weak` for coordination

```rust
// Key files:
// src/handshake.rs - Complete lock-free handshake implementation
// src/thread.rs - MutatorThread and ThreadRegistry with handshake integration

// Safepoint polling is guaranteed deadlock-free:
mutator.poll_safepoint(); // Never blocks, never deadlocks
```

### FUGC 8-Step Protocol Implementation

The project implements the complete FUGC 8-step concurrent collection protocol:

**Protocol Sequencing (Critical Order):**

1. **Step 1 - Idle State & Trigger**:

   - Coordinator waits in `FugcPhase::Idle`
   - External trigger via `coordinator.trigger_gc()` initiates collection

2. **Step 2 - Write Barrier Activation**:

   - Enable Dijkstra write barriers BEFORE any marking
   - `coordinator.write_barrier().activate()` called first
   - Prevents missed objects during concurrent marking

3. **Step 3 - Black Allocation**:

   - Switch to black allocation mode during marking
   - `coordinator.black_allocator().activate()` ensures new objects marked black
   - Maintains tricolor invariant for concurrent allocation

4. **Step 4 - Global Root Marking**:

   - Mark all global roots (static variables, VM globals) as grey
   - Uses `global_roots.lock().unwrap()` for thread-safe access
   - Seeds the marking work queues

5. **Step 5 - Stack Scanning**:

   - Perform soft handshakes with mutator threads
   - Cooperative stack scanning without stop-the-world
   - `mutator.poll_safepoint()` enables handshake coordination

6. **Step 6 - Tracing Termination**:

   - Complete concurrent marking phase
   - Ensure tricolor invariant satisfied (no white objects reachable from black)
   - Work-stealing coordination terminates marking

7. **Step 7 - Barrier Deactivation**:

   - Disable write barriers after marking complete
   - `coordinator.write_barrier().deactivate()` called
   - Prepare for sweep phase

8. **Step 8 - Page-Based Sweep**:
   - Sweep unmarked objects and update page allocation colors
   - `coordinator.page_allocation_color(page_index)` manages page state
   - Return to idle phase

**Required Coordinator APIs:**

The `FugcCoordinator` must expose these APIs for proper integration:

```rust
// Collection Control APIs
pub fn trigger_gc(&self);                           // Initiate 8-step protocol
pub fn wait_until_idle(&self, timeout: Duration) -> bool;  // Block until complete
pub fn current_phase(&self) -> FugcPhase;           // Get current step
pub fn is_collecting(&self) -> bool;                // Check if active

// Component Access APIs
pub fn write_barrier(&self) -> &WriteBarrier;       // Access write barrier
pub fn black_allocator(&self) -> &BlackAllocator;   // Access black allocator
pub fn tricolor_marking(&self) -> &TricolorMarking; // Access marking state

// Statistics APIs
pub fn get_cycle_stats(&self) -> FugcCycleStats;    // Collection metrics
pub fn page_allocation_color(&self, page: usize) -> AllocationColor;  // Page state

// Soft Handshake APIs (Internal)
pub fn request_handshake(&self, thread_id: usize);  // Request cooperation
pub fn complete_handshake(&self, thread_id: usize); // Signal completion
```

**Testing Infrastructure Requirements:**

For realistic protocol testing, the test suite requires:

1. **Background Mutator Simulation**:

   ```rust
   // Critical: mutators must poll safepoints for handshake realism
   fn spawn_mutator(mutator: MutatorThread) -> (JoinHandle<()>, Arc<AtomicBool>) {
       let handle = thread::spawn(move || {
           while running.load(Ordering::Relaxed) {
               worker.poll_safepoint();  // Enables soft handshakes
               //(Duration::from_millis(1));
           }
       });
   }
   ```

2. **Phase Manager Helpers**:

   ```rust
   // Test utilities for protocol validation
   impl FugcCoordinator {
       pub fn advance_to_phase(&self, target: FugcPhase) -> bool;
       pub fn wait_for_phase_transition(&self, from: FugcPhase, to: FugcPhase) -> bool;
   }
   ```

3. **Proper Root Registration**:
   ```rust
   // Tests must use proper global root registration, not manual color setting
   {
       let mut roots = global_roots.lock().unwrap();
       roots.register(object_address as *mut u8);  // Correct approach
   }
   // Avoid: tricolor.set_color(obj, Black);  // Bypasses protocol
   ```

### FUGC-Specific Adaptations

**Incremental Stack Scanning:**

- Use MMTk's concurrent marking with custom stack scanning
- Implement soft handshakes in VM's thread management
- Cooperative scanning via `poll_safepoint()` integration

**Parallel Marking:**

- MMTk provides parallel GC workers out of the box
- Configure work-stealing for object graph traversal
- FUGC coordinator manages work distribution and termination

## Testing Strategy

The test suite uses feature flags to organize different test categories:

- **smoke**: Lightweight tests for validating high-level GC semantics and infrastructure
- **segment_scan_linux**: Linux-specific segment scanning tests
- **legacy_tests**: Backward compatibility tests

Tests demonstrate key FUGC properties including handshake mechanisms, safepoint infrastructure, and concurrent collection phases.

### FUGC Protocol Testing Requirements

**Integration Testing Best Practices:**

1. **Protocol Step Validation**: Each test should validate specific protocol steps, not just overall functionality

   ```rust
   #[test]
   fn step_2_write_barrier_activation() {
       let coordinator = setup_coordinator();
       let write_barrier = coordinator.write_barrier();

       assert!(!write_barrier.is_active());  // Initially inactive
       coordinator.trigger_gc();
       // Validate Step 2: barriers activated before marking
       // Note: timing-dependent, may require phase introspection
   }
   ```

2. **Realistic Mutator Behavior**: All tests with handshakes must spawn background mutator threads

   ```rust
   // Required pattern for Step 5 (stack scanning) tests
   let (handle, running) = spawn_mutator(mutator.clone());
   coordinator.trigger_gc();
   // Test handshake coordination
   running.store(false, Ordering::Relaxed);
   handle.join().unwrap();
   ```

3. **Proper Root Management**: Tests must use coordinator's root registration, not bypass mechanisms

   ```rust
   // Correct: Register through global roots
   {
       let mut roots = global_roots.lock().unwrap();
       roots.register(object_address as *mut u8);
   }

   // Incorrect: Manual color manipulation bypasses protocol
   // tricolor.set_color(obj, ObjectColor::Black);  // Don't do this
   ```

4. **Plan Manager Integration**: Tests should validate both direct coordinator access and plan manager delegation

   ```rust
   // Test both access patterns
   let plan_manager = FugcPlanManager::new();
   let coordinator = plan_manager.get_fugc_coordinator();

   // Via plan manager
   plan_manager.gc();
   assert!(plan_manager.is_fugc_collecting());

   // Via coordinator
   coordinator.trigger_gc();
   assert_eq!(coordinator.current_phase(), FugcPhase::some_active_phase);
   ```

**Test Infrastructure Components:**

- `spawn_mutator()`: Background thread simulation with safepoint polling
- `FugcCoordinator::wait_until_idle()`: Synchronization for test determinism
- `ThreadRegistry`: Thread lifecycle management for handshake testing
- `GlobalRoots`: Thread-safe root registration for Step 4 validation

## Module Architecture

**Core Modules:**

- `handshake` - Lock-free coordination protocol (deadlock-impossible)
- `thread` - MutatorThread and ThreadRegistry with handshake integration
- `fugc_coordinator` - FUGC 8-step protocol coordinator
- `safepoint` - Pollcheck infrastructure and thread synchronization
- `concurrent` - Tricolor marking, write barriers, black allocation
- `binding` - MMTk VM binding layer (RustVM implementation)

**Memory Management:**

- `allocator` - MMTk allocator interface and stub implementations
- `memory_management` - Finalizers, weak references, free object management
- `simd_sweep` - SIMD-optimized bitvector sweeping algorithms
- `cache_optimization` - Cache-friendly allocation and marking strategies

**Supporting Infrastructure:**

- `roots` - Global and stack root scanning with thread integration
- `plan` - FugcPlanManager coordinating MMTk plans
- `weak` - Weak reference headers and registry
- `error` - GC-specific error types and result handling
- `pollcheck_macros` - Compiler integration macros for safepoint insertion

## Known Issues & Next Steps

**Current Failing Tests (7/259):**

1. **Fast Failures (0.008-0.016s):** API integration mismatches

   - `test_rust_scanning` - Root scanning API needs handshake integration
   - `coordinator_state_sharing_works` - Stats collection integration
   - `step_5_stack_scanning_mutator_roots` - Stack roots API compatibility

2. **Timeout Failures (0.5-1.0s):** Coordinator integration incomplete
   - `complete_fugc_8_step_protocol` - `wait_until_idle()` not wired to handshake protocol
   - `fugc_concurrent_collection_stress` - Same integration issue
   - `fugc_statistics_accuracy` - Stats collection during handshakes
   - `step_2_write_barrier_activation` - Barrier activation timing

**Priority Fix:** Wire `FugcCoordinator::wait_until_idle()` to the new lock-free handshake completion signals.

## Key Dependencies

- `mmtk` - Memory Management Toolkit integration
- `crossbeam` - Lock-free channels and epoch-based memory reclamation
- `parking_lot` - Fast mutex implementations for rare coordination
- `rayon` - Parallel work-stealing for marking phase
- `psm` & `stacker` - Stack manipulation for root scanning
- `bitflags` - Object header flag management
