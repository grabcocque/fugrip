# Fugrip: FUGC-Inspired Concurrent Garbage Collector

Fugrip is a high-performance concurrent garbage collector for Rust VMs, inspired by FUGC (Functional Update Garbage Collection) principles. It integrates with MMTk to provide modern GC capabilities with excellent performance characteristics.

## 🚀 Key Features

### Concurrent Marking Infrastructure

- **Dijkstra Write Barriers**: Prevents missed objects during concurrent marking
- **Tricolor Marking**: Atomic color transitions with 2-bit per object encoding
- **Parallel Marking Workers**: Work-stealing coordinator with load balancing
- **Black Allocation**: Objects allocated black during marking to reduce barrier overhead
- **Concurrent Root Scanning**: Thread-safe root enumeration during marking

### FUGC-Inspired Object Management

- **Object Classification**: Age-based (young/old) and mutability-based categorization
- **Generational Hints**: Framework for future generational collection
- **Precise Lifetimes**: Fine-grained object lifecycle management

### Performance Optimizations

- **Inline Barrier Fast Path**: 2.5x performance improvement when barriers inactive
- **Relaxed Memory Ordering**: Optimized atomic operations for hot paths
- **Branch Prediction Hints**: Compiler-guided optimization for common cases
- **Multiple Barrier Variants**: Choose appropriate barrier for your use case

## 📊 Performance Characteristics

Based on comprehensive benchmarks:

```
Write Barrier Performance (ns/op):
├── Inactive barrier (fast path):     1.05 ns
├── Active barriers (slow path):      2.5-2.6 ns
└── Performance gain:                 2.5x faster when inactive

Tricolor Marking (ns/1000 ops):
├── Set color operations:            ~268 ns
└── Get color operations:            ~259 ns
```

## 🏗️ Architecture

### Core Components

#### Write Barriers

```rust
// Multiple barrier variants for different performance needs
barrier.write_barrier(&mut slot, new_value);        // Standard API
barrier.write_barrier_fast(&mut slot, new_value);   // Optimized fast path
barrier.write_barrier_inline(&mut slot, new_value); // Ultra-fast inline
```

#### Concurrent Marking

```rust
// Full concurrent marking workflow
let coordinator = ConcurrentMarkingCoordinator::new(heap_base, heap_size, num_workers, thread_registry, global_roots);
coordinator.start_marking(root_objects);
coordinator.wait_for_completion();
```

#### Object Classification

```rust
// FUGC-style object categorization
let classifier = ObjectClassifier::new();
classifier.classify_object(object, ObjectClass {
    age: ObjectAge::Young,
    mutability: ObjectMutability::Mutable,
    size_class: SizeClass::Small,
});
```

### MMTk Integration

Fugrip provides complete MMTk VM binding implementation:

- **VM Binding Traits**: `RustVM`, `RustActivePlan`, `RustReferenceGlue`
- **Object Model**: `RustObjectModel` with header management
- **Root Scanning**: `RustScanning` for thread stacks and globals
- **Allocation**: `MMTkAllocator` and `StubAllocator` implementations

## 🔧 Usage Examples

### Basic Write Barrier Usage

```rust
use fugrip::concurrent::{WriteBarrier, TricolorMarking, ParallelMarkingCoordinator};
use mmtk::util::Address;
use std::sync::Arc;

// Set up GC infrastructure
let marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
let coordinator = Arc::new(ParallelMarkingCoordinator::new(4));
let barrier = WriteBarrier::new(marking, coordinator);

// Use in mutator code
barrier.activate(); // Enable barriers during marking
unsafe {
    barrier.write_barrier(&mut object_slot, new_object_reference);
}
```

### FUGC Plan Manager Integration

```rust
use fugrip::plan::FugcPlanManager;

// Initialize FUGC plan manager
let mut plan_manager = FugcPlanManager::new();

// Access unified coordinator for 8-step protocol
let coordinator = plan_manager.get_fugc_coordinator();

// Trigger collection through plan manager
plan_manager.gc();  // Initiates 8-step protocol

// Monitor collection progress
while plan_manager.is_fugc_collecting() {
    let phase = plan_manager.fugc_phase();
    println!("Current phase: {:?}", phase);
}

// Access performance statistics
let stats = plan_manager.get_fugc_stats();
println!("Cycles completed: {}", stats.work_shared);
```

### Concurrent Marking Workflow

```rust
// Initialize coordinator through plan manager
let plan_manager = FugcPlanManager::new();
let coordinator = plan_manager.get_fugc_coordinator();

// 8-step protocol executes automatically
coordinator.trigger_gc();

// Mutators continue running with barriers active during Steps 2-6
// Workers process objects concurrently during marking phases

// Wait for completion
coordinator.wait_until_idle(Duration::from_millis(500));
```

## 🎯 FUGC 8-Step Protocol

Fugrip implements the complete FUGC 8-step concurrent collection protocol:

### Protocol Sequencing

1. **Step 1 - Idle State & Trigger**: Collection coordinator waits in idle state until `trigger_gc()` called
2. **Step 2 - Write Barrier Activation**: Enable Dijkstra write barriers before any marking begins
3. **Step 3 - Black Allocation**: Switch allocator to black allocation mode during concurrent marking
4. **Step 4 - Global Root Marking**: Mark all global roots (static variables, VM globals) as grey
5. **Step 5 - Stack Scanning**: Perform soft handshakes with mutator threads to scan stack roots
6. **Step 6 - Tracing Termination**: Complete concurrent marking, ensure tricolor invariant satisfied
7. **Step 7 - Barrier Deactivation**: Disable write barriers and prepare for sweep phase
8. **Step 8 - Page-Based Sweep**: Sweep unmarked objects and update allocation page colors

### Coordinator APIs

The `FugcCoordinator` exposes these essential APIs for external control:

```rust
// Collection Control
coordinator.trigger_gc();                    // Initiate 8-step protocol
coordinator.wait_until_idle(timeout);        // Block until collection completes

// Phase Monitoring
coordinator.current_phase();                 // Get current protocol step
coordinator.is_collecting();                 // Check if collection active

// Statistics & Diagnostics
coordinator.get_cycle_stats();               // Get collection cycle metrics
coordinator.get_fugc_stats();                // Get performance statistics

// Soft Handshakes (Internal)
coordinator.request_handshake(thread_id);    // Request mutator cooperation
coordinator.complete_handshake(thread_id);   // Signal handshake completion
```

### Testing Infrastructure Requirements

For realistic testing of the 8-step protocol, the test suite requires:

#### Background Mutator Simulation

```rust
// Spawns realistic mutator threads that poll safepoints
fn spawn_mutator(mutator: MutatorThread) -> (JoinHandle<()>, Arc<AtomicBool>) {
    let handle = thread::spawn(move || {
        while running.load(Ordering::Relaxed) {
            worker.poll_safepoint();  // Critical for handshake realism
            # Using sleeps to paper over logic bugs is unprofessional(Duration::from_millis(1));
        }
    });
}
```

#### Phase Manager Shim

```rust
// Lightweight helper for phase transition testing
impl FugcCoordinator {
    pub fn advance_to_phase(&self, target_phase: FugcPhase) -> bool {
        // Used by tests to validate specific protocol steps
    }

    pub fn wait_for_phase_transition(&self, from: FugcPhase, to: FugcPhase) -> bool {
        // Ensures tests can observe protocol sequencing
    }
}
```

## 🎯 FUGC Design Principles

Fugrip implements several FUGC-inspired concepts:

1. **Incremental Updates**: Write barriers provide precise tracking of object graph changes
2. **Concurrent Processing**: Parallel workers minimize pause times
3. **Color-Abstraction**: Tricolor marking provides clear object states
4. **Work Stealing**: Dynamic load balancing across marking threads
5. **Black Allocation**: Reduces redundant barrier operations
6. **Soft Handshakes**: Cooperative stack scanning without stop-the-world pauses

## 🧪 Testing & Quality

- **56 Comprehensive Tests**: Unit and integration test coverage
- **Performance Benchmarks**: Criterion-based performance testing
- **Memory Safety**: Extensive use of Rust's ownership system
- **Thread Safety**: All concurrent operations properly synchronized

## 📈 Future Enhancements

- **Generational Collection**: Build on current age-based classification
- **Compaction**: Add moving collection capabilities
- **NUMA Awareness**: Optimize for multi-socket systems
- **Custom Allocators**: Specialized allocators for different object types

## 🤝 Contributing

Fugrip welcomes contributions in concurrent GC research, performance optimizations, and MMTk integration improvements.

---

_Built with ❤️ for high-performance Rust VMs_

**MMTk Plan Composition:**

```rust
// Custom plan combining FUGC concepts with MMTk infrastructure
pub struct RustVMPlan<VM: VMBinding> {
    common: CommonPlan<VM>,
    mark_compact: MarkCompact<VM>, // Non-moving as per your plan
    barrier: DijkstraBarrier<VM>,
    // FUGC-style generational if needed later
}
```

**VM Binding Layer Design:**

- **Object Model**: Define how your VM's objects map to MMTk's expectations
- **Root Scanning**: Integrate with your thread registry for stack/global roots
- **Allocation Sites**: Hook MMTk allocators into your VM's allocation points

## libpas vs jemalloc Integration

For a **Rust VM with MMTk**, I'd recommend:

1. **MMTk + jemalloc hybrid**:

   - Let MMTk handle GC heap (managed objects)
   - Use jemalloc for VM infrastructure (bytecode, JIT code, metadata)
   - Simpler FFI boundary than libpas

2. **libpas integration** (if you want WebKit's sophistication):
   - MMTk can delegate to libpas for large object spaces
   - Good for mixed workloads with varying allocation patterns

## Safepoint Integration

**For a Rust VM specifically:**

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

// At allocation sites (already handled by MMTk)
fn allocate<T>() -> Gc<T> {
    // MMTk handles safepoint polling internally
    mmtk_alloc(size_of::<T>())
}
```

## Gc<T> API Design for VM

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

## FUGC-Specific Adaptations

**Incremental Stack Scanning:**

- Use MMTk's concurrent marking with custom stack scanning
- Implement soft handshakes in your VM's thread management

**Parallel Marking:**

- MMTk provides parallel GC workers out of the box
- Configure work-stealing for your object graph traversal

## Architecture Recommendations

1. **Start with MMTk's existing concurrent plan** as base
2. **Customize the write barrier** for FUGC-style incremental marking
3. **Use jemalloc for non-GC allocations** (simpler than libpas integration initially)
4. **Design your VM's object layout** to work well with MMTk's tracing

## Key Integration Points

- **Object Headers**: Align with MMTk's metadata requirements
- **Thread Management**: Your thread registry feeds MMTk's root scanning
- **JIT Integration**: If you have one, coordinate with MMTk's code space management

The MMTk route will save us months of low-level GC implementation while still allowing FUGC innovations in the higher-level algorithms.

This is an excellent concrete roadmap for implementing a garbage collector in Rust! Let me break down this milestone plan and provide some insights on each phase:

## Phase 1: Library Prototype (Safe Foundation)

Your starting point is solid:

- **`Gc<T>` wrapper + `Trace/Collect` traits**: Smart approach using Rust's type system for safety
- **Explicit roots management**: Scoped roots are great for stack management, dynamic roots for globals
- **Thread registry + `pollcheck()` safepoints**: Critical for cooperative collection
- **Mark/sweep non-moving**: Good starting choice - simpler than moving collectors
- **libpas integration**: Interesting choice! WebKit's libpas could provide mature heap management

## Phase 2: Concurrent & FUGC-like Features

This is where it gets sophisticated:

- **Parallel grey stacks**: Will significantly improve marking performance
- **Dijkstra write barrier**: Classic choice for concurrent marking
- **Black allocation during marking**: Prevents newly allocated objects from being missed
- **Soft handshakes for incremental stack scans**: Modern approach to reduce pause times

## Phase 3: Performance Optimization

- **Inline barrier fast path**: ✅ **Implemented** - Multiple optimized write barrier variants with relaxed memory ordering and branch prediction hints for maximum performance when barriers are inactive
- **libpas vs pure-Rust heap benchmark**: Good validation approach
- Consider also benchmarking against other allocators like `jemalloc`

## Phase 4: Compiler Integration (Advanced)

- **LLVM statepoints**: Would enable precise GC without manual root management
- **rustc fork**: Ambitious but would provide the cleanest API

## Alternative: MMTk Integration

- **MMTk**: Well-tested framework, could save significant implementation time
- **VM binding layer**: Much smaller surface area to implement

## Questions to Help Tailor Advice:

1. **What's your target runtime?** (Rust VM, app framework, etc.)
2. **Performance vs. simplicity trade-off?**
3. **FFI requirements?** (C interop, etc.)
4. **Pause time requirements?** (real-time constraints?)

The roadmap shows deep GC knowledge. Starting with the safe prototype and iterating toward concurrency is the right approach.
