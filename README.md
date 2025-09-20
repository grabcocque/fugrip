# Fugrip: FUGC Implementation in Rust

**Fugrip** is a Rust implementation of FUGC (Fil's Unbelievable Garbage Collector), a sophisticated parallel concurrent on-the-fly grey-stack Dijkstra accurate non-moving garbage collector originally designed for the Verse programming language.

## What is FUGC?

FUGC is a state-of-the-art garbage collection algorithm that combines several advanced techniques:

- **Parallel**: Marking and sweeping across multiple threads for maximum throughput
- **Concurrent**: Collection happens on dedicated threads while mutators continue running
- **On-the-fly**: Uses "soft handshakes" instead of global stop-the-world pauses
- **Grey-stack**: Rescans thread stacks to fixpoint, eliminating load barriers
- **Dijkstra**: Simple store barrier with compare-and-swap on slow path
- **Accurate**: Precise pointer tracking via compiler integration
- **Non-moving**: Objects don't move, simplifying concurrency
- **Advancing wavefront**: Mutators cannot create new work during collection

## Key Features

### Zero-Cost Opaque Abstraction

- All MMTk types hidden behind opaque handles (just `usize` values)
- Swappable backends (jemalloc, MMTk, or custom allocators) via compile-time dispatch
- Perfect type safety - MMTk types never escape to external code
- Zero runtime overhead - all dispatch happens at compile time

### FUGC 8-Step Protocol

The complete FUGC concurrent collection protocol:

1. Idle State & Trigger
2. Write Barrier Activation
3. Black Allocation
4. Global Root Marking
5. Stack Scanning
6. Tracing Termination
7. Barrier Deactivation
8. Page-Based Sweep

### Deadlock-Free Handshakes

- Lock-free coordination protocol using atomic state machines
- Type-safe design where invalid states are unrepresentable
- Crossbeam channels for efficient thread communication

### SIMD-Optimized Operations

- High-performance bitvector sweeping using SIMD instructions
- Parallel marking with work-stealing
- Cache-friendly memory layouts

## Current Status: MMTk Blackwall Migration

🚧 **Active Development**: The project is undergoing a strategic refactoring to push all MMTk types behind an impermeable "blackwall" of opaque handles with zero-cost abstractions.

### Migration Strategy

We're migrating modules from foundation upward, creating a forcing function where external code must use opaque handles:

**Layer 1: Foundation (Completed)**

- ✅ **`alloc_facade.rs`** - Core opaque API
- ✅ **`types.rs`** - Custom types for non-MMTk backend
- ✅ **`core.rs`** - Object headers and basic types

**Layer 2: Core Services (In Progress)**

- 🔄 **`test_utils.rs`** - Testing infrastructure
- 🔄 **`debug_test.rs`** - Debugging utilities

**Layer 3: Allocation & Memory (Planned)**

- ⏳ **`allocator.rs`** - Main allocation interface
- ⏳ **`facade_allocator.rs`** - Facade-based allocator
- ⏳ **`modern_allocator.rs`** - Modern allocation interface

**Layer 4: GC Coordination (Planned)**

- ⏳ **`concurrent/`** modules - Marking, barriers, tricolor
- ⏳ **`fugc_coordinator/`** - FUGC protocol coordination
- ⏳ **`memory_management/`** - Finalizers, weak refs, free objects

**Layer 5: High-Level Features (Planned)**

- ⏳ **`safepoint/`** - Safepoint management
- ⏳ **`plan.rs`** - Plan management
- ⏳ **`binding/`** - MMTk binding layer

### Current Progress

**✅ Completed:**

- Opaque handle types (`MutatorHandle`, `PlanHandle`)
- Feature flag infrastructure (`use_mmtk` vs `use_jemalloc`)
- Basic allocation facade with handle registry
- Zero-cost dispatch architecture
- Public API cleanup (no MMTk re-exports in `lib.rs`)

**🔄 In Progress:**

- `alloc_facade.rs` core functionality completion
- Pure opaque example demonstration
- `compat.rs` elimination planning

**📋 Next Steps:**

1. Complete `alloc_facade.rs` - remove all `compat` imports
2. Create working `examples/pure_opaque_demo.rs`
3. Verify zero-cost: `assert_eq!(size_of::<MutatorHandle>(), size_of::<usize>())`
4. Begin Layer 2 migration (`test_utils.rs`)

## Getting Started

### Prerequisites

- Rust 1.70+ (stable)
- For MMTk backend: MMTk dependencies (see `Cargo.toml`)

### Building

```bash
# Build with jemalloc backend (opaque handles only)
cargo build --no-default-features --features use_jemalloc

# Build with MMTk backend (legacy + opaque)
cargo build --features use_mmtk

# Build with stub backend for testing
cargo build --features use_stub
```

### Running Examples

```bash
# Run pure opaque demonstration
cargo run --example pure_opaque_demo --features use_jemalloc

# Run zero-cost verification
cargo run --example zero_cost_verification --features use_jemalloc

# Run minimal opaque demo
cargo run --example minimal_opaque_demo --features use_jemalloc
```

### Testing

```bash
# Run all tests
cargo nextest run

# Run specific test categories
cargo nextest run --features smoke         # Lightweight GC semantics validation
cargo nextest run --features stress-tests  # Expensive stress tests

# Run benchmarks
cargo bench --bench comprehensive_gc_benchmarks
```

## Architecture

### Zero-Cost Abstraction

Our opaque handle system guarantees zero runtime overhead:

```rust
// Handles are just numbers - perfect zero-cost abstraction
assert_eq!(size_of::<MutatorHandle>(), size_of::<usize>());
assert_eq!(size_of::<PlanHandle>(), size_of::<usize>());

// All dispatch happens at compile-time via feature flags
#[cfg(feature = "use_mmtk")]
fn allocate_impl(handle: MutatorHandle) -> *mut u8 { /* MMTk path */ }

#[cfg(not(feature = "use_mmtk"))]
fn allocate_impl(handle: MutatorHandle) -> *mut u8 { /* jemalloc path */ }
```

### Before (Type Leakage)

```rust
// ❌ MMTk types exposed everywhere
use mmtk::util::{Address, ObjectReference};
use crate::compat::{Address, ObjectReference};

fn allocate(mutator: &Mutator<RustVM>) -> ObjectReference {
    // Direct MMTk usage - not swappable
}
```

### After (Opaque Blackwall)

```rust
// ✅ Only opaque handles exposed
use crate::frontend::alloc_facade::{MutatorHandle, allocate};

fn allocate_object(mutator: MutatorHandle) -> *mut u8 {
    // Handle-based allocation - swappable backends
}
```

## Usage Examples

### Basic Allocation

```rust
use fugrip::frontend::alloc_facade::{allocate, init_facade, register_mutator, register_plan};
use fugrip::core::ObjectHeader;
use fugrip::frontend::alloc_facade::AllocationSemantics;

// Initialize the facade
init_facade();

// Create opaque handles
let mutator = register_mutator(std::thread::current().id().as_u64().get());
let plan = register_plan();

// Allocate an object
let header = ObjectHeader::default();
match allocate(mutator, 64, 8, 0, AllocationSemantics::Default) {
    Ok(ptr) => {
        println!("Allocated object at: {:p}", ptr);
        // Use the object...
    }
    Err(e) => {
        println!("Allocation failed: {:?}", e);
    }
}
```

### Using the Zero-Cost Allocator

```rust
use fugrip::zero_cost_allocator::ZeroCostAllocator;
use fugrip::core::ObjectHeader;

// Create a zero-cost allocator
let allocator = ZeroCostAllocator::new()?;

// Allocate an object
let header = ObjectHeader::default();
let obj_ptr = allocator.alloc_object(header, 128)?;

// Use write barrier
allocator.write_barrier(obj_ptr, slot_ptr, Some(target_ptr));

// Trigger GC
allocator.trigger_gc();
```

### FUGC Collection

```rust
use fugrip::plan::FugcPlanManager;
use std::time::Duration;

// Create a plan manager
let plan_manager = FugcPlanManager::new();

// Trigger FUGC collection
plan_manager.gc();

// Wait for completion
let coordinator = plan_manager.get_fugc_coordinator();
coordinator.wait_until_idle(Duration::from_secs(5));

// Get statistics
let stats = plan_manager.get_fugc_stats();
println!("Collection completed: {:?}", stats);
```

## Performance Characteristics

### Zero-Cost Guarantees

- **No runtime overhead**: All dispatch happens at compile time
- **No memory overhead**: Handles are exactly `usize` values
- **No indirection**: Direct calls to backend implementations
- **No vtables**: Pure monomorphization via feature flags

### FUGC Advantages

- **Low pause times**: Concurrent collection minimizes stop-the-world pauses
- **High throughput**: Parallel marking and sweeping
- **Scalability**: Work-stealing algorithms scale to many cores
- **Predictability**: Non-moving design eliminates evacuation pauses
- **Efficiency**: Grey-stack scanning eliminates load barriers

## Key Benefits

1. **Zero Runtime Overhead**: All dispatch at compile-time
2. **Perfect Abstraction**: MMTk completely hidden behind blackwall
3. **Swappable Backends**: Easy to switch allocators without code changes
4. **Type Safety**: Invalid states unrepresentable
5. **Clean APIs**: Simple handle-based interface
6. **Future-Proof**: Easy to extend without breaking changes

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Migration Guidelines

When working on the MMTk blackwall migration:

1. **Never import `compat` types** in new code
2. **Use opaque handles exclusively** (`MutatorHandle`, `PlanHandle`)
3. **Test both backends** (`use_mmtk` and `use_jemalloc`)
4. **Verify zero-cost** with size assertions
5. **Follow bottom-up migration order**

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Documentation

- **Architecture**: See `CLAUDE.md` for detailed technical documentation
- **Migration**: See `MMTK_BLACKWALL_ROADMAP.md` for step-by-step migration guide
- **FUGC Protocol**: 8-step concurrent collection algorithm documentation
- **Testing**: Feature flag organization and protocol validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FUGC algorithm design by Fil (Epic Games)
- MMTk (Memory Management Toolkit) for production GC infrastructure
- Rust community for excellent tools and libraries
