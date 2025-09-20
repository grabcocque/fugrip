# Fugrip: FUGC Implementation in Rust

**Fugrip** is a Rust implementation of FUGC (Fil's Unbelievable Garbage Collector), a sophisticated parallel concurrent on-the-fly grey-stack Dijkstra accurate non-moving garbage collector originally designed for the Verse programming language.

## Current Status: MMTk Blackwall Migration

🚧 **Active Development**: The project is undergoing a strategic refactoring to push all MMTk types behind an impermeable "blackwall" of opaque handles with zero-cost abstractions.

### Key Features (Target Architecture)
- **Zero-Cost Opaque Abstraction**: All MMTk types hidden behind opaque handles
- **Swappable Backends**: jemalloc, MMTk, or custom allocators via compile-time dispatch
- **Perfect Type Safety**: MMTk types never escape to external code
- **Deadlock-Free Handshakes**: Lock-free coordination protocol
- **FUGC 8-Step Protocol**: Complete concurrent collection implementation

## MMTk Blackwall Migration Strategy

### Problem Statement
Legacy code mixed MMTk types (`Address`, `ObjectReference`) with opaque handles, creating:
- ❌ Type leakage defeating abstraction benefits
- ❌ Build failures from incompatible type systems
- ❌ Impossible backend swapping (MMTk types everywhere)
- ❌ Complex migration path for external users

### Solution: Bottom-Up Migration
**Strategy**: Migrate modules from foundation upward, creating a forcing function where external code must use opaque handles.

#### Migration Layers (Bottom-Up Order)

**Layer 1: Foundation (Start Here)**
1. ✅ **`alloc_facade.rs`** - Core opaque API ⭐ **FIRST TARGET**
2. 🔄 **`types.rs`** - Custom types for non-MMTk backend
3. 🔄 **`core.rs`** - Object headers and basic types

**Layer 2: Core Services**
4. ⏳ **`test_utils.rs`** - Testing infrastructure
5. ⏳ **`debug_test.rs`** - Debugging utilities

**Layer 3: Allocation & Memory**
6. ⏳ **`allocator.rs`** - Main allocation interface
7. ⏳ **`facade_allocator.rs`** - Facade-based allocator
8. ⏳ **`modern_allocator.rs`** - Modern allocation interface

**Layer 4: GC Coordination**
9. ⏳ **`concurrent/`** modules - Marking, barriers, tricolor
10. ⏳ **`fugc_coordinator/`** - FUGC protocol coordination
11. ⏳ **`memory_management/`** - Finalizers, weak refs, free objects

**Layer 5: High-Level Features**
12. ⏳ **`safepoint/`** - Safepoint management
13. ⏳ **`plan.rs`** - Plan management
14. ⏳ **`binding/`** - MMTk binding layer

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

## Build Commands

```bash
# Build with jemalloc backend (opaque handles only)
cargo build --no-default-features --features use_jemalloc

# Build with MMTk backend (legacy + opaque)
cargo build --features use_mmtk

# Test opaque abstraction
cargo run --example pure_opaque_demo --features use_jemalloc

# Run tests
cargo nextest run --features smoke
```

## Zero-Cost Verification

The opaque handle system guarantees zero runtime overhead:

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

## Architecture Goals

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
use crate::alloc_facade::{MutatorHandle, allocate};

fn allocate_object(mutator: MutatorHandle) -> *mut u8 {
    // Handle-based allocation - swappable backends
}
```

## Key Benefits

1. **Zero Runtime Overhead**: All dispatch at compile-time
2. **Perfect Abstraction**: MMTk completely hidden behind blackwall
3. **Swappable Backends**: Easy to switch allocators without code changes
4. **Type Safety**: Invalid states unrepresentable
5. **Clean APIs**: Simple handle-based interface
6. **Future-Proof**: Easy to extend without breaking changes

## Contributing

When working on migration:

1. **Never import `compat` types** in new code
2. **Use opaque handles exclusively** (`MutatorHandle`, `PlanHandle`)
3. **Test both backends** (`use_mmtk` and `use_jemalloc`)
4. **Verify zero-cost** with size assertions
5. **Follow bottom-up migration order**

See `MMTK_BLACKWALL_ROADMAP.md` for detailed migration instructions.

## Documentation

- **Architecture**: See `CLAUDE.md` for detailed technical documentation
- **Migration**: See `MMTK_BLACKWALL_ROADMAP.md` for step-by-step migration guide
- **FUGC Protocol**: 8-step concurrent collection algorithm documentation
- **Testing**: Feature flag organization and protocol validation