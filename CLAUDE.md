# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust implementation of a concurrent, non-moving garbage collector inspired by Epic Games' FUGC (Fil's Unbelievable Garbage Collector) from the Verse programming language. The project aims to bring innovative GC features to Rust, with potential integration paths for MMTk (Memory Management Toolkit) or pure-Rust implementations.

## Build & Development Commands

```bash
# Build the project
cargo build

# Run all tests
cargo test

# Run tests with smoke feature (lightweight GC semantics validation)
cargo test --features smoke

# Run a single test
cargo test test_name

# Check compilation without building
cargo check

# Format code
cargo fmt

# Run lints
cargo clippy
```

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

### FUGC-Specific Adaptations

**Incremental Stack Scanning:**
- Use MMTk's concurrent marking with custom stack scanning
- Implement soft handshakes in VM's thread management

**Parallel Marking:**
- MMTk provides parallel GC workers out of the box
- Configure work-stealing for object graph traversal

## Testing Strategy

The test suite uses feature flags to organize different test categories:

- **smoke**: Lightweight tests for validating high-level GC semantics and infrastructure
- **segment_scan_linux**: Linux-specific segment scanning tests
- **legacy_tests**: Backward compatibility tests

Tests demonstrate key FUGC properties including handshake mechanisms, safepoint infrastructure, and concurrent collection phases.

## Implementation Roadmap

### Architecture Recommendations

1. **Start with MMTk's existing concurrent plan** as base
2. **Customize the write barrier** for FUGC-style incremental marking
3. **Use jemalloc for non-GC allocations** (simpler than libpas integration initially)
4. **Design VM's object layout** to work well with MMTk's tracing

### Key Integration Points

- **Object Headers**: Align with MMTk's metadata requirements
- **Thread Management**: Thread registry feeds MMTk's root scanning
- **JIT Integration**: If present, coordinate with MMTk's code space management

### Development Phases

1. **Phase 1**: MMTk VM binding implementation
   - Define object model and layout
   - Implement root scanning hooks
   - Basic allocation integration

2. **Phase 2**: FUGC-specific features
   - Concurrent marking with custom barriers
   - Soft handshakes for incremental stack scanning
   - Parallel marking optimization

3. **Phase 3**: Performance tuning
   - Inline barrier fast paths
   - Work-stealing configuration
   - Benchmark against pure-Rust alternatives

4. **Phase 4**: Advanced features
   - FUGC-style generational collection
   - Advanced weak reference handling
   - Free singleton redirection

## Key Dependencies

- `crossbeam`: Lock-free data structures for concurrent collection
- `parking_lot`: Efficient synchronization primitives
- `rayon`: Parallel iteration for marking phase
- `psm` & `stacker`: Stack manipulation for root scanning
- `thread_local`: Thread-local storage for mutator state