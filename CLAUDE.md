# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust implementation of a concurrent, non-moving garbage collector inspired by Epic Games' FUGC (Fil's Unbelievable Garbage Collector) from the Verse programming language. The project aims to bring innovative GC features like free singleton redirection and advanced weak reference handling to Rust while maintaining compatibility with the borrow checker.

## Build & Development Commands

```bash
# Build the project
cargo build

# Run tests
cargo test

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

The garbage collector consists of several key components:

- **Core Smart Pointers**: `Gc<T>` for managed objects, with atomic redirection to `FREE_SINGLETON` for dead objects
- **Memory Management**: `SegmentedHeap` provides thread-local allocation with fixed-size segments
- **Collection Phases**: Concurrent marking, censusing (weak references), reviving (finalizers), and sweeping
- **Thread Coordination**: `CollectorState` manages GC phases and worker threads, `MutatorState` for thread-local operations
- **Object Classification**: Different object types (`Default`, `Destructor`, `Census`, `Finalizer`) allocated in distinct heaps

Key implementation details:

- Non-moving design ensures pointer stability for FFI compatibility
- Free singleton redirection prevents use-after-free by atomically redirecting dead object pointers
- Handshake mechanism for soft synchronization without full stop-the-world pauses
- Fork() safety through GC suspension mechanisms

## Module Structure

Current modules in `src/`:

- `segmented_heap.rs` - Low-level memory management
- `collector_phase.rs` - GC phase state machine
- `gc_allocator.rs` - Allocation interface
- `collector_state.rs` - Central GC coordination
- `type_info.rs` - Type metadata and vtables
- `free_singleton.rs` - Dead object redirection
- `object_class.rs` - Object categorization
- `weak.rs` - Weak reference implementation
- `finalizable.rs` - Finalizer support
- `sweeping_phase.rs` - Memory reclamation
- `suspend_for_fork.rs` - Fork safety mechanisms

## Implementation Notes

The project is in early development stages - the main lib.rs currently contains only placeholder code. The actual GC implementation will need to:

1. Implement the `GcTrace` trait (likely via derive macro) for tracing object graphs
2. Provide RAII guards (`GcRef<T>`, `GcRefMut<T>`) for borrow checker compliance
3. Ensure thread-safety through atomic operations and proper synchronization
4. Handle root registration for global/static GC pointers
