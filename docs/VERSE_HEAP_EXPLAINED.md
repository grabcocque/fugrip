# Verse Heap Implementation Overview

This document explains the high-level design and functionality of the Verse heap implementation found in `src/c/verse_heap.h`.

## What is the Verse Heap?

The Verse heap is a garbage-collected memory management system originally developed by Epic Games for the Verse programming language. It's built on top of the PAS (Parallel Allocation System) framework and implements a concurrent, mark-and-sweep garbage collector with several advanced features.

## Key Design Principles

### 1. Concurrent Garbage Collection
- **Non-blocking allocation**: Mutator threads can allocate while GC is running
- **Parallel marking**: Multiple threads can mark objects simultaneously
- **Incremental sweeping**: Collection work is distributed over time

### 2. Mark Bit Management
The heap uses a sophisticated mark bit system:
- **Chunk-based organization**: Memory is divided into chunks of `VERSE_HEAP_CHUNK_SIZE`
- **Bitvector storage**: Mark bits are stored as bitvectors at the start of each chunk
- **Atomic operations**: Thread-safe mark bit manipulation using atomic compare-and-swap

### 3. Memory Layout
```
[Chunk Start]
├── Mark bits bitvector
├── Object data...
└── [Next chunk]
```

## Core Components

### Mark Bit Operations
```c
// Fast path for checking if an object is marked
bool verse_heap_is_marked(void* object)

// Atomic mark bit setting with full memory barriers
bool verse_heap_set_is_marked(void* object, bool value)

// Relaxed atomic operations for performance-critical paths
bool verse_heap_set_is_marked_relaxed(void* object, bool value)
```

### Memory Accounting
The heap tracks live bytes and provides callbacks when thresholds are exceeded:
- `verse_heap_notify_allocation()` - Updates live byte count on allocation
- `verse_heap_notify_deallocation()` - Updates live byte count on deallocation
- `verse_heap_notify_sweep()` - Tracks bytes reclaimed during sweep

### GC State Management
- **Version tracking**: `verse_heap_latest_version` for GC cycle coordination
- **Allocation color**: `verse_heap_allocating_black_version` for tricolor marking
- **Sweep state**: `verse_heap_is_sweeping` flag for concurrent coordination

## Advanced Features

### 1. Iteration Support
The heap provides safe iteration over live objects during concurrent collection:
- **Iteration state**: Versioned snapshots for consistent iteration
- **Large object tracking**: Special handling for objects larger than page size
- **Thread-safe enumeration**: Lock-free iteration coordination

### 2. Page Cache System
- **Clean page management**: `verse_heap_page_cache` for zeroed pages
- **Demand paging support**: Designed to work with virtual memory systems
- **Large object headers**: Special metadata for oversized allocations

### 3. Thread-Local Caching
- **Layout nodes**: Per-thread allocation fast paths
- **Cache coordination**: Integration with global GC state
- **Allocator counts**: Statistics tracking per allocation site

## Memory Safety Guarantees

### Tricolor Marking Invariant
The implementation maintains the tricolor marking invariant:
- **White objects**: Unreachable (will be collected)
- **Grey objects**: Reachable but not yet scanned
- **Black objects**: Reachable and fully scanned

### Atomic Memory Barriers
- **Full barriers**: Used for conservative marking and allocation coordination
- **Relaxed atomics**: Used in performance-critical parallel marking
- **Memory ordering**: Ensures mark bit updates are visible across threads

## Integration Points

### Object Model Requirements
Objects in the Verse heap must:
- Be aligned to `VERSE_HEAP_MIN_ALIGN` boundaries
- Have discoverable size information
- Support conservative pointer scanning

### Allocation Interface
```c
// Core allocation functions (implementation in other files)
size_t verse_heap_get_size(pas_heap* heap);
size_t verse_heap_get_alignment(pas_heap* heap);
bool verse_heap_object_is_allocated(void* ptr);
```

## Performance Characteristics

### Strengths
- **Low allocation overhead**: Fast-path allocation with minimal synchronization
- **Scalable marking**: Parallel mark bit operations with minimal contention
- **Cache-friendly layout**: Mark bits colocated with object data
- **Incremental collection**: Bounded pause times through incremental work

### Design Trade-offs
- **No deallocation**: Objects can only be freed during GC sweep
- **Memory overhead**: Mark bits require additional space per chunk
- **Complexity**: Concurrent coordination adds implementation complexity

## Usage Context

This implementation is part of a larger garbage collection system and requires:
- **PAS framework**: Low-level memory management primitives
- **Heap configuration**: Size classes and allocation policies
- **Root scanning**: Stack and global variable enumeration
- **Write barriers**: Mutation tracking for concurrent collection

The code shows evidence of production use in a high-performance language runtime, with careful attention to memory barriers, atomic operations, and concurrent safety.
