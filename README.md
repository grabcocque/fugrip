# 🦀 Rust FUGC-Inspired Garbage Collector

A highly concurrent, non-moving garbage collector for Rust, heavily inspired by Epic Games' "Fil's Unbelievable Garbage Collector" (FUGC) from the Verse programming language. This implementation aims to bring FUGC's novel features, such as the "free singleton" redirection and advanced weak reference handling, into the Rust ecosystem while maintaining compatibility with the borrow checker's strict memory model.

## ✨ Features

This garbage collector provides a robust and innovative approach to memory management in Rust, with a strong focus on concurrency and performance:

- **Concurrent & Parallel Collection**: Marking, censusing (weak references), reviving (finalizers), and sweeping phases all run concurrently with mutator threads, utilizing multiple cores for maximum throughput.
- **Non-Moving GC**: Objects are never relocated in memory, which is beneficial for systems programming and FFI, as pointers remain stable.
- **Definitive Freeing with Free Singleton Redirection**: A core innovation from FUGC. Instead of deallocating dead objects, all `Gc<T>` pointers pointing to them are atomically redirected to a special "free singleton" object. This ensures that any subsequent access to a "freed" object will consistently point to the singleton, preventing use-after-free bugs and providing a definitive "null" state for dead objects without actually zeroing out the memory (which happens later during sweeping).
- **Sophisticated Weak Reference & Finalizer Handling**:
  - **Census Phases**: Dedicated phases for processing `Weak<T>` references, `WeakMap`s, and `ExactPtrTable`s to correctly determine reachability and nullify weak pointers whose targets are dead.
  - **Finalizer Revival**: Objects with finalizers are automatically "revived" (marked live) if they are found to be dead during the initial mark phase. They are then added to a global mark stack for a subsequent "remarking" phase to trace any objects they keep alive. After remarking, finalizers are executed, and then the object is allowed to be collected in the next cycle.
- **Object Classification and Multiple Heaps**: Objects can be classified (e.g., `Default`, `Destructor`, `Census`, `Finalizer`) and allocated into distinct `SegmentedHeap`s, allowing for specialized treatment and efficient iteration during different GC phases.
- **Fork() Safety**: The collector includes mechanisms to safely suspend all GC activities (collector thread and worker threads) before a `fork()` call and resume afterward, preventing memory corruption or deadlocks in the child process.
- **Borrow Checker Compatibility**: Provides RAII-based `GcRef<T>` and `GcRefMut<T>` guards that enforce Rust's borrowing rules, preventing concurrent mutable access during critical GC phases like marking.
- **Custom Segmented Allocator**: A highly concurrent, thread-local caching allocator built on top of fixed-size memory `Segment`s.
- **Atomic Operations**: Extensive use of `std::sync::atomic` for lock-free operations and efficient synchronization, especially in the hot paths of allocation and marking.

## 📐 Architecture Overview

The system is composed of several interacting components:

- **`Gc<T>`**: The primary smart pointer for garbage-collected objects. It encapsulates a raw pointer to `GcHeader<T>` and handles atomic redirection to the `FREE_SINGLETON` if the object is dead.
- **`GcHeader<T>`**: Stores GC metadata for each object, including `mark_bit`, `type_info`, and a `forwarding_ptr` used for redirection.
- **`SegmentedHeap`**: The low-level memory manager. It consists of multiple `Segment`s, each with its own allocation pointer and mark bits. Allocations are primarily thread-local for performance, falling back to global segment allocation with CAS.
- **`CollectorState`**: The central state machine for the GC. It manages the current `CollectorPhase` (e.g., `Marking`, `Sweeping`), orchestrates parallel workers, and handles global synchronization primitives like mutexes and condition variables for handshakes and suspension.
- **`MutatorState` (Thread-Local)**: Each application thread (`mutator`) maintains its own `local_mark_stack` for concurrent marking and `allocation_buffer`s for fast, lock-free allocations.
- **`ObjectType` and `ObjectSet`**: Allows the GC to categorize objects (e.g., `Weak`, `Finalizable`) and efficiently iterate over specific groups during collection phases.
- **`TypeInfo`**: A vtable-like structure generated for each `Gc<T>` type, containing function pointers for tracing outgoing `Gc<T>` pointers, running destructors, and, crucially, updating pointers to dead objects to point to the `FREE_SINGLETON`.
- **Handshake Mechanism**: A soft synchronization protocol that allows the collector to coordinate with mutator threads (e.g., stopping allocators, switching allocation color) without full stop-the-world pauses.

## 💡 How to Use (Conceptual)

1.  **Create `Gc<T>` Objects**:
    ```rust
    let my_data = Gc::new(MyStruct { /* ... */ });
    // Or, with classification:
    let my_finalizable_data = ALLOCATOR.allocate_classified(MyFinalizableStruct { /* ... */ }, ObjectClass::Finalizer);
    ```
2.  **Access Data**: Use `read()` for shared access and `write()` for exclusive mutable access. `write()` will return `None` if the collector is in a critical marking phase.

    ```rust
    let value = my_data.read().unwrap();
    println!("{}", value.field);

    if let Some(mut_value) = my_data.write() {
        mut_value.field = new_value;
    }
    ```

3.  **Implement `GcTrace`**: For custom types stored in `Gc<T>`, you would implement a trait (likely via a derive macro) that tells the GC how to find other `Gc<T>` pointers within your struct, allowing it to trace the object graph.
    ```rust
    // #[derive(GcTrace)] // Hypothetical macro
    struct MyStruct {
        nested_gc: Gc<AnotherStruct>,
        // ... other fields
    }
    ```
4.  **Implement `Finalizable`**: For types that require custom cleanup logic before being swept.
    ```rust
    impl Finalizable for MyFinalizableStruct {
        fn finalize(&mut self) {
            // Perform cleanup, e.g., close file handles
        }
    }
    ```
5.  **Register Roots**: Explicitly register global or static `Gc<T>` pointers as roots to prevent them from being collected.
    ```rust
    register_root(&my_global_object);
    ```

## ⚠️ Safety and Guarantees

This implementation is designed to be fully compatible with Rust's safety guarantees:

- **No Use-After-Free**: Achieved through the `FREE_SINGLETON` redirection, ensuring that dead objects are never truly "accessed" after being collected.
- **Data Race Prevention**: Extensive use of `Atomic` types, `RwLock`s, `Mutex`s, and RAII guards for `GcRef`/`GcRefMut` to prevent data races and ensure memory safety during concurrent operations.
- **Borrow Checker Compliance**: The `GcRef` and `GcRefMut` types ensure that Rust's borrowing rules are upheld, even when interacting with GC-managed memory.

This project is a theoretical exploration of bringing advanced GC techniques to Rust. A full-fledged production-ready implementation would require significant engineering effort, careful performance tuning, and comprehensive testing.
