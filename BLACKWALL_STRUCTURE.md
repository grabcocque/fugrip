# Blackwall Module Structure

This document explains the new module organization that makes the MMTk blackwall boundary explicit.

## Directory Structure

```
src/
├── frontend/              # OPAQUE HANDLES ONLY - No MMTk types allowed
│   ├── mod.rs
│   ├── alloc_facade.rs    # Zero-cost opaque handle APIs
│   ├── allocator.rs       # AllocatorInterface + MMTkAllocator/StubAllocator
│   └── types.rs           # Custom Address/ObjectReference for non-MMTk usage
├── backends/              # BACKEND-SPECIFIC - Can use native types freely
│   ├── mod.rs
│   ├── mmtk/              # MMTk backend implementation
│   │   ├── mod.rs
│   │   └── binding/       # MMTk VM binding (moved from src/binding/)
│   │       ├── mod.rs
│   │       ├── allocation.rs
│   │       ├── initialization.rs
│   │       ├── mutator.rs
│   │       ├── vm_impl.rs
│   │       └── tests.rs
│   ├── jemalloc/          # jemalloc backend implementation
│   │   └── mod.rs
│   └── stub/              # Testing stub backend implementation
│       └── mod.rs
└── [other modules]        # Core GC logic - should use frontend/ types
```

## Blackwall Rules

### ✅ Frontend Modules (`src/frontend/`)
- **ONLY opaque handles**: `MutatorHandle`, `PlanHandle`
- **ONLY custom types**: `frontend::types::{Address, ObjectReference}`
- **Zero-cost abstractions**: All dispatch via `#[cfg(feature)]`
- **NO MMTk types**: Cannot import `mmtk::*` directly

### ✅ Backend Modules (`src/backends/`)
- **Native types allowed**: Can use `mmtk::util::Address`, etc.
- **Backend-specific logic**: Each backend optimized for its environment
- **Feature-gated**: Only compiled when corresponding feature enabled

### ⚠️ Core Modules (everything else)
- **Should use frontend types**: Import from `crate::frontend::types::`
- **Transitional**: May still have some compat imports to clean up
- **Target**: Eventually all use opaque handles via frontend

## Import Patterns

### Frontend Code
```rust
// ✅ Correct - opaque handles only
use crate::frontend::alloc_facade::{MutatorHandle, allocate};
use crate::frontend::types::{Address, ObjectReference};
```

### Backend Code
```rust
// ✅ Correct - native MMTk types allowed
use mmtk::util::{Address, ObjectReference};
use mmtk::memory_manager;
```

### Core Code (transitional)
```rust
// ✅ Target pattern - use frontend types
use crate::frontend::types::{Address, ObjectReference};

// ❌ Old pattern - being phased out
use crate::compat::{Address, ObjectReference};
```

## Benefits

1. **Crystal Clear Boundary**: No confusion about what's frontend vs backend
2. **Zero-Cost Verification**: Frontend size assertions prove no abstraction overhead
3. **Backend Freedom**: MMTk code can use native types without restrictions
4. **Migration Clarity**: Easy to see what still needs frontend conversion
5. **Testing Simplicity**: Stub backend provides predictable test behavior

## Migration Status

- ✅ **Frontend**: alloc_facade.rs, allocator.rs, types.rs moved and working
- ✅ **Backend**: MMTk binding modules moved to backends/mmtk/binding/
- ⏳ **Core modules**: Need import path updates from compat → frontend
- 🎯 **Next**: Systematic cleanup of remaining compat imports