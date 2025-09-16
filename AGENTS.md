# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds Rust source modules: `concurrent.rs` and `fugc_coordinator.rs` implement GC mechanics, while `plan.rs` and `binding.rs` bridge to MMTk.
- `tests/` contains integration suites (`fugc_8_step_protocol.rs`, `concurrent_state_sharing_test.rs`) plus ancillary GC stress tests.
- `benches/`, `fuzz/`, and `scripts/` provide performance harnesses, fuzz regressions, and helper tooling. Generated artifacts live under `target/`.

## Build, Test, and Development Commands
- `cargo fmt` – enforce standard formatting before commits.
- `cargo test` – run all unit, integration, doc, and property tests; the default profile is sufficient for PR validation.
- `cargo bench` – execute benchmarking harnesses in `benches/`; only required for performance-related changes.
- `cargo run --bin <name>` – execute example binaries if added under `src/bin/`.

## Coding Style & Naming Conventions
- Rust code must follow `rustfmt` defaults (4-space indentation, trailing commas where applicable).
- Prefer descriptive `snake_case` for functions/variables, `UpperCamelCase` for types, and module-level docs using `//!`.
- Keep public APIs documented with Rustdoc comments; include examples when adding new GC endpoints.

## Testing Guidelines
- Integration tests reside in `tests/` and rely on `#[test]` harnesses; name tests with the pattern `<feature>_<behavior>`. Example: `fugc_concurrent_collection_stress`.
- Ensure new GC phases or barriers ship with unit coverage in `src/` and scenario coverage in `tests/`.
- Run `cargo test --test fugc_8_step_protocol` when touching FUGC protocol logic for faster feedback.

## Commit & Pull Request Guidelines
- Follow imperative mood commit subjects (e.g., “Implement page coloring sweep”). Group related changes into a single commit when possible.
- PRs should summarize behavior changes, list affected modules, and mention validation commands (e.g., `cargo test`). Link to tracking issues and attach screenshots only when UI tooling is touched.

## Architecture Notes
- The coordinator orchestrates an eight-step protocol: barrier activation, black allocation, root marking, stack handshakes, tracing, sweep prep, and page-color sweep.
- Mutator handshakes require registered `MutatorThread` instances polling safepoints; spawn helper threads in tests to simulate runtime conditions.
