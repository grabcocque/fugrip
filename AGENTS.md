# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the garbage-collection runtime: `concurrent.rs` and `fugc_coordinator.rs` drive the concurrent collector, while `plan.rs` and `binding.rs` integrate with MMTk.
- Integration scenarios live in `tests/`, including `fugc_8_step_protocol.rs` and `concurrent_state_sharing_test.rs` for protocol and state-sharing coverage.
- Performance harnesses sit in `benches/`, fuzz regressions in `fuzz/`, and helper scripts in `scripts/`. Cargo build outputs belong under `target/`.

## Build, Test, and Development Commands
- `cargo fmt` formats Rust sources before submission.
- `cargo nextest` runs the full unit, integration, doc, and property suite. Use `cargo nextest --test fugc_8_step_protocol` for fast checks while iterating on protocol logic.
- `cargo bench` executes benchmarks in `benches/` when validating performance-sensitive changes.
- `cargo run --bin <name>` launches any example binaries added under `src/bin/`.

## Coding Style & Naming Conventions
- Follow default `rustfmt` output: 4-space indentation, trailing commas, and module-level `//!` docs when applicable.
- Use `snake_case` for functions and variables, `UpperCamelCase` for types, and keep public APIs documented with Rustdoc examples that mirror actual GC usage.
- Add brief, purposeful comments only when the flow is non-obvious.

## Testing Guidelines
- Tests rely on Rust’s built-in harness with `#[test]`. Name integration cases using `<feature>_<behavior>` (e.g., `fugc_concurrent_collection_stress`).
- Ensure new GC phases or barriers gain unit tests in `src/` and scenario coverage in `tests/`.
- Prefer `cargo nextest` for CI parity; use targeted nextest invocations for protocol-specific work.

## Commit & Pull Request Guidelines
- Write imperative commit subjects such as “Implement page coloring sweep,” grouping related edits together.
- Pull requests should summarise behavioral changes, list touched modules, note validation commands (examples: `cargo nextest`, `cargo fmt`), and link tracking issues. Attach diagnostics or traces only when they clarify GC behavior.

## Architecture Overview
- The coordinator runs an eight-step cycle: activate barriers, switch to black allocation, mark roots, perform mutator handshakes, trace, prep sweep, and execute page-color sweep.
- Mutator threads must poll safepoints; tests often spawn helper threads to stand in for runtime mutators. Ensure new coordination logic keeps these lifecycles in sync.
