Test Harnesses and Fuzz/Bench Templates

This document describes the new templated harnesses for stress tests, benchmarks, and fuzz targets.

Design goals

- Use Rayon or crossbeam scoped threads; avoid `thread::spawn` in new harnesses
- Use lock-free primitives: `arc_swap::ArcSwap`, `crossbeam::queue::SegQueue`, atomic counters
- Keep harnesses lightweight and focused on measurable work and metrics

Files added by the scaffolding:

- `benches/template_gc_bench.rs` - a benchmark harness using `criterion` (optional, see README)
- `tests/stress/template_gc_stress.rs` - an integration-style stress harness using Rayon
- `fuzz/fuzz_targets/template_fuzz_target.rs` - a libFuzzer-compatible fuzz target template

How to use

- Bench: `cargo bench --bench template_gc_bench` (requires enabling the bench feature in Cargo.toml)
- Tests: `cargo test --test template_gc_stress` or `cargo nextest --test template_gc_stress`
- Fuzzing: Use `cargo fuzz run template_fuzz_target` (requires cargo-fuzz)

Notes

- These are templates; you should wire them to the crate's public testing API for real workloads.
- The templates intentionally avoid adding heavy dependencies. Add `criterion` or `cargo-fuzz` to `dev-dependencies` before enabling bench/fuzz runs.
