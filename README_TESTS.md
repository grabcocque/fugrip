Repository test & fuzz harness guidelines

This README explains how to use the generated template harnesses for benches, stress tests, and fuzz targets.

Benchmarks

- The `benches/template_gc_bench.rs` file is a minimal placeholder. To run realistic benches:
  - Add `criterion = "*"` to `dev-dependencies` in `Cargo.toml`.
  - Replace the `main()` in the template with `criterion` benchmark definitions.

Stress tests

- The `tests/stress/template_gc_stress.rs` is an example of using `crossbeam::queue::SegQueue` and Rayon for parallel processing.
  - Run: `cargo test --test template_gc_stress` or `cargo nextest --test template_gc_stress`.

Fuzzing

- The `fuzz/fuzz_targets/template_fuzz_target.rs` is a libFuzzer-compatible template.
  - Install `cargo-fuzz` and add `libfuzzer-sys` to `dev-dependencies` in the fuzz crate.
  - Run: `cargo fuzz run template_fuzz_target`.

Notes

- These are templates — wire them directly to this crate's public testing API (e.g., object model constructors, allocation helpers).
