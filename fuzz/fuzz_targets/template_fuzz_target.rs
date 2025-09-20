// Template fuzz target for `fugrip` to be used with `cargo-fuzz`.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Interpret `data` as an input to a small workload.
    // For now this is a placeholder — wire it to real API calls like root scanning or write-barrier operations.
    let _ = data.len();
});
