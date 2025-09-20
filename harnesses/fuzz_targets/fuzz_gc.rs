#![no_main]
use libfuzzer_sys::fuzz_target;

use fugrip_harnesses::run_mixed_allocation_workload;

fuzz_target!(|data: &[u8]| {
    // Use input length to drive a small workload
    let items = (data.len() % 1000) + 1;
    run_mixed_allocation_workload(items);
});
