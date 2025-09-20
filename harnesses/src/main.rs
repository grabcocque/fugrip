// CLI harness runner for stress and bench templates.
use anyhow::Result;
use fugrip_harnesses::{run_mixed_allocation_workload, run_parallel_queue_work};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "parallel" => {
            run_parallel_queue_work(200_000, 8);
            println!("parallel workload completed");
        }
        "mixed" => {
            run_mixed_allocation_workload(100_000);
            println!("mixed allocation workload completed");
        }
        _ => {
            println!("Usage: harnesses <parallel|mixed>");
        }
    }

    Ok(())
}
