//! Demonstration of soft handshakes and enter/exit functionality
//!
//! This example shows how threads can coordinate through soft handshakes
//! and how threads that are blocked in syscalls or long operations can
//! still participate in GC coordination through the enter/exit mechanism.

use crossbeam::channel::{self};
use fugrip::{SafepointManager, pollcheck, safepoint_enter, safepoint_exit};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

fn main() {
    println!("FUGC Soft Handshake & Enter/Exit Demo");
    println!("=====================================");

    let container = fugrip::di::DIContainer::new();
    let manager = container.safepoint_manager();
    let counter = Arc::new(AtomicUsize::new(0));

    // Demonstrate basic soft handshake
    demonstrate_basic_handshake(&manager, Arc::clone(&counter));

    // Demonstrate enter/exit functionality
    demonstrate_enter_exit_functionality(&manager, Arc::clone(&counter));

    // Demonstrate multi-threaded coordination
    demonstrate_multithreaded_coordination(&manager, Arc::clone(&counter));

    println!("\n✅ Soft handshake demo completed successfully!");
    println!("Final counter value: {}", counter.load(Ordering::Relaxed));
}

/// Demonstrate basic soft handshake functionality
fn demonstrate_basic_handshake(manager: &SafepointManager, counter: Arc<AtomicUsize>) {
    println!("\n1. Basic Soft Handshake:");
    println!("   Requesting soft handshake with callback...");

    let counter_clone = Arc::clone(&counter);
    manager.request_soft_handshake(Box::new(move || {
        let old_value = counter_clone.fetch_add(1, Ordering::Relaxed);
        println!(
            "   📞 Handshake callback executed! Counter: {} -> {}",
            old_value,
            old_value + 1
        );
    }));

    println!("   ✓ Soft handshake completed");
}

/// Demonstrate enter/exit functionality for blocking operations
fn demonstrate_enter_exit_functionality(manager: &SafepointManager, counter: Arc<AtomicUsize>) {
    println!("\n2. Enter/Exit Functionality:");
    println!("   Simulating thread blocking in syscall...");

    let counter_clone = Arc::clone(&counter);

    // Create channels for proper coordination
    let (work_sender, work_receiver) = channel::bounded(1);
    let (exit_sender, exit_receiver) = channel::bounded(1);
    let (handshake_sender, handshake_receiver) = channel::bounded(1);

    // Spawn a thread that will block
    let blocking_thread = thread::spawn(move || {
        println!("   🧵 Thread starting active work");

        // Do some work with pollchecks
        for i in 0..5 {
            pollcheck();

            // Small amount of actual work instead of sleep
            for _ in 0..1000 {
                std::hint::spin_loop();
            }

            if i == 2 {
                println!("   📤 Thread entering exited state (blocking operation)");
                safepoint_exit(); // Thread is about to block

                // Signal that we're entering exit state
                let _ = exit_sender.try_send(());

                // Wait for external coordination signal instead of sleep
                let _ = work_receiver.recv_timeout(Duration::from_millis(100));

                println!("   📥 Thread re-entering active state");
                safepoint_enter(); // Thread is active again
            }
        }

        counter_clone.fetch_add(10, Ordering::Relaxed);
        println!("   ✅ Thread completed work");

        // Signal completion
        let _ = handshake_sender.try_send(());
    });

    // Wait for thread to signal it's in exit state
    let _ = exit_receiver.recv_timeout(Duration::from_millis(200));
    println!("   📞 Requesting handshake while thread is in exited state...");

    let counter_clone2 = Arc::clone(&counter);
    manager.request_soft_handshake(Box::new(move || {
        counter_clone2.fetch_add(100, Ordering::Relaxed);
        println!("   🤝 Handshake executed (including for exited threads)");
    }));

    // Signal thread to continue
    let _ = work_sender.try_send(());

    // Wait for thread completion
    let _ = handshake_receiver.recv_timeout(Duration::from_millis(200));
    blocking_thread.join().unwrap();
    println!("   ✓ Enter/exit demonstration completed");
}

/// Demonstrate multi-threaded coordination with handshakes
fn demonstrate_multithreaded_coordination(manager: &SafepointManager, counter: Arc<AtomicUsize>) {
    println!("\n3. Multi-threaded Coordination:");
    println!("   Spawning threads with different behaviors...");

    let num_threads = 4;

    // Use barriers for proper thread synchronization
    let start_barrier = Arc::new(Barrier::new(num_threads + 1)); // +1 for main thread
    let end_barrier = Arc::new(Barrier::new(num_threads + 1));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let counter_clone = Arc::clone(&counter);
            let start_barrier_clone = Arc::clone(&start_barrier);
            let end_barrier_clone = Arc::clone(&end_barrier);

            thread::spawn(move || {
                println!("   🧵 Thread {} starting", thread_id);

                // Wait for all threads to be ready
                start_barrier_clone.wait();

                for i in 0..20 {
                    pollcheck(); // Regular pollchecks

                    // Some threads occasionally enter/exit
                    if thread_id % 2 == 0 && i % 8 == 0 {
                        safepoint_exit();

                        // Simulate blocking with actual work instead of sleep
                        for _ in 0..10000 {
                            std::hint::spin_loop();
                        }

                        safepoint_enter();
                    }

                    // Small amount of work instead of sleep
                    for _ in 0..1000 {
                        std::hint::spin_loop();
                    }
                }

                counter_clone.fetch_add(1, Ordering::Relaxed);
                println!("   ✅ Thread {} completed", thread_id);

                // Signal completion
                end_barrier_clone.wait();
            })
        })
        .collect();

    // Wait for all threads to be ready
    start_barrier.wait();

    // Give threads a moment to start their work
    for _ in 0..5000 {
        std::hint::spin_loop();
    }

    // Perform multiple handshakes with proper coordination
    for handshake_id in 0..3 {
        println!("   📞 Requesting handshake #{}", handshake_id + 1);
        let counter_clone = Arc::clone(&counter);

        manager.request_soft_handshake(Box::new(move || {
            counter_clone.fetch_add(1000, Ordering::Relaxed);
            println!("   🤝 Handshake #{} callback executed", handshake_id + 1);
        }));

        // Small delay between handshakes using work instead of sleep
        for _ in 0..20000 {
            std::hint::spin_loop();
        }
    }

    // Wait for all threads to complete
    end_barrier.wait();
    for handle in handles {
        handle.join().unwrap();
    }

    println!("   ✓ Multi-threaded coordination completed");
}

/// Demonstrate advanced coordination patterns
#[allow(dead_code)]
fn demonstrate_advanced_patterns(manager: &SafepointManager, counter: Arc<AtomicUsize>) {
    println!("\n4. Advanced Coordination Patterns:");

    // Pattern 1: Root scanning with handshakes
    println!("   Pattern 1: Root scanning coordination");
    let counter_clone = Arc::clone(&counter);
    manager.request_soft_handshake(Box::new(move || {
        // Simulate root scanning
        counter_clone.fetch_add(1, Ordering::Relaxed);
        println!("   🔍 Root scanning performed on thread");
    }));

    // Pattern 2: Write barrier coordination
    println!("   Pattern 2: Write barrier coordination");
    let counter_clone = Arc::clone(&counter);
    manager.request_soft_handshake(Box::new(move || {
        // Simulate barrier activation/deactivation
        counter_clone.fetch_add(1, Ordering::Relaxed);
        println!("   🚧 Write barrier state updated");
    }));

    // Pattern 3: Concurrent marking coordination
    println!("   Pattern 3: Concurrent marking handshake");
    let counter_clone = Arc::clone(&counter);
    manager.request_soft_handshake(Box::new(move || {
        // Simulate marking work sharing
        counter_clone.fetch_add(1, Ordering::Relaxed);
        println!("   🎨 Marking work coordinated");
    }));

    println!("   ✓ Advanced patterns demonstrated");
}

/// Demonstrate error handling and edge cases
#[allow(dead_code)]
fn demonstrate_error_handling() {
    println!("\n5. Error Handling & Edge Cases:");

    // Case 1: Multiple rapid handshakes
    println!("   Case 1: Rapid handshake requests");
    let container = fugrip::di::DIContainer::new();
    let manager = container.safepoint_manager();

    for i in 0..5 {
        manager.request_soft_handshake(Box::new(move || {
            println!("   ⚡ Rapid handshake {} executed", i);
        }));
    }

    // Case 2: Handshake with no active threads
    println!("   Case 2: Handshake with minimal thread activity");
    manager.request_soft_handshake(Box::new(|| {
        println!("   👤 Solo handshake executed");
    }));

    println!("   ✓ Error handling demonstrated");
}
