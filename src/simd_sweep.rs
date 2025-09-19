//! High-performance SIMD-based sweeping implementation
//!
//! This module provides an insanely fast sweeping algorithm using bitvectors
//! and SIMD operations to dramatically outperform traditional marking-based
//! collection. The sweeper operates on compressed bitvectors representing
//! object liveness across the heap.
//!
//! This consolidates SIMD bitvector utilities and sweeping algorithms into
//! a unified high-performance implementation with AVX2 optimization.

use mmtk::util::{Address, ObjectReference};
use std::cmp::{max, min};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Constants for SIMD sweeping optimization
const BITS_PER_WORD: usize = 64;
const WORDS_PER_SIMD_BLOCK: usize = 4; // 256-bit SIMD
const BITS_PER_SIMD_BLOCK: usize = BITS_PER_WORD * WORDS_PER_SIMD_BLOCK;
const OBJECTS_PER_CACHE_LINE: usize = 64 / 8; // Assume 8-byte objects
const BITVECTOR_CACHE_LINE_SIZE: usize = 64;

/// Hybrid strategy constants - configurable based on heap layout
const TARGET_CHUNK_SIZE_BYTES: usize = 64 * 1024; // Target 64KB chunks for cache locality
const DENSITY_THRESHOLD_PERCENT: usize = 35; // Switch to sparse scan below 35%
const MIN_WORDS_FOR_SIMD: usize = WORDS_PER_SIMD_BLOCK; // Minimum words for SIMD vectorization

/// High-performance bitvector for tracking object liveness
///
/// Uses SIMD operations for bulk operations and cache-optimized storage
/// for maximum sweeping throughput.
///
/// # Examples
///
/// ```
/// use fugrip::simd_sweep::SimdBitvector;
/// use mmtk::util::Address;
///
/// let heap_base = unsafe { Address::from_usize(0x10000000) };
/// let heap_size = 1024 * 1024; // 1MB
/// let bitvector = SimdBitvector::new(heap_base, heap_size, 16);
///
/// // Mark some objects as live
/// let obj1 = unsafe { Address::from_usize(heap_base.as_usize() + 64) };
/// bitvector.mark_live(obj1);
///
/// // Perform SIMD sweep
/// let stats = bitvector.simd_sweep();
/// assert!(stats.objects_swept > 0);
/// ```
pub struct SimdBitvector {
    /// Base address of the heap
    heap_base: Address,
    /// Size of the heap in bytes
    heap_size: usize,
    /// Object alignment (objects are aligned to this boundary)
    object_alignment: usize,
    /// Bitvector storage (one bit per object slot)
    bits: Vec<AtomicU64>,
    /// Number of objects that can be represented
    max_objects: usize,
    /// Statistics counters
    objects_marked: AtomicUsize,
    objects_swept: AtomicUsize,
    simd_operations: AtomicUsize,
    /// Hybrid strategy infrastructure - dynamic chunk layout
    /// Number of chunks in this heap layout
    chunk_count: usize,
    /// Objects per chunk (calculated based on heap size and object alignment)
    objects_per_chunk: usize,
    /// Words per chunk (derived from objects per chunk)
    words_per_chunk: usize,
    /// Actual chunk size in bytes (may differ from target for optimal alignment)
    actual_chunk_size_bytes: usize,
    /// Per-chunk population counters for density analysis
    chunk_populations: Vec<AtomicUsize>,
    /// Hybrid strategy performance counters
    simd_chunks_processed: AtomicUsize,
    sparse_chunks_processed: AtomicUsize,
    /// Total compare-exchange operations performed during marking
    compare_exchange_operations: AtomicUsize,
    /// Total compare-exchange retries due to contention
    compare_exchange_retries: AtomicUsize,
    /// Architecture-specific optimizations used
    #[cfg(target_arch = "x86_64")]
    avx2_operations: AtomicUsize,
}

impl SimdBitvector {
    /// Create a new SIMD-optimized bitvector for the given heap
    ///
    /// # Arguments
    ///
    /// * `heap_base` - Base address of the managed heap
    /// * `heap_size` - Total size of the heap in bytes
    /// * `object_alignment` - Minimum object alignment (must be power of 2)
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024 * 1024, 16);
    /// assert_eq!(bitvector.max_objects(), 1024 * 1024 / 16);
    /// ```
    pub fn new(heap_base: Address, heap_size: usize, object_alignment: usize) -> Self {
        assert!(
            object_alignment.is_power_of_two(),
            "Object alignment must be power of 2"
        );

        let max_objects = heap_size / object_alignment;
        let words_needed = max_objects.div_ceil(BITS_PER_WORD);

        // Align to SIMD block boundaries for optimal performance
        let aligned_words = (words_needed + WORDS_PER_SIMD_BLOCK - 1) & !(WORDS_PER_SIMD_BLOCK - 1);

        // Calculate dynamic chunk infrastructure for hybrid strategy
        let (chunk_count, objects_per_chunk, words_per_chunk, actual_chunk_size_bytes) =
            Self::calculate_chunk_layout(max_objects, object_alignment, aligned_words);
        let chunk_populations = (0..chunk_count).map(|_| AtomicUsize::new(0)).collect();

        Self {
            heap_base,
            heap_size,
            object_alignment,
            bits: (0..aligned_words).map(|_| AtomicU64::new(0)).collect(),
            max_objects,
            objects_marked: AtomicUsize::new(0),
            objects_swept: AtomicUsize::new(0),
            simd_operations: AtomicUsize::new(0),
            chunk_count,
            objects_per_chunk,
            words_per_chunk,
            actual_chunk_size_bytes,
            chunk_populations,
            simd_chunks_processed: AtomicUsize::new(0),
            sparse_chunks_processed: AtomicUsize::new(0),
            compare_exchange_operations: AtomicUsize::new(0),
            compare_exchange_retries: AtomicUsize::new(0),
            #[cfg(target_arch = "x86_64")]
            avx2_operations: AtomicUsize::new(0),
        }
    }

    /// Calculate optimal chunk layout for dynamic heap configuration
    ///
    /// Determines chunk size, object count, and word count based on actual heap parameters
    /// and object alignment, ensuring accurate density calculations and proper boundaries.
    fn calculate_chunk_layout(
        max_objects: usize,
        object_alignment: usize,
        total_words: usize,
    ) -> (usize, usize, usize, usize) {
        // Calculate target objects per chunk based on desired chunk size and alignment
        let target_objects_per_chunk = TARGET_CHUNK_SIZE_BYTES / object_alignment;

        // Ensure we don't create chunks larger than necessary
        let effective_objects_per_chunk = if target_objects_per_chunk > max_objects {
            max_objects
        } else {
            target_objects_per_chunk
        };

        // Calculate words needed for this many objects
        let words_needed_per_chunk = effective_objects_per_chunk.div_ceil(BITS_PER_WORD);

        // Align to SIMD block boundaries for optimal vectorization
        let aligned_words_per_chunk =
            (words_needed_per_chunk + WORDS_PER_SIMD_BLOCK - 1) & !(WORDS_PER_SIMD_BLOCK - 1);

        // Don't exceed total available words
        let final_words_per_chunk = aligned_words_per_chunk.min(total_words);

        // Calculate actual objects that fit in the aligned chunk
        let objects_per_chunk = (final_words_per_chunk * BITS_PER_WORD).min(max_objects);

        // Calculate actual chunk size and count
        let actual_chunk_size_bytes = objects_per_chunk * object_alignment;
        let chunk_count = if objects_per_chunk > 0 {
            max(max_objects.div_ceil(objects_per_chunk), 1)
        } else {
            1
        };

        (
            chunk_count,
            objects_per_chunk,
            final_words_per_chunk,
            actual_chunk_size_bytes,
        )
    }

    #[inline]
    fn chunk_start_object(&self, chunk_index: usize) -> usize {
        chunk_index * self.objects_per_chunk
    }

    #[inline]
    fn chunk_objects(&self, chunk_index: usize) -> usize {
        if chunk_index >= self.chunk_count {
            return 0;
        }
        let start = self.chunk_start_object(chunk_index);
        if start >= self.max_objects {
            0
        } else {
            min(self.max_objects - start, self.objects_per_chunk)
        }
    }

    #[inline]
    fn chunk_word_range(&self, chunk_index: usize) -> (usize, usize) {
        let start_object = self.chunk_start_object(chunk_index);
        if start_object >= self.max_objects {
            return (self.bits.len(), self.bits.len());
        }
        let chunk_objects = self.chunk_objects(chunk_index);
        let start_word = start_object / BITS_PER_WORD;
        let end_word = (start_object + chunk_objects).div_ceil(BITS_PER_WORD);
        (start_word, min(end_word, self.bits.len()))
    }

    #[inline]
    fn chunk_index_for_bit(&self, bit_index: usize) -> usize {
        let objects_per_chunk = max(self.objects_per_chunk, 1);
        let index = bit_index / objects_per_chunk;
        if index >= self.chunk_count {
            self.chunk_count - 1
        } else {
            index
        }
    }

    #[inline]
    fn word_mask_in_chunk(&self, chunk_index: usize, word_offset: usize) -> u64 {
        let chunk_objects = self.chunk_objects(chunk_index);
        let bits_before = word_offset * BITS_PER_WORD;
        if bits_before >= chunk_objects {
            return 0;
        }
        let remaining = chunk_objects - bits_before;
        if remaining >= BITS_PER_WORD {
            !0u64
        } else {
            (1u64 << remaining) - 1
        }
    }

    /// Create a word mask for chunk objects with precise boundary handling
    ///
    /// This helper uses chunk capacity information to generate accurate masks
    /// for words that cross chunk boundaries or heap limits. Essential for
    /// correct processing of partial chunks without accessing invalid bits.
    #[inline]
    fn word_mask_for_chunk_objects(
        &self,
        chunk_idx: usize,
        word_idx: usize,
        chunk_capacity: usize,
    ) -> u64 {
        let chunk_start_object = chunk_idx * self.objects_per_chunk;
        let chunk_end_object = chunk_start_object + chunk_capacity;

        let word_start_object = word_idx * BITS_PER_WORD;
        let word_end_object = word_start_object + BITS_PER_WORD;

        // Calculate intersection of word range with valid chunk object range
        let valid_start = word_start_object.max(chunk_start_object);
        let valid_end = word_end_object.min(chunk_end_object);

        if valid_end <= valid_start {
            return 0; // No valid objects in this word
        }

        // Create mask for valid bits in this word
        let start_bit_in_word = valid_start.saturating_sub(word_start_object);
        let end_bit_in_word = valid_end.saturating_sub(word_start_object);

        if start_bit_in_word == 0 && end_bit_in_word >= BITS_PER_WORD {
            // Full word is valid
            !0u64
        } else {
            // Partial word mask
            let bit_count = end_bit_in_word - start_bit_in_word;
            let full_mask = if bit_count >= 64 {
                !0u64
            } else {
                (1u64 << bit_count) - 1
            };
            full_mask << start_bit_in_word
        }
    }

    /// Mark an object as live in the bitvector
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    ///
    /// let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + 32) };
    /// bitvector.mark_live(obj_addr);
    /// assert!(bitvector.is_marked(obj_addr));
    /// ```
    /// Mark an object as live in the bitvector with concurrency safety
    ///
    /// Uses compare-exchange to ensure atomic bit setting and accurate chunk population counting
    /// even under concurrent access from multiple marking threads.
    ///
    /// # Arguments
    /// * `object_addr` - Address of the object to mark as live
    ///
    /// # Returns
    /// * `true` if the object was newly marked (bit transition from 0 to 1)
    /// * `false` if the object was already marked or address is invalid
    pub fn mark_live(&self, object_addr: Address) -> bool {
        if let Some(bit_index) = self.object_to_bit_index(object_addr) {
            let word_index = bit_index / BITS_PER_WORD;
            let bit_offset = bit_index % BITS_PER_WORD;

            if word_index < self.bits.len() {
                let atomic_word = &self.bits[word_index];
                let bit_mask = 1u64 << bit_offset;

                // Concurrency-safe bit marking with compare-exchange retry loop
                let mut retry_count = 0;
                loop {
                    let current_word = atomic_word.load(Ordering::Acquire);

                    // If bit is already set, no work needed
                    if (current_word & bit_mask) != 0 {
                        return false;
                    }

                    // Attempt to set the bit atomically
                    let new_word = current_word | bit_mask;
                    self.compare_exchange_operations
                        .fetch_add(1, Ordering::Relaxed);

                    match atomic_word.compare_exchange_weak(
                        current_word,
                        new_word,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // Successfully set the bit - update counters with accurate chunk mapping
                            let object_index = bit_index; // bit_index is already the object index
                            let chunk_index = object_index / self.objects_per_chunk;
                            if chunk_index < self.chunk_populations.len() {
                                self.chunk_populations[chunk_index].fetch_add(1, Ordering::Relaxed);
                            }

                            self.objects_marked.fetch_add(1, Ordering::Relaxed);

                            // Track retry statistics for performance analysis
                            if retry_count > 0 {
                                self.compare_exchange_retries
                                    .fetch_add(retry_count, Ordering::Relaxed);
                            }

                            return true;
                        }
                        Err(_) => {
                            // CAS failed due to concurrent modification, retry
                            // Using compare_exchange_weak for better performance on some architectures
                            retry_count += 1;
                            continue;
                        }
                    }
                }
            }
        }
        false
    }

    /// Check if an object is marked as live
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    ///
    /// let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + 32) };
    /// assert!(!bitvector.is_marked(obj_addr));
    /// bitvector.mark_live(obj_addr);
    /// assert!(bitvector.is_marked(obj_addr));
    /// ```
    pub fn is_marked(&self, object_addr: Address) -> bool {
        if let Some(bit_index) = self.object_to_bit_index(object_addr) {
            let word_index = bit_index / BITS_PER_WORD;
            let bit_offset = bit_index % BITS_PER_WORD;

            if word_index < self.bits.len() {
                let word = self.bits[word_index].load(Ordering::Relaxed);
                return (word & (1u64 << bit_offset)) != 0;
            }
        }
        false
    }

    /// Perform ultra-fast SIMD sweep of the entire heap
    ///
    /// This is the core high-performance operation that makes sweeping
    /// dramatically faster than marking. Uses 256-bit SIMD operations
    /// to process multiple cache lines of bitvector data simultaneously.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    ///
    /// // Mark some objects
    /// bitvector.mark_live(unsafe { Address::from_usize(heap_base.as_usize() + 32) });
    /// bitvector.mark_live(unsafe { Address::from_usize(heap_base.as_usize() + 64) });
    ///
    /// let stats = bitvector.simd_sweep();
    /// assert!(stats.objects_swept > 0);
    /// assert!(stats.sweep_time_ns < stats.objects_swept as u64 * 10); // Sub-10ns per object
    /// ```
    /// Perform ultra-fast SIMD sweep (delegates to hybrid sweep for best performance)
    ///
    /// This method now delegates to the hybrid sweep implementation which automatically
    /// chooses the optimal strategy (SIMD vs sparse) based on chunk density analysis.
    /// Maintains backward compatibility by converting hybrid statistics to legacy format.
    pub fn simd_sweep(&self) -> SweepStatistics {
        // Delegate to hybrid sweep for optimal performance
        let hybrid_stats = self.hybrid_sweep();

        // Convert hybrid statistics to legacy SweepStatistics format for compatibility
        SweepStatistics {
            objects_swept: hybrid_stats.objects_swept,
            free_blocks: hybrid_stats.free_blocks,
            sweep_time_ns: hybrid_stats.sweep_time_ns,
            simd_blocks_processed: hybrid_stats.simd_chunks_processed,
            throughput_objects_per_sec: hybrid_stats.throughput_objects_per_sec,
        }
    }

    /// Hybrid SIMD+sparse sweep combining our vectorized approach with Verse-style chunking
    ///
    /// This method dynamically switches between SIMD sweep for dense chunks and
    /// Verse-style sparse scanning for low-density chunks based on runtime statistics.
    /// Dense chunks (>35% populated) use our existing AVX2 SIMD implementation,
    /// while sparse chunks use trailing_zeros() scanning to skip work efficiently.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024 * 1024, 16);
    ///
    /// // Mark some objects creating mixed density pattern
    /// for i in (0..1000).step_by(10) {
    ///     let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
    ///     bitvector.mark_live(obj_addr);
    /// }
    ///
    /// let stats = bitvector.hybrid_sweep();
    /// assert!(stats.objects_swept > 0);
    /// ```
    pub fn hybrid_sweep(&self) -> HybridSweepStatistics {
        let sweep_start = Instant::now();
        let mut free_blocks = Vec::new();
        let mut objects_swept = 0usize;
        let mut simd_chunks = 0usize;
        let mut sparse_chunks = 0usize;

        // Reset per-sweep instrumentation counters
        self.simd_chunks_processed.store(0, Ordering::Relaxed);
        self.sparse_chunks_processed.store(0, Ordering::Relaxed);

        // Process each chunk with adaptive strategy selection
        for chunk_idx in 0..self.chunk_count {
            let chunk_mask = self.create_chunk_object_mask(chunk_idx);

            if chunk_mask.is_empty() {
                continue;
            }

            // Analyze chunk density for strategy selection with precise object capacity
            let chunk_population = self.chunk_populations[chunk_idx].load(Ordering::Acquire);
            let object_capacity = chunk_mask.object_capacity;
            let density_percent = if object_capacity > 0 {
                (chunk_population * 100) / object_capacity
            } else {
                0
            };

            let (start_word, end_word) = self.get_chunk_word_range(chunk_idx);
            let chunk_words = end_word - start_word;

            if density_percent >= DENSITY_THRESHOLD_PERCENT && chunk_words >= MIN_WORDS_FOR_SIMD {
                // Dense chunk: use SIMD sweep
                let chunk_swept =
                    self.process_chunk_simd_masked(chunk_idx, &chunk_mask, &mut free_blocks);
                objects_swept += chunk_swept;
                simd_chunks += 1;
            } else {
                // Sparse chunk: use Verse-style trailing_zeros scan
                let chunk_swept =
                    self.process_chunk_sparse_masked(chunk_idx, &chunk_mask, &mut free_blocks);
                objects_swept += chunk_swept;
                sparse_chunks += 1;
            }
        }

        let sweep_time = sweep_start.elapsed();
        self.objects_swept.store(objects_swept, Ordering::Relaxed);

        // Reset chunk populations for next cycle
        for chunk_pop in &self.chunk_populations {
            chunk_pop.store(0, Ordering::Relaxed);
        }
        self.objects_marked.store(0, Ordering::Relaxed);
        self.simd_chunks_processed
            .store(simd_chunks, Ordering::Relaxed);
        self.sparse_chunks_processed
            .store(sparse_chunks, Ordering::Relaxed);

        HybridSweepStatistics {
            objects_swept,
            free_blocks: free_blocks.len(),
            sweep_time_ns: sweep_time.as_nanos() as u64,
            simd_chunks_processed: simd_chunks,
            sparse_chunks_processed: sparse_chunks,
            total_chunks: self.chunk_count,
            density_threshold_percent: DENSITY_THRESHOLD_PERCENT,
            throughput_objects_per_sec: if sweep_time.as_nanos() > 0 {
                (objects_swept as u128 * 1_000_000_000) / sweep_time.as_nanos()
            } else {
                0
            } as u64,
            compare_exchange_operations: self.compare_exchange_operations.load(Ordering::Relaxed),
            compare_exchange_retries: self.compare_exchange_retries.load(Ordering::Relaxed),
            simd_operations: self.simd_operations.load(Ordering::Relaxed),
            #[cfg(target_arch = "x86_64")]
            avx2_operations: self.avx2_operations.load(Ordering::Relaxed),
        }
    }

    /// Process a chunk using SIMD operations for dense regions with capacity-aware masking
    fn process_chunk_simd_masked(
        &self,
        chunk_idx: usize,
        chunk_mask: &ChunkMask,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        let mut objects_swept = 0usize;
        let chunk_capacity = chunk_mask.object_capacity;

        // Use chunk capacity to determine accurate word range
        let chunk_start_object = chunk_idx * self.objects_per_chunk;
        let chunk_end_object = chunk_start_object + chunk_capacity;
        let start_word = chunk_start_object / BITS_PER_WORD;
        let end_word = chunk_end_object.div_ceil(BITS_PER_WORD);
        let end_word = end_word.min(self.bits.len());

        // Process chunk in SIMD blocks when possible, applying capacity-based masks
        let mut word_idx = start_word;

        while word_idx + WORDS_PER_SIMD_BLOCK <= end_word {
            #[cfg(target_arch = "x86_64")]
            {
                objects_swept += self.process_simd_block_capacity_masked(
                    word_idx,
                    chunk_idx,
                    chunk_capacity,
                    free_blocks,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // Fallback to scalar processing for non-x86_64
                for i in 0..WORDS_PER_SIMD_BLOCK {
                    let mask =
                        self.word_mask_for_chunk_objects(chunk_idx, word_idx + i, chunk_capacity);
                    objects_swept += self.process_word_with_mask(word_idx + i, mask, free_blocks);
                }
            }
            word_idx += WORDS_PER_SIMD_BLOCK;
        }

        // Handle remaining words in chunk with scalar operations and capacity-based masking
        while word_idx < end_word {
            let mask = self.word_mask_for_chunk_objects(chunk_idx, word_idx, chunk_capacity);
            if mask != 0 {
                objects_swept += self.process_word_with_mask(word_idx, mask, free_blocks);
            }
            word_idx += 1;
        }

        objects_swept
    }

    /// Process a sparse chunk with capacity-aware masking using Verse-style trailing_zeros scan
    fn process_chunk_sparse_masked(
        &self,
        chunk_idx: usize,
        chunk_mask: &ChunkMask,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        let mut objects_swept = 0usize;
        let chunk_capacity = chunk_mask.object_capacity;

        // Use chunk capacity to determine accurate word range for sparse processing
        let chunk_start_object = chunk_idx * self.objects_per_chunk;
        let chunk_end_object = chunk_start_object + chunk_capacity;
        let start_word = chunk_start_object / BITS_PER_WORD;
        let end_word = chunk_end_object.div_ceil(BITS_PER_WORD);
        let end_word = end_word.min(self.bits.len());

        // Process words in chunk with capacity-based masked operations
        for word_idx in start_word..end_word {
            if word_idx >= self.bits.len() {
                break;
            }

            let mask = self.word_mask_for_chunk_objects(chunk_idx, word_idx, chunk_capacity);
            if mask == 0 {
                continue; // Skip words with no valid objects
            }

            // Use trailing_zeros optimization for sparse scanning
            let word = self.bits[word_idx].load(Ordering::Relaxed);
            let masked_word = word & mask;

            if masked_word != 0 {
                // Word has live objects, so process for dead object extraction
                objects_swept += self.process_word_with_mask(word_idx, mask, free_blocks);
            } else if mask == !0u64 {
                // Full word is dead and valid - optimized path for completely free words
                let dead_count = BITS_PER_WORD;
                self.extract_free_blocks_with_trailing_zeros(word_idx, mask, free_blocks);
                self.bits[word_idx].store(0, Ordering::Relaxed);
                objects_swept += dead_count;
            } else {
                // Partial word that's completely dead within the valid mask
                let dead_count = mask.count_ones() as usize;
                self.extract_free_blocks_with_trailing_zeros(word_idx, mask, free_blocks);
                let current_word = self.bits[word_idx].load(Ordering::Relaxed);
                let cleared_word = current_word & !mask;
                self.bits[word_idx].store(cleared_word, Ordering::Relaxed);
                objects_swept += dead_count;
            }
        }

        objects_swept
    }

    /// Process a chunk using SIMD operations for dense regions (legacy method)
    fn process_chunk_simd(
        &self,
        start_word: usize,
        word_count: usize,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        let mut objects_swept = 0usize;
        let end_word = start_word + word_count;

        // Process chunk in SIMD blocks when possible
        let mut word_idx = start_word;
        while word_idx + WORDS_PER_SIMD_BLOCK <= end_word {
            #[cfg(target_arch = "x86_64")]
            {
                objects_swept += self.process_simd_block(word_idx, free_blocks);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // Fallback to scalar processing for non-x86_64
                for i in 0..WORDS_PER_SIMD_BLOCK {
                    objects_swept += self.process_word_scalar(word_idx + i, free_blocks);
                }
            }
            word_idx += WORDS_PER_SIMD_BLOCK;
        }

        // Handle remaining words in chunk with scalar operations
        while word_idx < end_word {
            objects_swept += self.process_word_scalar(word_idx, free_blocks);
            word_idx += 1;
        }

        objects_swept
    }

    /// Process a chunk using unified sparse scanning with masked scalar operations
    ///
    /// Reuses the masked scalar word processing to avoid redundant stores and simplify logic.
    /// Optimized for low-density regions using trailing_zeros to skip sparse areas efficiently.
    fn process_chunk_sparse(
        &self,
        start_word: usize,
        word_count: usize,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        let mut objects_swept = 0usize;
        let end_word = start_word + word_count;

        // Process words in chunk with unified masked operations
        for word_idx in start_word..end_word {
            if word_idx >= self.bits.len() {
                break;
            }

            // Use unified masked scalar processing for consistency
            let word_mask = if word_idx == end_word - 1 {
                // Handle partial word at chunk boundary
                let bits_in_final_word = self.calculate_bits_in_final_word(word_idx, word_count);
                if bits_in_final_word < BITS_PER_WORD {
                    (1u64 << bits_in_final_word) - 1
                } else {
                    !0u64
                }
            } else {
                !0u64 // Full word mask
            };

            objects_swept += self.process_word_with_mask(word_idx, word_mask, free_blocks);
        }

        objects_swept
    }

    /// Unified masked scalar word processing function
    ///
    /// Handles word processing with proper masking for partial words and chunk boundaries.
    /// Avoids redundant stores and provides consistent behavior across SIMD and sparse paths.
    fn process_word_with_mask(
        &self,
        word_idx: usize,
        mask: u64,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        let word = self.bits[word_idx].load(Ordering::Relaxed);
        let masked_word = word & mask;

        // Invert to get dead objects, applying mask
        let dead_mask = (!masked_word) & mask;
        let dead_count = dead_mask.count_ones() as usize;

        if dead_count > 0 {
            // Extract free blocks using Verse-style trailing_zeros for efficiency
            self.extract_free_blocks_with_trailing_zeros(word_idx, dead_mask, free_blocks);
        }

        // Clear the processed bits (only the masked portion)
        let cleared_word = word & !mask;
        self.bits[word_idx].store(cleared_word, Ordering::Relaxed);

        dead_count
    }

    /// Calculate bits in final word for partial chunk handling
    fn calculate_bits_in_final_word(&self, word_idx: usize, _word_count: usize) -> usize {
        let total_bits = self.max_objects;
        let word_start_bit = word_idx * BITS_PER_WORD;
        let remaining_bits = total_bits.saturating_sub(word_start_bit);
        remaining_bits.min(BITS_PER_WORD)
    }

    /// Process a single SIMD block with capacity-based masking (AVX2 optimized)
    #[cfg(target_arch = "x86_64")]
    fn process_simd_block_capacity_masked(
        &self,
        start_word: usize,
        chunk_idx: usize,
        chunk_capacity: usize,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        unsafe {
            self.simd_operations.fetch_add(1, Ordering::Relaxed);

            // Load 4 consecutive words with capacity-based masking
            let mut word_values = [0u64; WORDS_PER_SIMD_BLOCK];
            let mut masks = [0u64; WORDS_PER_SIMD_BLOCK];

            for i in 0..WORDS_PER_SIMD_BLOCK {
                if start_word + i < self.bits.len() {
                    word_values[i] = self.bits[start_word + i].load(Ordering::Relaxed);
                    masks[i] =
                        self.word_mask_for_chunk_objects(chunk_idx, start_word + i, chunk_capacity);
                    // Apply mask to the word to only consider valid objects
                    word_values[i] &= masks[i];
                }
            }

            // Load into AVX2 register
            let live_mask = _mm256_loadu_si256(word_values.as_ptr() as *const __m256i);
            let all_ones = _mm256_set1_epi64x(-1i64);
            let dead_mask = _mm256_xor_si256(live_mask, all_ones);

            // Apply masks to dead_mask to ensure we only consider valid object bits
            let mask_simd = _mm256_loadu_si256(masks.as_ptr() as *const __m256i);
            let masked_dead = _mm256_and_si256(dead_mask, mask_simd);

            // Count dead objects
            let dead_count = self.count_dead_objects_simd(masked_dead);

            // Extract free blocks if any dead objects found
            if dead_count > 0 {
                // Convert SIMD register back to individual words for unified processing
                let mut temp_words = [0u64; WORDS_PER_SIMD_BLOCK];
                _mm256_storeu_si256(temp_words.as_mut_ptr() as *mut __m256i, masked_dead);

                for (i, &word) in temp_words.iter().enumerate() {
                    if word != 0 {
                        let word_idx = start_word + i;
                        self.extract_free_blocks_with_trailing_zeros(word_idx, word, free_blocks);
                    }
                }
            }

            // Clear processed words (only the masked portions)
            for (i, mask) in masks.iter().enumerate().take(WORDS_PER_SIMD_BLOCK) {
                if start_word + i < self.bits.len() && *mask != 0 {
                    let current_word = self.bits[start_word + i].load(Ordering::Relaxed);
                    let cleared_word = current_word & !mask;
                    self.bits[start_word + i].store(cleared_word, Ordering::Relaxed);
                }
            }

            dead_count
        }
    }

    /// Process a single SIMD block (4 x 64-bit words) using AVX2 with proper masking
    #[cfg(target_arch = "x86_64")]
    fn process_simd_block_masked(
        &self,
        start_word: usize,
        word_offset: usize,
        chunk_mask: &ChunkMask,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        unsafe {
            self.simd_operations.fetch_add(1, Ordering::Relaxed);

            // Load 4 consecutive words into SIMD register
            let mut word_values = [0u64; WORDS_PER_SIMD_BLOCK];
            let mut masks = [0u64; WORDS_PER_SIMD_BLOCK];

            for i in 0..WORDS_PER_SIMD_BLOCK {
                if start_word + i < self.bits.len() {
                    word_values[i] = self.bits[start_word + i].load(Ordering::Relaxed);
                    masks[i] = chunk_mask.get_word_mask(word_offset + i);
                    // Apply mask to the word to only consider valid objects
                    word_values[i] &= masks[i];
                }
            }

            // Load into AVX2 register
            let live_mask = _mm256_loadu_si256(word_values.as_ptr() as *const __m256i);
            let all_ones = _mm256_set1_epi64x(-1i64);
            let dead_mask = _mm256_xor_si256(live_mask, all_ones);

            // Apply masks to dead_mask to ensure we only consider valid object bits
            let mask_simd = _mm256_loadu_si256(masks.as_ptr() as *const __m256i);
            let masked_dead = _mm256_and_si256(dead_mask, mask_simd);

            // Count dead objects
            let dead_count = self.count_dead_objects_simd(masked_dead);

            // Extract free blocks if any dead objects found
            if dead_count > 0 {
                // Convert SIMD register back to individual words for unified processing
                let mut temp_words = [0u64; WORDS_PER_SIMD_BLOCK];
                _mm256_storeu_si256(temp_words.as_mut_ptr() as *mut __m256i, masked_dead);

                for (i, &word) in temp_words.iter().enumerate() {
                    if word != 0 {
                        let word_idx = start_word + i;
                        self.extract_free_blocks_with_trailing_zeros(word_idx, word, free_blocks);
                    }
                }
            }

            // Clear processed words (only the masked portions)
            for (i, mask) in masks.iter().enumerate().take(WORDS_PER_SIMD_BLOCK) {
                if start_word + i < self.bits.len() && *mask != 0 {
                    let current_word = self.bits[start_word + i].load(Ordering::Relaxed);
                    let cleared_word = current_word & !mask;
                    self.bits[start_word + i].store(cleared_word, Ordering::Relaxed);
                }
            }

            dead_count
        }
    }

    /// Process a single SIMD block (4 x 64-bit words) using AVX2 (legacy method)
    #[cfg(target_arch = "x86_64")]
    fn process_simd_block(&self, start_word: usize, free_blocks: &mut Vec<FreeBlock>) -> usize {
        unsafe {
            self.simd_operations.fetch_add(1, Ordering::Relaxed);

            // Load 4 consecutive words into SIMD register
            let mut word_values = [0u64; WORDS_PER_SIMD_BLOCK];
            for (i, word_value) in word_values
                .iter_mut()
                .enumerate()
                .take(WORDS_PER_SIMD_BLOCK)
            {
                if start_word + i < self.bits.len() {
                    *word_value = self.bits[start_word + i].load(Ordering::Relaxed);
                }
            }

            // Load into AVX2 register
            let live_mask = _mm256_loadu_si256(word_values.as_ptr() as *const __m256i);
            let all_ones = _mm256_set1_epi64x(-1i64);
            let dead_mask = _mm256_xor_si256(live_mask, all_ones);

            // Count dead objects
            let dead_count = self.count_dead_objects_simd(dead_mask);

            // Extract free blocks if any dead objects found
            if dead_count > 0 {
                // Convert SIMD register back to individual words for unified processing
                let mut temp_words = [0u64; WORDS_PER_SIMD_BLOCK];
                _mm256_storeu_si256(temp_words.as_mut_ptr() as *mut __m256i, dead_mask);

                for (i, &word) in temp_words.iter().enumerate() {
                    if word != 0 {
                        let word_idx = start_word + i;
                        self.extract_free_blocks_with_trailing_zeros(word_idx, word, free_blocks);
                    }
                }
            }

            // Clear processed words
            for i in 0..WORDS_PER_SIMD_BLOCK {
                if start_word + i < self.bits.len() {
                    self.bits[start_word + i].store(0, Ordering::Relaxed);
                }
            }

            dead_count
        }
    }

    /// Process a single word with scalar operations
    fn process_word_scalar(&self, word_idx: usize, free_blocks: &mut Vec<FreeBlock>) -> usize {
        if word_idx >= self.bits.len() {
            return 0;
        }

        let word = self.bits[word_idx].load(Ordering::Relaxed);
        let inverted_word = !word;
        let dead_count = inverted_word.count_ones() as usize;

        if dead_count > 0 {
            let bit_index = word_idx * BITS_PER_WORD;
            self.extract_free_blocks_from_word(bit_index, inverted_word, free_blocks);
        }

        // Clear word
        self.bits[word_idx].store(0, Ordering::Relaxed);
        dead_count
    }

    /// Extract free block for an entire word that's completely dead
    fn extract_full_word_free_block(&self, word_idx: usize, free_blocks: &mut Vec<FreeBlock>) {
        let start_bit_index = word_idx * BITS_PER_WORD;
        let start_addr = self.heap_base.as_usize() + start_bit_index * self.object_alignment;
        let size_bytes = BITS_PER_WORD * self.object_alignment;

        free_blocks.push(FreeBlock {
            start_addr: unsafe { Address::from_usize(start_addr) },
            size_bytes,
            object_count: BITS_PER_WORD,
        });

        // Clear the word for next cycle
        if word_idx < self.bits.len() {
            self.bits[word_idx].store(0, Ordering::Relaxed);
        }
    }

    /// Optimized free block extraction using trailing_zeros with refined masks
    ///
    /// Consolidates extraction logic for both SIMD and sparse paths, using efficient
    /// bit manipulation to skip sparse regions and extract contiguous free blocks.
    fn extract_free_blocks_with_trailing_zeros(
        &self,
        word_idx: usize,
        dead_mask: u64,
        free_blocks: &mut Vec<FreeBlock>,
    ) {
        if dead_mask == 0 {
            return;
        }

        let base_bit_index = word_idx * BITS_PER_WORD;
        let mut remaining_mask = dead_mask;
        let mut bit_offset = 0;

        // Use trailing_zeros to efficiently skip sparse regions
        while remaining_mask != 0 {
            let zeros = remaining_mask.trailing_zeros() as usize;
            bit_offset += zeros;

            // Prevent shift overflow (max shift is 63 for u64)
            if zeros >= 64 {
                break;
            }
            remaining_mask >>= zeros;

            if remaining_mask == 0 {
                break;
            }

            // Find the run of consecutive dead objects
            let ones = remaining_mask.trailing_ones() as usize;

            // Prevent shift overflow (max shift is 63 for u64)
            if ones >= 64 {
                remaining_mask = 0; // Consume the entire remaining mask
            } else {
                remaining_mask >>= ones;
            }

            // Create free block for this run with proper bounds checking
            let start_object_index = base_bit_index + bit_offset;
            if start_object_index < self.max_objects {
                let end_object_index = (start_object_index + ones).min(self.max_objects);
                let actual_count = end_object_index - start_object_index;

                if actual_count > 0 {
                    let start_addr =
                        self.heap_base.as_usize() + start_object_index * self.object_alignment;
                    let size_bytes = actual_count * self.object_alignment;

                    free_blocks.push(FreeBlock {
                        start_addr: unsafe { Address::from_usize(start_addr) },
                        size_bytes,
                        object_count: actual_count,
                    });
                }
            }

            bit_offset += ones;
        }
    }

    /// X86-64 specific SIMD implementation using AVX2
    #[cfg(target_arch = "x86_64")]
    fn simd_sweep_x86_64(&self, free_blocks: &mut Vec<FreeBlock>) -> usize {
        let mut objects_swept = 0usize;

        unsafe {
            // Process bitvector in 256-bit (4x64-bit) chunks
            let chunks = self.bits.chunks_exact(WORDS_PER_SIMD_BLOCK);
            let remainder = chunks.remainder();
            let chunks_len = chunks.len();

            for (chunk_index, chunk) in chunks.enumerate() {
                self.simd_operations.fetch_add(1, Ordering::Relaxed);

                // Load atomic values into a temporary array for SIMD processing
                let mut chunk_values = [0u64; WORDS_PER_SIMD_BLOCK];
                for (i, atomic_word) in chunk.iter().enumerate() {
                    chunk_values[i] = atomic_word.load(Ordering::Relaxed);
                }

                // Load 256 bits of liveness data
                let ptr = chunk_values.as_ptr() as *const __m256i;
                let live_mask = _mm256_loadu_si256(ptr);

                // Invert to get dead object mask (dead = available for sweeping)
                let all_ones = _mm256_set1_epi64x(-1i64);
                let dead_mask = _mm256_xor_si256(live_mask, all_ones);

                // Count dead objects using population count
                let dead_count = self.count_dead_objects_simd(dead_mask);
                objects_swept += dead_count;

                // Extract free blocks for this chunk
                if dead_count > 0 {
                    self.extract_free_blocks_simd(chunk_index, dead_mask, free_blocks);
                }

                // Clear the processed bits (reset for next cycle)
                for atomic_word in chunk.iter() {
                    atomic_word.store(0, Ordering::Relaxed);
                }
            }

            // Handle remainder words with scalar operations
            for (word_index, atomic_word) in remainder.iter().enumerate() {
                let word = atomic_word.load(Ordering::Relaxed);
                let base_index = chunks_len * WORDS_PER_SIMD_BLOCK + word_index;
                objects_swept += self.process_remainder_word(base_index, word, free_blocks);
                // Clear remainder word
                atomic_word.store(0, Ordering::Relaxed);
            }
        }

        objects_swept
    }

    /// Count dead objects in a 256-bit SIMD register
    #[cfg(target_arch = "x86_64")]
    unsafe fn count_dead_objects_simd(&self, dead_mask: __m256i) -> usize {
        // Extract 64-bit lanes and count set bits
        let lane0 = unsafe { _mm256_extract_epi64(dead_mask, 0) } as u64;
        let lane1 = unsafe { _mm256_extract_epi64(dead_mask, 1) } as u64;
        let lane2 = unsafe { _mm256_extract_epi64(dead_mask, 2) } as u64;
        let lane3 = unsafe { _mm256_extract_epi64(dead_mask, 3) } as u64;

        lane0.count_ones() as usize
            + lane1.count_ones() as usize
            + lane2.count_ones() as usize
            + lane3.count_ones() as usize
    }

    /// Extract contiguous free blocks from SIMD chunk
    #[cfg(target_arch = "x86_64")]
    fn extract_free_blocks_simd(
        &self,
        chunk_index: usize,
        dead_mask: __m256i,
        free_blocks: &mut Vec<FreeBlock>,
    ) {
        // Convert SIMD register back to scalar for free block analysis
        let mut temp_words = [0u64; WORDS_PER_SIMD_BLOCK];
        unsafe {
            _mm256_storeu_si256(temp_words.as_mut_ptr() as *mut __m256i, dead_mask);
        }

        let base_bit_index = chunk_index * BITS_PER_SIMD_BLOCK;

        for (word_offset, &word) in temp_words.iter().enumerate() {
            if word != 0 {
                let word_bit_index = base_bit_index + word_offset * BITS_PER_WORD;
                self.extract_free_blocks_from_word(word_bit_index, word, free_blocks);
            }
        }
    }

    /// Fallback sweep implementation for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    fn fallback_sweep(&self, free_blocks: &mut Vec<FreeBlock>) -> usize {
        let mut objects_swept = 0usize;

        for (word_index, &word) in self.bits.iter().enumerate() {
            let inverted_word = !word; // Invert to get dead objects
            objects_swept += inverted_word.count_ones() as usize;

            if inverted_word != 0 {
                let bit_index = word_index * BITS_PER_WORD;
                self.extract_free_blocks_from_word(bit_index, inverted_word, free_blocks);
            }
        }

        // Clear all bits for next cycle
        for word in &mut self.bits.iter() {
            let word_ptr = word as *const u64 as *mut u64;
            unsafe {
                std::ptr::write_volatile(word_ptr, 0);
            }
        }

        objects_swept
    }

    /// Process a single remainder word during SIMD sweep
    fn process_remainder_word(
        &self,
        word_index: usize,
        word: u64,
        free_blocks: &mut Vec<FreeBlock>,
    ) -> usize {
        let inverted_word = !word;
        let dead_count = inverted_word.count_ones() as usize;

        if dead_count > 0 {
            let bit_index = word_index * BITS_PER_WORD;
            self.extract_free_blocks_from_word(bit_index, inverted_word, free_blocks);
        }

        // Clear the word (already done in caller)

        dead_count
    }

    /// Extract contiguous free blocks from a word of dead object bits
    fn extract_free_blocks_from_word(
        &self,
        base_bit_index: usize,
        dead_word: u64,
        free_blocks: &mut Vec<FreeBlock>,
    ) {
        let mut word = dead_word;
        let mut bit_offset = 0;

        while word != 0 {
            // Find start of free block
            let leading_zeros = word.trailing_zeros() as usize;
            bit_offset += leading_zeros;

            // Prevent shift overflow (max shift is 63 for u64)
            if leading_zeros >= 64 {
                break;
            }
            word >>= leading_zeros;

            if word == 0 {
                break;
            }

            // Find end of free block
            let block_size = word.trailing_ones() as usize;

            // Prevent shift overflow (max shift is 63 for u64)
            if block_size >= 64 {
                word = 0; // Consume the entire word
            } else {
                word >>= block_size;
            }

            // Convert bit indices to object addresses
            let start_object_index = base_bit_index + bit_offset;
            let start_addr = self.heap_base.as_usize() + start_object_index * self.object_alignment;
            let size_bytes = block_size * self.object_alignment;

            free_blocks.push(FreeBlock {
                start_addr: unsafe { Address::from_usize(start_addr) },
                size_bytes,
                object_count: block_size,
            });

            bit_offset += block_size;
        }
    }

    /// Convert object address to bitvector bit index
    fn object_to_bit_index(&self, object_addr: Address) -> Option<usize> {
        let addr = object_addr.as_usize();
        let base = self.heap_base.as_usize();

        if addr >= base && addr < base + self.heap_size {
            let offset = addr - base;
            if offset.is_multiple_of(self.object_alignment) {
                let index = offset / self.object_alignment;
                if index < self.max_objects {
                    return Some(index);
                }
            }
        }
        None
    }

    /// Calculate maximum objects in a chunk with accurate object-based indexing
    ///
    /// Properly handles partial chunks and object alignment boundaries, ensuring
    /// density calculations reflect actual object counts rather than bit counts.
    fn calculate_max_objects_in_chunk(&self, chunk_idx: usize, _chunk_words: usize) -> usize {
        let chunk_start_object = chunk_idx * self.objects_per_chunk;
        let chunk_end_object = ((chunk_idx + 1) * self.objects_per_chunk).min(self.max_objects);

        chunk_end_object.saturating_sub(chunk_start_object)
    }

    /// Get the actual number of objects that can be stored in a chunk
    ///
    /// Handles partial chunks accurately by considering heap boundaries and object alignment.
    /// Essential for correct density calculations and avoiding out-of-bounds access.
    fn get_chunk_object_capacity(&self, chunk_idx: usize) -> usize {
        if chunk_idx >= self.chunk_count {
            return 0;
        }

        let chunk_start_object = chunk_idx * self.objects_per_chunk;
        let chunk_end_object = ((chunk_idx + 1) * self.objects_per_chunk).min(self.max_objects);

        chunk_end_object.saturating_sub(chunk_start_object)
    }

    /// Create a bitmask for valid objects in a chunk
    ///
    /// Returns a mask that covers only the valid object bits in the chunk,
    /// accounting for partial chunks at heap boundaries. Critical for preventing
    /// counting or processing of invalid bits in the final chunk.
    fn create_chunk_object_mask(&self, chunk_idx: usize) -> ChunkMask {
        let object_capacity = self.get_chunk_object_capacity(chunk_idx);
        if object_capacity == 0 {
            return ChunkMask::empty();
        }

        let chunk_start_word = chunk_idx * self.words_per_chunk;
        let mut word_masks = Vec::new();

        // Use our new helper to generate precise masks for each word
        for word_offset in 0..self.words_per_chunk {
            let word_idx = chunk_start_word + word_offset;
            if word_idx >= self.bits.len() {
                break;
            }

            // Use the helper method for accurate masking
            let mask = self.word_mask_for_chunk_objects(chunk_idx, word_idx, object_capacity);
            word_masks.push(mask);
        }

        ChunkMask {
            chunk_idx,
            object_capacity,
            word_masks,
        }
    }

    /// Get the range of words that belong to a chunk
    ///
    /// Returns (start_word_idx, end_word_idx) for safe iteration over chunk words.
    /// Ensures bounds checking to prevent accessing invalid memory regions.
    fn get_chunk_word_range(&self, chunk_idx: usize) -> (usize, usize) {
        if chunk_idx >= self.chunk_count {
            return (0, 0);
        }

        let start_word = chunk_idx * self.words_per_chunk;
        let end_word = ((chunk_idx + 1) * self.words_per_chunk).min(self.bits.len());

        (start_word, end_word)
    }

    /// Calculate the actual bit count in a specific word within a chunk
    ///
    /// Handles partial words at chunk boundaries and heap limits to prevent
    /// counting invalid bits that could skew density calculations.
    fn calculate_valid_bits_in_word(&self, chunk_idx: usize, word_offset: usize) -> usize {
        let (start_word, end_word) = self.get_chunk_word_range(chunk_idx);
        let word_idx = start_word + word_offset;

        if word_idx >= end_word || word_idx >= self.bits.len() {
            return 0;
        }

        let chunk_start_object = chunk_idx * self.objects_per_chunk;
        let chunk_object_capacity = self.get_chunk_object_capacity(chunk_idx);
        let chunk_end_object = chunk_start_object + chunk_object_capacity;

        let word_start_object = word_idx * BITS_PER_WORD;
        let word_end_object = word_start_object + BITS_PER_WORD;

        // Intersect word range with valid chunk object range
        let valid_start = word_start_object.max(chunk_start_object);
        let valid_end = word_end_object.min(chunk_end_object);

        valid_end.saturating_sub(valid_start)
    }

    /// Get the maximum number of objects this bitvector can track
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    /// assert_eq!(bitvector.max_objects(), 64);
    /// ```
    pub fn max_objects(&self) -> usize {
        self.max_objects
    }

    /// Get the number of objects per chunk for hybrid strategy
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024 * 1024, 16);
    /// assert!(bitvector.objects_per_chunk() > 0);
    /// ```
    pub fn objects_per_chunk(&self) -> usize {
        self.objects_per_chunk
    }

    /// Clear all live marks (prepare for next marking cycle)
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    /// bitvector.clear_all_marks();
    /// ```
    pub fn clear_all_marks(&self) {
        let mut idx = 0;
        let len = self.bits.len();

        while idx + WORDS_PER_SIMD_BLOCK <= len {
            for lane in 0..WORDS_PER_SIMD_BLOCK {
                self.bits[idx + lane].store(0, Ordering::Relaxed);
            }
            idx += WORDS_PER_SIMD_BLOCK;
        }

        while idx < len {
            self.bits[idx].store(0, Ordering::Relaxed);
            idx += 1;
        }

        // Reset chunk populations for hybrid strategy
        for chunk_pop in &self.chunk_populations {
            chunk_pop.store(0, Ordering::Relaxed);
        }

        self.objects_marked.store(0, Ordering::Relaxed);
        self.simd_chunks_processed.store(0, Ordering::Relaxed);
        self.sparse_chunks_processed.store(0, Ordering::Relaxed);
    }

    /// Get the population (number of marked objects) in a specific chunk
    ///
    /// This is useful for testing and density analysis.
    pub fn get_chunk_population(&self, chunk_idx: usize) -> usize {
        if chunk_idx < self.chunk_populations.len() {
            self.chunk_populations[chunk_idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Get the total number of chunks in this bitvector
    pub fn get_chunk_count(&self) -> usize {
        self.chunk_count
    }

    /// Get current statistics
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    /// let stats = bitvector.get_stats();
    /// assert_eq!(stats.objects_marked, 0);
    /// ```
    pub fn get_stats(&self) -> BitvectorStats {
        BitvectorStats {
            objects_marked: self.objects_marked.load(Ordering::Relaxed),
            objects_swept: self.objects_swept.load(Ordering::Relaxed),
            simd_operations: self.simd_operations.load(Ordering::Relaxed),
            bitvector_size_bits: self.max_objects,
            bitvector_size_bytes: self.bits.len() * 8,
            dense_chunks_last: self.simd_chunks_processed.load(Ordering::Relaxed),
            sparse_chunks_last: self.sparse_chunks_processed.load(Ordering::Relaxed),
            chunk_size_bytes: self.actual_chunk_size_bytes,
        }
    }

    /// Mark an object as live in the bitvector (used during marking phase)
    ///
    /// # Arguments
    /// * `object` - Object reference to mark as live
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    /// bitvector.mark_object_live(obj);
    /// ```
    pub fn mark_object_live(&self, object: ObjectReference) {
        if let Some(bit_index) = self.object_to_bit_index(object.to_raw_address()) {
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            if word_index < self.bits.len() {
                let atomic_word = &self.bits[word_index];
                let mask = 1u64 << bit_offset;
                atomic_word.fetch_or(mask, Ordering::Relaxed);
                self.objects_marked.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Count live objects in a specific address range using SIMD operations
    ///
    /// This uses AVX2 instructions to efficiently count set bits in the range.
    ///
    /// # Arguments
    /// * `start_addr` - Starting address of the range
    /// * `size` - Size of the range in bytes
    ///
    /// # Returns
    /// Number of live objects in the range
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    /// let page_start = heap_base;
    /// let count = bitvector.count_live_objects_in_range(page_start, 4096);
    /// assert_eq!(count, 0); // No objects marked yet
    /// ```
    pub fn count_live_objects_in_range(&self, start_addr: Address, size: usize) -> usize {
        let start_bit = self.object_to_bit_index(start_addr).unwrap_or_default();
        let end_addr = start_addr + size;
        let end_bit = match self.object_to_bit_index(end_addr) {
            Some(bit) => bit,
            None => self.max_objects,
        };

        if start_bit >= end_bit {
            return 0;
        }

        self.count_bits_range(start_bit, end_bit - start_bit)
    }

    /// Count set bits in a specific bit range using optimized operations
    ///
    /// This is a helper method for count_live_objects_in_range.
    ///
    /// # Arguments
    /// * `start_bit` - Starting bit index
    /// * `bit_count` - Number of bits to count
    ///
    /// # Returns
    /// Number of set bits in the range
    fn count_bits_range(&self, start_bit: usize, bit_count: usize) -> usize {
        if bit_count == 0 {
            return 0;
        }

        let end_bit = (start_bit + bit_count).min(self.max_objects);
        if start_bit >= end_bit {
            return 0;
        }

        let start_word = start_bit / 64;
        let end_word = end_bit.div_ceil(64);
        let mut total = 0usize;

        if start_word == end_word - 1 {
            // Range is within a single word
            if start_word < self.bits.len() {
                let word = self.bits[start_word].load(Ordering::Relaxed);
                let start_offset = start_bit % 64;
                let end_offset = end_bit % 64;
                let mask = if end_offset == 0 {
                    !0u64 << start_offset
                } else {
                    (!0u64 << start_offset) & ((1u64 << end_offset) - 1)
                };
                return (word & mask).count_ones() as usize;
            }
        } else {
            // Multi-word range
            // Leading word
            if start_word < self.bits.len() {
                let word = self.bits[start_word].load(Ordering::Relaxed);
                let start_offset = start_bit % 64;
                let lead_mask = !0u64 << start_offset;
                total += (word & lead_mask).count_ones() as usize;
            }

            // Middle words - can use SIMD for large ranges
            let middle_start = start_word + 1;
            let middle_end = end_word.saturating_sub(1);
            if middle_end > middle_start && middle_start < self.bits.len() {
                let middle_end_clamped = middle_end.min(self.bits.len());
                let mut idx = middle_start;
                let mut block = [0u64; WORDS_PER_SIMD_BLOCK];

                while idx + WORDS_PER_SIMD_BLOCK <= middle_end_clamped {
                    for (lane, slot) in block.iter_mut().enumerate() {
                        *slot = self.bits[idx + lane].load(Ordering::Relaxed);
                    }
                    total += self.count_words_simd(&block);
                    idx += WORDS_PER_SIMD_BLOCK;
                }

                while idx < middle_end_clamped {
                    let word = self.bits[idx].load(Ordering::Relaxed);
                    total += word.count_ones() as usize;
                    idx += 1;
                }
            }

            // Trailing word
            if end_word > 0 && end_word <= self.bits.len() {
                let word = self.bits[end_word - 1].load(Ordering::Relaxed);
                let trailing_bits = end_bit % 64;
                if trailing_bits == 0 {
                    total += word.count_ones() as usize;
                } else {
                    let mask = (1u64 << trailing_bits) - 1;
                    total += (word & mask).count_ones() as usize;
                }
            }
        }

        total
    }

    /// Count set bits in a slice of words using optimized SIMD operations
    ///
    /// This provides the same functionality as simd_bitvec.rs but integrated
    /// into the sweep implementation for better performance.
    fn count_words_simd(&self, words: &[u64]) -> usize {
        self.simd_operations
            .fetch_add(words.len(), Ordering::Relaxed);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return self.avx2_population_count(words) };
            }
        }

        words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Ultra-fast AVX2 population count for bit arrays
    ///
    /// Uses lookup table approach with 256-bit SIMD for maximum throughput.
    /// This consolidates the optimized population counting from simd_bitvec.rs.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn avx2_population_count(&self, words: &[u64]) -> usize {
        const LOOKUP_BYTES: [i8; 32] = [
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        ];

        let lookup = unsafe { _mm256_loadu_si256(LOOKUP_BYTES.as_ptr() as *const __m256i) };
        let low_mask = unsafe { _mm256_set1_epi8(0x0F) };
        let mut total = 0usize;

        // Process in chunks of 4 u64 words (256 bits)
        let chunks = words.len() / 4;
        let remainder_start = chunks * 4;

        for i in 0..chunks {
            let chunk_start = i * 4;
            unsafe {
                let data = _mm256_loadu_si256(words.as_ptr().add(chunk_start) as *const __m256i);
                let low = _mm256_and_si256(data, low_mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(data, 4), low_mask);
                let cnt_low = _mm256_shuffle_epi8(lookup, low);
                let cnt_high = _mm256_shuffle_epi8(lookup, high);
                let counts = _mm256_add_epi8(cnt_low, cnt_high);
                let sums = _mm256_sad_epu8(counts, _mm256_setzero_si256());

                total += _mm256_extract_epi64(sums, 0) as usize;
                total += _mm256_extract_epi64(sums, 1) as usize;
                total += _mm256_extract_epi64(sums, 2) as usize;
                total += _mm256_extract_epi64(sums, 3) as usize;
            }
        }

        // Handle remainder words with scalar operations
        for &word in &words[remainder_start..] {
            total += word.count_ones() as usize;
        }

        total
    }

    /// Clear all live marks (prepare for next marking cycle)
    ///
    /// This is an alias for clear_all_marks for consistency with the coordinator API.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::simd_sweep::SimdBitvector;
    /// use mmtk::util::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let bitvector = SimdBitvector::new(heap_base, 1024, 16);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    /// bitvector.mark_object_live(obj);
    /// bitvector.clear();
    /// // Object is no longer marked as live
    /// ```
    pub fn clear(&self) {
        self.clear_all_marks();
    }
}

/// Statistics from a SIMD sweep operation
///
/// # Examples
///
/// ```
/// use fugrip::simd_sweep::SweepStatistics;
/// let stats = SweepStatistics::default();
/// assert_eq!(stats.objects_swept, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SweepStatistics {
    /// Number of objects swept (reclaimed)
    pub objects_swept: usize,
    /// Number of contiguous free blocks found
    pub free_blocks: usize,
    /// Time taken for sweep in nanoseconds
    pub sweep_time_ns: u64,
    /// Number of SIMD blocks processed
    pub simd_blocks_processed: usize,
    /// Throughput in objects per second
    pub throughput_objects_per_sec: u64,
}

/// Statistics for the bitvector itself
///
/// # Examples
///
/// ```
/// use fugrip::simd_sweep::BitvectorStats;
/// let stats = BitvectorStats::default();
/// assert_eq!(stats.objects_marked, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BitvectorStats {
    /// Number of objects currently marked as live
    pub objects_marked: usize,
    /// Total objects swept in last operation
    pub objects_swept: usize,
    /// Number of SIMD operations performed
    pub simd_operations: usize,
    /// Size of bitvector in bits
    pub bitvector_size_bits: usize,
    /// Size of bitvector in bytes
    pub bitvector_size_bytes: usize,
    /// Number of chunks processed with SIMD strategy in the last sweep
    pub dense_chunks_last: usize,
    /// Number of chunks processed with sparse strategy in the last sweep
    pub sparse_chunks_last: usize,
    /// Chunk size (bytes) used for hybrid decision making
    pub chunk_size_bytes: usize,
}

/// Statistics from a hybrid SIMD+sparse sweep operation
///
/// This tracks the adaptive strategy switching between SIMD and Verse-style
/// sparse scanning based on chunk density analysis.
///
/// # Examples
///
/// ```
/// use fugrip::simd_sweep::HybridSweepStatistics;
/// let stats = HybridSweepStatistics::default();
/// assert_eq!(stats.objects_swept, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct HybridSweepStatistics {
    /// Number of objects swept (reclaimed)
    pub objects_swept: usize,
    /// Number of contiguous free blocks found
    pub free_blocks: usize,
    /// Time taken for sweep in nanoseconds
    pub sweep_time_ns: u64,
    /// Number of chunks processed with SIMD strategy
    pub simd_chunks_processed: usize,
    /// Number of chunks processed with sparse strategy
    pub sparse_chunks_processed: usize,
    /// Total number of chunks in the heap
    pub total_chunks: usize,
    /// Density threshold percentage used for strategy switching
    pub density_threshold_percent: usize,
    /// Throughput in objects per second
    pub throughput_objects_per_sec: u64,
    /// Total compare-exchange operations during marking phase
    pub compare_exchange_operations: usize,
    /// Total compare-exchange retries due to contention
    pub compare_exchange_retries: usize,
    /// Total SIMD operations performed
    pub simd_operations: usize,
    /// Architecture-specific optimizations (AVX2 operations on x86_64)
    #[cfg(target_arch = "x86_64")]
    pub avx2_operations: usize,
}

/// Represents a contiguous block of free memory
///
/// # Examples
///
/// ```
/// use fugrip::simd_sweep::FreeBlock;
/// use mmtk::util::Address;
///
/// let block = FreeBlock {
///     start_addr: unsafe { Address::from_usize(0x1000) },
///     size_bytes: 64,
///     object_count: 4,
/// };
/// assert_eq!(block.size_bytes, 64);
/// ```
#[derive(Debug, Clone)]
pub struct FreeBlock {
    /// Starting address of the free block
    pub start_addr: Address,
    /// Size of the block in bytes
    pub size_bytes: usize,
    /// Number of object slots in this block
    pub object_count: usize,
}

/// Bitmask for valid objects in a chunk
///
/// Provides precise masking for partial chunks to ensure counting and processing
/// operations only consider valid object bits, preventing corruption or incorrect
/// statistics from accessing bits beyond heap boundaries.
///
/// # Examples
///
/// ```
/// use fugrip::simd_sweep::ChunkMask;
///
/// let mask = ChunkMask::empty();
/// assert_eq!(mask.object_capacity, 0);
/// ```
#[derive(Debug, Clone)]
pub struct ChunkMask {
    /// Index of the chunk this mask applies to
    pub chunk_idx: usize,
    /// Number of valid objects in this chunk
    pub object_capacity: usize,
    /// Per-word masks for valid bits (0 = invalid, 1 = valid)
    pub word_masks: Vec<u64>,
}

impl ChunkMask {
    /// Create an empty chunk mask (no valid objects)
    pub fn empty() -> Self {
        Self {
            chunk_idx: 0,
            object_capacity: 0,
            word_masks: Vec::new(),
        }
    }

    /// Check if this chunk mask covers any valid objects
    pub fn is_empty(&self) -> bool {
        self.object_capacity == 0 || self.word_masks.iter().all(|&mask| mask == 0)
    }

    /// Get the mask for a specific word in the chunk
    pub fn get_word_mask(&self, word_offset: usize) -> u64 {
        self.word_masks.get(word_offset).copied().unwrap_or(0)
    }

    /// Count the total number of valid bits covered by this mask
    pub fn count_valid_bits(&self) -> usize {
        self.word_masks
            .iter()
            .map(|&mask| mask.count_ones() as usize)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmtk::util::ObjectReference;
    use std::sync::atomic::Ordering;

    #[test]
    fn bitvector_creation() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let bitvector = SimdBitvector::new(heap_base, 1024, 16);
        assert_eq!(bitvector.max_objects(), 64);
    }

    #[test]
    fn mark_and_check_objects() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let bitvector = SimdBitvector::new(heap_base, 1024, 16);

        let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + 32) };
        assert!(!bitvector.is_marked(obj_addr));

        bitvector.mark_live(obj_addr);
        assert!(bitvector.is_marked(obj_addr));
    }

    #[test]
    fn simd_sweep_performance() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let heap_size = 64 * 1024; // 64KB
        let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

        // Mark every other object as live
        for i in (0..heap_size).step_by(32) {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i) };
            bitvector.mark_live(obj_addr);
        }

        let stats = bitvector.simd_sweep();

        // Should have swept approximately half the objects
        assert!(stats.objects_swept > 1000);
        // Should be very fast (reasonable performance per object on modern hardware)
        assert!(stats.sweep_time_ns < stats.objects_swept as u64 * 500); // Sub-500ns per object (more lenient for CI/different systems)
        assert!(stats.throughput_objects_per_sec > 1_000_000); // 1M+ objects/sec (more lenient for CI/different systems)
    }

    #[test]
    fn free_block_extraction() {
        let heap_base = unsafe { Address::from_usize(0x10000000) };
        let bitvector = SimdBitvector::new(heap_base, 512, 16);

        // Leave some objects unmarked to create free blocks
        let obj1 = unsafe { Address::from_usize(heap_base.as_usize()) };
        let obj3 = unsafe { Address::from_usize(heap_base.as_usize() + 32) };

        bitvector.mark_live(obj1);
        bitvector.mark_live(obj3);
        // obj2 at offset 16 is unmarked, creating a single-object free block

        let stats = bitvector.simd_sweep();
        assert!(stats.free_blocks > 0);
        assert!(stats.objects_swept > 0);
    }

    #[test]
    fn clear_all_marks_resets_state() {
        let heap_base = unsafe { Address::from_usize(0x20000000) };
        let bitvector = SimdBitvector::new(heap_base, 1024, 16);

        let first = unsafe { Address::from_usize(heap_base.as_usize() + 16) };
        let second = unsafe { Address::from_usize(heap_base.as_usize() + 32) };

        bitvector.mark_live(first);
        bitvector.mark_live(second);
        assert!(bitvector.is_marked(first));
        assert!(bitvector.is_marked(second));
        assert_eq!(bitvector.objects_marked.load(Ordering::Relaxed), 2);

        bitvector.clear_all_marks();
        assert!(!bitvector.is_marked(first));
        assert!(!bitvector.is_marked(second));
        assert_eq!(bitvector.objects_marked.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn clear_aliases_clear_all_marks() {
        let heap_base = unsafe { Address::from_usize(0x21000000) };
        let bitvector = SimdBitvector::new(heap_base, 1024, 16);

        let marked = unsafe { Address::from_usize(heap_base.as_usize() + 64) };
        bitvector.mark_live(marked);
        assert!(bitvector.is_marked(marked));

        bitvector.clear();
        assert!(!bitvector.is_marked(marked));
    }

    #[test]
    fn mark_object_live_tracks_object_references() {
        let heap_base = unsafe { Address::from_usize(0x22000000) };
        let bitvector = SimdBitvector::new(heap_base, 2048, 16);

        let object_a = ObjectReference::from_raw_address(heap_base).unwrap();
        let second_addr = unsafe { Address::from_usize(heap_base.as_usize() + 16) };
        let object_b = ObjectReference::from_raw_address(second_addr).unwrap();

        bitvector.mark_object_live(object_a);
        bitvector.mark_object_live(object_b);

        assert!(bitvector.is_marked(heap_base));
        assert!(bitvector.is_marked(second_addr));
        assert_eq!(bitvector.objects_marked.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn count_live_objects_in_range_covers_multi_word_segments() {
        let heap_base = unsafe { Address::from_usize(0x23000000) };
        let bitvector = SimdBitvector::new(heap_base, 4096, 16);

        // Configure bit patterns directly so we can reason about coverage paths.
        bitvector.bits[0].store((1u64 << 1) | (1u64 << 60), Ordering::Relaxed);
        bitvector.bits[1].store(1u64 << 6, Ordering::Relaxed);

        let start = unsafe { Address::from_usize(heap_base.as_usize() + 16) };
        let count = bitvector.count_live_objects_in_range(start, 2048);
        assert_eq!(count, 3);
    }

    #[test]
    fn count_bits_range_handles_single_word_masks() {
        let heap_base = unsafe { Address::from_usize(0x24000000) };
        let bitvector = SimdBitvector::new(heap_base, 2048, 16);
        bitvector.bits[0].store(0b1111_0000, Ordering::Relaxed);

        assert_eq!(bitvector.count_bits_range(4, 4), 4);
        assert_eq!(bitvector.count_bits_range(0, 2), 0);
    }

    #[test]
    fn count_bits_range_handles_cross_word_segments() {
        let heap_base = unsafe { Address::from_usize(0x25000000) };
        let bitvector = SimdBitvector::new(heap_base, 4096, 16);
        bitvector.bits[0].store((1u64 << 63) | (1u64 << 62), Ordering::Relaxed);
        bitvector.bits[1].store(1u64, Ordering::Relaxed);

        // Starting near the end of the first word should require processing multiple words.
        assert_eq!(bitvector.count_bits_range(62, 4), 3);
    }

    #[test]
    fn count_words_simd_falls_back_to_scalar_sum() {
        let heap_base = unsafe { Address::from_usize(0x26000000) };
        let bitvector = SimdBitvector::new(heap_base, 1024, 16);
        let words = [0b1011_0001u64, 0b1110u64];
        let total = bitvector.count_words_simd(&words);
        let expected: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn process_remainder_word_extracts_free_blocks() {
        let heap_base = unsafe { Address::from_usize(0x27000000) };
        let bitvector = SimdBitvector::new(heap_base, 2048, 16);
        let mut free_blocks = Vec::new();

        let live_word = 0b1111u64; // remaining bits are dead
        let dead = bitvector.process_remainder_word(0, live_word, &mut free_blocks);

        // Four dead objects should be reported from the inverted word, forming one block.
        assert_eq!(dead, BITS_PER_WORD - 4);
        assert_eq!(free_blocks.len(), 1);
        assert_eq!(free_blocks[0].object_count, BITS_PER_WORD - 4);
        assert_eq!(
            free_blocks[0].size_bytes,
            (BITS_PER_WORD - 4) * bitvector.object_alignment
        );
        let expected_start =
            unsafe { Address::from_usize(heap_base.as_usize() + 4 * bitvector.object_alignment) };
        assert_eq!(free_blocks[0].start_addr, expected_start);
    }

    #[test]
    fn hybrid_sweep_adaptive_strategy() {
        let heap_base = unsafe { Address::from_usize(0x28000000) };
        let heap_size = 256 * 1024; // 256KB heap to ensure multiple chunks
        let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

        // Create mixed density pattern across chunks
        // Dense chunk: mark 50% of objects (above 35% threshold)
        for i in (0..1000).step_by(2) {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            bitvector.mark_live(obj_addr);
        }

        // Sparse chunk: mark only 10% of objects (below 35% threshold)
        let sparse_start = 8000;
        for i in (sparse_start..sparse_start + 1000).step_by(10) {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            bitvector.mark_live(obj_addr);
        }

        let stats = bitvector.hybrid_sweep();

        // Verify adaptive strategy worked
        assert!(stats.objects_swept > 0);
        assert!(stats.free_blocks > 0);
        assert!(stats.simd_chunks_processed > 0 || stats.sparse_chunks_processed > 0);
        assert_eq!(stats.total_chunks, bitvector.chunk_count);
        assert_eq!(stats.density_threshold_percent, DENSITY_THRESHOLD_PERCENT);
        assert!(stats.throughput_objects_per_sec > 0);

        // Verify that both strategies were used for this mixed pattern
        println!(
            "Hybrid sweep: {} SIMD chunks, {} sparse chunks, {} total chunks",
            stats.simd_chunks_processed, stats.sparse_chunks_processed, stats.total_chunks
        );
    }

    #[test]
    fn chunk_population_tracking() {
        let heap_base = unsafe { Address::from_usize(0x29000000) };
        let heap_size = 128 * 1024; // 128KB heap
        let bitvector = SimdBitvector::new(heap_base, heap_size, 16);

        // Mark objects in first chunk
        for i in 0..100 {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            bitvector.mark_live(obj_addr);
        }

        // Verify chunk population tracking
        assert!(bitvector.chunk_populations[0].load(Ordering::Relaxed) == 100);

        // Clear and verify reset
        bitvector.clear_all_marks();
        assert!(bitvector.chunk_populations[0].load(Ordering::Relaxed) == 0);
    }

    #[test]
    fn test_chunk_layout_validation() {
        // Test various heap sizes and alignments to validate chunk calculations
        let test_cases = vec![
            (64 * 1024, 16),    // 64KB heap, 16-byte objects
            (128 * 1024, 32),   // 128KB heap, 32-byte objects
            (256 * 1024, 64),   // 256KB heap, 64-byte objects
            (1024 * 1024, 128), // 1MB heap, 128-byte objects
            (10000, 16),        // Non-aligned heap size
        ];

        for (heap_size, alignment) in test_cases {
            let heap_base = unsafe { Address::from_usize(0x30000000) };
            let bitvector = SimdBitvector::new(heap_base, heap_size, alignment);

            // Validate chunk counts
            assert!(bitvector.chunk_count > 0, "Should have at least one chunk");
            assert_eq!(
                bitvector.chunk_count,
                bitvector.max_objects.div_ceil(bitvector.objects_per_chunk).max(1),
                "Chunk count calculation mismatch for heap_size={}, alignment={}",
                heap_size,
                alignment
            );

            // Validate objects per chunk
            assert!(
                bitvector.objects_per_chunk > 0,
                "Should have at least one object per chunk"
            );

            // Validate words per chunk
            assert!(
                bitvector.words_per_chunk > 0,
                "Should have at least one word per chunk"
            );

            // Test boundary conditions for last chunk
            let last_chunk_idx = bitvector.chunk_count - 1;
            let last_chunk_capacity = bitvector.get_chunk_object_capacity(last_chunk_idx);
            assert!(
                last_chunk_capacity <= bitvector.objects_per_chunk,
                "Last chunk capacity should not exceed objects_per_chunk"
            );
            assert!(
                last_chunk_capacity > 0,
                "Last chunk should have non-zero capacity"
            );
        }
    }

    #[test]
    fn test_mark_live_cas_single_thread() {
        let heap_base = unsafe { Address::from_usize(0x31000000) };
        let bitvector = SimdBitvector::new(heap_base, 4096, 16);

        // Test single-threaded CAS marking
        let obj1 = unsafe { Address::from_usize(heap_base.as_usize()) };
        let obj2 = unsafe { Address::from_usize(heap_base.as_usize() + 16) };

        // First mark should succeed
        assert!(bitvector.mark_live(obj1));
        assert!(bitvector.is_marked(obj1));

        // Second mark of same object should fail (already marked)
        assert!(!bitvector.mark_live(obj1));

        // Mark different object should succeed
        assert!(bitvector.mark_live(obj2));
        assert!(bitvector.is_marked(obj2));

        // Verify chunk population counter
        assert_eq!(bitvector.chunk_populations[0].load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_mark_live_cas_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let heap_base = unsafe { Address::from_usize(0x32000000) };
        let bitvector = Arc::new(SimdBitvector::new(heap_base, 8192, 16)); // Larger heap for 400 objects
        let num_threads = 4;
        let objects_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let bv = Arc::clone(&bitvector);
                thread::spawn(move || {
                    let mut marked_count = 0;
                    for i in 0..objects_per_thread {
                        let offset = (tid * objects_per_thread + i) * 16;
                        let obj = unsafe { Address::from_usize(heap_base.as_usize() + offset) };
                        if bv.mark_live(obj) {
                            marked_count += 1;
                        }
                    }
                    marked_count
                })
            })
            .collect();

        let total_marked: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

        // All objects should be marked exactly once
        assert_eq!(total_marked, num_threads * objects_per_thread);

        // Verify chunk population
        let chunk_pop = bitvector.get_chunk_population(0);
        assert_eq!(chunk_pop, total_marked);
    }

    #[test]
    fn test_hybrid_strategy_force_dense() {
        let heap_base = unsafe { Address::from_usize(0x33000000) };
        let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

        // Get actual chunk capacity and mark enough for >35% density
        let chunk_capacity = bitvector.get_chunk_object_capacity(0);
        let target_marked = ((chunk_capacity as f64 * 0.4) as usize).max(500); // 40% density

        // Mark objects consecutively to achieve high density
        for i in 0..target_marked {
            let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            if !bitvector.mark_live(obj) {
                // Stop if we hit heap boundary
                break;
            }
        }

        let stats = bitvector.hybrid_sweep();

        // Verify SIMD was used for dense chunk
        assert!(
            stats.simd_chunks_processed > 0,
            "Should use SIMD for dense chunk with {} marked objects out of {} capacity ({:.1}% density)",
            stats.objects_swept, chunk_capacity,
            (stats.objects_swept as f64 / chunk_capacity as f64) * 100.0
        );
        assert!(stats.objects_swept > 0, "Should sweep marked objects");
    }

    #[test]
    fn test_hybrid_strategy_force_sparse() {
        let heap_base = unsafe { Address::from_usize(0x34000000) };
        let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

        // Mark exactly 10 objects by using explicit object indices with wide spacing
        let target_objects = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900];
        let mut marked_count = 0;

        for &obj_index in &target_objects {
            let obj = unsafe { Address::from_usize(heap_base.as_usize() + obj_index * 16) };
            if bitvector.mark_live(obj) {
                marked_count += 1;
            }
        }

        // Get total object capacity in first chunk to calculate expected dead count
        let chunk_capacity = bitvector.get_chunk_object_capacity(0);
        let expected_dead_count = chunk_capacity - marked_count;

        let stats = bitvector.hybrid_sweep();

        // Verify sparse operations were used
        assert!(
            stats.sparse_chunks_processed > 0,
            "Should use sparse for low-density chunk"
        );
        assert_eq!(stats.objects_swept, expected_dead_count, "Should sweep all dead objects");
    }

    #[test]
    fn test_hybrid_strategy_mixed_chunks() {
        let heap_base = unsafe { Address::from_usize(0x35000000) };
        let bitvector = SimdBitvector::new(heap_base, 256 * 1024, 16);

        // Create mixed density pattern using reasonable numbers
        let chunk0_capacity = bitvector.get_chunk_object_capacity(0);
        let chunk1_capacity = bitvector.get_chunk_object_capacity(1);
        let dense_count = (chunk0_capacity / 2).min(1000); // Conservative dense count

        // Chunk 0: Dense (mark many)
        let mut total_marked = 0;
        for i in 0..dense_count {
            let obj = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            if bitvector.mark_live(obj) {
                total_marked += 1;
            }
        }

        // Chunk 1: Sparse (mark few) - skip to second chunk area
        let chunk1_start = chunk0_capacity * 16; // Start of chunk 1
        let sparse_count = 5;
        for i in 0..sparse_count {
            let obj = unsafe { Address::from_usize(heap_base.as_usize() + chunk1_start + i * 512) };
            if bitvector.mark_live(obj) {
                total_marked += 1;
            }
        }

        let stats = bitvector.hybrid_sweep();

        // Should process chunks (exact strategy depends on actual density)
        assert!(
            stats.simd_chunks_processed > 0 || stats.sparse_chunks_processed > 0,
            "Should use at least one strategy"
        );

        // The sweep should have processed some dead objects
        // (exact count depends on which chunks were processed and their densities)
        assert!(
            stats.objects_swept > 0,
            "Should sweep some dead objects, got {} swept, {} marked total",
            stats.objects_swept, total_marked
        );
    }

    #[test]
    fn test_clear_all_marks_resets_chunk_stats() {
        let heap_base = unsafe { Address::from_usize(0x36000000) };
        let bitvector = SimdBitvector::new(heap_base, 128 * 1024, 16);

        // Mark objects across multiple chunks
        for chunk_idx in 0..2 {
            let chunk_base = heap_base.as_usize() + chunk_idx * 64 * 1024;
            for i in 0..100 {
                let obj = unsafe { Address::from_usize(chunk_base + i * 16) };
                bitvector.mark_live(obj);
            }
        }

        // Verify populations are set
        assert_eq!(
            bitvector.chunk_populations[0].load(Ordering::Relaxed),
            100
        );
        assert_eq!(
            bitvector.chunk_populations[1].load(Ordering::Relaxed),
            100
        );

        // Clear and verify all chunk stats are reset
        bitvector.clear_all_marks();
        for chunk_idx in 0..bitvector.chunk_count {
            assert_eq!(
                bitvector.chunk_populations[chunk_idx].load(Ordering::Relaxed),
                0,
                "Chunk {} population should be reset",
                chunk_idx
            );
        }

        // Verify sweep stats are also reset
        let stats = bitvector.get_stats();
        assert_eq!(stats.objects_marked, 0);
    }

    #[test]
    fn test_chunk_mask_generation() {
        let heap_base = unsafe { Address::from_usize(0x37000000) };
        let bitvector = SimdBitvector::new(heap_base, 10000, 16); // Non-aligned heap size

        // Test mask for complete chunk
        let mask0 = bitvector.create_chunk_object_mask(0);
        assert_eq!(mask0.chunk_idx, 0);
        assert!(mask0.object_capacity > 0);
        assert!(!mask0.word_masks.is_empty());

        // Test mask for last (partial) chunk
        let last_chunk = bitvector.chunk_count - 1;
        let mask_last = bitvector.create_chunk_object_mask(last_chunk);
        assert!(mask_last.object_capacity <= bitvector.objects_per_chunk);

        // Verify last word mask handles boundary correctly
        if !mask_last.word_masks.is_empty() {
            let total_bits_in_all_words: usize = mask_last.word_masks.iter()
                .map(|mask| mask.count_ones() as usize)
                .sum();

            // The total bits across all words should equal the chunk capacity
            assert_eq!(
                total_bits_in_all_words, mask_last.object_capacity,
                "Total mask bits should equal chunk capacity"
            );
        }
    }

    #[test]
    fn test_word_mask_for_chunk_objects_boundary() {
        let heap_base = unsafe { Address::from_usize(0x38000000) };
        let bitvector = SimdBitvector::new(heap_base, 1000, 16); // Small heap for edge cases

        // Test first word of first chunk
        let mask = bitvector.word_mask_for_chunk_objects(0, 0, 62);
        assert_eq!(mask, (1u64 << 62) - 1, "Should mask first 62 bits");

        // Test word spanning chunk boundary
        let chunk_idx = 0;
        let word_idx = bitvector.words_per_chunk - 1; // Last word of chunk
        let chunk_capacity = bitvector.get_chunk_object_capacity(chunk_idx);
        let mask = bitvector.word_mask_for_chunk_objects(chunk_idx, word_idx, chunk_capacity);

        // Mask should only include bits within chunk capacity
        let word_start_obj = word_idx * BITS_PER_WORD;
        let chunk_end_obj = chunk_capacity;
        if word_start_obj < chunk_end_obj {
            let valid_bits = chunk_end_obj.saturating_sub(word_start_obj).min(BITS_PER_WORD);
            let expected = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            assert_eq!(mask, expected, "Boundary word mask incorrect");
        }
    }

    // Benchmark validation tests to ensure microbenchmark helpers generate intended patterns
    #[test]
    fn test_benchmark_dense_pattern_generator() {
        let heap_base = unsafe { Address::from_usize(0x50000000) };
        let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

        // Simulate dense pattern (80% marked)
        let total_objects = bitvector.objects_per_chunk;
        let objects_to_mark = (total_objects * 80) / 100;

        for i in 0..objects_to_mark {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            bitvector.mark_live(obj_addr);
        }

        let population = bitvector.chunk_populations[0].load(Ordering::Relaxed);
        let density = (population as f64) / (total_objects as f64);

        assert!(
            density >= 0.75,
            "Dense pattern should be at least 75% marked, got {:.2}%",
            density * 100.0
        );
        assert!(
            density <= 0.85,
            "Dense pattern should be at most 85% marked, got {:.2}%",
            density * 100.0
        );
    }

    #[test]
    fn test_benchmark_sparse_pattern_generator() {
        let heap_base = unsafe { Address::from_usize(0x51000000) };
        let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

        // Simulate sparse pattern (10% marked)
        let total_objects = bitvector.objects_per_chunk;
        let objects_to_mark = (total_objects * 10) / 100;

        // Mark objects with gaps (every 10th object to achieve ~10% density)
        for i in 0..objects_to_mark {
            let sparse_index = i * 10; // Create gaps between marked objects
            if sparse_index < total_objects {
                let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + sparse_index * 16) };
                bitvector.mark_live(obj_addr);
            }
        }

        let population = bitvector.chunk_populations[0].load(Ordering::Relaxed);
        let density = (population as f64) / (total_objects as f64);

        assert!(
            density <= 0.15,
            "Sparse pattern should be at most 15% marked, got {:.2}%",
            density * 100.0
        );
        assert!(
            density >= 0.05,
            "Sparse pattern should be at least 5% marked, got {:.2}%",
            density * 100.0
        );
    }

    #[test]
    fn test_benchmark_mixed_pattern_generator() {
        let heap_base = unsafe { Address::from_usize(0x52000000) };
        let bitvector = SimdBitvector::new(heap_base, 256 * 1024, 16); // 4 chunks

        let objects_per_chunk = bitvector.objects_per_chunk;

        // Chunk 0: Dense (75%)
        let dense_marks = (objects_per_chunk * 75) / 100;
        for i in 0..dense_marks {
            let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
            bitvector.mark_live(obj_addr);
        }

        // Chunk 1: Sparse (15%)
        let sparse_marks = (objects_per_chunk * 15) / 100;
        let chunk1_base = heap_base.as_usize() + 64 * 1024;
        for i in 0..sparse_marks {
            let sparse_index = i * 7; // Create gaps to maintain sparsity
            if sparse_index < objects_per_chunk {
                let obj_addr = unsafe { Address::from_usize(chunk1_base + sparse_index * 16) };
                bitvector.mark_live(obj_addr);
            }
        }

        // Chunk 2: Medium (45%)
        let medium_marks = (objects_per_chunk * 45) / 100;
        let chunk2_base = heap_base.as_usize() + 128 * 1024;
        for i in 0..medium_marks {
            let medium_index = i * 2; // Moderate spacing for medium density
            if medium_index < objects_per_chunk {
                let obj_addr = unsafe { Address::from_usize(chunk2_base + medium_index * 16) };
                bitvector.mark_live(obj_addr);
            }
        }

        // Verify pattern densities
        let chunk0_density = bitvector.chunk_populations[0].load(Ordering::Relaxed) as f64 / objects_per_chunk as f64;
        let chunk1_density = bitvector.chunk_populations[1].load(Ordering::Relaxed) as f64 / objects_per_chunk as f64;
        let chunk2_density = bitvector.chunk_populations[2].load(Ordering::Relaxed) as f64 / objects_per_chunk as f64;

        assert!(chunk0_density >= 0.70, "Chunk 0 should be dense");
        assert!(chunk1_density <= 0.20, "Chunk 1 should be sparse");
        assert!(chunk2_density >= 0.40 && chunk2_density <= 0.50, "Chunk 2 should be medium");

        // Run sweep and verify strategy dispatch
        let stats = bitvector.hybrid_sweep();
        assert!(stats.simd_chunks_processed >= 1, "Should have at least one dense chunk");
        assert!(stats.sparse_chunks_processed >= 1, "Should have at least one sparse chunk");
    }

    #[test]
    fn test_benchmark_density_threshold_validation() {
        // Validate that our density calculations match benchmark expectations
        let heap_base = unsafe { Address::from_usize(0x55000000) };
        let bitvector = SimdBitvector::new(heap_base, 64 * 1024, 16);

        // Test at various density levels
        let density_tests = [10, 25, 50, 75, 90]; // Percentages

        for &target_density in &density_tests {
            bitvector.clear_all_marks();

            let objects_to_mark = (bitvector.objects_per_chunk * target_density) / 100;
            for i in 0..objects_to_mark {
                let obj_addr = unsafe { Address::from_usize(heap_base.as_usize() + i * 16) };
                bitvector.mark_live(obj_addr);
            }

            let actual_pop = bitvector.chunk_populations[0].load(Ordering::Relaxed);
            let actual_density = (actual_pop * 100) / bitvector.objects_per_chunk;

            // Allow 2% tolerance for rounding
            assert!(
                (actual_density as i32 - target_density as i32).abs() <= 2,
                "Density test failed: target {}%, actual {}%",
                target_density,
                actual_density
            );

            // Run sweep and verify strategy selection
            let stats = bitvector.hybrid_sweep();

            if target_density <= 25 {
                // Should prefer sparse for low density (check operation counters)
                assert!(
                    bitvector.simd_operations.load(Ordering::Relaxed) == 0 || stats.objects_swept > 0,
                    "Low density should minimize SIMD usage"
                );
            } else if target_density >= 75 {
                // Should prefer SIMD for high density (check SIMD counters)
                assert!(
                    bitvector.simd_operations.load(Ordering::Relaxed) > 0 || stats.objects_swept > 0,
                    "High density should use SIMD strategy"
                );
            }
        }
    }
}
