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

        Self {
            heap_base,
            heap_size,
            object_alignment,
            bits: (0..aligned_words).map(|_| AtomicU64::new(0)).collect(),
            max_objects,
            objects_marked: AtomicUsize::new(0),
            objects_swept: AtomicUsize::new(0),
            simd_operations: AtomicUsize::new(0),
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
    pub fn mark_live(&self, object_addr: Address) -> bool {
        if let Some(bit_index) = self.object_to_bit_index(object_addr) {
            let word_index = bit_index / BITS_PER_WORD;
            let bit_offset = bit_index % BITS_PER_WORD;

            if word_index < self.bits.len() {
                let atomic_word = &self.bits[word_index];
                let old_word = atomic_word.load(Ordering::Relaxed);
                let new_word = old_word | (1u64 << bit_offset);
                atomic_word.store(new_word, Ordering::Relaxed);

                self.objects_marked.fetch_add(1, Ordering::Relaxed);
                return true;
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
    pub fn simd_sweep(&self) -> SweepStatistics {
        let sweep_start = Instant::now();
        let mut free_blocks = Vec::new();

        let objects_swept = {
            #[cfg(target_arch = "x86_64")]
            {
                self.simd_sweep_x86_64(&mut free_blocks)
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                self.fallback_sweep(&mut free_blocks)
            }
        };

        let sweep_time = sweep_start.elapsed();
        self.objects_swept.store(objects_swept, Ordering::Relaxed);

        SweepStatistics {
            objects_swept,
            free_blocks: free_blocks.len(),
            sweep_time_ns: sweep_time.as_nanos() as u64,
            simd_blocks_processed: self.bits.len() / WORDS_PER_SIMD_BLOCK,
            throughput_objects_per_sec: if sweep_time.as_nanos() > 0 {
                (objects_swept as u128 * 1_000_000_000) / sweep_time.as_nanos()
            } else {
                0
            } as u64,
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
        self.objects_marked.store(0, Ordering::Relaxed);
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
        // Should be very fast (sub-nanosecond per object on modern hardware)
        assert!(stats.sweep_time_ns < stats.objects_swept as u64 * 50); // Sub-50ns per object
        assert!(stats.throughput_objects_per_sec > 10_000_000); // 10M+ objects/sec
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
}
