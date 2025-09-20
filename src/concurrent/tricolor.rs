//! Tricolor marking state management for concurrent garbage collection

use crate::compat::{Address, ObjectReference};
use crossbeam_utils::Backoff;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::core::ObjectColor;

/// Tricolor marking state manager that tracks object colors for concurrent garbage collection.
/// Uses a compact bit vector representation with atomic operations for thread safety.
///
/// # Examples
///
/// ```
/// use fugrip::concurrent::{TricolorMarking, ObjectColor};
/// use crate::compat::{Address, ObjectReference};
/// use std::sync::Arc;
///
/// let heap_base = unsafe { Address::from_usize(0x10000000) };
/// let marking = Arc::new(TricolorMarking::new(heap_base, 1024 * 1024));
///
/// // Create an object reference
/// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
///
/// // Objects start as white
/// assert_eq!(marking.get_color(obj), ObjectColor::White);
///
/// // Mark object as grey
/// marking.set_color(obj, ObjectColor::Grey);
/// assert_eq!(marking.get_color(obj), ObjectColor::Grey);
///
/// // Atomically transition from grey to black
/// let success = marking.transition_color(obj, ObjectColor::Grey, ObjectColor::Black);
/// assert!(success);
/// assert_eq!(marking.get_color(obj), ObjectColor::Black);
/// ```
pub struct TricolorMarking {
    /// Bit vector for object colors (2 bits per object)
    /// 00 = White, 01 = Grey, 10 = Black, 11 = Reserved
    color_bits: Vec<AtomicUsize>,
    /// Base address for address-to-index conversion
    heap_base: Address,
    /// Address space size covered by this marking state
    address_space_size: usize,
    /// Bits per color entry (2 bits for tricolor)
    bits_per_object: usize,
}

impl TricolorMarking {
    /// Create a new tricolor marking state manager for the given address space.
    ///
    /// # Arguments
    /// * `heap_base` - Base address of the heap region
    /// * `address_space_size` - Size of the address space to track
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::concurrent::TricolorMarking;
    /// use crate::compat::Address;
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let marking = TricolorMarking::new(heap_base, 64 * 1024 * 1024); // 64MB
    ///
    /// // Ready to track object colors in the 64MB address space
    /// ```
    pub fn new(heap_base: Address, address_space_size: usize) -> Self {
        let objects_per_word = std::mem::size_of::<usize>() * 8 / 2; // 2 bits per object
        let num_words = (address_space_size / 8).div_ceil(objects_per_word);

        Self {
            color_bits: (0..num_words).map(|_| AtomicUsize::new(0)).collect(),
            heap_base,
            address_space_size,
            bits_per_object: 2,
        }
    }

    /// Get the current color of an object.
    ///
    /// # Arguments
    /// * `object` - Object reference to query
    ///
    /// # Returns
    /// Current [`ObjectColor`] of the object
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::concurrent::{TricolorMarking, ObjectColor};
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let marking = TricolorMarking::new(heap_base, 1024 * 1024);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    ///
    /// // New objects are white by default
    /// assert_eq!(marking.get_color(obj), ObjectColor::White);
    /// ```
    pub fn get_color(&self, object: ObjectReference) -> ObjectColor {
        let index = self.object_to_index(object);
        let word_index = index / (std::mem::size_of::<usize>() * 8 / self.bits_per_object);
        let bit_offset = (index % (std::mem::size_of::<usize>() * 8 / self.bits_per_object))
            * self.bits_per_object;

        if word_index >= self.color_bits.len() {
            return ObjectColor::White; // Default for out-of-bounds
        }

        let word = self.color_bits[word_index].load(Ordering::Acquire);
        let color_bits = (word >> bit_offset) & 0b11;

        match color_bits {
            0b00 => ObjectColor::White,
            0b01 => ObjectColor::Grey,
            0b10 => ObjectColor::Black,
            _ => ObjectColor::White, // Reserved/invalid
        }
    }

    /// Set the color of an object atomically.
    ///
    /// # Arguments
    /// * `object` - Object reference to modify
    /// * `color` - New color to assign
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::concurrent::{TricolorMarking, ObjectColor};
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let marking = TricolorMarking::new(heap_base, 1024 * 1024);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    ///
    /// // Set object to grey (marked but not scanned)
    /// marking.set_color(obj, ObjectColor::Grey);
    /// assert_eq!(marking.get_color(obj), ObjectColor::Grey);
    ///
    /// // Set object to black (fully processed)
    /// marking.set_color(obj, ObjectColor::Black);
    /// assert_eq!(marking.get_color(obj), ObjectColor::Black);
    /// ```
    pub fn set_color(&self, object: ObjectReference, color: ObjectColor) {
        let index = self.object_to_index(object);
        let word_index = index / (std::mem::size_of::<usize>() * 8 / self.bits_per_object);
        let bit_offset = (index % (std::mem::size_of::<usize>() * 8 / self.bits_per_object))
            * self.bits_per_object;

        if word_index >= self.color_bits.len() {
            return; // Out-of-bounds, ignore
        }

        let color_bits = match color {
            ObjectColor::White => 0b00,
            ObjectColor::Grey => 0b01,
            ObjectColor::Black => 0b10,
        };

        // Atomic update with compare-and-swap loop
        let mask = 0b11usize << bit_offset;
        let new_bits = color_bits << bit_offset;

        // Use `snooze()` to prefer an escalating strategy: start with a
        // few tight spins but quickly escalate to yield/park under
        // contention. This reduces CPU waste for longer waits while
        // still being responsive for short waits.
        let backoff = Backoff::new();
        backoff.snooze();
        loop {
            let current = self.color_bits[word_index].load(Ordering::Acquire);
            let updated = (current & !mask) | new_bits;

            match self.color_bits[word_index].compare_exchange_weak(
                current,
                updated,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => {
                    // Use exponential backoff to reduce CPU spinning under contention
                    // Use `spin()` for short, tight retries on the CAS
                    // because contention windows here are expected to be
                    // short; spinning briefly keeps latency low.
                    backoff.spin();
                    continue;
                }
            }
        }
    }

    /// Atomically transition an object from one color to another.
    /// This operation is thread-safe and will only succeed if the object
    /// is currently in the expected `from` color.
    ///
    /// # Arguments
    /// * `object` - Object reference to modify
    /// * `from` - Expected current color
    /// * `to` - Desired new color
    ///
    /// # Returns
    /// `true` if the transition succeeded, `false` if the object was not in the expected color
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::concurrent::{TricolorMarking, ObjectColor};
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let marking = TricolorMarking::new(heap_base, 1024 * 1024);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    ///
    /// // Transition from white to grey (should succeed)
    /// let success = marking.transition_color(obj, ObjectColor::White, ObjectColor::Grey);
    /// assert!(success);
    /// assert_eq!(marking.get_color(obj), ObjectColor::Grey);
    ///
    /// // Try to transition from white to black (should fail - object is grey)
    /// let failed = marking.transition_color(obj, ObjectColor::White, ObjectColor::Black);
    /// assert!(!failed);
    /// assert_eq!(marking.get_color(obj), ObjectColor::Grey); // Unchanged
    ///
    /// // Transition from grey to black (should succeed)
    /// let success2 = marking.transition_color(obj, ObjectColor::Grey, ObjectColor::Black);
    /// assert!(success2);
    /// assert_eq!(marking.get_color(obj), ObjectColor::Black);
    /// ```
    pub fn transition_color(
        &self,
        object: ObjectReference,
        from: ObjectColor,
        to: ObjectColor,
    ) -> bool {
        let index = self.object_to_index(object);
        let word_index = index / (std::mem::size_of::<usize>() * 8 / self.bits_per_object);
        let bit_offset = (index % (std::mem::size_of::<usize>() * 8 / self.bits_per_object))
            * self.bits_per_object;

        if word_index >= self.color_bits.len() {
            return false; // Out-of-bounds
        }

        let from_bits = match from {
            ObjectColor::White => 0b00,
            ObjectColor::Grey => 0b01,
            ObjectColor::Black => 0b10,
        };

        let to_bits = match to {
            ObjectColor::White => 0b00,
            ObjectColor::Grey => 0b01,
            ObjectColor::Black => 0b10,
        };

        let mask = 0b11usize << bit_offset;
        let expected_bits = from_bits << bit_offset;
        let new_bits = to_bits << bit_offset;

        let backoff = Backoff::new();
        backoff.snooze();
        loop {
            let current = self.color_bits[word_index].load(Ordering::Acquire);

            // Check if the current color matches expected
            if (current & mask) != expected_bits {
                return false; // Color transition not valid
            }

            // ADVANCING WAVEFRONT INVARIANT: Enforce once-marked-always-marked property
            // Objects can only transition forward: White → Grey → Black
            // Backwards transitions violate the advancing wavefront property
            match (from, to) {
                // Valid forward transitions
                (ObjectColor::White, ObjectColor::Grey) => {} // Discovery: white → grey
                (ObjectColor::Grey, ObjectColor::Black) => {} // Processing: grey → black
                (ObjectColor::White, ObjectColor::Black) => {} // Direct marking: white → black

                // Self-transitions are allowed (idempotent)
                (ObjectColor::White, ObjectColor::White) => {}
                (ObjectColor::Grey, ObjectColor::Grey) => {}
                (ObjectColor::Black, ObjectColor::Black) => {}

                // INVALID backwards transitions - violate advancing wavefront
                (ObjectColor::Black, ObjectColor::White) => return false, // Never allow black → white
                (ObjectColor::Black, ObjectColor::Grey) => return false, // Never allow black → grey
                (ObjectColor::Grey, ObjectColor::White) => return false, // Never allow grey → white
            }

            let updated = (current & !mask) | new_bits;

            match self.color_bits[word_index].compare_exchange_weak(
                current,
                updated,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(_) => {
                    backoff.spin();
                    continue; // Retry on contention
                }
            }
        }
    }

    /// Get all objects that are currently marked as black (fully processed)
    ///
    /// This is used during sweep to build the SIMD bitvector of live objects.
    ///
    /// # Returns
    /// Vector of all ObjectReferences that are currently black
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::concurrent::{TricolorMarking, ObjectColor};
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let marking = TricolorMarking::new(heap_base, 1024 * 1024);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    ///
    /// // Mark object as black
    /// marking.set_color(obj, ObjectColor::Black);
    ///
    /// // Retrieve all black objects
    /// let black_objects = marking.get_black_objects();
    /// assert!(black_objects.contains(&obj));
    /// ```
    pub fn get_black_objects(&self) -> Vec<ObjectReference> {
        let mut black_objects = Vec::new();
        let objects_per_word = std::mem::size_of::<usize>() * 8 / self.bits_per_object;
        const HIGH_BIT_MASK: usize = 0xAAAAAAAAAAAAAAAAusize; // High bits of each 2-bit lane
        const LOW_BIT_MASK: usize = 0x5555555555555555usize; // Low bits of each 2-bit lane

        for (word_index, word) in self.color_bits.iter().enumerate() {
            let word_value = word.load(Ordering::Acquire);
            if word_value == 0 {
                continue;
            }

            let mut mask = (word_value & HIGH_BIT_MASK) & !(word_value & LOW_BIT_MASK);
            while mask != 0 {
                let high_bit = mask.trailing_zeros() as usize;
                let object_index = word_index * objects_per_word + (high_bit >> 1);
                let addr = self.heap_base + (object_index * 8);

                if let Some(obj_ref) = ObjectReference::from_raw_address(addr) {
                    black_objects.push(obj_ref);
                }

                mask &= mask - 1;
            }
        }

        black_objects
    }

    /// Clear all color markings (set everything to white)
    ///
    /// This resets all objects to white state, preparing for the next collection cycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use fugrip::concurrent::{TricolorMarking, ObjectColor};
    /// use crate::compat::{Address, ObjectReference};
    ///
    /// let heap_base = unsafe { Address::from_usize(0x10000000) };
    /// let marking = TricolorMarking::new(heap_base, 1024 * 1024);
    /// let obj = ObjectReference::from_raw_address(heap_base).unwrap();
    ///
    /// // Mark object as black
    /// marking.set_color(obj, ObjectColor::Black);
    /// assert_eq!(marking.get_color(obj), ObjectColor::Black);
    ///
    /// // Clear all markings
    /// marking.clear();
    /// assert_eq!(marking.get_color(obj), ObjectColor::White);
    /// ```
    pub fn clear(&self) {
        for word in &self.color_bits {
            word.store(0, Ordering::Release);
        }
    }

    /// Convert object reference to bit index
    fn object_to_index(&self, object: ObjectReference) -> usize {
        let addr = object.to_raw_address();
        if addr < self.heap_base {
            return 0;
        }

        let offset = addr - self.heap_base;
        offset / 8 // Assume 8-byte alignment
    }

    /// Reset marking state for new collection cycle
    /// This resets all objects to white (unmarked) state in preparation for GC
    pub fn reset_marking_state(&self) {
        // Reset all color bits to white (00) for new collection cycle
        for word in &self.color_bits {
            word.store(0, Ordering::Release);
        }
    }

    /// Mark an object as grey for concurrent marking
    /// This is used during root scanning to seed the marking work queue
    pub fn mark_grey(&self, object: ObjectReference) {
        let index = self.object_to_index(object);
        let word_index = index / (std::mem::size_of::<usize>() * 8 / self.bits_per_object);
        let bit_offset = (index % (std::mem::size_of::<usize>() * 8 / self.bits_per_object))
            * self.bits_per_object;

        if word_index < self.color_bits.len() {
            let grey_bits = 0b01 << bit_offset;
            self.color_bits[word_index].fetch_or(grey_bits, Ordering::Release);
        }
    }
}
