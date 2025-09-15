//! Concurrent marking infrastructure for FUGC-style garbage collection

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use mmtk::util::{Address, ObjectReference};

/// Color states for tricolor marking algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectColor {
    /// White objects are unmarked and candidates for collection
    White,
    /// Grey objects are marked but their children haven't been scanned yet
    Grey,
    /// Black objects are fully marked with all children scanned
    Black,
}

/// Thread-local grey stack for parallel marking
pub struct GreyStack {
    /// Local grey object queue for this worker
    local_stack: VecDeque<ObjectReference>,
    /// Capacity before triggering work sharing
    capacity_threshold: usize,
    /// Worker ID for debugging and coordination
    worker_id: usize,
}

impl GreyStack {
    pub fn new(worker_id: usize, capacity_threshold: usize) -> Self {
        Self {
            local_stack: VecDeque::with_capacity(capacity_threshold),
            capacity_threshold,
            worker_id,
        }
    }

    /// Push an object onto the grey stack
    pub fn push(&mut self, object: ObjectReference) {
        self.local_stack.push_back(object);
    }

    /// Pop an object from the grey stack
    pub fn pop(&mut self) -> Option<ObjectReference> {
        self.local_stack.pop_front()
    }

    /// Check if the local stack is empty
    pub fn is_empty(&self) -> bool {
        self.local_stack.is_empty()
    }

    /// Get the current size of the local stack
    pub fn len(&self) -> usize {
        self.local_stack.len()
    }

    /// Check if we should share work with other workers
    pub fn should_share_work(&self) -> bool {
        self.local_stack.len() > self.capacity_threshold
    }

    /// Extract half of the work for sharing with other workers
    pub fn extract_work(&mut self) -> Vec<ObjectReference> {
        let share_count = self.local_stack.len() / 2;
        let mut shared_work = Vec::with_capacity(share_count);

        for _ in 0..share_count {
            if let Some(obj) = self.local_stack.pop_back() {
                shared_work.push(obj);
            }
        }

        shared_work
    }

    /// Add shared work from another worker
    pub fn add_shared_work(&mut self, work: Vec<ObjectReference>) {
        for obj in work {
            self.local_stack.push_back(obj);
        }
    }
}

/// Global work-stealing coordinator for parallel marking
pub struct ParallelMarkingCoordinator {
    /// Shared work pool for work stealing
    shared_work_pool: Mutex<VecDeque<ObjectReference>>,
    /// Number of active marking workers
    active_workers: AtomicUsize,
    /// Total number of workers
    total_workers: usize,
    /// Work stealing statistics
    work_stolen_count: AtomicUsize,
    work_shared_count: AtomicUsize,
}

impl ParallelMarkingCoordinator {
    pub fn new(total_workers: usize) -> Self {
        Self {
            shared_work_pool: Mutex::new(VecDeque::with_capacity(1024)),
            active_workers: AtomicUsize::new(total_workers),
            total_workers,
            work_stolen_count: AtomicUsize::new(0),
            work_shared_count: AtomicUsize::new(0),
        }
    }

    /// Share work from a worker's local stack
    pub fn share_work(&self, work: Vec<ObjectReference>) {
        let mut pool = self.shared_work_pool.lock().unwrap();
        for obj in work {
            pool.push_back(obj);
        }
        self.work_shared_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Attempt to steal work for a worker
    pub fn steal_work(&self, target_count: usize) -> Vec<ObjectReference> {
        let mut pool = self.shared_work_pool.lock().unwrap();
        let steal_count = std::cmp::min(target_count, pool.len());
        let mut stolen_work = Vec::with_capacity(steal_count);

        for _ in 0..steal_count {
            if let Some(obj) = pool.pop_front() {
                stolen_work.push(obj);
            }
        }

        if !stolen_work.is_empty() {
            self.work_stolen_count.fetch_add(1, Ordering::Relaxed);
        }

        stolen_work
    }

    /// Check if there's any work available in the global pool
    pub fn has_work(&self) -> bool {
        !self.shared_work_pool.lock().unwrap().is_empty()
    }

    /// Signal that a worker has finished its local work
    pub fn worker_finished(&self) -> bool {
        let remaining = self.active_workers.fetch_sub(1, Ordering::SeqCst);
        remaining == 1 // True if this was the last active worker
    }

    /// Reset for a new marking phase
    pub fn reset(&self) {
        self.shared_work_pool.lock().unwrap().clear();
        self.active_workers
            .store(self.total_workers, Ordering::SeqCst);
        self.work_stolen_count.store(0, Ordering::Relaxed);
        self.work_shared_count.store(0, Ordering::Relaxed);
    }

    /// Get work stealing statistics
    pub fn get_stats(&self) -> (usize, usize) {
        (
            self.work_stolen_count.load(Ordering::Relaxed),
            self.work_shared_count.load(Ordering::Relaxed),
        )
    }
}

/// Parallel marking worker that processes grey objects
pub struct MarkingWorker {
    /// Worker ID
    id: usize,
    /// Local grey stack
    pub grey_stack: GreyStack,
    /// Reference to the global coordinator
    coordinator: Arc<ParallelMarkingCoordinator>,
    /// Objects marked by this worker
    objects_marked: usize,
}

impl MarkingWorker {
    pub fn new(
        id: usize,
        coordinator: Arc<ParallelMarkingCoordinator>,
        stack_capacity: usize,
    ) -> Self {
        Self {
            id,
            grey_stack: GreyStack::new(id, stack_capacity),
            coordinator,
            objects_marked: 0,
        }
    }

    /// Add initial grey objects to this worker
    pub fn add_initial_work(&mut self, objects: Vec<ObjectReference>) {
        for obj in objects {
            self.grey_stack.push(obj);
        }
    }

    /// Process grey objects until the local stack is empty
    pub fn process_grey_objects<F>(&mut self, mut mark_object: F)
    where
        F: FnMut(ObjectReference) -> Vec<ObjectReference>,
    {
        loop {
            // Process local work first
            while let Some(grey_obj) = self.grey_stack.pop() {
                // Mark the object black and get its children
                let children = mark_object(grey_obj);
                self.objects_marked += 1;

                // Add children to grey stack
                for child in children {
                    self.grey_stack.push(child);
                }

                // Share work if we have too much
                if self.grey_stack.should_share_work() {
                    let shared_work = self.grey_stack.extract_work();
                    self.coordinator.share_work(shared_work);
                }
            }

            // Try to steal work if we have no local work
            let stolen_work = self.coordinator.steal_work(64);
            if stolen_work.is_empty() {
                // No work available, check if all workers are done
                if self.coordinator.worker_finished() {
                    break; // Global termination
                }

                // Wait a bit and try again
                std::thread::yield_now();
                continue;
            }

            self.grey_stack.add_shared_work(stolen_work);
        }
    }

    /// Get the number of objects marked by this worker
    pub fn objects_marked(&self) -> usize {
        self.objects_marked
    }

    /// Reset worker state for a new marking phase
    pub fn reset(&mut self) {
        self.grey_stack.local_stack.clear();
        self.objects_marked = 0;
    }
}

/// Tricolor marking state manager
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
    pub fn new(heap_base: Address, address_space_size: usize) -> Self {
        let objects_per_word = std::mem::size_of::<usize>() * 8 / 2; // 2 bits per object
        let num_words = (address_space_size / 8 + objects_per_word - 1) / objects_per_word;

        Self {
            color_bits: (0..num_words).map(|_| AtomicUsize::new(0)).collect(),
            heap_base,
            address_space_size,
            bits_per_object: 2,
        }
    }

    /// Get the color of an object
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

    /// Set the color of an object
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
                Err(_) => continue, // Retry on contention
            }
        }
    }

    /// Atomically transition an object from one color to another
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

        loop {
            let current = self.color_bits[word_index].load(Ordering::Acquire);

            // Check if the current color matches expected
            if (current & mask) != expected_bits {
                return false; // Color transition not valid
            }

            let updated = (current & !mask) | new_bits;

            match self.color_bits[word_index].compare_exchange_weak(
                current,
                updated,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(_) => continue, // Retry on contention
            }
        }
    }

    /// Clear all color markings (set everything to white)
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
}

/// Dijkstra write barrier for concurrent marking
pub struct WriteBarrier {
    /// Tricolor marking state
    pub tricolor_marking: Arc<TricolorMarking>,
    /// Global grey stack coordinator
    coordinator: Arc<ParallelMarkingCoordinator>,
    /// Flag indicating if concurrent marking is active
    marking_active: std::sync::atomic::AtomicBool,
}

impl WriteBarrier {
    pub fn new(
        tricolor_marking: Arc<TricolorMarking>,
        coordinator: Arc<ParallelMarkingCoordinator>,
    ) -> Self {
        Self {
            tricolor_marking,
            coordinator,
            marking_active: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Activate the write barrier for concurrent marking
    pub fn activate(&self) {
        self.marking_active.store(true, Ordering::SeqCst);
    }

    /// Deactivate the write barrier after marking completes
    pub fn deactivate(&self) {
        self.marking_active.store(false, Ordering::SeqCst);
    }

    /// Check if the write barrier is currently active
    pub fn is_active(&self) -> bool {
        self.marking_active.load(Ordering::SeqCst)
    }

    /// Dijkstra write barrier: shade the old value when overwriting a reference
    /// This prevents the concurrent marker from missing objects that become unreachable
    /// during marking due to pointer updates by the mutator
    pub fn write_barrier(&self, slot: *mut ObjectReference, new_value: ObjectReference) {
        if !self.is_active() {
            // Write barrier is not active, just perform the store
            unsafe { *slot = new_value };
            return;
        }

        // Read the old value before overwriting
        let old_value = unsafe { *slot };

        // Perform the actual store
        unsafe { *slot = new_value };

        // Dijkstra write barrier: shade the old value if it's white
        if old_value.to_raw_address() != mmtk::util::Address::ZERO {
            let old_color = self.tricolor_marking.get_color(old_value);
            if old_color == ObjectColor::White {
                // Atomically transition from white to grey
                if self.tricolor_marking.transition_color(
                    old_value,
                    ObjectColor::White,
                    ObjectColor::Grey,
                ) {
                    // Successfully marked as grey, add to work pool
                    self.coordinator.share_work(vec![old_value]);
                }
            }
        }
    }

    /// Optimized write barrier for bulk operations
    pub fn write_barrier_bulk(&self, updates: &[(*mut ObjectReference, ObjectReference)]) {
        if !self.is_active() {
            // Write barrier is not active, just perform the stores
            for &(slot, new_value) in updates {
                unsafe { *slot = new_value };
            }
            return;
        }

        let mut grey_objects = Vec::new();

        for &(slot, new_value) in updates {
            // Read the old value before overwriting
            let old_value = unsafe { *slot };

            // Perform the actual store
            unsafe { *slot = new_value };

            // Check if old value needs to be shaded
            if old_value.to_raw_address() != mmtk::util::Address::ZERO {
                let old_color = self.tricolor_marking.get_color(old_value);
                if old_color == ObjectColor::White {
                    // Atomically transition from white to grey
                    if self.tricolor_marking.transition_color(
                        old_value,
                        ObjectColor::White,
                        ObjectColor::Grey,
                    ) {
                        grey_objects.push(old_value);
                    }
                }
            }
        }

        // Share all newly greyed objects in one batch
        if !grey_objects.is_empty() {
            self.coordinator.share_work(grey_objects);
        }
    }

    /// Write barrier for array element updates
    pub fn array_write_barrier(
        &self,
        array_base: *mut u8,
        index: usize,
        element_size: usize,
        new_value: ObjectReference,
    ) {
        let slot_ptr = unsafe {
            array_base
                .add(index * element_size)
                .cast::<ObjectReference>()
        };

        self.write_barrier(slot_ptr, new_value);
    }

    /// Reset write barrier state for a new marking phase
    pub fn reset(&self) {
        self.marking_active.store(false, Ordering::SeqCst);
    }
}

/// Black allocation manager for concurrent marking
pub struct BlackAllocator {
    /// Tricolor marking state
    pub tricolor_marking: Arc<TricolorMarking>,
    /// Flag indicating if black allocation is active
    black_allocation_active: std::sync::atomic::AtomicBool,
    /// Statistics
    objects_allocated_black: std::sync::atomic::AtomicUsize,
}

impl BlackAllocator {
    pub fn new(tricolor_marking: Arc<TricolorMarking>) -> Self {
        Self {
            tricolor_marking,
            black_allocation_active: std::sync::atomic::AtomicBool::new(false),
            objects_allocated_black: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Activate black allocation during marking
    pub fn activate(&self) {
        self.black_allocation_active.store(true, Ordering::SeqCst);
    }

    /// Deactivate black allocation after marking
    pub fn deactivate(&self) {
        self.black_allocation_active.store(false, Ordering::SeqCst);
    }

    /// Check if black allocation is active
    pub fn is_active(&self) -> bool {
        self.black_allocation_active.load(Ordering::SeqCst)
    }

    /// Mark a newly allocated object as black during concurrent marking
    /// This prevents the object from being collected in the current cycle
    pub fn allocate_black(&self, object: ObjectReference) {
        if !self.is_active() {
            // Black allocation not active, object starts white
            return;
        }

        // Mark the newly allocated object as black
        self.tricolor_marking.set_color(object, ObjectColor::Black);
        self.objects_allocated_black.fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics for black allocation
    pub fn get_stats(&self) -> usize {
        self.objects_allocated_black.load(Ordering::Relaxed)
    }

    /// Reset for a new marking phase
    pub fn reset(&self) {
        self.black_allocation_active.store(false, Ordering::SeqCst);
        self.objects_allocated_black.store(0, Ordering::Relaxed);
    }
}

/// Concurrent root scanner for parallel root enumeration during marking
pub struct ConcurrentRootScanner {
    /// Thread registry for accessing mutator threads
    thread_registry: Arc<crate::thread::ThreadRegistry>,
    /// Global roots manager
    global_roots: Arc<crate::roots::GlobalRoots>,
    /// Number of worker threads for root scanning
    num_workers: usize,
    /// Statistics
    roots_scanned: AtomicUsize,
}

/// Concurrent marking coordinator that orchestrates the entire marking process
pub struct ConcurrentMarkingCoordinator {
    /// Write barrier
    write_barrier: WriteBarrier,
    /// Black allocator
    black_allocator: BlackAllocator,
    /// Parallel marking coordinator
    parallel_coordinator: Arc<ParallelMarkingCoordinator>,
    /// Tricolor marking state
    pub tricolor_marking: Arc<TricolorMarking>,
    /// Concurrent root scanner
    root_scanner: ConcurrentRootScanner,
    /// Object classifier for FUGC-style classification
    object_classifier: ObjectClassifier,
    /// Marking worker threads
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Shutdown signal for workers
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl ConcurrentMarkingCoordinator {
    pub fn new(
        heap_base: mmtk::util::Address,
        heap_size: usize,
        num_workers: usize,
        thread_registry: Arc<crate::thread::ThreadRegistry>,
        global_roots: Arc<crate::roots::GlobalRoots>,
    ) -> Self {
        let tricolor_marking = Arc::new(TricolorMarking::new(heap_base, heap_size));
        let parallel_coordinator = Arc::new(ParallelMarkingCoordinator::new(num_workers));
        let write_barrier = WriteBarrier::new(
            Arc::clone(&tricolor_marking),
            Arc::clone(&parallel_coordinator),
        );
        let black_allocator = BlackAllocator::new(Arc::clone(&tricolor_marking));
        let root_scanner = ConcurrentRootScanner::new(
            Arc::clone(&thread_registry),
            Arc::clone(&global_roots),
            num_workers,
        );
        let object_classifier = ObjectClassifier::new();

        Self {
            write_barrier,
            black_allocator,
            parallel_coordinator,
            tricolor_marking,
            root_scanner,
            object_classifier,
            workers: Vec::new(),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start concurrent marking with the given root set
    pub fn start_marking(&mut self, roots: Vec<ObjectReference>) {
        // Reset all components
        self.tricolor_marking.clear();
        self.parallel_coordinator.reset();
        self.write_barrier.reset();
        self.black_allocator.reset();

        // Activate concurrent mechanisms
        self.write_barrier.activate();
        self.black_allocator.activate();

        // Initialize roots as grey
        for root in &roots {
            self.tricolor_marking.set_color(*root, ObjectColor::Grey);
        }
        self.parallel_coordinator.share_work(roots);

        // Start worker threads if not already running
        if self.workers.is_empty() {
            self.shutdown.store(false, Ordering::SeqCst);
            self.start_worker_threads();
        }
    }

    /// Stop concurrent marking and wait for completion
    pub fn stop_marking(&mut self) {
        // Deactivate concurrent mechanisms
        self.write_barrier.deactivate();
        self.black_allocator.deactivate();

        // Wait for workers to complete
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }

    /// Get marking statistics
    pub fn get_stats(&self) -> ConcurrentMarkingStats {
        let (stolen, shared) = self.parallel_coordinator.get_stats();
        ConcurrentMarkingStats {
            work_stolen: stolen,
            work_shared: shared,
            objects_allocated_black: self.black_allocator.get_stats(),
        }
    }

    /// Get references to components for external access
    pub fn write_barrier(&self) -> &WriteBarrier {
        &self.write_barrier
    }

    pub fn black_allocator(&self) -> &BlackAllocator {
        &self.black_allocator
    }

    fn start_worker_threads(&mut self) {
        let num_workers = self.parallel_coordinator.total_workers;

        for worker_id in 0..num_workers {
            let coordinator = Arc::clone(&self.parallel_coordinator);
            let tricolor_marking = Arc::clone(&self.tricolor_marking);
            let shutdown = Arc::clone(&self.shutdown);

            let worker = std::thread::spawn(move || {
                let mut marking_worker = MarkingWorker::new(worker_id, coordinator, 256);

                loop {
                    // Check shutdown signal first
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Check for work
                    if marking_worker.grey_stack.is_empty() {
                        let stolen_work = marking_worker.coordinator.steal_work(64);
                        if stolen_work.is_empty() {
                            // No work available, check if we should terminate
                            if shutdown.load(Ordering::Relaxed) {
                                break;
                            }
                            // Brief pause before checking again
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            continue;
                        }
                        marking_worker.grey_stack.add_shared_work(stolen_work);
                    }

                    // Process available work with shutdown checks
                    let mut processed_any = false;
                    for _ in 0..10 {
                        // Limit iterations to allow shutdown checks
                        if shutdown.load(Ordering::Relaxed) {
                            return; // Exit immediately on shutdown
                        }

                        if let Some(obj) = marking_worker.grey_stack.pop() {
                            tricolor_marking.set_color(obj, ObjectColor::Black);
                            marking_worker.objects_marked += 1;
                            processed_any = true;
                        } else {
                            break;
                        }
                    }

                    if !processed_any {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            });

            self.workers.push(worker);
        }
    }
}

impl ConcurrentRootScanner {
    pub fn new(
        thread_registry: Arc<crate::thread::ThreadRegistry>,
        global_roots: Arc<crate::roots::GlobalRoots>,
        num_workers: usize,
    ) -> Self {
        Self {
            thread_registry,
            global_roots,
            num_workers,
            roots_scanned: AtomicUsize::new(0),
        }
    }

    /// Scan roots concurrently using the registered mutator threads
    pub fn scan_roots(&self) {
        // TODO: Implement concurrent root scanning logic
    }

    /// Get the number of roots scanned
    pub fn get_scanned_count(&self) -> usize {
        self.roots_scanned.load(Ordering::Relaxed)
    }

    /// Reset the scanner state
    pub fn reset(&self) {
        self.roots_scanned.store(0, Ordering::Relaxed);
    }
}

/// FUGC-style object classification system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectAge {
    /// Young objects (recently allocated)
    Young,
    /// Old objects (survived multiple collections)
    Old,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectMutability {
    /// Immutable objects (cannot be modified after creation)
    Immutable,
    /// Mutable objects (can be modified)
    Mutable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectConnectivity {
    /// Low connectivity (few references)
    Low,
    /// Medium connectivity
    Medium,
    /// High connectivity (many references)
    High,
}

/// Complete object classification combining multiple dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectClass {
    pub age: ObjectAge,
    pub mutability: ObjectMutability,
    pub connectivity: ObjectConnectivity,
}

impl ObjectClass {
    /// Create a new object classification
    pub fn new(
        age: ObjectAge,
        mutability: ObjectMutability,
        connectivity: ObjectConnectivity,
    ) -> Self {
        Self {
            age,
            mutability,
            connectivity,
        }
    }

    /// Default classification for newly allocated objects
    pub fn default_young() -> Self {
        Self::new(
            ObjectAge::Young,
            ObjectMutability::Mutable,
            ObjectConnectivity::Low,
        )
    }

    /// Check if this is a "hot" object that should be prioritized in marking
    pub fn is_hot(&self) -> bool {
        matches!(self.age, ObjectAge::Young)
            && matches!(self.mutability, ObjectMutability::Mutable)
            && matches!(self.connectivity, ObjectConnectivity::High)
    }

    /// Check if this object should be scanned eagerly during concurrent marking
    pub fn should_scan_eagerly(&self) -> bool {
        matches!(self.age, ObjectAge::Old) || matches!(self.connectivity, ObjectConnectivity::High)
    }

    /// Get priority for marking (higher = more important to mark first)
    pub fn marking_priority(&self) -> u8 {
        let mut priority = 0;

        // Age priority (old objects get higher priority)
        if matches!(self.age, ObjectAge::Old) {
            priority += 4;
        }

        // Connectivity priority (highly connected objects get higher priority)
        match self.connectivity {
            ObjectConnectivity::High => priority += 3,
            ObjectConnectivity::Medium => priority += 2,
            ObjectConnectivity::Low => priority += 1,
        }

        // Mutability priority (mutable objects get slightly higher priority)
        if matches!(self.mutability, ObjectMutability::Mutable) {
            priority += 1;
        }

        priority
    }
}

/// Object classifier that assigns classifications to objects
pub struct ObjectClassifier {
    /// Classification storage (object -> class mapping)
    classifications: Mutex<std::collections::HashMap<ObjectReference, ObjectClass>>,
    /// Statistics
    young_objects: AtomicUsize,
    old_objects: AtomicUsize,
    immutable_objects: AtomicUsize,
    mutable_objects: AtomicUsize,
}

impl ObjectClassifier {
    pub fn new() -> Self {
        Self {
            classifications: Mutex::new(std::collections::HashMap::new()),
            young_objects: AtomicUsize::new(0),
            old_objects: AtomicUsize::new(0),
            immutable_objects: AtomicUsize::new(0),
            mutable_objects: AtomicUsize::new(0),
        }
    }

    /// Classify an object and store its classification
    pub fn classify_object(&self, object: ObjectReference, class: ObjectClass) {
        let mut classifications = self.classifications.lock().unwrap();
        classifications.insert(object, class);

        // Update statistics
        match class.age {
            ObjectAge::Young => {
                self.young_objects.fetch_add(1, Ordering::Relaxed);
            }
            ObjectAge::Old => {
                self.old_objects.fetch_add(1, Ordering::Relaxed);
            }
        }

        match class.mutability {
            ObjectMutability::Immutable => {
                self.immutable_objects.fetch_add(1, Ordering::Relaxed);
            }
            ObjectMutability::Mutable => {
                self.mutable_objects.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get the classification of an object
    pub fn get_classification(&self, object: ObjectReference) -> Option<ObjectClass> {
        let classifications = self.classifications.lock().unwrap();
        classifications.get(&object).copied()
    }

    /// Promote a young object to old (after surviving collections)
    pub fn promote_to_old(&self, object: ObjectReference) -> bool {
        let mut classifications = self.classifications.lock().unwrap();
        if let Some(class) = classifications.get_mut(&object) {
            if matches!(class.age, ObjectAge::Young) {
                class.age = ObjectAge::Old;
                self.young_objects.fetch_sub(1, Ordering::Relaxed);
                self.old_objects.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Update connectivity classification based on reference count
    pub fn update_connectivity(&self, object: ObjectReference, reference_count: usize) {
        let mut classifications = self.classifications.lock().unwrap();
        if let Some(class) = classifications.get_mut(&object) {
            class.connectivity = match reference_count {
                0..=2 => ObjectConnectivity::Low,
                3..=10 => ObjectConnectivity::Medium,
                _ => ObjectConnectivity::High,
            };
        }
    }

    /// Get objects that should be scanned eagerly during concurrent marking
    pub fn get_eager_scan_objects(&self) -> Vec<(ObjectReference, ObjectClass)> {
        let classifications = self.classifications.lock().unwrap();
        classifications
            .iter()
            .filter(|(_, class)| class.should_scan_eagerly())
            .map(|(&obj, &class)| (obj, class))
            .collect()
    }

    /// Get objects sorted by marking priority (highest first)
    pub fn get_prioritized_objects(&self) -> Vec<(ObjectReference, ObjectClass)> {
        let classifications = self.classifications.lock().unwrap();
        let mut objects: Vec<_> = classifications
            .iter()
            .map(|(&obj, &class)| (obj, class))
            .collect();

        // Sort by priority (highest first)
        objects.sort_by(|a, b| b.1.marking_priority().cmp(&a.1.marking_priority()));
        objects
    }

    /// Get classification statistics
    pub fn get_stats(&self) -> ObjectClassificationStats {
        ObjectClassificationStats {
            young_objects: self.young_objects.load(Ordering::Relaxed),
            old_objects: self.old_objects.load(Ordering::Relaxed),
            immutable_objects: self.immutable_objects.load(Ordering::Relaxed),
            mutable_objects: self.mutable_objects.load(Ordering::Relaxed),
            total_classified: {
                let classifications = self.classifications.lock().unwrap();
                classifications.len()
            },
        }
    }

    /// Clear all classifications (for new GC cycle)
    pub fn clear(&self) {
        let mut classifications = self.classifications.lock().unwrap();
        classifications.clear();
        self.young_objects.store(0, Ordering::Relaxed);
        self.old_objects.store(0, Ordering::Relaxed);
        self.immutable_objects.store(0, Ordering::Relaxed);
        self.mutable_objects.store(0, Ordering::Relaxed);
    }
}

/// Statistics for concurrent marking
#[derive(Debug, Clone)]
pub struct ConcurrentMarkingStats {
    pub work_stolen: usize,
    pub work_shared: usize,
    pub objects_allocated_black: usize,
}

/// Statistics for object classification
#[derive(Debug, Clone)]
pub struct ObjectClassificationStats {
    pub young_objects: usize,
    pub old_objects: usize,
    pub immutable_objects: usize,
    pub mutable_objects: usize,
    pub total_classified: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmtk::util::Address;

    #[test]
    fn grey_stack_basic_operations() {
        let mut stack = GreyStack::new(0, 100);
        assert!(stack.is_empty());

        let obj =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000)) };
        stack.push(obj);
        assert!(!stack.is_empty());
        assert_eq!(stack.len(), 1);

        let popped = stack.pop();
        assert_eq!(popped, Some(obj));
        assert!(stack.is_empty());
    }

    #[test]
    fn tricolor_marking_operations() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let marking = TricolorMarking::new(heap_base, 0x10000);

        let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };

        // Initially white
        assert_eq!(marking.get_color(obj), ObjectColor::White);

        // Set to grey
        marking.set_color(obj, ObjectColor::Grey);
        assert_eq!(marking.get_color(obj), ObjectColor::Grey);

        // Transition to black
        assert!(marking.transition_color(obj, ObjectColor::Grey, ObjectColor::Black));
        assert_eq!(marking.get_color(obj), ObjectColor::Black);

        // Invalid transition should fail
        assert!(!marking.transition_color(obj, ObjectColor::Grey, ObjectColor::White));
    }

    #[test]
    fn parallel_coordinator_work_stealing() {
        let coordinator = ParallelMarkingCoordinator::new(2);
        assert!(!coordinator.has_work());

        let obj1 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x1000)) };
        let obj2 =
            unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(0x2000)) };

        coordinator.share_work(vec![obj1, obj2]);
        assert!(coordinator.has_work());

        let stolen = coordinator.steal_work(1);
        assert_eq!(stolen.len(), 1);
        assert!(coordinator.has_work());

        let stolen2 = coordinator.steal_work(10);
        assert_eq!(stolen2.len(), 1);
        assert!(!coordinator.has_work());
    }

    #[test]
    fn write_barrier_activation() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let marking = Arc::new(TricolorMarking::new(heap_base, 0x10000));
        let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
        let barrier = WriteBarrier::new(marking, coordinator);

        assert!(!barrier.is_active());
        barrier.activate();
        assert!(barrier.is_active());
        barrier.deactivate();
        assert!(!barrier.is_active());
    }

    #[test]
    fn write_barrier_dijkstra_shading() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let marking = Arc::new(TricolorMarking::new(heap_base, 0x10000));
        let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
        let barrier = WriteBarrier::new(Arc::clone(&marking), coordinator);

        let obj1 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };
        let obj2 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x200usize) };

        // Set up initial state - obj1 is white
        marking.set_color(obj1, ObjectColor::White);
        assert_eq!(marking.get_color(obj1), ObjectColor::White);

        // Create a slot containing obj1
        let mut slot = obj1;
        let slot_ptr = &mut slot as *mut ObjectReference;

        // Activate write barrier
        barrier.activate();

        // Perform write barrier operation (overwrite obj1 with obj2)
        barrier.write_barrier(slot_ptr, obj2);

        // Check that obj1 was shaded to grey (Dijkstra write barrier)
        assert_eq!(marking.get_color(obj1), ObjectColor::Grey);
        assert_eq!(slot, obj2);
    }

    #[test]
    fn write_barrier_bulk_operations() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let marking = Arc::new(TricolorMarking::new(heap_base, 0x10000));
        let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
        let barrier = WriteBarrier::new(Arc::clone(&marking), coordinator);

        let obj1 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };
        let obj2 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x200usize) };
        let obj3 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x300usize) };
        let obj4 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x400usize) };

        // Set up initial state
        marking.set_color(obj1, ObjectColor::White);
        marking.set_color(obj2, ObjectColor::White);

        let mut slot1 = obj1;
        let mut slot2 = obj2;
        let updates = vec![
            (&mut slot1 as *mut ObjectReference, obj3),
            (&mut slot2 as *mut ObjectReference, obj4),
        ];

        barrier.activate();
        barrier.write_barrier_bulk(&updates);

        // Both old values should be shaded
        assert_eq!(marking.get_color(obj1), ObjectColor::Grey);
        assert_eq!(marking.get_color(obj2), ObjectColor::Grey);
        assert_eq!(slot1, obj3);
        assert_eq!(slot2, obj4);
    }

    #[test]
    fn black_allocator_operations() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let marking = Arc::new(TricolorMarking::new(heap_base, 0x10000));
        let allocator = BlackAllocator::new(Arc::clone(&marking));

        let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };

        assert!(!allocator.is_active());
        assert_eq!(allocator.get_stats(), 0);

        // Without activation, objects remain white
        allocator.allocate_black(obj);
        assert_eq!(marking.get_color(obj), ObjectColor::White);

        // Activate and allocate black
        allocator.activate();
        allocator.allocate_black(obj);
        assert_eq!(marking.get_color(obj), ObjectColor::Black);
        assert_eq!(allocator.get_stats(), 1);

        allocator.deactivate();
        assert!(!allocator.is_active());
    }

    #[test]
    fn concurrent_marking_coordinator_lifecycle() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let thread_registry = Arc::new(crate::thread::ThreadRegistry::new());
        let global_roots = Arc::new(crate::roots::GlobalRoots::default());
        let coordinator = ConcurrentMarkingCoordinator::new(
            heap_base,
            0x10000,
            1, // Use single worker
            thread_registry,
            global_roots,
        );

        let root1 = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };

        // Test activation/deactivation without starting worker threads
        assert!(!coordinator.write_barrier().is_active());
        assert!(!coordinator.black_allocator().is_active());

        coordinator.write_barrier.activate();
        coordinator.black_allocator.activate();
        assert!(coordinator.write_barrier().is_active());
        assert!(coordinator.black_allocator().is_active());

        coordinator.write_barrier.deactivate();
        coordinator.black_allocator.deactivate();
        assert!(!coordinator.write_barrier().is_active());
        assert!(!coordinator.black_allocator().is_active());

        // Test that roots are properly initialized
        coordinator
            .tricolor_marking
            .set_color(root1, ObjectColor::Grey);
        assert_eq!(
            coordinator.tricolor_marking.get_color(root1),
            ObjectColor::Grey
        );
    }

    #[test]
    fn array_write_barrier() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let marking = Arc::new(TricolorMarking::new(heap_base, 0x10000));
        let coordinator = Arc::new(ParallelMarkingCoordinator::new(1));
        let barrier = WriteBarrier::new(Arc::clone(&marking), coordinator);

        let old_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };
        let new_obj =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x200usize) };

        // Set up array with old object
        let null_ref =
            unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x1000usize) };
        let mut array = [old_obj, null_ref];
        marking.set_color(old_obj, ObjectColor::White);

        barrier.activate();

        // Update array element through write barrier
        barrier.array_write_barrier(
            array.as_mut_ptr() as *mut u8,
            0,
            std::mem::size_of::<ObjectReference>(),
            new_obj,
        );

        // Old object should be shaded
        assert_eq!(marking.get_color(old_obj), ObjectColor::Grey);
        assert_eq!(array[0], new_obj);
    }

    #[test]
    fn concurrent_marking_stats() {
        let heap_base = unsafe { Address::from_usize(0x10000) };
        let thread_registry = Arc::new(crate::thread::ThreadRegistry::new());
        let global_roots = Arc::new(crate::roots::GlobalRoots::default());
        let mut coordinator =
            ConcurrentMarkingCoordinator::new(heap_base, 0x10000, 2, thread_registry, global_roots);

        let stats = coordinator.get_stats();
        assert_eq!(stats.work_stolen, 0);
        assert_eq!(stats.work_shared, 0);
        assert_eq!(stats.objects_allocated_black, 0);

        // Start marking to activate systems
        let root = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x100usize) };
        coordinator.start_marking(vec![root]);

        // Simulate some allocations
        let obj = unsafe { ObjectReference::from_raw_address_unchecked(heap_base + 0x200usize) };
        coordinator.black_allocator().allocate_black(obj);

        let updated_stats = coordinator.get_stats();
        assert_eq!(updated_stats.objects_allocated_black, 1);

        coordinator.stop_marking();
    }
}
