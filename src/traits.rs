use crate::{GcHeader, SendPtr};

/// Trait for objects that can be traced during garbage collection.
///
/// Types implementing this trait can be stored in the garbage-collected heap.
/// The `trace` method is called during the marking phase to identify all
/// reachable objects in the object graph.
///
/// # Safety
///
/// This method is `unsafe` because it deals with raw pointers and is called
/// during garbage collection when normal borrowing rules may not apply.
///
/// # Examples
///
/// Implementing `GcTrace` for a custom type:
///
/// ```
/// use fugrip::{GcTrace, Gc, SendPtr, GcHeader};
///
/// struct Node {
///     value: i32,
///     next: Option<Gc<Node>>,
/// }
///
/// # // We can't actually implement GcTrace for Node in a doctest because it would
/// # // violate the orphan rule, but this shows the pattern
/// # trait MockGcTrace {
/// #     unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>);
/// # }
/// # impl MockGcTrace for Node {
/// #     unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
/// #         // In a real implementation, this would trace the next pointer
/// #         // if let Some(ref next) = self.next {
/// #         //     next.trace(stack);
/// #         // }
/// #     }
/// # }
///
/// // The implementation would look like this:
/// // unsafe impl GcTrace for Node {
/// //     unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
/// //         if let Some(ref next) = self.next {
/// //             next.trace(stack);
/// //         }
/// //     }
/// // }
/// ```
pub unsafe trait GcTrace {
    /// Traces all garbage-collected references within this object.
    ///
    /// This method should call `trace` on all `Gc<T>` fields within the object,
    /// allowing the garbage collector to discover all reachable objects.
    ///
    /// # Safety
    ///
    /// This method is called during garbage collection and must not:
    /// - Allocate new objects
    /// - Access objects that might be collected
    /// - Cause data races with concurrent collectors
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>);
}

// Provide GcTrace implementations for common primitive and standard types
unsafe impl GcTrace for i32 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for usize {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for String {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for () {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

// Additional primitive type implementations
unsafe impl GcTrace for u8 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for i8 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for u16 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for i16 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for u32 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for i64 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for u64 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for isize {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for f32 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for f64 {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for bool {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

unsafe impl GcTrace for char {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

// Container implementations
unsafe impl<T: GcTrace> GcTrace for Vec<T> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        for item in self.iter() {
            unsafe { item.trace(stack); }
        }
    }
}

unsafe impl<T: GcTrace> GcTrace for Option<T> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        if let Some(item) = self {
            unsafe { item.trace(stack); }
        }
    }
}

unsafe impl<T: GcTrace> GcTrace for Box<T> {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        unsafe { (**self).trace(stack); }
    }
}

// Weak references do not hold strong GC roots; tracing is a no-op
unsafe impl<T> GcTrace for crate::Weak<T> {
    unsafe fn trace(&self, _stack: &mut Vec<SendPtr<GcHeader<()>>>) {}
}

// Tuple implementations
unsafe impl<T: GcTrace> GcTrace for (T,) {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        unsafe { self.0.trace(stack); }
    }
}

unsafe impl<T: GcTrace, U: GcTrace> GcTrace for (T, U) {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        unsafe { 
            self.0.trace(stack);
            self.1.trace(stack);
        }
    }
}

unsafe impl<T: GcTrace, U: GcTrace, V: GcTrace> GcTrace for (T, U, V) {
    unsafe fn trace(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        unsafe {
            self.0.trace(stack);
            self.1.trace(stack);
            self.2.trace(stack);
        }
    }
}

/// Trait for objects that can be finalized before collection.
///
/// Objects implementing this trait have their `finalize` method called
/// when they are about to be garbage collected, allowing for cleanup
/// operations like closing files or network connections.
///
/// # Examples
///
/// ```
/// use fugrip::traits::GcFinalize;
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicBool, Ordering};
///
/// struct FileResource {
///     name: String,
///     closed: Arc<AtomicBool>,
/// }
///
/// impl GcFinalize for FileResource {
///     fn finalize(&mut self) {
///         // Clean up the resource
///         println!("Closing file: {}", self.name);
///         self.closed.store(true, Ordering::SeqCst);
///     }
/// }
///
/// let closed_flag = Arc::new(AtomicBool::new(false));
/// let mut resource = FileResource {
///     name: "test.txt".to_string(),
///     closed: closed_flag.clone(),
/// };
///
/// // This would normally be called automatically by the GC
/// resource.finalize();
/// assert!(closed_flag.load(Ordering::SeqCst));
/// ```
pub trait GcFinalize {
    /// Called when the object is about to be garbage collected.
    ///
    /// This method should perform any necessary cleanup operations.
    /// It should not panic or perform operations that could fail.
    fn finalize(&mut self);
}

/// Trait for objects that can be dropped with custom cleanup logic.
///
/// This trait provides a safe alternative to implementing `Drop` for
/// garbage-collected objects, as the standard `Drop` trait may not
/// be called at predictable times due to the garbage collector.
///
/// # Examples
///
/// ```
/// use fugrip::traits::GcDrop;
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicBool, Ordering};
///
/// struct NetworkConnection {
///     id: u32,
///     closed: Arc<AtomicBool>,
/// }
///
/// impl GcDrop for NetworkConnection {
///     fn gc_drop(&mut self) {
///         // Cleanup network resources
///         println!("Closing connection {}", self.id);
///         self.closed.store(true, Ordering::SeqCst);
///     }
/// }
///
/// let closed_flag = Arc::new(AtomicBool::new(false));
/// let mut conn = NetworkConnection {
///     id: 123,
///     closed: closed_flag.clone(),
/// };
///
/// // This would be called automatically during GC
/// conn.gc_drop();
/// assert!(closed_flag.load(Ordering::SeqCst));
/// ```
pub trait GcDrop {
    /// Called when the object is being dropped by the garbage collector.
    ///
    /// This method should perform any necessary cleanup operations
    /// that don't require the object to remain alive.
    fn gc_drop(&mut self);
}

/// Trait for objects that need special handling during census phase.
///
/// Objects implementing this trait are processed during the census phase
/// of garbage collection to handle weak reference invalidation and
/// similar operations that require coordination with the collector.
///
/// # Examples
///
/// ```
/// use fugrip::traits::GcCensus;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// struct CacheManager {
///     cache_size: AtomicUsize,
///     max_size: usize,
/// }
///
/// impl GcCensus for CacheManager {
///     fn census(&self) -> bool {
///         let current_size = self.cache_size.load(Ordering::Acquire);
///         if current_size > self.max_size {
///             // Indicate that cache needs cleanup
///             println!("Cache size {} exceeds max {}, needs cleanup", current_size, self.max_size);
///             true
///         } else {
///             false
///         }
///     }
/// }
///
/// let manager = CacheManager {
///     cache_size: AtomicUsize::new(1000),
///     max_size: 500,
/// };
///
/// // This would be called during GC census phase
/// let needs_cleanup = manager.census();
/// assert!(needs_cleanup); // Cache exceeds max size
/// ```
pub trait GcCensus {
    /// Called during the census phase of garbage collection.
    ///
    /// This method can be used to update weak references, clean up
    /// expired caches, or perform other census-related operations.
    fn census(&self) -> bool;
}

/// Trait for objects that can be revived during finalization.
///
/// Objects implementing this trait may be "revived" (marked as reachable again)
/// during the revival phase if their finalizer makes them reachable again.
///
/// # Examples
///
/// ```
/// use fugrip::traits::GcRevive;
/// use std::sync::atomic::{AtomicBool, Ordering};
///
/// struct ImportantData {
///     value: String,
///     preserve: AtomicBool,
/// }
///
/// impl GcRevive for ImportantData {
///     fn should_revive(&self) -> bool {
///         // Keep important data alive if preserve flag is set
///         self.preserve.load(Ordering::Acquire) || self.value.starts_with("IMPORTANT")
///     }
/// }
///
/// let data = ImportantData {
///     value: "IMPORTANT: Do not delete".to_string(),
///     preserve: AtomicBool::new(false),
/// };
///
/// // This would be called during GC revival phase
/// assert!(data.should_revive()); // Should be kept alive
///
/// let temp_data = ImportantData {
///     value: "temp data".to_string(),
///     preserve: AtomicBool::new(false),
/// };
///
/// assert!(!temp_data.should_revive()); // Can be collected
/// ```
pub trait GcRevive {
    /// Called during the revival phase to determine if this object should be revived.
    ///
    /// Returns `true` if the object should be kept alive for another collection cycle.
    fn should_revive(&self) -> bool;
}

/// Trait for custom marking strategies during garbage collection.
///
/// Advanced users can implement this trait to provide custom marking
/// logic for complex data structures or optimization purposes.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::{GcMark, GcMarker}, GcHeader};
///
/// struct CustomGraph {
///     nodes: Vec<*mut GcHeader<()>>,
/// }
///
/// impl GcMark for CustomGraph {
///     unsafe fn mark(&self, marker: &dyn GcMarker) {
///         // Custom marking logic for graph structure
///         for &node_ptr in &self.nodes {
///             if !node_ptr.is_null() {
///                 unsafe { marker.mark_object(node_ptr); }
///             }
///         }
///     }
/// }
/// ```
pub trait GcMark {
    /// Custom marking logic for this object.
    ///
    /// # Safety
    ///
    /// This method must correctly mark all reachable objects and not
    /// cause data races with concurrent markers.
    unsafe fn mark(&self, marker: &dyn GcMarker);
}

/// Trait for objects that provide custom marking functionality.
///
/// This trait is implemented by the garbage collector's marking subsystem
/// and provided to objects that implement `GcMark`.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::GcMarker, GcHeader};
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// struct TestMarker {
///     marked_count: AtomicUsize,
/// }
///
/// impl GcMarker for TestMarker {
///     unsafe fn mark_object(&self, ptr: *mut GcHeader<()>) {
///         // Mark the object (in real implementation would set mark bit)
///         self.marked_count.fetch_add(1, Ordering::Relaxed);
///     }
///
///     unsafe fn is_marked(&self, _ptr: *mut GcHeader<()>) -> bool {
///         // Check if marked (in real implementation would check mark bit)
///         self.marked_count.load(Ordering::Relaxed) > 0
///     }
/// }
/// ```
pub trait GcMarker {
    /// Mark an object as reachable.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid garbage-collected object.
    unsafe fn mark_object(&self, ptr: *mut GcHeader<()>);

    /// Check if an object is already marked.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid garbage-collected object.
    unsafe fn is_marked(&self, ptr: *mut GcHeader<()>) -> bool;
}

/// Trait for objects that can provide heap statistics.
///
/// This trait allows objects to report their memory usage and
/// contribute to overall heap statistics.
///
/// # Examples
///
/// ```
/// use fugrip::traits::GcStats;
/// use std::collections::HashMap;
///
/// struct TreeNode {
///     value: String,
///     children: Vec<Box<TreeNode>>,
/// }
///
/// impl GcStats for TreeNode {
///     fn size_bytes(&self) -> usize {
///         std::mem::size_of::<Self>() 
///             + self.value.len() 
///             + self.children.iter().map(|c| c.size_bytes()).sum::<usize>()
///     }
///
///     fn child_count(&self) -> usize {
///         self.children.len()
///     }
///
///     fn custom_stats(&self) -> HashMap<String, u64> {
///         let mut stats = HashMap::new();
///         stats.insert("value_len".to_string(), self.value.len() as u64);
///         stats.insert("depth".to_string(), 
///             self.children.iter().map(|c| c.custom_stats().get("depth")
///                 .unwrap_or(&0) + 1).max().unwrap_or(0));
///         stats
///     }
/// }
///
/// let node = TreeNode {
///     value: "root".to_string(),
///     children: vec![],
/// };
/// 
/// assert_eq!(node.child_count(), 0);
/// assert!(node.size_bytes() > 0);
/// ```
pub trait GcStats {
    /// Returns the size of this object in bytes.
    fn size_bytes(&self) -> usize;

    /// Returns the number of child objects referenced by this object.
    fn child_count(&self) -> usize;

    /// Returns additional statistics specific to this object type.
    fn custom_stats(&self) -> std::collections::HashMap<String, u64> {
        std::collections::HashMap::new()
    }
}

/// Trait for objects that can be visited during heap traversal.
///
/// This trait is used for debugging, profiling, and analysis tools
/// that need to traverse the entire heap structure.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::{GcVisitable, GcVisitor}, GcHeader};
///
/// struct Container {
///     items: Vec<*const GcHeader<()>>,
/// }
///
/// impl GcVisitable for Container {
///     fn accept_visitor<V: GcVisitor>(&self, visitor: &mut V) {
///         let self_ptr = self as *const Self as *const GcHeader<()>;
///         visitor.visit_object(self_ptr, std::mem::size_of::<Self>());
///         
///         for &item_ptr in &self.items {
///             visitor.visit_reference(self_ptr, item_ptr);
///         }
///     }
/// }
/// ```
pub trait GcVisitable {
    /// Accept a visitor for this object.
    fn accept_visitor<V: GcVisitor>(&self, visitor: &mut V);
}

/// Trait for visitors that can process objects during heap traversal.
///
/// This trait is used in conjunction with `GcVisitable` to implement
/// the visitor pattern for heap analysis.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::GcVisitor, GcHeader};
///
/// struct ObjectCounter {
///     object_count: usize,
///     reference_count: usize,
///     total_bytes: usize,
/// }
///
/// impl GcVisitor for ObjectCounter {
///     fn visit_object(&mut self, _ptr: *const GcHeader<()>, size: usize) {
///         self.object_count += 1;
///         self.total_bytes += size;
///     }
///
///     fn visit_reference(&mut self, _from: *const GcHeader<()>, _to: *const GcHeader<()>) {
///         self.reference_count += 1;
///     }
/// }
///
/// let mut counter = ObjectCounter {
///     object_count: 0,
///     reference_count: 0,
///     total_bytes: 0,
/// };
/// ```
pub trait GcVisitor {
    /// Visit an object during heap traversal.
    fn visit_object(&mut self, ptr: *const GcHeader<()>, size: usize);

    /// Visit a reference between objects.
    fn visit_reference(&mut self, from: *const GcHeader<()>, to: *const GcHeader<()>);
}

/// Trait for objects that can be serialized from the garbage-collected heap.
///
/// This trait provides a safe way to serialize objects that may contain
/// circular references or other complex structures.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::{GcSerialize, GcSerializeContext}, GcHeader};
/// use std::io::Write;
///
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// impl GcSerialize for Person {
///     fn serialize<W: Write>(
///         &self,
///         writer: &mut W,
///         _context: &mut dyn GcSerializeContext,
///     ) -> std::io::Result<()> {
///         write!(writer, "{}:{}", self.name, self.age)
///     }
/// }
///
/// // Usage would typically be with a proper serialization context
/// ```
pub trait GcSerialize {
    /// Serialize this object to the given writer.
    ///
    /// The serializer should handle circular references appropriately.
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
        context: &mut dyn GcSerializeContext,
    ) -> std::io::Result<()>;
}

/// Context for serialization operations that handles circular references.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::GcSerializeContext, GcHeader};
/// use std::collections::HashSet;
///
/// struct SimpleSerializeContext {
///     visited: HashSet<*const GcHeader<()>>,
///     next_id: u64,
/// }
///
/// impl GcSerializeContext for SimpleSerializeContext {
///     fn is_visited(&self, ptr: *const GcHeader<()>) -> bool {
///         self.visited.contains(&ptr)
///     }
///
///     fn mark_visited(&mut self, ptr: *const GcHeader<()>) {
///         self.visited.insert(ptr);
///     }
///
///     fn get_object_id(&self, ptr: *const GcHeader<()>) -> u64 {
///         ptr as u64
///     }
/// }
/// ```
pub trait GcSerializeContext {
    /// Check if an object has already been serialized.
    fn is_visited(&self, ptr: *const GcHeader<()>) -> bool;

    /// Mark an object as visited during serialization.
    fn mark_visited(&mut self, ptr: *const GcHeader<()>);

    /// Get a unique identifier for an object.
    fn get_object_id(&self, ptr: *const GcHeader<()>) -> u64;
}

/// Trait for objects that can be deserialized into the garbage-collected heap.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::{GcDeserialize, GcDeserializeContext}};
/// use std::io::{Read, Result};
///
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// impl GcDeserialize for Person {
///     fn deserialize<R: Read>(
///         reader: &mut R,
///         _context: &mut dyn GcDeserializeContext,
///     ) -> Result<Self> {
///         let mut buffer = String::new();
///         reader.read_to_string(&mut buffer)?;
///         
///         let parts: Vec<&str> = buffer.split(':').collect();
///         if parts.len() != 2 {
///             return Err(std::io::Error::new(
///                 std::io::ErrorKind::InvalidData,
///                 "Invalid format"
///             ));
///         }
///         
///         Ok(Person {
///             name: parts[0].to_string(),
///             age: parts[1].parse().map_err(|_| 
///                 std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid age"))?,
///         })
///     }
/// }
/// ```
pub trait GcDeserialize: Sized {
    /// Deserialize an object from the given reader.
    fn deserialize<R: std::io::Read>(
        reader: &mut R,
        context: &mut dyn GcDeserializeContext,
    ) -> std::io::Result<Self>;
}

/// Context for deserialization operations.
///
/// # Examples
///
/// ```
/// use fugrip::{traits::GcDeserializeContext, GcHeader};
/// use std::collections::HashMap;
///
/// struct SimpleDeserializeContext {
///     objects: HashMap<u64, *mut GcHeader<()>>,
/// }
///
/// impl GcDeserializeContext for SimpleDeserializeContext {
///     fn register_object(&mut self, id: u64, ptr: *mut GcHeader<()>) {
///         self.objects.insert(id, ptr);
///     }
///
///     fn get_object(&self, id: u64) -> Option<*mut GcHeader<()>> {
///         self.objects.get(&id).copied()
///     }
/// }
///
/// let mut context = SimpleDeserializeContext {
///     objects: HashMap::new(),
/// };
/// ```
pub trait GcDeserializeContext {
    /// Register a deserialized object with a given ID.
    fn register_object(&mut self, id: u64, ptr: *mut GcHeader<()>);

    /// Get a previously deserialized object by ID.
    fn get_object(&self, id: u64) -> Option<*mut GcHeader<()>>;
}

/// Helper macro to implement GcTrace for simple types.
///
/// This macro provides a convenient way to implement GcTrace for types
/// that don't contain any GC-managed pointers.
///
/// # Examples
///
/// ```
/// use fugrip::{gc_traceable, GcTrace, SendPtr, GcHeader};
///
/// #[derive(Debug)]
/// struct MyType {
///     value: i32,
/// }
///
/// gc_traceable!(MyType);
///
/// // Now MyType implements GcTrace with a no-op trace function
/// let obj = MyType { value: 42 };
/// let mut stack = Vec::<SendPtr<GcHeader<()>>>::new();
/// unsafe { obj.trace(&mut stack); }
/// assert!(stack.is_empty());
/// ```
#[macro_export]
macro_rules! gc_traceable {
    ($type:ty) => {
        unsafe impl $crate::traits::GcTrace for $type {
            unsafe fn trace(&self, _stack: &mut Vec<$crate::SendPtr<$crate::GcHeader<()>>>) {
                // No GC references to trace in this type
            }
        }
    };
}

/// Internal helper trait for tracing "strong" GC edges on specific field shapes.
///
/// Implementations push header pointers for strong references onto the mark stack.
/// Weak references intentionally do not implement this and should be skipped.
pub trait StrongTraceFields {
    fn trace_strong_fields(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>);
}

impl<T: GcTrace + 'static> StrongTraceFields for crate::Gc<T> {
    fn trace_strong_fields(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        // Safety: GC owns the header; we're only pushing the header pointer
        let ptr = self.as_ptr();
        if !ptr.is_null() {
            // Safety: header pointer comes from a managed Gc
            stack.push(unsafe { SendPtr::new(ptr as *mut GcHeader<()>) });
        }
    }
}

impl<T: GcTrace + 'static> StrongTraceFields for Option<crate::Gc<T>> {
    fn trace_strong_fields(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        if let Some(g) = self {
            g.trace_strong_fields(stack);
        }
    }
}

impl<T: GcTrace + 'static> StrongTraceFields for Vec<crate::Gc<T>> {
    fn trace_strong_fields(&self, stack: &mut Vec<SendPtr<GcHeader<()>>>) {
        for g in self.iter() {
            g.trace_strong_fields(stack);
        }
    }
}

/// Pragmatic hybrid macro: implement GcTrace for a type by listing strong fields.
///
/// This macro traces only the provided fields using StrongTraceFields, and skips
/// any weak references or other non-owning fields.
///
/// Example:
///
/// gc_trace_strong!(WeakHolderNode, next, children);
#[macro_export]
macro_rules! gc_trace_strong {
    ($type:ty, $( $field:ident ),+ $(,)? ) => {
        unsafe impl $crate::traits::GcTrace for $type {
            unsafe fn trace(&self, stack: &mut Vec<$crate::SendPtr<$crate::GcHeader<()>>>) {
                $( $crate::traits::StrongTraceFields::trace_strong_fields(&self.$field, stack); )+
            }
        }
    };
}
