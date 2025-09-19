#[test]
fn debug_chunk_mask() {
    use crate::simd_sweep::SimdBitvector;
    use mmtk::util::Address;
    
    let heap_base = unsafe { Address::from_usize(0x37000000) };
    let bitvector = SimdBitvector::new(heap_base, 10000, 16);
    
    println!("Total objects: {}", bitvector.max_objects);
    println!("Objects per chunk: {}", bitvector.objects_per_chunk);
    println!("Chunk count: {}", bitvector.chunk_count);
    println!("Words per chunk: {}", bitvector.words_per_chunk);
    
    let last_chunk = bitvector.chunk_count - 1;
    println!("Last chunk index: {}", last_chunk);
    
    let capacity = bitvector.get_chunk_object_capacity(last_chunk);
    println!("Last chunk capacity: {}", capacity);
    
    if capacity > 0 {
        let mask = bitvector.create_chunk_object_mask(last_chunk);
        println!("Mask created with {} words", mask.word_masks.len());
        if let Some(last_word) = mask.word_masks.last() {
            println!("Last word mask: {:064b}", last_word);
            println!("Bit count: {}", last_word.count_ones());
        }
    }
}
