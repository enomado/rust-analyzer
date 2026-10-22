//! Oracle for `base_db::file_text_heap_size`: a file text's allocation,
//! counted by the allocator itself (`test_utils::counting_alloc`; one
//! `#[test]`, for the reason given there).
//!
//! The function computes triomphe's `Arc<str>` layout instead of asking for
//! it; this measures `Arc::from(text)` — what every setter of `FileText` does —
//! and requires the two to agree to the byte.

use base_db::{file_text_heap_size, file_text_store::StoredText, loaded_file_text_heap_size};
use test_utils::counting_alloc::{CountingAllocator, live_bytes};
use triomphe::Arc;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn heap_size_of_a_file_text_is_what_the_allocator_holds_for_it() {
    // Lengths on both sides of the alignment, and a real file.
    let texts: [&str; 6] =
        ["", "a", "1234567", "12345678", "123456789", include_str!("../src/lib.rs")];
    // Reserved up front, so that pushing into it allocates outside every
    // measured window.
    let mut alive: Vec<Arc<str>> = Vec::with_capacity(texts.len());
    for text in texts {
        let before = live_bytes();
        let arc: Arc<str> = Arc::from(text);
        let live = live_bytes() - before;
        let reported =
            file_text_heap_size(&(StoredText::Inline(arc.clone()), vfs::FileId::from_raw(0)));
        // A text loaded out of the store is the same `Arc::from`, owned by the
        // memo instead of the input.
        let loaded = loaded_file_text_heap_size(&arc);
        alive.push(arc);
        assert_eq!(
            live,
            reported as isize,
            "a {}-byte text: the allocator holds {live} bytes and `file_text_heap_size` reports \
             {reported} — triomphe's `Arc<str>` layout no longer matches the arithmetic",
            text.len()
        );
        assert_eq!(
            live,
            loaded as isize,
            "a {}-byte loaded text: the allocator holds {live} bytes and \
             `loaded_file_text_heap_size` reports {loaded}",
            text.len()
        );
    }
}
