//! Oracle for `syntax::heap_size::parse`: the bytes a parse keeps alive,
//! counted by the allocator itself.
//!
//! `heap_size::parse` computes the size of every allocation of a parse from
//! rowan's layout arithmetic. That layout is rowan's private business, so the
//! number is only trustworthy while something checks it against reality. This
//! binary installs a global allocator that keeps a running total of requested
//! bytes; parsing a file and holding the result moves that total by exactly
//! what the parse owns, because everything else the parser allocated (the
//! builder, its node cache, the lexer's buffers) is freed by the time it
//! returns.
//!
//! An integration test and not a unit test, because the global allocator is
//! per binary: here it observes nothing but this file. There is one `#[test]`
//! for the same reason — two tests on two threads would see each other's
//! allocations in the shared total.
//!
//! 🚨 No parse may be dropped while the test runs. `Parse` drops its tree on a
//! background thread (`ParseNodeDropper`, `lib.rs`), so a tree dropped before a
//! measurement is freed *during* a later one, at a moment nobody chooses. The
//! first version did exactly that: the warm-up tree was freed while the same
//! file was parsed again, and the two cancelled to a delta of 0. Every parse
//! is kept alive until the end instead.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use syntax::{Edition, Parse, SourceFile};

/// Requested bytes currently allocated, process-wide.
static LIVE: AtomicIsize = AtomicIsize::new(0);

struct Counting;

// SAFETY: every call is forwarded to `System` unchanged; the counter only
// observes sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            LIVE.fetch_add(new_size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Parses `text` and returns the parse with the live bytes it added. The
/// caller keeps the parse alive; see the module comment.
fn parse_measured(text: &str) -> (Parse<SourceFile>, isize) {
    let before = LIVE.load(Ordering::Relaxed);
    let parse = SourceFile::parse(text, Edition::CURRENT);
    (parse, LIVE.load(Ordering::Relaxed) - before)
}

#[test]
fn heap_size_of_a_parse_is_what_the_allocator_holds_for_it() {
    // Real sources, the largest generated file among them, plus one broken
    // enough to carry an error list with messages.
    let fixtures: [(&str, &str); 4] = [
        ("ast/generated/nodes.rs", include_str!("../src/ast/generated/nodes.rs")),
        ("lib.rs", include_str!("../src/lib.rs")),
        ("heap_size.rs", include_str!("../src/heap_size.rs")),
        ("broken", "fn f( { let = ; struct } impl for {{ 1 +"),
    ];
    // Reserved up front, so that pushing into it allocates outside every
    // measured window.
    let mut alive: Vec<Parse<SourceFile>> = Vec::with_capacity(fixtures.len() + 1);

    // A first parse pays one-time costs (lazily initialised tables) that stay
    // allocated and belong to no parse; take them out of the comparison.
    alive.push(parse_measured(fixtures[0].1).0);

    let mut with_errors = 0;
    for (name, text) in fixtures {
        let (parse, live) = parse_measured(text);
        // Read after the measurement: the walk allocates visited sets of its own.
        let reported = syntax::heap_size::parse(&parse);
        if !parse.errors().is_empty() {
            with_errors += 1;
        }
        alive.push(parse);
        assert!(live > 0, "`{name}`: a parse that holds nothing measures nothing");
        assert_eq!(
            live, reported as isize,
            "`{name}`: the allocator holds {live} bytes for the parse and `heap_size` reports \
             {reported} — rowan's allocation layout (or the error list's) no longer matches the \
             arithmetic in `heap_size.rs`"
        );
    }
    assert!(with_errors > 0, "no fixture has errors: the error-list half went unchecked");
}
