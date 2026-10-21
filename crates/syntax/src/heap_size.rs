//! What a parsed file costs on the heap.
//!
//! salsa collects this through the `heap_size = <fn>` option on a tracked
//! function and reports it from `salsa::Database::memory_usage`. The parse
//! query is memoized per file and capped by an LRU, so its line in that report
//! is the difference between "3930 files are parsed" and "the parses are worth
//! N megabytes" — the second is the one a capacity can be argued from.
//!
//! # What the number counts, and what it leaves out
//!
//! Every allocation the parse owns, at the size it was requested with:
//!
//! * each green node and green token — rowan's `ThinArc`: the reference count,
//!   the header slice (header and length), then the child slots or the text,
//!   rounded up to the alignment. That is rowan's own arithmetic in
//!   `ThinArc::from_header_and_iter` (`arc.rs`); see [`thin_arc`];
//! * the error list — a `triomphe::Arc<[SyntaxError]>` — and each message.
//!
//! It leaves out what the allocator adds per chunk (its header, its size
//! classes): that belongs to the allocator, is the same for every ingredient,
//! and shows up in the process portrait as fragmentation, not here.
//!
//! An earlier version counted only the child slots and the token text, on the
//! grounds that rowan's allocation layout is private and a number built on it
//! would drift silently. It undercounted several times over — a `;` token is a
//! 32-byte allocation and was charged one byte — which left gigabytes of parse
//! trees in the daemon's "unattributed" line. The drift is now caught instead
//! of avoided: `tests/heap_size_oracle.rs` parses real files under a counting
//! global allocator and requires the live bytes to equal this number exactly,
//! so a rowan change to the layout fails that test rather than the report.
//!
//! # Why the walk deduplicates
//!
//! rowan interns identical subtrees while parsing, so one allocation can hang
//! under many parents — every `;` token in a file is typically the same token.
//! Following the tree naively would count those allocations once per reference
//! and report several times the real heap. Over-counting is worse than
//! under-counting: it would move an ingredient up the report on bytes that do
//! not exist. Hence the two sets of visited addresses.

use std::{
    mem::{align_of, size_of},
    ptr,
};

use rowan::{GreenNodeData, GreenTokenData, NodeOrToken};
use rustc_hash::FxHashSet;

use crate::{Parse, SyntaxError};

/// `heap_size` for the `parse` query.
pub fn parse<T>(parse: &Parse<T>) -> usize {
    let green = parse.green.as_ref().map_or(0, |green| green_tree(green));
    // The error list is owned here rather than shared with the `parse_errors`
    // query, which hands out a fresh `Box<[_]>`, so counting it here is not a
    // double count.
    let errors = parse.errors.as_deref().map_or(0, error_list);
    green + errors
}

/// One pointer pair per child, as asserted in rowan's `green/node.rs`.
const CHILD_SLOT: usize = 2 * size_of::<usize>();

/// The reference count both `ArcInner`s (rowan's and triomphe's) put before
/// the data.
const ARC_COUNT: usize = size_of::<usize>();

/// One rowan `ThinArc` allocation: the count, the header slice, `items` bytes
/// of slots or text, rounded up to the alignment.
///
/// `head` is the size of the public zero-length repr (`GreenNodeData` /
/// `GreenTokenData`, a `HeaderSlice<_, [_; 0]>`): header and length, with the
/// slice starting right after them. The alignment is the larger of the count's
/// and that repr's, which is how `ArcInner<HeaderSlice<H, [T; 0]>>` gets its own.
fn thin_arc(head: usize, head_align: usize, items: usize) -> usize {
    (ARC_COUNT + head + items).next_multiple_of(align_of::<usize>().max(head_align))
}

fn node_allocation(slots: usize) -> usize {
    thin_arc(size_of::<GreenNodeData>(), align_of::<GreenNodeData>(), slots * CHILD_SLOT)
}

fn token_allocation(token: &GreenTokenData) -> usize {
    thin_arc(size_of::<GreenTokenData>(), align_of::<GreenTokenData>(), token.text().len())
}

/// `triomphe::Arc<[SyntaxError]>`: the count, then the slice, padded; plus each
/// message's own buffer.
fn error_list(errors: &[SyntaxError]) -> usize {
    let list = (ARC_COUNT + size_of_val(errors))
        .next_multiple_of(align_of::<usize>().max(align_of::<SyntaxError>()));
    list + errors.iter().map(SyntaxError::message_heap).sum::<usize>()
}

fn green_tree(root: &GreenNodeData) -> usize {
    walk(root, true, true)
}

/// The walk behind [`green_tree`], with either visited-set switchable off so a
/// test can ask what each half of the deduplication is worth on its own.
/// Comparing against one walk with *both* off would pass while half the
/// filtering was broken — which is exactly what happened when this was written.
fn walk(root: &GreenNodeData, dedup_nodes: bool, dedup_tokens: bool) -> usize {
    let mut seen_nodes: FxHashSet<*const GreenNodeData> = FxHashSet::default();
    let mut seen_tokens: FxHashSet<*const GreenTokenData> = FxHashSet::default();
    // An explicit worklist rather than recursion: the depth here is the nesting
    // depth of the *parsed source*, so a generated or pathological file would
    // otherwise decide how much stack this needs.
    let mut worklist = vec![root];
    let mut total = 0;

    while let Some(node) = worklist.pop() {
        if dedup_nodes && !seen_nodes.insert(ptr::from_ref(node)) {
            continue;
        }
        let mut slots = 0;
        for child in node.children() {
            slots += 1;
            match child {
                NodeOrToken::Node(node) => worklist.push(node),
                NodeOrToken::Token(token) => {
                    if !dedup_tokens || seen_tokens.insert(ptr::from_ref(token)) {
                        total += token_allocation(token);
                    }
                }
            }
        }
        total += node_allocation(slots);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edition, SourceFile};

    fn parse_source(text: &str) -> crate::Parse<SourceFile> {
        SourceFile::parse(text, Edition::CURRENT)
    }

    /// Positive control for the deduplication, one assertion per half: rowan
    /// interns whole subtrees *and* individual tokens, so both sets have to
    /// filter something or the report is inflated by shared allocations
    /// charged once per parent.
    #[test]
    fn a_shared_allocation_is_charged_once() {
        // Repetition on purpose: these statements share both their subtrees
        // and their tokens.
        let source = "fn f() { let a = 1; let a = 1; let a = 1; let a = 1; let a = 1; }";
        let parse = parse_source(source);
        let green = parse.green.as_ref().unwrap();

        let deduped = green_tree(green);
        assert!(deduped > 0, "a parsed file cannot cost nothing");

        let without_node_dedup = walk(green, false, true);
        let without_token_dedup = walk(green, true, false);
        assert!(
            deduped < without_node_dedup,
            "charging shared *nodes* once per parent came to the same {deduped} bytes: either \
             this source shares no subtrees (pick a more repetitive one) or the node set has \
             stopped filtering"
        );
        assert!(
            deduped < without_token_dedup,
            "charging shared *tokens* once per parent came to the same {deduped} bytes: either \
             this source shares no tokens (pick a more repetitive one) or the token set has \
             stopped filtering. This is the half a single both-off comparison misses, because \
             node sharing alone is enough to make that one pass"
        );
    }

    /// Guards the direction of the number: it has to grow with the file, or it
    /// is not measuring the file.
    #[test]
    fn more_code_costs_more() {
        let small = parse_source("fn f() {}");
        let large = parse_source(&"fn f() { let x = 1; }\n".repeat(200));

        assert!(parse(&large) > parse(&small), "200 functions did not weigh more than one");
    }
}
