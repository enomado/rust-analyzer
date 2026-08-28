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
//! Two things, both of which rowan certainly stores inline:
//!
//! * the child slots of every green node, and
//! * the text of every green token.
//!
//! It leaves out the per-allocation headers (refcount, kind, text length).
//! Those are real bytes, but their layout is rowan's private business, and a
//! number built on a guess about a private layout would keep looking right
//! while quietly drifting. So this is a LOWER BOUND, like the rest of the
//! `heap_size` family — and the useful direction, since a caller comparing
//! ingredients wants "at least this much".
//!
//! The one size taken from rowan is the child slot, and rowan asserts it
//! itself (`static_assert!(size_of::<GreenChild>() == size_of::<usize>() * 2)`
//! in `green/node.rs`) rather than leaving it to be inferred here.
//!
//! # Why the walk deduplicates
//!
//! rowan interns identical subtrees while parsing, so one allocation can hang
//! under many parents — every `;` token in a file is typically the same token.
//! Following the tree naively would count those allocations once per reference
//! and report several times the real heap. Over-counting is worse than
//! under-counting: it would move an ingredient up the report on bytes that do
//! not exist. Hence the two sets of visited addresses.

use std::{mem::size_of, ptr};

use rowan::{GreenNodeData, GreenTokenData, NodeOrToken};
use rustc_hash::FxHashSet;

use crate::{Parse, SyntaxError};

/// `heap_size` for the `parse` query.
pub fn parse<T>(parse: &Parse<T>) -> usize {
    let green = parse.green.as_ref().map_or(0, |green| green_tree(green));
    // The error list is owned here rather than shared with the `parse_errors`
    // query, which hands out a fresh `Box<[_]>`, so counting it here is not a
    // double count.
    let errors =
        parse.errors.as_deref().map_or(0, |errors| errors.len() * size_of::<SyntaxError>());
    green + errors
}

/// One pointer pair per child, as asserted in rowan's `green/node.rs`.
const CHILD_SLOT: usize = 2 * size_of::<usize>();

fn green_tree(root: &GreenNodeData) -> usize {
    let mut seen_nodes: FxHashSet<*const GreenNodeData> = FxHashSet::default();
    let mut seen_tokens: FxHashSet<*const GreenTokenData> = FxHashSet::default();
    // An explicit worklist rather than recursion: the depth here is the nesting
    // depth of the *parsed source*, so a generated or pathological file would
    // otherwise decide how much stack this needs.
    let mut worklist = vec![root];
    let mut total = 0;

    while let Some(node) = worklist.pop() {
        if !seen_nodes.insert(ptr::from_ref(node)) {
            continue;
        }
        let mut slots = 0;
        for child in node.children() {
            slots += 1;
            match child {
                NodeOrToken::Node(node) => worklist.push(node),
                NodeOrToken::Token(token) => {
                    if seen_tokens.insert(ptr::from_ref(token)) {
                        total += token.text().len();
                    }
                }
            }
        }
        total += slots * CHILD_SLOT;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edition, SourceFile};

    /// The same walk with either visited-set switchable off, so a test can ask
    /// what each half of the deduplication is worth on its own. Comparing
    /// against one walk with *both* off would pass while half the filtering was
    /// broken — which is exactly what happened when this was written.
    fn walk(root: &GreenNodeData, dedup_nodes: bool, dedup_tokens: bool) -> usize {
        let mut seen_nodes: FxHashSet<*const GreenNodeData> = FxHashSet::default();
        let mut seen_tokens: FxHashSet<*const GreenTokenData> = FxHashSet::default();
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
                            total += token.text().len();
                        }
                    }
                }
            }
            total += slots * CHILD_SLOT;
        }
        total
    }

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
        assert_eq!(
            deduped,
            walk(green, true, true),
            "the test's own walk no longer agrees with the real one; whatever changed in \
             `green_tree` has left the comparisons below measuring something else"
        );

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
