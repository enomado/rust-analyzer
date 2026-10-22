//! Oracle for `DeclarativeMacro::heap_size`: what a compiled macro owns,
//! counted by the allocator itself (`test_utils::counting_alloc`; one
//! `#[test]`, for the reason given there).
//!
//! Each macro is compiled twice from the same token tree and only the second
//! compilation is measured: the first interns every new `Symbol`, which the
//! allocator would otherwise count here while `heap_size` (rightly) leaves it
//! to the interner.

use mbe::DeclarativeMacro;
use syntax::{
    AstNode,
    ast::{self, HasName},
};
use syntax_bridge::{
    DocCommentDesugarMode,
    dummy_test_span_utils::{DUMMY, DummyTestSpanMap},
    syntax_node_to_token_tree,
};
use test_utils::counting_alloc::{CountingAllocator, live_bytes};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Every kind of `Op` the parser makes, every separator kind, an empty rule
/// side, and a definition that fails to parse (its `ParseError` owns text).
const EVERY_OP: &str = r#"
macro_rules! every_op {
    ($name:ident, $($x:expr),* ; $($l:lifetime)'a* ; $($t:tt)=>+ ; $(_ $u:ident)?) => {
        ${ignore($x)} ${index()} ${len()} ${count($x)}
        ${concat($name, _suffix, 1)}
        fn $name() { [$($x),*]; ($($t)*) }
        "literal" 1u8 'c' ::<>=== ident
    };
    () => {};
    (attr() $i:item) => { $i };
}
macro_rules! broken {
    ($x:expr) => { $x } garbage more
}
macro_rules! no_subtree {
    x => {}
}
macro_rules! empty_repeat {
    ($()*) => {}
}
"#;

#[test]
fn heap_size_of_a_compiled_macro_is_what_the_allocator_holds_for_it() {
    let fixture = format!("{EVERY_OP}\n{}", test_utils::bench_fixture::numerous_macro_rules());
    let file = ast::SourceFile::parse(&fixture, span::Edition::CURRENT).ok().unwrap();
    let trees: Vec<(String, tt::TopSubtree)> = file
        .syntax()
        .descendants()
        .filter_map(ast::MacroRules::cast)
        .map(|rules| {
            let tt = syntax_node_to_token_tree(
                rules.token_tree().unwrap().syntax(),
                DummyTestSpanMap,
                DUMMY,
                DocCommentDesugarMode::Mbe,
            );
            (rules.name().unwrap().to_string(), tt)
        })
        .collect();
    // Against vacuity: the corpus is the real one, and the fixture's own
    // definitions are in it, the failing ones included.
    assert!(trees.len() > 100, "only {} macros in the corpus", trees.len());
    for name in ["every_op", "broken", "no_subtree", "empty_repeat"] {
        assert!(trees.iter().any(|(n, _)| n == name), "`{name}` missing from the corpus");
    }

    let edition = |_| span::Edition::CURRENT;
    let mut errors = 0;
    // Reserved up front, so that pushing into it allocates outside every
    // measured window.
    let mut alive: Vec<DeclarativeMacro> = Vec::with_capacity(trees.len());
    for (name, tt) in &trees {
        drop(DeclarativeMacro::parse_macro_rules(tt, edition));
        let before = live_bytes();
        let mac = DeclarativeMacro::parse_macro_rules(tt, edition);
        let live = live_bytes() - before;
        let reported = mac.heap_size();
        errors += usize::from(mac.err().is_some());
        assert_eq!(
            live, reported as isize,
            "`{name}`: the allocator holds {live} bytes for the compiled macro and \
             `heap_size` reports {reported}"
        );
        alive.push(mac);
    }
    assert!(errors >= 2, "the failing definitions must fail, or their error is not weighed");
}
