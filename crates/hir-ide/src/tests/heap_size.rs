//! The inference heap that `hir_ty::heap_size` claims, against fixtures.
//!
//! These live here rather than next to the arithmetic because `hir-ty`'s tests
//! moved into this crate: they need the full database, which only the leaf
//! crate can build.

use base_db::EditionedFileId;
use hir_def::{DefWithBodyId, ModuleDefId, nameres::crate_def_map};
use test_fixture::WithFixture;

use crate::{InferenceResult, test_db::TestDB};

/// The inference heap of every top-level function in `file_id`.
///
/// Inference runs under `attach_db` because the type interner reaches for
/// the database through a thread-local rather than through an argument, and
/// panics when it is not there.
fn heap_of(db: &TestDB, file_id: EditionedFileId) -> usize {
    crate::attach_db(db, || {
        let module = db.module_for_file(file_id.file_id(db));
        crate_def_map(db, module.krate(db))[module]
            .scope
            .declarations()
            .filter_map(|def| match def {
                ModuleDefId::FunctionId(it) => Some(DefWithBodyId::FunctionId(it)),
                _ => None,
            })
            .map(|def| InferenceResult::of(db, def).heap_size())
            .sum()
    })
}

fn heap(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> usize {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    heap_of(&db, file_id)
}

/// [`heap`] for a fixture that needs `minicore`, which makes it multi-file.
fn heap_with_core(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> usize {
    let (db, files) = TestDB::with_many_files(ra_fixture);
    heap_of(&db, files[0])
}

/// One statement per line of the same shape, as a fixture body.
fn body(statements: impl Iterator<Item = String>) -> String {
    format!("fn f() {{ {} }}", statements.collect::<String>())
}

/// Over-counting is the failure mode this family cannot afford: a number
/// too big moves its ingredient up the report on bytes that do not exist,
/// and every wrong answer still looks plausible. An empty body owns nearly
/// nothing, so its weight is where an over-count shows up undisguised.
///
/// Measured while this was written: 8 bytes per expression statement, flat
/// from 200 of them to 2000 — one `Option<StoredTy>` slot each, which is
/// what the arithmetic claims to count and nothing besides.
#[test]
fn an_empty_body_owns_almost_nothing() {
    let empty = heap("fn f() {}");
    assert!(
        empty < 1024,
        "an empty function body reported {empty} bytes of heap — that is an over-count, and \
         every other number in this family carries the same error"
    );
}

/// The dense half: a type is recorded for every expression in the body, and
/// that map is where most of a big body's bytes are.
///
/// Both fixtures are pure expression statements — no patterns, no bindings,
/// no closures — so the only term that can move between them is
/// `type_of_expr`, and dropping it takes the difference to zero.
#[test]
fn a_body_with_more_expressions_has_a_heavier_inference_result() {
    let one = heap("fn f() { 1; }");
    let many = heap(&body((0..200).map(|_| "1;".to_owned())));

    assert!(
        many > one,
        "two hundred expressions weighed no more than one ({one} bytes), so the per-expression \
         type map is not reaching the total"
    );
}

/// Patterns and bindings, pinned one at a time.
///
/// The three fixtures hold the expression count fixed — `1;`, `let _ = 1;`
/// and `let x = 1;` are one expression each — and add exactly one family
/// per step: first a pattern, then a binding for that pattern. A single
/// comparison would not do it: growing `let _` into `let x` moves the
/// pattern-keyed maps too, so only the pair separates the two halves.
///
/// # What the second assertion does NOT pin
///
/// `type_of_binding` and `binding_modes` move together and cannot be
/// separated by any fixture — a binding pattern always gets both — so the
/// second comparison is a claim about the two of them jointly. Verified by
/// mutation: dropping either one alone leaves this test green, dropping
/// both turns it red.
#[test]
fn patterns_and_bindings_are_counted_beside_the_expressions() {
    const COUNT: usize = 200;
    let expressions = heap(&body((0..COUNT).map(|_| "1;".to_owned())));
    let wildcards = heap(&body((0..COUNT).map(|_| "let _ = 1;".to_owned())));
    let named = heap(&body((0..COUNT).map(|i| format!("let x{i} = 1;"))));

    assert!(
        wildcards > expressions,
        "{COUNT} wildcard `let`s weighed no more than {COUNT} bare expressions \
         ({expressions} bytes) — the pattern map is going uncounted"
    );
    assert!(
        named > wildcards,
        "{COUNT} named `let`s weighed no more than {COUNT} wildcard ones ({wildcards} bytes) \
         — the binding-keyed maps are going uncounted"
    );
}

/// The one place a *value* in these maps owns a container of its own: the
/// capture list of a closure.
///
/// Both closures have the same shape — twenty variable references in a
/// twenty-element tuple — so expressions, patterns and bindings match on
/// both sides and only the capture set differs: one root variable against
/// twenty. Skipping `closures_data`, or skipping `min_captures` inside it,
/// makes the two equal.
#[test]
fn a_closure_that_captures_more_variables_weighs_more() {
    const COUNT: usize = 20;
    let lets = (0..COUNT).map(|i| format!("let a{i} = 1;")).collect::<String>();
    let closure = |captures: &dyn Fn(usize) -> String| {
        let elements = (0..COUNT).map(captures).collect::<Vec<_>>().join(", ");
        format!(
            "//- minicore: fn
//- /main.rs
fn f() {{ {lets} let c = || ({elements}); }}"
        )
    };

    let one_root = heap_with_core(&closure(&|_| "a0".to_owned()));
    let every_root = heap_with_core(&closure(&|i| format!("a{i}")));

    assert!(
        every_root > one_root,
        "a closure capturing {COUNT} variables weighed no more than one capturing a single \
         variable {COUNT} times ({one_root} bytes) — the per-closure capture list is not \
         reaching the total"
    );
}

/// The impl-lowering queries own their diagnostics box and nothing else: the
/// lowered value is a handle into the solver's interners. So a clean impl
/// header weighs zero — measured, not missing — and a header with a
/// diagnostic weighs the box. Read through salsa's report, so the
/// `heap_size = …` wiring is under test with the arithmetic.
#[test]
fn impl_lowering_weighs_its_diagnostics_and_nothing_else() {
    /// `(impl_self_ty, impl_trait)` heaps after lowering every impl in the file.
    fn heaps(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (usize, usize) {
        let (mut db, file_id) = TestDB::with_single_file(ra_fixture);
        crate::attach_db(&db, || {
            let module = db.module_for_file(file_id.file_id(&db));
            for impl_id in crate_def_map(&db, module.krate(&db))[module].scope.impls() {
                hir_ty::db::HirDatabase::impl_self_ty(&db, impl_id);
                hir_ty::db::HirDatabase::impl_trait(&db, impl_id);
            }
        });
        let usage = <dyn salsa::Database>::memory_usage(&mut db);
        let heap = |query: &str| {
            let info = &usage.queries[query];
            assert!(info.count() > 0, "positive control: `{query}` has no memos at all");
            info.heap_size_of_fields().unwrap_or_else(|| panic!("`{query}` reports no heap size"))
        };
        (heap("impl_self_ty_with_diagnostics"), heap("impl_trait_with_diagnostics"))
    }

    let clean = heaps("trait Tr {} struct S; impl Tr for S {} impl S {}");
    assert_eq!(clean, (0, 0), "a clean impl header owns no heap beyond interned handles");

    // Generic arguments on a primitive are a lowering diagnostic.
    let (self_ty, trait_ref) = heaps("trait Tr {} impl Tr for u8<i32> {} impl u16<i32> {}");
    assert!(self_ty > 0, "a self type lowered with a diagnostic weighed nothing");
    assert_eq!(trait_ref, 0, "the trait half of those impls is clean");
}
