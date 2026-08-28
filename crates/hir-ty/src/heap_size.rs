//! Weighing the heap that memoized values in this crate hang on to.
//!
//! Same contract as [`hir_def::heap_size`], whose container arithmetic this
//! reuses rather than restating: every number is a LOWER BOUND on owned
//! buffers, interned handles are followed by nobody, and a shared allocation is
//! charged to exactly one owner. That module's header is the long form.
//!
//! # Why inference is worth weighing
//!
//! [`crate::infer::InferenceResult`] is one struct with two dozen maps, arenas
//! and sets on it — a type for every expression, pattern and binding in a body,
//! plus the resolutions of every method call and field access. Salsa can see
//! none of that: without `heap_size` it reports the struct's own stack size and
//! a memo count, and a body with ten thousand expressions weighs the same as an
//! empty one.
//!
//! It is also the one query whose LRU capacity this fork picked for itself
//! (`lru = 2024` on `InferenceResult::for_body`, chosen by analogy with
//! borrowck rather than from a measurement). A capacity is only arguable
//! against bytes, and these are the bytes.

use hir_def::heap_size::{index_map_with, slice, small_vec, vec};

use crate::infer::{ClosureData, InferenceResult};

/// `heap_size` for [`InferenceResult::for_body`].
///
/// The arithmetic lives on the type itself, next to the fields, because most of
/// them are private to the `infer` module — and because a total assembled where
/// the fields are declared is the one that a new field breaks loudly.
pub(crate) fn inference_result(result: &InferenceResult<'_>) -> usize {
    result.heap_size()
}

/// The containers a single closure's capture analysis owns.
///
/// `liberated_sig` is an interned handle and owns nothing here. What is left
/// out inside the elements: a `CapturedPlace`'s projection is interned too, and
/// a `CaptureSourceStack` may hold more than its own bytes — both err low, like
/// the rest of the family.
pub(crate) fn closure_data(data: &ClosureData) -> usize {
    let ClosureData { min_captures, fake_reads, liberated_sig: _ } = data;

    slice(&fake_reads[..])
        + index_map_with(min_captures, vec)
        + fake_reads.iter().map(|(_, _, sources)| small_vec(sources)).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use base_db::EditionedFileId;
    use hir_def::{DefWithBodyId, ModuleDefId, nameres::crate_def_map};
    use test_fixture::WithFixture;

    use crate::{infer::InferenceResult, test_db::TestDB};

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
            let elements =
                (0..COUNT).map(captures).collect::<Vec<_>>().join(", ");
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
}
