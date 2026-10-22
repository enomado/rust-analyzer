use base_db::{
    CrateDisplayName, CrateGraphBuilder, CrateName, CrateOrigin, CrateWorkspaceData,
    DependencyBuilder, Env, SourceDatabase, TargetKind, all_crates,
};
use expect_test::{Expect, expect};
use intern::Symbol;
use span::Edition;
use test_fixture::WithFixture;
use triomphe::Arc;

use crate::{
    file_item_tree,
    nameres::{crate_def_map, tests::TestDB},
};

fn check_def_map_is_not_recomputed(
    #[rust_analyzer::rust_fixture] ra_fixture_initial: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_change: &str,
    expecta: Expect,
    expectb: Expect,
) {
    let (mut db, pos) = TestDB::with_position(ra_fixture_initial);
    let krate = db.fetch_test_crate();
    execute_assert_events(
        &db,
        || {
            crate_def_map(&db, krate);
        },
        &[],
        expecta,
    );
    db.set_file_text(pos.file_id.file_id(&db), ra_fixture_change);

    execute_assert_events(
        &db,
        || {
            crate_def_map(&db, krate);
        },
        &[("crate_local_def_map", 0)],
        expectb,
    );
}

#[test]
fn crate_metadata_changes_should_not_invalidate_unrelated_def_maps() {
    let (mut db, files) = TestDB::with_many_files(
        r#"
//- /a.rs crate:a
pub fn foo() {}

//- /b.rs crate:b
pub struct Bar;

//- /c.rs crate:c deps:b
pub const BAZ: u32 = 0;
    "#,
    );

    for &krate in all_crates(&db).iter() {
        crate_def_map(&db, krate);
    }

    let all_crates_before = all_crates(&db);

    {
        // Add dependencies: c -> b, b -> a.
        let mut new_crate_graph = CrateGraphBuilder::default();

        let mut add_crate = |crate_name, root_file_idx: usize| {
            new_crate_graph.add_crate_root(
                files[root_file_idx].file_id(&db),
                Edition::CURRENT,
                Some(CrateDisplayName::from_canonical_name(crate_name)),
                None,
                Default::default(),
                None,
                Env::default(),
                CrateOrigin::Local { repo: None, name: Some(Symbol::intern(crate_name)) },
                Vec::new(),
                TargetKind::Lib { is_proc_macro: false },
                Arc::new(
                    // FIXME: This is less than ideal
                    TryFrom::try_from(
                        &*std::env::current_dir().unwrap().as_path().to_string_lossy(),
                    )
                    .unwrap(),
                ),
                Arc::new(CrateWorkspaceData { target: Err("".into()), toolchain: None }),
            )
        };
        let a = add_crate("a", 0);
        let b = add_crate("b", 1);
        let c = add_crate("c", 2);
        new_crate_graph
            .add_dep(c, DependencyBuilder::new(CrateName::new("b").unwrap(), b))
            .unwrap();
        new_crate_graph
            .add_dep(b, DependencyBuilder::new(CrateName::new("a").unwrap(), a))
            .unwrap();
        new_crate_graph.set_in_db(&mut db);
    }

    let all_crates_after = all_crates(&db);
    assert!(
        std::sync::Arc::ptr_eq(&all_crates_before, &all_crates_after),
        "the all_crates list should not have been invalidated"
    );
    execute_assert_events(
        &db,
        || {
            for &krate in all_crates(&db).iter() {
                crate_def_map(&db, krate);
            }
        },
        // `c` gets invalidated as its dependency `b` changed
        // `b` gets invalidated due to its new dependency edge to `a`
        &[("crate_local_def_map", 2)],
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "crate_local_def_map",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_a_function_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- /lib.rs
mod foo;$0

use crate::foo::bar::Baz;

enum E { A, B }
use E::*;

fn foo() -> i32 {
    1 + 1
}

#[cfg(never)]
fn no() {}
//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
pub struct Baz;
",
        r"
mod foo;

use crate::foo::bar::Baz;

enum E { A, B }
use E::*;

fn foo() -> i32 { 92 }

#[cfg(never)]
fn no() {}
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "EnumVariants::of_",
            ]
        "#]],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
                "EnumVariants::of_",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_a_macro_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn f() {
            $ident + $ident;
        };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
m!(X);

pub struct S {}
",
        r"
m!(Y);

pub struct S {}
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "AstId < ast :: Macro >::decl_macro_expander_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "MacroId::definition_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
            ]
        "#]],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
                "MacroCallId::macro_arg_",
                "MacroCallId::parse_macro_expansion_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_an_attribute_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: identity
//- /lib.rs
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
#[proc_macros::identity]
fn f() {}
",
        r"
#[proc_macros::identity]
fn f() { foo }
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "crate_local_def_map",
                "ProcMacros::get_for_crate_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "MacroId::definition_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::expand_proc_macro_",
                "MacroCallId::macro_arg_",
                "proc_macro_span",
            ]
        "#]],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
                "MacroCallId::macro_arg_",
                "MacroCallId::expand_proc_macro_",
                "MacroCallId::parse_macro_expansion_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
            ]
        "#]],
    );
}

// Would be nice if this was the case, but as attribute inputs are stored in the item tree, this is
// not currently the case.
// #[test]
// fn typing_inside_an_attribute_arg_should_not_invalidate_def_map() {
//     check_def_map_is_not_recomputed(
//         r"
// //- proc_macros: identity
// //- /lib.rs
// mod foo;

// //- /foo/mod.rs
// pub mod bar;

// //- /foo/bar.rs
// $0
// #[proc_macros::identity]
// fn f() {}
// ",
//         r"
// #[proc_macros::identity(foo)]
// fn f() {}
// ",
//     );
// }

#[test]
fn typing_inside_macro_heavy_file_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: identity, derive_identity
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn fm() {
            $ident + $ident;
        };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
fn f() {}

m!(X);
macro_rules! m2 {
    ($ident:ident) => {
        fn f2() {
            $ident + $ident;
        };
    }
}
m2!(X);

#[proc_macros::identity]
#[derive(proc_macros::DeriveIdentity)]
pub struct S {}
",
        r"
fn f() {0}

m!(X);
macro_rules! m2 {
    ($ident:ident) => {
        fn f2() {
            $ident + $ident;
        };
    }
}
m2!(X);

#[proc_macros::identity]
#[derive(proc_macros::DeriveIdentity)]
pub struct S {}
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "crate_local_def_map",
                "ProcMacros::get_for_crate_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "AstId < ast :: Macro >::decl_macro_expander_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "MacroId::definition_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
                "AstId < ast :: Macro >::decl_macro_expander_",
                "MacroId::definition_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
                "MacroId::definition_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::expand_proc_macro_",
                "MacroCallId::macro_arg_",
                "proc_macro_span",
            ]
        "#]],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
                "MacroCallId::macro_arg_",
                "AstId < ast :: Macro >::decl_macro_expander_",
                "MacroCallId::macro_arg_",
                "MacroCallId::macro_arg_",
            ]
        "#]],
    );
}

// Would be nice if this was the case, but as attribute inputs are stored in the item tree, this is
// not currently the case.
// #[test]
// fn typing_inside_a_derive_should_not_invalidate_def_map() {
//     check_def_map_is_not_recomputed(
//         r"
// //- proc_macros: derive_identity
// //- minicore:derive
// //- /lib.rs
// mod foo;

// //- /foo/mod.rs
// pub mod bar;

// //- /foo/bar.rs
// $0
// #[derive(proc_macros::DeriveIdentity)]
// #[allow()]
// struct S;
// ",
//         r"
// #[derive(proc_macros::DeriveIdentity)]
// #[allow(dead_code)]
// struct S;
// ",
//     );
// }

#[test]
fn typing_inside_a_function_should_not_invalidate_item_expansions() {
    let (mut db, pos) = TestDB::with_position(
        r#"
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn $ident() { };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
m!(X);
fn quux() { 1$0 }
m!(Y);
m!(Z);
"#,
    );
    let krate = db.test_crate();
    execute_assert_events(
        &db,
        || {
            let crate_def_map = crate_def_map(&db, krate);
            let module_data = &crate_def_map
                [crate_def_map.modules_for_file(&db, pos.file_id.file_id(&db)).next().unwrap()];
            assert_eq!(module_data.scope.resolutions().count(), 4);
        },
        &[("file_item_tree_query", 6), ("parse_macro_expansion", 3)],
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "AstId < ast :: Macro >::decl_macro_expander_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
                "MacroId::definition_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
            ]
        "#]],
    );

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let crate_def_map = crate_def_map(&db, krate);
            let module_data = &crate_def_map
                [crate_def_map.modules_for_file(&db, pos.file_id.file_id(&db)).next().unwrap()];
            assert_eq!(module_data.scope.resolutions().count(), 4);
        },
        &[("file_item_tree_query", 1), ("parse_macro_expansion", 0)],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
                "MacroCallId::macro_arg_",
                "MacroCallId::macro_arg_",
                "MacroCallId::macro_arg_",
            ]
        "#]],
    );
}

/// What the LRU cap on `macro_arg` (bur) costs, priced on the fixture above:
/// an argument that was evicted cannot backdate, because salsa compares the
/// recomputed value with the old one and an evicted memo has none. So typing
/// in the calling file re-expands that call where the test above re-expands
/// nothing — and the cascade still stops at the expansion, which backdates, so
/// the crate def map is not recomputed either way.
#[test]
fn typing_beside_an_evicted_macro_arg_re_expands_only_that_call() {
    let (mut db, pos) = TestDB::with_position(
        r#"
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn $ident() { };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
m!(X);
fn quux() { 1$0 }
m!(Y);
m!(Z);
"#,
    );
    let krate = db.test_crate();
    let resolutions = |db: &TestDB| {
        let crate_def_map = crate_def_map(db, krate);
        let module_data = &crate_def_map
            [crate_def_map.modules_for_file(db, pos.file_id.file_id(db)).next().unwrap()];
        module_data.scope.resolutions().count()
    };
    assert_eq!(resolutions(&db), 4);

    // Keep one argument of the three; the other two lose their values.
    assert_eq!(salsa::Database::set_lru_capacity_by_name(&mut db, "MacroCallId::macro_arg_", 1), 1);
    salsa::Database::trigger_lru_eviction(&mut db);

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || assert_eq!(resolutions(&db), 4),
        &[
            ("MacroCallId::macro_arg_", 3),
            // The two evicted calls, and only those: the kept one backdates.
            ("parse_macro_expansion", 2),
            ("crate_local_def_map", 0),
        ],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
                "MacroCallId::parse_macro_expansion_",
                "MacroCallId::macro_arg_",
                "MacroCallId::macro_arg_",
            ]
        "#]],
    );
}

/// What the LRU cap on `Body::of` (bur; upstream keeps every body) costs. On
/// a large workspace the producer, `Body::with_source_map` (`lru = 512`), is
/// usually evicted, and an edit anywhere in the file re-lowers every body there
/// that is asked for again: an evicted producer cannot backdate. Uncapped,
/// `Body::of` is where that stops — it compares the fresh body by value — so
/// what reads the body stays put. An evicted `Body::of` has no old value
/// either, and its readers re-run too: here the scopes of the two evicted
/// bodies. The readers that walk every body of a module or a crate (symbol
/// index, impl indexes) read `Body::block_scopes` instead, which is not capped
/// and backdates — they must not re-run.
#[test]
fn typing_beside_evicted_bodies_re_runs_their_readers() {
    #[salsa::tracked]
    fn block_count(db: &dyn base_db::SourceDatabase, def: crate::DefWithBodyId) -> usize {
        crate::expr_store::Body::block_scopes(db, def).len()
    }

    let (mut db, pos) = TestDB::with_position(
        r#"
//- /lib.rs
fn a() { struct S; let x = 1; }
fn b() { struct S; let x = 1; }
fn c() { struct S; let x = 1; }
fn quux() { 1$0 }
"#,
    );
    let krate = db.test_crate();
    let bodies: Vec<crate::DefWithBodyId> = {
        let def_map = crate_def_map(&db, krate);
        def_map
            .modules()
            .flat_map(|(_, module)| module.scope.declarations())
            .filter_map(|def| match def {
                crate::ModuleDefId::FunctionId(it) => Some(it.into()),
                _ => None,
            })
            .collect()
    };
    // `quux` is not asked for: its body really changes.
    let bodies = &bodies[..3];
    let read = |db: &TestDB| {
        bodies
            .iter()
            .map(|&def| {
                let scopes = crate::expr_store::scope::ExprScopes::body_expr_scopes(db, def)
                    .scope_by_expr()
                    .iter()
                    .count();
                (scopes, block_count(db, def))
            })
            .fold((0, 0), |(s, b), (scopes, blocks)| (s + scopes, b + blocks))
    };
    let before = read(&db);
    assert!(before.0 > 0, "the scopes of three bodies cannot be empty");
    assert_eq!(before.1, 3, "each body has one block with an item");

    // Keep one producer and one body of the three; the others lose their values.
    for query in ["Body::with_source_map_", "Body::of_"] {
        assert_eq!(salsa::Database::set_lru_capacity_by_name(&mut db, query, 1), 1, "{query}");
    }
    salsa::Database::trigger_lru_eviction(&mut db);

    // An edit after the three bodies: their own text does not move.
    let file_id = pos.file_id.file_id(&db);
    let new_text = db.file_text(file_id).text(&db).replace("{ 1 }", "{ 92 }");
    db.set_file_text(file_id, &new_text);

    execute_assert_events(
        &db,
        || assert_eq!(read(&db), before),
        &[
            ("Body::with_source_map_", 3),
            // The two evicted producers, and only those: the kept one backdates.
            ("Body::of_", 2),
            // The price of the cap: an uncapped `Body::of` backdates here, and
            // no scopes are rebuilt.
            ("body_expr_scopes", 2),
            // ...and where it stops for the readers of whole modules and crates.
            ("Body::block_scopes_", 2),
            ("block_count", 0),
        ],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "ExprScopes::body_expr_scopes_",
                "Body::of_",
                "Body::with_source_map_",
                "file_item_tree_query",
                "real_span_map",
                "AttrFlags::query_",
                "block_item_tree_query",
                "Body::block_scopes_",
                "ExprScopes::body_expr_scopes_",
                "Body::of_",
                "Body::with_source_map_",
                "AttrFlags::query_",
                "block_item_tree_query",
                "Body::block_scopes_",
                "Body::with_source_map_",
                "AttrFlags::query_",
                "block_item_tree_query",
            ]
        "#]],
    );
}

/// What an evicted signature source map costs (bur caps every
/// `XSignature::with_source_map`; upstream caps none). An edit in the file
/// re-lowers every signature there that is asked for again — evicted or not,
/// since the producer reads the file's syntax. The difference is in `of`: a
/// kept producer backdates and `of` is not even re-run, an evicted one cannot
/// (there is no old value to compare with) and its `of` re-runs — but `of`
/// compares the fresh signature by value and backdates in turn, so a query
/// that reads only `of` stays put.
#[test]
fn typing_beside_evicted_signature_maps_stops_at_of() {
    #[salsa::tracked]
    fn where_clause_count(db: &dyn base_db::SourceDatabase, id: crate::ImplId) -> usize {
        crate::signatures::ImplSignature::of(db, id).generic_params.where_predicates.len()
    }

    let (mut db, pos) = TestDB::with_position(
        r#"
//- /lib.rs
trait Tr {}
struct A<T>(T);
struct B<T>(T);
struct C<T>(T);
impl<T> Tr for A<T> where T: Copy {}
impl<T> Tr for B<T> where T: Copy {}
impl<T> Tr for C<T> where T: Copy {}
fn quux() { 1$0 }
"#,
    );
    let krate = db.test_crate();
    let impls = || -> Vec<crate::ImplId> {
        let def_map = crate_def_map(&db, krate);
        def_map.modules().flat_map(|(_, module)| module.scope.impls()).collect()
    };
    let impls = impls();
    assert_eq!(impls.len(), 3);
    let counts = |db: &TestDB| impls.iter().map(|&id| where_clause_count(db, id)).sum::<usize>();
    assert_eq!(counts(&db), 3);

    // Keep one source map of the three; the other two lose their values.
    assert_eq!(
        salsa::Database::set_lru_capacity_by_name(&mut db, "ImplSignature::with_source_map_", 1),
        1
    );
    salsa::Database::trigger_lru_eviction(&mut db);

    // An edit AFTER the impls, and nothing else: the source maps hold absolute
    // text ranges, so anything that shifts the impls (a leading newline, say)
    // changes every map and nothing could backdate in the first place.
    let file_id = pos.file_id.file_id(&db);
    let new_text = db.file_text(file_id).text(&db).replace("{ 1 }", "{ 92 }");
    db.set_file_text(file_id, &new_text);

    execute_assert_events(
        &db,
        || assert_eq!(counts(&db), 3),
        &[
            ("ImplSignature::with_source_map_", 3),
            // The two evicted maps, and only those: the kept one backdates.
            ("ImplSignature::of_", 2),
            ("where_clause_count", 0),
        ],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "ImplSignature::of_",
                "ImplSignature::with_source_map_",
                "file_item_tree_query",
                "real_span_map",
                "ImplSignature::of_",
                "ImplSignature::with_source_map_",
                "ImplSignature::with_source_map_",
            ]
        "#]],
    );
}

#[test]
fn item_tree_prevents_reparsing() {
    let (mut db, pos) = TestDB::with_position(
        r#"
pub struct S;
pub union U {}
pub enum E {
    Variant,
}
pub fn f(_: S) { $0 }
pub trait Tr {}
impl Tr for () {}
pub const C: u8 = 0;
pub static ST: u8 = 0;
pub type Ty = ();
"#,
    );

    execute_assert_events(
        &db,
        || {
            file_item_tree(&db, pos.file_id.into(), db.test_crate());
        },
        &[("file_item_tree_query", 1), ("parse", 1)],
        expect![[r#"
            [
                "file_item_tree_query",
                "HirFileId::ast_id_map_",
                "EditionedFileId::parse_",
                "real_span_map",
            ]
        "#]],
    );

    let file_id = pos.file_id.file_id(&db);
    let file_text = db.file_text(file_id).text(&db);
    db.set_file_text(file_id, &format!("{file_text}\n"));

    execute_assert_events(
        &db,
        || {
            file_item_tree(&db, pos.file_id.into(), db.test_crate());
        },
        &[("file_item_tree_query", 1), ("parse", 1)],
        expect![[r#"
            [
                "EditionedFileId::parse_",
                "HirFileId::ast_id_map_",
                "file_item_tree_query",
                "real_span_map",
            ]
        "#]],
    );
}

fn execute_assert_events(
    db: &TestDB,
    f: impl FnOnce(),
    required: &[(&str, usize)],
    expect: Expect,
) {
    let events = db.log_executed(f);
    for (event, count) in required {
        let n = events.iter().filter(|it| it.contains(event)).count();
        assert_eq!(n, *count, "Expected {event} to be executed {count} times, but only got {n}");
    }
    expect.assert_debug_eq(&events);
}
