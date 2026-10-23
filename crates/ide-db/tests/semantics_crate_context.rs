//! The same source can be compiled independently by two Cargo targets.
use hir::{Crate, Semantics};
use ide_db::RootDatabase;
use syntax::{AstNode, ast};
use test_fixture::WithFixture;

#[test]
fn shared_file_resolves_in_the_selected_crate() {
    let (db, files) = RootDatabase::with_many_files(r#"
//- /lib.rs crate:lib deps:dep
pub mod shared;
pub fn leaf() {}
//- /main.rs crate:bin deps:dep
mod shared;
fn leaf() {}
//- /shared.rs
pub fn call() { crate::leaf(); dep::leaf(); }
//- /dep.rs crate:dep
pub fn leaf() {}
"#);
    for krate in Crate::all(&db) {
        let name = krate.display_name(&db).unwrap();
        if name.canonical_name().as_str() == "dep" { continue; }
        let sema = Semantics::new_for_crate(&db, krate);
        let source = sema.parse(files[2]);
        let function = source.syntax().descendants().find_map(ast::Fn::cast).unwrap();
        assert_eq!(sema.to_def(&function).unwrap().module(&db).krate(&db), krate);
        for call in source.syntax().descendants().filter_map(ast::CallExpr::cast) {
            let callee = call.expr().unwrap();
            let callable = sema.resolve_expr_as_callable(&callee).unwrap();
            let hir::CallableKind::Function(target) = callable.kind() else { panic!("function call"); };
            let target_crate = target.module(&db).krate(&db);
            if callee.syntax().to_string().starts_with("crate::") {
                assert_eq!(target_crate, krate);
            } else {
                assert_eq!(target_crate.display_name(&db).unwrap().canonical_name().as_str(), "dep");
                let target_source = sema.source(target).unwrap();
                assert_eq!(sema.to_def(&target_source.value), Some(target));
            }
        }
    }
}
