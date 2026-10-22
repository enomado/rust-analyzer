//! A file text store (`base_db::file_text_store`) on a real `RootDatabase`:
//! which texts leave the heap and how, that every reader still gets the exact
//! text it was given, that a changed original is refused and reported, and that
//! the loaded texts are weighed and evicted like any LRU-capped query.

use std::{collections::HashMap, path::Path};

use ide_db::{
    RootDatabase,
    base_db::{
        AbsPathBuf, FileChange, FileId, FileSet, SourceDatabase, SourceRoot, VfsPath,
        file_text_store::{FileTextStore, StoredText},
        salsa,
    },
};
use rustc_hash::FxHashMap;

/// One file of the fixture: where it lives, what the database is given, and
/// whether its root is a library.
struct Fixture {
    path: AbsPathBuf,
    text: String,
    library: bool,
}

fn abs(path: &Path) -> AbsPathBuf {
    AbsPathBuf::assert_utf8(path.to_path_buf())
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().expect("a file has a parent")).expect("mkdir");
    std::fs::write(path, text).expect("write a fixture file");
}

/// Load `files` the way `load-cargo` does — roots first, then texts, in one
/// change — into a database with `store`, or with no store at all.
fn load(store: Option<FileTextStore>, files: &[Fixture]) -> RootDatabase {
    let mut db = RootDatabase::new(None);
    if let Some(store) = store {
        db.set_file_text_store(store);
    }
    let mut library = FileSet::default();
    let mut local = FileSet::default();
    let mut change = FileChange::default();
    for (index, file) in files.iter().enumerate() {
        let file_id = FileId::from_raw(index as u32);
        let set = if file.library { &mut library } else { &mut local };
        set.insert(file_id, VfsPath::from(file.path.clone()));
        change.change_file(file_id, Some(file.text.clone()));
    }
    change.set_roots(vec![SourceRoot::new_library(library), SourceRoot::new_local(local)]);
    change.apply(&mut db);
    db
}

fn kind(db: &RootDatabase, file_id: FileId) -> &'static str {
    match db.file_text(file_id).stored(db) {
        StoredText::Inline(_) => "inline",
        StoredText::Mapped(_) => "mapped",
        StoredText::OnDisk(_) => "on disk",
    }
}

/// `(heap of the FileText inputs, heap of the loaded_file_text memos)`.
fn heaps(db: &mut RootDatabase) -> (usize, usize) {
    let db: &mut dyn salsa::Database = db;
    let info = db.memory_usage();
    let inputs = info
        .structs
        .iter()
        .find(|ingredient| ingredient.debug_name() == "FileText")
        .and_then(|ingredient| ingredient.heap_size_of_fields())
        .expect("FileText reports its heap");
    let loaded = info
        .queries
        .get("loaded_file_text")
        .map_or(0, |ingredient| ingredient.heap_size_of_fields().expect("loaded_file_text weighs"));
    (inputs, loaded)
}

/// A registry-like directory (read back), a library file there whose size on
/// disk is not the text it was given (copied: reading it back could never pass
/// the check), a library outside it (copied, like build-script output) and a
/// workspace file (copied).
fn fixture(root: &Path) -> (FileTextStore, Vec<Fixture>) {
    let registry = root.join("registry");
    let registry_text = "pub fn from_the_registry() -> u32 { 1 }\n".repeat(40);
    write(&registry.join("dep/src/lib.rs"), &registry_text);
    let unsaved = "pub struct Unsaved;\n";
    write(&registry.join("dep/src/unsaved.rs"), "pub struct SavedLongerThanTheBuffer;\n");
    // Multi-byte UTF-8, so a copy cut at a byte offset would show.
    let out_dir_text = "pub const GENERATED: &str = \"généré — ✓\";\n".to_owned();
    write(&root.join("target/out/generated.rs"), &out_dir_text);
    let local_text = "fn main() { dep::from_the_registry(); }\n".to_owned();
    write(&root.join("ws/src/main.rs"), &local_text);
    let store =
        FileTextStore { read_back_dirs: vec![abs(&registry)], copy_dir: abs(&root.join("copies")) };
    std::fs::create_dir_all(root.join("copies")).expect("mkdir copies");
    let files = vec![
        Fixture { path: abs(&registry.join("dep/src/lib.rs")), text: registry_text, library: true },
        Fixture {
            path: abs(&registry.join("dep/src/unsaved.rs")),
            text: unsaved.to_owned(),
            library: true,
        },
        Fixture {
            path: abs(&root.join("target/out/generated.rs")),
            text: out_dir_text,
            library: true,
        },
        Fixture { path: abs(&root.join("ws/src/main.rs")), text: local_text, library: false },
    ];
    (store, files)
}

#[test]
fn texts_out_of_the_heap_read_back_exactly_what_was_given() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (store, files) = fixture(dir.path());
    let mut db = load(Some(store), &files);

    let kinds: Vec<_> = (0..files.len()).map(|i| kind(&db, FileId::from_raw(i as u32))).collect();
    assert_eq!(kinds, ["on disk", "mapped", "mapped", "mapped"]);
    for (index, file) in files.iter().enumerate() {
        let text = db.file_text(FileId::from_raw(index as u32)).text(&db);
        assert_eq!(&**text, file.text, "{} reads back as given", file.path);
    }

    // The inputs hold only the read-back path; each loaded text is in the heap
    // once, owned by its memo.
    let (inputs, loaded) = heaps(&mut db);
    let arc = |text: &str| (8 + text.len()).next_multiple_of(8);
    assert_eq!(inputs, files[0].path.as_str().len());
    assert_eq!(loaded, files.iter().map(|file| arc(&file.text)).sum::<usize>());

    // An edit of a workspace file is a new copy, read back as the new text,
    // and the reader sees it (the memo depends on the input).
    let edited = "fn main() { /* edited */ }\n";
    let mut change = FileChange::default();
    change.change_file(FileId::from_raw(3), Some(edited.to_owned()));
    change.apply(&mut db);
    assert_eq!(kind(&db, FileId::from_raw(3)), "mapped");
    assert_eq!(&**db.file_text(FileId::from_raw(3)).text(&db), edited);
    assert_eq!(db.file_text_drift(), None);
}

/// Without a store nothing changes: every text is the input's own, and the
/// loading query is never run.
#[test]
fn without_a_store_every_text_stays_in_the_heap() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, files) = fixture(dir.path());
    let mut db = load(None, &files);
    for (index, file) in files.iter().enumerate() {
        let file_id = FileId::from_raw(index as u32);
        assert_eq!(kind(&db, file_id), "inline");
        assert_eq!(&**db.file_text(file_id).text(&db), file.text);
    }
    let (inputs, loaded) = heaps(&mut db);
    assert_eq!(loaded, 0);
    assert!(inputs >= files.iter().map(|file| file.text.len()).sum::<usize>());
}

/// An original changed behind the database's back — same length, other bytes,
/// which only the hash can tell — is refused with the path, and recorded for the
/// embedder; a deleted one likewise.
#[test]
fn a_changed_original_is_refused_and_reported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (store, files) = fixture(dir.path());
    let db = load(Some(store), &files);
    let original = &files[0];
    let tampered: String = original.text.chars().map(|c| if c == '1' { '2' } else { c }).collect();
    assert_eq!(tampered.len(), original.text.len());
    std::fs::write(original.path.as_path(), &tampered).expect("tamper with the original");

    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.file_text(FileId::from_raw(0)).text(&db).len()
    }));
    let payload = refused.expect_err("a changed original must not be read as the old text");
    let message =
        payload.downcast_ref::<String>().cloned().unwrap_or_else(|| "<not a string>".to_owned());
    assert!(message.contains(original.path.as_str()), "the refusal names the file: {message}");
    let drift = db.file_text_drift().expect("the drift is recorded");
    assert!(drift.contains("changed on disk"), "{drift}");

    let dir = tempfile::tempdir().expect("tempdir");
    let (store, files) = fixture(dir.path());
    let db = load(Some(store), &files);
    std::fs::remove_file(files[0].path.as_path()).expect("delete the original");
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        db.file_text(FileId::from_raw(0)).text(&db).len()
    }));
    assert!(refused.is_err(), "a deleted original must not be read");
    let drift = db.file_text_drift().expect("the deletion is recorded");
    assert!(drift.contains("can no longer be read"), "{drift}");
}

/// The loaded texts are an LRU-capped query under its real name, and eviction
/// gives their heap back.
#[test]
fn loaded_texts_are_evicted_by_their_lru() {
    let dir = tempfile::tempdir().expect("tempdir");
    let copies = dir.path().join("copies");
    std::fs::create_dir_all(&copies).expect("mkdir copies");
    let store = FileTextStore { read_back_dirs: vec![], copy_dir: abs(&copies) };
    let files: Vec<_> = (0..20)
        .map(|i| Fixture {
            path: abs(&dir.path().join(format!("ws/src/m{i}.rs"))),
            text: format!("pub fn f{i}() {{}}\n").repeat(10),
            library: false,
        })
        .collect();
    let mut db = load(Some(store), &files);
    let capacities: FxHashMap<Box<str>, u16> =
        [("loaded_file_text".into(), 1)].into_iter().collect();
    assert_eq!(db.update_lru_capacities(&capacities), 0, "the query is tunable by its name");

    for index in 0..files.len() {
        db.file_text(FileId::from_raw(index as u32)).text(&db);
    }
    let arcs: Vec<usize> =
        files.iter().map(|file| (8 + file.text.len()).next_multiple_of(8)).collect();
    let (_, loaded) = heaps(&mut db);
    assert_eq!(loaded, arcs.iter().sum::<usize>(), "every text read once is loaded");

    salsa::Database::trigger_lru_eviction(&mut db);
    let (_, evicted) = heaps(&mut db);
    let largest = *arcs.iter().max().expect("files");
    assert!(evicted <= largest, "a capacity of one keeps at most one text: {evicted} bytes");
    // Still readable after eviction: loaded again from the copy.
    let texts: HashMap<_, _> = (0..files.len())
        .map(|i| (i, db.file_text(FileId::from_raw(i as u32)).text(&db).to_string()))
        .collect();
    assert!(texts.iter().all(|(i, text)| *text == files[*i].text));
}
