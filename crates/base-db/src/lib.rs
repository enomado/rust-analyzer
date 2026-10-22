//! This crate defines the basic database trait for interacting with source code using [`salsa`].
//!
//! The concrete implementation DB is defined by ide.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

pub use salsa;
use span::TextSize;

mod change;
mod editioned_file_id;
pub mod file_text_store;
mod input;
pub mod target;

use std::{
    cell::RefCell,
    hash::BuildHasherDefault,
    panic,
    str::FromStr as _,
    sync::{Once, atomic::AtomicUsize},
};

pub use crate::{
    change::FileChange,
    editioned_file_id::EditionedFileId,
    input::{
        BuiltCrateData, BuiltDependency, Crate, CrateBuilder, CrateBuilderId, CrateDataBuilder,
        CrateDisplayName, CrateGraphBuilder, CrateName, CrateOrigin, CratesIdMap, CratesMap,
        DependencyBuilder, Env, ExtraCrateData, LangCrateOrigin, ProcMacroLoadingError,
        ProcMacroPaths, ReleaseChannel, SourceRoot, SourceRootId, TargetKind, UniqueCrateData,
    },
};
use dashmap::{DashMap, mapref::entry::Entry};
use rustc_hash::{FxHashSet, FxHasher};
use salsa::{Durability, Setter};
pub use semver::{BuildMetadata, Prerelease, Version, VersionReq};
use triomphe::Arc;
pub use vfs::{AbsPathBuf, AnchoredPath, AnchoredPathBuf, FileId, VfsPath, file_set::FileSet};

pub type FxIndexSet<T> = indexmap::IndexSet<T, rustc_hash::FxBuildHasher>;
pub type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[macro_export]
macro_rules! impl_intern_key {
    ($id:ident, $loc:ident) => {
        #[salsa::interned(unsafe(no_lifetime), revisions = usize::MAX)]
        #[derive(PartialOrd, Ord)]
        pub struct $id {
            #[returns(ref)]
            pub loc: $loc,
        }

        // If we derive this salsa prints the values recursively, and this causes us to blow.
        impl ::std::fmt::Debug for $id {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.debug_tuple(stringify!($id))
                    .field(&format_args!("{:04x}", self.0.index()))
                    .finish()
            }
        }
    };
}

pub const DEFAULT_FILE_TEXT_LRU_CAP: u16 = 16;
pub const DEFAULT_PARSE_LRU_CAP: u16 = 128;
pub const DEFAULT_BORROWCK_LRU_CAP: u16 = 2024;

#[derive(Debug, Default)]
pub struct Files {
    files: Arc<DashMap<vfs::FileId, FileText, BuildHasherDefault<FxHasher>>>,
    source_roots: Arc<DashMap<SourceRootId, SourceRootInput, BuildHasherDefault<FxHasher>>>,
    file_source_roots: Arc<DashMap<vfs::FileId, FileSourceRootInput, BuildHasherDefault<FxHasher>>>,
    /// `None`: every text in the heap, as upstream keeps them.
    text_store: Option<file_text_store::TextStore>,
}

impl Files {
    /// Store file texts set from now on as `store` says; see
    /// [`file_text_store`]. Texts already set stay where they are, so an
    /// embedder calls this before loading anything.
    pub fn set_text_store(&mut self, store: file_text_store::FileTextStore) {
        self.text_store = Some(file_text_store::TextStore::new(store));
    }

    /// The first library file whose text was read back and found changed on
    /// disk since it was set, if any. Once there is one, the database no longer
    /// describes the files on disk and the workspace has to be reloaded.
    pub fn text_drift(&self) -> Option<String> {
        self.text_store.as_ref()?.drift.first()
    }

    pub fn file_text(&self, file_id: vfs::FileId) -> FileText {
        match self.files.get(&file_id) {
            Some(text) => *text,
            None => {
                panic!("Unable to fetch file text for `vfs::FileId`: {file_id:?}; this is a bug")
            }
        }
    }

    pub fn set_file_text(&self, db: &mut dyn SourceDatabase, file_id: vfs::FileId, text: &str) {
        let stored = self.stored_text(db, file_id, text);
        match self.files.entry(file_id) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().set_stored(db).to(stored);
            }
            Entry::Vacant(vacant) => {
                let text = FileText::new(db, stored, file_id);
                vacant.insert(text);
            }
        };
    }

    pub fn set_file_text_with_durability(
        &self,
        db: &mut dyn SourceDatabase,
        file_id: vfs::FileId,
        text: &str,
        durability: Durability,
    ) {
        let stored = self.stored_text(db, file_id, text);
        match self.files.entry(file_id) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().set_stored(db).with_durability(durability).to(stored);
            }
            Entry::Vacant(vacant) => {
                let text = FileText::builder(stored, file_id).durability(durability).new(db);
                vacant.insert(text);
            }
        };
    }

    /// Where `text` goes. The file's source root says whether it is a library
    /// and what its path is; `FileChange::apply` sets the roots before the
    /// texts, so a loaded workspace always has them. A file without a recorded
    /// root (a test fixture setting texts directly) is stored as a workspace
    /// file without a path.
    fn stored_text(
        &self,
        db: &dyn SourceDatabase,
        file_id: vfs::FileId,
        text: &str,
    ) -> file_text_store::StoredText {
        let Some(store) = &self.text_store else {
            return file_text_store::StoredText::Inline(Arc::from(text));
        };
        let root_id = self.file_source_roots.get(&file_id).map(|root| root.source_root_id(db));
        let root = root_id
            .and_then(|id| self.source_roots.get(&id).map(|root| *root))
            .map(|root| root.source_root(db));
        let path = root.as_ref().and_then(|root| root.path_for_file(&file_id)?.as_path());
        let is_library = root.as_ref().is_some_and(|root| root.is_library);
        store.store(path, is_library, text)
    }

    /// Source root of the file.
    pub fn source_root(&self, source_root_id: SourceRootId) -> SourceRootInput {
        let source_root = match self.source_roots.get(&source_root_id) {
            Some(source_root) => source_root,
            None => panic!(
                "Unable to fetch `SourceRootInput` with `SourceRootId` ({source_root_id:?}); this is a bug"
            ),
        };

        *source_root
    }

    pub fn set_source_root_with_durability(
        &self,
        db: &mut dyn SourceDatabase,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    ) {
        match self.source_roots.entry(source_root_id) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().set_source_root(db).with_durability(durability).to(source_root);
            }
            Entry::Vacant(vacant) => {
                let source_root =
                    SourceRootInput::builder(source_root).durability(durability).new(db);
                vacant.insert(source_root);
            }
        };
    }

    pub fn file_source_root(
        &self,
        db: &dyn SourceDatabase,
        id: vfs::FileId,
    ) -> FileSourceRootInput {
        let file_source_root = match self.file_source_roots.get(&id) {
            Some(file_source_root) => file_source_root,
            None => panic!(
                "Unable to get `FileSourceRootInput` with `vfs::FileId` ({id:?}, path: {}); this is a bug",
                self.path_for_file(db, id)
                    .map_or_else(|| "<unknown>".to_owned(), |path| path.to_string()),
            ),
        };
        *file_source_root
    }

    fn path_for_file(&self, db: &dyn SourceDatabase, id: vfs::FileId) -> Option<vfs::VfsPath> {
        for source_root in &*self.source_roots {
            let source_root = *source_root.value();
            if let Some(path) = source_root.source_root(db).path_for_file(&id) {
                return Some(path.clone());
            }
        }
        None
    }

    pub fn set_file_source_root_with_durability(
        &self,
        db: &mut dyn SourceDatabase,
        id: vfs::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    ) {
        match self.file_source_roots.entry(id) {
            Entry::Occupied(mut occupied) => {
                occupied
                    .get_mut()
                    .set_source_root_id(db)
                    .with_durability(durability)
                    .to(source_root_id);
            }
            Entry::Vacant(vacant) => {
                let file_source_root =
                    FileSourceRootInput::builder(source_root_id).durability(durability).new(db);
                vacant.insert(file_source_root);
            }
        };
    }
}

/// The set of roots for crates.io libraries.
/// Files in libraries are assumed to never change.
#[salsa::input(singleton, debug)]
pub struct LibraryRoots {
    #[returns(ref)]
    pub roots: FxHashSet<SourceRootId>,
}

/// The set of "local" (that is, from the current workspace) roots.
/// Files in local roots are assumed to change frequently.
#[salsa::input(singleton, debug)]
pub struct LocalRoots {
    #[returns(ref)]
    pub roots: FxHashSet<SourceRootId>,
}

#[salsa::input(debug, heap_size = file_text_heap_size)]
pub struct FileText {
    /// The text, or a handle to it; read it with [`FileText::text`].
    #[returns(ref)]
    pub stored: file_text_store::StoredText,
    pub file_id: vfs::FileId,
}

impl FileText {
    /// The text of the file.
    ///
    /// An inline text is the input's own `Arc`; any other is loaded by
    /// [`loaded_file_text`], whose LRU decides how many stay in the heap. Either
    /// way the reader depends on the input, so a changed text invalidates it.
    pub fn text<'db, Db: ?Sized + salsa::Database>(self, db: &'db Db) -> &'db Arc<str> {
        match self.stored(db) {
            file_text_store::StoredText::Inline(text) => text,
            file_text_store::StoredText::Mapped(_) | file_text_store::StoredText::OnDisk(_) => {
                loaded_file_text(db.as_dyn_database(), self)
            }
        }
    }
}

/// The text of a file stored out of the heap, back in it until evicted.
///
/// The capacity is the old `FileTextQuery`'s, from when rust-analyzer kept
/// texts compressed: a text is wanted again only to build what is derived from
/// it, and most of that (`parse`, `ast_id_map`, item trees) is cached itself.
#[salsa::tracked(returns(ref), lru = 16, heap_size = loaded_file_text_heap_size)]
pub fn loaded_file_text(db: &dyn salsa::Database, file: FileText) -> Arc<str> {
    file.stored(db).load()
}

/// `heap_size` for [`FileText`]: what the input itself holds — the whole text
/// when it is inline, a path or nothing when it is not.
///
/// The input is an inline text's one owner: every setter in `Files` builds a
/// fresh `Arc::from(text)`, and readers borrow it (`returns(ref)`), so a clone a
/// query makes is charged here once. Checked against the allocator by
/// `tests/heap_size_oracle.rs`.
pub fn file_text_heap_size(
    (stored, _file_id): &(file_text_store::StoredText, vfs::FileId),
) -> usize {
    stored.heap_size()
}

/// `heap_size` for [`loaded_file_text`]: the loaded `Arc<str>`, which the memo
/// owns — it was built by the load, not cloned from the input.
pub fn loaded_file_text_heap_size(text: &Arc<str>) -> usize {
    file_text_store::arc_str_heap_size(text)
}

#[salsa::input(debug)]
pub struct FileSourceRootInput {
    #[returns(copy)]
    pub source_root_id: SourceRootId,
}

#[salsa::input(debug)]
pub struct SourceRootInput {
    #[returns(clone)]
    pub source_root: Arc<SourceRoot>,
}

#[salsa::db]
pub trait SourceDatabase: salsa::Database + std::fmt::Debug {
    /// Text of the file.
    fn file_text(&self, file_id: vfs::FileId) -> FileText;

    fn set_file_text(&mut self, file_id: vfs::FileId, text: &str);

    fn set_file_text_with_durability(
        &mut self,
        file_id: vfs::FileId,
        text: &str,
        durability: Durability,
    );

    /// Contents of the source root.
    fn source_root(&self, id: SourceRootId) -> SourceRootInput;

    fn file_source_root(&self, id: vfs::FileId) -> FileSourceRootInput;

    fn set_file_source_root_with_durability(
        &mut self,
        id: vfs::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    );

    /// Source root of the file.
    fn set_source_root_with_durability(
        &mut self,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    );

    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        // FIXME: this *somehow* should be platform agnostic...
        let source_root = self.file_source_root(path.anchor);
        let source_root = self.source_root(source_root.source_root_id(self));
        source_root.source_root(self).resolve_path(path)
    }

    #[doc(hidden)]
    fn crates_map(&self) -> Arc<CratesMap>;

    fn nonce_and_revision(&self) -> (Nonce, salsa::Revision);

    fn line_column(&self, file: FileId, offset: TextSize) -> Result<(u32, u32), ()>;
}

static NEXT_NONCE: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Nonce(usize);

impl Default for Nonce {
    #[inline]
    fn default() -> Self {
        Nonce::new()
    }
}

impl Nonce {
    #[inline]
    pub const fn invalid() -> Nonce {
        Nonce(usize::MAX)
    }

    #[inline]
    pub fn new() -> Nonce {
        Nonce(NEXT_NONCE.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

/// Crate related data shared by the whole workspace.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CrateWorkspaceData {
    pub target: Result<target::TargetData, target::TargetLoadError>,
    /// Toolchain version used to compile the crate.
    pub toolchain: Option<Version>,
}

impl CrateWorkspaceData {
    pub fn is_atleast_187(&self) -> bool {
        const VERSION_187: Version = Version {
            major: 1,
            minor: 87,
            patch: 0,
            pre: Prerelease::EMPTY,
            build: BuildMetadata::EMPTY,
        };
        self.toolchain.as_ref().map_or(false, |v| *v >= VERSION_187)
    }
}

pub fn toolchain_channel(db: &dyn salsa::Database, krate: Crate) -> Option<ReleaseChannel> {
    krate.workspace_data(db).toolchain.as_ref().and_then(|v| ReleaseChannel::from_str(&v.pre).ok())
}

#[salsa::input(singleton, debug)]
struct AllCrates {
    #[returns(clone)]
    crates: std::sync::Arc<[Crate]>,
}

pub fn set_all_crates_with_durability(
    db: &mut dyn salsa::Database,
    crates: impl IntoIterator<Item = Crate>,
    durability: Durability,
) {
    AllCrates::try_get(db)
        .unwrap_or_else(|| AllCrates::new(db, std::sync::Arc::default()))
        .set_crates(db)
        .with_durability(durability)
        .to(crates.into_iter().collect());
}

/// Returns the crates in topological order.
///
/// **Warning**: do not use this query in `hir-*` crates! It kills incrementality across crate metadata modifications.
pub fn all_crates(db: &dyn salsa::Database) -> std::sync::Arc<[Crate]> {
    AllCrates::try_get(db).map_or(std::sync::Arc::default(), |all_crates| all_crates.crates(db))
}

// FIXME: VFS rewrite should allow us to get rid of this wrapper
#[doc(hidden)]
#[salsa::interned]
pub struct InternedSourceRootId {
    #[returns(copy)]
    pub id: SourceRootId,
}

/// Crates whose root file is in `id`.
pub fn source_root_crates(db: &dyn SourceDatabase, id: SourceRootId) -> &[Crate] {
    #[salsa::tracked(returns(deref))]
    pub fn source_root_crates<'db>(
        db: &'db dyn SourceDatabase,
        id: InternedSourceRootId<'db>,
    ) -> Box<[Crate]> {
        let crates = AllCrates::get(db).crates(db);
        let id = id.id(db);
        crates
            .iter()
            .copied()
            .filter(|&krate| {
                let root_file = krate.data(db).root_file_id;
                db.file_source_root(root_file).source_root_id(db) == id
            })
            .collect()
    }
    source_root_crates(db, InternedSourceRootId::new(db, id))
}

pub fn relevant_crates(db: &dyn SourceDatabase, file_id: FileId) -> &[Crate] {
    let _p = tracing::info_span!("relevant_crates").entered();

    let source_root = db.file_source_root(file_id);
    source_root_crates(db, source_root.source_root_id(db))
}

#[must_use]
#[non_exhaustive]
pub struct DbPanicContext;

impl Drop for DbPanicContext {
    fn drop(&mut self) {
        Self::with_ctx(|ctx| assert!(ctx.pop().is_some()));
    }
}

impl DbPanicContext {
    pub fn enter(frame: String) -> DbPanicContext {
        #[expect(clippy::print_stderr, reason = "already panicking anyway")]
        fn set_hook() {
            let default_hook = panic::take_hook();
            panic::set_hook(Box::new(move |panic_info| {
                default_hook(panic_info);
                if let Some(backtrace) = salsa::Backtrace::capture() {
                    eprintln!("{backtrace:#}");
                }
                DbPanicContext::with_ctx(|ctx| {
                    if !ctx.is_empty() {
                        eprintln!("additional context:");
                        for (idx, frame) in ctx.iter().enumerate() {
                            eprintln!("{idx:>4}: {frame}\n");
                        }
                    }
                });
            }));
        }

        static SET_HOOK: Once = Once::new();
        SET_HOOK.call_once(set_hook);

        Self::with_ctx(|ctx| ctx.push(frame));
        DbPanicContext
    }

    fn with_ctx(f: impl FnOnce(&mut Vec<String>)) {
        thread_local! {
            static CTX: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
        }
        CTX.with(|ctx| f(&mut ctx.borrow_mut()));
    }
}
