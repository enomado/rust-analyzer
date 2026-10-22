//! Where a [`crate::FileText`] keeps its text.
//!
//! By default a file text lives in the heap for as long as the database does:
//! the input owns an `Arc<str>` and nothing ever evicts an input. On a large
//! workspace with its dependencies that is tens of thousands of files and
//! hundreds of MB, nearly all of it read once — to build the syntax tree, the
//! item tree and the AST id map — and then only on the rare re-parse after the
//! LRU dropped the tree.
//!
//! An embedder can opt into a [`FileTextStore`] instead. The input then keeps a
//! *handle* to the text, and [`crate::FileText::text`] loads the text through an
//! LRU-capped query, so the heap holds only the texts read recently:
//!
//! - **read back from disk** ([`StoredText::OnDisk`]): a library file under one
//!   of [`FileTextStore::read_back_dirs`] — directories whose files do not change
//!   under a loaded workspace (the cargo registry's unpacked sources, git
//!   checkouts by revision, toolchains). The input keeps the path, the length and
//!   a hash of the text; every load checks both, because "does not change" is a
//!   convention, not a guarantee (cargo's cache GC deletes old sources, a
//!   toolchain can be reinstalled in place).
//! - **a private copy, mapped** ([`StoredText::Mapped`]): every other file —
//!   the workspace's own, which change, library files outside the read-back
//!   directories (build-script output, path dependencies), and a read-back
//!   candidate whose size on disk is not the length of the text it was given
//!   (an editor's unsaved buffer: reading it back could never pass the check).
//!   The text is written to
//!   an unlinked file under [`FileTextStore::copy_dir`] and read through a
//!   read-only mapping of it. The copy is the snapshot the database was given, so
//!   it cannot drift; its pages are file-backed page cache the kernel can drop
//!   under pressure, not anonymous heap; and an unlinked file disappears with the
//!   process, however it exits.
//! - **in the heap** ([`StoredText::Inline`]): without a store, for an empty
//!   text, and when the copy cannot be written (logged).
//!
//! # When the original changed
//!
//! A load that finds a read-back file changed or gone cannot return the text the
//! database was built from: that text is gone. It cannot return the new one
//! either — item trees, AST ids and spans were computed from the old one. So it
//! records the [drift](Files::text_drift) and panics. Nothing inside a query can
//! repair an input; the embedder that reads the drift reloads the workspace.
//!
//! [`Files::text_drift`]: crate::Files::text_drift

use std::{
    fs::File,
    hash::Hasher as _,
    io,
    sync::{Mutex, PoisonError},
};

use rustc_hash::FxHasher;
use triomphe::Arc;
use vfs::{AbsPath, AbsPathBuf};

/// How an embedder wants file texts stored; see the [module](self) docs.
#[derive(Debug, Clone)]
pub struct FileTextStore {
    /// Library files under these directories are read back from disk when
    /// their text is needed again.
    pub read_back_dirs: Vec<AbsPathBuf>,
    /// Directory for the mapped copies of every other file text. It must be on
    /// a real file system: on a `tmpfs` the "file-backed" pages are memory
    /// again, and the copy saves nothing.
    pub copy_dir: AbsPathBuf,
}

/// Size of one copy segment. Large enough that a workspace of a few hundred MB
/// of text takes a handful of mappings rather than one per file (the number of
/// mappings a process may hold is limited), small enough that a segment kept
/// alive by one surviving text after edits wastes little disk.
const SEGMENT_BYTES: usize = 64 << 20;

/// The form a [`crate::FileText`] input keeps its text in.
#[derive(Debug)]
pub enum StoredText {
    Inline(Arc<str>),
    Mapped(MappedText),
    OnDisk(DiskText),
}

impl StoredText {
    /// Heap the input itself holds; a loaded text is charged to the query that
    /// loaded it, see `loaded_file_text_heap_size`.
    pub(crate) fn heap_size(&self) -> usize {
        match self {
            StoredText::Inline(text) => arc_str_heap_size(text),
            // The segment is shared by many texts and is a mapping, not heap;
            // its handful of bytes of bookkeeping are not worth a rule.
            StoredText::Mapped(_) => 0,
            // `Box<str>`: exactly the path's bytes.
            StoredText::OnDisk(disk) => disk.path.len(),
        }
    }

    /// The text of a handle. An inline text is never loaded: it is already in
    /// the heap, and `FileText::text` hands it out directly.
    pub(crate) fn load(&self) -> Arc<str> {
        match self {
            StoredText::Inline(_) => unreachable!("an inline file text is read directly"),
            StoredText::Mapped(mapped) => Arc::from(mapped.text()),
            StoredText::OnDisk(disk) => disk.load(),
        }
    }
}

/// Heap of a triomphe `Arc<str>`: the reference count, then the bytes, padded
/// to the count's alignment.
pub(crate) fn arc_str_heap_size(text: &Arc<str>) -> usize {
    (std::mem::size_of::<usize>() + text.len()).next_multiple_of(std::mem::align_of::<usize>())
}

/// A text in a copy segment.
#[derive(Debug)]
pub struct MappedText {
    segment: std::sync::Arc<Segment>,
    offset: usize,
    len: usize,
}

impl MappedText {
    fn text(&self) -> &str {
        // Written from a `&str` by `Copies::copy`, so valid; checked anyway,
        // because the check costs a fraction of the copy into the heap that
        // follows it, and a wrong offset would otherwise surface far away.
        std::str::from_utf8(&self.segment.map[self.offset..self.offset + self.len])
            .expect("a mapped file text is the UTF-8 that was written to it")
    }
}

/// One unlinked file holding copies of texts, mapped read-only.
#[derive(Debug)]
struct Segment {
    file: File,
    map: memmap2::Mmap,
}

impl Segment {
    fn create(dir: &AbsPath, len: usize) -> io::Result<Segment> {
        let file = tempfile::tempfile_in(dir)?;
        file.set_len(len as u64)?;
        // SAFETY: the file is unlinked and only this process has it open; it is
        // never truncated, and bytes are written only into ranges no
        // `MappedText` refers to yet (`Copies::copy` hands a range out after
        // writing it, and never writes it again).
        let map = unsafe { memmap2::MmapOptions::new().len(len).map(&file)? };
        Ok(Segment { file, map })
    }

    fn write_at(&self, offset: usize, bytes: &[u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::write_all_at(&self.file, bytes, offset as u64)
        }
        #[cfg(windows)]
        {
            let mut written = 0;
            while written < bytes.len() {
                let n = std::os::windows::fs::FileExt::seek_write(
                    &self.file,
                    &bytes[written..],
                    (offset + written) as u64,
                )?;
                if n == 0 {
                    return Err(io::ErrorKind::WriteZero.into());
                }
                written += n;
            }
            Ok(())
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

/// A library text left on disk.
#[derive(Debug)]
pub struct DiskText {
    path: Box<str>,
    len: usize,
    hash: u64,
    drift: std::sync::Arc<TextDrift>,
}

impl DiskText {
    fn load(&self) -> Arc<str> {
        let bytes = match std::fs::read(&*self.path) {
            Ok(bytes) => bytes,
            Err(error) => self.drifted(format_args!("can no longer be read ({error})")),
        };
        if bytes.len() != self.len || content_hash(&bytes) != self.hash {
            self.drifted(format_args!(
                "changed on disk ({} bytes now, {} when loaded)",
                bytes.len(),
                self.len
            ));
        }
        let text = String::from_utf8(bytes)
            .expect("the bytes hash to a text that was UTF-8 when it was loaded");
        Arc::from(text)
    }

    fn drifted(&self, what: std::fmt::Arguments<'_>) -> ! {
        let report = format!("{} {what}", self.path);
        self.drift.record(&report);
        panic!("file text of {report}; the workspace must be reloaded");
    }
}

/// The first read-back file found changed, for the embedder to act on.
#[derive(Debug, Default)]
pub(crate) struct TextDrift(Mutex<Option<String>>);

impl TextDrift {
    fn record(&self, report: &str) {
        let mut first = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if first.is_none() {
            *first = Some(report.to_owned());
        }
    }

    pub(crate) fn first(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
}

fn content_hash(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(bytes);
    hasher.finish()
}

/// A [`FileTextStore`] at work: the policy, the open copy segment, and the
/// drift report the read-back texts share.
#[derive(Debug)]
pub(crate) struct TextStore {
    read_back_dirs: Vec<AbsPathBuf>,
    copies: Mutex<Copies>,
    pub(crate) drift: std::sync::Arc<TextDrift>,
}

#[derive(Debug)]
struct Copies {
    dir: AbsPathBuf,
    /// The segment being filled and the first free byte in it.
    open: Option<(std::sync::Arc<Segment>, usize)>,
}

impl TextStore {
    pub(crate) fn new(store: FileTextStore) -> TextStore {
        TextStore {
            read_back_dirs: store.read_back_dirs,
            copies: Mutex::new(Copies { dir: store.copy_dir, open: None }),
            drift: Default::default(),
        }
    }

    /// Store `text`, the contents of the file at `path` (if the file has a
    /// real path) in a library root or not.
    pub(crate) fn store(&self, path: Option<&AbsPath>, is_library: bool, text: &str) -> StoredText {
        if text.is_empty() {
            return StoredText::Inline(Arc::from(text));
        }
        if is_library
            && let Some(path) = path
            && self.read_back_dirs.iter().any(|dir| path.starts_with(dir))
            && let Some(disk) = self.disk_text(path, text)
        {
            return StoredText::OnDisk(disk);
        }
        let copied = self.copies.lock().unwrap_or_else(PoisonError::into_inner).copy(text);
        match copied {
            Ok(mapped) => StoredText::Mapped(mapped),
            Err(error) => {
                tracing::warn!(
                    "cannot copy a file text out of the heap, keeping it there: {error}"
                );
                StoredText::Inline(Arc::from(text))
            }
        }
    }

    /// `None` when the file on disk cannot be the text: a loader that changed
    /// the bytes (or an editor's unsaved buffer) would fail every read-back.
    fn disk_text(&self, path: &AbsPath, text: &str) -> Option<DiskText> {
        let on_disk = std::fs::metadata(path).ok()?.len();
        if on_disk != text.len() as u64 {
            return None;
        }
        Some(DiskText {
            path: path.as_str().into(),
            len: text.len(),
            hash: content_hash(text.as_bytes()),
            drift: self.drift.clone(),
        })
    }
}

impl Copies {
    fn copy(&mut self, text: &str) -> io::Result<MappedText> {
        let fits =
            matches!(&self.open, Some((segment, free)) if segment.len() - free >= text.len());
        if !fits {
            // A text bigger than a segment gets one of its own, full at once.
            let segment = Segment::create(&self.dir, SEGMENT_BYTES.max(text.len()))?;
            self.open = Some((std::sync::Arc::new(segment), 0));
        }
        let (segment, free) = self.open.as_mut().expect("a segment was just opened");
        segment.write_at(*free, text.as_bytes())?;
        let mapped = MappedText { segment: segment.clone(), offset: *free, len: text.len() };
        *free += text.len();
        Ok(mapped)
    }
}
