//! What the global interners are holding, for a process memory report.
//!
//! The interners live outside salsa — every `impl_internable!` type and every
//! `impl_slice_internable!` tag has its own `static` storage, and `Symbol` has
//! one more — so salsa's `memory_usage` never sees them, and nothing else
//! enumerates them either. Each storage therefore registers itself here the
//! first time it is initialised, which is also the first moment it can hold
//! anything: a storage that was never touched has nothing to report.
//!
//! # What the numbers mean
//!
//! Lower bounds, in the same sense as salsa's `heap_size`:
//! - `table_bytes` is the hash table itself: one `(key, ())` slot per bucket,
//!   occupied or not, plus a control byte each and one trailing group;
//! - `value_bytes` is the allocation behind each key: the `Arc` refcount header,
//!   the value, and for slice interners the length and the slice. Buffers the
//!   value owns in turn (a spilled `SmallVec` in a `ModPath`, say) are not
//!   followed — most interned values point into *other* interners instead, and
//!   those are counted by their own storage, once.

use std::{alloc::Layout, sync::Mutex};

use dashmap::DashMap;

/// One interner's footprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternerMemory {
    /// The interned type (or slice tag), as `std::any::type_name` spells it.
    pub name: &'static str,
    /// Whether the storage is swept by the mark-and-sweep GC rather than freed
    /// when the last handle drops.
    pub gc: bool,
    /// Live entries.
    pub count: usize,
    /// Bucket array and control bytes of every shard, empty buckets included.
    pub table_bytes: usize,
    /// The allocations the entries point to; see the module docs for the bound.
    pub value_bytes: usize,
}

impl InternerMemory {
    pub fn bytes(&self) -> usize {
        self.table_bytes + self.value_bytes
    }
}

/// A storage that can report itself. Implemented by the two generic storages;
/// `Symbol`'s map is a single static and is measured directly.
pub(crate) trait MeasuredStorage: Sync {
    /// `None` until the storage is initialised. Uses `OnceLock::get`, never
    /// `get_or_init`: a report must not create the storage it describes.
    fn measure(&'static self) -> Option<InternerMemory>;
}

static REGISTRY: Mutex<Vec<&'static dyn MeasuredStorage>> = Mutex::new(Vec::new());

/// Called from inside the storage's `get_or_init`, exactly once per storage.
pub(crate) fn register(storage: &'static dyn MeasuredStorage) {
    REGISTRY.lock().unwrap().push(storage);
}

/// Every initialised interner in the process, `Symbol` included, largest first.
///
/// Takes each shard's read lock in turn while it counts, so it briefly
/// contends with interning on other threads; it never holds two at once.
pub fn interner_memory() -> Vec<InternerMemory> {
    // Copy the list out before measuring: a storage initialising on another
    // thread right now registers from inside its `OnceLock` initialiser, and
    // must not find the registry locked by a reader waiting on that same lock.
    let storages = REGISTRY.lock().unwrap().clone();
    let mut rows: Vec<InternerMemory> =
        storages.iter().filter_map(|storage| storage.measure()).collect();
    rows.push(crate::symbol::memory());
    rows.sort_by(|a, b| b.bytes().cmp(&a.bytes()).then(b.count.cmp(&a.count)));
    rows
}

/// Bytes of a DashMap's tables, whatever is in them.
///
/// hashbrown allocates `buckets` slots of `K` plus `buckets + GROUP_WIDTH`
/// control bytes in one block; an empty table points at a static singleton and
/// allocates nothing. `GROUP_WIDTH` is 16 with SSE2 and 8 otherwise — 16 is
/// used everywhere, the difference is eight bytes per shard.
pub(crate) fn table_bytes<K, V, S>(map: &DashMap<K, V, S>) -> usize
where
    K: Eq + std::hash::Hash,
    S: std::hash::BuildHasher + Clone,
{
    const GROUP_WIDTH: usize = 16;
    let slot = size_of::<(K, dashmap::SharedValue<V>)>();
    map.shards()
        .iter()
        .map(|shard| {
            let buckets = shard.read().buckets();
            if buckets <= 1 { 0 } else { buckets * (slot + 1) + GROUP_WIDTH }
        })
        .sum()
}

/// Size of the allocation behind a `triomphe::Arc<T>`: the refcount, then `T`.
pub(crate) fn arc_allocation<T>() -> usize {
    arc_allocation_of(Layout::new::<T>())
}

/// Same, for a payload whose layout is only known at runtime (a slice).
pub(crate) fn arc_allocation_of(payload: Layout) -> usize {
    Layout::new::<std::sync::atomic::AtomicUsize>()
        .extend(payload)
        .expect("an interned value's layout overflowed")
        .0
        .pad_to_align()
        .size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Interned, Symbol};

    #[derive(PartialEq, Eq, Hash)]
    struct Counted(u64, u64);
    crate::impl_internable!(Counted);

    fn row(name: &str) -> Option<InternerMemory> {
        interner_memory().into_iter().find(|row| row.name == name)
    }

    /// A storage nobody touched has nothing to report, and the report itself
    /// must not be what initialises it.
    #[test]
    fn untouched_storage_is_absent_and_stays_absent() {
        #[derive(PartialEq, Eq, Hash)]
        struct NeverInterned(u8);
        crate::impl_internable!(NeverInterned);

        let name = std::any::type_name::<NeverInterned>();
        assert_eq!(row(name), None);
        // Had the first report initialised the storage, it would have
        // registered itself and show up now.
        assert_eq!(row(name), None);
    }

    /// The counts follow the live entries and the bytes follow the counts:
    /// dropping the handles frees the non-GC entries, and the report sees it.
    #[test]
    fn interned_values_are_counted_and_weighed() {
        let name = std::any::type_name::<Counted>();
        let held: Vec<_> = (0..1000).map(|i| Interned::new(Counted(i, i))).collect();

        let full = row(name).expect("an initialised storage must report itself");
        assert_eq!(full.count, 1000);
        assert!(!full.gc);
        assert_eq!(full.value_bytes, 1000 * arc_allocation::<Counted>());
        assert!(
            full.table_bytes >= 1000 * size_of::<triomphe::Arc<Counted>>(),
            "a table holding 1000 keys has at least 1000 slots, got {} bytes",
            full.table_bytes
        );

        drop(held);
        let empty = row(name).unwrap();
        assert_eq!(empty.count, 0);
        assert_eq!(empty.value_bytes, 0);
    }

    /// `Symbol` is reported even though it never goes through a storage, and
    /// is reported once. How its bytes follow the text is checked at the end of
    /// `symbol::tests::smoke_test` — that test asserts exact map lengths, so a
    /// second test interning symbols in parallel would break it.
    #[test]
    fn symbol_is_reported_once() {
        let name = std::any::type_name::<Symbol>();
        let rows = interner_memory();
        assert_eq!(rows.iter().filter(|row| row.name == name).count(), 1);
    }
}
