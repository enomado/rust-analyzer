//! Weighing the heap that memoized values in this crate hang on to.
//!
//! salsa collects these through the `heap_size = <fn>` option on a tracked
//! function (or tracked struct) and reports them from
//! [`salsa::Database::memory_usage`]. Without the option an ingredient answers
//! `None`, and a memory report can then say how MANY memos a query has
//! accumulated but never how big they grew — which is the difference between
//! naming a suspect and convicting one.
//!
//! This module holds two things: the container arithmetic shared by every
//! `heap_size` function in the crate (including [`crate::expr_store::heap_size`],
//! which owns the body/signature half), and the entry points for the queries
//! whose result types live here.
//!
//! # What the numbers mean
//!
//! They count the *owned buffers*: arenas, map and slice allocations hanging
//! off a value. They deliberately do not count indirection *inside* an element,
//! nor the padding a container's own bucket layout adds. Every number here is
//! therefore a LOWER BOUND on the real heap.
//!
//! That is the useful direction. A caller comparing ingredients wants "at least
//! this much", and an over-count would be worse than useless: it would move an
//! ingredient up the report on bytes that do not exist.
//!
//! # Shared allocations are charged to one owner only
//!
//! Where an `Arc` is handed to many memos, exactly one of them counts it. See
//! [`def_map`] for the case that actually bites — a crate's `DefMapCrateData`
//! is shared with every block def map in that crate, and charging all of them
//! would multiply one allocation by the thousands.

use std::{collections::HashMap, hash::BuildHasher, mem::size_of};

use la_arena::{Arena, ArenaMap, Idx};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::{FxIndexMap, item_tree::ItemTree, nameres::DefMap, nameres::LocalDefMap};

// -------------------------------------------------------------- entry points

/// `heap_size` for [`crate::nameres::block_def_map`].
pub(crate) fn def_map(map: &DefMap) -> usize {
    map.heap_size()
}

/// `heap_size` for the `DefMapPair` tracked struct, which is where a *crate's*
/// def map lives — `crate_local_def_map` returns the struct, so its own memo
/// holds nothing but an id and the bytes have to be reported from here.
pub(crate) fn def_map_pair(fields: &(DefMap, LocalDefMap)) -> usize {
    let (def_map, local) = fields;
    def_map.heap_size() + local.heap_size()
}

/// `heap_size` for [`crate::item_tree::block_item_tree_query`].
pub(crate) fn item_tree(tree: &ItemTree) -> usize {
    tree.heap_size()
}

/// `heap_size` for `file_item_tree_query`, which answers `None` for a file that
/// declares nothing rather than storing an empty tree.
pub(crate) fn file_item_tree(tree: &Option<Box<ItemTree>>) -> usize {
    boxed(tree, ItemTree::heap_size)
}

// ---------------------------------------------------------------- containers

/// An arena is a `Vec<T>`, and lowering calls `shrink_to_fit` on these once it
/// is done, so the length is the allocation rather than merely the fill.
pub(crate) fn arena<T>(arena: &Arena<T>) -> usize {
    arena.len() * size_of::<T>()
}

/// An `ArenaMap` is a `Vec<Option<V>>` indexed by arena position, but it
/// exposes neither its length nor its capacity — only the occupied slots. Its
/// holes therefore go uncounted. The maps here are dense (a source for every
/// expression), so the gap is small, and it errs low like everything else.
pub(crate) fn arena_map<T, V>(map: &ArenaMap<Idx<T>, V>) -> usize {
    map.values().count() * size_of::<Option<V>>()
}

/// [`arena_map`] for a map whose values own heap of their own.
pub(crate) fn arena_map_with<T, V>(
    map: &ArenaMap<Idx<T>, V>,
    per_value: impl Fn(&V) -> usize,
) -> usize {
    map.values().map(|value| size_of::<Option<V>>() + per_value(value)).sum()
}

/// hashbrown allocates for `capacity`, not for `len`, and keeps one control
/// byte per bucket beside the bucket itself.
pub(crate) fn hash_map<K, V, S: BuildHasher>(map: &HashMap<K, V, S>) -> usize {
    map.capacity() * (size_of::<(K, V)>() + 1)
}

/// [`hash_map`] for a map whose values own heap of their own.
pub(crate) fn hash_map_with<K, V, S: BuildHasher>(
    map: &HashMap<K, V, S>,
    per_value: impl Fn(&V) -> usize,
) -> usize {
    hash_map(map) + map.values().map(per_value).sum::<usize>()
}

/// A set is a map to `()`; hashbrown stores the same control byte per bucket.
pub(crate) fn hash_set<T>(set: &FxHashSet<T>) -> usize {
    set.capacity() * (size_of::<T>() + 1)
}

/// An `IndexMap` is a `Vec` of `{hash, key, value}` buckets beside a hashbrown
/// table of positions into it. Neither the bucket type nor the table is
/// nameable from here, so this counts the entry vector as
/// `capacity * (usize + K + V)` and leaves the index table and any bucket
/// padding out — lower bound, as everywhere else.
pub(crate) fn index_map<K, V>(map: &FxIndexMap<K, V>) -> usize {
    map.capacity() * (size_of::<usize>() + size_of::<K>() + size_of::<V>())
}

/// [`index_map`] for a map whose values own heap of their own.
pub(crate) fn index_map_with<K, V>(
    map: &FxIndexMap<K, V>,
    per_value: impl Fn(&V) -> usize,
) -> usize {
    index_map(map) + map.values().map(per_value).sum::<usize>()
}

pub(crate) fn slice<T>(slice: &[T]) -> usize {
    slice.len() * size_of::<T>()
}

pub(crate) fn vec<T>(vec: &Vec<T>) -> usize {
    vec.capacity() * size_of::<T>()
}

/// A `SmallVec` that never spilled lives in its inline array and owns nothing —
/// counting it as heap would attribute the enclosing struct's own bytes twice.
pub(crate) fn small_vec<A: smallvec::Array>(vec: &SmallVec<A>) -> usize {
    if vec.spilled() { vec.capacity() * size_of::<A::Item>() } else { 0 }
}

/// An empty `ThinVec` points at a shared static and owns nothing; a non-empty
/// one owns a header plus its elements, and reports `capacity() == 0` when
/// empty, so the multiplication covers both cases.
pub(crate) fn thin_vec<T>(vec: &ThinVec<T>) -> usize {
    vec.capacity() * size_of::<T>()
}

/// The allocation a `Box` makes for the value it points at, plus whatever that
/// value owns in turn. Spelled out because it is easy to charge for the pointee
/// and forget the box itself.
pub(crate) fn boxed<T>(value: &Option<Box<T>>, owned: impl Fn(&T) -> usize) -> usize {
    value.as_ref().map_or(0, |value| size_of::<T>() + owned(value))
}

#[cfg(test)]
mod tests {
    use test_fixture::WithFixture;

    use crate::{nameres::crate_def_map, test_db::TestDB};

    /// The rule that a shared `Arc` is charged to exactly one owner, asserted
    /// from both sides — because either half alone stays green while the other
    /// is broken.
    ///
    /// Growing a crate's `DefMapCrateData` has to move the crate's def map (or
    /// the data is not counted at all) and must NOT move a block def map inside
    /// that crate (or one allocation is reported once per block, and a
    /// workspace has thousands of blocks to a couple of hundred crates).
    #[test]
    fn a_block_def_map_does_not_pay_for_the_crate_data_it_borrows() {
        /// `(block def map, crate def map)` for a fixture whose `$0` sits
        /// inside a block expression.
        fn heaps(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (usize, usize) {
            let (db, position) = TestDB::with_position(ra_fixture);
            let block_module = db.module_at_position(position);
            let block = block_module.def_map(&db).heap_size();
            let krate = crate_def_map(&db, db.fetch_test_crate()).heap_size();
            (block, krate)
        }

        let (plain_block, plain_crate) = heaps(
            r#"
//- /lib.rs
fn f() {
    {
        fn inside_the_block() { $0 }
    }
}
"#,
        );
        // `#![register_tool]` is the cheapest way to grow crate-wide data: the
        // names land in `DefMapCrateData::registered_tools`. One attribute per
        // name — the collector reads a single ident and ignores a list.
        let (loaded_block, loaded_crate) = heaps(
            r#"
//- /lib.rs
#![register_tool(one)]
#![register_tool(two)]
#![register_tool(three)]
#![register_tool(four)]
#![register_tool(five)]
#![register_tool(six)]
#![register_tool(seven)]
#![register_tool(eight)]
fn f() {
    {
        fn inside_the_block() { $0 }
    }
}
"#,
        );

        assert!(
            loaded_crate > plain_crate,
            "eight more registered tools did not move the crate def map ({plain_crate} bytes \
             either way), so the shared crate data is going uncounted by its one legitimate \
             owner and the assertion below proves nothing"
        );
        assert_eq!(
            plain_block, loaded_block,
            "the block def map moved with crate-wide data it only borrows an `Arc` to; every \
             block in the crate is now charged for the same allocation"
        );
    }

    /// The item tree is five vectors and nothing else. Two assertions, because
    /// the first one alone stays green while only `top_level` is counted:
    /// adding top-level items grows that vector *and* the data vectors at once,
    /// so the second pair keeps the file's top level at exactly one item and
    /// lets only the data vectors move.
    #[test]
    fn a_file_with_more_items_has_a_heavier_item_tree() {
        fn heap(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> usize {
            let (db, file_id) = TestDB::with_single_file(ra_fixture);
            crate::item_tree::file_item_tree(&db, file_id.into(), db.test_crate()).heap_size()
        }

        let structs =
            |count: usize| (0..count).map(|i| format!("struct S{i};\n")).collect::<String>();

        let one = heap("struct S;");
        let many = heap(&structs(200));
        assert!(
            many > one,
            "200 structs weighed no more than one ({one} bytes), so the tree's vectors are not \
             being counted"
        );

        let empty_module = heap("mod m {}");
        let loaded_module = heap(&format!("mod m {{\n{}}}", structs(200)));
        assert!(
            loaded_module > empty_module,
            "200 items inside the file's single top-level `mod` weighed no more than an empty one \
             ({empty_module} bytes): only `top_level` is reaching the total, and the assertion \
             above cannot tell"
        );
    }

    /// Nearly everything a def map owns hangs off the item scopes of its
    /// modules, which is the reason it is worth weighing at all.
    #[test]
    fn a_module_with_more_items_has_a_heavier_def_map() {
        fn heap(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> usize {
            let db = TestDB::with_files(ra_fixture);
            crate_def_map(&db, db.fetch_test_crate()).heap_size()
        }

        let one = heap("//- /lib.rs\nstruct S;");
        let many = heap(&format!(
            "//- /lib.rs\n{}",
            (0..200).map(|i| format!("struct S{i};\n")).collect::<String>()
        ));

        assert!(
            many > one,
            "200 declarations weighed no more than one ({one} bytes): the module scopes are not \
             reaching the total"
        );
    }
}
