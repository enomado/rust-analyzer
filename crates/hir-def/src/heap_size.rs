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
//! The container half is `pub` because `hir-ty` weighs inference results with
//! the same arithmetic. Subtle parts of it — hashbrown's per-bucket control
//! byte, a `ThinVec`'s shared empty static, a `SmallVec` that never spilled —
//! are exactly the places a second copy would get wrong in a way no test would
//! notice, since every answer is a plausible number.
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

use indexmap::IndexMap;
use la_arena::{Arena, ArenaMap, Idx};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::{item_tree::ItemTree, nameres::DefMap, nameres::LocalDefMap};

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
pub fn arena<T>(arena: &Arena<T>) -> usize {
    arena.len() * size_of::<T>()
}

/// An `ArenaMap` is a `Vec<Option<V>>` indexed by arena position, but it
/// exposes neither its length nor its capacity — only the occupied slots. Its
/// holes therefore go uncounted. The maps here are dense (a source for every
/// expression), so the gap is small, and it errs low like everything else.
pub fn arena_map<T, V>(map: &ArenaMap<Idx<T>, V>) -> usize {
    map.values().count() * size_of::<Option<V>>()
}

/// [`arena_map`] for a map whose values own heap of their own.
pub fn arena_map_with<T, V>(map: &ArenaMap<Idx<T>, V>, per_value: impl Fn(&V) -> usize) -> usize {
    map.values().map(|value| size_of::<Option<V>>() + per_value(value)).sum()
}

/// The one allocation of a hashbrown table of `buckets` slots of `T` (`std`'s
/// `HashMap`/`HashSet` and `indexmap`'s index table are all hashbrown): the
/// slots, padded to the control bytes' alignment, then one control byte per
/// slot plus a trailing group that mirrors the first, so a probe can read a
/// whole group at any position. A table with no buckets is a shared static.
///
/// The group is 16 bytes wide where hashbrown probes with SSE2 and 8 where it
/// probes a `u64` (NEON included); the allocator oracle confirms the former.
fn hash_table_bytes(buckets: usize, slot: usize) -> usize {
    const GROUP_WIDTH: usize = if cfg!(all(
        target_feature = "sse2",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    )) {
        16
    } else {
        8
    };
    if buckets == 0 {
        return 0;
    }
    let control_align = GROUP_WIDTH;
    let slots = (buckets * slot).next_multiple_of(control_align);
    slots + buckets + GROUP_WIDTH
}

/// How many buckets a hashbrown table that reports `capacity` has: it keeps
/// one bucket free below 8 buckets and an eighth free above.
fn hash_table_buckets(capacity: usize) -> usize {
    match capacity {
        0 => 0,
        1..8 => capacity + 1,
        _ => capacity / 7 * 8,
    }
}

/// How many buckets hashbrown gives a table sized for `len` elements — what
/// `with_capacity(len)` and `shrink_to_fit` produce, and so the fewest any
/// table holding `len` elements can have.
fn hash_table_buckets_for(len: usize) -> usize {
    match len {
        0 => 0,
        1..4 => 4,
        4..8 => 8,
        _ => (len * 8 / 7).next_power_of_two(),
    }
}

/// hashbrown allocates for its buckets, not for `len`; see [`hash_table_bytes`].
/// `capacity()` names the bucket count exactly, so this is exact.
pub fn hash_map<K, V, S: BuildHasher>(map: &HashMap<K, V, S>) -> usize {
    hash_table_bytes(hash_table_buckets(map.capacity()), size_of::<(K, V)>())
}

/// [`hash_map`] for a map whose values own heap of their own.
pub fn hash_map_with<K, V, S: BuildHasher>(
    map: &HashMap<K, V, S>,
    per_value: impl Fn(&V) -> usize,
) -> usize {
    hash_map(map) + map.values().map(per_value).sum::<usize>()
}

/// A set is a map to `()`: the same table.
pub fn hash_set<T>(set: &FxHashSet<T>) -> usize {
    hash_table_bytes(hash_table_buckets(set.capacity()), size_of::<T>())
}

/// An `IndexMap` is a `Vec` of `{hash, key, value}` entries beside a hashbrown
/// table of `usize` positions into it — two allocations. `capacity()` is the
/// smaller of the two capacities, and neither is exposed alone, so both are
/// sized from what can be known: the entry vector from `capacity()`, the
/// index table from `len()` (the fewest buckets that hold it). Both are exact
/// for a map that was shrunk to fit — every map in a finished def map is —
/// and a lower bound otherwise.
///
/// The index table was left out until 2026-10: on a large workspace's def maps
/// that was 5.9 million names at ≈ 10–20 bytes each.
///
/// Generic over the hasher, not written against [`crate::FxIndexMap`]: `base-db`
/// spells the same alias with `BuildHasherDefault<FxHasher>` where this crate
/// uses `FxBuildHasher`, and those are two distinct types.
pub fn index_map<K, V, S>(map: &IndexMap<K, V, S>) -> usize {
    // The entry is `indexmap::Bucket { hash: HashValue(usize), key, value }`,
    // which is not nameable here; a tuple of the same fields lays out the same.
    map.capacity() * size_of::<(usize, K, V)>()
        + hash_table_bytes(hash_table_buckets_for(map.len()), size_of::<usize>())
}

/// [`index_map`] for a map whose values own heap of their own.
pub fn index_map_with<K, V, S>(map: &IndexMap<K, V, S>, per_value: impl Fn(&V) -> usize) -> usize {
    index_map(map) + map.values().map(per_value).sum::<usize>()
}

pub fn slice<T>(slice: &[T]) -> usize {
    slice.len() * size_of::<T>()
}

pub fn vec<T>(vec: &Vec<T>) -> usize {
    vec.capacity() * size_of::<T>()
}

/// A `SmallVec` that never spilled lives in its inline array and owns nothing —
/// counting it as heap would attribute the enclosing struct's own bytes twice.
pub fn small_vec<A: smallvec::Array>(vec: &SmallVec<A>) -> usize {
    if vec.spilled() { vec.capacity() * size_of::<A::Item>() } else { 0 }
}

/// An empty `ThinVec` points at a shared static and owns nothing; a non-empty
/// one owns a header plus its elements, and reports `capacity() == 0` when
/// empty, so the multiplication covers both cases.
pub fn thin_vec<T>(vec: &ThinVec<T>) -> usize {
    vec.capacity() * size_of::<T>()
}

/// The allocation a `Box` makes for the value it points at, plus whatever that
/// value owns in turn. Spelled out because it is easy to charge for the pointee
/// and forget the box itself.
pub fn boxed<T>(value: &Option<Box<T>>, owned: impl Fn(&T) -> usize) -> usize {
    value.as_ref().map_or(0, |value| size_of::<T>() + owned(value))
}

// -------------------------------------------------------------------- census

/// One container's — or one field's, summed over many values — share of a
/// heap: what the `heap_size` arithmetic above charges for it, and what the
/// stored elements alone would take. The gap between the two is spare
/// capacity, which is the question a census asks and a total cannot answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Weight {
    /// Containers that own an allocation; an empty map or vector owns none.
    pub containers: usize,
    /// Elements stored.
    pub entries: usize,
    /// `entries` slots and nothing else (for hashbrown, with the control
    /// byte): the floor any layout of the same elements would pay.
    pub filled_bytes: usize,
    /// What `heap_size` charges: capacity, not fill.
    pub bytes: usize,
}

impl std::ops::AddAssign for Weight {
    fn add_assign(&mut self, other: Weight) {
        let Weight { containers, entries, filled_bytes, bytes } = other;
        self.containers += containers;
        self.entries += entries;
        self.filled_bytes += filled_bytes;
        self.bytes += bytes;
    }
}

impl Weight {
    fn of(len: usize, slot: usize, bytes: usize) -> Weight {
        Weight { containers: usize::from(bytes > 0), entries: len, filled_bytes: len * slot, bytes }
    }
}

/// [`hash_map`] as a [`Weight`].
pub fn hash_map_weight<K, V, S: BuildHasher>(map: &HashMap<K, V, S>) -> Weight {
    Weight::of(map.len(), size_of::<(K, V)>() + 1, hash_map(map))
}

/// [`hash_set`] as a [`Weight`].
pub fn hash_set_weight<T>(set: &FxHashSet<T>) -> Weight {
    Weight::of(set.len(), size_of::<T>() + 1, hash_set(set))
}

/// [`index_map`] as a [`Weight`].
pub fn index_map_weight<K, V, S>(map: &IndexMap<K, V, S>) -> Weight {
    Weight::of(map.len(), size_of::<(usize, K, V)>(), index_map(map))
}

/// [`vec`] as a [`Weight`].
pub fn vec_weight<T>(vec: &Vec<T>) -> Weight {
    Weight::of(vec.len(), size_of::<T>(), self::vec(vec))
}

/// [`thin_vec`] as a [`Weight`].
pub fn thin_vec_weight<T>(vec: &ThinVec<T>) -> Weight {
    Weight::of(vec.len(), size_of::<T>(), thin_vec(vec))
}

/// [`small_vec`] as a [`Weight`]: a vector still inline is no container here.
pub fn small_vec_weight<A: smallvec::Array>(vec: &SmallVec<A>) -> Weight {
    let bytes = small_vec(vec);
    Weight::of(if bytes > 0 { vec.len() } else { 0 }, size_of::<A::Item>(), bytes)
}

/// [`slice`] as a [`Weight`]; a boxed slice has no spare capacity.
pub fn slice_weight<T>(slice: &[T]) -> Weight {
    Weight::of(slice.len(), size_of::<T>(), self::slice(slice))
}

/// What def maps own, field by field — the same rows `heap_size` sums, kept
/// apart. Built for one question: of the bytes a workspace's def maps hold,
/// how much is spare capacity, which field holds it, and how many of the
/// names in item scopes are a module's own against copies a `use` brought in.
///
/// Rows are named `<owner>.<field>`; `scope.*` sums the item scopes of every
/// module of every def map added. A field whose values own buffers of their
/// own gets a second row for those (`… (spilled)`, `… (names)`).
#[derive(Debug, Default)]
pub struct DefMapCensus {
    pub def_maps: usize,
    pub modules: usize,
    pub rows: std::collections::BTreeMap<&'static str, Weight>,
    /// Entries of the `types`/`values`/`macros` maps of item scopes by how
    /// the name got there: `<ns> own` (declared or expanded in the module),
    /// `<ns> import`, `<ns> glob`, `<ns> extern crate`.
    pub scope_entries_by_origin: std::collections::BTreeMap<&'static str, usize>,
}

impl DefMapCensus {
    /// Adds a crate's def map *and* the crate-local half that
    /// `crate_local_def_map` keeps beside it — together, the `DefMapPair`
    /// row of a memory report.
    pub fn add_crate(&mut self, db: &dyn base_db::SourceDatabase, krate: base_db::Crate) {
        let pair = crate::nameres::crate_local_def_map(db, krate);
        self.add_def_map(pair.def_map(db));
        pair.local(db).weigh(&mut |row, weight| self.add_row(row, weight));
    }

    /// Adds one def map, a crate's or a block's; a block's does not pay for
    /// the crate data it shares (see [`DefMap::heap_size`]).
    pub fn add_def_map(&mut self, map: &DefMap) {
        self.def_maps += 1;
        self.modules += map.modules.len();
        map.weigh(&mut |row, weight| self.add_row(row, weight));
        for (_, module) in map.modules.iter() {
            module.scope.count_entries_by_origin(&mut |origin| {
                *self.scope_entries_by_origin.entry(origin).or_default() += 1
            });
        }
    }

    fn add_row(&mut self, row: &'static str, weight: Weight) {
        *self.rows.entry(row).or_default() += weight;
    }

    /// Everything the rows charge — what `heap_size` reports for the same
    /// def maps.
    pub fn bytes(&self) -> usize {
        self.rows.values().map(|weight| weight.bytes).sum()
    }
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

    /// `ast_id_map` and `macro_arg` (both in `hir-expand`) report a heap, and
    /// it follows the input: more items grow the AST id map, a longer macro
    /// argument grows its token tree. Read through salsa's own report, so the
    /// `heap_size = …` wiring is under test along with the arithmetic.
    #[test]
    fn ast_id_maps_and_macro_args_are_weighed() {
        fn heaps(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (usize, usize) {
            let mut db = TestDB::with_files(ra_fixture);
            // Collecting the def map lowers every item and expands every macro
            // call, which is what fills both queries.
            crate_def_map(&db, db.fetch_test_crate());
            let usage = <dyn salsa::Database>::memory_usage(&mut db);
            let heap = |query: &str| {
                let info = &usage.queries[query];
                assert!(info.count() > 0, "positive control: `{query}` has no memos at all");
                info.heap_size_of_fields()
                    .unwrap_or_else(|| panic!("`{query}` reports no heap size"))
            };
            (heap("HirFileId::ast_id_map_"), heap("MacroCallId::macro_arg_"))
        }

        let (small_map, small_arg) = heaps(
            r#"
//- /lib.rs
macro_rules! m { ($($t:tt)*) => {} }
m!(a);
"#,
        );
        let (big_map, big_arg) = heaps(&format!(
            "//- /lib.rs\nmacro_rules! m {{ ($($t:tt)*) => {{}} }}\nm!({});\n{}",
            "a ".repeat(300),
            (0..200).map(|i| format!("struct S{i};\n")).collect::<String>()
        ));

        assert!(big_map > small_map, "200 more items left the AST id maps at {small_map} bytes");
        assert!(
            big_arg > small_arg,
            "a 300-token macro argument weighed no more than a 1-token one ({small_arg} bytes)"
        );
    }

    /// The interned `MacroCallId` reports the buffers its call kind owns — here
    /// an eager call's expanded argument — and `real_span_map` its anchors;
    /// both follow the input. Read through salsa's own report, for the same
    /// reason as above.
    #[test]
    fn eager_macro_calls_and_real_span_maps_are_weighed() {
        fn heaps(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (usize, usize) {
            let mut db = TestDB::with_files(ra_fixture);
            crate_def_map(&db, db.fetch_test_crate());
            let usage = <dyn salsa::Database>::memory_usage(&mut db);
            let calls = usage
                .structs
                .iter()
                .find(|info| info.debug_name() == "MacroCallId" && info.count() > 0)
                .unwrap_or_else(|| {
                    let named: Vec<_> = usage
                        .structs
                        .iter()
                        .filter(|info| info.debug_name().contains("MacroCall"))
                        .map(|info| (info.debug_name(), info.count()))
                        .collect();
                    panic!("positive control: no `MacroCallId` was interned; similar: {named:?}")
                });
            let spans = &usage.queries["real_span_map"];
            assert!(spans.count() > 0, "positive control: `real_span_map` has no memos");
            (
                calls.heap_size_of_fields().expect("`MacroCallId` reports no heap size"),
                spans.heap_size_of_fields().expect("`real_span_map` reports no heap size"),
            )
        }
        let fixture = |args: usize, items: usize| {
            format!(
                // At item level: collecting the def map expands item macros,
                // while a call inside a `const` waits for its body.
                "//- /lib.rs\n#[rustc_builtin_macro]\nmacro_rules! concat {{}}\n\
                 concat!({});\n{}",
                vec!["\"a\""; args].join(", "),
                (0..items).map(|i| format!("struct S{i};\n")).collect::<String>()
            )
        };

        let (small_calls, small_spans) = heaps(&fixture(1, 0));
        let (big_calls, _) = heaps(&fixture(300, 0));
        let (_, big_spans) = heaps(&fixture(1, 200));

        assert!(small_calls > 0, "an eager call owns its expanded argument, yet weighed 0");
        assert!(
            big_calls > small_calls,
            "a 300-literal eager argument weighed no more than a 1-literal one ({small_calls} bytes)"
        );
        assert!(
            big_spans > small_spans,
            "200 more items left the real span maps at {small_spans} bytes"
        );
    }

    /// A signature (and a body) is one `Arc` held by two memos — its producer
    /// `with_source_map` and the `of` that clones it out — and is charged to
    /// exactly one of them, the one that survives eviction.
    ///
    /// Three sides, because each alone stays green while another breaks: the
    /// producer alone pays for object and map (`of` never ran); with `of`
    /// alive the two rows add up to that same total, not more (the object is
    /// not charged twice) and not less; and after evicting the producer the
    /// object is still in the report (the bug this rule fixes: 67 532 bodies
    /// in a live daemon weighed as 512).
    #[test]
    fn a_shared_signature_is_charged_once_and_survives_eviction() {
        use crate::{
            DefWithBodyId, FunctionId, ImplId, ModuleDefId, expr_store::body::Body,
            signatures::ImplSignature,
        };

        const IMPLS: &str = "ImplSignature::with_source_map_";
        const IMPL_OF: &str = "ImplSignature::of_";
        const BODIES: &str = "Body::with_source_map_";
        const BODY_OF: &str = "Body::of_";

        let fixture = format!(
            "//- /lib.rs\ntrait Tr<T> {{}}\n{}",
            (0..20)
                .map(|i| {
                    format!(
                        "struct S{i}<T>(T);\n\
                         impl<T: Clone, U: Copy> Tr<U> for S{i}<T> where T: Default {{}}\n\
                         fn f{i}(a: u32) -> u32 {{ let b = a + {i}; b * 2 }}\n"
                    )
                })
                .collect::<String>()
        );
        let db_with_items = || {
            let db = TestDB::with_files(&fixture);
            let def_map = crate_def_map(&db, db.fetch_test_crate());
            let (impls, functions): (Vec<ImplId>, Vec<FunctionId>) = def_map
                .modules()
                .map(|(_, module)| {
                    let functions = module.scope.declarations().filter_map(|it| match it {
                        ModuleDefId::FunctionId(it) => Some(it),
                        _ => None,
                    });
                    (module.scope.impls().collect::<Vec<_>>(), functions.collect::<Vec<_>>())
                })
                .fold((vec![], vec![]), |(mut impls, mut fns), (i, f)| {
                    impls.extend(i);
                    fns.extend(f);
                    (impls, fns)
                });
            assert_eq!((impls.len(), functions.len()), (20, 20), "positive control: fixture items");
            (db, impls, functions)
        };
        let heaps = |db: &mut TestDB, queries: &[&str]| -> Vec<Option<usize>> {
            let usage = <dyn salsa::Database>::memory_usage(db);
            queries
                .iter()
                .map(|query| usage.queries.get(*query).and_then(|info| info.heap_size_of_fields()))
                .collect()
        };

        // The producers alone: each pays for its objects and its maps.
        let (mut db, impls, functions) = db_with_items();
        for &id in &impls {
            ImplSignature::with_source_map(&db, id);
        }
        for &id in &functions {
            Body::with_source_map(&db, DefWithBodyId::FunctionId(id));
        }
        let [Some(impls_alone), Some(bodies_alone)] = heaps(&mut db, &[IMPLS, BODIES])[..] else {
            panic!("a producer reports no heap size")
        };

        // With `of` alive: the objects move to `of`, the maps stay.
        let (mut db, impls, functions) = db_with_items();
        for &id in &impls {
            ImplSignature::of(&db, id);
        }
        for &id in &functions {
            Body::of(&db, DefWithBodyId::FunctionId(id));
        }
        let [Some(impl_maps), Some(impl_objects), Some(body_maps), Some(body_objects)] =
            heaps(&mut db, &[IMPLS, IMPL_OF, BODIES, BODY_OF])[..]
        else {
            panic!("an `of` or a producer reports no heap size")
        };
        assert!(
            impl_objects > 0 && impl_maps > 0 && body_objects > 0 && body_maps > 0,
            "positive control: every half weighs something (impls {impl_objects} + \
             {impl_maps}, bodies {body_objects} + {body_maps})"
        );
        assert_eq!(
            (impl_objects + impl_maps, body_objects + body_maps),
            (impls_alone, bodies_alone),
            "with `of` alive the two rows must add up to what the producer reported alone: \
             more is the shared `Arc` charged twice, less is it charged to nobody"
        );

        // Evict the producers down to one memo each: the maps go, the objects
        // `of` still holds stay in the report.
        for query in [IMPLS, BODIES] {
            assert_eq!(salsa::Database::set_lru_capacity_by_name(&mut db, query, 1), 1);
        }
        salsa::Database::trigger_lru_eviction(&mut db);
        let after = heaps(&mut db, &[IMPLS, IMPL_OF, BODIES, BODY_OF]);
        let [
            Some(impl_maps_left),
            Some(impl_objects_left),
            Some(body_maps_left),
            Some(body_objects_left),
        ] = after[..]
        else {
            panic!("a report lost its heap size after eviction: {after:?}")
        };
        assert!(
            impl_maps_left < impl_maps && body_maps_left < body_maps,
            "positive control: eviction dropped no source map (impls {impl_maps} → \
             {impl_maps_left}, bodies {body_maps} → {body_maps_left})"
        );
        assert_eq!(
            (impl_objects_left, body_objects_left),
            (impl_objects, body_objects),
            "evicting the producer took objects out of the report that `of` still holds"
        );
    }

    /// The census is `heap_size`'s own walk with the rows kept apart: they add
    /// up to what the `DefMapPair` reports, and the names a glob import copies
    /// into a module are told apart from the module's own.
    #[test]
    fn the_def_map_census_adds_up_to_the_pair_and_counts_glob_copies() {
        use crate::{heap_size::DefMapCensus, nameres::crate_local_def_map};

        // Unit structs, so each name lands in two namespaces: types and values.
        let structs = (0..50).map(|i| format!("pub struct S{i};\n")).collect::<String>();
        let db = TestDB::with_files(&format!(
            "//- /lib.rs\nmod a {{\n{structs}}}\nmod b {{ use super::a::*; }}\n"
        ));
        let krate = db.fetch_test_crate();
        let mut census = DefMapCensus::default();
        census.add_crate(&db, krate);

        let pair = crate_local_def_map(&db, krate);
        // What `def_map_pair` sums for the tracked struct's two fields.
        let reported = pair.def_map(&db).heap_size() + pair.local(&db).heap_size();
        assert_eq!(census.bytes(), reported, "the census rows do not add up to the pair's heap");
        assert_eq!((census.def_maps, census.modules), (1, 3), "positive control: root, a, b");
        let origin = |key: &str| census.scope_entries_by_origin.get(key).copied().unwrap_or(0);
        assert_eq!(
            (origin("types glob"), origin("values glob")),
            (50, 50),
            "`use super::a::*` copies 50 names into each namespace of `b`"
        );
        assert_eq!(origin("types own"), 52, "50 structs in `a`, `a` and `b` in the root");
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
