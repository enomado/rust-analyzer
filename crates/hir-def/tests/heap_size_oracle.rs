//! Oracle for the container arithmetic of `hir_def::heap_size`: what a
//! hashbrown map, set and `IndexMap` hold, counted by the allocator itself
//! (`test_utils::counting_alloc`; one `#[test]`, for the reason given there).
//!
//! Every `heap_size` of hir-def and hir-ty is built on these few functions,
//! and the def maps alone put millions of names through them on a large
//! workspace — an error per table is multiplied by the tables of every
//! module of every crate.

use hir_def::heap_size::{hash_map, hash_set, index_map};
use indexmap::IndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use test_utils::counting_alloc::{CountingAllocator, live_bytes};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Bytes the allocator holds for what `build` returns, beside what the
/// function under test says, for one value.
fn held<T>(build: impl FnOnce() -> T, reported: impl FnOnce(&T) -> usize) -> (isize, usize, T) {
    let before = live_bytes();
    let value = build();
    let live = live_bytes() - before;
    let reported = reported(&value);
    (live, reported, value)
}

#[test]
fn container_arithmetic_is_what_the_allocator_holds() {
    // Sizes around every bucket-count step, including the small-table
    // special cases (4 and 8 buckets) and the empty static.
    let lens: Vec<usize> = (0..40).chain([55, 56, 57, 100, 112, 113, 1000, 5000]).collect();
    // Two slot sizes, one a multiple of the control alignment and one not.
    type Wide = [u64; 5];
    let mut checked = 0;
    for &len in &lens {
        // Grown by insertion: capacity says the bucket count exactly.
        let (live, reported, _map) = held(
            || (0..len as u64).map(|i| (i, [i; 5])).collect::<FxHashMap<u64, Wide>>(),
            hash_map,
        );
        assert_eq!(live, reported as isize, "grown FxHashMap<u64, [u64; 5]> of {len}");

        let (live, reported, _set) =
            held(|| (0..len as u32).collect::<FxHashSet<u32>>(), |set| hash_set(set));
        assert_eq!(live, reported as isize, "grown FxHashSet<u32> of {len}");

        // Shrunk to fit, as every map of a finished def map is: exact for
        // both of an `IndexMap`'s allocations.
        let (live, reported, _map) = held(
            || {
                let mut map: IndexMap<u32, Wide, FxBuildHasher> = IndexMap::default();
                map.extend((0..len as u32).map(|i| (i, [u64::from(i); 5])));
                map.shrink_to_fit();
                map
            },
            index_map,
        );
        assert_eq!(live, reported as isize, "shrunk IndexMap<u32, [u64; 5]> of {len}");

        let (live, reported, _map) = held(
            || {
                let mut map: IndexMap<u32, u8, FxBuildHasher> = IndexMap::default();
                map.extend((0..len as u32).map(|i| (i, i as u8)));
                map.shrink_to_fit();
                map
            },
            index_map,
        );
        assert_eq!(live, reported as isize, "shrunk IndexMap<u32, u8> of {len}");

        // Grown and never shrunk: the index table can only be larger than
        // what `len` needs, so the report stays a lower bound.
        let (live, reported, _map) = held(
            || (0..len as u32).map(|i| (i, i)).collect::<IndexMap<u32, u32, FxBuildHasher>>(),
            index_map,
        );
        assert!(reported as isize <= live, "grown IndexMap of {len}: {reported} over {live} held");
        checked += 1;
    }
    assert_eq!(checked, lens.len(), "positive control: every size was checked");
}
