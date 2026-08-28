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
