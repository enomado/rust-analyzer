//! Sizing the heap that a memoized body hangs on to.
//!
//! salsa collects this through the `heap_size = <fn>` option on a tracked
//! function and reports it from [`salsa::Database::memory_usage`]. Without the
//! option an ingredient answers `None`, and a memory report can then say how
//! MANY memos a query has accumulated but never how big they grew — which is
//! the difference between naming a suspect and convicting one.
//!
//! These functions run from `memory_usage()` alone, never on a memo write (see
//! `salsa::function::memo::Memo::memory_usage`), so they are free to walk as
//! much as they need to.
//!
//! # What the numbers mean
//!
//! They count the *owned buffers*: the arena, map and slice allocations
//! hanging off a store. They deliberately do not count indirection *inside* an
//! element — the `Box<[ExprId]>` in one `Expr` variant, say. Every number here
//! is therefore a LOWER BOUND on the real heap.
//!
//! That is the useful direction. A caller comparing ingredients wants "at
//! least this much", and an over-count would be worse than useless. The
//! alternative — a match over every HIR variant — would go stale on the next
//! upstream change with nothing to notice it had.
//!
//! # Which query pays for the shared `Arc`
//!
//! [`Body::of`] hands out a clone of the very `Arc<Body>` that
//! [`Body::with_source_map`] built, and every `XSignature::of` does the same
//! with its signature. Charging both memos would count one allocation twice,
//! so exactly one of them pays — and it has to be `of`.
//!
//! The producer is the query with an `lru`: once it is evicted its value is
//! gone, while `of` (never capped) keeps the very same `Arc` alive. A producer
//! that charged for the object would therefore make every evicted body vanish
//! from the report while its bytes stayed resident — which is exactly what the
//! report showed before (67 532 bodies held by `Body::of`, 10 MB charged to the
//! 512 the producer still had; see bur's `RA_MEMORY_MISREPORTS.md`, № 16).
//!
//! So `of` charges the object, the `Arc` block included, and the producer
//! charges its source map — plus the object only while it is the `Arc`'s
//! sole holder, i.e. before `of` ran or after `of` was collected. A third
//! holder elsewhere makes the object go uncounted, never counted twice: lower
//! bound, as everywhere in this module.

use triomphe::Arc;

use crate::{
    expr_store::{
        ExpressionOnlySourceMap, ExpressionOnlyStore, ExpressionStore, ExpressionStoreSourceMap,
        FormatTemplate,
        body::{Body, BodySourceMap},
    },
    // The container arithmetic is shared with the crate's other `heap_size`
    // entry points, and lives beside them.
    heap_size::{
        arena, arena_map, arena_map_with, boxed, hash_map, hash_map_with, slice, small_vec,
        thin_vec, vec,
    },
    hir::generics::GenericParams,
    signatures::{
        ConstSignature, EnumSignature, FunctionSignature, ImplSignature, StaticSignature,
        StructSignature, TraitSignature, TypeAliasSignature, UnionSignature, VariantFields,
    },
};

/// `heap_size` for [`Body::with_source_map`]: the source map, and the body
/// only while nothing else holds it (see the module docs).
pub(crate) fn body_with_source_map(value: &(Arc<Body>, BodySourceMap)) -> usize {
    let (body, source_map) = value;
    source_map_heap(source_map) + sole_holder(body, body_heap)
}

/// `heap_size` for [`Body::of`]: the body, `Arc` block included.
pub(crate) fn body_of(body: &Arc<Body>) -> usize {
    arc(body, body_heap)
}

/// `heap_size` for every `XSignature::with_source_map`. They all return the
/// same shape — the signature beside the source map it was lowered with — so
/// they share one entry point, and only the signature half differs.
pub(crate) fn signature_with_source_map<T: SignatureHeap>(
    value: &(Arc<T>, ExpressionStoreSourceMap),
) -> usize {
    let (signature, source_map) = value;
    store_source_map_heap(source_map) + sole_holder(signature, T::heap_size)
}

/// `heap_size` for every `XSignature::of`: the signature, `Arc` block included.
pub(crate) fn signature_of<T: SignatureHeap>(signature: &Arc<T>) -> usize {
    arc(signature, T::heap_size)
}

/// The allocation behind an `Arc` — its count beside the value — plus whatever
/// the value owns. Padding for an over-aligned value goes uncounted.
fn arc<T>(value: &Arc<T>, owned: impl Fn(&T) -> usize) -> usize {
    size_of::<usize>() + size_of::<T>() + owned(value)
}

/// [`arc`], for the producer of an `Arc` that its `of` query normally holds a
/// clone of: charged here only when this memo is the last holder left.
fn sole_holder<T>(value: &Arc<T>, owned: impl Fn(&T) -> usize) -> usize {
    if Arc::strong_count(value) == 1 { arc(value, owned) } else { 0 }
}

/// What a signature owns beside its source map.
///
/// Spelled out per signature rather than derived: the shared parts (the lowered
/// store, the generic parameters) sit under different names in each, and a
/// blanket impl would have to guess. An implementation that forgets a field
/// undercounts silently, which is why each one lists what it owns.
pub(crate) trait SignatureHeap {
    fn heap_size(&self) -> usize;
}

macro_rules! signature_heap {
    ($($signature:ty => |$this:ident| $owned:expr,)*) => {
        $(impl SignatureHeap for $signature {
            fn heap_size(&self) -> usize {
                let $this = self;
                store_heap(&$this.store) + $owned
            }
        })*
    };
}

signature_heap! {
    // `Name` is an interned symbol and `*Flags` are bitflags: neither allocates.
    StructSignature => |it| generic_params_heap(&it.generic_params),
    UnionSignature => |it| generic_params_heap(&it.generic_params),
    EnumSignature => |it| generic_params_heap(&it.generic_params),
    TraitSignature => |it| generic_params_heap(&it.generic_params),
    ImplSignature => |it| generic_params_heap(&it.generic_params),
    // These two carry no generic parameters at all (the field is commented out
    // upstream), so there is nothing beyond the store.
    ConstSignature => |_it| 0,
    StaticSignature => |_it| 0,
    FunctionSignature => |it| generic_params_heap(&it.generic_params) + slice(&it.params),
    TypeAliasSignature => |it| generic_params_heap(&it.generic_params) + slice(&it.bounds),
    // Not a signature by name, but the same query shape and the same owner
    // relationship, so it rides along. Its field arena is read through the
    // accessor because the field itself is private to `signatures`.
    VariantFields => |it| arena(it.fields()),
}

fn generic_params_heap(params: &GenericParams) -> usize {
    arena(&params.type_or_consts) + arena(&params.lifetimes) + slice(&params.where_predicates)
}

// -------------------------------------------------------------------- stores

fn body_heap(body: &Body) -> usize {
    // `self_param` is an inline `Option<Param<BindingId>>` and owns nothing.
    store_heap(&body.store) + slice(&body.params)
}

fn store_heap(store: &ExpressionStore) -> usize {
    arena(&store.types) + arena(&store.lifetimes) + boxed(&store.expr_only, expr_only_store_heap)
}

fn expr_only_store_heap(store: &ExpressionOnlyStore) -> usize {
    arena(&store.exprs)
        + arena(&store.pats)
        + arena(&store.bindings)
        + arena(&store.labels)
        + hash_map(&store.binding_owners)
        + slice(&store.block_scopes)
        + hash_map(&store.ident_hygiene)
        + small_vec(&store.expr_roots)
}

fn source_map_heap(source_map: &BodySourceMap) -> usize {
    store_source_map_heap(&source_map.store)
}

fn store_source_map_heap(source_map: &ExpressionStoreSourceMap) -> usize {
    arena_map(&source_map.types_map_back)
        + hash_map(&source_map.types_map)
        + arena_map(&source_map.lifetime_map_back)
        + hash_map(&source_map.lifetime_map)
        + boxed(&source_map.expr_only, expr_only_source_map_heap)
}

fn expr_only_source_map_heap(source_map: &ExpressionOnlySourceMap) -> usize {
    hash_map(&source_map.expr_map)
        + arena_map(&source_map.expr_map_back)
        + hash_map(&source_map.pat_map)
        + arena_map(&source_map.pat_map_back)
        + hash_map(&source_map.label_map)
        + arena_map(&source_map.label_map_back)
        + arena_map_with(&source_map.binding_definitions, small_vec)
        + hash_map(&source_map.field_map_back)
        + hash_map(&source_map.pat_field_map_back)
        + boxed(&source_map.template_map, format_template_heap)
        + hash_map(&source_map.expansions)
        + thin_vec(&source_map.diagnostics)
}

fn format_template_heap(template: &FormatTemplate) -> usize {
    hash_map_with(&template.format_args_to_captures, |(_hygiene, captures)| vec(captures))
        + hash_map_with(&template.asm_to_captures, |captures| {
            vec(captures) + captures.iter().map(vec).sum::<usize>()
        })
        + hash_map(&template.implicit_capture_to_source)
}
