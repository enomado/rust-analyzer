//! What a memoized macro expansion costs on the heap.
//!
//! Same contract as [`syntax::heap_size`], and the same reason: the expansion
//! query is LRU-capped, and a capacity is only arguable against bytes. Numbers
//! here are lower bounds — see that module for what the parse half leaves out.

use std::mem::size_of;

use span::{Span, TextSize};
use syntax::{Parse, SyntaxNode};

use crate::{
    AttrMacroAttrIds, AttrMacroAttrIdsRepr, EagerCallInfo, ExpandResult, MacroCallKind,
    MacroCallLoc, span_map::ExpansionSpanMap,
};

/// `heap_size` for the interned `MacroCallId`: the buffers its call kind owns.
///
/// An eager call owns its expanded argument and an attribute macro its
/// arguments, each in a `Box` of its own, plus the list of the attributes it
/// censors when there are several. Both destructurings are exhaustive, so a
/// field upstream adds is a compile error here rather than a silent zero.
/// The eager call's `error` is not counted: an `ExpandError` is an `Arc`
/// shared with whoever observed the same failure (as in
/// [`parse_macro_expansion`]).
pub(crate) fn macro_call_loc((loc,): &(MacroCallLoc,)) -> usize {
    let MacroCallLoc { def: _, krate: _, kind, ctxt: _, macro_depth: _ } = loc;
    match kind {
        MacroCallKind::FnLike { ast_id: _, expand_to: _, eager } => {
            eager.as_deref().map_or(0, |eager| {
                let EagerCallInfo { arg, arg_id: _, error: _, span: _ } = eager;
                size_of::<EagerCallInfo>() + arg.heap_size()
            })
        }
        MacroCallKind::Derive {
            ast_id: _,
            derive_attr_index: _,
            derive_index: _,
            derive_macro_id: _,
        } => 0,
        MacroCallKind::Attr { ast_id: _, attr_args, censored_attr_ids } => {
            attr_args.as_deref().map_or(0, |args| size_of::<tt::TopSubtree>() + args.heap_size())
                + attr_ids(censored_attr_ids)
        }
    }
}

/// `heap_size` for `decl_macro_expander`: the compiled rules of one
/// `macro_rules!` (or `macro`), which `mbe` weighs itself — the tree types are
/// private to it.
pub(crate) fn decl_macro_expander(
    expander: &crate::declarative::DeclarativeMacroExpander,
) -> usize {
    let crate::declarative::DeclarativeMacroExpander { mac, transparency: _, edition: _ } =
        expander;
    mac.heap_size()
}

/// The `ThinVec` behind several censored attributes: its two-word header and
/// the elements, by capacity. A single attribute is inline, and an empty
/// `ThinVec` points at a shared static header and owns nothing.
fn attr_ids(ids: &AttrMacroAttrIds) -> usize {
    match &ids.0 {
        AttrMacroAttrIdsRepr::One(_) => 0,
        AttrMacroAttrIdsRepr::ManyDerives(ids) if ids.capacity() == 0 => 0,
        AttrMacroAttrIdsRepr::ManyDerives(ids) => {
            2 * size_of::<usize>() + ids.capacity() * size_of::<crate::attrs::AttrId>()
        }
    }
}

/// `heap_size` for the `real_span_map` query; see [`span::RealSpanMap::heap_size`].
pub(crate) fn real_span_map(map: &span::RealSpanMap) -> usize {
    map.heap_size()
}

/// `heap_size` for the `parse_macro_expansion` query.
///
/// The `err` half is not counted: an `ExpandError` is an `Arc` shared with
/// whoever else observed the same failure, so charging it here would count one
/// allocation under several ingredients.
pub(crate) fn parse_macro_expansion(
    value: &ExpandResult<(Parse<SyntaxNode>, ExpansionSpanMap)>,
) -> usize {
    let (parse, span_map) = &value.value;
    syntax::heap_size::parse(parse) + span_map_heap(span_map)
}

/// `heap_size` for the `macro_arg` query: the argument's token tree and the
/// fixup undo info; the span is inline.
pub(crate) fn macro_arg(value: &crate::MacroArgResult) -> usize {
    let (tt, undo_info, _span) = value;
    tt.heap_size() + undo_info.heap_size()
}

/// `heap_size` for the `ast_id_map` query; see [`span::AstIdMap::heap_size`].
pub(crate) fn ast_id_map(map: &span::AstIdMap) -> usize {
    map.heap_size()
}

/// The span vector, counted by its length: `SpanMap` keeps the entries private
/// and exposes only an iterator, so spare capacity goes uncounted like every
/// other over-allocation in this family.
fn span_map_heap(span_map: &ExpansionSpanMap) -> usize {
    span_map.iter().count() * size_of::<(TextSize, Span)>()
}
