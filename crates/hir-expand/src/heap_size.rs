//! What a memoized macro expansion costs on the heap.
//!
//! Same contract as [`syntax::heap_size`], and the same reason: the expansion
//! query is LRU-capped, and a capacity is only arguable against bytes. Numbers
//! here are lower bounds — see that module for what the parse half leaves out.

use std::mem::size_of;

use span::{Span, TextSize};
use syntax::{Parse, SyntaxNode};

use crate::{ExpandResult, span_map::ExpansionSpanMap};

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

/// The span vector, counted by its length: `SpanMap` keeps the entries private
/// and exposes only an iterator, so spare capacity goes uncounted like every
/// other over-allocation in this family.
fn span_map_heap(span_map: &ExpansionSpanMap) -> usize {
    span_map.iter().count() * size_of::<(TextSize, Span)>()
}
