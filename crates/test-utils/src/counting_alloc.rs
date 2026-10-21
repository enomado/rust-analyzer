//! A global allocator that counts live bytes, for `heap_size` oracles.
//!
//! The `heap_size` functions compute allocation sizes from layout arithmetic
//! (rowan's `ThinArc`, triomphe's `Arc`), which nothing but a measurement can
//! confirm. An oracle installs this allocator in its own integration-test
//! binary —
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: test_utils::counting_alloc::CountingAllocator =
//!     test_utils::counting_alloc::CountingAllocator;
//! ```
//!
//! — and compares the change in [`live_bytes`] across building a value with
//! what its `heap_size` reports. One binary per oracle, with a single
//! `#[test]`: the counter is process-wide, so a second test on another thread
//! would land in the same total.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

/// Requested bytes currently allocated through [`CountingAllocator`].
static LIVE: AtomicIsize = AtomicIsize::new(0);

/// Requested bytes currently allocated, process-wide. Meaningful only in a
/// binary whose global allocator is [`CountingAllocator`].
pub fn live_bytes() -> isize {
    LIVE.load(Ordering::Relaxed)
}

/// Forwards every call to [`System`] and keeps [`live_bytes`] current.
pub struct CountingAllocator;

// SAFETY: every call is forwarded to `System` unchanged; the counter only
// observes sizes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            LIVE.fetch_add(new_size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        new
    }
}
