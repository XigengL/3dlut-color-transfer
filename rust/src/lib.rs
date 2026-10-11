//! lutfit — fit a 3D LUT from paired before/after photographs.
//!
//! See `DESIGN.md` for the method and the error analysis.

// Several hot loops index two or three parallel arrays at once (node values, weights,
// per-channel accumulators).  Clippy's suggested iterator rewrite is measurably less
// readable in those spots and does not change codegen, so they are allowed here with
// the rest of the lints left on.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
// A few lints are deliberately not followed:
//  * `field_reassign_with_default` — `LutApp` has ~20 fields; the screenshot harness
//    sets two of them after construction, which reads better than a 20-line literal.
//  * `type_complexity` — the evaluation pass returns `Option<(PairDe, [f64; 4])>`; a
//    named alias would be longer than the thing it names.
//  * `chunks_exact` — the .npy reader walks 8-byte records.
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::type_complexity)]
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod app;
pub mod core;
pub mod theme;
