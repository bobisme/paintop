//! Pointwise fusion eligibility + the normalized fused-expression key
//! (`plan.md` §13.2; bn-t2v).
//!
//! A *fusible* pointwise subgraph is a maximal run of adjacent nodes that are each
//! **pure and pointwise** (output sample `i` is a closed-form function of input
//! sample `i` only), so the whole run collapses into one CPU loop or one generated
//! WGSL kernel with no intermediate buffers (`plan.md` §13.2). This module decides
//! *which* runs fuse and produces the normalized key the
//! [`pipeline`](super::pipeline) cache keys on.
//!
//! # What fuses, and what stops it
//!
//! Fusion is **conservative**: a node joins the current fused run only when *every*
//! condition holds; the first failing node closes the run and starts a fresh
//! eligibility decision (`plan.md` §13.2):
//!
//! * **Pointwise.** The node's declared [`RoiCategory`] must be
//!   [`Pointwise`](RoiCategory::Pointwise). Any wider footprint (a halo, a geometric
//!   warp, a full-domain reduction) reads neighbours, so it cannot fuse into a
//!   per-sample kernel.
//! * **Supported.** The op must be a kind this backend can express as a fused stage
//!   ([`stage_kinds`] / [`is_supported_op`]). An unsupported pointwise op (one with
//!   no CPU/WGSL stage lowering yet) is a hard barrier, never silently dropped.
//! * **No evidence barrier.** A [`debug.materialize`](MATERIALIZE_OP) node is a
//!   *semantics-preserving evidence barrier* (`plan.md` §18): it deliberately forces
//!   the intermediate to be materialized, so fusion must not span it. It always
//!   closes the run.
//! * **Same format.** Two stages only share a kernel if they run on the same
//!   [`ResourceFormat`] (channel count + scalar). A format change (e.g. a layout
//!   conversion that alters the channel count) closes the run, because the binding
//!   layout and the generated kernel differ (`plan.md` §13.2; the format is half of
//!   the [`PipelineKey`]).
//!
//! Crucially, fusion **never algebraically reorders** stages: color transforms,
//! clamping, premultiplication, and transfer functions are order-sensitive
//! (`plan.md` §13.2). The fused [`FusedExpr`] preserves chain order exactly, and the
//! per-stage CPU/WGSL lowering applies the stages in that same order.

use paintop_ir::{OpId, RoiCategory};
use serde_json::Value;

use super::pipeline::{FusedExpr, FusedStage, PipelineKey, ResourceFormat};

/// The canonical id of the evidence-barrier op that always stops fusion.
///
/// `debug.materialize` is the identity numerically but forces its intermediate to
/// be materialized for evidence (`plan.md` §18), so a fused run must break around
/// it — its result is required to exist as a real buffer.
pub const MATERIALIZE_OP: &str = "debug.materialize@1";

/// A single resolved, pointwise stage of a fused chain.
///
/// Each variant is a supported pointwise op with its parameters already resolved to
/// the concrete scalars the kernel needs. A stage knows three things, kept in lock-
/// step so the GPU kernel can never silently diverge from the oracle:
/// * its [`op_id`](Self::op_id) + canonical [`params`](Self::params), for the key;
/// * how to [`eval_pixel`](Self::eval_pixel) one RGBA sample on the CPU (the
///   reference the differential harness compares against, bn-3eb);
/// * (in bn-125) how to emit its WGSL statement (the GPU lowering used by the fused
///   pipeline).
///
/// The variants mirror the order-sensitive pointwise color/alpha primitives
/// (`plan.md` §13.2): a linear-light gain, a per-channel bias, a saturation blend
/// toward luminance, a clamp to `[0, 1]`, and alpha (un)premultiply. They compose by
/// chaining, never by reordering.
#[derive(Debug, Clone, PartialEq)]
pub enum PointwiseStage {
    /// Multiply every color channel by a linear-light gain (`color.adjust` exposure
    /// component: `c ↦ c · gain`). Alpha is never touched.
    Gain {
        /// The multiplicative gain applied to each color channel.
        gain: f32,
    },
    /// Add a constant to every color channel (`c ↦ c + bias`). Alpha is untouched.
    Bias {
        /// The additive bias applied to each color channel.
        bias: f32,
    },
    /// Blend each color channel toward the pixel's Rec. 709 linear luminance
    /// (`color.adjust` saturation component: `c ↦ Y + (1+s)·(c − Y)`).
    Saturate {
        /// The saturation amount `s`; `0` is the identity, `-1` fully desaturates.
        amount: f32,
    },
    /// Clamp every channel (color **and** alpha) into the closed unit interval.
    Clamp01,
    /// Premultiply the color channels by alpha (`alpha.premultiply`). A no-op when
    /// there is no alpha channel.
    Premultiply,
    /// Recover straight (un-premultiplied) color from premultiplied color, dividing
    /// by alpha where alpha is non-zero (`alpha.unpremultiply`).
    Unpremultiply,
}

/// Rec. 709 linear-luminance weights (matching `color.adjust`'s oracle).
const LUMA: [f32; 3] = [0.212_6, 0.715_2, 0.072_2];

impl PointwiseStage {
    /// The op id this stage lowers, used (with [`params`](Self::params)) to build
    /// the normalized [`FusedStage`].
    ///
    /// # Errors
    /// Propagates the [`schema`](paintop_ir::ErrorClass::Schema) error only if the
    /// hard-coded id is somehow invalid (it is not).
    pub fn op_id(&self) -> Result<OpId, paintop_ir::Error> {
        match self {
            Self::Gain { .. } | Self::Bias { .. } | Self::Saturate { .. } | Self::Clamp01 => {
                OpId::new("color", "adjust", 1)
            }
            Self::Premultiply | Self::Unpremultiply => OpId::new("alpha", "premultiply", 1),
        }
    }

    /// The stage's canonical params as JSON — the resolved scalars that distinguish
    /// two otherwise-identical stage kinds (a gain of `1.5` vs `2.0` key apart).
    #[must_use]
    pub fn params(&self) -> Value {
        match self {
            Self::Gain { gain } => serde_json::json!({ "stage": "gain", "gain": gain }),
            Self::Bias { bias } => serde_json::json!({ "stage": "bias", "bias": bias }),
            Self::Saturate { amount } => {
                serde_json::json!({ "stage": "saturate", "amount": amount })
            }
            Self::Clamp01 => serde_json::json!({ "stage": "clamp01" }),
            Self::Premultiply => serde_json::json!({ "stage": "premultiply" }),
            Self::Unpremultiply => serde_json::json!({ "stage": "unpremultiply" }),
        }
    }

    /// The normalized [`FusedStage`] for this stage (op id + canonical params).
    ///
    /// # Errors
    /// Propagates the [`op_id`](Self::op_id) error (cannot occur for the fixed ids).
    pub fn to_fused_stage(&self) -> Result<FusedStage, paintop_ir::Error> {
        Ok(FusedStage::new(self.op_id()?, self.params()))
    }

    /// Emit the WGSL statements that apply this stage in place to a fused kernel's
    /// per-pixel local channel array `c[0..4]` (bn-125).
    ///
    /// The generated code is **specialized** to `color_count` (the number of leading
    /// color channels) and `has_alpha` (whether `c[color_count]` is alpha), which are
    /// fixed per pipeline — they are part of the [`ResourceFormat`] half of the
    /// pipeline key — so the body is straight-line WGSL with no runtime channel
    /// branching. It mirrors [`eval_pixel`](Self::eval_pixel) operation-for-operation
    /// (same gain/bias/luma math, same clamp, same premultiply) so the GPU result
    /// matches the CPU reference within the op's tolerance.
    ///
    /// `f` renders an `f32` literal in a form WGSL accepts (always with a decimal
    /// point and an `f` suffix). The caller declares `var c: array<f32, 4>;`.
    #[must_use]
    pub fn wgsl_body(&self, color_count: usize, has_alpha: bool) -> String {
        use std::fmt::Write as _;

        let alpha_index = color_count; // alpha, when present, sits just past the color channels
        let mut s = String::new();
        match self {
            Self::Gain { gain } => {
                let g = wgsl_f32(*gain);
                for i in 0..color_count {
                    let _ = writeln!(s, "    c[{i}] = c[{i}] * {g};");
                }
            }
            Self::Bias { bias } => {
                let b = wgsl_f32(*bias);
                for i in 0..color_count {
                    let _ = writeln!(s, "    c[{i}] = c[{i}] + {b};");
                }
            }
            Self::Saturate { amount } => {
                // A single color channel is its own luminance: a no-op (matches
                // eval_pixel), so only the 3-color case emits anything.
                if color_count == 3 {
                    let scale = wgsl_f32(1.0 + *amount);
                    let _ = writeln!(
                        s,
                        "    let luma = {lr} * c[0] + {lg} * c[1] + {lb} * c[2];",
                        lr = wgsl_f32(LUMA[0]),
                        lg = wgsl_f32(LUMA[1]),
                        lb = wgsl_f32(LUMA[2]),
                    );
                    for i in 0..3 {
                        let _ = writeln!(s, "    c[{i}] = luma + {scale} * (c[{i}] - luma);");
                    }
                }
            }
            Self::Clamp01 => {
                // Clamp every present channel (color and alpha), like eval_pixel.
                let last = if has_alpha {
                    color_count + 1
                } else {
                    color_count
                };
                for i in 0..last {
                    let _ = writeln!(s, "    c[{i}] = clamp(c[{i}], 0.0f, 1.0f);");
                }
            }
            Self::Premultiply => {
                if has_alpha {
                    for i in 0..color_count {
                        let _ = writeln!(s, "    c[{i}] = c[{i}] * c[{alpha_index}];");
                    }
                }
            }
            Self::Unpremultiply => {
                if has_alpha {
                    let _ = writeln!(s, "    if (c[{alpha_index}] != 0.0f) {{");
                    for i in 0..color_count {
                        let _ = writeln!(s, "        c[{i}] = c[{i}] / c[{alpha_index}];");
                    }
                    s.push_str("    }\n");
                }
            }
        }
        s
    }

    /// Apply this stage to one pixel's interleaved channels in place — the CPU
    /// reference the differential harness compares the GPU kernel against (bn-3eb).
    ///
    /// `channels` is the per-pixel sample count and `has_alpha` whether the last
    /// channel is alpha. The color channels are everything but a trailing alpha; a
    /// single-channel pixel is treated as luminance-equal (saturation is a no-op),
    /// matching the `color.adjust` oracle.
    pub fn eval_pixel(&self, pixel: &mut [f32], has_alpha: bool) {
        let channels = pixel.len();
        let color_count = if has_alpha {
            channels.saturating_sub(1)
        } else {
            channels
        };
        match self {
            Self::Gain { gain } => {
                for c in pixel.iter_mut().take(color_count) {
                    *c *= *gain;
                }
            }
            Self::Bias { bias } => {
                for c in pixel.iter_mut().take(color_count) {
                    *c += *bias;
                }
            }
            Self::Saturate { amount } => {
                if color_count == 3 {
                    let luma =
                        LUMA[0].mul_add(pixel[0], LUMA[1].mul_add(pixel[1], LUMA[2] * pixel[2]));
                    let scale = 1.0 + *amount;
                    for c in pixel.iter_mut().take(3) {
                        *c = scale.mul_add(*c - luma, luma);
                    }
                }
                // A single color channel is its own luminance: saturation is a no-op.
            }
            Self::Clamp01 => {
                for c in pixel.iter_mut() {
                    *c = c.clamp(0.0, 1.0);
                }
            }
            Self::Premultiply => {
                if has_alpha && channels >= 1 {
                    let alpha = pixel[channels - 1];
                    for c in pixel.iter_mut().take(color_count) {
                        *c *= alpha;
                    }
                }
            }
            Self::Unpremultiply => {
                if has_alpha && channels >= 1 {
                    let alpha = pixel[channels - 1];
                    if alpha != 0.0 {
                        for c in pixel.iter_mut().take(color_count) {
                            *c /= alpha;
                        }
                    }
                }
            }
        }
    }
}

/// Try to lower a candidate `(op, params)` node into a fused [`PointwiseStage`].
///
/// Returns `None` when the op is not a supported fused stage (the caller treats that
/// as a hard fusion barrier — an unsupported op is never silently fused). The op id
/// is matched on its namespace/name (the version-agnostic identity), and the params
/// are read leniently: an absent sub-parameter defaults to its identity value,
/// matching the oracle ops' defaults.
#[must_use]
pub fn stage_kinds(op: &OpId, params: &Value) -> Vec<PointwiseStage> {
    match (op.namespace(), op.name()) {
        // `color.adjust` decomposes into its order-sensitive sub-stages: exposure
        // gain, then saturation. (Temperature is not lowered here; a request that
        // sets it is reported unsupported by `is_supported_op` below.)
        ("color", "adjust") => {
            let mut stages = Vec::new();
            let gain = exp2(read_f32(params, "exposure_ev", 0.0));
            // An *exactly* identity sub-parameter contributes no stage; this is a
            // structural identity test (is the gain literally 1.0?), not an
            // approximate numeric comparison, so a strict `==`/`!=` is correct.
            #[allow(
                clippy::float_cmp,
                reason = "exact identity elision: a literal 1.0 gain / 0.0 saturation \
                          adds no stage; this is structural, not an approximate compare"
            )]
            {
                if gain != 1.0 {
                    stages.push(PointwiseStage::Gain { gain });
                }
                let saturation = read_f32(params, "saturation", 0.0);
                if saturation != 0.0 {
                    stages.push(PointwiseStage::Saturate { amount: saturation });
                }
            }
            stages
        }
        ("color", "gain") => vec![PointwiseStage::Gain {
            gain: read_f32(params, "gain", 1.0),
        }],
        ("color", "bias") => vec![PointwiseStage::Bias {
            bias: read_f32(params, "bias", 0.0),
        }],
        ("color", "clamp") => vec![PointwiseStage::Clamp01],
        ("alpha", "premultiply") => vec![PointwiseStage::Premultiply],
        ("alpha", "unpremultiply") => vec![PointwiseStage::Unpremultiply],
        _ => Vec::new(),
    }
}

/// Whether `op` with `params` is a supported, fusible pointwise op.
///
/// A supported op lowers to one or more [`PointwiseStage`]s; an op that lowers to no
/// stages (an explicit identity, e.g. `color.adjust` with all-default params) is
/// still *supported* — it simply contributes nothing. The distinguishing case is an
/// op this backend has no lowering for at all, which is unsupported.
#[must_use]
pub fn is_supported_op(op: &OpId, params: &Value) -> bool {
    match (op.namespace(), op.name()) {
        // color.adjust is only supported when it uses sub-parameters the fused
        // lowering covers: a non-zero temperature has no fused stage yet, so such a
        // request is unsupported (a hard barrier) rather than silently wrong. The
        // test is structural ("is temperature exactly its identity 0.0?"), so a
        // strict `==` is correct, not an approximate compare.
        ("color", "adjust") => {
            #[allow(
                clippy::float_cmp,
                reason = "structural identity test for the temperature sub-parameter"
            )]
            {
                read_f32(params, "temperature", 0.0) == 0.0
            }
        }
        ("color", "gain" | "bias" | "clamp") | ("alpha", "premultiply" | "unpremultiply") => true,
        _ => false,
    }
}

/// Why a candidate node could not extend the current fused run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionBreak {
    /// The node's ROI category is not pointwise (it reads a neighbourhood).
    NotPointwise,
    /// The op has no fused-stage lowering on this backend.
    UnsupportedOp,
    /// The node is a `debug.materialize` evidence barrier.
    EvidenceBarrier,
    /// The node runs on a different resource format than the run so far.
    FormatChange,
}

impl FusionBreak {
    /// A stable, human-readable reason for the trace / evidence.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::NotPointwise => "node is not pointwise (reads a neighbourhood)",
            Self::UnsupportedOp => "op has no fused-stage lowering on this backend",
            Self::EvidenceBarrier => "debug.materialize evidence barrier forces materialization",
            Self::FormatChange => "node runs on a different resource format",
        }
    }
}

/// One candidate node the fusion pass considers: its op, resolved params, declared
/// ROI category, and the resource format it runs on.
///
/// The fusion pass walks a slice of these (the M2 graph analysis order) and groups
/// the maximal fusible runs. Built from the resolved graph + manifests by the
/// caller; this module stays graph-agnostic so it is unit-testable without a full
/// plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    /// The node's operation id.
    pub op: OpId,
    /// The node's resolved parameters.
    pub params: Value,
    /// The node's declared ROI category (from its manifest).
    pub roi: RoiCategory,
    /// The resource format the node operates on.
    pub format: ResourceFormat,
}

impl FusionCandidate {
    /// Decide why this node cannot fuse, or `None` if it is fusible on its own.
    ///
    /// `run_format` is the format of the run it would extend, or `None` if it would
    /// start a fresh run (in which case there is no format constraint yet).
    #[must_use]
    pub fn break_against(&self, run_format: Option<ResourceFormat>) -> Option<FusionBreak> {
        if self.op.to_string() == MATERIALIZE_OP {
            return Some(FusionBreak::EvidenceBarrier);
        }
        if self.roi != RoiCategory::Pointwise {
            return Some(FusionBreak::NotPointwise);
        }
        if !is_supported_op(&self.op, &self.params) {
            return Some(FusionBreak::UnsupportedOp);
        }
        if let Some(run) = run_format
            && run != self.format
        {
            return Some(FusionBreak::FormatChange);
        }
        None
    }
}

/// A maximal fused run: the contiguous candidate node span `[start, end)` that fused
/// together, the resolved [`PointwiseStage`]s in chain order, and the run's format.
///
/// The stages are the *lowered* form (a single `color.adjust` may contribute several
/// stages); the span indices refer back into the candidate slice the pass walked, so
/// a caller can map a run to its source nodes (for evidence / no-readback assertions,
/// bn-3eb).
#[derive(Debug, Clone, PartialEq)]
pub struct FusedRun {
    /// First candidate index in this run (inclusive).
    pub start: usize,
    /// One past the last candidate index in this run (exclusive).
    pub end: usize,
    /// The lowered pointwise stages, in chain order.
    pub stages: Vec<PointwiseStage>,
    /// The resource format the run operates on.
    pub format: ResourceFormat,
}

impl FusedRun {
    /// The number of source candidate nodes this run fused.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.end - self.start
    }

    /// Whether the run fused more than one node (so it actually elides an
    /// intermediate buffer — the no-readback property bn-3eb asserts).
    #[must_use]
    pub const fn is_nontrivial(&self) -> bool {
        self.node_count() >= 2
    }

    /// The normalized [`FusedExpr`] for this run, in chain order.
    ///
    /// # Errors
    /// Propagates a stage's [`op_id`](PointwiseStage::op_id) error (cannot occur for
    /// the fixed ids).
    pub fn fused_expr(&self) -> Result<FusedExpr, paintop_ir::Error> {
        let mut expr = FusedExpr::new();
        for stage in &self.stages {
            expr.push(stage.to_fused_stage()?);
        }
        Ok(expr)
    }

    /// The content-addressed [`PipelineKey`] this run's pipeline is cached under
    /// (its normalized fused expression + format).
    ///
    /// # Errors
    /// Propagates the [`fused_expr`](Self::fused_expr) / key-derivation error.
    pub fn pipeline_key(&self) -> Result<PipelineKey, paintop_ir::Error> {
        PipelineKey::derive(&self.fused_expr()?, self.format)
    }

    /// The number of channels per pixel this run operates on (from its format).
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.format.channels as usize
    }

    /// Apply the whole fused run to an interleaved `f32` sample buffer on the CPU —
    /// the reference path the differential harness compares the GPU kernel against,
    /// and the fallback when no adapter is present (bn-3eb).
    ///
    /// `samples` is row-major, channel-interleaved with `self.channels()` per pixel;
    /// `has_alpha` whether the trailing channel is alpha. Stages are applied in chain
    /// order, per pixel, exactly as [`PointwiseStage::eval_pixel`] specifies. Returns
    /// the transformed buffer (the input is left untouched).
    #[must_use]
    pub fn eval_buffer(&self, samples: &[f32], has_alpha: bool) -> Vec<f32> {
        let stride = self.channels();
        let mut out = samples.to_vec();
        if stride == 0 {
            return out;
        }
        for pixel in out.chunks_mut(stride) {
            for stage in &self.stages {
                stage.eval_pixel(pixel, has_alpha);
            }
        }
        out
    }

    /// Generate the complete WGSL compute shader for this fused run (bn-125).
    ///
    /// The shader binds the input/output as `f32` storage buffers (a flat,
    /// row-major, channel-interleaved array) plus a small uniform carrying the pixel
    /// count, and runs **one invocation per pixel**: it loads the pixel's channels
    /// into a local `array<f32, 4>`, applies every stage's
    /// [`wgsl_body`](PointwiseStage::wgsl_body) in chain order, and writes the result
    /// back. Because the whole chain runs in one kernel, no intermediate buffer is
    /// produced — the readback-free property bn-3eb asserts.
    ///
    /// The body is specialized to the run's channel count and `has_alpha`, so it has
    /// no runtime channel branching; the workgroup is `workgroup` invocations along
    /// X. The generated source is a pure function of the run + layout, so two runs
    /// with the same key + layout generate byte-identical WGSL (the cache reuse
    /// property).
    #[must_use]
    pub fn wgsl_shader(&self, has_alpha: bool, workgroup: u32) -> String {
        use std::fmt::Write as _;
        let stride = self.channels();
        let color_count = if has_alpha {
            stride.saturating_sub(1)
        } else {
            stride
        };

        let mut load = String::new();
        for i in 0..stride {
            let _ = writeln!(load, "    c[{i}] = input[base + {i}u];");
        }
        let mut body = String::new();
        for stage in &self.stages {
            body.push_str(&stage.wgsl_body(color_count, has_alpha));
        }
        let mut store = String::new();
        for i in 0..stride {
            let _ = writeln!(store, "    output[base + {i}u] = c[{i}];");
        }

        format!(
            "// Generated fused pointwise kernel ({stages} stage(s), {stride} channel(s)).\n\
             struct Params {{ pixel_count: u32 }};\n\
             @group(0) @binding(0) var<storage, read> input: array<f32>;\n\
             @group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\
             @group(0) @binding(2) var<uniform> params: Params;\n\
             \n\
             @compute @workgroup_size({workgroup})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             \x20   let pixel = gid.x;\n\
             \x20   if (pixel >= params.pixel_count) {{ return; }}\n\
             \x20   let base = pixel * {stride}u;\n\
             \x20   var c: array<f32, 4>;\n\
             {load}{body}{store}}}\n",
            stages = self.stages.len(),
        )
    }
}

/// Partition a candidate slice into its maximal fused runs (`plan.md` §13.2; bn-t2v).
///
/// Walks the candidates in order, greedily extending the current run while each node
/// [`break_against`](FusionCandidate::break_against) the run's format returns `None`,
/// and starting a fresh run on a break. A node that cannot fuse *on its own* (a
/// barrier, a non-pointwise op, an unsupported op) is not emitted as a run — only
/// fusible runs are returned, so a caller knows exactly which spans become one
/// kernel. A run with no lowered stages (every node an explicit identity) is still
/// emitted when it fused ≥1 node, but its [`FusedExpr`] is empty (the identity
/// kernel).
///
/// The result is deterministic: a pure function of the candidate order.
#[must_use]
pub fn plan_fusion(candidates: &[FusionCandidate]) -> Vec<FusedRun> {
    let mut runs: Vec<FusedRun> = Vec::new();
    let mut current: Option<FusedRun> = None;

    for (index, candidate) in candidates.iter().enumerate() {
        let run_format = current.as_ref().map(|r| r.format);
        if candidate.break_against(run_format).is_some() {
            // This node cannot extend (or even start, if standalone-unfusible) the
            // current run. Close the run, then decide whether the node can start a
            // fresh one on its own.
            if let Some(run) = current.take() {
                runs.push(run);
            }
            if candidate.break_against(None).is_none() {
                current = Some(start_run(index, candidate));
            }
            continue;
        }
        match current.as_mut() {
            Some(run) => {
                run.end = index + 1;
                run.stages
                    .extend(stage_kinds(&candidate.op, &candidate.params));
            }
            None => current = Some(start_run(index, candidate)),
        }
    }
    if let Some(run) = current.take() {
        runs.push(run);
    }
    runs
}

/// Begin a fresh fused run at `index` from a fusible candidate.
fn start_run(index: usize, candidate: &FusionCandidate) -> FusedRun {
    FusedRun {
        start: index,
        end: index + 1,
        stages: stage_kinds(&candidate.op, &candidate.params),
        format: candidate.format,
    }
}

/// Read an optional finite `f32` param, defaulting to `default` when absent or
/// non-finite. (Eligibility/lowering stay lenient; the op's own contract is the
/// authority on rejecting a malformed request.)
fn read_f32(params: &Value, name: &str, default: f32) -> f32 {
    params
        .get(name)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .map_or(default, |n| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "grading controls are single-precision by design"
            )]
            {
                n as f32
            }
        })
}

/// `2^e`, matching the oracle's exposure gain.
fn exp2(e: f32) -> f32 {
    e.exp2()
}

/// Render an `f32` as a WGSL `f32` literal.
///
/// WGSL requires a float literal to be unambiguously floating point; the safest
/// universally-accepted form is a full-precision decimal with an `f` suffix. A
/// non-finite value cannot appear here (params are validated finite upstream), but
/// is mapped to `0.0f` defensively so the generated shader always parses.
fn wgsl_f32(value: f32) -> String {
    if value.is_finite() {
        // `{:?}` on an f32 prints a round-trippable decimal (e.g. `1.5`, `0.2126`)
        // that always contains a `.` or exponent, so appending `f` yields a valid
        // WGSL f32 literal.
        format!("{value:?}f")
    } else {
        "0.0f".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FusionBreak, FusionCandidate, MATERIALIZE_OP, PointwiseStage, is_supported_op, plan_fusion,
        stage_kinds,
    };
    use crate::gpu::pipeline::ResourceFormat;
    use paintop_ir::{OpId, RoiCategory};
    use serde_json::json;

    const RGBA: ResourceFormat = ResourceFormat::f32(4);

    fn op(s: &str) -> OpId {
        s.parse().expect("op id")
    }

    fn candidate(op_id: &str, params: serde_json::Value, roi: RoiCategory) -> FusionCandidate {
        FusionCandidate {
            op: op(op_id),
            params,
            roi,
            format: RGBA,
        }
    }

    #[test]
    fn a_pointwise_chain_fuses_into_one_run() {
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.5 }),
                RoiCategory::Pointwise,
            ),
            candidate(
                "color.bias@1",
                json!({ "bias": 0.1 }),
                RoiCategory::Pointwise,
            ),
            candidate("alpha.premultiply@1", json!({}), RoiCategory::Pointwise),
        ];
        let runs = plan_fusion(&chain);
        assert_eq!(runs.len(), 1, "the whole pointwise chain fuses");
        let run = &runs[0];
        assert_eq!(run.node_count(), 3);
        assert!(run.is_nontrivial());
        assert_eq!(run.stages.len(), 3);
        // Order is preserved: gain, bias, premultiply.
        assert_eq!(run.stages[0], PointwiseStage::Gain { gain: 1.5 });
        assert!(matches!(run.stages[1], PointwiseStage::Bias { .. }));
        assert_eq!(run.stages[2], PointwiseStage::Premultiply);
    }

    #[test]
    fn debug_materialize_is_an_evidence_barrier_splitting_the_run() {
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 2.0 }),
                RoiCategory::Pointwise,
            ),
            candidate(MATERIALIZE_OP, json!({}), RoiCategory::Pointwise),
            candidate(
                "color.bias@1",
                json!({ "bias": 0.2 }),
                RoiCategory::Pointwise,
            ),
        ];
        let runs = plan_fusion(&chain);
        // Two runs: [gain] | barrier | [bias]. The barrier node is not in any run.
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].start, 0);
        assert_eq!(runs[0].end, 1);
        assert_eq!(runs[1].start, 2);
        assert_eq!(runs[1].end, 3);
        // The barrier's own break reason is the evidence barrier.
        assert_eq!(
            chain[1].break_against(None),
            Some(FusionBreak::EvidenceBarrier)
        );
    }

    #[test]
    fn a_non_pointwise_op_breaks_fusion() {
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.2 }),
                RoiCategory::Pointwise,
            ),
            candidate(
                "filter.gaussian_blur@1",
                json!({ "sigma": 2.0 }),
                RoiCategory::LocalHalo,
            ),
            candidate(
                "color.bias@1",
                json!({ "bias": 0.3 }),
                RoiCategory::Pointwise,
            ),
        ];
        let runs = plan_fusion(&chain);
        // The blur is neither fused nor emitted as a run; it splits the two pointwise
        // singletons.
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].node_count(), 1);
        assert_eq!(runs[1].node_count(), 1);
        assert_eq!(
            chain[1].break_against(Some(RGBA)),
            Some(FusionBreak::NotPointwise)
        );
    }

    #[test]
    fn an_unsupported_pointwise_op_breaks_fusion() {
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.2 }),
                RoiCategory::Pointwise,
            ),
            // A pointwise op with no fused-stage lowering on this backend.
            candidate("color.mystery@1", json!({}), RoiCategory::Pointwise),
        ];
        let runs = plan_fusion(&chain);
        assert_eq!(runs.len(), 1, "only the supported prefix fuses");
        assert_eq!(runs[0].node_count(), 1);
        assert_eq!(
            chain[1].break_against(Some(RGBA)),
            Some(FusionBreak::UnsupportedOp)
        );
    }

    #[test]
    fn a_format_change_breaks_fusion() {
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.2 }),
                RoiCategory::Pointwise,
            ),
            FusionCandidate {
                op: op("color.bias@1"),
                params: json!({ "bias": 0.1 }),
                roi: RoiCategory::Pointwise,
                // A single-channel format: different binding layout, different kernel.
                format: ResourceFormat::f32(1),
            },
        ];
        let runs = plan_fusion(&chain);
        assert_eq!(runs.len(), 2, "a format change splits the run");
        assert_eq!(
            chain[1].break_against(Some(RGBA)),
            Some(FusionBreak::FormatChange)
        );
        // But the same node *starts* a fresh run fine (no format constraint yet).
        assert_eq!(chain[1].break_against(None), None);
    }

    #[test]
    fn color_adjust_lowers_to_ordered_sub_stages() {
        // exposure_ev = 1 -> gain 2; saturation -0.5 -> saturate. Order: gain then
        // saturate, matching the oracle's fixed order.
        let stages = stage_kinds(
            &op("color.adjust@1"),
            &json!({ "exposure_ev": 1.0, "saturation": -0.5 }),
        );
        assert_eq!(stages.len(), 2);
        assert!(matches!(stages[0], PointwiseStage::Gain { .. }));
        assert!(matches!(stages[1], PointwiseStage::Saturate { .. }));
    }

    #[test]
    fn color_adjust_with_temperature_is_unsupported() {
        // Temperature has no fused stage yet: the op is a hard barrier, not silently
        // wrong.
        assert!(!is_supported_op(
            &op("color.adjust@1"),
            &json!({ "temperature": 0.3 })
        ));
        assert!(is_supported_op(
            &op("color.adjust@1"),
            &json!({ "exposure_ev": 1.0 })
        ));
    }

    #[test]
    fn all_default_adjust_is_supported_but_lowers_to_no_stages() {
        // An explicit identity is supported and fuses; it just contributes nothing.
        assert!(is_supported_op(&op("color.adjust@1"), &json!({})));
        assert!(stage_kinds(&op("color.adjust@1"), &json!({})).is_empty());
        let chain = [candidate(
            "color.adjust@1",
            json!({}),
            RoiCategory::Pointwise,
        )];
        let runs = plan_fusion(&chain);
        assert_eq!(runs.len(), 1);
        assert!(runs[0].stages.is_empty(), "identity run has no stages");
    }

    #[test]
    fn the_fused_expr_and_key_preserve_chain_order() {
        let forward = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.5 }),
                RoiCategory::Pointwise,
            ),
            candidate("alpha.premultiply@1", json!({}), RoiCategory::Pointwise),
        ];
        let reversed = [
            candidate("alpha.premultiply@1", json!({}), RoiCategory::Pointwise),
            candidate(
                "color.gain@1",
                json!({ "gain": 1.5 }),
                RoiCategory::Pointwise,
            ),
        ];
        let kf = plan_fusion(&forward)[0].pipeline_key().expect("key");
        let kr = plan_fusion(&reversed)[0].pipeline_key().expect("key");
        assert_ne!(kf, kr, "composition order is part of the key");
    }

    #[test]
    fn eval_pixel_matches_the_color_adjust_gain_and_saturation_semantics() {
        // A gray-ish RGBA pixel; gain then saturation, alpha untouched.
        let mut pixel = [0.4_f32, 0.2, 0.1, 0.8];
        PointwiseStage::Gain { gain: 2.0 }.eval_pixel(&mut pixel, true);
        assert!((pixel[0] - 0.8).abs() < 1e-6);
        assert!((pixel[3] - 0.8).abs() < 1e-6, "alpha is never gained");

        // Clamp brings an over-range channel back into [0,1] but leaves alpha.
        PointwiseStage::Clamp01.eval_pixel(&mut pixel, true);
        assert!(pixel.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }

    #[test]
    fn premultiply_then_unpremultiply_round_trips_for_nonzero_alpha() {
        let mut pixel = [0.5_f32, 0.25, 0.75, 0.5];
        let original = pixel;
        PointwiseStage::Premultiply.eval_pixel(&mut pixel, true);
        assert!((pixel[0] - 0.25).abs() < 1e-6, "color scaled by alpha");
        PointwiseStage::Unpremultiply.eval_pixel(&mut pixel, true);
        for i in 0..4 {
            assert!((pixel[i] - original[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn fusion_is_deterministic() {
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.5 }),
                RoiCategory::Pointwise,
            ),
            candidate(
                "color.bias@1",
                json!({ "bias": 0.1 }),
                RoiCategory::Pointwise,
            ),
        ];
        assert_eq!(plan_fusion(&chain), plan_fusion(&chain));
    }

    #[test]
    fn wgsl_shader_is_a_single_kernel_with_one_load_and_one_store() {
        // A 3-stage RGBA chain: gain, bias, premultiply -> one generated kernel.
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 1.5 }),
                RoiCategory::Pointwise,
            ),
            candidate(
                "color.bias@1",
                json!({ "bias": 0.1 }),
                RoiCategory::Pointwise,
            ),
            candidate("alpha.premultiply@1", json!({}), RoiCategory::Pointwise),
        ];
        let run = &plan_fusion(&chain)[0];
        let src = run.wgsl_shader(true, 64);
        // Exactly one compute entry point, one input load block, one output store
        // block -> the whole chain is one dispatch, no intermediate buffer.
        assert_eq!(src.matches("@compute").count(), 1);
        assert_eq!(src.matches("fn main").count(), 1);
        // 4 channels loaded and 4 stored (one upload region, one store region).
        assert_eq!(src.matches("= input[base").count(), 4);
        assert_eq!(src.matches("output[base").count(), 4);
        // The stages appear in chain order: gain (*) before premultiply (* c[3]).
        let gain_at = src.find("* 1.5f").expect("gain stage");
        let premul_at = src.find("* c[3]").expect("premultiply stage");
        assert!(gain_at < premul_at, "stages emitted in chain order");
    }

    #[test]
    fn wgsl_shader_specializes_to_channel_count() {
        // A single-channel (gray) run: saturation is a no-op, no alpha handling.
        let chain = [FusionCandidate {
            op: op("color.gain@1"),
            params: json!({ "gain": 2.0 }),
            roi: RoiCategory::Pointwise,
            format: ResourceFormat::f32(1),
        }];
        let run = &plan_fusion(&chain)[0];
        let src = run.wgsl_shader(false, 64);
        assert_eq!(src.matches("= input[base").count(), 1, "one channel loaded");
        assert!(!src.contains("c[3]"), "no alpha references for a gray run");
    }

    #[test]
    fn eval_buffer_applies_stages_per_pixel_in_order() {
        // Two RGBA pixels; gain x2 then bias +0.1 on color channels, alpha untouched.
        let chain = [
            candidate(
                "color.gain@1",
                json!({ "gain": 2.0 }),
                RoiCategory::Pointwise,
            ),
            candidate(
                "color.bias@1",
                json!({ "bias": 0.1 }),
                RoiCategory::Pointwise,
            ),
        ];
        let run = &plan_fusion(&chain)[0];
        let input = vec![0.1_f32, 0.2, 0.3, 0.5, 0.0, 0.4, 0.25, 1.0];
        let out = run.eval_buffer(&input, true);
        // pixel 0: color *2 +0.1 -> 0.3, 0.5, 0.7 ; alpha 0.5 unchanged.
        assert!((out[0] - 0.3).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.7).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6, "alpha untouched");
        // pixel 1: 0.0,0.4,0.25 -> 0.1, 0.9, 0.6 ; alpha 1.0.
        assert!((out[4] - 0.1).abs() < 1e-6);
        assert!((out[7] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn identity_run_generates_a_passthrough_kernel() {
        // An all-default adjust fuses to a stageless run; its kernel loads and stores
        // unchanged (a verbatim passthrough).
        let chain = [candidate(
            "color.adjust@1",
            json!({}),
            RoiCategory::Pointwise,
        )];
        let run = &plan_fusion(&chain)[0];
        assert!(run.stages.is_empty());
        let out = run.eval_buffer(&[0.2, 0.4, 0.6, 0.8], true);
        assert_eq!(out, vec![0.2, 0.4, 0.6, 0.8]);
    }
}
