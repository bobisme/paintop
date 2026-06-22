//! The `repair.poisson_blend@1` operation: **gradient-domain seamless cloning**
//! (`OP_CATALOG` §12; `plan.md` §1444 — M4 classical magic, Poisson editing).
//!
//! Poisson blending pastes a `source` region into a `target` image so that the
//! seam is invisible: inside the `mask` the result reproduces the *gradients* of
//! the source rather than its absolute colours, while matching the target colour
//! exactly on the mask boundary. The reconstructed field `u` solves the Poisson
//! problem
//!
//! ```text
//! Δu = Δ(source)   inside the mask    (the result's gradient field = source's)
//! u  = target      on the mask edge   (Dirichlet boundary continuity)
//! ```
//!
//! per colour channel, via the shared [`crate::poisson`] Gauss-Seidel solver. The
//! output `candidate` equals the target wherever the mask does not select, and is
//! the seamlessly-blended source inside it.
//!
//! # Gradient-domain fixtures (M4 acceptance)
//!
//! Two analytic guarantees the test suite checks:
//!
//! - **gradients match inside the mask**: where the source has a constant
//!   gradient (e.g. a linear ramp), the blended interior reproduces that gradient
//!   up to a global offset fixing the boundary — so a uniform-gradient source over
//!   a uniform target reconstructs the source-shifted-to-the-seam exactly;
//! - **boundary continuity**: at the mask edge the result equals the target, so
//!   there is no visible seam (the defining property of seamless cloning).
//!
//! # Determinism + convergence metrics (M4 exit criteria 1 & 2)
//!
//! The solver runs fixed row-major Gauss-Seidel sweeps with no RNG, so a rerun is
//! bit-identical on a fixed backend. The `report` carries a
//! [`SolverData`] per the worst-converging channel:
//! iteration count, relative-residual history, stop reason, target tolerance, and
//! final residual.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, MaskDescriptor, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect,
    Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    SolverData, TestMetadata,
};

use crate::poisson::{
    Cell, PoissonSystem, SolveControls, SolveReport, classify_cells, guidance_laplacian, solve,
};

/// The canonical id of the Poisson-blend operation.
pub const POISSON_BLEND_OP_ID: &str = "repair.poisson_blend@1";

/// A required input port (`source`, `target`, or `mask`) was absent or carried
/// the wrong resource kind.
pub const E_POISSON_BLEND_INPUT: &str = "E_POISSON_BLEND_INPUT";

/// The three ports disagree on extent or channel layout.
pub const E_POISSON_BLEND_SHAPE: &str = "E_POISSON_BLEND_SHAPE";

/// A solver-control parameter was the wrong type or out of range.
pub const E_POISSON_BLEND_PARAM: &str = "E_POISSON_BLEND_PARAM";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_POISSON_BLEND_BUFFER: &str = "E_POISSON_BLEND_BUFFER";

/// The default maximum number of Gauss-Seidel sweeps.
pub const DEFAULT_MAX_ITERATIONS: u32 = 2_000;

/// The hard cap on requested iterations, bounding the work per op.
pub const MAX_ITERATIONS_LIMIT: u32 = 100_000;

/// The default relative-residual convergence tolerance.
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// The fixed successive-over-relaxation factor (`< 2` for stability); accelerates
/// the smooth Poisson problem while staying deterministic.
pub const OMEGA: f64 = 1.8;

/// The resolved solver controls plus (for the screened variant, reused here with
/// `lambda = 0`) the data weight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendControls {
    pub(crate) max_iterations: u32,
    pub(crate) tolerance: f64,
    pub(crate) lambda: f64,
}

impl BlendControls {
    /// The [`SolveControls`] this resolves to.
    pub(crate) const fn solve_controls(self) -> SolveControls {
        SolveControls {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            omega: OMEGA,
        }
    }
}

/// Resolve the optional `max_iterations` and `tolerance` solver-control params
/// for the pure-Poisson blend (`lambda = 0`).
fn resolve_controls(params: &serde_json::Value, op: &str) -> Result<BlendControls> {
    let max_iterations = match params.get("max_iterations") {
        None => DEFAULT_MAX_ITERATIONS,
        Some(value) => {
            let n = value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    param_error(op, "max_iterations must be a positive integer", value)
                })?;
            if n == 0 || n > MAX_ITERATIONS_LIMIT {
                return Err(param_error(
                    op,
                    &format!("max_iterations must be in 1..={MAX_ITERATIONS_LIMIT}"),
                    value,
                ));
            }
            n
        }
    };
    let tolerance = match params.get("tolerance") {
        None => DEFAULT_TOLERANCE,
        Some(value) => {
            let t = value
                .as_f64()
                .ok_or_else(|| param_error(op, "tolerance must be a number", value))?;
            if !t.is_finite() || t <= 0.0 || t >= 1.0 {
                return Err(param_error(op, "tolerance must be in (0, 1)", value));
            }
            t
        }
    };
    Ok(BlendControls {
        max_iterations,
        tolerance,
        lambda: 0.0,
    })
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(op: &str, detail: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_POISSON_BLEND_PARAM,
        format!("{op}: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// One channel's deinterleaved samples as `f64` (the solver's working type).
fn channel_f64(samples: &[f32], channels: usize, channel: usize, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    for pixel in 0..count {
        out.push(f64::from(samples[pixel * channels + channel]));
    }
    out
}

/// Run a gradient-domain blend for every channel and return the interleaved
/// blended samples plus the worst-converging channel's solver report.
///
/// For each channel the guidance is `Δ(source)` at every interior pixel, the
/// anchor is the `target` everywhere, and the cell classification (shared across
/// channels) comes from the mask. `lambda` selects pure Poisson (`0`) or the
/// screened variant (`> 0`).
pub(crate) fn blend_channels(
    source: &[f32],
    target: &[f32],
    mask: &[f32],
    channels: usize,
    extent: Extent,
    controls: BlendControls,
) -> (Vec<f32>, SolveReport) {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let count = width * height;
    let cells = classify_cells(mask, width, height);

    let mut out = vec![0.0_f32; count * channels];
    // Track the report of the channel that converged worst (most iterations,
    // then largest final residual) so the op's single report is conservative.
    let mut worst: Option<SolveReport> = None;

    for channel in 0..channels {
        let src = channel_f64(source, channels, channel, count);
        let anchor = channel_f64(target, channels, channel, count);

        // Guidance: the source Laplacian at every interior pixel (0 elsewhere).
        let mut rhs = vec![0.0_f64; count];
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                if cells[idx] == Cell::Interior {
                    rhs[idx] = guidance_laplacian(&src, col, row, width, height);
                }
            }
        }

        let system = PoissonSystem {
            width,
            height,
            cells: &cells,
            rhs: &rhs,
            anchor: &anchor,
            lambda: controls.lambda,
        };
        let (field, report) = solve(&system, controls.solve_controls());

        for pixel in 0..count {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the channel solved in f64, stored once as the op's f32 sample"
            )]
            {
                out[pixel * channels + channel] = field[pixel] as f32;
            }
        }

        worst = Some(match worst {
            None => report,
            Some(prev) => worse_of(prev, report),
        });
    }

    let report = worst.unwrap_or_else(|| SolveReport {
        residual_history: Vec::new(),
        iterations: 0,
        converged: true,
        stop_reason: paintop_ir::SolverStopReason::Converged,
        tolerance: controls.tolerance,
        final_residual: 0.0,
    });
    (out, report)
}

/// Pick the worse-converging of two channel reports (more iterations, then
/// larger final residual).
fn worse_of(a: SolveReport, b: SolveReport) -> SolveReport {
    if (b.iterations, b.final_residual) > (a.iterations, a.final_residual) {
        b
    } else {
        a
    }
}

/// Convert a solver [`SolveReport`] into the report's [`SolverData`] payload.
pub(crate) fn solver_data(kind: &str, report: &SolveReport) -> SolverData {
    SolverData {
        kind: kind.to_owned(),
        // `steps` mirrors the iteration count for an iterative solver.
        steps: report.iterations,
        stability_number: 0.0,
        stability_limit: 0.0,
        // Gauss-Seidel on this SPD system is unconditionally stable.
        stable: true,
        residual_history: report.residual_history.clone(),
        total_energy: 0.0,
        iterations: Some(report.iterations),
        stop_reason: Some(report.stop_reason),
        converged: Some(report.converged),
        tolerance: Some(report.tolerance),
        final_residual: Some(report.final_residual),
    }
}

/// The `repair.poisson_blend@1` operation: `source` + `target` + `mask` → the
/// seamlessly-cloned `candidate` plus a solver `report`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonBlend;

impl PoissonBlend {
    /// Construct the Poisson-blend operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `repair.poisson_blend@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: POISSON_BLEND_OP_ID.parse()?,
            impl_version: 1,
            summary: "Gradient-domain seamless cloning: paste a source region into a target image \
                      so the masked interior reproduces the source's gradients (Δu = Δsource) while \
                      matching the target on the mask boundary (Dirichlet continuity), solved per \
                      channel by row-major Gauss-Seidel. Deterministic (bit-identical reruns); the \
                      report carries SolverData (iterations, residual history, stop reason)."
                .to_owned(),
            // Solved in f64 and rounded once to the f32 sample: bounded, not
            // bit-exact across platforms.
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                // The masked region's solution couples across the whole interior,
                // so the honest footprint is the full domain.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "source".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The image whose gradients are cloned inside the mask.".to_owned(),
                },
                InputSpec {
                    name: "target".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The image the source is blended into; kept outside the mask and used as \
                          the Dirichlet boundary."
                        .to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc: "The coverage mask selecting the interior region to reconstruct (> 0.5 is \
                          interior)."
                        .to_owned(),
                },
            ],
            outputs: vec![
                OutputSpec {
                    name: "candidate".to_owned(),
                    kind: ResourceKind::Image,
                    doc: "The seamlessly-blended image (the target descriptor; same extent/layout)."
                        .to_owned(),
                },
                OutputSpec {
                    name: "report".to_owned(),
                    kind: ResourceKind::Report,
                    doc: "The solver report carrying SolverData (iterations, residual history, \
                          stop reason, tolerance, final residual)."
                        .to_owned(),
                },
            ],
            params: params_spec(),
            implementations: vec![reference_impl()?],
            test: blend_test_metadata(),
        })
    }
}

/// The declared solver-control parameter list.
fn params_spec() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            name: "max_iterations".to_owned(),
            ty: ParamType::Integer,
            unit: None,
            required: false,
            default: Some(serde_json::json!(DEFAULT_MAX_ITERATIONS)),
            choices: vec![],
            doc: format!(
                "The maximum Gauss-Seidel sweeps before the solver stops at the cap, in \
                 1..={MAX_ITERATIONS_LIMIT}."
            ),
        },
        ParamSpec {
            name: "tolerance".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(DEFAULT_TOLERANCE)),
            choices: vec![],
            doc: "The relative-residual convergence tolerance in (0, 1).".to_owned(),
        },
    ]
}

/// Read a required image port's descriptor.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_POISSON_BLEND_INPUT,
            format!("repair.poisson_blend requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_POISSON_BLEND_INPUT,
            format!("repair.poisson_blend `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read the `mask` port's descriptor.
fn mask_descriptor(inputs: &Descriptors) -> Result<&MaskDescriptor> {
    let resource = inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_POISSON_BLEND_INPUT,
            "repair.poisson_blend requires a `mask` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Mask(mask) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_POISSON_BLEND_INPUT,
            "repair.poisson_blend `mask` input must be a mask resource".to_owned(),
        ));
    };
    Ok(mask)
}

/// Validate the three ports share an extent and the images share a layout,
/// returning the output (target) descriptor.
pub(crate) fn check_shapes(
    source: &ImageDescriptor,
    target: &ImageDescriptor,
    mask_extent: Extent,
    op: &str,
) -> Result<ImageDescriptor> {
    if source.extent != target.extent {
        return Err(shape_error(
            op,
            "the source and target images must share an extent",
            format!("source {:?} vs target {:?}", source.extent, target.extent),
        ));
    }
    if source.layout != target.layout {
        return Err(shape_error(
            op,
            "the source and target images must share a channel layout",
            format!("source {:?} vs target {:?}", source.layout, target.layout),
        ));
    }
    if mask_extent != target.extent {
        return Err(shape_error(
            op,
            "the mask must share the images' extent",
            format!("mask {mask_extent:?} vs image {:?}", target.extent),
        ));
    }
    Ok(*target)
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_error(op: &str, detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_POISSON_BLEND_SHAPE,
        format!("{op}: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

impl OpContract for PoissonBlend {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("source".to_owned(), ResourceKind::Image),
            ("target".to_owned(), ResourceKind::Image),
            ("mask".to_owned(), ResourceKind::Mask),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("candidate".to_owned(), ResourceKind::Image),
            ("report".to_owned(), ResourceKind::Report),
        ]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        resolve_controls(params, "repair.poisson_blend")?;
        let source = image_descriptor(inputs, "source")?;
        let target = image_descriptor(inputs, "target")?;
        let mask = mask_descriptor(inputs)?;
        let out_descriptor = check_shapes(source, target, mask.extent, "repair.poisson_blend")?;

        let mut out = OutputDescriptors::new();
        out.insert(
            "candidate".to_owned(),
            ResourceDescriptor::Image(out_descriptor),
        );
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent: out_descriptor.extent,
                channels: out_descriptor.layout.channel_count(),
            }),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // FullDomain: the masked solve couples the whole interior, so every port
        // is read in full.
        let mut regions = InputRegions::new();
        for port in ["source", "target", "mask"] {
            if let Some(resource) = inputs.get(port) {
                regions.insert(port.to_owned(), Rect::from_extent(resource.extent()));
            }
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let image = matches!(outputs.get("candidate"), Some(ResourceDescriptor::Image(_)));
        let report = matches!(outputs.get("report"), Some(ResourceDescriptor::Report(_)));
        Ok(vec![if image && report {
            AssertionResult::pass("produces_candidate_and_report")
        } else {
            AssertionResult::fail(
                "produces_candidate_and_report",
                "expected a `candidate` image and a `report`",
            )
        }])
    }
}

impl OpImplementation for PoissonBlend {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let controls = resolve_controls(params, "repair.poisson_blend")?;
        let source = input_value(inputs, "source")?;
        let target = input_value(inputs, "target")?;
        let mask = input_value(inputs, "mask")?;

        let ResourceDescriptor::Image(source_descriptor) = source.descriptor() else {
            return Err(input_type_error("source", "image"));
        };
        let ResourceDescriptor::Image(target_descriptor) = target.descriptor() else {
            return Err(input_type_error("target", "image"));
        };
        let ResourceDescriptor::Mask(mask_descriptor) = mask.descriptor() else {
            return Err(input_type_error("mask", "mask"));
        };

        let out_descriptor = check_shapes(
            source_descriptor,
            target_descriptor,
            mask_descriptor.extent,
            "repair.poisson_blend",
        )?;
        let channels = out_descriptor.layout.channel_count() as usize;
        let extent = out_descriptor.extent;

        let (samples, report) = blend_channels(
            source.samples(),
            target.samples(),
            mask.samples(),
            channels,
            extent,
            controls,
        );

        let all_finite = samples.iter().all(|s| s.is_finite());
        let candidate = ResourceValue::new(
            ResourceDescriptor::Image(out_descriptor),
            out_descriptor.layout.channel_count(),
            samples,
        )
        .map_err(buffer_error)?;

        let report_value = Report {
            extent,
            channels: out_descriptor.layout.channel_count(),
            channel_stats: Vec::new(),
            all_finite,
            content_hash: String::new(),
            diff: None,
            assertion: None,
            histogram: None,
            components: None,
            frequency_energy: None,
            solver: Some(solver_data("poisson", &report)),
        };

        let mut out = OutputValues::new();
        out.insert("candidate".to_owned(), candidate);
        out.insert("report".to_owned(), ResourceValue::report(report_value));
        Ok(out)
    }
}

/// Read a required input *value* port, erroring if absent.
fn input_value<'a>(
    inputs: &'a InputValues,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_POISSON_BLEND_INPUT,
            format!("repair.poisson_blend requires a `{port}` input value"),
        )
    })
}

/// The wrong-resource-kind error for an input port.
fn input_type_error(port: &str, kind: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_POISSON_BLEND_INPUT,
        format!("repair.poisson_blend `{port}` input must be a {kind} resource"),
    )
}

/// A buffer-length-mismatch execution error for the candidate output.
fn buffer_error(actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_POISSON_BLEND_BUFFER,
        format!("repair.poisson_blend produced a candidate buffer of unexpected length {actual}"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `repair.poisson_blend@1`: an iterative
/// gradient-domain solver verified by analytic gradient/continuity fixtures
/// (uniform-gradient reconstruction, boundary continuity), seeded-free
/// determinism (bit-identical reruns), and the residual-history convergence
/// record. No perceptual metric applies.
fn blend_test_metadata() -> TestMetadata {
    use paintop_ir::{CategoryStatus, VerificationCategory, VerificationDeclarations};
    let mut decls = VerificationDeclarations::new();
    for category in [
        VerificationCategory::BuildHygiene,
        VerificationCategory::SchemaContract,
        VerificationCategory::AnalyticFixtures,
        VerificationCategory::PropertyTests,
        VerificationCategory::Metamorphic,
        VerificationCategory::Goldens,
        VerificationCategory::Fuzzing,
        VerificationCategory::Performance,
    ] {
        decls = decls.with(category, CategoryStatus::Covered);
    }
    decls = decls.with(
        VerificationCategory::Perceptual,
        CategoryStatus::not_applicable(
            "repair.poisson_blend is a gradient-domain Poisson solver verified by analytic \
             fixtures (gradients reproduced inside the mask, target continuity at the boundary), \
             bit-identical reruns, and a residual-history convergence record; there is no \
             perceptual-quality metric to apply",
        ),
    );
    TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

#[cfg(test)]
mod tests;
