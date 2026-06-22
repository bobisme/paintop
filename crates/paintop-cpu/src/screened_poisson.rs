//! The `repair.screened_poisson@1` operation: **screened-Poisson
//! reconstruction** with explicit `lambda` data-fidelity semantics (`OP_CATALOG`
//! §12; `plan.md` §1444 — M4 gradient-domain editing).
//!
//! The screened-Poisson problem reconstructs a field `u` that both matches a
//! `guidance` field's gradients and stays close to an `anchor` field, trading the
//! two off with a data weight `lambda >= 0`:
//!
//! ```text
//! (Δ - lambda·I) u = Δ(guidance) - lambda·anchor   inside the mask
//! u = anchor                                        on the mask boundary
//! ```
//!
//! per colour channel, via the shared [`crate::poisson`] Gauss-Seidel solver. The
//! `lambda` term is the explicit knob this op exposes:
//!
//! - **`lambda = 0`** is *pure Poisson*: the interior reproduces the guidance
//!   gradients with no pull toward the anchor (the seamless-clone limit, the same
//!   reconstruction `repair.poisson_blend` performs);
//! - **large `lambda`** approaches the *anchor*: the data term dominates and the
//!   interior collapses onto the anchor field, ignoring the guidance gradients;
//! - intermediate `lambda` smoothly interpolates between the two.
//!
//! These limits are the documented, tested semantics (the `lambda` boundary
//! cases are analytic fixtures). A negative or non-finite `lambda` is rejected.
//!
//! # Determinism + convergence metrics (M4 exit criteria 1 & 2)
//!
//! The solver runs fixed row-major Gauss-Seidel sweeps with no RNG, so a rerun is
//! bit-identical on a fixed backend. The `report` carries a
//! [`SolverData`](paintop_ir::SolverData) per the worst-converging channel:
//! iteration count, relative-residual history, stop reason, target tolerance, and
//! final residual.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    ImageDescriptor, ImplId, InputRegions, InputSpec, MaskDescriptor, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect,
    Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    TestMetadata,
};

use crate::poisson_blend::{
    BlendControls, DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE, MAX_ITERATIONS_LIMIT, blend_channels,
    check_shapes, solver_data,
};

/// The canonical id of the screened-Poisson operation.
pub const SCREENED_POISSON_OP_ID: &str = "repair.screened_poisson@1";

/// A required input port (`guidance`, `anchor`, or `mask`) was absent or carried
/// the wrong resource kind.
pub const E_SCREENED_POISSON_INPUT: &str = "E_SCREENED_POISSON_INPUT";

/// A solver-control or `lambda` parameter was the wrong type or out of range.
pub const E_SCREENED_POISSON_PARAM: &str = "E_SCREENED_POISSON_PARAM";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_SCREENED_POISSON_BUFFER: &str = "E_SCREENED_POISSON_BUFFER";

/// The default data weight (pure Poisson — no pull toward the anchor).
pub const DEFAULT_LAMBDA: f64 = 0.0;

/// Resolve the optional `lambda`, `max_iterations`, and `tolerance` params.
fn resolve_controls(params: &serde_json::Value) -> Result<BlendControls> {
    let lambda = match params.get("lambda") {
        None => DEFAULT_LAMBDA,
        Some(value) => {
            let l = value
                .as_f64()
                .ok_or_else(|| param_error("lambda must be a number", value))?;
            if !l.is_finite() || l < 0.0 {
                return Err(param_error("lambda must be a finite value >= 0", value));
            }
            l
        }
    };
    let max_iterations = match params.get("max_iterations") {
        None => DEFAULT_MAX_ITERATIONS,
        Some(value) => {
            let n = value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| param_error("max_iterations must be a positive integer", value))?;
            if n == 0 || n > MAX_ITERATIONS_LIMIT {
                return Err(param_error(
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
                .ok_or_else(|| param_error("tolerance must be a number", value))?;
            if !t.is_finite() || t <= 0.0 || t >= 1.0 {
                return Err(param_error("tolerance must be in (0, 1)", value));
            }
            t
        }
    };
    Ok(BlendControls {
        max_iterations,
        tolerance,
        lambda,
    })
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_SCREENED_POISSON_PARAM,
        format!("repair.screened_poisson: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// The `repair.screened_poisson@1` operation: `guidance` + `anchor` + `mask` →
/// the reconstructed `candidate` plus a solver `report`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScreenedPoisson;

impl ScreenedPoisson {
    /// Construct the screened-Poisson operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `repair.screened_poisson@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: SCREENED_POISSON_OP_ID.parse()?,
            impl_version: 1,
            summary: "Screened-Poisson reconstruction with explicit lambda data-fidelity weight: \
                      inside the mask solve (Δ - lambda·I)u = Δguidance - lambda·anchor with a \
                      Dirichlet anchor boundary, per channel by row-major Gauss-Seidel. lambda = 0 \
                      is pure Poisson (guidance gradients); large lambda approaches the anchor. \
                      Deterministic (bit-identical reruns); the report carries SolverData."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "guidance".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The image whose gradients drive the masked interior (the guidance \
                          divergence Δguidance)."
                        .to_owned(),
                },
                InputSpec {
                    name: "anchor".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The data-fidelity target: the Dirichlet boundary and the field the \
                          interior is pulled toward with weight lambda."
                        .to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc:
                        "The coverage mask selecting the interior region to reconstruct (> 0.5 is \
                          interior)."
                            .to_owned(),
                },
            ],
            outputs: vec![
                OutputSpec {
                    name: "candidate".to_owned(),
                    kind: ResourceKind::Image,
                    doc: "The reconstructed image (the anchor descriptor; same extent/layout)."
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
            test: screened_test_metadata(),
        })
    }
}

/// The declared parameter list (`lambda` plus the shared solver controls).
fn params_spec() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            name: "lambda".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(DEFAULT_LAMBDA)),
            choices: vec![],
            doc:
                "The data-fidelity weight (>= 0). 0 is pure Poisson (guidance gradients); a large \
                  value pulls the interior toward the anchor."
                    .to_owned(),
        },
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
            E_SCREENED_POISSON_INPUT,
            format!("repair.screened_poisson requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_SCREENED_POISSON_INPUT,
            format!("repair.screened_poisson `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read the `mask` port's descriptor.
fn mask_descriptor(inputs: &Descriptors) -> Result<&MaskDescriptor> {
    let resource = inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_SCREENED_POISSON_INPUT,
            "repair.screened_poisson requires a `mask` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Mask(mask) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_SCREENED_POISSON_INPUT,
            "repair.screened_poisson `mask` input must be a mask resource".to_owned(),
        ));
    };
    Ok(mask)
}

impl OpContract for ScreenedPoisson {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("guidance".to_owned(), ResourceKind::Image),
            ("anchor".to_owned(), ResourceKind::Image),
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
        resolve_controls(params)?;
        let guidance = image_descriptor(inputs, "guidance")?;
        let anchor = image_descriptor(inputs, "anchor")?;
        let mask = mask_descriptor(inputs)?;
        // The output keeps the anchor descriptor; guidance/anchor/mask must agree
        // on extent and the images on layout.
        let out_descriptor =
            check_shapes(guidance, anchor, mask.extent, "repair.screened_poisson")?;

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
        let mut regions = InputRegions::new();
        for port in ["guidance", "anchor", "mask"] {
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

impl OpImplementation for ScreenedPoisson {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let controls = resolve_controls(params)?;
        let guidance = input_value(inputs, "guidance")?;
        let anchor = input_value(inputs, "anchor")?;
        let mask = input_value(inputs, "mask")?;

        let ResourceDescriptor::Image(guidance_descriptor) = guidance.descriptor() else {
            return Err(input_type_error("guidance", "image"));
        };
        let ResourceDescriptor::Image(anchor_descriptor) = anchor.descriptor() else {
            return Err(input_type_error("anchor", "image"));
        };
        let ResourceDescriptor::Mask(mask_descriptor) = mask.descriptor() else {
            return Err(input_type_error("mask", "mask"));
        };

        let out_descriptor = check_shapes(
            guidance_descriptor,
            anchor_descriptor,
            mask_descriptor.extent,
            "repair.screened_poisson",
        )?;
        let channels = out_descriptor.layout.channel_count() as usize;
        let extent = out_descriptor.extent;

        // `blend_channels` reconstructs each channel from the guidance Laplacian
        // (the source role) toward the anchor (the target role), with the
        // screened `lambda` pulling the interior toward the anchor.
        let (samples, report) = blend_channels(
            guidance.samples(),
            anchor.samples(),
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
            solver: Some(solver_data("screened-poisson", &report)),
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
            E_SCREENED_POISSON_INPUT,
            format!("repair.screened_poisson requires a `{port}` input value"),
        )
    })
}

/// The wrong-resource-kind error for an input port.
fn input_type_error(port: &str, kind: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_SCREENED_POISSON_INPUT,
        format!("repair.screened_poisson `{port}` input must be a {kind} resource"),
    )
}

/// A buffer-length-mismatch execution error for the candidate output.
fn buffer_error(actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_SCREENED_POISSON_BUFFER,
        format!(
            "repair.screened_poisson produced a candidate buffer of unexpected length {actual}"
        ),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `repair.screened_poisson@1`: an iterative
/// screened-Poisson solver verified by analytic lambda-limit fixtures (lambda = 0
/// recovers pure Poisson, large lambda snaps to the anchor), boundary continuity,
/// determinism (bit-identical reruns), and the residual-history convergence
/// record. No perceptual metric applies.
fn screened_test_metadata() -> TestMetadata {
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
            "repair.screened_poisson is a screened-Poisson solver verified by analytic lambda-limit \
             fixtures (lambda = 0 recovers pure Poisson, large lambda snaps to the anchor), \
             boundary continuity, bit-identical reruns, and a residual-history convergence record; \
             there is no perceptual-quality metric to apply",
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
