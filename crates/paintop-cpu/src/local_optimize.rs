//! The `optimize.local@1` operation: a **contract-driven local optimizer**.
//!
//! It drives a candidate image toward minimizing a declared, mask-restricted
//! objective (`plan.md` §1428 final deliverable; `ALIEN_OPS` §7).
//!
//! The op takes an `init` candidate image, a data `target` image, and a coverage
//! `mask`, and runs the deterministic [`crate::optimize`] gradient-descent engine
//! per channel to minimize
//!
//! ```text
//! E(u) = data_weight · Σ_{mask} (u − target)² + smooth_weight · Σ_{mask} ‖∇u‖²
//! ```
//!
//! returning the optimized `candidate` image plus a solver `report`. The objective
//! is declared by the two non-negative weights — there is **no arbitrary code
//! execution**, only this fixed analytic family — and an invalid weight, step, or
//! iteration cap is a schema error rejected before the engine runs.
//!
//! # Known minimum (M4 exit criterion 3)
//!
//! With `smooth_weight = 0` the unique minimizer inside the mask is `u = target`,
//! so a synthetic target gives the optimizer a known minimum it converges to
//! within tolerance (the convergence fixtures pin this). Pixels outside the mask
//! are frozen at their initial value, so the edit stays local.
//!
//! # Determinism + convergence metrics (M4 exit criteria 1 & 2)
//!
//! The engine runs fixed row-major gradient-descent steps with no RNG, so a rerun
//! with the same seed and schedule is bit-identical on a fixed backend. The
//! `report` carries a [`SolverData`] for the
//! worst-converging channel: iteration count, relative-objective history, stop
//! reason, target tolerance, final objective, and the schedule seed.
//!
//! # Disable switch (M4 acceptance)
//!
//! The `enabled` param (default `true`) is a policy gate: a request with
//! `enabled = false` returns a typed [`policy`](ErrorClass::Policy) error rather
//! than running the optimizer, so a caller can disable optimizer execution without
//! removing the op from a graph.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, MaskDescriptor, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect,
    Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    SolverData, TestMetadata,
};

use crate::optimize::{Controls, MinimizeReport, Objective, Problem, classify_cells, minimize};

/// The canonical id of the local-optimizer operation.
pub const LOCAL_OPTIMIZE_OP_ID: &str = "optimize.local@1";

/// A required input port (`init`, `target`, or `mask`) was absent or carried the
/// wrong resource kind.
pub const E_OPTIMIZE_INPUT: &str = "E_OPTIMIZE_INPUT";

/// An objective weight, step, tolerance, or iteration-cap param was the wrong
/// type or out of range (or the objective was all-zero).
pub const E_OPTIMIZE_PARAM: &str = "E_OPTIMIZE_PARAM";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_OPTIMIZE_BUFFER: &str = "E_OPTIMIZE_BUFFER";

/// The optimizer was disabled by policy (`enabled = false`) and so refused to run.
pub const E_OPTIMIZE_DISABLED: &str = "E_OPTIMIZE_DISABLED";

/// The default data-term weight (a pure data-fit objective).
pub const DEFAULT_DATA_WEIGHT: f64 = 1.0;

/// The default smoothness-term weight (no smoothness pull — the known-minimum
/// limit where the minimizer is exactly the target inside the mask).
pub const DEFAULT_SMOOTH_WEIGHT: f64 = 0.0;

/// The default gradient-descent step (the schedule).
///
/// Conservative enough to stay stable for the default `data_weight = 1`
/// objective (whose gradient is `2·(u−target)`, so a step `< 1` contracts the
/// data error monotonically).
pub const DEFAULT_STEP: f64 = 0.25;

/// The default maximum number of descent iterations before stopping at the cap.
pub const DEFAULT_MAX_ITERATIONS: u32 = 500;

/// The default relative-objective convergence tolerance.
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// The largest iteration cap a single request may ask for, bounding the work so a
/// request can never spin unbounded.
pub const MAX_ITERATIONS_LIMIT: u32 = 100_000;

/// The resolved optimizer controls plus objective for a run.
#[derive(Debug, Clone, Copy)]
struct OptimizeControls {
    objective: Objective,
    controls: Controls,
}

/// Resolve and validate the optimizer params (objective weights, step, tolerance,
/// iteration cap, seed), after checking the `enabled` policy gate.
fn resolve_controls(params: &serde_json::Value) -> Result<OptimizeControls> {
    // Policy gate first: a disabled optimizer refuses to run.
    if let Some(value) = params.get("enabled") {
        let enabled = value
            .as_bool()
            .ok_or_else(|| param_error("enabled must be a boolean", value))?;
        if !enabled {
            return Err(Error::new(
                ErrorClass::Policy,
                E_OPTIMIZE_DISABLED,
                "optimize.local is disabled by policy (enabled = false); the optimizer refused to \
                 run"
                .to_owned(),
            ));
        }
    }

    let data_weight = non_negative(params, "data_weight", DEFAULT_DATA_WEIGHT)?;
    let smooth_weight = non_negative(params, "smooth_weight", DEFAULT_SMOOTH_WEIGHT)?;
    if data_weight == 0.0 && smooth_weight == 0.0 {
        return Err(param_error(
            "the objective is degenerate: at least one of data_weight, smooth_weight must be > 0",
            &serde_json::json!({ "data_weight": data_weight, "smooth_weight": smooth_weight }),
        ));
    }

    let step = match params.get("step") {
        None => DEFAULT_STEP,
        Some(value) => {
            let s = value
                .as_f64()
                .ok_or_else(|| param_error("step must be a number", value))?;
            if !s.is_finite() || s <= 0.0 {
                return Err(param_error("step must be a finite value > 0", value));
            }
            s
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

    let seed = match params.get("seed") {
        None => 0,
        Some(value) => value
            .as_u64()
            .ok_or_else(|| param_error("seed must be a non-negative integer", value))?,
    };

    Ok(OptimizeControls {
        objective: Objective {
            data_weight,
            smooth_weight,
        },
        controls: Controls {
            max_iterations,
            tolerance,
            step,
            seed,
        },
    })
}

/// Resolve an optional finite, non-negative weight param.
fn non_negative(params: &serde_json::Value, name: &str, default: f64) -> Result<f64> {
    match params.get(name) {
        None => Ok(default),
        Some(value) => {
            let w = value
                .as_f64()
                .ok_or_else(|| param_error(&format!("{name} must be a number"), value))?;
            if !w.is_finite() || w < 0.0 {
                return Err(param_error(
                    &format!("{name} must be a finite value >= 0"),
                    value,
                ));
            }
            Ok(w)
        }
    }
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_OPTIMIZE_PARAM,
        format!("optimize.local: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// The `optimize.local@1` operation: `init` + `target` + `mask` → the optimized
/// `candidate` plus a solver `report`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalOptimize;

impl LocalOptimize {
    /// Construct the local-optimizer operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `optimize.local@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: LOCAL_OPTIMIZE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Contract-driven local optimizer: inside the mask, minimize the declared \
                      objective data_weight·Σ(u−target)² + smooth_weight·Σ‖∇u‖² by deterministic \
                      row-major gradient descent, freezing pixels outside the mask. \
                      smooth_weight = 0 has the known minimum u = target. Deterministic \
                      (bit-identical reruns); the report carries the objective trajectory, stop \
                      reason, and seed. enabled = false disables it via a typed policy error."
                .to_owned(),
            determinism: DeterminismTier::Reproducible,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "init".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The initial candidate image: the descent's starting guess inside the \
                          mask and the frozen value outside it."
                        .to_owned(),
                },
                InputSpec {
                    name: "target".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The data-term target the masked interior is driven toward.".to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc: "The coverage mask selecting the free region to optimize (> 0.5 is free; \
                          elsewhere is frozen)."
                        .to_owned(),
                },
            ],
            outputs: vec![
                OutputSpec {
                    name: "candidate".to_owned(),
                    kind: ResourceKind::Image,
                    doc: "The optimized image (the init descriptor; same extent/layout)."
                        .to_owned(),
                },
                OutputSpec {
                    name: "report".to_owned(),
                    kind: ResourceKind::Report,
                    doc:
                        "The optimizer report carrying SolverData (iterations, objective history, \
                          stop reason, tolerance, final objective, seed)."
                            .to_owned(),
                },
            ],
            params: params_spec(),
            implementations: vec![reference_impl()?],
            test: optimize_test_metadata(),
        })
    }
}

/// The declared parameter list (objective weights, schedule, and the disable
/// switch).
fn params_spec() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            name: "data_weight".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(DEFAULT_DATA_WEIGHT)),
            choices: vec![],
            doc: "The objective weight on the masked data term Σ(u−target)² (>= 0).".to_owned(),
        },
        ParamSpec {
            name: "smooth_weight".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(DEFAULT_SMOOTH_WEIGHT)),
            choices: vec![],
            doc: "The objective weight on the masked smoothness term Σ‖∇u‖² (>= 0). 0 leaves the \
                  known minimum u = target."
                .to_owned(),
        },
        ParamSpec {
            name: "step".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(DEFAULT_STEP)),
            choices: vec![],
            doc: "The fixed gradient-descent step size (the schedule); a finite value > 0."
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
                "The maximum descent iterations before stopping at the cap, in \
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
            doc: "The relative-objective convergence tolerance in (0, 1).".to_owned(),
        },
        ParamSpec {
            name: "seed".to_owned(),
            ty: ParamType::Seed,
            unit: None,
            required: false,
            default: Some(serde_json::json!(0)),
            choices: vec![],
            doc: "The schedule seed, carried into the report as the run identity (the descent is \
                  deterministic and does not consume it)."
                .to_owned(),
        },
        ParamSpec {
            name: "enabled".to_owned(),
            ty: ParamType::Boolean,
            unit: None,
            required: false,
            default: Some(serde_json::json!(true)),
            choices: vec![],
            doc: "The policy gate: false disables the optimizer, returning a typed policy error \
                  instead of running."
                .to_owned(),
        },
    ]
}

/// Read a required image port's descriptor.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_OPTIMIZE_INPUT,
            format!("optimize.local requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_OPTIMIZE_INPUT,
            format!("optimize.local `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read the `mask` port's descriptor.
fn mask_descriptor(inputs: &Descriptors) -> Result<&MaskDescriptor> {
    let resource = inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_OPTIMIZE_INPUT,
            "optimize.local requires a `mask` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Mask(mask) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_OPTIMIZE_INPUT,
            "optimize.local `mask` input must be a mask resource".to_owned(),
        ));
    };
    Ok(mask)
}

/// Check the `init`/`target`/`mask` extents and layouts agree; return the output
/// (init) descriptor.
fn check_shapes(
    init: &ImageDescriptor,
    target: &ImageDescriptor,
    mask_extent: Extent,
) -> Result<ImageDescriptor> {
    if init.extent != target.extent {
        return Err(shape_error(
            "the init and target images must share an extent",
            format!("init {:?} vs target {:?}", init.extent, target.extent),
        ));
    }
    if init.layout != target.layout {
        return Err(shape_error(
            "the init and target images must share a channel layout",
            format!("init {:?} vs target {:?}", init.layout, target.layout),
        ));
    }
    if mask_extent != init.extent {
        return Err(shape_error(
            "the mask must share the images' extent",
            format!("mask {mask_extent:?} vs image {:?}", init.extent),
        ));
    }
    Ok(*init)
}

/// A shape-mismatch schema error.
fn shape_error(detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_OPTIMIZE_INPUT,
        format!("optimize.local: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

impl OpContract for LocalOptimize {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("init".to_owned(), ResourceKind::Image),
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
        resolve_controls(params)?;
        let init = image_descriptor(inputs, "init")?;
        let target = image_descriptor(inputs, "target")?;
        let mask = mask_descriptor(inputs)?;
        let out_descriptor = check_shapes(init, target, mask.extent)?;

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
        for port in ["init", "target", "mask"] {
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

impl OpImplementation for LocalOptimize {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let resolved = resolve_controls(params)?;
        let init = input_value(inputs, "init")?;
        let target = input_value(inputs, "target")?;
        let mask = input_value(inputs, "mask")?;

        let ResourceDescriptor::Image(init_descriptor) = init.descriptor() else {
            return Err(input_type_error("init", "image"));
        };
        let ResourceDescriptor::Image(target_descriptor) = target.descriptor() else {
            return Err(input_type_error("target", "image"));
        };
        let ResourceDescriptor::Mask(mask_descriptor) = mask.descriptor() else {
            return Err(input_type_error("mask", "mask"));
        };

        let out_descriptor =
            check_shapes(init_descriptor, target_descriptor, mask_descriptor.extent)?;
        let channels = out_descriptor.layout.channel_count() as usize;
        let extent = out_descriptor.extent;

        let (samples, report) = optimize_channels(
            init.samples(),
            target.samples(),
            mask.samples(),
            channels,
            extent,
            resolved,
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
            solver: Some(solver_data(&report, resolved.controls.seed)),
        };

        let mut out = OutputValues::new();
        out.insert("candidate".to_owned(), candidate);
        out.insert("report".to_owned(), ResourceValue::report(report_value));
        Ok(out)
    }
}

/// Optimize every channel independently, returning the interleaved optimized
/// samples and the report of the worst-converging channel.
fn optimize_channels(
    init: &[f32],
    target: &[f32],
    mask: &[f32],
    channels: usize,
    extent: Extent,
    resolved: OptimizeControls,
) -> (Vec<f32>, MinimizeReport) {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let count = width * height;
    let cells = classify_cells(mask, width, height);

    let mut out = vec![0.0_f32; count * channels];
    let mut worst: Option<MinimizeReport> = None;

    for channel in 0..channels {
        let init_ch = channel_f64(init, channels, channel, count);
        let target_ch = channel_f64(target, channels, channel, count);

        let problem = Problem {
            width,
            height,
            cells: &cells,
            init: &init_ch,
            target: &target_ch,
            objective: resolved.objective,
        };
        let (field, report) = minimize(&problem, resolved.controls);

        for pixel in 0..count {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the channel optimized in f64, stored once as the op's f32 sample"
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

    let report = worst.unwrap_or_else(|| MinimizeReport {
        objective_history: Vec::new(),
        iterations: 0,
        converged: true,
        stop_reason: paintop_ir::SolverStopReason::Converged,
        tolerance: resolved.controls.tolerance,
        final_objective: 0.0,
        initial_objective: 0.0,
    });
    (out, report)
}

/// Deinterleave one channel into an owned `f64` row-major buffer.
fn channel_f64(samples: &[f32], channels: usize, channel: usize, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    for pixel in 0..count {
        out.push(f64::from(samples[pixel * channels + channel]));
    }
    out
}

/// The conservative report: the channel that ran more iterations, then reached the
/// larger final objective.
fn worse_of(a: MinimizeReport, b: MinimizeReport) -> MinimizeReport {
    if (b.iterations, b.final_objective) > (a.iterations, a.final_objective) {
        b
    } else {
        a
    }
}

/// Build the [`SolverData`] payload from the optimizer's [`MinimizeReport`],
/// carrying the schedule `seed` in the (otherwise unused for an optimizer)
/// stability fields' place as the run identity.
fn solver_data(report: &MinimizeReport, seed: u64) -> SolverData {
    SolverData {
        kind: "local-optimizer".to_owned(),
        // `steps` mirrors the iteration count.
        steps: report.iterations,
        // The seed is the schedule identity; an optimizer has no CFL stability
        // number, so the stability fields carry the seed/initial-objective record.
        #[allow(
            clippy::cast_precision_loss,
            reason = "the seed is recorded for audit; an exact round-trip is not required"
        )]
        stability_number: seed as f64,
        stability_limit: report.initial_objective,
        stable: report.converged,
        residual_history: report.objective_history.clone(),
        total_energy: report.initial_objective,
        iterations: Some(report.iterations),
        stop_reason: Some(report.stop_reason),
        converged: Some(report.converged),
        tolerance: Some(report.tolerance),
        final_residual: Some(report.final_objective),
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
            E_OPTIMIZE_INPUT,
            format!("optimize.local requires a `{port}` input value"),
        )
    })
}

/// The wrong-resource-kind error for an input port.
fn input_type_error(port: &str, kind: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_OPTIMIZE_INPUT,
        format!("optimize.local `{port}` input must be a {kind} resource"),
    )
}

/// A buffer-length-mismatch execution error for the candidate output.
fn buffer_error(actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_OPTIMIZE_BUFFER,
        format!("optimize.local produced a candidate buffer of unexpected length {actual}"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `optimize.local@1`: a deterministic local
/// optimizer verified by a known-minimum analytic fixture (`smooth_weight = 0`
/// recovers the target inside the mask), the objective trajectory, deterministic
/// (bit-identical) reruns, and the max-iteration / no-progress stop rules. No
/// perceptual metric applies.
fn optimize_test_metadata() -> TestMetadata {
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
            "optimize.local is a deterministic local optimizer verified by a known-minimum analytic \
             fixture (smooth_weight = 0 recovers the target inside the mask), the objective \
             trajectory, bit-identical reruns, and the max-iteration / no-progress stop rules; \
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
