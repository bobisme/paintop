//! The `repair.patch_field@1` operation (`OP_CATALOG` §10, `PatchMatch`).
//!
//! Computes an approximate nearest-neighbour patch-correspondence field from a
//! `source` image into a `target` image: for every target pixel it finds the
//! source coordinate of a similar `(2·radius + 1)²` patch via deterministic,
//! seeded [`PatchMatch`](crate::patchmatch::patch_match). Two optional masks
//! shape the problem — a `target_mask` selecting which target pixels need a
//! correspondence (the hole of an inpainting target) and a `source_mask`
//! selecting which source pixels are eligible anchors (the known region). The
//! op emits a [`PatchField`](paintop_ir::ResourceDescriptor::PatchField) and a
//! [`Report`] carrying the search's convergence trace.
//!
//! # Determinism
//!
//! The search is **reproducible** ([`DeterminismTier::Reproducible`]): every
//! random choice is a hash of its `(seed, x, y, iteration, step)` coordinate, the
//! scan order is a fixed forward/backward alternation, and ties keep the
//! incumbent — so a fixed seed and scan order yield a bit-identical field on every
//! rerun on a fixed backend (the M4 determinism criterion). The op's report
//! exposes the per-iteration total-cost history, the iteration count, and whether
//! the search reached a fixed point (the M4 "solver exposes convergence metrics"
//! criterion).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, PatchFieldDescriptor, Rect,
    Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    SolverData, SolverStopReason, TestMetadata,
};

use crate::patchmatch::{PatchPlane, SearchConfig, SearchResult, patch_match};

/// The canonical id of the patch-field operation.
pub const PATCH_FIELD_OP_ID: &str = "repair.patch_field@1";

/// A required input was absent or carried an unsupported descriptor.
pub const E_PATCH_FIELD_INPUT: &str = "E_PATCH_FIELD_INPUT";

/// A parameter was missing, the wrong type, or out of range.
pub const E_PATCH_FIELD_PARAM: &str = "E_PATCH_FIELD_PARAM";

/// The largest patch radius a plan may request (mirrors the resource bound).
pub const RADIUS_MAX: u32 = paintop_ir::MAX_PATCH_RADIUS;

/// The largest iteration count a plan may request (a finite, generous cap).
pub const ITERATIONS_MAX: u32 = 256;

/// The resolved search parameters.
#[derive(Debug, Clone, Copy)]
pub struct PatchFieldParams {
    /// The patch half-window.
    pub radius: u32,
    /// The number of propagation/random-search iterations.
    pub iterations: u32,
    /// The deterministic seed.
    pub seed: u64,
}

impl PatchFieldParams {
    /// Resolve and validate `radius`, `iterations`, and `seed` from the param
    /// object.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error for a missing/ill-typed
    /// parameter or an out-of-range radius or iteration count.
    pub fn resolve(params: &serde_json::Value) -> Result<Self> {
        let radius = required_u64(params, "radius")?;
        if radius > u64::from(RADIUS_MAX) {
            return Err(param_err(&format!("`radius` must be in 0..={RADIUS_MAX}"))
                .with_context(ErrorContext::default().with_actual(radius.to_string())));
        }
        let iterations = match params.get("iterations") {
            None => 8,
            Some(v) => v
                .as_u64()
                .ok_or_else(|| param_err("`iterations` must be a non-negative integer"))?,
        };
        if iterations == 0 || iterations > u64::from(ITERATIONS_MAX) {
            return Err(
                param_err(&format!("`iterations` must be in 1..={ITERATIONS_MAX}"))
                    .with_context(ErrorContext::default().with_actual(iterations.to_string())),
            );
        }
        let seed = match params.get("seed") {
            None => 0,
            Some(v) => v
                .as_u64()
                .ok_or_else(|| param_err("`seed` must be a non-negative integer"))?,
        };
        #[allow(
            clippy::cast_possible_truncation,
            reason = "radius and iterations are validated within u32 bounds above"
        )]
        Ok(Self {
            radius: radius as u32,
            iterations: iterations as u32,
            seed,
        })
    }
}

/// Read a required non-negative integer parameter.
fn required_u64(params: &serde_json::Value, name: &str) -> Result<u64> {
    params
        .get(name)
        .ok_or_else(|| param_err(&format!("requires a `{name}` parameter")))?
        .as_u64()
        .ok_or_else(|| param_err(&format!("`{name}` must be a non-negative integer")))
}

/// Build a schema [`Error`] for a malformed patch-field parameter.
fn param_err(detail: &str) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_PATCH_FIELD_PARAM,
        format!("repair.patch_field {detail}"),
    )
}

/// Read a required image port's descriptor.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ResourceDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_PATCH_FIELD_INPUT,
            format!("repair.patch_field requires a `{port}` input"),
        )
    })?;
    match resource {
        ResourceDescriptor::Image(_) | ResourceDescriptor::Field1(_) => Ok(resource),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_PATCH_FIELD_INPUT,
            format!("repair.patch_field `{port}` must be an Image or Field1 resource"),
        )),
    }
}

/// The interleaved channel count of a supported image/field descriptor.
const fn descriptor_channels(descriptor: &ResourceDescriptor) -> u32 {
    match descriptor {
        ResourceDescriptor::Image(d) => d.layout.channel_count(),
        _ => 1,
    }
}

/// Build the [`PatchFieldDescriptor`] a search over `target`/`source` produces.
const fn field_descriptor(target: Extent, source: Extent, radius: u32) -> PatchFieldDescriptor {
    PatchFieldDescriptor {
        target_extent: target,
        source_extent: source,
        radius,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The `repair.patch_field@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PatchField;

impl PatchField {
    /// Construct the patch-field operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `repair.patch_field@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: PATCH_FIELD_OP_ID.parse()?,
            impl_version: 1,
            summary: "Compute an approximate nearest-neighbour patch-correspondence field \
                      (PatchField) from a source image into a target image via deterministic, \
                      seeded PatchMatch. Optional target/source masks select the hole pixels to \
                      match and the eligible source anchors. Emits the field plus a convergence \
                      report."
                .to_owned(),
            determinism: DeterminismTier::Reproducible,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "source".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The source image coherent patches are drawn from.".to_owned(),
                },
                InputSpec {
                    name: "target".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The target image each correspondence is matched against.".to_owned(),
                },
                InputSpec {
                    name: "target_mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: "Optional: target pixels (> 0.5) needing a correspondence (the hole). \
                          Absent means every target pixel is matched."
                        .to_owned(),
                },
                InputSpec {
                    name: "source_mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: "Optional: source pixels (> 0.5) eligible as anchors (the known \
                          region). Absent means every source pixel is eligible."
                        .to_owned(),
                },
            ],
            outputs: vec![
                OutputSpec {
                    name: "field".to_owned(),
                    kind: ResourceKind::PatchField,
                    doc: "The approximate nearest-neighbour field (src_x, src_y, cost per target \
                          pixel)."
                        .to_owned(),
                },
                OutputSpec {
                    name: "report".to_owned(),
                    kind: ResourceKind::Report,
                    doc: "The search report carrying SolverData (iterations, per-iteration cost \
                          history, fixed-point/stop reason)."
                        .to_owned(),
                },
            ],
            params: vec![
                ParamSpec {
                    name: "radius".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: format!(
                        "The patch half-window: a (2r+1)x(2r+1) patch (0..={RADIUS_MAX})."
                    ),
                },
                ParamSpec {
                    name: "iterations".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(8)),
                    choices: vec![],
                    doc: format!(
                        "The number of propagation/random-search sweeps (1..={ITERATIONS_MAX})."
                    ),
                },
                ParamSpec {
                    name: "seed".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(0)),
                    choices: vec![],
                    doc: "The deterministic search seed (hash-of-coordinate RNG).".to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: patch_field_test_metadata(),
        })
    }
}

impl OpContract for PatchField {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("source".to_owned(), ResourceKind::Image),
            ("target".to_owned(), ResourceKind::Image),
            ("target_mask".to_owned(), ResourceKind::Mask),
            ("source_mask".to_owned(), ResourceKind::Mask),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("field".to_owned(), ResourceKind::PatchField),
            ("report".to_owned(), ResourceKind::Report),
        ]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let resolved = PatchFieldParams::resolve(params)?;
        let source = image_descriptor(inputs, "source")?;
        let target = image_descriptor(inputs, "target")?;
        let descriptor = field_descriptor(target.extent(), source.extent(), resolved.radius);
        descriptor.validate()?;

        let mut out = OutputDescriptors::new();
        out.insert(
            "field".to_owned(),
            ResourceDescriptor::PatchField(descriptor),
        );
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent: target.extent(),
                channels: descriptor_channels(target),
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
        // The search reads every source and target pixel (a full-domain op).
        let mut regions = InputRegions::new();
        for port in ["source", "target", "target_mask", "source_mask"] {
            if let Some(resource) = inputs.get(port) {
                let e = resource.extent();
                regions.insert(
                    port.to_owned(),
                    Rect::new(0, 0, i64::from(e.width), i64::from(e.height)),
                );
            }
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("field") {
            Some(ResourceDescriptor::PatchField(_)) => {
                AssertionResult::pass("produces_patch_field")
            }
            _ => AssertionResult::fail("produces_patch_field", "no `field` output produced"),
        }])
    }
}

impl OpImplementation for PatchField {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let resolved = PatchFieldParams::resolve(params)?;
        let source = input_value(inputs, "source")?;
        let target = input_value(inputs, "target")?;

        let source_plane = plane_of(source, "source")?;
        let target_plane = plane_of(target, "target")?;

        // Optional masks: read each into a `> 0.5` predicate buffer the search
        // closures consult by coordinate.
        let target_mask = mask_predicate(inputs, "target_mask", target.extent())?;
        let source_mask = mask_predicate(inputs, "source_mask", source.extent())?;

        let descriptor = field_descriptor(target.extent(), source.extent(), resolved.radius);
        descriptor.validate()?;

        let (tw, sw) = (target.extent().width, source.extent().width);
        let config = SearchConfig {
            target: target_plane,
            source: source_plane,
            radius: resolved.radius,
            iterations: resolved.iterations,
            seed: resolved.seed,
            match_target: |x: u32, y: u32| {
                target_mask.as_ref().is_none_or(|m| m[mask_idx(x, y, tw)])
            },
            source_valid: |x: u32, y: u32| {
                source_mask.as_ref().is_none_or(|m| m[mask_idx(x, y, sw)])
            },
        };
        let result = patch_match(&config);

        let field_value =
            ResourceValue::patch_field(descriptor, pack_field(&result)).map_err(buffer_error)?;

        let report = build_report(
            target.extent(),
            descriptor_channels(target.descriptor()),
            &result,
        );

        let mut out = OutputValues::new();
        out.insert("field".to_owned(), field_value);
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// Flatten a search result into the packed `(src_x, src_y, cost)` field buffer.
fn pack_field(result: &SearchResult) -> Vec<f32> {
    let mut samples = Vec::with_capacity(result.field.matches.len() * 3);
    for m in &result.field.matches {
        #[allow(
            clippy::cast_precision_loss,
            reason = "source coordinates are small image indices, exact in f32"
        )]
        let (sx, sy) = (m.src_x as f32, m.src_y as f32);
        // An unreachable (no eligible anchor) cell carries an infinite cost;
        // store a large finite sentinel so the field stays all-finite.
        let cost = if m.cost.is_finite() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "cost is an SSD magnitude; f32 range is ample for a field cost"
            )]
            let c = m.cost as f32;
            c
        } else {
            f32::MAX
        };
        samples.push(sx);
        samples.push(sy);
        samples.push(cost);
    }
    samples
}

/// Build the convergence [`Report`] from a search result.
fn build_report(extent: Extent, channels: u32, result: &SearchResult) -> Report {
    let final_cost = result.cost_history.last().copied().unwrap_or(0.0);
    let stop_reason = if result.converged {
        SolverStopReason::Stalled
    } else {
        SolverStopReason::MaxIterations
    };
    Report {
        extent,
        channels,
        channel_stats: Vec::new(),
        all_finite: true,
        content_hash: String::new(),
        diff: None,
        assertion: None,
        histogram: None,
        components: None,
        frequency_energy: None,
        solver: Some(SolverData {
            kind: "patchmatch".to_owned(),
            steps: result.iterations,
            stability_number: 0.0,
            stability_limit: 0.0,
            stable: true,
            residual_history: result.cost_history.clone(),
            total_energy: final_cost,
            iterations: Some(result.iterations),
            stop_reason: Some(stop_reason),
            converged: Some(result.converged),
            tolerance: None,
            final_residual: Some(final_cost),
        }),
    }
}

/// The flat index of mask pixel `(x, y)` in a single-channel row-major buffer.
const fn mask_idx(x: u32, y: u32, width: u32) -> usize {
    (y as usize * width as usize) + x as usize
}

/// Read a required input value by port.
fn input_value<'a>(
    inputs: &'a InputValues,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_PATCH_FIELD_INPUT,
            format!("repair.patch_field requires a `{port}` value"),
        )
    })
}

/// Build a [`PatchPlane`] view over an input value's samples.
fn plane_of<'a>(
    value: &'a ResourceValue,
    port: &str,
) -> std::result::Result<PatchPlane<'a>, Error> {
    let e = value.extent();
    PatchPlane::new(value.samples(), e.width, e.height, value.channels()).ok_or_else(|| {
        Error::new(
            ErrorClass::Execution,
            E_PATCH_FIELD_INPUT,
            format!("repair.patch_field `{port}` sample buffer does not match its extent"),
        )
    })
}

/// Read an optional mask port into a `> 0.5` boolean buffer, or `None` if the
/// port is absent.
fn mask_predicate(
    inputs: &InputValues,
    port: &str,
    expected: Extent,
) -> std::result::Result<Option<Vec<bool>>, Error> {
    let Some(value) = inputs.get(port) else {
        return Ok(None);
    };
    if value.extent() != expected {
        return Err(Error::new(
            ErrorClass::Type,
            E_PATCH_FIELD_INPUT,
            format!(
                "repair.patch_field `{port}` extent {}x{} must match its image {}x{}",
                value.extent().width,
                value.extent().height,
                expected.width,
                expected.height
            ),
        ));
    }
    let predicate = value.samples().iter().map(|&s| s > 0.5).collect();
    Ok(Some(predicate))
}

/// Build an execution error for a mis-sized produced field buffer.
fn buffer_error(actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_PATCH_FIELD_INPUT,
        format!("repair.patch_field produced a field buffer of unexpected length {actual}"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `repair.patch_field@1`: a reproducible, seeded
/// iterative op verified by determinism (bit-identical reruns), a brute-force
/// differential oracle on tiny fixtures, and the convergence report. Differential
/// across backends is not applicable (a single reference today); perceptual does
/// not apply — correctness is the oracle agreement and determinism, not a
/// perceptual metric.
fn patch_field_test_metadata() -> TestMetadata {
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
        VerificationCategory::Differential,
        CategoryStatus::not_applicable(
            "patch_field has a single cpu.reference implementation; it is verified against an \
             independent brute-force NNF oracle, not a second backend",
        ),
    );
    decls = decls.with(
        VerificationCategory::Perceptual,
        CategoryStatus::not_applicable(
            "the patch field is verified by exact oracle agreement and seeded determinism, not a \
             perceptual metric",
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
