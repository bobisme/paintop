//! The morphology convenience macros `mask.grow@1`, `mask.shrink@1`, and
//! `mask.feather@1` (`OP_CATALOG` §4 note, `plan.md` §4.2 / §17).
//!
//! These are **compatibility macros**: each is exactly equivalent to the explicit
//! signed-distance subgraph
//!
//! ```text
//! mask.to_sdf  →  sdf.offset  →  sdf.to_mask
//! ```
//!
//! and the macro **hides nothing semantic**. A plan that uses a macro
//! [normalizes](expand_plan) during canonicalization into that three-node
//! subgraph, so the normalized plan always exposes the expanded SDF nodes
//! (`plan.md` §17 macro rule) and the macro plan's semantic hash is byte-identical
//! to the hand-written expansion. The macro param → SDF param mapping is fixed and
//! deterministic:
//!
//! | macro             | `mask.to_sdf` | `sdf.offset`              | `sdf.to_mask`                           |
//! |-------------------|---------------|--------------------------|-----------------------------------------|
//! | `mask.grow`       | `threshold`   | `distance_px = radius_px`| hard step (`half_width_px = 0`)         |
//! | `mask.shrink`     | `threshold`   | `distance_px = −radius_px`| hard step                              |
//! | `mask.feather`    | `threshold`   | `distance_px = 0`        | `smoothstep`, `half_width_px`           |
//!
//! Under the `negative-inside` convention (`IR_SPEC` §7.4) growing the region is a
//! *positive* offset (`φ' = φ − d` pushes more of the field below zero), so
//! `mask.grow(r)` offsets by `+r` and `mask.shrink(r)` by `−r`.
//!
//! # Two equivalent realizations
//!
//! For un-normalized execution each macro also carries a `cpu.reference` kernel
//! that composes the three op kernels in-process, so a macro node *also* computes
//! the right coverage directly. That direct kernel and the expanded subgraph are
//! definitionally the same function — [`expand_plan`] is the canonical form the
//! semantic hash is taken over, and the direct kernel is the un-expanded
//! shortcut; the test suite pins both to the same output.
//!
//! # Determinism
//!
//! [`DeterminismTier::Exact`]: every stage (threshold partition, exact EDT,
//! subtraction, clamp+smoothstep) is exact and pointwise-or-whole-domain
//! deterministic, so the macro is exact and platform-stable.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent,
    ImplId, InputRegions, InputSpec, MaskDescriptor, MaskMeaning, Node, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Plan, Rect, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    ScalarType, TestMetadata, ValidRange,
};

use crate::mask_to_sdf::{self, MaskToSdf};
use crate::sdf_ops::{SdfOffset, SdfToMask};

/// The canonical id of the grow macro.
pub const GROW_OP_ID: &str = "mask.grow@1";
/// The canonical id of the shrink macro.
pub const SHRINK_OP_ID: &str = "mask.shrink@1";
/// The canonical id of the feather macro.
pub const FEATHER_OP_ID: &str = "mask.feather@1";

/// A macro input/param was malformed.
pub const E_MACRO_PARAM: &str = "E_MACRO_PARAM";
/// A required macro mask input was absent.
pub const E_MACRO_INPUT: &str = "E_MACRO_INPUT";

/// The kind of morphology macro, fixing its param schema and SDF mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MacroKind {
    /// Grow the region by `radius_px` (`distance_px = +radius_px`, hard edge).
    Grow,
    /// Shrink the region by `radius_px` (`distance_px = −radius_px`, hard edge).
    Shrink,
    /// Feather the edge by `half_width_px` (`distance_px = 0`, smoothstep edge).
    Feather,
}

impl MacroKind {
    /// The macro's canonical op id.
    const fn op_id(self) -> &'static str {
        match self {
            Self::Grow => GROW_OP_ID,
            Self::Shrink => SHRINK_OP_ID,
            Self::Feather => FEATHER_OP_ID,
        }
    }

    /// Identify a macro from an op id string.
    fn from_op_id(op: &str) -> Option<Self> {
        match op {
            GROW_OP_ID => Some(Self::Grow),
            SHRINK_OP_ID => Some(Self::Shrink),
            FEATHER_OP_ID => Some(Self::Feather),
            _ => None,
        }
    }
}

/// Whether `op` is one of the morphology macro op ids.
#[must_use]
pub fn is_macro_op(op: &str) -> bool {
    MacroKind::from_op_id(op).is_some()
}

// ---------------------------------------------------------------------------
// param resolution (shared by the impl kernel and the plan expander)
// ---------------------------------------------------------------------------

/// The resolved SDF parameters a macro maps to.
#[derive(Debug, Clone, Copy)]
struct MacroParams {
    /// The contour threshold passed to `mask.to_sdf`.
    threshold: f64,
    /// The signed offset distance passed to `sdf.offset`.
    distance_px: f64,
    /// The reconstruction feather half-width passed to `sdf.to_mask`.
    half_width_px: f64,
}

/// Read a required non-negative finite float param.
fn require_non_negative(params: &serde_json::Value, name: &str, op: &str) -> Result<f64> {
    let value = params
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_MACRO_PARAM,
                format!("{op} requires a numeric `{name}` parameter"),
            )
        })?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_MACRO_PARAM,
            format!("{op} `{name}` must be a finite value >= 0, got {value}"),
        ))
    }
}

/// Read the optional contour `threshold`, defaulting to the half-coverage level.
fn resolve_threshold(params: &serde_json::Value, op: &str) -> Result<f64> {
    let value = match params.get("threshold") {
        None | Some(serde_json::Value::Null) => mask_to_sdf::DEFAULT_THRESHOLD,
        Some(v) => v.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_MACRO_PARAM,
                format!("{op} `threshold` must be a number"),
            )
        })?,
    };
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_MACRO_PARAM,
            format!("{op} `threshold` must be in [0, 1], got {value}"),
        ))
    }
}

/// Resolve a macro's params to the SDF subgraph params.
fn resolve_macro_params(kind: MacroKind, params: &serde_json::Value) -> Result<MacroParams> {
    let op = kind.op_id();
    let threshold = resolve_threshold(params, op)?;
    match kind {
        MacroKind::Grow => Ok(MacroParams {
            threshold,
            distance_px: require_non_negative(params, "radius_px", op)?,
            half_width_px: 0.0,
        }),
        MacroKind::Shrink => Ok(MacroParams {
            threshold,
            distance_px: -require_non_negative(params, "radius_px", op)?,
            half_width_px: 0.0,
        }),
        MacroKind::Feather => Ok(MacroParams {
            threshold,
            distance_px: 0.0,
            half_width_px: require_non_negative(params, "half_width_px", op)?,
        }),
    }
}

// ---------------------------------------------------------------------------
// plan expansion (the §17 canonicalization rule)
// ---------------------------------------------------------------------------

/// The id of the synthetic `mask.to_sdf` node for a macro node `id`.
#[must_use]
pub fn to_sdf_node_id(id: &str) -> String {
    format!("{id}.to_sdf")
}

/// The id of the synthetic `sdf.offset` node for a macro node `id`.
#[must_use]
pub fn offset_node_id(id: &str) -> String {
    format!("{id}.offset")
}

/// Expand every macro node in `plan` into its explicit `mask.to_sdf → sdf.offset
/// → sdf.to_mask` subgraph, leaving non-macro nodes untouched.
///
/// The terminal `sdf.to_mask` node keeps the macro node's original `id` and
/// `mask` output port, so every downstream reference `node:<id>/mask` stays valid
/// with no rewrite. The two synthetic upstream nodes take deterministic
/// `<id>.to_sdf` / `<id>.offset` ids. The result is the canonical normalized form
/// whose semantic hash equals a hand-written expansion's.
///
/// # Errors
/// [`E_MACRO_INPUT`] if a macro node has no `mask` input, or [`E_MACRO_PARAM`] if
/// its params are malformed.
pub fn expand_plan(plan: &Plan) -> Result<Plan> {
    let mut expanded = plan.clone();
    let mut nodes = Vec::with_capacity(expanded.nodes.len());
    for node in &expanded.nodes {
        if let Some(kind) = MacroKind::from_op_id(&node.op) {
            let params =
                resolve_macro_params(kind, &serde_json::Value::Object(node.params.clone()))?;
            nodes.extend(expand_node(node, params)?);
        } else {
            nodes.push(node.clone());
        }
    }
    expanded.nodes = nodes;
    Ok(expanded)
}

/// Expand a single macro `node` into its three SDF nodes.
fn expand_node(node: &Node, params: MacroParams) -> Result<Vec<Node>> {
    let mask_ref = node.inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_MACRO_INPUT,
            format!("{} node `{}` requires a `mask` input", node.op, node.id),
        )
    })?;

    let to_sdf_id = to_sdf_node_id(&node.id);
    let offset_id = offset_node_id(&node.id);

    let to_sdf = Node {
        id: to_sdf_id.clone(),
        op: mask_to_sdf::OP_ID.to_owned(),
        inputs: btree_one("mask", mask_ref),
        params: json_object([("threshold", serde_json::json!(params.threshold))]),
        hints: serde_json::Map::new(),
        extensions: paintop_ir::Extensions::default(),
    };
    let offset = Node {
        id: offset_id.clone(),
        op: crate::sdf_ops::OFFSET_OP_ID.to_owned(),
        inputs: btree_one("sdf", &format!("node:{to_sdf_id}/sdf")),
        params: json_object([("distance_px", serde_json::json!(params.distance_px))]),
        hints: serde_json::Map::new(),
        extensions: paintop_ir::Extensions::default(),
    };
    let to_mask = Node {
        id: node.id.clone(),
        op: crate::sdf_ops::TO_MASK_OP_ID.to_owned(),
        inputs: btree_one("sdf", &format!("node:{offset_id}/sdf")),
        params: json_object([
            ("profile", serde_json::json!("smoothstep")),
            ("half_width_px", serde_json::json!(params.half_width_px)),
        ]),
        hints: serde_json::Map::new(),
        extensions: paintop_ir::Extensions::default(),
    };
    Ok(vec![to_sdf, offset, to_mask])
}

/// A single-entry `in` map.
fn btree_one(port: &str, reference: &str) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    map.insert(port.to_owned(), reference.to_owned());
    map
}

/// Build a JSON object param map from key/value pairs.
fn json_object<const N: usize>(
    entries: [(&str, serde_json::Value); N],
) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (k, v) in entries {
        map.insert(k.to_owned(), v);
    }
    map
}

// ---------------------------------------------------------------------------
// the macro op (manifest + direct kernel)
// ---------------------------------------------------------------------------

/// A morphology macro operation (`mask.grow` / `mask.shrink` / `mask.feather`).
#[derive(Debug, Clone, Copy)]
pub struct MaskMacro {
    kind: MacroKind,
}

impl MaskMacro {
    /// The `mask.grow@1` macro.
    #[must_use]
    pub const fn grow() -> Self {
        Self {
            kind: MacroKind::Grow,
        }
    }

    /// The `mask.shrink@1` macro.
    #[must_use]
    pub const fn shrink() -> Self {
        Self {
            kind: MacroKind::Shrink,
        }
    }

    /// The `mask.feather@1` macro.
    #[must_use]
    pub const fn feather() -> Self {
        Self {
            kind: MacroKind::Feather,
        }
    }

    /// The declared manifest for `mask.grow@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn grow_manifest() -> Result<OperationManifest> {
        Self::grow().manifest(
            "Grow (dilate) a mask region by a physical pixel radius. Compatibility macro: \
             normalizes to mask.to_sdf -> sdf.offset(distance_px = +radius_px) -> sdf.to_mask \
             (hard edge).",
            "The grown coverage mask in [0, 1].",
            radius_param("The physical pixel radius to grow the region by; >= 0."),
            "mask.grow is an exact macro over the EDT/SDF path verified by hash-identity with the \
             hand-written mask.to_sdf -> sdf.offset -> sdf.to_mask expansion; there is no \
             perceptual metric",
        )
    }

    /// The declared manifest for `mask.shrink@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn shrink_manifest() -> Result<OperationManifest> {
        Self::shrink().manifest(
            "Shrink (erode) a mask region by a physical pixel radius. Compatibility macro: \
             normalizes to mask.to_sdf -> sdf.offset(distance_px = -radius_px) -> sdf.to_mask \
             (hard edge).",
            "The shrunk coverage mask in [0, 1].",
            radius_param("The physical pixel radius to shrink the region by; >= 0."),
            "mask.shrink is an exact macro over the EDT/SDF path verified by hash-identity with \
             the hand-written expansion; there is no perceptual metric",
        )
    }

    /// The declared manifest for `mask.feather@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn feather_manifest() -> Result<OperationManifest> {
        Self::feather().manifest(
            "Feather (soften) a mask edge by a physical pixel half-width. Compatibility macro: \
             normalizes to mask.to_sdf -> sdf.offset(0) -> sdf.to_mask(smoothstep, half_width_px).",
            "The feathered coverage mask in [0, 1].",
            ParamSpec {
                name: "half_width_px".to_owned(),
                ty: ParamType::Float,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "Half the physical feather width in pixels; the soft edge spans \
                      2*half_width_px centered on the contour. >= 0."
                    .to_owned(),
            },
            "mask.feather is an exact macro over the EDT/SDF path verified by hash-identity with \
             the hand-written expansion and a measured feather width; there is no perceptual \
             metric",
        )
    }

    /// Build the manifest for this macro.
    fn manifest(
        self,
        summary: &str,
        out_doc: &str,
        size_param: ParamSpec,
        perceptual_reason: &str,
    ) -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: self.kind.op_id().parse()?,
            impl_version: 1,
            summary: summary.to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                required: true,
                doc: "The coverage or selection mask to transform.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                doc: out_doc.to_owned(),
            }],
            params: vec![size_param, threshold_param()],
            implementations: vec![reference_impl()?],
            test: macro_test_metadata(perceptual_reason),
        })
    }
}

impl OpContract for MaskMacro {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = match inputs.get("mask") {
            Some(ResourceDescriptor::Mask(d)) => d.extent,
            Some(_) => {
                return Err(Error::new(
                    ErrorClass::Type,
                    E_MACRO_INPUT,
                    format!("{} input `mask` must be a Mask", self.kind.op_id()),
                ));
            }
            None => {
                return Err(Error::new(
                    ErrorClass::Reference,
                    E_MACRO_INPUT,
                    format!("{} requires a `mask` input", self.kind.op_id()),
                ));
            }
        };
        let mut out = OutputDescriptors::new();
        out.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(mask_descriptor(extent)),
        );
        Ok(out)
    }
    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // The macro expands to an EDT-backed subgraph, so it needs the whole mask.
        let mut regions = InputRegions::new();
        if let Some(d) = inputs.get("mask") {
            let extent = d.extent();
            regions.insert(
                "mask".to_owned(),
                Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
            );
        }
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Mask(mask)) = outputs.get("mask") else {
            return Ok(vec![AssertionResult::fail(
                "produces_mask",
                "no `mask` output produced",
            )]);
        };
        let unit = ValidRange::Bounded { min: 0.0, max: 1.0 };
        Ok(vec![
            AssertionResult::pass("produces_mask"),
            if mask.range == unit {
                AssertionResult::pass("coverage_in_unit_range")
            } else {
                AssertionResult::fail("coverage_in_unit_range", "mask range is not [0, 1]")
            },
        ])
    }
}

impl OpImplementation for MaskMacro {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = inputs.get("mask").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_MACRO_INPUT,
                format!("{} requires a `mask` input value", self.kind.op_id()),
            )
        })?;
        let p = resolve_macro_params(self.kind, params)?;

        // Compose the three op kernels in-process: mask.to_sdf -> sdf.offset ->
        // sdf.to_mask. This is definitionally the expanded subgraph.
        let sdf = run_single(
            &MaskToSdf::new(),
            "mask",
            mask.clone(),
            &serde_json::json!({ "threshold": p.threshold }),
            "sdf",
        )?;
        let offset = run_single(
            &SdfOffset::new(),
            "sdf",
            sdf,
            &serde_json::json!({ "distance_px": p.distance_px }),
            "sdf",
        )?;
        let mask_out = run_single(
            &SdfToMask::new(),
            "sdf",
            offset,
            &serde_json::json!({ "profile": "smoothstep", "half_width_px": p.half_width_px }),
            "mask",
        )?;

        let mut out = OutputValues::new();
        out.insert("mask".to_owned(), mask_out);
        Ok(out)
    }
}

/// Run one op kernel on a single named input/output, returning the output value.
fn run_single(
    op: &dyn OpImplementation,
    in_port: &str,
    value: ResourceValue,
    params: &serde_json::Value,
    out_port: &str,
) -> std::result::Result<ResourceValue, Error> {
    let mut inputs = InputValues::new();
    inputs.insert(in_port.to_owned(), value);
    let mut out = op.compute(&inputs, params)?;
    out.remove(out_port).ok_or_else(|| {
        Error::new(
            ErrorClass::Execution,
            E_MACRO_PARAM,
            format!("macro sub-op produced no `{out_port}` output"),
        )
    })
}

// ---------------------------------------------------------------------------
// shared manifest helpers
// ---------------------------------------------------------------------------

/// The coverage-mask descriptor produced by a macro.
const fn mask_descriptor(extent: Extent) -> MaskDescriptor {
    MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The `radius_px` size param shared by grow/shrink.
fn radius_param(doc: &str) -> ParamSpec {
    ParamSpec {
        name: "radius_px".to_owned(),
        ty: ParamType::Float,
        unit: Some(ParamUnit::Pixels),
        required: true,
        default: None,
        choices: vec![],
        doc: doc.to_owned(),
    }
}

/// The optional `threshold` param shared by every macro (forwarded to
/// `mask.to_sdf`).
fn threshold_param() -> ParamSpec {
    ParamSpec {
        name: "threshold".to_owned(),
        ty: ParamType::Float,
        unit: None,
        required: false,
        default: Some(serde_json::json!(mask_to_sdf::DEFAULT_THRESHOLD)),
        choices: vec![],
        doc: "The contour threshold forwarded to mask.to_sdf (coverage >= threshold is inside); \
              default 0.5."
            .to_owned(),
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for an exact macro op.
fn macro_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
        CategoryStatus::not_applicable(perceptual_reason.to_owned()),
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
