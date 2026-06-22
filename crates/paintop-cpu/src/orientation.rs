//! The `field.orientation@1` operation: the **orientation field** and
//! **coherence** of a structure tensor (`OP_CATALOG` §10.4; `plan.md` §1428).
//!
//! Given the symmetric structure tensor `J = [[Jxx, Jxy], [Jxy, Jyy]]` (the
//! three-component `Field3` produced by
//! [`filter.structure_tensor`](crate::structure_tensor)), this op diagonalizes
//! `J` per pixel and emits:
//!
//! - **`orientation`** — a unit [`Field2`](paintop_ir::ResourceKind::Field2) along
//!   the **dominant local orientation**: the eigenvector of the *smaller*
//!   eigenvalue, i.e. the direction along which the signal varies *least* (the
//!   direction of an edge / the stripes of a grating, not across them). For an
//!   isotropic (degenerate) tensor there is no preferred direction and the vector
//!   is the zero vector.
//! - **`coherence`** — a scalar [`Field1`](paintop_ir::ResourceKind::Field1) in
//!   `[0, 1]` measuring how anisotropic the tensor is:
//!   `((λ_max − λ_min) / (λ_max + λ_min))²`, defined to `0` when the trace is
//!   `0`. Coherence is `≈ 1` on a clean straight edge (one eigenvalue dominates)
//!   and `≈ 0` on isotropic content (equal eigenvalues).
//!
//! # Eigenvector sign convention (explicit and deterministic)
//!
//! An eigenvector is only defined up to sign, so the op fixes a **canonical
//! representative**: the unit eigenvector `(ux, uy)` is flipped, if necessary, so
//! that `ux > 0`, or `ux == 0 && uy >= 0` when `ux` is exactly zero. This makes
//! the orientation field a deterministic single-valued function of the tensor (no
//! arbitrary per-pixel sign), at the cost of a discontinuity where `ux` changes
//! sign — orientation is a *line* field, and any sign convention must cut it
//! somewhere; this is the cut, documented and fixed. The zero vector (degenerate
//! tensor) is left unflipped.
//!
//! # Eigenanalysis (closed form)
//!
//! For the symmetric 2×2 tensor the eigenvalues are
//! `λ± = m ± sqrt(d² + Jxy²)` with `m = (Jxx + Jyy)/2`, `d = (Jxx − Jyy)/2`. The
//! smaller eigenvalue's eigenvector is taken from the standard 2×2 symmetric
//! formula and normalized. The whole computation is a fixed-order `f64`
//! evaluation rounded once to `f32`, so the op is bit-identical on reruns; it
//! involves a `sqrt`/divide and so declares [`Bounded`](DeterminismTier::Bounded)
//! against an independent reference.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent,
    FieldArity, FieldDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, Rect, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, ScalarType, SemanticRole, TestMetadata, VectorEncoding,
    VectorNormalization, VectorSpace,
};

/// The canonical id of the orientation operation.
pub const ORIENTATION_OP_ID: &str = "field.orientation@1";

/// The `tensor` input was absent or carried an unsupported descriptor.
pub const E_ORIENTATION_INPUT: &str = "E_ORIENTATION_INPUT";

/// The `field.orientation@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Orientation;

impl Orientation {
    /// Construct the orientation operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `field.orientation@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ORIENTATION_OP_ID.parse()?,
            impl_version: 1,
            summary: "Diagonalize a structure-tensor Field3 (Jxx, Jxy, Jyy) per pixel into a unit \
                      Field2 orientation (smaller-eigenvalue eigenvector, canonical sign ux>=0) \
                      and a Field1 coherence ((l_max - l_min)/(l_max + l_min))^2."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "tensor".to_owned(),
                kind: ResourceKind::Field3,
                required: true,
                doc: "The structure tensor's (Jxx, Jxy, Jyy) per pixel (a Field3).".to_owned(),
            }],
            outputs: vec![
                OutputSpec {
                    name: "orientation".to_owned(),
                    kind: ResourceKind::Field2,
                    doc: "Unit orientation vector (smaller-eigenvalue eigenvector; zero where the \
                          tensor is isotropic)."
                        .to_owned(),
                },
                OutputSpec {
                    name: "coherence".to_owned(),
                    kind: ResourceKind::Field1,
                    doc: "Anisotropy coherence in [0, 1]: ((l_max - l_min)/(l_max + l_min))^2."
                        .to_owned(),
                },
            ],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: orientation_test_metadata(),
        })
    }
}

/// The input tensor's extent and Field3 descriptor.
fn tensor_extent(descriptor: &ResourceDescriptor) -> Result<Extent> {
    match descriptor {
        ResourceDescriptor::Field3(d) => Ok(d.extent),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_ORIENTATION_INPUT,
            "field.orientation `tensor` must be a Field3 (Jxx, Jxy, Jyy) resource".to_owned(),
        )),
    }
}

/// The unit-vector `Field2` descriptor an orientation field uses for `extent`.
const fn orientation_descriptor(extent: Extent) -> FieldDescriptor {
    FieldDescriptor {
        arity: FieldArity::Field2,
        extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Flow,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: Some(VectorSpace::Tangent),
        normalization: Some(VectorNormalization::Unit),
        encoding: Some(VectorEncoding::SignedVector),
    }
}

/// The scalar `Field1` descriptor a coherence map uses for `extent`.
const fn coherence_descriptor(extent: Extent) -> FieldDescriptor {
    FieldDescriptor {
        arity: FieldArity::Field1,
        extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Confidence,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    }
}

impl OpContract for Orientation {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("tensor".to_owned(), ResourceKind::Field3)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("orientation".to_owned(), ResourceKind::Field2),
            ("coherence".to_owned(), ResourceKind::Field1),
        ]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("tensor").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ORIENTATION_INPUT,
                "field.orientation requires a `tensor` resource".to_owned(),
            )
        })?;
        let extent = tensor_extent(input)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "orientation".to_owned(),
            ResourceDescriptor::Field2(orientation_descriptor(extent)),
        );
        out.insert(
            "coherence".to_owned(),
            ResourceDescriptor::Field1(coherence_descriptor(extent)),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: each output pixel reads exactly its own tensor sample.
        let input = inputs.get("tensor").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ORIENTATION_INPUT,
                "field.orientation requires a `tensor` resource".to_owned(),
            )
        })?;
        let extent = tensor_extent(input)?;
        let full = Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height));
        // The union of both requested output regions (they share one extent).
        let mut needed: Option<Rect> = None;
        for port in ["orientation", "coherence"] {
            if let Some(region) = requested_outputs.get(port) {
                needed = Some(needed.map_or(*region, |acc| acc.union(*region)));
            }
        }
        let mut regions = InputRegions::new();
        if let Some(region) = needed {
            regions.insert("tensor".to_owned(), region.intersect(full));
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let orientation = matches!(
            outputs.get("orientation"),
            Some(ResourceDescriptor::Field2(_))
        );
        let coherence = matches!(
            outputs.get("coherence"),
            Some(ResourceDescriptor::Field1(_))
        );
        Ok(vec![if orientation && coherence {
            AssertionResult::pass("produces_orientation_and_coherence")
        } else {
            AssertionResult::fail(
                "produces_orientation_and_coherence",
                "expected an `orientation` Field2 and a `coherence` Field1",
            )
        }])
    }
}

impl OpImplementation for Orientation {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("tensor").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ORIENTATION_INPUT,
                "field.orientation requires a `tensor` value".to_owned(),
            )
        })?;
        let extent = tensor_extent(input.descriptor())?;
        let tensor = input.samples();
        let n = (extent.width as usize) * (extent.height as usize);

        let mut orientation = vec![0.0_f32; n * 2];
        let mut coherence = vec![0.0_f32; n];
        for i in 0..n {
            let jxx = f64::from(tensor[i * 3]);
            let jxy = f64::from(tensor[i * 3 + 1]);
            let jyy = f64::from(tensor[i * 3 + 2]);
            let (ux, uy, coh) = analyze(jxx, jxy, jyy);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "unit components and coherence computed in f64, stored as f32 samples"
            )]
            {
                orientation[i * 2] = ux as f32;
                orientation[i * 2 + 1] = uy as f32;
                coherence[i] = coh as f32;
            }
        }

        let mut out = OutputValues::new();
        out.insert(
            "orientation".to_owned(),
            ResourceValue::new(
                ResourceDescriptor::Field2(orientation_descriptor(extent)),
                2,
                orientation,
            )
            .map_err(|actual| buffer_error("orientation", actual))?,
        );
        out.insert(
            "coherence".to_owned(),
            ResourceValue::new(
                ResourceDescriptor::Field1(coherence_descriptor(extent)),
                1,
                coherence,
            )
            .map_err(|actual| buffer_error("coherence", actual))?,
        );
        Ok(out)
    }
}

/// A buffer-length-mismatch execution error for an output port.
fn buffer_error(port: &str, actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_ORIENTATION_INPUT,
        format!("field.orientation produced a `{port}` buffer of unexpected length {actual}"),
    )
}

/// Diagonalize the symmetric tensor `[[jxx, jxy], [jxy, jyy]]` and return the
/// canonical unit orientation `(ux, uy)` (smaller-eigenvalue eigenvector) and the
/// coherence in `[0, 1]`.
///
/// The orientation is the zero vector and coherence `0` for a degenerate
/// (isotropic / all-zero) tensor.
pub(crate) fn analyze(jxx: f64, jxy: f64, jyy: f64) -> (f64, f64, f64) {
    let m = (jxx + jyy) * 0.5;
    let d = (jxx - jyy) * 0.5;
    let disc = d.mul_add(d, jxy * jxy).max(0.0).sqrt();
    let lambda_max = m + disc;
    let lambda_min = m - disc;

    // Coherence: ((l_max - l_min) / (l_max + l_min))^2 = (disc / m)^2, defined to
    // 0 when the trace (2m) is non-positive (a zero or degenerate tensor).
    let trace = lambda_max + lambda_min; // == 2m
    let coherence = if trace > 0.0 {
        let c = disc / m;
        (c * c).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Orientation: the eigenvector of the *smaller* eigenvalue (the direction of
    // least variation — along edges). For a symmetric 2x2 matrix the eigenvector
    // of eigenvalue λ satisfies (Jxx - λ) ux + Jxy uy = 0, giving the direction
    // (Jxy, λ - Jxx) (or the perpendicular when Jxy is ~0). An isotropic tensor
    // (disc ~ 0) has no preferred direction -> the zero vector.
    if disc <= f64::EPSILON {
        return (0.0, 0.0, coherence);
    }
    // Eigenvector for lambda_min: direction (Jxy, lambda_min - Jxx).
    let mut vx = jxy;
    let mut vy = lambda_min - jxx;
    let len = vx.hypot(vy);
    if len <= f64::EPSILON {
        // Diagonal tensor: pick the axis of the smaller diagonal entry.
        if jxx <= jyy {
            // Jxx is the smaller eigenvalue -> least variation along x.
            vx = 1.0;
            vy = 0.0;
        } else {
            vx = 0.0;
            vy = 1.0;
        }
    } else {
        vx /= len;
        vy /= len;
    }
    let (ux, uy) = canonical_sign(vx, vy);
    (ux, uy, coherence)
}

/// Flip a unit vector to its canonical representative: `ux > 0`, or `ux == 0 &&
/// uy >= 0` when `ux` is exactly zero. The zero vector is returned unchanged.
fn canonical_sign(ux: f64, uy: f64) -> (f64, f64) {
    if ux > 0.0 {
        (ux, uy)
    } else if ux < 0.0 {
        (-ux, -uy)
    } else if uy >= 0.0 {
        (ux, uy)
    } else {
        (ux, -uy)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `field.orientation@1`: a bounded, single-
/// reference pointwise op. Differential does not apply (one implementation).
/// Perceptual is not applicable: correctness is the analytic eigenanalysis
/// property set (grating-angle recovery, coherence extremes, sign-convention
/// determinism, rotation covariance), not a perceptual metric.
fn orientation_test_metadata() -> TestMetadata {
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
            "field.orientation is a closed-form per-pixel eigenanalysis verified by analytic \
             properties (oriented-grating angle recovery, coherence ~1 on a clean edge and ~0 on \
             isotropic noise, deterministic eigenvector sign convention, rotation covariance); \
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
