//! The `field.noise@1` and `field.fbm@1` operations: **hash-based deterministic
//! procedural noise** (`OP_CATALOG` §11; `plan.md` §1428, §1444).
//!
//! Both ops synthesize a scalar [`Field1`](paintop_ir::ResourceKind::Field1) from
//! an explicit extent and a 64-bit `seed`. Their defining property is that every
//! sample is a pure **hash of its integer-lattice coordinate** (and the seed) —
//! never drawn from a stateful sequential RNG. That makes the value at a lattice
//! point `(gx, gy)` a function of `(gx, gy, seed)` *alone*:
//!
//! - **reproducible** — bit-identical across runs on a fixed backend;
//! - **tiling / evaluation-order invariant** — the value at a coordinate does not
//!   depend on which tile, in what order, or how many neighbors were evaluated, so
//!   a tiled or reordered evaluation produces the *same* field as a whole-frame
//!   one (M4 exit criterion 4).
//!
//! # Value noise (`field.noise@1`)
//!
//! A classic **value-noise** lattice: each integer lattice point hashes to a
//! pseudo-random value in `[-1, 1]`, and a continuous sample is the smoothstep
//! (quintic Perlin fade) bilinear interpolation of the four surrounding lattice
//! values. `frequency` scales how many lattice cells span a pixel; the lattice is
//! evaluated in a fixed `f64` order and rounded once to `f32`, so the op is
//! [`Bounded`](DeterminismTier::Bounded) (the fade uses no transcendental, but the
//! `frequency` multiply is a float scale, asserted within tolerance against an
//! independent reference rather than bit-exactly across platforms).
//!
//! # Fractional Brownian motion (`field.fbm@1`)
//!
//! `field.fbm@1` sums `octaves` copies of the value noise, each at
//! `frequency * lacunarity^o` and amplitude `gain^o`. The sum is **normalized**
//! by the total amplitude `Σ gain^o` so the result stays in `[-1, 1]` regardless
//! of the octave count — the documented octave normalization (M4 acceptance).
//! Each octave is seed-decorrelated by mixing the octave index into the seed, so
//! octaves do not align into visible banding.
//!
//! # Range
//!
//! Both ops emit a `Field1` whose samples lie in `[-1, 1]` by construction (value
//! noise interpolates values in `[-1, 1]`; fbm is amplitude-normalized). The range
//! is documented here rather than enforced by a `ValidRange` (a `Field1` carries
//! no range policy), and is asserted by the property tests.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, FieldArity, FieldDescriptor, ImplId, InputRegions, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Result, RoiCategory, RoiPolicy, ScalarType, SemanticRole, TestMetadata,
};

/// The canonical id of the value-noise operation.
pub const NOISE_OP_ID: &str = "field.noise@1";

/// The canonical id of the fbm operation.
pub const FBM_OP_ID: &str = "field.fbm@1";

/// A noise parameter (`width`, `height`, `frequency`, `seed`, `octaves`, …) was
/// missing, the wrong shape, or held a non-finite / out-of-range value.
pub const E_NOISE_PARAM: &str = "E_NOISE_PARAM";

/// The execution buffer length disagreed with the declared extent.
pub const E_NOISE_BUFFER: &str = "E_NOISE_BUFFER";

/// The largest octave count an fbm request may ask for, bounding the per-sample
/// cost. Beyond a few dozen octaves the amplitude `gain^o` underflows to nothing
/// anyway, so this is a generous cap rather than a tight one.
const MAX_OCTAVES: u32 = 32;

/// Mix a 64-bit value (the `SplitMix64` finalizer). A bijective avalanche mix
/// used to fold lattice coordinates and the seed into a well-distributed 64-bit
/// hash.
const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hash an integer lattice coordinate `(gx, gy)` and `seed` to a value in
/// `[-1, 1]`.
///
/// The coordinates are taken as `i64` and reinterpreted as `u64` (two's
/// complement) so negative lattice indices hash cleanly; they are folded with the
/// seed through [`mix64`] in a fixed order. The top 24 bits of the final hash map
/// to a value in `[-1, 1]` with a uniform step, independent of platform endianness
/// or float rounding (the division is by an exact power of two).
fn lattice_value(gx: i64, gy: i64, seed: u64) -> f64 {
    // Two's-complement bit-reinterpretation of the signed lattice index; this is
    // a value-preserving bit cast (not a lossy numeric one), so negative indices
    // hash cleanly and deterministically.
    #[allow(
        clippy::cast_sign_loss,
        reason = "deliberate two's-complement bit reinterpretation of the lattice index for hashing"
    )]
    let ux = gx as u64;
    #[allow(
        clippy::cast_sign_loss,
        reason = "deliberate two's-complement bit reinterpretation of the lattice index for hashing"
    )]
    let uy = gy as u64;
    let hx = mix64(seed ^ ux.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let h = mix64(hx ^ uy.wrapping_mul(0xc2b2_ae3d_27d4_eb4f));
    // Top 24 bits -> [0, 2^24), exactly representable in f64, mapped to [-1, 1].
    // The shifted value is < 2^24 so the u64->f64 cast is exact.
    let top24 = h >> 40;
    let unit = f64::from(u32::try_from(top24).unwrap_or(0)) / f64::from(1u32 << 24); // [0, 1)
    unit.mul_add(2.0, -1.0)
}

/// The quintic Perlin fade `6t^5 - 15t^4 + 10t^3`: a smoothstep with zero first
/// and second derivatives at `0` and `1`, so the interpolated noise is `C^2`.
fn fade(t: f64) -> f64 {
    t * t * t * t.mul_add(t.mul_add(6.0, -15.0), 10.0)
}

/// Linear interpolation `a + t (b - a)`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    (b - a).mul_add(t, a)
}

/// A single octave of value noise at continuous lattice coordinate `(x, y)`.
///
/// The four surrounding lattice points are hashed and bilinearly blended with the
/// quintic fade. The result lies in `[-1, 1]` (a convex blend of values already in
/// `[-1, 1]`).
fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let x0 = x.floor();
    let y0 = y.floor();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "floored continuous coordinate to its integer lattice cell; extents are bounded \
                  far below i64 range so no truncation occurs in practice"
    )]
    let gx0 = x0 as i64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "floored continuous coordinate to its integer lattice cell; extents are bounded \
                  far below i64 range so no truncation occurs in practice"
    )]
    let gy0 = y0 as i64;
    let tx = fade(x - x0);
    let ty = fade(y - y0);

    let v00 = lattice_value(gx0, gy0, seed);
    let v10 = lattice_value(gx0 + 1, gy0, seed);
    let v01 = lattice_value(gx0, gy0 + 1, seed);
    let v11 = lattice_value(gx0 + 1, gy0 + 1, seed);

    let top = lerp(v00, v10, tx);
    let bottom = lerp(v01, v11, tx);
    lerp(top, bottom, ty)
}

/// The resolved geometry/params common to both noise ops.
#[derive(Debug, Clone, Copy)]
struct NoiseRequest {
    extent: Extent,
    frequency: f64,
    seed: u64,
}

impl NoiseRequest {
    /// Resolve the shared `width`/`height`/`frequency`/`seed` params.
    fn resolve(op: &str, params: &serde_json::Value) -> Result<Self> {
        let width = u32_param(op, params, "width")?;
        let height = u32_param(op, params, "height")?;
        let frequency = optional_positive(op, params, "frequency", 1.0)?;
        let seed = seed_param(op, params)?;
        Ok(Self {
            extent: Extent::new(width, height),
            frequency,
            seed,
        })
    }
}

/// The fbm-specific params (octaves, lacunarity, gain) plus the shared base.
#[derive(Debug, Clone, Copy)]
struct FbmRequest {
    base: NoiseRequest,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
}

impl FbmRequest {
    /// Resolve the fbm params, validating octave/lacunarity/gain bounds.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let base = NoiseRequest::resolve(FBM_OP_ID, params)?;
        let octaves = params.get("octaves").map_or(Ok(4_u32), |v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    param_error(
                        FBM_OP_ID,
                        "`octaves` must be a non-negative integer",
                        "octaves",
                        v,
                    )
                })
        })?;
        if octaves == 0 || octaves > MAX_OCTAVES {
            return Err(Error::new(
                ErrorClass::Schema,
                E_NOISE_PARAM,
                format!("field.fbm `octaves` must be in 1..={MAX_OCTAVES}, got {octaves}"),
            ));
        }
        let lacunarity = optional_positive(FBM_OP_ID, params, "lacunarity", 2.0)?;
        let gain = optional_gain(params)?;
        Ok(Self {
            base,
            octaves,
            lacunarity,
            gain,
        })
    }

    /// Evaluate the amplitude-normalized fbm at continuous coordinate `(x, y)`,
    /// where `(x, y)` are already scaled by the base frequency.
    ///
    /// The sum runs in a fixed octave order (coarse seed-decorrelation per octave)
    /// and is divided by the total amplitude so the value stays in `[-1, 1]`.
    fn evaluate(&self, x: f64, y: f64) -> f64 {
        let mut frequency = 1.0_f64;
        let mut amplitude = 1.0_f64;
        let mut sum = 0.0_f64;
        let mut total_amplitude = 0.0_f64;
        for octave in 0..self.octaves {
            // Decorrelate octaves: fold the octave index into the seed so the
            // lattices do not align into a visible grid.
            let octave_seed = mix64(self.base.seed ^ (u64::from(octave)).wrapping_add(0x1));
            sum = amplitude.mul_add(value_noise(x * frequency, y * frequency, octave_seed), sum);
            total_amplitude += amplitude;
            frequency *= self.lacunarity;
            amplitude *= self.gain;
        }
        if total_amplitude > 0.0 {
            sum / total_amplitude
        } else {
            0.0
        }
    }
}

/// The scalar `Field1` descriptor a procedural-noise field uses for `extent`: a
/// generic-material scalar in `[-1, 1]` under the fixed coordinate convention.
const fn noise_descriptor(extent: Extent) -> FieldDescriptor {
    FieldDescriptor {
        arity: FieldArity::Field1,
        extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    }
}

/// Read a required non-negative-integer param as a `u32`.
fn u32_param(op: &str, params: &serde_json::Value, name: &str) -> Result<u32> {
    let value = params.get(name).ok_or_else(|| {
        param_error(
            op,
            "missing required parameter",
            name,
            &serde_json::Value::Null,
        )
    })?;
    let n = value
        .as_u64()
        .ok_or_else(|| param_error(op, "must be a non-negative integer", name, value))?;
    u32::try_from(n).map_err(|_| param_error(op, "does not fit in u32", name, value))
}

/// Read the optional `seed` param (a non-negative integer), defaulting to `0`.
fn seed_param(op: &str, params: &serde_json::Value) -> Result<u64> {
    params.get("seed").map_or(Ok(0), |value| {
        value
            .as_u64()
            .ok_or_else(|| param_error(op, "`seed` must be a non-negative integer", "seed", value))
    })
}

/// Read an optional strictly-positive finite float param, defaulting when absent.
fn optional_positive(
    op: &str,
    params: &serde_json::Value,
    name: &str,
    default: f64,
) -> Result<f64> {
    let Some(value) = params.get(name) else {
        return Ok(default);
    };
    let n = value
        .as_f64()
        .ok_or_else(|| param_error(op, "must be a number", name, value))?;
    if n.is_finite() && n > 0.0 {
        Ok(n)
    } else {
        Err(param_error(
            op,
            "must be a finite positive number",
            name,
            value,
        ))
    }
}

/// Read the optional `gain` param: a finite value in `(0, 1]` (a persistence
/// factor; `>1` would diverge the amplitude-normalized sum's intent), default
/// `0.5`.
fn optional_gain(params: &serde_json::Value) -> Result<f64> {
    let Some(value) = params.get("gain") else {
        return Ok(0.5);
    };
    let n = value
        .as_f64()
        .ok_or_else(|| param_error(FBM_OP_ID, "must be a number", "gain", value))?;
    if n.is_finite() && n > 0.0 && n <= 1.0 {
        Ok(n)
    } else {
        Err(param_error(
            FBM_OP_ID,
            "must be a finite number in (0, 1]",
            "gain",
            value,
        ))
    }
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(op: &str, detail: &str, name: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_NOISE_PARAM,
        format!("{op} parameter `{name}`: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// A buffer-length-mismatch execution error.
fn buffer_error(op: &str, actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_NOISE_BUFFER,
        format!("{op} produced a field buffer of unexpected length {actual}"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations shared by both noise ops: a bounded source op
/// verified by analytic properties (tiling invariance, seed sensitivity, range,
/// fbm normalization), with no input to diff against and no perceptual metric.
fn noise_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
        CategoryStatus::not_applicable(perceptual_reason),
    );
    TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

/// The shared `width`/`height`/`frequency`/`seed` param specs.
fn base_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            name: "width".to_owned(),
            ty: ParamType::Integer,
            unit: Some(ParamUnit::Pixels),
            required: true,
            default: None,
            choices: vec![],
            doc: "The output field width in pixels.".to_owned(),
        },
        ParamSpec {
            name: "height".to_owned(),
            ty: ParamType::Integer,
            unit: Some(ParamUnit::Pixels),
            required: true,
            default: None,
            choices: vec![],
            doc: "The output field height in pixels.".to_owned(),
        },
        ParamSpec {
            name: "frequency".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(1.0)),
            choices: vec![],
            doc: "Lattice cells per 64 pixels (the base spatial frequency); strictly positive."
                .to_owned(),
        },
        ParamSpec {
            name: "seed".to_owned(),
            ty: ParamType::Seed,
            unit: None,
            required: false,
            default: Some(serde_json::json!(0)),
            choices: vec![],
            doc:
                "The 64-bit seed; the noise is a hash of (coordinate, seed), not a sequential RNG."
                    .to_owned(),
        },
    ]
}

/// The base spatial scale: `frequency` lattice cells per this many pixels. Fixing
/// it as a constant keeps `frequency = 1` a gentle, visible noise rather than one
/// lattice cell per pixel (which would alias).
const FREQUENCY_PIXELS: f64 = 64.0;

/// The `field.noise@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Noise;

impl Noise {
    /// Construct the value-noise operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `field.noise@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: NOISE_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Synthesize a scalar Field1 of hash-based value noise in [-1, 1]: each lattice \
                      point is a hash of (coordinate, seed) (tiling- and order-invariant, not a \
                      sequential RNG), bilinearly blended with a quintic fade."
                    .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![],
            outputs: vec![OutputSpec {
                name: "field".to_owned(),
                kind: paintop_ir::ResourceKind::Field1,
                doc: "The value-noise scalar field in [-1, 1].".to_owned(),
            }],
            params: base_params(),
            implementations: vec![reference_impl()?],
            test: noise_test_metadata(
                "field.noise is hash-based value noise verified by analytic properties (bit-exact \
                 reruns, tiling/evaluation-order invariance via hash-of-coordinate, seed \
                 sensitivity, [-1, 1] range); there is no perceptual-quality metric to apply",
            ),
        })
    }
}

impl OpContract for Noise {
    fn declared_inputs(&self) -> Vec<(String, paintop_ir::ResourceKind)> {
        vec![]
    }

    fn declared_outputs(&self) -> Vec<(String, paintop_ir::ResourceKind)> {
        vec![("field".to_owned(), paintop_ir::ResourceKind::Field1)]
    }

    fn infer_outputs(
        &self,
        _inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let request = NoiseRequest::resolve(NOISE_OP_ID, params)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "field".to_owned(),
            paintop_ir::ResourceDescriptor::Field1(noise_descriptor(request.extent)),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(InputRegions::new())
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![produces_field(outputs)])
    }
}

impl OpImplementation for Noise {
    fn compute(
        &self,
        _inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let request = NoiseRequest::resolve(NOISE_OP_ID, params)?;
        let scale = request.frequency / FREQUENCY_PIXELS;
        let samples = synthesize(request.extent, |px, py| {
            value_noise(px * scale, py * scale, request.seed)
        });
        wrap_field(NOISE_OP_ID, request.extent, samples)
    }
}

/// The `field.fbm@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fbm;

impl Fbm {
    /// Construct the fbm operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `field.fbm@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        let mut params = base_params();
        params.push(ParamSpec {
            name: "octaves".to_owned(),
            ty: ParamType::Integer,
            unit: None,
            required: false,
            default: Some(serde_json::json!(4)),
            choices: vec![],
            doc: format!("The number of summed octaves, in 1..={MAX_OCTAVES}."),
        });
        params.push(ParamSpec {
            name: "lacunarity".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(2.0)),
            choices: vec![],
            doc: "The per-octave frequency multiplier (>0); classically 2.".to_owned(),
        });
        params.push(ParamSpec {
            name: "gain".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(0.5)),
            choices: vec![],
            doc: "The per-octave amplitude multiplier (persistence) in (0, 1]; classically 0.5."
                .to_owned(),
        });
        Ok(OperationManifest {
            id: FBM_OP_ID.parse()?,
            impl_version: 1,
            summary: "Sum `octaves` of hash-based value noise at frequency*lacunarity^o and \
                      amplitude gain^o, amplitude-normalized so the Field1 stays in [-1, 1] \
                      regardless of octave count (hash-of-coordinate, not a sequential RNG)."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![],
            outputs: vec![OutputSpec {
                name: "field".to_owned(),
                kind: paintop_ir::ResourceKind::Field1,
                doc: "The amplitude-normalized fbm scalar field in [-1, 1].".to_owned(),
            }],
            params,
            implementations: vec![reference_impl()?],
            test: noise_test_metadata(
                "field.fbm is an amplitude-normalized sum of hash-based value noise verified by \
                 analytic properties (bit-exact reruns, tiling/order invariance, seed sensitivity, \
                 octave-count-independent [-1, 1] range via amplitude normalization); there is no \
                 perceptual-quality metric to apply",
            ),
        })
    }
}

impl OpContract for Fbm {
    fn declared_inputs(&self) -> Vec<(String, paintop_ir::ResourceKind)> {
        vec![]
    }

    fn declared_outputs(&self) -> Vec<(String, paintop_ir::ResourceKind)> {
        vec![("field".to_owned(), paintop_ir::ResourceKind::Field1)]
    }

    fn infer_outputs(
        &self,
        _inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let request = FbmRequest::resolve(params)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "field".to_owned(),
            paintop_ir::ResourceDescriptor::Field1(noise_descriptor(request.base.extent)),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(InputRegions::new())
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![produces_field(outputs)])
    }
}

impl OpImplementation for Fbm {
    fn compute(
        &self,
        _inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let request = FbmRequest::resolve(params)?;
        let scale = request.base.frequency / FREQUENCY_PIXELS;
        let samples = synthesize(request.base.extent, |px, py| {
            request.evaluate(px * scale, py * scale)
        });
        wrap_field(FBM_OP_ID, request.base.extent, samples)
    }
}

/// Evaluate `f` at every pixel center of `extent` in row-major order, rounding
/// each `f64` value once to `f32`. The sample value at a pixel is a pure function
/// of its own (fixed) center coordinate, so the loop order does not matter — the
/// op is tiling/evaluation-order invariant by construction.
fn synthesize(extent: Extent, f: impl Fn(f64, f64) -> f64) -> Vec<f32> {
    let n = (extent.width as usize) * (extent.height as usize);
    let mut samples = vec![0.0_f32; n];
    let mut i = 0;
    for y in 0..extent.height {
        for x in 0..extent.width {
            let (px, py) = CoordinateConvention::PixelCenterUpperLeft.pixel_center(x, y);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "noise value computed in f64, stored once as the f32 sample"
            )]
            {
                samples[i] = f(px, py) as f32;
            }
            i += 1;
        }
    }
    samples
}

/// Wrap a synthesized scalar buffer as the single `field` output.
fn wrap_field(
    op: &str,
    extent: Extent,
    samples: Vec<f32>,
) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(
        paintop_ir::ResourceDescriptor::Field1(noise_descriptor(extent)),
        1,
        samples,
    )
    .map_err(|actual| buffer_error(op, actual))?;
    let mut out = OutputValues::new();
    out.insert("field".to_owned(), value);
    Ok(out)
}

/// Postcondition: the op produced a `field` Field1 output.
fn produces_field(outputs: &OutputDescriptors) -> AssertionResult {
    if matches!(
        outputs.get("field"),
        Some(paintop_ir::ResourceDescriptor::Field1(_))
    ) {
        AssertionResult::pass("produces_field")
    } else {
        AssertionResult::fail("produces_field", "no `field` Field1 output produced")
    }
}

#[cfg(test)]
mod tests;
