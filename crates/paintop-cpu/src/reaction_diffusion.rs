//! The `field.reaction_diffusion@1` operation: an explicit-step **Gray-Scott
//! reaction-diffusion** solver (`OP_CATALOG` §11, P2; `plan.md` §1428, §1444).
//!
//! The Gray-Scott model evolves two scalar concentration fields `u` and `v` under
//! diffusion plus the autocatalytic reaction `u + 2v -> 3v`:
//!
//! ```text
//! u_{n+1} = u_n + dt * ( Du * lap(u_n) - u_n v_n^2 + feed * (1 - u_n) )
//! v_{n+1} = v_n + dt * ( Dv * lap(v_n) + u_n v_n^2 - (feed + kill) * v_n )
//! ```
//!
//! where `lap` is the 5-point discrete Laplacian under a `wrap` (toroidal)
//! boundary, and `Du`, `Dv`, `feed`, `kill`, `dt` are the model parameters. With a
//! suitable `feed`/`kill` pair the system self-organizes into the classic Turing
//! patterns (spots, stripes, mazes).
//!
//! # Determinism (M4 exit criterion 2)
//!
//! The op is deterministic given the seed and the step count: the initial state is
//! a **hash-of-coordinate** seeding (a uniform `u = 1` background with a seeded
//! square of `v`-perturbation whose jitter is hashed from the pixel coordinate,
//! never a sequential RNG), and every step is a fixed row-major `f64` evaluation
//! rounded once to `f32`. A rerun with the same seed and step count is therefore
//! bit-identical (asserted by the test suite). It is declared
//! [`Bounded`](DeterminismTier::Bounded): the per-step float arithmetic matches an
//! independent reference within tolerance rather than bit-for-bit across
//! platforms.
//!
//! # Stability guard (M4 acceptance)
//!
//! The explicit forward-Euler diffusion step is conditionally stable: the
//! dimensionless **stability number** `s = max(Du, Dv) * dt * 4` (4 = the 5-point
//! Laplacian's neighbour weight) must stay at or below the scheme's bound `1`.
//! A request whose `(Du, Dv, dt)` exceed the bound is **rejected** as a
//! [`policy`](ErrorClass::Policy) error rather than silently integrated into a
//! `NaN` blow-up; the realized stability number and bound are recorded in the
//! report.
//!
//! # Convergence metrics (M4 exit criterion 1)
//!
//! The op's report carries a [`SolverData`] payload: the
//! step count, the stability number/bound and a `stable` flag, the per-step L2
//! change `||state_{n+1} - state_n||_2` (the residual history, a decaying series
//! as the pattern settles), and the final total energy. These are the solver's
//! exposed convergence metrics.
//!
//! # Outputs
//!
//! Three outputs: the solved `u` and `v` scalar [`Field1`](paintop_ir::ResourceKind::Field1)s
//! and a `report` carrying the [`SolverData`].

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, FieldArity, FieldDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Rect, Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, ScalarType, SemanticRole, SolverData, TestMetadata,
};

/// The canonical id of the reaction-diffusion operation.
pub const REACTION_DIFFUSION_OP_ID: &str = "field.reaction_diffusion@1";

/// The `extent_from` input was absent or carried no descriptor to size the
/// fields.
pub const E_RD_INPUT: &str = "E_RD_INPUT";

/// A solver parameter was missing, the wrong shape, or non-finite.
pub const E_RD_PARAM: &str = "E_RD_PARAM";

/// The requested `(Du, Dv, dt)` exceeds the explicit-stability bound.
pub const E_RD_UNSTABLE: &str = "E_RD_UNSTABLE";

/// The execution buffer length disagreed with the declared extent.
pub const E_RD_BUFFER: &str = "E_RD_BUFFER";

/// The 5-point Laplacian's total neighbour weight; the explicit-Euler diffusion
/// step is stable while `max_diffusion * dt * NEIGHBOURS <= STABILITY_LIMIT`.
const NEIGHBOURS: f64 = 4.0;

/// The explicit-stability bound for the normalized 5-point scheme.
const STABILITY_LIMIT: f64 = 1.0;

/// The largest number of steps a single request may run, bounding the work.
const MAX_STEPS: u32 = 100_000;

/// The resolved Gray-Scott parameters.
#[derive(Debug, Clone, Copy)]
struct RdParams {
    diffusion_u: f64,
    diffusion_v: f64,
    feed: f64,
    kill: f64,
    dt: f64,
    steps: u32,
    seed: u64,
}

impl RdParams {
    /// Resolve and validate the solver params, including the stability guard.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let diffusion_u = positive_or_zero(params, "diffusion_u", 0.16)?;
        let diffusion_v = positive_or_zero(params, "diffusion_v", 0.08)?;
        let feed = finite(params, "feed", 0.060)?;
        let kill = finite(params, "kill", 0.062)?;
        let dt = strictly_positive(params, "dt", 1.0)?;
        let steps = steps_param(params)?;
        let seed = seed_param(params)?;

        // Stability guard: reject a request that would diverge under explicit
        // forward Euler rather than integrate it into a NaN.
        let stability_number = diffusion_u.max(diffusion_v) * dt * NEIGHBOURS;
        if stability_number > STABILITY_LIMIT {
            return Err(Error::new(
                ErrorClass::Policy,
                E_RD_UNSTABLE,
                format!(
                    "field.reaction_diffusion is explicitly unstable: max(Du, Dv)*dt*4 = \
                     {stability_number:.4} exceeds the stability bound {STABILITY_LIMIT}; reduce \
                     dt or the diffusion rates"
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("{stability_number:.6}"))
                    .with_expected(format!("<= {STABILITY_LIMIT}")),
            ));
        }

        Ok(Self {
            diffusion_u,
            diffusion_v,
            feed,
            kill,
            dt,
            steps,
            seed,
        })
    }

    /// The realized dimensionless stability number.
    fn stability_number(&self) -> f64 {
        self.diffusion_u.max(self.diffusion_v) * self.dt * NEIGHBOURS
    }
}

/// Read an optional finite, non-negative float param (a diffusion rate).
fn positive_or_zero(params: &serde_json::Value, name: &str, default: f64) -> Result<f64> {
    let Some(value) = params.get(name) else {
        return Ok(default);
    };
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("must be a number", name, value))?;
    if n.is_finite() && n >= 0.0 {
        Ok(n)
    } else {
        Err(param_error(
            "must be a finite, non-negative number",
            name,
            value,
        ))
    }
}

/// Read an optional strictly-positive finite float param.
fn strictly_positive(params: &serde_json::Value, name: &str, default: f64) -> Result<f64> {
    let Some(value) = params.get(name) else {
        return Ok(default);
    };
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("must be a number", name, value))?;
    if n.is_finite() && n > 0.0 {
        Ok(n)
    } else {
        Err(param_error(
            "must be a finite, strictly-positive number",
            name,
            value,
        ))
    }
}

/// Read an optional finite float param (a feed/kill rate, possibly negative).
fn finite(params: &serde_json::Value, name: &str, default: f64) -> Result<f64> {
    let Some(value) = params.get(name) else {
        return Ok(default);
    };
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("must be a number", name, value))?;
    if n.is_finite() {
        Ok(n)
    } else {
        Err(param_error("must be finite", name, value))
    }
}

/// Read the `steps` param (a positive integer, default 1000, capped).
fn steps_param(params: &serde_json::Value) -> Result<u32> {
    let steps = match params.get("steps") {
        None => 1000_u32,
        Some(value) => value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| param_error("must be a non-negative integer", "steps", value))?,
    };
    if steps == 0 || steps > MAX_STEPS {
        return Err(Error::new(
            ErrorClass::Schema,
            E_RD_PARAM,
            format!("field.reaction_diffusion `steps` must be in 1..={MAX_STEPS}, got {steps}"),
        ));
    }
    Ok(steps)
}

/// Read the optional `seed` param (a non-negative integer), defaulting to `0`.
fn seed_param(params: &serde_json::Value) -> Result<u64> {
    params.get("seed").map_or(Ok(0), |value| {
        value
            .as_u64()
            .ok_or_else(|| param_error("`seed` must be a non-negative integer", "seed", value))
    })
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, name: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_RD_PARAM,
        format!("field.reaction_diffusion parameter `{name}`: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// The `SplitMix64` finalizer, used to seed the initial `v`-perturbation jitter
/// as a hash of the pixel coordinate (deterministic, order-invariant).
const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A hashed jitter in `[0, 1)` for pixel `(x, y)` under `seed`.
fn jitter(x: u32, y: u32, seed: u64) -> f64 {
    let h = mix64(seed ^ (u64::from(x) << 32 | u64::from(y)).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    f64::from(u32::try_from(h >> 40).unwrap_or(0)) / f64::from(1u32 << 24)
}

/// The scalar `Field1` descriptor a concentration field uses for `extent`.
const fn field_descriptor(extent: Extent) -> FieldDescriptor {
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

/// The solved Gray-Scott state plus the convergence metrics.
struct Solution {
    u: Vec<f32>,
    v: Vec<f32>,
    residual_history: Vec<f64>,
    total_energy: f64,
}

/// Seed the initial state and run `steps` explicit Gray-Scott steps over
/// `extent`, collecting the per-step L2 residual and the final energy.
fn solve(extent: Extent, params: RdParams) -> Solution {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let count = width * height;

    // Initial state: u = 1 everywhere, v = 0 except a seeded central square where
    // v ~ 0.25 + small hashed jitter and u ~ 0.5 (the classic Gray-Scott seed).
    let mut conc_u = vec![1.0_f64; count];
    let mut conc_v = vec![0.0_f64; count];
    let center_x = width / 2;
    let center_y = height / 2;
    let half = (width.min(height) / 8).max(1);
    for row in 0..height {
        for col in 0..width {
            let inside = col.abs_diff(center_x) <= half && row.abs_diff(center_y) <= half;
            if inside {
                let jit = jitter(
                    u32::try_from(col).unwrap_or(0),
                    u32::try_from(row).unwrap_or(0),
                    params.seed,
                );
                let idx = row * width + col;
                conc_u[idx] = 0.5;
                conc_v[idx] = 0.04_f64.mul_add(jit, 0.25);
            }
        }
    }

    let mut next_u = vec![0.0_f64; count];
    let mut next_v = vec![0.0_f64; count];
    let mut residual_history = Vec::with_capacity(params.steps as usize);

    for _ in 0..params.steps {
        let mut residual_sq = 0.0_f64;
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let lap_u = laplacian(&conc_u, col, row, width, height);
                let lap_v = laplacian(&conc_v, col, row, width, height);
                let reaction = conc_u[idx] * conc_v[idx] * conc_v[idx];
                let delta_u = params
                    .diffusion_u
                    .mul_add(lap_u, params.feed.mul_add(1.0 - conc_u[idx], -reaction));
                let delta_v = params.diffusion_v.mul_add(
                    lap_v,
                    (params.feed + params.kill).mul_add(-conc_v[idx], reaction),
                );
                let updated_u = params.dt.mul_add(delta_u, conc_u[idx]);
                let updated_v = params.dt.mul_add(delta_v, conc_v[idx]);
                next_u[idx] = updated_u;
                next_v[idx] = updated_v;
                let res_u = updated_u - conc_u[idx];
                let res_v = updated_v - conc_v[idx];
                residual_sq += res_u.mul_add(res_u, res_v * res_v);
            }
        }
        std::mem::swap(&mut conc_u, &mut next_u);
        std::mem::swap(&mut conc_v, &mut next_v);
        residual_history.push(residual_sq.sqrt());
    }

    let mut total_energy = 0.0_f64;
    let mut samples_u = Vec::with_capacity(count);
    let mut samples_v = Vec::with_capacity(count);
    for idx in 0..count {
        total_energy += conc_u[idx].mul_add(conc_u[idx], conc_v[idx] * conc_v[idx]);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "state evolved in f64, stored once as the f32 sample"
        )]
        {
            samples_u.push(conc_u[idx] as f32);
            samples_v.push(conc_v[idx] as f32);
        }
    }

    Solution {
        u: samples_u,
        v: samples_v,
        residual_history,
        total_energy,
    }
}

/// The 5-point discrete Laplacian of `field` at `(col, row)` under a `wrap`
/// boundary (toroidal): `sum(neighbours) - 4*center`.
fn laplacian(field: &[f64], col: usize, row: usize, width: usize, height: usize) -> f64 {
    let left = if col == 0 { width - 1 } else { col - 1 };
    let right = if col + 1 == width { 0 } else { col + 1 };
    let up = if row == 0 { height - 1 } else { row - 1 };
    let down = if row + 1 == height { 0 } else { row + 1 };
    let center = field[row * width + col];
    let sum = field[row * width + left]
        + field[row * width + right]
        + field[up * width + col]
        + field[down * width + col];
    NEIGHBOURS.mul_add(-center, sum)
}

/// The `field.reaction_diffusion@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReactionDiffusion;

impl ReactionDiffusion {
    /// Construct the reaction-diffusion operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `field.reaction_diffusion@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: REACTION_DIFFUSION_OP_ID.parse()?,
            impl_version: 1,
            summary: "Explicit-step Gray-Scott reaction-diffusion solver: evolve two Field1 \
                      concentrations (u, v) for `steps` forward-Euler steps under a wrap-boundary \
                      Laplacian, deterministic given seed + step count, with a stability guard and \
                      a SolverData report (step count, stability number, residual history, energy)."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                // Each step reads a 1px neighbourhood, but `steps` steps propagate
                // information across the whole field, so the honest footprint is
                // the whole domain.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "extent_from".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The resource whose pixel extent the solved fields match.".to_owned(),
            }],
            outputs: vec![
                OutputSpec {
                    name: "u".to_owned(),
                    kind: ResourceKind::Field1,
                    doc: "The solved u concentration field.".to_owned(),
                },
                OutputSpec {
                    name: "v".to_owned(),
                    kind: ResourceKind::Field1,
                    doc: "The solved v concentration field (the patterned species).".to_owned(),
                },
                OutputSpec {
                    name: "report".to_owned(),
                    kind: ResourceKind::Report,
                    doc: "The solver report carrying SolverData (steps, stability number/limit, \
                          residual history, total energy)."
                        .to_owned(),
                },
            ],
            params: params_spec(),
            implementations: vec![reference_impl()?],
            test: rd_test_metadata(),
        })
    }
}

/// The declared parameter list.
fn params_spec() -> Vec<ParamSpec> {
    let float = |name: &str, default: f64, doc: &str| ParamSpec {
        name: name.to_owned(),
        ty: ParamType::Float,
        unit: None,
        required: false,
        default: Some(serde_json::json!(default)),
        choices: vec![],
        doc: doc.to_owned(),
    };
    vec![
        float(
            "diffusion_u",
            0.16,
            "The u diffusion rate Du (>= 0); Du*dt*4 must stay <= 1 for stability.",
        ),
        float(
            "diffusion_v",
            0.08,
            "The v diffusion rate Dv (>= 0); Dv*dt*4 must stay <= 1 for stability.",
        ),
        float("feed", 0.060, "The Gray-Scott feed rate F."),
        float("kill", 0.062, "The Gray-Scott kill rate k."),
        float("dt", 1.0, "The explicit time step (> 0)."),
        ParamSpec {
            name: "steps".to_owned(),
            ty: ParamType::Integer,
            unit: None,
            required: false,
            default: Some(serde_json::json!(1000)),
            choices: vec![],
            doc: format!("The number of explicit steps, in 1..={MAX_STEPS}."),
        },
        ParamSpec {
            name: "seed".to_owned(),
            ty: ParamType::Seed,
            unit: Some(ParamUnit::Pixels),
            required: false,
            default: Some(serde_json::json!(0)),
            choices: vec![],
            doc: "The seed for the initial v-perturbation jitter (hash-of-coordinate, not an RNG)."
                .to_owned(),
        },
    ]
}

/// The `extent_from` input's extent.
fn extent_of(inputs: &Descriptors) -> Result<Extent> {
    let input = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_RD_INPUT,
            "field.reaction_diffusion requires an `extent_from` resource".to_owned(),
        )
    })?;
    Ok(input.extent())
}

impl OpContract for ReactionDiffusion {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("extent_from".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("u".to_owned(), ResourceKind::Field1),
            ("v".to_owned(), ResourceKind::Field1),
            ("report".to_owned(), ResourceKind::Report),
        ]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // Validate params (and the stability guard) at infer time.
        RdParams::resolve(params)?;
        let extent = extent_of(inputs)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "u".to_owned(),
            ResourceDescriptor::Field1(field_descriptor(extent)),
        );
        out.insert(
            "v".to_owned(),
            ResourceDescriptor::Field1(field_descriptor(extent)),
        );
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent,
                channels: 2,
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
        // The solver reads no input *samples* — only the `extent_from` size — so it
        // demands no input pixels.
        let mut regions = InputRegions::new();
        if inputs.contains_key("extent_from") {
            regions.insert("extent_from".to_owned(), Rect::new(0, 0, 0, 0));
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let u = matches!(outputs.get("u"), Some(ResourceDescriptor::Field1(_)));
        let v = matches!(outputs.get("v"), Some(ResourceDescriptor::Field1(_)));
        let report = matches!(outputs.get("report"), Some(ResourceDescriptor::Report(_)));
        Ok(vec![if u && v && report {
            AssertionResult::pass("produces_u_v_report")
        } else {
            AssertionResult::fail(
                "produces_u_v_report",
                "expected `u` and `v` Field1 outputs and a `report`",
            )
        }])
    }
}

impl OpImplementation for ReactionDiffusion {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let source = inputs.get("extent_from").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_RD_INPUT,
                "field.reaction_diffusion requires an `extent_from` input value".to_owned(),
            )
        })?;
        let extent = source.extent();
        let resolved = RdParams::resolve(params)?;

        let solution = solve(extent, resolved);

        let stability_number = resolved.stability_number();
        let report = Report {
            extent,
            channels: 2,
            channel_stats: Vec::new(),
            all_finite: solution.u.iter().chain(&solution.v).all(|s| s.is_finite()),
            content_hash: String::new(),
            diff: None,
            assertion: None,
            histogram: None,
            components: None,
            frequency_energy: None,
            solver: Some(SolverData {
                kind: "gray-scott".to_owned(),
                steps: resolved.steps,
                stability_number,
                stability_limit: STABILITY_LIMIT,
                stable: stability_number <= STABILITY_LIMIT,
                residual_history: solution.residual_history,
                total_energy: solution.total_energy,
                iterations: None,
                stop_reason: None,
                converged: None,
                tolerance: None,
                final_residual: None,
            }),
        };

        let mut out = OutputValues::new();
        out.insert(
            "u".to_owned(),
            ResourceValue::new(
                ResourceDescriptor::Field1(field_descriptor(extent)),
                1,
                solution.u,
            )
            .map_err(|actual| buffer_error("u", actual))?,
        );
        out.insert(
            "v".to_owned(),
            ResourceValue::new(
                ResourceDescriptor::Field1(field_descriptor(extent)),
                1,
                solution.v,
            )
            .map_err(|actual| buffer_error("v", actual))?,
        );
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// A buffer-length-mismatch execution error for an output port.
fn buffer_error(port: &str, actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_RD_BUFFER,
        format!(
            "field.reaction_diffusion produced a `{port}` buffer of unexpected length {actual}"
        ),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `field.reaction_diffusion@1`: a bounded iterative
/// solver verified by analytic/statistical properties (seeded determinism, the
/// stability guard, pattern statistics on a known Turing parameter set, residual-
/// history presence). No perceptual metric applies.
fn rd_test_metadata() -> TestMetadata {
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
            "field.reaction_diffusion is an explicit-step PDE solver verified by analytic and \
             statistical properties (seeded bit-identical reruns, the explicit-stability guard \
             rejecting unstable steps, bounded pattern statistics on a known Turing parameter set, \
             a residual-history convergence record); there is no perceptual-quality metric to apply",
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
