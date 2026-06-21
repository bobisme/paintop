//! Cross-backend SIMD differential suite (`bn-yhf`, `bn-1pu`, `bn-2ja`).
//!
//! Every `cpu.optimized` pointwise kernel must reproduce the `cpu.reference`
//! oracle within the op's declared tolerance. This suite drives the cluster-1
//! differential harness ([`paintop_testkit::differential`]) against the *real* op
//! registries built by [`paintop_cpu::registry`], forcing each optimized backend
//! and comparing it to the oracle on representative inputs:
//!
//! * **bn-yhf** — color.convert, color.adjust, alpha.premultiply/unpremultiply;
//! * **bn-1pu** — composite.over, composite.blend (premultiplied-linear edge
//!   cases: transparent coloured pixels, exact-tier subsets);
//! * **bn-2ja** — the full matrix as one table plus the perf artifact pairing each
//!   kernel's tolerance, speedup, and implementation id.
//!
//! The harness reads the tolerance *tier* from the manifest, so an exact-tier op
//! (premultiply, over, blend) must match the oracle **bit-for-bit** and a
//! bounded-tier op (convert, adjust, unpremultiply) within its envelope — the test
//! never hard-codes a tolerance.

use paintop_core::executor::value::ResourceValue;
use paintop_core::executor::{BackendId, ImplRegistry, InputValues};
use paintop_cpu::optimized::bench::{KernelKind, measure};
use paintop_cpu::registry::{implementation_registry, operation_registry};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, MaskDescriptor, MaskMeaning, OpId, OperationRegistry, ResourceDescriptor,
    ScalarType, SemanticRole, ValidRange,
};
use paintop_testkit::differential::{
    AllAvailable, DifferentialReport, OpInvocation, differential_check,
};

const EXTENT: Extent = Extent::new(8, 8);
const PIXELS: usize = (EXTENT.width * EXTENT.height) as usize;

/// A deterministic pseudo-random buffer in `[0, 1)` (xorshift64*), so the inputs
/// are non-degenerate without a `rand` dependency.
fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mantissa = u16::try_from((r >> 40) & 0xFFFF).unwrap_or(0);
        out.push(f32::from(mantissa) / f32::from(u16::MAX));
    }
    out
}

/// An RGBA color image value with the given encoding / alpha representation.
fn rgba(color: ColorEncoding, alpha: AlphaRepresentation, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: EXTENT,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color,
        range: ColorRange::SceneReferred,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 4, samples).expect("sized rgba")
}

/// A single-channel coverage mask value.
fn mask(samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: EXTENT,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("sized mask")
}

/// A premultiplied-linear RGBA buffer of `PIXELS` pixels: each pixel's colour is
/// its straight colour scaled by its alpha (`C' = C * a`), so the buffer is a valid
/// premultiplied image (`|colour| <= alpha`).
fn premultiplied_buffer(seed: u64) -> Vec<f32> {
    let mut out = fill(PIXELS * 4, seed);
    for px in out.chunks_exact_mut(4) {
        let a = px[3];
        px[0] *= a;
        px[1] *= a;
        px[2] *= a;
    }
    out
}

/// Run the differential harness for `op` with the given inputs/params, treating
/// every backend as available (the optimized backend is pure-CPU).
fn check(
    manifests: &OperationRegistry,
    impls: &ImplRegistry,
    op: &OpId,
    inputs: &InputValues,
    params: &serde_json::Value,
) -> DifferentialReport {
    let invocation = OpInvocation {
        inputs,
        params,
        output_port: "image",
    };
    differential_check(op, manifests, impls, &invocation, &AllAvailable, None)
        .expect("differential check runs")
}

/// Assert the optimized backend was actually exercised (not silently skipped) and
/// matched the oracle within tier tolerance.
fn assert_optimized_passed(report: &DifferentialReport) {
    let optimized = BackendId::new("cpu", "optimized");
    let (_, outcome) = report
        .backends
        .iter()
        .find(|(b, _)| b == &optimized)
        .expect("cpu.optimized backend present in the report");
    assert!(
        outcome.is_pass(),
        "cpu.optimized diverged from the oracle for {} (tolerance {:?}): {:?}",
        report.op,
        report.tolerance,
        outcome.error_map(),
    );
    assert!(report.all_pass(), "every backend passed for {}", report.op);
}

// ---------------------------------------------------------------------------
// bn-yhf: color + alpha kernels.
// ---------------------------------------------------------------------------

#[test]
fn alpha_premultiply_optimized_is_bit_exact() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "alpha.premultiply@1".parse().expect("op");

    // Straight-alpha linear RGBA, including hidden colour under zero coverage.
    let mut samples = fill(PIXELS * 4, 0xA11);
    // Force a transparent coloured pixel (hidden RGB, alpha 0) at pixel 0.
    samples[0..4].copy_from_slice(&[0.7, 0.3, 0.9, 0.0]);
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Straight,
            samples,
        ),
    );

    let report = check(&manifests, &impls, &op, &inputs, &serde_json::Value::Null);
    assert!(report.tolerance.exact, "premultiply is exact-tier");
    assert_optimized_passed(&report);
}

#[test]
fn alpha_unpremultiply_optimized_within_bounds() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "alpha.unpremultiply@1".parse().expect("op");

    // Premultiplied linear RGBA with a near-zero-alpha pixel (unrecoverable colour
    // clamped to zero on both backends).
    let mut samples = fill(PIXELS * 4, 0xB22);
    samples[0..4].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
    samples[4..8].copy_from_slice(&[0.01, 0.02, 0.03, 0.04]);
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Premultiplied,
            samples,
        ),
    );

    let report = check(&manifests, &impls, &op, &inputs, &serde_json::Value::Null);
    assert!(!report.tolerance.exact, "unpremultiply is bounded-tier");
    assert_optimized_passed(&report);
}

#[test]
fn color_convert_optimized_within_bounds() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "color.convert@1".parse().expect("op");

    let samples = fill(PIXELS * 4, 0xC33);
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(ColorEncoding::Srgb, AlphaRepresentation::Straight, samples),
    );
    let params = serde_json::json!({"from": "srgb", "to": "linear-srgb"});

    let report = check(&manifests, &impls, &op, &inputs, &params);
    assert_optimized_passed(&report);
}

#[test]
fn color_convert_identity_optimized_is_exact_passthrough() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "color.convert@1".parse().expect("op");

    let samples = fill(PIXELS * 4, 0xC34);
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Straight,
            samples,
        ),
    );
    // An identity (from == to) conversion: both backends clone the input.
    let params = serde_json::json!({"from": "linear-srgb", "to": "linear-srgb"});

    let report = check(&manifests, &impls, &op, &inputs, &params);
    assert_optimized_passed(&report);
}

#[test]
fn color_adjust_optimized_within_bounds() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "color.adjust@1".parse().expect("op");

    let samples = fill(PIXELS * 4, 0xD44);
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Straight,
            samples,
        ),
    );
    let params = serde_json::json!({
        "exposure_ev": 0.75,
        "saturation": 0.3,
        "temperature": -0.2,
    });

    let report = check(&manifests, &impls, &op, &inputs, &params);
    assert_optimized_passed(&report);
}

#[test]
fn color_adjust_masked_optimized_within_bounds() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "color.adjust@1".parse().expect("op");

    let samples = fill(PIXELS * 4, 0xD45);
    let mut mask_samples = fill(PIXELS, 0xD46);
    // Pin the gate extremes (fully off and fully on) to exercise both blend ends.
    mask_samples[0] = 0.0;
    mask_samples[1] = 1.0;
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Straight,
            samples,
        ),
    );
    inputs.insert("mask".to_owned(), mask(mask_samples));
    let params = serde_json::json!({"exposure_ev": 1.0, "saturation": -0.5});

    let report = check(&manifests, &impls, &op, &inputs, &params);
    assert_optimized_passed(&report);
}

#[test]
fn color_adjust_identity_optimized_is_passthrough() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "color.adjust@1".parse().expect("op");

    let samples = fill(PIXELS * 4, 0xD47);
    let mut inputs = InputValues::new();
    inputs.insert(
        "image".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Straight,
            samples,
        ),
    );
    // All-default params with no mask: a verbatim passthrough on both backends.
    let report = check(&manifests, &impls, &op, &inputs, &serde_json::Value::Null);
    assert_optimized_passed(&report);
}

// ---------------------------------------------------------------------------
// bn-1pu: compositing + blend kernels.
// ---------------------------------------------------------------------------

/// Build the premultiplied-linear `src`/`dst` inputs both compositing ops share,
/// pinning the premultiplied-linear edge cases: a fully transparent source pixel
/// (identity on dst), a fully opaque source pixel (replaces dst), and a
/// transparent-but-zero source colour (no colour fringe).
fn over_blend_inputs(with_mask: bool) -> InputValues {
    let mut src = premultiplied_buffer(0xE55);
    let dst = premultiplied_buffer(0xF66);
    // Pixel 0: fully transparent source (premultiplied colour 0, alpha 0) -> dst
    // passes through unchanged. Pixel 1: fully opaque source -> replaces dst.
    src[0..4].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
    src[4..8].copy_from_slice(&[0.4, 0.5, 0.6, 1.0]);

    let mut inputs = InputValues::new();
    inputs.insert(
        "src".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Premultiplied,
            src,
        ),
    );
    inputs.insert(
        "dst".to_owned(),
        rgba(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Premultiplied,
            dst,
        ),
    );
    if with_mask {
        let mut mask_samples = fill(PIXELS, 0x1A2B);
        mask_samples[0] = 0.0; // gate fully off (identity on dst)
        mask_samples[1] = 1.0; // gate fully on (pure blend)
        inputs.insert("mask".to_owned(), mask(mask_samples));
    }
    inputs
}

#[test]
fn composite_over_optimized_is_bit_exact() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "composite.over@1".parse().expect("op");

    let inputs = over_blend_inputs(false);
    let report = check(&manifests, &impls, &op, &inputs, &serde_json::Value::Null);
    assert!(report.tolerance.exact, "composite.over is exact-tier");
    assert_optimized_passed(&report);
}

#[test]
fn composite_blend_optimized_is_bit_exact_across_modes() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "composite.blend@1".parse().expect("op");

    let inputs = over_blend_inputs(true);
    // Every restricted, exactly-pinned mode must match the oracle bit-for-bit, at a
    // partial opacity so the dst-mix path (not just k in {0,1}) is exercised.
    for mode in [
        "normal",
        "over",
        "add",
        "subtract",
        "multiply",
        "screen",
        "darken",
        "lighten",
        "difference",
    ] {
        let params = serde_json::json!({"mode": mode, "opacity": 0.6});
        let report = check(&manifests, &impls, &op, &inputs, &params);
        assert!(
            report.tolerance.exact,
            "composite.blend is exact-tier ({mode})"
        );
        assert_optimized_passed(&report);
    }
}

#[test]
fn composite_blend_zero_opacity_optimized_is_dst_identity() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let op: OpId = "composite.blend@1".parse().expect("op");

    let inputs = over_blend_inputs(true);
    // opacity 0 -> k == 0 everywhere -> dst passes through verbatim on both
    // backends (the bit-pattern identity guarantee).
    let params = serde_json::json!({"mode": "add", "opacity": 0.0});
    let report = check(&manifests, &impls, &op, &inputs, &params);
    assert_optimized_passed(&report);
}

// ---------------------------------------------------------------------------
// bn-2ja: the full SIMD differential matrix + perf artifact.
//
// One table covering every optimized kernel vs the reference oracle, paired with
// the kernel's benchmark speedup, and written as the evidence artifact that pairs
// each kernel's tolerance, speedup, and implementation id.
// ---------------------------------------------------------------------------

/// A representative invocation for one op: its inputs and resolved params.
struct Case {
    inputs: InputValues,
    params: serde_json::Value,
}

/// Build the representative differential invocation for `kernel`'s op, reusing the
/// same fixtures the per-op tests exercise (premultiplied edge cases, masks, etc.).
fn case_for(kernel: KernelKind) -> Case {
    match kernel {
        KernelKind::AlphaPremultiply => {
            let mut s = fill(PIXELS * 4, 0xA11);
            s[0..4].copy_from_slice(&[0.7, 0.3, 0.9, 0.0]);
            let mut inputs = InputValues::new();
            inputs.insert(
                "image".to_owned(),
                rgba(ColorEncoding::LinearSrgb, AlphaRepresentation::Straight, s),
            );
            Case {
                inputs,
                params: serde_json::Value::Null,
            }
        }
        KernelKind::AlphaUnpremultiply => {
            let mut s = fill(PIXELS * 4, 0xB22);
            s[0..4].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            let mut inputs = InputValues::new();
            inputs.insert(
                "image".to_owned(),
                rgba(
                    ColorEncoding::LinearSrgb,
                    AlphaRepresentation::Premultiplied,
                    s,
                ),
            );
            Case {
                inputs,
                params: serde_json::Value::Null,
            }
        }
        KernelKind::ColorConvert => {
            let mut inputs = InputValues::new();
            inputs.insert(
                "image".to_owned(),
                rgba(
                    ColorEncoding::Srgb,
                    AlphaRepresentation::Straight,
                    fill(PIXELS * 4, 0xC33),
                ),
            );
            Case {
                inputs,
                params: serde_json::json!({"from": "srgb", "to": "linear-srgb"}),
            }
        }
        KernelKind::ColorAdjust => {
            let mut inputs = InputValues::new();
            inputs.insert(
                "image".to_owned(),
                rgba(
                    ColorEncoding::LinearSrgb,
                    AlphaRepresentation::Straight,
                    fill(PIXELS * 4, 0xD44),
                ),
            );
            Case {
                inputs,
                params: serde_json::json!({
                    "exposure_ev": 0.75, "saturation": 0.3, "temperature": -0.2
                }),
            }
        }
        KernelKind::CompositeOver => Case {
            inputs: over_blend_inputs(false),
            params: serde_json::Value::Null,
        },
        KernelKind::CompositeBlend => Case {
            inputs: over_blend_inputs(true),
            params: serde_json::json!({"mode": "screen", "opacity": 0.6}),
        },
    }
}

#[test]
fn simd_differential_matrix_and_perf_artifact() {
    // A modest working set keeps the in-suite speedup probe fast; the differential
    // correctness check is independent of the timing.
    const BENCH_PIXELS: usize = 4_096;

    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let optimized = BackendId::new("cpu", "optimized");
    let mut rows = Vec::new();

    for kernel in KernelKind::ALL {
        let op: OpId = kernel.op_id().parse().expect("kernel op id");
        let case = case_for(kernel);

        // 1. Differential: the optimized kernel vs the oracle, within tier
        //    tolerance read from the manifest.
        let report = check(&manifests, &impls, &op, &case.inputs, &case.params);
        let (_, outcome) = report
            .backends
            .iter()
            .find(|(b, _)| b == &optimized)
            .expect("cpu.optimized present");
        assert!(
            outcome.is_pass(),
            "matrix: cpu.optimized diverged for {op} (tier tol {:?})",
            report.tolerance,
        );
        let error_map = outcome.error_map().expect("pass carries an error map");

        // 2. Speedup: the benchmark ratio for this kernel.
        let measurement = measure(kernel, BENCH_PIXELS);

        // 3. The evidence row pairs tolerance, speedup, and implementation id.
        rows.push(serde_json::json!({
            "op": kernel.op_id(),
            "impl_id": "cpu.optimized@1",
            "precision_tier": format!("{:?}", kernel.precision()),
            "tolerance": {
                "exact": report.tolerance.exact,
                "max_abs": report.tolerance.max_abs,
                "max_rms": report.tolerance.max_rms,
            },
            "differential": {
                "passed": true,
                "max_abs_diff": error_map.max_abs,
                "rms_diff": error_map.rms,
            },
            "speedup": measurement.speedup,
            "scalar_ns": measurement.scalar_ns,
            "optimized_ns": measurement.optimized_ns,
        }));
    }

    // Every optimized kernel is represented in the matrix.
    assert_eq!(rows.len(), KernelKind::ALL.len());

    let artifact = serde_json::json!({
        "backend": "cpu.optimized",
        "oracle": "cpu.reference",
        "kernels": rows,
    });

    // Each row pairs tolerance + speedup + implementation id (the bone's evidence
    // requirement).
    for row in artifact["kernels"].as_array().expect("kernels array") {
        assert_eq!(row["impl_id"], "cpu.optimized@1");
        assert!(row["tolerance"].is_object());
        assert!(row["speedup"].is_number());
        assert_eq!(row["differential"]["passed"], true);
    }

    // Write the matrix/perf evidence under the op-backend differential path.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("target/verification/cpu.optimized/differential");
    std::fs::create_dir_all(&root).expect("create differential dir");
    let path = root.join("matrix.json");
    let json = serde_json::to_string_pretty(&artifact).expect("serialise matrix");
    std::fs::write(&path, format!("{json}\n")).expect("write matrix artifact");
    assert!(path.exists());
}
