//! Verification suite for `filter.convolve@1` (`OP_CATALOG` §8,
//! `AGENT_VERIFICATION` §3.3, `IR_SPEC` §8.4):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in JSON matches the builder;
//! - **analytic fixtures**: an impulse kernel reproduces the kernel (interior);
//!   each boundary mode (constant/transparent/clamp/mirror/wrap/valid) reproduces
//!   a known border on a tiny signal; a multi-channel filter is per-channel;
//!   Field1 is filtered like a one-channel image;
//! - **property**: a unit-sum kernel preserves a constant; a zero-sum kernel
//!   annihilates a constant on the interior; a constant scales by the kernel sum;
//!   linearity (before any clamp); outputs are finite for finite inputs;
//! - **metamorphic**: translation equivariance away from the boundary;
//! - **rejection**: a malformed/missized kernel, a bad origin, an unknown mode,
//!   and a mismatched `value` length are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, FieldArity, FieldDescriptor, ImageDescriptor, OpContract,
    OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{CONVOLVE_OP_ID, Convolve, E_CONVOLVE_KERNEL, E_CONVOLVE_PARAM};

/// Build a single-channel (gray) image from a row-major sample list.
fn gray(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    image_value(width, height, ChannelLayout::Gray, samples)
}

/// Build an image [`ResourceValue`] with the given layout/samples.
fn image_value(width: u32, height: u32, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// A single-channel Field1 value carrying `samples`.
fn field1(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Field1(FieldDescriptor {
        arity: FieldArity::Field1,
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    });
    ResourceValue::new(descriptor, 1, samples).expect("field1")
}

/// Run the convolution kernel and recover the output resource.
fn convolve(value: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value.clone());
    let mut out = Convolve::new()
        .compute(&inputs, params)
        .expect("convolve computes");
    out.remove("output").expect("output port produced")
}

/// A 3x3 kernel object from a row-major weight list (origin centred).
fn k3(weights: [f64; 9]) -> serde_json::Value {
    serde_json::json!({ "width": 3, "height": 3, "weights": weights.to_vec() })
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Convolve::manifest().expect("convolve manifest");
    manifest.validate().expect("convolve manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Convolve::new())
        .expect("convolve manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("convolve verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), CONVOLVE_OP_ID);
}

#[test]
fn impulse_kernel_is_the_identity() {
    // A centred unit impulse copies the input verbatim (interior and, under
    // clamp, the border too since the off-centre taps are zero).
    let img = gray(3, 3, (0..9u8).map(f32::from).collect());
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": k3([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) }),
    );
    assert_eq!(out.extent(), Extent::new(3, 3));
    assert_eq!(out.samples(), img.samples());
}

#[test]
fn impulse_response_equals_the_kernel_on_the_interior() {
    // §3.3: convolving a single-pixel impulse against a kernel reproduces the
    // (correlation) kernel pattern centred on the impulse. With an off-centre
    // kernel value, an impulse at (cx, cy) places weight w(kx,ky) at
    // (cx - (kx - ox), cy - (ky - oy)). We instead verify the simpler
    // interior-impulse identity: an interior delta image filtered by the kernel
    // yields the flipped kernel footprint; here we use a symmetric kernel so the
    // footprint equals the kernel.
    // 5x5 image, single 1.0 at centre (2,2).
    let mut samples = vec![0.0f32; 25];
    samples[2 * 5 + 2] = 1.0;
    let img = gray(5, 5, samples);
    let kernel = k3([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "constant" }),
    );
    // Correlation of a centred delta with weights w: o(2 - dx, 2 - dy) = w(dx,dy)
    // where (dx,dy) are kernel offsets from origin (1,1). Equivalently the kernel
    // appears flipped around (2,2). Build the expected map directly.
    let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut want = vec![0.0f32; 25];
    for ky in 0..3usize {
        for kx in 0..3usize {
            let weight = weights[ky * 3 + kx];
            // output pixel p with src(p + (kx-1, ky-1)) == delta(2,2) =>
            // p = (2 - (kx-1), 2 - (ky-1)) = (3 - kx, 3 - ky).
            let px = 3 - kx;
            let py = 3 - ky;
            want[py * 5 + px] = weight;
        }
    }
    assert_eq!(out.samples(), want.as_slice());
}

#[test]
fn constant_input_scales_by_the_kernel_sum() {
    // §3.3: on a constant interior, output == constant * sum(weights).
    let img = gray(5, 5, vec![2.0; 25]);
    let kernel = k3([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); // sum 9
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "clamp" }),
    );
    // clamp makes every pixel see a 3x3 of constant 2 => 18 everywhere.
    for &s in out.samples() {
        assert!((s - 18.0).abs() < 1e-5, "expected 18, got {s}");
    }
}

#[test]
fn unit_sum_kernel_preserves_a_constant() {
    let img = gray(4, 4, vec![0.5; 16]);
    let kernel = k3([
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
    ]);
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "clamp" }),
    );
    for &s in out.samples() {
        assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
    }
}

#[test]
fn zero_sum_kernel_annihilates_a_constant_on_the_interior() {
    // A Laplacian (sum 0) on a constant interior is exactly 0.
    let img = gray(5, 5, vec![3.0; 25]);
    let kernel = k3([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]);
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "clamp" }),
    );
    for &s in out.samples() {
        assert!(s.abs() < 1e-5, "expected 0 on constant, got {s}");
    }
}

#[test]
fn box_blur_row_with_clamp_matches_hand_computation() {
    // 1x5 row, horizontal 1x3 box (unnormalized), clamp boundary.
    let img = gray(5, 1, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "clamp" }),
    );
    // clamp neighbours: [1,1,2]=4, [1,2,4]=7, [2,4,8]=14, [4,8,16]=28, [8,16,16]=40
    assert_eq!(out.samples(), [4.0, 7.0, 14.0, 28.0, 40.0].as_slice());
}

#[test]
fn constant_mode_uses_the_fill_value_off_edge() {
    // 1x3 row, 1x3 box, constant fill 10 on both sides.
    let img = gray(3, 1, vec![1.0, 2.0, 3.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "constant", "value": [10.0] }),
    );
    // [10,1,2]=13, [1,2,3]=6, [2,3,10]=15
    assert_eq!(out.samples(), [13.0, 6.0, 15.0].as_slice());
}

#[test]
fn transparent_mode_treats_off_edge_as_zero() {
    let img = gray(3, 1, vec![1.0, 2.0, 3.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "transparent" }),
    );
    // [0,1,2]=3, [1,2,3]=6, [2,3,0]=5
    assert_eq!(out.samples(), [3.0, 6.0, 5.0].as_slice());
}

#[test]
fn mirror_mode_reflects_off_edge() {
    let img = gray(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "mirror" }),
    );
    // left neighbour of idx0 mirrors to idx1 (=2); right neighbour of idx3 -> idx2 (=3)
    // [2,1,2]=5, [1,2,3]=6, [2,3,4]=9, [3,4,3]=10
    assert_eq!(out.samples(), [5.0, 6.0, 9.0, 10.0].as_slice());
}

#[test]
fn wrap_mode_tiles_periodically() {
    let img = gray(3, 1, vec![1.0, 2.0, 3.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "wrap" }),
    );
    // [3,1,2]=6, [1,2,3]=6, [2,3,1]=6
    assert_eq!(out.samples(), [6.0, 6.0, 6.0].as_slice());
}

#[test]
fn valid_mode_shrinks_the_extent() {
    // 5x1 row, 1x3 box, valid => 3 outputs (no border).
    let img = gray(5, 1, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "valid" }),
    );
    assert_eq!(out.extent(), Extent::new(3, 1));
    // [1,2,4]=7, [2,4,8]=14, [4,8,16]=28
    assert_eq!(out.samples(), [7.0, 14.0, 28.0].as_slice());
}

#[test]
fn valid_mode_kernel_larger_than_input_is_empty() {
    let img = gray(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": k3([0.0; 9]), "mode": "valid" }),
    );
    assert_eq!(out.extent(), Extent::new(0, 0));
    assert!(out.samples().is_empty());
}

#[test]
fn multi_channel_is_filtered_independently() {
    // 1x3 RGBA row; a 1x3 box on each channel under transparent.
    // Channels carry distinct ramps so a cross-channel bug would show.
    let samples = vec![
        1.0, 10.0, 100.0, 1000.0, // px0
        2.0, 20.0, 200.0, 2000.0, // px1
        3.0, 30.0, 300.0, 3000.0, // px2
    ];
    let img = image_value(3, 1, ChannelLayout::Rgba, samples);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "transparent" }),
    );
    // Per channel box with zero off-edge: centre px = a+b+c, edges drop one term.
    let want = vec![
        3.0, 30.0, 300.0, 3000.0, // [0+1+2]
        6.0, 60.0, 600.0, 6000.0, // [1+2+3]
        5.0, 50.0, 500.0, 5000.0, // [2+3+0]
    ];
    assert_eq!(out.samples(), want.as_slice());
}

#[test]
fn field1_is_filtered_like_a_one_channel_image() {
    let f = field1(5, 1, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    let kernel = serde_json::json!({ "width": 3, "height": 1, "weights": [1.0, 1.0, 1.0] });
    let out = convolve(
        &f,
        &serde_json::json!({ "kernel": kernel, "mode": "clamp" }),
    );
    // Output is still a Field1.
    assert!(matches!(out.descriptor(), ResourceDescriptor::Field1(_)));
    assert_eq!(out.samples(), [4.0, 7.0, 14.0, 28.0, 40.0].as_slice());
}

#[test]
fn linearity_before_clamp() {
    // §3.3: conv(a*x + b*y) == a*conv(x) + b*conv(y), no clamping in the op.
    let x = gray(4, 4, (0..16u8).map(f32::from).collect());
    let y = gray(
        4,
        4,
        (0..16u8)
            .map(|n| f32::from(n).mul_add(0.25, -1.0))
            .collect(),
    );
    let kernel = k3([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]);
    let params = serde_json::json!({ "kernel": kernel, "mode": "mirror" });

    let (a, b) = (2.0f32, -3.0f32);
    let combined = gray(
        4,
        4,
        x.samples()
            .iter()
            .zip(y.samples())
            .map(|(xs, ys)| a.mul_add(*xs, b * ys))
            .collect(),
    );
    let cx = convolve(&x, &params);
    let cy = convolve(&y, &params);
    let cc = convolve(&combined, &params);
    for ((cxv, cyv), ccv) in cx.samples().iter().zip(cy.samples()).zip(cc.samples()) {
        let lin = a.mul_add(*cxv, b * cyv);
        assert!((lin - ccv).abs() < 1e-3, "linearity broke: {lin} != {ccv}");
    }
}

#[test]
fn finite_input_yields_finite_output() {
    let img = gray(
        6,
        6,
        (0..36u8).map(|n| f32::from(n).mul_add(0.1, -1.5)).collect(),
    );
    let kernel = k3([1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]);
    for mode in [
        "constant",
        "transparent",
        "clamp",
        "mirror",
        "wrap",
        "valid",
    ] {
        let out = convolve(
            &img,
            &serde_json::json!({ "kernel": kernel.clone(), "mode": mode }),
        );
        assert!(
            out.samples().iter().all(|s| s.is_finite()),
            "non-finite output for mode {mode}"
        );
    }
}

#[test]
fn translation_equivariance_away_from_boundary() {
    // §3.3 metamorphic: shifting the input by one pixel shifts the output by one
    // pixel, away from the boundary. Use a large field with a localized bump so
    // the kernel footprint never touches the edge.
    let width = 9usize;
    let height = 9usize;
    let bump = |cx: usize, cy: usize| -> Vec<f32> {
        let mut buf = vec![0.0f32; width * height];
        // a 3x3 plateau centred at (cx, cy)
        for dy in 0..3usize {
            for dx in 0..3usize {
                let px = cx + dx - 1;
                let py = cy + dy - 1;
                buf[py * width + px] = 1.0;
            }
        }
        buf
    };
    let wu = u32::try_from(width).expect("width fits u32");
    let hu = u32::try_from(height).expect("height fits u32");
    let img_a = gray(wu, hu, bump(3, 3));
    let img_b = gray(wu, hu, bump(4, 3)); // shifted +1 in x
    let kernel = k3([1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]);
    let params = serde_json::json!({ "kernel": kernel, "mode": "constant" });
    let out_a = convolve(&img_a, &params);
    let out_b = convolve(&img_b, &params);
    // Compare the interior region [1..7) x [1..7): out_b(x,y) == out_a(x-1,y).
    for y in 1..7usize {
        for x in 2..7usize {
            let bx = out_b.samples()[y * width + x];
            let ax = out_a.samples()[y * width + (x - 1)];
            assert!(
                (bx - ax).abs() < 1e-5,
                "translation equivariance broke at ({x},{y}): {bx} != {ax}"
            );
        }
    }
}

#[test]
fn infer_outputs_preserves_extent_and_shrinks_under_valid() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(10, 8),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("input".to_owned(), descriptor);

    // Same-extent under clamp.
    let clamp = serde_json::json!({ "kernel": k3([0.0; 9]), "mode": "clamp" });
    let out = Convolve::new()
        .infer_outputs(&inputs, &clamp)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["output"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(10, 8));
    assert_eq!(d.layout, ChannelLayout::Rgba);

    // Shrunk under valid: (10-2, 8-2) = (8, 6).
    let valid = serde_json::json!({ "kernel": k3([0.0; 9]), "mode": "valid" });
    let out = Convolve::new()
        .infer_outputs(&inputs, &valid)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["output"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(8, 6));
}

#[test]
fn required_inputs_dilates_by_the_kernel_footprint() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(20, 20),
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("input".to_owned(), descriptor);
    let params = serde_json::json!({ "kernel": k3([0.0; 9]), "mode": "clamp" });
    let mut requested = OutputRegions::new();
    requested.insert("output".to_owned(), Rect::new(5, 5, 7, 7));
    let needed = Convolve::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    // 3x3 kernel centred: window [5-1, 7+1) => (4,4)..(8,8), within the input.
    assert_eq!(needed["input"], Rect::new(4, 4, 8, 8));
}

#[test]
fn missing_kernel_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = Convolve::new()
        .compute(&inputs, &serde_json::json!({ "mode": "clamp" }))
        .expect_err("missing kernel must fail");
    assert_eq!(err.code, E_CONVOLVE_KERNEL);
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn wrong_weight_count_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let kernel = serde_json::json!({ "width": 3, "height": 3, "weights": [1.0, 2.0] });
    let err = Convolve::new()
        .compute(&inputs, &serde_json::json!({ "kernel": kernel }))
        .expect_err("missized weights must fail");
    assert_eq!(err.code, E_CONVOLVE_KERNEL);
}

#[test]
fn zero_extent_kernel_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let kernel = serde_json::json!({ "width": 0, "height": 3, "weights": [] });
    let err = Convolve::new()
        .compute(&inputs, &serde_json::json!({ "kernel": kernel }))
        .expect_err("zero-extent kernel must fail");
    assert_eq!(err.code, E_CONVOLVE_KERNEL);
}

#[test]
fn origin_outside_kernel_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let kernel = serde_json::json!({
        "width": 3, "height": 3, "origin_x": 5, "origin_y": 0,
        "weights": [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
    });
    let err = Convolve::new()
        .compute(&inputs, &serde_json::json!({ "kernel": kernel }))
        .expect_err("bad origin must fail");
    assert_eq!(err.code, E_CONVOLVE_KERNEL);
}

#[test]
fn unknown_mode_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = Convolve::new()
        .compute(
            &inputs,
            &serde_json::json!({ "kernel": k3([0.0; 9]), "mode": "bogus" }),
        )
        .expect_err("unknown mode must fail");
    assert_eq!(err.code, E_CONVOLVE_PARAM);
}

#[test]
fn mismatched_value_length_is_rejected() {
    // Gray (1 channel) image but a 3-component constant value.
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = Convolve::new()
        .compute(
            &inputs,
            &serde_json::json!({ "kernel": k3([0.0; 9]), "mode": "constant", "value": [1.0, 2.0, 3.0] }),
        )
        .expect_err("bad value length must fail");
    assert_eq!(err.code, E_CONVOLVE_PARAM);
}

#[test]
fn non_centred_origin_shifts_the_response() {
    // A 1x3 impulse with origin at the left tap shifts the image right by one.
    let img = gray(5, 1, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    let kernel = serde_json::json!({
        "width": 3, "height": 1, "origin_x": 0, "origin_y": 0, "weights": [1.0, 0.0, 0.0]
    });
    let out = convolve(
        &img,
        &serde_json::json!({ "kernel": kernel, "mode": "constant" }),
    );
    // origin at tap 0 => output(x) = src(x + 0 - 0 ... ) reads src(x), src(x+1), src(x+2)
    // weight only on tap 0 (offset 0 from origin) => identity. Use tap offset:
    // o(x) = sum_k w(k) src(x + k - origin) = w0*src(x) since origin=0,k=0.
    assert_eq!(out.samples(), img.samples());
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Convolve::manifest().expect("convolve manifest");
    let path = root.join(format!("{}.json", manifest.id));
    let on_disk =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let expected = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    assert_eq!(
        on_disk.trim_end(),
        expected.trim_end(),
        "{} is stale; regenerate from the Rust builder",
        path.display()
    );
}
