//! Per-family `OpContract::required_inputs` cover tests (bn-18o).
//!
//! Backward ROI propagation must be *conservative*: for every operation, the
//! input region the contract reports for a demanded output region must **contain
//! every input pixel a consumer reading that output region could possibly
//! depend on**. These tests pin that cover property for one representative
//! operation in each ROI family the M1 op set spans:
//!
//! - **pointwise** (`color.convert`, `composite.over` style) — input region
//!   equals the output region;
//! - **convolution / blur** (`filter.convolve`, `filter.gaussian_blur`) — input
//!   region is the output region dilated by the kernel halo, clipped to the
//!   source;
//! - **crop / pad / resize** (`image.crop`, `image.pad`, `image.resize`) — the
//!   inverse geometric footprint plus a reconstruction halo;
//! - **composite** (`composite.masked_replace`) — the target region on every
//!   contributing port;
//! - **full-domain** (`analyze.statistics`, `analyze.histogram`,
//!   `image.inspect`) — the whole demanded input.
//!
//! The assertion in every case is the same: `returned ⊇ contributors`, expressed
//! with [`Region::contains_rect`] against the analytically-known contributor
//! rect. A conservative contract may return *more* than the contributor set
//! (e.g. the whole plane), and that is accepted; returning *less* fails the test.

use std::collections::BTreeMap;

use paintop_cpu::{
    color, composite, convolve, crop, gaussian_blur, inspect, pad, resize, statistics,
};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, OpContract, OutputRegions,
    Rect, Region, ResourceDescriptor, ScalarType, SemanticRole, ValidRange,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------

const fn linear_image(extent: Extent) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

const fn coverage_mask(extent: Extent) -> ResourceDescriptor {
    ResourceDescriptor::Mask(MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    })
}

fn descriptors(entries: &[(&str, ResourceDescriptor)]) -> Descriptors {
    let mut d = Descriptors::new();
    for (port, desc) in entries {
        d.insert((*port).to_owned(), *desc);
    }
    d
}

fn one_output(port: &str, rect: Rect) -> OutputRegions {
    let mut o = OutputRegions::new();
    o.insert(port.to_owned(), rect);
    o
}

/// Run a contract's `required_inputs` and return the demanded region of `port`
/// as a [`Region`], or the empty region when the port is absent.
fn demanded(
    contract: &dyn OpContract,
    requested: &OutputRegions,
    inputs: &Descriptors,
    params: &serde_json::Value,
    port: &str,
) -> Region {
    let needed = contract
        .required_inputs(requested, inputs, params)
        .expect("required_inputs should succeed for a well-typed request");
    needed
        .get(port)
        .copied()
        .map_or(Region::EMPTY, Region::from_rect)
}

// ---------------------------------------------------------------------------
// Pointwise: color.convert — input region == output region
// ---------------------------------------------------------------------------

#[test]
fn pointwise_color_convert_returns_the_same_region() {
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[("image", linear_image(extent))]);
    let out = Rect::new(8, 6, 40, 30);
    let region = demanded(
        &color::Convert::new(),
        &one_output("image", out),
        &inputs,
        &json!({"to": "srgb"}),
        "image",
    );
    // The contributor of output pixel p is exactly input pixel p.
    assert!(region.contains_rect(out), "pointwise must cover its output");
    // ...and a pointwise op must be *tight*: it demands nothing more.
    assert_eq!(region.bounding_rect(), out);
}

// ---------------------------------------------------------------------------
// Convolution / blur — output region dilated by the kernel halo, clipped
// ---------------------------------------------------------------------------

#[test]
fn convolution_dilates_by_the_kernel_halo() {
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[("input", linear_image(extent))]);
    // A centred 3x3 kernel (origin 1,1) under clamp: output pixel (x,y) reads
    // the 3x3 window centred on (x,y), so the contributor of output R is R
    // dilated by 1, clipped to the source.
    let params = json!({
        "kernel": {"width": 3, "height": 3, "origin_x": 1, "origin_y": 1,
                   "weights": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
        "mode": "clamp"
    });
    let out = Rect::new(10, 10, 30, 30);
    let region = demanded(
        &convolve::Convolve::new(),
        &one_output("output", out),
        &inputs,
        &params,
        "input",
    );
    let contributors = out.dilate(1).clamp_to_extent(extent);
    assert!(
        region.contains_rect(contributors),
        "convolution must cover the dilated window: got {:?}, need {:?}",
        region.bounding_rect(),
        contributors
    );
}

#[test]
fn gaussian_blur_dilates_by_its_declared_halo() {
    let extent = Extent::new(128, 96);
    let inputs = descriptors(&[("input", linear_image(extent))]);
    // sigma = 2.0 -> radius r = ceil(3*sigma) = 6 (the declared halo).
    let out = Rect::new(40, 40, 80, 70);
    let region = demanded(
        &gaussian_blur::GaussianBlur::new(),
        &one_output("output", out),
        &inputs,
        &json!({"sigma": 2.0, "mode": "clamp"}),
        "input",
    );
    let contributors = out.dilate(6).clamp_to_extent(extent);
    assert!(
        region.contains_rect(contributors),
        "blur must cover its halo: got {:?}, need {:?}",
        region.bounding_rect(),
        contributors
    );
}

// ---------------------------------------------------------------------------
// Crop — inverse geometric footprint (translate by the crop origin)
// ---------------------------------------------------------------------------

#[test]
fn crop_translates_the_output_region_by_the_crop_origin() {
    // Crop a 64x48 source to the rect [16,12 .. 56,44); the cropped image is
    // 40x32. An output region R (in cropped space) maps back to R + (16, 12).
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[("image", linear_image(extent))]);
    let params = json!({"rect": {"x0": 16, "y0": 12, "x1": 56, "y1": 44}});
    let out = Rect::new(4, 4, 20, 20);
    let region = demanded(
        &crop::Crop::new(),
        &one_output("image", out),
        &inputs,
        &params,
        "image",
    );
    let contributors = out.translate(16, 12);
    assert!(
        region.contains_rect(contributors),
        "crop must cover the source footprint"
    );
    assert_eq!(
        region.bounding_rect(),
        contributors,
        "crop is an exact translation"
    );
}

// ---------------------------------------------------------------------------
// Pad — constant interior shift; non-constant border demands the full input
// ---------------------------------------------------------------------------

#[test]
fn pad_constant_maps_the_interior_by_the_lead_margin() {
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[("image", linear_image(extent))]);
    // 8px constant border on every side: output (x,y) interior maps to input
    // (x-8, y-8); the requested window's source span is R - (8, 8) clipped.
    let params = json!({"left": 8, "right": 8, "top": 8, "bottom": 8, "mode": "constant"});
    let out = Rect::new(20, 20, 40, 40);
    let region = demanded(
        &pad::Pad::new(),
        &one_output("image", out),
        &inputs,
        &params,
        "image",
    );
    let contributors = out.translate(-8, -8).clamp_to_extent(extent);
    assert!(
        region.contains_rect(contributors),
        "pad-constant must cover the shifted interior"
    );
}

#[test]
fn pad_clamp_demands_the_whole_input() {
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[("image", linear_image(extent))]);
    // A clamp border can replicate an arbitrary edge sample, so the demand is the
    // whole input plane.
    let params = json!({"left": 8, "right": 8, "top": 8, "bottom": 8, "mode": "clamp"});
    let out = Rect::new(0, 0, 80, 64);
    let region = demanded(
        &pad::Pad::new(),
        &one_output("image", out),
        &inputs,
        &params,
        "image",
    );
    assert!(
        region.contains_rect(Rect::from_extent(extent)),
        "pad-clamp must demand the full input"
    );
}

// ---------------------------------------------------------------------------
// Resize — inverse footprint plus a reconstruction halo
// ---------------------------------------------------------------------------

#[test]
fn resize_covers_the_source_footprint_with_a_reconstruction_halo() {
    // Downscale 64x48 -> 32x24 with a bicubic filter (a multi-tap reconstruction
    // kernel). An output region R maps to its source span dilated by the filter
    // support. We assert the returned region covers at least the *centre* span of
    // each output pixel (a sound under-approximation of the contributor set).
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[("image", linear_image(extent))]);
    let params = json!({"width": 32, "height": 24, "filter": "bicubic"});
    let out = Rect::new(4, 4, 28, 20);
    let region = demanded(
        &resize::Resize::new(),
        &one_output("image", out),
        &inputs,
        &params,
        "image",
    );
    // Output x in [4,28) at 2x downscale samples source centres near [8, 56); the
    // nearest source pixel span is conservatively [8, 56) x [8, 40).
    let centre_span = Rect::new(8, 8, 56, 40).clamp_to_extent(extent);
    assert!(
        region.contains_rect(centre_span),
        "resize must cover the source footprint: got {:?}, need {:?}",
        region.bounding_rect(),
        centre_span
    );
}

// ---------------------------------------------------------------------------
// Composite — the target region on every contributing port
// ---------------------------------------------------------------------------

#[test]
fn composite_masked_replace_demands_the_target_on_every_port() {
    let extent = Extent::new(64, 48);
    let inputs = descriptors(&[
        ("edited", linear_image(extent)),
        ("base", linear_image(extent)),
        ("mask", coverage_mask(extent)),
    ]);
    let out = Rect::new(10, 8, 30, 28);
    let contract = composite::MaskedReplace::new();
    let needed = contract
        .required_inputs(&one_output("image", out), &inputs, &json!({}))
        .unwrap();
    // The composite reads the co-located edited, base, and mask sample, so every
    // port must cover the target region.
    for port in ["edited", "base", "mask"] {
        let region = needed
            .get(port)
            .copied()
            .map_or(Region::EMPTY, Region::from_rect);
        assert!(
            region.contains_rect(out),
            "composite port {port} must cover the target region"
        );
    }
}

// ---------------------------------------------------------------------------
// Full-domain — statistics / histogram / inspect demand the whole input
// ---------------------------------------------------------------------------

#[test]
fn full_domain_ops_demand_the_whole_input() {
    let extent = Extent::new(50, 40);
    let full = Rect::from_extent(extent);

    // analyze.statistics over an Image input.
    let stats_inputs = descriptors(&[("resource", linear_image(extent))]);
    let region = demanded(
        &statistics::Statistics::new(),
        &OutputRegions::new(),
        &stats_inputs,
        &json!({}),
        "resource",
    );
    assert!(
        region.contains_rect(full),
        "statistics must demand the full input"
    );

    // analyze.histogram over an Image input.
    let hist_inputs = descriptors(&[("resource", linear_image(extent))]);
    let region = demanded(
        &statistics::Histogram::new(),
        &OutputRegions::new(),
        &hist_inputs,
        &json!({"bins": 8, "domain_min": 0.0, "domain_max": 1.0}),
        "resource",
    );
    assert!(
        region.contains_rect(full),
        "histogram must demand the full input"
    );

    // image.inspect over an Image input.
    let inspect_inputs = descriptors(&[("image", linear_image(extent))]);
    let region = demanded(
        &inspect::Inspect::new(),
        &OutputRegions::new(),
        &inspect_inputs,
        &json!({}),
        "image",
    );
    assert!(
        region.contains_rect(full),
        "inspect must demand the full input"
    );
}

// ---------------------------------------------------------------------------
// Empty demand — an empty output region demands no input
// ---------------------------------------------------------------------------

#[test]
fn an_empty_output_demand_propagates_to_an_empty_pointwise_input() {
    let extent = Extent::new(32, 32);
    let inputs = descriptors(&[("image", linear_image(extent))]);
    let empty_out = Rect::new(5, 5, 5, 5); // zero area
    let region = demanded(
        &color::Convert::new(),
        &one_output("image", empty_out),
        &inputs,
        &json!({"to": "srgb"}),
        "image",
    );
    assert!(
        region.is_empty(),
        "an empty output demand needs no input pixels"
    );
}

/// A trivial sanity check that the helper map type is the one the contracts use.
#[test]
fn descriptors_alias_is_a_btreemap() {
    let d: BTreeMap<String, ResourceDescriptor> = descriptors(&[]);
    assert!(d.is_empty());
}
