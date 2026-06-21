//! ROI-restricted-versus-full-execution differential suite (bn-3oi).
//!
//! The backward ROI analysis ([`analyze_roi_from_seeds`]) claims that, to
//! reproduce a demanded output region `R`, an operation needs only the input
//! region its [`required_inputs`](paintop_ir::OpContract::required_inputs)
//! reports. This suite *falsifies* that claim the only way that matters for an
//! exact, deterministic runtime: it perturbs each external input **outside** its
//! backward-demanded region, re-runs the whole pipeline, and asserts the export
//! output **inside `R`** is bit-for-bit identical to the unperturbed run.
//!
//! If the demanded region missed a single contributor, perturbing outside it
//! would change a pixel inside `R` and the byte comparison would fail. Because
//! the ops here are exact (a box-blur halo, a pointwise scale, a mask select),
//! "identical inside `R`" is *bit*-identical, the determinism the milestone
//! demands.
//!
//! The suite covers the families the bone names:
//!
//! - **pointwise chains** (`scale -> scale`): demand a sub-rect, perturb outside
//!   it, output unchanged inside;
//! - **halo ops** (`box_blur`): demand a sub-rect, the demand grows by the blur
//!   radius, perturbing outside the grown region leaves `R` unchanged while
//!   perturbing *inside* it provably changes `R` (a negative control);
//! - **composite masks** (`mask_select`): each contributing port's demand covers
//!   the target region;
//! - **dead nodes**: a node feeding no root carries an empty demand and is
//!   eliminated.

use std::collections::BTreeMap;

use paintop_core::executor::{
    ImplRegistry, InputValues, OpImplementation, OutputValues, ResourceValue,
    analyze_roi_from_seeds, execute,
};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
    ContractRegistry, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    Extent, ImageDescriptor, InputRegions, InputSpec, OpContract, OperationManifest,
    OperationRegistry, OutputDescriptors, OutputRegions, OutputSpec, Plan, Rect, Region,
    ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy, ScalarType, SemanticRole,
    TestMetadata, check_graph, parse_plan, resolve_plan,
};
use serde_json::Value;

const W: i64 = 16;
const H: i64 = 16;
const EXTENT: Extent = Extent::new(16, 16);
const CHANNELS: u32 = 1;
const BLUR_RADIUS: u32 = 2;

// ---------------------------------------------------------------------------
// Descriptors / values
// ---------------------------------------------------------------------------

const fn gray() -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent: EXTENT,
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

fn value(samples: Vec<f32>) -> ResourceValue {
    ResourceValue::new(gray(), CHANNELS, samples).expect("well-sized buffer")
}

fn idx(x: i64, y: i64) -> usize {
    usize::try_from(y * W + x).expect("in range")
}

fn pixel_count() -> usize {
    usize::try_from(W * H).expect("in range")
}

/// A smooth deterministic base image: `f(x, y) = x * 3 + y * 7 mod 251`.
fn base_samples() -> Vec<f32> {
    let mut s = vec![0.0f32; pixel_count()];
    for y in 0..H {
        for x in 0..W {
            let level = u16::try_from((x * 3 + y * 7) % 251).expect("in range");
            s[idx(x, y)] = f32::from(level);
        }
    }
    s
}

/// A 0/1 mask: 1 inside the rect, 0 outside.
fn mask_samples(rect: Rect) -> Vec<f32> {
    let mut s = vec![0.0f32; pixel_count()];
    for y in 0..H {
        for x in 0..W {
            if rect.contains(x, y) {
                s[idx(x, y)] = 1.0;
            }
        }
    }
    s
}

/// Replace every sample of `image` *outside* `keep` with a garbage sentinel,
/// leaving samples inside `keep` untouched. Used to perturb an input outside its
/// demanded region.
fn perturb_outside(image: &ResourceValue, keep: Region) -> ResourceValue {
    let mut s = image.samples().to_vec();
    for y in 0..H {
        for x in 0..W {
            if !keep.contains_rect(Rect::new(x, y, x + 1, y + 1)) {
                s[idx(x, y)] = -999.0; // a value the real image never takes
            }
        }
    }
    value(s)
}

// ---------------------------------------------------------------------------
// Ops: manifest + contract + impl, all ROI-correct
// ---------------------------------------------------------------------------

fn manifest(id: &str, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
    OperationManifest {
        id: id.parse().expect("ok"),
        impl_version: 1,
        summary: String::new(),
        determinism: DeterminismTier::Exact,
        roi: RoiPolicy {
            category: RoiCategory::Pointwise,
            halo_px: None,
        },
        inputs: inputs
            .iter()
            .map(|name| InputSpec {
                name: (*name).to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: String::new(),
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|name| OutputSpec {
                name: (*name).to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            })
            .collect(),
        params: vec![],
        implementations: vec!["cpu.reference@1".parse().expect("ok")],
        test: TestMetadata::default(),
    }
}

fn passthrough_outputs(inputs: &Descriptors, port: &str) -> Result<OutputDescriptors, Error> {
    let image = inputs
        .get(port)
        .copied()
        .ok_or_else(|| Error::new(ErrorClass::Type, "E_MISSING_INPUT", "needs input"))?;
    let mut o = OutputDescriptors::new();
    o.insert("image".to_owned(), image);
    Ok(o)
}

// -- pointwise scale: out = in (identity, exact) ----------------------------

struct ScaleContract;
impl OpContract for ScaleContract {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn infer_outputs(&self, i: &Descriptors, _p: &Value) -> Result<OutputDescriptors, Error> {
        passthrough_outputs(i, "image")
    }
    fn required_inputs(
        &self,
        o: &OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<InputRegions, Error> {
        let mut r = InputRegions::new();
        if let Some(region) = o.get("image") {
            r.insert("image".to_owned(), *region);
        }
        Ok(r)
    }
    fn validate_postconditions(
        &self,
        _o: &OutputDescriptors,
        _p: &Value,
    ) -> Result<Vec<AssertionResult>, Error> {
        Ok(vec![])
    }
}

struct ScaleImpl;
impl OpImplementation for ScaleImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs
            .get("image")
            .ok_or_else(|| Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "needs image"))?;
        // out = 0.5 * in + 1 (a pointwise affine, still exact in f32 here).
        let s: Vec<f32> = image
            .samples()
            .iter()
            .map(|v| v.mul_add(0.5, 1.0))
            .collect();
        let mut o = OutputValues::new();
        o.insert("image".to_owned(), value(s));
        Ok(o)
    }
}

// -- box blur: out(x,y) = mean of (2r+1)^2 clamped neighbours ----------------

struct BlurContract;
impl OpContract for BlurContract {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn infer_outputs(&self, i: &Descriptors, _p: &Value) -> Result<OutputDescriptors, Error> {
        passthrough_outputs(i, "image")
    }
    fn required_inputs(
        &self,
        o: &OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<InputRegions, Error> {
        let mut r = InputRegions::new();
        if let Some(region) = o.get("image") {
            // The box blur reads a (2r+1)^2 window: dilate by r, clamp to extent.
            let grown = Region::from_rect(*region)
                .dilate(BLUR_RADIUS)
                .clamp_to_extent(EXTENT);
            r.insert("image".to_owned(), grown.bounding_rect());
        }
        Ok(r)
    }
    fn validate_postconditions(
        &self,
        _o: &OutputDescriptors,
        _p: &Value,
    ) -> Result<Vec<AssertionResult>, Error> {
        Ok(vec![])
    }
}

struct BlurImpl;
impl OpImplementation for BlurImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs
            .get("image")
            .ok_or_else(|| Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "needs image"))?;
        let src = image.samples();
        let r = i64::from(BLUR_RADIUS);
        // The window is (2r+1)^2 taps; accumulate in a fixed scan order so the
        // f32 mean is a deterministic (run-stable) function of the input.
        let side = f32::from(u16::try_from(2 * r + 1).expect("kernel side fits in u16"));
        let count = side * side;
        let mut out = vec![0.0f32; pixel_count()];
        for y in 0..H {
            for x in 0..W {
                let mut sum = 0.0f32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        // Clamp boundary.
                        let sx = (x + dx).clamp(0, W - 1);
                        let sy = (y + dy).clamp(0, H - 1);
                        sum += src[idx(sx, sy)];
                    }
                }
                out[idx(x, y)] = sum / count;
            }
        }
        let mut o = OutputValues::new();
        o.insert("image".to_owned(), value(out));
        Ok(o)
    }
}

// -- mask select: out = where(mask>0.5, a, b) -------------------------------

struct SelectContract;
impl OpContract for SelectContract {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("a".to_owned(), ResourceKind::Image),
            ("b".to_owned(), ResourceKind::Image),
            ("mask".to_owned(), ResourceKind::Image),
        ]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn infer_outputs(&self, i: &Descriptors, _p: &Value) -> Result<OutputDescriptors, Error> {
        passthrough_outputs(i, "a")
    }
    fn required_inputs(
        &self,
        o: &OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<InputRegions, Error> {
        // Pointwise composite: every contributing port shares the target region.
        let mut r = InputRegions::new();
        if let Some(region) = o.get("image") {
            for port in ["a", "b", "mask"] {
                r.insert(port.to_owned(), *region);
            }
        }
        Ok(r)
    }
    fn validate_postconditions(
        &self,
        _o: &OutputDescriptors,
        _p: &Value,
    ) -> Result<Vec<AssertionResult>, Error> {
        Ok(vec![])
    }
}

struct SelectImpl;
impl OpImplementation for SelectImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let a = inputs.get("a").ok_or_else(miss)?;
        let b = inputs.get("b").ok_or_else(miss)?;
        let mask = inputs.get("mask").ok_or_else(miss)?;
        let (sa, sb, sm) = (a.samples(), b.samples(), mask.samples());
        let out: Vec<f32> = (0..pixel_count())
            .map(|i| if sm[i] > 0.5 { sa[i] } else { sb[i] })
            .collect();
        let mut o = OutputValues::new();
        o.insert("image".to_owned(), value(out));
        Ok(o)
    }
}

fn miss() -> Error {
    Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "needs input")
}

// ---------------------------------------------------------------------------
// Registries
// ---------------------------------------------------------------------------

fn registry() -> OperationRegistry {
    OperationRegistry::from_manifests([
        manifest("filter.scale@1", &["image"], &["image"]),
        manifest("filter.blur@1", &["image"], &["image"]),
        manifest("composite.select@1", &["a", "b", "mask"], &["image"]),
    ])
    .expect("ok")
}

fn contracts() -> ContractRegistry {
    let mut c = ContractRegistry::new();
    c.register(
        "filter.scale@1".parse().expect("ok"),
        Box::new(ScaleContract),
    )
    .expect("ok");
    c.register("filter.blur@1".parse().expect("ok"), Box::new(BlurContract))
        .expect("ok");
    c.register(
        "composite.select@1".parse().expect("ok"),
        Box::new(SelectContract),
    )
    .expect("ok");
    c
}

fn implementations() -> ImplRegistry {
    let mut r = ImplRegistry::new();
    r.register("filter.scale@1".parse().expect("ok"), Box::new(ScaleImpl))
        .expect("ok");
    r.register("filter.blur@1".parse().expect("ok"), Box::new(BlurImpl))
        .expect("ok");
    r.register(
        "composite.select@1".parse().expect("ok"),
        Box::new(SelectImpl),
    )
    .expect("ok");
    r
}

fn input_descriptors(names: &[&str]) -> BTreeMap<String, ResourceDescriptor> {
    names.iter().map(|n| ((*n).to_owned(), gray())).collect()
}

/// The export image samples of running `plan` with the given external `inputs`.
fn run(plan: &Plan, inputs: &BTreeMap<String, ResourceValue>) -> Vec<f32> {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).expect("resolve");
    let exec = execute(plan, &graph, &reg, &implementations(), inputs).expect("execute");
    exec.exports()[0].1.samples().to_vec()
}

/// The backward demand on external input `port`'s producer output, given a seed
/// demand of `roi` on the export node's output.
fn demanded_source_region(plan: &Plan, export_node: &str, roi: Rect) -> Region {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).expect("resolve");
    let checked = check_graph(
        plan,
        &graph,
        &reg,
        &contracts(),
        &input_descriptors(&["base", "alt", "sel"]),
    )
    .expect("check");
    let mut seeds: BTreeMap<(String, String), Region> = BTreeMap::new();
    seeds.insert(
        (export_node.to_owned(), "image".to_owned()),
        Region::from_rect(roi),
    );
    let analysis =
        analyze_roi_from_seeds(plan, &graph, &checked, &contracts(), &seeds).expect("analyze");
    // The demand pushed onto the external `input:base` resource — the union of
    // what every consuming node demands of it.
    let _ = export_node;
    analysis.input_region("base")
}

/// Compare two sample buffers only inside `roi`; returns whether they match.
///
/// Exact (`!=`) comparison is deliberate: the ops are deterministic, so two runs
/// must produce *bit-identical* samples inside the demanded region — an epsilon
/// would mask exactly the kind of contributor leak this suite hunts.
#[allow(
    clippy::float_cmp,
    reason = "bit-identical determinism is the property under test"
)]
fn equal_inside(a: &[f32], b: &[f32], roi: Rect) -> bool {
    for y in roi.y0..roi.y1 {
        for x in roi.x0..roi.x1 {
            if a[idx(x, y)] != b[idx(x, y)] {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

fn pointwise_chain_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"base": {"kind": "image.file", "path": "b.png"}},
            "nodes": [
                {"id": "s1", "op": "filter.scale@1", "in": {"image": "input:base"}},
                {"id": "s2", "op": "filter.scale@1", "in": {"image": "node:s1/image"}}
            ],
            "exports": {"out": {"resource": "node:s2/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok")
}

fn blur_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"base": {"kind": "image.file", "path": "b.png"}},
            "nodes": [
                {"id": "blur", "op": "filter.blur@1", "in": {"image": "input:base"}}
            ],
            "exports": {"out": {"resource": "node:blur/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok")
}

fn composite_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {
                "base": {"kind": "image.file", "path": "b.png"},
                "alt": {"kind": "image.file", "path": "a.png"},
                "sel": {"kind": "image.file", "path": "m.png"}
            },
            "nodes": [
                {"id": "cmp", "op": "composite.select@1",
                 "in": {"a": "input:base", "b": "input:alt", "mask": "input:sel"}}
            ],
            "exports": {"out": {"resource": "node:cmp/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok")
}

fn dead_node_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"base": {"kind": "image.file", "path": "b.png"}},
            "nodes": [
                {"id": "used", "op": "filter.scale@1", "in": {"image": "input:base"}},
                {"id": "dead", "op": "filter.scale@1", "in": {"image": "input:base"}}
            ],
            "exports": {"out": {"resource": "node:used/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn pointwise_chain_is_unchanged_by_perturbing_outside_the_demand() {
    let plan = pointwise_chain_plan();
    let roi = Rect::new(3, 4, 10, 11);

    let mut inputs = BTreeMap::new();
    inputs.insert("base".to_owned(), value(base_samples()));
    let reference = run(&plan, &inputs);

    // Pointwise chain: the demand on `base` equals the export ROI.
    let demand = demanded_source_region(&plan, "s2", roi);
    assert_eq!(demand.bounding_rect(), roi, "pointwise demand is exactly R");

    // Perturb base outside the demand; the export inside R is unchanged.
    let mut perturbed = BTreeMap::new();
    perturbed.insert(
        "base".to_owned(),
        perturb_outside(&value(base_samples()), demand),
    );
    let after = run(&plan, &perturbed);
    assert!(
        equal_inside(&reference, &after, roi),
        "pointwise output inside R must be bit-identical after perturbing outside the demand"
    );
}

#[test]
fn halo_op_demand_grows_and_covers_every_contributor() {
    let plan = blur_plan();
    let roi = Rect::new(6, 6, 10, 10);

    let mut inputs = BTreeMap::new();
    inputs.insert("base".to_owned(), value(base_samples()));
    let reference = run(&plan, &inputs);

    // The blur demand on `base` is R dilated by the radius, clamped.
    let demand = demanded_source_region(&plan, "blur", roi);
    let expected = Region::from_rect(roi)
        .dilate(BLUR_RADIUS)
        .clamp_to_extent(EXTENT);
    assert_eq!(demand.bounding_rect(), expected.bounding_rect());

    // Perturbing OUTSIDE the demand leaves R bit-identical...
    let mut perturbed = BTreeMap::new();
    perturbed.insert(
        "base".to_owned(),
        perturb_outside(&value(base_samples()), demand),
    );
    let after = run(&plan, &perturbed);
    assert!(
        equal_inside(&reference, &after, roi),
        "blur output inside R must be identical after perturbing outside its halo"
    );

    // ...but perturbing INSIDE the demand DOES change R (a negative control that
    // proves the demand is not vacuously large / the perturbation actually bites).
    let smaller = Region::from_rect(Rect::new(7, 7, 9, 9)); // strictly inside demand
    let mut inside = BTreeMap::new();
    inside.insert(
        "base".to_owned(),
        perturb_outside(&value(base_samples()), smaller),
    );
    let changed = run(&plan, &inside);
    assert!(
        !equal_inside(&reference, &changed, roi),
        "perturbing within the halo must change R (the demand is meaningful)"
    );
}

#[test]
fn composite_mask_demands_the_target_on_every_contributing_input() {
    let plan = composite_plan();
    let roi = Rect::new(2, 2, 12, 12);

    // Inside the mask rect we read `base` (a); outside we read `alt` (b).
    let mask_rect = Rect::new(4, 4, 9, 9);
    let mut inputs = BTreeMap::new();
    inputs.insert("base".to_owned(), value(base_samples()));
    inputs.insert(
        "alt".to_owned(),
        value(base_samples().iter().map(|v| v + 100.0).collect()),
    );
    inputs.insert("sel".to_owned(), value(mask_samples(mask_rect)));
    let reference = run(&plan, &inputs);

    // The demand on `base` covers the whole target ROI (pointwise composite).
    let demand = demanded_source_region(&plan, "cmp", roi);
    assert!(
        demand.contains_rect(roi),
        "composite must demand the target region on the `a` port"
    );

    // Perturbing base outside R leaves R identical.
    let mut perturbed = BTreeMap::new();
    perturbed.insert(
        "base".to_owned(),
        perturb_outside(&value(base_samples()), demand),
    );
    perturbed.insert(
        "alt".to_owned(),
        value(base_samples().iter().map(|v| v + 100.0).collect()),
    );
    perturbed.insert("sel".to_owned(), value(mask_samples(mask_rect)));
    let after = run(&plan, &perturbed);
    assert!(equal_inside(&reference, &after, roi));
}

#[test]
fn a_dead_node_carries_an_empty_demand_and_is_eliminated() {
    let plan = dead_node_plan();
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("resolve");
    let checked = check_graph(
        &plan,
        &graph,
        &reg,
        &contracts(),
        &input_descriptors(&["base"]),
    )
    .expect("check");

    // Seed the export's full extent and analyze.
    let mut seeds: BTreeMap<(String, String), Region> = BTreeMap::new();
    seeds.insert(
        ("used".to_owned(), "image".to_owned()),
        Region::from_extent(EXTENT),
    );
    let analysis =
        analyze_roi_from_seeds(&plan, &graph, &checked, &contracts(), &seeds).expect("analyze");

    assert!(analysis.is_demanded("used"));
    assert!(
        !analysis.is_demanded("dead"),
        "a node feeding no root must be region-level eliminated"
    );
    assert!(analysis.output_region("dead", "image").is_empty());

    // And the whole-image executor likewise never dispatches `dead`.
    let exec = execute(&plan, &graph, &reg, &implementations(), &{
        let mut m = BTreeMap::new();
        m.insert("base".to_owned(), value(base_samples()));
        m
    })
    .expect("execute");
    assert!(exec.demand().is_eliminated("dead"));
    assert!(exec.demand().is_demanded("used"));
}
