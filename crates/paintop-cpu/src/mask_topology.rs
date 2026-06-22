//! Hard-mask topology operations `mask.connected_components@1`,
//! `mask.fill_holes@1`, and `mask.remove_components@1` (`OP_CATALOG` §4,
//! `AGENT_VERIFICATION` §2.3).
//!
//! These three ops reason about the *connectivity* of a hard (selection) mask —
//! which foreground pixels form one blob, which background pixels are enclosed
//! holes — rather than about per-pixel coverage. They share one foreground
//! predicate, one pixel-connectivity model, and one deterministic component
//! labeling, defined once here.
//!
//! # Hard-mask predicate
//!
//! A pixel is **foreground** iff its coverage sample is `>= 0.5`. On a hard mask
//! (every sample exactly `0` or `1`) this is unambiguous; on a soft mask it is the
//! half-coverage threshold, matching `mask.to_sdf`'s default contour. The
//! predicate is the only place a coverage value is consulted; everything
//! downstream is purely topological.
//!
//! # Connectivity
//!
//! Adjacency is either **4-connectivity** (the von Neumann neighborhood: up,
//! down, left, right) or **8-connectivity** (the Moore neighborhood, adding the
//! four diagonals), selected by the explicit `connectivity` param. The two give
//! genuinely different component counts (a diagonal chain of pixels is one
//! 8-component but `n` separate 4-components), so the choice is never implicit.
//!
//! # Label stability (`mask.connected_components@1`)
//!
//! Components are numbered `1, 2, 3, …` in **raster-scan order of their first
//! pixel**: the labeler scans pixels top-to-bottom, left-to-right, and the first
//! time it meets an unlabeled foreground pixel it assigns the next free label and
//! floods the whole component with it. Two pixels get the same label iff they are
//! connected. Label `0` is reserved for the background. This order is a total
//! function of the input bitmap and the connectivity, so the labeling is
//! deterministic and stable — the same mask always yields the same IDs
//! (`OP_CATALOG` §4 "label stability policy").
//!
//! # The `LabelMap` resource and large IDs
//!
//! `mask.connected_components@1` produces a [`ResourceKind::LabelMap`]: a
//! single-channel `u32` raster of component IDs plus a [`Report`] carrying the
//! component count and per-label areas. The run-time value stores each `u32` ID
//! **losslessly** by reinterpreting its bit pattern as an `f32`
//! ([`f32::from_bits`]); IDs above `2^24` (which `f32` cannot represent as a
//! number) therefore survive a round trip unchanged, the integer-encoding-loss
//! case of `AGENT_VERIFICATION` §2.3.
//!
//! # Determinism
//!
//! Every op is [`DeterminismTier::Exact`]: the predicate is a threshold compare,
//! the flood fill visits pixels in a fixed order, and label assignment is the
//! raster-scan rule above. No floating-point arithmetic enters the topology, so a
//! tile boundary cannot change a result — the connectivity footprint is the whole
//! connected domain ([`RoiCategory::FullDomain`]).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, ComponentsData, CoordinateConvention, Descriptors, DeterminismTier, Error,
    ErrorClass, Extent, ImplId, InputRegions, InputSpec, LabelMapDescriptor, MaskDescriptor,
    MaskMeaning, OpContract, OperationManifest, OutputDescriptors, OutputRegions, OutputSpec,
    ParamSpec, ParamType, Rect, Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, ScalarType, TestMetadata, ValidRange,
};

/// The canonical id of the connected-components labeling operation.
pub const CONNECTED_COMPONENTS_OP_ID: &str = "mask.connected_components@1";
/// The canonical id of the hole-filling operation.
pub const FILL_HOLES_OP_ID: &str = "mask.fill_holes@1";
/// The canonical id of the component-removal operation.
pub const REMOVE_COMPONENTS_OP_ID: &str = "mask.remove_components@1";

/// A required mask input was absent or was not a mask.
pub const E_TOPOLOGY_INPUT: &str = "E_TOPOLOGY_INPUT";
/// A topology parameter was malformed or out of range.
pub const E_TOPOLOGY_PARAM: &str = "E_TOPOLOGY_PARAM";
/// A produced buffer had an unexpected length.
pub const E_TOPOLOGY_BUFFER: &str = "E_TOPOLOGY_BUFFER";

/// The foreground threshold: a pixel is foreground iff its coverage is `>= 0.5`.
const FOREGROUND_THRESHOLD: f32 = 0.5;

// ---------------------------------------------------------------------------
// shared connectivity + labeling core
// ---------------------------------------------------------------------------

/// The pixel adjacency model a topology op uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Connectivity {
    /// 4-connectivity: up, down, left, right (von Neumann).
    Four,
    /// 8-connectivity: the four orthogonal plus the four diagonal neighbors
    /// (Moore).
    Eight,
}

impl Connectivity {
    /// The numeric tag (`4` or `8`) recorded in the report.
    const fn tag(self) -> u8 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

/// Resolve the optional `connectivity` param (`4` or `8`, default `8`).
fn resolve_connectivity(params: &serde_json::Value, op: &str) -> Result<Connectivity> {
    match params.get("connectivity") {
        None | Some(serde_json::Value::Null) => Ok(Connectivity::Eight),
        Some(v) => match v.as_i64() {
            Some(4) => Ok(Connectivity::Four),
            Some(8) => Ok(Connectivity::Eight),
            _ => Err(Error::new(
                ErrorClass::Schema,
                E_TOPOLOGY_PARAM,
                format!("{op} `connectivity` must be 4 or 8"),
            )),
        },
    }
}

/// A binary foreground bitmap with its extent, the common substrate of every
/// topology op.
struct Bitmap {
    width: usize,
    height: usize,
    /// Row-major foreground membership, length `width * height`.
    fore: Vec<bool>,
}

impl Bitmap {
    /// Build the foreground bitmap of a coverage buffer under the half-coverage
    /// predicate.
    fn from_coverage(extent: Extent, coverage: &[f32]) -> Self {
        let fore: Vec<bool> = coverage
            .iter()
            .map(|&c| c >= FOREGROUND_THRESHOLD)
            .collect();
        Self {
            width: extent.width as usize,
            height: extent.height as usize,
            fore,
        }
    }

    /// The number of pixels.
    const fn len(&self) -> usize {
        self.width * self.height
    }

    /// Push the (up to 8) neighbor indices of `(x, y)` under `conn` onto `out`.
    fn neighbors(&self, x: usize, y: usize, conn: Connectivity, out: &mut Vec<usize>) {
        out.clear();
        let w = self.width;
        let h = self.height;
        // Orthogonal neighbors (both connectivities).
        if x > 0 {
            out.push(y * w + (x - 1));
        }
        if x + 1 < w {
            out.push(y * w + (x + 1));
        }
        if y > 0 {
            out.push((y - 1) * w + x);
        }
        if y + 1 < h {
            out.push((y + 1) * w + x);
        }
        if conn == Connectivity::Eight {
            if x > 0 && y > 0 {
                out.push((y - 1) * w + (x - 1));
            }
            if x + 1 < w && y > 0 {
                out.push((y - 1) * w + (x + 1));
            }
            if x > 0 && y + 1 < h {
                out.push((y + 1) * w + (x - 1));
            }
            if x + 1 < w && y + 1 < h {
                out.push((y + 1) * w + (x + 1));
            }
        }
    }
}

/// The result of labeling a bitmap's foreground components.
struct Labeling {
    /// Per-pixel component label, `0` for background, `1..=count` for foreground,
    /// length `width * height`.
    labels: Vec<u32>,
    /// The number of components.
    count: u32,
    /// The pixel area of each component in label order (`areas[i]` is label
    /// `i + 1`), length `count`.
    areas: Vec<u64>,
}

/// Label the foreground components of `bitmap` under `conn`, numbering them in
/// raster-scan order of their first pixel (the stable-label policy).
fn label_components(bitmap: &Bitmap, conn: Connectivity) -> Labeling {
    let mut labels = vec![0_u32; bitmap.len()];
    let mut areas: Vec<u64> = Vec::new();
    let mut count: u32 = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut nbrs: Vec<usize> = Vec::with_capacity(8);

    for y in 0..bitmap.height {
        for x in 0..bitmap.width {
            let seed = y * bitmap.width + x;
            if !bitmap.fore[seed] || labels[seed] != 0 {
                continue;
            }
            // New component: assign the next label and flood it.
            count += 1;
            let label = count;
            let mut area: u64 = 0;
            labels[seed] = label;
            stack.push(seed);
            while let Some(idx) = stack.pop() {
                area += 1;
                let px = idx % bitmap.width;
                let py = idx / bitmap.width;
                bitmap.neighbors(px, py, conn, &mut nbrs);
                for &n in &nbrs {
                    if bitmap.fore[n] && labels[n] == 0 {
                        labels[n] = label;
                        stack.push(n);
                    }
                }
            }
            areas.push(area);
        }
    }
    Labeling {
        labels,
        count,
        areas,
    }
}

// ---------------------------------------------------------------------------
// shared input/output helpers
// ---------------------------------------------------------------------------

/// The descriptor of the `mask` input, or a typed error if absent / wrong kind.
fn mask_descriptor_of<'a>(inputs: &'a Descriptors, op: &str) -> Result<&'a MaskDescriptor> {
    match inputs.get("mask") {
        Some(ResourceDescriptor::Mask(d)) => Ok(d),
        Some(_) => Err(Error::new(
            ErrorClass::Type,
            E_TOPOLOGY_INPUT,
            format!("{op} input `mask` must be a Mask"),
        )),
        None => Err(Error::new(
            ErrorClass::Reference,
            E_TOPOLOGY_INPUT,
            format!("{op} requires a `mask` input"),
        )),
    }
}

/// The value of the `mask` input, or a typed error if absent.
fn mask_value_of<'a>(
    inputs: &'a InputValues,
    op: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_TOPOLOGY_INPUT,
            format!("{op} requires a `mask` input value"),
        )
    })
}

/// The required `mask` input port declaration.
fn mask_input_port(doc: &str) -> InputSpec {
    InputSpec {
        name: "mask".to_owned(),
        kind: ResourceKind::Mask,
        required: true,
        doc: doc.to_owned(),
    }
}

/// The `connectivity` param declaration shared by every topology op.
fn connectivity_param() -> ParamSpec {
    ParamSpec {
        name: "connectivity".to_owned(),
        ty: ParamType::Integer,
        unit: None,
        required: false,
        default: Some(serde_json::json!(8)),
        choices: vec![],
        doc: "Pixel adjacency: 4 (orthogonal) or 8 (orthogonal + diagonal). \
              Default 8; any other value is rejected."
            .to_owned(),
    }
}

/// The full-domain input region a topology op demands of `mask`, if present.
fn full_region(inputs: &Descriptors, regions: &mut InputRegions) {
    if let Some(d) = inputs.get("mask") {
        let extent = d.extent();
        regions.insert(
            "mask".to_owned(),
            Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
        );
    }
}

/// The hard-selection mask descriptor produced for `extent` (output of the
/// hole-filling and component-removal ops).
const fn selection_mask_descriptor(extent: Extent) -> MaskDescriptor {
    MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Selection,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The `u32` label-map descriptor produced for `extent`.
const fn label_map_descriptor(extent: Extent) -> LabelMapDescriptor {
    LabelMapDescriptor {
        extent,
        scalar: ScalarType::U32,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// Wrap a hard `{0, 1}` selection buffer as a mask value, mapping a length
/// mismatch to a typed error.
fn finish_mask(
    extent: Extent,
    samples: Vec<f32>,
    op: &str,
) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(
        ResourceDescriptor::Mask(selection_mask_descriptor(extent)),
        1,
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_TOPOLOGY_BUFFER,
            format!("{op} produced a mask buffer of unexpected length {actual}"),
        )
    })?;
    let mut out = OutputValues::new();
    out.insert("mask".to_owned(), value);
    Ok(out)
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for an exact, single-reference topology op:
/// perceptual does not apply; every other applicable category is covered by this
/// module's synthetic-blob fixtures, property/connectivity tests, and metamorphic
/// laws.
fn topology_test_metadata(perceptual_reason: &str) -> TestMetadata {
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

// ---------------------------------------------------------------------------
// mask.connected_components@1
// ---------------------------------------------------------------------------

/// The `mask.connected_components@1` operation: a hard mask → a `LabelMap` of
/// `u32` component IDs plus a component-summary `Report`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnectedComponents;

impl ConnectedComponents {
    /// Construct the op.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.connected_components@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the hard-coded op or
    /// impl ids are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: CONNECTED_COMPONENTS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Label the connected foreground components of a hard mask (coverage >= 0.5) \
                      into a u32 LabelMap (0 = background) plus a report of the component count \
                      and per-label areas. Connectivity (4 or 8) is an explicit param; labels \
                      are numbered in raster-scan order of each component's first pixel (a \
                      stable, deterministic policy)."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![mask_input_port(
                "The hard mask whose foreground (coverage >= 0.5) components are labeled.",
            )],
            outputs: vec![
                OutputSpec {
                    name: "labels".to_owned(),
                    kind: ResourceKind::LabelMap,
                    doc: "The u32 label map: 0 for background, 1..=count for the \
                          raster-scan-ordered components."
                        .to_owned(),
                },
                OutputSpec {
                    name: "report".to_owned(),
                    kind: ResourceKind::Report,
                    doc: "The component count, connectivity, and per-label pixel areas.".to_owned(),
                },
            ],
            params: vec![connectivity_param()],
            implementations: vec![reference_impl()?],
            test: topology_test_metadata(
                "connected-component labeling is exact integer topology verified by synthetic \
                 multi-blob fixtures, 4- vs 8-connectivity counts, label stability, and \
                 large-ID round trips; there is no perceptual metric",
            ),
        })
    }
}

impl OpContract for ConnectedComponents {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("labels".to_owned(), ResourceKind::LabelMap),
            ("report".to_owned(), ResourceKind::Report),
        ]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = mask_descriptor_of(inputs, CONNECTED_COMPONENTS_OP_ID)?.extent;
        let mut out = OutputDescriptors::new();
        out.insert(
            "labels".to_owned(),
            ResourceDescriptor::LabelMap(label_map_descriptor(extent)),
        );
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent,
                channels: 1,
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
        full_region(inputs, &mut regions);
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let labels_ok = matches!(outputs.get("labels"), Some(ResourceDescriptor::LabelMap(_)));
        let report_ok = matches!(outputs.get("report"), Some(ResourceDescriptor::Report(_)));
        Ok(vec![
            if labels_ok {
                AssertionResult::pass("produces_label_map")
            } else {
                AssertionResult::fail("produces_label_map", "no `labels` LabelMap output produced")
            },
            if report_ok {
                AssertionResult::pass("produces_report")
            } else {
                AssertionResult::fail("produces_report", "no `report` output produced")
            },
        ])
    }
}

impl OpImplementation for ConnectedComponents {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = mask_value_of(inputs, CONNECTED_COMPONENTS_OP_ID)?;
        let conn = resolve_connectivity(params, CONNECTED_COMPONENTS_OP_ID)?;
        let bitmap = Bitmap::from_coverage(mask.extent(), mask.samples());
        let labeling = label_components(&bitmap, conn);

        // Store each u32 label losslessly as the f32 bit pattern.
        let samples: Vec<f32> = labeling.labels.iter().map(|&l| f32::from_bits(l)).collect();
        let labels_value = ResourceValue::new(
            ResourceDescriptor::LabelMap(label_map_descriptor(mask.extent())),
            1,
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_TOPOLOGY_BUFFER,
                format!(
                    "{CONNECTED_COMPONENTS_OP_ID} produced a label buffer of unexpected length \
                     {actual}"
                ),
            )
        })?;

        let report = component_report(mask.extent(), conn, &labeling);

        let mut out = OutputValues::new();
        out.insert("labels".to_owned(), labels_value);
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// Build the component-summary report for a labeling.
fn component_report(extent: Extent, conn: Connectivity, labeling: &Labeling) -> Report {
    Report {
        extent,
        channels: 1,
        channel_stats: Vec::new(),
        all_finite: true,
        content_hash: String::new(),
        diff: None,
        assertion: None,
        histogram: None,
        components: Some(ComponentsData {
            connectivity: conn.tag(),
            count: labeling.count,
            areas: labeling.areas.clone(),
        }),
        frequency_energy: None,
        solver: None,
    }
}

/// Recover the `u32` label of an `f32` sample (the inverse of the lossless
/// bit-pattern encoding).
#[must_use]
pub const fn label_of_sample(sample: f32) -> u32 {
    sample.to_bits()
}

// ---------------------------------------------------------------------------
// mask.fill_holes@1
// ---------------------------------------------------------------------------

/// The `mask.fill_holes@1` operation: a hard mask → a hard mask with every
/// enclosed background hole filled to foreground.
#[derive(Debug, Clone, Copy, Default)]
pub struct FillHoles;

impl FillHoles {
    /// Construct the op.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.fill_holes@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the hard-coded op or
    /// impl ids are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FILL_HOLES_OP_ID.parse()?,
            impl_version: 1,
            summary: "Fill the enclosed holes of a hard mask: every background pixel (coverage < \
                      0.5) that is NOT connected to the image border becomes foreground. \
                      Border-connected background is left untouched. Connectivity of the \
                      background flood is an explicit param (4 or 8)."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![mask_input_port(
                "The hard mask whose non-border-connected background holes are filled.",
            )],
            outputs: vec![OutputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                doc: "The hard selection mask with enclosed holes filled to foreground.".to_owned(),
            }],
            params: vec![connectivity_param()],
            implementations: vec![reference_impl()?],
            test: topology_test_metadata(
                "hole filling is exact integer topology verified by synthetic ring/hole \
                 fixtures, the border-connected definition, and 4- vs 8-connectivity; there is \
                 no perceptual metric",
            ),
        })
    }
}

impl OpContract for FillHoles {
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
        let extent = mask_descriptor_of(inputs, FILL_HOLES_OP_ID)?.extent;
        let mut out = OutputDescriptors::new();
        out.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(selection_mask_descriptor(extent)),
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
        full_region(inputs, &mut regions);
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(selection_postcondition(outputs))
    }
}

impl OpImplementation for FillHoles {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = mask_value_of(inputs, FILL_HOLES_OP_ID)?;
        let conn = resolve_connectivity(params, FILL_HOLES_OP_ID)?;
        let bitmap = Bitmap::from_coverage(mask.extent(), mask.samples());
        let filled = fill_holes(&bitmap, conn);
        let samples: Vec<f32> = filled.iter().map(|&f| if f { 1.0 } else { 0.0 }).collect();
        finish_mask(mask.extent(), samples, FILL_HOLES_OP_ID)
    }
}

/// Compute the hole-filled foreground bitmap.
///
/// A background pixel is part of a *hole* iff it cannot be reached from the image
/// border through background pixels (under `conn`). The output foreground is the
/// original foreground plus every such hole pixel.
fn fill_holes(bitmap: &Bitmap, conn: Connectivity) -> Vec<bool> {
    let n = bitmap.len();
    // Flood the *background* from the border inward.
    let mut border_bg = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut nbrs: Vec<usize> = Vec::with_capacity(8);

    let seed_border = |x: usize, y: usize, stack: &mut Vec<usize>, border_bg: &mut Vec<bool>| {
        let idx = y * bitmap.width + x;
        if !bitmap.fore[idx] && !border_bg[idx] {
            border_bg[idx] = true;
            stack.push(idx);
        }
    };
    if bitmap.width > 0 && bitmap.height > 0 {
        for x in 0..bitmap.width {
            seed_border(x, 0, &mut stack, &mut border_bg);
            seed_border(x, bitmap.height - 1, &mut stack, &mut border_bg);
        }
        for y in 0..bitmap.height {
            seed_border(0, y, &mut stack, &mut border_bg);
            seed_border(bitmap.width - 1, y, &mut stack, &mut border_bg);
        }
    }
    while let Some(idx) = stack.pop() {
        let px = idx % bitmap.width;
        let py = idx / bitmap.width;
        bitmap.neighbors(px, py, conn, &mut nbrs);
        for &nb in &nbrs {
            if !bitmap.fore[nb] && !border_bg[nb] {
                border_bg[nb] = true;
                stack.push(nb);
            }
        }
    }
    // Output: foreground OR (background AND not border-connected) = NOT
    // border-connected-background.
    (0..n).map(|i| bitmap.fore[i] || !border_bg[i]).collect()
}

// ---------------------------------------------------------------------------
// mask.remove_components@1
// ---------------------------------------------------------------------------

/// The `mask.remove_components@1` operation: a hard mask → a hard mask with every
/// foreground component smaller than a minimum pixel area removed.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveComponents;

impl RemoveComponents {
    /// Construct the op.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.remove_components@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the hard-coded op or
    /// impl ids are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: REMOVE_COMPONENTS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Remove the small foreground components of a hard mask: every connected \
                      component with a pixel area strictly below `min_area` is cleared to \
                      background. Connectivity (4 or 8) is an explicit param; `min_area` is a \
                      required non-negative integer."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![mask_input_port(
                "The hard mask whose below-threshold foreground components are removed.",
            )],
            outputs: vec![OutputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                doc: "The hard selection mask with small components cleared to background."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "min_area".to_owned(),
                    ty: ParamType::Integer,
                    unit: Some(paintop_ir::ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The minimum component area in pixels: a component with fewer than \
                          `min_area` pixels is removed."
                        .to_owned(),
                },
                connectivity_param(),
            ],
            implementations: vec![reference_impl()?],
            test: topology_test_metadata(
                "component removal is exact integer topology verified by synthetic multi-blob \
                 fixtures at threshold boundaries and 4- vs 8-connectivity; there is no \
                 perceptual metric",
            ),
        })
    }
}

impl OpContract for RemoveComponents {
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
        let extent = mask_descriptor_of(inputs, REMOVE_COMPONENTS_OP_ID)?.extent;
        let mut out = OutputDescriptors::new();
        out.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(selection_mask_descriptor(extent)),
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
        full_region(inputs, &mut regions);
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(selection_postcondition(outputs))
    }
}

impl OpImplementation for RemoveComponents {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = mask_value_of(inputs, REMOVE_COMPONENTS_OP_ID)?;
        let conn = resolve_connectivity(params, REMOVE_COMPONENTS_OP_ID)?;
        let min_area = resolve_min_area(params)?;
        let bitmap = Bitmap::from_coverage(mask.extent(), mask.samples());
        let labeling = label_components(&bitmap, conn);

        let samples: Vec<f32> = labeling
            .labels
            .iter()
            .map(|&label| {
                if label == 0 {
                    0.0
                } else {
                    // areas[label - 1] is the component's pixel area.
                    let area = labeling.areas[(label - 1) as usize];
                    if area >= min_area { 1.0 } else { 0.0 }
                }
            })
            .collect();
        finish_mask(mask.extent(), samples, REMOVE_COMPONENTS_OP_ID)
    }
}

/// Resolve the required non-negative `min_area` integer param.
fn resolve_min_area(params: &serde_json::Value) -> Result<u64> {
    let value = params
        .get("min_area")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_TOPOLOGY_PARAM,
                format!("{REMOVE_COMPONENTS_OP_ID} requires an integer `min_area` parameter"),
            )
        })?;
    u64::try_from(value).map_err(|_| {
        Error::new(
            ErrorClass::Schema,
            E_TOPOLOGY_PARAM,
            format!("{REMOVE_COMPONENTS_OP_ID} `min_area` must be >= 0, got {value}"),
        )
    })
}

// ---------------------------------------------------------------------------
// shared postcondition
// ---------------------------------------------------------------------------

/// The selection-mask postcondition: a `[0, 1]` hard-selection mask output.
fn selection_postcondition(outputs: &OutputDescriptors) -> Vec<AssertionResult> {
    let Some(ResourceDescriptor::Mask(mask)) = outputs.get("mask") else {
        return vec![AssertionResult::fail(
            "produces_mask",
            "no `mask` output produced",
        )];
    };
    vec![
        AssertionResult::pass("produces_mask"),
        if mask.meaning == MaskMeaning::Selection {
            AssertionResult::pass("hard_selection")
        } else {
            AssertionResult::fail(
                "hard_selection",
                format!("mask meaning {:?} is not a hard Selection", mask.meaning),
            )
        },
    ]
}

#[cfg(test)]
mod tests;
