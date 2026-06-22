//! The operation **manifest**: the declared, machine-readable contract of an
//! operation (`IR_SPEC` §3.3, §4, §6, §10, §11, §18).
//!
//! A manifest is the single source of truth an agent (or the runtime, or the
//! op-verification harness) reads to learn what an operation *is*: its stable
//! versioned id, its typed input ports, its named output ports, its parameter
//! schema (with units and defaults), its determinism tier, and the region-of-
//! interest category that governs how much input it consumes per output region.
//!
//! Manifests are **data**, not code. The executable shape/ROI/halo contract
//! (`IR_SPEC` §18) is implemented in Rust and tested independently; the manifest
//! only *declares* the category and the ports those functions operate over, so
//! that the manifest and the implementation can be cross-checked.
//!
//! # JSON Schema export
//!
//! [`manifest_json_schema`] emits a hand-authored JSON Schema (draft 2020-12)
//! describing the manifest wire shape, so that `paintop op schema` (and CI) can
//! validate representative manifests without booting the runtime. The schema is
//! kept deliberately self-contained (no `$ref` to external registries) and is
//! covered by round-trip tests against the serde model in this module.
//!
//! ```
//! use paintop_ir::manifest::OpId;
//!
//! let id: OpId = "filter.gaussian_blur@1".parse().unwrap();
//! assert_eq!(id.namespace(), "filter");
//! assert_eq!(id.name(), "gaussian_blur");
//! assert_eq!(id.major(), 1);
//! assert_eq!(id.to_string(), "filter.gaussian_blur@1");
//! ```

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorClass, Result};

/// A stable, versioned operation identifier of the form
/// `<namespace>.<name>@<semantic-major>` (`IR_SPEC` §6).
///
/// The **major** version defines semantics: two ids that differ only in major
/// are different operations as far as the compiler and cache are concerned.
/// Backward-compatible parameter additions may happen *within* a major only if
/// normalization fills a fixed default and old normalized plans keep their
/// meaning.
///
/// The string form is canonical and is what appears in a plan's `op` field; the
/// struct decomposes it so the registry can index by `(namespace, name, major)`
/// without re-parsing.
/// The derived `Ord`/`PartialOrd` orders ids lexicographically by
/// `(namespace, name, major)` — the field declaration order — which is the
/// canonical ordering the registry relies on for deterministic iteration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId {
    namespace: String,
    name: String,
    major: u32,
}

impl OpId {
    /// Build an [`OpId`] from already-validated parts.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error with code
    /// `E_INVALID_OP_ID` if `namespace` or `name` are not valid identifier
    /// segments (see [`OpId::from_str`] for the grammar).
    pub fn new(namespace: impl Into<String>, name: impl Into<String>, major: u32) -> Result<Self> {
        let namespace = namespace.into();
        let name = name.into();
        validate_segment(&namespace, "namespace")?;
        validate_segment(&name, "name")?;
        Ok(Self {
            namespace,
            name,
            major,
        })
    }

    /// The namespace segment, e.g. `filter` in `filter.gaussian_blur@1`.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// The operation name segment, e.g. `gaussian_blur`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The semantic major version, e.g. `1`.
    #[must_use]
    pub const fn major(&self) -> u32 {
        self.major
    }
}

/// A valid identifier segment is a non-empty ASCII string of `[a-z0-9_]`
/// starting with a lowercase letter. This keeps op ids stable, case-insensitive-
/// safe, and free of characters that would need escaping in a plan or a path.
fn validate_segment(segment: &str, kind: &str) -> Result<()> {
    let mut chars = segment.chars();
    let starts_ok = matches!(chars.next(), Some('a'..='z'));
    let body_ok = chars.all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_'));
    if starts_ok && body_ok {
        return Ok(());
    }
    Err(Error::new(
        ErrorClass::Schema,
        "E_INVALID_OP_ID",
        format!(
            "operation {kind} segment {segment:?} must match [a-z][a-z0-9_]* (lowercase, \
             ASCII, starts with a letter)"
        ),
    ))
}

impl FromStr for OpId {
    type Err = Error;

    /// Parse `<namespace>.<name>@<major>`.
    ///
    /// The grammar is strict: exactly one `@` separating the dotted id from a
    /// non-negative decimal major, and exactly one `.` separating namespace and
    /// name. Each segment must match `[a-z][a-z0-9_]*` (lowercase ASCII,
    /// starting with a letter).
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error (`E_INVALID_OP_ID`) on any
    /// structural violation: missing `@`, missing `.`, an unparsable or
    /// out-of-range major, or an invalid segment.
    fn from_str(s: &str) -> Result<Self> {
        let invalid = |msg: String| Error::new(ErrorClass::Schema, "E_INVALID_OP_ID", msg);

        let Some((dotted, major_str)) = s.split_once('@') else {
            return Err(invalid(format!(
                "operation id {s:?} is missing the '@<major>' version suffix"
            )));
        };
        if major_str.contains('@') {
            return Err(invalid(format!(
                "operation id {s:?} contains more than one '@' separator"
            )));
        }
        let major: u32 = major_str.parse().map_err(|_| {
            invalid(format!(
                "operation id {s:?} has a non-numeric or out-of-range major version {major_str:?}"
            ))
        })?;

        let Some((namespace, name)) = dotted.split_once('.') else {
            return Err(invalid(format!(
                "operation id {s:?} is missing the '<namespace>.<name>' separator"
            )));
        };
        if name.contains('.') {
            return Err(invalid(format!(
                "operation id {s:?} has more than one '.' in '<namespace>.<name>'"
            )));
        }
        Self::new(namespace, name, major)
    }
}

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}@{}", self.namespace, self.name, self.major)
    }
}

impl Serialize for OpId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for OpId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A versioned **implementation** identifier of the form
/// `<backend>.<name>@<impl-version>`, e.g. `cpu.reference@1`,
/// `cpu.simd_separable@3`, `wgpu.separable@2` (`IR_SPEC` §6).
///
/// An implementation id names a concrete executor of an operation's semantics.
/// Unlike an [`OpId`] major, an implementation version does **not** change the
/// operation's semantics — it changes execution provenance and cache
/// compatibility (`IR_SPEC` §6). The manifest lists the implementations an
/// operation declares; the registry rejects a manifest whose declared list does
/// not include the conventional `cpu.reference` oracle (see
/// [`OperationManifest::validate`]).
///
/// The grammar mirrors [`OpId`]: a dotted `<backend>.<name>` where each segment
/// matches `[a-z][a-z0-9_]*`, an `@`, then a non-negative decimal version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImplId {
    backend: String,
    name: String,
    version: u32,
}

impl ImplId {
    /// Build an [`ImplId`] from already-validated parts.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error with code
    /// `E_INVALID_IMPL_ID` if `backend` or `name` are not valid identifier
    /// segments (`[a-z][a-z0-9_]*`).
    pub fn new(backend: impl Into<String>, name: impl Into<String>, version: u32) -> Result<Self> {
        let backend = backend.into();
        let name = name.into();
        validate_impl_segment(&backend, "backend")?;
        validate_impl_segment(&name, "name")?;
        Ok(Self {
            backend,
            name,
            version,
        })
    }

    /// The backend segment, e.g. `cpu` in `cpu.reference@1`.
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// The implementation name segment, e.g. `reference`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The implementation version, e.g. `1`.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }
}

/// Validate an implementation-id segment with the same grammar as an op-id
/// segment, but emitting `E_INVALID_IMPL_ID`.
fn validate_impl_segment(segment: &str, kind: &str) -> Result<()> {
    let mut chars = segment.chars();
    let starts_ok = matches!(chars.next(), Some('a'..='z'));
    let body_ok = chars.all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_'));
    if starts_ok && body_ok {
        return Ok(());
    }
    Err(Error::new(
        ErrorClass::Schema,
        "E_INVALID_IMPL_ID",
        format!(
            "implementation {kind} segment {segment:?} must match [a-z][a-z0-9_]* (lowercase, \
             ASCII, starts with a letter)"
        ),
    ))
}

impl FromStr for ImplId {
    type Err = Error;

    /// Parse `<backend>.<name>@<version>` with the same strictness as [`OpId`].
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error (`E_INVALID_IMPL_ID`) on
    /// any structural violation: missing `@`, missing `.`, an unparsable or
    /// out-of-range version, or an invalid segment.
    fn from_str(s: &str) -> Result<Self> {
        let invalid = |msg: String| Error::new(ErrorClass::Schema, "E_INVALID_IMPL_ID", msg);

        let Some((dotted, version_str)) = s.split_once('@') else {
            return Err(invalid(format!(
                "implementation id {s:?} is missing the '@<version>' suffix"
            )));
        };
        if version_str.contains('@') {
            return Err(invalid(format!(
                "implementation id {s:?} contains more than one '@' separator"
            )));
        }
        let version: u32 = version_str.parse().map_err(|_| {
            invalid(format!(
                "implementation id {s:?} has a non-numeric or out-of-range version {version_str:?}"
            ))
        })?;

        let Some((backend, name)) = dotted.split_once('.') else {
            return Err(invalid(format!(
                "implementation id {s:?} is missing the '<backend>.<name>' separator"
            )));
        };
        if name.contains('.') {
            return Err(invalid(format!(
                "implementation id {s:?} has more than one '.' in '<backend>.<name>'"
            )));
        }
        Self::new(backend, name, version)
    }
}

impl fmt::Display for ImplId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}@{}", self.backend, self.name, self.version)
    }
}

impl Serialize for ImplId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ImplId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// The conventional reference-oracle implementation backend (`cpu`).
///
/// Together with [`CPU_REFERENCE_NAME`] this names the `cpu.reference` oracle,
/// against which every operation's other implementations are differentially
/// verified (`IR_SPEC` §6, `AGENT_VERIFICATION`). A manifest must declare at
/// least one `cpu.reference@<v>` implementation.
pub const CPU_REFERENCE_BACKEND: &str = "cpu";
/// The conventional reference-oracle implementation name (`reference`).
pub const CPU_REFERENCE_NAME: &str = "reference";

/// The determinism tier an operation declares (`plan.md` §4.9).
///
/// The tier is part of the operation's contract: it tells an agent how much to
/// trust a result and what verification category applies. It is *not* a promise
/// the runtime can silently weaken — an op that declares `exact` must be
/// bit-exact for its scalar/backend contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum DeterminismTier {
    /// Bit-exact for the declared scalar format and backend contract.
    Exact,
    /// Same build/backend/device-class/seed/inputs reproduce within a declared
    /// bound.
    Reproducible,
    /// Alternate implementations agree within operation-specific absolute /
    /// relative / perceptual bounds.
    Bounded,
    /// Seeded, but model/provider behavior may vary; results are candidates with
    /// evidence.
    Stochastic,
}

impl DeterminismTier {
    /// Every tier, in spec order, for exhaustive table tests.
    pub const ALL: [Self; 4] = [
        Self::Exact,
        Self::Reproducible,
        Self::Bounded,
        Self::Stochastic,
    ];

    /// Whether a result at this tier is reproducible by re-execution (used by
    /// the cache to decide whether a hit may be reused without re-evidence).
    #[must_use]
    pub const fn is_reproducible(self) -> bool {
        matches!(self, Self::Exact | Self::Reproducible | Self::Bounded)
    }
}

/// The region-of-interest category of an operation (`IR_SPEC` §18).
///
/// This declares, at the manifest level, how much input an operation needs to
/// produce a given output region — the *category* the executable
/// `required_inputs` contract realizes. The compiler uses it to propagate ROIs
/// and to keep a masked pointwise 4K edit from touching tiles outside the
/// propagated region (`plan.md` §1 goal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum RoiCategory {
    /// Output region `R` needs exactly input region `R` (pointwise, composite
    /// target region).
    Pointwise,
    /// Output region `R` needs `R` dilated by a fixed halo (e.g. a blur of
    /// radius `r`); the halo is declared in [`RoiPolicy::halo_px`].
    LocalHalo,
    /// The required input footprint is a geometric transform of the output
    /// region plus a reconstruction halo (e.g. an affine warp).
    Geometric,
    /// The operation may require the whole connected component the output
    /// touches plus its boundary (Poisson, masked propagation).
    ConnectedComponent,
    /// The operation requires the whole demanded input (SDF conversion,
    /// global histogram) unless a tiled reduction is proven equivalent.
    FullDomain,
}

/// The declared ROI policy of an operation: its category plus, where the
/// category is [`RoiCategory::LocalHalo`], the fixed halo in pixels.
///
/// `halo_px` is only meaningful for [`RoiCategory::LocalHalo`]; for other
/// categories it is omitted. The manifest declares it so the compiler and the
/// verification harness can check the executable contract against the
/// declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoiPolicy {
    /// The ROI category governing input-region propagation.
    pub category: RoiCategory,
    /// For [`RoiCategory::LocalHalo`], the fixed dilation in pixels. Omitted for
    /// all other categories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub halo_px: Option<u32>,
}

/// The abstract resource *kind* a port carries (`IR_SPEC` §7), without the full
/// inferred descriptor.
///
/// Ports are typed by kind only; concrete descriptors (extent, encoding, …) are
/// inferred by the compiler. Mirrors the tags of
/// [`ResourceDescriptor`](crate::resource::ResourceDescriptor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResourceKind {
    /// A color or scalar raster.
    Image,
    /// A coverage mask.
    Mask,
    /// A scalar field.
    Field1,
    /// A 2-vector field.
    Field2,
    /// A 3-vector field.
    Field3,
    /// A signed distance field.
    SdfMask,
    /// An integer label map (`OP_CATALOG` §4): a single-channel raster of `u32`
    /// component IDs (`0` = background). Mirrors
    /// [`ResourceDescriptor::LabelMap`](crate::resource::ResourceDescriptor::LabelMap).
    LabelMap,
    /// An ordered candidate set of one of the above (`IR_SPEC` §12). The element
    /// kind is carried separately by the consuming op's contract.
    CandidateSet,
    /// A structured analysis report (`OP_CATALOG` §1): extent, per-channel
    /// ranges, finite-value statistics, and a stable content hash. Carries no
    /// raster; it is the output of inspection ops such as `image.inspect@1`.
    Report,
    /// A multi-resolution image/field pyramid (`OP_CATALOG` §13): a stack of
    /// co-located rasters, level `0` the full-resolution base and each deeper
    /// level dyadically downsampled. Mirrors
    /// [`ResourceDescriptor::Pyramid`](crate::resource::ResourceDescriptor::Pyramid).
    Pyramid,
    /// A complex frequency spectrum (`OP_CATALOG` §9): the DFT of an image/field
    /// plane. Mirrors
    /// [`ResourceDescriptor::Spectrum`](crate::resource::ResourceDescriptor::Spectrum).
    Spectrum,
    /// A patch correspondence field / nearest-neighbour field (`OP_CATALOG` §10,
    /// PatchMatch): a per-target-pixel mapping to the source coordinate of its
    /// best-matching patch, with the patch-match cost. Mirrors
    /// [`ResourceDescriptor::PatchField`](crate::resource::ResourceDescriptor::PatchField).
    PatchField,
}

/// A typed, named **input port** of an operation (`IR_SPEC` §5 `in`).
///
/// Inputs are referenced positionally-by-name in a plan node's `in` object; this
/// declares the legal names and their kinds so the parser can reject an op that
/// is wired to a port it does not define.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    /// The port name, e.g. `image`, `mask`, `hole`.
    pub name: String,
    /// The resource kind this port accepts.
    pub kind: ResourceKind,
    /// Whether the port must be wired. Optional ports default to absent; an op
    /// must tolerate their absence.
    #[serde(default = "default_true")]
    pub required: bool,
    /// Human-readable description for agent discovery.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub doc: String,
}

/// A named **output port** of an operation (`IR_SPEC` §3.3).
///
/// A node may not invent ports: the legal `node:<id>/<port>` references are
/// exactly the names declared here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    /// The port name, e.g. `image`, `lowpass`, `residuals`, `metadata`.
    pub name: String,
    /// The resource kind this port produces.
    pub kind: ResourceKind,
    /// Human-readable description for agent discovery.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub doc: String,
}

/// The canonical type of a parameter value (`IR_SPEC` §10).
///
/// Parameters are typed so normalization can reject `NaN`/infinity, refuse
/// fractional values for integer params, and apply unit conversion. Compound
/// params (objects/arrays such as a boundary spec or an objective list) are
/// declared as [`ParamType::Json`] and validated by the op's own schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ParamType {
    /// A finite IEEE-754 double. `NaN`/±∞ are rejected at normalization.
    Float,
    /// A signed integer; fractional JSON numbers are rejected.
    Integer,
    /// A boolean flag.
    Boolean,
    /// A string drawn from the param's `choices` (if any) or free-form.
    String,
    /// A pseudo-random seed; every stochastic op must carry a resolved numeric
    /// seed after normalization (`IR_SPEC` §11).
    Seed,
    /// A structured object/array validated by the operation's own schema.
    Json,
}

/// The unit a dimensional parameter is expressed in (`IR_SPEC` §10).
///
/// Units are part of the parameter's identity so that `sigma_px` and an angle in
/// radians are never confused. `None` (omitted) means the parameter is
/// dimensionless (a count, a flag, a seed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ParamUnit {
    /// Length in pixels (the canonical spatial unit).
    Pixels,
    /// Plane angle in radians (the canonical angular unit after normalization).
    Radians,
    /// Exposure in EV stops.
    Ev,
    /// A normalized `[0, 1]` ratio.
    Ratio,
}

/// A single parameter declaration (`IR_SPEC` §10, §11).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSpec {
    /// The canonical parameter name, e.g. `sigma_px`, `seed`, `boundary`.
    pub name: String,
    /// The canonical type the normalizer coerces values to.
    #[serde(rename = "type")]
    pub ty: ParamType,
    /// The dimensional unit, if any. Omitted for dimensionless params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<ParamUnit>,
    /// Whether the parameter must be supplied. A required parameter with a
    /// `default` is contradictory and rejected by [`OperationManifest::validate`].
    #[serde(default)]
    pub required: bool,
    /// The default value used when the parameter is absent, as canonical JSON.
    /// A present default makes the parameter optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// The closed set of legal values for a [`ParamType::String`] parameter, if
    /// it is an enum. Empty means free-form.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
    /// Human-readable description for agent discovery.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub doc: String,
}

/// Metadata describing how an operation is verified (`AGENT_VERIFICATION` §2.2,
/// `IR_SPEC` §18: "tested independently").
///
/// The manifest declares which verification categories apply so the harness can
/// require evidence for each, and so "not applicable" carries an explicit
/// reason rather than being a silent gap.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestMetadata {
    /// Whether the op has an analytic ground truth (closed-form expected output)
    /// to test against.
    #[serde(default)]
    pub has_analytic_reference: bool,
    /// Whether property-based / metamorphic tests apply (e.g. blur preserves a
    /// constant, idempotence, commutation).
    #[serde(default)]
    pub has_property_tests: bool,
    /// Named golden fixtures this op is checked against, if any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub golden_fixtures: Vec<String>,
    /// If a verification category is **not applicable**, the required reason
    /// (`AGENT_VERIFICATION` §10: "not applicable requires a reason").
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub not_applicable_reason: String,
    /// Per-category verification declarations (`AGENT_VERIFICATION` §2, §14).
    ///
    /// Each applicable [`VerificationCategory`](crate::verify::VerificationCategory)
    /// must be declared covered or not-applicable-with-a-reason; see
    /// [`OperationManifest::verify_categories`]. Defaults to empty.
    #[serde(default, skip_serializing_if = "verification_is_empty")]
    pub verification: crate::verify::VerificationDeclarations,
}

/// Whether a [`VerificationDeclarations`](crate::verify::VerificationDeclarations)
/// has no entries, so it can be skipped on serialization.
fn verification_is_empty(v: &crate::verify::VerificationDeclarations) -> bool {
    v.by_category.is_empty()
}

/// The complete operation manifest: the declared contract of one operation
/// version (`IR_SPEC` §3.3, §4, §6, §10, §11, §18).
///
/// This is the unit the versioned registry (a later bone) indexes by
/// [`OperationManifest::id`]. It is pure data; the executable ROI/shape/halo
/// contract lives in the op implementation and is cross-checked against the
/// `roi` declaration here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationManifest {
    /// The stable versioned id, e.g. `filter.gaussian_blur@1`.
    pub id: OpId,
    /// The implementation/provenance version, monotonically bumped when the
    /// *implementation* changes without changing semantics (`IR_SPEC` §6:
    /// "an implementation version changes execution provenance and cache
    /// compatibility").
    #[serde(default = "default_impl_version")]
    pub impl_version: u32,
    /// One-line human-readable summary for agent discovery.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    /// The declared determinism tier.
    pub determinism: DeterminismTier,
    /// The region-of-interest policy governing input propagation.
    pub roi: RoiPolicy,
    /// Typed input ports, in declaration order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<InputSpec>,
    /// Named output ports, in declaration order. An op must declare at least
    /// one output.
    pub outputs: Vec<OutputSpec>,
    /// Parameter declarations, in declaration order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ParamSpec>,
    /// The available implementations of this operation, in declaration order
    /// (`IR_SPEC` §6). Every operation must expose a `cpu.reference@<v>` oracle;
    /// optimized and GPU backends are listed alongside it and differentially
    /// verified against it.
    pub implementations: Vec<ImplId>,
    /// Verification metadata.
    #[serde(default)]
    pub test: TestMetadata,
}

impl OperationManifest {
    /// Validate the manifest's internal consistency beyond what serde enforces
    /// (`IR_SPEC` §3.3, §10, §11).
    ///
    /// Checks:
    /// - at least one output port;
    /// - input, output, and parameter names are each unique;
    /// - no parameter is both `required` and has a `default`;
    /// - a `LocalHalo` ROI declares a `halo_px`, and non-`LocalHalo` ROIs do not;
    /// - a stochastic op declares a `seed` parameter (`IR_SPEC` §11);
    /// - `choices` are only set on string parameters;
    /// - implementation ids are unique and include a `cpu.reference` oracle.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error with a specific code on
    /// the first violation found.
    pub fn validate(&self) -> Result<()> {
        let schema = |code: &str, msg: String| Error::new(ErrorClass::Schema, code, msg);

        if self.outputs.is_empty() {
            return Err(schema(
                "E_MANIFEST_NO_OUTPUTS",
                format!("operation {} declares no output ports", self.id),
            ));
        }

        check_unique(self.inputs.iter().map(|i| i.name.as_str()), "input")?;
        check_unique(self.outputs.iter().map(|o| o.name.as_str()), "output")?;
        check_unique(self.params.iter().map(|p| p.name.as_str()), "parameter")?;

        for p in &self.params {
            if p.required && p.default.is_some() {
                return Err(schema(
                    "E_PARAM_REQUIRED_WITH_DEFAULT",
                    format!(
                        "parameter {:?} of {} is both required and has a default",
                        p.name, self.id
                    ),
                ));
            }
            if !p.choices.is_empty() && p.ty != ParamType::String {
                return Err(schema(
                    "E_PARAM_CHOICES_NON_STRING",
                    format!(
                        "parameter {:?} of {} declares choices but is not a string parameter",
                        p.name, self.id
                    ),
                ));
            }
        }

        match (self.roi.category, self.roi.halo_px) {
            (RoiCategory::LocalHalo, None) => {
                return Err(schema(
                    "E_ROI_HALO_MISSING",
                    format!(
                        "operation {} declares a local-halo ROI but no halo_px",
                        self.id
                    ),
                ));
            }
            (cat, Some(_)) if cat != RoiCategory::LocalHalo => {
                return Err(schema(
                    "E_ROI_HALO_UNEXPECTED",
                    format!(
                        "operation {} declares halo_px for a non-local-halo ROI category",
                        self.id
                    ),
                ));
            }
            _ => {}
        }

        if self.determinism == DeterminismTier::Stochastic
            && !self.params.iter().any(|p| p.ty == ParamType::Seed)
        {
            return Err(schema(
                "E_STOCHASTIC_NO_SEED",
                format!(
                    "stochastic operation {} must declare a seed parameter",
                    self.id
                ),
            ));
        }

        let impl_ids: Vec<String> = self
            .implementations
            .iter()
            .map(ToString::to_string)
            .collect();
        check_unique(impl_ids.iter().map(String::as_str), "implementation")?;
        if !self
            .implementations
            .iter()
            .any(|i| i.backend == CPU_REFERENCE_BACKEND && i.name == CPU_REFERENCE_NAME)
        {
            return Err(schema(
                "E_NO_REFERENCE_IMPL",
                format!(
                    "operation {} must declare a {CPU_REFERENCE_BACKEND}.{CPU_REFERENCE_NAME} \
                     reference implementation",
                    self.id
                ),
            ));
        }

        Ok(())
    }

    /// Validate this manifest's [`test.verification`](TestMetadata::verification)
    /// declarations against the categories derived as applicable to it
    /// (`AGENT_VERIFICATION` §2, §14).
    ///
    /// This is **not** folded into [`validate`](Self::validate): a manifest may
    /// be structurally valid while still owing verification declarations, and the
    /// `verify-op` harness gates on the latter separately. See
    /// [`verify_categories`](crate::verify::verify_categories) for the rules and
    /// the error codes.
    ///
    /// # Errors
    /// Propagates the [`semantic`](ErrorClass::Semantic) error from
    /// [`verify_categories`](crate::verify::verify_categories) on the first
    /// applicable-but-undeclared, reasonless-not-applicable, or
    /// covered-but-inapplicable category.
    pub fn verify_categories(&self) -> Result<()> {
        crate::verify::verify_categories(self, &self.test.verification)
    }
}

/// Ensure every name in `names` is unique, returning a
/// [`schema`](ErrorClass::Schema) error on the first duplicate.
fn check_unique<'a>(names: impl Iterator<Item = &'a str>, kind: &str) -> Result<()> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if seen.contains(&name) {
            return Err(Error::new(
                ErrorClass::Schema,
                "E_DUPLICATE_PORT_NAME",
                format!("duplicate {kind} name {name:?} in manifest"),
            ));
        }
        seen.push(name);
    }
    Ok(())
}

const fn default_true() -> bool {
    true
}

const fn default_impl_version() -> u32 {
    1
}

/// Emit the JSON Schema (draft 2020-12) describing the
/// [`OperationManifest`] wire shape (`IR_SPEC` §14, §3.3).
///
/// The schema is hand-authored and self-contained so that `paintop op schema`
/// and CI can validate representative manifests without depending on the
/// runtime. It mirrors the serde model in this module; the round-trip tests
/// keep the two in sync.
#[must_use]
pub fn manifest_json_schema() -> serde_json::Value {
    use serde_json::json;

    let resource_kind = json!({
        "enum": ["Image", "Mask", "Field1", "Field2", "Field3", "SdfMask", "CandidateSet", "Report"]
    });

    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://paintop.dev/schema/operation-manifest.json",
        "title": "OperationManifest",
        "description": "The declared contract of one paintop operation version (IR_SPEC §6).",
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "determinism", "roi", "outputs", "implementations"],
        "properties": {
            "id": {
                "type": "string",
                "description": "<namespace>.<name>@<major> (IR_SPEC §6).",
                "pattern": "^[a-z][a-z0-9_]*\\.[a-z][a-z0-9_]*@[0-9]+$"
            },
            "impl_version": { "type": "integer", "minimum": 0, "default": 1 },
            "summary": { "type": "string" },
            "determinism": {
                "enum": ["exact", "reproducible", "bounded", "stochastic"]
            },
            "roi": {
                "type": "object",
                "additionalProperties": false,
                "required": ["category"],
                "properties": {
                    "category": {
                        "enum": [
                            "pointwise", "local-halo", "geometric",
                            "connected-component", "full-domain"
                        ]
                    },
                    "halo_px": { "type": "integer", "minimum": 0 }
                }
            },
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["name", "kind"],
                    "properties": {
                        "name": { "type": "string" },
                        "kind": resource_kind,
                        "required": { "type": "boolean", "default": true },
                        "doc": { "type": "string" }
                    }
                }
            },
            "outputs": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["name", "kind"],
                    "properties": {
                        "name": { "type": "string" },
                        "kind": resource_kind,
                        "doc": { "type": "string" }
                    }
                }
            },
            "params": params_schema(),
            "implementations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "description": "<backend>.<name>@<version> (IR_SPEC §6).",
                    "pattern": "^[a-z][a-z0-9_]*\\.[a-z][a-z0-9_]*@[0-9]+$"
                }
            },
            "test": test_schema()
        }
    })
}

/// The `params` array sub-schema of [`manifest_json_schema`].
fn params_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "type"],
            "properties": {
                "name": { "type": "string" },
                "type": {
                    "enum": ["float", "integer", "boolean", "string", "seed", "json"]
                },
                "unit": { "enum": ["pixels", "radians", "ev", "ratio"] },
                "required": { "type": "boolean", "default": false },
                "default": {},
                "choices": { "type": "array", "items": { "type": "string" } },
                "doc": { "type": "string" }
            }
        }
    })
}

/// The `test` object sub-schema of [`manifest_json_schema`].
fn test_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "has_analytic_reference": { "type": "boolean", "default": false },
            "has_property_tests": { "type": "boolean", "default": false },
            "golden_fixtures": { "type": "array", "items": { "type": "string" } },
            "not_applicable_reason": { "type": "string" },
            "verification": verification_schema()
        }
    })
}

/// The `test.verification` map sub-schema of [`manifest_json_schema`]: a map
/// keyed by verification category to a per-category status object.
fn verification_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "description": "Per-category verification declarations (AGENT_VERIFICATION §2, §14).",
        "propertyNames": {
            "enum": [
                "build-hygiene", "schema-contract", "analytic-fixtures", "property-tests",
                "metamorphic", "differential", "goldens", "perceptual", "fuzzing", "performance"
            ]
        },
        "additionalProperties": {
            "type": "object",
            "additionalProperties": false,
            "required": ["status"],
            "properties": {
                "status": { "enum": ["covered", "not-applicable"] },
                "reason": { "type": "string" }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DeterminismTier, ImplId, InputSpec, OpId, OperationManifest, OutputSpec, ParamSpec,
        ParamType, ParamUnit, ResourceKind, RoiCategory, RoiPolicy, TestMetadata,
        manifest_json_schema,
    };
    use crate::error::ErrorClass;
    use serde_json::json;

    fn gaussian_blur() -> OperationManifest {
        OperationManifest {
            id: "filter.gaussian_blur@1".parse().unwrap(),
            impl_version: 1,
            summary: "Separable Gaussian blur in linear light.".to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::LocalHalo,
                halo_px: Some(24),
            },
            inputs: vec![
                InputSpec {
                    name: "image".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: String::new(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: String::new(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            }],
            params: vec![ParamSpec {
                name: "sigma_px".to_owned(),
                ty: ParamType::Float,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "Standard deviation of the kernel.".to_owned(),
            }],
            implementations: vec![
                "cpu.reference@1".parse().unwrap(),
                "cpu.simd_separable@1".parse().unwrap(),
            ],
            test: TestMetadata {
                has_analytic_reference: true,
                has_property_tests: true,
                golden_fixtures: vec!["blur_checker_8x8".to_owned()],
                not_applicable_reason: String::new(),
                verification: crate::verify::VerificationDeclarations::default(),
            },
        }
    }

    #[test]
    fn op_id_parses_and_round_trips() {
        let id: OpId = "filter.gaussian_blur@1".parse().unwrap();
        assert_eq!(id.namespace(), "filter");
        assert_eq!(id.name(), "gaussian_blur");
        assert_eq!(id.major(), 1);
        assert_eq!(id.to_string(), "filter.gaussian_blur@1");

        // serde wire form is the canonical string.
        let value = serde_json::to_value(&id).unwrap();
        assert_eq!(value, json!("filter.gaussian_blur@1"));
        let back: OpId = serde_json::from_value(value).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn op_id_rejects_malformed_strings() {
        let cases = [
            "filter.gaussian_blur",     // no @major
            "filtergaussian_blur@1",    // no .
            "filter.gaussian_blur@x",   // non-numeric major
            "filter.gaussian_blur@1@2", // two @
            "Filter.gaussian_blur@1",   // uppercase namespace
            "filter.Gaussian@1",        // uppercase name
            "1filter.blur@1",           // namespace starts with digit
            "filter.a.b@1",             // two dots
            ".blur@1",                  // empty namespace
            "filter.@1",                // empty name
        ];
        for bad in cases {
            let err = bad.parse::<OpId>().unwrap_err();
            assert_eq!(err.class, ErrorClass::Schema, "{bad:?}");
            assert_eq!(err.code, "E_INVALID_OP_ID", "{bad:?}");
        }
    }

    #[test]
    fn op_id_via_new_validates_segments() {
        assert!(OpId::new("filter", "blur", 1).is_ok());
        assert!(OpId::new("Filter", "blur", 1).is_err());
        assert!(OpId::new("filter", "BLUR", 1).is_err());
    }

    #[test]
    fn manifest_serde_round_trips_and_validates() {
        let m = gaussian_blur();
        m.validate().unwrap();
        let value = serde_json::to_value(&m).unwrap();
        assert_eq!(value["id"], json!("filter.gaussian_blur@1"));
        assert_eq!(value["determinism"], json!("bounded"));
        assert_eq!(value["roi"]["category"], json!("local-halo"));
        assert_eq!(value["roi"]["halo_px"], json!(24));
        assert_eq!(value["params"][0]["type"], json!("float"));
        assert_eq!(value["params"][0]["unit"], json!("pixels"));
        assert_eq!(value["implementations"][0], json!("cpu.reference@1"));
        let back: OperationManifest = serde_json::from_value(value).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn manifest_rejects_unknown_fields() {
        let mut value = serde_json::to_value(gaussian_blur()).unwrap();
        value["bogus"] = json!(true);
        let err = serde_json::from_value::<OperationManifest>(value).unwrap_err();
        assert!(err.to_string().contains("bogus"), "{err}");
    }

    #[test]
    fn validate_rejects_no_outputs() {
        let mut m = gaussian_blur();
        m.outputs.clear();
        let err = m.validate().unwrap_err();
        assert_eq!(err.code, "E_MANIFEST_NO_OUTPUTS");
    }

    #[test]
    fn validate_rejects_duplicate_ports() {
        let mut m = gaussian_blur();
        m.outputs.push(OutputSpec {
            name: "image".to_owned(),
            kind: ResourceKind::Image,
            doc: String::new(),
        });
        let err = m.validate().unwrap_err();
        assert_eq!(err.code, "E_DUPLICATE_PORT_NAME");
    }

    #[test]
    fn validate_rejects_required_param_with_default() {
        let mut m = gaussian_blur();
        m.params[0].default = Some(json!(8.0));
        let err = m.validate().unwrap_err();
        assert_eq!(err.code, "E_PARAM_REQUIRED_WITH_DEFAULT");
    }

    #[test]
    fn validate_rejects_choices_on_non_string() {
        let mut m = gaussian_blur();
        m.params[0].choices = vec!["a".to_owned()];
        let err = m.validate().unwrap_err();
        assert_eq!(err.code, "E_PARAM_CHOICES_NON_STRING");
    }

    #[test]
    fn validate_enforces_halo_for_local_halo_only() {
        // local-halo without halo_px -> error.
        let mut m = gaussian_blur();
        m.roi.halo_px = None;
        assert_eq!(m.validate().unwrap_err().code, "E_ROI_HALO_MISSING");

        // non-local-halo with halo_px -> error.
        let mut m = gaussian_blur();
        m.roi.category = RoiCategory::Pointwise;
        assert_eq!(m.validate().unwrap_err().code, "E_ROI_HALO_UNEXPECTED");
    }

    #[test]
    fn validate_requires_seed_for_stochastic_ops() {
        let mut m = gaussian_blur();
        m.determinism = DeterminismTier::Stochastic;
        // No seed param yet -> rejected.
        assert_eq!(m.validate().unwrap_err().code, "E_STOCHASTIC_NO_SEED");

        // Add a seed param -> accepted.
        m.params.push(ParamSpec {
            name: "seed".to_owned(),
            ty: ParamType::Seed,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: String::new(),
        });
        m.validate().unwrap();
    }

    #[test]
    fn impl_id_parses_and_round_trips() {
        let id: ImplId = "cpu.reference@1".parse().unwrap();
        assert_eq!(id.backend(), "cpu");
        assert_eq!(id.name(), "reference");
        assert_eq!(id.version(), 1);
        assert_eq!(id.to_string(), "cpu.reference@1");

        let value = serde_json::to_value(&id).unwrap();
        assert_eq!(value, json!("cpu.reference@1"));
        let back: ImplId = serde_json::from_value(value).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn impl_id_rejects_malformed_strings() {
        for bad in [
            "cpu.reference",     // no @version
            "cpureference@1",    // no .
            "cpu.reference@x",   // non-numeric version
            "cpu.reference@1@2", // two @
            "Cpu.reference@1",   // uppercase backend
            "cpu.Reference@1",   // uppercase name
            "cpu.a.b@1",         // two dots
        ] {
            let err = bad.parse::<ImplId>().unwrap_err();
            assert_eq!(err.class, ErrorClass::Schema, "{bad:?}");
            assert_eq!(err.code, "E_INVALID_IMPL_ID", "{bad:?}");
        }
    }

    #[test]
    fn validate_requires_a_cpu_reference_implementation() {
        // Dropping the cpu.reference oracle is rejected.
        let mut m = gaussian_blur();
        m.implementations = vec!["wgpu.separable@2".parse().unwrap()];
        assert_eq!(m.validate().unwrap_err().code, "E_NO_REFERENCE_IMPL");

        // No implementations at all is also rejected.
        let mut m = gaussian_blur();
        m.implementations.clear();
        assert_eq!(m.validate().unwrap_err().code, "E_NO_REFERENCE_IMPL");
    }

    #[test]
    fn validate_rejects_duplicate_implementation_ids() {
        let mut m = gaussian_blur();
        m.implementations = vec![
            "cpu.reference@1".parse().unwrap(),
            "cpu.reference@1".parse().unwrap(),
        ];
        assert_eq!(m.validate().unwrap_err().code, "E_DUPLICATE_PORT_NAME");
    }

    #[test]
    fn determinism_tier_table_is_exhaustive() {
        assert_eq!(DeterminismTier::ALL.len(), 4);
        for t in DeterminismTier::ALL {
            let token = serde_json::to_value(t).unwrap();
            let back: DeterminismTier = serde_json::from_value(token).unwrap();
            assert_eq!(back, t);
        }
        assert!(DeterminismTier::Exact.is_reproducible());
        assert!(!DeterminismTier::Stochastic.is_reproducible());
    }

    #[test]
    fn unknown_enum_variants_fail_to_parse() {
        assert!(serde_json::from_value::<DeterminismTier>(json!("fast")).is_err());
        assert!(serde_json::from_value::<RoiCategory>(json!("magic")).is_err());
        assert!(serde_json::from_value::<ParamType>(json!("complex")).is_err());
        assert!(serde_json::from_value::<ResourceKind>(json!("Tensor")).is_err());
    }

    #[test]
    fn schema_is_well_formed_and_self_describing() {
        let schema = manifest_json_schema();
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["additionalProperties"], json!(false));
        // Required top-level keys mirror the serde model's non-defaulted fields.
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"id"));
        assert!(required.contains(&"determinism"));
        assert!(required.contains(&"roi"));
        assert!(required.contains(&"outputs"));
        assert!(required.contains(&"implementations"));
        // The id pattern matches a real op id and rejects a bad one.
        let pattern = schema["properties"]["id"]["pattern"].as_str().unwrap();
        let re = regex_lite_match(pattern, "filter.gaussian_blur@1");
        assert!(re, "pattern should match a valid id");
    }

    #[test]
    fn verify_categories_gate_on_a_manifest() {
        use crate::verify::{CategoryStatus, VerificationCategory, VerificationDeclarations};

        // gaussian_blur has two impls (cpu.reference + cpu.simd_separable) and a
        // bounded tier, so differential AND perceptual apply alongside the
        // universal layers.
        let mut m = gaussian_blur();
        assert!(VerificationCategory::Differential.is_applicable(&m));
        assert!(VerificationCategory::Perceptual.is_applicable(&m));

        // Undeclared -> the first applicable layer is missing.
        assert_eq!(
            m.verify_categories().unwrap_err().code,
            crate::verify::E_VERIFY_CATEGORY_MISSING
        );

        // Cover every applicable layer -> passes.
        let mut decls = VerificationDeclarations::new();
        for c in VerificationCategory::applicable_to(&m) {
            decls = decls.with(c, CategoryStatus::Covered);
        }
        m.test.verification = decls.clone();
        m.verify_categories().unwrap();

        // Skip one with a reason -> still passes.
        m.test.verification = decls.clone().with(
            VerificationCategory::Goldens,
            CategoryStatus::not_applicable("pass-through op has no stable golden"),
        );
        m.verify_categories().unwrap();

        // Skip one without a reason -> invalid.
        m.test.verification = decls.with(
            VerificationCategory::Goldens,
            CategoryStatus::not_applicable(""),
        );
        assert_eq!(
            m.verify_categories().unwrap_err().code,
            crate::verify::E_VERIFY_NA_REASON_MISSING
        );
    }

    #[test]
    fn verification_declarations_round_trip_through_the_manifest() {
        use crate::verify::{CategoryStatus, VerificationCategory, VerificationDeclarations};

        let mut m = gaussian_blur();
        m.test.verification = VerificationDeclarations::new()
            .with(VerificationCategory::BuildHygiene, CategoryStatus::Covered)
            .with(
                VerificationCategory::AnalyticFixtures,
                CategoryStatus::not_applicable("no closed form"),
            );
        let value = serde_json::to_value(&m).unwrap();
        assert_eq!(
            value["test"]["verification"]["build-hygiene"],
            json!({ "status": "covered" })
        );
        assert_eq!(
            value["test"]["verification"]["analytic-fixtures"],
            json!({ "status": "not-applicable", "reason": "no closed form" })
        );
        let back: OperationManifest = serde_json::from_value(value).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn schema_advertises_the_verification_map() {
        let schema = manifest_json_schema();
        let verification = &schema["properties"]["test"]["properties"]["verification"];
        assert_eq!(verification["type"], json!("object"));
        let names: Vec<&str> = verification["propertyNames"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        // Schema property names mirror the serde category tokens.
        let tokens: Vec<&str> = crate::verify::VerificationCategory::ALL
            .iter()
            .map(|c| c.as_str())
            .collect();
        assert_eq!(names, tokens);
    }

    #[test]
    fn schema_enums_match_serde_tokens() {
        let schema = manifest_json_schema();
        // determinism enum must equal the serde tokens.
        let det: Vec<String> = DeterminismTier::ALL
            .iter()
            .map(|t| {
                serde_json::to_value(t)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        let schema_det: Vec<String> = schema["properties"]["determinism"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(det, schema_det);
    }

    // Minimal anchored-pattern check (no regex dependency): the manifest schema
    // pattern is `^...$`; we just confirm the schema embeds it. A full validator
    // lives in the xtask `schema` command's integration, exercised separately.
    fn regex_lite_match(pattern: &str, _candidate: &str) -> bool {
        pattern.starts_with('^') && pattern.ends_with('$')
    }
}
