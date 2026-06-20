//! Typed resource descriptors and the enums they reference.
//!
//! Typed resources are the vocabulary every operation and the plan parser build
//! on. They let the compiler reject nonsense statically — hue-rotating a depth
//! field, sRGB-decoding a normal map, alpha-compositing an integer label map —
//! instead of producing a silently wrong raster (`plan.md` §4.5, §7.2).
//!
//! # Coordinate convention (`plan.md` §8.1)
//!
//! Every raster in paintop uses a single, fixed convention so that operations
//! never disagree off-by-half:
//!
//! - an integer `(x, y)` identifies a pixel *cell*;
//! - the **center** of cell `(x, y)` is the continuous point `(x + 0.5, y + 0.5)`;
//! - the image **origin** is the upper-left corner; positive `x` is rightward and
//!   positive `y` is downward;
//! - rectangles are **half-open**: a [`Rect`] covers `[x0, x1) × [y0, y1)`, so its
//!   width is `x1 - x0` and the pixel column `x1` is *excluded*.
//!
//! This is captured by the single [`CoordinateConvention::PixelCenterUpperLeft`]
//! variant; the enum exists so the convention is an explicit, serialized field
//! rather than an unstated assumption.
//!
//! ```
//! use paintop_ir::resource::{CoordinateConvention, Rect};
//!
//! // The only convention paintop supports today.
//! let conv = CoordinateConvention::PixelCenterUpperLeft;
//! assert_eq!(conv.pixel_center(3, 5), (3.5, 5.5));
//!
//! // Half-open rect: width is the exclusive span, the top-left pixel is (2, 4).
//! let roi = Rect::new(2, 4, 10, 9);
//! assert_eq!(roi.width(), 8);
//! assert_eq!(roi.height(), 5);
//! assert!(roi.contains(2, 4)); // inclusive lower bound
//! assert!(!roi.contains(10, 4)); // exclusive upper bound
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorClass, Result};

/// The pixel scalar storage type a resource is materialized in (`plan.md` §8.3).
///
/// Only the three formats reference semantics are defined against are
/// representable; `f16` GPU storage is deferred until equivalence tests exist
/// and is intentionally *not* a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ScalarType {
    /// Unsigned 8-bit, used for import/export and exact integer fixtures.
    U8,
    /// 32-bit float, the internal type for color, masks, fields, and reference
    /// semantics.
    F32,
    /// Unsigned 32-bit, used for label maps and IDs.
    U32,
}

impl ScalarType {
    /// Every scalar type, for exhaustive table tests.
    pub const ALL: [Self; 3] = [Self::U8, Self::F32, Self::U32];

    /// The size in bytes of a single scalar of this type.
    #[must_use]
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::F32 | Self::U32 => 4,
        }
    }
}

/// The supported color transfer encodings (`plan.md` §8.2).
///
/// This set is intentionally narrow: paintop implements a small color pipeline
/// correctly rather than half an ICC color-management system. `display-p3` and
/// arbitrary ICC profiles are *rejected*, not silently approximated — see
/// [`RequestedColorEncoding`], which can parse a request for them but converts
/// to a [`semantic`](ErrorClass::Semantic) error rather than a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ColorEncoding {
    /// Standard sRGB transfer function. An import/export encoding, never an
    /// arithmetic space.
    Srgb,
    /// sRGB primaries with a linear transfer function: the space color math
    /// happens in.
    LinearSrgb,
    /// Raw linear data with no color meaning, for material scalar maps
    /// (roughness, metallic, etc.). Not a color encoding to be decoded.
    RawLinear,
}

impl ColorEncoding {
    /// Every supported encoding, for exhaustive table tests.
    pub const ALL: [Self; 3] = [Self::Srgb, Self::LinearSrgb, Self::RawLinear];

    /// Whether this encoding represents light in linear light (so that
    /// premultiplication and neighborhood filters are physically meaningful).
    #[must_use]
    pub const fn is_linear_light(self) -> bool {
        matches!(self, Self::LinearSrgb | Self::RawLinear)
    }
}

/// A color encoding *as requested by a plan*, including encodings paintop can
/// name but does not implement.
///
/// This type exists so the parser can accept the token without crashing, then
/// surface a precise [`semantic`](ErrorClass::Semantic) error when the
/// requested encoding cannot be honored — rather than silently substituting a
/// supported encoding (`plan.md` §8.2: "reject unsupported ICC/profile
/// behavior").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum RequestedColorEncoding {
    /// Standard sRGB transfer function.
    Srgb,
    /// Linear-light sRGB primaries.
    LinearSrgb,
    /// Raw linear material data.
    RawLinear,
    /// Display P3. Representable so a request is *rejected*, not silently
    /// approximated.
    DisplayP3,
    /// An arbitrary embedded ICC profile. Representable so a request is
    /// *rejected*, not silently approximated.
    Icc,
}

impl RequestedColorEncoding {
    /// Resolve a requested encoding to a supported [`ColorEncoding`].
    ///
    /// # Errors
    /// Returns a [`semantic`](ErrorClass::Semantic) [`Error`] with code
    /// `E_UNSUPPORTED_COLOR_ENCODING` for `display-p3` and ICC requests: these
    /// are explicitly rejected for now and must never silently fall back to a
    /// supported encoding.
    pub fn resolve(self) -> Result<ColorEncoding> {
        match self {
            Self::Srgb => Ok(ColorEncoding::Srgb),
            Self::LinearSrgb => Ok(ColorEncoding::LinearSrgb),
            Self::RawLinear => Ok(ColorEncoding::RawLinear),
            Self::DisplayP3 => Err(Self::reject("display-p3")),
            Self::Icc => Err(Self::reject("icc")),
        }
    }

    fn reject(actual: &str) -> Error {
        use crate::error::ErrorContext;
        Error::new(
            ErrorClass::Semantic,
            "E_UNSUPPORTED_COLOR_ENCODING",
            format!("color encoding `{actual}` is not supported"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(actual)
                .with_expected("srgb | linear-srgb | raw-linear"),
        )
    }
}

/// Whether a color resource's alpha is associated with its color channels
/// (`plan.md` §8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AlphaRepresentation {
    /// Color channels are pre-scaled by alpha (the internal compositing form).
    Premultiplied,
    /// Color channels are independent of alpha (unassociated); converted to
    /// premultiplied explicitly at boundaries.
    Straight,
}

/// The reference-light range a color resource lives in (`IR_SPEC` §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ColorRange {
    /// Values are bounded to the display range `[0, 1]`.
    DisplayReferred,
    /// Values may exceed `1.0` (high dynamic range / scene light).
    SceneReferred,
}

/// The channel layout of a color image (`IR_SPEC` §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ChannelLayout {
    /// Single grayscale channel.
    Gray,
    /// Grayscale plus alpha.
    GrayA,
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue, alpha.
    Rgba,
}

impl ChannelLayout {
    /// The number of channels in this layout.
    #[must_use]
    pub const fn channel_count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::GrayA => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Whether this layout carries an alpha channel.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayA | Self::Rgba)
    }
}

/// The semantic role of an image-like resource (`plan.md` §4.5).
///
/// The role drives which operations are legal: hue-rotating a `depth` resource
/// or sRGB-decoding a `normal` resource is rejected at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum SemanticRole {
    /// Ordinary displayable color.
    Color,
    /// Linear scalar material data (roughness, metallic, etc.).
    Material,
    /// Surface normals.
    Normal,
    /// Scene depth.
    Depth,
    /// Per-pixel confidence / weight.
    Confidence,
    /// Generic distance field values.
    Distance,
    /// Generic 2-vector flow / orientation / displacement.
    Flow,
}

/// The meaning of a [`MaskDescriptor`]'s values (`IR_SPEC` §7.2, §9.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum MaskMeaning {
    /// Fractional coverage in `[0, 1]`, *not* boolean truth.
    Coverage,
    /// A hard selection thresholded to `{0, 1}`.
    Selection,
}

/// The reference frame a vector field's components are expressed in
/// (`IR_SPEC` §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum VectorSpace {
    /// Tangent space relative to the surface.
    Tangent,
    /// World space.
    World,
    /// Object/local space.
    Object,
}

/// How a vector field's components are packed (`IR_SPEC` §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum VectorEncoding {
    /// Components stored directly as signed values in `[-1, 1]`.
    SignedVector,
    /// Components stored as unsigned `[0, 1]` and remapped to `[-1, 1]` on
    /// decode (the classic normal-map packing).
    UnsignedNormalized,
}

/// Whether a vector field is constrained to unit length (`IR_SPEC` §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum VectorNormalization {
    /// Vectors are unit length.
    Unit,
    /// Vectors are unconstrained.
    None,
}

/// The physical units a signed distance field is measured in (`IR_SPEC` §7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum SdfUnits {
    /// Distance in physical pixel units.
    Pixels,
}

/// The sign convention of a signed distance field (`IR_SPEC` §7.4).
///
/// The sign convention must never be implicit, so this field has no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum SdfSign {
    /// Distance is negative inside the region, positive outside.
    NegativeInside,
    /// Distance is positive inside the region, negative outside.
    PositiveInside,
}

/// The boundary condition a neighborhood operation samples outside the valid
/// region with (`plan.md` §8.4).
///
/// The boundary mode is part of the operation hash and conformance tests, so it
/// must always be declared explicitly — there is no implicit default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum BoundaryMode {
    /// Out-of-bounds samples take a fixed constant value.
    Constant {
        /// The constant scalar returned for out-of-bounds samples.
        value: f32,
    },
    /// Out-of-bounds samples take the nearest edge value.
    Clamp,
    /// Out-of-bounds coordinates reflect across the edge.
    Mirror,
    /// Out-of-bounds coordinates wrap around (toroidal).
    Wrap,
    /// Out-of-bounds samples are treated as fully transparent (zero coverage).
    Transparent,
    /// Out-of-bounds samples are undefined; the output is only valid where every
    /// input sample is in-bounds.
    ValidOnly,
}

/// The valid-range policy a resource's scalar values must satisfy
/// (`plan.md` §8.3).
///
/// Clamping is never implicit: a resource declares its range policy, and any
/// clamp is an explicit node or policy decision, so silent clamps cannot hide
/// bugs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ValidRange {
    /// Values must lie in `[min, max]`; going outside is clamped or rejected by
    /// explicit policy.
    Bounded {
        /// Inclusive lower bound.
        min: f32,
        /// Inclusive upper bound.
        max: f32,
    },
    /// Values are unconstrained in magnitude but must be finite.
    Unbounded,
    /// Values form a finite, norm-constrained vector (e.g. unit normals).
    NormalizedVector,
}

/// The fixed coordinate convention every raster uses (`plan.md` §8.1).
///
/// See the [module documentation](self) for the full convention. The enum has a
/// single variant today; it exists so the convention is an explicit, serialized
/// field rather than an unstated assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum CoordinateConvention {
    /// Pixel `(x, y)` has center `(x + 0.5, y + 0.5)`, origin upper-left,
    /// positive `x` right and positive `y` down, with half-open rectangles.
    PixelCenterUpperLeft,
}

impl CoordinateConvention {
    /// The continuous center of integer pixel cell `(x, y)`.
    ///
    /// ```
    /// use paintop_ir::resource::CoordinateConvention;
    /// let c = CoordinateConvention::PixelCenterUpperLeft;
    /// assert_eq!(c.pixel_center(0, 0), (0.5, 0.5));
    /// ```
    #[must_use]
    pub fn pixel_center(self, x: u32, y: u32) -> (f64, f64) {
        match self {
            Self::PixelCenterUpperLeft => (f64::from(x) + 0.5, f64::from(y) + 0.5),
        }
    }
}

/// A 2D pixel extent with checked pixel/byte arithmetic (`IR_SPEC` §7.1).
///
/// `width`/`height` are pixel counts. The product helpers are *checked*: a
/// resource that would overflow `u64` pixels or bytes is rejected with a
/// [`policy`](ErrorClass::Policy) error rather than wrapping silently into a
/// tiny allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extent {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Extent {
    /// Construct an extent from a width and height in pixels.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// The total pixel count `width * height`, computed in `u64`.
    ///
    /// # Errors
    /// Returns a [`policy`](ErrorClass::Policy) [`Error`] with code
    /// `E_EXTENT_OVERFLOW` if `width * height` does not fit in `u64`.
    pub fn pixel_count(self) -> Result<u64> {
        u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .ok_or_else(|| self.overflow("pixel count width*height overflows u64"))
    }

    /// The total byte count `width * height * channels * scalar_bytes`, computed
    /// in `u64`.
    ///
    /// # Errors
    /// Returns a [`policy`](ErrorClass::Policy) [`Error`] with code
    /// `E_EXTENT_OVERFLOW` if any factor of the product overflows `u64`.
    pub fn byte_count(self, channels: u32, scalar: ScalarType) -> Result<u64> {
        self.pixel_count()?
            .checked_mul(u64::from(channels))
            .and_then(|n| n.checked_mul(u64::from(scalar.byte_size())))
            .ok_or_else(|| self.overflow("byte count width*height*channels*bytes overflows u64"))
    }

    fn overflow(self, message: &str) -> Error {
        use crate::error::ErrorContext;
        Error::new(ErrorClass::Policy, "E_EXTENT_OVERFLOW", message).with_context(
            ErrorContext::default().with_actual(format!("{}x{}", self.width, self.height)),
        )
    }
}

/// A half-open axis-aligned rectangle in pixel space: `[x0, x1) × [y0, y1)`
/// (`plan.md` §8.1).
///
/// The upper bounds are *exclusive*: `Rect::new(0, 0, w, h)` covers exactly the
/// `w × h` pixels with top-left corner `(0, 0)`. See the
/// [module documentation](self) for the full coordinate convention.
///
/// ```
/// use paintop_ir::resource::Rect;
/// let r = Rect::new(2, 3, 5, 7);
/// assert_eq!(r.width(), 3);
/// assert_eq!(r.height(), 4);
/// assert!(r.is_valid());
/// assert!(!Rect::new(5, 0, 2, 1).is_valid()); // x1 < x0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rect {
    /// Inclusive left edge.
    pub x0: i64,
    /// Inclusive top edge.
    pub y0: i64,
    /// Exclusive right edge.
    pub x1: i64,
    /// Exclusive bottom edge.
    pub y1: i64,
}

impl Rect {
    /// Construct a half-open rect `[x0, x1) × [y0, y1)`.
    #[must_use]
    pub const fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Whether the rect is well-formed (non-negative width and height).
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.x1 >= self.x0 && self.y1 >= self.y0
    }

    /// Whether the rect is empty (zero width or height). Empty rects are valid
    /// but contain no pixels.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    /// The width `x1 - x0`, saturating to `0` for an ill-formed rect.
    #[must_use]
    pub const fn width(self) -> i64 {
        let w = self.x1 - self.x0;
        if w > 0 { w } else { 0 }
    }

    /// The height `y1 - y0`, saturating to `0` for an ill-formed rect.
    #[must_use]
    pub const fn height(self) -> i64 {
        let h = self.y1 - self.y0;
        if h > 0 { h } else { 0 }
    }

    /// The pixel area `width * height`.
    #[must_use]
    pub const fn area(self) -> i64 {
        self.width() * self.height()
    }

    /// Whether pixel cell `(x, y)` lies in this half-open rect: `x0 <= x < x1`
    /// and `y0 <= y < y1`.
    #[must_use]
    pub const fn contains(self, x: i64, y: i64) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    /// The intersection of two half-open rects.
    ///
    /// The result is the largest rect contained in both. If they do not overlap
    /// the result [`is_empty`](Rect::is_empty) (the bounds may be ill-formed,
    /// but the area is zero).
    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        Self {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }

    /// The smallest half-open rect containing both inputs (their bounding box).
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

/// A typed color/scalar raster descriptor (`IR_SPEC` §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDescriptor {
    /// Pixel extent.
    pub extent: Extent,
    /// Channel layout.
    pub layout: ChannelLayout,
    /// Scalar storage type.
    pub scalar: ScalarType,
    /// Color transfer encoding.
    pub color: ColorEncoding,
    /// Reference-light range.
    pub range: ColorRange,
    /// Alpha representation.
    pub alpha: AlphaRepresentation,
    /// Coordinate convention.
    pub coordinates: CoordinateConvention,
    /// Semantic role.
    pub semantic: SemanticRole,
}

/// A coverage mask descriptor (`IR_SPEC` §7.2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskDescriptor {
    /// Pixel extent.
    pub extent: Extent,
    /// Scalar storage type.
    pub scalar: ScalarType,
    /// Valid-range policy (coverage is normally `bounded [0, 1]`).
    pub range: ValidRange,
    /// What the mask values mean.
    pub meaning: MaskMeaning,
    /// Coordinate convention.
    pub coordinates: CoordinateConvention,
}

/// The dimensionality of a [`FieldDescriptor`]'s vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum FieldArity {
    /// Scalar field (`Field1`): depth, roughness, confidence, distance.
    Field1,
    /// 2-vector field (`Field2`): flow, orientation, displacement.
    Field2,
    /// 3-vector field (`Field3`): normals or 3-vector features.
    Field3,
}

impl FieldArity {
    /// The number of components per sample.
    #[must_use]
    pub const fn component_count(self) -> u32 {
        match self {
            Self::Field1 => 1,
            Self::Field2 => 2,
            Self::Field3 => 3,
        }
    }
}

/// A scalar or vector field descriptor (`IR_SPEC` §7.3).
///
/// `space`, `normalization`, and `encoding` are only meaningful for vector
/// fields; they are optional so a `Field1` scalar field can omit them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldDescriptor {
    /// Field dimensionality.
    pub arity: FieldArity,
    /// Pixel extent.
    pub extent: Extent,
    /// Scalar storage type.
    pub scalar: ScalarType,
    /// Semantic role.
    pub semantic: SemanticRole,
    /// Coordinate convention.
    pub coordinates: CoordinateConvention,
    /// Reference frame of the vector components (vector fields only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<VectorSpace>,
    /// Norm constraint on the vectors (vector fields only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization: Option<VectorNormalization>,
    /// Component packing (vector fields only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<VectorEncoding>,
}

/// A signed distance field descriptor (`IR_SPEC` §7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdfDescriptor {
    /// Pixel extent.
    pub extent: Extent,
    /// Scalar storage type (`f32` for reference semantics).
    pub scalar: ScalarType,
    /// Physical distance units.
    pub units: SdfUnits,
    /// Sign convention; never implicit.
    pub sign: SdfSign,
    /// Coordinate convention.
    pub coordinates: CoordinateConvention,
}

/// A typed resource descriptor: the tagged union of every resource kind this
/// bone defines (`IR_SPEC` §7).
///
/// The `kind` tag matches the spec's JSON (`"Image"`, `"Mask"`, `"Field1"`,
/// `"Field2"`, `"Field3"`, `"SdfMask"`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum ResourceDescriptor {
    /// A color or scalar raster.
    Image(ImageDescriptor),
    /// A coverage mask.
    Mask(MaskDescriptor),
    /// A scalar field.
    Field1(FieldDescriptor),
    /// A 2-vector field.
    Field2(FieldDescriptor),
    /// A 3-vector field.
    Field3(FieldDescriptor),
    /// A signed distance field.
    SdfMask(SdfDescriptor),
}

impl ResourceDescriptor {
    /// The pixel extent of any resource descriptor.
    #[must_use]
    pub const fn extent(&self) -> Extent {
        match self {
            Self::Image(d) => d.extent,
            Self::Mask(d) => d.extent,
            Self::Field1(d) | Self::Field2(d) | Self::Field3(d) => d.extent,
            Self::SdfMask(d) => d.extent,
        }
    }

    /// The abstract [`ResourceKind`](crate::manifest::ResourceKind) this concrete
    /// descriptor realizes.
    ///
    /// A descriptor is one inferred instance of a kind: a port declares only the
    /// kind, and the checker compares the kind of the descriptor flowing into a
    /// port against the port's declared kind. `CandidateSet` has no single
    /// descriptor variant, so it is never produced here.
    #[must_use]
    pub const fn kind(&self) -> crate::manifest::ResourceKind {
        use crate::manifest::ResourceKind;
        match self {
            Self::Image(_) => ResourceKind::Image,
            Self::Mask(_) => ResourceKind::Mask,
            Self::Field1(_) => ResourceKind::Field1,
            Self::Field2(_) => ResourceKind::Field2,
            Self::Field3(_) => ResourceKind::Field3,
            Self::SdfMask(_) => ResourceKind::SdfMask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AlphaRepresentation, BoundaryMode, ChannelLayout, ColorEncoding, ColorRange,
        CoordinateConvention, Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, Rect,
        RequestedColorEncoding, ResourceDescriptor, ScalarType, SemanticRole, ValidRange,
    };
    use crate::error::ErrorClass;
    use serde_json::json;

    fn sample_image() -> ImageDescriptor {
        ImageDescriptor {
            extent: Extent::new(2048, 2048),
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        }
    }

    #[test]
    fn image_descriptor_serde_round_trips() {
        let d = sample_image();
        let value = serde_json::to_value(d).unwrap();
        // Kebab-case wire tokens per the spec example.
        assert_eq!(value["color"], json!("linear-srgb"));
        assert_eq!(value["coordinates"], json!("pixel-center-upper-left"));
        let back: ImageDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn resource_descriptor_tag_matches_spec_kinds() {
        let d = ResourceDescriptor::Image(sample_image());
        let value = serde_json::to_value(d).unwrap();
        assert_eq!(value["kind"], json!("Image"));
        let back: ResourceDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(back, d);
        assert_eq!(back.extent(), Extent::new(2048, 2048));
    }

    #[test]
    fn deny_unknown_fields_on_descriptor() {
        let mut value = serde_json::to_value(sample_image()).unwrap();
        value["bogus"] = json!(true);
        let err = serde_json::from_value::<ImageDescriptor>(value).unwrap_err();
        assert!(err.to_string().contains("bogus"), "{err}");
    }

    #[test]
    fn unknown_enum_variant_is_a_parse_error() {
        // An encoding token that is not in the supported set must fail to parse,
        // rather than silently selecting a neighbor.
        let err = serde_json::from_value::<ColorEncoding>(json!("cmyk")).unwrap_err();
        assert!(err.to_string().contains("unknown variant"), "{err}");

        let err = serde_json::from_value::<ScalarType>(json!("f64")).unwrap_err();
        assert!(err.to_string().contains("unknown variant"), "{err}");

        let err = serde_json::from_value::<MaskMeaning>(json!("truth")).unwrap_err();
        assert!(err.to_string().contains("unknown variant"), "{err}");
    }

    #[test]
    fn display_p3_and_icc_are_representable_as_rejected() {
        // Parseable as a *request*...
        let p3: RequestedColorEncoding = serde_json::from_value(json!("display-p3")).unwrap();
        assert_eq!(p3, RequestedColorEncoding::DisplayP3);
        let icc: RequestedColorEncoding = serde_json::from_value(json!("icc")).unwrap();
        assert_eq!(icc, RequestedColorEncoding::Icc);

        // ...but resolving yields a semantic error, never a silent fallback.
        for enc in [
            RequestedColorEncoding::DisplayP3,
            RequestedColorEncoding::Icc,
        ] {
            let err = enc.resolve().unwrap_err();
            assert_eq!(err.class, ErrorClass::Semantic);
            assert_eq!(err.code, "E_UNSUPPORTED_COLOR_ENCODING");
        }

        // The supported encodings round-trip through resolve.
        assert_eq!(
            RequestedColorEncoding::Srgb.resolve().unwrap(),
            ColorEncoding::Srgb
        );
        assert_eq!(
            RequestedColorEncoding::LinearSrgb.resolve().unwrap(),
            ColorEncoding::LinearSrgb
        );
        assert_eq!(
            RequestedColorEncoding::RawLinear.resolve().unwrap(),
            ColorEncoding::RawLinear
        );
    }

    #[test]
    fn boundary_mode_serde_is_tagged() {
        let value = serde_json::to_value(BoundaryMode::Constant { value: 0.25 }).unwrap();
        assert_eq!(value, json!({"mode": "constant", "value": 0.25}));
        assert_eq!(
            serde_json::to_value(BoundaryMode::Mirror).unwrap(),
            json!({"mode": "mirror"})
        );
        let back: BoundaryMode = serde_json::from_value(json!({"mode": "valid-only"})).unwrap();
        assert_eq!(back, BoundaryMode::ValidOnly);

        let err =
            serde_json::from_value::<BoundaryMode>(json!({"mode": "reflect101"})).unwrap_err();
        assert!(err.to_string().contains("unknown variant"), "{err}");
    }

    #[test]
    fn valid_range_serde_is_tagged() {
        let value = serde_json::to_value(ValidRange::Bounded { min: 0.0, max: 1.0 }).unwrap();
        assert_eq!(value, json!({"policy": "bounded", "min": 0.0, "max": 1.0}));
        assert_eq!(
            serde_json::to_value(ValidRange::NormalizedVector).unwrap(),
            json!({"policy": "normalized-vector"})
        );
        let back: ValidRange = serde_json::from_value(json!({"policy": "unbounded"})).unwrap();
        assert_eq!(back, ValidRange::Unbounded);
    }

    #[test]
    fn mask_descriptor_round_trips() {
        let d = MaskDescriptor {
            extent: Extent::new(64, 64),
            scalar: ScalarType::F32,
            range: ValidRange::Bounded { min: 0.0, max: 1.0 },
            meaning: MaskMeaning::Coverage,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        let back: MaskDescriptor =
            serde_json::from_value(serde_json::to_value(d).unwrap()).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn extent_pixel_and_byte_counts() {
        let e = Extent::new(2048, 2048);
        assert_eq!(e.pixel_count().unwrap(), 2048 * 2048);
        // 4 channels * 4 bytes per f32.
        assert_eq!(
            e.byte_count(4, ScalarType::F32).unwrap(),
            2048 * 2048 * 4 * 4
        );
    }

    #[test]
    fn extent_rejects_overflowing_pixel_product() {
        // u32::MAX * u32::MAX fits in u64, so pixel_count succeeds...
        let huge = Extent::new(u32::MAX, u32::MAX);
        let pixels = huge.pixel_count().unwrap();
        assert_eq!(pixels, u64::from(u32::MAX) * u64::from(u32::MAX));

        // ...but multiplying by 4 channels * 4 bytes overflows u64 and is rejected.
        let err = huge.byte_count(4, ScalarType::F32).unwrap_err();
        assert_eq!(err.class, ErrorClass::Policy);
        assert_eq!(err.code, "E_EXTENT_OVERFLOW");
    }

    #[test]
    fn extent_byte_count_overflow_is_checked_not_wrapping() {
        // A width*height that already saturates near u64::MAX: ensure we never
        // wrap into a small value. Pick factors whose channel product overflows.
        let e = Extent::new(u32::MAX, 5);
        // pixels fit
        let _ = e.pixel_count().unwrap();
        // bytes with a large channel count overflow.
        let err = e.byte_count(u32::MAX, ScalarType::U32).unwrap_err();
        assert_eq!(err.code, "E_EXTENT_OVERFLOW");
    }

    #[test]
    fn half_open_rect_math() {
        let r = Rect::new(2, 4, 10, 9);
        assert_eq!(r.width(), 8);
        assert_eq!(r.height(), 5);
        assert_eq!(r.area(), 40);
        assert!(r.is_valid());
        assert!(!r.is_empty());

        // Inclusive lower bound, exclusive upper bound.
        assert!(r.contains(2, 4));
        assert!(r.contains(9, 8));
        assert!(!r.contains(10, 4)); // x1 excluded
        assert!(!r.contains(2, 9)); // y1 excluded
        assert!(!r.contains(1, 4)); // below x0
    }

    #[test]
    fn rect_empty_and_invalid() {
        let empty = Rect::new(3, 3, 3, 8); // zero width
        assert!(empty.is_valid());
        assert!(empty.is_empty());
        assert_eq!(empty.area(), 0);

        let invalid = Rect::new(10, 0, 2, 4); // x1 < x0
        assert!(!invalid.is_valid());
        assert!(invalid.is_empty());
        assert_eq!(invalid.width(), 0); // saturates to 0
        assert_eq!(invalid.area(), 0);
    }

    #[test]
    fn rect_intersect_and_union() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 20, 20);
        let i = a.intersect(b);
        assert_eq!(i, Rect::new(5, 5, 10, 10));
        assert_eq!(i.area(), 25);

        let u = a.union(b);
        assert_eq!(u, Rect::new(0, 0, 20, 20));

        // Disjoint rects intersect to an empty region.
        let disjoint = Rect::new(0, 0, 4, 4).intersect(Rect::new(8, 8, 12, 12));
        assert!(disjoint.is_empty());
        assert_eq!(disjoint.area(), 0);
    }

    #[test]
    fn rect_serde_round_trips_with_deny_unknown_fields() {
        let r = Rect::new(1, 2, 3, 4);
        let back: Rect = serde_json::from_value(serde_json::to_value(r).unwrap()).unwrap();
        assert_eq!(back, r);
        let err = serde_json::from_value::<Rect>(json!({
            "x0": 0, "y0": 0, "x1": 1, "y1": 1, "z": 2
        }))
        .unwrap_err();
        assert!(err.to_string().contains('z'), "{err}");
    }

    #[test]
    fn coordinate_convention_pixel_center() {
        let c = CoordinateConvention::PixelCenterUpperLeft;
        assert_eq!(c.pixel_center(0, 0), (0.5, 0.5));
        assert_eq!(c.pixel_center(3, 5), (3.5, 5.5));
    }

    #[test]
    fn scalar_and_layout_sizes() {
        assert_eq!(ScalarType::U8.byte_size(), 1);
        assert_eq!(ScalarType::F32.byte_size(), 4);
        assert_eq!(ScalarType::U32.byte_size(), 4);
        assert_eq!(ChannelLayout::Rgba.channel_count(), 4);
        assert!(ChannelLayout::Rgba.has_alpha());
        assert!(!ChannelLayout::Rgb.has_alpha());
    }
}
