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
    ///
    /// An *empty* operand contributes nothing: the union of an empty rect with a
    /// non-empty one is the non-empty one (an empty rect covers no pixels, so its
    /// degenerate bounds must not inflate the result). The union of two empty
    /// rects is the canonical [`Rect::EMPTY`].
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        match (self.is_empty(), other.is_empty()) {
            (true, true) => Self::EMPTY,
            (true, false) => other,
            (false, true) => self,
            (false, false) => Self {
                x0: self.x0.min(other.x0),
                y0: self.y0.min(other.y0),
                x1: self.x1.max(other.x1),
                y1: self.y1.max(other.y1),
            },
        }
    }

    /// The canonical empty rect `[0, 0) × [0, 0)`: valid, zero-area, the identity
    /// element of [`union`](Rect::union).
    pub const EMPTY: Self = Self {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    /// A rect covering exactly the `extent.width × extent.height` pixels of an
    /// image with its origin at `(0, 0)` — the *full* domain of a resource.
    #[must_use]
    pub const fn from_extent(extent: Extent) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: extent.width as i64,
            y1: extent.height as i64,
        }
    }

    /// This rect translated by `(dx, dy)` (a pure geometric shift; bounds are not
    /// clamped). Used to map an output region back to its co-located input region
    /// under a crop/pad/composite offset.
    #[must_use]
    pub const fn translate(self, dx: i64, dy: i64) -> Self {
        Self {
            x0: self.x0 + dx,
            y0: self.y0 + dy,
            x1: self.x1 + dx,
            y1: self.y1 + dy,
        }
    }

    /// This rect dilated outward by a uniform `halo` of pixels on every side: a
    /// neighbourhood operation's output region `R` reads input region `R` grown by
    /// the kernel halo (`IR_SPEC` §18, [`RoiCategory::LocalHalo`]).
    ///
    /// An *empty* rect dilates to [`Rect::EMPTY`] (no output pixels demand any
    /// input). A non-empty rect grows by `halo` on each edge; the grown bounds are
    /// clamped to `i64` and never overflow for realistic halos.
    ///
    /// [`RoiCategory::LocalHalo`]: crate::manifest::RoiCategory::LocalHalo
    #[must_use]
    pub fn dilate(self, halo: u32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let h = i64::from(halo);
        Self {
            x0: self.x0.saturating_sub(h),
            y0: self.y0.saturating_sub(h),
            x1: self.x1.saturating_add(h),
            y1: self.y1.saturating_add(h),
        }
    }

    /// This rect dilated by independent positive `dx`/`dy` halos on the horizontal
    /// and vertical axes (an anisotropic kernel). Empty in, [`Rect::EMPTY`] out.
    #[must_use]
    pub fn dilate_xy(self, dx: u32, dy: u32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let (hx, hy) = (i64::from(dx), i64::from(dy));
        Self {
            x0: self.x0.saturating_sub(hx),
            y0: self.y0.saturating_sub(hy),
            x1: self.x1.saturating_add(hx),
            y1: self.y1.saturating_add(hy),
        }
    }

    /// This rect clipped (intersected) to the full domain of `extent`, the
    /// canonical "clamp a demanded region to the producer's actual pixels"
    /// operation. The result [`is_empty`](Rect::is_empty) when the rect lies
    /// wholly outside the extent.
    #[must_use]
    pub fn clamp_to_extent(self, extent: Extent) -> Self {
        self.intersect(Self::from_extent(extent))
    }

    /// Whether this rect fully contains `other`: every pixel of `other` is a pixel
    /// of `self`. An empty `other` is contained by any rect (it has no pixels to
    /// fall outside). Used by the ROI suite to prove a demanded source region
    /// *covers* every contributor.
    #[must_use]
    pub const fn contains_rect(self, other: Self) -> bool {
        if other.is_empty() {
            return true;
        }
        !self.is_empty()
            && self.x0 <= other.x0
            && self.y0 <= other.y0
            && self.x1 >= other.x1
            && self.y1 >= other.y1
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

/// An integer label-map descriptor (`OP_CATALOG` §4): a single-channel raster of
/// `u32` component IDs.
///
/// A label map is the output of connected-component labeling: every pixel carries
/// the integer ID of the component it belongs to, with `0` reserved for the
/// background (no component). Unlike a [`MaskDescriptor`] its samples are exact
/// integers, not fractional coverage, so its scalar type is always
/// [`ScalarType::U32`]; the run-time value stores each ID losslessly (the `f32`
/// sample buffer holds the raw `u32` bit pattern via
/// [`f32::from_bits`]/[`f32::to_bits`], so IDs above `2^24` survive a round trip
/// — `AGENT_VERIFICATION` §2.3 "integer attachment/encoding loss").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelMapDescriptor {
    /// Pixel extent.
    pub extent: Extent,
    /// Scalar storage type; always [`ScalarType::U32`] for an integer label map.
    pub scalar: ScalarType,
    /// Coordinate convention.
    pub coordinates: CoordinateConvention,
}

/// The type-level descriptor of a [`Report`] resource (`OP_CATALOG` §1).
///
/// A report carries no raster, so its descriptor records only the shape of the
/// resource it summarizes: the source extent and channel count. The statistical
/// payload (ranges, finite stats, content hash) is the resource *value*
/// ([`Report`]), produced at execution rather than inferred from metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportDescriptor {
    /// The pixel extent of the summarized resource.
    pub extent: Extent,
    /// The channel count of the summarized resource.
    pub channels: u32,
}

/// Per-channel statistics carried by a [`Report`] (`OP_CATALOG` §1).
///
/// All extrema and the sum are computed over the channel's **finite** samples
/// only; `NaN`/`±∞` samples are excluded from the range and counted in
/// [`nonfinite`](ChannelStats::nonfinite) instead, so a single bad sample cannot
/// poison the reported range while still being flagged.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelStats {
    /// The minimum finite sample, or `None` if the channel has no finite sample.
    pub min: Option<f32>,
    /// The maximum finite sample, or `None` if the channel has no finite sample.
    pub max: Option<f32>,
    /// The sum of the finite samples (used to derive the mean).
    pub sum: f64,
    /// The sum of the squares of the finite samples (used to derive the
    /// population variance, `OP_CATALOG` §12 `analyze.statistics`). Defaults to
    /// `0.0` so a report serialized before this field existed still parses.
    #[serde(default)]
    pub sum_sq: f64,
    /// The number of finite samples.
    pub finite: u64,
    /// The number of non-finite (`NaN`/`±∞`) samples.
    pub nonfinite: u64,
}

impl ChannelStats {
    /// The arithmetic mean of the finite samples, or `None` if there are none.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.finite == 0 {
            None
        } else {
            Some(self.sum / self.finite_f64())
        }
    }

    /// The population variance of the finite samples — the mean of the squares
    /// minus the square of the mean — or `None` if there are no finite samples.
    ///
    /// The result is clamped at zero so floating-point cancellation on a
    /// (near-)constant channel can never report a spuriously negative variance.
    #[must_use]
    pub fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        let mean_sq = self.sum_sq / self.finite_f64();
        Some(mean.mul_add(-mean, mean_sq).max(0.0))
    }

    /// The finite-sample count as an `f64` denominator.
    #[must_use]
    const fn finite_f64(&self) -> f64 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "finite is a sample count; f64 mantissa covers realistic image sizes"
        )]
        let denom = self.finite as f64;
        denom
    }

    /// Whether every sample in the channel is finite.
    #[must_use]
    pub const fn all_finite(&self) -> bool {
        self.nonfinite == 0
    }
}

/// The whole-image difference summary a difference op (`analyze.diff@1`)
/// attaches to its [`Report`] (`OP_CATALOG` §12, `AGENT_VERIFICATION` §2.6).
///
/// The metrics are reductions over the per-pixel, per-channel **absolute**
/// difference field `|after − before|` computed in the op's declared comparison
/// space. Identical inputs produce all-zero metrics and an empty
/// [`changed_bounds`](DiffMetrics::changed_bounds); a known injected delta
/// produces exact metrics. Errors above [`threshold`](DiffMetrics::threshold)
/// define the *changed* pixels whose count and bounding box are reported.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffMetrics {
    /// The maximum absolute difference across every channel and pixel.
    pub max_abs_error: f64,
    /// The mean absolute difference across every channel and pixel.
    pub mean_abs_error: f64,
    /// The root-mean-square difference across every channel and pixel.
    pub rms_error: f64,
    /// The error threshold a pixel must exceed (strictly) to count as *changed*.
    pub threshold: f64,
    /// The number of *changed* pixels: pixels with at least one channel whose
    /// absolute difference exceeds [`threshold`](Self::threshold).
    pub changed_count: u64,
    /// The tight bounding box of the changed pixels in pixel space, or `None`
    /// when no pixel changed (an empty diff).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_bounds: Option<Rect>,
}

/// A per-channel histogram of an image or field over an explicit value domain
/// (`OP_CATALOG` §12 `analyze.histogram@1`).
///
/// The domain `[domain_min, domain_max)` is split into `bins` equal-width bins.
/// A finite sample `v` falls in bin `floor((v - domain_min) / width)`, clamped so
/// the upper domain edge `domain_max` lands in the last bin (a half-open domain
/// with an inclusive top edge). Samples strictly below `domain_min` are counted
/// in [`below`](Self::below); samples strictly above `domain_max` in
/// [`above`](Self::above); non-finite samples in [`nonfinite`](Self::nonfinite).
/// The per-channel `counts` are row-major: channel `c`'s bin `b` is
/// `counts[c * bins + b]`. Every reduction runs in a fixed row-major order, so
/// the histogram is a deterministic function of the input and the domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistogramData {
    /// The number of channels the histogram covers.
    pub channels: u32,
    /// The number of bins per channel.
    pub bins: u32,
    /// The inclusive lower edge of the value domain.
    pub domain_min: f64,
    /// The inclusive upper edge of the value domain (`domain_max > domain_min`).
    pub domain_max: f64,
    /// The per-channel bin counts, row-major (`counts[c * bins + b]`).
    pub counts: Vec<u64>,
    /// The per-channel count of finite samples strictly below `domain_min`.
    pub below: Vec<u64>,
    /// The per-channel count of finite samples strictly above `domain_max`.
    pub above: Vec<u64>,
    /// The per-channel count of non-finite (`NaN`/`±∞`) samples.
    pub nonfinite: Vec<u64>,
}

/// The severity of an assertion's verdict (`IR_SPEC` §13).
///
/// Severity governs how an assertion's failure affects the run, and is *explicit*
/// rather than implied by the assertion kind: an `error` fails the run (exit
/// class 6), a `warning` retains the output but marks the evidence, and a
/// `metric` never fails the run — it only records the measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AssertionSeverity {
    /// A failure fails the run (maps to exit class 6).
    Error,
    /// A failure retains the output but marks the evidence as suspect.
    Warning,
    /// A measurement only: a failure never fails the run.
    Metric,
}

impl AssertionSeverity {
    /// Whether a *failure* at this severity should fail the run.
    ///
    /// Only [`Error`](Self::Error) fails the run; [`Warning`](Self::Warning) and
    /// [`Metric`](Self::Metric) record the verdict without failing it.
    #[must_use]
    pub const fn fails_run(self) -> bool {
        matches!(self, Self::Error)
    }
}

/// The structured verdict an assertion operation attaches to its [`Report`]
/// (`IR_SPEC` §13, `OP_CATALOG` §12, `AGENT_VERIFICATION` §5.3).
///
/// An assertion is an ordinary typed node that produces a [`Report`]; this block
/// records whether the predicate *held* ([`passed`](Self::passed)), the explicit
/// [`severity`](Self::severity) that decides whether a failure fails the run, and
/// the failure evidence the bundle surfaces: the worst offending pixel, a capped
/// list of offending pixel locations, and the assertion-specific metrics
/// (`max_abs_delta_outside` / `changed_pixels_outside` for
/// `assert.no_change_outside_mask`, `nonfinite_count` for `assert.finite`).
///
/// The fields are deterministic functions of the inputs (every reduction runs in
/// a fixed row-major order), so the verdict is reproducible across runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionOutcome {
    /// The canonical id of the assertion that produced this verdict, e.g.
    /// `assert.no_change_outside_mask@1`.
    pub assertion: String,
    /// Whether the asserted predicate held.
    pub passed: bool,
    /// The explicit severity governing whether a failure fails the run.
    pub severity: AssertionSeverity,
    /// The maximum absolute delta found *outside* the allowed region
    /// (`assert.no_change_outside_mask` only), in the comparison space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_delta_outside: Option<f64>,
    /// The number of pixels that changed *outside* the allowed region
    /// (`assert.no_change_outside_mask` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_pixels_outside: Option<u64>,
    /// The number of non-finite (`NaN`/`±∞`) samples found (`assert.finite`
    /// only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonfinite_count: Option<u64>,
    /// The `[x, y]` location of the single worst offending pixel, or `None` when
    /// the assertion passed (no offending pixel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_pixel: Option<[i64; 2]>,
    /// A capped, row-major-ordered list of offending pixel `[x, y]` locations
    /// (leaking pixels, or non-finite-sample pixels), for the failure artifact.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<[i64; 2]>,
    /// The number of samples/pixels that violated the predicate
    /// (`assert.range` out-of-range count, `assert.alpha_valid` invalid-pixel
    /// count). Absent for assertions that report their count in a more specific
    /// field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub violations: Option<u64>,
    /// The single worst offending *value* (`assert.range`: the in-image sample
    /// furthest outside `[min, max]`; `assert.alpha_valid`: the largest
    /// premultiplied-constraint excess `|C| - α`). Absent when the assertion
    /// reports no scalar worst value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_value: Option<f64>,
    /// The tight bounding box of the *actual* changed region
    /// (`assert.changed_bounds` only), or `None` when nothing changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_bounds: Option<Rect>,
    /// The *expected* bounding box the changed region must stay within
    /// (`assert.changed_bounds` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_bounds: Option<Rect>,
}

impl AssertionOutcome {
    /// A bare verdict with no failure evidence: the assertion id, whether it
    /// passed, and its severity. Every assertion-specific evidence field is
    /// `None`/empty; the caller fills in the ones its assertion populates.
    #[must_use]
    pub fn new(assertion: impl Into<String>, passed: bool, severity: AssertionSeverity) -> Self {
        Self {
            assertion: assertion.into(),
            passed,
            severity,
            max_abs_delta_outside: None,
            changed_pixels_outside: None,
            nonfinite_count: None,
            worst_pixel: None,
            locations: Vec::new(),
            violations: None,
            worst_value: None,
            changed_bounds: None,
            expected_bounds: None,
        }
    }
}

/// A structured analysis report (`OP_CATALOG` §1): the resource *value* an
/// inspection op (`image.inspect@1`) produces.
///
/// A report records the summarized resource's extent, its per-channel finite
/// statistics, and a stable [`content_hash`](Report::content_hash) of the source
/// samples. The content hash is computed by the canonical hashing module
/// ([`crate::hash`], domain [`Content`](crate::hash::HashDomain::Content)) over a
/// canonical encoding of the samples, so it is deterministic and matches the
/// hashing module's identity for the same bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    /// The pixel extent of the summarized resource.
    pub extent: Extent,
    /// The channel count of the summarized resource.
    pub channels: u32,
    /// Per-channel finite statistics, in channel order.
    pub channel_stats: Vec<ChannelStats>,
    /// Whether every sample of every channel is finite.
    pub all_finite: bool,
    /// The algorithm-prefixed content hash of the summarized samples
    /// (`blake3:<hex>`), as produced by [`crate::hash`].
    pub content_hash: String,
    /// The whole-image difference summary, present only on the report a
    /// difference op (`analyze.diff@1`) produces (`OP_CATALOG` §12,
    /// `AGENT_VERIFICATION` §2.6). Absent (`None`) for an ordinary inspection
    /// report, so it is omitted on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<DiffMetrics>,
    /// The structured assertion verdict, present only on the report an assertion
    /// op (`assert.no_change_outside_mask@1`, `assert.finite@1`) produces
    /// (`IR_SPEC` §13, `AGENT_VERIFICATION` §5.3). Absent (`None`) for an ordinary
    /// inspection or diff report, so it is omitted on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assertion: Option<AssertionOutcome>,
    /// The per-channel histogram, present only on the report a histogram op
    /// (`analyze.histogram@1`) produces (`OP_CATALOG` §12). Absent (`None`) for
    /// every other report, so it is omitted on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub histogram: Option<HistogramData>,
    /// The connected-component summary, present only on the report a labeling op
    /// (`mask.connected_components@1`) produces (`OP_CATALOG` §4). Absent (`None`)
    /// for every other report, so it is omitted on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub components: Option<ComponentsData>,
    /// The per-band frequency-energy summary, present only on the report a
    /// frequency-analysis op (`analyze.frequency_energy@1`) produces
    /// (`OP_CATALOG` §13). Absent (`None`) for every other report, so it is
    /// omitted on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_energy: Option<FrequencyEnergyData>,
    /// The iterative-solver convergence summary, present only on the report an
    /// explicit-step solver (`field.reaction_diffusion@1`) produces
    /// (`OP_CATALOG` §11). Carries the step count, the stability bound, and an
    /// energy/residual history so a consumer can audit convergence. Absent
    /// (`None`) for every other report, so it is omitted on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver: Option<SolverData>,
}

/// The convergence summary an explicit-step iterative solver
/// (`field.reaction_diffusion@1`) attaches to its [`Report`] (`OP_CATALOG` §11;
/// `plan.md` §1444 — every solver exposes its convergence metrics).
///
/// The solver runs a fixed number of explicit forward-Euler steps; this record
/// makes its run auditable: the number of `steps` taken, the explicit-stability
/// bound `stability_limit` and the actual `stability_number` the chosen time step
/// realized (a value `> 1` would diverge — the op rejects such a request), the
/// per-step total-variation `residual_history` (the L2 change between successive
/// states, length `steps`), and the final `total_energy` (the sum of squared
/// samples across the solved fields). Every reduction runs in a fixed row-major
/// order, so the summary is a deterministic function of the inputs, the seed, and
/// the step count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverData {
    /// The solver kind, e.g. `"gray-scott"`.
    pub kind: String,
    /// The number of explicit steps taken.
    pub steps: u32,
    /// The dimensionless explicit-stability number the chosen step realized
    /// (`= max_diffusion * dt * neighbours`); must be `<= stability_limit`.
    pub stability_number: f64,
    /// The explicit-stability bound the scheme stays stable below (`1.0` for the
    /// normalized 5-point Laplacian scheme).
    pub stability_limit: f64,
    /// Whether the run stayed within the stability bound for its whole duration.
    pub stable: bool,
    /// The per-step L2 change `||state_{n+1} - state_n||_2` (length `steps`); a
    /// decaying history evidences convergence toward a steady state.
    pub residual_history: Vec<f64>,
    /// The final total energy (sum of squared samples across the solved fields).
    pub total_energy: f64,
    /// The number of iterations a *convergence-driven* iterative solver (e.g. the
    /// Poisson / screened-Poisson Gauss-Seidel solver) actually ran before it
    /// stopped — distinct from a fixed-`steps` explicit scheme. Absent (`None`,
    /// omitted on serialization) for a fixed-step solver, where `steps` already
    /// carries the iteration count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u32>,
    /// The reason a convergence-driven solver halted: it reached the residual
    /// tolerance, hit the iteration cap, or detected no further progress. Absent
    /// (`None`) for a fixed-step solver.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<SolverStopReason>,
    /// Whether a convergence-driven solver reached its residual `tolerance` before
    /// the iteration cap. Absent (`None`) for a fixed-step solver.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converged: Option<bool>,
    /// The residual `tolerance` a convergence-driven solver targeted (the
    /// relative-residual threshold below which it declares convergence). Absent
    /// (`None`) for a fixed-step solver.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    /// The final relative residual a convergence-driven solver reached
    /// (`||r_final||_2 / ||r_0||_2`, or the absolute final residual when the
    /// initial residual is zero). Absent (`None`) for a fixed-step solver.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_residual: Option<f64>,
}

/// Why a convergence-driven iterative solver halted (`OP_CATALOG` §12;
/// `plan.md` §1444 — every solver exposes a stop reason).
///
/// A Gauss-Seidel / SOR Poisson solver iterates until one of three terminal
/// conditions is met; the report records exactly which one so a consumer can
/// distinguish a converged result from a capped or stalled one. The enum is
/// `#[non_exhaustive]` so a future stop condition is an additive variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum SolverStopReason {
    /// The relative residual fell to or below the requested tolerance.
    Converged,
    /// The iteration cap was reached before the tolerance was met.
    MaxIterations,
    /// The residual stopped decreasing (no measurable progress between
    /// successive sweeps) before the tolerance was met.
    Stalled,
}

/// The per-band frequency-energy summary an analysis op
/// (`analyze.frequency_energy@1`) attaches to its [`Report`] (`OP_CATALOG` §13).
///
/// A multi-resolution decomposition (a Gaussian pyramid's blurred levels or a
/// Laplacian pyramid's band-pass residuals) carries the image's energy split by
/// spatial frequency. `band_energy[l]` is the **sum of squared samples** of band
/// `l` (finest first), optionally restricted to the pixels an analysis mask
/// admits; `band_pixels[l]` is the admitted sample count for that band, so a
/// consumer can form a mean-energy density. The `total_energy` is the sum across
/// every band. Every reduction runs in a fixed (finest-first, row-major) order
/// with a stable pairwise sum, so the summary is a deterministic function of the
/// input and the band policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencyEnergyData {
    /// The decomposition kind the bands came from: `"gaussian"` (per-level
    /// low-pass energy) or `"laplacian"` (per-band band-pass energy).
    pub decomposition: String,
    /// The number of bands (= pyramid levels).
    pub bands: u32,
    /// The sum of squared samples of each band, finest first (length `bands`).
    pub band_energy: Vec<f64>,
    /// The admitted sample count of each band, finest first (length `bands`).
    pub band_pixels: Vec<u64>,
    /// The total energy summed across every band.
    pub total_energy: f64,
}

/// The connected-component summary a labeling op (`mask.connected_components@1`)
/// attaches to its [`Report`] (`OP_CATALOG` §4).
///
/// `count` is the number of foreground components found (each labeled with an ID
/// in `1..=count`; `0` is the background). `areas` carries the pixel area of each
/// component in **label order** (`areas[i]` is the area of component `i + 1`), so
/// a consumer can apply a size policy without re-scanning the label map. The
/// labeling is deterministic: components are numbered in raster-scan order of
/// their first (top-most, then left-most) pixel, the stable policy this op
/// guarantees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentsData {
    /// The pixel connectivity used to define adjacency (`4` or `8`).
    pub connectivity: u8,
    /// The number of foreground components (labels `1..=count`).
    pub count: u32,
    /// The pixel area of each component in label order (`areas[i]` is label
    /// `i + 1`), length `count`.
    pub areas: Vec<u64>,
}

impl Report {
    /// The type-level [`ReportDescriptor`] this report realizes.
    #[must_use]
    pub const fn descriptor(&self) -> ReportDescriptor {
        ReportDescriptor {
            extent: self.extent,
            channels: self.channels,
        }
    }
}

/// The per-level downsample ratio of a [`PyramidDescriptor`] (`OP_CATALOG` §13).
///
/// Classical image pyramids halve the resolution at every level; the factor is
/// an explicit, serialized field (no implicit `2`) so the convention is part of
/// the resource type and the op hash, never an unstated assumption. Only the
/// dyadic (factor-2) pyramid is defined today; the enum exists so a future
/// factor is an additive variant rather than a breaking reinterpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum DownsampleFactor {
    /// Each level halves both axes: level `l` has extent `f(base / 2^l)` for the
    /// descriptor's [`PyramidPhase`] rounding rule `f`.
    Half,
}

impl DownsampleFactor {
    /// The integer divisor applied to each axis per level (2 for [`Half`]).
    ///
    /// [`Half`]: Self::Half
    #[must_use]
    pub const fn divisor(self) -> u32 {
        match self {
            Self::Half => 2,
        }
    }
}

/// The rounding rule mapping a parent level's extent to its child's
/// (`OP_CATALOG` §13).
///
/// When an axis has an odd length, halving it is not exact: the child axis is
/// either rounded down (`Floor`, the classical Gaussian-pyramid convention) or
/// up (`Ceil`). The phase is part of the resource type so a reconstruction op
/// (`frequency.recombine`) knows exactly which child extent each level uses and
/// can invert the split bit-for-bit. There is no implicit default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum PyramidPhase {
    /// A child axis is `floor(parent / divisor)`, clamped to a minimum of 1
    /// (the classical convention: `(n + 1) / 2` is *not* used; an odd `n`
    /// loses its last column/row).
    Floor,
    /// A child axis is `ceil(parent / divisor)`, clamped to a minimum of 1
    /// (an odd `n` keeps a half-populated last column/row).
    Ceil,
}

impl PyramidPhase {
    /// The child axis length for a parent axis `n` under this phase and
    /// `divisor`, never below 1 (a degenerate `n == 1` stays `1`).
    #[must_use]
    pub const fn child_axis(self, n: u32, divisor: u32) -> u32 {
        // `divisor` is at least 2 for every `DownsampleFactor`, so the division
        // is well defined; the `max(1)` keeps a 1-px axis from collapsing to 0.
        let child = match self {
            Self::Floor => n / divisor,
            Self::Ceil => n.div_ceil(divisor),
        };
        if child < 1 { 1 } else { child }
    }
}

/// The maximum number of levels a single pyramid may declare.
///
/// This bounds the derived per-level table and any allocation a pyramid drives.
/// A dyadic pyramid over a `u32`-extent base reaches a 1×1 apex in at most 32
/// halvings per axis, so this ceiling is never restrictive for a real image
/// while still rejecting a pathological request.
pub const MAX_PYRAMID_LEVELS: u32 = 64;

/// A multi-resolution image/field **pyramid** descriptor (`OP_CATALOG` §13).
///
/// A pyramid is a stack of `levels` co-located rasters, level `0` the full-
/// resolution base and each deeper level downsampled by [`factor`] under the
/// [`phase`] rounding rule. The descriptor stores only the *base* extent and the
/// level count; every level's extent is **derived** by
/// [`level_extent`](Self::level_extent) from the base, factor, and phase, so the
/// level chain is a deterministic function of the descriptor and cannot drift
/// out of sync with a separately stored table. This also keeps the descriptor
/// `Copy` (no per-level `Vec`), matching every other [`ResourceDescriptor`].
///
/// A Gaussian pyramid (`frequency.gaussian_pyramid`) stores a blurred-and-
/// downsampled image at each level; a Laplacian pyramid
/// (`frequency.laplacian_split`) stores the band-pass residual at each level
/// plus the coarsest low-pass — both share this one descriptor shape, the
/// content distinction is the producing op's, not the type's.
///
/// [`factor`]: Self::factor
/// [`phase`]: Self::phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PyramidDescriptor {
    /// The level-0 (full-resolution) pixel extent. Every deeper level's extent
    /// is derived from this base.
    pub base_extent: Extent,
    /// The number of levels, level `0` the base. Always at least `1`; a
    /// one-level pyramid is just the base image wrapped as a pyramid.
    pub levels: u32,
    /// The number of interleaved channels per sample at every level (the base
    /// resource's channel count; constant across levels).
    pub channels: u32,
    /// The scalar storage type of every level.
    pub scalar: ScalarType,
    /// The per-level downsample ratio; never implicit.
    pub factor: DownsampleFactor,
    /// The odd-size rounding rule mapping a parent extent to its child; never
    /// implicit.
    pub phase: PyramidPhase,
    /// Coordinate convention (shared by every level).
    pub coordinates: CoordinateConvention,
}

impl PyramidDescriptor {
    /// The pixel extent of level `level` (level `0` is the base), each axis
    /// downsampled `level` times by the factor under the phase rounding rule.
    ///
    /// Returns `None` if `level >= self.levels` (the level is out of range).
    /// Every in-range level has a well-defined, non-degenerate (≥ 1×1) extent.
    #[must_use]
    pub fn level_extent(&self, level: u32) -> Option<Extent> {
        if level >= self.levels {
            return None;
        }
        let divisor = self.factor.divisor();
        let mut width = self.base_extent.width.max(1);
        let mut height = self.base_extent.height.max(1);
        for _ in 0..level {
            width = self.phase.child_axis(width, divisor);
            height = self.phase.child_axis(height, divisor);
        }
        Some(Extent::new(width, height))
    }

    /// The total sample count across every level: `Σ_l width_l·height_l·channels`.
    ///
    /// This is the length a flat, level-major concatenated pyramid sample buffer
    /// must have; the executor stores a pyramid value as the finest-first
    /// concatenation of its levels.
    ///
    /// # Errors
    /// Returns a [`policy`](ErrorClass::Policy) [`Error`] (`E_PYRAMID_OVERFLOW`)
    /// if the accumulated count overflows `u64`.
    pub fn total_samples(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for level in 0..self.levels {
            // Every level in `0..levels` is in range, so `level_extent` is `Some`.
            let extent = self
                .level_extent(level)
                .ok_or_else(|| self.overflow("pyramid level index out of range"))?;
            let level_samples = extent
                .pixel_count()?
                .checked_mul(u64::from(self.channels))
                .ok_or_else(|| self.overflow("pyramid level sample count overflows u64"))?;
            total = total
                .checked_add(level_samples)
                .ok_or_else(|| self.overflow("pyramid total sample count overflows u64"))?;
        }
        Ok(total)
    }

    /// Validate the level chain: a positive, bounded level count, and a base
    /// extent that does not collapse before the declared depth.
    ///
    /// A pyramid is well-formed when `1 <= levels <= MAX_PYRAMID_LEVELS` and the
    /// derived per-level extents never overflow. A 1-level pyramid is always
    /// valid (it is the base wrapped as a pyramid); odd base sizes are valid
    /// under either phase (the phase fixes the rounding). The total sample count
    /// is checked so a pathological descriptor cannot drive an overflowing
    /// allocation.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) [`Error`] (`E_PYRAMID_LEVELS`)
    /// for a zero or over-deep level count, or a [`policy`](ErrorClass::Policy)
    /// [`Error`] (`E_PYRAMID_OVERFLOW`) if the level chain overflows.
    pub fn validate(&self) -> Result<()> {
        if self.levels == 0 {
            return Err(Error::new(
                ErrorClass::Schema,
                "E_PYRAMID_LEVELS",
                "a pyramid must declare at least one level".to_owned(),
            ));
        }
        if self.levels > MAX_PYRAMID_LEVELS {
            return Err(Error::new(
                ErrorClass::Schema,
                "E_PYRAMID_LEVELS",
                format!(
                    "a pyramid may declare at most {MAX_PYRAMID_LEVELS} levels, got {}",
                    self.levels
                ),
            )
            .with_context(
                crate::error::ErrorContext::default()
                    .with_actual(self.levels.to_string())
                    .with_expected(format!("1..={MAX_PYRAMID_LEVELS}")),
            ));
        }
        // Drives the per-level extent derivation and overflow checks.
        let _ = self.total_samples()?;
        Ok(())
    }

    fn overflow(&self, message: &str) -> Error {
        Error::new(ErrorClass::Policy, "E_PYRAMID_OVERFLOW", message.to_owned()).with_context(
            crate::error::ErrorContext::default().with_actual(format!(
                "{}x{} x{} levels",
                self.base_extent.width, self.base_extent.height, self.levels
            )),
        )
    }
}

/// A typed **complex frequency-spectrum** descriptor (`OP_CATALOG` §9).
///
/// A spectrum is the discrete Fourier transform of a single-channel-or-multi-
/// channel image/field plane: the same `W×H` grid as its spatial source, but
/// every sample is a *complex* number `(re, im)` rather than a real scalar. The
/// transform is the centered, **non-normalized forward DFT** — the `1/(W·H)`
/// scale lives entirely on the inverse — so a forward-then-inverse round trip
/// reconstructs the source up to floating-point rounding.
///
/// The descriptor stores the spatial extent and the *logical* (complex) channel
/// count; the run-time value packs each complex sample as two interleaved `f32`
/// (real then imaginary), so a spectrum value's flat buffer has length
/// `W·H·channels·2`. The DC term (zero frequency) is **not** shifted to the
/// center: bin `(0, 0)` is the array origin, the natural ("un-shifted") DFT
/// layout, so the producing/consuming ops agree on bin indexing without an
/// implicit `fftshift`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectrumDescriptor {
    /// The spatial pixel extent of the transformed plane (the spectrum shares
    /// the source grid: `W×H` complex bins).
    pub extent: Extent,
    /// The number of logical (complex) channels; each is transformed
    /// independently. Equals the source plane's channel count.
    pub channels: u32,
    /// The scalar storage type of the interleaved real/imaginary components
    /// (always [`ScalarType::F32`] for reference semantics).
    pub scalar: ScalarType,
    /// Coordinate convention of the spatial source grid.
    pub coordinates: CoordinateConvention,
}

impl SpectrumDescriptor {
    /// The flat sample count of a packed spectrum value:
    /// `W·H·channels·2` (two `f32` — real then imaginary — per complex bin).
    ///
    /// # Errors
    /// Returns a [`policy`](ErrorClass::Policy) [`Error`] (`E_SPECTRUM_OVERFLOW`)
    /// if the product overflows `u64`.
    pub fn total_samples(&self) -> Result<u64> {
        self.extent
            .pixel_count()?
            .checked_mul(u64::from(self.channels))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                Error::new(
                    ErrorClass::Policy,
                    "E_SPECTRUM_OVERFLOW",
                    "spectrum sample count W*H*channels*2 overflows u64".to_owned(),
                )
                .with_context(crate::error::ErrorContext::default().with_actual(
                    format!(
                        "{}x{} x{} channels",
                        self.extent.width, self.extent.height, self.channels
                    ),
                ))
            })
    }

    /// Validate the descriptor: a positive channel count and a non-overflowing
    /// sample count.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) [`Error`] (`E_SPECTRUM_CHANNELS`)
    /// for a zero channel count, or the overflow error from
    /// [`total_samples`](Self::total_samples).
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 {
            return Err(Error::new(
                ErrorClass::Schema,
                "E_SPECTRUM_CHANNELS",
                "a spectrum must declare at least one channel".to_owned(),
            ));
        }
        let _ = self.total_samples()?;
        Ok(())
    }
}

/// The largest patch half-window (radius) a [`PatchFieldDescriptor`] may declare
/// (`OP_CATALOG` §10).
///
/// The matched patch is the `(2·r + 1) × (2·r + 1)` square centred on a pixel;
/// the radius is bounded so a descriptor cannot drive an unbounded comparison
/// window. The cap is generous (a 257×257 patch) yet finite, so a pathological
/// descriptor is rejected at validation rather than at allocation.
pub const MAX_PATCH_RADIUS: u32 = 128;

/// The number of interleaved samples a [`PatchField`](ResourceDescriptor::PatchField)
/// stores per target pixel: the matched source coordinate `(src_x, src_y)` and
/// the patch-match `cost`, in that order.
///
/// `src_x`/`src_y` are integer source coordinates carried losslessly in an `f32`
/// (every coordinate up to `2^24` is exactly representable); `cost` is the
/// patch distance the producing op minimised (smaller is a closer match).
pub const PATCH_FIELD_CHANNELS: u32 = 3;

/// A typed **patch correspondence field** (nearest-neighbour field) descriptor
/// (`OP_CATALOG` §10, `PatchMatch`).
///
/// A `PatchField` is the result of approximate nearest-neighbour patch matching:
/// for every pixel of a `target_extent` grid it records the source coordinate of
/// the most similar `(2·radius + 1)²` patch in a `source_extent` image, plus the
/// patch-match cost (the distance the producer minimised). It is the input a
/// patch-synthesis / inpainting op consumes to fill a hole from coherent source
/// texture.
///
/// The run-time value packs `PATCH_FIELD_CHANNELS` interleaved `f32` per target
/// pixel — `src_x`, `src_y`, then `cost` — in row-major, channel-interleaved
/// order: target pixel `(x, y)` occupies indices `3·(y·W + x) + {0,1,2}`. The
/// source coordinates are *absolute* pixel positions in the source image (the
/// anchor of the matched patch), not relative offsets, so a consumer never needs
/// the producer's offset convention to dereference a correspondence. Both extents
/// and the patch radius are explicit, serialized fields — the matching window is
/// part of the resource type and the op hash, never an unstated assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchFieldDescriptor {
    /// The pixel extent of the target grid: one correspondence per target pixel.
    pub target_extent: Extent,
    /// The pixel extent of the source image the correspondences point into.
    pub source_extent: Extent,
    /// The patch half-window (radius `r`): the matched patch is the
    /// `(2·r + 1) × (2·r + 1)` square centred on a pixel. Never implicit.
    pub radius: u32,
    /// Coordinate convention shared by the target grid and the source image.
    pub coordinates: CoordinateConvention,
}

impl PatchFieldDescriptor {
    /// The flat sample count of a packed patch-field value:
    /// `target_W·target_H·3` ([`PATCH_FIELD_CHANNELS`], the `(src_x, src_y, cost)`
    /// triple per target pixel).
    ///
    /// # Errors
    /// Returns a [`policy`](ErrorClass::Policy) [`Error`] (`E_PATCH_FIELD_OVERFLOW`)
    /// if the product overflows `u64`.
    pub fn total_samples(&self) -> Result<u64> {
        self.target_extent
            .pixel_count()?
            .checked_mul(u64::from(PATCH_FIELD_CHANNELS))
            .ok_or_else(|| {
                Error::new(
                    ErrorClass::Policy,
                    "E_PATCH_FIELD_OVERFLOW",
                    "patch-field sample count target_W*target_H*channels overflows u64".to_owned(),
                )
                .with_context(crate::error::ErrorContext::default().with_actual(
                    format!("{}x{}", self.target_extent.width, self.target_extent.height),
                ))
            })
    }

    /// Validate the descriptor: a non-degenerate source the correspondences can
    /// point into, a bounded patch radius, and a non-overflowing sample count.
    ///
    /// A valid `PatchField` has a non-empty `source_extent` (a correspondence must
    /// have at least one source pixel to reference) and a `radius` within
    /// [`MAX_PATCH_RADIUS`]. An empty `target_extent` is permitted (a zero-area
    /// target yields an empty field). The total sample count is checked so a
    /// pathological descriptor cannot drive an overflowing allocation.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) [`Error`] (`E_PATCH_FIELD_SOURCE`)
    /// for a degenerate source, (`E_PATCH_FIELD_RADIUS`) for an over-large radius,
    /// or the overflow error from [`total_samples`](Self::total_samples).
    pub fn validate(&self) -> Result<()> {
        if self.source_extent.width == 0 || self.source_extent.height == 0 {
            return Err(Error::new(
                ErrorClass::Schema,
                "E_PATCH_FIELD_SOURCE",
                "a patch field must have a non-empty source extent".to_owned(),
            )
            .with_context(crate::error::ErrorContext::default().with_actual(format!(
                "{}x{}",
                self.source_extent.width, self.source_extent.height
            ))));
        }
        if self.radius > MAX_PATCH_RADIUS {
            return Err(Error::new(
                ErrorClass::Schema,
                "E_PATCH_FIELD_RADIUS",
                format!(
                    "a patch field radius may be at most {MAX_PATCH_RADIUS}, got {}",
                    self.radius
                ),
            )
            .with_context(
                crate::error::ErrorContext::default()
                    .with_actual(self.radius.to_string())
                    .with_expected(format!("0..={MAX_PATCH_RADIUS}")),
            ));
        }
        let _ = self.total_samples()?;
        Ok(())
    }

    /// Whether `(x, y)` is a valid anchor coordinate in the source image: an
    /// in-bounds source pixel a correspondence may legally reference. The bounds
    /// are the half-open source domain `[0, source_W) × [0, source_H)`.
    #[must_use]
    pub const fn source_contains(&self, x: i64, y: i64) -> bool {
        x >= 0
            && y >= 0
            && x < self.source_extent.width as i64
            && y < self.source_extent.height as i64
    }
}

/// A typed resource descriptor: the tagged union of every resource kind this
/// bone defines (`IR_SPEC` §7).
///
/// The `kind` tag matches the spec's JSON (`"Image"`, `"Mask"`, `"Field1"`,
/// `"Field2"`, `"Field3"`, `"SdfMask"`, `"Report"`, `"Pyramid"`, `"Spectrum"`).
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
    /// An integer label map of `u32` component IDs.
    LabelMap(LabelMapDescriptor),
    /// A structured analysis report (carries no raster).
    Report(ReportDescriptor),
    /// A multi-resolution image/field pyramid (`OP_CATALOG` §13).
    Pyramid(PyramidDescriptor),
    /// A complex frequency spectrum (`OP_CATALOG` §9): the DFT of an image/field
    /// plane, two interleaved `f32` (real/imaginary) per bin.
    Spectrum(SpectrumDescriptor),
    /// A patch correspondence field (`OP_CATALOG` §10): a per-target-pixel source
    /// coordinate and patch-match cost, three interleaved `f32` per target pixel.
    PatchField(PatchFieldDescriptor),
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
            Self::LabelMap(d) => d.extent,
            Self::Report(d) => d.extent,
            // A pyramid's extent is its level-0 base extent.
            Self::Pyramid(d) => d.base_extent,
            // A spectrum shares the spatial extent of its source plane.
            Self::Spectrum(d) => d.extent,
            // A patch field's extent is its target grid.
            Self::PatchField(d) => d.target_extent,
        }
    }

    /// This descriptor with its pixel extent replaced by `extent`, leaving every
    /// other type field (layout, scalar, color, …) unchanged.
    ///
    /// The descriptor of a *window* of a resource (a tile, an ROI crop) is the
    /// resource's descriptor at the window's extent: the type semantics are
    /// identical, only the pixel count differs. A [`Report`](Self::Report) carries
    /// no raster, so its summarized extent is replaced verbatim.
    #[must_use]
    pub const fn with_extent(self, extent: Extent) -> Self {
        match self {
            Self::Image(mut d) => {
                d.extent = extent;
                Self::Image(d)
            }
            Self::Mask(mut d) => {
                d.extent = extent;
                Self::Mask(d)
            }
            Self::Field1(mut d) => {
                d.extent = extent;
                Self::Field1(d)
            }
            Self::Field2(mut d) => {
                d.extent = extent;
                Self::Field2(d)
            }
            Self::Field3(mut d) => {
                d.extent = extent;
                Self::Field3(d)
            }
            Self::SdfMask(mut d) => {
                d.extent = extent;
                Self::SdfMask(d)
            }
            Self::Report(mut d) => {
                d.extent = extent;
                Self::Report(d)
            }
            Self::LabelMap(mut d) => {
                d.extent = extent;
                Self::LabelMap(d)
            }
            // Replacing a pyramid's extent re-bases it: the level count, factor,
            // and phase are preserved and every deeper level is re-derived from
            // the new base by `level_extent`.
            Self::Pyramid(mut d) => {
                d.base_extent = extent;
                Self::Pyramid(d)
            }
            // Re-windowing a spectrum replaces its spatial extent; the bin grid
            // of the re-windowed transform is that new extent.
            Self::Spectrum(mut d) => {
                d.extent = extent;
                Self::Spectrum(d)
            }
            // Re-windowing a patch field replaces its target grid; the source it
            // points into and the patch radius are preserved.
            Self::PatchField(mut d) => {
                d.target_extent = extent;
                Self::PatchField(d)
            }
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
            Self::LabelMap(_) => ResourceKind::LabelMap,
            Self::Report(_) => ResourceKind::Report,
            Self::Pyramid(_) => ResourceKind::Pyramid,
            Self::Spectrum(_) => ResourceKind::Spectrum,
            Self::PatchField(_) => ResourceKind::PatchField,
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
    fn with_extent_replaces_only_the_extent() {
        let d = ResourceDescriptor::Image(sample_image());
        let windowed = d.with_extent(Extent::new(128, 128));
        assert_eq!(windowed.extent(), Extent::new(128, 128));
        assert_eq!(windowed.kind(), d.kind());
        // Every other type field is preserved.
        if let (ResourceDescriptor::Image(a), ResourceDescriptor::Image(b)) = (d, windowed) {
            assert_eq!(a.layout, b.layout);
            assert_eq!(a.color, b.color);
            assert_eq!(a.alpha, b.alpha);
            assert_eq!(a.semantic, b.semantic);
        } else {
            panic!("variant changed");
        }
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
    fn label_map_descriptor_round_trips_and_reports_kind() {
        use super::{LabelMapDescriptor, ResourceDescriptor};
        use crate::manifest::ResourceKind;
        let d = LabelMapDescriptor {
            extent: Extent::new(64, 48),
            scalar: ScalarType::U32,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        let back: LabelMapDescriptor =
            serde_json::from_value(serde_json::to_value(d).unwrap()).unwrap();
        assert_eq!(back, d);

        // The descriptor tag round-trips and reports the LabelMap kind/extent.
        let resource = ResourceDescriptor::LabelMap(d);
        assert_eq!(resource.kind(), ResourceKind::LabelMap);
        assert_eq!(resource.extent(), Extent::new(64, 48));
        let json = serde_json::to_value(resource).unwrap();
        assert_eq!(json["kind"], json!("LabelMap"));
        let recovered: ResourceDescriptor = serde_json::from_value(json).unwrap();
        assert_eq!(recovered, resource);
    }

    #[test]
    fn components_data_round_trips_with_large_areas() {
        use super::ComponentsData;
        // Areas span past 2^24 to exercise integer fidelity through serde.
        let data = ComponentsData {
            connectivity: 8,
            count: 3,
            areas: vec![1, 16_777_217, u64::from(u32::MAX) + 7],
        };
        let back: ComponentsData =
            serde_json::from_value(serde_json::to_value(&data).unwrap()).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn frequency_energy_data_round_trips() {
        use super::FrequencyEnergyData;
        let data = FrequencyEnergyData {
            decomposition: "laplacian".to_owned(),
            bands: 3,
            band_energy: vec![1.5, 0.25, 4.0],
            band_pixels: vec![64, 16, 4],
            total_energy: 5.75,
        };
        let back: FrequencyEnergyData =
            serde_json::from_value(serde_json::to_value(&data).unwrap()).unwrap();
        assert_eq!(back, data);
        // deny_unknown_fields rejects an unexpected field.
        let mut value = serde_json::to_value(&data).unwrap();
        value["bogus"] = json!(1);
        assert!(serde_json::from_value::<FrequencyEnergyData>(value).is_err());
    }

    #[test]
    fn solver_data_round_trips() {
        use super::SolverData;
        let data = SolverData {
            kind: "gray-scott".to_owned(),
            steps: 250,
            stability_number: 0.64,
            stability_limit: 1.0,
            stable: true,
            residual_history: vec![1.0, 0.5, 0.25],
            total_energy: 12.5,
            iterations: None,
            stop_reason: None,
            converged: None,
            tolerance: None,
            final_residual: None,
        };
        let back: SolverData =
            serde_json::from_value(serde_json::to_value(&data).unwrap()).unwrap();
        assert_eq!(back, data);
        // deny_unknown_fields rejects an unexpected field.
        let mut value = serde_json::to_value(&data).unwrap();
        value["bogus"] = json!(1);
        assert!(serde_json::from_value::<SolverData>(value).is_err());
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
    fn pyramid_descriptor_round_trips_and_reports_kind() {
        use super::{
            DownsampleFactor, PyramidDescriptor, PyramidPhase, ResourceDescriptor, ScalarType,
        };
        use crate::manifest::ResourceKind;
        let d = PyramidDescriptor {
            base_extent: Extent::new(64, 48),
            levels: 4,
            channels: 3,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        let value = serde_json::to_value(d).unwrap();
        assert_eq!(value["factor"], json!("half"));
        assert_eq!(value["phase"], json!("floor"));
        let back: PyramidDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(back, d);

        let resource = ResourceDescriptor::Pyramid(d);
        assert_eq!(resource.kind(), ResourceKind::Pyramid);
        assert_eq!(resource.extent(), Extent::new(64, 48));
        let json = serde_json::to_value(resource).unwrap();
        assert_eq!(json["kind"], json!("Pyramid"));
        let recovered: ResourceDescriptor = serde_json::from_value(json).unwrap();
        assert_eq!(recovered, resource);
    }

    #[test]
    fn pyramid_deny_unknown_fields() {
        use super::{DownsampleFactor, PyramidDescriptor, PyramidPhase, ScalarType};
        let d = PyramidDescriptor {
            base_extent: Extent::new(8, 8),
            levels: 2,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        let mut value = serde_json::to_value(d).unwrap();
        value["bogus"] = json!(1);
        let err = serde_json::from_value::<PyramidDescriptor>(value).unwrap_err();
        assert!(err.to_string().contains("bogus"), "{err}");
    }

    #[test]
    fn pyramid_floor_phase_derives_even_and_odd_levels() {
        use super::{DownsampleFactor, PyramidDescriptor, PyramidPhase, ScalarType};
        // Odd base sizes: floor halves and drops the last column/row.
        let d = PyramidDescriptor {
            base_extent: Extent::new(7, 5),
            levels: 4,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        assert_eq!(d.level_extent(0), Some(Extent::new(7, 5)));
        assert_eq!(d.level_extent(1), Some(Extent::new(3, 2))); // floor(7/2)=3, floor(5/2)=2
        assert_eq!(d.level_extent(2), Some(Extent::new(1, 1))); // floor(3/2)=1, floor(2/2)=1
        assert_eq!(d.level_extent(3), Some(Extent::new(1, 1))); // clamped at 1x1
        assert_eq!(d.level_extent(4), None); // out of range
    }

    #[test]
    fn pyramid_ceil_phase_keeps_half_populated_axes() {
        use super::{DownsampleFactor, PyramidDescriptor, PyramidPhase, ScalarType};
        let d = PyramidDescriptor {
            base_extent: Extent::new(7, 5),
            levels: 3,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Ceil,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        assert_eq!(d.level_extent(0), Some(Extent::new(7, 5)));
        assert_eq!(d.level_extent(1), Some(Extent::new(4, 3))); // ceil(7/2)=4, ceil(5/2)=3
        assert_eq!(d.level_extent(2), Some(Extent::new(2, 2))); // ceil(4/2)=2, ceil(3/2)=2
    }

    #[test]
    fn one_level_pyramid_is_valid_and_has_base_samples() {
        use super::{DownsampleFactor, PyramidDescriptor, PyramidPhase, ScalarType};
        let d = PyramidDescriptor {
            base_extent: Extent::new(5, 3),
            levels: 1,
            channels: 2,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        d.validate().unwrap();
        assert_eq!(d.level_extent(0), Some(Extent::new(5, 3)));
        assert_eq!(d.level_extent(1), None);
        // Only the base level: 5*3*2 = 30 samples.
        assert_eq!(d.total_samples().unwrap(), 30);
    }

    #[test]
    fn pyramid_total_samples_sums_every_level() {
        use super::{DownsampleFactor, PyramidDescriptor, PyramidPhase, ScalarType};
        let d = PyramidDescriptor {
            base_extent: Extent::new(4, 4),
            levels: 3,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        // 4x4=16, 2x2=4, 1x1=1 -> 21.
        assert_eq!(d.total_samples().unwrap(), 21);
    }

    #[test]
    fn pyramid_rejects_zero_and_overdeep_level_counts() {
        use super::{
            DownsampleFactor, MAX_PYRAMID_LEVELS, PyramidDescriptor, PyramidPhase, ScalarType,
        };
        let base = PyramidDescriptor {
            base_extent: Extent::new(8, 8),
            levels: 0,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        let err = base.validate().unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_PYRAMID_LEVELS");

        let too_deep = PyramidDescriptor {
            levels: MAX_PYRAMID_LEVELS + 1,
            ..base
        };
        let err = too_deep.validate().unwrap_err();
        assert_eq!(err.code, "E_PYRAMID_LEVELS");

        // A maximal-but-legal level count validates (extents clamp at 1x1).
        let ok = PyramidDescriptor {
            levels: MAX_PYRAMID_LEVELS,
            ..base
        };
        ok.validate().unwrap();
    }

    #[test]
    fn pyramid_with_extent_rebases_and_preserves_chain() {
        use super::{
            DownsampleFactor, PyramidDescriptor, PyramidPhase, ResourceDescriptor, ScalarType,
        };
        let d = PyramidDescriptor {
            base_extent: Extent::new(64, 64),
            levels: 3,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        let rebased = ResourceDescriptor::Pyramid(d).with_extent(Extent::new(16, 16));
        assert_eq!(rebased.extent(), Extent::new(16, 16));
        if let ResourceDescriptor::Pyramid(p) = rebased {
            assert_eq!(p.levels, 3);
            assert_eq!(p.phase, PyramidPhase::Floor);
            assert_eq!(p.level_extent(2), Some(Extent::new(4, 4)));
        } else {
            panic!("variant changed");
        }
    }

    fn sample_patch_field() -> super::PatchFieldDescriptor {
        super::PatchFieldDescriptor {
            target_extent: Extent::new(8, 6),
            source_extent: Extent::new(12, 10),
            radius: 2,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        }
    }

    #[test]
    fn patch_field_round_trips_and_reports_kind() {
        use super::ResourceDescriptor;
        use crate::manifest::ResourceKind;
        let d = sample_patch_field();
        let value = serde_json::to_value(d).unwrap();
        let back: super::PatchFieldDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(back, d);

        let resource = ResourceDescriptor::PatchField(d);
        assert_eq!(resource.kind(), ResourceKind::PatchField);
        assert_eq!(resource.extent(), Extent::new(8, 6));
        let json = serde_json::to_value(resource).unwrap();
        assert_eq!(json["kind"], json!("PatchField"));
        let recovered: ResourceDescriptor = serde_json::from_value(json).unwrap();
        assert_eq!(recovered, resource);
    }

    #[test]
    fn patch_field_deny_unknown_fields() {
        let d = sample_patch_field();
        let mut value = serde_json::to_value(d).unwrap();
        value["bogus"] = json!(7);
        let err = serde_json::from_value::<super::PatchFieldDescriptor>(value).unwrap_err();
        assert!(err.to_string().contains("bogus"), "{err}");
    }

    #[test]
    fn patch_field_total_samples_is_target_pixels_times_three() {
        let d = sample_patch_field();
        // 8*6 target pixels * 3 channels = 144.
        assert_eq!(d.total_samples().unwrap(), 8 * 6 * 3);
    }

    #[test]
    fn patch_field_validation_accepts_valid_and_rejects_malformed() {
        sample_patch_field().validate().unwrap();

        // Degenerate source: a correspondence has nowhere to point.
        let mut bad = sample_patch_field();
        bad.source_extent = Extent::new(0, 10);
        let err = bad.validate().unwrap_err();
        assert_eq!(err.code, "E_PATCH_FIELD_SOURCE");

        // Over-large radius.
        let mut huge = sample_patch_field();
        huge.radius = super::MAX_PATCH_RADIUS + 1;
        let err = huge.validate().unwrap_err();
        assert_eq!(err.code, "E_PATCH_FIELD_RADIUS");

        // Empty target is permitted: a zero-area field is well-formed.
        let mut empty = sample_patch_field();
        empty.target_extent = Extent::new(0, 0);
        empty.validate().unwrap();
        assert_eq!(empty.total_samples().unwrap(), 0);
    }

    #[test]
    fn patch_field_source_contains_uses_half_open_source_domain() {
        let d = sample_patch_field(); // source 12x10
        assert!(d.source_contains(0, 0));
        assert!(d.source_contains(11, 9));
        assert!(!d.source_contains(12, 0));
        assert!(!d.source_contains(0, 10));
        assert!(!d.source_contains(-1, 0));
    }

    #[test]
    fn patch_field_rewindow_replaces_target_keeps_source() {
        use super::ResourceDescriptor;
        let d = sample_patch_field();
        let rebased = ResourceDescriptor::PatchField(d).with_extent(Extent::new(4, 4));
        assert_eq!(rebased.extent(), Extent::new(4, 4));
        if let ResourceDescriptor::PatchField(p) = rebased {
            assert_eq!(p.target_extent, Extent::new(4, 4));
            assert_eq!(p.source_extent, Extent::new(12, 10));
            assert_eq!(p.radius, 2);
        } else {
            panic!("variant changed");
        }
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
