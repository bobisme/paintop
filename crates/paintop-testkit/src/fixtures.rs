//! Analytic fixture generation from fixed, versioned formulas.
//!
//! Every fixture kind here (`AGENT_VERIFICATION` §2.3) is produced by an exact
//! formula over integer pixel coordinates, materialized as an exact numeric
//! array. The array — not any rendered PNG — is the source of truth; the PNG is
//! a human preview only.
//!
//! # Determinism
//!
//! All arithmetic that feeds the stored array is either integer or computed in
//! `f64` through deterministic, library-independent routines (see
//! [`det_sin`]) before being cast to the stored scalar type. We deliberately
//! avoid `f32::sin`/`f64::sin`, whose results are not guaranteed bit-identical
//! across platforms, so a fixture's bytes — and therefore its `sha256` — are
//! reproducible on any machine.
//!
//! # Hashing
//!
//! The fixture digest is taken over a small canonical header (a scalar-type
//! tag plus width, height, and channel count as little-endian `u32`s) followed
//! by the array's raw little-endian element bytes. Hashing canonical bytes —
//! never a float's textual rendering — keeps the digest independent of JSON
//! float formatting (`M0_DECISIONS`: never hash raw `serde_json` output).

// Fixture math must be *bit-reproducible across targets*. Fused multiply-add
// (`mul_add`) is permitted by IEEE-754 to differ from a separate multiply and
// add, and whether the compiler emits an FMA is target-dependent, so using it
// would break the cross-machine byte-identity the fixtures guarantee. We
// therefore keep the explicit `a*b + c` form and allow `suboptimal_flops`.
#![allow(
    clippy::suboptimal_flops,
    reason = "explicit multiply-then-add is required for cross-target bit-reproducibility; mul_add/FMA is not deterministic across machines"
)]

use std::collections::BTreeMap;

use paintop_ir::error::{Error, ErrorClass, ErrorContext, Result};
use paintop_ir::resource::{Extent, ScalarType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The schema/format version stamped into every generated [`Manifest`].
///
/// Bump this whenever a formula changes in a way that alters output bytes, so
/// stale checked-in digests are detectable.
pub const FIXTURE_FORMAT_VERSION: u32 = 1;

/// The exact numeric payload of a fixture, in its declared scalar type.
///
/// Values are stored channel-interleaved in row-major order
/// (`y` outer, `x` inner, channel innermost). The variant matches the
/// fixture's [`ScalarType`].
#[derive(Debug, Clone, PartialEq)]
pub enum FixtureData {
    /// 8-bit unsigned samples (e.g. label-map previews, binary geometry).
    U8(Vec<u8>),
    /// 32-bit unsigned samples (e.g. large-ID label maps).
    U32(Vec<u32>),
    /// 32-bit float samples (the reference type for color/mask/field math).
    F32(Vec<f32>),
}

impl FixtureData {
    /// The scalar type of this payload.
    #[must_use]
    pub const fn scalar(&self) -> ScalarType {
        match self {
            Self::U8(_) => ScalarType::U8,
            Self::U32(_) => ScalarType::U32,
            Self::F32(_) => ScalarType::F32,
        }
    }

    /// The number of scalar elements in the payload.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    /// Whether the payload has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The canonical little-endian byte serialization of the raw samples.
    ///
    /// `f32` elements are serialized via [`f32::to_le_bytes`], so the exact
    /// bit pattern (including any `NaN`/`Inf` payload) is preserved.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Self::U8(v) => v.clone(),
            Self::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

/// A generated analytic fixture: its formula identity, dimensions, the exact
/// numeric array, and the digest of its canonical bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Fixture {
    /// The formula name (e.g. `"impulse"`), matching the CLI kind.
    pub formula: String,
    /// The parameters the formula was evaluated with, in a stable key order.
    pub params: BTreeMap<String, FixtureParam>,
    /// The pixel extent.
    pub extent: Extent,
    /// The number of interleaved channels per pixel.
    pub channels: u32,
    /// The exact numeric payload.
    pub data: FixtureData,
}

/// A single fixture parameter value, recorded in the manifest for provenance.
///
/// Kept as a small closed enum (rather than `serde_json::Value`) so the
/// manifest has a stable, `deny_unknown_fields`-friendly shape and floats are
/// recorded losslessly as their textual value alongside the digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, untagged)]
pub enum FixtureParam {
    /// An unsigned integer parameter (dimensions, coordinates, ids, periods).
    Uint(u64),
    /// A floating parameter (values, frequencies, radii).
    Float(f64),
    /// A string parameter (an enumerated mode).
    Text(String),
}

impl From<u64> for FixtureParam {
    fn from(v: u64) -> Self {
        Self::Uint(v)
    }
}

impl From<u32> for FixtureParam {
    fn from(v: u32) -> Self {
        Self::Uint(u64::from(v))
    }
}

impl From<f64> for FixtureParam {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<&str> for FixtureParam {
    fn from(v: &str) -> Self {
        Self::Text(v.to_owned())
    }
}

/// The `AGENT_VERIFICATION` §4.2-shaped manifest recorded next to a generated
/// fixture: which formula and parameters produced it, at what version, and the
/// `sha256` of its canonical bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The formula name.
    pub formula: String,
    /// The fixture-format version (see [`FIXTURE_FORMAT_VERSION`]).
    pub version: u32,
    /// The parameters used, in a stable key order.
    pub params: BTreeMap<String, FixtureParam>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Channels per pixel.
    pub channels: u32,
    /// The scalar storage type of the array (`u8` / `u32` / `f32`).
    pub scalar: ScalarType,
    /// Lowercase hex `sha256` of the fixture's canonical bytes.
    pub sha256: String,
}

impl Fixture {
    /// The canonical bytes the digest is taken over: a header (scalar tag,
    /// width, height, channels) followed by the raw little-endian samples.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        // Scalar tag distinguishes equal byte-lengths across types.
        let tag: u8 = match &self.data {
            FixtureData::U8(_) => 0,
            FixtureData::U32(_) => 1,
            FixtureData::F32(_) => 2,
        };
        bytes.push(tag);
        bytes.extend_from_slice(&self.extent.width.to_le_bytes());
        bytes.extend_from_slice(&self.extent.height.to_le_bytes());
        bytes.extend_from_slice(&self.channels.to_le_bytes());
        bytes.extend_from_slice(&self.data.to_le_bytes());
        bytes
    }

    /// The lowercase hex `sha256` of [`Fixture::canonical_bytes`].
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_bytes());
        let digest = hasher.finalize();
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write;
            // Writing to a String never fails.
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    /// Build the §4.2-shaped [`Manifest`] for this fixture.
    #[must_use]
    pub fn manifest(&self) -> Manifest {
        Manifest {
            formula: self.formula.clone(),
            version: FIXTURE_FORMAT_VERSION,
            params: self.params.clone(),
            width: self.extent.width,
            height: self.extent.height,
            channels: self.channels,
            scalar: self.data.scalar(),
            sha256: self.sha256_hex(),
        }
    }

    /// The exact numeric array serialized as canonical JSON (pretty-printed),
    /// suitable for writing alongside the manifest.
    ///
    /// The shape is `{"formula","version","width","height","channels",
    /// "scalar","data":[...]}`; `f32` arrays are emitted as JSON numbers, which
    /// are auxiliary — the digest is taken over the binary bytes, not this
    /// text.
    ///
    /// # Errors
    /// Returns an [`export`](ErrorClass::Export) error only if the array cannot
    /// be rendered to JSON, which does not occur for the finite numeric vectors
    /// here (non-finite `f32` are emitted via their bit pattern, see below).
    pub fn to_json(&self) -> Result<String> {
        // `serde_json` cannot represent NaN/Inf as JSON numbers, so encode the
        // exact f32 array as its little-endian u32 bit patterns to stay lossless
        // and parseable. Integer arrays serialize directly.
        let data = match &self.data {
            FixtureData::U8(v) => {
                serde_json::Value::Array(v.iter().map(|&x| serde_json::json!(x)).collect())
            }
            FixtureData::U32(v) => {
                serde_json::Value::Array(v.iter().map(|&x| serde_json::json!(x)).collect())
            }
            FixtureData::F32(v) => serde_json::Value::Array(
                v.iter().map(|&x| serde_json::json!(x.to_bits())).collect(),
            ),
        };
        let encoding = match &self.data {
            FixtureData::F32(_) => "f32-le-bits",
            FixtureData::U8(_) | FixtureData::U32(_) => "int",
        };
        let doc = serde_json::json!({
            "formula": self.formula,
            "version": FIXTURE_FORMAT_VERSION,
            "width": self.extent.width,
            "height": self.extent.height,
            "channels": self.channels,
            "scalar": self.data.scalar(),
            "encoding": encoding,
            "data": data,
        });
        serde_json::to_string_pretty(&doc).map_err(|e| {
            Error::new(
                ErrorClass::Export,
                "E_FIXTURE_JSON",
                format!("failed to render fixture array to JSON: {e}"),
            )
        })
    }

    /// Render an 8-bit preview PNG of the fixture for human inspection.
    ///
    /// The preview is **auxiliary** and lossy: float data is clamped to
    /// `[0, 1]` and quantized to 8-bit, and multi-channel data is mapped to
    /// grayscale/RGBA as appropriate. It must never be used as a test oracle.
    ///
    /// # Errors
    /// Returns an [`export`](ErrorClass::Export) error if the PNG encoder
    /// fails.
    pub fn to_preview_png(&self) -> Result<Vec<u8>> {
        use image::{ImageEncoder, ImageError, codecs::png::PngEncoder};

        let width = self.extent.width;
        let height = self.extent.height;
        // Map every fixture onto an RGBA8 preview buffer.
        let px_count = (width as usize) * (height as usize);
        let mut rgba = vec![0u8; px_count * 4];
        for i in 0..px_count {
            let (red, green, blue, alpha) = self.preview_pixel(i);
            let base = i * 4;
            rgba[base] = red;
            rgba[base + 1] = green;
            rgba[base + 2] = blue;
            rgba[base + 3] = alpha;
        }
        let mut out = Vec::new();
        let encoder = PngEncoder::new(&mut out);
        encoder
            .write_image(&rgba, width, height, image::ExtendedColorType::Rgba8)
            .map_err(|e: ImageError| {
                Error::new(
                    ErrorClass::Export,
                    "E_FIXTURE_PNG",
                    format!("failed to encode preview PNG: {e}"),
                )
            })?;
        Ok(out)
    }

    /// Compute the 8-bit RGBA preview tuple for pixel index `i`.
    fn preview_pixel(&self, i: usize) -> (u8, u8, u8, u8) {
        let ch = self.channels as usize;
        let base = i * ch;
        let sample = |c: usize| -> u8 {
            let idx = base + c;
            match &self.data {
                FixtureData::U8(v) => v.get(idx).copied().unwrap_or(0),
                FixtureData::U32(v) => {
                    // Show low byte so large ids are at least visible.
                    u8::try_from(v.get(idx).copied().unwrap_or(0) & 0xff).unwrap_or(0)
                }
                FixtureData::F32(v) => {
                    let f = v.get(idx).copied().unwrap_or(0.0);
                    quantize_unit_f32(f)
                }
            }
        };
        match ch {
            1 => {
                let l = sample(0);
                (l, l, l, 255)
            }
            2 => {
                let l = sample(0);
                (l, l, l, sample(1))
            }
            3 => (sample(0), sample(1), sample(2), 255),
            _ => (sample(0), sample(1), sample(2), sample(3)),
        }
    }
}

/// Clamp a float to `[0, 1]` and quantize to 8-bit (preview only). Non-finite
/// values render as magenta-ish full white so they stand out.
fn quantize_unit_f32(f: f32) -> u8 {
    if f.is_nan() {
        return 255;
    }
    let c = f.clamp(0.0, 1.0);
    // Round-to-nearest; deterministic. The product lies in [0, 255], so the cast
    // never truncates or loses sign.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "preview-only quantization; c in [0,1] so c*255+0.5 in [0.5,255.5] casts to a valid u8"
    )]
    let q = (f64::from(c) * 255.0 + 0.5) as u8;
    q
}

/// A deterministic, platform-independent sine for `x` in radians.
///
/// `f64::sin` is not guaranteed bit-identical across targets, which would make
/// the sine-grating fixture non-reproducible. This routine performs Cody–Waite
/// range reduction to `[-pi/4, pi/4]` and evaluates a fixed minimax-style
/// polynomial, giving the same bits everywhere for the same input.
#[must_use]
pub fn det_sin(x: f64) -> f64 {
    use std::f64::consts::FRAC_PI_2;
    // Reduce to nearest multiple of pi/2: x = steps*(pi/2) + rem, |rem| <= pi/4.
    let steps = (x / FRAC_PI_2).round();
    let rem = x - steps * FRAC_PI_2;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "steps.rem_euclid(4.0) is exactly one of 0.0/1.0/2.0/3.0 for the finite inputs the fixtures use, so the i64 cast is exact"
    )]
    let quadrant = steps.rem_euclid(4.0) as i64;
    let rem2 = rem * rem;
    // 7th-order Taylor for sin and 6th-order for cos on |rem|<=pi/4; the error is
    // below ~1e-9, ample for analytic fixtures and fully deterministic.
    let sin_r = rem * (1.0 - rem2 * (1.0 / 6.0 - rem2 * (1.0 / 120.0 - rem2 * (1.0 / 5040.0))));
    let cos_r =
        1.0 - rem2 * (0.5 - rem2 * (1.0 / 24.0 - rem2 * (1.0 / 720.0 - rem2 * (1.0 / 40320.0))));
    match quadrant {
        0 => sin_r,
        1 => cos_r,
        2 => -sin_r,
        _ => -cos_r,
    }
}

/// Index helper: the flat element offset of channel `c` of pixel `(x, y)` in a
/// `channels`-interleaved, row-major buffer of width `w`.
const fn idx(x: u32, y: u32, w: u32, channels: u32, c: u32) -> usize {
    (((y as usize) * (w as usize) + (x as usize)) * channels as usize) + c as usize
}

/// Validate a non-zero, non-overflowing extent for a single-channel fixture.
fn checked_len(extent: Extent, channels: u32) -> Result<usize> {
    if extent.width == 0 || extent.height == 0 {
        return Err(Error::new(
            ErrorClass::Policy,
            "E_FIXTURE_EMPTY",
            "fixture extent must be non-empty (width and height >= 1)",
        )
        .with_context(
            ErrorContext::default().with_actual(format!("{}x{}", extent.width, extent.height)),
        ));
    }
    let pixels = extent.pixel_count()?;
    pixels
        .checked_mul(u64::from(channels))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Policy,
                "E_FIXTURE_TOO_LARGE",
                "fixture element count overflows addressable memory",
            )
        })
}

/// A constant scalar/color field: every channel of every pixel takes `value`.
///
/// Detects bias, non-unit kernels, and boundary errors (`AGENT_VERIFICATION`
/// §2.3).
///
/// # Errors
/// Fails with a [`policy`](ErrorClass::Policy) error for an empty or
/// overflowing extent.
pub fn constant(extent: Extent, channels: u32, value: f32) -> Result<Fixture> {
    let len = checked_len(extent, channels)?;
    let data = vec![value; len];
    let mut params = BTreeMap::new();
    params.insert("channels".to_owned(), FixtureParam::from(channels));
    params.insert("value".to_owned(), FixtureParam::from(f64::from(value)));
    Ok(Fixture {
        formula: "constant".to_owned(),
        params,
        extent,
        channels,
        data: FixtureData::F32(data),
    })
}

/// A single unit impulse at `(x, y)`: that pixel is `1.0`, all others `0.0`.
///
/// Detects kernel shape, centering, support, and splat normalization
/// (`AGENT_VERIFICATION` §2.3). Single-channel.
///
/// # Errors
/// Fails if the extent is empty/overflowing, or if `(x, y)` is out of bounds.
pub fn impulse(extent: Extent, x: u32, y: u32) -> Result<Fixture> {
    let len = checked_len(extent, 1)?;
    if x >= extent.width || y >= extent.height {
        return Err(Error::new(
            ErrorClass::Policy,
            "E_FIXTURE_OUT_OF_BOUNDS",
            "impulse coordinate lies outside the extent",
        )
        .with_context(
            ErrorContext::default()
                .with_actual(format!("({x},{y})"))
                .with_expected(format!("0..{} x 0..{}", extent.width, extent.height)),
        ));
    }
    let mut data = vec![0.0_f32; len];
    data[idx(x, y, extent.width, 1, 0)] = 1.0;
    let mut params = BTreeMap::new();
    params.insert("x".to_owned(), FixtureParam::from(x));
    params.insert("y".to_owned(), FixtureParam::from(y));
    Ok(Fixture {
        formula: "impulse".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// The axis a [`ramp`] increases along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampAxis {
    /// Value increases with `x` (left to right).
    Horizontal,
    /// Value increases with `y` (top to bottom).
    Vertical,
}

impl RampAxis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
        }
    }
}

/// A linear ramp from `0.0` to `1.0` along `axis`.
///
/// Value at coordinate `t` (in `0..n`) is `t / (n - 1)`, so the endpoints are
/// exactly `0.0` and `1.0`. For a degenerate `n == 1` the single column/row is
/// `0.0`. Detects derivative, interpolation, coordinate, and gamma errors
/// (`AGENT_VERIFICATION` §2.3). Single-channel.
///
/// # Errors
/// Fails with a [`policy`](ErrorClass::Policy) error for an empty/overflowing
/// extent.
pub fn ramp(extent: Extent, axis: RampAxis) -> Result<Fixture> {
    let len = checked_len(extent, 1)?;
    let mut data = vec![0.0_f32; len];
    let denom = match axis {
        RampAxis::Horizontal => extent.width.saturating_sub(1),
        RampAxis::Vertical => extent.height.saturating_sub(1),
    };
    for y in 0..extent.height {
        for x in 0..extent.width {
            let t = match axis {
                RampAxis::Horizontal => x,
                RampAxis::Vertical => y,
            };
            #[allow(
                clippy::cast_possible_truncation,
                reason = "ramp is computed in f64 for determinism then narrowed to the f32 storage type; the value is in [0,1] and fits exactly"
            )]
            let v = if denom == 0 {
                0.0_f32
            } else {
                (f64::from(t) / f64::from(denom)) as f32
            };
            data[idx(x, y, extent.width, 1, 0)] = v;
        }
    }
    let mut params = BTreeMap::new();
    params.insert("axis".to_owned(), FixtureParam::from(axis.as_str()));
    Ok(Fixture {
        formula: "ramp".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// A checkerboard of `tile`-pixel squares alternating `0.0` and `1.0`.
///
/// Pixel `(x, y)` is `1.0` when `(x / tile + y / tile)` is even. Detects
/// aliasing, phase, resampling, and tile-seam errors (`AGENT_VERIFICATION`
/// §2.3). Single-channel.
///
/// # Errors
/// Fails for an empty/overflowing extent or `tile == 0`.
pub fn checkerboard(extent: Extent, tile: u32) -> Result<Fixture> {
    if tile == 0 {
        return Err(Error::new(
            ErrorClass::Policy,
            "E_FIXTURE_BAD_PARAM",
            "checkerboard tile size must be >= 1",
        ));
    }
    let len = checked_len(extent, 1)?;
    let mut data = vec![0.0_f32; len];
    for y in 0..extent.height {
        for x in 0..extent.width {
            let cell = (x / tile + y / tile) % 2;
            let v = if cell == 0 { 1.0_f32 } else { 0.0_f32 };
            data[idx(x, y, extent.width, 1, 0)] = v;
        }
    }
    let mut params = BTreeMap::new();
    params.insert("tile".to_owned(), FixtureParam::from(tile));
    Ok(Fixture {
        formula: "checkerboard".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// A horizontal sine grating with `periods` full cycles across the width,
/// remapped to `[0, 1]`.
///
/// Value at column `x` is `0.5 + 0.5 * sin(2*pi*periods*(x + 0.5)/width)`,
/// evaluated through the deterministic [`det_sin`]. Detects frequency response
/// and blur/sharpen behavior (`AGENT_VERIFICATION` §2.3). Single-channel.
///
/// # Errors
/// Fails for an empty/overflowing extent.
pub fn sine_grating(extent: Extent, periods: f64) -> Result<Fixture> {
    use std::f64::consts::TAU;
    let len = checked_len(extent, 1)?;
    let mut data = vec![0.0_f32; len];
    let w = f64::from(extent.width);
    for y in 0..extent.height {
        for x in 0..extent.width {
            // Sample at the pixel center (x + 0.5) per the coordinate convention.
            let phase = TAU * periods * (f64::from(x) + 0.5) / w;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the grating is computed in f64 for determinism then narrowed to f32 storage; value lies in [0,1]"
            )]
            let v = (0.5 + 0.5 * det_sin(phase)) as f32;
            data[idx(x, y, extent.width, 1, 0)] = v;
        }
    }
    let mut params = BTreeMap::new();
    params.insert("periods".to_owned(), FixtureParam::from(periods));
    Ok(Fixture {
        formula: "sine_grating".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// A binary rectangle: `1.0` inside the half-open box `[x0, x0+rw) ×
/// [y0, y0+rh)`, `0.0` outside.
///
/// Detects rasterization and morphology errors (`AGENT_VERIFICATION` §2.3).
/// Single-channel; no antialiasing (exact integer membership).
///
/// # Errors
/// Fails for an empty/overflowing extent.
pub fn rectangle(extent: Extent, x0: u32, y0: u32, rw: u32, rh: u32) -> Result<Fixture> {
    let len = checked_len(extent, 1)?;
    let mut data = vec![0.0_f32; len];
    let x1 = x0.saturating_add(rw);
    let y1 = y0.saturating_add(rh);
    for y in 0..extent.height {
        for x in 0..extent.width {
            let inside = x >= x0 && x < x1 && y >= y0 && y < y1;
            data[idx(x, y, extent.width, 1, 0)] = if inside { 1.0 } else { 0.0 };
        }
    }
    let mut params = BTreeMap::new();
    params.insert("x0".to_owned(), FixtureParam::from(x0));
    params.insert("y0".to_owned(), FixtureParam::from(y0));
    params.insert("rw".to_owned(), FixtureParam::from(rw));
    params.insert("rh".to_owned(), FixtureParam::from(rh));
    Ok(Fixture {
        formula: "rectangle".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// A binary disc of radius `radius` centered at `(cx, cy)` (pixel centers).
///
/// Pixel `(x, y)` is `1.0` when `(x+0.5-cx-0.5)^2 + ...` — concretely when the
/// squared distance from the pixel center to `(cx + 0.5, cy + 0.5)` is `<=
/// radius^2`. Computed in integer/`f64` for exact, deterministic membership.
/// Detects rasterization, SDF, and antialiasing errors (`AGENT_VERIFICATION`
/// §2.3). Single-channel.
///
/// # Errors
/// Fails for an empty/overflowing extent.
pub fn circle(extent: Extent, cx: u32, cy: u32, radius: f64) -> Result<Fixture> {
    let len = checked_len(extent, 1)?;
    let mut data = vec![0.0_f32; len];
    let r2 = radius * radius;
    let ccx = f64::from(cx) + 0.5;
    let ccy = f64::from(cy) + 0.5;
    for y in 0..extent.height {
        for x in 0..extent.width {
            let px = f64::from(x) + 0.5;
            let py = f64::from(y) + 0.5;
            let d2 = (px - ccx) * (px - ccx) + (py - ccy) * (py - ccy);
            data[idx(x, y, extent.width, 1, 0)] = if d2 <= r2 { 1.0 } else { 0.0 };
        }
    }
    let mut params = BTreeMap::new();
    params.insert("cx".to_owned(), FixtureParam::from(cx));
    params.insert("cy".to_owned(), FixtureParam::from(cy));
    params.insert("radius".to_owned(), FixtureParam::from(radius));
    Ok(Fixture {
        formula: "circle".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// An RGBA alpha-edge fixture with *hidden* RGB under transparent pixels.
///
/// The left half (`x < width/2`) is opaque (`alpha = 1`) with the visible
/// color `(0.2, 0.4, 0.6)`; the right half is fully transparent (`alpha = 0`)
/// but stores a non-zero "hidden" RGB `(0.9, 0.1, 0.1)`. The straight-alpha
/// representation deliberately keeps colour under zero coverage so
/// premultiplication and fringe bugs are exposed (`AGENT_VERIFICATION` §2.3,
/// §3.2). RGBA, straight alpha.
///
/// # Errors
/// Fails for an empty/overflowing extent.
pub fn alpha_edge(extent: Extent) -> Result<Fixture> {
    let len = checked_len(extent, 4)?;
    let mut data = vec![0.0_f32; len];
    let half = extent.width / 2;
    for y in 0..extent.height {
        for x in 0..extent.width {
            let base = idx(x, y, extent.width, 4, 0);
            if x < half {
                data[base] = 0.2;
                data[base + 1] = 0.4;
                data[base + 2] = 0.6;
                data[base + 3] = 1.0;
            } else {
                // Hidden RGB beneath zero coverage.
                data[base] = 0.9;
                data[base + 1] = 0.1;
                data[base + 2] = 0.1;
                data[base + 3] = 0.0;
            }
        }
    }
    let params = BTreeMap::new();
    Ok(Fixture {
        formula: "alpha_edge".to_owned(),
        params,
        extent,
        channels: 4,
        data: FixtureData::F32(data),
    })
}

/// A single-channel `f32` field with `NaN`/`Inf` deliberately injected.
///
/// Pixel `0` is `NaN`, pixel `1` (if present) is `+Inf`, pixel `2` is `-Inf`;
/// every other pixel is a finite ramp value `x_index / count`. Exercises
/// finite-value validation (`AGENT_VERIFICATION` §2.3). Single-channel.
///
/// # Errors
/// Fails for an empty/overflowing extent.
pub fn nan_inf_field(extent: Extent) -> Result<Fixture> {
    let len = checked_len(extent, 1)?;
    let mut data = vec![0.0_f32; len];
    #[allow(
        clippy::cast_precision_loss,
        reason = "fixtures are small; the index-to-f32 ramp need not be exact, only deterministic"
    )]
    let denom = len as f32;
    for (i, slot) in data.iter_mut().enumerate() {
        *slot = match i {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            #[allow(
                clippy::cast_precision_loss,
                reason = "deterministic index ramp; precision is irrelevant"
            )]
            other => (other as f32) / denom,
        };
    }
    let params = BTreeMap::new();
    Ok(Fixture {
        formula: "nan_inf_field".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::F32(data),
    })
}

/// A `u32` label map whose ids start at `base` and increment per pixel.
///
/// Pixel `(x, y)` gets id `base + y*width + x`, so with a large `base` the ids
/// exceed the `u16`/`f16` range and expose integer attachment/encoding loss
/// (`AGENT_VERIFICATION` §2.3). Single-channel `u32`.
///
/// # Errors
/// Fails for an empty/overflowing extent, or if `base + last_index` overflows
/// `u32`.
pub fn label_map(extent: Extent, base: u32) -> Result<Fixture> {
    let len = checked_len(extent, 1)?;
    let last = u32::try_from(len - 1).map_err(|_| {
        Error::new(
            ErrorClass::Policy,
            "E_FIXTURE_TOO_LARGE",
            "label-map index exceeds u32",
        )
    })?;
    base.checked_add(last).ok_or_else(|| {
        Error::new(
            ErrorClass::Policy,
            "E_FIXTURE_ID_OVERFLOW",
            "label-map base + max index overflows u32",
        )
        .with_context(ErrorContext::default().with_actual(format!("base={base}, max_index={last}")))
    })?;
    let mut data = vec![0_u32; len];
    for (i, slot) in data.iter_mut().enumerate() {
        // Safe: i <= last <= u32::MAX and base + last does not overflow.
        *slot = base + u32::try_from(i).unwrap_or(0);
    }
    let mut params = BTreeMap::new();
    params.insert("base".to_owned(), FixtureParam::from(base));
    Ok(Fixture {
        formula: "label_map".to_owned(),
        params,
        extent,
        channels: 1,
        data: FixtureData::U32(data),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(w: u32, h: u32) -> Extent {
        Extent::new(w, h)
    }

    /// Exact `f32` equality via bit pattern, avoiding the pedantic `float_cmp`
    /// lint. Fixture values like `0.0`/`1.0` are exact, so bit equality is the
    /// correct, lint-clean comparison.
    fn bits_eq(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    fn impulse_places_a_single_one() {
        let f = impulse(e(65, 65), 32, 32).unwrap();
        let FixtureData::F32(v) = &f.data else {
            panic!("impulse is f32")
        };
        assert_eq!(v.len(), 65 * 65);
        let mut ones = 0;
        for (i, &x) in v.iter().enumerate() {
            if bits_eq(x, 1.0) {
                ones += 1;
                assert_eq!(i, 32 * 65 + 32);
            } else {
                assert!(bits_eq(x, 0.0));
            }
        }
        assert_eq!(ones, 1);
    }

    #[test]
    fn impulse_rejects_out_of_bounds() {
        let err = impulse(e(8, 8), 8, 0).unwrap_err();
        assert_eq!(err.code, "E_FIXTURE_OUT_OF_BOUNDS");
    }

    #[test]
    fn empty_extent_is_rejected() {
        let err = constant(e(0, 4), 1, 1.0).unwrap_err();
        assert_eq!(err.code, "E_FIXTURE_EMPTY");
    }

    #[test]
    fn ramp_endpoints_are_exact() {
        let f = ramp(e(5, 1), RampAxis::Horizontal).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        assert!(bits_eq(v[0], 0.0));
        assert!(bits_eq(v[4], 1.0));
        assert!((v[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ramp_degenerate_single_column_is_zero() {
        let f = ramp(e(1, 4), RampAxis::Horizontal).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        assert!(v.iter().all(|&x| bits_eq(x, 0.0)));
    }

    #[test]
    fn checkerboard_top_left_is_one() {
        let f = checkerboard(e(4, 4), 1).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        assert!(bits_eq(v[0], 1.0));
        assert!(bits_eq(v[1], 0.0));
        assert!(bits_eq(v[4], 0.0)); // (0,1)
    }

    #[test]
    fn checkerboard_rejects_zero_tile() {
        assert_eq!(
            checkerboard(e(4, 4), 0).unwrap_err().code,
            "E_FIXTURE_BAD_PARAM"
        );
    }

    #[test]
    fn det_sin_matches_std_within_tolerance() {
        // Sample [-10, 10] on a fixed integer grid to avoid float loop counters.
        for i in -1000..=1000 {
            let x = f64::from(i) * 0.01;
            assert!((det_sin(x) - x.sin()).abs() < 1e-6, "x={x}");
        }
    }

    #[test]
    fn sine_grating_is_in_unit_range() {
        let f = sine_grating(e(32, 1), 2.0).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        for &s in v {
            assert!((0.0..=1.0).contains(&s), "{s}");
        }
    }

    #[test]
    fn rectangle_membership_is_half_open() {
        let f = rectangle(e(8, 8), 2, 2, 3, 3).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        let at = |x: u32, y: u32| v[(y * 8 + x) as usize];
        assert!(bits_eq(at(2, 2), 1.0));
        assert!(bits_eq(at(4, 4), 1.0));
        assert!(bits_eq(at(5, 4), 0.0)); // x == x0 + rw is excluded
        assert!(bits_eq(at(1, 2), 0.0));
    }

    #[test]
    fn circle_center_is_inside() {
        let f = circle(e(9, 9), 4, 4, 3.0).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        assert!(bits_eq(v[(4 * 9 + 4) as usize], 1.0));
        assert!(bits_eq(v[0], 0.0)); // far corner
    }

    #[test]
    fn alpha_edge_hides_rgb_under_transparency() {
        let f = alpha_edge(e(4, 1)).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        // Right half: alpha 0 but RGB non-zero.
        let base = (2 * 4) as usize;
        assert!(bits_eq(v[base + 3], 0.0));
        assert!(v[base] > 0.0);
    }

    #[test]
    fn nan_inf_field_injects_non_finite() {
        let f = nan_inf_field(e(4, 1)).unwrap();
        let FixtureData::F32(v) = &f.data else {
            unreachable!()
        };
        assert!(v[0].is_nan());
        assert!(v[1].is_infinite() && v[1] > 0.0);
        assert!(v[2].is_infinite() && v[2] < 0.0);
        assert!(v[3].is_finite());
    }

    #[test]
    fn label_map_increments_from_base() {
        let f = label_map(e(2, 2), 1_000_000).unwrap();
        let FixtureData::U32(v) = &f.data else {
            unreachable!()
        };
        assert_eq!(v, &[1_000_000, 1_000_001, 1_000_002, 1_000_003]);
    }

    #[test]
    fn label_map_rejects_id_overflow() {
        let err = label_map(e(4, 4), u32::MAX).unwrap_err();
        assert_eq!(err.code, "E_FIXTURE_ID_OVERFLOW");
    }

    #[test]
    fn tiny_images_generate() {
        // 1x1, 1xN, 2x2 boundary cases.
        assert!(constant(e(1, 1), 1, 0.5).is_ok());
        assert!(ramp(e(1, 7), RampAxis::Vertical).is_ok());
        assert!(checkerboard(e(2, 2), 1).is_ok());
    }

    #[test]
    fn manifest_records_formula_params_version_and_digest() {
        let f = impulse(e(8, 8), 1, 1).unwrap();
        let m = f.manifest();
        assert_eq!(m.formula, "impulse");
        assert_eq!(m.version, FIXTURE_FORMAT_VERSION);
        assert_eq!(m.scalar, ScalarType::F32);
        assert_eq!(m.width, 8);
        assert_eq!(m.sha256.len(), 64);
        assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(m.params.get("x"), Some(&FixtureParam::Uint(1)));
    }

    #[test]
    fn manifest_serde_round_trips_with_deny_unknown_fields() {
        let m = impulse(e(8, 8), 1, 1).unwrap().manifest();
        let json = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        // Unknown field is rejected.
        let bad = json.replace("\"formula\"", "\"bogus\":1,\"formula\"");
        assert!(serde_json::from_str::<Manifest>(&bad).is_err());
    }

    #[test]
    fn digest_is_stable_across_regeneration() {
        // Every fixture kind must hash identically when regenerated.
        let pairs: Vec<(Fixture, Fixture)> = vec![
            (
                constant(e(16, 16), 3, 0.25).unwrap(),
                constant(e(16, 16), 3, 0.25).unwrap(),
            ),
            (
                impulse(e(65, 65), 32, 32).unwrap(),
                impulse(e(65, 65), 32, 32).unwrap(),
            ),
            (
                ramp(e(16, 16), RampAxis::Horizontal).unwrap(),
                ramp(e(16, 16), RampAxis::Horizontal).unwrap(),
            ),
            (
                checkerboard(e(16, 16), 4).unwrap(),
                checkerboard(e(16, 16), 4).unwrap(),
            ),
            (
                sine_grating(e(64, 1), 3.0).unwrap(),
                sine_grating(e(64, 1), 3.0).unwrap(),
            ),
            (
                rectangle(e(16, 16), 2, 2, 5, 5).unwrap(),
                rectangle(e(16, 16), 2, 2, 5, 5).unwrap(),
            ),
            (
                circle(e(33, 33), 16, 16, 8.0).unwrap(),
                circle(e(33, 33), 16, 16, 8.0).unwrap(),
            ),
            (alpha_edge(e(8, 4)).unwrap(), alpha_edge(e(8, 4)).unwrap()),
            (
                nan_inf_field(e(8, 4)).unwrap(),
                nan_inf_field(e(8, 4)).unwrap(),
            ),
            (
                label_map(e(8, 4), 100_000).unwrap(),
                label_map(e(8, 4), 100_000).unwrap(),
            ),
        ];
        for (a, b) in pairs {
            assert_eq!(a.sha256_hex(), b.sha256_hex(), "{} not stable", a.formula);
            assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        }
    }

    #[test]
    fn distinct_kinds_have_distinct_digests() {
        let a = constant(e(16, 16), 1, 1.0).unwrap().sha256_hex();
        let b = constant(e(16, 16), 1, 0.0).unwrap().sha256_hex();
        assert_ne!(a, b);
    }

    #[test]
    fn nan_field_digest_is_deterministic_despite_non_finite() {
        // The canonical bytes preserve NaN's exact bit pattern, so two runs
        // agree even though NaN != NaN under float comparison.
        let a = nan_inf_field(e(8, 8)).unwrap();
        let b = nan_inf_field(e(8, 8)).unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.sha256_hex(), b.sha256_hex());
    }

    #[test]
    fn preview_png_encodes() {
        let f = checkerboard(e(8, 8), 2).unwrap();
        let png = f.to_preview_png().unwrap();
        // PNG magic.
        assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn json_round_trip_preserves_f32_bits() {
        let f = nan_inf_field(e(4, 1)).unwrap();
        let json = f.to_json().unwrap();
        let doc: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(doc["encoding"], "f32-le-bits");
        let bits = u32::try_from(doc["data"][0].as_u64().unwrap()).unwrap();
        assert!(f32::from_bits(bits).is_nan());
    }
}
