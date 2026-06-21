//! **Content hashing** of a resource value — the input identity a cache key is
//! built from (`plan.md` §10.3: "hash content and relevant metadata").
//!
//! A producer's cache key folds in the *content hash* of every input, never its
//! path. The content hash is the BLAKE3 identity of a value's bytes **and** its
//! declared semantics: the descriptor (extent, color encoding, alpha, …) framed
//! ahead of the raw sample bytes. Two values with identical pixels but different
//! declared meaning therefore hash apart, exactly as the cache key requires.
//!
//! The framing is mixed under [`HashDomain::Content`] so a content hash can never
//! collide with a plan, resource, manifest, or cache-entry hash even on identical
//! bytes. The raw `f32` samples are hashed in fixed little-endian byte order so
//! the hash is stable across machines; non-finite samples (a `NaN` an op may
//! produce) are hashed by their bit pattern and never routed through the
//! JSON-canonical path, which forbids them.

use std::fmt;

use paintop_ir::{HashDomain, ResourceDescriptor, SemanticHash, hash_canonical_bytes};

use crate::executor::ResourceValue;

/// A resource value's content hash — its bytes-plus-semantics BLAKE3 identity.
///
/// A thin newtype over [`SemanticHash`] so a content hash is never silently
/// confused with a plan or cache key in an API signature. Its serialized form is
/// the underlying `blake3:<hex>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(SemanticHash);

impl ContentHash {
    /// The underlying [`SemanticHash`].
    #[must_use]
    pub const fn hash(self) -> SemanticHash {
        self.0
    }

    /// Parse a serialized `blake3:<hex>` content hash.
    ///
    /// # Errors
    /// Returns the [`parse`](paintop_ir::ErrorClass::Parse) error if `text` is not
    /// a valid `blake3:<hex>` id.
    pub fn parse(text: &str) -> paintop_ir::Result<Self> {
        SemanticHash::parse(text).map(Self)
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The wire tag distinguishing a raster-value framing from a report framing, so a
/// report and a same-descriptor empty raster can never collide.
const RASTER_TAG: u8 = 0x01;
/// The wire tag for a report-value framing.
const REPORT_TAG: u8 = 0x02;

/// Hash a whole-image [`ResourceValue`]'s content: its descriptor semantics
/// framed ahead of its raw `f32` sample bytes.
///
/// The bytes the digest absorbs are, in order: a kind tag, the length-prefixed
/// canonical-JSON descriptor, the channel count, the sample count, and the
/// little-endian `f32` sample bytes. The length prefixes make every boundary
/// unambiguous, so no two distinct (descriptor, samples) pairs can frame to the
/// same byte stream.
///
/// A report value (no raster) is framed by its descriptor plus the report's own
/// canonical content hash string, since a report carries no sample buffer.
#[must_use]
pub fn content_hash_value(value: &ResourceValue) -> ContentHash {
    if let Some(report) = value.as_report() {
        return content_hash_report(value.descriptor(), &report.content_hash);
    }
    content_hash_descriptor(value.descriptor(), value.samples())
}

/// Hash a (descriptor, samples) pair directly, the raster path of
/// [`content_hash_value`].
///
/// Exposed so callers (and tests) can compute a content hash from a descriptor
/// and a sample slice without first wrapping a [`ResourceValue`].
#[must_use]
pub fn content_hash_descriptor(descriptor: &ResourceDescriptor, samples: &[f32]) -> ContentHash {
    let mut bytes = Vec::new();
    bytes.push(RASTER_TAG);
    push_descriptor(&mut bytes, descriptor);
    // Sample count, then the little-endian f32 bytes. The count is redundant with
    // the descriptor for a well-formed value, but framing it explicitly keeps the
    // stream unambiguous even for a degenerate buffer.
    push_len(&mut bytes, samples.len());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    ContentHash(hash_canonical_bytes(HashDomain::Content, &bytes))
}

/// Hash a report value: its descriptor plus the report's own content-hash string.
fn content_hash_report(descriptor: &ResourceDescriptor, report_hash: &str) -> ContentHash {
    let mut bytes = Vec::new();
    bytes.push(REPORT_TAG);
    push_descriptor(&mut bytes, descriptor);
    let report_bytes = report_hash.as_bytes();
    push_len(&mut bytes, report_bytes.len());
    bytes.extend_from_slice(report_bytes);
    ContentHash(hash_canonical_bytes(HashDomain::Content, &bytes))
}

/// Append the length-prefixed canonical-JSON descriptor to `bytes`.
///
/// A descriptor serializes to finite JSON (its fields are enums and integers), so
/// canonicalization never fails; the fallback empty framing only exists to keep
/// this total without a panic and is unreachable for a real descriptor.
fn push_descriptor(bytes: &mut Vec<u8>, descriptor: &ResourceDescriptor) {
    let descriptor_bytes = serde_json::to_value(descriptor)
        .ok()
        .and_then(|v| paintop_ir::to_canonical_bytes(&v).ok())
        .unwrap_or_default();
    push_len(bytes, descriptor_bytes.len());
    bytes.extend_from_slice(&descriptor_bytes);
}

/// Append a little-endian `u64` length prefix to `bytes`.
fn push_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend_from_slice(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::{ContentHash, content_hash_descriptor, content_hash_value};
    use crate::executor::ResourceValue;
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
    };

    fn image(color: ColorEncoding) -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(2, 2),
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    fn value(color: ColorEncoding, fill: f32) -> ResourceValue {
        ResourceValue::new(image(color), 4, vec![fill; 2 * 2 * 4]).expect("value")
    }

    #[test]
    fn identical_values_hash_identically() {
        assert_eq!(
            content_hash_value(&value(ColorEncoding::LinearSrgb, 0.5)),
            content_hash_value(&value(ColorEncoding::LinearSrgb, 0.5)),
        );
    }

    #[test]
    fn differing_samples_hash_apart() {
        assert_ne!(
            content_hash_value(&value(ColorEncoding::LinearSrgb, 0.5)),
            content_hash_value(&value(ColorEncoding::LinearSrgb, 0.6)),
        );
    }

    #[test]
    fn differing_semantics_hash_apart() {
        // Same bytes, different declared color encoding.
        assert_ne!(
            content_hash_value(&value(ColorEncoding::LinearSrgb, 0.5)),
            content_hash_value(&value(ColorEncoding::Srgb, 0.5)),
        );
    }

    #[test]
    fn non_finite_samples_hash_stably() {
        let nan = content_hash_descriptor(&image(ColorEncoding::LinearSrgb), &[f32::NAN]);
        let nan_again = content_hash_descriptor(&image(ColorEncoding::LinearSrgb), &[f32::NAN]);
        assert_eq!(nan, nan_again, "NaN bit pattern hashes deterministically");
    }

    #[test]
    fn hash_round_trips_through_parse() {
        let h = content_hash_value(&value(ColorEncoding::LinearSrgb, 0.5));
        assert_eq!(ContentHash::parse(&h.to_string()).expect("parse"), h);
        assert!(h.to_string().starts_with("blake3:"));
    }
}
