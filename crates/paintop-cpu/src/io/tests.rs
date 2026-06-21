//! Verification suite for `io.decode_image@1` / `io.encode_image@1`
//! (`AGENT_VERIFICATION` §2.2–§2.9, §2.5 differential, §2.9 fuzzing):
//!
//! - **schema/contract**: the manifests validate, agree with their contracts,
//!   and the verification declarations gate clean;
//! - **analytic/round-trip**: a constructed image round-trips
//!   decode∘encode∘decode byte-exactly, and metadata (layout/alpha) is preserved;
//! - **differential**: paintop's decode of an `image`-crate-encoded PNG matches
//!   the bytes `image` itself decodes (the §2.5 "internal PNG path vs known
//!   decoder" comparator);
//! - **fuzzing / adversarial**: a seed corpus of malformed, truncated, oversize,
//!   and decompression-bomb PNGs is rejected with the typed `asset` / `policy`
//!   codes rather than panicking or exhausting memory.

use std::path::PathBuf;

use image::{ColorType, ImageEncoder};
use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    ChannelLayout, ImageDescriptor, ResourceDescriptor, check_contract_consistency,
    verify_categories,
};

use super::{
    DecodeImage, DecodeLimits, E_DECODE_LIMIT_EXCEEDED, E_DECODE_MALFORMED,
    E_DECODE_UNSUPPORTED_FORMAT, EncodeImage, decode_png, encode_png,
};

/// A unique scratch path in the temp dir, so parallel tests never collide.
fn scratch(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("paintop_io_{tag}_{}_{n}.png", std::process::id()))
}

/// Build a normalized-`f32` [`ResourceValue`] from raw 8-bit samples and a layout.
fn image_value(width: u32, height: u32, layout: ChannelLayout, raw: &[u8]) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(super::decoded_image_descriptor(
        paintop_ir::Extent::new(width, height),
        layout,
    ));
    let samples: Vec<f32> = raw.iter().map(|&b| f32::from(b) / 255.0).collect();
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// A small RGBA gradient with a varying alpha edge.
fn rgba_fixture() -> (u32, u32, Vec<u8>) {
    let (w, h): (u8, u8) = (4, 3);
    let mut raw = Vec::with_capacity(usize::from(w) * usize::from(h) * 4);
    for y in 0..h {
        for x in 0..w {
            raw.push(x * 60);
            raw.push(y * 80);
            raw.push((x + y) * 30);
            raw.push(if x == 0 { 0 } else { 255 });
        }
    }
    (u32::from(w), u32::from(h), raw)
}

/// Re-quantize a value's samples back to the 8-bit raster they encode.
fn requantize(value: &ResourceValue) -> Vec<u8> {
    value
        .samples()
        .iter()
        .map(|&s| super::quantize(s))
        .collect()
}

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let decode_manifest = DecodeImage::manifest().expect("decode manifest");
    decode_manifest.validate().expect("decode manifest valid");
    check_contract_consistency(&decode_manifest, &DecodeImage::new())
        .expect("decode manifest agrees with contract");
    verify_categories(&decode_manifest, &decode_manifest.test.verification)
        .expect("decode verification declarations gate clean");

    let encode_manifest = EncodeImage::manifest().expect("encode manifest");
    encode_manifest.validate().expect("encode manifest valid");
    check_contract_consistency(&encode_manifest, &EncodeImage::new())
        .expect("encode manifest agrees with contract");
    verify_categories(&encode_manifest, &encode_manifest.test.verification)
        .expect("encode verification declarations gate clean");
}

#[test]
fn decode_then_encode_round_trips() {
    let (w, h, raw) = rgba_fixture();
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);

    // encode -> decode -> the raster is byte-identical.
    let png = encode_png(&value).expect("encode");
    let decoded = decode_png(&png, &DecodeLimits::DEFAULT).expect("decode");
    assert_eq!(decoded.extent(), value.extent());
    assert_eq!(requantize(&decoded), raw);
}

#[test]
fn round_trip_preserves_layout_and_alpha() {
    for (layout, channels) in [
        (ChannelLayout::Gray, 1u32),
        (ChannelLayout::GrayA, 2),
        (ChannelLayout::Rgb, 3),
        (ChannelLayout::Rgba, 4),
    ] {
        let (w, h) = (3u32, 2u32);
        let raw: Vec<u8> = (0..(w * h * channels))
            .map(|i| u8::try_from(i.wrapping_mul(7) % 256).expect("fits in u8"))
            .collect();
        let value = image_value(w, h, layout, &raw);

        let png = encode_png(&value).expect("encode");
        let decoded = decode_png(&png, &DecodeLimits::DEFAULT).expect("decode");
        let ResourceDescriptor::Image(ImageDescriptor {
            layout: decoded_layout,
            alpha,
            color,
            ..
        }) = *decoded.descriptor()
        else {
            panic!("decoded resource is an image");
        };
        assert_eq!(decoded_layout, layout, "layout preserved for {layout:?}");
        assert_eq!(alpha, paintop_ir::AlphaRepresentation::Straight);
        assert_eq!(color, paintop_ir::ColorEncoding::Srgb);
        assert_eq!(
            requantize(&decoded),
            raw,
            "samples preserved for {layout:?}"
        );
    }
}

#[test]
fn encode_is_deterministic() {
    let (w, h, raw) = rgba_fixture();
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);
    let a = encode_png(&value).expect("encode a");
    let b = encode_png(&value).expect("encode b");
    assert_eq!(a, b, "encoding the same image twice is byte-identical");
}

/// §2.5 differential: paintop's decode of an `image`-crate-encoded PNG matches
/// the raster the `image` crate decodes from the same bytes.
#[test]
fn differential_against_image_crate_decoder() {
    let (w, h, raw) = rgba_fixture();

    // Encode the same raster with the upstream `image` crate directly.
    let mut reference_png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut reference_png)
        .write_image(&raw, w, h, ColorType::Rgba8.into())
        .expect("reference encode");

    // paintop decode.
    let ours = decode_png(&reference_png, &DecodeLimits::DEFAULT).expect("paintop decode");

    // upstream decode of the same bytes.
    let reference = image::load_from_memory(&reference_png).expect("reference decode");
    let reference_rgba = reference.to_rgba8();

    assert_eq!(ours.extent().width, reference_rgba.width());
    assert_eq!(ours.extent().height, reference_rgba.height());
    assert_eq!(requantize(&ours), reference_rgba.as_raw().as_slice());
}

#[test]
fn execute_decode_then_encode_end_to_end() {
    let (w, h, raw) = rgba_fixture();
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);

    // Encode to a file via the op implementation (atomic write).
    let path = scratch("e2e");
    let encode = EncodeImage::new();
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let params = serde_json::json!({ "path": path.to_str().expect("utf8 path") });
    let produced = encode.compute(&inputs, &params).expect("encode op");
    assert!(
        produced.contains_key("image"),
        "encode passes image through"
    );
    assert!(path.exists(), "encode wrote the file atomically");

    // Decode it back via the op implementation.
    let decode = DecodeImage::new();
    let decoded = decode
        .compute(&InputValues::new(), &params)
        .expect("decode op");
    let decoded_image = decoded.get("image").expect("decode produced image");
    assert_eq!(requantize(decoded_image), raw);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn missing_path_param_is_rejected() {
    let decode = DecodeImage::new();
    let err = decode
        .compute(&InputValues::new(), &serde_json::json!({}))
        .expect_err("missing path must fail");
    assert_eq!(err.code, super::E_IO_PATH_PARAM);
}

// --- §2.9 fuzzing / adversarial seed corpus -------------------------------

#[test]
fn truncated_png_is_rejected() {
    let (w, h, raw) = rgba_fixture();
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);
    let png = encode_png(&value).expect("encode");

    // Truncate to the signature + part of the header: not a decodable image.
    let truncated = &png[..png.len() / 2];
    let err = decode_png(truncated, &DecodeLimits::DEFAULT).expect_err("truncated must fail");
    assert_eq!(err.code, E_DECODE_MALFORMED);
    assert_eq!(err.class, paintop_ir::ErrorClass::Asset);
}

#[test]
fn garbage_bytes_are_rejected() {
    let garbage = [0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03];
    let err = decode_png(&garbage, &DecodeLimits::DEFAULT).expect_err("garbage must fail");
    assert_eq!(err.code, E_DECODE_MALFORMED);
}

#[test]
fn empty_input_is_rejected() {
    let err = decode_png(&[], &DecodeLimits::DEFAULT).expect_err("empty must fail");
    assert_eq!(err.code, E_DECODE_MALFORMED);
}

#[test]
fn oversize_dimensions_are_rejected() {
    // A genuine 64x64 image, but a limit that forbids anything past 16px: the
    // codec's own dimension limit rejects it as a `policy` violation.
    let (w, h) = (64u32, 64u32);
    let raw = vec![128u8; (w * h * 4) as usize];
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);
    let png = encode_png(&value).expect("encode");

    let tight = DecodeLimits {
        max_width: 16,
        max_height: 16,
        max_pixels: 256,
        max_alloc_bytes: DecodeLimits::DEFAULT.max_alloc_bytes,
    };
    let err = decode_png(&png, &tight).expect_err("oversize must fail");
    assert_eq!(err.code, E_DECODE_LIMIT_EXCEEDED);
    assert_eq!(err.class, paintop_ir::ErrorClass::Policy);
}

#[test]
fn decompression_bomb_is_rejected_by_alloc_ceiling() {
    // A real, large image but a tiny allocation ceiling: the codec refuses to
    // inflate it, surfacing a `policy` limit error rather than exhausting RAM.
    let (w, h) = (256u32, 256u32);
    let raw = vec![200u8; (w * h * 4) as usize];
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);
    let png = encode_png(&value).expect("encode");

    let bomb_guard = DecodeLimits {
        max_width: DecodeLimits::DEFAULT.max_width,
        max_height: DecodeLimits::DEFAULT.max_height,
        max_pixels: DecodeLimits::DEFAULT.max_pixels,
        max_alloc_bytes: 1024, // 1 KiB ceiling: far below the decoded raster.
    };
    let err = decode_png(&png, &bomb_guard).expect_err("bomb must fail");
    assert_eq!(err.code, E_DECODE_LIMIT_EXCEEDED);
    assert_eq!(err.class, paintop_ir::ErrorClass::Policy);
}

#[test]
fn sixteen_bit_png_is_rejected_as_unsupported() {
    // Encode a 16-bit grayscale PNG with the upstream crate; paintop's M0 import
    // set is 8-bit only, so it is rejected as an unsupported format (not silently
    // truncated).
    let (w, h) = (2u32, 2u32);
    let samples: Vec<u16> = vec![0, 16_000, 32_000, 65_535];
    let raw: Vec<u8> = samples.iter().flat_map(|s| s.to_be_bytes()).collect();
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&raw, w, h, ColorType::L16.into())
        .expect("16-bit encode");

    let err = decode_png(&png, &DecodeLimits::DEFAULT).expect_err("16-bit must be rejected");
    assert_eq!(err.code, E_DECODE_UNSUPPORTED_FORMAT);
    assert_eq!(err.class, paintop_ir::ErrorClass::Asset);
}

#[test]
fn atomic_write_replaces_existing_file() {
    let path = scratch("atomic");
    std::fs::write(&path, b"stale contents").expect("seed file");

    let (w, h, raw) = rgba_fixture();
    let value = image_value(w, h, ChannelLayout::Rgba, &raw);
    let png = encode_png(&value).expect("encode");
    super::atomic_write(&path, &png).expect("atomic write");

    let written = std::fs::read(&path).expect("read back");
    assert_eq!(
        written, png,
        "atomic write replaced the file with the new bytes"
    );
    // No temp artifact is left behind.
    let tmp = super::temp_sibling(&path);
    assert!(!tmp.exists(), "temp sibling cleaned up");

    let _ = std::fs::remove_file(&path);
}

/// The checked-in `ops/manifests/<id>.json` files (read by `cargo xtask
/// verify-op`) must stay byte-identical to the Rust manifest builders, which are
/// the source of truth. If this fails, regenerate with `serde_json::to_string_pretty`.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        DecodeImage::manifest().expect("decode manifest"),
        EncodeImage::manifest().expect("encode manifest"),
    ] {
        let path = root.join(format!("{}.json", manifest.id));
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let expected = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
        assert_eq!(
            on_disk.trim_end(),
            expected.trim_end(),
            "{} is stale; regenerate from the Rust builder",
            path.display()
        );
    }
}
