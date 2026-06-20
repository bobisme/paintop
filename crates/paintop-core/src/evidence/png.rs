//! A tiny, dependency-free, deterministic RGBA8 PNG encoder for evidence
//! artifacts (the contact sheet and failure diffs).
//!
//! Evidence images must be **byte-reproducible**: re-running a plan and
//! re-compositing the same pixels must yield the same `contact-sheet.png` bytes
//! so a bundle diffs cleanly. A general image library optimizes for size with
//! adaptive filtering and DEFLATE, neither of which is guaranteed stable across
//! versions. Instead we emit the simplest *valid* PNG: 8-bit RGBA, filter type 0
//! on every scanline, wrapped in a single **stored** (uncompressed) zlib stream.
//! The result is larger than an optimized PNG but is small (contact sheets are
//! thumbnails), spec-correct, and identical byte-for-byte for identical pixels.
//!
//! Only the encoder lives here; the compositor that decides *what* pixels go in
//! the sheet is [`contact`](crate::evidence::contact).

/// The 8-byte PNG signature.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Encode `rgba` (row-major, 4 bytes per pixel, `width * height * 4` bytes) as an
/// 8-bit RGBA PNG.
///
/// # Panics
/// Never: a `width`/`height` mismatch with the buffer length is reported as an
/// error rather than indexing out of bounds.
///
/// # Errors
/// Returns `None` if `rgba.len() != width * height * 4` or either dimension is
/// zero — the caller has built an inconsistent raster.
#[must_use]
pub fn encode_rgba(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    let expected = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    if width == 0 || height == 0 || rgba.len() != expected {
        return None;
    }

    let mut out = Vec::with_capacity(expected + 256);
    out.extend_from_slice(&PNG_SIGNATURE);
    write_chunk(&mut out, *b"IHDR", &ihdr(width, height));
    write_chunk(&mut out, *b"IDAT", &idat(width, height, rgba));
    write_chunk(&mut out, *b"IEND", &[]);
    Some(out)
}

/// The IHDR payload: width, height, 8-bit depth, color type 6 (RGBA), default
/// compression / filter / interlace.
fn ihdr(width: u32, height: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(13);
    v.extend_from_slice(&width.to_be_bytes());
    v.extend_from_slice(&height.to_be_bytes());
    v.push(8); // bit depth
    v.push(6); // color type: truecolor with alpha
    v.push(0); // compression: deflate
    v.push(0); // filter: adaptive (we use filter 0 per scanline)
    v.push(0); // interlace: none
    v
}

/// The IDAT payload: the filtered scanlines wrapped in a stored zlib stream.
fn idat(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let stride = (width as usize) * 4;
    // Prepend a filter-type byte (0 = None) to each scanline.
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0);
        let start = row * stride;
        raw.extend_from_slice(&rgba[start..start + stride]);
    }
    zlib_stored(&raw)
}

/// Wrap `data` in a zlib stream using only **stored** (type 0) DEFLATE blocks,
/// so encoding is trivial and deterministic.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    // zlib header: CMF=0x78 (deflate, 32K window), FLG chosen so (CMF<<8|FLG) % 31 == 0.
    out.push(0x78);
    out.push(0x01);

    // DEFLATE stored blocks: each carries up to 65 535 bytes.
    let mut chunks = data.chunks(0xFFFF).peekable();
    if chunks.peek().is_none() {
        // Empty input still needs one final empty stored block.
        out.push(0x01); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(!0u16).to_le_bytes());
    } else {
        while let Some(chunk) = chunks.next() {
            let last = chunks.peek().is_none();
            out.push(u8::from(last)); // BFINAL bit, BTYPE=00
            #[expect(
                clippy::cast_possible_truncation,
                reason = "chunk length is bounded by 0xFFFF by construction"
            )]
            let len = chunk.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }

    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Write a PNG chunk: 4-byte big-endian length, 4-byte type, payload, 4-byte
/// CRC over type+payload.
fn write_chunk(out: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "PNG chunk lengths are bounded well under u32::MAX for our artifacts"
    )]
    let len = payload.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(payload);
    let mut crc_input = Vec::with_capacity(4 + payload.len());
    crc_input.extend_from_slice(&kind);
    crc_input.extend_from_slice(payload);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// The Adler-32 checksum used by zlib.
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// The CRC-32 used by PNG chunks (IEEE polynomial, reflected).
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::{PNG_SIGNATURE, adler32, crc32, encode_rgba};

    #[test]
    fn rejects_inconsistent_dimensions() {
        // 2x2 RGBA needs 16 bytes.
        assert!(encode_rgba(2, 2, &[0; 15]).is_none());
        assert!(encode_rgba(0, 2, &[]).is_none());
        assert!(encode_rgba(2, 0, &[]).is_none());
    }

    #[test]
    fn encodes_signature_and_required_chunks() {
        let png = encode_rgba(1, 1, &[10, 20, 30, 255]).expect("encode");
        assert!(png.starts_with(&PNG_SIGNATURE));
        // The three required chunk type tags appear in order.
        let body = &png[8..];
        let ihdr = body.windows(4).position(|w| w == b"IHDR");
        let idat = body.windows(4).position(|w| w == b"IDAT");
        let iend = body.windows(4).position(|w| w == b"IEND");
        assert!(ihdr < idat);
        assert!(idat < iend);
    }

    #[test]
    fn encoding_is_byte_deterministic() {
        let pixels: Vec<u8> = (0u16..4 * 4 * 4)
            .map(|i| u8::try_from(i % 256).unwrap_or(0))
            .collect();
        let a = encode_rgba(4, 4, &pixels).expect("a");
        let b = encode_rgba(4, 4, &pixels).expect("b");
        assert_eq!(a, b, "identical pixels must yield identical bytes");
    }

    #[test]
    fn known_checksum_vectors() {
        // Wikipedia/zlib reference vectors for the ASCII string "Wikipedia".
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        // CRC-32 of the empty input is 0.
        assert_eq!(crc32(&[]), 0);
        // CRC-32/ISO-HDLC of "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn decodes_with_a_real_png_reader() {
        // Confirm the bytes are a valid PNG a real decoder accepts and
        // round-trips losslessly (the `image` crate is a dev-dependency).
        let png = encode_rgba(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("encode");
        let decoded = image::load_from_memory(&png).expect("decode").to_rgba8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw().as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
