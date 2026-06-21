//! Fuzz target 2: the PNG decode/encode codec boundary
//! (`plan.md` §17.2, §19 M0; `AGENT_VERIFICATION` §2.2).
//!
//! Drives arbitrary bytes through [`paintop_cpu::io::decode_png`] under the
//! hardened [`DecodeLimits`] the runtime uses — the dimension/pixel/allocation
//! ceilings that refuse decompression bombs at the header before any raster is
//! allocated. Malformed, truncated, oversized, and adversarial-header inputs
//! must all resolve to a classified `Ok`/`Err` without panicking, aborting, or
//! exhausting memory.
//!
//! On a successful decode the value is re-encoded with
//! [`encode_png`](paintop_cpu::io::encode_png), exercising the encode boundary
//! on decoder-produced rasters (a cheap round-trip that catches encode-side
//! panics on otherwise-valid images).
#![no_main]

use libfuzzer_sys::fuzz_target;
use paintop_cpu::io::{DecodeLimits, decode_png, encode_png};

fuzz_target!(|data: &[u8]| {
    // Use a tightened ceiling so the fuzzer cannot drive a (validly small but)
    // multi-megapixel allocation per case and starve the campaign; this still
    // exercises every limit branch and the malformed/truncated paths. The real
    // runtime default is larger (`DecodeLimits::DEFAULT`).
    let limits = DecodeLimits {
        max_width: 1024,
        max_height: 1024,
        max_pixels: 1024 * 1024,
        max_alloc_bytes: 16 * 1024 * 1024,
    };

    if let Ok(value) = decode_png(data, &limits) {
        // A raster the decoder accepted must re-encode without panicking.
        let _ = encode_png(&value);
    }
});
