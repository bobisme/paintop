//! Fuzz target 1: the strict plan parse/canonicalize boundary
//! (`plan.md` §10.1, §17.1, §19 M0; `AGENT_VERIFICATION` §2.1).
//!
//! Drives arbitrary bytes through [`paintop_ir::parse_plan`] — the exact
//! hardened pipeline the CLI uses — which layers, in order:
//!
//! 1. `check_limits`: byte-size, nesting-depth, node-count, and inline-payload
//!    ceilings enforced *before* allocation, so a deeply nested or oversized
//!    document fails fast (no unbounded allocation / stack blow-up).
//! 2. `scan_json`: duplicate-key and `NaN`/`Infinity`/overflow number rejection.
//! 3. `serde_json` deserialization with `deny_unknown_fields`.
//!
//! The invariant under test is liveness: for *any* input the call must return
//! `Ok`/`Err` without panicking, aborting, or running away on memory. On a
//! successful parse we additionally canonicalize and semantic-hash the plan,
//! exercising the §17 normalizer and BLAKE3 hashing on attacker-influenced (but
//! structurally valid) graphs.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The parser takes `&str`; non-UTF-8 inputs are simply not plans. (The
    // codec target covers raw-byte boundaries.)
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    match paintop_ir::parse_plan(text) {
        Ok(plan) => {
            // A structurally valid plan must normalize, canonicalize, and hash
            // without panicking: exercise the §17 normalizer + canonical-bytes +
            // BLAKE3 path on graphs the fuzzer steered into the typed model.
            if let Ok(value) = paintop_ir::normalized_value(&plan) {
                let _ = paintop_ir::to_canonical_bytes(&value);
            }
            let _ = paintop_ir::semantic_hash(&plan);
        }
        Err(_) => {
            // Rejection is the expected outcome for malformed/hostile inputs;
            // the only bug class we hunt here is a panic/abort/OOM, not a
            // particular error code.
        }
    }
});
