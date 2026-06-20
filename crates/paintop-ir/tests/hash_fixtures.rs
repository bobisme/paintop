//! Golden hash fixtures for the semantic / content hash API (`bn-1m3`;
//! `M0_DECISIONS` D; `plan.md §10.3`, §17).
//!
//! These fixtures are the bone's exit gate, made executable:
//!
//! 1. **Stable hashes.** A fixed set of `(domain, plan)` inputs hashes to the
//!    exact `blake3:<hex>` ids recorded in `fixtures/hashes/golden.json`. If the
//!    hashing, canonicalization, or domain-separation framing ever drifts, these
//!    bytes change and the test fails — a regression guard against silently
//!    invalidating every on-disk hash.
//! 2. **Changed semantic fields change the hash.** The fixture set deliberately
//!    pairs near-identical plans (e.g. `sigma: 8.0` vs `sigma: 8.5`); their
//!    golden ids differ, proving a real semantic edit moves the hash.
//! 3. **No wall-clock / provenance.** The goldens were recorded once and never
//!    move: recomputing them on any machine, at any time, reproduces them
//!    exactly, because the API mixes in nothing but the domain label and the
//!    canonical bytes.
//!
//! To regenerate after an *intentional* change, run with
//! `UPDATE_HASH_FIXTURES=1` and review the diff.

use std::collections::BTreeMap;

use paintop_ir::{HashDomain, SemanticHash, hash_value};
use serde_json::{Value, json};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/hashes/golden.json");

/// The named, ordered fixture inputs. Each entry pairs a stable case name with a
/// domain and the JSON value to canonicalize-and-hash. Near-duplicate cases
/// exist on purpose so the goldens prove semantic sensitivity.
fn cases() -> Vec<(&'static str, HashDomain, Value)> {
    vec![
        (
            "empty-plan",
            HashDomain::Plan,
            json!({"paintop": "1.0", "inputs": {}, "nodes": [], "exports": {}}),
        ),
        (
            "blur-sigma-8_0",
            HashDomain::Plan,
            json!({"paintop": "1.0", "nodes": [{"op": "filter.gaussian_blur@1", "params": {"sigma": 8.0}}]}),
        ),
        (
            // Same shape, one changed semantic field: must differ from above.
            "blur-sigma-8_5",
            HashDomain::Plan,
            json!({"paintop": "1.0", "nodes": [{"op": "filter.gaussian_blur@1", "params": {"sigma": 8.5}}]}),
        ),
        (
            // Integer vs. integral float: a semantic distinction the canonical
            // emitter preserves, so the hash must differ from `blur-sigma-8_0`.
            "blur-sigma-8-int",
            HashDomain::Plan,
            json!({"paintop": "1.0", "nodes": [{"op": "filter.gaussian_blur@1", "params": {"sigma": 8}}]}),
        ),
        (
            "resource-rgba8",
            HashDomain::Resource,
            json!({"kind": "image", "extent": [256, 256], "encoding": "linear-srgb"}),
        ),
        (
            // Identical bytes to `resource-rgba8` would be hashed under a
            // different domain here; this content case must not collide.
            "content-rgba8",
            HashDomain::Content,
            json!({"kind": "image", "extent": [256, 256], "encoding": "linear-srgb"}),
        ),
        (
            "manifest-blur-v1",
            HashDomain::Manifest,
            json!({"op": "filter.gaussian_blur", "version": 1, "params": {"sigma": "f32"}}),
        ),
        (
            "cache-entry-blur",
            HashDomain::CacheEntry,
            json!({
                "op": "filter.gaussian_blur@1",
                "params": {"sigma": 8.0},
                "inputs": ["blake3:00", "blake3:11"],
                "seed": 7,
                "backend": "cpu-reference@1"
            }),
        ),
    ]
}

/// Read the recorded goldens, or an empty map if regenerating from scratch.
fn load_golden() -> BTreeMap<String, String> {
    std::fs::read_to_string(GOLDEN).map_or_else(
        |_| BTreeMap::new(),
        |text| serde_json::from_str(&text).expect("golden.json must be a string->string map"),
    )
}

#[test]
fn golden_hashes_are_stable_and_well_formed() {
    let updating = std::env::var_os("UPDATE_HASH_FIXTURES").is_some();
    let golden = load_golden();
    let mut produced: BTreeMap<String, String> = BTreeMap::new();

    for (name, domain, value) in cases() {
        let hash = hash_value(domain, &value).expect("fixture value canonicalizes and hashes");
        let text = hash.to_string();

        // Self-describing, prefixed, lowercase-hex, and round-trippable.
        assert!(
            text.starts_with("blake3:"),
            "{name}: hash `{text}` lacks the algorithm prefix"
        );
        let reparsed = SemanticHash::parse(&text).expect("emitted hash must re-parse");
        assert_eq!(
            reparsed, hash,
            "{name}: hash did not round-trip through parse"
        );

        if !updating {
            let expected = golden.get(name).unwrap_or_else(|| {
                panic!("{name}: no golden recorded (run with UPDATE_HASH_FIXTURES=1)")
            });
            assert_eq!(
                &text, expected,
                "{name}: hash drifted from golden; if intentional, regenerate with UPDATE_HASH_FIXTURES=1"
            );
        }

        produced.insert(name.to_owned(), text);
    }

    if updating {
        let serialized = serde_json::to_string_pretty(&produced).expect("serialize goldens");
        if let Some(parent) = std::path::Path::new(GOLDEN).parent() {
            std::fs::create_dir_all(parent).expect("create fixtures/hashes dir");
        }
        std::fs::write(GOLDEN, format!("{serialized}\n")).expect("write golden.json");
    }
}

#[test]
fn changed_semantic_field_changes_the_golden() {
    // The exit criterion, read straight off the goldens: the three blur cases
    // differ only in the `sigma` value's semantics, yet each has a distinct id.
    let golden = load_golden();
    if golden.is_empty() {
        // Fresh tree before the first regen; the stability test owns generation.
        return;
    }
    let f = |k: &str| golden.get(k).cloned().unwrap_or_default();
    let v80 = f("blur-sigma-8_0");
    let v85 = f("blur-sigma-8_5");
    let vint = f("blur-sigma-8-int");
    assert_ne!(v80, v85, "sigma 8.0 vs 8.5 must hash differently");
    assert_ne!(v80, vint, "float 8.0 vs integer 8 must hash differently");
    // And cross-domain identical bytes must not collide.
    assert_ne!(
        f("resource-rgba8"),
        f("content-rgba8"),
        "identical bytes in Resource vs Content domains must not collide"
    );
}
