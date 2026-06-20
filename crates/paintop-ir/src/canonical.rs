//! Canonical JSON byte emitter (`IR_SPEC` §17 r5, r11; `plan.md` §10.2, §10.3).
//!
//! Hashing, content-addressed caching, replay, and meaningful diffs all require
//! that two *equivalent* plans/manifests serialize to **byte-identical** output.
//! `serde_json::to_string` is close but not sufficient on its own: it leaves the
//! float format up to `ryu` (which is round-trippable but renders integral
//! floats as `8.0` while the central model must pin a single, stable shape), and
//! it offers no guarantee that object keys are emitted in a fixed order
//! independent of the in-memory map type.
//!
//! [`to_canonical_bytes`] / [`to_canonical_string`] walk a [`serde_json::Value`]
//! and emit:
//!
//! 1. **Lexicographically sorted object keys** (`IR_SPEC` §17 r5) — sorted by the
//!    UTF-8 byte sequence of the key, which equals Unicode scalar order for the
//!    well-formed `String` keys `serde_json` produces. The sort is performed here
//!    explicitly so the result does not depend on whether the source map preserves
//!    insertion order.
//! 2. **No insignificant whitespace** — no spaces after `:` or `,`, no newlines.
//! 3. **Stable string escaping** — the minimal escape set required by RFC 8259
//!    (`"`, `\`, and the C0 control range, with the short forms `\b \t \n \f \r`
//!    and `\u00XX` for the rest). All other code points, including non-ASCII, are
//!    emitted verbatim as UTF-8 so the same string always yields the same bytes.
//! 4. **A single round-trippable float format** (`IR_SPEC` §17 r11) — every JSON
//!    `f64` is rendered with `serde_json`'s shortest round-trippable decimal
//!    (its `Number` `Display`, backed by `ryu`), `-0.0` is normalized to `0.0`,
//!    and an integral float (`8.0`) keeps its trailing `.0` so a float never
//!    collides with the integer `8`. Integers that arrived as JSON integers
//!    (`u64`/`i64`) are emitted without a decimal point.
//!
//! Non-finite floats (`NaN`/`Infinity`) cannot occur in a value that came through
//! the strict parser (they are rejected by [`scan_json`](crate::scan::scan_json)),
//! but [`to_canonical_bytes`] guards against them anyway and returns a typed
//! [`parse`](crate::ErrorClass::Parse) error rather than emitting non-JSON.
//!
//! ```
//! use paintop_ir::canonical::to_canonical_string;
//! use serde_json::json;
//!
//! // Keys are sorted and whitespace is stripped, regardless of source order.
//! let a = to_canonical_string(&json!({"b": 1, "a": 2})).unwrap();
//! let b = to_canonical_string(&json!({"a": 2, "b": 1})).unwrap();
//! assert_eq!(a, b);
//! assert_eq!(a, r#"{"a":2,"b":1}"#);
//!
//! // Integral floats keep a single, stable shape distinct from integers.
//! assert_eq!(to_canonical_string(&json!(8.0)).unwrap(), "8.0");
//! assert_eq!(to_canonical_string(&json!(8)).unwrap(), "8");
//! ```

use serde_json::Value;

use crate::error::{Error, ErrorClass, Result};

/// Stable code for a non-finite float encountered while canonicalizing
/// (`NaN`/`Infinity`). Such values are rejected by the strict parser, so this is
/// a defensive guard on values constructed in-memory.
pub const E_NON_FINITE_FLOAT: &str = "E_NON_FINITE_FLOAT";

/// Serialize a [`serde_json::Value`] to its canonical JSON **bytes** (`IR_SPEC` §17).
///
/// The output has lexicographically sorted object keys, no insignificant
/// whitespace, the minimal stable string escaping, and the single round-trippable
/// float format, so equivalent values always produce byte-identical output.
///
/// The returned bytes are valid UTF-8; [`to_canonical_string`] is a thin wrapper
/// that hands back a `String` for callers that want one.
///
/// # Errors
/// Returns a [`parse`](ErrorClass::Parse) / [`E_NON_FINITE_FLOAT`] error if the
/// value (or any nested value) carries a non-finite float (`NaN`/`Infinity`).
/// A value that came through [`parse_plan`](crate::plan::parse_plan) cannot hit
/// this case.
pub fn to_canonical_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    write_value(value, &mut out)?;
    Ok(out)
}

/// Serialize a [`serde_json::Value`] to its canonical JSON **string**.
///
/// See [`to_canonical_bytes`]; this is the same emitter returning a `String`.
///
/// # Errors
/// See [`to_canonical_bytes`].
pub fn to_canonical_string(value: &Value) -> Result<String> {
    let bytes = to_canonical_bytes(value)?;
    // The emitter only ever writes valid UTF-8 (ASCII structure + UTF-8 string
    // payloads), so this conversion cannot fail; surface a typed error rather
    // than `expect` to honor the no-unwrap rule.
    String::from_utf8(bytes).map_err(|err| {
        Error::new(
            ErrorClass::Parse,
            "E_INVALID_UTF8",
            format!("canonical bytes were not valid UTF-8: {err}"),
        )
    })
}

/// Recursively emit one value into `out` in canonical form.
fn write_value(value: &Value, out: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(number) => write_number(number, out)?,
        Value::String(string) => write_string(string, out),
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                write_value(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            // Sort keys lexicographically by their UTF-8 bytes (== Unicode scalar
            // order for well-formed keys), independent of the map's own order.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push(b'{');
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                write_string(key, out);
                out.push(b':');
                // The key came from the map, so the lookup always succeeds.
                if let Some(child) = map.get(key) {
                    write_value(child, out)?;
                }
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// Emit a JSON number in canonical form.
///
/// `serde_json::Number`'s `Display` is already the canonical single
/// representation we want: integers render as a plain decimal (no point) and
/// floats render with `ryu`'s shortest round-trippable decimal *and* a guaranteed
/// decimal point (`8.0`, `0.1`, `1e-300`), so a float never aliases an integer.
/// `serde_json` cannot construct a `Number` holding `NaN`/`Infinity`
/// (`Number::from_f64` rejects them), so the only normalization needed on top is
/// collapsing `-0.0` to `0.0`; the non-finite guard is defensive.
fn write_number(number: &serde_json::Number, out: &mut Vec<u8>) -> Result<()> {
    // Integers (signed or unsigned) are emitted verbatim from their stable
    // decimal `Display`, which never carries a decimal point.
    if number.is_u64() || number.is_i64() {
        out.extend_from_slice(number.to_string().as_bytes());
        return Ok(());
    }
    // Otherwise it is a float. Normalize `-0.0` to `0.0` so the two zeros share a
    // single representation, then emit the shortest round-trippable float text.
    let float = number.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Parse,
            E_NON_FINITE_FLOAT,
            "number is neither a JSON integer nor a representable float",
        )
    })?;
    if !float.is_finite() {
        return Err(Error::new(
            ErrorClass::Parse,
            E_NON_FINITE_FLOAT,
            format!("non-finite float `{float}` cannot be canonicalized"),
        ));
    }
    if float == 0.0 {
        // Covers both `0.0` and `-0.0`; emit the single canonical zero float.
        out.extend_from_slice(b"0.0");
        return Ok(());
    }
    out.extend_from_slice(number.to_string().as_bytes());
    Ok(())
}

/// Append a JSON string literal for `text` with the minimal stable escape set.
///
/// Only the characters RFC 8259 *requires* escaping are escaped: the quote, the
/// backslash, and the C0 control range. The short escapes `\b \t \n \f \r` are
/// used where defined; the remaining controls use `\u00XX`. Every other code
/// point — crucially all non-ASCII — is written verbatim as UTF-8, so the same
/// logical string always produces the same bytes.
fn write_string(text: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for ch in text.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{0008}' => out.extend_from_slice(b"\\b"),
            '\u{0009}' => out.extend_from_slice(b"\\t"),
            '\u{000A}' => out.extend_from_slice(b"\\n"),
            '\u{000C}' => out.extend_from_slice(b"\\f"),
            '\u{000D}' => out.extend_from_slice(b"\\r"),
            c if (c as u32) < 0x20 => {
                // Remaining C0 controls: `\u00XX` with lowercase hex digits. The
                // guard bounds the code point to `0x00..=0x1F`, so it fits in a
                // `u8` and the nibbles are in `0x0..=0xF`.
                let code = c as u8;
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(code >> 4));
                out.push(hex_digit(code & 0xF));
            }
            c => {
                let mut buffer = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
            }
        }
    }
    out.push(b'"');
}

/// Map a nibble (`0x0..=0xF`) to its lowercase ASCII hex digit.
const fn hex_digit(nibble: u8) -> u8 {
    match nibble {
        0..=9 => b'0' + nibble,
        _ => b'a' + (nibble - 10),
    }
}

#[cfg(test)]
mod tests {
    use super::{E_NON_FINITE_FLOAT, to_canonical_bytes, to_canonical_string};
    use serde_json::json;

    #[test]
    fn sorts_object_keys_lexicographically() {
        let value = json!({"z": 1, "a": 2, "m": 3});
        assert_eq!(
            to_canonical_string(&value).unwrap(),
            r#"{"a":2,"m":3,"z":1}"#
        );
    }

    #[test]
    fn key_order_does_not_affect_output() {
        // Two objects equal up to key order canonicalize identically.
        let a = json!({"b": {"d": 1, "c": 2}, "a": [3, 2, 1]});
        let b = json!({"a": [3, 2, 1], "b": {"c": 2, "d": 1}});
        assert_eq!(
            to_canonical_bytes(&a).unwrap(),
            to_canonical_bytes(&b).unwrap()
        );
    }

    #[test]
    fn no_insignificant_whitespace() {
        let value = json!({"a": [1, 2, {"b": 3}], "c": "x"});
        let out = to_canonical_string(&value).unwrap();
        assert_eq!(out, r#"{"a":[1,2,{"b":3}],"c":"x"}"#);
        assert!(!out.contains(' '));
        assert!(!out.contains('\n'));
    }

    #[test]
    fn nested_keys_are_sorted_recursively() {
        let value = json!({"outer": {"y": {"b": 1, "a": 2}, "x": 0}});
        assert_eq!(
            to_canonical_string(&value).unwrap(),
            r#"{"outer":{"x":0,"y":{"a":2,"b":1}}}"#
        );
    }

    #[test]
    fn array_order_is_preserved() {
        // Arrays are sequence-significant; order must NOT be sorted.
        let value = json!([3, 1, 2, "z", "a"]);
        assert_eq!(to_canonical_string(&value).unwrap(), r#"[3,1,2,"z","a"]"#);
    }

    #[test]
    fn integers_have_no_decimal_point() {
        assert_eq!(to_canonical_string(&json!(0)).unwrap(), "0");
        assert_eq!(to_canonical_string(&json!(8)).unwrap(), "8");
        assert_eq!(to_canonical_string(&json!(-17)).unwrap(), "-17");
        assert_eq!(
            to_canonical_string(&json!(12_345_678_901_234_567_890_u64)).unwrap(),
            "12345678901234567890"
        );
    }

    #[test]
    fn integral_floats_keep_a_single_decimal_form() {
        // A float that happens to be integral keeps `.0` so it never collides
        // with the integer of the same magnitude.
        assert_eq!(to_canonical_string(&json!(8.0)).unwrap(), "8.0");
        assert_ne!(
            to_canonical_string(&json!(8.0)).unwrap(),
            to_canonical_string(&json!(8)).unwrap()
        );
    }

    #[test]
    fn negative_zero_float_normalizes_to_zero() {
        let neg_zero = json!(-0.0_f64);
        let pos_zero = json!(0.0_f64);
        assert_eq!(to_canonical_string(&neg_zero).unwrap(), "0.0");
        assert_eq!(
            to_canonical_bytes(&neg_zero).unwrap(),
            to_canonical_bytes(&pos_zero).unwrap()
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "round-tripping a shortest-decimal float must reproduce the exact bit pattern, so bit-exact equality is the property under test"
    )]
    fn floats_round_trip_through_canonical_text() {
        for raw in [
            0.1_f64,
            8.5,
            -3.25,
            1e-300,
            2.225_073_858_507_201_4e-308,
            123.456,
        ] {
            let text = to_canonical_string(&json!(raw)).unwrap();
            let parsed: f64 = text.parse().unwrap();
            assert_eq!(parsed, raw, "`{text}` did not round-trip to {raw}");
        }
    }

    #[test]
    fn single_float_format_is_deterministic() {
        // The same float must always render to the same bytes.
        let a = to_canonical_bytes(&json!(0.1_f64)).unwrap();
        let b = to_canonical_bytes(&json!(0.1_f64)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn strings_escape_only_the_required_characters() {
        // Short escapes are used where defined; `\u{0001}` (no short form) uses
        // the lowercase `\u00XX` long form.
        let value = json!("a\"b\\c\n\t\r\u{0008}\u{000C}\u{0001}");
        let expected = "\"a\\\"b\\\\c\\n\\t\\r\\b\\f\\u0001\"";
        assert_eq!(to_canonical_string(&value).unwrap(), expected);
    }

    #[test]
    fn non_ascii_is_emitted_verbatim_as_utf8() {
        // No `\u` escaping of ordinary Unicode: the bytes are the UTF-8 of the
        // source so the same string always yields the same canonical bytes.
        let value = json!("héllo — 世界 🎨");
        let bytes = to_canonical_bytes(&value).unwrap();
        let expected = format!("\"{}\"", "héllo — 世界 🎨");
        assert_eq!(bytes, expected.as_bytes());
    }

    #[test]
    fn null_and_bools() {
        assert_eq!(to_canonical_string(&json!(null)).unwrap(), "null");
        assert_eq!(to_canonical_string(&json!(true)).unwrap(), "true");
        assert_eq!(to_canonical_string(&json!(false)).unwrap(), "false");
    }

    #[test]
    fn empty_containers() {
        assert_eq!(to_canonical_string(&json!({})).unwrap(), "{}");
        assert_eq!(to_canonical_string(&json!([])).unwrap(), "[]");
    }

    #[test]
    fn output_is_valid_json_and_reparses_equal() {
        let value = json!({
            "paintop": "1.0",
            "nodes": [{"id": "n", "params": {"sigma": 8.0, "count": 3}}],
            "tag": "α/β"
        });
        let text = to_canonical_string(&value).unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(reparsed, value);
    }

    #[test]
    fn equivalent_plans_produce_byte_identical_output() {
        // The bone's exit criterion: two equivalent documents differing only in
        // key order and whitespace canonicalize to identical bytes.
        let pretty = r#"{
            "b": { "two": 2, "one": 1 },
            "a": [ 1, 2, 3 ]
        }"#;
        let compact_other_order = r#"{"a":[1,2,3],"b":{"one":1,"two":2}}"#;
        let v1: serde_json::Value = serde_json::from_str(pretty).unwrap();
        let v2: serde_json::Value = serde_json::from_str(compact_other_order).unwrap();
        assert_eq!(
            to_canonical_bytes(&v1).unwrap(),
            to_canonical_bytes(&v2).unwrap()
        );
    }

    #[test]
    fn serde_json_cannot_construct_a_non_finite_number() {
        // The non-finite guard in `write_number` is defensive: `serde_json`
        // refuses to build a `Number` from `NaN`/`Infinity`, so a `Value` reaching
        // the emitter can never hold one. This documents that invariant.
        assert!(serde_json::Number::from_f64(f64::NAN).is_none());
        assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
        assert!(serde_json::Number::from_f64(f64::NEG_INFINITY).is_none());
        // The stable code constant exists for the guarded branch and downstream
        // taxonomy use.
        assert_eq!(E_NON_FINITE_FLOAT, "E_NON_FINITE_FLOAT");
    }
}
