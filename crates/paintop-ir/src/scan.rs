//! Pre-deserialization JSON scanning: duplicate-key and invalid-number
//! rejection (`IR_SPEC` §10, §17 r11/r12; `plan.md` §10.1 phase 1).
//!
//! `serde_json` is permissive in two ways that the plan parser must not be:
//!
//! 1. **Duplicate object keys are silently last-wins.** `{"sigma": 1, "sigma": 2}`
//!    deserializes to `2` with no error, so a typo or an adversarial plan can
//!    silently change the canonical bytes (and therefore the semantic hash) of a
//!    plan. `IR_SPEC` §17 r12 requires duplicate keys be *rejected before
//!    canonicalization*; this module is that rejection.
//! 2. **Some non-round-trippable numbers are silently accepted.** A magnitude
//!    that underflows the `f64` range (`1e-400`) is silently coerced to `0.0`,
//!    losing the authored value. `IR_SPEC` §10 / §17 r11 require a single
//!    round-trippable float representation, so such tokens are rejected. (`NaN`,
//!    `Infinity`, and overflowing magnitudes are already rejected by
//!    `serde_json`'s tokenizer; this module reclassifies that failure onto the
//!    central taxonomy with a stable code.)
//!
//! [`scan_json`] performs a single recursive walk *before* the [`Plan`](crate::Plan)
//! structs are built, so an adversarial document fails fast — with a stable
//! [`parse`](crate::ErrorClass::Parse) error and a JSON-pointer path locating the
//! offending value — rather than allocating the full typed model first. The
//! [`parse_plan`](crate::plan::parse_plan) front door runs this scan ahead of
//! `serde` deserialization.
//!
//! ```
//! use paintop_ir::scan::scan_json;
//!
//! // Duplicate keys are rejected (serde_json would keep the last silently).
//! let err = scan_json(r#"{"sigma": 1, "sigma": 2}"#).unwrap_err();
//! assert_eq!(err.code, "E_DUPLICATE_KEY");
//!
//! // A well-formed document passes the scan untouched.
//! scan_json(r#"{"nodes": [{"id": "n", "params": {"sigma": 8.0}}]}"#)
//!     .expect("clean document scans");
//! ```

use std::collections::BTreeSet;
use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;

use crate::error::{Error, ErrorClass, ErrorContext, Result};

/// Stable code for a JSON object carrying the same key twice (`IR_SPEC` §17 r12).
pub const E_DUPLICATE_KEY: &str = "E_DUPLICATE_KEY";
/// Stable code for a numeric token that is not finite or not round-trippable
/// (`NaN`, `Infinity`, an over/underflowing magnitude; `IR_SPEC` §10).
pub const E_INVALID_NUMBER: &str = "E_INVALID_NUMBER";
/// Stable code for syntactically malformed JSON surfaced by the scan pass.
pub const E_INVALID_JSON: &str = "E_INVALID_JSON";

/// Scan raw plan JSON for duplicate object keys and invalid numeric forms,
/// failing before the typed model is allocated (`plan.md` §10.1 phase 1).
///
/// On success the input is left untouched for the subsequent `serde`
/// deserialization; this function neither builds nor returns the plan.
///
/// # Errors
/// - [`parse`](ErrorClass::Parse) / [`E_DUPLICATE_KEY`] if any object carries the
///   same key twice. The error's [`path`](ErrorContext::path) is a JSON pointer
///   to the offending object and its [`actual`](ErrorContext::actual) is the
///   repeated key.
/// - [`parse`](ErrorClass::Parse) / [`E_INVALID_NUMBER`] for `NaN`, `Infinity`,
///   or a magnitude that does not round-trip through `f64` (over/underflow).
/// - [`parse`](ErrorClass::Parse) / [`E_INVALID_JSON`] for syntactically invalid
///   JSON (this scan runs first, so it owns the syntax error here).
pub fn scan_json(json: &str) -> Result<()> {
    let mut deserializer = serde_json::Deserializer::from_str(json);
    let visitor = NodeVisitor {
        pointer: String::new(),
    };
    let outcome = deserializer.deserialize_any(visitor);
    match outcome {
        Ok(()) => deserializer
            .end()
            .map_err(|err| map_syntax_error(&err, json)),
        // A `Reject` smuggled out through `serde`'s `custom` error carries our
        // already-classified failure; a plain `serde` error is a syntax failure.
        Err(err) => Err(Reject::recover(&err).unwrap_or_else(|| map_syntax_error(&err, json))),
    }
}

/// A classified scan failure, round-tripped through `serde`'s `Error::custom`.
///
/// `serde`'s visitor callbacks can only surface failures as the deserializer's
/// own error type, so a structured rejection is encoded into the `custom`
/// message with a private sentinel prefix and recovered by [`Reject::recover`].
/// The sentinel is an internal implementation detail and never reaches a caller.
struct Reject {
    code: &'static str,
    message: String,
    pointer: String,
    actual: Option<String>,
}

/// Internal sentinel marking a scan rejection encoded inside a `serde` custom
/// error message. Chosen to be improbable in genuine `serde` diagnostics.
const SENTINEL: &str = "\u{1}paintop-scan\u{1}";
/// Field separator inside an encoded [`Reject`].
const SEP: &str = "\u{1f}";

impl Reject {
    fn new(
        code: &'static str,
        message: impl Into<String>,
        pointer: impl Into<String>,
        actual: Option<String>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            pointer: pointer.into(),
            actual,
        }
    }

    /// Encode this rejection into a `serde` custom error message.
    fn encode(&self) -> String {
        format!(
            "{SENTINEL}{code}{SEP}{pointer}{SEP}{actual}{SEP}{message}",
            code = self.code,
            pointer = self.pointer,
            actual = self.actual.as_deref().unwrap_or(""),
            message = self.message,
        )
    }

    /// Recover a classified [`Error`] from a `serde` error, if it was one of ours.
    fn recover(err: &serde_json::Error) -> Option<Error> {
        let rendered = err.to_string();
        let payload = rendered.split(SENTINEL).nth(1)?;
        // `serde_json` appends `" at line L column C"`; trim it back off so the
        // encoded fields are recovered cleanly.
        let payload = payload.split(" at line ").next().unwrap_or(payload);
        let mut parts = payload.splitn(4, SEP);
        let code = parts.next()?;
        let pointer = parts.next().unwrap_or("");
        let actual = parts.next().unwrap_or("");
        let message = parts.next().unwrap_or("");
        let code = match code {
            E_DUPLICATE_KEY => E_DUPLICATE_KEY,
            E_INVALID_NUMBER => E_INVALID_NUMBER,
            _ => E_INVALID_JSON,
        };
        let mut context = ErrorContext::default();
        if !pointer.is_empty() {
            context = context.with_path(pointer.to_owned());
        }
        if !actual.is_empty() {
            context = context.with_actual(actual.to_owned());
        }
        Some(Error::new(ErrorClass::Parse, code, message.to_owned()).with_context(context))
    }

    /// Raise this rejection as the deserializer's error type.
    fn raise<E: de::Error>(&self) -> E {
        E::custom(self.encode())
    }
}

/// Map a genuine `serde_json` syntax/number error onto the central taxonomy.
///
/// `serde_json` rejects `NaN`, `Infinity`, and over-range magnitudes at the
/// tokenizer before any visitor runs; those surface here. The bare `NaN` /
/// `Infinity` / `-Infinity` literals are reported only as a generic
/// "expected value", so the offending token is recovered from the source at the
/// error's column to distinguish an invalid *number* from arbitrary bad syntax.
/// Such number failures are reclassified to [`E_INVALID_NUMBER`]; everything
/// else is a generic JSON syntax failure.
fn map_syntax_error(err: &serde_json::Error, source: &str) -> Error {
    let message = err.to_string();
    let line = err.line();
    let column = err.column();
    let lowered = message.to_ascii_lowercase();
    let is_numberish_message =
        lowered.contains("number") || lowered.contains("nan") || lowered.contains("infinit");
    let code = if is_numberish_message || token_at_looks_like_number(source, line, column) {
        E_INVALID_NUMBER
    } else {
        E_INVALID_JSON
    };
    let mut context = ErrorContext::default();
    if line > 0 {
        context = context.with_path(format!("line {line}, column {column}"));
    }
    Error::new(ErrorClass::Parse, code, message).with_context(context)
}

/// Whether the token at `serde_json`'s reported `(line, column)` in `source`
/// begins one of the rejected non-finite literals (`NaN`, `Infinity`,
/// `-Infinity`). `serde_json` reports these only as "expected value", so the
/// source is consulted to reclassify them as invalid *numbers*.
fn token_at_looks_like_number(source: &str, line: usize, column: usize) -> bool {
    if line == 0 || column == 0 {
        return false;
    }
    let Some(line_text) = source.lines().nth(line - 1) else {
        return false;
    };
    // `serde_json` columns are 1-based and point at (just past) the offending
    // byte; clamp into range and look at the token starting there.
    let start = column.saturating_sub(1).min(line_text.len());
    let rest = line_text[start..].trim_start();
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    rest.starts_with("NaN") || rest.starts_with("Infinity")
}

/// Recursive scanner over one JSON value at JSON-pointer `pointer`.
///
/// Implemented as a `serde` [`Visitor`] driven by `deserialize_any` so the walk
/// streams over the original token stream once without materializing a
/// `serde_json::Value` tree. Object keys are tracked in a [`BTreeSet`] to reject
/// duplicates; scalar values are captured as [`RawValue`] so number leaves can
/// be validated against their *source token* (round-trip), not just the parsed
/// `f64`.
struct NodeVisitor {
    pointer: String,
}

/// Seed that captures the next value as a borrowed [`RawValue`] (its source
/// text) and recursively scans it under the child `pointer`.
struct ChildSeed {
    pointer: String,
}

impl<'de> de::DeserializeSeed<'de> for ChildSeed {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        let raw = <&RawValue>::deserialize(deserializer)?;
        scan_raw(raw.get(), &self.pointer).map_err(|reject| reject.raise())
    }
}

impl<'de> Visitor<'de> for NodeVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, _v: bool) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _v: i64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _v: u64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<(), E> {
        // `serde_json` rejects non-finite tokens before this point, but guard
        // anyway so the contract holds regardless of upstream behavior.
        if value.is_finite() {
            Ok(())
        } else {
            Err(Reject::new(
                E_INVALID_NUMBER,
                "non-finite number is not permitted",
                &self.pointer,
                Some(value.to_string()),
            )
            .raise())
        }
    }

    fn visit_str<E: de::Error>(self, _v: &str) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_none<E: de::Error>(self) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_some<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        let mut index = 0usize;
        loop {
            let child = ChildSeed {
                pointer: format!("{}/{index}", self.pointer),
            };
            if seq.next_element_seed(child)?.is_none() {
                break;
            }
            index += 1;
        }
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<(), A::Error> {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(Reject::new(
                    E_DUPLICATE_KEY,
                    format!("duplicate object key `{key}`"),
                    &self.pointer,
                    Some(key),
                )
                .raise());
            }
            let child = ChildSeed {
                pointer: format!("{}/{}", self.pointer, escape_pointer_token(&key)),
            };
            map.next_value_seed(child)?;
        }
        Ok(())
    }
}

/// Recursively scan the source text of one JSON value at `pointer`.
///
/// Containers are re-driven through the streaming [`NodeVisitor`]; number leaves
/// are validated against their raw token so non-round-trippable magnitudes are
/// caught. Returns a [`Reject`] (never a bare `serde` error) so the caller can
/// surface a classified failure.
fn scan_raw(text: &str, pointer: &str) -> std::result::Result<(), Reject> {
    let trimmed = text.trim_start();
    match trimmed.as_bytes().first() {
        Some(b'{' | b'[') => {
            let mut deserializer = serde_json::Deserializer::from_str(text);
            let visitor = NodeVisitor {
                pointer: pointer.to_owned(),
            };
            deserializer
                .deserialize_any(visitor)
                .and_then(|()| deserializer.end())
                .map_err(|err| {
                    Reject::recover(&err).map_or_else(
                        || Reject::new(E_INVALID_JSON, err.to_string(), pointer, None),
                        |error| {
                            let code = classify_code(&error);
                            let ErrorContext { path, actual, .. } = *error.context;
                            Reject::new(
                                code,
                                error.message,
                                path.unwrap_or_else(|| pointer.to_owned()),
                                actual,
                            )
                        },
                    )
                })
        }
        Some(c) if c.is_ascii_digit() || *c == b'-' => check_number(trimmed.trim_end(), pointer),
        _ => Ok(()),
    }
}

/// Map a recovered [`Error`]'s code string back onto one of the static scan codes.
const fn classify_code(error: &Error) -> &'static str {
    // The recovered error always carries one of our three codes; match on the
    // first byte that distinguishes them to avoid allocating.
    match error.code.as_bytes() {
        [b'E', b'_', b'D', ..] => E_DUPLICATE_KEY,
        [
            b'E',
            b'_',
            b'I',
            b'N',
            b'V',
            b'A',
            b'L',
            b'I',
            b'D',
            b'_',
            b'N',
            ..,
        ] => E_INVALID_NUMBER,
        _ => E_INVALID_JSON,
    }
}

/// Validate a JSON number *token* for round-trippability (`IR_SPEC` §10 / §17 r11).
///
/// `serde_json` already rejects `NaN`, `Infinity`, and over-range magnitudes at
/// the tokenizer, so the only silently-lossy case reaching here is a magnitude
/// that underflows `f64` to zero (`1e-400`). Such a token parses to `0.0` while
/// its source text is not `0`, so the authored value would be lost: reject it.
fn check_number(token: &str, pointer: &str) -> std::result::Result<(), Reject> {
    // Integer tokens (no `.`, `e`, or `E`) never underflow; nothing to check.
    let is_float = token.bytes().any(|b| b == b'.' || b == b'e' || b == b'E');
    if !is_float {
        return Ok(());
    }
    let Ok(value) = token.parse::<f64>() else {
        // Unparseable as f64 (should not happen for a serde-accepted token).
        return Ok(());
    };
    if !value.is_finite() {
        return Err(Reject::new(
            E_INVALID_NUMBER,
            "non-finite number is not permitted",
            pointer,
            Some(token.to_owned()),
        ));
    }
    if value == 0.0 && !is_zero_token(token) {
        return Err(Reject::new(
            E_INVALID_NUMBER,
            format!("number `{token}` underflows to zero and is not round-trippable"),
            pointer,
            Some(token.to_owned()),
        ));
    }
    Ok(())
}

/// Whether a numeric token genuinely denotes zero (so a parsed `0.0` is not an
/// underflow). Covers `0`, `-0`, `0.0`, `0e9`, `-0.0E-3`, etc.
fn is_zero_token(token: &str) -> bool {
    let body = token.strip_prefix('-').unwrap_or(token);
    // Strip an exponent; a zero mantissa is zero regardless of exponent.
    let mantissa = body.split(['e', 'E']).next().unwrap_or(body);
    mantissa.bytes().all(|b| b == b'0' || b == b'.')
}

/// Escape a map key for use as a JSON-pointer reference token (RFC 6901):
/// `~` -> `~0`, `/` -> `~1`.
fn escape_pointer_token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::{E_DUPLICATE_KEY, E_INVALID_JSON, E_INVALID_NUMBER, scan_json};
    use crate::error::ErrorClass;

    #[test]
    fn clean_document_scans_ok() {
        scan_json(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "n", "op": "filter.gaussian_blur@1", "params": {"sigma": 8.0}}],
                "exports": {}
            }"#,
        )
        .expect("clean document must scan");
    }

    #[test]
    fn top_level_duplicate_key_is_parse_error() {
        let err = scan_json(r#"{"a": 1, "a": 2}"#).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_DUPLICATE_KEY);
        assert_eq!(err.context.actual.as_deref(), Some("a"));
    }

    #[test]
    fn nested_object_duplicate_key_is_rejected_with_pointer() {
        let err = scan_json(r#"{"params": {"sigma": 1, "sigma": 2}}"#).unwrap_err();
        assert_eq!(err.code, E_DUPLICATE_KEY);
        assert_eq!(err.context.actual.as_deref(), Some("sigma"));
        assert_eq!(err.context.path.as_deref(), Some("/params"));
    }

    #[test]
    fn duplicate_key_inside_array_element_is_rejected() {
        let err = scan_json(r#"{"nodes": [{"id": "a", "id": "b"}]}"#).unwrap_err();
        assert_eq!(err.code, E_DUPLICATE_KEY);
        assert_eq!(err.context.actual.as_deref(), Some("id"));
        assert_eq!(err.context.path.as_deref(), Some("/nodes/0"));
    }

    #[test]
    fn duplicate_key_does_not_silently_keep_last() {
        // The whole point of §17 r12: serde_json would accept this as `{"x": 2}`.
        let value: serde_json::Value =
            serde_json::from_str(r#"{"x": 1, "x": 2}"#).expect("serde accepts last-wins");
        assert_eq!(value["x"], serde_json::json!(2));
        // Our scan must reject it instead.
        assert_eq!(
            scan_json(r#"{"x": 1, "x": 2}"#).unwrap_err().code,
            E_DUPLICATE_KEY
        );
    }

    #[test]
    fn nan_literal_is_invalid_number() {
        let err = scan_json(r#"{"sigma": NaN}"#).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_NUMBER);
    }

    #[test]
    fn infinity_literal_is_invalid_number() {
        let err = scan_json(r#"{"sigma": Infinity}"#).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_NUMBER);
    }

    #[test]
    fn negative_infinity_literal_is_invalid_number() {
        let err = scan_json(r#"{"sigma": -Infinity}"#).unwrap_err();
        assert_eq!(err.code, E_INVALID_NUMBER);
    }

    #[test]
    fn overflowing_magnitude_is_invalid_number() {
        let err = scan_json(r#"{"sigma": 1e400}"#).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_NUMBER);
    }

    #[test]
    fn underflowing_magnitude_is_not_round_trippable() {
        let err = scan_json(r#"{"sigma": 1e-400}"#).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_NUMBER);
        assert_eq!(err.context.actual.as_deref(), Some("1e-400"));
    }

    #[test]
    fn nested_underflow_is_rejected_with_pointer() {
        let err = scan_json(r#"{"a": [0, {"b": 2.5e-310000}]}"#).unwrap_err();
        assert_eq!(err.code, E_INVALID_NUMBER);
        assert_eq!(err.context.path.as_deref(), Some("/a/1/b"));
    }

    #[test]
    fn legitimate_zero_forms_are_accepted() {
        for token in ["0", "-0", "0.0", "0e9", "-0.0E-3", "0.0e0"] {
            scan_json(&format!(r#"{{"z": {token}}}"#))
                .unwrap_or_else(|e| panic!("zero token {token} should scan: {e}"));
        }
    }

    #[test]
    fn small_but_representable_floats_are_accepted() {
        for token in ["1e-300", "0.1", "2.2250738585072014e-308"] {
            scan_json(&format!(r#"{{"v": {token}}}"#))
                .unwrap_or_else(|e| panic!("representable {token} should scan: {e}"));
        }
    }

    #[test]
    fn large_integer_token_is_accepted() {
        scan_json(r#"{"n": 12345678901234567890}"#).expect("u64-range integer scans");
    }

    #[test]
    fn malformed_json_is_invalid_json() {
        let err = scan_json("{ not json").unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_JSON);
    }

    #[test]
    fn trailing_garbage_after_value_is_rejected() {
        let err = scan_json(r#"{"a": 1} trailing"#).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_JSON);
    }

    #[test]
    fn pointer_tokens_escape_slash_and_tilde() {
        // A key containing `/` and `~` must be escaped per RFC 6901 in the path.
        let err = scan_json(r#"{"a/b~c": {"k": 1, "k": 2}}"#).unwrap_err();
        assert_eq!(err.code, E_DUPLICATE_KEY);
        assert_eq!(err.context.path.as_deref(), Some("/a~1b~0c"));
    }

    #[test]
    fn array_of_scalars_with_no_dups_scans() {
        scan_json(r#"{"xs": [1, 2, 3, "a", true, null, 4.5]}"#).expect("scalar array scans");
    }
}
