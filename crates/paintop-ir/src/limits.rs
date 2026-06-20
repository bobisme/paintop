//! Pre-deserialization resource limits: plan byte size, nesting depth, node
//! count, and inline payload caps (`plan.md` §10.1 phase 1, §17.1).
//!
//! `plan.md` §10.1 phase 1 mandates that the parser "apply strict size/depth
//! limits *before* allocating large structures". A hostile or accidentally huge
//! plan must fail predictably and cheaply rather than driving the typed model
//! builder to exhaust memory. §17.1 enumerates the budgets a load must enforce:
//! maximum graph nodes, maximum parameter nesting, and maximum inline payload
//! sizes among them.
//!
//! This module enforces the four cheapest-to-check, allocation-independent of
//! those budgets, in a single streaming pass over the raw token stream:
//!
//! 1. **Byte size.** `{json}.len()` is checked first, before anything is walked,
//!    so a multi-gigabyte blob is rejected in O(1).
//! 2. **Nesting depth.** Every `{`/`[` increments a depth counter; exceeding the
//!    cap rejects before a deeply-recursive structure is materialized (this also
//!    bounds the recursion of the later typed parse).
//! 3. **Node count.** The number of elements in the top-level `nodes` array is
//!    capped, mirroring the `policy.resources.max_nodes` budget (`IR_SPEC` §15);
//!    a plan with more nodes than the hard ceiling is rejected before the
//!    `Vec<Node>` is grown.
//! 4. **Inline payload.** Any single inline array (e.g. a `paint.gaussian_splats`
//!    `splats` batch, `IR_SPEC` §6) or string longer than the cap is rejected,
//!    bounding the largest single allocation a node's `params` can request.
//!
//! These are *hard structural ceilings*, deliberately generous and distinct from
//! the per-plan `policy.resources` budgets enforced later (`plan.md` §10.1 phase
//! 5): policy can only tighten, never exceed, these. The walk runs *before* the
//! [`Plan`](crate::Plan) structs are built, so an oversized document fails fast
//! with a stable [`policy`](crate::ErrorClass::Policy) error and a JSON-pointer
//! path locating the offending value.
//!
//! ```
//! use paintop_ir::limits::{check_limits, PlanLimits};
//!
//! // A well-formed, small document passes.
//! check_limits(r#"{"nodes": [{"id": "n"}]}"#, &PlanLimits::DEFAULT)
//!     .expect("small plan is within limits");
//!
//! // A document nested past the depth ceiling is rejected.
//! let limits = PlanLimits { max_depth: 3, ..PlanLimits::DEFAULT };
//! let err = check_limits("[[[[1]]]]", &limits).unwrap_err();
//! assert_eq!(err.code, "E_MAX_DEPTH");
//! ```

use std::fmt;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;

use crate::error::{Error, ErrorClass, ErrorContext, Result};

/// Stable code for a plan whose raw byte length exceeds [`PlanLimits::max_bytes`].
pub const E_MAX_PLAN_BYTES: &str = "E_MAX_PLAN_BYTES";
/// Stable code for a plan nested deeper than [`PlanLimits::max_depth`].
pub const E_MAX_DEPTH: &str = "E_MAX_DEPTH";
/// Stable code for a plan whose `nodes` array exceeds [`PlanLimits::max_nodes`].
pub const E_MAX_NODES: &str = "E_MAX_NODES";
/// Stable code for an inline array or string exceeding [`PlanLimits::max_inline_len`].
pub const E_MAX_INLINE_PAYLOAD: &str = "E_MAX_INLINE_PAYLOAD";

/// The hard structural ceilings applied to a plan before its typed model is
/// allocated (`plan.md` §10.1 phase 1, §17.1).
///
/// These are fixed runtime safety limits, intentionally far larger than any
/// realistic hand- or agent-authored plan; the per-plan `policy.resources`
/// budgets (`IR_SPEC` §15) are validated later and may only tighten them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanLimits {
    /// Maximum raw plan length in bytes. Checked before any walk (O(1)).
    pub max_bytes: usize,
    /// Maximum JSON container nesting depth (`{`/`[` levels).
    pub max_depth: usize,
    /// Maximum number of elements in the top-level `nodes` array.
    pub max_nodes: usize,
    /// Maximum length of any single inline array (element count) or string
    /// (byte length) carried anywhere in the plan.
    pub max_inline_len: usize,
}

impl PlanLimits {
    /// The default hard ceilings applied by [`parse_plan`](crate::plan::parse_plan).
    ///
    /// - 64 MiB of raw plan bytes,
    /// - 64 levels of nesting,
    /// - 100 000 nodes (an order of magnitude above the §15 `max_nodes` example),
    /// - 10 000 000 inline array elements / string bytes (a single large splat
    ///   batch fits; an adversarial unbounded one does not).
    pub const DEFAULT: Self = Self {
        max_bytes: 64 * 1024 * 1024,
        max_depth: 64,
        max_nodes: 100_000,
        max_inline_len: 10_000_000,
    };
}

impl Default for PlanLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Enforce the structural ceilings in `limits` over raw plan JSON, failing
/// before the typed model is allocated (`plan.md` §10.1 phase 1).
///
/// The byte-size ceiling is checked first (O(1)); the remaining ceilings are
/// enforced during a single streaming walk that materializes no
/// `serde_json::Value` tree. On success the input is left untouched for the
/// subsequent scan and `serde` deserialization.
///
/// # Errors
/// - [`policy`](ErrorClass::Policy) / [`E_MAX_PLAN_BYTES`] if the input is longer
///   than [`PlanLimits::max_bytes`].
/// - [`policy`](ErrorClass::Policy) / [`E_MAX_DEPTH`] if container nesting exceeds
///   [`PlanLimits::max_depth`].
/// - [`policy`](ErrorClass::Policy) / [`E_MAX_NODES`] if the top-level `nodes`
///   array holds more than [`PlanLimits::max_nodes`] elements.
/// - [`policy`](ErrorClass::Policy) / [`E_MAX_INLINE_PAYLOAD`] if any inline array
///   or string exceeds [`PlanLimits::max_inline_len`].
///
/// Syntactically invalid JSON is *not* this function's concern: it is reported by
/// the [`scan`](crate::scan) pass that owns parse-error classification, so any
/// `serde` syntax failure here is swallowed (returns `Ok`) and surfaced there.
pub fn check_limits(json: &str, limits: &PlanLimits) -> Result<()> {
    if json.len() > limits.max_bytes {
        return Err(Error::new(
            ErrorClass::Policy,
            E_MAX_PLAN_BYTES,
            format!(
                "plan is {} bytes, exceeding the {}-byte limit",
                json.len(),
                limits.max_bytes
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(json.len().to_string())
                .with_expected(format!("<= {}", limits.max_bytes)),
        ));
    }

    let mut deserializer = serde_json::Deserializer::from_str(json);
    let visitor = LimitVisitor {
        pointer: String::new(),
        depth: 1,
        limits,
        location: Location::Root,
    };
    match deserializer.deserialize_any(visitor) {
        Ok(()) => Ok(()),
        // A `Reject` smuggled through `serde`'s `custom` error is a real limit
        // failure; a plain `serde` syntax error is left for the scan pass to
        // classify, so it is swallowed here.
        Err(err) => Reject::recover(&err).map_or(Ok(()), Err),
    }
}

/// Where in the plan a value sits, so the `nodes` array can be singled out for
/// the node-count ceiling without tracking the full pointer semantically.
#[derive(Debug, Clone, Copy)]
enum Location {
    /// The document root value.
    Root,
    /// The value bound to the top-level `nodes` key.
    TopLevelNodes,
    /// Anywhere else.
    Other,
}

/// A classified limit failure, round-tripped through `serde`'s `Error::custom`
/// (the same encode/recover trick used by [`crate::scan`], kept private here).
struct Reject {
    code: &'static str,
    message: String,
    pointer: String,
    actual: Option<String>,
}

/// Internal sentinel marking a limit rejection encoded inside a `serde` custom
/// error message; an implementation detail that never reaches a caller.
const SENTINEL: &str = "\u{1}paintop-limit\u{1}";
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
        // `serde_json` appends `" at line L column C"`; trim it back off.
        let payload = payload.split(" at line ").next().unwrap_or(payload);
        let mut parts = payload.splitn(4, SEP);
        let code = parts.next()?;
        let pointer = parts.next().unwrap_or("");
        let actual = parts.next().unwrap_or("");
        let message = parts.next().unwrap_or("");
        let code = match code {
            E_MAX_PLAN_BYTES => E_MAX_PLAN_BYTES,
            E_MAX_DEPTH => E_MAX_DEPTH,
            E_MAX_NODES => E_MAX_NODES,
            _ => E_MAX_INLINE_PAYLOAD,
        };
        let mut context = ErrorContext::default();
        if !pointer.is_empty() {
            context = context.with_path(pointer.to_owned());
        }
        if !actual.is_empty() {
            context = context.with_actual(actual.to_owned());
        }
        Some(Error::new(ErrorClass::Policy, code, message.to_owned()).with_context(context))
    }

    /// Raise this rejection as the deserializer's error type.
    fn raise<E: de::Error>(&self) -> E {
        E::custom(self.encode())
    }
}

/// Recursive limit-enforcing walk over one JSON value at JSON-pointer `pointer`
/// and nesting `depth`. Streams over the original token stream once.
struct LimitVisitor<'a> {
    pointer: String,
    depth: usize,
    limits: &'a PlanLimits,
    location: Location,
}

/// Seed that captures the next value and recursively limit-checks it under the
/// child `pointer`, `depth`, and `location`.
struct ChildSeed<'a> {
    pointer: String,
    depth: usize,
    limits: &'a PlanLimits,
    location: Location,
}

impl<'de> de::DeserializeSeed<'de> for ChildSeed<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_any(LimitVisitor {
            pointer: self.pointer,
            depth: self.depth,
            limits: self.limits,
            location: self.location,
        })
    }
}

impl<'de> Visitor<'de> for LimitVisitor<'_> {
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

    fn visit_f64<E: de::Error>(self, _v: f64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<(), E> {
        if value.len() > self.limits.max_inline_len {
            return Err(Reject::new(
                E_MAX_INLINE_PAYLOAD,
                format!(
                    "inline string is {} bytes, exceeding the {}-byte limit",
                    value.len(),
                    self.limits.max_inline_len
                ),
                &self.pointer,
                Some(value.len().to_string()),
            )
            .raise());
        }
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
        let child_depth = self.enter()?;
        // The top-level `nodes` array is capped by `max_nodes`; every inline
        // array is additionally capped by `max_inline_len`.
        let (cap, code, label) = match self.location {
            Location::TopLevelNodes => (self.limits.max_nodes, E_MAX_NODES, "nodes"),
            _ => (self.limits.max_inline_len, E_MAX_INLINE_PAYLOAD, "array"),
        };
        let mut index = 0usize;
        loop {
            let child = ChildSeed {
                pointer: format!("{}/{index}", self.pointer),
                depth: child_depth,
                limits: self.limits,
                location: Location::Other,
            };
            if seq.next_element_seed(child)?.is_none() {
                break;
            }
            index += 1;
            if index > cap {
                return Err(Reject::new(
                    code,
                    format!("{label} array exceeds the {cap}-element limit"),
                    &self.pointer,
                    Some(format!("> {cap}")),
                )
                .raise());
            }
        }
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<(), A::Error> {
        let child_depth = self.enter()?;
        let at_root = matches!(self.location, Location::Root);
        while let Some(key) = map.next_key::<&RawValue>()? {
            // Borrow the key text without allocating a `String`; the leading and
            // trailing quotes of the JSON string token are stripped for the path.
            let key_text = key.get();
            let key_name = key_text
                .strip_prefix('"')
                .and_then(|k| k.strip_suffix('"'))
                .unwrap_or(key_text);
            let location = if at_root && key_name == "nodes" {
                Location::TopLevelNodes
            } else {
                Location::Other
            };
            let child = ChildSeed {
                pointer: format!("{}/{}", self.pointer, escape_pointer_token(key_name)),
                depth: child_depth,
                limits: self.limits,
                location,
            };
            map.next_value_seed(child)?;
        }
        Ok(())
    }
}

impl LimitVisitor<'_> {
    /// Enter one container level, enforcing the depth ceiling, and return the
    /// child depth for nested values.
    fn enter<E: de::Error>(&self) -> std::result::Result<usize, E> {
        if self.depth > self.limits.max_depth {
            return Err(Reject::new(
                E_MAX_DEPTH,
                format!(
                    "nesting depth {} exceeds the {} level limit",
                    self.depth, self.limits.max_depth
                ),
                &self.pointer,
                Some(self.depth.to_string()),
            )
            .raise());
        }
        Ok(self.depth + 1)
    }
}

/// Escape a map key for use as a JSON-pointer reference token (RFC 6901):
/// `~` -> `~0`, `/` -> `~1`.
fn escape_pointer_token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::{
        E_MAX_DEPTH, E_MAX_INLINE_PAYLOAD, E_MAX_NODES, E_MAX_PLAN_BYTES, PlanLimits, check_limits,
    };
    use crate::error::ErrorClass;

    fn small_limits() -> PlanLimits {
        PlanLimits {
            max_bytes: 1024,
            max_depth: 8,
            max_nodes: 3,
            max_inline_len: 5,
        }
    }

    #[test]
    fn within_limits_passes() {
        check_limits(
            r#"{"paintop": "1.0", "nodes": [{"id": "a"}, {"id": "b"}]}"#,
            &small_limits(),
        )
        .expect("plan within all limits must pass");
    }

    #[test]
    fn oversized_plan_is_policy_error() {
        let limits = PlanLimits {
            max_bytes: 8,
            ..small_limits()
        };
        let err = check_limits(r#"{"nodes": []}"#, &limits).unwrap_err();
        assert_eq!(err.class, ErrorClass::Policy);
        assert_eq!(err.code, E_MAX_PLAN_BYTES);
    }

    #[test]
    fn byte_check_runs_before_walk() {
        // Syntactically invalid JSON that is also oversized must fail on bytes,
        // since the byte check is the first thing `check_limits` does.
        let limits = PlanLimits {
            max_bytes: 4,
            ..small_limits()
        };
        let err = check_limits("not even json at all", &limits).unwrap_err();
        assert_eq!(err.code, E_MAX_PLAN_BYTES);
    }

    fn depth_4() -> PlanLimits {
        PlanLimits {
            max_depth: 4,
            ..small_limits()
        }
    }

    #[test]
    fn excessive_depth_is_rejected() {
        // max_depth = 4; this nests 5 array levels.
        let err = check_limits("[[[[[1]]]]]", &depth_4()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Policy);
        assert_eq!(err.code, E_MAX_DEPTH);
    }

    #[test]
    fn depth_at_limit_is_accepted() {
        // Exactly max_depth (4) levels must pass.
        check_limits("[[[[1]]]]", &depth_4()).expect("depth at the limit is allowed");
    }

    #[test]
    fn mixed_object_array_depth_counts_every_container() {
        // object(1) -> array(2) -> object(3) -> array(4) -> array(5) > 4.
        let err = check_limits(r#"{"a": [{"b": [[1]]}]}"#, &depth_4()).unwrap_err();
        assert_eq!(err.code, E_MAX_DEPTH);
    }

    #[test]
    fn too_many_nodes_is_rejected() {
        // max_nodes = 3; four node entries.
        let json = r#"{"nodes": [{"id":"a"},{"id":"b"},{"id":"c"},{"id":"d"}]}"#;
        let err = check_limits(json, &small_limits()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Policy);
        assert_eq!(err.code, E_MAX_NODES);
        assert_eq!(err.context.path.as_deref(), Some("/nodes"));
    }

    #[test]
    fn node_count_at_limit_is_accepted() {
        let json = r#"{"nodes": [{"id":"a"},{"id":"b"},{"id":"c"}]}"#;
        check_limits(json, &small_limits()).expect("exactly max_nodes is allowed");
    }

    #[test]
    fn only_top_level_nodes_array_is_node_capped() {
        // A `nodes` key nested inside params is NOT the graph node array, so its
        // length is bounded by the (larger, here equal) inline cap, not max_nodes;
        // four short elements exceed the inline cap of 5? No: 4 <= 5, so it passes.
        let json = r#"{"params": {"nodes": [1, 2, 3, 4]}}"#;
        check_limits(json, &small_limits()).expect("nested `nodes` is not the graph array");
    }

    #[test]
    fn oversized_inline_array_is_rejected() {
        // max_inline_len = 5; a six-element params array.
        let json = r#"{"nodes": [{"id": "a", "params": {"splats": [1,2,3,4,5,6]}}]}"#;
        let err = check_limits(json, &small_limits()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Policy);
        assert_eq!(err.code, E_MAX_INLINE_PAYLOAD);
        assert_eq!(err.context.path.as_deref(), Some("/nodes/0/params/splats"));
    }

    #[test]
    fn oversized_inline_string_is_rejected() {
        // max_inline_len = 5; a six-byte string.
        let err = check_limits(r#"{"note": "abcdef"}"#, &small_limits()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Policy);
        assert_eq!(err.code, E_MAX_INLINE_PAYLOAD);
        assert_eq!(err.context.path.as_deref(), Some("/note"));
    }

    #[test]
    fn string_at_inline_limit_is_accepted() {
        check_limits(r#"{"note": "abcde"}"#, &small_limits()).expect("string at the limit is ok");
    }

    #[test]
    fn syntactically_invalid_json_is_not_a_limit_error() {
        // The scan pass owns parse errors; check_limits must swallow them so the
        // pipeline does not surface a misclassified policy error for bad syntax.
        check_limits("{ not json", &small_limits())
            .expect("limit check ignores syntax errors (the scan pass owns them)");
    }

    #[test]
    fn default_limits_accept_a_realistic_plan() {
        let json = r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "n", "op": "filter.gaussian_blur@1", "params": {"sigma": 8.0}}],
            "exports": {}
        }"#;
        check_limits(json, &PlanLimits::DEFAULT).expect("a normal plan is within default limits");
    }

    #[test]
    fn pointer_tokens_escape_slash_and_tilde() {
        let json = r#"{"a/b~c": "abcdef"}"#;
        let err = check_limits(json, &small_limits()).unwrap_err();
        assert_eq!(err.code, E_MAX_INLINE_PAYLOAD);
        assert_eq!(err.context.path.as_deref(), Some("/a~1b~0c"));
    }
}
