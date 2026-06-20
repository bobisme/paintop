//! Central error taxonomy for paintop.
//!
//! Every fallible subsystem in the workspace surfaces failures through the
//! types defined here, so that the agent-facing contract (`IR_SPEC` §19) and
//! the stable CLI exit classes (`plan.md` §15.4) have a single source of truth.
//!
//! The structured error *is* part of the agent-facing contract: it serializes
//! to the exact §19 JSON envelope and maps deterministically to one of the
//! stable exit codes. Downstream bones build their typed errors out of these
//! pieces rather than inventing parallel taxonomies.

use serde::{Deserialize, Serialize};

/// The error class: a coarse, stable bucket that an agent can match on without
/// knowing the full set of specific codes (`IR_SPEC` §19).
///
/// The set is closed and ordered to mirror the spec listing. Each class maps
/// to exactly one stable exit code via [`ErrorClass::exit_code`] (`plan.md`
/// §15.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorClass {
    /// The input could not be tokenized/parsed as JSON (or the wire format).
    Parse,
    /// The input parsed but violated the plan/manifest schema.
    Schema,
    /// A reference (node id, input handle, op id) did not resolve.
    Reference,
    /// A value or port had the wrong type/shape.
    Type,
    /// A type-correct plan violated a semantic rule (e.g. color encoding).
    Semantic,
    /// A resource or execution policy limit rejected the plan.
    Policy,
    /// A backend failed while executing an otherwise-valid plan.
    Execution,
    /// A runtime assertion (e.g. `assert.no_change_outside_mask`) failed.
    Assertion,
    /// A differential / conformance comparison failed.
    Conformance,
    /// A model adapter failed (load, verify, or invoke).
    Model,
    /// An asset (input image, sidecar) failed integrity or availability.
    Asset,
    /// Exporting / writing a result failed its integrity contract.
    Export,
}

impl ErrorClass {
    /// Every error class, in spec order. Useful for exhaustive table tests.
    pub const ALL: [Self; 12] = [
        Self::Parse,
        Self::Schema,
        Self::Reference,
        Self::Type,
        Self::Semantic,
        Self::Policy,
        Self::Execution,
        Self::Assertion,
        Self::Conformance,
        Self::Model,
        Self::Asset,
        Self::Export,
    ];

    /// The stable, lowercase wire token for this class (matches the `serde`
    /// `snake_case` representation used in the §19 JSON envelope).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Schema => "schema",
            Self::Reference => "reference",
            Self::Type => "type",
            Self::Semantic => "semantic",
            Self::Policy => "policy",
            Self::Execution => "execution",
            Self::Assertion => "assertion",
            Self::Conformance => "conformance",
            Self::Model => "model",
            Self::Asset => "asset",
            Self::Export => "export",
        }
    }

    /// The stable process exit code for this class (`plan.md` §15.4).
    ///
    /// The mapping collapses related classes onto a single code:
    /// - `parse`, `schema` -> `2`
    /// - `type`, `semantic` -> `3`
    /// - `policy` -> `4`
    /// - `execution` -> `5`
    /// - `assertion` -> `6`
    /// - `conformance` -> `7`
    /// - `model` -> `8`
    /// - `asset`, `export` -> `9`
    ///
    /// `reference` is a kind of schema/wiring failure and shares the `schema`
    /// code (`2`): a dangling reference is detected during the same load/check
    /// phase as schema validation.
    #[must_use]
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::Parse | Self::Schema | Self::Reference => 2,
            Self::Type | Self::Semantic => 3,
            Self::Policy => 4,
            Self::Execution => 5,
            Self::Assertion => 6,
            Self::Conformance => 7,
            Self::Model => 8,
            Self::Asset | Self::Export => 9,
        }
    }
}

/// A non-authoritative remediation hint attached to an error (`IR_SPEC` §19).
///
/// Suggestions are *data*, never auto-applied edits: an agent may choose to act
/// on them, but the runtime never mutates a plan on their behalf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Suggestion {
    /// The op that would address the error, e.g. `color.convert@1`.
    pub op: String,
    /// Parameters the suggested op would be invoked with. Kept as canonical
    /// JSON so suggestions can carry arbitrary op params without this crate
    /// depending on every op's param schema.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, serde_json::Value>,
}

/// The structured context that accompanies an [`Error`] in the §19 envelope.
///
/// All locating fields are optional because not every failure has a node, a
/// JSON path, or a concrete `actual`/`expected` pair.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorContext {
    /// The plan node id the error is attributed to, if any (e.g. `"blurred"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// A JSON pointer into the plan locating the offending value, if known
    /// (e.g. `"/nodes/7/in/image"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The actual value/encoding that was found, rendered as a stable string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
    /// The value/encoding that was expected, rendered as a stable string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Non-authoritative remediation hints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<Suggestion>,
}

impl ErrorContext {
    /// Attach the node id the error is attributed to.
    #[must_use]
    pub fn with_node(mut self, node: impl Into<String>) -> Self {
        self.node = Some(node.into());
        self
    }

    /// Attach a JSON pointer locating the offending value.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Attach the actual value that was found.
    #[must_use]
    pub fn with_actual(mut self, actual: impl Into<String>) -> Self {
        self.actual = Some(actual.into());
        self
    }

    /// Attach the value that was expected.
    #[must_use]
    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    /// Append a remediation suggestion.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }
}

/// A structured paintop error: a stable class + code, a human message, and
/// optional locating context (`IR_SPEC` §19).
///
/// This is the central, typed error every library crate surfaces. It is a
/// `thiserror` enum-of-one over a struct so that the `Display`/`std::error`
/// machinery is derived while the rich fields stay directly addressable.
///
/// The locating [`ErrorContext`] (five owned-`String`/`Vec` fields) is the
/// heavy part of the payload, so it is **boxed**: this keeps `size_of::<Error>()`
/// small (`<= 128`) and `Result<T, Error>` cheap to move, without forcing every
/// fallible function in the workspace to paste a large-error clippy allow. The
/// `Box` derefs transparently, so `err.context.path` and friends read exactly as
/// before. The common (context-free) error allocates no context box.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{class}/{code}: {message}", class = .class.as_str())]
pub struct Error {
    /// The coarse, stable error class.
    pub class: ErrorClass,
    /// The stable machine code, e.g. `E_COLOR_ENCODING_MISMATCH`.
    pub code: String,
    /// A human-readable message describing the failure.
    pub message: String,
    /// Structured locating context, boxed to keep [`Error`] small.
    pub context: Box<ErrorContext>,
}

impl Error {
    /// Construct an error from its class, stable code, and message, with empty
    /// context. Use the [`ErrorContext`] builders or [`Error::with_context`] to
    /// attach locating information.
    #[must_use]
    pub fn new(class: ErrorClass, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            class,
            code: code.into(),
            message: message.into(),
            context: Box::default(),
        }
    }

    /// Replace this error's context.
    #[must_use]
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = Box::new(context);
        self
    }

    /// The stable process exit code for this error (`plan.md` §15.4).
    #[must_use]
    pub const fn exit_code(&self) -> i32 {
        self.class.exit_code()
    }

    /// Serialize to the §19 JSON envelope:
    /// `{"ok":false,"error":{class,code,message,node,path,actual,expected,suggestions}}`.
    ///
    /// # Errors
    /// Returns the underlying `serde_json` error only if the in-memory error
    /// structure cannot be represented as JSON, which does not occur for the
    /// owned `String`/`Value` fields used here.
    pub fn to_json_value(&self) -> std::result::Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(ErrorEnvelope::from(self))
    }

    /// Serialize the §19 envelope to a compact JSON string.
    ///
    /// # Errors
    /// See [`Error::to_json_value`].
    pub fn to_json_string(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string(&ErrorEnvelope::from(self))
    }
}

/// The exact §19 wire shape: `{"ok": false, "error": { ... }}`.
///
/// `ok` is always `false` for an error envelope; the literal is enforced by
/// the serializer below so the field cannot drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorEnvelope {
    /// Always `false`. Present so a single envelope type can describe both the
    /// success (`ok: true`, defined by later bones) and failure cases.
    pub ok: bool,
    /// The error payload, flattening class/code/message and context into the
    /// flat §19 object.
    pub error: ErrorPayload,
}

/// The flat `error` object of the §19 envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorPayload {
    /// The stable error class.
    pub class: ErrorClass,
    /// The stable machine code.
    pub code: String,
    /// The human-readable message.
    pub message: String,
    /// The node id, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// The JSON pointer, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The actual value, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
    /// The expected value, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Remediation suggestions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<Suggestion>,
}

impl From<&Error> for ErrorEnvelope {
    fn from(error: &Error) -> Self {
        Self {
            ok: false,
            error: ErrorPayload {
                class: error.class,
                code: error.code.clone(),
                message: error.message.clone(),
                node: error.context.node.clone(),
                path: error.context.path.clone(),
                actual: error.context.actual.clone(),
                expected: error.context.expected.clone(),
                suggestions: error.context.suggestions.clone(),
            },
        }
    }
}

/// Convenience `Result` alias for the central paintop error type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::{Error, ErrorClass, ErrorContext, Suggestion};
    use serde_json::json;

    #[test]
    fn exit_code_table_matches_spec_for_all_twelve_classes() {
        // `plan.md` §15.4 stable exit classes. Every one of the 12 §19 error
        // classes must map to exactly the code below.
        let expected: [(ErrorClass, i32); 12] = [
            (ErrorClass::Parse, 2),
            (ErrorClass::Schema, 2),
            (ErrorClass::Reference, 2),
            (ErrorClass::Type, 3),
            (ErrorClass::Semantic, 3),
            (ErrorClass::Policy, 4),
            (ErrorClass::Execution, 5),
            (ErrorClass::Assertion, 6),
            (ErrorClass::Conformance, 7),
            (ErrorClass::Model, 8),
            (ErrorClass::Asset, 9),
            (ErrorClass::Export, 9),
        ];
        // Guard against a class being added without a table entry.
        assert_eq!(ErrorClass::ALL.len(), expected.len());
        for (class, code) in expected {
            assert_eq!(
                class.exit_code(),
                code,
                "exit code mismatch for class {class:?}"
            );
            // The struct-level helper must agree with the class-level one.
            let err = Error::new(class, "E_X", "x");
            assert_eq!(err.exit_code(), code);
        }
    }

    #[test]
    fn every_class_has_a_stable_wire_token() {
        for class in ErrorClass::ALL {
            let token = serde_json::to_value(class).unwrap();
            assert_eq!(token, json!(class.as_str()));
        }
    }

    #[test]
    fn golden_section_19_json_envelope() {
        // Reconstructs the exact example from `IR_SPEC` §19.
        let err = Error::new(
            ErrorClass::Semantic,
            "E_COLOR_ENCODING_MISMATCH",
            "filter.gaussian_blur@1 requires a linear image",
        )
        .with_context(
            ErrorContext::default()
                .with_node("blurred")
                .with_path("/nodes/7/in/image")
                .with_actual("srgb")
                .with_expected("linear-*")
                .with_suggestion(Suggestion {
                    op: "color.convert@1".to_owned(),
                    params: {
                        let mut m = serde_json::Map::new();
                        m.insert("to".to_owned(), json!("linear-srgb"));
                        m
                    },
                }),
        );

        let expected = json!({
            "ok": false,
            "error": {
                "class": "semantic",
                "code": "E_COLOR_ENCODING_MISMATCH",
                "message": "filter.gaussian_blur@1 requires a linear image",
                "node": "blurred",
                "path": "/nodes/7/in/image",
                "actual": "srgb",
                "expected": "linear-*",
                "suggestions": [
                    {
                        "op": "color.convert@1",
                        "params": {"to": "linear-srgb"}
                    }
                ]
            }
        });

        assert_eq!(err.to_json_value().unwrap(), expected);
    }

    #[test]
    fn minimal_error_omits_empty_optional_fields() {
        // A context-free error must not emit null/empty locating fields.
        let err = Error::new(ErrorClass::Parse, "E_INVALID_JSON", "unexpected EOF");
        let value = err.to_json_value().unwrap();
        let obj = value["error"].as_object().unwrap();
        assert_eq!(value["ok"], json!(false));
        assert_eq!(obj["class"], json!("parse"));
        assert_eq!(obj["code"], json!("E_INVALID_JSON"));
        assert!(!obj.contains_key("node"));
        assert!(!obj.contains_key("path"));
        assert!(!obj.contains_key("actual"));
        assert!(!obj.contains_key("expected"));
        assert!(!obj.contains_key("suggestions"));
    }

    #[test]
    fn each_class_is_constructible_with_context() {
        for class in ErrorClass::ALL {
            let err = Error::new(class, "E_X", "msg")
                .with_context(ErrorContext::default().with_node("n"));
            assert_eq!(err.class, class);
            assert_eq!(err.context.node.as_deref(), Some("n"));
            // Round-trips through the envelope.
            assert!(err.to_json_string().unwrap().contains(class.as_str()));
        }
    }

    #[test]
    fn display_includes_class_and_code() {
        let err = Error::new(ErrorClass::Policy, "E_MAX_NODES", "too many nodes");
        assert_eq!(err.to_string(), "policy/E_MAX_NODES: too many nodes");
    }

    #[test]
    fn error_is_small_so_results_stay_cheap() {
        // The heavy `ErrorContext` is boxed so `Result<T, Error>` is cheap to
        // move and no large-error clippy allow is needed anywhere. Clippy's
        // large-error threshold is 128 bytes; keep `Error` at or below it.
        assert!(
            std::mem::size_of::<Error>() <= 128,
            "Error grew to {} bytes; box more of its payload to stay <= 128",
            std::mem::size_of::<Error>()
        );
    }
}
