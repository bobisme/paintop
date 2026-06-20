//! Integration tests for the `paintop` CLI (`plan.md` §15.4 contract).
//!
//! These run the built binary as a subprocess and assert the agent-facing
//! contract: the documented exit codes, that **stdout is pure JSON** in machine
//! mode, and that **logs go to stderr** (never stdout). The two `validate`
//! invocations are the M0 exit criterion (`plan.md` §19).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Absolute path to the freshly built `paintop` binary under test.
const fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_paintop")
}

/// Absolute path to a checked-in fixture plan.
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/plans")
        .join(name)
}

/// Run `paintop <args>` and capture its output.
fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("the paintop binary must be runnable")
}

/// Parse captured stdout bytes as JSON, asserting stdout is exactly one JSON
/// document (the machine-mode purity contract).
fn stdout_json(output: &Output) -> serde_json::Value {
    let text = std::str::from_utf8(&output.stdout).expect("stdout must be valid UTF-8");
    serde_json::from_str(text)
        .unwrap_or_else(|err| panic!("stdout must be pure JSON, got {text:?}: {err}"))
}

#[test]
fn validate_empty_plan_exits_zero_with_json() {
    let output = run(&["validate", fixture("empty-valid.json").to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["ok"], serde_json::json!(true));
    assert_eq!(value["valid"], serde_json::json!(true));
}

#[test]
fn validate_unknown_field_exits_two_with_stable_code() {
    // M0 exit criterion: an unknown top-level field is a schema error -> exit 2
    // with a stable machine code.
    let output = run(&["validate", fixture("unknown-field.json").to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["ok"], serde_json::json!(false));
    assert_eq!(value["error"]["class"], serde_json::json!("schema"));
    assert_eq!(value["error"]["code"], serde_json::json!("E_UNKNOWN_FIELD"));
}

#[test]
fn validate_missing_file_is_asset_error_exit_nine() {
    let output = run(&["validate", "/nonexistent/plan/does-not-exist.json"]);
    assert_eq!(output.status.code(), Some(9), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["error"]["class"], serde_json::json!("asset"));
}

#[test]
fn op_list_emits_valid_json() {
    let output = run(&["op", "list", "--format", "json"]);
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["ok"], serde_json::json!(true));
    let ops = value["operations"].as_array().expect("operations array");
    assert!(!ops.is_empty(), "stub registry must list operations");
    assert!(
        ops.iter()
            .any(|o| o["id"] == serde_json::json!("filter.gaussian_blur@1"))
    );
}

#[test]
fn op_schema_known_op_emits_manifest_and_schema() {
    let output = run(&["op", "schema", "filter.gaussian_blur@1"]);
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(
        value["manifest"]["id"],
        serde_json::json!("filter.gaussian_blur@1")
    );
    assert!(value["schema"].is_object(), "schema must be a JSON object");
}

#[test]
fn op_schema_unknown_op_is_reference_error_exit_two() {
    let output = run(&["op", "schema", "filter.no_such_op@1"]);
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["error"]["class"], serde_json::json!("reference"));
}

#[test]
fn op_schema_malformed_id_is_schema_error() {
    let output = run(&["op", "schema", "not a valid id"]);
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["ok"], serde_json::json!(false));
}

#[test]
fn explain_emits_semantic_hash() {
    let output = run(&["explain", fixture("empty-valid.json").to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    let hash = value["semantic_hash"]
        .as_str()
        .expect("semantic_hash string");
    assert!(hash.starts_with("blake3:"), "got {hash}");
}

#[test]
fn stdout_is_pure_json_and_logs_go_to_stderr() {
    // The defining machine-mode contract: stdout parses as a single JSON value
    // (asserted by stdout_json) and the human log line lives only on stderr.
    let output = run(&["validate", fixture("empty-valid.json").to_str().unwrap()]);
    let _ = stdout_json(&output); // panics if stdout carries a non-JSON log line
    let err = stderr(&output);
    assert!(
        err.contains("validate"),
        "stderr should carry the log line, got {err:?}"
    );
    // And the log line must NOT have leaked onto stdout.
    let out = std::str::from_utf8(&output.stdout).unwrap();
    assert!(
        !out.contains("validate:"),
        "stdout leaked a log line: {out:?}"
    );
}

#[test]
fn selftest_is_a_stub_that_exits_zero() {
    let output = run(&["selftest", "--backend", "cpu-reference"]);
    assert_eq!(output.status.code(), Some(0), "stderr: {}", stderr(&output));
    let value = stdout_json(&output);
    assert_eq!(value["status"], serde_json::json!("stub"));
}

/// Decode captured stderr for assertion messages.
fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
