//! The machine-output contract: pure JSON on stdout, logs on stderr, and the
//! stable §15.4 exit-code mapping.
//!
//! Every command resolves to a [`CommandOutcome`]: either a success `Value`
//! (rendered as a compact `{"ok": true, ...}`-shaped object — the command builds
//! the object) or a structured failure that carries the §19 error envelope and
//! the §15.4 exit code derived from its [`ErrorClass`]. The `main` shim prints
//! the JSON to **stdout** and returns the exit code; human-readable logging is
//! the caller's job and always goes to **stderr** via [`log`].

use std::io::Write;

use paintop_ir::{Error, ErrorClass};

/// The result of running a subcommand: a JSON value to print on stdout and the
/// process exit code (`0` for success, a §15.4 class code for failure).
pub struct CommandOutcome {
    /// The JSON document to write to stdout.
    pub value: serde_json::Value,
    /// The process exit code.
    pub exit_code: i32,
}

impl CommandOutcome {
    /// A successful outcome carrying `value` and exit code `0`.
    #[must_use]
    pub const fn success(value: serde_json::Value) -> Self {
        Self {
            value,
            exit_code: 0,
        }
    }

    /// An outcome carrying `value` and an explicit `exit_code`.
    ///
    /// Used by `run` to surface a completed run whose terminal status maps to a
    /// non-zero §15.4 class (e.g. a failed `error`-severity assertion → `6`)
    /// without treating it as an internal tool error.
    #[must_use]
    pub const fn with_exit_code(value: serde_json::Value, exit_code: i32) -> Self {
        Self { value, exit_code }
    }

    /// A failure outcome built from a paintop [`Error`]: the §19 error envelope
    /// on stdout and the class's stable §15.4 exit code.
    #[must_use]
    pub fn failure(error: &Error) -> Self {
        let value = error.to_json_value().unwrap_or_else(|_| {
            // The envelope is owned `String`/`Value` data, so this never fails;
            // fall back to a minimal literal rather than unwrap.
            serde_json::json!({
                "ok": false,
                "error": { "class": "execution", "code": "E_ENVELOPE", "message": "error" }
            })
        });
        Self {
            value,
            exit_code: error.exit_code(),
        }
    }

    /// Print the JSON document to stdout as a single compact line and return the
    /// exit code. stdout therefore stays pure JSON in machine mode.
    ///
    /// # Errors
    /// Returns an [`std::io::Error`] only if writing to stdout fails.
    pub fn emit(&self) -> std::io::Result<i32> {
        let mut stdout = std::io::stdout().lock();
        // `to_string` on a `serde_json::Value` is infallible.
        writeln!(stdout, "{}", self.value)?;
        stdout.flush()?;
        Ok(self.exit_code)
    }
}

/// Lift an I/O failure (e.g. a missing plan or image file) into the central
/// taxonomy as an [`asset`](ErrorClass::Asset) error (§15.4 exit code `9`), so
/// even filesystem failures surface as a structured stdout envelope.
#[must_use]
pub fn io_error(path: &std::path::Path, err: &std::io::Error, code: &str) -> Error {
    Error::new(
        ErrorClass::Asset,
        code,
        format!("{}: {err}", path.display()),
    )
    .with_context(paintop_ir::ErrorContext::default().with_path(path.display().to_string()))
}

/// Write a human-readable log line to **stderr** (never stdout), keeping the
/// machine output channel clean.
pub fn log(message: &str) {
    let mut stderr = std::io::stderr().lock();
    // Best-effort logging; a failure to write a log line must not change the
    // command's outcome, so the result is intentionally ignored.
    let _ = writeln!(stderr, "{message}");
}
