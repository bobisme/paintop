//! The `clap` command-line surface for the `paintop` binary (`plan.md` §15.4).
//!
//! Only the argument *shape* lives here; each subcommand's behavior is in
//! [`crate::commands`]. The grammar mirrors the §15.4 invocation table exactly so
//! the agent-facing contract is stable.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// `paintop`: the stable, machine-facing image-editing runtime CLI.
///
/// In machine modes stdout is pure JSON and every log line goes to stderr, so a
/// caller can parse stdout unconditionally. Process exit codes are the stable
/// §15.4 classes (`0` success, `2` parse/schema, `3` type/semantic, `4` policy,
/// `5` execution, `6` assertion, `7` conformance, `8` model, `9` asset/export).
#[derive(Debug, Parser)]
#[command(name = "paintop", version, about, long_about = None)]
pub struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The output rendering for commands that support a `--format` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Format {
    /// Machine-readable JSON on stdout (the default and only M0 format).
    #[default]
    Json,
}

/// The top-level subcommands (`plan.md` §15.4).
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Parse, resolve, and type-check a plan, reporting the first failure.
    Validate(ValidateArgs),
    /// Print a structured explanation of a plan (semantic hash, nodes, exports).
    Explain(ExplainArgs),
    /// Execute a plan whole-image and report the run outcome.
    Run(RunArgs),
    /// Emit the plan's dependency graph (DOT) to a file.
    Graph(GraphArgs),
    /// Diff two images and report whether they are byte-identical.
    Diff(DiffArgs),
    /// Operation registry discovery.
    #[command(subcommand)]
    Op(OpCommand),
    /// Run a backend self-test (stub in M0).
    Selftest(SelftestArgs),
}

/// `paintop validate <plan>`.
#[derive(Debug, Args)]
pub struct ValidateArgs {
    /// Path to the plan JSON file.
    pub plan: PathBuf,
}

/// `paintop explain <plan> [--format json]`.
#[derive(Debug, Args)]
pub struct ExplainArgs {
    /// Path to the plan JSON file.
    pub plan: PathBuf,
    /// Output format (machine JSON).
    #[arg(long, value_enum, default_value_t = Format::Json)]
    pub format: Format,
}

/// `paintop run <plan> [--bundle <dir>]`.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Path to the plan JSON file.
    pub plan: PathBuf,
    /// Directory the evidence bundle would be written to (recorded, not yet
    /// materialized in M0).
    #[arg(long)]
    pub bundle: Option<PathBuf>,
}

/// `paintop graph <plan> --out <file>`.
#[derive(Debug, Args)]
pub struct GraphArgs {
    /// Path to the plan JSON file.
    pub plan: PathBuf,
    /// Destination file for the emitted DOT graph.
    #[arg(long)]
    pub out: PathBuf,
}

/// `paintop diff <before> <after> [--bundle <dir>]`.
#[derive(Debug, Args)]
pub struct DiffArgs {
    /// The reference (before) image path.
    pub before: PathBuf,
    /// The candidate (after) image path.
    pub after: PathBuf,
    /// Directory a diff bundle would be written to (recorded, not yet
    /// materialized in M0).
    #[arg(long)]
    pub bundle: Option<PathBuf>,
}

/// `paintop op <list|schema>`.
#[derive(Debug, Subcommand)]
pub enum OpCommand {
    /// List every registered operation as JSON.
    List(OpListArgs),
    /// Print one operation's manifest and JSON schema.
    Schema(OpSchemaArgs),
}

/// `paintop op list [--format json]`.
#[derive(Debug, Args)]
pub struct OpListArgs {
    /// Output format (machine JSON).
    #[arg(long, value_enum, default_value_t = Format::Json)]
    pub format: Format,
}

/// `paintop op schema <id>`.
#[derive(Debug, Args)]
pub struct OpSchemaArgs {
    /// The canonical operation id, e.g. `filter.gaussian_blur@1`.
    pub id: String,
}

/// `paintop selftest [--backend <name>]`.
#[derive(Debug, Args)]
pub struct SelftestArgs {
    /// The backend to self-test (M0 accepts the value but runs a stub).
    #[arg(long, default_value = "cpu-reference")]
    pub backend: String,
}
