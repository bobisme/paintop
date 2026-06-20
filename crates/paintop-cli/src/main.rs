//! `paintop`: the stable, machine-facing command-line interface.
//!
//! Per `plan.md` §6.1 the CLI reuses the canonical IR from [`paintop_ir`] rather
//! than inventing parallel JSON structs, and per §15.4 it speaks a stable
//! contract: in machine mode **stdout is pure JSON** and every log line goes to
//! **stderr**, and the process exit code is one of the stable §15.4 classes.
//!
//! This module is only the dispatch shim: it parses arguments ([`cli`]), runs the
//! selected subcommand ([`commands`]), prints the resulting JSON to stdout, and
//! returns the subcommand's exit code. All the behavior lives in [`commands`].

mod cli;
mod commands;
mod output;
mod stub_ops;

// Keep the §6.1-legal crate edges live: the CLI may read core/cpu, which read
// the IR. Concrete use arrives as the executor wiring lands in segment 2.
use paintop_core as _;
use paintop_cpu as _;

use clap::Parser;

use crate::cli::{Cli, Command, OpCommand};
use crate::output::{CommandOutcome, log};

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let outcome = dispatch(cli.command);
    let code = outcome.emit().unwrap_or_else(|err| {
        // stdout is closed/broken; report on stderr and use the export-integrity
        // class (`9`) since the result could not be delivered.
        log(&format!("failed to write output: {err}"));
        9
    });
    exit_code(code)
}

/// Route a parsed [`Command`] to its handler, returning the command's outcome.
fn dispatch(command: Command) -> CommandOutcome {
    match command {
        Command::Validate(args) => commands::validate(&args.plan),
        Command::Explain(args) => commands::explain(&args.plan),
        Command::Run(args) => commands::run(&args.plan, args.bundle.as_deref()),
        Command::Graph(args) => commands::graph(&args.plan, &args.out),
        Command::Diff(args) => commands::diff(&args.before, &args.after),
        Command::Op(OpCommand::List(_)) => commands::op_list(),
        Command::Op(OpCommand::Schema(args)) => commands::op_schema(&args.id),
        Command::Selftest(args) => commands::selftest(&args.backend),
    }
}

/// Convert a stable §15.4 integer exit code into a process
/// [`ExitCode`](std::process::ExitCode).
///
/// [`std::process::ExitCode::from`] takes a `u8`; the §15.4 codes are all small
/// non-negative values, so a negative or oversized code (which the taxonomy
/// never produces) is clamped to `1` rather than wrapping.
fn exit_code(code: i32) -> std::process::ExitCode {
    let byte = u8::try_from(code).unwrap_or(1);
    std::process::ExitCode::from(byte)
}
