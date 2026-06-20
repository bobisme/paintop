//! `xtask`: repository automation entry point (op verification and fixture
//! generation, plus future schema-export / CI helpers).
//!
//! `verify-op` runs an operation's definition-of-done report (manifest
//! locate/validate, the verification-category gate, and the
//! `index.json`/`summary.md`/`test-results.json` report tree; see
//! [`mod@verify_op`]). `fixture generate` builds analytic fixtures. Stages that
//! need the executor and per-op test suites (real property/differential runs,
//! contact sheets) land in later segments and slot into the same report layout.
//!
//! `anyhow` is used here because `xtask` is a workspace *binary*, where the
//! M0 decisions permit it (library crates use the typed `paintop-ir` taxonomy).

mod verify_op;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

/// Repository automation tasks for paintop.
#[derive(Debug, Parser)]
#[command(
    name = "xtask",
    about = "Repository automation (op verification, fixture generation, CI helpers).",
    long_about = None,
)]
struct Cli {
    /// The task to run.
    #[command(subcommand)]
    command: Command,
}

/// Top-level `xtask` subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Verify an operation's implementation against its manifest contract.
    VerifyOp(VerifyOpArgs),
    /// Manage analytic / conformance fixtures.
    #[command(subcommand)]
    Fixture(FixtureCommand),
    /// Emit the operation-manifest JSON Schema to stdout.
    Schema,
    /// Validate an operation manifest file against the schema and its internal
    /// consistency rules.
    ValidateManifest(ValidateManifestArgs),
}

/// Arguments for `xtask validate-manifest`.
#[derive(Debug, Parser)]
struct ValidateManifestArgs {
    /// Path to a JSON operation manifest to validate.
    path: std::path::PathBuf,
}

/// Arguments for `xtask verify-op`.
#[derive(Debug, Parser)]
struct VerifyOpArgs {
    /// The operation id to verify, e.g. `filter.gaussian_blur@1`.
    op: String,
    /// Path to the operation manifest. M0 has no on-disk registry, so the
    /// manifest location is supplied rather than discovered.
    #[arg(long)]
    manifest: std::path::PathBuf,
    /// Report root; defaults to `target/verification`. The report is written to
    /// `<out-dir>/<op-id>/`.
    #[arg(long)]
    out_dir: Option<std::path::PathBuf>,
}

/// `xtask fixture` subcommands.
#[derive(Debug, Subcommand)]
enum FixtureCommand {
    /// Generate an analytic fixture from a fixed, versioned formula.
    Generate(FixtureGenerateArgs),
}

/// The analytic fixture kinds the generator can produce (`AGENT_VERIFICATION`
/// §2.3). Each maps to a formula in [`paintop_testkit::fixtures`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "kebab-case")]
enum FixtureKind {
    /// Constant scalar/color field (`--value`, `--channels`).
    Constant,
    /// Single unit impulse at `(--x, --y)`.
    Impulse,
    /// Horizontal `0..1` ramp.
    RampH,
    /// Vertical `0..1` ramp.
    RampV,
    /// Checkerboard of `--tile`-pixel squares.
    Checker,
    /// Horizontal sine grating of `--periods` cycles.
    Sine,
    /// Binary rectangle `[--x,--x+--rw) × [--y,--y+--rh)`.
    Rect,
    /// Binary disc of `--radius` centered at `(--x, --y)`.
    Circle,
    /// RGBA alpha edge with hidden RGB under transparency.
    AlphaEdge,
    /// Single-channel field with NaN/Inf injected.
    NanInf,
    /// `u32` label map with ids starting at `--base`.
    LabelMap,
}

/// The scalar output type requested on the CLI. The generator picks the natural
/// type per kind; `--format` is accepted for forward-compatibility and must
/// agree with that natural type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "kebab-case")]
enum FixtureFormat {
    /// 32-bit float (color / mask / field fixtures).
    F32,
    /// 8-bit unsigned.
    U8,
    /// 32-bit unsigned (label maps).
    U32,
}

/// Arguments for `xtask fixture generate <kind> ...`.
#[derive(Debug, Parser)]
struct FixtureGenerateArgs {
    /// The fixture kind to generate.
    kind: FixtureKind,
    /// Image width in pixels.
    #[arg(long, default_value_t = 64)]
    width: u32,
    /// Image height in pixels.
    #[arg(long, default_value_t = 64)]
    height: u32,
    /// X coordinate (impulse/circle center/rect origin).
    #[arg(long, default_value_t = 0)]
    x: u32,
    /// Y coordinate (impulse/circle center/rect origin).
    #[arg(long, default_value_t = 0)]
    y: u32,
    /// Rectangle width (rect only).
    #[arg(long, default_value_t = 1)]
    rw: u32,
    /// Rectangle height (rect only).
    #[arg(long, default_value_t = 1)]
    rh: u32,
    /// Disc radius (circle only).
    #[arg(long, default_value_t = 1.0)]
    radius: f64,
    /// Constant value (constant only).
    #[arg(long, default_value_t = 0.5)]
    value: f32,
    /// Channel count (constant only).
    #[arg(long, default_value_t = 1)]
    channels: u32,
    /// Checkerboard tile size in pixels (checker only).
    #[arg(long, default_value_t = 8)]
    tile: u32,
    /// Number of sine cycles across the width (sine only).
    #[arg(long, default_value_t = 4.0)]
    periods: f64,
    /// Starting id for the label map (label-map only).
    #[arg(long, default_value_t = 0)]
    base: u32,
    /// The requested scalar storage type; must match the kind's natural type.
    #[arg(long, value_enum, default_value_t = FixtureFormat::F32)]
    format: FixtureFormat,
    /// Output path for the exact numeric array (JSON). A `<out>.manifest.json`
    /// and `<out>.png` preview are written alongside it.
    #[arg(long)]
    out: std::path::PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    dispatch(cli.command)
}

/// Route a parsed command to its handler.
///
/// Split out from `main` so the routing is unit-testable without spawning a
/// process.
fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::VerifyOp(args) => verify_op(&args),
        Command::Fixture(FixtureCommand::Generate(args)) => fixture_generate(&args),
        Command::Schema => schema(),
        Command::ValidateManifest(args) => validate_manifest(&args.path),
    }
}

/// Print the operation-manifest JSON Schema to stdout.
fn schema() -> Result<()> {
    let schema = paintop_ir::manifest::manifest_json_schema();
    let rendered = serde_json::to_string_pretty(&schema)
        .map_err(|e| anyhow::anyhow!("failed to render manifest schema: {e}"))?;
    println!("{rendered}");
    Ok(())
}

/// Load and validate an operation manifest file: it must deserialize into the
/// typed model (which enforces the wire schema via `deny_unknown_fields` and the
/// `OpId` grammar) and then pass `OperationManifest::validate`.
fn validate_manifest(path: &std::path::Path) -> Result<()> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read manifest {}: {e}", path.display()))?;
    let manifest: paintop_ir::manifest::OperationManifest = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("manifest {} failed schema parse: {e}", path.display()))?;
    manifest
        .validate()
        .map_err(|e| anyhow::anyhow!("manifest {} is inconsistent: {e}", path.display()))?;
    println!("ok: {} ({})", manifest.id, path.display());
    Ok(())
}

/// Verify an op against its manifest: run the definition-of-done report and
/// fail (nonzero exit) if any applicable verification category is missing.
fn verify_op(args: &VerifyOpArgs) -> Result<()> {
    verify_op::run(&verify_op::VerifyOpOptions {
        op: args.op.clone(),
        manifest: args.manifest.clone(),
        out_dir: args.out_dir.clone(),
    })
}

/// Build the requested analytic fixture and write its exact numeric array, the
/// §4.2-shaped manifest, and a preview PNG.
///
/// The fixture array (`--out`) is the source of truth; the PNG is auxiliary.
/// Generation is deterministic, so re-running yields byte-identical files.
fn fixture_generate(args: &FixtureGenerateArgs) -> Result<()> {
    use paintop_ir::resource::{Extent, ScalarType};
    use paintop_testkit::fixtures::{self, RampAxis};

    let extent = Extent::new(args.width, args.height);
    let fixture = match args.kind {
        FixtureKind::Constant => fixtures::constant(extent, args.channels, args.value),
        FixtureKind::Impulse => fixtures::impulse(extent, args.x, args.y),
        FixtureKind::RampH => fixtures::ramp(extent, RampAxis::Horizontal),
        FixtureKind::RampV => fixtures::ramp(extent, RampAxis::Vertical),
        FixtureKind::Checker => fixtures::checkerboard(extent, args.tile),
        FixtureKind::Sine => fixtures::sine_grating(extent, args.periods),
        FixtureKind::Rect => fixtures::rectangle(extent, args.x, args.y, args.rw, args.rh),
        FixtureKind::Circle => fixtures::circle(extent, args.x, args.y, args.radius),
        FixtureKind::AlphaEdge => fixtures::alpha_edge(extent),
        FixtureKind::NanInf => fixtures::nan_inf_field(extent),
        FixtureKind::LabelMap => fixtures::label_map(extent, args.base),
    }
    .map_err(|e| anyhow::anyhow!("fixture generation failed: {e}"))?;

    // The requested --format must match the kind's natural scalar type, so the
    // output type is never silently coerced.
    let want = match args.format {
        FixtureFormat::F32 => ScalarType::F32,
        FixtureFormat::U8 => ScalarType::U8,
        FixtureFormat::U32 => ScalarType::U32,
    };
    let got = fixture.data.scalar();
    if want != got {
        bail!(
            "fixture kind {:?} produces {got:?}, but --format requested {want:?}",
            args.kind
        );
    }

    let manifest = fixture.manifest();
    let array_json = fixture
        .to_json()
        .map_err(|e| anyhow::anyhow!("failed to serialize fixture array: {e}"))?;
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| anyhow::anyhow!("failed to serialize manifest: {e}"))?;
    let png = fixture
        .to_preview_png()
        .map_err(|e| anyhow::anyhow!("failed to render preview PNG: {e}"))?;

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("failed to create {}: {e}", parent.display()))?;
    }
    std::fs::write(&args.out, array_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", args.out.display()))?;
    let manifest_path = sibling(&args.out, "manifest.json");
    std::fs::write(&manifest_path, manifest_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", manifest_path.display()))?;
    let png_path = sibling(&args.out, "png");
    std::fs::write(&png_path, &png)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", png_path.display()))?;

    println!(
        "generated {} ({}x{}x{} {:?}) sha256={}",
        manifest.formula,
        manifest.width,
        manifest.height,
        manifest.channels,
        manifest.scalar,
        manifest.sha256
    );
    Ok(())
}

/// Replace (or append) the extension of `path` with `ext` to form a sibling
/// output path (`fixture.json` -> `fixture.manifest.json` / `fixture.png`).
fn sibling(path: &std::path::Path, ext: &str) -> std::path::PathBuf {
    path.with_extension(ext)
}

#[cfg(test)]
mod tests {
    use super::{Cli, Command, FixtureCommand, dispatch};
    use clap::Parser;

    #[test]
    fn cli_definition_is_valid() {
        // `debug_assert` validates the whole derived command graph; a malformed
        // definition would panic here rather than at runtime.
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_verify_op() {
        let cli = Cli::try_parse_from([
            "xtask",
            "verify-op",
            "filter.gaussian_blur@1",
            "--manifest",
            "/tmp/gaussian.json",
        ])
        .unwrap();
        match cli.command {
            Command::VerifyOp(args) => {
                assert_eq!(args.op, "filter.gaussian_blur@1");
                assert_eq!(
                    args.manifest,
                    std::path::PathBuf::from("/tmp/gaussian.json")
                );
            }
            other => panic!("expected verify-op, got {other:?}"),
        }
    }

    #[test]
    fn parses_fixture_generate() {
        let cli = Cli::try_parse_from([
            "xtask",
            "fixture",
            "generate",
            "impulse",
            "--width",
            "65",
            "--height",
            "65",
            "--x",
            "32",
            "--y",
            "32",
            "--format",
            "f32",
            "--out",
            "/tmp/impulse-65.json",
        ])
        .unwrap();
        match cli.command {
            Command::Fixture(FixtureCommand::Generate(args)) => {
                assert_eq!(args.kind, super::FixtureKind::Impulse);
                assert_eq!(args.width, 65);
                assert_eq!(args.x, 32);
                assert_eq!(args.format, super::FixtureFormat::F32);
                assert_eq!(args.out, std::path::PathBuf::from("/tmp/impulse-65.json"));
            }
            other => panic!("expected fixture generate, got {other:?}"),
        }
    }

    #[test]
    fn verify_op_requires_a_manifest_path() {
        // The manifest path is mandatory: there is no on-disk registry in M0.
        assert!(Cli::try_parse_from(["xtask", "verify-op", "filter.invert@1"]).is_err());
    }

    #[test]
    fn verify_op_missing_manifest_errors_cleanly() {
        let cli = Cli::try_parse_from([
            "xtask",
            "verify-op",
            "filter.invert@1",
            "--manifest",
            "/nonexistent/paintop/xtask/manifest.json",
        ])
        .unwrap();
        let err = dispatch(cli.command).unwrap_err();
        assert!(err.to_string().contains("failed to read manifest"), "{err}");
    }

    /// Exit-gate command (`AGENT_VERIFICATION` §4.3): generating the same
    /// impulse twice yields byte-identical files, and the manifest digest
    /// matches the checked-in expected.
    #[test]
    fn fixture_generate_is_deterministic_and_matches_expected_digest() {
        let dir = std::env::temp_dir();
        let out_a = dir.join(format!("paintop_impulse_a_{}.json", std::process::id()));
        let out_b = dir.join(format!("paintop_impulse_b_{}.json", std::process::id()));
        let run = |out: &std::path::Path| {
            let cli = Cli::try_parse_from([
                "xtask",
                "fixture",
                "generate",
                "impulse",
                "--width",
                "65",
                "--height",
                "65",
                "--x",
                "32",
                "--y",
                "32",
                "--format",
                "f32",
                "--out",
                out.to_str().unwrap(),
            ])
            .unwrap();
            dispatch(cli.command).unwrap();
        };
        run(&out_a);
        run(&out_b);

        let a = std::fs::read(&out_a).unwrap();
        let b = std::fs::read(&out_b).unwrap();
        assert_eq!(a, b, "fixture array must be byte-identical across runs");

        // The manifest records the digest of the canonical binary bytes; it must
        // match the checked-in expected value for the §4.3 exit-gate command.
        let manifest_path = out_a.with_extension("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path).unwrap();
        let manifest: paintop_testkit::fixtures::Manifest =
            serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest.sha256, EXPECTED_IMPULSE_65_SHA256);

        for p in [
            out_a,
            out_b,
            manifest_path,
            dir.join(format!("paintop_impulse_a_{}.png", std::process::id())),
            dir.join(format!(
                "paintop_impulse_b_{}.manifest.json",
                std::process::id()
            )),
            dir.join(format!("paintop_impulse_b_{}.png", std::process::id())),
        ] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn fixture_format_mismatch_is_rejected() {
        let dir = std::env::temp_dir();
        let out = dir.join(format!("paintop_fmt_mismatch_{}.json", std::process::id()));
        // label-map is u32; requesting f32 must fail rather than coerce.
        let cli = Cli::try_parse_from([
            "xtask",
            "fixture",
            "generate",
            "label-map",
            "--width",
            "4",
            "--height",
            "4",
            "--format",
            "f32",
            "--out",
            out.to_str().unwrap(),
        ])
        .unwrap();
        let err = dispatch(cli.command).unwrap_err();
        assert!(err.to_string().contains("requested"), "{err}");
        let _ = std::fs::remove_file(out);
    }

    /// The sha256 of the canonical bytes of the §4.3 exit-gate impulse fixture
    /// (`impulse --width 65 --height 65 --x 32 --y 32 --format f32`). This is the
    /// checked-in expected value the exit gate compares against; a change here
    /// signals a formula or canonicalization drift.
    const EXPECTED_IMPULSE_65_SHA256: &str =
        "023aa0fd2c5fbf5cf11900a295169cfa992855643ae45fbaf37e16b6067bf178";

    #[test]
    fn missing_subcommand_is_an_error() {
        assert!(Cli::try_parse_from(["xtask"]).is_err());
    }

    #[test]
    fn parses_schema_command() {
        let cli = Cli::try_parse_from(["xtask", "schema"]).unwrap();
        assert!(matches!(cli.command, Command::Schema));
        // The schema command runs cleanly and prints valid JSON.
        dispatch(cli.command).unwrap();
    }

    #[test]
    fn validate_manifest_accepts_a_valid_manifest() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("paintop_xtask_valid_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{
                "id": "filter.gaussian_blur@1",
                "determinism": "bounded",
                "roi": { "category": "local-halo", "halo_px": 24 },
                "inputs": [{ "name": "image", "kind": "Image" }],
                "outputs": [{ "name": "image", "kind": "Image" }],
                "params": [
                    { "name": "sigma_px", "type": "float", "unit": "pixels", "required": true }
                ],
                "implementations": ["cpu.reference@1"]
            }"#,
        )
        .unwrap();
        let cli =
            Cli::try_parse_from(["xtask", "validate-manifest", path.to_str().unwrap()]).unwrap();
        let result = dispatch(cli.command);
        let _ = std::fs::remove_file(&path);
        result.unwrap();
    }

    #[test]
    fn validate_manifest_rejects_an_inconsistent_manifest() {
        // Stochastic op with no seed parameter -> internal-consistency failure.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("paintop_xtask_invalid_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{
                "id": "synthesis.patchmatch@1",
                "determinism": "stochastic",
                "roi": { "category": "full-domain" },
                "inputs": [{ "name": "image", "kind": "Image" }],
                "outputs": [{ "name": "image", "kind": "Image" }],
                "implementations": ["cpu.reference@1"]
            }"#,
        )
        .unwrap();
        let cli =
            Cli::try_parse_from(["xtask", "validate-manifest", path.to_str().unwrap()]).unwrap();
        let result = dispatch(cli.command);
        let _ = std::fs::remove_file(&path);
        let err = result.unwrap_err();
        assert!(err.to_string().contains("inconsistent"), "{err}");
    }

    #[test]
    fn validate_manifest_rejects_an_unknown_field() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("paintop_xtask_unknown_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{
                "id": "filter.gaussian_blur@1",
                "determinism": "bounded",
                "roi": { "category": "local-halo", "halo_px": 24 },
                "outputs": [{ "name": "image", "kind": "Image" }],
                "bogus": true
            }"#,
        )
        .unwrap();
        let cli =
            Cli::try_parse_from(["xtask", "validate-manifest", path.to_str().unwrap()]).unwrap();
        let result = dispatch(cli.command);
        let _ = std::fs::remove_file(&path);
        let err = result.unwrap_err();
        assert!(err.to_string().contains("schema parse"), "{err}");
    }
}
