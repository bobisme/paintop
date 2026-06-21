//! The M0 keystone end-to-end conformance test: "an agent edits a blemish".
//!
//! This drives the real public surface — the checked-in conformance plan
//! (`conformance/plans/blemish.json`), a deterministic procedural input fixture,
//! and the [`paintop_cpu::pipeline::run_plan`] orchestrator the CLI's `paintop
//! run` calls — to prove the M0 exit criteria (`plan.md` S19, S25;
//! `AGENT_VERIFICATION` S5/S6):
//!
//! 1. the fourteen-op non-SDF loop runs to completion with exit code 0;
//! 2. `assert.no_change_outside_mask` passes (the edit stayed inside the
//!    authorized ellipse, byte-for-byte outside it);
//! 3. a full evidence bundle is written (manifest, normalized plan, trace,
//!    assertions, graph, outputs, masks, contact sheet, diff);
//! 4. a second run produces a byte-identical output image and an identical plan
//!    semantic hash (reproducibility).
//!
//! The fixture is generated from a fixed formula (no external assets, no RNG) and
//! checked into `conformance/fixtures/`; the test regenerates it when missing and
//! always asserts the on-disk bytes match the formula, so the conformance input
//! can never silently drift.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use paintop_core::evidence::layout::{dirs, files};
use paintop_cpu::pipeline::{RunOutcome, run_plan};

/// Serializes the working-directory-sensitive runs. Cargo runs the test
/// functions in this file on parallel threads, and the conformance plan
/// references its fixture / encode target by workspace-relative path, so each run
/// must hold the process CWD for its duration. The mutex makes that mutual
/// exclusion explicit rather than relying on `--test-threads=1`.
static RUN_LOCK: Mutex<()> = Mutex::new(());

/// The fixture's dimensions: a small, fast, deterministic RGBA8 canvas.
const FIXTURE_WIDTH: u32 = 256;
const FIXTURE_HEIGHT: u32 = 192;

/// The workspace root, derived from this crate's manifest directory
/// (`<root>/crates/paintop-cpu`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root resolves")
}

/// A deterministic procedural RGBA8 image: a smooth two-axis gradient with an
/// off-center darker "blemish" lobe near the authorized ellipse, fully opaque.
///
/// Every channel is an exact integer function of the pixel coordinates, so the
/// raster — and therefore the encoded PNG — is byte-reproducible on any machine
/// without floating-point or RNG.
fn procedural_rgba8() -> Vec<u8> {
    let width = i64::from(FIXTURE_WIDTH);
    let height = i64::from(FIXTURE_HEIGHT);
    let pixel_count = usize::try_from(width * height).expect("extent fits in usize");
    let mut rgba = vec![0u8; pixel_count * 4];
    // Blemish lobe centered near the authorized ellipse center (128, 96).
    let (blemish_x, blemish_y) = (128_i64, 96_i64);
    // Map an integer level in [0, 255] to a byte (exact, never out of range).
    let to_byte = |level: i64| -> u8 { u8::try_from(level.clamp(0, 255)).unwrap_or(0) };
    for y in 0..height {
        for x in 0..width {
            // Base gradient: red rises with x, green with y, blue a fixed mid.
            let red = x * 255 / (width - 1);
            let green = y * 255 / (height - 1);
            let blue = 96_i64;
            // A bounded quadratic darkening inside a radius-32 disc around the
            // blemish center, computed in integers for exact reproducibility.
            let dx = x - blemish_x;
            let dy = y - blemish_y;
            let dist2 = dx * dx + dy * dy;
            let darken = if dist2 < 32 * 32 {
                // Up to ~70 levels at the center, fading to 0 at the rim.
                70 * (32 * 32 - dist2) / (32 * 32)
            } else {
                0
            };
            let base = usize::try_from((y * width + x) * 4).expect("offset fits in usize");
            rgba[base] = to_byte(red - darken);
            rgba[base + 1] = to_byte(green - darken);
            rgba[base + 2] = to_byte(blue - darken);
            rgba[base + 3] = 255;
        }
    }
    rgba
}

/// Ensure the checked-in fixture PNG exists and matches the formula, returning
/// its path. Regenerates it when absent (first run / fresh checkout that lost the
/// binary) and asserts byte-identity otherwise.
fn ensure_fixture(root: &Path) -> PathBuf {
    let rgba = procedural_rgba8();
    let png = paintop_core::evidence::encode_rgba(FIXTURE_WIDTH, FIXTURE_HEIGHT, &rgba)
        .expect("fixture raster encodes");
    let path = root
        .join("conformance")
        .join("fixtures")
        .join("blemish-input.png");
    if path.exists() {
        let on_disk = std::fs::read(&path).expect("read checked-in fixture");
        assert_eq!(
            on_disk, png,
            "checked-in fixture drifted from its generating formula; regenerate it"
        );
    } else {
        std::fs::create_dir_all(path.parent().expect("fixture parent")).expect("mkdir fixtures");
        std::fs::write(&path, &png).expect("write fixture");
    }
    path
}

/// Run a named conformance plan (`<plan>.json`) from the workspace root into a
/// fresh bundle dir, returning the outcome.
fn run_blemish(root: &Path, bundle: &Path) -> RunOutcome {
    run_named(root, "blemish", bundle)
}

/// Run the plan `conformance/plans/<name>.json` from the workspace root.
fn run_named(root: &Path, name: &str, bundle: &Path) -> RunOutcome {
    // The plan references its fixture and encode target by workspace-relative
    // path, so the run must execute with the workspace as the working directory.
    let plan = root
        .join("conformance")
        .join("plans")
        .join(format!("{name}.json"));
    std::fs::create_dir_all(root.join("conformance").join("out")).expect("mkdir conformance/out");
    // Hold the process CWD for the whole run; the lock serializes against the
    // sibling test so the relative fixture / encode paths resolve correctly.
    let _lock = RUN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _guard = CwdGuard::enter(root);
    run_plan(&plan, bundle).expect("the blemish plan runs without an internal error")
}

/// Scoped working-directory change, restored on drop, so the test never leaks a
/// changed CWD onto a sibling test (Cargo runs integration tests in one process
/// per file, but the guard keeps this honest).
struct CwdGuard {
    previous: PathBuf,
}

impl CwdGuard {
    fn enter(dir: &Path) -> Self {
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(dir).expect("set cwd to workspace root");
        Self { previous }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.previous);
    }
}

#[test]
fn blemish_loop_runs_green_with_a_complete_bundle() {
    let root = workspace_root();
    ensure_fixture(&root);
    let bundle = root.join("target").join("conformance-blemish-run");
    let _ = std::fs::remove_dir_all(&bundle);

    let outcome = run_blemish(&root, &bundle);

    // 1. The whole loop ran green.
    assert!(outcome.ok(), "run status: {:?}", outcome.status);
    assert_eq!(outcome.exit_code, 0);
    assert!(
        outcome.failures.is_empty(),
        "no assertion should fail: {:?}",
        outcome.failures
    );

    // 2. The evidence bundle is complete.
    for artifact in [
        files::MANIFEST,
        files::NORMALIZED_PLAN,
        files::GRAPH_DOT,
        files::TRACE,
        files::ASSERTIONS,
        files::CONTACT_SHEET,
    ] {
        assert!(
            bundle.join(artifact).is_file(),
            "missing bundle artifact {artifact}"
        );
    }
    assert!(
        bundle.join(dirs::OUTPUTS).join("final.png").is_file(),
        "the encoded export image is missing"
    );
    // The materialized mask and the edit-layer intermediate landed.
    assert!(
        bundle.join(dirs::MASKS).join("allowed-mask.png").is_file(),
        "the authorized mask was not materialized"
    );

    // 3. The no_change_outside_mask assertion is recorded as passed.
    let assertions =
        std::fs::read_to_string(bundle.join(files::ASSERTIONS)).expect("read assertions.json");
    let report: serde_json::Value =
        serde_json::from_str(&assertions).expect("assertions.json parses");
    let localized = report["assertions"]
        .as_array()
        .expect("assertions array")
        .iter()
        .find(|a| a["id"] == "localized")
        .expect("localized assertion present");
    assert_eq!(
        localized["status"], "passed",
        "no_change_outside_mask must pass: {localized}"
    );

    // The manifest records the same semantic hash and a zero exit code.
    let manifest =
        std::fs::read_to_string(bundle.join(files::MANIFEST)).expect("read manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(&manifest).expect("manifest parses");
    assert_eq!(manifest["exit_code"], 0);
    assert_eq!(manifest["status"], "success");
    assert_eq!(
        manifest["plan_semantic_hash"], outcome.plan_semantic_hash,
        "manifest hash must match the outcome"
    );
}

#[test]
fn second_run_is_byte_identical_and_hash_stable() {
    let root = workspace_root();
    ensure_fixture(&root);

    let bundle_a = root.join("target").join("conformance-blemish-a");
    let bundle_b = root.join("target").join("conformance-blemish-b");
    let _ = std::fs::remove_dir_all(&bundle_a);
    let _ = std::fs::remove_dir_all(&bundle_b);

    let a = run_blemish(&root, &bundle_a);
    let b = run_blemish(&root, &bundle_b);

    // The plan's semantic identity is stable across runs.
    assert_eq!(
        a.plan_semantic_hash, b.plan_semantic_hash,
        "plan semantic hash must be reproducible"
    );
    // The CPU-reference output content hash is stable across runs.
    assert_eq!(
        a.output_content_hash, b.output_content_hash,
        "output content hash must be reproducible"
    );
    assert!(a.output_content_hash.is_some(), "an output was produced");

    // And the encoded output PNGs are byte-for-byte identical.
    let out_a = std::fs::read(bundle_a.join(dirs::OUTPUTS).join("final.png")).expect("output a");
    let out_b = std::fs::read(bundle_b.join(dirs::OUTPUTS).join("final.png")).expect("output b");
    assert_eq!(
        out_a, out_b,
        "a re-run must produce a byte-identical output image"
    );
}

#[test]
fn leaking_variant_fails_with_exit_6_and_a_replay() {
    let root = workspace_root();
    ensure_fixture(&root);
    let bundle = root.join("target").join("conformance-blemish-leak");
    let _ = std::fs::remove_dir_all(&bundle);

    let outcome = run_named(&root, "blemish-leak", &bundle);

    // The leaking variant must fail the authorization-boundary assertion with the
    // stable assertion exit class (6), not crash and not silently pass.
    assert!(!outcome.ok(), "the leaking variant must not pass");
    assert_eq!(
        outcome.exit_code, 6,
        "a failed error-severity assertion maps to exit class 6"
    );
    assert!(
        outcome.failures.iter().any(|f| f == "localized"),
        "the localized no_change assertion must be reported failing: {:?}",
        outcome.failures
    );

    // assertions.json records the failure with its leaking metrics.
    let assertions =
        std::fs::read_to_string(bundle.join(files::ASSERTIONS)).expect("read assertions.json");
    let report: serde_json::Value =
        serde_json::from_str(&assertions).expect("assertions.json parses");
    let localized = report["assertions"]
        .as_array()
        .expect("assertions array")
        .iter()
        .find(|a| a["id"] == "localized")
        .expect("localized assertion present");
    assert_eq!(localized["status"], "failed", "{localized}");
    assert!(
        localized["metrics"]["changed_pixels_outside"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "the failure must record leaking pixels: {localized}"
    );

    // A minimal replay reproducing the failure is emitted under replays/.
    assert!(
        bundle.join(dirs::REPLAYS).join("localized.json").is_file(),
        "a minimal replay must be written for the failed assertion"
    );

    // The manifest stamps the assertion-failed status and exit code.
    let manifest =
        std::fs::read_to_string(bundle.join(files::MANIFEST)).expect("read manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(&manifest).expect("manifest parses");
    assert_eq!(manifest["status"], "assertion-failed");
    assert_eq!(manifest["exit_code"], 6);
    assert!(
        manifest["failures"]
            .as_array()
            .is_some_and(|f| f.iter().any(|x| x == "localized")),
        "manifest must list the failed assertion"
    );
}
