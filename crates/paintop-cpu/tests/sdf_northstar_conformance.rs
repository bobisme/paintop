//! The M1.5 SDF north-star end-to-end conformance test: "an agent edits a
//! blemish, authorizing the region through the signed-distance feather chain".
//!
//! This is the scenario the M0 decision deferred (`M0_DECISIONS` D1, "SDF lands
//! second"; `plan.md` S25 original SDF variant; `OP_CATALOG` S4; `ALIEN_OPS`
//! S2). It drives the real public surface — the checked-in conformance plan
//! (`conformance/plans/blemish-sdf.json`), the shared deterministic fixture, and
//! the [`paintop_cpu::pipeline::run_plan`] orchestrator the CLI's `paintop run`
//! calls — to prove the M1.5 exit criteria.
//!
//! Where the M0 blemish keystone authorizes its edit with an analytic *soft*
//! ellipse, this scenario builds the identical authorization boundary through the
//! new SDF mask calculus instead:
//!
//! * `mask.ellipse@1`  — a HARD ellipse coverage mask;
//! * `mask.to_sdf@1`   — exact Euclidean SDF, negative-inside (`IR_SPEC` S7.4);
//! * `sdf.offset@1`    — grow the region by a physical pixel offset;
//! * `sdf.to_mask@1`   — reconstruct a smoothstep-feathered coverage mask.
//!
//! That SDF-derived mask is the single authorization boundary feeding both
//! `composite.masked_replace@1` and `assert.no_change_outside_mask@1`. The test
//! asserts:
//!
//! 1. the loop runs to completion with exit code 0;
//! 2. `assert.no_change_outside_mask` passes (the edit stayed inside the
//!    SDF-authorized region, byte-for-byte outside it);
//! 3. a full evidence bundle is written (manifest, normalized plan, trace,
//!    assertions, graph, outputs, masks, contact sheet);
//! 4. a second run produces a byte-identical output image and identical plan +
//!    output content hashes (the reproducible rerun hash).
//!
//! The fixture is shared with the blemish/banner scenarios (a fixed integer
//! formula, no external assets, no RNG) and regenerated when missing.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use paintop_core::evidence::layout::{dirs, files};
use paintop_cpu::pipeline::{RunOutcome, run_plan};

/// Serializes the working-directory-sensitive runs in this file (the conformance
/// plan references its fixture / encode target by workspace-relative path, so
/// each run must hold the process CWD for its duration).
static RUN_LOCK: Mutex<()> = Mutex::new(());

/// The shared fixture's dimensions: a small, fast, deterministic RGBA8 canvas.
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

/// The shared deterministic procedural RGBA8 fixture (a smooth two-axis gradient
/// with an off-center darker lobe), byte-identical to the blemish/banner
/// scenarios' so every conformance loop reads the same checked-in input.
fn procedural_rgba8() -> Vec<u8> {
    let width = i64::from(FIXTURE_WIDTH);
    let height = i64::from(FIXTURE_HEIGHT);
    let pixel_count = usize::try_from(width * height).expect("extent fits in usize");
    let mut rgba = vec![0u8; pixel_count * 4];
    let (blemish_x, blemish_y) = (128_i64, 96_i64);
    let to_byte = |level: i64| -> u8 { u8::try_from(level.clamp(0, 255)).unwrap_or(0) };
    for y in 0..height {
        for x in 0..width {
            let red = x * 255 / (width - 1);
            let green = y * 255 / (height - 1);
            let blue = 96_i64;
            let dx = x - blemish_x;
            let dy = y - blemish_y;
            let dist2 = dx * dx + dy * dy;
            let darken = if dist2 < 32 * 32 {
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
/// its path. Regenerates it when absent and asserts byte-identity otherwise.
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

/// Run the SDF north-star plan from the workspace root into a fresh bundle dir.
fn run_sdf(root: &Path, bundle: &Path) -> RunOutcome {
    let plan = root
        .join("conformance")
        .join("plans")
        .join("blemish-sdf.json");
    std::fs::create_dir_all(root.join("conformance").join("out")).expect("mkdir conformance/out");
    let _lock = RUN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _guard = CwdGuard::enter(root);
    run_plan(&plan, bundle).expect("the SDF north-star plan runs without an internal error")
}

/// Scoped working-directory change, restored on drop.
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
fn sdf_northstar_loop_runs_green_with_a_complete_bundle() {
    let root = workspace_root();
    ensure_fixture(&root);
    let bundle = root.join("target").join("conformance-sdf-run");
    let _ = std::fs::remove_dir_all(&bundle);

    let outcome = run_sdf(&root, &bundle);

    // 1. The whole SDF feather-chain loop ran green.
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
    // The SDF-derived authorized mask was materialized.
    assert!(
        bundle.join(dirs::MASKS).join("allowed-mask.png").is_file(),
        "the SDF-derived authorized mask was not materialized"
    );

    // 3. The no_change_outside_mask assertion is recorded as passed with zero
    //    pixels changed outside the SDF-authorized region.
    let report = read_assertions(&bundle);
    let localized = find_assertion(&report, "localized");
    assert_eq!(
        localized["status"], "passed",
        "no_change_outside_mask must pass: {localized}"
    );
    assert_eq!(
        localized["metrics"]["changed_pixels_outside"]
            .as_u64()
            .expect("changed_pixels_outside metric"),
        0,
        "nothing may change outside the SDF-authorized region: {localized}"
    );

    // The manifest records the same semantic hash and a zero exit code.
    let manifest = read_manifest(&bundle);
    assert_eq!(manifest["exit_code"], 0);
    assert_eq!(manifest["status"], "success");
    assert_eq!(
        manifest["plan_semantic_hash"], outcome.plan_semantic_hash,
        "manifest hash must match the outcome"
    );
}

#[test]
fn sdf_northstar_second_run_is_byte_identical_and_hash_stable() {
    let root = workspace_root();
    ensure_fixture(&root);

    let bundle_a = root.join("target").join("conformance-sdf-a");
    let bundle_b = root.join("target").join("conformance-sdf-b");
    let _ = std::fs::remove_dir_all(&bundle_a);
    let _ = std::fs::remove_dir_all(&bundle_b);

    let a = run_sdf(&root, &bundle_a);
    let b = run_sdf(&root, &bundle_b);

    // The plan's semantic identity is stable across runs.
    assert_eq!(
        a.plan_semantic_hash, b.plan_semantic_hash,
        "plan semantic hash must be reproducible"
    );
    // The CPU-reference output content hash is stable across runs (the
    // reproducible rerun hash the M1.5 exit gate requires).
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

/// Read and parse `assertions.json` from a bundle.
fn read_assertions(bundle: &Path) -> serde_json::Value {
    let raw =
        std::fs::read_to_string(bundle.join(files::ASSERTIONS)).expect("read assertions.json");
    serde_json::from_str(&raw).expect("assertions.json parses")
}

/// Read and parse `manifest.json` from a bundle.
fn read_manifest(bundle: &Path) -> serde_json::Value {
    let raw = std::fs::read_to_string(bundle.join(files::MANIFEST)).expect("read manifest.json");
    serde_json::from_str(&raw).expect("manifest.json parses")
}

/// Find the assertion with the given id in a parsed `assertions.json` report.
fn find_assertion<'a>(report: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
    report["assertions"]
        .as_array()
        .expect("assertions array")
        .iter()
        .find(|a| a["id"] == id)
        .unwrap_or_else(|| panic!("assertion {id} present"))
}
