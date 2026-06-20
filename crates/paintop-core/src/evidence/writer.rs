//! The evidence-bundle writer: lay out the directory and atomically write the
//! canonical artifacts (`plan.md` §15.1, `AGENT_VERIFICATION` §5).
//!
//! This bone owns the **manifest + canonical artifact** half of the bundle: it
//! creates the directory skeleton, writes `manifest.json` and
//! `normalized-plan.json` atomically (temp-then-rename), and exposes an atomic
//! JSON helper ([`BundleWriter::write_artifact`]) the later trace / metrics /
//! assertion stages reuse. Every canonical artifact is serialized through the IR
//! [`to_canonical_bytes`] emitter (sorted keys, single float format) — we never
//! hash or persist raw `serde_json` output — so a re-run produces byte-identical
//! files and the bundle diffs cleanly.
//!
//! Atomicity is the load-bearing property (`plan.md` §15.1): a crash mid-write
//! leaves either no file or a complete file, never a truncated canonical
//! artifact. See [`atomic`](crate::evidence::atomic).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use paintop_ir::{Plan, normalized_value, semantic_hash, to_canonical_bytes};

use crate::evidence::atomic::write_atomic;
use crate::evidence::error::{BundleError, BundleResult};
use crate::evidence::layout::BundleLayout;
use crate::evidence::manifest::BundleManifest;

/// Writes an evidence bundle to a directory.
///
/// A writer is bound to one bundle root. [`BundleWriter::create`] lays out the
/// directory skeleton (root + well-known subdirectories); the `write_*` methods
/// then publish individual artifacts atomically. The writer is intentionally
/// thin: it does not decide *what* a run produced, only *how* artifacts land on
/// disk completely and canonically.
#[derive(Debug, Clone)]
pub struct BundleWriter {
    layout: BundleLayout,
}

impl BundleWriter {
    /// Create the bundle directory skeleton at `root` and return a writer bound
    /// to it.
    ///
    /// Creates the root and every well-known subdirectory
    /// ([`SUBDIRS`](crate::evidence::layout::SUBDIRS)); existing directories are
    /// reused (the operation is idempotent). No artifact files are written yet —
    /// missing optional artifacts stay absent until their stage materializes
    /// them.
    ///
    /// # Errors
    /// Returns [`BundleError::Io`] if the root or any subdirectory cannot be
    /// created.
    pub fn create(root: impl Into<PathBuf>) -> BundleResult<Self> {
        let layout = BundleLayout::new(root);
        let root = layout.root();
        fs::create_dir_all(root)
            .map_err(|e| BundleError::io_source(root, "creating bundle root directory", e))?;
        for dir in layout.subdirs() {
            fs::create_dir_all(&dir)
                .map_err(|e| BundleError::io_source(&dir, "creating bundle subdirectory", e))?;
        }
        Ok(Self { layout })
    }

    /// The layout describing where artifacts live under this bundle.
    #[must_use]
    pub const fn layout(&self) -> &BundleLayout {
        &self.layout
    }

    /// Atomically write `manifest.json`.
    ///
    /// The manifest is serialized to a [`Value`] and then through the canonical
    /// emitter, so it is byte-stable across runs. Because the timestamp and
    /// platform live in the manifest as *provenance* (never folded into the
    /// `plan_semantic_hash`), this canonical form still excludes wall-clock time
    /// from the bundle's semantic identity.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the manifest cannot be canonicalized
    /// or [`BundleError::Io`] if the atomic write fails.
    pub fn write_manifest(&self, manifest: &BundleManifest) -> BundleResult<()> {
        Self::write_json(&self.layout.manifest(), manifest)
    }

    /// Atomically write `normalized-plan.json` — the exact graph executed —
    /// alongside returning its `blake3:…` semantic hash.
    ///
    /// The plan is normalized (`IR_SPEC` §17: defaults resolved, prose stripped)
    /// and emitted in canonical form, so the on-disk artifact re-parses to a plan
    /// with the *same* semantic hash. Returning the hash lets the caller stamp it
    /// into the manifest without recomputing it.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the plan cannot be normalized /
    /// canonicalized or [`BundleError::Io`] if the atomic write fails.
    pub fn write_normalized_plan(&self, plan: &Plan) -> BundleResult<String> {
        let value = normalized_value(plan)
            .map_err(|e| BundleError::serialize("normalizing plan for the bundle", e))?;
        Self::write_canonical_value(&self.layout.normalized_plan(), &value)?;
        let hash = semantic_hash(plan)
            .map_err(|e| BundleError::serialize("computing plan semantic hash", e))?;
        Ok(hash.to_string())
    }

    /// Atomically write a `serde`-serializable value as canonical JSON to a
    /// bundle-relative `path`.
    ///
    /// This is the shared entry point for the later trace / metrics / assertion
    /// stages: hand it the artifact-relative name and a serializable payload and
    /// it lands atomically and canonically.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the value cannot be canonicalized or
    /// [`BundleError::Io`] if the atomic write fails.
    pub fn write_artifact<T: Serialize>(
        &self,
        relative: impl AsRef<Path>,
        value: &T,
    ) -> BundleResult<()> {
        let path = self.layout.join(relative);
        Self::write_json(&path, value)
    }

    /// Atomically write raw `bytes` to a bundle-relative `path` (for binary
    /// artifacts such as images, where there is no canonical JSON form).
    ///
    /// # Errors
    /// Returns [`BundleError::Io`] if the atomic write fails.
    pub fn write_bytes(&self, relative: impl AsRef<Path>, bytes: &[u8]) -> BundleResult<()> {
        let path = self.layout.join(relative);
        write_atomic(&path, bytes)
    }

    /// Atomically write the `assertions.json` report (`AGENT_VERIFICATION` §5.3).
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the report cannot be canonicalized or
    /// [`BundleError::Io`] if the atomic write fails.
    pub fn write_assertions(
        &self,
        report: &crate::evidence::assertions::AssertionReport,
    ) -> BundleResult<()> {
        self.write_artifact(crate::evidence::assertions::AssertionReport::PATH, report)
    }

    /// Atomically write a minimal replay under `replays/<assertion-id>.json`
    /// (`AGENT_VERIFICATION` §5.4) and return its bundle-relative path.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the replay cannot be canonicalized or
    /// [`BundleError::Io`] if the atomic write fails.
    pub fn write_replay(
        &self,
        replay: &crate::evidence::replay::MinimalReplay,
    ) -> BundleResult<String> {
        let relative = crate::evidence::replay::MinimalReplay::path_for(&replay.spec.assertion);
        self.write_artifact(&relative, replay)?;
        Ok(relative)
    }

    /// Encode and atomically write the before/after/diff `contact-sheet.png`
    /// (`plan.md` §15.1), returning its bundle-relative path.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the sheet's raster is degenerate
    /// (cannot encode) or [`BundleError::Io`] if the atomic write fails.
    pub fn write_contact_sheet(
        &self,
        sheet: &crate::evidence::contact::ContactSheet,
    ) -> BundleResult<String> {
        let png = sheet.encode_png().ok_or_else(|| {
            BundleError::serialize(
                "encoding the contact sheet",
                paintop_ir::Error::new(
                    paintop_ir::ErrorClass::Export,
                    crate::evidence::error::E_BUNDLE_SERIALIZE,
                    "contact sheet raster has zero area".to_owned(),
                ),
            )
        })?;
        let relative = crate::evidence::contact::ContactSheet::PATH;
        self.write_bytes(relative, &png)?;
        Ok(relative.to_owned())
    }

    /// Serialize `value` to canonical JSON and atomically write it to an absolute
    /// `path`.
    fn write_json<T: Serialize>(path: &Path, value: &T) -> BundleResult<()> {
        let json = serde_json::to_value(value).map_err(|e| {
            BundleError::serialize(
                "serializing artifact to a json value",
                paintop_ir::Error::new(
                    paintop_ir::ErrorClass::Export,
                    crate::evidence::error::E_BUNDLE_SERIALIZE,
                    e.to_string(),
                ),
            )
        })?;
        Self::write_canonical_value(path, &json)
    }

    /// Canonicalize a [`Value`] and atomically write it to an absolute `path`.
    fn write_canonical_value(path: &Path, value: &Value) -> BundleResult<()> {
        let bytes = to_canonical_bytes(value)
            .map_err(|e| BundleError::serialize("canonicalizing artifact bytes", e))?;
        write_atomic(path, &bytes)
    }
}
