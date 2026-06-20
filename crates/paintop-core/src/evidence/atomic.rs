//! Atomic (temp-then-rename) file writes for evidence-bundle artifacts.
//!
//! Every canonical artifact in a bundle (`manifest.json`, `normalized-plan.json`,
//! …) must be written **atomically**: a crash mid-write must never leave a
//! truncated or partially serialized file in the bundle (`plan.md` §15.1
//! "Atomic writes (temp-then-rename)"). We achieve this the classic way: write
//! the full contents to a sibling temp file under the *same directory*, flush it
//! to the OS, then `rename` it onto the final path. `rename(2)` within one
//! filesystem is atomic, so an observer (or a re-run after a crash) sees either
//! the old file / nothing or the complete new file — never a half-written one.
//!
//! Writing the temp file as a *sibling* (not in `/tmp`) guarantees the rename
//! stays on the same filesystem, which is what makes it atomic; a cross-device
//! rename would silently degrade to a copy.

use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::evidence::error::{BundleError, BundleResult};

/// Atomically write `contents` to `path`.
///
/// The bytes are first written to a uniquely named sibling temp file in the same
/// directory, flushed and synced, then renamed onto `path`. On success `path`
/// holds exactly `contents`; on any error the temp file is removed and `path` is
/// left untouched. The parent directory must already exist.
///
/// # Errors
/// Returns a [`BundleError::Io`] if the parent directory is missing, the temp
/// file cannot be created/written/synced, or the final rename fails.
pub fn write_atomic(path: &Path, contents: &[u8]) -> BundleResult<()> {
    let parent = path.parent().ok_or_else(|| {
        BundleError::io_at(
            path,
            "destination path has no parent directory for atomic write",
        )
    })?;
    let temp = temp_sibling(path)?;

    // Scope the file so it is flushed and closed before the rename; on any
    // failure we remove the partial temp file so the directory stays clean.
    let write_result = write_and_sync(&temp, contents);
    if let Err(err) = write_result {
        // Best-effort cleanup: the original error is the one worth surfacing.
        let _ = fs::remove_file(&temp);
        return Err(err);
    }

    if let Err(err) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(BundleError::io_source(
            path,
            format!(
                "renaming temp file into place (parent: {})",
                parent.display()
            ),
            err,
        ));
    }
    Ok(())
}

/// Write `contents` to `temp`, flushing the buffer and syncing the file's data
/// to the OS before it is closed so the subsequent rename publishes complete
/// bytes.
fn write_and_sync(temp: &Path, contents: &[u8]) -> BundleResult<()> {
    let mut file =
        File::create(temp).map_err(|e| BundleError::io_source(temp, "creating temp file", e))?;
    file.write_all(contents)
        .map_err(|e| BundleError::io_source(temp, "writing temp file", e))?;
    file.flush()
        .map_err(|e| BundleError::io_source(temp, "flushing temp file", e))?;
    file.sync_all()
        .map_err(|e| BundleError::io_source(temp, "syncing temp file", e))?;
    Ok(())
}

/// Build a unique temp path that is a sibling of `path` (same directory, so the
/// rename stays on one filesystem and is therefore atomic).
///
/// Uniqueness comes from the process id plus a monotonic per-process counter, so
/// concurrent writers within or across processes never collide on the same temp
/// name.
fn temp_sibling(path: &Path) -> BundleResult<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let parent = path
        .parent()
        .ok_or_else(|| BundleError::io_at(path, "destination path has no parent directory"))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| BundleError::io_at(path, "destination path has no file name"))?;
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut temp_name = std::ffi::OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(".tmp.{pid}.{seq}"));
    Ok(parent.join(temp_name))
}

#[cfg(test)]
mod tests {
    use super::write_atomic;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!("paintop-atomic-{}-{tag}-{seq}", std::process::id()));
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn writes_complete_contents() {
        let dir = scratch_dir("write");
        let path = dir.join("artifact.json");
        write_atomic(&path, b"{\"a\":1}").expect("write");
        assert_eq!(fs::read(&path).expect("read"), b"{\"a\":1}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn overwrite_replaces_atomically_and_leaves_no_temp() {
        let dir = scratch_dir("overwrite");
        let path = dir.join("artifact.json");
        write_atomic(&path, b"old").expect("write 1");
        write_atomic(&path, b"newer-contents").expect("write 2");
        assert_eq!(fs::read(&path).expect("read"), b"newer-contents");
        // No sidecar temp file survives the rename.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_parent_directory_is_an_error_not_a_partial_write() {
        let dir = scratch_dir("missing-parent");
        let path = dir.join("does-not-exist").join("artifact.json");
        let err = write_atomic(&path, b"data").expect_err("must fail");
        // The error names the I/O failure code; nothing was written.
        assert_eq!(err.code(), super::super::error::E_BUNDLE_IO);
        assert!(!path.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
