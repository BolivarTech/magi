// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Verified `.xz` compression of a finished log file.
//!
//! # Verified, and why that is not paranoia
//!
//! The compressed bytes are written to a temporary, read back, decompressed and
//! compared against the original **before** anything is renamed. That ordering
//! is what lets a failure halfway through cost nothing: the original is still
//! there, the temporary is garbage nobody will read, and the next run tries
//! again. Compressing in place, or renaming before verifying, turns a partial
//! write into permanent data loss on the one file that was supposed to be the
//! record.

use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use lzma_rust2::{XzOptions, XzReader, XzWriter};

use crate::logging::LoggingError;

/// Compression preset. 6 is the xz default: the knee of the ratio/time curve.
const XZ_PRESET: u32 = 6;
/// Extension appended to the source name for the compressed file.
const XZ_EXTENSION: &str = "xz";
/// Restrictive mode for every file this module creates (REQ-L65).
#[cfg(unix)]
const OWNER_ONLY_MODE: u32 = 0o600;

/// Compresses `src` to `<src>.xz`, verifying the round trip before renaming.
///
/// # Parameters
///
/// * `src` — the finished log file. It is never modified, and never removed by
///   this function: deleting the original is the caller's decision.
/// * `dst_tmp` — where the compressed bytes are staged. Must be on the same
///   filesystem as `src`, or the final rename is not atomic.
///
/// # Returns
///
/// `Ok(())` once `<src>.xz` exists and has been proven to decompress to the
/// original bytes.
///
/// # Errors
///
/// * [`LoggingError::Compress`] if compression, the read-back or the comparison
///   fails. The original is untouched and `<src>.xz` does not exist.
/// * [`LoggingError::Write`] if the staged file cannot be created or renamed.
///
/// # Complexity
///
/// `O(n)` over the file, with two passes: one to compress, one to verify.
pub fn compress_verified(src: &Path, dst_tmp: &Path) -> Result<(), LoggingError> {
    let original = fs::read(src).map_err(|e| LoggingError::Compress {
        path: src.to_path_buf(),
        source: e,
    })?;

    compress_to(&original, dst_tmp)?;
    verify_round_trip(dst_tmp, &original)?;

    let final_path = compressed_path(src);
    fs::rename(dst_tmp, &final_path).map_err(|e| LoggingError::Write {
        path: final_path,
        source: e,
    })?;
    Ok(())
}

/// Writes `bytes` compressed into `dst`, with owner-only permissions.
///
/// # Errors
///
/// [`LoggingError::Compress`] on any I/O or encoder failure.
///
/// # Complexity
///
/// `O(n)`.
fn compress_to(bytes: &[u8], dst: &Path) -> Result<(), LoggingError> {
    let fail = |e: std::io::Error| LoggingError::Compress {
        path: dst.to_path_buf(),
        source: e,
    };

    let file = fs::File::create(dst).map_err(fail)?;
    restrict(dst)?;
    let mut writer = XzWriter::new(file, XzOptions::with_preset(XZ_PRESET)).map_err(fail)?;
    writer.write_all(bytes).map_err(fail)?;
    let mut finished = writer.finish().map_err(fail)?;
    finished.flush().map_err(fail)?;
    Ok(())
}

/// Reads `staged` back, decompresses it and compares against `expected`.
///
/// # Errors
///
/// [`LoggingError::Compress`] when the read-back fails or the bytes differ.
///
/// # Complexity
///
/// `O(n)`.
fn verify_round_trip(staged: &Path, expected: &[u8]) -> Result<(), LoggingError> {
    let fail = |e: std::io::Error| LoggingError::Compress {
        path: staged.to_path_buf(),
        source: e,
    };

    let file = fs::File::open(staged).map_err(fail)?;
    let mut reader = XzReader::new(file, false);
    let mut back = Vec::with_capacity(expected.len());
    reader.read_to_end(&mut back).map_err(fail)?;

    if back == expected {
        return Ok(());
    }
    Err(LoggingError::Compress {
        path: staged.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "the compressed file does not decompress to the original bytes",
        ),
    })
}

/// Applies owner-only permissions to a file this module created (REQ-L65).
///
/// # Errors
///
/// [`LoggingError::Write`] if the mode cannot be set.
///
/// # Platform
///
/// On Windows this is a no-op: the file inherits the ACL of the `.magi/`
/// directory, which `magi init` already creates restricted. Stated rather than
/// silently skipped, because "0600 everywhere" would be a claim this function
/// does not keep on that platform.
///
/// # Complexity
///
/// `O(1)`.
fn restrict(path: &Path) -> Result<(), LoggingError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(OWNER_ONLY_MODE)).map_err(|e| {
            LoggingError::Write {
                path: path.to_path_buf(),
                source: e,
            }
        })?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// The `.xz` name a source file compresses into.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn compressed_path(src: &Path) -> PathBuf {
    let mut name = src.as_os_str().to_os_string();
    name.push(".");
    name.push(XZ_EXTENSION);
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    fn write_src(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, body).expect("writing a fixture file");
        p
    }

    #[test]
    fn a_verified_compression_round_trips_to_the_original_bytes() {
        let dir = tempdir().unwrap();
        let body = b"2026-08-14T00:00:00Z INFO magi_rs::agent: hello\n".repeat(200);
        let src = write_src(dir.path(), "magi-2026-08-14.log", &body);

        compress_verified(&src, &dir.path().join("staging.tmp")).unwrap();

        let xz = compressed_path(&src);
        assert!(xz.exists(), "the compressed file must exist after success");
        assert!(src.exists(), "compress_verified never removes the original");

        let mut back = Vec::new();
        XzReader::new(fs::File::open(&xz).unwrap(), false)
            .read_to_end(&mut back)
            .unwrap();
        assert_eq!(back, body, "the round trip must be exact");
    }

    #[test]
    fn the_compressed_file_is_smaller_than_a_repetitive_original() {
        // Without this the round-trip test above passes just as well against an
        // implementation that copies the bytes and calls it compression.
        let dir = tempdir().unwrap();
        let body = b"the same line over and over\n".repeat(500);
        let src = write_src(dir.path(), "magi-2026-08-15.log", &body);
        compress_verified(&src, &dir.path().join("staging.tmp")).unwrap();
        let compressed = fs::metadata(compressed_path(&src)).unwrap().len();
        assert!(
            compressed < body.len() as u64 / 2,
            "500 identical lines compressed to {compressed} of {} bytes",
            body.len()
        );
    }

    #[test]
    fn an_interrupted_compression_leaves_the_original_intact_and_no_xz_behind() {
        let dir = tempdir().unwrap();
        let body = b"payload".to_vec();
        let src = write_src(dir.path(), "magi-2026-08-16.log", &body);

        // A staging path inside a directory that does not exist: creating the
        // temporary fails, which is the earliest point compression can break.
        let doomed = dir.path().join("no-such-dir").join("staging.tmp");
        let outcome = compress_verified(&src, &doomed);

        assert!(
            outcome.is_err(),
            "the failure must be reported, not swallowed"
        );
        assert_eq!(fs::read(&src).unwrap(), body, "the original must survive");
        assert!(
            !compressed_path(&src).exists(),
            "nothing is renamed into place when compression failed"
        );
    }

    #[test]
    fn a_missing_source_is_an_error_rather_than_an_empty_archive() {
        let dir = tempdir().unwrap();
        let outcome = compress_verified(
            &dir.path().join("magi-2026-08-17.log"),
            &dir.path().join("staging.tmp"),
        );
        assert!(outcome.is_err());
    }

    #[test]
    fn verification_rejects_a_staged_file_that_is_not_valid_xz_at_all() {
        let dir = tempdir().unwrap();
        let staged = dir.path().join("garbage.tmp");
        fs::write(&staged, b"this is not an xz stream").unwrap();
        assert!(verify_round_trip(&staged, b"payload").is_err());
    }

    #[test]
    fn verification_rejects_a_staged_file_that_decompresses_to_different_bytes() {
        let dir = tempdir().unwrap();
        let staged = dir.path().join("wrong.tmp");
        compress_to(b"what was actually compressed", &staged).unwrap();
        assert!(
            verify_round_trip(&staged, b"what the caller expected").is_err(),
            "a mismatch must be reported, not accepted"
        );
        // And the same staged file verifies against its real contents, so the
        // rejection above is about the comparison and not about the reader.
        assert!(verify_round_trip(&staged, b"what was actually compressed").is_ok());
    }

    #[test]
    fn compress_verified_actually_calls_the_verification_before_renaming() {
        // **No behavioural test can catch this, and that is why it is a source
        // check.** With a working compressor the round trip always succeeds, so
        // deleting the verification changes nothing any assertion can observe —
        // it was run as a mutation and every test stayed green. The guarantee
        // REQ-L14 buys only shows up when compression is corrupt, which this
        // process cannot produce on demand.
        let src = include_str!("xz.rs");
        let body = src
            .split("pub fn compress_verified")
            .nth(1)
            .expect("the function is in this file");
        let body = body
            .split(
                "
}",
            )
            .next()
            .expect("its body ends somewhere");
        let verify_at = body
            .find("verify_round_trip(")
            .expect("verification is called");
        let rename_at = body.find("fs::rename(").expect("the rename is there");
        assert!(
            verify_at < rename_at,
            "the round trip must be verified BEFORE the rename, or a corrupt              archive replaces a good original"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_compressed_file_is_readable_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        let src = write_src(dir.path(), "magi-2026-08-18.log", b"x");
        compress_verified(&src, &dir.path().join("staging.tmp")).unwrap();
        let mode = fs::metadata(compressed_path(&src))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, OWNER_ONLY_MODE, "REQ-L65 wants 0600");
    }
}
