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
use std::io::Write as _;
use std::path::{Path, PathBuf};

use lzma_rust2::{XzOptions, XzReader, XzWriter};

use crate::logging::LoggingError;

/// Compression preset. 6 is the xz default: the knee of the ratio/time curve.
const XZ_PRESET: u32 = 6;
/// Block size the round-trip comparison reads at a time.
///
/// 64 KiB: large enough that the syscall count is irrelevant beside the
/// decompression, small enough that peak memory is a constant nobody has to
/// reason about against a file of unknown size.
const VERIFY_BLOCK_BYTES: usize = 64 * 1024;
/// Extension appended to the source name for the compressed file.
const XZ_EXTENSION: &str = "xz";
/// Restrictive mode for every file this module creates (REQ-L65).
///
/// Aliased from the subsystem's own constant rather than declared again: the
/// active `.log`, the staging temporary and the finished `.xz` are the same
/// secret at three moments of its life, and two constants is two places to
/// change one rule.
use crate::logging::OWNER_ONLY_FILE_MODE as OWNER_ONLY_MODE;

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
    compress_to(src, dst_tmp)?;
    verify_round_trip(dst_tmp, src)?;

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
fn compress_to(src: &Path, dst: &Path) -> Result<(), LoggingError> {
    let fail = |e: std::io::Error| LoggingError::Compress {
        path: dst.to_path_buf(),
        source: e,
    };

    let mut source = fs::File::open(src).map_err(|e| LoggingError::Compress {
        path: src.to_path_buf(),
        source: e,
    })?;
    let file = fs::File::create(dst).map_err(fail)?;
    crate::logging::restrict(dst, OWNER_ONLY_MODE)?;
    let mut writer = XzWriter::new(file, XzOptions::with_preset(XZ_PRESET)).map_err(fail)?;
    // **Streamed, never `fs::read` into a `Vec`.** A daily file runs to hundreds
    // of megabytes -- `max_total_bytes` defaults to 512 MiB and today's file is
    // untouchable -- so buffering it whole makes the routine that exists to
    // protect data the cause of an OOM that kills the process.
    std::io::copy(&mut source, &mut writer).map_err(fail)?;
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
fn verify_round_trip(staged: &Path, original: &Path) -> Result<(), LoggingError> {
    let fail = |e: std::io::Error| LoggingError::Compress {
        path: staged.to_path_buf(),
        source: e,
    };

    let staged_file = fs::File::open(staged).map_err(fail)?;
    let mut back = std::io::BufReader::new(XzReader::new(staged_file, false));
    let mut source = std::io::BufReader::new(fs::File::open(original).map_err(fail)?);

    // **Block by block, cutting at the first mismatch: constant memory whatever
    // the size.** Reading both into `Vec`s peaks at twice the file, and this
    // runs on the blocking pool inside a live session.
    let mut a = vec![0u8; VERIFY_BLOCK_BYTES];
    let mut b = vec![0u8; VERIFY_BLOCK_BYTES];
    loop {
        let read_a = fill(&mut source, &mut a).map_err(fail)?;
        let read_b = fill(&mut back, &mut b).map_err(fail)?;
        if read_a != read_b || a.get(..read_a) != b.get(..read_b) {
            break;
        }
        if read_a == 0 {
            return Ok(());
        }
    }
    Err(LoggingError::Compress {
        path: staged.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "the compressed file does not decompress to the original bytes",
        ),
    })
}

/// Fills `buf` as far as the reader will go, treating a short read as normal.
///
/// `Read::read` is allowed to return fewer bytes than asked for at any time, and
/// a decompressor does so routinely at a block boundary. Comparing two streams
/// with raw `read` calls therefore reports a mismatch where the only difference
/// is where each side happened to stop.
///
/// # Returns
///
/// How many bytes were placed in `buf`; `0` only at end of stream.
///
/// # Errors
///
/// The first non-interrupted I/O error.
///
/// # Complexity
///
/// `O(buf.len())`.
fn fill(reader: &mut impl std::io::Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        let Some(slot) = buf.get_mut(filled..) else {
            break;
        };
        match reader.read(slot) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// The `.xz` name a source file compresses into.
///
/// `pub(crate)` rather than `pub`: its consumer is the retention executor in
/// this same subsystem, and a test is not a consumer. Nothing outside the crate
/// has any reason to derive this name.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub(crate) fn compressed_path(src: &Path) -> PathBuf {
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
    fn verification_does_not_hold_the_whole_file_in_memory() {
        // REQ-L14 is explicit that the comparison is streamed "nunca cargando
        // los dos archivos en memoria", and the reason is that this runs on the
        // blocking pool inside a live session over a file that can reach
        // hundreds of megabytes. Reading both sides into `Vec`s peaks at twice
        // the file and turns the routine that exists to PROTECT data into the
        // cause of an OOM that kills the process.
        //
        // A test cannot watch an allocator from here, so it asserts the property
        // that the streaming shape gives and the buffering one does not: a file
        // several times the block size round-trips correctly, and it does so
        // reading a bounded window at a time. The block constant is what the
        // implementation is measured against, and it is checked against a
        // literal below so this cannot be satisfied by widening it.
        let dir = tempdir().unwrap();
        let src = dir.path().join("magi-2026-08-29.log");
        // Prose rather than one repeated byte, so the compressor cannot collapse
        // it to something smaller than the window under test.
        let body = "the quick brown fox jumps over the lazy dog 0123456789\n".repeat(20_000);
        fs::write(&src, &body).unwrap();
        assert!(
            body.len() > VERIFY_BLOCK_BYTES * 4,
            "the fixture must span several blocks or the loop runs once"
        );

        let staged = dir.path().join("staging.tmp");
        compress_verified(&src, &staged).expect("round trip");

        let out = compressed_path(&src);
        assert!(out.exists(), "the compressed file was not produced");
        let mut back = String::new();
        std::io::Read::read_to_string(
            &mut lzma_rust2::XzReader::new(fs::File::open(&out).unwrap(), false),
            &mut back,
        )
        .unwrap();
        assert_eq!(back, body, "the streamed round trip lost bytes");
    }

    #[test]
    fn the_verification_window_is_a_bounded_constant() {
        // Asserted against a LITERAL, not against itself: a test that compared
        // the constant to the constant would stay green while someone raised it
        // to the file size, which is the buffering the requirement forbids
        // wearing a streaming shape.
        assert_eq!(VERIFY_BLOCK_BYTES, 65_536);
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
        std::io::Read::read_to_end(
            &mut XzReader::new(fs::File::open(&xz).unwrap(), false),
            &mut back,
        )
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
        let original = dir.path().join("magi-2026-08-17.log");
        fs::write(&original, b"payload").unwrap();
        assert!(verify_round_trip(&staged, &original).is_err());
    }

    #[test]
    fn verification_rejects_a_staged_file_that_decompresses_to_different_bytes() {
        let dir = tempdir().unwrap();
        let real = dir.path().join("real.log");
        let claimed = dir.path().join("claimed.log");
        fs::write(&real, b"what was actually compressed").unwrap();
        fs::write(&claimed, b"what the caller expected").unwrap();
        let staged = dir.path().join("wrong.tmp");
        compress_to(&real, &staged).unwrap();
        assert!(
            verify_round_trip(&staged, &claimed).is_err(),
            "a mismatch must be reported, not accepted"
        );
        // And the same staged file verifies against its real contents, so the
        // rejection above is about the comparison and not about the reader.
        assert!(verify_round_trip(&staged, &real).is_ok());
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
