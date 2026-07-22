//! ring_residency — NVMe-as-RAM kernel host (RingKit kernels/ silicon layer, D9).
//!
//! Generic byte-addressable file residency: open/read/write/close over exact sector-aligned
//! ranges, durable (write-through/unbuffered where the platform supports it), byte-for-byte
//! round-trip. NO key/value/attention/position/LSH semantics here — those are a consumer's
//! concern (a KV-cache tier, weight residency, content-addressed memory), each its own later
//! phase built ON this module, not baked into it.
//!
//! Depends on gevhv's verified `tkv` crate for the actual unbuffered I/O (`open_unbuffered`,
//! `open_unbuffered_ro`, `SECTOR`) instead of reimplementing it — same reuse discipline
//! `kernels/rust` already established for its `gevhv`/`zring` dependency. `tkv::store`'s own
//! cross-platform behavior: Windows gets `FILE_FLAG_NO_BUFFERING|FILE_FLAG_WRITE_THROUGH` (true
//! page-cache bypass); other platforms get a plain buffered file (same addressing, same
//! correctness — the unbuffered SPEED property is Windows-only today, stated honestly, not hidden).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use tkv::store::{open_unbuffered, SECTOR};

#[inline]
fn check_aligned(who: &str, offset: u64, len: u64) -> PyResult<()> {
    let sector = SECTOR as u64;
    if offset % sector != 0 {
        return Err(PyValueError::new_err(format!(
            "{who}: offset {offset} is not sector-aligned ({sector} bytes)"
        )));
    }
    if len % sector != 0 {
        return Err(PyValueError::new_err(format!(
            "{who}: length {len} is not a sector multiple ({sector} bytes)"
        )));
    }
    Ok(())
}

/// One open residency file. Persistent read+write handle for the life of the object — no
/// per-call open/close, matching the KvTier lesson already learned elsewhere in this project
/// (open cost dominates when paid per call).
#[pyclass]
struct Residency {
    file: File,
    path: String,
}

#[pymethods]
impl Residency {
    /// Create-or-open `path`, sized to exactly `size` bytes (a sector multiple). If `path`
    /// already exists and is LARGER than `size`, this refuses (raises) rather than truncating —
    /// `tkv::store::open_unbuffered` calls `set_len(size)` unconditionally, which is safe for
    /// gevhv's own callers (always a consistent, computed size) but would silently destroy data
    /// for a general-purpose caller who passes the wrong size. Growing an existing file (or
    /// creating a new one) is fine; shrinking one is not, ever, implicitly.
    #[new]
    fn open(path: String, size: u64) -> PyResult<Self> {
        if size % (SECTOR as u64) != 0 {
            return Err(PyValueError::new_err(format!(
                "Residency.open: size {size} is not a sector multiple ({SECTOR} bytes)"
            )));
        }
        let p = Path::new(&path);
        if let Ok(meta) = std::fs::metadata(p) {
            if meta.len() > size {
                return Err(PyValueError::new_err(format!(
                    "Residency.open({path}): existing file is {} bytes, larger than the requested \
                     size {size} — refusing to truncate. Pass the existing size (or larger).",
                    meta.len()
                )));
            }
        }
        let file = open_unbuffered(p, size)
            .map_err(|e| PyValueError::new_err(format!("Residency.open({path}): {e}")))?;
        Ok(Residency { file, path })
    }

    /// Durable positioned write. `offset` and `data.len()` must both be sector multiples.
    fn write(&mut self, offset: u64, data: &[u8]) -> PyResult<()> {
        check_aligned("Residency.write", offset, data.len() as u64)?;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| PyValueError::new_err(format!("Residency.write seek({offset}): {e}")))?;
        self.file
            .write_all(data)
            .map_err(|e| PyValueError::new_err(format!("Residency.write({offset}, {} bytes): {e}", data.len())))?;
        Ok(())
    }

    /// Positioned read of exactly `length` bytes at `offset`. Both must be sector multiples.
    fn read(&mut self, py: Python<'_>, offset: u64, length: u64) -> PyResult<Py<PyBytes>> {
        check_aligned("Residency.read", offset, length)?;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| PyValueError::new_err(format!("Residency.read seek({offset}): {e}")))?;
        let mut buf = vec![0u8; length as usize];
        self.file
            .read_exact(&mut buf)
            .map_err(|e| PyValueError::new_err(format!("Residency.read({offset}, {length}): {e}")))?;
        Ok(PyBytes::new_bound(py, &buf).into())
    }

    /// Explicit close (Rust drops the handle on GC too; this exists for callers that want a
    /// deterministic lifecycle point, e.g. a Python context manager).
    fn close(&mut self) -> PyResult<()> {
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("Residency({:?})", self.path)
    }
}

/// The sector size every offset/length in this module must be a multiple of. Exposed so Python
/// callers can align their own ranges rather than hardcoding 4096.
#[pyfunction]
fn sector_size() -> u64 {
    SECTOR as u64
}

/// Liveness probe (matches ring_rust's `ring_rust_probe` convention): confirms the extension
/// module loaded, before the D9 self-test runs the real round-trip checks.
#[pyfunction]
fn ring_residency_probe() -> i32 {
    42
}

#[pymodule]
fn ring_residency(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Residency>()?;
    m.add_function(wrap_pyfunction!(sector_size, m)?)?;
    m.add_function(wrap_pyfunction!(ring_residency_probe, m)?)?;
    Ok(())
}
