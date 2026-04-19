//! `CallbackReader` — `std::io::{Read, Seek}` adapter over the C IO callback
//! struct. This is the seam the whole crate pivots on: the Ogg demuxer and
//! `OpusTags` parser both take `Read + Seek`, and the C++ shim only knows
//! about the `RopusFb2kIo` layout, so this type is where the two meet.
//!
//! The C struct is defined in `lib.rs` as `#[repr(C)]` and copied by value
//! into each reader (it's four function pointers + one `*mut c_void`), so
//! the Rust side doesn't have to worry about the caller mutating it after
//! `ropus_fb2k_open` returns.

use std::io::{self, Read, Seek, SeekFrom};

use crate::RopusFb2kIo;

/// Zero-sized marker type carried inside `io::Error` when the caller's
/// `abort` callback reported cancellation. The reader layer recovers it via
/// `downcast_ref` (walking the `Error::source()` chain so a wrapper layer
/// like the `ogg` crate can't hide it). Using a typed marker replaces the
/// older string-match ("aborted") heuristic, which would silently break if
/// any intermediate layer wrapped the error in its own `Display` impl.
#[derive(Debug)]
pub(crate) struct AbortTag;

impl std::fmt::Display for AbortTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("aborted")
    }
}

impl std::error::Error for AbortTag {}

/// `Read + Seek` over a caller-supplied C IO struct. Polls `abort` before
/// every operation so cancellation is responsive even inside long Ogg-page
/// walks.
///
/// `pos` is advanced on every successful `read` / `seek` so callers that
/// only hand us `read` + `abort` (no `seek`) still get a plausible `Seek`
/// response to `SeekFrom::Current(0)` — the Ogg crate relies on that for
/// stream-position queries.
pub(crate) struct CallbackReader {
    io: RopusFb2kIo,
    pos: u64,
    size: Option<u64>,
}

impl CallbackReader {
    /// Build a reader from a caller-supplied IO struct, probing `size` once
    /// up front so `Seek::stream_len`-style queries don't re-enter C on every
    /// call.
    ///
    /// Returns `None` if the caller did not supply a `read` callback — that's
    /// a programmer error on the C side (we guarantee `read` is non-null in
    /// the header contract). `ropus_fb2k_open` treats this as `BAD_ARG`.
    ///
    /// The `io.size` callback is optional (HTTP live streams supply no length);
    /// when absent or failing, `size` is `None` and we report `ErrorKind::Other`
    /// on `SeekFrom::End(_)`.
    pub(crate) fn new(io: RopusFb2kIo) -> Option<Self> {
        io.read?;
        let size = size_from(&io);
        Some(Self { io, pos: 0, size })
    }

    /// Whether the underlying stream advertised a `seek` callback. Used by
    /// `ropus_fb2k_open` to decide whether to attempt the reverse-scan for
    /// last-page granule; live HTTP streams with no `seek` get `false` and
    /// we fall through to the unseekable path (zero duration, no bitrate).
    pub(crate) fn can_seek(&self) -> bool {
        self.io.seek.is_some()
    }

    /// Best-effort size query cached at construction time.
    pub(crate) fn size(&self) -> Option<u64> {
        self.size
    }

    fn check_abort(&self) -> io::Result<()> {
        if let Some(abort) = self.io.abort {
            // `abort` is a safe `extern "C" fn` pointer; the caller owns
            // `ctx` and promised the callback is cheap/side-effect-free.
            let flagged = abort(self.io.ctx);
            if flagged != 0 {
                return Err(io::Error::other(AbortTag));
            }
        }
        Ok(())
    }
}

fn size_from(io: &RopusFb2kIo) -> Option<u64> {
    let f = io.size?;
    let mut out: u64 = 0;
    // `size` is a safe `extern "C" fn` pointer. The callee writes at most
    // one `u64` into the out-pointer we pass.
    let rc = f(io.ctx, &mut out as *mut u64);
    if rc == 0 { Some(out) } else { None }
}

impl Read for CallbackReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.check_abort()?;
        if buf.is_empty() {
            return Ok(0);
        }
        // `read` is a safe `extern "C" fn` pointer. `CallbackReader::new`
        // rejects a `None` read callback, so unwrap here is infallible. We
        // pass a valid buffer of known length; the callback may write up to
        // `buf.len()` bytes and returns the count (>=0) or -1 on error.
        let read = self
            .io
            .read
            .expect("CallbackReader invariant: read is Some");
        let n = read(self.io.ctx, buf.as_mut_ptr(), buf.len());
        if n < 0 {
            return Err(io::Error::other("read callback failed"));
        }
        let n = n as usize;
        if n > buf.len() {
            // A misbehaving callback would be a host bug, but we guard it
            // here rather than trusting. Silently truncating would corrupt
            // the stream position.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "read callback returned more bytes than buffer size",
            ));
        }
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for CallbackReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.check_abort()?;

        let abs = match pos {
            SeekFrom::Start(off) => off,
            SeekFrom::Current(delta) => {
                let signed = self.pos as i128 + delta as i128;
                if signed < 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "negative seek position",
                    ));
                }
                signed as u64
            }
            SeekFrom::End(delta) => {
                let Some(len) = self.size else {
                    return Err(io::Error::other(
                        "seek from end requires a known stream size",
                    ));
                };
                let signed = len as i128 + delta as i128;
                if signed < 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "negative seek position",
                    ));
                }
                signed as u64
            }
        };

        let Some(seek) = self.io.seek else {
            return Err(io::Error::other("stream does not support seeking"));
        };

        // `seek` is a safe `extern "C" fn` pointer; 0 = OK, -1 = error per
        // the header contract.
        let rc = seek(self.io.ctx, abs);
        if rc != 0 {
            return Err(io::Error::other("seek callback failed"));
        }
        self.pos = abs;
        Ok(abs)
    }

    /// Override the default impl (which routes through `seek(Current(0))`)
    /// to return our cached position directly. The default would poll
    /// `abort` and round-trip through the C `seek` callback just to learn
    /// a value we already track locally — cheap to shave.
    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::raw::c_void;
    use std::sync::Mutex;

    /// Shared state the C-shaped callbacks below read through a raw pointer
    /// stashed in the `ctx` slot. We park the lifetime inside a `Mutex` so
    /// parallel test runs can't stomp on each other.
    struct ByteBuffer {
        bytes: Vec<u8>,
        pos: usize,
        aborting: bool,
    }

    extern "C" fn read_cb(ctx: *mut c_void, out: *mut u8, n: usize) -> i64 {
        // SAFETY: `ctx` was set to `&Mutex<ByteBuffer>` by `make_io`, and
        // lives for the duration of the caller test.
        let m = unsafe { &*(ctx as *const Mutex<ByteBuffer>) };
        let mut b = m.lock().expect("not poisoned");
        let remaining = b.bytes.len().saturating_sub(b.pos);
        let take = remaining.min(n);
        if take > 0 {
            let src = &b.bytes[b.pos..b.pos + take];
            // SAFETY: `out` has capacity for at least `n >= take` bytes.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), out, take) };
        }
        b.pos += take;
        take as i64
    }

    extern "C" fn seek_cb(ctx: *mut c_void, off: u64) -> i32 {
        let m = unsafe { &*(ctx as *const Mutex<ByteBuffer>) };
        let mut b = m.lock().expect("not poisoned");
        if (off as usize) > b.bytes.len() {
            return -1;
        }
        b.pos = off as usize;
        0
    }

    extern "C" fn size_cb(ctx: *mut c_void, out: *mut u64) -> i32 {
        let m = unsafe { &*(ctx as *const Mutex<ByteBuffer>) };
        let b = m.lock().expect("not poisoned");
        // SAFETY: caller supplies a writable `u64` destination.
        unsafe { *out = b.bytes.len() as u64 };
        0
    }

    extern "C" fn abort_cb(ctx: *mut c_void) -> i32 {
        let m = unsafe { &*(ctx as *const Mutex<ByteBuffer>) };
        let b = m.lock().expect("not poisoned");
        if b.aborting { 1 } else { 0 }
    }

    fn make_io(buf: &Mutex<ByteBuffer>) -> RopusFb2kIo {
        RopusFb2kIo {
            ctx: buf as *const Mutex<ByteBuffer> as *mut c_void,
            read: Some(read_cb),
            seek: Some(seek_cb),
            size: Some(size_cb),
            abort: Some(abort_cb),
        }
    }

    fn new_buf(bytes: &[u8]) -> Mutex<ByteBuffer> {
        Mutex::new(ByteBuffer {
            bytes: bytes.to_vec(),
            pos: 0,
            aborting: false,
        })
    }

    #[test]
    fn reads_all_bytes() {
        let buf = new_buf(b"hello world");
        let mut r = CallbackReader::new(make_io(&buf)).expect("read is Some");
        assert_eq!(r.size(), Some(11));
        let mut out = [0u8; 11];
        r.read_exact(&mut out).unwrap();
        assert_eq!(&out, b"hello world");
    }

    #[test]
    fn seek_round_trip() {
        let buf = new_buf(b"0123456789");
        let mut r = CallbackReader::new(make_io(&buf)).expect("read is Some");
        let got = r.seek(SeekFrom::Start(4)).unwrap();
        assert_eq!(got, 4);
        let mut b = [0u8; 3];
        r.read_exact(&mut b).unwrap();
        assert_eq!(&b, b"456");

        let got = r.seek(SeekFrom::End(-2)).unwrap();
        assert_eq!(got, 8);
        let mut b = [0u8; 2];
        r.read_exact(&mut b).unwrap();
        assert_eq!(&b, b"89");
    }

    #[test]
    fn abort_halts_read() {
        let buf = new_buf(b"anything");
        buf.lock().unwrap().aborting = true;
        let mut r = CallbackReader::new(make_io(&buf)).expect("read is Some");
        let err = r.read(&mut [0u8; 4]).unwrap_err();
        // Upstream downcasts to `AbortTag`; its Display renders as "aborted"
        // but the reader layer doesn't rely on that string.
        assert!(
            err.get_ref()
                .and_then(|e| e.downcast_ref::<AbortTag>())
                .is_some(),
            "expected AbortTag marker, got {err:?}"
        );
    }

    #[test]
    fn null_read_rejected_by_constructor() {
        let buf = new_buf(b"anything");
        let mut io = make_io(&buf);
        io.read = None;
        assert!(CallbackReader::new(io).is_none());
    }
}
