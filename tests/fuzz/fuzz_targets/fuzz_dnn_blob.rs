#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::OpusDecoder;
use std::cell::RefCell;
use std::sync::Once;

// --------------------------------------------------------------------------- //
// Panic-capture: on Windows, Rust assertions in libfuzzer-sys trigger __fastfail
// which bypasses libFuzzer's crash-artifact writer. Install a panic hook that
// saves the current input to FUZZ_PANIC_CAPTURE_DIR (or `fuzz_crashes/`) so we
// get reproducible crash files even when libFuzzer fails to persist them.
// --------------------------------------------------------------------------- //
thread_local! {
    static CURRENT_INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn init_panic_capture() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            CURRENT_INPUT.with(|cell| {
                let bytes = cell.borrow();
                if !bytes.is_empty() {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    let hash = hasher.finish();
                    let dir = std::env::var("FUZZ_PANIC_CAPTURE_DIR")
                        .unwrap_or_else(|_| "fuzz_crashes".to_string());
                    let _ = std::fs::create_dir_all(&dir);
                    let path = std::path::Path::new(&dir).join(format!("crash_{:016x}.bin", hash));
                    match std::fs::write(&path, bytes.as_slice()) {
                        Ok(()) => eprintln!(
                            "[PANIC CAPTURE] Saved {} bytes to {}",
                            bytes.len(),
                            path.display()
                        ),
                        Err(e) => eprintln!(
                            "[PANIC CAPTURE] Failed to write {}: {}",
                            path.display(),
                            e
                        ),
                    }
                }
            });
            prev(info);
        }));
    });
}

// Pure safety target for `OpusDecoder::set_dnn_blob` — a parser surface that
// returns `Result<(), i32>` and is otherwise unfuzzed. No differential: the C
// reference does not expose set_dnn_blob with the same semantics; panic-free
// is the only oracle. We additionally exercise a single small decode after the
// blob load to cover post-blob state transitions.
fuzz_target!(|data: &[u8]| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
    });

    let mut dec = match OpusDecoder::new(48000, 2) {
        Ok(d) => d,
        Err(_) => return,
    };
    let _ = dec.set_dnn_blob(data);

    // Optional: try a tiny PLC step afterwards to exercise post-blob state.
    // PLC with `None` packet must not panic regardless of blob load result.
    let plc_frame = 48000 / 50; // 20 ms at 48 kHz
    let mut plc_pcm = vec![0i16; plc_frame as usize * 2];
    let _ = dec.decode(None, &mut plc_pcm, plc_frame, false);
});
