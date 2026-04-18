#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::repacketizer::OpusRepacketizer;
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;
use c_reference::{
    opus_repacketizer_cat, opus_repacketizer_create, opus_repacketizer_destroy,
    opus_repacketizer_init, opus_repacketizer_out, OPUS_OK,
};

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

fuzz_target!(|data: &[u8]| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
    });

    if data.is_empty() {
        return;
    }

    // --- Rust repacketizer ---
    let mut rust_rp = OpusRepacketizer::new();
    rust_rp.init();

    let rust_cat = rust_rp.cat(data, data.len() as i32);

    let mut rust_out = vec![0u8; data.len() + 256];
    let rust_out_maxlen = rust_out.len() as i32;
    let rust_out_len = rust_rp.out(&mut rust_out, rust_out_maxlen);

    // --- C reference repacketizer ---
    let (c_cat, c_out_result) = unsafe {
        let c_rp = opus_repacketizer_create();
        if c_rp.is_null() {
            return;
        }
        opus_repacketizer_init(c_rp);

        let c_cat_ret = opus_repacketizer_cat(c_rp, data.as_ptr(), data.len() as i32);

        let mut c_out = vec![0u8; data.len() + 256];
        let c_out_ret = opus_repacketizer_out(c_rp, c_out.as_mut_ptr(), c_out.len() as i32);

        opus_repacketizer_destroy(c_rp);
        (c_cat_ret, (c_out_ret, c_out))
    };

    let (c_out_len, c_out) = c_out_result;

    // --- Differential comparison ---

    // cat() return value must agree on success/failure
    let rust_cat_ok = rust_cat == OPUS_OK as i32;
    let c_cat_ok = c_cat == OPUS_OK as i32;
    assert_eq!(
        rust_cat_ok, c_cat_ok,
        "cat() agreement mismatch: Rust={rust_cat} (ok={rust_cat_ok}), C={c_cat} (ok={c_cat_ok}), \
         data_len={}",
        data.len()
    );

    // If both cat succeeded, out() must produce identical results
    if rust_cat_ok && c_cat_ok {
        assert_eq!(
            rust_out_len, c_out_len,
            "out() length mismatch: Rust={rust_out_len}, C={c_out_len}, data_len={}",
            data.len()
        );

        if rust_out_len > 0 && c_out_len > 0 {
            assert_eq!(
                &rust_out[..rust_out_len as usize],
                &c_out[..c_out_len as usize],
                "out() content mismatch at data_len={}",
                data.len()
            );
        }
    }
});
