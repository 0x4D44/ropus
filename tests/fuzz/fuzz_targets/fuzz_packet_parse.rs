#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, opus_packet_get_nb_frames,
    opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
};
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

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

    // --- Bandwidth ---
    let rust_bw = opus_packet_get_bandwidth(data);
    let c_bw = c_reference::c_packet_get_bandwidth(data);
    assert_eq!(
        rust_bw, c_bw,
        "get_bandwidth mismatch: Rust={rust_bw}, C={c_bw}, first_byte=0x{:02x}, len={}",
        data[0],
        data.len()
    );

    // --- Nb channels ---
    let rust_ch = opus_packet_get_nb_channels(data);
    let c_ch = c_reference::c_packet_get_nb_channels(data);
    assert_eq!(
        rust_ch, c_ch,
        "get_nb_channels mismatch: Rust={rust_ch}, C={c_ch}, first_byte=0x{:02x}",
        data[0]
    );

    // --- Nb frames ---
    let rust_frames = opus_packet_get_nb_frames(data).unwrap_or_else(|e| e);
    let c_frames = c_reference::c_packet_get_nb_frames(data);
    assert_eq!(
        rust_frames, c_frames,
        "get_nb_frames mismatch: Rust={rust_frames}, C={c_frames}, \
         first_byte=0x{:02x}, len={}",
        data[0],
        data.len()
    );

    // --- Samples per frame and nb samples at ALL sample rates ---
    for &sr in &SAMPLE_RATES {
        let rust_spf = opus_packet_get_samples_per_frame(data, sr);
        let c_spf = c_reference::c_packet_get_samples_per_frame(data, sr);
        assert_eq!(
            rust_spf, c_spf,
            "get_samples_per_frame mismatch at sr={sr}: Rust={rust_spf}, C={c_spf}, \
             first_byte=0x{:02x}",
            data[0]
        );

        let rust_ns = opus_packet_get_nb_samples(data, sr).unwrap_or_else(|e| e);
        let c_ns = c_reference::c_packet_get_nb_samples(data, sr);
        assert_eq!(
            rust_ns, c_ns,
            "get_nb_samples mismatch at sr={sr}: Rust={rust_ns}, C={c_ns}, \
             first_byte=0x{:02x}, len={}",
            data[0],
            data.len()
        );
    }
});
