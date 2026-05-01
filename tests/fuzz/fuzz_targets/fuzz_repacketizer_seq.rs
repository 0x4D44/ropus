#![no_main]
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ropus::opus::repacketizer::OpusRepacketizer;
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

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

// --------------------------------------------------------------------------- //
// Structured input: 1..=8 packets cat-ed into one repacketizer, then both
// `out()` and `out_range(begin, end)` exercised. `out_range` bounds clamp at
// runtime against the actual frame count returned by the repacketizer, since
// `Arbitrary` cannot know that ahead of time without re-implementing the
// repacketizer's frame accounting.
// --------------------------------------------------------------------------- //
#[derive(Debug)]
struct RepacketizerSeq {
    packets: Vec<Vec<u8>>,
    out_range_begin: u8,
    out_range_end: u8,
}

impl<'a> Arbitrary<'a> for RepacketizerSeq {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let n = u.int_in_range(1..=8)?;
        let mut packets = Vec::with_capacity(n);
        for _ in 0..n {
            let len = u.int_in_range(1..=2048)?;
            packets.push(u.bytes(len)?.to_vec());
        }
        let out_range_begin = u.arbitrary()?;
        let out_range_end = u.arbitrary()?;
        Ok(Self {
            packets,
            out_range_begin,
            out_range_end,
        })
    }
}

fuzz_target!(|input: RepacketizerSeq| {
    init_panic_capture();
    // Best-effort fingerprint: total payload size + packet count + out_range
    // raw bounds. libFuzzer keeps the raw byte input separately.
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        let total_len: usize = input.packets.iter().map(|p| p.len()).sum();
        buf.extend_from_slice(&(total_len as u32).to_le_bytes());
        buf.push(input.packets.len() as u8);
        buf.push(input.out_range_begin);
        buf.push(input.out_range_end);
    });

    let packet_refs: Vec<&[u8]> = input.packets.iter().map(|p| p.as_slice()).collect();

    // --- Rust repacketizer: cat all packets, query nb_frames, then out + out_range. ---
    let mut rust_rp = OpusRepacketizer::new();
    rust_rp.init();

    let mut rust_cat_rets = Vec::with_capacity(packet_refs.len());
    for pkt in &packet_refs {
        let r = rust_rp.cat(pkt, pkt.len() as i32);
        rust_cat_rets.push(r);
    }

    let rust_nb_frames = rust_rp.get_nb_frames();

    // Clamp out_range bounds against the actual frame count. If nb_frames == 0
    // we still pass through bogus bounds to exercise the OPUS_BAD_ARG path.
    let (begin, end) = if rust_nb_frames > 0 {
        let nf = rust_nb_frames as u32;
        let b = (input.out_range_begin as u32) % nf;
        // end ∈ (begin, nb_frames]
        let span = nf - b;
        let e = b + 1 + (input.out_range_end as u32) % span;
        (b as i32, e as i32)
    } else {
        // Drive validation paths: begin/end remain as raw small bytes.
        (input.out_range_begin as i32, input.out_range_end as i32)
    };

    let total_payload: usize = packet_refs.iter().map(|p| p.len()).sum();
    let buf_cap = total_payload + 1024;
    let mut rust_out = vec![0u8; buf_cap];
    let rust_out_ret = rust_rp.out(&mut rust_out, buf_cap as i32);

    let mut rust_out_range = vec![0u8; buf_cap];
    let rust_out_range_ret =
        rust_rp.out_range(begin as usize, end as usize, &mut rust_out_range, buf_cap as i32);

    // --- C reference: same sequence. ---
    let c_outcome = c_reference::c_repack_cat_then_out_range(&packet_refs, begin, end);

    // --- Differential checks ---

    // cat() agreement on success/failure per packet.
    assert_eq!(
        rust_cat_rets.len(),
        c_outcome.cat_rets.len(),
        "cat ret count mismatch"
    );
    for (i, (r, c)) in rust_cat_rets
        .iter()
        .zip(c_outcome.cat_rets.iter())
        .enumerate()
    {
        let r_ok = *r == c_reference::OPUS_OK;
        let c_ok = *c == c_reference::OPUS_OK;
        assert_eq!(
            r_ok, c_ok,
            "cat[{i}] agreement mismatch: Rust={r}, C={c}, packet_len={}",
            packet_refs[i].len()
        );
    }

    // Frame-count agreement after all cats.
    assert_eq!(
        rust_nb_frames, c_outcome.nb_frames,
        "nb_frames mismatch: Rust={rust_nb_frames}, C={}, n_packets={}",
        c_outcome.nb_frames,
        packet_refs.len()
    );

    // out() must agree.
    assert_eq!(
        rust_out_ret, c_outcome.out_ret,
        "out() length mismatch: Rust={rust_out_ret}, C={}, nb_frames={rust_nb_frames}",
        c_outcome.out_ret
    );
    if rust_out_ret > 0 {
        assert_eq!(
            &rust_out[..rust_out_ret as usize],
            &c_outcome.out_buf[..],
            "out() bytes mismatch, len={rust_out_ret}, nb_frames={rust_nb_frames}"
        );
    }

    // out_range() must agree.
    assert_eq!(
        rust_out_range_ret, c_outcome.out_range_ret,
        "out_range() length mismatch: Rust={rust_out_range_ret}, C={}, \
         begin={begin}, end={end}, nb_frames={rust_nb_frames}",
        c_outcome.out_range_ret
    );
    if rust_out_range_ret > 0 {
        assert_eq!(
            &rust_out_range[..rust_out_range_ret as usize],
            &c_outcome.out_range_buf[..],
            "out_range() bytes mismatch, len={rust_out_range_ret}, \
             begin={begin}, end={end}, nb_frames={rust_nb_frames}"
        );
    }
});
