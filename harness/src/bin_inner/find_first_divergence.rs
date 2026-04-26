// Decode testvector02.bit (or arg[1]) as stereo through both Rust and C,
// find the first diverging packet.
use std::fs::File;
use std::io::Read;

use ropus::opus::decoder::OpusDecoder as RustDec;

unsafe extern "C" {
    fn opus_decoder_create(fs: i32, ch: i32, err: *mut i32) -> *mut u8;
    fn opus_decoder_destroy(st: *mut u8);
    fn opus_decode(
        st: *mut u8,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn opus_decoder_ctl(st: *mut u8, req: i32, ...) -> i32;
}
const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;

pub fn main() {
    let bit_path = std::env::args().nth(1).expect("pass .bit path");
    let channels: i32 = std::env::args().nth(2).map_or(2, |s| s.parse().unwrap());
    let mut f = File::open(&bit_path).expect("open");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("read");

    let sr = 48000;
    let max_frame = 5760;
    let mut rust_dec = RustDec::new(sr, channels).unwrap();
    let c_dec = unsafe {
        let mut err = 0;
        let d = opus_decoder_create(sr, channels, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    let mut rust_pcm = vec![0i16; max_frame as usize * channels as usize];
    let mut c_pcm = vec![0i16; max_frame as usize * channels as usize];

    let stop_after: usize = std::env::args()
        .nth(3)
        .map_or(usize::MAX, |s| s.parse().unwrap());
    let mut off = 0usize;
    let mut frame_no = 0usize;
    while off + 8 <= buf.len() && frame_no <= stop_after {
        let len_word = u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        let fin_range_file =
            u32::from_be_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
        off += 8;
        let _lost = (len_word & 0x80000000) != 0;
        let len = (len_word & 0x7FFFFFFF) as usize;
        if off + len > buf.len() {
            break;
        }
        let data = &buf[off..off + len];
        off += len;

        let rn = rust_dec
            .decode(Some(data), &mut rust_pcm, max_frame, false)
            .unwrap();
        let rust_range = rust_dec.get_final_range();

        let cn = unsafe {
            opus_decode(
                c_dec,
                data.as_ptr(),
                data.len() as i32,
                c_pcm.as_mut_ptr(),
                max_frame,
                0,
            )
        };
        let mut c_range: u32 = 0;
        unsafe {
            opus_decoder_ctl(c_dec, OPUS_GET_FINAL_RANGE_REQUEST, &mut c_range);
        }

        if rn != cn {
            println!(
                "frame {}: sample counts differ: rust={} c={}",
                frame_no, rn, cn
            );
            break;
        }

        let n_samples = rn as usize * channels as usize;
        let pcm_match = rust_pcm[..n_samples] == c_pcm[..n_samples];
        let range_match = rust_range == c_range;

        let toc = data[0];
        let cfg = toc >> 3;
        let stereo = (toc >> 2) & 1;
        let code = toc & 0x3;
        if (595..=605).contains(&frame_no) {
            eprintln!(
                "frame {}: len={} toc=0x{:02x} cfg={} stereo={} code={} rn={}",
                frame_no,
                data.len(),
                toc,
                cfg,
                stereo,
                code,
                rn
            );
        }
        if !pcm_match || !range_match {
            println!(
                "frame {}: FIRST DIVERGENCE pcm_match={} range_match={} rust_range=0x{:08x} c_range=0x{:08x} file_fin_range=0x{:08x}",
                frame_no, pcm_match, range_match, rust_range, c_range, fin_range_file
            );
            // find first mismatched sample
            let mut first_diff = n_samples;
            for i in 0..n_samples {
                if rust_pcm[i] != c_pcm[i] {
                    first_diff = i;
                    println!(
                        "  first diff at sample {}: rust={} c={}",
                        i, rust_pcm[i], c_pcm[i]
                    );
                    break;
                }
            }
            println!(
                "  frame_size={} channels={} first_diff_sample_pair={}",
                rn,
                channels,
                first_diff / channels as usize
            );
            println!("  first 16 samples rust: {:?}", &rust_pcm[..16]);
            println!("  first 16 samples c:    {:?}", &c_pcm[..16]);
            println!(
                "  packet bytes[0..16] = {:02x?}",
                &data[..data.len().min(16)]
            );
            println!("  packet len = {}", data.len());
            break;
        }
        frame_no += 1;
    }
    println!("Processed {} frames without divergence", frame_no);

    unsafe {
        opus_decoder_destroy(c_dec);
    }
}
