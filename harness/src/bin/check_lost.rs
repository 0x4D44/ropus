use std::fs::File;
use std::io::Read;

fn main() {
    let bit_path = std::env::args().nth(1).expect("pass .bit path");
    let mut f = File::open(&bit_path).expect("open");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("read");

    // opus_demo format: each frame has
    // 32-bit big-endian length, 32-bit big-endian final_range, then len bytes
    // Special case: length with 0x80000000 bit set => packet loss marker
    let mut off = 0usize;
    let mut frame = 0usize;
    let mut lost_count = 0;
    while off + 8 <= buf.len() {
        let len_word = u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        let fin_range =
            u32::from_be_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
        off += 8;
        let lost = (len_word & 0x80000000) != 0;
        let len = (len_word & 0x7FFFFFFF) as usize;
        if lost {
            lost_count += 1;
        }
        if frame < 50 || lost {
            println!(
                "frame {}: {}bytes{}, final_range=0x{:08x}",
                frame,
                len,
                if lost { " [LOST]" } else { "" },
                fin_range
            );
        }
        if off + len > buf.len() {
            println!("frame {}: truncated", frame);
            break;
        }
        off += len;
        frame += 1;
    }
    println!("Total frames: {}, lost: {}", frame, lost_count);
}
