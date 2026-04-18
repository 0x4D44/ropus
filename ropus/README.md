# ropus — Safe Rust port of the xiph Opus audio codec

Bit-exact against the C reference. Safe Rust throughout the codec path.

## What it is

`ropus` is a Rust port of the [xiph/opus](https://github.com/xiph/opus)
fixed-point audio codec. Encoded and decoded output is bit-exact against the
upstream C reference implementation across the full parameter matrix we test.
The codec path contains no `unsafe`; the crate builds as pure Rust with no
external C toolchain and no build script.

## Install

```
cargo add ropus
```

## Quickstart

Encode 20 ms of silence at 48 kHz mono in VOIP mode and decode it back:

```rust
use ropus::{OpusEncoder, OpusDecoder, OPUS_APPLICATION_VOIP};

let mut encoder = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_VOIP).unwrap();
let pcm_in = [0i16; 960]; // 20 ms at 48 kHz mono
let mut packet = [0u8; 4000];
let max_bytes = packet.len() as i32;
let len = encoder.encode(&pcm_in, 960, &mut packet, max_bytes).unwrap();

let mut decoder = OpusDecoder::new(48_000, 1).unwrap();
let mut pcm_out = [0i16; 960];
let samples = decoder.decode(Some(&packet[..len as usize]), &mut pcm_out, 960, false).unwrap();
assert_eq!(samples, 960);
```

## Status

Works today:

- All sample rates (8, 12, 16, 24, 48 kHz) and frame sizes (2.5, 5, 10, 20, 40, 60 ms)
- SILK, CELT, and hybrid modes; VBR and CBR
- FEC (in-band forward error correction) and DTX (discontinuous transmission)
- Multistream encode/decode and the repacketizer

Deferred:

- Platform-specific SIMD (ARM NEON, x86 AVX) — portable `wide` SIMD only
- DNN wiring into the encode/decode pipeline (module ports exist but are inert)
- OSCE / DRED neural post-processing
- Analysis / tonality detection (encoder runs without it)

## License

BSD-3-Clause. See `LICENSE`. This port derives from xiph/opus; upstream
copyright is preserved verbatim.

## Acknowledgements

Credit to the [Xiph.Org Foundation](https://xiph.org/) and the Opus IETF IPR
contributors — Xiph.Org, Microsoft, Skype, and Broadcom — whose royalty-free
patent grants make the codec freely usable. Upstream source:
<https://github.com/xiph/opus>. This port is independent and is not affiliated
with or endorsed by Xiph.Org.
