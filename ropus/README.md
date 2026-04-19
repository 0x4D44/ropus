# ropus — Rust port of the xiph Opus audio codec

Bit-exact against the C reference. Almost entirely safe Rust.

## What it is

`ropus` is a Rust port of the [xiph/opus](https://github.com/xiph/opus)
fixed-point audio codec. Encoded and decoded output is bit-exact against the
upstream C reference implementation across the full parameter matrix we test.
The crate builds as pure Rust with no external C toolchain and no build script.

Safety-wise, the codec is almost entirely safe Rust. A small set of
`unsafe { get_unchecked[_mut] }` macros remain in SILK and CELT hot loops
(NSQ LPC, Burg, autocorrelation, LPC analysis, FFT butterflies, MDCT
rotations) for parity with the reference's hand-tuned SSE intrinsics. Each
call site operates under statically-known iteration bounds against a slice
at least that long, and is covered by fuzzing and differential tests
against the C reference with no out-of-bounds findings.

## Install

```
cargo add ropus
```

## Quickstart

Encode 20 ms of silence at 48 kHz mono in VOIP mode and decode it back:

```rust
use ropus::{Application, Channels, DecodeMode, Decoder, Encoder};

let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Voip)
    .build()
    .unwrap();
let pcm_in = [0i16; 960]; // 20 ms at 48 kHz mono
let mut packet = [0u8; 4000];
let len = encoder.encode(&pcm_in, &mut packet).unwrap();

let mut decoder = Decoder::new(48_000, Channels::Mono).unwrap();
let mut pcm_out = [0i16; 960];
let samples = decoder.decode(&packet[..len], &mut pcm_out, DecodeMode::Normal).unwrap();
assert_eq!(samples, 960);
```

## Status

Works today:

- All sample rates (8, 12, 16, 24, 48 kHz) and frame sizes (2.5, 5, 10, 20, 40, 60 ms)
- SILK, CELT, and hybrid modes; VBR and CBR
- FEC (in-band forward error correction) and DTX (discontinuous transmission)
- Multistream encode/decode and the repacketizer

Additional shipped features (tier-2 where noted):

- Analysis / tonality detection — module-level bit-exact; integrated encode
  accepts ~5% byte-drift on music under a decoded-PCM SNR-equivalence gate
  (0.000 dB codec-SNR delta on music + speech vectors)
- DNN wiring into the decode pipeline — real weight blob embedded; LPCNet,
  FARGAN, and classical-PLC paths live for packet-loss concealment at a
  tier-2 50 dB SNR gate (calibrated against a 42.33 dB C-fixed-vs-C-float
  classical-PLC ceiling)
- DRED (Deep REDundancy) — encoder and decoder-parse shipped; decoder PCM
  reconstruction via FARGAN synthesis is the one open follow-up (see
  `wrk_docs/2026.04.19 - HLD - fargan-dred-joint-followup.md`)

Deferred:

- Platform-specific SIMD (ARM NEON, x86 AVX) — portable `wide` SIMD only
- OSCE neural post-processing

## Performance & SIMD

ropus uses portable SIMD through the [`wide`](https://crates.io/crates/wide)
crate and runs at approximate parity with the xiph/opus C reference. Platform-
specific intrinsics (ARM NEON, x86 AVX) are intentionally deferred: at C-parity,
a hand-tuned per-architecture backend is optimization-above-parity rather than
a correctness gate, so it is not a shipping requirement and not a blocker to
adoption. The portable path is a single implementation that compiles everywhere
the Rust target does.

## License

BSD-3-Clause. See `LICENSE`. This port derives from xiph/opus; upstream
copyright is preserved verbatim.

## Acknowledgements

Credit to the [Xiph.Org Foundation](https://xiph.org/) and the Opus IETF IPR
contributors — Xiph.Org, Microsoft, Skype, and Broadcom — whose royalty-free
patent grants make the codec freely usable. Upstream source:
<https://github.com/xiph/opus>. This port is independent and is not affiliated
with or endorsed by Xiph.Org.
