# ropus — Rust port of the xiph Opus audio codec

Bit-exact against the C reference. Almost entirely safe Rust.

API docs: <https://docs.rs/ropus>

## What it is

`ropus` is a Rust port of the [xiph/opus](https://github.com/xiph/opus)
fixed-point audio codec. Encoded and decoded output is bit-exact against the
upstream C reference implementation across the full parameter matrix we test.
A crates.io install (`cargo add ropus`) needs no external C toolchain: the
build script embeds DNN weights from the xiph reference sources when they're
present on disk (in-tree development), and emits an empty blob otherwise, in
which case callers supply a blob through `Decoder::set_dnn_blob` (or the
lower-level `OpusDecoder::set_dnn_blob`). The only runtime dependency is
[`wide`](https://crates.io/crates/wide) for portable SIMD.

Safety-wise, the codec is almost entirely safe Rust. A small set of
`unsafe { get_unchecked[_mut] }` macros remain in SILK and CELT hot loops
(NSQ LPC, Burg, autocorrelation, LPC analysis, FFT butterflies, MDCT
rotations) for parity with the reference's hand-tuned SSE intrinsics. A few
other `unsafe` blocks cover byte↔PCM slice reinterpretation on the downmix
boundary in `opus/encoder.rs` and one zero-init in the analysis module's
boxed scratch allocator. Each call site operates under statically-known
iteration bounds against a slice at least that long, and is covered by
fuzzing and differential tests against the C reference with no out-of-bounds
findings.

## Install

```
cargo add ropus
```

Requires Rust 1.88 or newer (edition 2024). No external C toolchain required
for the default build.

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

Runnable end-to-end samples live in [`examples/`](examples/):
[`encode.rs`](examples/encode.rs), [`decode.rs`](examples/decode.rs), and
[`roundtrip.rs`](examples/roundtrip.rs). Run with `cargo run --example encode`
etc.

ropus produces raw Opus packets. The examples use the
[`ogg`](https://crates.io/crates/ogg) crate (dev-dependency only) to read
and write standard Ogg-Opus files per RFC 7845 — containerisation is the
caller's choice, so you can equally feed ropus packets into WebM, Matroska,
RTP, or a custom transport.

## Thread safety

`Encoder` and `Decoder` are both `Send` and `Sync`. The
`encoder_decoder_are_send_sync` test in `src/api.rs` asserts this at compile
time, so any future `!Send`/`!Sync` field (e.g., an accidentally introduced
`Rc` or `RefCell`) will fail the build rather than silently drift.

Note that `encode` and `decode` take `&mut self`: you'll typically move a
codec into a thread or serialise access through a `Mutex` / channel rather
than share an instance by reference.

## Status

Works today:

- All sample rates (8, 12, 16, 24, 48 kHz) and all nine legal Opus frame
  sizes (2.5, 5, 10, 20, 40, 60, 80, 100, 120 ms; 80/100/120 ms are
  multi-frame packets)
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
- DRED (Deep REDundancy) — encoder, decoder-parse, and decoder-process
  (latent → fec_features) shipped; final FARGAN-driven PCM reconstruction
  is deferred pending an in-tree consumer (see
  `wrk_docs/2026.04.19 - HLD - fargan-dred-joint-followup.md` for scope
  and rationale)

Deferred:

- Hand-tuned platform SIMD intrinsics (NEON, direct AVX/AVX-512). The shipped
  binary still uses AVX2 — see the SIMD section below — just not via
  hand-written backends.
- OSCE neural post-processing.

Not exposed:

- Custom modes (`opus_custom_*`, the `#ifdef CUSTOM_MODES` surface in
  libopus) — non-RFC 6716 frame sizes that don't interoperate with
  standard Opus streams. Effectively deprecated upstream.
- QEXT (Opus 2.0 extensions) — the feature-flagged surface in recent
  xiph/opus trunk is not ported.

## Performance

Ratio of Rust encode/decode wall time over the xiph/opus C 1.5.2 reference
(lower is better; <1.0× means Rust is faster). Measured via
`tools/bench_sweep.sh --iters=30` on the in-tree benchmark corpus. Both sides
dispatch to AVX2 at runtime.

| Vector                   | Encode    | Decode    |
|--------------------------|:---------:|:---------:|
| SILK NB 8k mono noise    | 1.05×     | 0.69×     |
| SILK WB 16k mono noise   | 1.14×     | 0.89×     |
| Hybrid 24k mono noise    | 1.11×     | 0.90×     |
| CELT FB 48k mono noise   | 1.08×     | 0.92×     |
| CELT FB 48k stereo noise | 1.04×     | 0.94×     |
| CELT 48k sine 1k loud    | 0.94×     | 1.05×     |
| CELT 48k sweep           | 0.96×     | 0.99×     |
| CELT 48k square 1k       | 1.03×     | 1.03×     |
| SPEECH 48k mono (TTS)    | 0.84×     | 0.98×     |
| MUSIC 48k stereo         | 1.01×     | 0.98×     |
| **Mean**                 | **1.02×** | **0.95×** |

Three vectors encode *faster* than C (sine, sweep, SPEECH) with MUSIC
essentially at parity (1.01×); the remaining six run 3-14% slower, dominated
by SILK where the C reference dispatches to hand-tuned SSE. Decode is faster
or at parity on eight of ten vectors, with CELT full-band sine (1.05×) and
square (1.03×) running slightly behind. Full measurement log and methodology:
[`wrk_journals/2026.04.19 - JRN - avx2-baseline.md`](https://github.com/0x4D44/ropus/blob/main/wrk_journals/2026.04.19%20-%20JRN%20-%20avx2-baseline.md).

## SIMD

ropus uses portable SIMD through the [`wide`](https://crates.io/crates/wide)
crate. The published binary also sets `target-cpu = "x86-64-v3"` in
`.cargo/config.toml` — the AVX2 + FMA3 + BMI1/2 baseline (Haswell 2013 /
Excavator 2015). This unblocks LLVM to emit AVX2 from the existing
`wide::i32x4` kernels and to auto-vectorise scalar hot paths at 256-bit
width, which is what keeps encode/decode at or ahead of the C reference's
runtime-dispatched AVX2.

Hand-written platform intrinsics (NEON, direct AVX/AVX-512) are intentionally
deferred: at C-parity through the portable path, a per-architecture backend
is optimization-above-parity rather than a correctness gate. Users on
pre-2013 x86 (~2% of x86 machines per Steam HW survey) can rebuild with
`RUSTFLAGS="-C target-cpu=x86-64-v2"` or `=x86-64`.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for version history.

## License

BSD-3-Clause. See `LICENSE`. This port derives from xiph/opus; upstream
copyright is preserved verbatim.

## Acknowledgements

Credit to the [Xiph.Org Foundation](https://xiph.org/) and the Opus IETF IPR
contributors — Xiph.Org, Microsoft, Skype, and Broadcom — whose royalty-free
patent grants make the codec freely usable. Upstream source:
<https://github.com/xiph/opus>. This port is independent and is not affiliated
with or endorsed by Xiph.Org.
