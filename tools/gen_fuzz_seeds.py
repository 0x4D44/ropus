#!/usr/bin/env python3
"""Generate seed corpus files for ropus fuzz targets.

Reads WAV test vectors from tests/vectors/ and produces binary seed files
for each fuzz target. Per HLD V2 (Stream D, gap 9), seeds are emitted
into ``tests/fuzz/seeds/<target>/`` (committed) — NOT
``tests/fuzz/corpus/<target>/`` (gitignored). The overnight launcher
(``tools/overnight_fuzz_launch.sh``) copies seeds into the corpus tree
with ``cp -n`` at run start.

Per-target structured-input layout (Stream A's 8-byte prologue):

  fuzz_decode / fuzz_decode_safety
      [sr_idx, ch_byte] + Opus packet bytes  (2-byte header)

  fuzz_encode / fuzz_encode_safety / fuzz_roundtrip / fuzz_roundtrip_safety
      8 config bytes + i16 LE PCM:
        byte 0: sample rate index (0..=4)
        byte 1: channel/float byte (bit 0 = stereo, bit 1 = float-PCM mode)
        byte 2: application index (0..=2)
        bytes 3-4: bitrate raw (u16 LE; target = 6000 + raw % 504001)
        byte 5: complexity (target = byte % 11)
        byte 6: vbr/fec config (bit 0 = vbr, bit 1 = vbr_constraint,
                                 bits 2-3 = inband_fec, 3 -> 0)
        byte 7: dtx/loss-perc (bit 0 = dtx,
                                bits 1-7 = loss_perc raw, mod 101)

The pathological generator covers a stress-test palette per HLD V2:
DC offset, max-amp sines at codec-stressing frequencies, white noise,
step click, periodic impulse train, and near-silence with single-LSB
toggling. Total seed budget is constrained to <1 MB committed.

Usage:
    python3 tools/gen_fuzz_seeds.py
"""

import math
import os
import random
import struct
import wave

VECTORS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")
SEEDS_BASE = os.path.join(os.path.dirname(__file__), "..", "tests", "fuzz", "seeds")

# Encode-style targets all consume Stream A's 8-byte prologue + i16 LE PCM
# (or f32 LE PCM when bit 1 of the channel byte is set — we leave that
# branch to runtime mutation; pathological seeds use i16).
ENCODE_TARGETS = [
    "fuzz_encode",
    "fuzz_encode_safety",
    "fuzz_roundtrip",
    "fuzz_roundtrip_safety",
    # fuzz_encode_multiframe consumes via Arbitrary; raw bytes still work as
    # libFuzzer mutator seed material though they don't deserialise cleanly.
    # Skip explicit seeding there — the mutator finds shape on its own.
]

DECODE_TARGETS = [
    "fuzz_decode",
    "fuzz_decode_safety",
]

# Sample rates supported by Opus and their frame sizes for 20 ms.
SAMPLE_RATES = [8000, 12000, 16000, 24000, 48000]
FRAME_SIZES_20MS = {8000: 160, 12000: 240, 16000: 320, 24000: 480, 48000: 960}

# Application indexes (matching APPLICATIONS = [VOIP, AUDIO, LOWDELAY] in
# the fuzz target prologues).
APP_VOIP = 0
APP_AUDIO = 1
APP_LOWDELAY = 2


# ---------------------------------------------------------------------------
# Header builders
# ---------------------------------------------------------------------------

def make_decode_header(sr_idx, channels):
    """Build 2-byte config header for fuzz_decode: [sr_idx, ch_byte]."""
    return bytes([sr_idx % 5, 0 if channels == 1 else 1])


def make_encode_header(
    sr_idx,
    channels,
    app_idx,
    bitrate,
    complexity,
    vbr=False,
    vbr_constraint=False,
    inband_fec=0,
    dtx=False,
    loss_perc=0,
):
    """Build the 8-byte structured-input header used by Stream A's encode/roundtrip
    target prologues. Mirrors the parsing in fuzz_encode.rs / fuzz_roundtrip.rs.

    Byte 0: sample rate index (0..=4)
    Byte 1: channel byte — bit 0 = stereo, bit 1 = float-PCM (left clear here)
    Byte 2: application index (0..=2)
    Bytes 3-4: bitrate raw u16 LE (target = 6000 + raw % 504001)
    Byte 5: complexity (target = byte % 11)
    Byte 6: vbr flags + inband_fec (bit 0 = vbr, bit 1 = vbr_constraint,
                                     bits 2-3 = inband_fec, 3 -> 0)
    Byte 7: dtx + loss_perc raw (bit 0 = dtx, bits 1-7 = loss_perc, mod 101)
    """
    br_raw = max(0, min(65535, bitrate - 6000))
    ch_byte = 0 if channels == 1 else 1  # i16 PCM, bit 1 stays clear
    byte6 = 0
    if vbr:
        byte6 |= 0b0001
    if vbr_constraint:
        byte6 |= 0b0010
    fec_clamped = max(0, min(2, inband_fec))
    byte6 |= (fec_clamped & 0b11) << 2
    byte7 = 0
    if dtx:
        byte7 |= 0b0001
    lp = max(0, min(100, loss_perc))
    byte7 |= (lp << 1) & 0b1111_1110
    return struct.pack(
        "<BBBHBBB",
        sr_idx % 5,
        ch_byte,
        app_idx % 3,
        br_raw,
        complexity % 11,
        byte6,
        byte7,
    )


# Opus TOC byte layout:
#   bits 7-3: config (0-31)  -- encodes bandwidth + frame size
#   bit 2:    stereo flag
#   bits 1-0: code (0-3)     -- frame packing mode

def make_toc(config, stereo, code):
    return ((config & 0x1F) << 3) | ((1 if stereo else 0) << 2) | (code & 0x03)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create seed directories if they don't exist."""
    for target in ENCODE_TARGETS + DECODE_TARGETS:
        d = os.path.join(SEEDS_BASE, target)
        os.makedirs(d, exist_ok=True)


def write_seed(target, name, data):
    path = os.path.join(SEEDS_BASE, target, name)
    with open(path, "wb") as f:
        f.write(data)


def read_wav(path):
    with wave.open(path, "rb") as f:
        nchannels = f.getnchannels()
        framerate = f.getframerate()
        nframes = f.getnframes()
        data = f.readframes(nframes)
    return nchannels, framerate, data


def extract_pcm_frame(pcm_bytes, src_channels, src_rate, target_rate, target_channels):
    """Extract one 20ms frame of PCM at target_rate/target_channels via
    nearest-neighbour resampling (quality is irrelevant for fuzzing).

    Returns raw i16 LE bytes."""
    samples_per_channel = len(pcm_bytes) // (2 * src_channels)
    if samples_per_channel == 0:
        target_frame = FRAME_SIZES_20MS.get(target_rate, target_rate // 50)
        return bytes(target_frame * target_channels * 2)

    n_total = samples_per_channel * src_channels
    fmt = f"<{n_total}h"
    all_samples = struct.unpack(fmt, pcm_bytes[:n_total * 2])

    channels_data = []
    for ch in range(src_channels):
        channels_data.append(
            [all_samples[i] for i in range(ch, len(all_samples), src_channels)]
        )

    src_frame = FRAME_SIZES_20MS.get(src_rate, src_rate // 50)
    for ch in range(src_channels):
        channels_data[ch] = channels_data[ch][:src_frame]

    target_frame = FRAME_SIZES_20MS.get(target_rate, target_rate // 50)
    resampled = []
    for ch in range(min(src_channels, target_channels)):
        src = channels_data[ch]
        src_len = len(src) if src else 1
        out = []
        for i in range(target_frame):
            idx = min(i * src_len // target_frame, src_len - 1)
            out.append(src[idx] if idx < len(src) else 0)
        resampled.append(out)

    while len(resampled) < target_channels:
        resampled.append(resampled[0] if resampled else [0] * target_frame)

    interleaved = []
    for i in range(target_frame):
        for ch in range(target_channels):
            s = resampled[ch][i] if i < len(resampled[ch]) else 0
            interleaved.append(s)

    return struct.pack(f"<{len(interleaved)}h", *interleaved)


# ---------------------------------------------------------------------------
# Pathological seed generators (HLD V2 Stream D, gap 9)
# ---------------------------------------------------------------------------

def _pcm_bytes(samples):
    """Pack a list of i16 sample values to LE bytes, clipping at int16 range."""
    clipped = [max(-32768, min(32767, int(s))) for s in samples]
    return struct.pack(f"<{len(clipped)}h", *clipped)


def _pathological_patterns(sample_rate, channels):
    """Return a list of (label, i16-LE-PCM-bytes) pairs covering the
    pathological palette specified in HLD V2."""
    frame = FRAME_SIZES_20MS.get(sample_rate, sample_rate // 50)
    n = frame * channels
    out = []

    # 1-3: DC offset (constant max +i16, min, zero)
    out.append(("dc_max_pos", _pcm_bytes([32767] * n)))
    out.append(("dc_max_neg", _pcm_bytes([-32768] * n)))
    out.append(("dc_zero", bytes(n * 2)))

    # 4-7: Max-amplitude sines at 100 Hz, 1 kHz, 8 kHz, 22 kHz (Nyquist for 48k)
    # Multi-channel: identical content per channel, interleaved.
    for freq, label in [
        (100, "sine_100hz_max"),
        (1000, "sine_1khz_max"),
        (8000, "sine_8khz_max"),
        (22000, "sine_22khz_max"),
    ]:
        # If freq > Nyquist (sample_rate / 2), still emit but expect aliasing —
        # this is intentional fuzz stress.
        samples = []
        for i in range(frame):
            v = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            for _ in range(channels):
                samples.append(v)
        out.append((label, _pcm_bytes(samples)))

    # 8: White noise at full scale (deterministic via seed for reproducibility).
    rng = random.Random(0xCAFEBABE)
    samples = [rng.randint(-32768, 32767) for _ in range(n)]
    out.append(("noise_fullscale", _pcm_bytes(samples)))

    # 9: Step click — silence -> max -> silence
    samples = [0] * n
    if n >= 2:
        # Place the click at ~25 % into the frame so it's neither at boundary
        # nor centred (better attack-pitch coverage).
        click_idx = (n // 4) * channels
        for ch in range(channels):
            if click_idx + ch < n:
                samples[click_idx + ch] = 32767
    out.append(("step_click", _pcm_bytes(samples)))

    # 10: Periodic impulse train — 80 Hz pitch (in the SILK/CELT pitch range).
    samples = [0] * n
    period_samples = max(1, sample_rate // 80)
    for i in range(0, frame, period_samples):
        for ch in range(channels):
            idx = i * channels + ch
            if idx < n:
                samples[idx] = 32767
    out.append(("impulse_train_80hz", _pcm_bytes(samples)))

    # 11: LSB toggle riding a small DC bias (quantizer cell-boundary
    # stress). At amplitude ~100 the SILK and CELT quantizers actually
    # have to make a quantization decision; alternating ±1 LSB without
    # being silence exercises off-by-one rounding paths in the encoder.
    samples = []
    for i in range(frame):
        v = 100 if (i & 1) == 0 else 101
        for _ in range(channels):
            samples.append(v)
    out.append(("near_silence_lsb_toggle", _pcm_bytes(samples)))

    return out


def gen_pathological_seeds(target, sample_rate, channels, configs=None):
    """Emit the pathological palette for ``target`` at ``sample_rate``/
    ``channels``, prefixed with the appropriate Stream A 8-byte header.

    ``configs`` is a list of (app_idx, bitrate, complexity, vbr, fec, dtx,
    loss_perc) tuples — we emit one seed per (pattern, config). Default
    config exercises mid-range CELT.
    """
    if configs is None:
        # One default config per pattern keeps the seed count tight.
        configs = [(APP_AUDIO, 64000, 5, False, 0, False, 0)]

    sr_idx = SAMPLE_RATES.index(sample_rate)
    count = 0
    for app_idx, br, cx, vbr, fec, dtx, lp in configs:
        cfg_label = f"app{app_idx}_br{br}_cx{cx}_vbr{int(vbr)}_fec{fec}_dtx{int(dtx)}_lp{lp}"
        hdr = make_encode_header(
            sr_idx, channels, app_idx, br, cx,
            vbr=vbr, inband_fec=fec, dtx=dtx, loss_perc=lp,
        )
        for label, pcm in _pathological_patterns(sample_rate, channels):
            ch_label = "mono" if channels == 1 else "stereo"
            sr_label = f"{sample_rate // 1000}k"
            name = f"path_{label}_{sr_label}_{ch_label}_{cfg_label}.bin"
            write_seed(target, name, hdr + pcm)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Decode-target seed generators
# ---------------------------------------------------------------------------

def make_opus_packet_code0(config, stereo, payload_len=20):
    toc = make_toc(config, stereo, 0)
    payload = bytes([i % 256 for i in range(payload_len)]) if payload_len > 0 else b""
    return bytes([toc]) + payload


def make_opus_packet_code1(config, stereo, frame_len=10):
    toc = make_toc(config, stereo, 1)
    frame = bytes([0xAA] * frame_len)
    return bytes([toc]) + frame + frame


def make_opus_packet_code2(config, stereo, len1=10, len2=15):
    toc = make_toc(config, stereo, 2)
    return bytes([toc, len1 & 0xFF]) + bytes([0xBB] * len1) + bytes([0xCC] * len2)


def make_opus_packet_code3(config, stereo, nframes=3, vbr=False, padding=False):
    toc = make_toc(config, stereo, 3)
    count = nframes & 0x3F
    if vbr:
        count |= 0x80
    if padding:
        count |= 0x40
    frame_data = bytes([0xDD] * 8)
    if vbr:
        parts = bytes([toc, count])
        if padding:
            parts += bytes([2])
        for _ in range(nframes - 1):
            parts += bytes([len(frame_data)])
        for _ in range(nframes):
            parts += frame_data
        if padding:
            parts += bytes([0x00, 0x00])
        return parts
    parts = bytes([toc, count])
    if padding:
        parts += bytes([2])
    for _ in range(nframes):
        parts += frame_data
    if padding:
        parts += bytes([0x00, 0x00])
    return parts


def gen_decode_seeds(target):
    """Synthetic Opus packet seeds for the decode targets.

    Format: [sr_idx, ch_byte] + packet bytes. Trimmed selection — the full
    TOC config space is reachable via mutator from these representatives.
    """
    count = 0
    # One config per mode class (SILK / Hybrid / CELT) is enough — libFuzzer
    # mutates the TOC byte trivially.
    representative_configs = [0, 12, 16, 31]

    # Cover the (sr_idx, channels) header space with one packet each.
    for sr_idx in range(5):
        for ch in (1, 2):
            hdr = make_decode_header(sr_idx, ch)
            stereo = ch == 2
            s = "stereo" if stereo else "mono"
            sr = SAMPLE_RATES[sr_idx]
            pkt = make_opus_packet_code0(0, stereo, payload_len=20)
            write_seed(target, f"sr{sr}_{s}_silk_c0.bin", hdr + pkt)
            count += 1

    hdr_48m = make_decode_header(4, 1)
    for config in representative_configs:
        for code_fn, code_name in [
            (lambda c, st: make_opus_packet_code0(c, st, 20), "c0"),
            (lambda c, st: make_opus_packet_code1(c, st, 10), "c1"),
            (lambda c, st: make_opus_packet_code2(c, st, 8, 12), "c2"),
            (lambda c, st: make_opus_packet_code3(c, st, 2, False), "c3"),
        ]:
            pkt = code_fn(config, False)
            write_seed(target, f"48m_cfg{config}_{code_name}.bin", hdr_48m + pkt)
            count += 1

    # Edge cases.
    write_seed(
        target,
        "toc_only_cfg0.bin",
        hdr_48m + bytes([make_toc(0, False, 0)]),
    )
    count += 1
    write_seed(
        target,
        "zeros_pad.bin",
        hdr_48m + bytes([make_toc(12, False, 0)]) + bytes(64),
    )
    count += 1
    write_seed(
        target,
        "ones_pad.bin",
        hdr_48m + bytes([make_toc(12, False, 0)]) + bytes([0xFF] * 64),
    )
    count += 1

    # Mode coverage at 48 kHz stereo (CELT).
    hdr_48s = make_decode_header(4, 2)
    for config in [16, 24, 31]:
        pkt = make_opus_packet_code0(config, True, payload_len=50)
        write_seed(target, f"celt_cfg{config}_stereo.bin", hdr_48s + pkt)
        count += 1

    # Code 3 variants — these exercise the multi-frame parser, which has
    # historically been the source of TOC-decode bugs.
    for vbr in (False, True):
        for padding in (False, True):
            v = "vbr" if vbr else "cbr"
            p = "pad" if padding else "nopad"
            pkt = make_opus_packet_code3(16, False, 3, vbr, padding)
            write_seed(target, f"code3_{v}_{p}_3fr.bin", hdr_48m + pkt)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Encode-target seed generators
# ---------------------------------------------------------------------------

def gen_encode_wav_seeds(target):
    """WAV-derived seeds across mode-stressing configs for encode/roundtrip
    targets. Uses the new 8-byte header. Trimmed list keeps the seed budget
    well under 1 MB."""
    count = 0
    configs = [
        # (sr_idx, channels, app, bitrate, complexity, vbr, fec, dtx, loss, label)
        (0, 1, APP_VOIP, 12000, 5, False, 0, False, 0, "8m_voip_12k"),
        (2, 1, APP_VOIP, 16000, 5, False, 0, False, 0, "16m_voip_16k_silk"),
        (3, 1, APP_AUDIO, 32000, 5, False, 0, False, 0, "24m_audio_32k_hybrid"),
        (4, 1, APP_AUDIO, 64000, 5, False, 0, False, 0, "48m_audio_64k_celt"),
        (4, 2, APP_AUDIO, 96000, 5, True, 1, False, 5, "48s_audio_96k_vbr_fec"),
        (4, 1, APP_LOWDELAY, 48000, 5, False, 0, False, 0, "48m_lowdel_48k"),
        (4, 1, APP_AUDIO, 6000, 0, False, 0, True, 50, "48m_audio_6k_dtx_lp50"),
    ]
    # Minimal WAV set — pathological seeds cover the synthetic stress space.
    wav_files = [
        "48k_sine1k_loud.wav",
        "48k_impulse.wav",
        "48000hz_mono_silence.wav",
    ]
    for sr_idx, ch, app, br, cx, vbr, fec, dtx, lp, label in configs:
        target_rate = SAMPLE_RATES[sr_idx]
        hdr = make_encode_header(
            sr_idx, ch, app, br, cx,
            vbr=vbr, inband_fec=fec, dtx=dtx, loss_perc=lp,
        )
        for fname in wav_files:
            path = os.path.join(VECTORS_DIR, fname)
            if not os.path.exists(path):
                continue
            src_ch, src_rate, pcm_data = read_wav(path)
            pcm = extract_pcm_frame(pcm_data, src_ch, src_rate, target_rate, ch)
            base = fname.replace(".wav", "")
            write_seed(target, f"{label}_{base}.bin", hdr + pcm)
            count += 1
    return count


def gen_encode_synth_seeds(target):
    """A few synthetic single-frame seeds at 48 kHz mono with the default
    config. Pathological seeds cover the rest of the stress palette."""
    count = 0
    hdr = make_encode_header(4, 1, APP_AUDIO, 64000, 5)
    fs = 960
    write_seed(target, "synth_silence.bin", hdr + bytes(fs * 2))
    count += 1
    sine = [int(16000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(fs)]
    write_seed(target, "synth_sine440.bin", hdr + struct.pack(f"<{fs}h", *sine))
    count += 1
    return count


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("Generating fuzz seed corpus files (HLD V2 Stream D)...")
    print(f"  Vectors dir: {os.path.abspath(VECTORS_DIR)}")
    print(f"  Seeds base:  {os.path.abspath(SEEDS_BASE)}")
    print()

    ensure_dirs()
    total = 0

    # Decode-side targets get synthetic Opus packet seeds.
    for target in DECODE_TARGETS:
        n = gen_decode_seeds(target)
        print(f"  {target:25s}: {n} synthetic packet seeds")
        total += n

    # Encode-side targets: pathological + WAV + synth. Pathological palette is
    # emitted at 48 kHz mono only — that's the primary CELT exercise. SILK-band
    # coverage is left to the WAV-derived seeds, which already span 8/16/24 kHz.
    # Keeping pathological to one (sr, ch) pair holds the seed budget below
    # the 1 MB ceiling specified in HLD V2.
    for target in ENCODE_TARGETS:
        n_path = gen_pathological_seeds(target, 48000, 1)
        n_wav = gen_encode_wav_seeds(target)
        n_synth = gen_encode_synth_seeds(target)
        print(
            f"  {target:25s}: {n_path} pathological + {n_wav} wav + {n_synth} synth"
        )
        total += n_path + n_wav + n_synth

    print(f"\n  Total: {total} seed files generated.")


if __name__ == "__main__":
    main()
