#!/usr/bin/env python3
"""Generate seed corpus files for mdopus fuzz targets.

Reads WAV test vectors from tests/vectors/ and produces binary seed files
for each of the 5 fuzz targets. Seeds use structured fuzzing format:

  fuzz_decode      - [sr_idx, ch_byte] + packet bytes
  fuzz_encode      - [sr_idx, ch_byte, app_idx, br_lo, br_hi, cx] + PCM i16 LE
  fuzz_roundtrip   - same as fuzz_encode
  fuzz_repacketizer - raw Opus packet bytes (no header)
  fuzz_packet_parse - raw bytes (no header)

Usage:
    python tools/gen_fuzz_seeds.py
"""

import math
import os
import struct
import wave

VECTORS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")
CORPUS_BASE = os.path.join(os.path.dirname(__file__), "..", "tests", "fuzz", "corpus")

TARGETS = [
    "fuzz_decode",
    "fuzz_encode",
    "fuzz_roundtrip",
    "fuzz_repacketizer",
    "fuzz_packet_parse",
]

# Sample rates supported by Opus and their frame sizes for 20ms
SAMPLE_RATES = [8000, 12000, 16000, 24000, 48000]
FRAME_SIZES_20MS = {8000: 160, 12000: 240, 16000: 320, 24000: 480, 48000: 960}

# Application modes
APP_VOIP = 0
APP_AUDIO = 1
APP_LOWDELAY = 2


def make_decode_header(sr_idx, channels):
    """Build 2-byte config header for fuzz_decode: [sr_idx, ch_byte]."""
    return bytes([sr_idx % 5, 0 if channels == 1 else 1])


def make_encode_header(sr_idx, channels, app_idx, bitrate, complexity):
    """Build 6-byte config header for fuzz_encode/roundtrip.

    Byte 0: sample rate index (0-4)
    Byte 1: channels (low bit: 0=mono, 1=stereo)
    Byte 2: application index (0-2)
    Bytes 3-4: bitrate encoded as u16 LE (target maps: 6000 + (u16 % 504001))
    Byte 5: complexity (target maps: byte % 11)
    """
    # Reverse-map bitrate to u16: we want 6000 + (u16 % 504001) == bitrate
    # Simplest: br_raw = bitrate - 6000, clamped to u16 range
    br_raw = max(0, min(65535, bitrate - 6000))
    return struct.pack("<BBBHB",
                       sr_idx % 5,
                       0 if channels == 1 else 1,
                       app_idx % 3,
                       br_raw,
                       complexity % 11)

# Opus TOC byte layout:
#   bits 7-3: config (0-31)  -- encodes bandwidth + frame size
#   bit 2:    stereo flag
#   bits 1-0: code (0-3)     -- frame packing mode


def ensure_dirs():
    """Create corpus directories if they don't exist."""
    for target in TARGETS:
        d = os.path.join(CORPUS_BASE, target)
        os.makedirs(d, exist_ok=True)


def write_seed(target, name, data):
    """Write a seed file to the corpus directory for the given target."""
    path = os.path.join(CORPUS_BASE, target, name)
    with open(path, "wb") as f:
        f.write(data)


def read_wav(path):
    """Read a WAV file and return (nchannels, framerate, raw_pcm_bytes)."""
    with wave.open(path, "rb") as f:
        nchannels = f.getnchannels()
        framerate = f.getframerate()
        nframes = f.getnframes()
        data = f.readframes(nframes)
    return nchannels, framerate, data


def extract_pcm_frame(pcm_bytes, src_channels, src_rate, target_rate, target_channels):
    """Extract one 20ms frame of PCM at target_rate/target_channels.

    Resamples via nearest-neighbor (quality doesn't matter for fuzzing).
    Returns raw i16 LE bytes.
    """
    samples_per_channel = len(pcm_bytes) // (2 * src_channels)
    if samples_per_channel == 0:
        target_frame = FRAME_SIZES_20MS.get(target_rate, target_rate // 50)
        return bytes(target_frame * target_channels * 2)

    n_total = samples_per_channel * src_channels
    fmt = f"<{n_total}h"
    all_samples = struct.unpack(fmt, pcm_bytes[:n_total * 2])

    # Deinterleave to per-channel lists
    channels_data = []
    for ch in range(src_channels):
        channels_data.append([all_samples[i] for i in range(ch, len(all_samples), src_channels)])

    # Take 20ms at source rate
    src_frame = FRAME_SIZES_20MS.get(src_rate, src_rate // 50)
    for ch in range(src_channels):
        channels_data[ch] = channels_data[ch][:src_frame]

    # Resample each channel to target_rate
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

    # Duplicate mono to stereo if needed
    while len(resampled) < target_channels:
        resampled.append(resampled[0] if resampled else [0] * target_frame)

    # Interleave
    interleaved = []
    for i in range(target_frame):
        for ch in range(target_channels):
            s = resampled[ch][i] if i < len(resampled[ch]) else 0
            interleaved.append(s)

    return struct.pack(f"<{len(interleaved)}h", *interleaved)


def make_toc(config, stereo, code):
    """Build an Opus TOC byte."""
    return ((config & 0x1F) << 3) | ((1 if stereo else 0) << 2) | (code & 0x03)


def make_opus_packet_code0(config, stereo, payload_len=20):
    """Single frame packet (code 0): TOC + payload."""
    toc = make_toc(config, stereo, 0)
    payload = bytes([i % 256 for i in range(payload_len)]) if payload_len > 0 else b""
    return bytes([toc]) + payload


def make_opus_packet_code1(config, stereo, frame_len=10):
    """Two equal-size frames (code 1): TOC + frame1 + frame2 (equal size)."""
    toc = make_toc(config, stereo, 1)
    frame = bytes([0xAA] * frame_len)
    return bytes([toc]) + frame + frame


def make_opus_packet_code2(config, stereo, len1=10, len2=15):
    """Two different-size frames (code 2): TOC + len1_byte + frame1 + frame2."""
    toc = make_toc(config, stereo, 2)
    # Length coding: values < 252 are literal
    return bytes([toc, len1 & 0xFF]) + bytes([0xBB] * len1) + bytes([0xCC] * len2)


def make_opus_packet_code3(config, stereo, nframes=3, vbr=False, padding=False):
    """Arbitrary frame count (code 3): TOC + count_byte + [lengths] + frames."""
    toc = make_toc(config, stereo, 3)
    # Count byte: bit 7 = VBR, bit 6 = padding, bits 5-0 = frame count
    count = (nframes & 0x3F)
    if vbr:
        count |= 0x80
    if padding:
        count |= 0x40

    frame_data = bytes([0xDD] * 8)

    if vbr:
        # VBR: each frame except last has a length prefix
        parts = bytes([toc, count])
        if padding:
            parts += bytes([2])  # 2 bytes of padding
        for _ in range(nframes - 1):
            parts += bytes([len(frame_data)])  # length of each frame
        for _ in range(nframes):
            parts += frame_data
        if padding:
            parts += bytes([0x00, 0x00])
        return parts
    else:
        # CBR: all frames same size, no length bytes
        parts = bytes([toc, count])
        if padding:
            parts += bytes([2])  # 2 bytes of padding
        for _ in range(nframes):
            parts += frame_data
        if padding:
            parts += bytes([0x00, 0x00])
        return parts


# ---------------------------------------------------------------------------
# Seed generators per target
# ---------------------------------------------------------------------------

def gen_decode_seeds():
    """Generate synthetic Opus packet seeds for the decode target.

    Format: [sr_idx, ch_byte] + packet_bytes
    Covers TOC config space across sample rates and channel counts.
    """
    count = 0

    representative_configs = [0, 1, 4, 8, 12, 15, 16, 20, 24, 28, 31]

    # Vary sample rate and channels in the header
    for sr_idx in range(5):
        for ch in [1, 2]:
            ch_byte = 0 if ch == 1 else 1
            hdr = bytes([sr_idx, ch_byte])
            stereo = (ch == 2)
            s = "stereo" if stereo else "mono"
            sr = SAMPLE_RATES[sr_idx]

            # Pick a subset of configs for each sr/ch combination
            for config in representative_configs[:4]:
                pkt = make_opus_packet_code0(config, stereo, payload_len=20)
                write_seed("fuzz_decode", f"sr{sr}_{s}_cfg{config}_c0.bin", hdr + pkt)
                count += 1

    # Full config coverage at 48kHz mono (most important)
    hdr_48m = make_decode_header(4, 1)
    for config in representative_configs:
        for code_fn, code_name in [
            (lambda c, st: make_opus_packet_code0(c, st, 20), "c0"),
            (lambda c, st: make_opus_packet_code1(c, st, 10), "c1"),
            (lambda c, st: make_opus_packet_code2(c, st, 8, 12), "c2"),
            (lambda c, st: make_opus_packet_code3(c, st, 2, False), "c3"),
        ]:
            pkt = code_fn(config, False)
            write_seed("fuzz_decode", f"48m_cfg{config}_{code_name}.bin", hdr_48m + pkt)
            count += 1

    # Edge cases with 48kHz mono header
    for config in [0, 16, 31]:
        write_seed("fuzz_decode", f"toc_only_cfg{config}.bin",
                    hdr_48m + bytes([make_toc(config, False, 0)]))
        count += 1

    for size in [1, 2, 4, 16, 64, 255, 1275]:
        toc = make_toc(12, False, 0)
        write_seed("fuzz_decode", f"zeros_{size}b.bin", hdr_48m + bytes([toc]) + bytes(size))
        count += 1

    for size in [1, 16, 255]:
        toc = make_toc(12, False, 0)
        write_seed("fuzz_decode", f"ones_{size}b.bin",
                    hdr_48m + bytes([toc]) + bytes([0xFF] * size))
        count += 1

    # SILK/Hybrid/CELT mode coverage
    for config in [0, 4, 8]:  # SILK
        pkt = make_opus_packet_code0(config, False, payload_len=50)
        write_seed("fuzz_decode", f"silk_cfg{config}.bin", hdr_48m + pkt)
        count += 1
    for config in [12, 13, 14, 15]:  # Hybrid
        pkt = make_opus_packet_code0(config, False, payload_len=50)
        write_seed("fuzz_decode", f"hybrid_cfg{config}.bin", hdr_48m + pkt)
        count += 1

    hdr_48s = make_decode_header(4, 2)
    for config in [16, 20, 24, 28, 31]:  # CELT
        pkt = make_opus_packet_code0(config, True, payload_len=50)
        write_seed("fuzz_decode", f"celt_cfg{config}_stereo.bin", hdr_48s + pkt)
        count += 1

    # Code 3 variations
    for vbr in [False, True]:
        for padding in [False, True]:
            for nframes in [1, 2, 3, 6]:
                v = "vbr" if vbr else "cbr"
                p = "pad" if padding else "nopad"
                pkt = make_opus_packet_code3(16, False, nframes, vbr, padding)
                write_seed("fuzz_decode", f"code3_{v}_{p}_{nframes}fr.bin", hdr_48m + pkt)
                count += 1

    return count


def gen_encode_seeds():
    """Generate PCM seeds for the encode target.

    Format: [sr_idx, ch_byte, app_idx, br_lo, br_hi, cx] + PCM i16 LE
    Generate across multiple configs and WAV sources.
    """
    count = 0

    # Config presets: (sr_idx, channels, app_idx, bitrate, complexity, label)
    configs = [
        (4, 1, APP_AUDIO, 64000, 5, "48m_audio_64k_c5"),
        (4, 1, APP_VOIP, 24000, 5, "48m_voip_24k_c5"),
        (4, 2, APP_AUDIO, 96000, 5, "48s_audio_96k_c5"),
        (4, 1, APP_AUDIO, 128000, 10, "48m_audio_128k_c10"),
        (4, 1, APP_LOWDELAY, 32000, 3, "48m_lowdelay_32k_c3"),
        (3, 1, APP_AUDIO, 48000, 5, "24m_audio_48k_c5"),
        (2, 1, APP_VOIP, 16000, 5, "16m_voip_16k_c5"),
        (0, 1, APP_VOIP, 12000, 5, "8m_voip_12k_c5"),
        (1, 1, APP_VOIP, 16000, 5, "12m_voip_16k_c5"),
        (4, 1, APP_AUDIO, 6000, 0, "48m_audio_6k_c0"),
        (4, 1, APP_AUDIO, 510000, 10, "48m_audio_510k_c10"),
    ]

    # Generate from WAV files with various configs
    wav_files = sorted(f for f in os.listdir(VECTORS_DIR) if f.endswith(".wav"))

    for sr_idx, ch, app, br, cx, label in configs:
        target_rate = SAMPLE_RATES[sr_idx]
        hdr = make_encode_header(sr_idx, ch, app, br, cx)

        # Use representative WAV files for each config
        for fname in wav_files[:5]:  # first 5 WAVs per config
            path = os.path.join(VECTORS_DIR, fname)
            src_ch, src_rate, pcm_data = read_wav(path)
            pcm = extract_pcm_frame(pcm_data, src_ch, src_rate, target_rate, ch)
            base = fname.replace(".wav", "")
            write_seed("fuzz_encode", f"{label}_{base}.bin", hdr + pcm)
            count += 1

    # Synthetic patterns at 48kHz mono, default config
    hdr_default = make_encode_header(4, 1, APP_AUDIO, 64000, 5)
    frame_samples = 960

    write_seed("fuzz_encode", "synth_silence.bin",
               hdr_default + bytes(frame_samples * 2))
    count += 1

    write_seed("fuzz_encode", "synth_max_pos.bin",
               hdr_default + struct.pack(f"<{frame_samples}h", *([32767] * frame_samples)))
    count += 1

    write_seed("fuzz_encode", "synth_max_neg.bin",
               hdr_default + struct.pack(f"<{frame_samples}h", *([-32768] * frame_samples)))
    count += 1

    alt = [16000 if i % 2 == 0 else -16000 for i in range(frame_samples)]
    write_seed("fuzz_encode", "synth_alternating.bin",
               hdr_default + struct.pack(f"<{frame_samples}h", *alt))
    count += 1

    sine = [int(16000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(frame_samples)]
    write_seed("fuzz_encode", "synth_sine440.bin",
               hdr_default + struct.pack(f"<{frame_samples}h", *sine))
    count += 1

    impulse = [0] * frame_samples
    impulse[0] = 32767
    write_seed("fuzz_encode", "synth_impulse.bin",
               hdr_default + struct.pack(f"<{frame_samples}h", *impulse))
    count += 1

    ramp = [int(32767 * i / (frame_samples - 1)) for i in range(frame_samples)]
    write_seed("fuzz_encode", "synth_ramp.bin",
               hdr_default + struct.pack(f"<{frame_samples}h", *ramp))
    count += 1

    return count


def gen_roundtrip_seeds():
    """Generate PCM seeds for the roundtrip target.

    Format: [sr_idx, ch_byte, app_idx, br_lo, br_hi, cx] + PCM i16 LE
    Focus on configs that exercise different codec modes (SILK, hybrid, CELT).
    """
    count = 0

    # Configs designed to hit different codec modes:
    # - Low bitrate + low rate = SILK-only
    # - Mid bitrate + mid rate = Hybrid
    # - High bitrate + high rate = CELT-only
    configs = [
        (0, 1, APP_VOIP, 12000, 5, "8m_voip_12k"),      # SILK NB
        (2, 1, APP_VOIP, 16000, 5, "16m_voip_16k"),      # SILK WB
        (3, 1, APP_AUDIO, 32000, 5, "24m_audio_32k"),     # Hybrid
        (4, 1, APP_AUDIO, 64000, 5, "48m_audio_64k"),     # CELT
        (4, 2, APP_AUDIO, 96000, 5, "48s_audio_96k"),     # CELT stereo
        (4, 1, APP_LOWDELAY, 48000, 5, "48m_lowdel_48k"), # CELT lowdelay
        (4, 1, APP_AUDIO, 128000, 10, "48m_audio_128k"),  # CELT high quality
        (1, 1, APP_VOIP, 10000, 3, "12m_voip_10k"),       # SILK low complexity
    ]

    wav_files = [
        "48000hz_mono_sine440.wav",
        "48000hz_stereo_noise.wav",
        "48000hz_mono_silence.wav",
        "48k_impulse.wav",
        "48k_sweep.wav",
        "48k_sine1k_loud.wav",
        "48k_sine1k_quiet.wav",
        "8000hz_mono_sine440.wav",
        "16000hz_mono_noise.wav",
    ]

    for sr_idx, ch, app, br, cx, label in configs:
        target_rate = SAMPLE_RATES[sr_idx]
        hdr = make_encode_header(sr_idx, ch, app, br, cx)

        for fname in wav_files:
            path = os.path.join(VECTORS_DIR, fname)
            if not os.path.exists(path):
                continue
            src_ch, src_rate, pcm_data = read_wav(path)
            pcm = extract_pcm_frame(pcm_data, src_ch, src_rate, target_rate, ch)
            base = fname.replace(".wav", "")
            write_seed("fuzz_roundtrip", f"{label}_{base}.bin", hdr + pcm)
            count += 1

    # Synthetic at default config
    hdr = make_encode_header(4, 1, APP_AUDIO, 64000, 5)
    fs = 960

    write_seed("fuzz_roundtrip", "synth_silence.bin", hdr + bytes(fs * 2))
    count += 1

    sine = [int(16000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(fs)]
    write_seed("fuzz_roundtrip", "synth_sine440.bin",
               hdr + struct.pack(f"<{fs}h", *sine))
    count += 1

    sine1k = [int(16000 * math.sin(2 * math.pi * 1000 * i / 48000)) for i in range(fs)]
    write_seed("fuzz_roundtrip", "synth_sine1k.bin",
               hdr + struct.pack(f"<{fs}h", *sine1k))
    count += 1

    return count


def gen_repacketizer_seeds():
    """Generate valid Opus packet seeds for the repacketizer target.

    The repacketizer expects valid packets: it calls rp.cat(data) which
    parses the TOC and frames. We generate well-formed packets with
    various code types.
    """
    count = 0

    # Code 0: single frame packets with different configs
    for config in [0, 4, 8, 12, 16, 20, 24, 28, 31]:
        for stereo in [False, True]:
            s = "stereo" if stereo else "mono"
            pkt = make_opus_packet_code0(config, stereo, payload_len=20)
            write_seed("fuzz_repacketizer", f"c0_cfg{config}_{s}.bin", pkt)
            count += 1

    # Code 1: two equal frames
    for config in [0, 12, 16, 28]:
        pkt = make_opus_packet_code1(config, False, frame_len=15)
        write_seed("fuzz_repacketizer", f"c1_cfg{config}_mono.bin", pkt)
        count += 1

    # Code 2: two different frames
    for config in [0, 12, 16, 28]:
        pkt = make_opus_packet_code2(config, False, len1=10, len2=20)
        write_seed("fuzz_repacketizer", f"c2_cfg{config}_mono.bin", pkt)
        count += 1

    # Code 3 CBR: various frame counts
    for nframes in [1, 2, 3, 4, 6]:
        pkt = make_opus_packet_code3(16, False, nframes, vbr=False, padding=False)
        write_seed("fuzz_repacketizer", f"c3_cbr_{nframes}fr.bin", pkt)
        count += 1

    # Code 3 VBR: various frame counts
    for nframes in [2, 3, 4]:
        pkt = make_opus_packet_code3(16, False, nframes, vbr=True, padding=False)
        write_seed("fuzz_repacketizer", f"c3_vbr_{nframes}fr.bin", pkt)
        count += 1

    # Code 3 with padding
    for nframes in [1, 2, 3]:
        pkt = make_opus_packet_code3(16, False, nframes, vbr=False, padding=True)
        write_seed("fuzz_repacketizer", f"c3_cbr_pad_{nframes}fr.bin", pkt)
        count += 1

    # Minimal valid packets (just TOC + 1 byte)
    for config in [0, 8, 16, 24]:
        pkt = bytes([make_toc(config, False, 0), 0x00])
        write_seed("fuzz_repacketizer", f"minimal_cfg{config}.bin", pkt)
        count += 1

    # Larger payloads
    for payload_len in [100, 200, 500]:
        pkt = make_opus_packet_code0(16, False, payload_len)
        write_seed("fuzz_repacketizer", f"large_{payload_len}b.bin", pkt)
        count += 1

    # Stereo code 1 and code 2
    for config in [0, 12, 16, 28]:
        pkt = make_opus_packet_code1(config, True, frame_len=15)
        write_seed("fuzz_repacketizer", f"c1_cfg{config}_stereo.bin", pkt)
        count += 1
        pkt = make_opus_packet_code2(config, True, len1=10, len2=20)
        write_seed("fuzz_repacketizer", f"c2_cfg{config}_stereo.bin", pkt)
        count += 1

    return count


def gen_packet_parse_seeds():
    """Generate seeds for the packet parsing target.

    This target calls opus_packet_get_bandwidth, get_nb_channels,
    get_nb_frames, get_samples_per_frame, get_nb_samples on any bytes.
    We want good coverage of the TOC byte space plus edge cases.
    """
    count = 0

    # Every possible TOC byte value (256 values, 1 byte each)
    for toc in range(256):
        write_seed("fuzz_packet_parse", f"toc_{toc:02x}.bin", bytes([toc]))
        count += 1

    # TOC + small payloads for each code type
    for config in [0, 8, 16, 24, 31]:
        for stereo in [False, True]:
            s = "s" if stereo else "m"

            # code 0: TOC + payload
            toc = make_toc(config, stereo, 0)
            write_seed("fuzz_packet_parse", f"p_cfg{config}_{s}_c0.bin",
                        bytes([toc]) + bytes(10))
            count += 1

            # code 1: TOC + 2 equal frames
            toc = make_toc(config, stereo, 1)
            write_seed("fuzz_packet_parse", f"p_cfg{config}_{s}_c1.bin",
                        bytes([toc]) + bytes(20))
            count += 1

            # code 2: TOC + len + 2 frames
            toc = make_toc(config, stereo, 2)
            write_seed("fuzz_packet_parse", f"p_cfg{config}_{s}_c2.bin",
                        bytes([toc, 8]) + bytes(20))
            count += 1

            # code 3: TOC + count_byte + frames
            toc = make_toc(config, stereo, 3)
            write_seed("fuzz_packet_parse", f"p_cfg{config}_{s}_c3.bin",
                        bytes([toc, 0x03]) + bytes(30))
            count += 1

    # Edge cases
    # Empty after TOC
    write_seed("fuzz_packet_parse", "empty_after_toc.bin", bytes([0x00]))
    count += 1

    # Code 3 with VBR flag set
    toc = make_toc(16, False, 3)
    write_seed("fuzz_packet_parse", "code3_vbr.bin", bytes([toc, 0x83]) + bytes(30))
    count += 1

    # Code 3 with padding flag set
    write_seed("fuzz_packet_parse", "code3_pad.bin", bytes([toc, 0x43, 5]) + bytes(30))
    count += 1

    # Code 3 with both VBR and padding
    write_seed("fuzz_packet_parse", "code3_vbr_pad.bin", bytes([toc, 0xC3, 5]) + bytes(30))
    count += 1

    # Maximum padding (code 3, padding flag, large padding count)
    write_seed("fuzz_packet_parse", "max_padding.bin",
                bytes([toc, 0x41, 254, 255]) + bytes(300))
    count += 1

    # Code 3 with zero frames
    write_seed("fuzz_packet_parse", "code3_0frames.bin", bytes([toc, 0x00]))
    count += 1

    # Multi-byte length encoding (>= 252)
    toc2 = make_toc(16, False, 2)
    # Length of 300: first byte = 252, second byte = (300-252) = 48
    write_seed("fuzz_packet_parse", "long_length.bin",
                bytes([toc2, 252, 48]) + bytes(350))
    count += 1

    # All 0xFF bytes
    write_seed("fuzz_packet_parse", "all_ff.bin", bytes([0xFF] * 16))
    count += 1

    # All 0x00 bytes
    write_seed("fuzz_packet_parse", "all_00.bin", bytes([0x00] * 16))
    count += 1

    # Single byte (minimum input)
    write_seed("fuzz_packet_parse", "single_byte.bin", bytes([0x42]))
    count += 1

    # Two bytes
    write_seed("fuzz_packet_parse", "two_bytes.bin", bytes([0x42, 0x00]))
    count += 1

    return count


def main():
    print("Generating fuzz seed corpus files...")
    print(f"  Vectors dir: {os.path.abspath(VECTORS_DIR)}")
    print(f"  Corpus base: {os.path.abspath(CORPUS_BASE)}")
    print()

    ensure_dirs()

    total = 0

    n = gen_decode_seeds()
    print(f"  fuzz_decode:      {n} seeds")
    total += n

    n = gen_encode_seeds()
    print(f"  fuzz_encode:      {n} seeds")
    total += n

    n = gen_roundtrip_seeds()
    print(f"  fuzz_roundtrip:   {n} seeds")
    total += n

    n = gen_repacketizer_seeds()
    print(f"  fuzz_repacketizer: {n} seeds")
    total += n

    n = gen_packet_parse_seeds()
    print(f"  fuzz_packet_parse: {n} seeds")
    total += n

    print(f"\n  Total: {total} seed files generated.")


if __name__ == "__main__":
    main()
