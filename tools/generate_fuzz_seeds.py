#!/usr/bin/env python3
"""
Generate diverse synthetic fuzz seed corpus for mdopus.

Generates structurally rich PCM patterns (chirps, impulses, speech-like modulated
noise, varying amplitudes, transitions) and packs them with various encoder configs.

Seeds are written directly to tests/fuzz/corpus/<target>/.

Usage:
    python tools/generate_fuzz_seeds.py
"""

import math
import os
import struct
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR = os.path.join(ROOT, "tests", "fuzz", "corpus")

SAMPLE_RATES = [8000, 12000, 16000, 24000, 48000]
# Application indices: 0=VOIP, 1=AUDIO, 2=RESTRICTED_LOWDELAY
APPLICATIONS = [0, 1, 2]

random.seed(42)  # reproducible

def i16_clamp(x):
    return max(-32768, min(32767, int(x)))

def pcm_to_bytes(samples):
    return b"".join(struct.pack("<h", i16_clamp(s)) for s in samples)

# ---------------------------------------------------------------------------
# PCM pattern generators — each returns a list of i16 samples
# ---------------------------------------------------------------------------

def gen_sine(n, freq, sr, amplitude=16000):
    """Pure sine wave."""
    return [amplitude * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]

def gen_chirp(n, f_start, f_end, sr, amplitude=16000):
    """Linear frequency sweep from f_start to f_end Hz."""
    samples = []
    for i in range(n):
        t = i / sr
        freq = f_start + (f_end - f_start) * i / n
        samples.append(amplitude * math.sin(2 * math.pi * freq * t))
    return samples

def gen_impulse(n, positions=None, amplitude=30000):
    """Sparse impulses at specified positions."""
    samples = [0.0] * n
    if positions is None:
        positions = [0, n // 4, n // 2, 3 * n // 4]
    for p in positions:
        if p < n:
            samples[p] = amplitude
    return samples

def gen_modulated_noise(n, mod_freq, sr, amplitude=12000):
    """White noise amplitude-modulated by a low-frequency sine (speech-like envelope)."""
    samples = []
    for i in range(n):
        env = 0.5 + 0.5 * math.sin(2 * math.pi * mod_freq * i / sr)
        noise = random.uniform(-1.0, 1.0)
        samples.append(amplitude * env * noise)
    return samples

def gen_square_wave(n, freq, sr, amplitude=16000):
    """Square wave."""
    return [amplitude * (1.0 if math.sin(2 * math.pi * freq * i / sr) >= 0 else -1.0) for i in range(n)]

def gen_sawtooth(n, freq, sr, amplitude=16000):
    """Sawtooth wave."""
    period = sr / freq
    return [amplitude * (2.0 * ((i % period) / period) - 1.0) for i in range(n)]

def gen_amplitude_ramp(n, amplitude_start=0, amplitude_end=32000):
    """Ramp from silence to loud (or vice versa) with noise content."""
    samples = []
    for i in range(n):
        t = i / n
        amp = amplitude_start + (amplitude_end - amplitude_start) * t
        samples.append(amp * random.uniform(-1.0, 1.0))
    return samples

def gen_dc_offset(n, offset=16000):
    """DC offset with small noise — stresses highpass filters."""
    return [offset + random.uniform(-500, 500) for _ in range(n)]

def gen_alternating_tones(n, freqs, sr, amplitude=16000):
    """Alternate between frequencies every segment_len samples."""
    segment_len = n // len(freqs)
    samples = []
    for idx, freq in enumerate(freqs):
        seg_n = segment_len if idx < len(freqs) - 1 else n - len(samples)
        samples.extend(gen_sine(seg_n, freq, sr, amplitude))
    return samples

def gen_silence_burst(n, burst_start_frac=0.3, burst_end_frac=0.6, amplitude=24000):
    """Silence with a burst of noise in the middle — tests transient detection."""
    samples = [0.0] * n
    burst_start = int(n * burst_start_frac)
    burst_end = int(n * burst_end_frac)
    for i in range(burst_start, burst_end):
        samples[i] = amplitude * random.uniform(-1.0, 1.0)
    return samples

def gen_full_scale(n):
    """Alternating +/- full scale — worst-case clipping."""
    return [32767.0 if i % 2 == 0 else -32768.0 for i in range(n)]

# ---------------------------------------------------------------------------
# Config packing
# ---------------------------------------------------------------------------

def encode_config(sr_idx, channels, app_idx, bitrate, complexity):
    """Pack 6-byte config header for fuzz_encode / fuzz_roundtrip format."""
    # Map bitrate back to u16: bitrate = 6000 + (raw % 504001)
    raw = min(bitrate - 6000, 65535)
    b3, b4 = struct.pack("<H", raw)
    ch_byte = 0 if channels == 1 else 1
    return bytes([sr_idx, ch_byte, app_idx, b3, b4, complexity])

def decode_config(sr_idx, channels):
    """Pack 2-byte config header for fuzz_decode format."""
    ch_byte = 0 if channels == 1 else 1
    return bytes([sr_idx, ch_byte])

def multiframe_config(sr_idx, channels, app_idx, bitrate, complexity, num_frames):
    """Pack 7-byte config header for fuzz_encode_multiframe format."""
    base = encode_config(sr_idx, channels, app_idx, bitrate, complexity)
    frame_byte = num_frames - 5  # maps to 0-5, target adds 5
    return base + bytes([frame_byte])

# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

PATTERN_GENERATORS = [
    ("chirp_up",    lambda n, sr: gen_chirp(n, 100, sr // 2 - 100, sr)),
    ("chirp_down",  lambda n, sr: gen_chirp(n, sr // 2 - 100, 100, sr)),
    ("impulse",     lambda n, sr: gen_impulse(n)),
    ("mod_noise_4", lambda n, sr: gen_modulated_noise(n, 4, sr)),
    ("mod_noise_8", lambda n, sr: gen_modulated_noise(n, 8, sr)),
    ("square_200",  lambda n, sr: gen_square_wave(n, 200, sr)),
    ("sawtooth_300",lambda n, sr: gen_sawtooth(n, 300, sr)),
    ("ramp_up",     lambda n, sr: gen_amplitude_ramp(n, 0, 32000)),
    ("ramp_down",   lambda n, sr: gen_amplitude_ramp(n, 32000, 0)),
    ("dc_offset",   lambda n, sr: gen_dc_offset(n)),
    ("tones_mix",   lambda n, sr: gen_alternating_tones(n, [200, 1000, 3000], sr)),
    ("silence_burst",lambda n, sr: gen_silence_burst(n)),
    ("full_scale",  lambda n, sr: gen_full_scale(n)),
    ("sine_100",    lambda n, sr: gen_sine(n, 100, sr, 8000)),
    ("sine_1k_loud",lambda n, sr: gen_sine(n, 1000, sr, 30000)),
    ("sine_4k",     lambda n, sr: gen_sine(n, min(4000, sr // 2 - 1), sr)),
]

# Representative config combinations
ENCODE_CONFIGS = [
    # (sr_idx, channels, app_idx, bitrate, complexity)
    (0, 1, 0, 12000, 5),   # 8kHz mono VOIP mid
    (0, 2, 0, 16000, 5),   # 8kHz stereo VOIP
    (1, 1, 1, 24000, 8),   # 12kHz mono AUDIO high-cx
    (2, 1, 0, 16000, 3),   # 16kHz mono VOIP low-cx
    (2, 2, 1, 32000, 10),  # 16kHz stereo AUDIO max-cx
    (3, 1, 1, 48000, 5),   # 24kHz mono AUDIO mid
    (3, 2, 0, 24000, 7),   # 24kHz stereo VOIP
    (4, 1, 1, 64000, 10),  # 48kHz mono AUDIO max
    (4, 2, 1, 128000, 5),  # 48kHz stereo AUDIO high-br
    (4, 1, 2, 32000, 5),   # 48kHz mono LOWDELAY
    (4, 2, 2, 64000, 8),   # 48kHz stereo LOWDELAY
    (0, 1, 0, 6000, 0),    # 8kHz mono minimum everything
    (4, 2, 1, 510000, 10), # 48kHz stereo maximum everything
]

def generate_encode_seeds():
    """Generate seeds for fuzz_encode and fuzz_roundtrip targets."""
    target_dirs = {
        "fuzz_encode": os.path.join(CORPUS_DIR, "fuzz_encode"),
        "fuzz_roundtrip": os.path.join(CORPUS_DIR, "fuzz_roundtrip"),
    }
    for d in target_dirs.values():
        os.makedirs(d, exist_ok=True)

    count = 0
    for sr_idx, channels, app_idx, bitrate, complexity in ENCODE_CONFIGS:
        sr = SAMPLE_RATES[sr_idx]
        frame_size = sr // 50  # 20ms
        n_samples = frame_size * channels

        for pat_name, gen_fn in PATTERN_GENERATORS:
            # Generate mono pattern, duplicate for stereo
            mono_samples = gen_fn(frame_size, sr)
            if channels == 2:
                # Interleave: L=pattern, R=slightly phase-shifted pattern
                samples = []
                shift = len(mono_samples) // 7
                for i in range(frame_size):
                    samples.append(mono_samples[i])
                    samples.append(mono_samples[(i + shift) % frame_size])
            else:
                samples = mono_samples

            config = encode_config(sr_idx, channels, app_idx, bitrate, complexity)
            pcm = pcm_to_bytes(samples)
            seed = config + pcm

            sr_label = f"{sr // 1000}k"
            ch_label = "stereo" if channels == 2 else "mono"
            name = f"rich_{sr_label}_{ch_label}_{pat_name}_br{bitrate}_cx{complexity}.bin"

            for target, d in target_dirs.items():
                path = os.path.join(d, name)
                with open(path, "wb") as f:
                    f.write(seed)
                count += 1

    return count

def generate_multiframe_seeds():
    """Generate seeds for fuzz_encode_multiframe target."""
    target_dir = os.path.join(CORPUS_DIR, "fuzz_encode_multiframe")
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    # Subset of configs with multiple frame counts
    configs = [
        (0, 1, 0, 12000, 5),   # 8kHz mono VOIP
        (2, 1, 1, 24000, 8),   # 16kHz mono AUDIO
        (4, 1, 1, 64000, 10),  # 48kHz mono AUDIO max
        (4, 2, 1, 128000, 5),  # 48kHz stereo AUDIO
        (3, 2, 0, 24000, 7),   # 24kHz stereo VOIP
        (4, 1, 2, 32000, 5),   # 48kHz mono LOWDELAY
    ]
    frame_counts = [5, 7, 10]

    for sr_idx, channels, app_idx, bitrate, complexity in configs:
        sr = SAMPLE_RATES[sr_idx]
        frame_size = sr // 50
        n_samples_per_frame = frame_size * channels

        for num_frames in frame_counts:
            for pat_name, gen_fn in PATTERN_GENERATORS[:8]:  # Use first 8 patterns
                # Generate continuous PCM across all frames
                total_mono = frame_size * num_frames
                mono_samples = gen_fn(total_mono, sr)

                if channels == 2:
                    samples = []
                    shift = len(mono_samples) // 7
                    for i in range(total_mono):
                        samples.append(mono_samples[i])
                        samples.append(mono_samples[(i + shift) % total_mono])
                else:
                    samples = mono_samples

                config = multiframe_config(sr_idx, channels, app_idx, bitrate, complexity, num_frames)
                pcm = pcm_to_bytes(samples)
                seed = config + pcm

                sr_label = f"{sr // 1000}k"
                ch_label = "stereo" if channels == 2 else "mono"
                name = f"mf{num_frames}_{sr_label}_{ch_label}_{pat_name}_br{bitrate}.bin"

                path = os.path.join(target_dir, name)
                with open(path, "wb") as f:
                    f.write(seed)
                count += 1

    return count

def generate_decode_seeds():
    """Generate structurally valid Opus-like decode seeds.

    These create valid TOC byte structures with varying frame sizes, bandwidth,
    and mode configurations. The payload is random but the header structure helps
    the fuzzer explore valid decode paths faster.
    """
    target_dir = os.path.join(CORPUS_DIR, "fuzz_decode")
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    # TOC byte format: config(5 bits) | s(1 bit) | c(2 bits)
    # config = bandwidth(4 values) * mode(3 values: SILK/Hybrid/CELT) + frame_size_idx
    # c = 0 (1 frame), 1 (2 frames same compressed size), 2 (2 frames different), 3 (arbitrary)

    for sr_idx in range(5):
        for ch_val in [0, 1]:  # mono, stereo
            config_header = decode_config(sr_idx, ch_val + 1)

            # Generate TOC bytes covering all bandwidths and modes
            for toc_config in range(32):  # 5 bits of config
                for code in range(3):  # c=0,1,2 (skip 3=CBR/VBR with count byte)
                    s = 0
                    toc = (toc_config << 3) | (s << 2) | code
                    # Random payload
                    payload_len = random.randint(10, 200)
                    payload = bytes(random.getrandbits(8) for _ in range(payload_len))
                    seed = config_header + bytes([toc]) + payload

                    name = f"rich_sr{sr_idx}_ch{ch_val+1}_toc{toc:02x}_c{code}.bin"
                    path = os.path.join(target_dir, name)
                    with open(path, "wb") as f:
                        f.write(seed)
                    count += 1

            # Also generate code=3 packets (variable frame count)
            for toc_config in [0, 8, 16, 24, 31]:  # sample of configs
                for frame_count in [3, 5, 8]:
                    toc = (toc_config << 3) | 3  # code=3
                    # Frame count byte: VBR=0, padding=0, M=frame_count
                    count_byte = frame_count & 0x3F
                    # Frame lengths + payload
                    payload = bytes([count_byte])
                    for _ in range(frame_count):
                        flen = random.randint(5, 50)
                        payload += bytes([flen])  # 1-byte length
                        payload += bytes(random.getrandbits(8) for _ in range(flen))

                    seed = config_header + bytes([toc]) + payload
                    name = f"rich_sr{sr_idx}_ch{ch_val+1}_toc{toc:02x}_c3_f{frame_count}.bin"
                    path = os.path.join(target_dir, name)
                    with open(path, "wb") as f:
                        f.write(seed)
                    count += 1

    return count

def generate_packet_parse_seeds():
    """Generate seeds for fuzz_packet_parse — covers all TOC byte patterns."""
    target_dir = os.path.join(CORPUS_DIR, "fuzz_packet_parse")
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    # All 256 possible first bytes with various payload lengths
    for first_byte in range(256):
        for payload_len in [0, 1, 10, 50, 200]:
            payload = bytes(random.getrandbits(8) for _ in range(payload_len))
            seed = bytes([first_byte]) + payload
            name = f"rich_toc{first_byte:02x}_len{payload_len}.bin"
            path = os.path.join(target_dir, name)
            with open(path, "wb") as f:
                f.write(seed)
            count += 1

    return count

def main():
    print("Generating diverse synthetic fuzz seeds...")
    print()

    n = generate_encode_seeds()
    print(f"  Encode + Roundtrip seeds: {n} files ({n // 2} per target)")

    n = generate_multiframe_seeds()
    print(f"  Multiframe seeds:         {n} files")

    n = generate_decode_seeds()
    print(f"  Decode seeds:             {n} files")

    n = generate_packet_parse_seeds()
    print(f"  Packet parse seeds:       {n} files")

    # Print corpus stats
    print()
    print("Corpus directory sizes:")
    for name in sorted(os.listdir(CORPUS_DIR)):
        d = os.path.join(CORPUS_DIR, name)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            total_bytes = sum(os.path.getsize(os.path.join(d, f)) for f in files)
            print(f"  {name:30s}  {len(files):5d} files  {total_bytes // 1024:5d} KB")

if __name__ == "__main__":
    main()
