#!/usr/bin/env python3
"""
mdopus coordinator - Orchestrates Claude and Codex for Opus C->Rust port.

Usage:
    python coordinator.py run                # Run all phases sequentially
    python coordinator.py phase <name>       # Run a specific phase
    python coordinator.py status             # Show current progress
    python coordinator.py resume             # Resume from last checkpoint

Phases: document, hld, test_harness, implement, integrate
"""

import argparse
import subprocess
import json
import os
import sys
import time
import logging
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# How many agents to run in parallel
MAX_PARALLEL_AGENTS = 4

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference"
ASSETS = ROOT / "wrk_docs" / "design_docs"
TOOLS = ROOT / "tools"
NOTES = ROOT / "notes"
LOGS = ROOT / "logs"
SRC = ROOT / "src"
STATE_FILE = ROOT / "tools" / "coordinator_state.json"

# ---------------------------------------------------------------------------
# Module map - implementation order and C source groupings
# ---------------------------------------------------------------------------

MODULES = [
    {
        "name": "range_coder",
        "description": "Range coder / entropy coding foundation",
        "c_files": ["celt/entcode.c", "celt/entcode.h", "celt/entenc.c",
                     "celt/entenc.h", "celt/entdec.c", "celt/entdec.h",
                     "celt/mfrngcod.h", "celt/ecintrin.h"],
        "rust_module": "celt/range_coder",
        "deps": [],
    },
    {
        "name": "math_ops",
        "description": "Math operations, fixed-point arithmetic, LPC",
        "c_files": ["celt/mathops.c", "celt/mathops.h", "celt/arch.h",
                     "celt/fixed_generic.h", "celt/float_cast.h",
                     "celt/celt_lpc.c", "celt/celt_lpc.h"],
        "rust_module": "celt/math_ops",
        "deps": [],
    },
    {
        "name": "cwrs",
        "description": "Combinatorial (PVQ) coding",
        "c_files": ["celt/cwrs.c", "celt/cwrs.h"],
        "rust_module": "celt/cwrs",
        "deps": ["range_coder", "math_ops"],
    },
    {
        "name": "bands",
        "description": "Band energy processing and normalization",
        "c_files": ["celt/bands.c", "celt/bands.h"],
        "rust_module": "celt/bands",
        "deps": ["math_ops", "range_coder"],
    },
    {
        "name": "fft_mdct",
        "description": "Kiss FFT and MDCT transforms",
        "c_files": ["celt/kiss_fft.c", "celt/kiss_fft.h",
                     "celt/mdct.c", "celt/mdct.h",
                     "celt/mini_kfft.c"],
        "rust_module": "celt/fft",
        "deps": ["math_ops"],
    },
    {
        "name": "pitch",
        "description": "Pitch analysis and detection",
        "c_files": ["celt/pitch.c", "celt/pitch.h"],
        "rust_module": "celt/pitch",
        "deps": ["math_ops", "fft_mdct"],
    },
    {
        "name": "quant_bands",
        "description": "Band quantization with Laplace coding",
        "c_files": ["celt/quant_bands.c", "celt/quant_bands.h",
                     "celt/laplace.c", "celt/laplace.h"],
        "rust_module": "celt/quant_bands",
        "deps": ["bands", "range_coder", "math_ops"],
    },
    {
        "name": "vq",
        "description": "Vector quantization (PVQ search)",
        "c_files": ["celt/vq.c", "celt/vq.h", "celt/rate.c", "celt/rate.h"],
        "rust_module": "celt/vq",
        "deps": ["bands", "cwrs", "math_ops"],
    },
    {
        "name": "celt_modes",
        "description": "CELT mode configuration and static tables",
        "c_files": ["celt/modes.c", "celt/modes.h",
                     "celt/celt.c", "celt/celt.h",
                     "celt/cpu_support.h"],
        "rust_module": "celt/modes",
        "deps": ["fft_mdct", "bands", "math_ops"],
    },
    {
        "name": "celt_decoder",
        "description": "CELT decoder",
        "c_files": ["celt/celt_decoder.c"],
        "rust_module": "celt/decoder",
        "deps": ["celt_modes", "range_coder", "bands", "quant_bands",
                 "fft_mdct", "pitch", "vq", "cwrs", "math_ops"],
    },
    {
        "name": "celt_encoder",
        "description": "CELT encoder",
        "c_files": ["celt/celt_encoder.c"],
        "rust_module": "celt/encoder",
        "deps": ["celt_modes", "range_coder", "bands", "quant_bands",
                 "fft_mdct", "pitch", "vq", "cwrs", "math_ops"],
    },
    {
        "name": "silk_common",
        "description": "SILK shared tables, types, and utilities",
        "c_files": [
            "silk/tables_other.c", "silk/tables_gain.c",
            "silk/tables_pulses_per_block.c", "silk/tables_pitch_lag.c",
            "silk/tables_NLSF_CB_NB_MB.c", "silk/tables_NLSF_CB_WB.c",
            "silk/tables_LTP.c", "silk/table_LSF_cos.c",
            "silk/pitch_est_tables.c", "silk/resampler_rom.c",
            "silk/sort.c", "silk/sigm_Q15.c",
            "silk/lin2log.c", "silk/log2lin.c",
            "silk/bwexpander.c", "silk/bwexpander_32.c",
            "silk/inner_prod_aligned.c", "silk/sum_sqr_shift.c",
            "silk/LPC_inv_pred_gain.c", "silk/LPC_fit.c",
            "silk/interpolate.c",
        ],
        "rust_module": "silk/common",
        "deps": ["math_ops"],
    },
    {
        "name": "silk_decoder",
        "description": "SILK decoder pipeline",
        "c_files": [
            "silk/dec_API.c", "silk/init_decoder.c",
            "silk/decode_core.c", "silk/decode_frame.c",
            "silk/decode_indices.c", "silk/decode_parameters.c",
            "silk/decode_pitch.c", "silk/decode_pulses.c",
            "silk/decoder_set_fs.c",
            "silk/NLSF2A.c", "silk/NLSF_decode.c",
            "silk/NLSF_unpack.c", "silk/NLSF_stabilize.c",
            "silk/NLSF_VQ.c", "silk/NLSF_VQ_weights_laroia.c",
            "silk/shell_coder.c", "silk/code_signs.c",
            "silk/gain_quant.c", "silk/PLC.c", "silk/CNG.c",
            "silk/stereo_MS_to_LR.c", "silk/stereo_decode_pred.c",
            "silk/resampler.c", "silk/resampler_down2.c",
            "silk/resampler_down2_3.c",
            "silk/resampler_private_AR2.c",
            "silk/resampler_private_IIR_FIR.c",
            "silk/resampler_private_down_FIR.c",
            "silk/resampler_private_up2_HQ.c",
            "silk/biquad_alt.c",
        ],
        "rust_module": "silk/decoder",
        "deps": ["silk_common", "range_coder"],
    },
    {
        "name": "silk_encoder",
        "description": "SILK encoder pipeline",
        "c_files": [
            "silk/enc_API.c", "silk/init_encoder.c",
            "silk/encode_indices.c", "silk/encode_pulses.c",
            "silk/control_codec.c", "silk/control_audio_bandwidth.c",
            "silk/control_SNR.c", "silk/check_control_input.c",
            "silk/HP_variable_cutoff.c", "silk/LP_variable_cutoff.c",
            "silk/LPC_analysis_filter.c", "silk/ana_filt_bank_1.c",
            "silk/NSQ.c", "silk/NSQ_del_dec.c",
            "silk/VAD.c", "silk/VQ_WMat_EC.c",
            "silk/A2NLSF.c", "silk/NLSF_encode.c",
            "silk/NLSF_del_dec_quant.c", "silk/process_NLSFs.c",
            "silk/quant_LTP_gains.c",
            "silk/stereo_LR_to_MS.c", "silk/stereo_encode_pred.c",
            "silk/stereo_find_predictor.c", "silk/stereo_quant_pred.c",
        ],
        "rust_module": "silk/encoder",
        "deps": ["silk_common", "silk_decoder", "range_coder"],
    },
    {
        "name": "opus_decoder",
        "description": "Top-level Opus decoder (CELT + SILK hybrid)",
        "c_files": ["src/opus_decoder.c", "src/opus.c",
                     "src/opus_private.h"],
        "rust_module": "opus/decoder",
        "deps": ["celt_decoder", "silk_decoder"],
    },
    {
        "name": "opus_encoder",
        "description": "Top-level Opus encoder (CELT + SILK hybrid)",
        "c_files": ["src/opus_encoder.c", "src/analysis.c",
                     "src/analysis.h", "src/mlp.c", "src/mlp.h",
                     "src/mlp_data.c", "src/tansig_table.h"],
        "rust_module": "opus/encoder",
        "deps": ["celt_encoder", "silk_encoder", "opus_decoder"],
    },
    {
        "name": "opus_multistream",
        "description": "Multistream and projection encoder/decoder",
        "c_files": ["src/opus_multistream.c",
                     "src/opus_multistream_decoder.c",
                     "src/opus_multistream_encoder.c",
                     "src/opus_projection_decoder.c",
                     "src/opus_projection_encoder.c",
                     "src/mapping_matrix.c", "src/mapping_matrix.h"],
        "rust_module": "opus/multistream",
        "deps": ["opus_encoder", "opus_decoder"],
    },
    {
        "name": "repacketizer",
        "description": "Opus repacketizer",
        "c_files": ["src/repacketizer.c", "src/extensions.c"],
        "rust_module": "opus/repacketizer",
        "deps": ["opus_decoder"],
    },
    # -- DNN modules (Opus 1.4+ neural enhancement) --
    {
        "name": "dnn_core",
        "description": "DNN inference engine: nnet, nndsp, vector ops, activation tables",
        "c_files": [
            "dnn/nnet.c", "dnn/nnet.h", "dnn/nnet_arch.h",
            "dnn/nnet_default.c",
            "dnn/nndsp.c", "dnn/nndsp.h",
            "dnn/vec.h", "dnn/common.h",
            "dnn/tansig_table.h",
        ],
        "rust_module": "dnn/core",
        "deps": ["math_ops"],
    },
    {
        "name": "dnn_lpcnet",
        "description": "LPCNet: low-bitrate neural speech codec and tables",
        "c_files": [
            "dnn/lpcnet.c", "dnn/lpcnet.h", "dnn/lpcnet_private.h",
            "dnn/lpcnet_enc.c",
            "dnn/lpcnet_tables.c",
            "dnn/lpcnet_plc.c",
            "dnn/freq.c", "dnn/freq.h",
            "dnn/burg.c", "dnn/burg.h",
            "dnn/kiss99.c", "dnn/kiss99.h",
        ],
        "rust_module": "dnn/lpcnet",
        "deps": ["dnn_core", "math_ops"],
    },
    {
        "name": "dnn_osce",
        "description": "Opus Speech Coding Enhancement (OSCE) with neural post-filter",
        "c_files": [
            "dnn/osce.c", "dnn/osce.h",
            "dnn/osce_config.h", "dnn/osce_structs.h",
            "dnn/osce_features.c", "dnn/osce_features.h",
        ],
        "rust_module": "dnn/osce",
        "deps": ["dnn_core", "dnn_lpcnet", "silk_decoder", "celt_decoder"],
    },
    {
        "name": "dnn_fargan",
        "description": "FARGAN: neural waveform generator",
        "c_files": [
            "dnn/fargan.c", "dnn/fargan.h",
        ],
        "rust_module": "dnn/fargan",
        "deps": ["dnn_core", "dnn_lpcnet"],
    },
    {
        "name": "dnn_fwgan",
        "description": "FWGAN: frequency-domain waveform generator",
        "c_files": [
            "dnn/fwgan.c", "dnn/fwgan.h",
        ],
        "rust_module": "dnn/fwgan",
        "deps": ["dnn_core", "dnn_lpcnet"],
    },
    {
        "name": "dnn_dred",
        "description": "Deep REDundancy (DRED): neural redundancy coding",
        "c_files": [
            "dnn/dred_coding.c", "dnn/dred_coding.h",
            "dnn/dred_config.h",
            "dnn/dred_decoder.c", "dnn/dred_decoder.h",
            "dnn/dred_encoder.c", "dnn/dred_encoder.h",
            "dnn/dred_rdovae.h",
            "dnn/dred_rdovae_dec.c", "dnn/dred_rdovae_dec.h",
            "dnn/dred_rdovae_enc.c", "dnn/dred_rdovae_enc.h",
        ],
        "rust_module": "dnn/dred",
        "deps": ["dnn_core", "dnn_lpcnet", "range_coder"],
    },
    {
        "name": "dnn_lossgen",
        "description": "Packet loss generator for PLC training/testing",
        "c_files": [
            "dnn/lossgen.c", "dnn/lossgen.h",
        ],
        "rust_module": "dnn/lossgen",
        "deps": ["dnn_core"],
    },
    {
        "name": "dnn_pitchdnn",
        "description": "Neural pitch detection",
        "c_files": [
            "dnn/pitchdnn.c", "dnn/pitchdnn.h",
        ],
        "rust_module": "dnn/pitchdnn",
        "deps": ["dnn_core"],
    },
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPTS = {}

PROMPTS["document_module"] = textwrap.dedent("""\
    You are documenting the {module_name} module of the xiph/opus C codec
    for a Rust porting effort. Read these C source files carefully:

    {file_list}

    Generate detailed architecture documentation covering:
    1. **Purpose**: What this module does in the Opus pipeline
    2. **Public API**: Functions, their signatures, parameters, return values
    3. **Internal state**: Structs, their fields, lifecycle
    4. **Algorithm**: Step-by-step description of the core algorithm(s)
    5. **Data flow**: What goes in, what comes out, buffer layouts
    6. **Numerical details**: Fixed-point formats (Q-formats), overflow guards,
       precision requirements, rounding behavior
    7. **Dependencies**: What other modules this calls, what calls this
    8. **Constants and tables**: Static data, magic numbers, their derivation
    9. **Edge cases**: Error conditions, special input handling
    10. **Porting notes**: Anything that will be tricky in Rust (pointer
        arithmetic, in-place mutation, macro-generated code, conditional
        compilation)

    Be precise and thorough. This documentation will be the primary reference
    for the Rust implementation. Include relevant code snippets where they
    clarify the algorithm.

    IMPORTANT: Do NOT write any files. Output the complete Markdown document
    directly as your response. The coordinator will save it to disk.
""")

PROMPTS["hld"] = textwrap.dedent("""\
    You are creating a High-Level Design document for porting the xiph/opus
    audio codec from C to Rust. You have access to the architecture
    documentation in the wrk_docs/design_docs/ directory.

    Read all the architecture docs first, then produce an HLD covering:

    1. **Rust module structure**: Map C modules to Rust modules/crates.
       Show the dependency graph.
    2. **Type system design**: How to represent opus_int16, opus_int32,
       opus_val16, opus_val32, Q-format fixed-point types. Consider
       newtypes for safety.
    3. **Error handling**: Map C error codes to Rust Result types.
    4. **Memory management**: How to handle C's manual alloc/free patterns.
       Stack buffers, arena patterns, etc.
    5. **State machines**: How to model encoder/decoder state. Builder pattern?
    6. **FFI boundary**: Design for the test harness FFI layer to the C
       reference.
    7. **Testing strategy**: Unit tests per module, integration tests via
       comparison harness, property-based tests for numerical invariants.
    8. **Implementation phases**: Detailed ordering with milestones and
       gate criteria for each phase.
    9. **Risk register**: What's most likely to cause bit-inexact output
       and how to mitigate.
    10. **Performance considerations**: Where the C code uses SIMD or
        platform tricks that we'll replace with scalar Rust initially.

    The target is production quality, bit-exact output matching the C
    reference. No unsafe Rust except for the FFI test harness.

    IMPORTANT: Do NOT write any files. Output the complete HLD Markdown
    document directly as your response. The coordinator will save it to disk.
""")

PROMPTS["implement_module"] = textwrap.dedent("""\
    You are implementing the **{module_name}** module of the Opus codec in Rust.

    ## Context
    - Architecture doc: wrk_docs/design_docs/{module_name}_architecture.md
    - HLD: wrk_docs/design_docs/hld.md
    - C reference files (in reference/): {file_list}
    - Already implemented modules: {completed_modules}
    - Rust module path: src/{rust_module}.rs

    ## Requirements
    1. **Bit-exact output** matching the C reference - this is non-negotiable
    2. Safe Rust only (no unsafe)
    3. Idiomatic Rust where possible, but correctness over style
    4. Match the C algorithm precisely - do not optimize or simplify
    5. Preserve all edge case handling from the C code
    6. Use the type mappings from the HLD
    7. Include inline comments for non-obvious numerical operations

    ## Process
    1. Read the architecture doc and C source carefully
    2. Design the Rust types and function signatures
    3. Implement function by function, preserving C logic exactly
    4. Add unit tests that exercise the public API

    Write the complete module. Do not leave TODOs or placeholders.
""")

PROMPTS["review_module"] = textwrap.dedent("""\
    You are an adversarial code reviewer checking a C-to-Rust port of the
    Opus audio codec for correctness.

    ## Your task
    Review the Rust implementation of the **{module_name}** module against
    the C reference.

    Rust implementation: src/{rust_module}.rs
    C reference files: {file_list}

    ## Check for
    1. **Numerical correctness**: Will this produce bit-exact output?
       - Integer overflow/underflow differences (C wraps, Rust panics)
       - Signed right-shift behavior (arithmetic in C, check Rust)
       - Division rounding (C truncates toward zero)
       - Cast truncation differences
       - Fixed-point Q-format precision loss
    2. **Logic fidelity**: Does every branch, loop bound, and edge case
       match the C exactly?
    3. **Missing code**: Any C functions or branches not ported?
    4. **Buffer sizing**: Array lengths, stack buffer sizes match?
    5. **State management**: Struct fields all present? Init values match?
    6. **API compatibility**: Function signatures accept/return equivalent types?

    ## Output format
    Produce a structured review:
    - CRITICAL: Issues that WILL cause incorrect output
    - WARNING: Issues that MIGHT cause incorrect output
    - INFO: Style/idiom suggestions (low priority)

    Be thorough and adversarial. Assume bugs exist until proven otherwise.
""")

PROMPTS["test_harness"] = textwrap.dedent("""\
    Build a test comparison harness for the mdopus project. This is a Rust
    binary that links the C reference opus library via FFI and compares its
    output against our Rust implementation.

    ## Structure
    - `tests/harness/build.rs` - Build script that compiles C reference via cc crate
    - `tests/harness/bindings.rs` - Manual FFI bindings to key C functions
    - `tests/harness/main.rs` - CLI comparison tool

    ## CLI interface
    ```
    ropus-compare encode <input.wav> [--bitrate N] [--complexity N]
    ropus-compare decode <input.opus>
    ropus-compare roundtrip <input.wav> [--bitrate N]
    ropus-compare unit <module_name>
    ```

    ## What it does
    - `encode`: Encode WAV with both C and Rust, compare compressed bytes
    - `decode`: Decode Opus with both C and Rust, compare PCM samples
    - `roundtrip`: Encode then decode with both, compare final PCM
    - `unit`: Run module-level comparison tests (e.g., range coder, FFT)

    ## Output
    - PASS/FAIL with byte/sample offset of first difference
    - Statistics: total samples, matching samples, max difference
    - Hex dump of divergent regions

    ## Build setup
    Use the `cc` crate to compile the C reference from reference/ directory.
    Include only platform-independent C files (no ARM/MIPS/x86 intrinsics).
    Define OPUS_BUILD and HAVE_CONFIG_H=0 or provide a minimal config.h.

    The Cargo.toml needs a [[test]] or [[bin]] entry for the harness.
    Put FFI bindings in a separate module.

    Write all necessary files. The harness must compile and run.
""")

PROMPTS["fix_errors"] = textwrap.dedent("""\
    The {module_name} module has test failures. Fix them.

    ## Error output
    ```
    {error_output}
    ```

    ## Files
    - Rust implementation: src/{rust_module}.rs
    - C reference: {file_list}
    - Architecture doc: wrk_docs/design_docs/{module_name}_architecture.md

    ## Rules
    1. The C reference is always correct - match its behavior exactly
    2. Make the smallest change that fixes the issue
    3. Do not refactor or improve - just fix the bug
    4. If the error is a numerical mismatch, trace the computation
       step by step in both C and Rust to find where they diverge
""")

# ---------------------------------------------------------------------------
# State management (thread-safe)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_build_lock = threading.Lock()  # Serialize cargo build/test across threads

def load_state() -> dict:
    """Load coordinator state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "phase": "document",
        "completed_phases": [],
        "module_status": {},  # module_name -> "documented" | "implemented" | "reviewed" | "passing"
        "attempts": {},       # module_name -> int (fix attempts)
        "started_at": None,
        "last_updated": None,
    }


def save_state(state: dict):
    """Persist coordinator state to disk (thread-safe)."""
    with _state_lock:
        state["last_updated"] = datetime.now().isoformat()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"coordinator_{timestamp}.log"

    logger = logging.getLogger("coordinator")
    logger.setLevel(logging.DEBUG)

    # File handler - verbose
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

    # Console handler - concise
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

MAX_AGENT_TIMEOUT = 3600  # seconds per invocation (1 hour - Opus + max effort is thorough)

def invoke_claude(prompt: str, cwd: Path = ROOT, timeout: int = MAX_AGENT_TIMEOUT,
                  add_dirs: list[str] | None = None) -> tuple[bool, str]:
    """
    Invoke Claude CLI in headless mode.
    Returns (success: bool, output: str).
    """
    cmd = [
        "claude", "-p",
        "--model", "claude-opus-4-6",
        "--effort", "max",
        "--permission-mode", "bypassPermissions",
        "--output-format", "text",
    ]
    if add_dirs:
        for d in add_dirs:
            cmd.extend(["--add-dir", d])

    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True,
            input=prompt, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"ERROR: {e}"


def invoke_codex(prompt: str, cwd: Path = ROOT, timeout: int = MAX_AGENT_TIMEOUT,
                 model: str = "o3") -> tuple[bool, str]:
    """
    Invoke Codex CLI in headless (exec) mode for review.
    Returns (success: bool, output: str).
    """
    output_file = LOGS / f"codex_output_{int(time.time())}.txt"
    cmd = [
        "codex", "exec",
        "--model", model,
        "--dangerously-bypass-approvals-and-sandbox",
        "-o", str(output_file),
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        # Read the output file if it exists
        output = ""
        if output_file.exists():
            output = output_file.read_text(encoding="utf-8", errors="replace")
        if not output:
            output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def c_file_paths(module: dict) -> str:
    """Return newline-separated list of C file paths for a module."""
    return "\n".join(f"  reference/{f}" for f in module["c_files"])


def read_file_contents(paths: list[str], base: Path = REFERENCE) -> str:
    """Read and concatenate file contents with headers."""
    parts = []
    for p in paths:
        full = base / p
        if full.exists():
            content = full.read_text(encoding="utf-8", errors="replace")
            parts.append(f"=== {p} ===\n{content}\n")
        else:
            parts.append(f"=== {p} === (FILE NOT FOUND)\n")
    return "\n".join(parts)


def completed_module_names(state: dict) -> str:
    """Return comma-separated list of modules that are at least implemented."""
    done = [name for name, status in state.get("module_status", {}).items()
            if status in ("implemented", "reviewed", "passing")]
    return ", ".join(done) if done else "(none yet)"


def write_artifact(name: str, content: str, directory: Path = ASSETS):
    """Write a generated artifact to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Phase: DOCUMENT
# ---------------------------------------------------------------------------

def _document_one_module(module: dict, state: dict, log: logging.Logger) -> tuple[str, bool]:
    """Document a single module. Returns (name, success)."""
    name = module["name"]
    doc_file = ASSETS / f"{name}_architecture.md"

    if doc_file.exists() and doc_file.stat().st_size > 500:
        log.info(f"  [{name}] Already documented, skipping")
        with _state_lock:
            state["module_status"].setdefault(name, "documented")
        return name, True

    log.info(f"  [{name}] Generating architecture doc...")
    prompt = PROMPTS["document_module"].format(
        module_name=name,
        file_list=c_file_paths(module),
    )

    success, output = invoke_claude(
        prompt, add_dirs=[str(REFERENCE)], timeout=MAX_AGENT_TIMEOUT
    )

    if success and len(output) > 500:
        write_artifact(f"{name}_architecture.md", output)
        with _state_lock:
            state["module_status"][name] = "documented"
        save_state(state)
        log.info(f"  [{name}] Done ({len(output)} chars)")
        return name, True
    else:
        log.error(f"  [{name}] FAILED - output length: {len(output)}")
        log.debug(f"  Output: {output[:500]}")
        return name, False


def phase_document(state: dict, log: logging.Logger) -> bool:
    """Generate architecture documentation for each module (parallel)."""
    log.info("=" * 60)
    log.info(f"PHASE: DOCUMENT - Generating architecture docs ({MAX_PARALLEL_AGENTS} parallel)")
    log.info("=" * 60)

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as pool:
        futures = {
            pool.submit(_document_one_module, mod, state, log): mod["name"]
            for mod in MODULES
        }
        for future in as_completed(futures):
            name, success = future.result()
            if not success:
                failed.append(name)

    if failed:
        log.error(f"  Documentation FAILED for: {', '.join(failed)}")
        return False

    state["completed_phases"].append("document")
    save_state(state)
    log.info("PHASE: DOCUMENT complete")
    return True


# ---------------------------------------------------------------------------
# Phase: HLD
# ---------------------------------------------------------------------------

def phase_hld(state: dict, log: logging.Logger) -> bool:
    """Generate High-Level Design document."""
    log.info("=" * 60)
    log.info("PHASE: HLD - Generating High-Level Design")
    log.info("=" * 60)

    hld_file = ASSETS / "hld.md"
    if hld_file.exists():
        log.info("  HLD already exists, skipping")
        if "hld" not in state["completed_phases"]:
            state["completed_phases"].append("hld")
            save_state(state)
        return True

    prompt = PROMPTS["hld"]
    success, output = invoke_claude(
        prompt, add_dirs=[str(ASSETS), str(REFERENCE)], timeout=MAX_AGENT_TIMEOUT
    )

    if success and len(output) > 2000:
        write_artifact("hld.md", output)
        state["completed_phases"].append("hld")
        save_state(state)
        log.info(f"  HLD generated ({len(output)} chars)")
        return True
    else:
        log.error(f"  HLD generation FAILED - output length: {len(output)}")
        log.debug(f"  Output: {output[:500]}")
        return False


# ---------------------------------------------------------------------------
# Phase: TEST HARNESS
# ---------------------------------------------------------------------------

def phase_test_harness(state: dict, log: logging.Logger) -> bool:
    """Build the comparison test harness."""
    log.info("=" * 60)
    log.info("PHASE: TEST_HARNESS - Building comparison framework")
    log.info("=" * 60)

    prompt = PROMPTS["test_harness"]
    success, output = invoke_claude(
        prompt, add_dirs=[str(REFERENCE), str(ASSETS)], timeout=600
    )

    if not success:
        log.error("  Test harness generation FAILED")
        log.debug(f"  Output: {output[:1000]}")
        return False

    log.info("  Test harness files generated, verifying build...")

    # Try to compile
    build_result = subprocess.run(
        ["cargo", "build"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace",
    )

    if build_result.returncode == 0:
        state["completed_phases"].append("test_harness")
        save_state(state)
        log.info("  Test harness builds successfully")
        return True
    else:
        log.warning("  Build failed, asking Claude to fix...")
        log.debug(f"  Build errors: {build_result.stderr[:2000]}")

        fix_prompt = (
            f"The test harness build failed with these errors. Fix them.\n\n"
            f"```\n{build_result.stderr[:3000]}\n```"
        )
        fix_ok, fix_output = invoke_claude(fix_prompt, timeout=MAX_AGENT_TIMEOUT)

        # Retry build
        retry = subprocess.run(
            ["cargo", "build"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
        if retry.returncode == 0:
            state["completed_phases"].append("test_harness")
            save_state(state)
            log.info("  Test harness builds after fix")
            return True
        else:
            log.error("  Test harness still broken after fix attempt")
            log.debug(f"  Errors: {retry.stderr[:1000]}")
            return False


# ---------------------------------------------------------------------------
# Phase: IMPLEMENT (iterative)
# ---------------------------------------------------------------------------

MAX_FIX_ATTEMPTS = 10

def implement_module(module: dict, state: dict, log: logging.Logger) -> bool:
    """Implement a single module: code -> review -> test -> fix loop (thread-safe)."""
    name = module["name"]
    with _state_lock:
        status = state["module_status"].get(name, "")

    if status == "passing":
        log.info(f"  [{name}] Already passing, skipping")
        return True

    # Step 1: Implement (parallel-safe: each module writes its own files)
    if status not in ("implemented", "reviewed"):
        log.info(f"  [{name}] Implementing...")
        prompt = PROMPTS["implement_module"].format(
            module_name=name,
            file_list=c_file_paths(module),
            completed_modules=completed_module_names(state),
            rust_module=module["rust_module"],
        )
        success, output = invoke_claude(
            prompt, add_dirs=[str(REFERENCE), str(ASSETS)], timeout=MAX_AGENT_TIMEOUT
        )
        if not success:
            # Check if Claude wrote code files before timing out
            rust_path = SRC / module["rust_module"].replace("/", os.sep)
            wrote_files = any(
                f.suffix == ".rs" and f.stat().st_size > 200
                for f in rust_path.parent.rglob("*.rs")
            ) if rust_path.parent.exists() else False
            if wrote_files:
                log.warning(f"  [{name}] Claude returned failure but wrote code files - treating as success")
            else:
                log.error(f"  [{name}] Implementation FAILED (no code files written)")
                return False
        with _state_lock:
            state["module_status"][name] = "implemented"
        save_state(state)
        log.info(f"  [{name}] Implemented")

    # Step 2: Adversarial review via Codex
    with _state_lock:
        status = state["module_status"].get(name, "")
    if status != "reviewed":
        log.info(f"  [{name}] Requesting adversarial review from Codex...")
        review_prompt = PROMPTS["review_module"].format(
            module_name=name,
            file_list=c_file_paths(module),
            rust_module=module["rust_module"],
        )
        review_ok, review_output = invoke_codex(review_prompt, timeout=MAX_AGENT_TIMEOUT)
        review_path = write_artifact(f"{name}_review.md", review_output)
        log.info(f"  [{name}] Review saved to {review_path}")

        # Check if review found critical issues
        has_critical = "CRITICAL" in review_output.upper()
        if has_critical:
            log.warning(f"  [{name}] Review found CRITICAL issues, feeding back to Claude")
            fix_prompt = (
                f"Codex review found critical issues in the {name} module.\n\n"
                f"## Review\n{review_output}\n\n"
                f"Fix all CRITICAL issues. The C reference is always correct."
            )
            fix_ok, fix_output = invoke_claude(
                fix_prompt, add_dirs=[str(REFERENCE), str(ASSETS)], timeout=MAX_AGENT_TIMEOUT
            )

        with _state_lock:
            state["module_status"][name] = "reviewed"
        save_state(state)

    # Step 3: Build and test loop (serialized — cargo can't run concurrently)
    with _state_lock:
        attempts = state["attempts"].get(name, 0)

    while attempts < MAX_FIX_ATTEMPTS:
        log.info(f"  [{name}] Build/test attempt {attempts + 1}/{MAX_FIX_ATTEMPTS}")

        with _build_lock:
            build = subprocess.run(
                ["cargo", "build"], cwd=str(ROOT),
                capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace",
            )
            if build.returncode != 0:
                error_output = build.stderr[:3000]
                log.warning(f"  [{name}] Build failed, attempting fix...")
            else:
                # Run tests
                test = subprocess.run(
                    ["cargo", "test", "--", "--nocapture"],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=600,
                    encoding="utf-8", errors="replace",
                )
                if test.returncode == 0:
                    with _state_lock:
                        state["module_status"][name] = "passing"
                        state["attempts"][name] = attempts
                    save_state(state)
                    log.info(f"  [{name}] All tests PASS")
                    return True
                error_output = test.stdout + test.stderr
                error_output = error_output[-3000:]  # Keep tail
                log.warning(f"  [{name}] Tests failed, attempting fix...")

        # Ask Claude to fix (outside build lock — doesn't need cargo)
        fix_prompt = PROMPTS["fix_errors"].format(
            module_name=name,
            error_output=error_output,
            rust_module=module["rust_module"],
            file_list=c_file_paths(module),
        )
        log.info(f"  [{name}] Sending fix to Claude...")
        t0 = time.time()
        fix_ok, fix_out = invoke_claude(
            fix_prompt, add_dirs=[str(REFERENCE), str(ASSETS)], timeout=MAX_AGENT_TIMEOUT
        )
        elapsed = time.time() - t0
        log.info(f"  [{name}] Fix returned in {elapsed:.0f}s, ok={fix_ok}, len={len(fix_out)}")
        if elapsed < 10:
            log.warning(f"  [{name}] Fix returned suspiciously fast — likely no changes made")
        attempts += 1
        with _state_lock:
            state["attempts"][name] = attempts
        save_state(state)

    log.error(f"  [{name}] Exhausted {MAX_FIX_ATTEMPTS} fix attempts")
    return False


def _deps_satisfied(module: dict, state: dict) -> bool:
    """Check if all dependencies of a module are in 'passing' state."""
    for dep in module["deps"]:
        if state["module_status"].get(dep, "") != "passing":
            return False
    return True


def phase_implement(state: dict, log: logging.Logger) -> bool:
    """Implement all modules in dependency-wave parallel order."""
    log.info("=" * 60)
    log.info(f"PHASE: IMPLEMENT - Porting modules to Rust ({MAX_PARALLEL_AGENTS} parallel)")
    log.info("=" * 60)

    remaining = [m for m in MODULES
                 if state["module_status"].get(m["name"], "") != "passing"]

    while remaining:
        # Find modules whose deps are all satisfied
        ready = [m for m in remaining if _deps_satisfied(m, state)]
        if not ready:
            blocked = [m["name"] for m in remaining]
            log.error(f"  Deadlock: no modules ready. Blocked: {blocked}")
            return False

        ready_names = [m["name"] for m in ready]
        log.info(f"  Wave: implementing {len(ready)} modules in parallel: {ready_names}")

        failed = []
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as pool:
            futures = {
                pool.submit(implement_module, mod, state, log): mod["name"]
                for mod in ready
            }
            for future in as_completed(futures):
                name = futures[future]
                success = future.result()
                if not success:
                    failed.append(name)

        if failed:
            log.error(f"  Wave FAILED for: {', '.join(failed)}")
            return False

        remaining = [m for m in MODULES
                     if state["module_status"].get(m["name"], "") != "passing"]

    state["completed_phases"].append("implement")
    save_state(state)
    log.info("PHASE: IMPLEMENT complete - all modules passing")
    return True


# ---------------------------------------------------------------------------
# Phase: INTEGRATE
# ---------------------------------------------------------------------------

def phase_integrate(state: dict, log: logging.Logger) -> bool:
    """Full integration testing with WAV/Opus files."""
    log.info("=" * 60)
    log.info("PHASE: INTEGRATE - Full pipeline testing")
    log.info("=" * 60)

    # Generate test vectors if needed
    test_dir = ROOT / "tests" / "vectors"
    if not test_dir.exists():
        log.info("  Generating test vectors...")
        prompt = textwrap.dedent("""\
            Create test audio vectors for the opus comparison harness.
            Generate small WAV files programmatically (Rust code) covering:
            1. Silence (all zeros)
            2. Single frequency sine waves (440Hz, 1kHz)
            3. White noise
            4. Impulse (single sample spike)
            5. Full-scale square wave
            Each should be 1 second, 48kHz, mono, 16-bit PCM.
            Write them to tests/vectors/ directory.
            Create a Rust test that generates these files if they don't exist.
        """)
        invoke_claude(prompt, add_dirs=[str(REFERENCE)], timeout=600)

    # Run full comparison suite
    log.info("  Running full comparison suite...")
    test_result = subprocess.run(
        ["cargo", "test", "--test", "integration", "--", "--nocapture"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace",
    )

    if test_result.returncode == 0:
        state["completed_phases"].append("integrate")
        save_state(state)
        log.info("PHASE: INTEGRATE complete - 100% match")
        return True
    else:
        log.error("  Integration tests FAILED")
        log.info(f"  Output: {test_result.stdout[-2000:]}")
        log.info(f"  Errors: {test_result.stderr[-2000:]}")
        return False


# ---------------------------------------------------------------------------
# Phase dispatch
# ---------------------------------------------------------------------------

PHASES = {
    "document": phase_document,
    "hld": phase_hld,
    "test_harness": phase_test_harness,
    "implement": phase_implement,
    "integrate": phase_integrate,
}

PHASE_ORDER = ["document", "hld", "test_harness", "implement", "integrate"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Run all phases from current checkpoint."""
    log = setup_logging()
    state = load_state()

    if state["started_at"] is None:
        state["started_at"] = datetime.now().isoformat()
        save_state(state)

    log.info(f"mdopus coordinator starting at {datetime.now().isoformat()}")
    log.info(f"Project root: {ROOT}")
    log.info(f"State: {json.dumps(state, indent=2)}")

    for phase_name in PHASE_ORDER:
        if phase_name in state.get("completed_phases", []):
            log.info(f"Phase '{phase_name}' already completed, skipping")
            continue

        phase_fn = PHASES[phase_name]
        state["phase"] = phase_name
        save_state(state)

        success = phase_fn(state, log)
        if not success:
            log.error(f"Phase '{phase_name}' FAILED - coordinator stopping")
            log.error(f"Resume with: python coordinator.py resume")
            return 1

    log.info("=" * 60)
    log.info("ALL PHASES COMPLETE")
    log.info("=" * 60)
    return 0


def cmd_phase(args):
    """Run a single named phase."""
    log = setup_logging()
    state = load_state()

    phase_name = args.name
    if phase_name not in PHASES:
        print(f"Unknown phase: {phase_name}")
        print(f"Available: {', '.join(PHASE_ORDER)}")
        return 1

    log.info(f"Running single phase: {phase_name}")
    success = PHASES[phase_name](state, log)
    return 0 if success else 1


def cmd_status(args):
    """Print current coordinator status."""
    state = load_state()
    print(f"Current phase: {state.get('phase', 'not started')}")
    print(f"Completed phases: {', '.join(state.get('completed_phases', [])) or '(none)'}")
    print(f"Started: {state.get('started_at', 'never')}")
    print(f"Last updated: {state.get('last_updated', 'never')}")
    print()
    print("Module status:")
    for module in MODULES:
        name = module["name"]
        status = state.get("module_status", {}).get(name, "pending")
        attempts = state.get("attempts", {}).get(name, 0)
        attempt_str = f" ({attempts} fix attempts)" if attempts > 0 else ""
        print(f"  {name:25s} {status}{attempt_str}")
    return 0


def cmd_resume(args):
    """Resume from last checkpoint (alias for run)."""
    return cmd_run(args)


def main():
    parser = argparse.ArgumentParser(
        description="mdopus coordinator - multi-agent Opus C->Rust port"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run all phases")
    p_phase = sub.add_parser("phase", help="Run a specific phase")
    p_phase.add_argument("name", choices=PHASE_ORDER)
    sub.add_parser("status", help="Show current status")
    sub.add_parser("resume", help="Resume from checkpoint")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "run": cmd_run,
        "phase": cmd_phase,
        "status": cmd_status,
        "resume": cmd_resume,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
