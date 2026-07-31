# ROP-BUG-FLUX-00033 — Decode command accepts truncated framed input as a passing comparison

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/decode-framing
- **Raised:** 2026-07-31
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. Both framed decoders stop and return prior PCM when a declared packet overruns EOF at `/Users/md/language/ropus/harness/src/cli.rs:558` through `:588` and `:714` through `:742`; a trailing single byte is ignored because both loops require a complete length field. `cmd_decode` at `:1119` through `:1151` compares only those equal partial outputs and can return Pass. Expected: the documented length-prefixed input must end exactly on a packet boundary. Actual: a valid prefix plus truncated tail can print an error and still exit 0 as a bit-exact comparison. Fix: parse framing once into a fallible packet list, require exact EOF, propagate framing/decode failures to the command outcome, and add truncated-length and truncated-payload cases. Static review only; no decode command ran.

## Fix

<unfixed — raised only>

## Notes

### Float control framing also accepts ambiguous records (2026-07-31)

`/Users/md/language/ropus/harness-deep-plc/src/bin_inner/ctrl_decode_float.rs:94-121`
stops after the declared frame count without requiring exact EOF. A LOST record
with a nonzero low-bit length is treated as PLC without consuming its declared
payload, desynchronising later records. Later short reads panic after partial
PCM has already been written. Share one fallible framing parser with the fixed
control path, require LOST length zero and exact EOF, and publish output only
after a complete valid stream. Static review at `origin/main` `b65f812`; no
control decoder or packet stream ran.

### Lossless control accepts matching truncated output (2026-07-31)

The lossy control asserts the full expected output length at
`/Users/md/language/ropus/harness-control/tests/control_snr.rs:304-315`, but
the lossless path at `:368-386` checks only that both children have the same
length. If both decoders return the same nonempty prefix after a shared framing
or loop regression, SNR is infinite and the test passes despite missing audio.
Use one shared validator that requires exactly
`TOTAL_FRAMES * FRAME_SIZE * CHANNELS` samples, nonempty signal energy, and a
finite-or-explicitly-identical SNR policy before either comparison. Static
review at `origin/main` `1ae9e50`; no control decoder or test ran.
