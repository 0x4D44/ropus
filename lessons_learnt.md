- Build provenance must validate Git top-level ownership and watch HEAD plus its resolved ref (`ropus-tools-core/src/build_provenance.rs:discover`).

  A linked worktree stores `.git` as a file and branch commits move `refs/heads/*`, while vendored crates can inherit a consumer repository. Resolve Git paths through Git itself, compare the top-level to the package workspace, and emit `unknown` outside it.

- Validate auxiliary files before opening destinations, then atomically rename temp output (`commands/encode.rs:prepare_picture`).

  Read a cover image once through a `MAX+1` limit, reject output aliases, and keep regular-file writes in a same-directory temporary until the final flush succeeds. This preserves sentinels when the picture is missing, malformed, oversized, or replaced mid-read.

- Required behavior tests must synthesize small fixtures, not return success when checkout assets are absent (`tests/round_trip.rs`).

  Keep large/reference vectors optional, but make core CLI and library oracles deterministic in-test. If an external differential truly needs optional assets, mark it ignored and fail clearly when run manually; never print a skip and pass.

- Empty Ogg Opus packets are malformed, not PLC (`container/ogg.rs:validate_opus_audio_packet`); reject before decode or TOC parsing.

  The decoder's empty-slice contract deliberately means packet-loss concealment. Validate every container audio packet first so malformed zero-length payloads cannot fabricate audio, duration, or extended info.

- Never create a derived output before comparing filesystem identity (`util.rs:reject_input_output_alias`); metadata catches links.

  Compare lexical paths plus platform file identities before decoding or opening a writer. If a default extension equals the source, select a suffixed destination; explicit aliases must fail.

- Info scalar queries must select a bounded plan before opening payload (`commands/info.rs:QueryKey`); skip tags and TOCs.

  Parse the query first, read only OpusHead for fixed metadata, read OpusTags only for tag lookups, and stream raw Ogg pages rather than loading the whole file. Keep packet TOCs only for extended human output.

- Terminal-facing text needs one reversible control-escaping policy (`ui::escape_terminal_text`); raw metadata is safe only for non-TTY queries.

  Escape C0/C1, line separators, and bidi overrides in paths, tags, status lines, device listings, and error chains before printing.

- Public interleaved audio helpers must reject incomplete frames before `chunks_exact` or WAV header math (`audio/{downmix,resample,wav}.rs`).

  Validate channel/rate shape first, then use checked block-align, byte-rate, data-size, and frame-count calculations.

- JPEG MIME sniffing must match the SOI signature, not one marker layout (`container/picture.rs:detect_format`); APP2/APP14 are valid.

  Keep format detection deliberately shallow: accept `FF D8 FF` and leave full image parsing to the eventual consumer.

- Public command options need validation before any input/output work (`commands/{encode,decode,play}.rs`); Clap checks alone miss GUI/plugin callers and invalid Q8 casts can panic.

  Reject sentinel frame durations, complexity/bitrate/loss bounds, non-finite volume, and gains above Q8 32767 before opening files or devices.

- Reverse Ogg duration scans must validate full CRC-checked EOS pages (`container/ogg.rs:parse_duration_page`), not `OggS` bytes.

  Skip malformed, non-EOS, and unknown-granule candidates so payload data cannot fabricate duration or bitrate.

- Shared Opus decode must apply `OpusHead.output_gain` in `audio/decode.rs:decode_reader_with_gain`; keep CLI gain in Q8 for Opus and linear post-gain for other codecs.

- Shared OpusHead parsing must reject unsupported versions, channels, mappings, and family-0 trailing bytes (`container/ogg.rs:parse_opus_head`).

  Direct decode and info share this boundary; playback/transcode must parse the same header instead of extracting only pre-skip.

- PLC simulation must derive every lost packet's duration from its own TOC (`commands/decode.rs:packet_duration_samples`).

  A previous-packet fallback makes first-loss and duration-switch streams drift; validate code-3 counts and the 120 ms cap.

- Ogg Opus decode must clamp to the absolute EOS granule (`audio/decode.rs:decode_reader`), not packet padding.

  The direct reader gets the endpoint from `Packet::last_in_stream`; Symphonia exposes it as `codec_params.n_frames`.
  Reject a selected stream that reaches physical EOF without EOS before publishing playback, transcode, or WAV output.

- Clear fb2k last-error state with `Option<CString>` plus a static NUL pointer (`error.rs:clear_last_error`), not a new empty allocation.

  Successful decode calls can drop stale error text and reset the code without touching the heap; test the static pointer as the steady-state oracle.

- Validate OpusTags field-name bytes before ASCII uppercasing (`tags.rs:parse`); UTF-8 validity alone permits illegal keys.

  Vorbis keys must stay in ASCII `0x20..=0x7D` excluding `=`; reject controls and non-ASCII bytes with a typed error.

- Stop fb2k decode at `Packet::last_in_stream()` before reading a chained OpusHead (`reader.rs:decode_next`).

  PacketReader spans physical pages and logical chains; track sticky selected-stream EOS so later headers never reach OpusDecoder.

- Reverse Ogg duration scans must validate complete CRC-checked EOS pages (`reader.rs:parse_duration_page`), not `OggS` bytes.

  Check lacing extent, reserved flags, stream serial, and checksum before trusting a granule; payloads can contain header-shaped bytes.

- Transactional fb2k seek rollback must restore logical samples, not a raw Ogg cursor (`reader.rs:restore_pending_seek`).

  `PacketReader` can buffer a whole page, so replay from the audio start and discard to the saved absolute sample after cancellation.

- Apply signed OpusHead Q7.8 output gain during fb2k decoder init (`reader.rs:decode_next`); reset preserves it.

  Set the header gain once on lazy `OpusDecoder` construction; its reset path keeps `decode_gain`, so seek does not drop it.

- Phase-C stereo traces need occurrence keys (`fuzz_repro_diff.rs:keyed_v2_tuples`) or channel-0 mismatches get overwritten.

  Rust uses a `-1` channel sentinel, so key V2 records by boundary, iteration, and trace-order occurrence; keep C channel labels in diagnostics.

- Ogg Opus EOS granules cap decoded output (`ropus-fb2k/src/reader.rs:decode_next`); a full final packet can contain padding.

  Keep the absolute EOS granule separate from unknown duration, reject it when it precedes pre-skip, and make EOF sticky after clamping.

- Decode framing must be validated once before comparison (`harness/src/cli.rs:parse_framed_packets`).

  A partial length or payload can leave both decoders with the same prefix; require exact EOF and return `Fail` before comparing that prefix.

- Validate numeric CLI values before dispatch (`harness/src/cli.rs:parse_bitrate`).

  Typed range checks stop negative casts, modulo-zero intervals, unsupported rates, and unbounded duration/iteration work; keep zero as an explicit disable only where the command documents it.

- Keep RIFF parsing fallible and shared (`harness/src/wav.rs:parse_pcm16_wav`).

  Check the container and every padded chunk extent before slicing, then validate PCM format and sample alignment once for all harness callers.

- Validate control packet streams before decoder/output allocation (`ctrl_decode_fixed.rs:validate_packets_file`).

  Bound Opus headers, packet lengths, frame/output counts, lost flags, payload extents, and trailing bytes; reuse the same contract in float mode.

- RSS leak checks must baseline after fixed setup and tear down before sampling (`harness/src/cli.rs:cmd_torture`).

  Stream deterministic input through a bounded frame buffer; otherwise duration-sized PCM dominates the delta and masks real growth.

- Benchmark timers must bracket only configured work; centralize C lifecycle ordering (`harness/src/cli.rs:timed_c_lifecycle`).

  Inject timer callbacks in the lifecycle test so construction/configuration and destruction cannot drift back into the measured interval.

- Preserve one-sided fuzz errors; only `(Err, Err)` is ignored (`replay_fuzz_decode.rs:classify_decoder_status`).

  A replay oracle must model every status quadrant enforced by the fuzz target, not only the PCM-success path.

- Validate Ogg pages before corpus diff; `PacketReader` hides truncated tails (`corpus_diff.rs:validate_ogg_container`).

  A valid decoded prefix is not a complete corpus match. Validate `OpusTags`, classify structural errors separately from exploratory skips, and require clean EOF before recording a match.

- `Command::output()` buffers untrusted child output before any cap; drain stdout/stderr concurrently into bounded prefix+tail buffers (`full-test/src/process_capture.rs:output`).

  Apply the helper to every runner subprocess, bound coverage JSON by file size, and read temp-file logs through the same bounded reader.

- The full-test HTML report is a required output: atomically rename a complete temporary file and fail the run if delivery fails (`full-test/src/main.rs:attempt_report`).

  PASS/WARN describes validation only; a missing primary report must still return nonzero and show its write error in the summary.

- C ABI version strings should use `env!("CARGO_PKG_VERSION")` so package bumps cannot leave stale diagnostics (`capi/src/lib.rs:VERSION_STRING`).

  Keep the C-compatible NUL terminator in the compile-time string and test the exported pointer against the same package metadata.

- C ABI constructors must stage fallible handle/state allocations before publishing outputs (`capi/src/alloc.rs:try_box`).

  `handle_alloc_error`, `Box::new`, and `Vec::with_capacity` can abort instead of
  returning `OPUS_ALLOC_FAIL`. Allocate through the fallible helpers, release any
  staged sub-handles on failure, and only write C output parameters after the
  complete handle is ready.

- Multistream decoder CTLs need matching C varargs cases and typed Rust arms (`capi/src/ctl_shim.c:opus_multistream_decoder_ctl`).

  A request can be implemented by `OpusMSDecoder` yet still return `OPUS_UNIMPLEMENTED` when the C switch omits it; test the complete C-to-Rust round trip and reject invalid values before fan-out.

- Extension parsers must validate slice capacities and checked frame-count prefixes before indexing (`ropus/src/opus/repacketizer.rs:opus_packet_extensions_parse`).

  Safe slices do not make caller-provided counts safe: reject lengths beyond the input, capacities beyond output slices, negative counts, and overflowing frame prefixes before iterating.

- Multistream decoder resets must clear neural PLC history via `lpcnet.reset()` (`ropus/src/opus/decoder.rs:OpusDecoder::ms_reset`).

  Keep `ms_reset` aligned with the canonical decoder reset: clear FEC/GRU and analysis state while preserving loaded neural model weights.

- Validate expert frame-duration controls before storing them (`ropus/src/opus/multistream.rs:OpusMSEncoder::set_expert_frame_duration`).

  Multistream setters should mirror the single-stream CTL's accepted sentinel and duration range, return `OPUS_BAD_ARG`, and leave the prior duration unchanged; C callers must receive that status instead of an unconditional success.

- Route multistream decoder CTLs through validated OpusDecoder setters (`ropus/src/opus/multistream.rs:OpusMSDecoder::set_complexity`).

  Internal fan-out helpers must preserve public range checks and side effects. Calling raw field assignments can accept invalid values and leave the CELT decoder's complexity out of sync.

- Guard projection frame sizes before buffer allocation (`ropus/src/opus/multistream.rs:OpusProjectionEncoder::encode`).

  Wrapper methods must reject non-positive frame sizes before converting them to `usize`; delegating validation to the underlying multistream codec is too late when the wrapper sizes a temporary buffer first.

- Validate safe-slice lengths before fixed-size constructor copies (`ropus/src/opus/multistream.rs:OpusMSEncoder::new_impl`).

  Safe slices prevent memory unsafety, but direct indexing can still panic when a caller passes a short mapping or projection matrix. Return `OPUS_BAD_ARG` before copying, and use checked arithmetic for projection dimensions.

- Provision ignored C-reference assets before fuzz checks (`.deltic-integrate.toml:harness/fuzz`).

  A clean task worktree has no tracked `reference/` tree, so the fuzz build
  script cannot assume a prior manual fetch. Reuse the pinned, idempotent
  `fetch-assets reference` command in every gate that enters the fuzz crate.

- Check excluded Cargo workspaces with their own manifest (`.deltic-integrate.toml:fuzz`).

  Root `cargo fmt --all` can miss a standalone workspace, while shared source
  included by another package may use a different edition's import ordering.
  Gate the nested manifest explicitly and document any unavoidable shared-file
  formatter conflict narrowly.

- Route CLI banners from typed paths, never raw argv scans (`ropus-tools-core/src/prelude.rs:output_is_stdout`).

  A raw scan cannot distinguish positional inputs from values consumed by
  earlier options. Parse once with Clap, then derive stdout and query routing
  from the resulting fields so text never contaminates binary or scalar output.

- Drain Opus lookahead, but set EOS to source + pre-skip (`ropus-tools-core/src/commands/encode.rs:encode`).

  Packet input must cover the source plus encoder delay and round up to a full
  frame. The EOS granule deliberately excludes packet padding, so subtracting
  `pre_skip` recovers the exact source duration.

- Benchmark guards must inspect dependency features after Cargo unification (`ropus/src/lib.rs:SILK_ENCODE_TRACE_ENABLED`).

  A harness-local feature flag is not enough: another selected workspace package
  can enable instrumentation on the shared dependency. Export the dependency's
  compiled state and make the benchmark reject any instrumented build.

- Cache FFI capacities before range calls; slice types alone cannot prove C writes fit (`harness-deep-plc/src/lib.rs`).

  A safe wrapper must validate signed dimensions, checked products, integer
  conversions, and runtime buffer capacities before crossing the FFI boundary.
  Null-pointer sentinel tests can prove rejection happens before the C call.

- ARM DNN parity needs `DISABLE_NEON`: header kernels bypass source filters (`harness-deep-plc/build.rs`).

  Excluding `dnn/arm/*.c` does not select scalar inference because
  `reference/dnn/vec.h` dispatches to inline NEON code. Differential harnesses
  must pin the intended arithmetic path explicitly.
