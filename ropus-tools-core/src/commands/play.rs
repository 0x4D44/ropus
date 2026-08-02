//! Play: directory-aware playlist player with status line + keyboard controls.
//!
//! Top-level flow:
//! - `build_playlist` expands a file or directory into a sorted list of paths.
//! - For each entry we decode through `decode_to_f32_with_gain`, read its `OpusTags`,
//!   hand the samples to a fresh `rodio::Sink`, and enter `run_track_loop`
//!   (interactive) or `run_track_noninteractive` (piped/quiet) to watch for
//!   track end + keyboard input.
//! - Per-track decode errors are logged with a yellow `warning:` prefix and
//!   skipped via `advance_on_error` so one corrupt file does not end a session.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use colored::*;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::tty::IsTty;
use ogg::reading::PacketReader;
use rodio::cpal::traits::{DeviceTrait, HostTrait};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::audio::decode::{DecodedAudio, MAX_GAIN_DB, MIN_GAIN_DB, decode_to_f32_with_gain};
use crate::container::ogg::OpusTags;
use crate::options::{LoopMode, PlayOptions};
use crate::ui::{escape_terminal_path, escape_terminal_text};

/// What ended a track. Drives the index-advancement logic in the main FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Sink emptied naturally — respect `loop_mode` for what comes next.
    TrackFinished,
    /// User pressed `n`.
    Next,
    /// User pressed `p`.
    Prev,
    /// User pressed `q` or Ctrl-C.
    Quit,
}

/// Terminal mode operations used by [`RawModeGuard`]. A small seam keeps the
/// restore path unit-testable without requiring a real interactive terminal.
trait TerminalMode {
    fn enable_raw_mode(&mut self) -> io::Result<()>;
    fn disable_raw_mode(&mut self) -> io::Result<()>;
}

struct CrosstermTerminal;

impl TerminalMode for CrosstermTerminal {
    fn enable_raw_mode(&mut self) -> io::Result<()> {
        crossterm::terminal::enable_raw_mode()
    }

    fn disable_raw_mode(&mut self) -> io::Result<()> {
        crossterm::terminal::disable_raw_mode()
    }
}

/// Enables terminal raw mode on construction. Normal/error exits call
/// [`RawModeGuard::restore`] explicitly so failures are reported; `Drop`
/// remains the best-effort fallback for panic/unwind paths.
struct RawModeGuard<'a, T: TerminalMode> {
    terminal: &'a mut T,
    active: bool,
}

impl<'a, T: TerminalMode> RawModeGuard<'a, T> {
    fn enable(terminal: &'a mut T) -> Result<Self> {
        terminal
            .enable_raw_mode()
            .context("enabling terminal raw mode")?;
        Ok(Self {
            terminal,
            active: true,
        })
    }

    fn restore(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        self.terminal
            .disable_raw_mode()
            .context("restoring terminal raw mode")?;
        self.active = false;
        Ok(())
    }
}

impl<T: TerminalMode> Drop for RawModeGuard<'_, T> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.terminal.disable_raw_mode();
        }
    }
}

pub fn play(opts: PlayOptions) -> Result<()> {
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    play_with_io(opts, &mut stdout, &mut stderr)
}

fn play_with_io<W: Write, E: Write>(
    opts: PlayOptions,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<()> {
    validate_volume(opts.volume)?;
    // Validate gain before even the device-list shortcut. Direct library
    // callers must not be able to bypass the same option boundary as Clap.
    validate_and_compute_gain(opts.gain_db)?;

    // `--list-devices` short-circuits before the banner, any file I/O, and any
    // audio-output initialisation: we only need a host to walk for names.
    if opts.list_devices {
        return list_output_devices(stdout);
    }

    // Resolve the audio device *before* touching the filesystem. A bogus
    // device name should surface as "wrong config" without first reporting a
    // missing input file.
    let (_stream, handle) = match &opts.device {
        Some(name) => open_named_output_stream(name)?,
        None => rodio::OutputStream::try_default()
            .map_err(|e| anyhow!("no default audio output device available: {e}"))?,
    };

    write_output(stdout, format_args!("{}\n", "play".bright_yellow().bold()))?;

    let playlist = build_playlist(&opts.input)?;
    if playlist.len() == 1 {
        write_output(
            stdout,
            format_args!("file     {}\n", escape_terminal_path(&playlist[0]).cyan()),
        )?;
    } else {
        write_output(
            stdout,
            format_args!(
                "playlist {} tracks from {}\n",
                playlist.len().to_string().bright_white(),
                escape_terminal_path(&opts.input).cyan(),
            ),
        )?;
    }

    let interactive = interactive_enabled(opts.quiet);
    let mut terminal = CrosstermTerminal;
    // Keep the guard alive for the complete playback session. The closure
    // below captures all fallible output and playback errors; explicit
    // restoration afterward can therefore preserve both error causes.
    let mut raw = if interactive {
        Some(RawModeGuard::enable(&mut terminal)?)
    } else {
        None
    };

    let playback_result = (|| -> Result<()> {
        if interactive {
            // Raw mode disables the implicit line ending for `println!`, so
            // stitch `\r\n` explicitly for the help row + blank status row.
            write_output(
                stdout,
                format_args!("[space] pause  [n] next  [p] prev  [q] quit\r\n\r\n"),
            )?;
        }

        let mut idx: usize = 0;
        let mut consecutive_errors: usize = 0;
        let playlist_len = playlist.len();

        'outer: loop {
            let path = &playlist[idx];

            // Decode-latency UX: print an ephemeral `decoding …` line only on the
            // interactive TTY path. Redirected and quiet playback must stay plain
            // text; carriage returns and erase escapes make logs unreadable.
            if interactive {
                let stem = path
                    .file_stem()
                    .map(|s| escape_terminal_text(&s.to_string_lossy()))
                    .unwrap_or_else(|| escape_terminal_path(path));
                let cols = crossterm::terminal::size()
                    .map(|(w, _)| w as usize)
                    .unwrap_or(80);
                let prefix = format!("decoding {}/{}  ", idx + 1, playlist_len);
                let stem_fit = truncate_to_fit(&stem, cols, prefix.width(), 0);
                write_output(stdout, format_args!("\r{prefix}{stem_fit}"))?;
            }

            let (decoded, tags) = match decode_track(path, opts.gain_db) {
                Ok(ok) => ok,
                Err(e) => {
                    if interactive {
                        // Clear the ephemeral decoding line; the warning goes to
                        // stderr so it sits above the next status-line repaint.
                        write_output(stdout, format_args!("\r\x1b[K"))?;
                    }
                    write_output(
                        stderr,
                        format_args!(
                            "{} skipping {}: {}\n",
                            "warning:".yellow(),
                            escape_terminal_path(path),
                            escape_terminal_text(&e.to_string())
                        ),
                    )?;
                    consecutive_errors += 1;
                    if consecutive_errors >= playlist_len {
                        bail!("all {playlist_len} tracks failed to decode");
                    }
                    idx = match advance_on_error(idx, playlist_len, opts.loop_mode) {
                        Some(next) => next,
                        None => break 'outer,
                    };
                    continue 'outer;
                }
            };
            consecutive_errors = 0;

            let channels_u16 =
                u16::try_from(decoded.channels).map_err(|_| anyhow!("channel count overflow"))?;
            let duration = Duration::from_secs_f64(
                decoded.samples.len() as f64 / decoded.channels as f64 / decoded.sample_rate as f64,
            );
            let file_len = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let avg_kbps = compute_avg_kbps(file_len, duration.as_secs_f64());
            let display_name = resolve_display_name(&tags, path);

            let sink =
                rodio::Sink::try_new(&handle).map_err(|e| anyhow!("creating sink failed: {e}"))?;
            if let Some(v) = opts.volume {
                sink.set_volume(v.clamp(0.0, 1.0));
            }
            sink.append(rodio::buffer::SamplesBuffer::new(
                channels_u16,
                decoded.sample_rate,
                decoded.samples,
            ));

            let action = if interactive {
                run_track_loop(
                    stdout,
                    &sink,
                    &display_name,
                    duration,
                    idx,
                    playlist_len,
                    opts.loop_mode,
                    avg_kbps,
                )?
            } else {
                run_track_noninteractive(
                    stdout,
                    &sink,
                    &display_name,
                    duration,
                    idx,
                    playlist_len,
                )?;
                Action::TrackFinished
            };

            // Read the elapsed position while the sink is still in scope —
            // Prev's "restart-current vs previous-track" branch uses it.
            let pos_at_exit = sink.get_pos();

            match action {
                Action::TrackFinished => {
                    idx = match opts.loop_mode {
                        LoopMode::Single => idx,
                        LoopMode::All => (idx + 1) % playlist_len,
                        LoopMode::Off => {
                            if idx + 1 < playlist_len {
                                idx + 1
                            } else {
                                break 'outer;
                            }
                        }
                    };
                }
                Action::Next => idx = (idx + 1) % playlist_len,
                Action::Prev => {
                    // Convention: >2s means restart current, else step back.
                    if pos_at_exit <= Duration::from_secs(2) {
                        idx = if idx == 0 { playlist_len - 1 } else { idx - 1 };
                    }
                }
                Action::Quit => break 'outer,
            }
        }
        Ok(())
    })();

    let restore_result = raw
        .as_mut()
        .map(RawModeGuard::restore)
        .transpose()
        .map(|_| ());
    combine_playback_and_restore(playback_result, restore_result)
}

fn write_output<W: Write>(writer: &mut W, args: fmt::Arguments<'_>) -> Result<()> {
    writer.write_fmt(args).context("writing playback output")?;
    writer.flush().context("flushing playback output")?;
    Ok(())
}

fn combine_playback_and_restore<T>(
    playback_result: Result<T>,
    restore_result: Result<()>,
) -> Result<T> {
    match (playback_result, restore_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(playback), Ok(())) => Err(playback),
        (Ok(_), Err(restore)) => Err(restore),
        (Err(playback), Err(restore)) => {
            Err(playback.context(format!("terminal restoration also failed: {restore:#}")))
        }
    }
}

/// Validate `--gain DB` and convert it to a linear multiplier. Returns 1.0 for
/// the common `db == 0.0` case so callers can branch on "no-op" without a
/// floating-point equality check of their own. NaN / ±∞ and values outside
/// `[MIN_GAIN_DB, MAX_GAIN_DB]` dB both surface as a user-visible error. The
/// positive bound is the largest Q8 gain accepted by libopus.
pub(crate) fn validate_and_compute_gain(db: f32) -> Result<f32> {
    if !db.is_finite() {
        bail!("--gain must be a finite dB value (got {db})");
    }
    if !(MIN_GAIN_DB..=MAX_GAIN_DB).contains(&db) {
        bail!("--gain {db} dB out of range [{MIN_GAIN_DB}, {MAX_GAIN_DB}]");
    }
    if db == 0.0 {
        return Ok(1.0);
    }
    Ok(10f32.powf(db / 20.0))
}

fn validate_volume(volume: Option<f32>) -> Result<()> {
    if let Some(volume) = volume
        && !volume.is_finite()
    {
        bail!("--volume must be a finite value (got {volume})");
    }
    Ok(())
}

/// Legacy post-decode multiplier oracle retained for the gain unit tests. The
/// playback path now passes dB gain into the shared decoder, which applies
/// Opus gain before f32 conversion and keeps non-Opus policy in one place.
#[cfg(test)]
fn apply_gain(samples: &mut [f32], multiplier: f32) {
    if multiplier == 1.0 {
        return;
    }
    for s in samples.iter_mut() {
        *s *= multiplier;
    }
}

/// IO-free half of `list_output_devices`: format a slice of device names into
/// the stdout block, or return an error if the slice is empty. Split out so
/// the empty-list contract is unit-testable on headless CI where
/// `cpal::default_host().output_devices()` genuinely returns zero devices.
///
/// Output shape is one escaped name per line with a trailing newline, so
/// downstream scripts can match ordinary names with
/// `ropusplay --list-devices | grep -x "Speakers (Realtek)"` without worrying
/// about a missing final `\n`. C0/C1 controls are encoded as `\\u{NNNN}`;
/// matching in `--device` remains against the raw cpal name.
fn format_device_list(names: &[String]) -> Result<String> {
    if names.is_empty() {
        bail!("no output devices available on this host");
    }
    let mut out = String::new();
    for name in names {
        out.push_str(&escape_terminal_text(name));
        out.push('\n');
    }
    Ok(out)
}

/// Print every cpal output-device name (one per line) on stdout and return
/// `Ok(())`. Empty host list is treated as an error with exit code 1 so
/// scripts can `if ropusplay --list-devices; then …` without parsing stdout.
/// Enumeration + printing live here; the formatting + empty-list contract is
/// delegated to `format_device_list` so it is unit-testable.
fn list_output_devices<W: Write>(stdout: &mut W) -> Result<()> {
    let host = rodio::cpal::default_host();
    let devices = host
        .output_devices()
        .context("enumerating cpal output devices")?;
    let mut names: Vec<String> = Vec::new();
    for device in devices {
        // `device.name()` can fail (e.g. a device disconnected between enum
        // and query) — skip silently; we do not want one disconnected device
        // to abort the listing of the rest.
        if let Ok(name) = device.name() {
            names.push(name);
        }
    }
    let formatted = format_device_list(&names)?;
    write_output(stdout, format_args!("{formatted}"))
}

/// Find an output device by exact (case-sensitive) name and hand the caller a
/// fresh rodio stream + handle pair bound to it. Not-found errors list the
/// available devices on stderr so the user can spot typos without re-running
/// with `--list-devices`.
fn open_named_output_stream(
    name: &str,
) -> Result<(rodio::OutputStream, rodio::OutputStreamHandle)> {
    let host = rodio::cpal::default_host();
    let devices = host
        .output_devices()
        .context("enumerating cpal output devices")?;
    let mut available: Vec<String> = Vec::new();
    let mut matched: Option<rodio::cpal::Device> = None;
    for device in devices {
        if let Ok(dn) = device.name() {
            if dn == name {
                matched = Some(device);
                break;
            }
            available.push(dn);
        }
    }
    let device = matched.ok_or_else(|| {
        let mut msg = format!(
            "ropusplay: device '{}' not found. Available:",
            escape_terminal_text(name)
        );
        if available.is_empty() {
            msg.push_str("\n  (no cpal output devices)");
        } else {
            for d in &available {
                msg.push_str("\n  ");
                msg.push_str(&escape_terminal_text(d));
            }
        }
        anyhow!(msg)
    })?;
    rodio::OutputStream::try_from_device(&device).map_err(|e| {
        anyhow!(
            "opening output device '{}' failed: {e}",
            escape_terminal_text(name)
        )
    })
}

/// Interactive track loop. Polls for key events at 100 ms cadence, repainting
/// the status line only when its content changes. Returns when the sink drains
/// naturally (`Action::TrackFinished`) or the user presses `n` / `p` / `q` /
/// Ctrl-C.
#[allow(clippy::too_many_arguments)]
fn run_track_loop<W: Write>(
    stdout: &mut W,
    sink: &rodio::Sink,
    display_name: &str,
    duration: Duration,
    track_idx: usize,
    playlist_len: usize,
    loop_mode: LoopMode,
    avg_kbps: f64,
) -> Result<Action> {
    let mut paused = false;
    let mut last_rendered: Option<String> = None;

    loop {
        let cols = crossterm::terminal::size()
            .map(|(w, _)| w as usize)
            .unwrap_or(80);
        let pos = sink.get_pos();
        let status = format_status_line(
            cols,
            paused,
            loop_mode,
            display_name,
            track_idx,
            playlist_len,
            pos,
            duration,
            avg_kbps,
        );
        if last_rendered.as_deref() != Some(status.as_str()) {
            write_output(stdout, format_args!("\r\x1b[K{status}"))?;
            last_rendered = Some(status);
        }

        if sink.empty() {
            // Move off the status line so subsequent output (next track's
            // decoding line or shell prompt) does not overwrite it.
            write_output(stdout, format_args!("\r\n"))?;
            return Ok(Action::TrackFinished);
        }

        if event::poll(Duration::from_millis(100))? {
            let ev = event::read()?;
            if let Event::Key(key) = ev {
                // Windows fires both Press and Release; act on Press only so
                // each tap produces exactly one action.
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                // `KeyModifiers` is a bitflag set — Caps Lock + Ctrl-C arrives
                // as `CONTROL | SHIFT` on some platforms, so match on the code
                // first and use `.contains()` for the Ctrl-C guard instead of
                // exact modifier equality.
                match key.code {
                    KeyCode::Char(' ') => {
                        paused = !paused;
                        if paused {
                            sink.pause();
                        } else {
                            sink.play();
                        }
                        // Force a repaint on the next tick so the glyph flips
                        // immediately even within the same wall-clock second.
                        last_rendered = None;
                    }
                    KeyCode::Char('n') => {
                        write_output(stdout, format_args!("\r\n"))?;
                        return Ok(Action::Next);
                    }
                    KeyCode::Char('p') => {
                        write_output(stdout, format_args!("\r\n"))?;
                        return Ok(Action::Prev);
                    }
                    KeyCode::Char('q') => {
                        write_output(stdout, format_args!("\r\n"))?;
                        return Ok(Action::Quit);
                    }
                    KeyCode::Char('c') | KeyCode::Char('C')
                        if key.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        // In raw mode on Windows, Ctrl-C is a KeyEvent, not a
                        // signal. Without this branch the user would have no
                        // way to exit.
                        write_output(stdout, format_args!("\r\n"))?;
                        return Ok(Action::Quit);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Non-interactive fallback: print one summary line and block until the sink
/// drains. Used when stdout or stdin is not a tty, or when `-q/--quiet` is
/// set — both cases where a repainting status line would be noise.
fn run_track_noninteractive<W: Write>(
    stdout: &mut W,
    sink: &rodio::Sink,
    display_name: &str,
    duration: Duration,
    track_idx: usize,
    playlist_len: usize,
) -> Result<()> {
    let total = duration.as_secs();
    let clock = if total >= 3600 {
        format!(
            "{}:{:02}:{:02}",
            total / 3600,
            (total % 3600) / 60,
            total % 60
        )
    } else {
        format!("{}:{:02}", total / 60, total % 60)
    };
    write_output(
        stdout,
        format_args!(
            "playing {}/{}  {}  ({clock})\n",
            track_idx + 1,
            playlist_len,
            escape_terminal_text(display_name),
        ),
    )?;
    sink.sleep_until_end();
    Ok(())
}

/// Average bitrate in kbps computed from on-disk byte size and decoded
/// duration. Returns 0.0 for zero-duration inputs rather than producing a
/// NaN/infinity that would poison the `{:.0}` format in the status line.
fn compute_avg_kbps(file_len_bytes: u64, duration_secs: f64) -> f64 {
    if duration_secs > 0.0 {
        (file_len_bytes as f64 * 8.0) / duration_secs / 1000.0
    } else {
        0.0
    }
}

/// Decode a file to interleaved f32 and (best-effort) read its OpusTags. Tags
/// are optional — non-Opus inputs and tag-read failures both yield
/// `OpusTags::default()` rather than failing the whole track. A failed decode
/// is a hard error; callers map it into a per-track warning + skip.
fn decode_track(path: &Path, gain_db: f32) -> Result<(DecodedAudio, OpusTags)> {
    let decoded = decode_to_f32_with_gain(path, gain_db)?;
    let tags = read_opus_tags_from_file(path).unwrap_or_default();
    Ok((decoded, tags))
}

/// Open `path` as Ogg and try to parse the second packet as `OpusTags`. Any
/// failure (wrong extension, truncated file, non-Opus container, malformed
/// tags) returns `None` — the caller falls back to the filename stem for
/// display, which is exactly what we want for MP3/FLAC inputs.
fn read_opus_tags_from_file(path: &Path) -> Option<OpusTags> {
    let is_opus = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("opus"));
    if !is_opus {
        return None;
    }
    let file = File::open(path).ok()?;
    let mut reader = PacketReader::new(BufReader::new(file));
    // First packet is OpusHead; skip it.
    let _head = reader.read_packet().ok().flatten()?;
    let tags_pkt = reader.read_packet().ok().flatten()?;
    OpusTags::parse(&tags_pkt.data).ok()
}

/// Raw mode + status line only make sense when both ends of the pipe are a
/// real terminal. Piped stdout, redirected stdin, or `-q/--quiet` all fall
/// through to the non-interactive path.
fn interactive_enabled(quiet_flag: bool) -> bool {
    !quiet_flag && std::io::stdout().is_tty() && std::io::stdin().is_tty()
}

/// Expand a CLI `input` into an ordered playlist. A file yields a single-entry
/// vec; a directory yields every `.opus` child (case-insensitive match, not
/// recursive) sorted lexicographically. Empty directories are rejected early
/// so the caller can report a clear error instead of silently exiting.
pub(crate) fn build_playlist(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let entries =
        fs::read_dir(input).with_context(|| format!("reading {}", escape_terminal_path(input)))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("opus"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no .opus files in {}", escape_terminal_path(input));
    }
    Ok(files)
}

/// Derive the on-screen track label from OpusTags, falling back to the file
/// stem when tags are absent or incomplete. Produces `"ARTIST — TITLE"` when
/// both are present (em-dash U+2014, matching the HLD example).
pub(crate) fn resolve_display_name(tags: &OpusTags, path: &Path) -> String {
    match (tags.get("ARTIST"), tags.get("TITLE")) {
        (Some(a), Some(t)) => format!("{a} \u{2014} {t}"),
        (None, Some(t)) => t.to_string(),
        (Some(a), None) => a.to_string(),
        (None, None) => path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".into()),
    }
}

/// Render the single repainting status line. Pure: takes all inputs by value
/// and returns the final string with no trailing newline. Respects `cols` —
/// truncates the display name with `…` when the full render would overflow,
/// and drops the bar+bitrate entirely when `cols < 50`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn format_status_line(
    cols: usize,
    paused: bool,
    loop_mode: LoopMode,
    display_name: &str,
    track_idx: usize,
    playlist_len: usize,
    pos: Duration,
    dur: Duration,
    avg_kbps: f64,
) -> String {
    let display_name = escape_terminal_text(display_name);
    let glyph: char = if paused { '\u{2016}' } else { '\u{25B6}' };
    let track = format_track(track_idx, playlist_len);
    let loop_ind = loop_indicator(loop_mode);
    let pos_str = format_duration(pos, dur);
    let dur_str = format_duration(dur, dur);

    if cols < 50 {
        // Minimal fallback: glyph + track + name + clock, truncating the name
        // with `…` to fit the exact width.
        let prefix = format!("{glyph} {track}  ");
        let suffix = format!("  {pos_str} / {dur_str}");
        let name = truncate_to_fit(&display_name, cols, prefix.width(), suffix.width());
        return format!("{prefix}{name}{suffix}");
    }

    let bar = progress_bar(pos, dur);
    let prefix = format!("{glyph} {track}{loop_ind}  ");
    let suffix = format!("  [{bar}]  {pos_str} / {dur_str}  {avg_kbps:.0} kbps");
    let name = truncate_to_fit(&display_name, cols, prefix.width(), suffix.width());
    format!("{prefix}{name}{suffix}")
}

/// `NN/MM`, zero-padded to the width of the larger side. `len < 10` means
/// single-digit rendering (`1/1`, `3/9`) — padding only kicks in once the
/// playlist itself requires it.
fn format_track(track_idx: usize, playlist_len: usize) -> String {
    let width = playlist_len.to_string().len();
    format!(
        "{:0>width$}/{:0>width$}",
        track_idx + 1,
        playlist_len,
        width = width
    )
}

fn loop_indicator(mode: LoopMode) -> &'static str {
    match mode {
        LoopMode::Off => "",
        LoopMode::All => " \u{27F3}all",
        LoopMode::Single => " \u{27F3}one",
    }
}

/// 10-cell progress bar using U+2593 (filled) and U+2591 (empty). Filled count
/// is clamped to 0..=10 so rodio's end-of-track position overshoot cannot
/// produce an 11-cell bar.
fn progress_bar(pos: Duration, dur: Duration) -> String {
    let filled = if dur.is_zero() {
        0
    } else {
        (pos.as_secs_f64() / dur.as_secs_f64() * 10.0).round() as i64
    };
    let filled = filled.clamp(0, 10) as usize;
    let mut bar = String::with_capacity(10 * 3);
    for _ in 0..filled {
        bar.push('\u{2593}');
    }
    for _ in filled..10 {
        bar.push('\u{2591}');
    }
    bar
}

/// `M:SS` when the reference `dur < 3600s`, else `H:MM:SS`. Both `pos` and
/// `dur` should use the same format; callers thread `dur` as the reference
/// for both to keep them consistent.
fn format_duration(value: Duration, reference: Duration) -> String {
    let total = value.as_secs();
    if reference.as_secs() < 3600 {
        let m = total / 60;
        let s = total % 60;
        format!("{m}:{s:02}")
    } else {
        let h = total / 3600;
        let m = (total % 3600) / 60;
        let s = total % 60;
        format!("{h}:{m:02}:{s:02}")
    }
}

/// Truncate `name` by grapheme clusters so `prefix_cols + name_cols +
/// suffix_cols <= cols`, using Unicode terminal-cell widths. Appends U+2026
/// when anything was removed. The scan stops at the first cluster beyond the
/// visible budget; an untruncated label is the only case that scans its full
/// input.
fn truncate_to_fit(name: &str, cols: usize, prefix_cols: usize, suffix_cols: usize) -> String {
    let fixed_cols = prefix_cols.saturating_add(suffix_cols);
    let budget = cols.saturating_sub(fixed_cols);
    if budget == 0 {
        return String::new();
    }

    let ellipsis = "…";
    let keep_budget = budget.saturating_sub(ellipsis.width());
    let mut used_cols = 0usize;
    let mut keep_end = 0usize;

    for (start, grapheme) in name.grapheme_indices(true) {
        let grapheme_cols = grapheme.width();
        let next_cols = used_cols.saturating_add(grapheme_cols);
        if next_cols > budget {
            let mut out = String::with_capacity(keep_end + ellipsis.len());
            out.push_str(&name[..keep_end]);
            out.push_str(ellipsis);
            return out;
        }

        used_cols = next_cols;
        if used_cols <= keep_budget {
            keep_end = start + grapheme.len();
        }
    }

    name.to_string()
}

/// Pick the next playlist index after a decode error. `Off` stops when we run
/// off the end; `All` and `Single` wrap. `Single` still advances on error —
/// we never retry a broken track in a tight loop.
pub(crate) fn advance_on_error(
    idx: usize,
    playlist_len: usize,
    loop_mode: LoopMode,
) -> Option<usize> {
    if playlist_len == 0 {
        return None;
    }
    match loop_mode {
        LoopMode::Off => {
            if idx + 1 < playlist_len {
                Some(idx + 1)
            } else {
                None
            }
        }
        LoopMode::All | LoopMode::Single => Some((idx + 1) % playlist_len),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::PathBuf;

    /// Per-test scratch directory under `std::env::temp_dir()`. We avoid the
    /// `tempfile` crate (not a declared dep) and manage cleanup on Drop so a
    /// panicking assertion still removes the directory.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(tag: &str) -> Self {
            let nonce = format!(
                "{}_{}_{}",
                tag,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let dir = std::env::temp_dir().join(format!("ropus_play_{nonce}"));
            fs::create_dir_all(&dir).expect("create scratch dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn touch(&self, name: &str) -> PathBuf {
            let p = self.0.join(name);
            File::create(&p).expect("create file");
            p
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("injected writer failure"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("injected flush failure"))
        }
    }

    struct FlushFailingWriter(Vec<u8>);

    impl Write for FlushFailingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("injected flush failure"))
        }
    }

    struct FakeTerminal {
        disable_calls: usize,
        fail_disable: bool,
    }

    impl TerminalMode for FakeTerminal {
        fn enable_raw_mode(&mut self) -> io::Result<()> {
            Ok(())
        }

        fn disable_raw_mode(&mut self) -> io::Result<()> {
            self.disable_calls += 1;
            if self.fail_disable {
                Err(io::Error::other("injected restore failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn write_output_propagates_write_failures() {
        let mut writer = FailingWriter;
        let error = write_output(&mut writer, format_args!("hello"))
            .expect_err("injected output failure must be reported");
        assert!(format!("{error:#}").contains("writing playback output"));
    }

    #[test]
    fn write_output_propagates_flush_failures() {
        let mut writer = FlushFailingWriter(Vec::new());
        let error = write_output(&mut writer, format_args!("hello"))
            .expect_err("injected flush failure must be reported");
        assert!(format!("{error:#}").contains("flushing playback output"));
        assert_eq!(writer.0, b"hello");
    }

    #[test]
    fn raw_mode_restore_reports_failure_and_drop_retries() {
        let mut terminal = FakeTerminal {
            disable_calls: 0,
            fail_disable: true,
        };
        {
            let mut guard = RawModeGuard::enable(&mut terminal).expect("enable raw mode");
            let error = guard
                .restore()
                .expect_err("injected restore failure must be reported");
            assert!(format!("{error:#}").contains("restoring terminal raw mode"));
        }
        assert_eq!(terminal.disable_calls, 2, "Drop remains an unwind fallback");
    }

    #[test]
    fn playback_and_restore_errors_are_both_preserved() {
        let error = combine_playback_and_restore::<()>(
            Err(anyhow!("injected playback failure")),
            Err(anyhow!("restoring terminal raw mode: injected failure")),
        )
        .expect_err("both injected failures must be returned");
        let message = format!("{error:#}");
        assert!(message.contains("injected playback failure"));
        assert!(message.contains("restoring terminal raw mode: injected failure"));
    }

    // -- build_playlist ----------------------------------------------------

    #[test]
    fn build_playlist_single_file_returns_one_entry() {
        let scratch = ScratchDir::new("single");
        let file = scratch.touch("only.opus");
        let list = build_playlist(&file).expect("single file");
        assert_eq!(list, vec![file]);
    }

    #[test]
    fn build_playlist_directory_filters_and_sorts() {
        let scratch = ScratchDir::new("dir");
        let a = scratch.touch("a.opus");
        let b = scratch.touch("b.opus");
        let c = scratch.touch("c.opus");
        scratch.touch("not_opus.mp3");
        fs::create_dir_all(scratch.path().join("subdir")).expect("subdir");

        let list = build_playlist(scratch.path()).expect("dir scan");
        assert_eq!(list, vec![a, b, c], "only .opus files, sorted, no subdir");
    }

    #[test]
    fn build_playlist_matches_extension_case_insensitively() {
        let scratch = ScratchDir::new("case");
        let lower = scratch.touch("lower.opus");
        let upper = scratch.touch("UPPER.OPUS");
        let mixed = scratch.touch("Mixed.Opus");

        let list = build_playlist(scratch.path()).expect("dir scan");
        // Plain lexicographic sort places uppercase before lowercase on most
        // filesystems — assert containment rather than ordering.
        assert_eq!(list.len(), 3);
        for p in &[lower, upper, mixed] {
            assert!(list.contains(p), "missing {} from playlist", p.display());
        }
    }

    #[test]
    fn build_playlist_empty_directory_errors_with_message() {
        let scratch = ScratchDir::new("empty");
        let err = build_playlist(scratch.path()).expect_err("empty dir");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no .opus files"),
            "expected 'no .opus files' in error, got: {msg}"
        );
    }

    #[test]
    fn build_playlist_nonexistent_path_errors_cleanly() {
        let bogus = std::env::temp_dir().join(format!(
            "ropus_play_does_not_exist_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let err = build_playlist(&bogus).expect_err("nonexistent");
        // Just assert it returned an error (no panic) — the exact message is
        // OS-dependent and not worth pinning.
        let _ = format!("{err:#}");
    }

    // -- resolve_display_name ---------------------------------------------

    #[test]
    fn resolve_display_name_uses_artist_and_title_when_both_set() {
        let tags = OpusTags {
            vendor: "v".into(),
            comments: vec!["ARTIST=Liszt".into(), "TITLE=Sonata".into()],
        };
        let path = Path::new("/music/ignored.opus");
        assert_eq!(resolve_display_name(&tags, path), "Liszt \u{2014} Sonata");
    }

    #[test]
    fn resolve_display_name_title_only() {
        let tags = OpusTags {
            vendor: "v".into(),
            comments: vec!["TITLE=Solo".into()],
        };
        assert_eq!(resolve_display_name(&tags, Path::new("/x.opus")), "Solo");
    }

    #[test]
    fn resolve_display_name_artist_only() {
        let tags = OpusTags {
            vendor: "v".into(),
            comments: vec!["ARTIST=Solo".into()],
        };
        assert_eq!(resolve_display_name(&tags, Path::new("/x.opus")), "Solo");
    }

    #[test]
    fn resolve_display_name_falls_back_to_filename_stem() {
        let tags = OpusTags::default();
        assert_eq!(
            resolve_display_name(&tags, Path::new("/music/track01.opus")),
            "track01"
        );
    }

    #[test]
    fn resolve_display_name_fallback_unknown_when_no_stem() {
        let tags = OpusTags::default();
        // An empty path has no file_stem → we emit "unknown" rather than
        // panicking or returning "".
        assert_eq!(resolve_display_name(&tags, Path::new("")), "unknown");
    }

    // -- format_status_line: glyph ----------------------------------------

    #[test]
    fn status_line_glyph_reflects_pause_state() {
        let line_playing = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(line_playing.starts_with('\u{25B6}'), "▶ when playing");

        let line_paused = format_status_line(
            120,
            true,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(line_paused.starts_with('\u{2016}'), "‖ when paused");
    }

    // -- format_status_line: loop indicator -------------------------------

    #[test]
    fn status_line_loop_indicator_off_has_none() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(!line.contains('\u{27F3}'), "no loop glyph when Off");
    }

    #[test]
    fn status_line_loop_indicator_all_and_single() {
        let all = format_status_line(
            120,
            false,
            LoopMode::All,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(all.contains("\u{27F3}all"));

        let single = format_status_line(
            120,
            false,
            LoopMode::Single,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(single.contains("\u{27F3}one"));
    }

    // -- format_status_line: track formatting -----------------------------

    #[test]
    fn status_line_track_pads_to_playlist_width() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            2,
            17,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(line.contains("03/17"), "expected zero-padded 03/17: {line}");
    }

    #[test]
    fn status_line_track_single_digit_is_unpadded() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(line.contains("1/1"), "expected unpadded 1/1: {line}");
        assert!(!line.contains("01/01"));
    }

    // -- format_status_line: progress bar ---------------------------------

    #[test]
    fn status_line_progress_bar_filled_cells() {
        let filled_char = '\u{2593}';
        let empty_char = '\u{2591}';

        let at_zero = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert_eq!(at_zero.matches(filled_char).count(), 0);
        assert_eq!(at_zero.matches(empty_char).count(), 10);

        let at_half = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(30),
            Duration::from_secs(60),
            128.0,
        );
        assert_eq!(at_half.matches(filled_char).count(), 5);
        assert_eq!(at_half.matches(empty_char).count(), 5);

        let at_full = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(60),
            Duration::from_secs(60),
            128.0,
        );
        assert_eq!(at_full.matches(filled_char).count(), 10);
        assert_eq!(at_full.matches(empty_char).count(), 0);

        // Past end (rodio overshoot) — must clamp, not produce 11 cells.
        let overshoot = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(75),
            Duration::from_secs(60),
            128.0,
        );
        assert_eq!(overshoot.matches(filled_char).count(), 10);
        assert_eq!(overshoot.matches(empty_char).count(), 0);
    }

    // -- format_status_line: duration format ------------------------------

    #[test]
    fn status_line_short_duration_uses_m_ss() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(250),
            128.0,
        );
        assert!(line.contains("0:00 / 4:10"), "short format: {line}");
    }

    #[test]
    fn status_line_threshold_duration_uses_h_mm_ss() {
        // At exactly 3600s, spec says H:MM:SS.
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(3600),
            128.0,
        );
        assert!(line.contains("0:00:00 / 1:00:00"), "H:MM:SS format: {line}");
    }

    #[test]
    fn status_line_over_hour_uses_h_mm_ss() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(3661),
            Duration::from_secs(3700),
            128.0,
        );
        assert!(line.contains("1:01:01 / 1:01:40"), "H:MM:SS format: {line}");
    }

    #[test]
    fn status_line_short_seconds_renders_zero_minute() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(59),
            128.0,
        );
        assert!(line.contains("0:00 / 0:59"), "M:SS with 0 minute: {line}");
    }

    // -- format_status_line: width handling -------------------------------

    #[test]
    fn status_line_truncates_long_title_to_fit_cols() {
        // Build a name that clearly blows past the budget.
        let long_name: String = "x".repeat(200);
        let line = format_status_line(
            60,
            false,
            LoopMode::Off,
            &long_name,
            0,
            1,
            Duration::from_secs(30),
            Duration::from_secs(240),
            192.0,
        );
        assert!(
            line.width() <= 60,
            "rendered line must fit cols=60, got {} cells: {line}",
            line.width()
        );
        assert!(
            line.contains('\u{2026}'),
            "expected … in truncated title: {line}"
        );
    }

    #[test]
    fn status_line_narrow_mode_drops_bar_and_bitrate() {
        let line = format_status_line(
            40,
            false,
            LoopMode::Off,
            "Name",
            0,
            1,
            Duration::from_secs(10),
            Duration::from_secs(60),
            192.0,
        );
        assert!(!line.contains('\u{2593}'), "no filled bar cell: {line}");
        assert!(!line.contains('\u{2591}'), "no empty bar cell: {line}");
        assert!(!line.contains("kbps"), "no bitrate: {line}");
    }

    #[test]
    fn status_line_wide_labels_fit_terminal_cells() {
        let wide_name = "東京".repeat(100);
        let line = format_status_line(
            60,
            false,
            LoopMode::Off,
            &wide_name,
            0,
            1,
            Duration::from_secs(30),
            Duration::from_secs(240),
            192.0,
        );
        assert!(line.width() <= 60, "wide label overflowed: {line}");
        assert!(line.ends_with(" kbps"));
        assert!(line.contains('\u{2026}'), "expected title ellipsis: {line}");
    }

    #[test]
    fn status_line_resize_recomputes_unicode_budget() {
        let name = "東京 👩\u{200D}💻 e\u{301} ".repeat(30);
        let wide = format_status_line(
            120,
            false,
            LoopMode::Off,
            &name,
            0,
            1,
            Duration::from_secs(30),
            Duration::from_secs(240),
            192.0,
        );
        let narrow = format_status_line(
            40,
            false,
            LoopMode::Off,
            &name,
            0,
            1,
            Duration::from_secs(30),
            Duration::from_secs(240),
            192.0,
        );
        assert!(wide.width() <= 120, "wide render overflowed: {wide}");
        assert!(narrow.width() <= 40, "narrow render overflowed: {narrow}");
        assert_ne!(wide, narrow, "resize must change the rendered budget");
    }

    #[test]
    fn truncate_to_fit_preserves_combining_and_emoji_graphemes() {
        let combining = "e\u{301}".repeat(20);
        let combining_fit = truncate_to_fit(&combining, 5, 0, 0);
        assert_eq!(combining_fit.width(), 5);
        assert_eq!(combining_fit, "e\u{301}e\u{301}e\u{301}e\u{301}…");
        assert!(!combining_fit.ends_with('\u{301}'));

        let emoji = "👩\u{200D}💻".repeat(20);
        let emoji_fit = truncate_to_fit(&emoji, 5, 0, 0);
        assert_eq!(emoji_fit.width(), 5);
        assert_eq!(emoji_fit, "👩\u{200D}💻👩\u{200D}💻…");
        assert!(!emoji_fit.contains("👩\u{200D}…"));
    }

    #[test]
    fn status_line_escapes_untrusted_track_labels() {
        let line = format_status_line(
            120,
            false,
            LoopMode::Off,
            "album\n\x1B]0;forged\x07\u{0085}",
            0,
            1,
            Duration::from_secs(0),
            Duration::from_secs(60),
            128.0,
        );
        assert!(!line.contains('\n'));
        assert!(!line.contains('\x1B'));
        assert!(line.contains(r"album\u{000A}\u{001B}]0;forged\u{0007}\u{0085}"));
    }

    // -- advance_on_error --------------------------------------------------

    #[test]
    fn advance_on_error_off_advances_mid_list() {
        assert_eq!(advance_on_error(0, 3, LoopMode::Off), Some(1));
    }

    #[test]
    fn advance_on_error_off_stops_at_end() {
        assert_eq!(advance_on_error(2, 3, LoopMode::Off), None);
    }

    #[test]
    fn advance_on_error_all_wraps_at_end() {
        assert_eq!(advance_on_error(2, 3, LoopMode::All), Some(0));
    }

    #[test]
    fn advance_on_error_single_still_advances() {
        // Spec: Single advances on error, does not retry.
        assert_eq!(advance_on_error(0, 3, LoopMode::Single), Some(1));
    }

    #[test]
    fn advance_on_error_single_item_playlist_off_terminates() {
        assert_eq!(advance_on_error(0, 1, LoopMode::Off), None);
    }

    // -- compute_avg_kbps --------------------------------------------------

    #[test]
    fn compute_avg_kbps_standard_case() {
        // 128 kbps over 60s is 128_000 bits/s * 60 / 8 = 960_000 bytes.
        let kbps = compute_avg_kbps(960_000, 60.0);
        assert!(
            (kbps - 128.0).abs() < 0.01,
            "expected ~128 kbps, got {kbps}"
        );
    }

    #[test]
    fn compute_avg_kbps_zero_duration_returns_zero() {
        // Avoid NaN/Inf that would poison the `{:.0}` format in the status line.
        assert_eq!(compute_avg_kbps(12_345, 0.0), 0.0);
    }

    // -- validate_and_compute_gain / apply_gain ---------------------------

    #[test]
    fn apply_gain_multiplier_matches_formula() {
        // +6 dB is a factor of 10^(6/20) ≈ 1.99526. Applied to 0.5 gives
        // ≈ 0.99763. Assert both the intermediate multiplier and the
        // post-multiply sample land within 0.01 of the textbook answer.
        let mult = validate_and_compute_gain(6.0).expect("+6 dB is valid");
        assert!((mult - 1.995).abs() < 0.01, "+6 dB ≈ 1.995, got {mult}");

        let mut samples = [0.5f32];
        apply_gain(&mut samples, mult);
        assert!(
            (samples[0] - 0.998).abs() < 0.01,
            "expected ~0.998, got {}",
            samples[0]
        );
    }

    #[test]
    fn apply_gain_zero_db_is_exact_identity() {
        // 0 dB is a no-op: validator returns 1.0 and apply_gain short-circuits
        // to avoid iterating the buffer. The expected behaviour is exact
        // bit-equality, not "within epsilon".
        let mult = validate_and_compute_gain(0.0).expect("0 dB is valid");
        assert_eq!(mult, 1.0, "0 dB must round-trip to multiplier == 1.0");

        let orig = [0.5f32, -0.25, 0.0, 1.0, -1.0];
        let mut samples = orig;
        apply_gain(&mut samples, mult);
        assert_eq!(samples, orig, "0 dB must leave samples bit-identical");
    }

    #[test]
    fn gain_db_nan_rejected() {
        let err = validate_and_compute_gain(f32::NAN).expect_err("NaN must surface as an error");
        let msg = format!("{err:#}").to_ascii_lowercase();
        assert!(msg.contains("finite"), "error should mention finite: {msg}");
    }

    #[test]
    fn gain_db_infinity_rejected() {
        assert!(
            validate_and_compute_gain(f32::INFINITY).is_err(),
            "+∞ must surface as an error"
        );
        assert!(
            validate_and_compute_gain(f32::NEG_INFINITY).is_err(),
            "-∞ must surface as an error"
        );
    }

    #[test]
    fn gain_db_out_of_range_rejected() {
        // +128 dB converts to Q8 32768, one above libopus's maximum. Keep the
        // upper boundary representable rather than allowing a later setter
        // error after input decoding has already started.
        assert!(validate_and_compute_gain(127.9).is_ok());
        assert!(validate_and_compute_gain(128.0).is_err());
        assert!(validate_and_compute_gain(-128.0).is_ok());
        let err_hi = validate_and_compute_gain(200.0).expect_err("200 dB above clamp must error");
        let err_lo = validate_and_compute_gain(-200.0).expect_err("-200 dB below clamp must error");
        let msg_hi = format!("{err_hi:#}").to_ascii_lowercase();
        let msg_lo = format!("{err_lo:#}").to_ascii_lowercase();
        assert!(
            msg_hi.contains("range") || msg_hi.contains("out of"),
            "high-side error should mention range: {msg_hi}"
        );
        assert!(
            msg_lo.contains("range") || msg_lo.contains("out of"),
            "low-side error should mention range: {msg_lo}"
        );
    }

    #[test]
    fn play_rejects_nonfinite_volume_before_device_or_input() {
        let err = play(PlayOptions {
            input: PathBuf::from("does-not-exist.opus"),
            volume: Some(f32::NAN),
            loop_mode: LoopMode::Off,
            quiet: true,
            device: None,
            list_devices: true,
            gain_db: 0.0,
        })
        .expect_err("NaN volume must fail before --list-devices");
        assert!(
            format!("{err:#}").contains("volume"),
            "error should mention volume: {err:#}"
        );
    }

    // -- format_device_list -----------------------------------------------

    #[test]
    fn format_device_list_empty_list_errors() {
        let err = format_device_list(&[]).expect_err("empty slice must error");
        let msg = format!("{err:#}").to_ascii_lowercase();
        assert!(
            msg.contains("no output devices"),
            "error should mention 'no output devices': {msg}"
        );
    }

    #[test]
    fn format_device_list_prints_one_per_line() {
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let out = format_device_list(&names).expect("non-empty slice formats cleanly");
        assert_eq!(out, "A\nB\nC\n");
    }

    #[test]
    fn format_device_list_uses_reversible_control_escaping() {
        let names = vec![
            "Speakers\r\nforged".to_string(),
            "\x1B]0;title\x07".to_string(),
            "icc\u{0085}profile\u{009B}".to_string(),
        ];
        let out = format_device_list(&names).expect("device names format cleanly");
        assert_eq!(
            out,
            "Speakers\\u{000D}\\u{000A}forged\n\\u{001B}]0;title\\u{0007}\nicc\\u{0085}profile\\u{009B}\n"
        );
    }

    #[test]
    fn gain_and_volume_compose_multiplicatively() {
        // Decoder gain happens before sink.set_volume(v), so the effective
        // scaling is still `v * mult`. Here we pick +6 dB (≈ ×1.995) and a
        // sink-volume equivalent of 0.5, and assert the composed scale matches
        // the input to within a small tolerance.
        let mult = validate_and_compute_gain(6.0).expect("+6 dB is valid");
        let sink_volume = 0.5f32;

        let orig = [0.5f32, -0.25, 0.0, 1.0, -1.0];
        let mut samples = orig;
        apply_gain(&mut samples, mult);
        // Model the sink volume as the second multiplicative stage.
        for s in samples.iter_mut() {
            *s *= sink_volume;
        }

        // Expected composed scale is mult * sink_volume ≈ 0.9977. Compare each
        // sample against `orig * composed_scale` within a tight tolerance.
        let composed = mult * sink_volume;
        for (i, (got, want)) in samples
            .iter()
            .zip(orig.iter().map(|s| s * composed))
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-5,
                "sample {i}: got {got}, expected {want} (composed = {composed})"
            );
        }
    }
}
