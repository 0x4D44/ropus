//! Play: any symphonia-supported input → default audio output.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use colored::*;

use crate::audio::decode::{DecodedAudio, decode_to_f32};
use crate::container::ogg::OpusTags;
use crate::options::{LoopMode, PlayOptions};
use crate::ui::{format_num, heading, ok};

pub fn play(opts: PlayOptions) -> Result<()> {
    heading("play");
    println!("file     {}", opts.input.display().to_string().cyan());

    // Decode to interleaved f32 through one unified pipeline: symphonia demuxes
    // every container, and `decode_to_f32` routes Opus tracks through ropus
    // while everything else uses symphonia's native decoders.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&opts.input)?;

    let channels_u16 = u16::try_from(channels).map_err(|_| anyhow!("channel count overflow"))?;
    println!(
        "audio    {} samples, {} Hz, {} ch",
        format_num(samples.len() as u64).bright_white(),
        sample_rate.to_string().bright_white(),
        channels_u16.to_string().bright_white(),
    );

    // Try to open the default audio device. If that fails (no device, headless
    // environment, etc.) print a clear message instead of panicking.
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(pair) => pair,
        Err(e) => {
            return Err(anyhow!("no default audio output device available: {e}"));
        }
    };
    let sink = rodio::Sink::try_new(&handle).map_err(|e| anyhow!("creating sink failed: {e}"))?;

    if let Some(v) = opts.volume {
        sink.set_volume(v.clamp(0.0, 1.0));
    }

    let source = rodio::buffer::SamplesBuffer::new(channels_u16, sample_rate, samples);
    sink.append(source);
    println!("playing  (Ctrl-C to stop)");
    sink.sleep_until_end();
    ok("playback finished");
    Ok(())
}

/// Expand a CLI `input` into an ordered playlist. A file yields a single-entry
/// vec; a directory yields every `.opus` child (case-insensitive match, not
/// recursive) sorted lexicographically. Empty directories are rejected early
/// so the caller can report a clear error instead of silently exiting.
// `allow(dead_code)`: stage-3 wires these helpers into the main FSM. Tests
// exercise them today.
#[allow(dead_code)]
pub(crate) fn build_playlist(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let entries =
        fs::read_dir(input).with_context(|| format!("reading {}", input.display()))?;
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
        bail!("no .opus files in {}", input.display());
    }
    Ok(files)
}

/// Derive the on-screen track label from OpusTags, falling back to the file
/// stem when tags are absent or incomplete. Produces `"ARTIST — TITLE"` when
/// both are present (em-dash U+2014, matching the HLD example).
#[allow(dead_code)]
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
#[allow(clippy::too_many_arguments, dead_code)]
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
    // Both glyphs are cell-width 1 so chars().count() matches display cells.
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
        let name = truncate_to_fit(display_name, cols, prefix.chars().count(), suffix.chars().count());
        return format!("{prefix}{name}{suffix}");
    }

    let bar = progress_bar(pos, dur);
    let prefix = format!("{glyph} {track}{loop_ind}  ");
    let suffix = format!("  [{bar}]  {pos_str} / {dur_str}  {avg_kbps:.0} kbps");
    let name = truncate_to_fit(display_name, cols, prefix.chars().count(), suffix.chars().count());
    format!("{prefix}{name}{suffix}")
}

/// `NN/MM`, zero-padded to the width of the larger side. `len < 10` means
/// single-digit rendering (`1/1`, `3/9`) — padding only kicks in once the
/// playlist itself requires it.
#[allow(dead_code)]
fn format_track(track_idx: usize, playlist_len: usize) -> String {
    let width = playlist_len.to_string().len();
    format!(
        "{:0>width$}/{:0>width$}",
        track_idx + 1,
        playlist_len,
        width = width
    )
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Truncate `name` (by chars) so `prefix_cols + name_cols + suffix_cols <=
/// cols`. Appends U+2026 when anything was removed. Returns the original when
/// it already fits, or an empty string when prefix+suffix alone already
/// overflow — this keeps the render degradation monotonic instead of panicking.
#[allow(dead_code)]
fn truncate_to_fit(name: &str, cols: usize, prefix_cols: usize, suffix_cols: usize) -> String {
    let budget = cols.saturating_sub(prefix_cols + suffix_cols);
    let name_cols = name.chars().count();
    if name_cols <= budget {
        return name.to_string();
    }
    if budget == 0 {
        return String::new();
    }
    // Reserve one column for the `…`.
    let keep = budget.saturating_sub(1);
    let mut out: String = name.chars().take(keep).collect();
    out.push('\u{2026}');
    out
}

/// Pick the next playlist index after a decode error. `Off` stops when we run
/// off the end; `All` and `Single` wrap. `Single` still advances on error —
/// we never retry a broken track in a tight loop.
#[allow(dead_code)]
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
        assert_eq!(
            resolve_display_name(&tags, Path::new("/x.opus")),
            "Solo"
        );
    }

    #[test]
    fn resolve_display_name_artist_only() {
        let tags = OpusTags {
            vendor: "v".into(),
            comments: vec!["ARTIST=Solo".into()],
        };
        assert_eq!(
            resolve_display_name(&tags, Path::new("/x.opus")),
            "Solo"
        );
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
            line.chars().count() <= 60,
            "rendered line must fit cols=60, got {} chars: {line}",
            line.chars().count()
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
}
