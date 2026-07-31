# ROP-BUG-FLUX-00047 — Reverse Ogg duration scan trusts header-shaped payload bytes

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** ropus-tools-core/ogg-scan
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

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:324-379 accepts a reverse-scan candidate after checking only OggS, version zero, and serial. It does not validate lacing, full page extent, checksum, sequence, flags, or EOS. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165 trusts the resulting granule for duration and bitrate. Crafted trailing or payload bytes can therefore supply an arbitrary final position. Fix: use the validated page-metadata parser tracked by ROP-REQ-FLUX-00041, adopt an explicit CRC/EOS policy, reject final granule below pre_skip, and add fake-header cases. Static review only; no file or test ran.

## Fix

<unfixed — raised only>

## Notes

### Additional `ropusinfo` final-page trigger (2026-07-31)

`/Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:346-373`
also accepts a matching non-EOS candidate and does not prove the candidate's
declared page extent or checksum before
`/Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165`
publishes its granule as the stream duration. Add fake `OggS`, truncated-page,
bad-CRC, and non-EOS trailing candidates to the shared validated-page oracle.
Static review at `origin/main` `6a312e1`; no command or test ran.
