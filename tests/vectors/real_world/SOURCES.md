# Real-world corpus sources

This file records the provenance and license of every fixture either:

- committed under `tests/vectors/real_world/` (overriding the .gitignore default), or
- referenced as a remote source URL by `corpus_manifest.toml` or `tools/fetch_corpus.sh`.

## Policy

Any new source requires Arthur's review before commit. PRs that add a manifest
entry whose path is committed media, or whose recipe references a third-party
URL, must update this file in the same PR.

## Sources

| id | path-or-url | origin | license / rights | reviewer | review date |
|---|---|---|---|---|---|

(empty - no third-party committed media or remote sources are admitted by this
slice. Generated entries derived from existing checked-in WAV fixtures do not
require an entry here unless the resulting `.opus` is committed.)
