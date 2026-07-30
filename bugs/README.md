# Bugs — `bugs/` directory ledger

**Purpose.** One markdown file per bug under `bugs/`, the filename == the bug
ID; deltic assembles the ledger in memory by globbing `bugs/*.md`. There is
**no master table** by design, and **there is no `BUGS.md` — never create one.**

**ID grammar.** `PREFIX-TYPE-HOST-NNNNN`.
- **PREFIX** — this repo's acronym: `ROP`.
- **TYPE** — `BUG` in this ledger; the sibling `reqs/` ledger mints `REQ`
  (see `reqs/README.md`). Both ledgers share one ID parser, so deltic never
  inspects the token.
- **HOST** — this machine's canonical token: the **first DNS label** of its
  hostname (`<name>.local`, `<name>.home.arpa`, and a bare `<name>` all collapse
  to `<NAME>`), uppercased and reduced to `[A-Z0-9]`. It is derived **in code**
  by `deltic bugs new` — not by a hand-run `hostname`, which would bake the DNS
  domain into the token and split one machine into two. A short 3-letter typing
  code per host lives in deltic's host registry.
- **NNNNN** — a per-host sequence, zero-padded to at least 5 digits, **unified
  across the sibling ledgers**: the next number is `max(NNNNN)+1` over `bugs/`
  **and** `reqs/` **and** `issues/` for this host, so a bare host+number
  (`<HOST>-00042`) maps to exactly one item whatever its type.
- Legacy `PREFIX-NNNNN` ids (pre-conversion) remain valid, verbatim, forever.

**Per-host allocation.** Raise a bug with
**`deltic bugs new --title "<summary>"`** (`--severity`/`--priority`/`--area`/
`--body` optional; `--json` for tooling). It derives this machine's HOST token,
mints the next number across **all** sibling ledgers for that host, and writes
the record from this binary's own template. No central allocator, no lock; bugs
and reqs never share a number.

**Always mint with the command — never by hand.** A hand-run `max(NNNNN)+1` over
one ledger's glob collides with the sibling ledgers by construction, and a
hand-run `hostname` bakes the DNS domain into the token, splitting one machine
into two (MDK-BUG-KILN-00152). The command derives HOST in code and reads the
whole number space.

**How concurrent minting is made safe (R-SAMEHOST).** The unit of concurrency on
one box is the **worktree**, not the machine — a host runs many task worktrees at
once, each a fork of trunk with its own `bugs/` directory. The number space is per
HOST and every worktree for a host lives on that host's filesystem, so the
allocator reads a **host-wide** high-water: this working tree, every sibling
worktree (`git worktree list`), and every id ever added on any ref. That covers a
peer agent's not-yet-committed id and an id sitting on an unintegrated branch. An
error reading any view aborts the mint rather than silently narrowing it — under-
counting mints a duplicate, while over-counting only skips a number
(MDK-BUG-KILN-00150).

Two backstops sit behind it, and it is worth knowing what each does *not* cover:
- A same-directory race is caught by the write itself: publication is atomic and
  no-clobbering, so the loser gets `AlreadyExists` and retries at the next number
  rather than overwriting the winner. This is a no-replace rename, **not**
  `O_EXCL`, and it compares filenames within **one directory** — it cannot see a
  peer worktree, and it cannot see a bug and a req holding the same number.
- Same number in the same ledger is the same path, so git raises an **add/add
  conflict** at integration — loud, and never a silent overwrite. The resolver
  renumbers.
- Same number in **different** sibling ledgers is *not* the same path, so git
  merges it silently and nothing flags it. Only the host-wide high-water prevents
  that class (MDK-BUG-KILN-00151); a structural oracle in
  `crates/deltic-repos/src/mint.rs` fails the gate on a new occurrence.

**States & transitions.** `Open → Blocked → Fixed → Closed`, each dated and
attributed in the append-only `State history:` line. State is a **field inside
the file** (`- **State:** …`) — never rename a file to change its state.

**Two-eyes rule.** A bug moves to `Closed` only after a second pair of eyes
verifies the fix (regression test green, root cause understood).

**Priority vs severity.** `Priority` is fix urgency (`Must` / `Should` /
`Could`). `Severity` is user impact (`Critical` / `High` / `Medium` / `Low`).
Automation picks by priority first, then severity; malformed or missing
priority is automation-ineligible until corrected.

**Ownership.** The current owner lives in the bug file. `Owner role: human`
parks automation indefinitely for a named human owner. `Owner role: fix` and
`Owner role: verify` are leased automation claims; every automation owner must
carry the matching `Owner run`, `Owner host`, `Owner branch`, `Owner base`,
timestamps, and, for verify owners, `Owner fingerprint`. The unowned marker is
the ASCII hyphen `-` in every owner field. Partial, blank, or inconsistent owner
data is treated as malformed and skipped by automation.

**Attempts and parking.** `Attempts: fix=N, doubt=N, indeterminate=N` records
durable unattended retry history. `Held branch` preserves useful fixer work that
needs human follow-up. `Verify retry after` temporarily parks indeterminate
verification without blocking future fix attempts; after three indeterminate
passes, automation leaves the bug parked as `Fixed` for human follow-up.
`Legacy fixed run` is set on pre-schema `Fixed` bugs so the verify loop has
explicit provenance.

**File format.** Each `bugs/<ID>.md`:

```markdown
# ROP-BUG-HOST-00001 — Short title

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ui
- **Raised:** YYYY-MM-DD
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
- **State history:** Open (YYYY-MM-DD, raised by …) → Fixed (YYYY-MM-DD, `sha`)

## Observation
<symptom, repro, expected vs actual>

## Fix
<accepted fix summary and verification notes>

## Notes
<other notes, links, failed attempts>
```

> This ledger was bootstrapped by `deltic bugs init`. Edit it freely — it is your
> repo's own copy, not a fleet-managed cache.

Deltic appends successful `### Fix summary (...)` and `### Verification summary (...)`
sections under `## Fix`. Failed autonomous attempts still append `### Fix attempt
summary (...)` under `## Notes`. Older ledgers may have `### Fix summary (...)`
under `## Notes`; the Repos browser treats that legacy content as Fix prose. The
modeled header fields and `State history` remain the machine-readable ledger truth.

A transition may carry a `model=<id>@<effort>` token in its detail, recording who did
that step: an autonomous fix/verify records the agent family (with the effort deltic
pins for Codex — `model=codex@xhigh`; Claude records `model=claude`), and a raise via
`deltic bugs new --model <id> --effort <tier>` records the precise raising model.
