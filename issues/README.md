# Customer issues — `issues/` directory ledger

One markdown file per **customer-reported issue** under `issues/`, the filename
== the issue ID, the state a field inside. Mirrors the `bugs/`/`reqs/` ledgers
(filename-as-ID, bold-markdown fields, append-only history) so the same tooling
and muscle memory apply; deltic assembles it by globbing `issues/*.md` (`deltic
issues`). There is **no master table** by design — concurrent authoring never
collides, and a transition is a single-line edit, never a rename.

An **issue** is a customer's *report* — a **Bug**, an **Enhancement** request, or
a **Question** (its `Kind`). The underlying engineering work lives in two sibling
ledgers, linked via `Work:`: `ROP-BUG-…` defects in `bugs/` and
`ROP-REQ-…` accepted enhancements in `reqs/`. An issue closes when the
customer's *report* is resolved; a work item closes when its own fix/build is
verified. The generic lifecycle, privacy architecture, resolution sweep, and
release manifest are the fleet **`issue-tracking`** skill; a repo's intake /
privacy specifics live in its **`ISSUES-GUIDE.md`** (if present).

## ID grammar & allocation
`ROP-ISS-HOST-NNNNN` — the bug scheme with `TYPE = ISS`. The `HOST`
shortcode registry, the per-host `max(NNNNN)+1` allocation (unified across
`bugs/`+`reqs/`+`issues/`), and the "commit one new file" rule are defined
canonically in `bugs/README.md`; follow that, substituting `ISS` for `BUG`. Mint
one with **`deltic issues new --title "<summary>"`** — it derives the host token,
allocates the next unified number, and writes a New record from this binary's own
template (the `## Report` is a de-identify-reminder placeholder — see the privacy
boundary above; never paste raw customer prose).

## States
`New → Triaged → Resolved → Closed`, plus off-ramps `NotABug / WontFix /
Duplicate / Answered` (legal at `New` **or** later). State is the `- **State:**`
field, dated + attributed in the append-only `State history` line — never a
filename rename.
- **New** — received, untriaged.
- **Triaged** — understood; `Kind` / `Customer-impact` / `Area` set; linked to
  work item(s) or off-ramped.
- **Resolved** — the report is **fully** addressed in a shipped version (last
  `Resolution log` entry `Fully fixed` for a bug, or `Added` for an enhancement).
- **Closed** — confirmed (impact-tiered: Low/Medium self-close; Critical/High
  two-eyes).
- **Off-ramps** — `NotABug` / `WontFix` / `Duplicate` / `Answered` are terminal;
  they are **not** live work.

Triage branches on **`Kind`**: **Bug** → link a `ROP-BUG` defect;
**Enhancement (accepted)** → link a `ROP-REQ`; **Enhancement (declined)**
→ `WontFix`; **Question** → `Answered`.

## Fields
| Field | Values / format | Notes |
|---|---|---|
| `State` | New / Triaged / Resolved / Closed / NotABug / WontFix / Duplicate / Answered | Must match the latest `State history` entry. |
| `Kind` | Bug / Enhancement / Question | The **triaged** classification — provisional at `New`, confirmed at Triage. Drives the linked work-item type. |
| `Customer-impact` | Critical / High / Medium / Low | Customer-facing impact (bug) or value (enhancement). Distinct from a bug's engineering `Severity`. Gates closure rigour. |
| `Area` | hierarchical tag | Reuse a quality-map Area where one fits (quality-prioritisation signal). |
| `Reported-version` | `X.Y.Z` or `?` | The app version the customer ran. |
| `Received` | `YYYY-MM-DD` | When the report arrived. |
| `Source-ref` | submission token / `unknown` | The **only** identity value in the repo — a per-submission token (never a cross-report correlator). If lost at intake, record `unknown`; never synthesize a token. |
| `Work` | work ids / `—` / `none …` | Linked underlying work: defect ids in `bugs/`, enhancement ids in `reqs/`. `—` until triaged. |
| `State history` | append-only, dated, attributed | Same idiom as bugs. |
| `Resolution log` | append-only per-version entries, or `— none yet` | Drives the release manifest. |

## Privacy (hard rule)
It must be impossible to identify a customer by reading this repo. Record exactly
one identity value per issue — `Source-ref`, a per-submission token — and **never**
a stable per-customer id or a customer name, in any field or in prose. De-identify
the customer's free-text description **before** it enters `## Report`; never paste
it verbatim. The diagnostic bundle never enters the repo. Correlation and
de-anonymisation live in a secured out-of-repo store keyed on `Source-ref`. Full
detail: the fleet `issue-tracking` skill + this repo's `ISSUES-GUIDE.md`.

## File format
Each `issues/<ID>.md` — labelled field lines, free prose below, no YAML front-matter:

```
# ROP-ISS-HOST-00001 — <short title: the customer's problem, in our words>

- **State:** New
- **Kind:** <Bug | Enhancement | Question>
- **Customer-impact:** <Critical | High | Medium | Low>
- **Area:** <area tag>
- **Reported-version:** <X.Y.Z | ?>
- **Received:** <YYYY-MM-DD>
- **Source-ref:** <submission token | unknown>
- **Work:** —
- **State history:** New (<YYYY-MM-DD>, received by <who>)
- **Resolution log:** — none yet

## Report
<a DE-IDENTIFIED symptom statement — paraphrase the customer's words; strip
org / deal / person / product names, quoted content, and figures. Never paste the
raw description. Reference the report by Source-ref + the reported app version.>

## Notes
<triage: root-cause understanding, mapping to work item(s), duplicate-of pointer
or not-a-bug rationale.>
```
