# Section-by-section refinement ledger

Engineering document. Not part of the reader-facing PDF. Its purpose is to
make the work resumable: a new session reads **Current position** and the
status ledger, then continues with exactly one section.

## Protocol (repeat for ONE section at a time)

1. `python3 tools/check_coverage.py SLOT` — compare the slot's Part I
   (`tutorial/`) and Part III (`documentation/`) source sections with the
   generated fragments, and read the upstream Markdown itself. Record any
   lost information **before** changing anything.
2. Read the matching Triton / TileLang material for this topic and consult the
   style baseline below.
3. Design the rewrite: restore missing substance first, then fix the
   storyline, transitions and tone. Content is never cut for brevity.
4. Rebuild, review and verify:
   `python3 tools/build_guide.py`
   `python3 tools/review.py --round N --compare`
   `make all`
   `pdftotext -layout main.pdf -` — read that section in the PDF.
   Only that section's units may change in the generated output.
5. `git add -A && git commit -m "Refine section SLOT: ..."`

No other section, the Preface, the appendices, or shared tooling behaviour
may change during a section's iteration. Each section gets its own commit.

## Style baseline

Sources: Triton Programming Guide ch.1 Introduction (Motivations /
Challenges), Triton MAPL 2019 Introduction, Triton docs home; TileLang
Programming Guides Overview and Language Basics.

Storyline: observable phenomenon → causal chain with plain connectives
("Yet", "As a consequence", "Regrettably", "Of course") → name the tension →
ONE premise → one-sentence definition → a concrete comparison or the smallest
program → "A key benefit ..." → honest account of the remaining difficulty →
"Where to go next".

Construct sections (TileLang Language Basics): one construct per block; the
first sentence says what it is FOR; code follows immediately; constraints and
failure modes appear as short notes in place, not in a separate dump;
cross-references point forward; continuous prose, no label lists.

Avoid (the "AI flavour"): opening with a triad ("three families", "three
promises"), meta narration ("In this section, we will explore"), symmetrical
restatements, abrupt topic changes, dangling clauses after a removed passage.

Hard constraints: target-platform / tuning vocabulary stays archived
(`hardware` check at 0); no editorial provenance, no `we/our`, no tutorial
chapter numbers; every unit stays in its registered slot.

## Current position

- Next section: **1.9**
- Review rounds: from 16 onward (1.2→16 … 1.8→22)
- Baseline commit: `98bc7e1` (plus the pending 1.1 substance pass, committed
  with the infrastructure)

## Status ledger

`x` done, `~` in progress, `.` pending. Section titles are in
`SLOT_TITLES` in `tools/build_guide.py`.

| Slot | State | Commit | Note |
|---|---|---|---|
| 1.1 | x | baseline | Triton-style arc + Part I/III substance (rounds 14-15) |
| 1.2 | x | round 16 | Part I table + Part III duplicate list merged; version rationale restored; table set in place to avoid a stranded lead-in |
| 1.3 | x | round 17 | No source information lost; removed the heading repeated from the slot title, gave the three listings distinct captions, replaced the meta-narration intro with the command/verification chain, and handed off to 1.4 |
| 1.4 | x | round 18 | Both source listings survived; dropped the duplicated subsection heading, distinguished the two captions, said what the four-element copy verifies before showing the code, and handed off to the command-line section |
| 1.5 | x | round 19 | Restored the source-to-source/transpilation model that had been archived with the vendor names; repaired the "However"/"Therefore, Once"/"Species"/"a example" breaks; gave the three overlapping units distinct openings and six distinct listing captions; all six source code blocks and the flag table survived |
| 1.6 | x | round 20 | All 7 source command blocks kept; rewrote the Part III sentence fragments into continuous prose, fixed "corrupted version", gave all 7 listings distinct captions and separated the two units by role. The 4 weak-coverage hits are paraphrases (verified phrase by phrase, no information lost) |
| 1.7 | x | round 21 | Both source listings survived; removed the tutorial slogan heading, replaced the casual opener with an explicit account of what the first program leaves out and what skeleton it shows, distinguished the two captions and replaced the vague hand-off with the construct-by-construct storyline |
| 1.8 | x | round 22 | All 5 listings and the figure survived (the second source image is its dark/light variant); replaced both placeholder openings with sentences that name what the listing demonstrates and why the host code is shown twice, and rewrote the slot intro around the three-part structure |
| 1.9 | . | | |
| 1.10 | . | | |
| 2.1 | . | | |
| 2.2 | . | | |
| 2.3 | . | | |
| 2.4 | . | | |
| 2.5 | . | | |
| 2.6 | . | | |
| 2.7 | . | | |
| 2.8 | . | | |
| 2.9 | . | | |
| 2.10 | . | | |
| 2.11 | . | | |
| 2.12 | . | | |
| 2.13 | . | | |
| 2.14 | . | | |
| 2.15 | . | | |
| 3.1 | . | | |
| 3.2 | . | | |
| 3.3 | . | | |
| 3.4 | . | | |
| 3.4.1 | . | | |
| 3.4.2 | . | | |
| 3.5 | . | | |
| 3.5.1 | . | | |
| 3.5.2 | . | | |
| 3.5.3 | . | | |
| 3.5.4 | . | | |
| 3.5.10 | . | | |
| 3.6 | . | | |
| 3.6.1 | . | | |
| 3.6.2 | . | | |
| 3.6.3 | . | | |
| 3.7 | . | | |
| 3.7.1 | . | | |
| 3.7.2 | . | | |
| 3.7.3 | . | | |
| 3.7.9 | . | | |
| 3.7.10 | . | | |
| 3.8 | . | | |
| 3.8.1 | . | | |
| 3.8.2 | . | | |
| 3.8.3 | . | | |
| 3.8.4 | . | | |
| 3.8.5 | . | | |
| 3.8.10 | . | | |
| 3.9 | . | | |
| 3.9.1 | . | | |
| 3.9.2 | . | | |
| 3.9.10 | . | | |
| 3.10 | . | | |
| 3.10.1 | . | | |
| 3.10.2 | . | | |
| 3.11 | . | | |
