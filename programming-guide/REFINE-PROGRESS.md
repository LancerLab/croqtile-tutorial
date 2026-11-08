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

- Next section: **3.5** (chapters 1–2 complete; chapter 3 in progress through 3.4.2)
- Review rounds: from 16 onward (1.2→16 … 1.10→24, 2.1→25, 2.2→26, 2.3→27, 2.4→28, 2.5→29, 2.6→30, 2.7→31, 2.8→32, 2.9→33, 2.10→34, 2.11→35, 2.12→36, 2.13→37, 2.14→38, 2.15→39, 3.1→40, 3.2→41, 3.3→42, 3.4→43, 3.4.1→44, 3.4.2→45)
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
| 1.9 | x | round 23 | All 6 listings, 2 tables and the workflow figure survived; rewrote the intro as a cheap-to-expensive debugging chain, bridged the isolated imported "Debugging tip" into that chain, and gave the workflow page's two listings distinct captions |
| 1.10 | x | round 24 | The only source content was a bare five-item checklist with an unexplained first item; rewrote it as continuous prose in the order a reader meets the regions in a .co file, keeping all five regions and their descriptions |
| 2.1 | x | round 25 | Reconciled the tutorial's two-part account with the coding reference's host/device/tileflow regions; restored the translation step that had been archived with a platform name; repaired "targeting :", "an keyword from ,", "handel" and the dangling "Those familiar with" opener; distinguished the listing caption from its heading and read the three regions in call order |
| 2.2 | x | round 26 | Four kernel fragments kept their code but gained distinct captions; replaced the empty "heart of the program" opener with the signature/structure lead-in, removed a duplicated element-type sentence, repaired the dangling "short answer" sentence, and re-anchored the misplaced s32/swap note to Sections 3.6.2 and 3.7 |
| 2.3 | x | round 27 | Gave the three repeated caption groups distinct names, repaired broken host-side grammar (value semantics, includes, RVO, runtime dimensions), replaced the "Nothing new here" aside with what the verification loop actually checks, and removed the meta-narration from the slot intro |
| 2.4 | x | round 28 | Named all eight shape examples individually, replaced the "first-class citizen / in this section" opener with an account of what a shape value does, re-anchored the dangling "This comes up" sentence, and removed a "Note that" opener without losing the dimensional-arithmetic rule |
| 2.5 | x | round 29 | Gave the ten integer/i-tuple listings distinct captions, replaced both "In this section" bookends with statements about what the constructs do, removed a dangling "However" and a dangling "As described", and converted leftover Markdown asterisks into a plain constraint sentence |
| 2.6 | x | round 30 | Named the repeated declaration, parameter and default-storage listings, replaced the "In this section" opening and the "This section covered" summary with the declaration/specifier facts they stood for, and turned a bare "Example:" into a lead-in for the shared qualifier |
| 2.7 | x | round 31 | Gave twenty-one repeated iteration listings distinct names, removed the "And / However / Note that" openers, and replaced both tutorial bookends with statements about what bounded variables do |
| 2.8 | x | round 32 | Re-anchored the "Here" pointer, removed the "However" opener and the tutorial "Wish you good luck" sign-off, and replaced the section-intro/summary meta narration with what chunkat actually selects |
| 2.9 | x | round 33 | Named the DMA syntax, future, and event listings individually, and removed the "So far / However" openers plus a duplicated "In the next section, this section introduces" phrase |
| 2.10 | x | round 34 | Named the four event listings, turned four "Note that / Note it / Therefore" openers into plain statements, fixed the "ochestrate" typo and removed the tutorial summary opener |
| 2.11 | x | round 35 | Replaced the meta-narration slot intro with the distinction between a predicate and an early return, and made the yield description say which instance leaves the code |
| 2.12 | x | round 36 | Replaced the meta-narration slot intro with the four contraction steps and made the rhythm sentence point at the spelling instead of an unstated list |
| 2.13 | x | round 37 | Named the repeated macro/call/cpp-string listings, corrected the contradictory macro-parameter assertion, removed four tutorial "Note that / Therefore / In this section" openers and repaired "code in cautious" |
| 2.14 | x | round 38 | Gave the compile-time and runtime print listings distinct captions and replaced the meta-narration intro with the cheap-to-expensive inspection order |
| 2.15 | x | round 39 | Removed the "part of the guide" meta narration and the dangling "next chapter" pointer in favour of an explicit cross-reference to the contraction section |
| 3.1 | x | round 40 | Named the two device-code listings, replaced the "This section introduces" opener with the three-layer structure, removed a dangling "Thus" and fixed "a end-to-end" |
| 3.2 | x | round 41 | Named the four partial-type examples, removed the "However" opener and made the intro state what each value family contributes instead of announcing the section |
| 3.3 | x | round 42 | Named the anonymous-dimension, value-numbering and simplification listings, removed the "So far / However / In this section" openers and replaced the intro with the two ways a shape can stay symbolic |
| 3.4 | x | round 43 | Restored the C++ if-else control-structure sentence that the coverage check flagged as lost, repaired the "implies" agreement error, and replaced three tutorial bookends with the layering/masking facts |
| 3.4.1 | x | round 44 | Replaced the meta-narration intro with the build-up order of the worked example and turned the bare "Start with just" opener into a sentence about the first version |
| 3.4.2 | x | round 45 | Replaced the meta-narration intro with the chain of indices and turned the rhetorical question into the statement it was meant to prompt |
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
