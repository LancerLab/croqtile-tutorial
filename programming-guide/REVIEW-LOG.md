# Review loop log

Engineering document. Nothing here is typeset in the Programming Guide PDF.

`tools/review.py` turns "review the current version, find problems, fix them"
into a repeatable loop. It scans only files that reach the PDF: every generated
unit, front/back matter, the quick-reference appendix and the three chapter
files. Engineering documents (`README.md`, `UNIT-REGISTRY.md`,
`HARDWARE-NOTES.md`, `build/SOURCE-MAP.md`) are deliberately excluded, and
verbatim code listings are blanked out before scanning so that sample code
does not create false positives.

```
python3 tools/build_guide.py
python3 tools/review.py --round N               # baseline, or a new dimension
... edit tools/build_guide.py or content/*.tex ...
python3 tools/build_guide.py
python3 tools/review.py --round N+1 --compare  # must not regress
```

Each round writes `build/review-roundN.json`. `--compare` prints a
per-category diff against the previous round and exits non-zero if a non-zero
total did not fall; two consecutive zero rounds report convergence. A round
is **not** considered finished merely because its score
is zero: the loop keeps going by finding a dimension the checker does not yet
measure and adding it. Rounds 4, 6 and 8 were opened exactly that way, after
reading the generated prose and the compiled PDF rather than trusting the
previous round's zero. Rounds 11, 12 and 13 came from direct reader requests
about the Preface: first it needed language positioning, then a proper
argument from background trends through human productivity to agent-readiness,
and finally natural paragraph breaks and continuous prose instead of
label-style fragments. Each request became a regression check.

The reader-facing target is TileLang's Programming Guides: the subject is the
language, the reader is "you", every construct is introduced before it is
defined, and no sentence explains where the documentation came from.

## Check catalogue

| Category | What it catches |
|---|---|
| `meta` | Editorial provenance, build-tool names, source-map pointers |
| `pov` | "we / our" narration, "this chapter", old `Chapter N` numbers |
| `generic_headings` | Dozens of identical `Overview` / `Summary` headings |
| `typos` | Known spelling and grammar defects from the source pages |
| `broken` | Sentence fragments left by clause removal |
| `dangling_refs` | Tutorial navigation links whose targets disappeared |
| `dup_headings` | Source headings that merely repeat their section title |
| `repeated_sentences` | The same long sentence used in two different units |
| `stale_titles` | `Ch 0:`-style tutorial chapter titles |
| `inline_number_emph` | Markdown multiplication read as emphasis (`6 * 10 * 10`) |
| `heading_only` | A typeset heading with no body beneath it |
| `merged_paragraphs` | Two `\input`s on consecutive lines, typeset as one paragraph |
| `hardware` | Target-platform / tuning vocabulary (`build_guide.HW`) |
| `placeholders` | Leaked conversion placeholders (`QZ000QZ`, `QZC0000QZ`) |
| `empty_markup` | `\emph{}`, empty `\item`s, article-only paragraphs |
| `emph_ref` | A cross-reference still wrapped in link-label emphasis |
| `tutorial_voice` | "Welcome to the tutorial", "coding reference", "Each chapter" |
| `fragments` | `vs.`-truncated clauses, cut-off `protocols`, fused sentences |
| `preface_positioning` | Preface argument order (background trend → human productivity → agent-ready), heterogeneous positioning, natural paragraph breaks and no label-style lists |
| `narrative_gap` | Missing chapter roadmap / slot intro / "where to go next" |

## Rounds

| Round | Total | What the review found | What changed |
|---|---:|---|---|
| 0 | 319 | 57 provenance sentences, 127 "we/our" phrases, 39 generic headings, 12 typos, 16 fragments, 68 missing narrative anchors | Baseline before the reader-facing rewrite |
| 1 | 10 | Leftover metadata and POV | Rewrote `frontmatter.tex`, deleted reader-facing Appendix B, moved provenance to `build/SOURCE-MAP.md`, cleaned `backmatter.tex`, normalised POV and chapter references, added `\guidechapterintro` / `\guideslotintro` / `\guidechapternext` |
| 2 | 15 | 8 stale navigation links, 7 headings duplicating their slot title (new checks) | Real `Section~\ref{...}` targets and heading de-duplication |
| 3 | 0 | — | All three editions compiled clean |
| 4 | 22 | 4 empty markup / article-only paragraphs, 7 emphasised references, 7 tutorial-voice sentences, 4 truncated clauses (independent prose scan) | `clean_fragment()`, `scrub()` keeps dash/semicolon joiners, cross-references lose emphasis, tutorial welcome pages rewritten, old ten-chapter list replaced, three broken sentences repaired |
| 5 | 0 | — | All new categories clear |
| 6 | 1 | The same definition sentence appeared twice (new `repeated_sentences` check) | Opening pages split into distinct angles: motivation, how to read the examples, the three construct families, the language's promises, and a transition to the next section |
| 7 | 0 | — | No repeated long sentences |
| 8 | 271 | 1 stale `Ch 0:` title, 1 Markdown multiplication read as emphasis, 16 bare headings, 253 adjacent units typeset as one paragraph (independent PDF reading) | Blank line after every `\input`, empty group headings suppressed, tutorial chapter titles dropped, the empty unit given a transition, the `chunkat` paragraph rewritten, `\emergencystretch` / `\hfuzz` tuned |
| 9 | 0 | — | All structural categories clear |
| 10 | 0 | Final acceptance round after rebuilding all three editions | No regressions in any category |
| 11 | 0 | Reader feedback: the Preface opened too technically, with no sense of what Croqtile is for (new `preface_positioning` check) | Replaced the opening with one sustained paragraph on heterogeneous computing and AI-native design, followed by a short paragraph on program shape; all three editions rebuilt |
| 12 | 0 | Reader feedback: the paragraph argued from AI-native alone, with no background trend and no human-first motivation (`preface_positioning` extended to enforce concept order) | Rewrote the Preface around background trend → why high-level kernel DSLs emerged → human productivity → agent-readiness, drawing on `../croqtile-tuner-paper/paper/main.tex` and the first two ATC background slides; all three editions rebuilt |
| 13 | 0 | Reader feedback: one paragraph was too long, and the rest of the Preface was a set of label-style fragments rather than prose (`preface_positioning` extended to enforce segmentation) | Split the opening into five natural paragraphs, merged the two overlapping reading-order sections, and rewrote audience, prerequisites, scope, chapter guide and conventions as continuous prose; all three editions rebuilt |

## Final acceptance

- Editions A / B / C: **145 / 151 / 149 pages**; 0 LaTeX errors,
  0 overfull boxes, 0 undefined references.
- `pdftotext`: 0 target-platform / tuning terms, 0 conversion placeholders,
  0 editorial metadata, 0 "we / our" phrasing, 0 tutorial voice,
  0 stale chapter headings, 0 empty cross-references.
- Content: 315 units kept, 16 whole sections moved out, 6 registered empty
  pages; 62 chapter slots, 250 code listings, 21 tables, 33 figures,
  347 archived hardware passages.
- Narrative scaffolding: 3/3 chapter roadmaps, 62/62 slot introductions,
  3/3 "Where to go next" blocks.
