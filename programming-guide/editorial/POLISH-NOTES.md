# Text and figure polish — November 2026

Scope: the English programming guide (all three PDF editions), the Chinese
edition, and the 28 Manim figures used by these editions. The chapter order,
registered units, and executable examples are preserved.

- Reviewed prose, headings, tables, captions, and the front matter. English
  sentence and grammar corrections are recorded as exact before/after edits
  in `editorial/prose-polish.json`; Chinese edits live in `translation/zh/`.
- Unified figure fonts, title and caption positions, panel alignment, borders,
  and spacing. All publication renders use a 2560 × 1440 canvas, preserving
  relative type sizes when embedded at the same width.
- Recreated Chinese Figure 10 with `figures/guide_chunkat_3d.py` (relative to
  the repository root). The original partition and highlighted chunk are
  preserved; labels distinguish chunk coordinates from element coordinates.
- Added a static compile-and-run layout and fixed the Memoir header height
  so wrapped running headers fit.

Validation: all 239 generated code-listing bodies match the starting snapshot;
all 377 translation keys are covered with matching token sequences; every
figure reference resolves. Rendered contact sheets and the Figure 10 pages
were inspected. All English editions compile without errors, undefined
references, or overfull boxes; the Chinese PDF builds with WeasyPrint 70.

The legacy editorial checker has no regression. Its existing 62 missing
slot-introduction flags and one hardware-vocabulary flag were present before
this pass; the slot-introduction check expects an earlier book structure.

Rebuild commands are in README.md. In this workspace, the cached Chinese PDF
dependencies are available with `PYTHONPATH=/tmp/croqtile-zh-deps`.

A subsequent narrative pass expands thin explanations in both languages and
restores Chinese paragraphs lost by the old inline-formatting tokenizer.
See `NARRATIVE-REVIEW.md` for all 56 sections, the reordered introductions,
validation, and the limits of the unchanged source examples.
