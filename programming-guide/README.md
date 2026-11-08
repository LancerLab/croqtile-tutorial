# Croqtile Programming Guide

A printable, language-first guide to Croqtile: 3 chapters (Getting Started →
Language Basics → Programming Guide), a syntax quick-reference appendix and a
bibliography. The subject of the book is the language itself; provenance and
build details live in the engineering documents, not in the PDF.

All three chapters are generated from the upstream source.
`UNIT-REGISTRY.md` records every source section, `tools/build_guide.py` turns
those sections into LaTeX fragments and assembles the chapters, and
`HARDWARE-NOTES.md` archives the passages moved out because they describe
target platforms or tuning rather than the Croqtile language.
`tools/review.py` drives a repeatable review → fix loop (see
`REVIEW-LOG.md`). Current output is 147 / 153 / 149 pages for A / B / C.

Built with **pdfLaTeX + BibTeX only** (no Tectonic, no `latexmk`), following the
directory and build conventions of `../gemm_sp_paper/camera-ready`,
`../gemm_sp_paper/sc26-pap1104` and `../croqtile-tuner-paper/sigopsatc2026-35`.

## Build

```bash
make                # default: edition A only, main.tex -> output/croqtile-programming-guide.pdf
make all             # build all three editions
make ebook           # same as plain `make`
make manual          # output/croqtile-programming-guide-manual.pdf
make memoir          # output/croqtile-programming-guide-memoir.pdf
make check           # build all three, then pdfinfo summary
make clean           # remove aux/log files and all output PDFs
```

Naming rule: the default edition uses `main.tex` and the unsuffixed PDF name;
the two optional editions add a suffix (`main-manual.tex` →
`...-manual.pdf`, `main-memoir.tex` → `...-memoir.pdf`). A plain `make`
never rewrites the other editions' PDFs; use `make all` to refresh all three.

Each target runs `pdflatex → bibtex → pdflatex → pdflatex` and copies the
result into `output/`. Requires TeX Live with `pdflatex`, `bibtex`,
`tcolorbox`, `listings`, `booktabs`, `tabularx`, `natbib`, `microtype`,
`geometry`, `titlesec`, `fancyhdr`, `memoir`, `hyperref` and `lmodern`.

## Three compiled editions (A is the default)

All three share `body.tex` and every file in `content/`; only the template skin
changes. Open the PDFs below to compare the designs.

| Edition | Root file | Output | Template |
|---|---|---|---|
| A (default) | `main.tex` | `output/croqtile-programming-guide.pdf` | [LaTeXTemplates eBook](https://latextemplates.com/template/ebook-book/) — technical eBook with code listings + callout boxes, MIT, pdfLaTeX |
| B | `main-manual.tex` | `output/croqtile-programming-guide-manual.pdf` | Software-manual skin modelled on [Overleaf Technical Document Template](https://www.overleaf.com/latex/templates/technical-document-template/mdgftpdfbvbs) |
| C | `main-memoir.tex` | `output/croqtile-programming-guide-memoir.pdf` | [LaTeXTemplates Memoir](https://latextemplates.com/template/memoir/) — classic long-form book |

## How the content is assembled

The guide is **generated from the upstream Markdown**, not hand-written. The
pipeline is:

1. `UNIT-REGISTRY.md` — the content-unit register. All 46 upstream pages
   (`tutorial/` 11 + `documentation/` 35) are decomposed into **337 numbered
   sections** (`F01-S01` … `F46-S01`). Each row records the original heading,
   the paragraphs / code blocks / tables / lists / figures / admonitions in
   that section, its bucket, and the exact chapter slot it goes to.
2. `tools/build_guide.py` — converts each registered section into one LaTeX
   fragment under `build/units/`, preserving the original text, code, tables,
   figures, captions and checklists. It then assembles
   `content/ch01-getting-started.tex`, `content/ch02-language-basics.tex` and
   `content/ch03-programming-guide.tex` in slot order.
   `build/manifest.tsv` is the machine-readable mapping; the source map lives
   in `build/SOURCE-MAP.md` and is never typeset in the guide.
3. `HARDWARE-NOTES.md` — the verbatim passages removed because they describe
   target platforms or tuning rather than the language, keyed by unit ID.
   Nothing is silently dropped.

Editorial operations are limited to `Choreo → Croqtile` naming, heading-level
normalisation, reader-facing point of view, real cross references, narrative
transition wording, and removal of target-platform sentences. The three
buckets are:

| Bucket | Chapter | What goes in |
|---|---|---|
| A 示例代码 + 使用流程 | 1 Getting Started | 配置、安装、编译运行、CLI、完整示例与调试工作流 |
| B 基本用法 + 如何理解程序 | 2 Language Basics | 程序味例子，逐项讲语法「是什么意思」，不深入抽象 |
| C 高级用法、抽象、概念、类型、设计 | 3 Programming Guide | 按功能分类的语法深入、类型系统、设计理由与完整 kernel |
| X | — | 整节移到 `HARDWARE-NOTES.md` |

Current counts: 315 units kept, 16 whole sections moved out, 6 empty/TODO
pages registered but not typeset; 62 chapter slots, 250 code listings, 21
tables, 33 figures, 347 archived hardware passages. DSL statement names
(`dma.copy`, `tma.copy`, `mma.row.row`, `: group-4`, `.swiz<N>`,
`global/shared/local`) are kept as language syntax.

### Rebuilding the content

```bash
python3 tools/build_guide.py   # regenerate fragments + chapters
make all                       # compile all three editions
make check                     # pdfinfo summary
```

Edit `UNIT-REGISTRY.md` to move a unit between slots, or edit the upstream
Markdown to change wording; then rerun the two commands. Do not edit the
generated `content/ch0*.tex` or `build/units/*.tex` by hand.

### Review loop

`tools/review.py` scores the reader-facing text on 21 categories (editorial
metadata, point of view, narrative scaffolding, broken prose, duplicated
headings and sentences, stale tutorial titles, empty markup, target-platform
vocabulary, conversion placeholders, paragraph structure, the Preface's
argument and prose, and the framing of section 1.1). Every round is comparable
and must not regress:

```bash
python3 tools/review.py --round N               # measure, or add a new check
... fix tools/build_guide.py / content/*.tex ...
python3 tools/review.py --round N+1 --compare  # total must fall
```

Scores are kept in `build/review-roundN.json`; the full history, the check
catalogue and the final acceptance numbers are in `REVIEW-LOG.md`. A zero
score is not the end of the loop: each new round starts by reading the
generated prose and the compiled PDF for problems the checker does not yet
measure.

Acceptance checks after the last build: 0 LaTeX errors, 0 overfull boxes,
0 undefined references, and 0 target-platform / tuning terms, conversion
placeholders or editorial metadata in `pdftotext` output for all three PDFs.
The review loop reports `TOTAL 0` across all categories with 3/3 chapter
roadmaps, 62/62 slot introductions and 3/3 "Where to go next" blocks.

## Web templates with online previews (candidates for a further switch)

| Template | Info page | PDF preview | PNG preview | Notes |
|---|---|---|---|---|
| LaTeXTemplates **eBook** (used in A) | [page](https://latextemplates.com/template/ebook-book/) | [PDF](https://latextemplates.com/pdfs/ebook-book.pdf) | [PNG](https://latextemplates.com/previews/ebook-book.png) | MIT, `book` class, code + tcolorbox callouts, pdfLaTeX |
| Overleaf **Technical Document / Software Manual** (basis of B) | [page](https://www.overleaf.com/latex/templates/technical-document-template/mdgftpdfbvbs) | [PDF](https://www.overleaf.com/latex/templates/technical-document-template/mdgftpdfbvbs.pdf) | — | Official software-manual template, `ol-softwaremanual` class (Overleaf-only), uses `minted` |
| Overleaf **Coding Book Template** | [page](https://www.overleaf.com/latex/templates/coding-book-template/ksbrfmrgmjgy) | [PDF](https://www.overleaf.com/latex/templates/coding-book-template/ksbrfmrgmjgy.pdf) | — | Explicitly made for *programming guide books* (`bbe` class) |
| LaTeXTemplates **nostarch** | [page](https://latextemplates.com/template/nostarch/) | [PDF](https://latextemplates.com/pdfs/nostarch.pdf) | [PNG](https://latextemplates.com/previews/nostarch.png) | No Starch Press-style technical book, pdfLaTeX |
| LaTeXTemplates **kaobook** | [page](https://latextemplates.com/template/kaobook/) | [PDF](https://latextemplates.com/pdfs/kaobook.pdf) | [PNG](https://latextemplates.com/previews/kaobook.png) | Wide margins, sidenotes, KOMA-Script |
| LaTeXTemplates **ElegantBook** | [page](https://latextemplates.com/template/elegantbook/) | [PDF](https://latextemplates.com/pdfs/elegantbook.pdf) | [PNG](https://latextemplates.com/previews/elegantbook.png) | Coloured chapter styling; upstream also on [GitHub](https://github.com/ElegantLaTeX/ElegantBook) |
| LaTeXTemplates **Legrand Orange Book** | [page](https://latextemplates.com/template/legrand-orange-book/) | [PDF](https://latextemplates.com/pdfs/legrand-orange-book.pdf) | [PNG](https://latextemplates.com/previews/legrand-orange-book.png) | Multi-part book, orange parts pages |
| LaTeXTemplates **MathsCodeBook** | [page](https://latextemplates.com/template/mathscodebook/) | [PDF](https://latextemplates.com/pdfs/mathscodebook.pdf) | [PNG](https://latextemplates.com/previews/mathscodebook.png) | Theorems + Python listings |
| LaTeXTemplates **Memoir** (used in C) | [page](https://latextemplates.com/template/memoir/) | [PDF](https://latextemplates.com/pdfs/memoir.pdf) | [PNG](https://latextemplates.com/previews/memoir.png) | Classic book/report class |
| Overleaf gallery: technical manuals | [gallery](https://www.overleaf.com/gallery/tagged/manual) | — | — | More technical-manual templates |

**Local-TeX caveat:** `ElegantBook` needs `newtxtext.sty` and `kaobook` needs
`newpxtext.sty`/`newpxmath.sty`; those font packages are missing from this
machine's TeX Live, so they are offered as preview links rather than compiled
editions. They can be swapped in after installing the fonts. The Overleaf
`bbe`/`ol-softwaremanual` classes are Overleaf-only, so edition B reproduces
that template's visual grammar on the standard `book` class.

## Directory structure

```
programming-guide/
├── Makefile                 # pdfLaTeX -> BibTeX -> pdfLaTeX x2 -> output/
├── README.md
├── REVIEW-LOG.md            # review-loop rounds, checks, final acceptance
├── UNIT-REGISTRY.md         # approved 46-file / 337-section content-unit map
├── HARDWARE-NOTES.md        # verbatim passages moved out of the guide
├── tools/build_guide.py     # Markdown -> LaTeX fragments -> chapters
├── tools/review.py          # repeatable review -> fix loop driver
├── main.tex                 # edition A root (eBook)
├── main-manual.tex          # edition B root (software manual)
├── main-memoir.tex          # edition C root (memoir)
├── body.tex                 # shared chapter/include order (3 chapters + appendix A)
├── croqguide.sty            # shared macros, colours, listings language, callouts
├── references.bib
├── preamble/                 # one skin per root document
│   ├── ebook.tex
│   ├── manual.tex
│   └── memoir.tex
├── content/                 # front/back matter, appendix A; ch01-03 generated
├── build/units/             # one generated .tex fragment per registered section
├── build/manifest.tsv       # uid -> source -> slot -> status
├── build/SOURCE-MAP.md      # engineering provenance map (not in the PDF)
├── build/review-roundN.json # per-round review scores
├── figures/                 # diagrams copied from upstream for the generated figures
├── templates/ebook-book/    # pristine upstream eBook template + MIT licence
└── output/                  # compiled PDFs and template previews
```

## Credits and licence

- Text and code: transcribed and regrouped from the Croqtile tutorial (see
  `build/SOURCE-MAP.md` and `UNIT-REGISTRY.md` for the full
  section-by-section source map).
- Edition A is based on the [LaTeXTemplates eBook](https://latextemplates.com/template/ebook-book/)
  template (MIT, © Anton Mnatsakanov / AnMnv); the pristine files and the full
  licence text are in `templates/ebook-book/`.
- Edition B is modelled on Overleaf's Technical Document Template (LPPL-1.3c).
- Template sources and preview links were verified November 2026.
