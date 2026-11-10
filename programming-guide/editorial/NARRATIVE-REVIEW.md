# Narrative review — 10 November 2026

This pass follows the user’s request to expand thin explanations and make each section readable on its own. It adds 246 reviewed English edits across 152 units, with corresponding Chinese prose updates. Existing examples and registered section placement are retained.

## What changed

- Added the missing reasoning around coordinate composition, ownership, shape evaluation, asynchronous readiness, buffer reuse, and accumulator lifetime.
- Restored explanations that the Chinese tokenizer had skipped when paragraphs began with inline LaTeX formatting; the tokenizer now preserves these forms.
- Moved event basics before event arrays, program structure before the topic index, and the contraction introduction before layout details. These changes stay within the original sections.
- Replaced unsupported absolute claims about static checking, dynamic shapes, utilization, and transfer overhead with the actual conditions.
- Completed the RGB-to-grayscale walkthrough and the pipeline/interop comparison tables; repaired detached transitions, duplicate explanations, and stale chapter references.

## Review limits

All 239 listing bodies match the pre-pass snapshot. This was a prose and publication pass, not compilation or runtime testing of kernels. The text now identifies omissions in the scalar/DMA accumulation sketches, the final wait in the swap example, the MMA geometry macro definitions, and the schematic multi-level matmul. The examples remain unchanged.

The front matter and appendix already provide their own reading context and were reviewed without adding filler. Syntax tables remain references rather than being padded into tutorials.

## Section coverage

The table records where the English text changed; sections without unit edits were reviewed in context and retained. Chinese editorial hooks were also checked for repetition after restoring the missing text.

| Section | Topic | Units revised |
|---|---|---:|
| 1.1 | What is Croqtile | 0 |
| 1.2 | Setting Up: Requirements, Build, and Verification | 0 |
| 1.3 | The Toolchain: Command Line and Build Utilities | 0 |
| 1.4 | Inspecting and Debugging a Build | 3 |
| 2.1 | The Two Parts of a Croqtile Program | 0 |
| 2.2 | Reading a Kernel Line by Line | 2 |
| 2.3 | The Host-Side API: Inputs and Outputs | 2 |
| 2.4 | Shapes and mdspan | 4 |
| 2.5 | Integers, I-Tuples, and Bounded Values | 4 |
| 2.6 | Spanned Data and Storage Specifiers | 9 |
| 2.7 | Iteration: parallel-by, with-in, and foreach | 10 |
| 2.8 | Tile Selection: chunkat, subspan, and view/from | 7 |
| 2.9 | Data Movement: the DMA Statement and Futures | 5 |
| 2.10 | Coordination: inthreads, Events, and Async Blocks | 4 |
| 2.11 | Branching and Early Return | 2 |
| 2.12 | The Four-Step MMA Syntax | 1 |
| 2.13 | C++ Embedding, Calls, Macros, and the Preprocessor | 5 |
| 2.14 | Printing, Assertions, and Compile-Time Inspection | 1 |
| 2.15 | Syntax Quick Reference | 2 |
| 3.1 | Program Structure and Compilation Models | 4 |
| 3.2 | Syntax Design and the Type System | 11 |
| 3.3 | Symbolic Shapes, Shape Evaluation, and Value Numbering | 12 |
| 3.4 | Parallelism, Space Specifiers, and Masking | 5 |
| 3.4.1 | Worked Example: Tiled Matrix Multiplication | 1 |
| 3.4.2 | Tracing One Output Element | 1 |
| 3.5 | Tile Selection and Data Expressions | 5 |
| 3.5.1 | Arbitrary-Offset Windows: view and from | 1 |
| 3.5.2 | Strided Tiles: subspan, step, and at | 1 |
| 3.5.3 | Zero-Padding with zfill | 1 |
| 3.5.4 | Layout Reinterpretation with span_as | 1 |
| 3.5.5 | Tile Selection: Summary and Syntax | 1 |
| 3.6 | Data Movement: Operations, Configuration, and Async Patterns | 6 |
| 3.6.1 | Worked Example: DMA Matmul | 1 |
| 3.6.2 | Multi-Buffering: dma.any, swap, and rotate | 4 |
| 3.6.3 | Worked Example: A Pipelined Kernel | 1 |
| 3.7 | Contraction and Layout Contracts | 3 |
| 3.7.1 | Worked Example: One Cooperative Group | 1 |
| 3.7.2 | Worked Example: Widening the Cooperation Scope | 1 |
| 3.7.3 | Worked Example: Two Cooperative Groups | 1 |
| 3.7.4 | MMA Syntax Summary | 1 |
| 3.7.5 | MMA Design Summary | 0 |
| 3.8 | Coordination, Pipelines, and Persistent Scheduling | 3 |
| 3.8.1 | Canonical Pattern: One Producer and One Consumer | 1 |
| 3.8.2 | Worked Example: A 1P1C Kernel Skeleton | 1 |
| 3.8.3 | Persistent Scheduling | 2 |
| 3.8.4 | Host-Level Concurrency and Stream Binding | 2 |
| 3.8.5 | Worked Example: A Full Event Pipeline | 1 |
| 3.8.6 | Coordination Pattern Summary | 2 |
| 3.9 | C++ Interoperability and the Preprocessor | 3 |
| 3.9.1 | Worked Example: Calling a Device Helper | 1 |
| 3.9.2 | Verbatim C++ Injection | 1 |
| 3.9.3 | Interoperability Summary | 1 |
| 3.10 | Tileflow Patterns and Design Rationales | 6 |
| 3.10.1 | Worked Example: Tiled matmul | 2 |
| 3.10.2 | Worked Example: rgb2gray | 1 |
| 3.11 | Diagnostics and Methodology | 1 |

## Publication validation

- English ebook: 139 pages; manual: 143 pages; memoir: 141 pages.
- Chinese edition: 132 pages, 235 listings, 31 figures, and 21 tables.
- All 377 translation keys are present, with zero tag mismatches or forbidden terminology hits.
- No LaTeX errors, undefined references, or overfull boxes in the final English builds.
- All 239 source listing bodies are unchanged. The Chinese edition’s listing count differs because of its existing editorial composition.
- Inspected rendered pages for the contraction introduction, event pipeline, RGB conversion, diagnostics, and Figure 10. Contact sheet: `output/preview/narrative-review.jpg`.
- `git diff --check` passes, excluding generated TeX logs.
