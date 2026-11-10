# Croqtile Programming Guide — 内容单元登记表

本表是「原素材完整分解 → 归类 → 填放 → 串联」的第一步产物。逐篇逐小节编号，
登记的是**上游原文的内容单元**，不是摘要、不是改写、不是 placeholder。
填正文时按 `单元 ID` 回源取整段 / 整段代码 / 整张表 / 整组图注，只做命名统一、
去重交叉引用、补过渡句、统一标题层级。

- 统计口径：`P` 散文段、`C` 代码块、`T` 表格、`L` 列表/清单、`F` 图/图注/动画、
  `A` 提示框/指令、`M` 展示数学块。`×n` 为该节内该类型单元个数。
- 归类：**A → Ch1**（示例代码 + 使用流程）、**B → Ch2**（基本用法 / 如何理解程序）、
  **C → Ch3**（抽象、概念、类型、设计、高级用法）、**X → 移出本 guide**。
- 硬件处理：`NONE` 直接保留；`MIXED` 逐单元拆——DSL 语法、代码、表格、规则句保留，
  硬件句/段/图注移到独立文档；`DOMINANT` 仅保留其中的语法行或抽象结论，其余移出。
  DSL 语句名（`dma.copy`、`tma.copy`、`mma.row.row`、`: group-4`、`.swiz<N>`、
  `global/shared/local`）作为语言语法保留，不作为硬件内容删除。
- 上游空文件、TODO 占位、空标题：登记在案但不填正文，仅在附录 B 的素材映射中说明。
- 唯一允许的串联改动：`Choreo → Croqtile`、`choreo → croqtile`；重复单元合并或交叉引用；
  章/节过渡句；标题层级与行文统一。

## 目标骨架（填放位置）

Ch1 Getting Started — A：示例代码 + 使用流程
  1.1 What is Croqtile
  1.2 Setting Up: Requirements, Build, and Verification
  1.3 The Toolchain: Command Line and Build Utilities
  1.4 Your First Programs                 1.5 Inspecting and Debugging a Build

Ch2 Language Basics — B：基本用法 + 如何理解程序
  2.1 程序两部分（__co__ / host）          2.2 逐行读一个 kernel        2.3 Host API 与输入/输出
  2.4 Shape 与 mdspan                     2.5 整数、I-Tuple、Bounded   2.6 Spanned Data 与存储说明符
  2.7 迭代：parallel-by / with-in / foreach / range / 空域说明符
  2.8 选块：chunkat / subspan / view.from / #    2.9 DMA 语句 / future / 同步异步 / wait
  2.10 协调：inthreads / event / trigger-wait / .async
  2.11 分支与提前返回（if / yield）       2.12 四步 mma 语法（基础）
  2.13 C++ 嵌入：__device__ / __cpp__ / call / 宏与预处理
  2.14 打印、断言与编译期检查             2.15 语法速查表（各章 New syntax / Checklist 汇总）

Ch3 Programming Guide — C：抽象、概念、类型、设计、高级用法
  3.1 程序结构与编译模型                  3.2 语法设计与类型系统        3.3 符号形状、求值与值编号
  3.4 并行与空域说明符（3.4.1/3.4.2 完整示例与追踪）
  3.5 选块与数据表达式（3.5.1–3.5.4、3.5.10 小结）
  3.6 数据移动：操作族/配置/异步模式（3.6.1–3.6.3 完整示例）
  3.7 收缩与布局契约（3.7.1–3.7.3 完整 kernel、3.7.9/10 表）
  3.8 协调、流水线与持久调度（3.8.1–3.8.5、3.8.10）
  3.9 C++ 互操作与预处理（3.9.1/3.9.2、3.9.10）
  3.10 Tileflow 模式与设计论证（3.10.1 matmul / 3.10.2 rgb2gray）
  3.11 诊断与方法论                       X = 整节移出本 guide → 独立硬件文档

### F01 `tutorial/index.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F01-S01` | # Croqtile Tutorial | P×2 (welcome to Croqtile tutorial, running-example method plus reference link) | A | 1.1 | 拆分：移出硬件句，保留 DSL 单元；开篇 P1「high-performance GPU kernels」改为「Croqtile programs」 |
| `F01-S02` | ## Chapters | L×1 (10 items, "0. Installation: Setting Up the Croqtile Compiler") | A | 1.1 | 拆分：移出硬件句，保留 DSL 单元；10 章清单保留为本指南导读；章名中的硬件词去掉 |
| `F01-S03` | ## Prerequisites | L×1 (3 items, "- Basic C++ knowledge (functions, pointers, arrays)") | A | 1.1 | 拆分：移出硬件句，保留 DSL 单元；前置条件第 2 项改为「并行编程基本概念」 |

### F02 `tutorial/ch00-installation.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F02-S01` | # Ch 0: Installing Croqtile | P×1 (compiler needed, chapter walks dependencies) | A | 1.2 | — |
| `F02-S02` | ## System Requirements | P×1 (need C++17 compiler and tools) ; T×1 (headers: Dependency\|Minimum Version, 4 rows) ; P×1 (older Bison auto-fetched by setup) | A | 1.2 | 拆分：移出硬件句，保留 DSL 单元；依赖表保留版本数字；删除「(for GPU targets)」括注 |
| `F02-S03` | ## Build from Source | P×1 (clone repo, run automated setup) ; C×1 (bash, ~3 lines, "git clone https://github.com/LancerLab/croqtile.git") ; P×1 (setup pulls submodules, then build) ; C×1 (bash, 1 line, "make") ; P×1 (run test suite) ; C×1 (bash, 1 line, "make test") ; P×1 (binary ready, add to PATH) | A | 1.2 | — |
| `F02-S04` | ## Verify the Installation | P×1 (create minimal .co file) ; C×1 (choreo, ~15 lines, "__co__ s32 [4] identity(s32 [4] input) {") ; P×1 (compile and run) ; C×1 (bash, 2 lines, "croq verify.co -o verify") ; P×1 (you should see OK) | A | 1.2 | — |
| `F02-S05` | ## Compiler Usage | P×1 (command works like gcc/clang) ; C×1 (bash, 4 lines, "croq program.co                     # compile and link → a.out") ; P×1 ("Key flags:") ; T×1 (headers: Flag\|Effect, 8 rows) | A | 1.3 | 拆分：移出硬件句，保留 DSL 单元；CLI 表保留全部 flag；`-t/-es` 的 CUDA 说明改为「目标源码」 |
| `F02-S06` | ## Development Utilities | P×1 (Makefile shortcuts for test suites) ; C×1 (bash, 3 lines, "make help                              # list all available targets") ; P×2 (useful when modifying Croqtile, ready for Chapter 1) | A | 1.3 | — |

### F03 `tutorial/ch01-hello-croqtile.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F03-S01` | # Hello Croqtile: From Zero to Running Kernel | P×1 (write/compile/run complete elementwise addition) | A | 1.3 | — |
| `F03-S02` | ## The Two Parts of a Croqtile Program | P×1 (every program has two parts) ; L×1 (2 items, "1. **Croqtile Function** (`__co__`) — the kernel logic") ; P×1 (both live in one .co file) | B | 2.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F03-S03` | ## A Complete Example: Element-Wise Addition | P×1 (full program adds two [4,8] matrices) ; C×1 (choreo, ~24 lines, "__co__ s32 [4, 8] ele_add(s32 [4, 8] lhs, s32 [4, 8] rhs) {") ; P×1 (save as ele_add.co, compile and run) ; C×1 (bash, 2 lines, "croq ele_add.co -o ele_add") ; P×1 (you should see Test Passed) | A | 1.3 | — |
| `F03-S04` | ## The Croqtile Function | P×1 (heart of program, piece by piece) ; P×1 (**Function signature.**) ; C×1 (choreo, 1 line, "__co__ s32 [4, 8] ele_add(...) {") ; P×2 (`__co__` prefix, shaped signature in-bounds; `// TODO` datatype list s32/f16/bf16/f32/f8_e4m3/f8_e5m2) ; P×1 (**Output buffer declaration.**) ; C×1 (choreo, 1 line, "s32 [lhs.span] output;") ; P×1 (lhs.span copies full shape) ; P×1 (**The computation.**) ; C×1 (choreo, 2 lines, "parallel {i, j} by [4, 8]") ; P×2 (32 parallel instances one element per thread; CUDA-thread discussion, space specifiers : block/: thread, warps, warpgroups, GPU) ; P×1 (**Return.**) ; C×1 (choreo, 1 line, "return output;") ; P×1 (returns result tensor, croq::spanned_data, [i][j]) | B | 2.2 | 拆分：移出硬件句，保留 DSL 单元；删除「If you have written CUDA before…」整段；其余语法讲解全留 |
| `F03-S05` | ## The Host Program | P×1 (plain C++ with Croqtile API calls) ; C×1 (choreo, 5 lines, "auto lhs = croq::make_spandata<croq::s32>(4, 8);") ; P×1 (make_spandata creates owning buffer) ; C×1 (choreo, 1 line, "auto res = ele_add(lhs.view(), rhs.view());") ; P×2 (.view() produces spanned_view, returns spanned_data; rest of main is verification) | B | 2.3 | — |
| `F03-S06` | ## Build and Compile | P×1 (.co extension, compiler like gcc/clang) ; C×1 (bash, 3 lines, "croq ele_add.co                          # produces a.out") ; P×2 (-es/-t flags, run `croq --help`) ; C×1 (bash, 1 line, "croq --help") ; P×1 (prints full compiler flag set) ; F×4 (compile_and_run_dark.png#only-dark "Terminal: compile and run", compile_and_run_light.png#only-light "same", *caption* "Compiling/running plus --help output", <details> compile_and_run_dark.mp4/compile_and_run_light.mp4 "animated version") | A | 1.3 | 拆分：移出硬件句，保留 DSL 单元 |
| `F03-S07` | ## What You Have So Far | P×2 (skeleton recap: __co__, parallel/.at(), lhs.span, make_spandata/.view(); element-at-a-time limits, memory hierarchy, GPUs move data in bulk, next chapter dma.copy/chunkat) | C | 3.6 | 拆分：移出硬件句，保留 DSL 单元；删除 memory hierarchy / 吞吐段落，保留已学语法清单 |

### F04 `tutorial/ch02-data-movement.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F04-S01` | # Data Movement: The Tile Movement Engine | P×1 (scalar indexing vs how hardware moves data) ; P×1 (Croqtile makes tile-movement statements explicit) ; L×1 (4 items, "- `dma.copy` moves a tile as-is.") ; P×1 (model, core syntax, DMA/TMA lowering rules) ; F×4 (fig1_element_vs_block_dark.png#only-dark, fig1_element_vs_block_light.png#only-light, *caption* "Left element-by-element, Right tile-by-tile fast memory", <details> anim1_element_vs_block_dark/light.mp4 "animated version") | C | 3.6 | 拆分：移出硬件句，保留 DSL 单元；删除 GPU/TMA/带宽句，保留「tile movement 语句显式化」论断与 4 项清单 |
| `F04-S02` | ## The DMA Statement Family | P×1 (all operations share one form) ; C×1 (choreo, 1 line, "future = engine.operation.modifiers source => destination;") ; P×1 ("The important pieces are:") ; T×1 (headers: Part\|Meaning, 5 rows) ; P×1 ("Minimal examples:") ; C×1 (choreo, 3 lines, "f = dma.copy input.chunkat(block) => shared;") ; P×1 (key shift: shaped regions not elements) | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S03` | ## First Example: Tiled Addition Through Shared Memory | P×1 (same addition staged through shared memory) ; C×1 (choreo, ~14 lines, "__co__ s32 [64, 128] tiled_add_2d(...) {") ; P×1 ("What changed is the data path:") ; L×1 (4 items, "1. `chunkat(tr, tc)` selects one logical tile") ; P×1 (core pattern select/move/compute) ; F×4 (fig2_tiled_add_dark.png#only-dark, fig2_tiled_add_light.png#only-light, *caption* "Load tile, compute, write global coords", <details> anim2_tiled_add_dark/light.mp4 "animated version") | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S04` | ## `chunkat`, `subspan`, and `view().from()`: Three Ways to Select a Tile | P×1 (three ways to describe the tile) | C | 3.5 | — |
| `F04-S05` | ### `chunkat(...)`: equal partitioning | C×1 (choreo, 1 line, "lhs.chunkat(tr, tc)") ; P×2 (equal chunks, [16,16] tile from [64,128] by [4,8]; natural when tiling follows parallel decomposition) ; F×4 (fig3_chunkat_dark.png#only-dark, fig3_chunkat_light.png#only-light, *caption* "chunkat(1, 3) selects one equal tile", <details> anim3_chunkat_dark/light.mp4 "animated version") | B | 2.8 | — |
| `F04-S06` | ### `subspan(...).at(...)`: explicit tile extents on a tile grid | C×1 (choreo, 1 line, "lhs.subspan(64, 64).at(block_m, iv_k)") ; P×1 (subspan = tile shape, .at = instance, GEMM [64,64] tile) | B | 2.8 | — |
| `F04-S07` | ### `view(...).from(...)`: explicit tile extents with explicit element offsets | C×1 (choreo, 1 line, "lhs.view(TILE_M, TILE_K).from(offset_m, offset_k)") ; P×2 (most general selection form; right tool for element offsets/irregular access where chunkat is awkward) ; P×1 ("You can think of the three forms like this:" [closing 3-form summary, document order]) ; L×1 (3 items, "- Use `chunkat(...)` when the surrounding parallel decomposition already defines an equal tiling.") | B | 2.8 | — |
| `F04-S08` | ## Futures, `.data`, and `.span` | P×1 (result is a future-like handle) ; C×1 (choreo, 1 line, "f = dma.copy input => shared;") ; P×1 ("two important things:") ; L×1 (2 items, "- `f.data`: the moved tile, as a spanned tensor.") ; P×1 ("Example:") ; C×1 (choreo, 3 lines, "f = dma.copy input => shared;") ; P×1 (even sync copies worth naming) ; F×3 (fig_future_data_dark.png#only-dark, fig_future_data_light.png#only-light, *caption* "Handle exposes .data tile and .span shape") | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S09` | ## Synchronous and Asynchronous Copies | P×1 (copies synchronous by default) ; C×1 (choreo, 2 lines, "f = dma.copy input => shared;") ; P×1 (add .async for background) ; C×1 (choreo, 3 lines, "f = dma.copy.async input => shared;") ; P×1 (same pattern applies to TMA) ; C×1 (choreo, 3 lines, "f = tma.copy.async input => shared;") ; P×2 (Rule: wait before reading f.data; .async creates in-flight transfer, wait makes safe) | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S10` | ## Other Operations: `transp` and `pad` | P×1 (copy is simplest family member) | C | 3.6 | — |
| `F04-S11` | ### `dma.transp<...>` | C×1 (choreo, 2 lines, "fa = dma.transp<1, 0, 2> a => local;") ; P×1 (permutes dims during movement, produces transposed layout) | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S12` | ### `dma.pad<...>` | C×1 (choreo, 3 lines, "f = dma.pad<{2, 1}, {3, 2}, {0, 0}, V>.async input => shared;") ; P×1 (larger dest tile; low/high/stride padding + fill value V) | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S13` | ## `#`: Compose and Extent | P×1 (Croqtile uses # in two ways) | C | 3.5 | — |
| `F04-S14` | ### Compose: `outer # inner` | C×1 (choreo, 1 line, "output.at(tr # i, tc # j)") ; P×2 (combines tile + in-tile coordinate; read tr # i as row i inside tile tr) ; F×3 (fig_compose_dark.png#only-dark, fig_compose_light.png#only-light, *caption* "Compose tile index with in-tile offset") | B | 2.8 | — |
| `F04-S15` | ### Extent: `#name` | C×1 (choreo, 1 line, "foreach i in [128 / #tile]") ; P×1 (#tile is extent of tile axis; parallel tile by 8 → #tile is 8) ; F×3 (fig_extent_dark.png#only-dark, fig_extent_light.png#only-light, *caption* "Prefix # extent, infix # compose") | B | 2.8 | — |
| `F04-S16` | ## `span` and `span(i)` | P×1 (span = full shape, span(i) = one dim) ; C×1 (choreo, 1 line, "s32 [lhs.span(0), rhs.span(1)] output;") ; P×1 (output shapes from different inputs) ; F×3 (fig_span_dark.png#only-dark, fig_span_light.png#only-light, *caption* "span whole shape, span(i) single axis") | B | 2.4 | — |
| `F04-S17` | ## Memory Specifiers: `global`, `shared`, and `local` | P×1 (destination states where tile lives) ; L×1 (3 items, "- `global`: device memory. Large, slow, visible across the device.") ; C×1 (choreo, 3 lines, "f0 = dma.copy input => shared;") ; P×1 (shared is important level; global↔shared lowering, local not fast path) ; F×3 (fig_memory_hierarchy_dark.png#only-dark, fig_memory_hierarchy_light.png#only-light, *caption* "Specifiers map onto GPU memory hierarchy") | X | X | 整节移出 → 独立硬件文档；整节移出；`global/shared/local` 三个说明符名已在 §2.6 保留 |
| `F04-S18` | ## Partial Tiles, Guarding, and `.zfill` | P×2 (partial tile, source vs dest tile shape; "For example:") ; C×1 (choreo, 4 lines, "dma.copy.swiz<32>.zfill") ; P×3 (shape mismatch explained, .zfill writes zeros, "That is the right use") ; P×1 ("What happens when the shapes are the same:") ; L×1 (3 items, "- If the source view and destination tile have the same shape, the compiler already guards runtime tails automatically.") ; P×2 (.zfill NOT general mechanism; "Why `.zfill` matters:") ; L×1 (3 items, "- If the consumer reads only the valid region, the compiler's automatic masking is usually enough.") ; P×2 (use .zfill for full dense tile consumer/GEMM pipelines; compiler warns on redundant or missing .zfill) | C | 3.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S19` | ## `.swiz<N>`: Shared-Memory Layout for the Consumer | P×1 (swizzle = how tile lives in shared memory) ; P×1 ("Typical GEMM staging code looks like this:") ; C×1 (choreo, 2 lines, "dma.copy.swiz<128> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;") ; P×2 (load/compute swizzles must agree; MMA chapters go deeper) | C | 3.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F04-S20` | ## DMA vs TMA | P×1 (both move tiles, differ in how/constraints) ; T×1 (headers: (blank)\|`dma`\|`tma`, 5 rows) ; P×1 ("Example TMA use:") ; C×1 (choreo, 2 lines, "f = tma.copy.async lhs.subspan(64, 64).at(block_m, iv_k) => shared;") ; P×1 ("The practical rule is simple:") ; L×1 (2 items, "- Use `dma` when tile sizes are dynamic, irregular, or you want the portable path.") | C | 3.6 | 拆分：仅留语法行/抽象结论，硬件段落移出；保留「动态/不规则形状用 dma」语言规则；Hopper/SM90/带宽表行移出 |
| `F04-S21` | ### The main TMA constraint | P×1 (TMA shapes must be host-computable) ; C×1 (choreo, 1 line, "tma.copy lhs.subspan(64, 64).at(block_m, iv_k) => shared;") ; P×1 (kernel-runtime value cannot define tile shape) ; C×1 (choreo, 2 lines, "// Not valid for TMA if TILE_M is only known inside the kernel") ; P×1 (use dma.copy if runtime-dependent) | C | 3.6 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F04-S22` | ## Writing DMA and TMA That Lower Well | P×1 (few lowering rules matter) | X | X | 整节移出 → 独立硬件文档；(22–24) lowering/性能检查表整组移出 |
| `F04-S23` | ### For high-throughput DMA | L×1 (5 items, "- Prefer `global <-> shared` movement for cooperative staging.") ; P×1 (otherwise compiler falls back to simpler path) | X | X | 整节移出 → 独立硬件文档 |
| `F04-S24` | ### For TMA | L×1 (4 items, "- Use it on SM90+.") | X | X | 整节移出 → 独立硬件文档 |
| `F04-S25` | ## Checklist | P×1 (by end of chapter syntax should mean) ; T×1 (headers: Syntax\|Meaning, 15 rows) ; P×1 (next chapter overlaps movement with compute; double buffering, pipelining) | B | 2.15 | 拆分：移出硬件句，保留 DSL 单元；15 行速查表保留 13 行；2 行硬件描述改写为语法说明 |

### F05 `tutorial/ch03-parallelism.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F05-S01` | # Parallelism: How Croqtile Represents Parallel Work | P×2 (previous chapters used default parallel; GPU has deep execution hierarchy, mapping matters; chapter introduces logical vs physical + space specifiers) | C | 3.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S02` | ## How We Think About Parallel Tasks | P×1 (what "parallel" actually means) ; P×1 (eight tasks: CPU core, 4-core CPU, GPU warp lockstep) ; F×3 (fig1_virtual_parallelism_dark.png#only-dark, fig1_virtual_parallelism_light.png#only-light, *caption* "Same tasks, 1 core/4-wide CPU/8-wide GPU") ; P×3 (parallelism is virtual concept, hardware decides; CUDA `<<<blocks, threads>>>` tangles logical+physical; "Croqtile untangles them.") | C | 3.4 | 拆分：仅留语法行/抽象结论，硬件段落移出；仅保留「parallelism 是虚拟概念」论断；CPU/GPU/warp/CUDA 例子移出 |
| `F05-S03` | ## Croqtile's Two-Layer Parallelism Model | P×1 ("two separate knobs:") ; L×1 (2 items, "1. **`parallel`** — declares that iterations are independent") ; P×1 (foreach is opposite of parallel: sequential) ; C×1 (choreo, 6 lines, "parallel {px, py} by [8, 16] : block    // concurrent — mapped to thread blocks") ; P×1 (parallel lines independent, : block/: thread physical, foreach ordered) ; F×3 (fig2_logical_vs_physical_dark.png#only-dark, fig2_logical_vs_physical_light.png#only-light, *caption* "Logical nesting vs GPU hardware hierarchy") ; P×2 (omitting specifier → compiler default; `{a, b}`, `#`, `#p` supported) | B | 2.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S04` | ## Space Specifiers and Hardware Mapping | P×1 ("A GPU has a clear execution hierarchy. From coarsest to finest:") ; L×1 (4 items, "- A **thread block** (CTA) is a group of up to 1024 threads") ; P×1 ("Croqtile maps to each level with a space specifier:") ; F×3 (fig3_space_specifiers_dark.png#only-dark, fig3_space_specifiers_light.png#only-light, *caption* "Four specifiers, thread counts, typical operations") | X | X | 整节移出 → 独立硬件文档；GPU 执行层级枚举整节移出 |
| `F05-S05` | ### `: block` — Thread Blocks | C×1 (choreo, 1 line, "parallel {px, py} by [8, 16] : block") ; P×2 (128 block grid, each block on an SM with shared memory, blocks independent; use : block for outermost tiling, 128 blocks example) | B | 2.7 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F05-S06` | ### `: thread` — Threads Within a Block | C×1 (choreo, 1 line, "parallel {qx, qy} by [16, 16] : thread") ; P×2 (256 threads within enclosing block, registers vs on-chip memory; finest-grain per-element work) | B | 2.7 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F05-S07` | ### `: group` — Warps | C×1 (choreo, 1 line, "parallel w by 4 : group") ; P×1 (4 warps × 32 threads lockstep; `mma.sync` Ampere-era tensor core at this granularity) | X | X | 整节移出 → 独立硬件文档；`parallel w by 4 : group` 语法行并入 §2.7；warp/张量核说明移出 |
| `F05-S08` | ### `: group-4` — Warpgroups | C×1 (choreo, 1 line, "parallel g by 2 : group-4") ; P×2 (2 warpgroups of 4 warps/128 threads; Hopper cc 9.0+ WGMMA needs full warpgroup, `parallel p by 1 : group-4`; multiple warpgroups share shared memory, independent accumulators, tensor-core ops in Ch4) | C | 3.4 | 拆分：仅留语法行/抽象结论，硬件段落移出；保留 `: group-4` 语法与「合作域」抽象；Hopper/WGMMA 段落移出 |
| `F05-S09` | ### Nesting Specifiers | P×1 ("Real kernels nest these levels. A common pattern:") ; C×1 (choreo, 6 lines, "parallel {px, py} by [8, 16] : block {") ; P×1 (outer block grid, inner threads; `{px, py}` shorthand for two nested parallel lines) | B | 2.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S10` | ### How `shared` Enables Reuse | P×1 (space specifiers interact with memory specifiers; `=> shared` vs `=> local`) ; P×1 (data reuse: tile every thread reads) ; F×3 (fig4_shared_reuse_dark.png#only-dark, fig4_shared_reuse_light.png#only-light, *caption* "local = 4× bandwidth, shared = 1× bandwidth") ; P×2 (rule of thumb: shared vs local, bank contention; matmul K-tiles `=> shared` for 256 threads, accumulator in registers) | X | X | 整节移出 → 独立硬件文档；数据复用/带宽/bank contention 整节移出 |
| `F05-S11` | ## Type System Aside: `mdspan` and `ituple` | P×5 (two type-system features; `mdspan` shape-in-type compile-time verification; `span(i)` vs `lhs.span`; `ituple` `{px, py}`, `px#qx`, `#p`, compile-time; references + shape mismatches caught at compile time) | C | 3.2 | — |
| `F05-S12` | ## Matrix Multiply: Putting It Together | P×2 (tiled matmul within reach, "first program ... GPUs are famous for"; the plan [128,256]×[256,256]→[128,256]) | C | 3.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S13` | ### Scalar Matmul (No DMA) | P×1 (start with parallel + .at(), global memory) ; C×1 (choreo, ~10 lines, "__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {") ; F×3 (fig5_scalar_matmul_dark.png#only-dark, fig5_scalar_matmul_light.png#only-light, *caption* "[128,256] output into 16×64 tile grid") ; P×4 ("This is dense"; output shape from operand dims; multi-axis parallel; named tuple destructuring "The trip counts are:") ; L×1 (3 items, "- `128 / #p = 128 / 16 = 8` rows per tile") ; P×2 (composed global indices `p#m`; the arithmetic dot product reads global memory) | C | 3.4.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S14` | ### DMA Matmul: Tiles in Shared Memory | P×1 (scalar version reads global; K-tiles into shared memory reused by all threads) ; C×1 (choreo, ~17 lines, "__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {") ; F×3 (fig6_dma_matmul_dark.png#only-dark, fig6_dma_matmul_light.png#only-light, *caption* "One block: K-loop DMA into shared, 256 threads") ; P×1 ("What changed from the scalar version:") ; L×1 (4 items, "1. The outer `parallel` now uses `: block` with brace-form indices `{px, py}`.") ; P×2 (composed indices two layers `px#qx`/`py#qy`; **Dimension arithmetic** 16 rows/16 cols, K 16 tiles) | C | 3.6.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S15` | ### GPU Resource Layout | P×1 ("Here is how the matmul code maps to actual GPU hardware:") ; F×3 (fig7_matmul_gpu_layout_dark.png#only-dark, fig7_matmul_gpu_layout_light.png#only-light, *caption* "128 blocks across SMs, shared memory, 256 threads") | X | X | 整节移出 → 独立硬件文档；GPU 资源布局图整节移出 |
| `F05-S16` | ### Host Code | P×1 (host uses make_spandata/.view() pattern) ; C×1 (choreo, ~18 lines, "int main() {") ; P×1 ("Nothing new here. The host program stays boring...") | B | 2.3 | — |
| `F05-S17` | ## Tracing One Output Element | P×3 (pick (row=37,col=50), which block/thread; block (2,3), thread (5,2); K-loop 16 iterations, iteration tile_k=0 loads rows 32..47/K 0..15, foreach k accumulates, same as scalar reference) | C | 3.4.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F05-S18` | ## Summary of New Syntax | T×1 (headers: Syntax\|Meaning, 11 rows) ; P×1 (tiled DMA matmul is backbone; next chapter tensor-core 16×16×16 in one instruction) | B | 2.15 | 拆分：移出硬件句，保留 DSL 单元；11 行语法表保留；4 行 CUDA/warpgroup 映射改写为语法说明 |

### F06 `tutorial/ch04-mma.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F06-S01` | # Tensor Contraction: the `mma` Syntax | P×1 (Ch3 scalar contraction ignores dedicated hardware unit) ; P×1 (2D tensor contraction C += A × B; tensor cores / MXUs / AMX tiles / DSAs, tile sizes 16×16×16, 128×128, 16×64) ; F×3 (fig1_tensor_contraction_dark.png#only-dark, fig1_tensor_contraction_light.png#only-light, *caption* "Same math maps to different hardware") ; P×1 (register layout: wmma::fragment, load_matrix_sync, mma_sync, store_matrix_sync, 32-thread warp fragments) ; F×3 (fig2_register_loading_dark.png#only-dark, fig2_register_loading_light.png#only-light, *caption* "32 threads own scattered register fragments") ; P×1 (Croqtile sidesteps via four abstract operations fill/load/multiply/store) ; F×3 (fig3_mma_syntax_dark.png#only-dark, fig3_mma_syntax_light.png#only-light, *caption* "Four-step MMA is abstract interface, any DSA") | C | 3.7 | 拆分：仅留语法行/抽象结论，硬件段落移出；删除张量核/TPU/AMX/warp 寄存器段落与 3 组图注；保留「四步抽象接口」论断 |
| `F06-S02` | ## The four-step MMA syntax | P×1 ("Every tensor-contraction kernel follows the same rhythm:") ; L×1 (4 items, "1. **`mma.fill 0.0`** — allocate an accumulator tile `mc` in registers") ; P×2 (loop steps 2–3 over K, names mc/ma/mb opaque register tiles; `.row.row` layout contract, row.col/col.row/col.col, wrong variant = correctness bug) | B | 2.12 | 拆分：移出硬件句，保留 DSL 单元 |
| `F06-S03` | ## Scaling the cooperation scope | P×1 ("cooperation scope — one warp, one warpgroup, two warpgroups") | C | 3.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F06-S04` | ### One warp, one tile: the simplest MMA matmul | P×1 (Ampere SM86 MMA scoped to single warp = : group) ; C×1 (choreo, ~32 lines, "#define TILE_M 16") ; P×4 (**`__co__ void` and in-place output**; **Block grid** cdiv; **Warp grid and mma.fill**; **K loop and DMA**) ; A×1 (!!! note "Quick `subspan` syntax reminder" — subspan(h,w)/.at(i,j) vs chunkat reminder) ; P×3 (**Operand loads**; **Store and epilogue**; kernel simple because 32 threads/one warp, wider cooperation window?) | C | 3.7.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F06-S05` | ### Widening the scope: warpgroups and WGMMA | P×1 (Hopper SM90 WGMMA, four warps/128 threads, : group-4) ; F×3 (fig4_sm86_vs_sm90_dark.png#only-dark, fig4_sm86_vs_sm90_light.png#only-light, *caption* "SM86 one warp per MMA, SM90 four warps") ; C×1 (choreo, ~33 lines, "#define TILE_M 128") ; P×5 ("Read it side by side... differences are mechanical:"; **`mma.fill.f16 0.0f`** precision; **`parallel p by 1 : group-4`**; **`chunkat(_, iv_warp)`** `_` = no tiling; same four ops, compiler maps to SM86/SM90) | C | 3.7.2 | 拆分：仅留语法行/抽象结论，硬件段落移出；SM86/SM90 对比图移出；两侧 33 行 kernel 与差异清单全留 |
| `F06-S06` | ### Tiling further: two warpgroups sharing operands | P×1 (two warpgroups share B tile, split A rows) ; C×1 (choreo, ~35 lines, "#define TILE_M 128") ; P×3 (**Splitting M with `p1`**; **Mirrored store**; pattern scales to 3/4 warpgroups, syntax invariant) ; T×1 (headers: Aspect\|One warp(`group`)\|One warpgroup(`group-4`)\|Two warpgroups, 4 rows) | C | 3.7.3 | 拆分：移出硬件句，保留 DSL 单元 |
| `F06-S07` | ## Beyond dense FP16: what else the four steps express | P×2 (four-step extends beyond dense FP16; **Structured sparsity** 2:4 zero pattern, Ampere+, metadata `me`) ; C×1 (choreo, 1 line, "mma.row.row.sp mc, ma, mb, me;") ; P×3 (.sp suffix, `mma.row.col.sp`; **Quantized operands** FP8 `f8_e4m3`/`f8_e5m2` per-tile dequant; "Each result element is scaled...") ; C×1 (choreo, 1 line, "mma.row.row.scale mc, ma, mb, sc_a, sc_b;") ; P×1 (standalone scaling when scale source differs) ; C×1 (choreo, 2 lines, "mma.row.row mc, ma, mb;") ; P×4 (standalone mma.scale in MoE/mixed-precision; **Swizzled loads and transposed stores** mma.load.swiz<N>/mma.store.transp, bank conflicts; **Pipeline synchronization** mma.commit; design principle: suffix = hardware contract, table below) | C | 3.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F06-S08` | ## New syntax | T×1 (headers: Syntax\|Meaning, 14 rows) | C | 3.7.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F06-S09` | ## Chapter summary | T×1 (headers: Topic\|Takeaway, 7 rows) ; P×1 (contraction fast but loads/compute take turns; tensor cores multiply while memory idles; next chapter role specialization) | C | 3.7.10 | 拆分：移出硬件句，保留 DSL 单元 |

### F07 `tutorial/ch05-branch-control.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F07-S01` | # Control Flow: Divergence, Concurrency, and Coordination | P×1 (question: different threads do different things) ; P×1 (CPU program counters vs GPU SIMT warps of 32, serialization, hardware wants uniformity) ; P×1 (CUDA exposes only `if`, predicated execution, warp divergence) ; P×1 (predication insufficient: two structurally different programs, tensor cores) ; P×1 ("Croqtile introduces two additional control-flow primitives") ; L×1 (2 items, "- **`inthreads.async`** — **structured concurrent regions**") ; P×1 (three primitives cover full spectrum of GPU kernel control flow) | C | 3.8 | 拆分：仅留语法行/抽象结论，硬件段落移出；删除 SIMT/warp/CUDA 发散动机段；保留三种控制流原语清单 |
| `F07-S02` | ## Predicated execution with `if` | P×1 ("Croqtile's `if` behaves like its C counterpart:") ; C×1 (choreo, 3 lines, "if (tile_id < total_tiles) {") ; P×3 (all threads evaluate, false skip, warp serializes = warp divergence; **When to use `if`** data-dependent decisions; **Cost model** keep conditions warp-uniform, 32 threads agree) | B | 2.11 | 拆分：移出硬件句，保留 DSL 单元；保留 `if` 示例与「数据相关分支」规则；warp 串行化成本段移出 |
| `F07-S03` | ## Structured concurrent regions with `inthreads.async` | P×1 (solves different problem than `if`, compile-time partition) ; C×1 (choreo, ~11 lines, "parallel p1 by 2 : group-4 {") ; P×1 ("The distinction from `if` is structural, not just performance:") ; T×1 (headers: (blank)\|`if` (predicated execution)\|`inthreads.async` (structured concurrency), 5 rows) ; P×3 (**Why "structured"?** lexical scope, register allocation; **The `.async` modifier** concurrency; "The figure below shows the effect...") ; F×3 (fig1_role_comparison_dark.png#only-dark, fig1_role_comparison_light.png#only-light, *caption* "Top sequential DMA/MMA, Bottom concurrent overlap") | C | 3.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S04` | ### The canonical pattern: 1 producer + 1 consumer | P×1 (most common use is 1P1C split) ; C×1 (choreo, ~13 lines, "parallel p1 by 2 : group-4 {") ; P×4 (**`parallel p1 by 2 : group-4`** two warpgroups 128 threads each; **`inthreads.async (p1 == 0)`**; **`inthreads.async (p1 == 1)`**; "But sharing an address space is exactly what makes this dangerous... events") | C | 3.8.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S05` | ## Inter-region signaling with events | P×1 (concurrent regions need communication, events in shared memory) ; C×1 (choreo, 2 lines, "shared event full;") ; P×1 ("Events have two operations:") ; L×1 (2 items, "- **`trigger name`** — signal that a condition is met") ; P×2 (producer `trigger full`, consumer `wait full`, symmetric `empty`; "This is a **credit-based bounded buffer** protocol — semaphores, TCP window, warp barriers") | C | 3.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S06` | ### Event arrays for multi-stage pipelines | P×1 ("For pipelines with multiple buffered stages, declare event arrays:") ; C×1 (choreo, 1 line, "shared event full[STAGES], empty[STAGES];") ; P×1 (each slot gets pair, `stage = iv_k % STAGES`, four stages run ahead) | B | 2.10 | — |
| `F07-S07` | ### Bootstrap protocol | P×1 (consumer must seed empty credits) ; C×1 (choreo, 3 lines, "foreach {s} in [STAGES] {") ; P×2 (without bootstrap first `wait empty[0]` deadlocks, initial credits; Chapter 6 develops full pipelines with `inthreads.async`, events, `swap`, `mma.commit`) | B | 2.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S08` | ## A 1P1C matmul skeleton | P×1 (three primitives inside Hopper matmul, event sync omitted) ; C×1 (choreo, ~31 lines, "__co__ void matmul(global f16 [M, K] lhs, ...)") ; P×3 (**Producer `foreach`** warpgroup 0 dma.copy; **Consumer `mma` path** warpgroup 1 mc; **Missing coordination** events needed) | C | 3.8.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S09` | ## Persistent scheduling and the `if` guard | P×2 (grid grew with problem, tail underutilization, SMs idle; persistent kernel fixes launch near SM count) ; C×1 (choreo, ~35 lines, "__co__ void matmul(global f16 [M, K] lhs, ...)") ; P×3 (**`parallel block_id by NUM_SMS : block`**; **`tile_id = tile_iter # block_id`**; **`if (tile_id < total_tiles)`** guard) ; F×2 (fig2_persistent_kernel_dark.png#only-dark, fig2_persistent_kernel_light.png#only-light; NO caption) | C | 3.8.3 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S10` | ### Data-dependent vs persistent grids | T×1 (headers: Aspect\|One block per tile\|Persistent (`NUM_SMS` blocks), 4 rows) | C | 3.8.3 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S11` | ## `parallel.async`: host-level concurrency | P×1 (concurrency at host level, launch without blocking CPU) ; C×1 (choreo, 3 lines, "parallel.async {px, py} by [grid_m, grid_n] : block {") ; P×3 (**`parallel.async`** returns immediately, `cudaLaunchKernel` non-default stream; **`parallel.async<s>`** pins to CUDA stream s, overlap if enough SMs; "This is **host orchestration**...") | C | 3.8.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S12` | ## Early return from `parallel` | P×1 ("yield generates a `return;` to get out of device code") ; C×1 (choreo, 6 lines, "parallel p by BLK_CNT: block {") ; A×1 (!!! warning — `yield` does no compile-time/runtime checks, beware synchronization safety) | B | 2.11 | — |
| `F07-S13` | ## New syntax | T×1 (headers: Syntax\|Meaning, 11 rows) | B | 2.15 | 拆分：移出硬件句，保留 DSL 单元 |
| `F07-S14` | ## Chapter summary | T×1 (headers: Concept\|Primitive\|When to use, 5 rows) ; P×1 (1P1C skeleton incomplete without wait/trigger; Chapter 6 adds `swap` and event protocols) | C | 3.8.10 | 拆分：移出硬件句，保留 DSL 单元 |

### F08 `tutorial/ch06-synchronization.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F08-S01` | # Synchronization in Practice: Pipelines, Buffers, and Events | P×3 (Ch5 primitives recap, two progressively complex patterns; `swap` double buffering vs 1P1C event pipeline; both solve "cannot read buffer while someone is writing") | C | 3.8 | — |
| `F08-S02` | ## Double buffering with `swap` | P×2 (two logical buffers, swap handles; Croqtile spells it with dma.copy.async/dma.any/swap/three-phase loop) ; C×1 (choreo, ~38 lines, "__co__ auto matmul(s32 [M, K] lhs, s32 [K, N] rhs) {") | C | 3.6.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S03` | ### The three phases | P×3 (**Prologue** loads tile 0; **Steady state** load next + compute previous + swap; **Epilogue** final compute pass) | C | 3.6.2 | — |
| `F08-S04` | ### `swap`: names, not bytes | P×1 (exchanges future handles, not bytes; CUDA `^ 1` idiom; `rotate(f0, f1, f2)` triple buffering) | C | 3.6.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S05` | ### `with tile_k in 16` | P×1 (scoped region binds tile_k extent 16, chunkat index, `#tile_k` is 16) | B | 2.7 | — |
| `F08-S06` | ### `dma.any`: placeholder futures | P×1 (creates future with no transfer, gives type system something to swap) | C | 3.6.2 | — |
| `F08-S07` | ### `foreach tile_k(1:)`: sliced iteration | P×1 (`(1:)` = indices 1,2,... to end; tile 0 loaded in prologue) | B | 2.7 | — |
| `F08-S08` | ### `__co__ auto` return type | P×1 (compiler infers return type from `return output`) ; P×1 (italic note: s32 scalar accumulation simplification, applies to FP16/MMA kernels Ch4) | B | 2.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S09` | ## Why you cannot "just overlap" without events | P×2 (swap works because one thread group knows order; warp specialization puts them on different warpgroups, needs signaling; picture contrasts staircase with double-buffered overlap) ; F×2 (fig1_pipeline_timeline_dark.png#only-dark, fig1_pipeline_timeline_light.png#only-light; NO caption) | C | 3.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S10` | ## The full 1P1C event pipeline | P×1 (combines inthreads.async + event arrays, coordinated via wait/trigger) ; C×1 (choreo, ~50 lines, "#define TILE_M 128") | C | 3.8.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S11` | ### Walking through the kernel | P×5 (**Ring index** stage = iv_k % STAGES; **Producer path** wait empty/dma.copy/trigger full; **Consumer bootstrap** empty credits before K-loop, deadlock; **Consumer path** wait full/MMA/mma.commit/trigger empty; **`mma.commit`** Hopper WGMMA fence, stale data risk) | C | 3.8.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S12` | ### Credit flow for one stage | P×1 (bootstrap grants empty credits, producer waits empty/fills/signals full, consumer waits full/computes/signals empty, modulo STAGES) ; F×2 (fig2_event_credit_flow_dark.png#only-dark, fig2_event_credit_flow_light.png#only-light; NO caption) | C | 3.8.5 | — |
| `F08-S13` | ### Debugging tip | P×1 (verify event order and trip counts before chasing MMA layout bugs; same cdiv(K,TILE_K) bound, too few stages) | A | 1.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S14` | ## New syntax | T×1 (headers: Syntax\|Meaning, 8 rows) | B | 2.15 | 拆分：移出硬件句，保留 DSL 单元 |
| `F08-S15` | ## Summary | T×1 (headers: Pattern\|Primitives used\|When to use, 2 rows) ; P×2 (both overlap memory and compute, start with swap then events for warp specialization; next chapter hardware-accelerated TMA, swizzled shared layouts, view/from) | C | 3.8.10 | 拆分：移出硬件句，保留 DSL 单元 |

### F09 `tutorial/ch07-advanced-movement.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F09-S01` | # Advanced Data Movement: TMA, Swizzle, and Irregular Access | P×1 (Ch2 dma.copy is software-driven, warpgroup threads issue loads) ; P×1 (Hopper SM90+ TMA unit near L2/shared memory, descriptor-based instruction) ; P×1 (TMA not faster than DMA: same HBM→L2→shared bandwidth; dedicated engine enables overlap; warp-specialized pipeline) ; P×1 ("The two paths differ in interface, not in throughput:") ; L×1 (2 items, "- **`dma.copy`** — Threads cooperatively issue loads...") ; P×1 (Croqtile exposes TMA via same arrow syntax) | X | X | 整节移出 → 独立硬件文档；TMA 单元/L2/HBM/带宽整节移出；仅保留「同一箭头语法」一句入 §3.6 |
| `F09-S02` | ## `tma.copy`: hardware tensor movement | P×1 ("The surface syntax mirrors `dma.copy`:") ; C×1 (choreo, 1 line, "tma.copy lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;") ; P×1 ("Same source expression... difference is **who does the work**:") ; F×2 (fig3_tma_descriptor_dark.png#only-dark, fig3_tma_descriptor_light.png#only-light; NO caption) ; L×1 (2 items, "- **DMA path.** Threads cooperate to cover the tile") ; P×1 (**What this buys you** whole-tile sync, drop per-thread load choreography, compiler builds descriptor) ; F×2 (fig1_tma_vs_dma_dark.png#only-dark, fig1_tma_vs_dma_light.png#only-light; NO caption) | C | 3.6 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F09-S03` | ## Swizzle and bank conflicts | P×1 (32 banks, 4 bytes/bank, warp lanes, bank conflict, bandwidth) ; P×1 (**Swizzle** XOR remapping, exposed on both DMA and TMA) ; C×1 (choreo, 2 lines, "dma.copy.swiz<3> src => dst;") ; P×1 (MMA read path must use same `swiz<N>`) ; C×1 (choreo, 1 line, "ma = mma.load.swiz<3> lhs_load_s.chunkat(_, iv_warp);") ; P×3 (swizzle not TMA-specific, identical layouts; **Swizzle levels** swiz<0> identity, 64B/128B/256B; **Matching rule** compiler does not enforce pairing) ; F×2 (fig2_swizzle_dark.png#only-dark, fig2_swizzle_light.png#only-light; NO caption) | C | 3.7 | 拆分：仅留语法行/抽象结论，硬件段落移出；保留 `swiz<N>` 两条语法与「生产者/消费者必须一致」规则；32-bank 模型移出 |
| `F09-S04` | ## Why the restricted interface works: expressiveness vs performance | P×1 (raw CUDA freedom, coalesced loads, conflict-free shared access, 2~32× throughput loss) ; P×1 (Croqtile restricts to performance sweet spot, compiler handles coalescing/swizzle) ; F×2 (fig8_expressiveness_dark.png#only-dark, fig8_expressiveness_light.png#only-light; NO caption) ; P×1 (TMA descriptor = extreme instance; `dma.copy` generates only coalesced/conflict-free patterns; `LDG`/`STS` trade) | C | 3.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S05` | ## Example: TMA in a pipelined matmul | P×1 (Chapter 6 skeleton unchanged, producer swaps dma.copy→tma.copy.swiz<3>, consumer mma.load→mma.load.swiz<3>) ; C×1 (choreo, ~41 lines, "__co__ void matmul(global f16 [M, K] lhs, ...)") ; P×1 (only ingress/operand lines change; writeback still `dma.copy`) | C | 3.6.3 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S06` | ## Handling irregular access | P×2 (chunkat/subspan assume even division; MoE dynamic offsets, convolution overlap, partial K-tile, flat buffers; Croqtile provides four tools, DMA/TMA/MMA/events/swizzle unchanged) | C | 3.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S07` | ### Arbitrary-offset windows: `view` and `from` | P×2 (chunkat regular grid vs MoE runtime offsets; `view(M, N).from(row, col)` defines M×N rectangle) ; C×1 (choreo, 1 line, "patch = matrix.view(16, 16).from(37, 50);") ; P×1 ([16,16] slice at row 37 col 50, runtime origin) ; F×2 (fig4_view_from_dark.png#only-dark, fig4_view_from_light.png#only-light; NO caption) ; P×1 (composability: subspan/chunkat/dma.copy/tma.copy apply to window) ; C×1 (choreo, 2 lines, "expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);") ; P×2 (MoE expert_offset rewires operand, pipeline unchanged; **When to use it** chunkat vs view/from) | C | 3.5.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S08` | ### Strided tiles: `.subspan`, `.step`, and `.at` | P×2 (subspan(M,K).at(i,j) packed tiles; `.step(sM, sK)` overrides stride) ; C×1 (choreo, 1 line, "matrix.subspan(16, 16).step(32, 32).at(i, j);") ; F×2 (fig5_subspan_step_dark.png#only-dark, fig5_subspan_step_light.png#only-light; NO caption) ; P×2 (tile (1,0) starts at row 32, 16-row gap, packed default; **When you need this** padding/guard bands, overlapping stencils, metadata-interleaved layouts) | C | 3.5.2 | — |
| `F09-S09` | ### Zero-padding: `.zfill` | P×2 (partial tile edge case, undefined behavior on GPU; brute-force special code path breaks MMA uniformity, .zfill elegant) ; C×1 (choreo, 2 lines, "tma.copy.swiz<3> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill") ; P×1 (out-of-range elements zero, zeros contribute nothing, MMA loop uniform) ; F×2 (fig6_zfill_dark.png#only-dark, fig6_zfill_light.png#only-light; NO caption) | C | 3.5.3 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S10` | ### Layout reinterpretation: `span_as` | P×1 (flat 1D strip to 2D MMA matrix, `span_as` reinterprets in place) ; C×1 (choreo, 1 line, "flat_buffer.span_as([rows, cols])") ; P×1 (element count preserved, rank changes, no data moved) ; C×1 (choreo, 3 lines, "strip_load = dma.copy data.chunkat(tile) => shared;") ; P×1 (1D strip as 2D for chunkat/MMA, rows*cols invariant checked at compile time) ; F×2 (fig7_span_as_dark.png#only-dark, fig7_span_as_light.png#only-light; NO caption) | C | 3.5.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S11` | ## Chapter summary | T×1 (headers: Concept\|Syntax\|Role, 8 rows) | C | 3.5.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F09-S12` | ## New syntax | T×1 (headers: Syntax\|Meaning, 8 rows) ; P×1 (next chapter: C++ interop, `__device__`, register hints, preprocessor guards, inline PTX) | C | 3.5.10 | 拆分：移出硬件句，保留 DSL 单元 |

### F10 `tutorial/ch08-cpp-interop.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F10-S01` | # C++ Interop: Device Functions, Inline Code, and the Preprocessor | P×1 (every language needs escape hatch: ctypes, unsafe/extern "C", JNI) ; P×1 ("Croqtile's interoperability story has three parts:") ; L×1 (3 items, "1. **`__device__` functions** — Standard CUDA device functions") ; P×1 (built stack of abstractions; chapter is about stepping outside) | C | 3.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S02` | ## How `.co` files compile | P×1 (.co mixes three kinds of code, skeleton) ; C×1 (choreo, ~16 lines, "__device__ __forceinline__ float fast_rsqrt(float x) {   // ①") ; P×1 (compiler splits into single .cu, hands to nvcc) ; F×2 (fig1_compilation_flow_dark.png#only-dark, fig1_compilation_flow_light.png#only-light; NO caption) ; P×1 ("What happens to each region:") ; L×1 (4 items, "- **① `__device__` functions** are copied into the generated `.cu` file **verbatim**") ; P×1 (all four regions merge, nvcc compiles device/host, `croq kernel.co -v` shows command) | A | 1.4 | 拆分：移出硬件句，保留 DSL 单元；保留 .co 三种代码骨架与编译流水线；nvcc/wgmma/TMA 描述改为「目标代码」 |
| `F10-S03` | ## `__device__` vs ordinary C++ inline | P×4 (both passed through but different compilation paths; `__device__` → GPU machine code, CUDA intrinsics `__shfl_xor_sync`/`__frsqrt_rn`/`atomicAdd`, `__shared__`, `call` keyword; ordinary C++ → CPU host side, cannot call from `__co__`; strict boundary, `__co__` is bridge) | C | 3.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S04` | ## `__device__` functions: CUDA alongside Croqtile | P×1 (helper at warp/thread level → standard CUDA `__device__`) ; P×1 ("`__co__` can call a `__device__` function using the `call` keyword:") ; C×1 (choreo, ~37 lines, "template <int K>") ; P×5 (**`__device__ void warp_topk<K>`**; **`__device__ T SHFL_XOR`**; **`call warp_topk<8>(...)`**; **`inthreads (tid == 0)`** note inthreads without .async; "This pattern — Croqtile handles parallelism... MoE top-K") | C | 3.9.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S05` | ## `__cpp__`: verbatim C++ injection | P×1 (`__cpp__` pastes string literal, must be valid at splice point) ; P×1 ("**Two forms:**") ; L×1 (2 items, "- **`__cpp__(\"...\")`** — Ordinary string; best for short one-liners.") | B | 2.13 | — |
| `F10-S06` | ### Register hints: `setmaxnreg` | P×1 (canonical use: register redistribution in warp-specialized pipelines, PTX setmaxnreg) ; C×1 (choreo, ~15 lines, "parallel p1 by 2 : group-4 {") ; P×1 (**Placement** put hint at top of inthreads.async branch) | C | 3.9.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S07` | ### Macros inside `__cpp__` strings | P×1 (common mistake: macros expand inside string literals) ; C×1 (choreo, 6 lines, "#define PRODUCER_MAXNREG 40") ; P×1 (fails; use numeric literals + `#if`/`#error` outside) ; C×1 (choreo, 9 lines, "#define PRODUCER_MAXNREG 40") | B | 2.13 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S08` | ## The preprocessor | P×1 (preprocessor runs before main pass, macros expand in `__co__` and host) ; T×1 (headers: Directive\|Role, 4 rows) ; P×1 (function-like macros not supported, use `constexpr` helpers) | B | 2.13 | — |
| `F10-S09` | ### Tile geometry as macros | P×1 (production matmul centralizes tile dims at top) ; C×1 (choreo, 5 lines, "#define MATMUL_WARP_M 64") ; P×1 (same names in parallel, shared-memory declarations, foreach bounds, host verification) | B | 2.13 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S10` | ### Compile-time assertions with `#error` | C×1 (choreo, 8 lines, "#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)") ; P×1 ("Treat these guards as documentation of hardware constraints.") | B | 2.13 | 拆分：移出硬件句，保留 DSL 单元 |
| `F10-S11` | ### Conditional code paths | C×1 (choreo, ~10 lines, "#define PATH0") ; P×1 (preprocessor keeps one branch; command-line defines `croq kernel.co -DMATMUL_TILE_K=128`) | B | 2.13 | — |
| `F10-S12` | ## Reading a production `.co` file | P×1 ("When you open a benchmark kernel, read top down:") ; L×1 (5 items, "1. **Macros and `#error` guards** — The contract: allowed tile sizes") | A | 1.4 | 拆分：移出硬件句，保留 DSL 单元；5 项阅读清单保留；architecture flags / PTX 字样去掉 |
| `F10-S13` | ## Chapter summary | T×1 (headers: Syntax\|What it does\|Runs on, 8 rows) ; P×1 (next chapter: debugging, verbose modes, systematic narrowing) | C | 3.9.10 | 拆分：移出硬件句，保留 DSL 单元 |

### F11 `tutorial/ch09-debug-verbose.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F11-S01` | # Debug and Verbose: Printing, Assertions, and GDB | P×3 (GPU kernels are opaque systems, thousands of threads, shared memory invisible; universal discipline = systematically narrow the search space; Croqtile tools at each level: `print!`/`println!`, `assert`, guarded `println`, debug RTTI for `cuda-gdb`, `-v`) | C | 3.11 | 拆分：移出硬件句，保留 DSL 单元 |
| `F11-S02` | ## `print!` and `println!`: compile-time inspection | P×1 (bang variants run at compile time) ; C×1 (choreo, 4 lines, "__co__ void check_shapes(f32 [3, 2] b) {") ; P×1 ("When you compile this file, the compiler emits:") ; C×1 (plain/text, 1 line, "shape of b: b.span = [3, 2]") ; P×1 (no GPU needed, `"wor" "ld!"` concatenation, verify chunkat/subspan tile sizes) | B | 2.14 | 拆分：移出硬件句，保留 DSL 单元 |
| `F11-S03` | ## `print` and `println`: runtime device output | P×1 (no bang → device-side `printf` in generated CUDA) ; C×1 (choreo, 5 lines, "__co__ void inspect(s32 [4, 8] data) {") ; P×1 (comma-separated mix, nondeterministic order, GPU printf buffers) ; P×1 ("**Guarding output.** For a specific check, guard the print with a condition:") ; C×1 (choreo, 7 lines, "parallel {px, py} by [8, 16] : block") ; P×1 ("Without the guard you get one line per thread") | B | 2.14 | 拆分：移出硬件句，保留 DSL 单元 |
| `F11-S04` | ## `assert`: runtime invariant checks | P×1 (`assert` checks invariant, aborts kernel with message) ; C×1 (choreo, 1 line, "assert(stage < MATMUL_STAGES, \"stage index out of bounds\");") ; P×1 (compiles to printf + abort; index-out-of-range, null pointer) | B | 2.14 | — |
| `F11-S05` | ## Verbose mode: `-v` | P×1 (`-v`/`--verbose` shows external commands, `nvcc` invocation) ; C×1 (bash, 1 line, "croq kernel.co -v -o kernel") ; P×1 (useful for wrong flags to nvcc, generated file paths) | A | 1.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F11-S06` | ## Runtime checks: `-rtc` | P×1 (graduated runtime checking with `-rtc`/`--runtime-check`) ; C×1 (bash, 1 line, "croq kernel.co -rtc=high -o kernel") ; P×1 (levels none/entry/low/medium/high/all, superset, high/all dev, none production) | A | 1.5 | — |
| `F11-S07` | ## Debugging MMA-heavy kernels | P×2 (suspect layout first row/col-major `[N, K]` vs `[K, N]`, then indexing `.at`/`chunkat`, then async ordering; common mistake mislabeling `mma.row.row`, swizzled loads mismatch = regular error pattern every sixteenth element) | A | 1.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F11-S08` | ## Debug RTTI and `cuda-gdb` | P×1 (debug RTTI makes types visible to cuda-gdb) ; P×1 ("Compile with `-g` to enable debug symbols:") ; C×1 (bash, 1 line, "croq kernel.co -g -o kernel_debug") ; P×1 ("The generated code includes RTTI structs for Croqtile types:") ; T×1 (headers: Croqtile type\|GDB type\|Fields, 4 rows) ; P×1 ("**Example session:**") ; C×1 (bash, ~11 lines, "cuda-gdb -q ./kernel_debug") ; P×1 (`__dbg_` prefix generated by compiler, makes variables visible) | A | 1.5 | 拆分：移出硬件句，保留 DSL 单元；`cuda-gdb` 统一为 debugger；RTTI 表与会话代码块全留 |
| `F11-S09` | ## Putting it together | F×2 (fig1_debug_workflow_dark.png#only-dark, fig1_debug_workflow_light.png#only-light; NO caption) ; P×1 (cheap→expensive: `println!` shapes → guarded `println` one tile → event/stage prints → pattern analysis layout → `cuda-gdb`; guard prints behind `#ifdef`) ; C×1 (choreo, 3 lines, "#ifdef DEBUG_PRINT") ; C×1 (bash, 2 lines, "croq kernel.co -DDEBUG_PRINT -o kernel    # debug build") | A | 1.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F11-S10` | ## Summary | T×1 (headers: Tool\|Level\|Cost, 6 rows) ; P×1 ("You have learned the following concepts:") ; L×1 (9 items, "- simple element-wise addition in Croqtile (Chapter 1)") ; P×1 (next step: open benchmark kernel in `croqtile/benchmark/`, change one constant, rebuild, measure) ; F×n 仅出现在 choreo-program-structure.md（共 2 张图）； ; A×n 共 5 处：2 个 **Note:** 加粗提示、2 个 !INCLUDE_IF_EXISTS 指令、1 个 HTML 注释块。 | A | 1.5 | 拆分：移出硬件句，保留 DSL 单元 |

### F12 `documentation/index.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F12-S01` | [# Croqtile Coding Reference] | P×1: welcome and DSL overview | A | 1.1 | — |
| `F12-S02` | [## What You Will Find Here] | P×1: coverage statement, no detail ; L×1: (7, "**Program structure**: ... `__co__` function, host-side APIs") | C | 3.1 | 拆分：移出硬件句，保留 DSL 单元 |

### F13 `documentation/what-is-choreo.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F13-S01` | [# What is Good in CroqTile?] | P×2: GPU/DSA kernels, transpiles to CUDA; four design pillars intro | A | 1.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F13-S02` | [## Easy to Use] | P×1: tensors vs raw buffers, `dma.copy` | C | 3.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F13-S03` | [## Compile-Time Safety] | P×1: 353 compile-time checks, 1,319 assertions | C | 3.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F13-S04` | [## Dynamic Shapes] | P×1: symbolic dimensions, no recompilation | C | 3.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F13-S05` | [## Born for AI Tuning] | P×1: compact syntax, AI agents, TFLOPS | C | 3.10 | 拆分：移出硬件句，保留 DSL 单元 |

### F14 `documentation/choreo-program-structure.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F14-S01` | [## Overview] | P×1: introduces structure, terminology, example | C | 3.1 | — |
| `F14-S02` | [## Construct a Choreo-C++ Program] | P×1: three parts, heterogeneous platform ; L×1: (3, "The Device Program") | C | 3.1 | — |
| `F14-S03` | [### Host, Device and Tileflow] | P×9: CUDA/Cute example intro; review each part; Host Program label; host program entry point; host data preparation; Device Program label; device kernel `__device__` keyword; Tileflow Program label; tileflow orchestrates data movement ; C×1: (choreo, ~20, "// Device program: typically runs on GPU/NPU") | B | 2.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F14-S04` | [### Transpilation and Compilation] | P×6: three compilation steps, figure; transpilation leaves host/device unchanged; end-to-end compiler, key step; two compilation models intro; separate model needs `__cok__`; OpenCL separate file, `__cok__` needed ; F×1: (assets/figures/compile-workflow.drawio.png, "Choreo-Workflow" = 编译流程图) ; L×1: (2, "**Single Source Compilation Model**: Similar to CUDA/Cute") ; C×1: (choreo, ~9, "__cok__ {") | C | 3.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F14-S05` | [## A Full Choreo-Cute C++ Code Example] | P×1: complete element-wise addition example ; C×1: (choreo, ~56, "// Device Program") ; P×1: subsequent sections explain parts | A | 1.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F14-S06` | [### Host Program - the Control Center] | P×8: host program entry, code repeat; main uses make_spanview API; make_spanview declaration intro; usage example intro; spanned data shape safety; ele_add call, result verify; spanned_data owns buffer; row-major ordering note ; C×3: (choreo, ~26, "int main() {"); (cpp, ~4, "template <size_t Rank, typename T>"); (cpp, ~3, "choreo::make_spanview<3>(&a[0][0][0], {6, 17, 128});") ; A×1: "**Note:** The most significant dimension value comes first in the `initializer_list` shape..." | A | 1.4 | — |
| `F14-S07` | [### Device Program: Parallel Computation] | P×6: device kernel on GPU; code repeat intro; `__cok__` for separated model; extern C replaces `__device__`; target hardware, intrinsics; SPMD paradigm, tileflow moves data ; C×2: (choreo, ~5, "__device__ void kernel(int * a, int * b, int * c, int n) {"); (choreo, ~7, "__cok__ {") | C | 3.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F14-S08` | [### Tileflow Program: Orchestrating the Data Movement] | P×11: tileflow manages host-device movement; code revisit intro; `__co__` inputs, s32 shapes; parallel-by analogous kernel launch; with-in, bounded-ituple; foreach equivalent C loops; dma.copy statement intro; figure intro; chunks host to local memory; local buffer, call kernel, DMA back; 17x4 iterations, 6 threads ; C×2: (choreo, ~24, "__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs, s32 [6, 17, 128] rhs) {"); (cpp, ~4, "for (int x = 0; x < 17; x++)  // assume 'x' represents the 1st element of 'index'") ; L×1: (4, "The symbol `lhs_load` is called **future** of the DMA operation") ; F×1: (assets/figures/chunkat-6-17-4.drawio.png, "Choreo-Structure-Chunkat" = chunkat 6/17/4 切块示意) | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F14-S09` | [## Quick Summary] | P×2: three parts, call chain; DMA abstraction, tiling, next chapters | C | 3.1 | — |

### F15 `documentation/syntax-design-of-choreo.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F15-S01` | [# Syntax Design of Choreo] | —（分组/占位标题，无直属正文） | C | 3.2 | — |
| `F15-S02` | [## Introduction] | P×1: eDSL for accelerator data movement | C | 3.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F15-S03` | [## Embedding Choreo within C++] | P×3: DSL embedded, includes choreo.h; `__co__` prefix, translation scope; deferred type details ; C×1: (plain 无语言标注, ~13, "// some C++ code") | C | 3.2 | — |
| `F15-S04` | [## Variables and Types] | P×2: four type categories; domain-specific spanned, bounded ; L×1: (4, "**Scalar Type**. It is designed to fullfil the requirement of program control.") | C | 3.2 | — |
| `F15-S05` | [### Scalar Types] | P×4: integer 32-bit, `$-2^{31}$` range; integer operations like C++; no unsigned/narrow ints; boolean type like C++ ; C×1: (plain, ~4, "int a;") | B | 2.5 | — |
| `F15-S06` | [### Spanned Types] | P×3: composite of fundamental, mdspan; ML tiling, storage hierarchy; following sections preview | C | 3.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F15-S07` | [#### The Partial Type: **mdspan**] | P×11: mdspan partial type rationale; compile-time mdspan type checking; mdspan bracket syntax; explicit or deduced, dependent type; mdspan as multi-level ranges; element-of () operation; sp(0) integer arithmetic; derive mdspan from existing; tiling, syntax sugar; minimal code philosophy; other methods with i-tuple ; C×4: (plain, ~5, "mdspan sp : [7, 8];           // defined a mdspan of 2-dimensions"); (plain, ~4, "sp : [7, 8];"); (plain, ~4, "sp : [6, 8];"); (plain, ~4, "sp : [6, 8];") ; A×1: "**Note, a mdspan can only be defined. No modification to an existing mdspan is allowed...**" | C | 3.2 | — |
| `F15-S08` | [#### Fully-Typing] | P×3: fully-typed needs fundamental and mdspan; fundamental types list intro; s32 vs int distinction ; C×1: (plain, ~5, "ndims : [20, 15];") ; L×1: (3, "Unsigned Integers: *u8/u16/u32*") | B | 2.6 | — |
| `F15-S09` | [#### The Storage Qualifier] | P×4: move across memory hierarchy; three storage qualifiers intro; usage code intro; default global memory ; L×1: (3, "**global**,") ; C×1: (plain, ~5, "ndims : [20, 15];") | B | 2.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F15-S10` | [### I-Tuple Types] | P×2: ituple unordered integer set; brace definition syntax ; C×1: (plain, ~4, "ituple index = {5, 4, 3, 2, 1};  // It defines a tuple of 5 elements") | B | 2.5 | — |
| `F15-S11` | [### Operations over *mdspan* and *i-tuple*] | P×3: tuple-span operation example intro; supported operations intro; readability rationale ; C×1: (plain, ~6, "sp : [6, 8];") ; L×1: (5, "*mdspan* $/$ *i-tuple*") | B | 2.5 | — |
| `F15-S12` | [### Bounded Types] | P×2: bounded scalar [0, ub); bounded ituple with mdspan | C | 3.2 | — |
| `F15-S13` | [## Control Structures] | P×1: differs from C++ control ; A×1: HTML 注释块（"Choreo follows C++ to involve 'if-else' blocks... difference on parallelization, loop"） | C | 3.4 | — |
| `F15-S14` | [### Parallel Region: the 'parallel-by' Block] | P×4: SPMD model vs CPU threads; C-style parallel-by block; 6 threads, p as thread index; bounded integer, chunkat needs bound ; C×1: (plain, ~5, "parallel p by 6 {") | B | 2.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F15-S15` | [### The 'with-in' Block and 'where' Clause] | P×7: with-in binds ituple to mdspan; naming elements syntax; name tuple and elements; bounded ituple naming; sequential, no parallelism; where clause intro; `<->` equality relation, AI kernels ; C×4: (plain, ~5, "with index in [10, 10] {"); (plain, ~3, "with {x, y} in [10, 10] {...}"); (plain, ~3, "with index = {x, y} in [10, 10] {...}"); (plain, ~5, "with {m, n} in [M, N], {n_p, k} in [N_P, K] where n_p <-> n {") | B | 2.7 | — |
| `F15-S16` | [### The 'foreach' Block] | P×1: foreach loops bounded ituple ; C×1: (plain, ~7, "with x in [10] {") | B | 2.7 | — |
| `F15-S17` | [### The 'upper-bound' Operation] | —（分组/占位标题，无直属正文） | C | 3.2 | —；上游空标题（The upper-bound Operation）：登记但不填正文 |
| `F15-S18` | [### Async Operation: the DMA Statement] | P×7: only async form, DMA; async like CPU thread, DTE config; basic DMA example intro; DTE, future, wait; simpler DMA syntax intro; compiler allocates local, f.data; static/runtime checks, DMA manual ; C×2: (plain, ~7, "global f32 [10] g_data;"); (plain, ~5, "global f32 [10] data;") | C | 3.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F15-S19` | [### Bounded-ituple/integer and 'chunkat' Operation] | P×2: chunkat creates subview mdspan; divide into 6*10*10 chunks ; C×1: (plain, ~12, "global f32 [6, 10, 100] data;") | B | 2.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F15-S20` | [## Function Calls and Call Choreo Function] | P×2: no nested choreo calls; call keyword invokes C++ ; C×1: (plain, ~9, "void bar() {...}    // C++ kernel function") | B | 2.13 | — |
| `F15-S21` | [## Parameters Passing between Choreo and C++ Function] | P×3: choreo.h, wrap pointer with dims; make_spanview, \|d\| size; scalar passable, ituple internal ; C×1: (plain, ~15, "#include \"choreo.h\"") | B | 2.3 | — |
| `F15-S22` | [## Summary] | P×2: SPMD, dataflow DSL summary; closing, feedback request | C | 3.2 | 拆分：移出硬件句，保留 DSL 单元 |

### F16 `documentation/shape-in-choreo.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F16-S01` | [## Overview] | P×1: shape first-class, learn shapes | B | 2.4 | — |
| `F16-S02` | [## First-Class Citizen: Shape] | P×2: data movements, ML/HPC datasets; all data must have shape | C | 3.3 | — |
| `F16-S03` | [## Defining Shapes with `mdspan`] | P×7: mdspan keyword, Multi-Dimensional Span; explicit definition intro; initialization expression, `:` symbol; s0 2D, s1 1D; optional rank, consistency check; type inference possible; initialized in declaration, inference preferred ; C×3: (choreo, ~4, "mdspan s0 : [7, 8]; // Defines a 2D shape with dimensions [7, 8]"); (choreo, ~3, "mdspan<3> s2 : [64, 32]; // error: the rank of mdspan is inconsistent"); (choreo, ~3, "s3 : [7, 8, 9]") | B | 2.4 | — |
| `F16-S04` | [## Deriving *mdspan*s] | P×16: derive shapes, tiling, padding; arithmetic derivation example intro; new-shape0 tiling and reshaping; equivalent full form intro; element-of (), syntactic sugar; new-shape1 pads and swaps; whole-mdspan derivation intro; arithmetic applied dimensionally; `shape + 1` inline statement; equivalence statement; expanded `new-shape1` inline code; concatenation behavior explained; `[shape, 6]` inline statement; equivalence statement; expanded `new-shape3` inline code; prior version syntactic sugar ; C×3: (choreo, ~5, "shape : [128, 64]; // initial shape "); (plain, ~3, "new-shape0: [shape(0) / 2, shape(1) / 4, 1];"); (choreo, ~7, "shape : [32, 72]") | B | 2.4 | — |
| `F16-S05` | [## Evaluation of *mdspan*] | P×2: compile-time eval, fixed shapes; runtime shapes, symbolic dimensions, host code | C | 3.3 | — |

### F17 `documentation/integer-ituple.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F17-S01` | [## Overview] | P×1: integers and i-tuples intro | B | 2.5 | — |
| `F17-S02` | [## Integers for Program Control] | P×7: integers for control, 32-bit signed; C-like integer declaration; type inference style; multiple declarations; integers passable to device; no uninitialized or reassignment; immutable, constant ; C×5: (choreo, ~3, "int a = 1;"); (choreo, ~3, "a = 1;"); (choreo, ~4, "int a = 1, b = 2, c = 3;"); (choreo, ~6, "__co__ void foo(int a) {"); (choreo, ~5, "int a;  // error: declaration without initialization") | B | 2.5 | — |
| `F17-S03` | [## Group of Ranges and Group of Integers] | P×3: mdspan ranges not integers; `[7,14]` means two ranges; motivation for ituple ; L×1: (2, "When considering the delta of two mdspans, multiple integers are needed") | C | 3.2 | — |
| `F17-S04` | [## Group Integers as Integer-Tuple (I-Tuple)] | —（分组/占位标题，无直属正文） | B | 2.5 | — |
| `F17-S05` | [### Defining I-Tuples] | P×3: ituple keyword, definition; type inference, distinguish from mdspan; rank check intro ; C×2: (plain, ~4, "ituple a = {1, 2, 3};"); (plain, ~3, "ituple<3> a = {1, 2};  // error: inconsistent rank") ; L×1: (2, "The *initialization expression* of an *ituple* follows an assignment operation `=`") | B | 2.5 | — |
| `F17-S06` | [### Operations on I-Tuples] | P×4: element-of () and whole ituple; ituple immutable; ituple with mdspan example; rank consistency rule ; C×3: (choreo, ~7, "a = {3, 4};"); (choreo, ~6, "shape : [7, 18, 28];"); (choreo, ~3, "shape : [7, 8, 9] + {1, 2}; // error: inconsistent rank") | B | 2.5 | — |
| `F17-S07` | [### Evaluation of Integers and I-Tuples] | P×1: compile-time eval, minimal cost | C | 3.3 | — |
| `F17-S08` | [### Look-Ahead: Bounded I-Tuples] | P×1: bounded i-tuples for loops | C | 3.2 | — |
| `F17-S09` | [## Quick Summary] | P×2: integer/ituple syntax recap; `=` is initialization, immutability | B | 2.5 | — |

### F18 `documentation/spanned-data.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F18-S01` | [## Overview] | P×1: declaring data/buffers are fundamental | B | 2.6 | — |
| `F18-S02` | [## *Spanned*: Data and Buffers] | P×1: input/output vs buffers, spanned | C | 3.2 | — |
| `F18-S03` | [## Defining a *Spanned Data*] | P×7: element type plus shape; f32 IEEE-754, named mdspan; element-of, arith, concatenation; fundamental type list intro; prefix/suffix naming, bf16 ML; s32 vs int; no conversion, compile-time failure ; C×3: (plain, ~3, "f32 [32, 16] data;"); (plain, ~7, "shape : [72, 14];"); (choreo, ~3, "s32 a;  // error: the fundamental type can not be used for variable definition alone") ; L×1: (3, "**Unsigned Integers**: `u8`, `u16`, `u32`") | B | 2.6 | — |
| `F18-S04` | [## Storage Specifier] | P×3: storage specifier required; heterogenous, CPU default, Cuda/Cute GPU; target-specific limitations ; C×1: (choreo, ~5, "global f32 [32, 7, 2] a;") ; L×1: (3, "*global*: Refers to the device's global storage") | B | 2.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F18-S05` | [## Initialization] | P×2: declaration-site initialization; fixed value, limited functionality ; C×1: (choreo, ~4, "local s32 [17, 6] b1 {0};       // elements are initialized to 0") | B | 2.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F18-S06` | [## Declaring the Parameters and Return Values] | P×3: parameters between host and tileflow; no storage specifier, no ops; `.span` member function ; C×2: (choreo, ~3, "__co__ f16 [7, 8] foo(f32 [16, 17, 5] input) {...}"); (choreo, ~6, "__co__ auto foo(f32 [16, 17, 5] input) {") | B | 2.6 | — |
| `F18-S07` | [## Buffer Lifetime Management] | P×1: compiler reuses buffers, no management | C | 3.2 | — |
| `F18-S08` | [## Quick Summary] | P×1: recap, next dynamic shapes | B | 2.6 | — |

### F19 `documentation/symbolic-shape.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F19-S01` | [## Overview] | P×1: anonymous and symbolic dimensions intro（原文有 "n this section" 拼写残误，保留） | C | 3.3 | — |
| `F19-S02` | [## Fixed and Dynamic Shape] | P×1: fixed shapes, dynamic runtime shapes | C | 3.3 | — |
| `F19-S03` | [## *mdspan* with **Anonymous Dimension**] | P×4: anonymous dimension simplest method; `?` unknown, runtime value; runtime from config/framework; anonymous insufficient sometimes ; C×2: (choreo, ~3, "__co__ auto foo(s32 [?, 1, 2] input) { ... }"); (cpp, ~5, "void bar(int * data, int dim0) {") | B | 2.4 | — |
| `F19-S04` | [## Step Further: **Symbolic Dimension**] | P×2: named symbolic dimensions intro; shape relationships, safety ; C×1: (choreo, ~10, "__co__ auto Matmul(s32 [M, K] lhs, s32 [K, N] rhs) {") | B | 2.4 | — |
| `F19-S05` | [## Improved Code Safety] | P×6: more checks, anonymous Matmul; checks intro; choreo_abort early check; additional checks intro; dimension consistency verified; recommend symbolic dimensions ; C×3: (choreo, ~10, "__co__ auto Matmul(s32 [?, ?] lhs, s32 [?, ?] rhs) {"); (cpp, ~9, "void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,"); (cpp, ~13, "void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,") ; L×1: (2, "**Compile-time Checks**: Choreo verifies rank consistency") | C | 3.3 | — |
| `F19-S06` | [## Cost and Limitations] | P×2: conservative compile, runtime checks; only parameter lists ; C×1: (choreo, ~3, "__co__ void foo() { f32 [M] d;} // compile-time error: can not applied to the spanned buffer") | C | 3.3 | — |
| `F19-S07` | [## Quick Summary] | P×1: anonymous vs symbolic recap | C | 3.3 | — |

### F20 `documentation/why-symbolic-shapes.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F20-S01` | [# **Why Symbolic Dimensions are Better Than `?` or `-1`**] | —（分组/占位标题，无直属正文） | C | 3.3 | — |
| `F20-S02` | [### **a. Improved Readability and Clarity**] | P×3: readability, maintainability, cognitive load; `?` ambiguity example; `?` version ambiguous ; C×1: (choreo, ~9, "// Using symbolic dimensions ") | C | 3.3 | — |
| `F20-S03` | [### **b. Type Safety and Compile-Time Checking**] | P×3: type-safe, algebraic manipulation; runtime check overhead; type consistency benefit ; C×1: (choreo, ~7, "mdspan sp : [M, N]; ") | C | 3.3 | — |
| `F20-S04` | [### **c. Expressiveness and Flexibility**] | P×4: richer shape expressions; `?` brittle, runtime inference; relationship example intro; workarounds needed with `?` ; C×1: (choreo, ~4, "mdspan sp : [M, N]; // Symbolic dimensions ") | C | 3.3 | — |
| `F20-S05` | [### **d. Compile-Time Optimization**] | P×1: memory layouts, kernel plans | C | 3.3 | 拆分：移出硬件句，保留 DSL 单元 |

### F21 `documentation/value-numbering.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F21-S01` | [# Introduction to Choreo Value Numbering] | P×3: value numbering for shape inference; constant propagation, algebraic simplification; scope-based, AST, no SSA | C | 3.3 | — |
| `F21-S02` | [## Quick Example] | P×5: value numbers for mdspan/ituple; produced VN list intro; VN number semantics; #4 addition expression; a(0) resolves to 1 ; C×2: (plain, ~3, "mdspan a : {1, b, c + 3}"); (plain, ~8, "   VN #0:  const_1") | C | 3.3 | — |
| `F21-S03` | [# Implementations] | —（分组/占位标题，无直属正文） | C | 3.3 | — |
| `F21-S04` | [## From Expressions to Value Numbers] | P×6: syntax-directed AST traversal; expression to value number; steps intro; string signature, future hashing; symbol rules intro; scoped name requirement ; L×2: (3, "Generates a signature from an expression"); (2, "When an expression is to initialize a symbol...") | C | 3.3 | — |
| `F21-S05` | [## Simplifications] | P×4: simplify each signature; unoptimized VN intro; unoptimized vs optimized; constant folding successful ; C×3: (plain, ~3, "   int a = 0 + 1;"); (plain, ~6, "   VN #0:  const_3"); (plain, ~6, "   VN #0:  const_3") | C | 3.3 | — |
| `F21-S06` | [## From Value Numbers to Expression] | P×4: needed for shape inference; mdspan decides memory size; static allocation, codegen shapes; generate expressions back | C | 3.3 | — |
| `F21-S07` | [# Design Considerations] | —（分组/占位标题，无直属正文） | C | 3.3 | — |
| `F21-S08` | [## Lexical Scopes and Restriction] | P×1: global AST, no data flow | C | 3.3 | — |
| `F21-S09` | [## Limitations] | P×1: two limitations intro ; L×1: (2, "Any mdspan can not have element with bounded integer type") | C | 3.3 | — |
| `F21-S10` | [# Reference] | P×1: value numbering paper citation | C | 3.3 | — |

### F22 `documentation/getting-started-with-choreo.src.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F22-S01` | [# Build Choreo From Source Code and Compile Choreo Program] | —（分组/占位标题，无直属正文） | A | 1.2 | — |
| `F22-S02` | [## Build Choreo from Scratch] | —（分组/占位标题，无直属正文） | A | 1.2 | — |
| `F22-S03` | [### Prerequisitions] | L×1: (2, "GCC >=8.1 or clang >=6, where c++17 is fully supported.") | A | 1.2 | — |
| `F22-S04` | [### Environment] | P×9: make setup intro; fetches flex, bison, FileCheck, gtest; make build intro; unit tests intro; tests halt on errors; additional Makefile utilities; help description, sample-test intro; sample-test description; specific operator list ; C×6: (plain, ~3, "make setup"); (plain, ~3, "make"); (plain, ~3, "make test"); (plain, ~3, "make help"); (plain, ~3, "make sample-test"); (plain, ~3, "make sample-test-operator OPERATOR=add") ; A×1: !INCLUDE_IF_EXISTS "target/env_setting_up.md" | A | 1.3 | — |
| `F22-S05` | [## Compile Choreo-C++ Program] | P×12: transpilation to CUDA/Cute; end-to-end compiler; output forms intro; output forms, target limits; gcc-like usage; a.out, -o, -c, -S; -t, -es options; CUDA C++ output; -E preprocessor, #if/#ifdef; -gs work-script; --help options; help output intro ; L×1: (5, "Target source code") ; C×3: (plain, ~3, "croq your_program.co"); (bash, ~3, "croq -t cuda your_program.co -es -o cuda_source.co"); (bash, ~20, "// --help") | A | 1.3 | 拆分：移出硬件句，保留 DSL 单元 |

### F23 `documentation/macro-in-choreo.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F23-S01` | [## Overview] | P×1: macro expansion for host/tileflow | B | 2.13 | — |
| `F23-S02` | [## Object-Like Macros] | P×4: object-like macros, text replacement; static shapes via substitution; parameterized macros unsupported; MAX macro would not work ; C×2: (choreo, ~16, "#define M 256"); (cpp, ~3, "#define MAX(a, b) (((a) > (b)) ? (a) : (b))") | B | 2.13 | — |
| `F23-S03` | [## Comments] | P×1: C-style comments supported | B | 2.13 | — |
| `F23-S04` | [## Conditional Compilation] | P×3: conditional directives supported; same preprocess for both; no full C preprocessing ; C×1: (choreo, ~18, "#define PATH0") | B | 2.13 | — |
| `F23-S05` | [## Difference with C++ pre-processing] | P×2: earlier than C++ preprocessing; only tileflow scope, limited ; C×1: (plain, ~3, "chore-preprocessing -> choreo compilation -> c/c++ preprocessing -> c/c++ compilation") | C | 3.9 | — |
| `F23-S06` | [## Pre-defined Macros] | P×2: target builtin macros; usable in tileflow and host | C | 3.9 | 拆分：移出硬件句，保留 DSL 单元 |

### F24 `documentation/call-in-choreo.src.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F24-S01` | [## Overview] | P×1: call statement details intro | B | 2.13 | — |
| `F24-S02` | [## Call Device Functions] | P×1: call keyword required | B | 2.13 | — |
| `F24-S03` | [### Basic Syntax of Call Statements] | P×4: general syntax intro; template args and arguments; no caller/callee checks; calls must be in parallel-by ; C×2: (choreo, ~3, "call func-name <optional-template-args> (arguments);"); (choreo, ~10, "__co__ void foo() {") | B | 2.13 | 拆分：移出硬件句，保留 DSL 单元 |
| `F24-S04` | [### The Convention: Allowed Argument Types] | P×6: allowed types intro; int, float types, target variance; no float ops in tileflow; example intro; decayed pointer, shape dropped; choreo::f16/bf16 fallback ; L×1: (3, "*Spanned Data* type") ; C×1: (choreo, ~12, "__device__ void bar(float *p, int m, int n) {}") | B | 2.13 | 拆分：移出硬件句，保留 DSL 单元 |
| `F24-S05` | [## Instantiate the Function Template for Call] | —（分组/占位标题，无直属正文） | C | 3.9 | — |
| `F24-S06` | [### Trigger C++ Template Instantiation] | P×4: template call possible; instantiates matmul<16, 4, 72>; compile-time constant required; abort if inference fails ; C×2: (choreo, ~15, "template<int M, int N, int K>"); (choreo, ~7, "__co__ void foo(int M) {") ; A×1: !INCLUDE_IF_EXISTS "target/calls.md" | C | 3.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F24-S07` | [### Quick Summary] | P×1: call syntax recap, caution | B | 2.13 | — |

### F25 `documentation/input-output.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F25-S01` | [## Overview] | —（分组/占位标题，无直属正文） | B | 2.3 | — |
| `F25-S02` | [## Choreo Function Input] | —（分组/占位标题，无直属正文） | B | 2.3 | — |
| `F25-S03` | [### Input Parameters] | P×3: only spanned data, integer; named/unnamed parameters; unnamed param valid ; C×1: (choreo, ~3, "__co__ void foo(f32 [7] a, int) {...}") | B | 2.3 | — |
| `F25-S04` | [### C++ Host: Data Argument with Shape] | P×6: shaped spanned, need conversion; choreo.h API, implicit include; make_spanned, spanned_view; variable dimensions example; runtime shape check, terminate; inputs are references, no copy ; C×3: (choreo, ~10, "__co__ void foo(f32 [7, 16] input, f32 [7, 16] output) {...}"); (cpp, ~7, "template <typename T, size_t Rank>"); (choreo, ~7, "__co__ void foo(f32 [7, 16] input) {...}") | B | 2.3 | — |
| `F25-S05` | [## Choreo Function Output] | —（分组/占位标题，无直属正文） | B | 2.3 | — |
| `F25-S06` | [### Return Type and Type Deduction] | P×3: output types, void allowed; auto return inference; auto infers f32 [7,16] ; C×1: (choreo, ~7, "__co__ auto foo() {") | B | 2.3 | — |
| `F25-S07` | [### C++ Host: Shape Return] | P×4: semantic copy, buffer; spanned_data, no copy; owns buffer, RVO; .rank(), .shape() query ; C×2: (choreo, ~12, "__co__ auto foo() {"); (cpp, ~7, "template <typename T, size_t Rank>") | B | 2.3 | — |
| `F25-S08` | [## More Questions] | P×3: function signature question; C++ calling convention, no extern C; unsupported features intro ; L×1: (2, "Choreo function declaration without definition") | C | 3.9 | — |

### F26 `documentation/target/calls.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F26-S01` | [### *Factor* Target Specific Details] | P×4: factor target C-linkage, wrapper tricks; turn bar into template; generated specialization, no extern C; -kt for choreo-factor ; C×2: (choreo, ~9, "__cok__ {"); (choreo, ~12, "__cok__ {") | C | 3.9 | 拆分：移出硬件句，保留 DSL 单元 |

### F27 `documentation/target/env_setting_up.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F27-S01` | (无标题，整篇) | P×6: real hardware GCU setup; GCU-2.x compiler/runtime; GCU-2.x hardware driver; GCU-3.x commands intro; conjunction "and"; GCU-3.x compiler, runtime, driver ; C×4: (plain, ~3, "make setup-gcu2"); (plain, ~3, "make gcu2-kmd"); (plain, ~3, "make setup-gcu3"); (plain, ~3, "make gcu3-kmd") | X | X | 整节移出 → 独立硬件文档；整篇 GCU 真机/驱动/运行时配置移出到独立硬件文档 |

### F28 `documentation/basic-dma.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F28-S01` | **## Overview** (L1) | 1) P×1 — DMA statement most complex, future introduced | B | 2.9 | — |
| `F28-S02` | **## Data Movement Statement** (L4) | 1) P×1 — heterogeneous hardware, memory hierarchies, DMA terminology | C | 3.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F28-S03` | **### Basic Syntax** (L8) | 1) P×2 — async DMA runs parallel, syntax organized as follows; 2) C×1 — (choreo, ~3行, `future = dma-op src-expr => dst-expr;`); 3) P×2 — future is handle + result, RHS op-type/src/dst split by `=>` | B | 2.9 | — |
| `F28-S04` | **### Synchronous and Asynchronous** (L22) | 1) P×2 — async = non-blocking / sync = blocking, `.async` suffix example; 2) C×1 — (choreo, ~6行, `f0 = dma.copy.async data0 => shared; // non-blocking`); 3) P×2 — f0 async vs f1 sync, `wait f0` sync + dangling-future warning | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F28-S05` | **### Operation Type** (L39) | 1) P×1 — abstraction for modern hardware, TMA shaped transfers; 2) L×1 — (3 项, "`dma.copy`: Copies flat memory directly"); 3) P×1 — configurations programmed as DMA parameters; 4) C×1 — (choreo, ~4行, `global f32 [32, 16, 9] input;`); 5) P×2 — `<>` encloses DMA configuration, "listed below as an example"（悬空引用，无表）; 6) P×1 — platform-dependent support, CUDA/Cute TMA mapping | C | 3.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F28-S06` | **### Data Expression** (L59) | 1) P×1 — simple spanned data / storage location only; 2) C×1 — (choreo, ~6行, `global f32 [32, 16] input;`); 3) P×2 — storage location = compiler-allocated destination, recommend storage destinations; 4) P×1 — ChunkAt introduced next | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F28-S07` | **### Future and Wait** (L74) | 1) P×2 — both DMA kinds define future, multi-future wait / sync error; 2) C×1 — (choreo, ~8行, `f0 = dma.copy.async input0 => shared;`); 3) P×1 — future retrieves destination buffer; 4) L×1 — (2 项, "`.span` to retrieve the mdspan of DMA destination"); 5) P×1 — "For example:"; 6) C×1 — (choreo, ~7行, `f0 = dma.copy.async input0 => shared;`); 7) P×1 — span/data usage, ElementCount on mdspan | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |
| `F28-S08` | **## Quick Summary** (L106) | 1) P×2 — recap DMA/async/futures, next section data expression tiling | B | 2.9 | 拆分：移出硬件句，保留 DSL 单元 |

### F29 `documentation/tiling-chunkat.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F29-S01` | **## Overview** (L1) | 1) P×1 — introduces chunkat expression, core concept | B | 2.8 | — |
| `F29-S02` | **## *Ubound* Operation** (L4) | 1) P×2 — bounded vars & implicit upper-bound in foreach, `#` prefix ubound; 2) C×1 — (choreo, ~6行, `with size in [32] {`); 3) P×1 — `#size` = 32, current value gives wrong shape | B | 2.8 | — |
| `F29-S03` | **## Use `chunkat` Expression to Tile Data** (L18) | 1) P×1 — chunkat enables tiling in data expressions; 2) C×1 — (choreo, ~7行, `__co__ void foo(f32 [36] input) {`); 3) P×1 — `.chunkat` creates chunk, = `.chunk().at()`; 4) F×1 — (`assets/figures/chunkat.png`, "Choreo-Structure-Chunkat：chunk/at 语义示意"); 5) P×1 — `input.chunkat(p)` ≡ `input.chunk(#p).at(p)`; 6) L×1 — (4 项, "`input` is the spanned data to be tiled"); 7) P×1 — chunkat takes bounded vars, upper-bounds = tiling factors | B | 2.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F29-S04` | **## Use Multiple Bounded Variables** (L42) | 1) P×1 — example with multiple bounded variables; 2) C×1 — (choreo, ~11行, `__co__ void foo(f32 [36, 14, 8] input) {`); 3) P×3 — two bounded vars & chunkat concatenation, 6 threads + 7×4 iterations, compiler transpiles to C++ subscripting | B | 2.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F29-S05` | **## Tile and Flow** (L63) | 1) P×1 — chunkat on source vs destination semantics; 2) C×1 — (choreo, ~10行, `__co__ auto foo(f32 [36] input) {`); 3) P×2 — un-tile destination + same shape/type rule, both-sided rarely supported | C | 3.5 | 拆分：移出硬件句，保留 DSL 单元 |
| `F29-S06` | **## Quick Summary** (L81) | 1) P×2 — recap chunkat tiling src/dst, encourage full tileflow program | B | 2.8 | 拆分：移出硬件句，保留 DSL 单元 |

### F30 `documentation/intro-loop-parallel.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F30-S01` | **## Sequentialism and Parallelism** (L1) | 1) P×4 — repetitive tasks: loops vs parallel, SPMD paradigm + `parallel-by`, `with-in`+`foreach` do-while style, section roadmap | C | 3.4 | — |

### F31 `documentation/parallel-in-choreo.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F31-S01` | **## Overview** (L1) | 1) P×1 — utilize parallel hardware, guide to `parallel-by` | C | 3.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F31-S02` | **## Parallel Execution Block** (L5) | 1) P×1 — definition: parallel execution block, simultaneous instances | C | 3.4 | — |
| `F31-S03` | **### Basic Syntax: `parallel-by`** (L8) | 1) P×1 — basic syntax follows; 2) C×1 — (choreo, ~5行, `parallel p by 6 {`); 3) P×1 — (2 行连续) SPMD-style block named, instances = parallel thread, "In this example:"; 4) L×1 — (3 项, "`parallel`: keyword initiates the parallel execution block"); 5) P×1 — parallel variable can be omitted; 6) C×1 — (choreo, ~3行, `parallel by 2 { ... }`); 7) P×1 — two threads but cannot work on different data | B | 2.7 | — |
| `F31-S04` | **### Multiple-Level Parallelism** (L32) | 1) P×1 — example of multiple levels; 2) C×1 — (choreo, ~8行, `parallel p by 2 {`); 3) P×2 — 2×12=24 threads, CUDA grid/block mapping; 4) P×1 — hierarchical memory architecture, synchronization cost, CUDA block vs grids; 5) P×1 — single-line comma-separated syntax; 6) C×1 — (choreo, ~3行, `parallel p by 2, q by 12 { ... }`); 7) P×1 — equivalent, assumes no code between levels | B | 2.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F31-S05` | **### Sub-Level Parallelism** (L57) | 1) P×1 — CUDA allows three sub-levels, ituple/mdspan syntax; 2) C×1 — (choreo, ~11行, `parallel {px, py, pz} by [1, 1, 2] {`); 3) P×2 — level-0/level-1 sub-levels & mdspan counts, CUDA GPU workload + least-significant-var convention | B | 2.7 | 拆分：移出硬件句，保留 DSL 单元 |
| `F31-S06` | **### Bounded Variable** (L77) | 1) P×1 — variables after `parallel` = Bounded Variables; 2) L×1 — (2 项, "integer → bounded integer"); 3) P×2 — bound = value + bounds, `parallel p by 6` → `[0, 6)`; 4) P×1 — concept explored further later | C | 3.2 | — |
| `F31-S07` | **## The Implications** (L88) | 无直属正文（仅含 4 个 ### 子标题） | C | 3.4 | — |
| `F31-S08` | **### Heterogeneity: Transpiling to Kernel Launch** (L90) | 1) P×1 — parallel-by = host/device boundary for CUDA/Cute; 2) L×1 — (2 项, "outside the parallel-by block → host code"); 3) C×1 — (choreo, ~11行, `__co__ void foo(...) {`); 4) P×1 — analogous to CUDA Kernel Launch, unified SPMD abstraction | C | 3.1 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F31-S09` | **### Restrictions on Storage Specifiers** (L112) | 1) P×3 — parallel-by tied to hardware/heterogeneity/memory hierarchy, CUDA/Cute scratchpad shared/local vs global host memory, shared/local lifetime = kernel launch | X | X | 整节移出 → 独立硬件文档；存储说明符限制的硬件约束段整节移出；语言规则一句并入 §3.2 |
| `F31-S10` | **### Asynchronous Kernel Launch and Stream Binding** (L120) | 1) P×1 — default host waits, `parallel.async`, angle-bracket stream; 2) C×1 — (choreo, ~8行, `__co__ void foo(stream s, f32[M] a) {`); 3) P×1 — cross-link to `general-async.md` | C | 3.8.4 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F31-S11` | **### Quick Summary** (L135) | 1) P×2 — recap parallel-by/multi-sub-level/kernel launch/memory constraints, next: loops | C | 3.4 | 拆分：移出硬件句，保留 DSL 单元 |

### F32 `documentation/within-loops.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F32-S01` | **## Overview** (L1) | 1) P×1 — learn `with-in` + `foreach` | B | 2.7 | — |
| `F32-S02` | **## Loops in Choreo** (L5) | 1) P×2 — C++ for/while/do-while flexibility, Choreo constrained DSL loop; 2) C×1 — (choreo, ~5行, `foreach index {`); 3) C×1 — (cpp, ~5行, `for (int index = 0; index < 6; ++index) {`); 4) P×2 — bounded vars sufficient for data movement, parallel-by vars immutable → use `with-in` | B | 2.7 | — |
| `F32-S03` | **## The **With-In** Block** (L29) | 无直属正文（仅 1 个 ### 子标题） | B | 2.7 | — |
| `F32-S04` | **### Defining *Bounded Variable*s:** (L31) | 1) P×1 — with-in defines bounded integer/ituple, no parallel instances; 2) C×1 — (choreo, ~11行, `with x in 128 {`); 3) P×1 — x/y/index upper bounds, named ituple elements; 4) C×1 — (choreo, ~8行, `with {x, y} in [10, 10] {`); 5) P×1 — comma-separated multiple with-in; 6) C×1 — (choreo, ~3行, `with index = {x, y} in [10, 10], idx in [100, 10] { }`); 7) P×1 — scope limited to following block | B | 2.7 | — |
| `F32-S05` | **## The `foreach` Block** (L66) | 无直属正文（仅 3 个 ### 子标题） | B | 2.7 | — |
| `F32-S06` | **### Basic Syntax** (L68) | 1) P×1 — with-in vars iterated by foreach; 2) C×1 — (choreo, ~7行, `with index in [6] {`); 3) P×1 — iterates 6 times 0..5; 4) P×1 — ituple iteration possible; 5) C×1 — (choreo, ~5行, `with index = {x, y} in [6, 17] {`); 6) P×1 — ituple iteration = nested loops; 7) C×1 — (cpp, ~4行, `for (int x = 0; x < 6; ++x)`); 8) P×2 — left-to-right ↔ outer-to-inner nesting, applies to comma-separated foreach list; 9) C×1 — (choreo, ~5行, `with index = {x, y} in [6, 17], iv in [128] {`); 10) P×1 — equivalent multi-level nested loop; 11) C×1 — (cpp, ~5行, `for (int iv = 0; iv < 128; ++iv)`) | B | 2.7 | — |
| `F32-S07` | **### Syntactic Sugar** (L114) | 1) C×1 — (choreo, ~5行, `foreach x in 128 { }`); 2) P×1 — equivalent to the following code; 3) C×1 — (choreo, ~13行, `with x in 128 {`) | B | 2.7 | — |
| `F32-S08` | **### Deriving the Loop From a Bounded Integer** (L139) | 1) P×1 — pipelining may need modified iterations; 2) C×1 — (choreo, ~5行, `with {x, y} in [6, 17] {`); 3) P×1 — Range Expression `y(1::)` derives loop; 4) C×1 — (cpp, ~4行, `for (int x = 0; x < 6; ++x)`); 5) P×1 — y-loop starts at 1, range expression form; 6) C×1 — (无语言标注, ~3行, `bounded-variable(lower-offset:upper-offset:stride)`); 7) P×1 — derives loop, bounded var = iteration variable; 8) L×1 — (3 项, "initial value = lower-offset + lower bound"); 9) P×2 — `y(1:-1:2)` example + defaults, range expr only on bounded vars / ituple error | B | 2.7 | — |
| `F32-S09` | **## Values of Bounded Variables** (L172) | 1) P×1 — two key values: current & upper-bound; 2) L×1 — (2 顶层项含 3 子项, "**Current Value**: within foreach determined by iteration"); 3) P×1 — immutability outside foreach, code illustrates; 4) C×1 — (choreo, ~9行, `with x in 6 {`) | C | 3.2 | — |
| `F32-S10` | **## Quick Summary** (L194) | 1) P×2 — recap with-in/foreach/range ops, loop deriving important for multi-buffering | B | 2.7 | — |

### F33 `documentation/loop-in-advance.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F33-S01` | **## Overview** (L2) | **P×0 — 空标题，无任何正文**（仅 L3 空行） | C | 3.4 | — |
| `F33-S02` | **## Parallelization and Masking** (L4) | 1) P×2 — mask off parallel threads for uneven tasks, implicit vs explicit masking | C | 3.4 | — |
| `F33-S03` | **### Explicit Masking/Condition** (L10) | 1) P×2 — masking = full view of SPMD instances, boolean on thread identifiers; 2) C×1 — (choreo, ~11行, `__co__  void foo() {`)；（另有 P×1 实验性说明并入第 1 项同段语义：explicit masking of `foreach` block） | C | 3.4 | — |
| `F33-S04` | **### Implicit Masking** (L27) | 1) P×1 — some structures imply implicit masking, multi-level parallel-by; 2) C×1 — (无语言标注, ~14行, `__co__  void foo() {`); 3) P×1 — CUDA/Cute may invoke 2 threads, level-0 code as if `if (q == 0)` | C | 3.4 | —；该文件含一段无标题草稿残段（L47–L49），随本节原样保留并标注「上游未完成」 |
| `F33-S05` | **## Experimental: Optional *Where-Binding* for `within`** (L51) | 1) P×1 — where-clause aliases bounded variables; 2) C×1 — (choreo, ~6行, `with {m, n} in [M, N], {n_p, k} in [N_P, K]`); 3) P×1 — `<->` establishes where-binding, aliases both ways | C | 3.4 | — |

### F34 `documentation/thread-masking.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F34-S01` | **## Overview** (L1) | 1) P×1 — SPMD → MPMD via masking, `inthreads` block | C | 3.4 | — |
| `F34-S02` | **## Mask the Parallel Threads** (L4) | 无直属正文（仅 2 个 ### 子标题） | C | 3.4 | — |
| `F34-S03` | **### Masking in SPMD** (L6) | 1) P×1 — mask threads for uneven data chunks; 2) F×1 — (`assets/figures/masking-datachunk-dispatch.png`, "13 chunks / 4 threads tile dispatch"); 3) P×1 — figure: 13 chunks, 4 threads, masks `1111`→`1000` | C | 3.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F34-S04` | **### Uniform and Divergence** (L13) | 1) P×1 — divergence definition from SPMD view; 2) L×1 — (2 项, "**Uniform Code**: All threads execute same code path"); 3) P×2 — divergent code needed, mimics MPMD; 4) P×1 — Choreo `inthreads`, dive into detail | C | 3.4 | — |
| `F34-S05` | **## `inthreads`: Divergent Code with the Thread Mask** (L23) | 无直属正文（仅 2 个 ### 子标题） | C | 3.4 | — |
| `F34-S06` | **### Parallel Perspective of Tileflow Programs** (L24) | 1) P×2 — `inthreads` + compare expression, look at example; 2) C×1 — (choreo, ~20行, `__co__  void foo() {`); 3) P×1 — re-examine tileflow from parallelism perspective ("It consists of"); 4) L×1 — (2 顶层项含 2 子项, "*Sequential Code*"); 5) P×1 — `p <= 2` creates mask `000111`, forced synchronization after block; 6) P×1 — `inthreads` looks like C++ `if` but claims divergence; 7) L×1 — (2 项, "only in SPMD code region"); 8) P×1 — compiler checks comparison & placement | B | 2.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F34-S07` | **### Make it Asynchronous** (L66) | 1) P×1 — `inthreads.async` example for MPMD in SPMD; 2) C×1 — (choreo, ~14行, `__co__  void foo() {`); 3) P×1 — two sub-groups without forced sync, `sync.shared` = sync point; 4) P×1 — only outer `inthreads` async, inner = compile-time error | C | 3.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F34-S08` | **## Masking for Multi-Levels** (L87) | 1) P×1 — masking at different parallel levels; 2) C×1 — (choreo, ~14行, `__co__  void foo() {`); 3) P×1 — "This code uses two levels of parallelization:"; 4) L×1 — (2 项, "Outer level (`p`): Parallel count of 3"); 5) P×1 — first predicate restricts execution; 6) L×1 — (2 项, "Outer parallel IDs: `0, 1` (excluding 2)"); 7) P×1 — mask `011-0001`, levels separated by hyphen; 8) P×1 — second predicate `q == 1` no outer restriction | C | 3.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F34-S09` | **## Implicit Masking** (L118) | 1) P×1 — code between parallel-by levels = implicit masking; 2) C×1 — (无语言标注, ~11行, `__co__  void foo() {`); 3) P×1 — equivalent to `inthreads (q == 0)`, single thread executes outer code | B | 2.10 | — |
| `F34-S10` | **## Quick Summary** (L135) | 1) P×2 — recap masking/inthreads/implicit, next: synchronization challenges | C | 3.4 | — |

### F35 `documentation/events.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F35-S01` | **## Overview** (L1) | 1) P×1 — events synchronize async code from `inthreads` | B | 2.10 | — |
| `F35-S02` | **## Event Basics** (L4) | 1) P×1 — event definitions vary, Choreo event for async interaction; 2) P×1 — SET/UNSET states, declares `shared` event; 3) C×1 — (无语言标注, ~3行, `shared event e;`); 4) P×2 — shared event only inside parallel-by + platform restrictions, C-style event array; 5) C×1 — (无语言标注, ~3行, `shared event e[4], e1;`); 6) P×1 — comma-separated event/array in one line; 7) P×1 — `trigger` changes UNSET → SET; 8) C×1 — (无语言标注, ~5行, `shared event e[4], e1;`); 9) P×1 — trigger each event in array + single event; 10) P×1 — `wait` blocks until SET; 11) P×1 — trigger unblocks wait, example; 12) C×1 — (无语言标注, ~10行, `parallel p by 2 {`); 13) P×1 — wait-thread/trigger-thread chains async `inthreads`; 14) P×1 — event auto-reset by `wait` | B | 2.10 | 拆分：移出硬件句，保留 DSL 单元 |
| `F35-S03` | **## Event Instances** (L52) | 1) P×1 — events associated with storage specifier; 2) C×1 — (无语言标注, ~12行, `__co__ void foo() {`); 3) P×1 — comments explain instances, same as spanned data, leveled boolean in storage | C | 3.8 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F35-S04` | **## DeadLocks** (L70) | 1) P×1 — event lock-free by default, risky multi-thread update; 2) P×1 — "For example,"; 3) C×1 — (无语言标注, ~9行, `parallel p by 1 {`); 4) P×1 — two threads wait one event, reset race → deadlock; 5) P×1 — less likely on GPU-like SIMT lock-step, unsafe for CPU threading, assign per-thread event; 6) C×1 — (无语言标注, ~9行, `qc = 2;`); 7) P×1 — event array guarantees single-thread wait/reset | C | 3.8 | 拆分：移出硬件句，保留 DSL 单元 |
| `F35-S05` | **## Quick Summary** (L102) | 1) P×2 — recap trigger/wait/reset, deadlock warning | B | 2.10 | — |

### F36 `documentation/general-async.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F36-S01` | **### Synchronization and Asynchronous `parallel-by`** (L1) | 1) P×1 — `parallel-by` async via `.async`; 2) C×1 — (choreo, ~5行, `parallel.async p by 6 : block {`); 3) P×1 — without `.async` host waits; 4) C×1 — (choreo, ~6行, `parallel p by 6 : block {`); 5) P×1 — (3 行连续) host continues immediately, overlap compute with transfers, concurrent kernels | C | 3.8.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F36-S02` | **### Stream Binding** (L24) | 1) P×1 — (3 行连续) CUDA streams allow concurrent kernels/transfers, angle-bracket binding; 2) C×1 — (choreo, ~10行, `__co__ void foo(stream s0, stream s1, f32[M] a, f32[N] b) {`); 3) P×1 — "In this example:"; 4) L×1 — (3 项, "`stream s0, stream s1`: Stream parameters passed from the host"); 5) P×1 — (2 行连续) different streams + `.async` → concurrent on GPU | C | 3.8.4 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F36-S03` | **### Stream Binding without `.async`** (L50) | 1) P×1 — (3 行连续) sync `parallel-by` + stream: host launches then waits on stream; 2) C×1 — (choreo, ~8行, `__co__ void bar(stream _s, f32[M] a) {`) | C | 3.8.4 | 拆分：移出硬件句，保留 DSL 单元 |
| `F36-S04` | **### Rules** (L65) | 1) L×1 — (3 项, "Stream binding is only allowed on **block-level** parallel (`: block`)") | C | 3.8.4 | 拆分：移出硬件句，保留 DSL 单元 |

### F37 `documentation/memory-in-choreo.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F37-S01` | **# Memory in Choreo** (L2) | 1) P×1 — memory system for large datasets/HPC, spanned type + storage qualifiers across memory hierarchy | C | 3.2 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F37-S02` | **## Spanned Types in Choreo** (L6) | 1) P×2 — spanned type = fundamental type + mdspan, no storage allocation / blueprint + qualifiers → memory hierarchy placement | C | 3.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F37-S03` | **## Storage Qualifiers in Choreo** (L12) | 1) P×1 — qualifiers annotate memory allocation, specify memory region, three primary qualifiers | C | 3.2 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F37-S04` | **### 1. `global`** (L16) | 1) L×1 — (3 项, "default memory space if no storage qualifier"); 2) P×1 — "**Example:**"; 3) C×1 — (choreo, ~3行, `global f32 [100, 200] matrix;`) | B | 2.6 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F37-S05` | **### 2. `shared`** (L26) | 1) P×1 — (4 行连续段落) shared memory 用于单计算节点高速访问、比 global 小但更快、适合中间数据复用、含 "Example:"; 2) C×1 — (choreo, ~3行, `shared f32 [10, 10] tile_data;`) | B | 2.6 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F37-S06` | **### 3. `local`** (L35) | 1) P×1 — (4 行连续段落) smallest/fastest memory、per-thread scalars/small arrays、private data、含 "Example:"; 2) C×1 — (choreo, ~3行, `local f16 [5] thread_local_data;`) | B | 2.6 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F37-S07` | **## Default Behavior and Usage** (L43) | 1) P×1 — no qualifier → global memory, higher latency; 2) P×1 — d0 without qualifier = global; 3) C×1 — (choreo, ~3行, `f32 [100, 100] d0;`); 4) P×1 — place data in local/shared, d0 local / d1 shared; 5) C×1 — (choreo, ~4行, `local f32 [10, 10] d0;     // Small data, local to the thread`) | B | 2.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F37-S08` | **### Example Code: Using Storage Qualifiers** (L55) | 1) P×1 — "To better understand how storage qualifiers work..."; 2) C×1 — (choreo, ~10行, `ndims : [20, 15];`); 3) P×1 — "In this code:"; 4) P×1 — (3 行连续，无项目符号，**不是 Markdown 列表**) d0/d1/matrix 分别位于 local/shared/global | B | 2.6 | 拆分：移出硬件句，保留 DSL 单元 |

### F38 `documentation/multi-buffering-patterns.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F38-S01` | **# Multi-Buffering Optimization with Choreo** (L1) | 1) P×2 — overlap compute with DMA, needs `dma.any` + `swap`/`rotate`; 2) C×1 — (choreo, ~32行, `__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs,`); 3) P×1 — `ele_add` double-buffered A/B in local storage, roles swapped; 4) P×1 — "three stages"; 5) L×1 — (3 项, "**prologue** (line 8-11) pre-loads first chunk"); 6) P×4 — dummy futures & lexical scope, `swap` + `foreach y(1:)` + `y - 1` tuning, `y(-1)` last value, `select` for odd/even bounds; 7) P×1 — >2 buffers → `rotate` replaces `swap` | C | 3.6.2 | 拆分：移出硬件句，保留 DSL 单元 |

### F39 `documentation/dma-in-advance.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F39-S01` | **## Overview** (L1) | 1) P×1 — advanced DMA statement syntax for tileflow programs | C | 3.6 | — |
| `F39-S02` | **## Advances in *Data Expression*** (L4) | 无直属正文（仅 3 个 ### 子标题，其中 2 个为空占位） | C | 3.6 | — |
| `F39-S03` | **### Reshape with `.span_as`** (L5) | **P×0 — 空标题占位，无任何正文** | C | 3.6 | —；空标题占位（Reshape with .span_as / Dimension Composition / The Placeholders / Mutate the Current Value） |
| `F39-S04` | **### Dimension Composition inside `chunkat`** (L8) | **P×0 — 空标题占位，无任何正文** | C | 3.6 | — |
| `F39-S05` | **### Full Non-Blocking DMA Mode (Chain Mode in Choreo)** (L10) | 1) P×1 — chain async DMA with `after` event notifications; 2) C×1 — (choreo, ~3行, `out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;`); 3) P×1 — "In this example:"; 4) L×1 — (3 项, "`out_store` is the asynchronous DMA operation"); 5) P×1 — neither blocks main flow, event-driven dependency | C | 3.6 | 拆分：移出硬件句，保留 DSL 单元 |
| `F39-S06` | **## Dynamic Shape** (L24) | 无直属正文（仅 1 个空 ### 子标题） | C | 3.6 | — |
| `F39-S07` | **### The Placeholders** (L26) | **P×0 — 空标题占位，无任何正文** | C | 3.6 | — |
| `F39-S08` | **## Mutate the Current Value** (L29) | **P×0 — 空标题占位，无任何正文** | C | 3.6 | — |

### F40 `documentation/tileflow-opt.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F40-S01` | **# Tiling Optimisation in Choreo** (L1) | 1) P×1 — extends tiling optimization with practical examples | C | 3.10 | — |
| `F40-S02` | **## Example 1: `matmul`** (L5) | 1) P×1 — matmul device program, tiling + memory management + DMA；另含水平分隔线 `---` (L8) | C | 3.10.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F40-S03` | **### Example Code: Matrix Multiplication with Tiling** (L10) | 1) C×1 — (choreo, ~44行, `__co__ s32 [512, 1024] matmul(s32 [512, 1024] lhs, s32 [1024, 1024] rhs) { `)；另含水平分隔线 `---` (L56) | C | 3.10.1 | 拆分：移出硬件句，保留 DSL 单元 |
| `F40-S04` | **### Memory Tiling with `chunkat`** (L58) | 1) P×1 — memory tiling achieved via chunkat slicing lhs/rhs ; 1) L×1 — (2 项, "lhs and rhs tiled using chunkat by m_tile/k_tile/n_tile") ; 1) L×1 — (2 项, "m_tile/k_tile/n_tile are bounded variables") | C | 3.10.1 | — |
| `F40-S05` | **## Example 2: rgb2gray** (L73) | 1) P×1 — RGB→Grayscale weighted average, physical + virtual parallelism；另含水平分隔线 `---` (L76) | C | 3.10.2 | — |
| `F40-S06` | **### Example Code** (L77) | 1) C×1 — (choreo, ~18行, `__co__ auto rgb2gray(f32 [N, 3, H, W] input) {`)；另含水平分隔线 `---` (L97) | C | 3.10.2 | 拆分：移出硬件句，保留 DSL 单元 |
| `F40-S07` | **### Compose Two Bounded Variables for Tiling with `#`** (L98) | 1) P×1 — `#` operator = Cartesian product of two bounded vars, not commutative ; 1) L×1 — (4 项, "four bounded variables q, n, h, w with upper bounds 6, N/6, H/16, W/512") ; 1) L×1 — (3 项, "q and n combine physical and virtual parallelism") | C | 3.5 | 拆分：仅留语法行/抽象结论，硬件段落移出 |
| `F40-S08` | **## Virtual Parallelism and Physical DMA Mapping** (L117) | 1) P×1 — TVM schedules & loop transformations intro; 2) C×1 — (python, ~10行, `# Example of TVM-style loop schedule` / `s = tvm.create_schedule(A.op)`) — **注意 L122 是代码块内 Python 注释，不是标题**; 3) P×1 — TVM split/bind abstraction, limitations with memory hierarchies | X | X | 整节移出 → 独立硬件文档；(08–12) TVM/物理内存层级/DRAM-SRAM 论证整组移出；仅留「虚拟并行 + 显式数据移动」结论句 |
| `F40-S09` | **#### The Limitation of Virtual Parallelism in TVM-style Schedules** (L134) | 1) P×2 — threadIdx/blockIdx map virtual→physical + data movement hidden, suboptimal for complex memory hierarchies | X | X | 整节移出 → 独立硬件文档 |
| `F40-S10` | **#### Freedom Left to the System** (L139) | 1) P×1 — decision-making left to auto optimization; 2) L×1 — (3 项, "Memory layout (whether data should be placed in local, shared, or global memory)"); 3) P×1 — freedom hides physical behavior, strength and weakness; 4) L×1 — (2 项, "Strength: automatic optimizations based on heuristics") | X | X | 整节移出 → 独立硬件文档 |
| `F40-S11` | **#### The Importance of Explicit Physical Mapping in Choreo** (L151) | 1) P×2 — Choreo models virtual parallelism + physical DMA/memory ops, explicit coupling optimizes for underlying hardware | X | X | 整节移出 → 独立硬件文档 |
| `F40-S12` | **#### How Choreo Works to address this limitation?** (L156) | 1) P×6 — virtual parallel via bounded vars, physical DMA mapping, DRAM→SRAM→registers + `#` operator, composite variable/cache locality + `dma.copy.async` to shared/registers, mapping advantage over TVM, fine-grained control/hierarchy alignment + 末段 L169 编辑残留说明 "This compact version integrates all the important points without using subheadings..." | X | X | 整节移出 → 独立硬件文档 |

### F41 `documentation/async-dma-patterns.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F41-S01` | (空文件 / 无内容) | —（分组/占位标题，无直属正文） | — | 占位 | — |

### F42 `documentation/contribute.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F42-S01` | (空文件 / 无内容) | —（分组/占位标题，无直属正文） | — | 占位 | — |

### F43 `documentation/dynamic-design.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F43-S01` | (空文件 / 无内容) | —（分组/占位标题，无直属正文） | — | 占位 | — |

### F44 `documentation/elementwise-add-sample.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F44-S01` | (空文件 / 无内容) | —（分组/占位标题，无直属正文） | — | 占位 | — |

### F45 `documentation/operator-samples.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F45-S01` | (空文件 / 无内容) | —（分组/占位标题，无直属正文） | — | 占位 | —；上游仅 1 行 `// TODO: add link, discussion on each operator example…`；登记但不填正文 |

### F46 `documentation/symbolic-evaluation.md`

| 单元 ID | 原小节 | 内容单元（原样登记） | 归类 | 目标位置 | 硬件处理 |
|---|---|---|---|---|---|
| `F46-S01` | (空文件 / 无内容) | —（分组/占位标题，无直属正文） | — | 占位 | — |

## 统计与待办

- 素材文件：46 篇（tutorial/ 11 + documentation/ 35，含 target/ 2）
- 登记小节：337 个；空/TODO 占位文件：6 个；整节移出：16 个
- 内容单元合计：P×892，C×257，T×21，L×71，F×97，A×7，M×0
- 环境无 pandoc；填正文时需自建 Markdown→LaTeX 片段转换（代码块/表格/图注/提示框原样保留）。
- 归类确认后按本表逐节填充；附录 A 由 §2.15 速查表汇总生成，附录 B 由本表生成。
