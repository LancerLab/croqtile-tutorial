# Croqtile 编程指南中文版：术语表与翻译规范

工程文档，不进入中文版正文。它固定全书译名，供译文编写与一致性检查使用。

## 一、查证依据

译名优先采用国内权威机构与厂商官方中文技术文档，其次采用中文社区已成惯例的
计算机科学译名。本轮实际查阅的来源：

| 来源 | 用途 |
|---|---|
| 全国科学技术名词审定委员会「术语在线」 <https://www.cnterm.cn>；《中国科技术语》 <https://www.term.org.cn/> | 计算机科学技术名词审定口径 |
| 昇腾 CANN 官方中文文档 <https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse/autofuse_1_0050.html> | `Kernel`＝核函数、`Host`/`Device`＝主机/设备、`Operator`＝算子、`Global Memory` 等 |
| NVIDIA 开发者中文博客《附录 C 协作组》 <https://developer.nvidia.com/zh-cn/blog/cooperative-groups-cn>、《CUDA 11 功能揭晓》 | `Cooperative Groups`＝协作组、`tiled partition`＝分片 |
| Linux 内核中文术语表 <https://www.kernel.org/doc/html/latest/translations/zh_TW/glossary.html> | 操作系统语境 `kernel`＝内核，与本书语境区分 |
| 中文 C++ 术语表 <https://awesome-embedded-learning-studio.github.io/Tutorial_AwesomeModernCPP/appendix/terminology> | `zero-overhead abstraction`＝零开销抽象、`RVO`＝返回值优化、`copy elision`＝拷贝消除 |
| zh.cppreference <https://zh.cppreference.com/>（`constexpr`、`常量表达式`条目） | `constexpr`＝常量表达式、编译时求值 |
| 中文维基百科《源到源编译器》与术语表 <https://github.com/FrankHB/YSLib/wiki/Terminology.zh-CN> | `source-to-source compiler`＝源到源编译器、`transpiler`＝转译器、`transcompiler`＝转译编译器 |
| 天津大学《CUDA 编程简介：基础与实践》课程讲义 <https://cam.tju.edu.cn/homepage/zhangyong/ToSeminars/CUDA20230203.pdf> | `kernel`＝CUDA 核函数、`host`/`device`＝主机/设备、`block`＝线程块、`warp`＝线程束 |
| TileLang / Triton 中文技术文档、灵犀 ISA《张量指令集》 <https://linxisa.github.io/isa/header/tileblock/intro> | `copy`/`data movement`＝数据搬运、`tile`＝数据块／分块、`tiled GEMM`＝分块矩阵乘法 |

说明：本书中的 `kernel` 指在设备上并行执行的函数，按昇腾官方中文文档与高校 CUDA
课程讲义译为 **核函数**；`kernel` 作操作系统核心时译「内核」，本书不涉及该义项，
故「内核」列为禁用译名。`data movement` 在国内 GPU／算子开发文档中通行「数据搬运」，
本书统一采用该译名；`tile` 统一为「分块」，不用「瓦片」。`Croqtile` 为语言专名，
按学术技术文档惯例保留原文，不采用项目中文站的「鳄霸」。

## 二、翻译方法（学术／技术文档）

1. **术语一致**：同一英文术语在全书中只有一个中文译名。只有核心领域术语**在首次
   出现时**写作「中文(英文)」，通用词一律只用中文；完整规则见第五节。语言构造名、
   关键字、API、命令、文件名一律保留原文并置于等宽字体。
2. **不增不删**：不补写原文没有的背景解释，不省略限定、条件与否定；代码清单、命令输出、
   表格中的代码与标识符保持原样。
3. **句式**：英语被动语态改中文无主句或主动句；长句按意群拆成短句；避免「被……所……」
   的翻译腔与「本节将介绍……」一类元叙述的机械复制（原文若已去掉元叙述，译文同样不添）。
4. **结构**：章／节／附录标题、导语、清单题、图题、表题、交叉引用全部中文化，编号保持
   与原版一致（第 1 章、2.1 节、清单 2.1、图 3.2、表 1.1）。
5. **标点与混排**：正文用中文全角标点；术语双语括号固定为半角 `( )`，括号内保留英文
   原文的大小写与连字符，括号前后不加空格；中英文之间不额外加空格；行内代码用反引号；
   数值、版本号、单位保留阿拉伯数字。
6. **代码注释**：清单内注释保持英文，不翻译，以免与代码语义脱节；正文若解释注释含义，
   则在正文中用中文说明。
7. **专名**：语言名、框架名与项目名（`Croqtile`、`Triton`、`TileLang`、`CUDA`、C++）以及
   语言构造名（`parallel-by`、`__co__`、`spanned data` 等）无中文译名、保留原文，
   不套用「中文(英文)」格式；缩略语按第五节处理。

## 三、术语对照表

### 语言与编译器

| English | 中文 | 备注 |
|---|---|---|
| programming guide | 编程指南 | 书名 |
| kernel | 核函数 | 设备侧并行函数；非「内核」 |
| host / device | 主机 / 设备 | |
| host program / device program / tileflow program | 主机程序 / 设备程序 / tileflow 程序 | `tileflow` 为语言构造名，保留 |
| operator | 算子 | |
| domain-specific language (DSL) | 领域专用语言 | |
| embedded DSL | 嵌入式领域专用语言 | |
| abstraction | 抽象 | |
| zero-cost abstraction | 零开销抽象 | 不用「零成本抽象」 |
| compile-time / runtime | 编译时 / 运行时 | 与原版 `compile-time` / `runtime` 一一对应 |
| compile-time safety check | 编译时安全检查 | |
| compiler / downstream compiler | 编译器 / 下游编译器 | |
| transpile / transpilation | 转译 / 源到源翻译 | |
| source-to-source compiler / translator | 源到源编译器 / 源到源翻译器 | |
| target compilation | 目标编译 | |
| pre-processing / preprocessor | 预处理 / 预处理器 | |
| macro / preprocessor guard | 宏 / 预处理守卫 | |
| C++ interop / verbatim injection | C++ 互操作 / 原样注入 | |
| template instantiation / type deduction | 模板实例化 / 类型推导 | |
| return value optimization (RVO) | 返回值优化 | |
| immutable / invariant / assertion | 不可变 / 不变式 / 断言 | |
| diagnostic / verbose build | 诊断 / 详细构建 | |
| symbolic dimension / anonymous dimension | 符号维度 / 匿名维度 | |
| dynamic shape / bounded variable | 动态形状 / 有界变量 | |
| value numbering | 值编号 | |

### 并行与执行

| English | 中文 | 备注 |
|---|---|---|
| parallel / parallelism | 并行 / 并行性 | |
| iteration / instance | 迭代 / 实例 | |
| parallel-by / with-in / foreach | `parallel-by` / `with-in` / `foreach` | 构造名不译 |
| space specifier / specifier | 空间说明符 / 说明符 | |
| cooperative group | 协作组 | NVIDIA 官方中文译名 |
| masking / mask | 加掩码 / 掩码 | |
| SPMD / MPMD | 单程序多数据 / 多程序多数据 | 写作 `单程序多数据(SPMD)`、`多程序多数据(MPMD)` |
| producer / consumer | 生产者 / 消费者 | |
| pipeline / pipelining | 流水线 / 流水化 | |
| multi-buffering / double buffering | 多缓冲 / 双缓冲 | |
| event / wait / trigger | 事件 / 等待 / 触发 | 语句名 `wait`、`trigger` 保留 |
| future | future | 类型名保留原文；需说明语义时写 `期值(future)` |
| stream / stream binding | 流 / 流绑定 | |
| persistent kernel / scheduling | 持久核函数 / 调度 | |
| deadlock | 死锁 | |

### 数据与形状

| English | 中文 | 备注 |
|---|---|---|
| shape / dimension / rank | 形状 / 维度 / 秩 | |
| tensor / element-wise | 张量 / 逐元素 | |
| buffer / storage / storage level | 缓冲区 / 存储 / 存储层次 | |
| spanned data / spanned view | `spanned data` / `spanned view` | 语言内建类型名保留，首次说明为「带形状的数据／视图」 |
| mdspan / span | `mdspan`（多维跨度）/ 跨度 | 类型名保留 |
| ituple / integer | `ituple`（整数元组）/ 整数 | |
| storage specifier / storage location | 存储说明符 / 存储位置 | |
| tile / tiling / tiled | 分块 / 分块 / 分块的 | 不译「瓦片」 |
| tile selection / data expression | 分块选择 / 数据表达式 | |
| window / view / offset / stride | 窗口 / 视图 / 偏移 / 步长 | |
| zero-padding / padding | 零填充 / 填充 | |
| layout / reinterpretation | 布局 / 重新解释 | |
| swizzle / layout conflict | 重排 / 布局冲突 | 构造名 `swiz` 保留 |
| DMA (direct memory access) | 直接内存访问(DMA) | 语句名 `dma.copy` 保留 |
| MMA | 矩阵乘加(MMA) | 四步语法名 `mma.*` 保留 |
| contraction / accumulation | 收缩 / 累加 | 张量收缩义 |
| future / `.data` / `.span` | future / `.data` / `.span` | 成员名保留 |
| coalescing / memory access | 合并访问 / 内存访问 | |

### 工具链与文档结构

| English | 中文 | 备注 |
|---|---|---|
| system requirements / build from source | 系统要求 / 从源码构建 | |
| command line / option / flag | 命令行 / 选项 / 标志 | |
| test suite / Makefile target | 测试集 / Makefile 目标 | |
| listing / figure / table | 清单 / 图 / 表 | |
| chapter / section / appendix | 章 / 节 / 附录 | |
| quick reference / syntax | 速查 / 语法 | |
| worked example / skeleton / pattern | 示例剖析 / 骨架 / 模式 | |

### 固定不译

`Croqtile`、`.co`、`__co__`、`__device__`、`__cpp__`、`__cok__`、`__co__ auto`、
`chunkat`、`subspan`、`view()`、`from()`、`step()`、`at()`、`span_as`、`zfill`、
`dma.copy`、`tma.copy`、`dma.any`、`swap`、`rotate`、`inthreads.async`、
`parallel.async`、`print!`、`assert`、`make_spandata`、`croq::`、
`choreo.h`、`croq`（命令）、`MDSpan`、`FP16`、`FP32`、`MMA`、`TMA`、
所有 CLI 选项、文件路径、代码与代码注释、参考文献题名。

## 四、一致性检查

`python3 tools/build_guide_zh.py check` 检查：译文是否覆盖全部内容单元、tag 序列是否与
英文模板一致、行内代码是否与原文对应、禁用译名（如「内核」「瓦片」「零成本抽象」）
是否出现、术语双语括号是否误用全角、清单与图题编号是否连续。首次规则由第五节规定，
由 `normalize` 命令自动落实。

## 五、双语书写规则（中文(英文)）

**只有核心领域术语**在全书首次出现时写作「中文(英文)」，即中文译名后紧跟半角括号与
英文原文；其余一律只写中文。保留范围由 `tools/build_guide_zh.py` 的 `CORE_TERMS`
固定，例如 `核函数(kernel)`、`分块(tile)`、`期值(future)`、`协作组(cooperative group)`、
`单程序多数据(SPMD)`；同一术语全书只标注一次。

- **不标注**：通用计算机词汇与日常用词（性能问题、模型、程序、机器、函数、变量、类型、
  数组、指针、索引、语句、表达式、参数、线程、编译、编译器、构建、测试、版本、文件、
  命令行、代码、数据、值、形状等），以及描述性短语、文档结构性词语。这类词中文译名
  直观且无歧义，附英文只会打断阅读。

- **首次判定**：以全书为范围，按正文顺序（前言 → 第 1 章 → 第 2 章 → 第 3 章 → 附录）
  判定；同一英文术语只在第一处附英文，其后即使跨章、跨节也不重复。
- **一致性**：同一英文术语在全书中只有一个中文译名，且括号内英文只有一种形式，
  不得一处写 `分块(tile)`、另一处写 `分块(tiling)`。
- **括号内**：保留英文原文的大小写与连字符，不写中文释义。缩略语首次出现写
  「中文全称(英文全称, ABBR)」，如 `领域专用语言(domain-specific language, DSL)`；
  后文只用中文全称或该缩略语。
- **自动化**：`python3 tools/build_guide_zh.py normalize` 删除 `CORE_TERMS`
  之外的一切英文标注，并对核心术语按全书顺序只保留首次一处，写回
  `translation/zh/*.md`。该命令幂等，`build` 也会在渲染前自动调用它。
- **不适用**：代码清单及其注释、命令与命令行选项、文件路径、API／关键字／语言构造名、
  保留原文的专名（`Croqtile`、`Triton`、`TileLang`、`CUDA`、C++）、参考文献题名、
  交叉引用编号词（第 1 章、2.1 节、清单 2.1、图 3.2、表 1.1）、文档结构性词语
  （示例剖析、小结、表头「构造」「含义」等）、本术语对照表本身，以及表格中已单独
  列出英文的列。
- **格式**：括号用半角 `( )`，前后不加空格；中文与括号之间不插入标点或空格；
  不使用「中文（English）」的全角形式；术语后接中文标点时，括号在前、标点在后。
