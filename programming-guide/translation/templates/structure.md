=== FRONT
H: Preface
P: As large models have become widely used, efficient operator implementations have become a growing part of the development work. Model architectures may share a few common designs, but the computations within them continue to vary. Attention mechanisms, sparsity patterns, positional encodings and quantisation schemes all introduce new implementation requirements. Even the same operator may need different kernels for different input shapes or numerical formats. A shared model design therefore leaves substantial work to be done at the operator level.
P: That work also grows with the variety of computing platforms. A program may need to run on several kinds of machine, each with its own memory organisation and execution model. Within a single family of machines, new instructions and programming interfaces can require an existing kernel to be reconsidered and retuned. Operator fusion adds another source of variation: combining several operations can reduce intermediate data transfers and launch costs, but each combination brings its own implementation choices. Supporting these workloads calls for a continuing supply of kernels and the means to develop them efficiently.
P: Writing those kernels is demanding because performance depends on decisions that interact: tile sizes, data layouts, memory staging, pipeline depth and cooperation between parallel regions. A change in one can require changes in several others. Higher-level kernel languages have emerged to improve author productivity by expressing more of this work with less code. Their central challenge is to simplify programming while keeping the decisions needed to reach high performance available to the programmer. When an abstraction hides those decisions, it also limits the implementations that can be explored.
P: Croqtile is a language for high-performance heterogeneous computing built around this challenge, with human productivity as its starting point. It gives shapes, tile selection, data movement and coordination direct language forms, so a programmer can describe how a kernel works in one compact program. Related sizes and indices are derived from shaped data, reducing the bookkeeping that otherwise accompanies a structural change. Symbolic dimensions allow the same source to handle different input sizes, while compile-time checks identify inconsistent shapes and indices before execution. These abstractions add no runtime overhead. Together, they make kernels easier to read and revise while leaving performance decisions explicit.
P: The same design also supports AI-assisted kernel development and tuning. Compact source leaves room in an agent's context for profiling results and previous attempts. Independently expressed operations let the agent change a pipeline or a parallel arrangement through a small number of edits, and compiler diagnostics help it locate and correct invalid combinations. Croqtile is agent-ready in this practical sense: the language supports repeated cycles of proposing a change, checking it and measuring its effect. Its AI-native design carries the goal of efficient kernel development into an automated workflow.
P: Croqtile is embedded in C++. In a `.co` file, ordinary C++ prepares shaped buffers and calls a `__co__` function; that function describes the iterations, tiles, data movement and coordination of the parallel work. This guide introduces those constructs through examples and shows how to combine them into complete programs.
H: Reading guide
P: This guide is for developers who want to read and write Croqtile kernels. It assumes familiarity with C++ functions, pointers and arrays, and with the shape of a tensor. The examples begin with basic language constructs and show how to organise parallel computation, select and move tiles, and coordinate execution. Working through them will help you read a `.co` program and combine these constructs into kernels of your own.
P: If you are new to Croqtile, begin with getting.started to build the compiler and run a first program. Then read language.basics in order. Its short examples start with function declarations and shapes, then introduce iteration, tiling, data movement and coordination. Look first at the structure of each program, then use the explanation to understand the role of each statement. Focus on how the data is divided, moved and used in computation.
P: Once you are comfortable with the basic syntax, consult programming.guide as needed. It groups the language by function and explains the semantics, constraints and design choices behind the constructs. Complete kernels appear alongside the features they use, so you can study a particular topic or return to it while writing a program. quickref collects the syntax for quick lookup; the corresponding sections explain the meaning of a construct and the conditions for using it.

=== CH1
H: Getting Started
I: This chapter takes you from installing Croqtile to a program you have built and run yourself. After the language's motivation, set up and verify the compiler, run a complete element-wise addition example, and use the command-line options to inspect its build. Finally, learn the tools for checking compilation commands, runtime values, and incorrect results.
N: With a program running, continue to language.basics to learn what each construct in it means. If you already know the syntax, programming.guide is arranged for lookup.

=== CHTITLE1
H: Getting Started

=== CH2
H: Language Basics
I: You will learn to read and write Croqtile programs by working through short examples. Start with the structure of a complete program and the roles of its functions, then follow how shapes, iteration, tile selection, data movement and coordination are expressed. Each example adds a construct and explains its role, so you can build up a working understanding of the syntax.
N: You can now read most `.co` files. programming.guide explains why the constructs are shaped this way and covers the advanced forms; quickref is a compact lookup table.

=== CHTITLE2
H: Language Basics

=== CH3
H: Programming Guide
I: This chapter organises the same language by function. It explains the structure of a Croqtile program, its type system, symbolic shapes, parallelism, tile selection, data movement, contraction and coordination, and the reasoning behind them. Complete kernels sit beside the feature they demonstrate, so each section also stands on its own.
N: quickref collects the syntax in one place. For the compiler design behind the language, see [参考文献].

=== CHTITLE3
H: Programming Guide

=== SEC:1.1
H: What is Croqtile

=== SEC:1.2
H: Setting Up: Requirements, Build, and Verification

=== SEC:1.3
H: The Toolchain: Command Line and Build Utilities

=== SEC:1.5
H: Inspecting and Debugging a Build

=== SEC:2.1
H: The Two Parts of a Croqtile Program

=== SEC:2.2
H: Reading a Kernel Line by Line

=== SEC:2.3
H: The Host-Side API: Inputs and Outputs

=== SEC:2.4
H: Shapes and mdspan

=== SEC:2.5
H: Integers, I-Tuples, and Bounded Values

=== SEC:2.6
H: Spanned Data and Storage Specifiers

=== SEC:2.7
H: Iteration: parallel-by, with-in, and foreach

=== SEC:2.8
H: Tile Selection: chunkat, subspan, and view/from

=== SEC:2.9
H: Data Movement: the DMA Statement and Futures

=== SEC:2.10
H: Coordination: inthreads, Events, and Async Blocks

=== SEC:2.11
H: Branching and Early Return

=== SEC:2.12
H: The Four-Step MMA Syntax

=== SEC:2.13
H: C++ Embedding, Calls, Macros, and the Preprocessor

=== SEC:2.14
H: Printing, Assertions, and Compile-Time Inspection

=== SEC:2.15
H: Syntax Quick Reference

=== SEC:3.1
H: Program Structure and Compilation Models

=== SEC:3.2
H: Syntax Design and the Type System

=== SEC:3.3
H: Symbolic Shapes, Shape Evaluation, and Value Numbering

=== SEC:3.4
H: Parallelism, Space Specifiers, and Masking

=== SEC:3.4.1
H: Worked Example: Tiled Matrix Multiplication

=== SEC:3.4.2
H: Tracing One Output Element

=== SEC:3.5
H: Tile Selection and Data Expressions

=== SEC:3.5.1
H: Arbitrary-Offset Windows: view and from

=== SEC:3.5.2
H: Strided Tiles: subspan, step, and at

=== SEC:3.5.3
H: Zero-Padding with zfill

=== SEC:3.5.4
H: Layout Reinterpretation with span_as

=== SEC:3.5.10
H: Tile Selection: Summary and Syntax

=== SEC:3.6
H: Data Movement: Operations, Configuration, and Async Patterns

=== SEC:3.6.1
H: Worked Example: DMA Matmul

=== SEC:3.6.2
H: Multi-Buffering: dma.any, swap, and rotate

=== SEC:3.6.3
H: Worked Example: A Pipelined Kernel

=== SEC:3.7
H: Contraction and Layout Contracts

=== SEC:3.7.1
H: Worked Example: One Cooperative Group

=== SEC:3.7.2
H: Worked Example: Widening the Cooperation Scope

=== SEC:3.7.3
H: Worked Example: Two Cooperative Groups

=== SEC:3.7.9
H: MMA Syntax Summary

=== SEC:3.7.10
H: MMA Design Summary

=== SEC:3.8
H: Coordination, Pipelines, and Persistent Scheduling

=== SEC:3.8.1
H: Canonical Pattern: One Producer and One Consumer

=== SEC:3.8.2
H: Worked Example: A 1P1C Kernel Skeleton

=== SEC:3.8.3
H: Persistent Scheduling

=== SEC:3.8.4
H: Host-Level Concurrency and Stream Binding

=== SEC:3.8.5
H: Worked Example: A Full Event Pipeline

=== SEC:3.8.10
H: Coordination Pattern Summary

=== SEC:3.9
H: C++ Interoperability and the Preprocessor

=== SEC:3.9.1
H: Worked Example: Calling a Device Helper

=== SEC:3.9.2
H: Verbatim C++ Injection

=== SEC:3.9.10
H: Interoperability Summary

=== SEC:3.10
H: Tileflow Patterns and Design Rationales

=== SEC:3.10.1
H: Worked Example: Tiled matmul

=== SEC:3.10.2
H: Worked Example: rgb2gray

=== SEC:3.11
H: Diagnostics and Methodology

=== APP
H: Syntax Quick Reference
P: A condensed index of the constructs used in this guide, grouped the way you meet them: declarations first, then iteration, tile selection and movement, then statements, coordination and diagnostics. programming.guide explains what each construct means and why it is defined that way; this appendix is for looking up a spelling without re-reading an example.
H: Declarations and shapes
T: | Construct | Meaning | <ROW> | --- | --- | <ROW> | `__co__ T [shape] f(T [shape] a)` | Croqtile function with shaped parameters. | <ROW> | `__co__ void` / `__co__ auto` | In-place result / inferred return type. | <ROW> | `T [a.span] b;` | Buffer with a copied shape. | <ROW> | `a.span` / `a.span(i)` | Whole shape / one dimension. | <ROW> | `x : [7, 8]` | Named `mdspan` declaration. | <ROW> | `a = 1, 2, 3` | `ituple` declaration. | <ROW> | `tensor.at(i, j)` | Position access. |
H: Iteration
T: | Construct | Meaning | <ROW> | --- | --- | <ROW> | `parallel p by N` / `parallel p, q by [M, N]` | Independent instances. | <ROW> | `: block` / `: thread` / `: group` / `: group-4` | Annotations on a parallel axis. | <ROW> | `foreach idx in [N]` / `foreach t(1:)` | Ordered / sliced iteration. | <ROW> | `with idx in [a, b]` | Scoped index binding. | <ROW> | `p # m` / `#p` | Compose indices / axis extent. | <ROW> | `cdiv(a, b)` | Ceiling division. |
H: Tile selection and movement
T: | Construct | Meaning | <ROW> | --- | --- | <ROW> | `f = dma.copy src => shared` | Tile movement statement. | <ROW> | `f = tma.copy src => shared` | The same statement form on the other engine. | <ROW> | `a.chunkat(t)` | One tile of a regular partition. | <ROW> | `a.subspan(M, K).at(i, j)` | Explicit extents and tile coordinates. | <ROW> | `a.subspan(M, K).step(sM, sK).at(i, j)` | Strided tile selection. | <ROW> | `a.view(M, N).from(r, c)` | Window with explicit element offsets. | <ROW> | `a.span_as([dims])` | Shape reinterpretation without moving data. | <ROW> | `f.data` / `f.span` / `dma.any` | Moved tile / its shape / placeholder handle. | <ROW> | `.async` / `wait f` / `swap` / `rotate` | Async transfer and handle rotation. | <ROW> | `global` / `shared` / `local` | Storage specifiers. |
H: Statement modifiers and contraction
T: | Construct | Meaning | <ROW> | --- | --- | <ROW> | `dma.transp<...>` / `dma.pad<...>` | Transform while moving. | <ROW> | `.zfill` / `.swiz<N>` | Fill and layout modifiers. | <ROW> | `mc = mma.fill 0.0` | Zero an accumulator. | <ROW> | `ma = mma.load src.chunkat(...)` | Load an operand. | <ROW> | `mma.row.row mc, ma, mb` | Contraction with a layout contract. | <ROW> | `mma.store mc, dst` | Store the accumulator. | <ROW> | `.sp` / `.scale` / `mma.commit` / `mma.scale` | Contraction variants and stage fence. |
H: Coordination, interop and diagnostics
T: | Construct | Meaning | <ROW> | --- | --- | <ROW> | `if (expr) ... ` | Runtime predicate. | <ROW> | `inthreads.async (cond)` / `inthreads (cond)` | Concurrent region / sequential filter. | <ROW> | `shared event e[N]` / `trigger e` / `wait e` | Synchronisation tokens. | <ROW> | `parallel.async ...` / `stream s` / `yield` | Host scheduling and early return. | <ROW> | `__device__ fn()` / `call fn(args)` | Interoperating C++ function and its invocation. | <ROW> | `__cpp__("...")` / `__cpp__(R"(...)")` | Injected source text. | <ROW> | `#define` / `#if` / `#ifdef` / `#error` / `-D` | Preprocessor. | <ROW> | `print!` / `println!` / `print` / `println` | Compile-time and runtime output. | <ROW> | `assert` / `croq_assert` | Device and host assertions. |
P: For the meaning behind a construct, return to the section that introduced it: language.basics for the first reading and programming.guide for the design. To see a construct in a complete program, start with the worked kernels in 3.4.1, 3.6.1 and 3.8.5. brief

=== BACK
P: center croqgray 0.50.4pt\\[6pt] Croqtile Programming Guide, version , . center

=== GLOSS
H: Glossary
P: English term and the Chinese rendering fixed for this edition.
T: | English | 中文 | <ROW> | --- | --- | <ROW> | kernel | 核函数 |

