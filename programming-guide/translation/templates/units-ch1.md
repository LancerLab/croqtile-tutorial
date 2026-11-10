=== UID:F01-S01
P: Much of the development effort in a high-performance kernel goes into organising the computation. Beyond implementing the arithmetic, the programmer has to arrange data storage, divide parallel work and coordinate execution. These choices interact, so a change can require updates in several places. Different input sizes or target machines call for another round of validation and tuning. Implementing, maintaining and revising these arrangements accounts for much of the cost.

=== UID:F01-S02
P: Croqtile is a C++ embedded DSL for high-performance kernel programming, designed to make this work easier while preserving control over performance. It provides a direct way to describe the organisation of computation, data movement and coordination in a compact program. The programmer chooses how the kernel works; the compiler derives related information, checks consistency and generates the underlying code. This reduces the repetitive work involved in expressing and maintaining those choices.

=== UID:F01-S03
P: The resulting programs are easier to read as a whole and revise as requirements change. They can accommodate different input sizes, and the abstractions introduce no additional runtime overhead. Compact source and useful compiler feedback also make the same process accessible to AI-assisted development and tuning.

=== UID:F12-S01

=== UID:F13-S01

=== UID:F02-S01
P: Croqtile is built from source rather than installed as a prebuilt package. The compiler is itself a C++ program whose parser is generated from a grammar, so the first thing to check is the toolchain that builds it. The table below gives the versions that are known to work: a C++17-capable compiler and two parser-generator tools.

=== UID:F02-S02
T: | **Dependency** | **Minimum Version** | <ROW> | --- | --- | <ROW> | GCC | 8.1+ (or Clang 6+) | <ROW> | Bison | 3.8+ | <ROW> | Flex | 2.6.4+ |
P: Most Linux distributions ship Flex but an older Bison. If the Bison on your system is below 3.8, the setup step below fetches a compatible version on its own, so the distribution package does not have to be replaced by hand.

=== UID:F22-S01

=== UID:F22-S03
P: The version floor is not arbitrary. GCC 8.1 and Clang 6 are the first releases with complete C++17 support, and Bison 3.8 and Flex 2.6.4 are the versions whose generated code supports the C++ features the compiler uses. An older Bison is the usual obstacle, and it is the one the setup step resolves for you.

=== UID:F02-S03
P: Clone the repository and run the automated setup:
C: Clone and prepare
P: `make setup-`core pulls git submodules and downloads the required versions of Flex, Bison, FileCheck, and GoogleTest if they are not already present. Once setup finishes, build the compiler:
C: Build the compiler
P: Then run the test suite; it is the check that the freshly built compiler is sound:
C: Run the test suite
P: If all tests pass, the `croq` binary is ready in your build directory. Add it to your `PATH` or invoke it by its full path. The next step compiles a minimal program to confirm the installation end to end.

=== UID:F22-S02

=== UID:F02-S04
P: The smallest useful check is a kernel that copies four integers. Save it as `verify.co`:
C: A minimal Croqtile program
P: Compile and run:
C: Compile and run the check
P: You should see `OK`. If you do, the compiler, linker, and runtime are all working. 1.3 lists the options that change what the command produces.

=== UID:F03-S01
P: The shortest program that shows the shape of the language is an element-wise addition of two small matrices. It has no tiling, no explicit data movement and no coordination between regions; those are the constructs every later example adds. What it does have is the full skeleton of a Croqtile program: a `__co__` function that describes the parallel work, and an ordinary C++ `main()` that prepares the inputs, calls it and checks the result. Every program in the guide is built from these two halves.

=== UID:F03-S03
H: A Complete Example: Element-Wise Addition
P: Here is the full program. It adds two `[4, 8]` matrices of 32-bit integers element by element:
C: A complete element-wise addition
P: Save this as `ele_add.co`, compile, and run:
C: Compile and run the example
P: You should see `Test Passed`. The chapters that follow take this same program apart, one construct at a time.

=== UID:F03-S06
F: Compiling and running ele_add, followed by compiler help

=== UID:F02-S05
H: Compiler Options
P: After running `ele_add`, you can use the same command pattern for other `.co` files. Croqtile first generates target source code, then invokes the target toolchain to compile and link it. By default the output is `a.out`; `-o` gives it an explicit name.
P: The following options select the output form or expose a step in this process:
T: | **Option** | **Purpose** | <ROW> | --- | --- | <ROW> | `-`o <file> | Set the output filename. | <ROW> | `-`t <platform> | Select the target platform; list supported targets with `--help-target`. | <ROW> | `-es` | Emit target source without compiling it further. | <ROW> | `-E` | Emit preprocessed Croqtile source only. | <ROW> | `-c` | Compile to an object file without linking. | <ROW> | `-S` | Emit assembly when supported by the target toolchain. | <ROW> | `-gs` | Generate a script for subsequent compilation and execution. | <ROW> | `-v` / `--verbose` | Show the external commands invoked by the compiler. | <ROW> | `--help` / `--help-hidden` | List standard or advanced compiler options. |

=== UID:F22-S05
P: For example, you can inspect the generated source or the result of preprocessing before building an executable:
C: Inspecting intermediate output and compiler help
P: Replace `target` with a name reported by `croq -`-help-target. Available output forms depend on the selected toolchain. Use `-v` when you need to see the actual compilation commands; Section 1.4 covers further build diagnostics.

=== UID:F02-S06
H: Development Utilities
P: The commands above build your `.co` program. The repository's Makefile commands are useful when testing sample operators or changing the compiler itself:
C: Testing sample operators
P: `make help` lists the available targets. `make sample-`test runs all sample operators; `OPERATOR=add` restricts the run to addition. Other operator names include `mul`, `relu`, `sigmoid`, `softmax`, and `tanh`. With the program built, Section 1.4 explains how to inspect the compilation commands and check its execution.

=== UID:F22-S04

=== UID:F11-S05
H: Verbose mode: `-v`
P: Pass `-v` (or `--verbose`) to the compiler to see the external commands it invokes --- which preprocessor, which code generator, which `c++` invocation:
C: Verbose mode: -v
P: This is useful when you suspect the compiler is passing wrong flags to `c++` or when you need to see the exact generated file paths for manual inspection.

=== UID:F11-S06
H: Runtime checks: `-rtc`
P: The compiler supports graduated runtime checking with `-rtc` (or `-`-runtime-check):
C: Runtime checks: -rtc
P: Levels: `none`, `entry` (default), `low`, `medium`, `high`, `all`. Each level is a superset of the previous. Use `high` or `all` during development, `none` for production.

=== UID:F11-S08
H: Debug RTTI and `debugger`
P: When `print` / `println` is not enough, Croqtile supports debug RTTI (Runtime Type Information) that makes its types visible to `debugger`.
P: Compile with `-g` to enable debug symbols:
C: Debug RTTI and debugger
P: The generated code includes RTTI structs for Croqtile types:
T: | **Croqtile type** | **GDB type** | **Fields** | <ROW> | --- | --- | --- | <ROW> | Shaped tensor (`s32 [M, N]`) | `croq:`:rtti::spanned<int, 2> | `.span.data[]` (dimensions), `.stride.data[]` (strides), `.data` (pointer) | <ROW> | Index tuple | `croq:`:rtti::bounded_ituple<N> | `.data[]` (values), `.ub[]` (upper bounds) | <ROW> | Integer tuple | `croq:`:rtti::ituple<N> | `.data[]` (values) | <ROW> | Multi-dim span | `croq:`:rtti::mdspan<N> | `.data[]` (extents) |
C: Debug RTTI and debugger
P: The `__dbg_` prefix on variable names is generated by the compiler --- it makes Croqtile variables visible alongside the generated C++ intermediates.

=== UID:F11-S09
H: Putting it together
P: Choose the first check from the symptom. If a tile has the wrong dimensions, inspect its shape with `println!` before running the program. If the shape is right but the values are wrong, print one tile or add an assertion near the suspect operation. A stalled pipeline calls for event and iteration counts; a regular pattern of incorrect matrix elements calls for a layout check. Use a debugger when these checks point to an address or lifetime problem. Keep temporary output behind a preprocessor guard so it can be removed from a normal build:
C: Guarding debug prints
C: Debug and production builds

=== UID:F11-S07
H: Debugging MMA-heavy kernels
P: When the wrong answer comes from a tensor-core path, suspect **layout first** --- row versus column major, and whether the RHS is `[N, K]` versus `[K, N]` in memory. Then check **indexing** (which `block_m`, `block_n`, and K slice you attached with `.at` / `chunkat`).
P: A common mistake is mislabeling `mma.row.row` when the staged data is actually column-major, or using `chunkat` indices that do not align with the MMA tile geometry. If you use swizzled loads (`tma.copy.swiz` / `mma.load.swiz`), a mismatch shows up as a regular pattern in the error (e.g., every sixteenth element correct, the rest shifted).

=== UID:F08-S13
H: Debugging tip
P: Before reaching for the compiler's inspection flags, look at the program itself. If something looks wrong after editing a pipeline, verify **event order and trip counts** before chasing MMA layout bugs: producer and consumer must use the same `cdiv(K, TILE_`K) loop bound, and too few stages shifts pressure to `wait full` when the consumer outruns the producer.

=== UID:F11-S10
H: In short
T: | **Tool** | **Level** | **Cost** | <ROW> | --- | --- | --- | <ROW> | `print!` / `println!` | Compile-time | Free | <ROW> | `assert(expr, "msg")` | Runtime | Low --- aborts on violation | <ROW> | `print` / `println` | Runtime | Medium --- serialized `printf` | <ROW> | `-rtc=high` | Runtime | Medium --- bounds checks | <ROW> | `-v` / `--verbose` | Compiler | Free --- shows subprocess invocations | <ROW> | `-g` + `debugger` + RTTI | Runtime | High --- debug symbols, no optimization |
P: These tools observe different stages of the same program. Compiler output explains the shapes and types Croqtile inferred; runtime checks expose the values used in an actual call. Verbose mode is useful when the build itself fails, because it shows which external command failed and how to reproduce it. A successful build and a correct result are separate checks.
P: Once the small example passes, apply the same checks to a kernel you want to change. Keep a reference result, change one part of the program, and compare the new output before measuring performance. This makes it possible to connect a regression to a specific change.

