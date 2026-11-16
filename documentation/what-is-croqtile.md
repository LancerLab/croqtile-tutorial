# What is Good in Croqtile?

Croqtile is a C++ embedded DSL for writing high-performance kernels. You write Croqtile functions in `.co` files alongside your existing C++ code, and the compiler transpiles them into efficient target code, with full interoperability with C++ and its libraries.

Four design pillars set Croqtile apart from lower-level approaches.

## Easy to Use

You work on tensors and tiles, not raw buffers and pointers. DMA transfers, memory allocation, and synchronization that take dozens of lines of hand-written target code reduce to a single `dma.copy ... => ...` statement. Every construct lowers to the same code you would write by hand: zero-cost abstraction, no runtime overhead, no hidden allocations.

## Compile-Time Safety

Compile-time diagnostic checks catch shape mismatches, tiling errors, and DMA violations before any code runs. In development builds, runtime assertions guard every transfer and memory access. Together they eliminate entire classes of bugs that would otherwise need a memory checker and hours of printf debugging.

## Dynamic Shapes

First-class symbolic dimensions let you write a kernel once and run it on any shape without recompilation. The compiler derives packed-K, metadata columns, grid dimensions, and staged buffer size automatically. Static and runtime memory are unified — no template metaprogramming, no boilerplate.

## Born for AI Tuning

Compact, structured syntax keeps entire kernels inside AI context windows (30–60 lines instead of hundreds). Structured error messages and well-documented CLI arguments let autonomous agents compile, profile, and iterate without human intervention.
