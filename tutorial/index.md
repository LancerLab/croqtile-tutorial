# Croqtile Tutorial

Welcome to the Croqtile tutorial. This guide walks you through writing high-performance kernels using Croqtile, starting from scratch and building up to production-grade patterns.

Each chapter introduces a small set of new concepts by evolving a running example. By the end, you will have encountered every major Croqtile construct in a concrete, working program. For detailed syntax design and language reference, see the [Coding Reference](../documentation/index.md).

## Chapters

0. [Installation: Setting Up the Croqtile Compiler](ch00-installation.md)
1. [Hello Croqtile: From Zero to Running Kernel](ch01-hello-croqtile.md)
2. [Data Movement: Tiles Instead of Elements](ch02-data-movement.md)
3. [Parallelism: Mapping Work to Levels](ch03-parallelism.md)
4. [Tensor Contraction: The `mma` Operations](ch04-mma.md)
5. [Branch and Control: Roles and Persistent Kernels](ch05-branch-control.md)
6. [Synchronization: Pipelines, Events, and Double Buffering](ch06-synchronization.md)
7. [Advanced Data Movement: Swizzle and Irregular Access](ch07-advanced-movement.md)
8. [C++ Interop: Inline Code and the Preprocessor](ch08-cpp-interop.md)
9. [Debug and Verbose: Printing, RTTI, and GDB](ch09-debug-verbose.md)

## Prerequisites

- Basic C++ knowledge (functions, pointers, arrays)
- Familiarity with parallel programming concepts (threads, blocks, staged buffers)
- A working Croqtile compiler (see [Chapter 0](ch00-installation.md))
