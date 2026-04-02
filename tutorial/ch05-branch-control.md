# Role Specialization and Control Flow

Chapter 4's matmul kernels have a structural limitation: every thread in the block runs the same program — load, multiply, store, repeat. While the tensor cores are busy multiplying, the memory system idles. While DMA fetches the next K-slab, the tensor cores have nothing to do. Load and compute take turns; neither runs at full throughput.

This is not a GPU-specific problem. It is the universal bottleneck of **pipeline-stage serialization**. Any system where a single worker alternates between two activities — reading data and processing data — wastes time during each transition. The classic solution, whether in CPU thread pools, Unix pipelines, or factory assembly lines, is **role specialization**: assign different workers to different stages, let each worker focus on one job, and let the stages overlap in time.

The GPU twist is that CUDA's default execution model is **SPMD** — Single Program, Multiple Data. Every thread in a warp runs the same instruction stream. Overlapping memory and compute within a single kernel requires either instruction-level interleaving (which the hardware scheduler does poorly for large tiles) or explicit **warp specialization** where the programmer manually partitions warps and coordinates them with barriers. In raw CUDA, this means raw `__syncthreads()`, shared-memory flags, and careful reasoning about which warp is doing what.

Croktile takes a different approach. Instead of leaving role boundaries implicit in control-flow tricks, it makes them a **first-class language construct**: `inthreads.async` assigns different instruction streams to different thread subsets at compile time. The compiler can emit truly separate programs for each role, and the runtime schedules them concurrently. The figure below shows the difference:

![Uniform vs role-specialized execution: sequential alternation vs overlapping stages](../assets/images/ch05/fig1_role_comparison_dark.png#only-dark)
![Uniform vs role-specialized execution: sequential alternation vs overlapping stages](../assets/images/ch05/fig1_role_comparison_light.png#only-light)

*Left: one warpgroup alternates DMA and MMA — no overlap. Right: two warpgroups with static roles — producer DMA and consumer MMA execute concurrently, roughly halving wall-clock time.*

Croktile provides three control-flow primitives, each for a different purpose:

- **`inthreads.async`** — **static role partitioning**: compile-time assignment of different programs to different thread subsets. Analogous to MPMD (Multiple Program, Multiple Data) within a single kernel.
- **`if`** — **guarded execution**: runtime predication where all threads evaluate a condition and divergent threads are masked. Standard SPMD control flow.
- **`shared event`** / **`wait`** / **`trigger`** — **inter-role signaling**: the coordination mechanism that lets statically-partitioned roles communicate safely.

## Static role partitioning with `inthreads.async`

`inthreads.async (condition)` means: only threads for which `condition` is true **have this block in their program at all**. It is not "every thread evaluates the condition and some skip the body" — that is what `if` does. The distinction is fundamental:

- **`inthreads.async`**: the compiler generates separate instruction streams for each role. Threads in the false subset never see the body — it is not in their binary. The `.async` suffix means the roles execute **concurrently and independently** on different hardware resources (different warpgroups, different functional units).
- **`if`**: one instruction stream, one program. All threads evaluate the predicate; threads where it is false are masked. Divergence within a warp stalls one side.

Why both? Because they solve different problems. `inthreads.async` is for **persistent role assignment** — the producer stays a producer for the entire kernel lifetime. `if` is for **data-dependent decisions** — skip this tile if the index is out of bounds. You would not use `if` for warp specialization (it does not produce real concurrency), and you would not use `inthreads.async` for a bounds check (it is a compile-time construct, not a runtime predicate).

Without `.async`, `inthreads` would mean sequential, blocking role execution — thread subsets take turns. The `.async` modifier is what enables the overlapping pipeline execution in the figure above.

The canonical pattern is **one producer + one consumer (1P1C)** for matmul:

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // producer: only warpgroup 0 runs this
    // issue DMA / TMA loads, fill shared memory
  }

  inthreads.async (p1 == 1) {
    // consumer: only warpgroup 1 runs this
    // run MMA on shared memory, accumulate results
  }
}
```

**`parallel p1 by 2 : group-4`** — Two warpgroups, four warps each (128 threads per warpgroup), indexed by `p1`.

**`inthreads.async (p1 == 0)`** — Warpgroup 0 compiles and executes the producer body; warpgroup 1 never sees this code.

**`inthreads.async (p1 == 1)`** — Warpgroup 1 runs the consumer body. The two blocks are separate programs sharing an address space.

## Coordinating roles: events at a glance

Static role partitioning gives you two concurrent programs — but they share the same shared memory. The producer writes data that the consumer reads. Without coordination, the consumer might read a tile before the producer has finished writing it (a race condition), or the producer might overwrite a tile the consumer is still reading (a data hazard).

Croktile uses **events** as the signaling mechanism between roles. An event is a lightweight synchronization token declared in shared memory:

```choreo
shared event full;
shared event empty;
```

The producer calls `trigger full` after writing a tile to signal "data is ready." The consumer calls `wait full` before reading, blocking until the signal arrives. Symmetrically, the consumer triggers `empty` after it finishes reading (the buffer can be reused), and the producer waits on `empty` before writing the next tile.

This is a **credit-based bounded buffer** protocol — the same pattern used in network flow control and OS bounded queues. `full` is the "data available" credit; `empty` is the "buffer free" credit.

Chapter 6 develops this into the full double-buffering and software-pipelined matmul. For now, the important point is that events are the glue between `inthreads.async` roles — they turn two independent programs into a coordinated pipeline.

## Role specialization in practice: a 1P1C matmul

Here is how the split sits inside a Hopper matmul. Event-based synchronization is omitted on purpose; [Chapter 6](ch06-synchronization.md) adds the full pipeline protocol. Focus on who does what:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        // Producer: walk K, load tiles into shared
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        // Consumer: walk K, MMA on loaded tiles
        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**Producer `foreach`** — Walks the K dimension with `cdiv(K, MATMUL_TILE_K)` steps; only warpgroup 0 issues `dma.copy` into `lhs_load_s` and `rhs_load_s`.

**Consumer `mma.fill` / `mma.row.row` / `mma.store`** — Warpgroup 1 never issues those DMA fills; it only reads shared memory, accumulates in `mc`, and writes the result tile.

**Missing coordination** — The two sides both loop over K independently. The consumer assumes each K-slab is ready when it reads; making that assumption true is synchronization ([Chapter 6](ch06-synchronization.md)).

## Guarded execution with `if`

Sometimes you need a predicate every thread evaluates at runtime. Croktile's `if` behaves like C:

```choreo
if (tile_id < total_tiles) {
  // only execute this body when the condition is true
}
```

All threads in the scope test the condition; threads where it is false skip the body. This is the opposite of `inthreads.async`: one program, divergent execution — not two separate programs.

This shows up most often in **persistent kernels**, where the number of loop iterations can leave some blocks with a "padding" iteration that does not correspond to a real tile.

## Persistent scheduling and the `if` guard

In Chapters 3–4, the grid grew with the problem: roughly one block per output tile. For large matrices that can mean huge launch counts. The GPU runs blocks in **waves**; the last wave often leaves SMs partially idle — **tail underutilization**.

A **persistent kernel** fixes the launch size (often near the SM count) and lets each block iterate over multiple tiles:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);

  parallel block_id by NUM_SMS : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
      tile_id = tile_iter # block_id;

      if (tile_id < total_tiles) {
        block_m = tile_id / cdiv(N, MATMUL_WARP_N);
        block_n = tile_id % cdiv(N, MATMUL_WARP_N);

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k) => rhs_load_s;

          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            parallel p by 1 : group-4 {
              ma = mma.load lhs_load_s.chunkat(_, iv_warp);
              mb = mma.load rhs_load_s.chunkat(_, iv_warp);
              mma.row.row mc, ma, mb;
            }
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**`parallel block_id by NUM_SMS : block`** — Fixed worker count; `block_id` names which persistent worker this is, not a single output tile.

**`foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)]`** — Each block steps through its share of iterations; the ceiling may add one extra iteration for some blocks.

**`tile_id = tile_iter # block_id`** — Composes iteration with block index to stripe across the linear tile list (same `#` operator as Chapter 2, used here for scheduling).

**`if (tile_id < total_tiles)`** — Skips DMA, MMA, and store when the stripe walks past the last real tile. This is a runtime guard, not a role split.

![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_dark.png#only-dark)
![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_light.png#only-light)

### Data-dependent vs persistent grids

| Aspect | One block per tile | Persistent (`NUM_SMS` blocks) |
|--------|-------------------|-------------------------------|
| Grid size | Grows with problem | Fixed |
| Tail utilization | Last wave may leave SMs idle | All SMs stay busy |
| Extra constructs | Minimal | `total_tiles`, `tile_iter # block_id`, `if` |
| Complexity | Lower | Higher |

Neither layout changes the mathematical result by itself; both match modulo floating-point associativity. Persistent scheduling tends to pay off when `total_tiles` is much larger than the number of SMs.

## `parallel.async` and `stream s`: non-blocking launch

Everything above runs inside the kernel. Sometimes the control you need is at the **host level**: launch a grid without blocking the host thread, or pin different grids to different CUDA streams so they can execute concurrently on the GPU.

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

**`parallel.async`** returns control to the host immediately — the kernel is enqueued but the host does not wait for it to finish. This is the Croktile equivalent of `cudaLaunchKernel` with a non-default stream.

**`stream s`** inside the block body pins the kernel to CUDA stream `s`. Multiple `parallel.async` blocks with different streams can overlap on the GPU if there are enough SMs. Without `stream s`, the default stream serializes launches.

This is **host orchestration**, not in-kernel control flow. It does not replace `inthreads.async` for role partitioning or `if` for runtime predicates — it decides *when* and *where* a grid runs relative to other grids.

## New syntax

| Syntax | Meaning |
|--------|---------|
| `inthreads.async (condition)` | Static role partitioning — only threads satisfying `condition` include this block |
| `if (expr) { ... }` | Guarded execution — runtime conditional, skip body when `expr` is false |
| `shared event name` | Declare an event token in shared memory |
| `trigger name` | Signal that a condition is met (e.g., "data ready") |
| `wait name` | Block until the corresponding `trigger` fires |
| `tile_id = tile_iter # block_id` | Compose iteration index with block index for tile striping |
| `int total_tiles = expr` | Local integer in a Croktile function |
| `parallel.async ... : block` | Non-blocking async kernel launch |
| `stream s` | Bind kernel body to CUDA stream `s` |

## Chapter summary

| Topic | Takeaway |
|-------|----------|
| Pipeline serialization | One worker alternating load/compute wastes time; role specialization overlaps stages |
| Static role partitioning | `inthreads.async` — compile-time MPMD within a kernel; separate programs for separate thread subsets |
| Guarded execution | `if` — runtime SPMD predication; all threads evaluate, divergent threads are masked |
| Events (preview) | `shared event` / `wait` / `trigger` — inter-role signaling; credit-based bounded buffer protocol |
| Persistent kernels | Fixed `NUM_SMS` blocks, linear tile ids, striping with `#`, guard with `if` |
| Host orchestration | `parallel.async` / `stream s` — orthogonal to in-kernel specialization |

The 1P1C skeleton above is incomplete: without `wait` / `trigger`, the consumer can read before the producer has written. [Chapter 6](ch06-synchronization.md) adds the full synchronization protocol — events, `swap`, and double-buffering — so the pipeline runs safely and at full throughput.
