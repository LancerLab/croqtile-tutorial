# Material moved out of the Programming Guide

These are the exact upstream passages removed while assembling the
language-only guide, keyed by the content-unit IDs in
`UNIT-REGISTRY.md`. Nothing here is rewritten: it is the raw source
text, kept so that the hardware/architecture material can become its
own document.

## F01-S01

- **prose**:

    This guide walks you through writing high-performance GPU kernels using Croqtile, starting from scratch and building up to production-grade patterns.

## F01-S02

- **list**:

    Parallelism: Mapping Work to Hardware

- **list**:

    Tensor Cores: The QZ000QZ Operations

- **list**:

    Branch and Control: Warp Roles and Persistent Kernels

## F01-S03

- **list**:

    Familiarity with GPU programming concepts (threads, blocks, shared memory)

## F02-S02

- **table**:

    CUDA Toolkit

- **table**:

    11.8+ (for GPU targets)

- **table row**:

    CUDA Toolkit | 11.8+ (for GPU targets)

## F03-S02

- **list**:

    The compiler transpiles this into GPU-ready code.

## F03-S04

- **prose**:

    If you have written CUDA before, you might wonder: are these instances CUDA threads? Thread blocks? Under the hood, the 32 instances here become 32 CUDA threads inside a single thread block In QZ005QZ you will learn to control the mapping explicitly with QZ006QZ like QZ001QZ and QZ002QZ, nest multiple QZ003QZ axes to build a block-and-thread hierarchy, and map them onto warps and warpgroups. For now, treat QZ004QZ as "run all instances concurrently on the GPU" and move on.

## F03-S06

- **prose**:

    The QZ000QZ flag stops after transpilation, letting you inspect the generated CUDA code.

## F03-S07

- **prose**:

    The example works, but it processes one element at a time with no control over how data moves through the memory hierarchy. On real hardware that matters GPUs achieve peak throughput by moving data in bulk, not element by element.

## F04-S01

- **prose**:

    That is a good way to explain correctness, but it is not how the hardware wants to move data. GPUs fetch and stage contiguous blocks, not isolated elements. The hard part of GPU programming is usually not the arithmetic. It is organizing those block transfers so the compute units see data in the right memory level, in the right shape, at the right time.

- **list**:

    QZ000QZ uses Hopper's Tensor Memory Accelerator for the same style of transfers when the tile shape is compatible.

## F04-S02

- **table**:

    QZ000QZ for software cooperative copies, QZ001QZ for Hopper TMA copies.

- **table row**:

    `engine` | `dma` for software cooperative copies, `tma` for Hopper TMA copies.

## F04-S03

- **prose**:

    Here is the same addition from Chapter 1, but staged through shared memory in tiles:

- **list**:

    The inner loop reads from QZ000QZ and QZ001QZ, not from global memory.

## F04-S17

- **whole section**:

    ## Memory Specifiers: `global`, `shared`, and `local`
    
    The destination of a movement statement also states where the moved tile should live.
    
    - `global`: device memory. Large, slow, visible across the device.
    - `shared`: block-visible on-chip memory. This is the usual staging area for cooperative tiles.
    - `local`: thread-private storage. Good for small private tiles or non-cooperative use.
    
    ```choreo
    f0 = dma.copy input => shared;
    f1 = dma.copy matrix_tile => local;
    dma.copy out_tile => output;
    ```
    
    For performance-sensitive kernels, `shared` is the important level. The compiler's cooperative tiled lowering is aimed at global-to-shared and shared-to-global movement. `local` copies are still useful, but they are not the main fast path for cooperative block transfers.
    
    ![Memory Specifiers → GPU Hardware](../assets/images/ch02/fig_memory_hierarchy_dark.png#only-dark)
    ![Memory Specifiers → GPU Hardware](../assets/images/ch02/fig_memory_hierarchy_light.png#only-light)
    
    *Croqtile's memory specifiers map directly onto the GPU memory hierarchy.*
    

## F04-S19

- **prose**:

    Swizzle is mainly about how the tile should live in shared memory so later operations can read it efficiently.

## F04-S20

- **table**:

    Hopper Tensor Memory Accelerator

- **table row**:

    Implementation | Cooperative software copy | Hopper Tensor Memory Accelerator

- **table**:

    Hardware requirement

- **table**:

    General GPU path

- **table**:

    SM90+

- **table row**:

    Hardware requirement | General GPU path | SM90+

- **table**:

    Fixed-shape high-bandwidth staging

- **table row**:

    Good fit | Dynamic or irregular tiles, portable code | Fixed-shape high-bandwidth staging

- **list**:

    Use QZ000QZ on Hopper when the tile shape is fixed and you want hardware-assisted movement.

## F04-S22

- **whole section**:

    ## Writing DMA and TMA That Lower Well
    
    You do not need to know the whole lowering pass to write good code, but a few rules matter.
    

## F04-S23

- **whole section**:

    ### For high-throughput DMA
    
    - Prefer `global <-> shared` movement for cooperative staging.
    - Use enough participating threads. A warp-sized or larger cooperative group is the common case.
    - Keep source and destination ranks aligned.
    - Use ordinary byte-sized or larger element types.
    - Reserve `local` for private tiles, not as the default performance path.
    
    If these conditions do not hold, the compiler can still generate a correct copy, but it may fall back to a simpler path.
    

## F04-S24

- **whole section**:

    ### For TMA
    
    - Use it on SM90+.
    - Keep the tile shape fixed from the host's point of view.
    - Stage into shared memory.
    - Add `.zfill` for partial tiles that feed full-tile consumers.
    

## F04-S25

- **table**:

    Move a tile into shared memory.

- **table row**:

    `dma.copy src => shared` | Move a tile into shared memory.

- **table**:

    Use Hopper TMA to move a tile into shared memory.

- **table row**:

    `tma.copy src => shared` | Use Hopper TMA to move a tile into shared memory.

## F05-S01

- **prose**:

    But a GPU is not a flat bag of identical processors. It has a deep hierarchy of execution units, and mapping your work to the right level of that hierarchy is the difference between a toy demo and a real kernel.

- **prose**:

    This chapter introduces parallelism as a first-class concept in Croqtile: what it means, how Croqtile separates the QZ001QZ structure (what runs concurrently) from the QZ002QZ mapping (which hardware units run it), and how you control that mapping with QZ000QZ.

## F05-S02

- **prose**:

    On a GPU warp, all eight might genuinely execute in lockstep.

- **figure caption**:

    The same eight tasks scheduled sequentially (1 core), 4-wide (CPU), and 8-wide (GPU). only the hardware changes.

- **prose**:

    When you write "these eight tasks are independent," you are making a statement about the QZ001QZ of your program, not about the hardware. The hardware decides how many actually run simultaneously, based on how many execution units it has and how the scheduler assigns work.

- **prose**:

    This distinction matters because GPU programming languages traditionally conflate the two. In CUDA, you launch a kernel with QZ000QZ that is simultaneously declaring the logical structure (how many instances) and the physical mapping (how many thread blocks, how many threads per block).

## F05-S03

- **list**:

    which GPU hardware unit

- **code comment**:

    concurrent -- mapped to thread blocks

- **prose**:

    The QZ000QZ lines say "these iterations are independent." The QZ001QZ and QZ002QZ annotations say "put the outer ones on separate thread blocks and the inner ones on separate threads within a block." The QZ003QZ lines say "run these in order." No ambiguity, no conflation.

- **figure caption**:

    Right: the GPU hardware hierarchy.

## F05-S04

- **whole section**:

    ## Space Specifiers and Hardware Mapping
    
    A GPU has a clear execution hierarchy. From coarsest to finest:
    
    - A **thread block** (CTA) is a group of up to 1024 threads that share on-chip memory and can synchronize with each other.
    - A **warpgroup** is four warps — 128 threads — that cooperate on wide matrix instructions (WGMMA on Hopper GPUs).
    - A **warp** is 32 threads that execute in lockstep. This is the GPU's fundamental SIMD unit.
    - A **thread** is a single execution context with its own registers.
    
    Croqtile maps to each level with a space specifier:
    
    ![Space specifiers mapped to GPU execution hierarchy](../assets/images/ch03/fig3_space_specifiers_dark.png#only-dark)
    ![Space specifiers mapped to GPU execution hierarchy](../assets/images/ch03/fig3_space_specifiers_light.png#only-light)
    
    *The four space specifiers and the GPU hardware units they map to, with thread counts and typical operations at each level.*
    

## F05-S05

- **prose**:

    Each block runs on a streaming multiprocessor (SM) and has its own allocation of shared memory.

## F05-S06

- **prose**:

    Each thread has its own registers but shares on-chip memory with all other threads in the same block.

## F05-S07

- **whole section**:

    ### `: group` — Warps
    
    ```choreo
    parallel w by 4 : group
    ```
    
    This creates 4 warps, each of 32 threads. Warps execute in lockstep (all 32 threads run the same instruction simultaneously). Warp-level operations like `mma.sync` (Ampere-era tensor core instructions) operate at this granularity.
    

## F05-S08

- **prose**:

    This creates 2 warpgroups, each containing 4 warps (128 threads). On Hopper GPUs (compute capability 9.0+), WGMMA instructions require a full warpgroup to cooperate. Even when you have just one warpgroup (QZ000QZ), the annotation tells the compiler that these 128 threads form a cooperative unit.

- **prose**:

    Multiple warpgroups per block is how you scale work along one dimension without adding more blocks both warpgroups share the same shared memory but maintain independent accumulators.

## F05-S10

- **whole section**:

    ### How `shared` Enables Reuse
    
    Space specifiers interact directly with memory specifiers. Recall from Chapter 2 that `dma.copy ... => shared` places data in block-scoped shared memory, while `=> local` places data in thread-private local memory.
    
    The choice matters because of data reuse. Consider a tile that every thread in a block needs to read:
    
    ![Local vs shared memory reuse](../assets/images/ch03/fig4_shared_reuse_dark.png#only-dark)
    ![Local vs shared memory reuse](../assets/images/ch03/fig4_shared_reuse_light.png#only-light)
    
    *Left: with `=> local`, each thread loads its own copy — 4× the bandwidth. Right: with `=> shared`, one DMA fills shared memory and all threads read from it — 1× the bandwidth.*
    
    The rule of thumb: if multiple threads in a block need the same data, copy it into `shared`. If each thread's working set is independent, `local` keeps data closer and avoids shared-memory bank contention.
    
    In the matmul later in this chapter, the K-tiles of `lhs` and `rhs` are loaded `=> shared` because all 256 threads in the block read from them. The output accumulator stays in registers (thread-private) because each thread writes to its own element.
    

## F05-S12

- **prose**:

    This is the first program in the tutorial that does something GPUs are famous for.

## F05-S13

- **prose**:

    global memory, no explicit data movement:

- **prose**:

    Every QZ001QZ here reads from global memory.

## F05-S14

- **prose**:

    The scalar version works but every multiply reads from global memory. Bringing K-tiles into shared memory lets all threads in a block reuse them:

- **figure caption**:

    One block in detail: the K-loop loads lhs and rhs tiles into shared memory via QZ000QZ, then 256 threads compute partial products from the shared copies.

- **figure caption**:

    One block in detail: the K-loop loads lhs and rhs tiles into shared memory via QZ000QZ, then 256 threads compute partial products from the shared copies.

## F05-S15

- **whole section**:

    ### GPU Resource Layout
    
    Here is how the matmul code maps to actual GPU hardware:
    
    ![Matmul code mapped to GPU resources](../assets/images/ch03/fig7_matmul_gpu_layout_dark.png#only-dark)
    ![Matmul code mapped to GPU resources](../assets/images/ch03/fig7_matmul_gpu_layout_light.png#only-light)
    
    *Left: Croqtile code with nested parallel and foreach. Right: GPU hardware — 128 blocks distributed across SMs, each with shared memory and 256 threads.*
    

## F05-S18

- **table**:

    Map to CUDA thread blocks

- **table row**:

    `parallel p by N : block` | Map to CUDA thread blocks

- **table**:

    Map to warps (32 threads each)

- **table row**:

    `parallel w by N : group` | Map to warps (32 threads each)

- **table**:

    Map to warpgroups (128 threads each)

- **table row**:

    `parallel g by N : group-4` | Map to warpgroups (128 threads each)

- **table**:

    DMA destination: block-scoped shared memory

- **table row**:

    `=> shared` | DMA destination: block-scoped shared memory

- **prose**:

    The tiled DMA matmul you built in this chapter is the structural backbone of every high-performance GPU kernel. hardware-accelerated matrix multiply that processes a 16×16×16 tile in a single instruction.

## F06-S01

- **prose**:

    It is also the slowest way to use a modern accelerator, because it ignores the hardware unit designed specifically for this job.

- **prose**:

    The operation is so central that hardware vendors build dedicated units to execute it in one macro-instruction: NVIDIA calls them QZ001QZ, Google calls them QZ002QZ, Intel has QZ003QZ, and custom DSAs have their own variants. The tile sizes differ (16×16×16 for FP16 on NVIDIA, 128×128 systolic on TPU, 16×64 on AMX), but the mathematical shape is the same everywhere: take a tile of A, a tile of B, multiply, accumulate into C.

- **prose**:

    tensor cores

- **prose**:

    AMX tiles

- **prose**:

    ![2D tensor contraction: A[M,K] × B[K,N] → C[M,N], with different hardware implementations](../assets/images/ch04/fig1_tensor_contraction_dark.png#only-dark) ![2D tensor contraction: A[M,K] × B[K,N] → C[M,N], with different hardware implementations](../assets/images/ch04/fig1_tensor_contraction_light.png#only-light)

- **figure caption**:

    maps to different hardware implementations on different accelerators.

- **prose**:

    On a GPU tensor core, the tile is not stored contiguously in one thread's registers. It is QZ005QZ: 32 threads in a warp each own scattered pieces of the tile, and the exact scatter pattern depends on the datatype, the architecture generation, and whether the operand is A, B, or C. Writing raw CUDA means declaring QZ000QZ objects, calling QZ001QZ to distribute a shared-memory tile across lanes with the correct pattern, issuing QZ002QZ, and then calling QZ003QZ to reassemble the output.

- **prose**:

    register layout

- **figure caption**:

    Simplified view of how 32 threads in a warp own scattered register fragments of an MMA tile. The exact pattern is architecture-specific and deliberately opaque.

- **figure caption**:

    Simplified view of how 32 threads in a warp own scattered register fragments of an MMA tile. The exact pattern is architecture-specific and deliberately opaque.

- **prose**:

    Instead of exposing architecture-specific fragment types, it provides QZ000QZ that work on opaque register tiles: QZ001QZ, QZ002QZ, QZ003QZ, and QZ004QZ. These operations describe the same 2D contraction workflow regardless of which hardware backend runs them. The compiler handles fragment layouts, lane mappings, and instruction selection for the target architecture you describe QZ005QZ contraction to perform, not QZ006QZ registers are scattered.

- **figure caption**:

    not hardwired to GPU tensor cores. Any DSA that supports 2D tile contraction can map to these operations.

## F06-S02

- **list**:

    allocate an accumulator tile QZ001QZ in registers and zero it.

- **list**:

    bring operand tiles from shared memory into opaque MMA registers QZ001QZ and QZ002QZ.

- **list**:

    write QZ001QZ from registers into shared memory.

- **prose**:

    The names QZ001QZ, QZ002QZ, and QZ003QZ are opaque register tiles

- **prose**:

    it tells the hardware how to interpret the bits in QZ001QZ and QZ002QZ. If B is stored column-major in shared memory, you write QZ003QZ instead. the hardware interprets register bits differently for each layout.

## F06-S03

- **prose**:

    one warp, one warpgroup, two warpgroups

## F06-S04

- **prose**:

    On Ampere (SM86), tensor-core MMA is scoped to a QZ001QZ (32 threads).

- **prose**:

    single warp

- **prose**:

    results go through QZ001QZ, matching the usual GPU pattern of writing through a global pointer.

- **prose**:

    QZ003QZ QZ001QZ maps MMA tiles to warps. one warp covers the whole block tile. Wider block tiles would add warps, each with its own QZ002QZ.

- **prose**:

    Warp grid and QZ000QZ.

- **prose**:

    QZ005QZ Each QZ000QZ stage pulls A and B strips into shared memory via QZ001QZ with QZ002QZ.

- **prose**:

    QZ004QZ QZ000QZ moves the warp's tile from shared memory into QZ001QZ / QZ002QZ. QZ003QZ selects the M×K slice for this warp and inner-K step.

- **prose**:

    QZ004QZ After K completes, QZ000QZ writes QZ001QZ into the warp's sub-rectangle of QZ002QZ, then QZ003QZ sends the full block tile to global memory.

- **prose**:

    This kernel is simple because the cooperation scope is narrow: 32 threads, one warp, one MMA tile at a time. But what happens when the hardware offers a wider cooperation window?

## F06-S05

- **prose**:

    Hopper (SM90) adds QZ002QZ: the same C += A × B contraction, but issued cooperatively by QZ003QZ (128 threads). The hardware instruction is wider, the tiles are bigger, and throughput improves

- **prose**:

    Warpgroup Matrix Multiply-Accumulate (WGMMA)

- **prose**:

    four warps

- **figure caption**:

    SM86: one warp per MMA. SM90: four warps (QZ000QZ) cooperate on WGMMA.

- **figure caption**:

    SM86: one warp per MMA. SM90: four warps (QZ000QZ) cooperate on WGMMA.

- **prose**:

    Read it side by side with the SM86 kernel.

- **prose**:

    QZ010QZ Hopper often spells accumulator precision explicitly

- **prose**:

    QZ002QZ One warpgroup (four warps) executes the inner loads and MMA. The mnemonic QZ001QZ matches Ampere, but the hardware issue is wider.

- **prose**:

    keep the full M (or N) extent already resident in shared memory;

- **prose**:

    The compiler maps them to different hardware instructions depending on whether the target is SM86 or SM90.

## F06-S06

- **prose**:

    two warpgroups in one block. This is how large block tiles get split into multiple MMA tiles without duplicating the B operand in shared memory:

- **prose**:

    QZ004QZ gives each warpgroup its A rows;

- **prose**:

    QZ003QZ QZ000QZ targets QZ001QZ so each warpgroup writes its half of the staging buffer, then one QZ002QZ emits the combined tile.

- **prose**:

    The pattern scales: three warpgroups, four warpgroups, or any count that divides your block tile.

- **table**:

    One warp(QZ000QZ)

- **table**:

    One warpgroup(QZ000QZ)

- **table**:

    Two warpgroups

- **table**:

    One MMA tile per warp

- **table**:

    One MMA tile per warpgroup

- **table**:

    Block tile split across warpgroups

- **table row**:

    Tile split | One MMA tile per warp | One MMA tile per warpgroup | Block tile split across warpgroups

## F06-S07

- **prose**:

    QZ001QZ When half the elements of A follow a 2:4 zero pattern (Ampere and later), the hardware can skip the zero products and roughly double throughput

- **prose**:

    QZ006QZ When shared memory uses a swizzle pattern to avoid bank conflicts (QZ005QZ), the MMA load must use the matching swizzle mode: QZ000QZ.

- **prose**:

    QZ002QZ In pipelined kernels where producer and consumer warpgroups overlap (QZ001QZ), QZ000QZ marks the boundary between "done reading this K-slab's operands" and "safe to reuse the shared-memory buffer." It is mandatory glue in event-driven pipelines.

- **prose**:

    These extensions all follow the same design principle: the four-step skeleton stays fixed, and the variant suffix communicates a specific contract to the hardware.

## F06-S08

- **table**:

    Load operand tile from shared into MMA registers

- **table row**:

    `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers

- **table**:

    Write accumulator to shared memory

- **table row**:

    `mma.store mc, dst` | Write accumulator to shared memory

- **table**:

    Pipeline stage fence for WGMMA (see QZ000QZ)

- **table row**:

    `mma.commit` | Pipeline stage fence for WGMMA (see [Ch 6](ch06-synchronization.md))

## F06-S09

- **table**:

    Hardware diversity

- **table**:

    GPU tensor cores, TPU MXU, Intel AMX, custom DSAs all implement this; tile sizes and register layouts differ

- **table row**:

    Hardware diversity | GPU tensor cores, TPU MXU, Intel AMX, custom DSAs all implement this; tile sizes and register layouts differ

- **table**:

    QZ000QZ (one warp) → QZ001QZ (one warpgroup) → multiple warpgroups

- **table**:

    — must match data in shared memory

- **prose**:

    while tensor cores multiply, the memory system idles.

## F07-S01

- **prose**:

    On a GPU, the answer is deeply constrained. The SIMT (Single Instruction, Multiple Thread) execution model groups threads into QZ000QZ of 32 that share a single instruction pointer. When a branch sends some threads one way and other threads another way, the hardware cannot simply fork This is the fundamental tension of GPU programming languages: the hardware wants uniformity, but real programs need heterogeneity.

- **prose**:

    warps

- **prose**:

    CUDA exposes exactly one control-flow primitive for this: QZ000QZ. both paths execute sequentially within the warp. A warp that diverges on an QZ001QZ runs both sides, throwing away half its throughput on each.

- **prose**:

    Consider a matmul kernel where one group of threads should continuously fetch tiles from global memory while another group continuously multiplies tiles on the tensor cores.

- **list**:

    the hardware schedules them independently.

- **prose**:

    Together with QZ000QZ, these three primitives cover the full spectrum of control flow in a GPU kernel: data-dependent branching, structural program composition, and inter-program coordination.

## F07-S02

- **prose**:

    Within a single warp, if some threads take the branch and others do not, the hardware QZ000QZ the two paths

- **prose**:

    warp divergence

- **prose**:

    QZ001QZ divergence within a warp serializes both paths. Divergence across warps (where all threads in each warp agree) costs nothing the hardware simply skips the not-taken path.

- **prose**:

    warp-uniform

## F07-S03

- **code comment**:

    program A: only warpgroup 0 compiles and runs this

- **code comment**:

    program B: only warpgroup 1 compiles and runs this

- **table**:

    Serial within a warp if divergent

- **table**:

    Concurrent across warpgroups

- **table row**:

    **Execution** | Serial within a warp if divergent | Concurrent across warpgroups

- **table**:

    GPU analogy

- **table**:

    MPMD within a single kernel launch

- **table row**:

    **GPU analogy** | SPMD with masking | MPMD within a single kernel launch

- **prose**:

    This is what makes it amenable to compile-time analysis: the compiler can allocate registers differently for each region, emit different instruction schedules, and verify that shared resources are used safely.

- **prose**:

    The QZ003QZ suffix is the concurrency modifier: it tells the compiler and hardware that the regions may overlap in time.

- **prose**:

    The top timeline shows a single warpgroup alternating between DMA and MMA (sequential, no overlap). The bottom shows two warpgroups with QZ000QZ

- **figure caption**:

    Top: one warpgroup alternates DMA and MMA

## F07-S04

- **code comment**:

    producer: only warpgroup 0 runs this

- **code comment**:

    issue DMA / TMA loads, fill shared memory

- **code comment**:

    consumer: only warpgroup 1 runs this

- **code comment**:

    run MMA on shared memory, accumulate results

- **prose**:

    Two warpgroups (128 threads each), indexed by QZ001QZ.

- **prose**:

    Warpgroup 0 compiles and runs the producer body; warpgroup 1 never sees this code.

- **prose**:

    Warpgroup 1 runs the consumer body.

## F07-S05

- **prose**:

    lightweight synchronization tokens declared in shared memory:

## F07-S08

- **prose**:

    Here is how the three primitives sit together inside a Hopper matmul.

- **prose**:

    warpgroup 0 issues QZ002QZ into shared memory.

- **prose**:

    Warpgroup 1 never touches those DMAs; it reads shared memory, accumulates in QZ001QZ, and writes the result.

## F07-S11

- **prose**:

    QZ003QZ pins the kernel to CUDA stream QZ001QZ. Multiple QZ002QZ blocks with different streams can overlap on the GPU if there are enough SMs.

## F07-S13

- **table**:

    Declare a synchronization token in shared memory

- **table row**:

    `shared event name` | Declare a synchronization token in shared memory

- **table**:

    Non-blocking kernel launch

- **table row**:

    `parallel.async ... : block` | Non-blocking kernel launch

- **table**:

    Bind kernel to CUDA stream QZ000QZ

- **table row**:

    `stream s` | Bind kernel to CUDA stream `s`

## F07-S14

- **prose**:

    QZ002QZ for single-schedule double buffering, events for multi-warpgroup pipelines so the pipeline runs safely and at full throughput.

## F08-S04

- **prose**:

    Shared-memory contents stay where the hardware placed them; In CUDA, the same idiom is often a QZ001QZ buffer index or a boolean phase variable.

## F08-S09

- **prose**:

    Warp specialization (QZ002QZ) puts them on QZ003QZ warpgroups with different program counters.

## F08-S14

- **table**:

    Fence between pipeline stages for WGMMA

- **table row**:

    `mma.commit` | Fence between pipeline stages for WGMMA

## F08-S15

- **table**:

    Separate producer/consumer warpgroups with multi-stage pipeline

- **table row**:

    1P1C event pipeline | `inthreads.async`, `shared event[]`, `wait`, `trigger`, `mma.commit` | Separate producer/consumer warpgroups with multi-stage pipeline

- **prose**:

    graduate to events when you need warp specialization.

- **prose**:

    The QZ002QZ moves on to hardware-accelerated QZ003QZ, QZ004QZ shared layouts, and QZ005QZ for irregular access

## F09-S01

- **whole section**:

    # Advanced Data Movement: TMA, Swizzle, and Irregular Access
    
    Chapter 2 introduced `dma.copy` as Croqtile's general data movement primitive — a way to move rectangular tiles between memory spaces using a simple `src => dst` arrow syntax. Under the hood, DMA copies are **software-driven**: every thread in the warpgroup participates in address computation and load issuance. The hardware sees dozens of individual load instructions, one per thread, that collectively assemble a tile in shared memory.
    
    NVIDIA's Hopper GPUs (SM90+) add a second mechanism: the **Tensor Memory Accelerator (TMA)**. TMA is a physical hardware unit sitting near the L2 cache / shared memory interface. Instead of threads cooperatively issuing loads, a single thread issues one **descriptor-based** instruction, and the TMA engine handles everything: multi-dimensional address computation, boundary clamping, and the actual data transfer.
    
    A common misconception is that TMA moves data faster than DMA. It does not — both paths ultimately travel the same HBM → L2 → shared-memory data highway at the same bandwidth. The advantage is elsewhere: TMA has its own **dedicated engine** that runs independently of the SM's instruction pipeline. Once the descriptor-based instruction is issued, the TMA hardware performs the transfer in the background while the issuing thread (and all other threads in the warpgroup) are free to execute compute instructions. With `dma.copy`, the threads that participate in address math and load issuance are occupied for the duration of the transfer; with `tma.copy`, they can overlap the transfer with MMA or other work. In a warp-specialized pipeline (Chapter 5–6), this is the difference between a producer that blocks on loads and a producer that fires-and-forgets.
    
    The two paths differ in interface, not in throughput:
    
    - **`dma.copy`** — Threads cooperatively issue loads. The programmer controls nothing about address patterns — Croqtile handles coalescing automatically. Flexible: works on any CUDA GPU since it compiles to standard load instructions.
    - **`tma.copy`** — One descriptor-based instruction. The TMA hardware expands it into multi-dimensional addressing, applies swizzle, and handles boundary clamping. Hopper (SM90+) only. The descriptor is built by the compiler from your `__co__` signature and global layout.
    
    Croqtile exposes TMA through the same arrow syntax as DMA — `tma.copy` instead of `dma.copy`. The rest of this chapter covers TMA, swizzle, the irregular-access utilities for real-world edge cases, and how Croqtile's design philosophy trades generality for guaranteed performance.
    

## F09-S02

- **list**:

    Throughput scales with how well you keep those loads bank-friendly.

- **list**:

    the TMA hardware expands it into multi-dimensional addressing and moves the whole tile as a unit. Producer threads can overlap other work because the hardware, not a warp's worth of threads, owns the transfer.

## F09-S03

- **prose**:

    Shared memory is striped into QZ000QZ (4 bytes per bank). When multiple lanes in a warp touch different addresses that map to the same bank in the same cycle, the hardware QZ001QZ those accesses Dense row-major tiles often create 2-way, 4-way, or worse conflicts that cut effective bandwidth.

- **prose**:

    bank conflict

- **code comment**:

    hardware TMA with swizzle

- **prose**:

    The copy lands bytes in shared memory using swizzle pattern QZ000QZ.

- **prose**:

    descriptor-based hardware transfer), not in how the data is arranged once it arrives.

## F09-S04

- **prose**:

    In raw CUDA, a programmer implementing data movement has enormous freedom: arbitrary pointer arithmetic, variable-stride access, hand-computed bank-conflict avoidance, custom swizzle formulas. The space of possible data movement patterns is vast, but the subset that actually performs well on GPU hardware is narrow it requires coalesced global loads, conflict-free shared-memory access, and correct swizzle alignment. Getting any of these wrong silently degrades throughput by 2~32×.

- **prose**:

    When you write QZ000QZ or QZ001QZ, the compiler automatically handles coalesced access, bank-conflict-free layout, and swizzle alignment.

- **prose**:

    TMA's descriptor-based interface is an extreme instance of this philosophy: it supports only a few tile-aligned transfer patterns, but those patterns are exactly the ones the hardware is optimized for. it generates only patterns that are coalesced and conflict-free, even though the underlying QZ001QZ/QZ002QZ instructions can express far more (including many slow patterns).

## F09-S05

- **prose**:

    The writeback to global memory still uses QZ001QZ

## F09-S09

- **prose**:

    Reading those addresses is undefined behavior on the GPU.

## F09-S10

- **prose**:

    wasting shared memory and bandwidth.

- **prose**:

    The loaded 1D strip is exposed as a 2D matrix for QZ000QZ and MMA operand loading, without an extra copy or additional shared memory.

## F09-S11

- **table**:

    works on all CUDA GPUs

- **table**:

    Hardware TMA

- **table**:

    Descriptor-driven Hopper ingress;

- **table row**:

    Hardware TMA | `tma.copy` / `tma.copy.swiz<N>` | Descriptor-driven Hopper ingress; dedicated engine enables async overlap

- **table**:

    Croqtile restricts patterns to guarantee coalesced, conflict-free transfers

- **table row**:

    Expressiveness trade | — | Croqtile restricts patterns to guarantee coalesced, conflict-free transfers

## F09-S12

- **table**:

    TMA hardware tensor copy (Hopper SM90+)

- **table row**:

    `tma.copy src => dst` | TMA hardware tensor copy (Hopper SM90+)

- **prose**:

    register hints

- **prose**:

    inline PTX

## F10-S01

- **prose**:

    The reason is always the same: no matter how expressive the abstraction, some hardware feature, some legacy library, some performance-critical intrinsic lives outside the language's domain.

- **list**:

    Standard CUDA device functions that coexist with QZ001QZ functions in the same QZ002QZ file.

- **list**:

    Inject verbatim C++ or PTX into the generated code for hardware-specific instructions the DSL does not emit.

## F10-S02

- **list**:

    The Croqtile compiler does not parse their bodies, does not rewrite them, does not manage their register allocation. This is the key distinction from QZ002QZ functions: a QZ003QZ function is pure CUDA that happens to live in a QZ004QZ file.

- **list**:

    The compiler transforms them into QZ001QZ CUDA kernels: QZ002QZ becomes grid/block launch dimensions, QZ003QZ becomes cooperative load sequences (or TMA descriptor issues), QZ004QZ becomes QZ005QZ instructions, QZ006QZ declarations become QZ007QZ allocations with computed sizes, events become barrier/mbarrier synchronization. The generated code is typically hundreds of lines of CUDA for a few dozen lines of Croqtile.

- **list**:

    QZ002QZ are spliced character-for-character into the generated CUDA at the point where they appear in the QZ001QZ body.

- **prose**:

    QZ001QZ compiles the device code (QZ002QZ, QZ003QZ) into GPU machine code, compiles the host code into CPU code, and links them into a single binary.

## F10-S03

- **prose**:

    It can use CUDA intrinsics (QZ001QZ, QZ002QZ, QZ003QZ), access QZ004QZ memory, and run inside a QZ005QZ kernel.

- **prose**:

    GPU machine code

- **prose**:

    the compiler would reject it because the generated QZ003QZ kernel runs on the GPU.

## F10-S04

- **prose**:

    When an algorithm needs a helper function that operates at the warp or thread level writing it as a standard CUDA QZ000QZ function is often the natural choice. they are plain CUDA.

- **prose**:

    A templated device function implementing warp-level top-K selection using shuffle instructions.

- **prose**:

    A warp shuffle intrinsic wrapper. The QZ001QZ and QZ002QZ are standard CUDA

- **prose**:

    Only thread 0 in each warp writes back results.

- **prose**:

    QZ000QZ functions handle the per-warp algorithm

## F10-S05

- **prose**:

    QZ000QZ takes a string literal and pastes it, character for character, into the generated CUDA file.

## F10-S06

- **prose**:

    The canonical QZ000QZ use case is register redistribution in warp-specialized pipelines (QZ002QZ). The producer warpgroup is register-light (mostly TMA loads), while the consumer is register-heavy (MMA accumulators). PTX's QZ001QZ instruction moves the register budget:

- **code comment**:

    ... WGMMA compute ...

## F10-S10

- **code comment**:

    #error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"

- **prose**:

    Treat these guards as documentation of hardware constraints.

## F10-S12

- **list**:

    The contract: allowed tile sizes, swizzle rules, architecture flags.

- **list**:

    Pause on each and ask what the hardware receives that the DSL does not spell.

## F10-S13

- **table**:

    Standard CUDA device function;

- **table**:

    GPU

- **table row**:

    `__device__ fn()` | Standard CUDA device function; passed through unchanged to `.cu` | GPU

- **table**:

    GPU

- **table row**:

    `call fn(args)` | Invoke a `__device__` function from a `__co__` body | GPU

- **table**:

    GPU

- **table row**:

    `__cpp__("...")` | Inject verbatim C++ at the splice point (ordinary string) | GPU

- **table**:

    GPU

- **table row**:

    `__cpp__(R"(...)")` | Inject verbatim C++ (raw string; use for `asm volatile`) | GPU

- **table**:

    Register redistribution via QZ000QZ: QZ001QZ on producer, QZ002QZ on consumer

- **table**:

    GPU

- **table row**:

    `setmaxnreg` | Register redistribution via `__cpp__`: `dec` on producer, `inc` on consumer | GPU

## F11-S01

- **prose**:

    GPU kernels are QZ000QZ. Thousands of threads execute concurrently, shared memory is invisible from the host, and when results are wrong, there is no stack trace pointing to the bad line. This opacity is not unique to GPUs any system where the programmer cannot directly observe intermediate state faces the same challenge: distributed systems (where is the lost message?), embedded firmware (which interrupt handler corrupted the register?), optimizing compilers (which pass broke the semantics?).

## F11-S02

- **prose**:

    No GPU needed. Use QZ002QZ / QZ003QZ to verify that QZ004QZ and QZ005QZ produce the tile sizes you expect before you spend time on a full kernel launch.

## F11-S03

- **prose**:

    Without the bang, QZ000QZ and QZ001QZ emit device-side QZ002QZ calls in the generated CUDA:

- **prose**:

    GPU QZ000QZ buffers are flushed asynchronously.

## F11-S07

- **prose**:

    Then check QZ008QZ if you introduced asynchronous copies or split producer and consumer warps.

## F11-S10

- **table**:

    no kernel launch

- **list**:

    tensor cores (Chapter 4)

## F12-S02

- **list**:

    QZ000QZ: Thread masking, events, async execution, and warp specialization.

## F13-S01

- **prose**:

    Croqtile is a C++ embedded DSL for writing high-performance GPU and DSA kernels. You write Croqtile functions in QZ000QZ files alongside your existing C++ code, and the compiler transpiles them into efficient target code (CUDA today, more backends planned) with full interoperability to CUDA, CuTe, and any C++ library.

- **prose**:

    Four design pillars set Croqtile apart from raw CUDA, CuTe, or Triton.

## F13-S02

- **prose**:

    DMA transfers, memory allocation, and synchronization that take dozens of lines of CUDA boilerplate reduce to a single QZ000QZ statement roughly 40% of the equivalent CUDA code. Every construct compiles down to the same PTX you would write by hand: zero-cost abstraction, no runtime overhead, no hidden allocations.

## F13-S03

- **prose**:

    353 compile-time diagnostic checks across 7 compiler modules catch shape mismatches, tiling errors, and DMA violations before any code runs on the GPU.

## F13-S05

- **prose**:

    hundreds in CUDA/CuTe). In practice, AI agents have pushed FP16 matmul from 671 to 1,127 TFLOPS in a single session.

## F14-S01

- **prose**:

    Additionally, we will present a parallel element-wise addition QZ001QZ program to demonstrate how Choreo simplifies data orchestration across heterogeneous hardware.

## F14-S02

- **prose**:

    For a Choreo-supported platform, which is usually a programming environment leveraging heterogeneous parallel hardware, the Choreo-C++ program typically contains three parts:

## F14-S03

- **prose**:

    CUDA/Cute

- **code comment**:

    Device program: typically runs on GPU/NPU

- **prose**:

    CUDA/Cute

- **prose**:

    CUDA/Cute

## F14-S04

- **list**:

    CUDA/Cute

- **list**:

    OpenCL

- **prose**:

    The OpenCL compiler requires the device program (QZ000QZ in the code) to be stored in a separate file from the host program.

- **prose**:

    Choreo-OpenCL C++

## F14-S05

- **prose**:

    Choreo-Cute C++

- **code comment**:

    first `parallel` indicates the kernel launch

## F14-S07

- **prose**:

    The QZ000QZ defines the computational kernel that will be executed on the target device such as a GPU.

- **prose**:

    Choreo-OpenCL C++

- **prose**:

    CUDA/Cute

- **prose**:

    OpenCL

- **prose**:

    Choreo's device programming model varies depending on the target hardware and its supported features. For example, some private targets allows the use of vectorizing programming interfaces, or QZ000QZ to fully leverage the computational power of the parallel target hardware.

- **prose**:

    In this paradigm, multiple instances of the same device program are executed in parallel, making it highly efficient for exploiting data-level parallelism on target hardware.

- **prose**:

    CUDA/Cute

## F14-S08

- **code comment**:

    first `parallel` indicates the kernel launch

- **prose**:

    For those familiar with QZ001QZ, this concept is analogous to a kernel launch, where multiple threads or processes are initiated to perform computations simultaneously on the device.

- **prose**:

    CUDA

## F15-S02

- **prose**:

    Choreo is an Embedded Domain Specific Language (eDSL) designed for streamlining the manipulation of data transmission within accelerator hardware. This includes navigating the complexities of data transmission, such as tiling strategies, which is subjected to the hardware capabilities.

## F15-S09

- **prose**:

    For such large data, programmers could move it across different memory hierachy of accelerator to best utilize hardware resource.

## F15-S14

- **prose**:

    This is similar to some OpenMP parallel directive, and some parallel programming language like OpenCL/CUDA.

- **prose**:

    If you are familiar with programming CUDA, you may think 'p' is a equivalence of 'thread index'.

## F15-S18

- **prose**:

    Here, we utilize the data transfer engine (DTE) to invoke a linear copy, which moves the 'data' residing in global memory to 'data1' in local memory.

- **prose**:

    Programmers often prefer to avoid dealing with scratchpad memory (SPM) management. In this code, the local memory allocation is left to the compiler.

## F18-S04

- **prose**:

    For example, QZ000QZ (for GPU hardware) supports:

- **prose**:

    Cuda/Cute

## F22-S05

- **prose**:

    In the current implementation, Choreo performs QZ000QZ (or QZ001QZ) to convert QZ002QZ programs into vendor-supported C++ language code and APIs (such as CUDA/Cute, and more).

- **prose**:

    This command produces CUDA C++ source code, which can be useful for specific development tasks.

## F23-S06

- **prose**:

    For example, QZ000QZ is globally defined to generate CUDA/Cute code, while QZ001QZ is only set for CUDA/Cute device code compilation.

## F24-S03

- **prose**:

    Beside that, in all the currently supported platform, such calls must be made inside QZ000QZ, since it calls device function which only runs on heterogeneous hardware.

## F28-S02

- **prose**:

    Currently, the primary focus is on data movement between heterogeneous hardware and across memory hierarchies. In hardware terminology, such movements are known as QZ000QZ.

## F28-S04

- **prose**:

    Note, if the QZ004QZ is not waited, it typically results in unexpected hardware issues.

## F28-S05

- **prose**:

    Choreo's DMA statement is an abstraction designed to support modern hardware. Beyond simple linear memory copies, advanced hardware such as the QZ000QZ can transfer shaped data and apply shape transformations in-flight.

- **prose**:

    Tensor Memory Accelerator (TMA)

- **prose**:

    CUDA/Cute

- **prose**:

    It is also possible to implement advanced DMA operations using software-only methods or software-hardware cooperation, though these are not yet supported.

- **prose**:

    CUDA/Cute

## F28-S08

- **prose**:

    In this section, we learned how Choreo simplifies data movement using QZ000QZ, making it easier to work with modern hardware.

## F30-S01

- **prose**:

    These tasks can be executed either iteratively, using loops, or in parallel, which requires both hardware and software support for parallelization.

## F31-S01

- **prose**:

    To fully utilize parallel hardware, it is crucial to write parallel code in Choreo.

## F31-S04

- **prose**:

    For those familiar with CUDA, QZ011QZ corresponds to the CUDA QZ007QZ level, and QZ012QZ corresponds to the CUDA QZ008QZ level.

- **prose**:

    The concept of multiple-level parallelism originates from hardware architecture with hierarchical memory. For instance, in CUDA, threads within a QZ001QZ have a much lower synchronization cost compared to those across different QZ002QZ.

## F31-S05

- **prose**:

    CUDA

- **prose**:

    The concept of sub-levels originates from QZ002QZ's GPU workload management, which often involves data with 2D or 3D elements.

- **prose**:

    CUDA

- **prose**:

    CUDA

## F31-S08

- **prose**:

    CUDA/Cute

- **code comment**:

    Kernel launch

- **prose**:

    Kernel Launch

- **prose**:

    CUDA

## F31-S09

- **whole section**:

    ### Restrictions on Storage Specifiers
    
    Since the `parallel-by` construct in Choreo is related to hardware, including heterogeneity and memory hierarchy, it is subject to the constraints of the target platform.
    
    For example, for *CUDA/Cute* targets, a *spanned* buffer must be annotated with the appropriate storage specifier within the *parallel execution block*. This is because the target platform allows only *scratchpad memory*, such as *shared* and *local* memory, to be allocated by device code. Conversely, code outside the *parallel execution block* can only declare *global* or default host memory, which the target platform allows the host code to manage.
    
    Additionally, *shared* and *local* memory types have a lifetime limited to the duration of the kernel launch. Therefore, referencing them outside the *parallel execution block* is not possible. This constraint ensures proper lifetime management of these memory types.
    

## F31-S10

- **prose**:

    To bind a kernel to a specific CUDA stream, use angle-bracket syntax:

## F31-S11

- **prose**:

    We also discussed how Choreo abstracts kernel launches and imposes memory management constraints, enabling us to leverage parallel hardware effectively.

## F33-S04

- **prose**:

    CUDA/Cute

## F34-S03

- **prose**:

    In such cases, masking some threads is a proper way to leverage parallel hardware while maintaining correct functionality.

## F35-S04

- **prose**:

    Though such deadlocks are less possible for GPU-like SIMT hardware, where both threads execute in lock-step, it is not safe code for highly asynchronous executing like threading in CPUs. Since Choreo supports multiple levels of asynchronous execution, including different hardware/software, a more robust way is to assign each thread an event for QZ000QZ statements.

## F36-S01

- **code comment**:

    asynchronous kernel launch (no host-side synchronization)

- **code comment**:

    synchronous kernel launch (host blocks until completion)

## F36-S02

- **prose**:

    CUDA streams allow multiple kernels and memory transfers to execute concurrently on the GPU.

- **prose**:

    Since both kernels are launched on different streams with QZ000QZ, they can execute concurrently on the GPU.

## F37-S01

- **prose**:

    To achieve optimal resource utilization, Choreo employs a QZ000QZ with QZ001QZ that allow programmers to manage how data is distributed and accessed across various levels of memory hierarchy.

## F37-S02

- **prose**:

    When combined with QZ000QZ, the spanned type defines where the data will reside in the memory hierarchy, such as global, shared, or local memory spaces. This distinction enables more efficient memory management, especially when working with accelerators (e.g., GPUs or specialized hardware).

## F37-S03

- **prose**:

    The storage qualifiers help guide the placement of data across the different levels of memory hierarchy, improving memory access patterns and performance.

## F37-S04

- **list**:

    Data defined in the global memory space is typically the most abundant and has the largest capacity. However, accessing global memory can be slower compared to more localized memory spaces.

- **list**:

    Global memory is suitable for storing large datasets that need to be shared across multiple processing units or components, such as matrices in deep learning applications or large scientific datasets.

## F37-S05

- **prose**:

    The shared memory space is typically used for fast, high-throughput access by processing units within a single compute node (e.g., threads in a GPU block or a CPU core). Data stored in shared memory is generally smaller in size than global memory but offers much faster access times. It's ideal for storing intermediate data, which needs to be accessed frequently during computation, but is not shared across different compute units. Shared memory allows for efficient data transfer within compute units and can be crucial for optimizing algorithms like matrix multiplication, where intermediate results are reused multiple times.

## F37-S06

- **prose**:

    Local memory is the smallest and fastest memory space, typically used for storing very small data that only needs to be accessed by a specific thread or computation unit. Local memory is often used in parallel computing scenarios where data needs to be private to each thread and accessed very quickly, but it is not meant for large datasets.

## F37-S07

- **prose**:

    By default, when a storage qualifier is not specified, the data is considered to be in global memory.

- **prose**:

    For example, the following code defines an array d0 without any storage qualifier, implying it resides in global memory:

- **prose**:

    For example, the following code places the matrix d0 in local memory, and the matrix d1 in shared memory:

## F37-S08

- **code comment**:

    Local memory for thread-specific data

- **code comment**:

    Shared memory for intermediate data across threads

- **code comment**:

    Global memory for large dataset

- **code comment**:

    Copy from local to shared memory

- **code comment**:

    Copy from shared to global memory

- **prose**:

    d0 is a small, thread-specific data array placed in local memory. d1 is an intermediate array placed in shared memory to be used by multiple threads within a block. matrix is a large dataset placed in global memory, where it can be accessed by all threads.

## F40-S04

- **list**:

    shared memory

- **list**:

    global memory

- **list**:

    shared memory

## F40-S07

- **list**:

    Choreo’s approach here is more aligned with QZ000QZ (e.g., local/shared memory on GPUs or accelerators).

- **list**:

    physical memory hierarchies

## F40-S08

- **whole section**:

    ## Virtual Parallelism and Physical DMA Mapping
    
    In traditional systems like **TVM** (Tensor Virtual Machine), scheduling and loop transformations are abstracted to optimize computation. For example, a typical **TVM schedule** for a matrix multiplication operation might look like the following:
    
    ```python
    # Example of TVM-style loop schedule
    s = tvm.create_schedule(A.op)
    xo, xi = s[A].split(A.op.axis[0], factor=64)
    yo, yi = s[A].split(A.op.axis[1], factor=64)
    s[A].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[A].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[A].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[A].bind(yi, tvm.thread_axis("threadIdx.y"))
    ```
    
    Here, the computation of a matrix A is split into multiple loops with the axis split by some factor (e.g., 64). The bind method binds loop variables to hardware threads, providing an abstraction of parallelism. This approach works well in a variety of cases, but it comes with some limitations when dealing with specific memory hierarchies or when performance is tightly coupled with the underlying hardware behaviour.
    

## F40-S09

- **whole section**:

    #### The Limitation of Virtual Parallelism in TVM-style Schedules
    In TVM’s loop schedule, virtual parallelism is represented via abstraction layers like threadIdx and blockIdx, which map high-level parallelism to physical hardware threads. However, the physical behavior of data movement, particularly in the case of complex memory hierarchies (e.g., local, shared, or global memory), is not explicitly represented. TVM abstracts away the details of data movement, leaving some of these behaviors to be automatically determined by the system, either through rules or optimizations.
    
    The main limitation here is that the physical behavior and memory access patterns may not be fully optimized because they are hidden behind the abstraction. As a result, the system may perform well in general cases, but it may not always exploit the full potential of the hardware, especially when fine-grained control over memory access is required (e.g., in high-performance computing scenarios).
    

## F40-S10

- **whole section**:

    #### Freedom Left to the System
    In this abstraction, much of the decision-making about memory and parallelism is left to the automatic optimization process in the system. This includes decisions about:
    
    - Memory layout (whether data should be placed in local, shared, or global memory),
    - DMA operations (how data should be transferred across different levels of memory),
    - Parallel thread mapping (how virtual loops are mapped to physical threads).
    
    While this freedom allows for a broad set of optimizations, it hides some of the underlying physical behavior from the user. This can be both a strength and a weakness:
    
    - Strength: The system can automatically make optimizations based on heuristics or rules, allowing it to adapt to different hardware architectures or runtime conditions.
    - Weakness: The user has less control over how parallelism is mapped to hardware, which may lead to suboptimal performance in certain cases, especially for complex memory and data movement patterns.
    

## F40-S11

- **whole section**:

    #### The Importance of Explicit Physical Mapping in Choreo
    In contrast to the abstracted approach in TVM, Choreo explicitly represents both virtual parallelism (e.g., through parallel-by, with-in, and foreach constructs) and physical memory operations (e.g., using DMA operations, chunkat, and memory qualifiers).
    
    By explicitly modeling the mapping of virtual loops to physical DMA operations and memory hierarchies, Choreo ensures that memory access patterns are optimized for the underlying hardware. This allows for fine-grained control over memory management, which can lead to better performance, especially on hardware accelerators like GPUs or custom accelerators.
    

## F40-S12

- **whole section**:

    #### How Choreo Works to address this limitation?
    In Choreo, the virtual parallelism is clearly defined through bounded variables and loops (e.g., parallel by, foreach).
    The physical behavior (e.g., data transfer between memory hierarchies via DMA) is directly controlled and mapped.
    This explicit coupling between virtual parallelism and physical data movement gives users more control over performance optimizations, especially in complex scenarios where manual tuning is required to exploit specific hardware features like memory hierarchies, cache optimizations, and data locality.
    
    In **Choreo**, **DMA operations** explicitly represent data movement between different memory levels, such as from **DRAM** (main memory) to **SRAM** (on-chip memory) and from **SRAM** to **registers**. This explicit control over data movement is crucial for optimizing memory access patterns, particularly on hardware accelerators like GPUs. When performing **tiling**, the **# operator** is used to calculate the **Cartesian product** of bounded variables, which enables the tiling of multiple loop dimensions and directly binds them to **DMA operations**. This provides a tight coupling between **virtual loop dimensions** (e.g., m_tile, k_tile, n_tile) and the actual **physical memory transactions** involved in data movement. 
    
    The **# operator** combines multiple bounded variables into a **composite variable**, defining a tiling pattern for the data. The resulting tiling improves cache locality and reduces memory access latency by optimizing the movement of data across memory spaces. Once the tiling is defined, **DMA operations** move data from **DRAM** to **SRAM** and from **SRAM** to **local registers** at the correct times. In matrix multiplication, for instance, **dma.copy.async** is used to load chunks of the matrix into **shared memory (SRAM)**, and later from **SRAM** into **registers** for computation. Each **DMA operation** is tied to the tiling pattern defined by the **bounded variables**, ensuring efficient memory access during computation.
    
    The **# operator** for tiling allows multiple **virtual loop dimensions** to be mapped directly to **memory operations**, offering a clear advantage over systems like **TVM**, which abstract away the details of memory operations and parallelism. This explicit mapping of **virtual parallelism** to **physical DMA operations** ensures that data is moved efficiently across the memory hierarchy, improving performance. 
    
    By using the **# operator** to define **multi-dimensional tiling** and explicitly mapping it to **DMA operations**, Choreo gives programmers **fine-grained control** over memory access, leading to improved cache locality and reduced latency. It aligns virtual parallelism with the physical hardware memory hierarchy, providing **better performance** on accelerators by optimizing data movement. This approach of **explicit memory tiling** and **DMA control** ensures that Choreo can leverage the full potential of the hardware, unlike higher-level abstractions that may limit the ability to directly control data movement.
    
    This compact version integrates all the important points without using subheadings, maintaining a more concise, flow-like structure for easier reading.
    

