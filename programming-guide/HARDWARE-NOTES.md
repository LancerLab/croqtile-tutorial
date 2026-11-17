# Material moved out of the Programming Guide

These are the exact upstream passages removed while assembling the
language-only guide, keyed by the content-unit IDs in
`UNIT-REGISTRY.md`. Nothing here is rewritten: it is the raw source
text, kept so that the hardware/architecture material can become its
own document.

## F03-S04

- **prose**:

    Under the hood, the 32 instances here map onto 32 lanes of one execution unit

## F03-S07

- **prose**:

    The example works, but it processes one element at a time with no control over how data moves through the memory hierarchy. On real hardware that matters a machine reaches peak throughput by moving data in bulk, not element by element.

## F04-S01

- **prose**:

    It is organizing those block transfers so the compute units see data in the right memory level, in the right shape, at the right time.

## F05-S03

- **code comment**:

    concurrent -- mapped to thread blocks

- **prose**:

    The QZ000QZ lines say "these iterations are independent." The QZ001QZ and QZ002QZ annotations say "put the outer ones on separate thread blocks and the inner ones on separate threads within a block." The QZ003QZ lines say "run these in order." No ambiguity, no conflation.

## F06-S01

- **prose**:

    It is also the slowest way to use a modern accelerator, because it ignores the unit designed specifically for this job.

- **prose**:

    The operation is central enough that accelerators execute it as one macro-instruction: take a tile of A, a tile of B, multiply, accumulate into C.

## F06-S02

- **list**:

    allocate an accumulator tile QZ001QZ in registers and zero it.

- **prose**:

    The names QZ001QZ, QZ002QZ, and QZ003QZ are opaque register tiles

## F06-S08

- **table**:

    Load operand tile from shared into MMA registers

- **table row**:

    `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers

## F07-S01

- **prose**:

    A group that diverges on an QZ001QZ runs both sides, throwing away half its throughput on each.

## F07-S03

- **table**:

    MPMD within a single kernel launch

- **table row**:

    **Model** | SPMD with masking | MPMD within a single kernel launch

## F07-S09

- **prose**:

    A QZ000QZ fixes the launch size (often close to the number of available execution units) and lets each block iterate over multiple tiles:

## F07-S13

- **table**:

    Non-blocking kernel launch

- **table row**:

    `parallel.async ... : block` | Non-blocking kernel launch

## F07-S14

- **prose**:

    so the pipeline runs safely and at full throughput.

## F09-S04

- **prose**:

    it requires coalesced reads, conflict-free staged access, and correct swizzle alignment. Getting any of these wrong silently degrades throughput by an order of magnitude.

- **prose**:

    When you write QZ000QZ, the compiler handles coalescing, conflict-free layout, and swizzle alignment.

## F09-S11

- **table**:

    Croqtile restricts patterns to guarantee coalesced, conflict-free transfers

- **table row**:

    Expressiveness trade | — | Croqtile restricts patterns to guarantee coalesced, conflict-free transfers

## F11-S01

- **prose**:

    any system where the programmer cannot directly observe intermediate state faces the same challenge: distributed systems (where is the lost message?), embedded firmware (which interrupt handler corrupted the register?), optimizing compilers (which pass broke the semantics?).

## F11-S02

- **prose**:

    Use QZ002QZ / QZ003QZ to verify that QZ004QZ and QZ005QZ produce the tile sizes you expect before you spend time on a full kernel launch.

## F11-S10

- **table**:

    no kernel launch

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
    

