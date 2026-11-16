# Material moved out of the Programming Guide

These are the exact upstream passages removed while assembling the
language-only guide, keyed by the content-unit IDs in
`UNIT-REGISTRY.md`. Nothing here is rewritten: it is the raw source
text, kept so that the hardware/architecture material can become its
own document.

## F01-S02

- **list**:

    Parallelism: Mapping Work to Hardware

- **list**:

    Tensor Cores: The QZ000QZ Operations

## F03-S04

- **prose**:

    Under the hood, the 32 instances here map onto 32 lanes of one execution unit

## F03-S07

- **prose**:

    The example works, but it processes one element at a time with no control over how data moves through the memory hierarchy. On real hardware that matters an execution engine reaches peak throughput by moving data in bulk, not element by element.

## F04-S01

- **prose**:

    That is a good way to explain correctness, but it is not how the hardware wants to move data. It is organizing those block transfers so the compute units see data in the right memory level, in the right shape, at the right time.

## F05-S02

- **prose**:

    When you write "these eight tasks are independent," you are making a statement about the QZ001QZ of your program, not about the hardware. The hardware decides how many actually run simultaneously, based on how many execution units it has and how the scheduler assigns work.

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

## F06-S07

- **prose**:

    These extensions all follow the same design principle: the four-step skeleton stays fixed, and the variant suffix communicates a specific contract to the hardware.

## F06-S08

- **table**:

    Load operand tile from shared into MMA registers

- **table row**:

    `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers

## F07-S01

- **prose**:

    When a branch sends some threads one way and other threads another way, the hardware cannot simply fork This is the fundamental tension: the hardware wants uniformity, but real programs need heterogeneity.

- **prose**:

    A group that diverges on an QZ001QZ runs both sides, throwing away half its throughput on each.

- **list**:

    the hardware schedules them independently.

## F07-S02

- **prose**:

    Inside one cooperating group, if some threads take the branch and others do not, the hardware QZ000QZ the two paths

- **prose**:

    the hardware simply skips the not-taken path.

## F07-S03

- **table**:

    MPMD within a single kernel launch

- **table row**:

    **Model** | SPMD with masking | MPMD within a single kernel launch

- **prose**:

    The QZ003QZ suffix is the concurrency modifier: it tells the compiler and hardware that the regions may overlap in time.

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

## F08-S04

- **prose**:

    The staged contents stay where the hardware placed them;

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

## F09-S12

- **prose**:

    register hints

- **prose**:

    inline PTX

## F10-S01

- **prose**:

    The reason is always the same: no matter how expressive the abstraction, some hardware feature, some legacy library, some performance-critical intrinsic lives outside the language's domain.

## F11-S01

- **prose**:

    any system where the programmer cannot directly observe intermediate state faces the same challenge: distributed systems (where is the lost message?), embedded firmware (which interrupt handler corrupted the register?), optimizing compilers (which pass broke the semantics?).

## F11-S02

- **prose**:

    Use QZ002QZ / QZ003QZ to verify that QZ004QZ and QZ005QZ produce the tile sizes you expect before you spend time on a full kernel launch.

## F11-S10

- **table**:

    no kernel launch

## F14-S01

- **prose**:

    Additionally, we will present a parallel element-wise addition QZ001QZ program to demonstrate how Choreo simplifies data orchestration across heterogeneous hardware.

## F14-S02

- **prose**:

    For a Choreo-supported platform, which is usually a programming environment leveraging heterogeneous parallel hardware, the Choreo-C++ program typically contains three parts:

## F14-S04

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

    Choreo-OpenCL C++

- **prose**:

    Choreo's device programming model varies depending on the target hardware and its supported features. For example, some private targets allows the use of vectorizing programming interfaces, or QZ000QZ to fully leverage the computational power of the parallel target hardware.

- **prose**:

    In this paradigm, multiple instances of the same device program are executed in parallel, making it highly efficient for exploiting data-level parallelism on target hardware.

## F14-S08

- **code comment**:

    first `parallel` indicates the kernel launch

- **prose**:

    This is also where the kernel launch happens: many instances are initiated to perform computations simultaneously on the device.

## F15-S02

- **prose**:

    Choreo is an Embedded Domain Specific Language (eDSL) designed for streamlining the manipulation of data transmission within accelerator hardware. This includes navigating the complexities of data transmission, such as tiling strategies, which is subjected to the hardware capabilities.

## F15-S09

- **prose**:

    For such large data, programmers could move it across different memory hierachy of accelerator to best utilize hardware resource.

## F15-S18

- **prose**:

    Programmers often prefer to avoid dealing with scratchpad memory (SPM) management. In this code, the local memory allocation is left to the compiler.

## F23-S06

- **prose**:

    To mimic a target native compilation, choreo preprocess also takes the builtin macros from the target: the target identity macros are globally defined, and the architecture macro is set only while compiling device code.

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

    Choreo's DMA statement is an abstraction designed to support modern hardware.

- **prose**:

    It is also possible to implement advanced DMA operations using software-only methods or software-hardware cooperation, though these are not yet supported.

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

    The concept of multiple-level parallelism originates from hardware architecture with hierarchical memory.

## F31-S08

- **code comment**:

    Kernel launch

## F31-S09

- **whole section**:

    ### Restrictions on Storage Specifiers
    
    Since the `parallel-by` construct in Choreo is related to hardware, including heterogeneity and memory hierarchy, it is subject to the constraints of the target platform.
    
    For example, on targets that separate host and device execution, a *spanned* buffer must be annotated with the appropriate storage specifier within the *parallel execution block*. This is because the target allows only scratchpad storage, such as *shared* and *local*, to be allocated by device code. Conversely, code outside the *parallel execution block* can only declare *global* or default host storage, which the host code manages.
    
    Additionally, *shared* and *local* memory types have a lifetime limited to the duration of the kernel launch. Therefore, referencing them outside the *parallel execution block* is not possible. This constraint ensures proper lifetime management of these memory types.
    

## F31-S11

- **prose**:

    We also discussed how Choreo abstracts kernel launches and imposes memory management constraints, enabling us to leverage parallel hardware effectively.

## F34-S03

- **prose**:

    In such cases, masking some threads is a proper way to leverage parallel hardware while maintaining correct functionality.

## F35-S04

- **prose**:

    Since Choreo supports multiple levels of asynchronous execution, including different hardware/software, a more robust way is to assign each thread an event for QZ000QZ statements.

## F36-S01

- **code comment**:

    asynchronous kernel launch (no host-side synchronization)

- **code comment**:

    synchronous kernel launch (host blocks until completion)

## F37-S01

- **prose**:

    To achieve optimal resource utilization, Choreo employs a QZ000QZ with QZ001QZ that allow programmers to manage how data is distributed and accessed across various levels of memory hierarchy.

## F37-S02

- **prose**:

    When combined with QZ000QZ, the spanned type defines where the data will reside in the memory hierarchy, such as global, shared, or local memory spaces. This distinction enables more efficient memory management, especially when working with accelerators.

## F37-S03

- **prose**:

    The storage qualifiers help guide the placement of data across the different levels of memory hierarchy, improving memory access patterns and performance.

## F37-S05

- **prose**:

    The shared storage space is typically used for fast, high-throughput access by the processing units of a single compute unit (for example, the threads in one block). It is ideal for intermediate data that is accessed frequently during computation but is not shared across different compute units. Shared storage allows for efficient data transfer within a compute unit and can be crucial for optimizing algorithms like matrix multiplication, where intermediate results are reused multiple times.

## F37-S06

- **prose**:

    Local memory is the smallest and fastest memory space, typically used for storing very small data that only needs to be accessed by a specific thread or computation unit. Local memory is often used in parallel computing scenarios where data needs to be private to each thread and accessed very quickly, but it is not meant for large datasets.

## F37-S08

- **code comment**:

    Local memory for thread-specific data

- **prose**:

    d0 is a small, thread-specific data array placed in local memory.

