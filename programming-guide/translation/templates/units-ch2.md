=== UID:F12-S02
P: This reference covers every aspect of programming with Croqtile:
L: **Program structure**: How Croqtile programs are organized --- the `__co__` function, host-side APIs, and C++ interoperability.
L: **Shaped data**: Croqtile's type system for multi-dimensional tensors --- shapes, MDSpan, spanned data, symbolic dimensions, and dynamic shapes.
L: **Loop and parallelism**: SPMD parallelism, loop control, tiling, and iteration constructs.
L: **Data movement**: DMA statements, tiling with `chunkat`, TMA, swizzle, and advanced data movement patterns.
L: **C++ embeddings**: Input/output conventions, `__cpp__` inline blocks, macros, and preprocessor integration.
L: **Optimization patterns**: Tileflow optimizations, async DMA patterns, multi-buffering, and performance tuning techniques.

=== UID:F14-S01
P: A `.co` program has a fixed structure: host code, device code and a tileflow function, each treated differently by the compiler.

=== UID:F14-S02
H: Construct a Croqtile Program
P: A typical *Croqtile* program is composed of multiple parts, depending on the target platform.
L: The Device Program | The Host Program | The Tileflow Program

=== UID:F14-S04
H: Transpilation and Compilation
P: The Croqtile compilation process typically involves three major steps: **Pre-processing**, **Transpilation**, and **Target Compilation**. To better understand how different parts of a Croqtile program work together, the full compilation workflow is illustrated below:
F: f14-compile-workflow.drawio.png
P: As shown in the figure, immediately after *pre-processing*, Croqtile *transpiles* (*source-to-source* compilation) the *tileflow program* into target code, leaving the user-provided *host program* and *device program* unchanged. The *tileflow program* is *transpiled* into host and device source code, which you refer to as *croqtile-generated code*. The compiler then combines the user-provided code and *croqtile-generated* code to perform the *target compilation* process. This process can generate various outputs, such as transpiled source code, workscripts, target modules, target assembly, and executables.
P: The Croqtile compiler functions as an end-to-end compiler, with the key step being the transpilation of the *tileflow program* into *croqtile-generated code*.
P: One notable feature of Croqtile compilation is its support for both:
L: **Single Source Compilation Model**: Similar to , where the target compiler allows device and host programs to be in a single source file for target compilation. | **Separate Source Compilation Model**: Similar to , where host and device code must be compiled separately.
P: The code shown above naturally supports the *Single Source Compilation Model*. However, to support the *Separate Source Compilation Model*, Croqtile requires wrapping the Device Program with the `__cok__` block, as shown below:
C: Transpilation and Compilation
P: This is the code structure for a program. The `_`_cok__ wrapper enables the Croqtile compiler to handle user-provided device code properly. It helps Croqtile separate device and host code from a single Croqtile source file for different compilation processes. Therefore, do not be surprised if you encounter `__cok__` in certain Croqtile code; it is necessary for integrating support for the *Separate Source Compilation Model*.

=== UID:F14-S07
H: Device Program: Parallel Computation
P: The kernel is designed to operate on input data, process it in parallel, and generate the output.
P: You repeat the code as below:
C: A device helper called from tileflow
P: For targets that only support the _Separated Source Programming Model_, the code may need to be wrapped within a `_`_cok__ block. The equivalent code is shown below:
C: The cok wrapper for separate sources
P: Here, the `extern "C"` annotation replaces the `__device__` keyword used in target, as requires C-linkage for the device functions.
P: Programmers must be aware that the device program follows the *Single-Program-Multiple-Data (SPMD)* paradigm. However, unlike traditional programs, the device program does not manage data movement ---whether between the host and device or across multiple storage levels within the device. Instead, the *tileflow program* orchestrates these tasks in a much simpler and safer manner.

=== UID:F14-S09
H: In short
P: You are now aware that a Croqtile program consists of three parts, with the *tileflow program* being the core. This part is transpiled into target source code during the compilation process. The call chain typically flows from the host to the tileflow program, and then from the tileflow program to the device code.
P: You may have noticed that Croqtile not only abstracts DMA operations into higher-level semantics but also combines iteration and tiling for ease of use. This makes Croqtile code concise and expressive. In the following chapters, you will delve deeper into Croqtile's syntax and semantics to explore its full potential.

=== UID:F31-S08
H: Heterogeneity: Transpiling to program launch
P: For targets like , the `parallel-by` block not only specifies the number of parallel threads but also defines the boundary between host and device code, leveraging Croqtile's ability to manage heterogeneity. Specifically:
L: Code **outside** the `parallel-by` block is transpiled into **host code**. | Code **inside** the block is transpiled into **device code**, as illustrated below:
C: Heterogeneity: Transpiling to program launch
P: For those familiar with , this is analogous to a occurring at the `parallel-by` statement. However, Croqtile abstracts these details with the unified SPMD parallelism model, allowing developers to focus on their algorithms without worrying about the underlying heterogeneity.

=== UID:F05-S11
H: Type System Aside: `mdspan` and `ituple`
P: Before you build the matmul, two features of Croqtile's type system are worth mentioning. You have already used them implicitly; now is a good time to name them.
P: The `span(i)` syntax extracts one dimension: `lhs.span(0)` is 128, `rhs.span(1)` is 256. The shorthand `lhs.span` copies the entire shape.
P: For the full details, see 2.4 and 2.5. In the examples that follow, the key takeaway is: Croqtile catches shape mismatches at compile time, not at runtime.

=== UID:F15-S01

=== UID:F15-S02
P: It aims to simplify the daily tasks of engineers responsible for crafting high-performance kernels. By leveraging Croqtile, the programming process becomes more accessible, allowing engineers to optimize these critical processes with greater ease and efficiency.

=== UID:F15-S03
H: Embedding Croqtile within C++
P: Croqtile is the DSL code embedded within C++. The Croqtile compiler performs the source-to-source translation of the Croqtile function to be C++ code. It implicitly includes "croqtile.h", which bridges the translated code to work properly with other C++ code. The below code snippet showcases an example.
C: Embedding Croqtile within C
P: Note, a Croqtile function is prefixed with the "__co__" keyword. All code in the function scope is translated by the Croqtile compiler. Other C++ code calls the Croqtile function by its name as declared.
P: In addition, certain Croqtile-specific types are introduced to ensure consistency between Croqtile function arguments and their respective callers. Considering that there are some important programming elements missing, you will defer the introduction of such details until the data types are explained.

=== UID:F15-S04
H: Variables and Types
P: Croqtile introduces four fundamental type categories: scalar-type, spanned-type, integer-tuple-type (ituple-type) and bounded-type. These types serve for different purposes.
L: **Scalar Type**. It is designed to fullfil the requirement of program control. It consist of *Integer Type* and *Boolean Type*. | **Spanned Type**. It represents the data type (normally the tensor) for computation. Apart from referencing the raw data, a *spanned type* also associates data with a shape representing by multi-dimensional ranges. The 'multi-dimensional ranges' is named 'mdspan'. It is useful for tiling, etc, which is introduced soon. | **Integer Tuple (I-Tuple) Type**. It represents a group of integer values. A common usage of *i-tuple* is to introduce the bound of multi-dimensional data. | **Bounded (Integer/ITuple) Type**. This is the special type that is used to simplify data (sub-zone) reference. And it works with loop construct to iteratively process data.
P: Among the four type categories, *scalar* and *ituple* could be well accepted consider it maps to elements of existing general purpose programming languages. However, the *spanned type*, and *bounded type* are specific to the domain. The following sections will show the detail.

=== UID:F15-S06
H: Spanned Types
P: A spanned type is a **Composite Type**. It consists of a **fundamental type**, and a **multi-dimensional-span(mdspan) type**. Neither the *fundamental type* nor the *mdspan* is a *complete type*. It implies that neither of them can type a data with storage. In Croqtile, you name them as *partial types*.
P: The reason for making the type be composition of two is to specifically manipulate *mdspan* alone. This is crucial in ML scenarios where data are always organized as multi-dimensional. It is very common for the programmers to tile the data into smaller blocks/chunks, and move them across storage hierarchy. Therefore, manipulating the _shape_ (represented by _mdspan_) of multi-dimensional data plays a vital role.
P: In the following sections, this section shows how to define such partial types and how to compose partial types into a complete type for declaring/defining multi-dimensional data purpose.

=== UID:F15-S07
H: The Partial Type: **mdspan**
P: Unlike most type systems, *mdspan*, comes as an partial entity for typing. Here you claims it as *partial* because Croqtile program is unable to define data as a *mdspan* type. However, *mdspan* itself can be defined solely. Such design is based on the oberservation that loop tiling/blocking cares nothing but the shape of multiple dimension data. Therefore, Croqtile allows programmers to manipulate *mdspan* regardless the fundanmental type it associated with, to ease the programming about shapes.
P: Meanwhile, croqtile compiler could apply type checking on *mdspan* to reveal code errors as early (ahead of execution, when applies) as possble.
P: In Croqtile, a mdspan is braced by '[' and ']'. The upper-bound of dimensions inside are separated by ','. That is,
C: An mdspan declaration
P: As illustrated, a 'mdspan' can be explicitly defined with *mdspan* partial-type keyword as annotated, optionally with a '<' '>' enclosed total dimension value. It is also possible to define a mdspan without type annotation. As shown in definition of 'd' above. (Note: 'mds' and 'd' are of the **dependent type**, considering the type depends on the evaluation of 'a' and 'b'. Programers may provides 'a' and 'b' with runtime values. However, in this way, some runtime check are paid consequently)
P: For C++ programmers, you may think a mdspan as the trait of a multiple dimensional array, where it describes a multi-level-range. For example, 'sp' in the above example defines two-level of ranges, ranging from 0 to 6, and 0 to 7 separately.
P: When a mdspan is defined, it is possible to get the integer dimension value using operator '()' over the mdspan.
C: A shape literal
P: In the above example, the expression 'sp(0)' reasons about the first dimension value of mdspan 'sp'. The dimension values are in essence integer values, which are suitable for integer arithmetics.
P: As a conseqeunce, you may derive a mdspan from existing one as the below code snippet:
C: A shape with one dimension changed
P: With such a facility, it is very convinient to apply tiling over multiple-dimensional spans in your Croqtile code. Considering that tiling operation is always required in constructing the high-performance kernels, Croqtile has provided syntax suger to make the work even easier:
C: The same shape again
P: This code works the same way as the previous one, but obviously in a much simpler syntax. This follows one of Croqtile's design philosophy --- To enable functionalities with minimal code whenever possible.
P: Except for defining a mdspan in the above *dimension-wise* way, Croqtile also support to define mdspan definition with other methods. You will introduce such methods when *i-tuple* is revealed.

=== UID:F15-S12
H: Bounded Types
P: Bounded types consists of **Bounded Scalar** and **Bounded ITuple**. Bounded Scalar takes a range of [0, ub), where 'ub' represents its upper bound. Therefore, if an integer 'p' is set as bounded, it should also be associated with a specific upper bound. To establish such associations, programmers must code explicitly inside the *Control Structures* of 'parallel-by' and 'with-in', which will be introduced later.
P: Similarly, since *ITuple* is a group of *Integer*s, it can also be associated with a group of bounds. Specifically, in Croqtile, the *Bounded ITuple* is associated with a *mdspan* value, where a group of upper bounds are settled. In later sections, you shall illustrate the detailed syntax.

=== UID:F15-S17

=== UID:F15-S22
H: In short
P: Croqtile introduces a novel approach to SPMD programming. It favors C++-style coding and is embedded within C++. However, its primary focus is on alleviating the burden of low-level programming details, particularly those related to data manipulation across various memory layers through DMA operations. At times, it is also referred to as the dataflow programming DSL. You developed this tool to support the daily task of constructing high-performance kernels. Your aim is to enable programmers to focus less on the intricacies of language construction and more on higher-level conceptual thinking.
P: You wish you find it functions as expected. And you are looking forward to any feedback for continuous improvement.

=== UID:F17-S03
H: Group of Ranges and Group of Integers
P: In Croqtile, *mdspan* can be used to represent dimensional shapes. However, as introduced, *mdspan* stands for **Multi-Dimensional Span**. The elements inside are *Dimensional Spans*, which imply ranges of values.
P: For example, an *mdspan* `[7, 14]` represents a group of two ranges `[0, 7), [0, 14)`. Therefore, an *mdspan* does not represent a list of integer values. However, grouping integer values is useful in certain scenarios:
L: When considering the delta of two mdspans, multiple integers are needed. In high-level semantics, the delta may reflect padding differences when adjusting shape dimensions. | When tiling a shape, the multiple tiling factors are integers, not ranges.
P: These situations motivate Croqtile to have a type for grouping integer values.

=== UID:F17-S08
H: Look-Ahead: Bounded I-Tuples

=== UID:F18-S02
H: *Spanned*: Data and Buffers
P: The primary focus of a Tileflow program is to manipulate large datasets by moving them around. In some terminology, the input and output of a Croqtile function are classified as "input data" and "output data," respectively. Meanwhile, any other storage is referred to as "buffers." This conceptually distinguishes between "external" and "internal" memory from a function's perspective. However, both "data" and "buffer" refer to storage positions. In Croqtile, both are typed as **spanned**, reflecting that they are data/buffers associated with an

=== UID:F18-S07
H: Buffer Lifetime Management
P: The lifetime of storage allocated inside a *croqtile function* is managed by the Croqtile compiler for efficient use. The compiler attempts to reuse buffers as much as possible if their lifetimes do not overlap. Therefore users of Croqtile are not necessary to do buffer management.

=== UID:F31-S06
H: Bounded Variable
P: In a `parallel-by` block, the variables defined after the `parallel` keyword are referred to as **Bounded Variables**. Specifically:
L: If the variable is an *integer*, it is called a **bounded integer**. | If the variable is an *ituple*, it is called a **bounded ituple**.
P: This terminology indicates that the variable not only has a value but also has defined bounds. For example, in the statement `parallel p by 6`, `p` is a *bounded integer* with a bound `[0, 6)`, where `6` is the exclusive upper bound.
P: The concept of bounded variables will be explored further in later sections, where their usage will be detailed.

=== UID:F32-S09
H: Values of Bounded Variables
P: Understanding the values associated with a bounded variable is crucial. A bounded variable has two key values:
L: **Current Value**: itemize[leftmargin=1.6em] | Within a `foreach` statement, the current value is determined by the loop iteration, as the bounded variable serves as the iteration variable. | Outside of a `foreach` statement, the bounded variable always has a value of zero.
L: **Upper-Bound Value**: This is specified in the `with-in` or `parallel-by` statements.
P: Programmers may encounter issues when using the current value of a bounded variable outside of a `foreach` statement, particularly after the loop has completed. By definition, the current value of a bounded variable is immutable except within a `foreach` loop. The following code illustrates this:
C: Values of Bounded Variables

=== UID:F37-S01
H: Memory in Croqtile
P: In Croqtile, the memory system is designed to handle large datasets effectively, which is crucial for high-performance computing scenarios, especially in machine learning and scientific computing.

=== UID:F37-S02
H: Spanned Types in Croqtile
P: A **spanned type** is a composite type that includes both the fundamental data type (such as `f32`, `s32`, or `f16`) and a **multi-dimensional span (mdspan)** that describes the shape and organization of the data. The spanned type allows Croqtile to define large arrays or matrices with an efficient layout, while also enabling the programmer to manipulate the shape (i.e., dimensions) independently of the data type. This flexibility is essential for managing multi-dimensional datasets that are frequently encountered in parallel computation and machine learning workloads.
P: The spanned type by itself does not allocate storage for the data. Instead, it serves as a blueprint for defining how the data will be organized in memory, allowing for runtime flexibility and optimizations.

=== UID:F37-S03
H: Storage Qualifiers in Croqtile
P: In Croqtile, **storage qualifiers** annotate the memory allocation and specify which memory region the data will reside in. Croqtile defines three primary storage qualifiers:

=== UID:F16-S02
H: First-Class Citizen: Shape
P: Croqtile's primary function is to manage data movements, which are crucial for efficiently organizing and processing large datasets, especially in machine learning and high-performance computing scenarios. However, most C++ programming environments handle data in a casual manner ---either as flat (pointers) or hierarchical structures (arrays), without a native representation of associated shapes.
P: In contrast, Croqtile enforces code safety and simplifies the programming of shaped data by requiring that **any data declared or used must be associated with a shape**. This motivates Croqtile to treat shape as a first-class citizen.

=== UID:F16-S05
H: Evaluation of *mdspan*
P: An mdspan with constant values is sufficient for many scenarios, as high-performance device kernels often require fine-tuning based on fixed input data shapes. In these cases, the `mdspan`s are evaluated at compile-time, meaning their values do not incur extra execution time or storage overhead.
P: In some scenarios, a runtime shape (where some dimensions are determined at execution) is required for building the kernel. Croqtile supports this with **Symbolic Dimensions** for `mdspan`, which will be introduced later. In such scenarios, runtime evaluation of dimension values may be necessary. Fortunately, Croqtile manages this evaluation immediately after entering a croqtile function in the Croqtile-generated host code, resulting in negligible startup cost. **Therefore, programmers can ignore overheads related to _mdspan_ normally**.

=== UID:F17-S07
H: Evaluation of Integers and I-Tuples
P: Similar to mdspan, both *integer*s and *i-tuple*s incur minimal runtime cost. Their values are evaluated at compile time whenever possible, so programmers can generally ignore their cost.

=== UID:F19-S01
P: n this section, you will learn about the *anonymous dimension* and *symbolic dimension* in Croqtile and how they support dynamic shapes required by certain systems.

=== UID:F19-S02
H: Fixed and Dynamic Shape
P: Data with fixed shapes is commonly used in high-performance computing kernels, allowing for aggressive optimization with known dimensions. However, certain scenarios require handling shapes with runtime dimensions, known as dynamic shapes. This necessitates kernel generality to manage inputs of varying shapes, a feature highlighted by many machine learning frameworks.

=== UID:F19-S05
H: Improved Code Safety
P: The reason for improved code safety of using *symbolic dimension* is that more comprehensive code checks can be applied by Croqtile compiler. For example, consider an *anonymous dimension* version of the `Matmul` function:
C: A kernel with anonymous dimensions
P: Croqtile performs both compile-time and runtime checks to ensure safety:
L: **Compile-time Checks**: Croqtile verifies rank consistency. In this example, `lhs.span` has a ranked of `2`, matching the rank of `tile_`m, tile_k, so there are no issues. | **Runtime Checks**: Croqtile generates code to validate dimensions at runtime. For instance, in the declaration of `tiled_lhs`, its shape must not have a dimension of `0`, which would result in an invalid zero-sized buffer. Croqtile's *transpilation* process generates the following *target host code*:
C: The transpiled signature
P: Here, the `croqtile_`abort function aborts program execution if the condition is not met. This early check helps Croqtile detect issues promptly.
P: For the *symbolic dimension* version, Croqtile compiler generates additional checks in the *target host code*:
C: The generated shape check
P: In addition to the checks performed in the *anonymous dimension* version, the *symbolic dimension* version verifies the consistency of dimensions between `lhs` and `rhs`, resulting in safer code.
P: Therefore, it is recommended that programmers use symbolic dimensions not only for ease of use but also to enhance code safety.

=== UID:F19-S06
H: Cost and Limitations
P: To support dynamic shapes, Croqtile compile has to be conservative at compile time. This limitation restricts certain compile-time checks and necessitates some checks to be performed at runtime instead. Although the implementation aims to minimize overhead by placing these checks in the host code, they **remain more conservative compared to code with fixed-sized shapes**. Programmers should be mindful of this situation and may need to add user-level assertions to enhance code safety when necessary.
P: A significant limitation of both *anonymous dimension* and *symbolic dimension* is that they can **only be applied to the *spanned data* in parameter lists**. Dimensions of buffers declared within Croqtile functions cannot be made dynamic. For example, the following code will result in a compile-time error:
C: Cost and Limitations

=== UID:F19-S07
H: In short
P: Anonymous and symbolic dimensions are the two ways Croqtile represents a shape that is not fully known when the program is written. Compared to *anonymous dimension*, using *symbolic dimension* enhance code safety through additional compile-time checks, making them the recommended choice for ease of use and improved safety.

=== UID:F20-S01

=== UID:F20-S02
H: **a. Improved Readability and Clarity**
P: Using symbolic dimensions like `M` and `N` explicitly in the shape definition makes the program **easier to read** and understand. Instead of seeing a `?` or `-1` and wondering what the dimension is supposed to represent, the reader can immediately grasp that `M` and `N` are variables that depend on runtime values. This improves **maintainability** and reduces the cognitive load for developers who are reading and modifying the code.
P: For example, if a program uses `?` for dynamic dimensions, it's unclear whether it represents an uninitialized value, a wildcard dimension, or something else. This can lead to confusion when trying to reason about the program. In contrast, symbolic dimensions are explicit and show exactly how the shape is determined:
C: a. Improved Readability and Clarity
P: The latter version is ambiguous and lacks the clarity that symbolic dimensions provide.

=== UID:F20-S03
H: **b. Type Safety and Compile-Time Checking**
P: One of the major benefits of symbolic dimensions in Croqtile is that they are **type-safe** and **checked at compile-time**. Croqtile allows you to declare symbolic dimensions and even manipulate them algebraically (e.g., `M / 2`, `N / 4`), ensuring that they adhere to the expected types and constraints. This eliminates the need for runtime checks, which can often lead to errors or performance penalties.
P: For example, using `?` or `-1` as dynamic dimensions in other systems often leads to runtime checks for consistency, such as verifying that the dimensions are properly initialized before use. These checks can introduce overhead and make the program harder to optimize. In contrast, symbolic dimensions allow Croqtile to **check constraints at compile-time**, ensuring that shape expressions are valid before execution:
C: b. Type Safety and Compile-Time Checking
P: This ensures type consistency and reduces the potential for errors during runtime.

=== UID:F20-S04
H: **c. Expressiveness and Flexibility**
P: Symbolic dimensions enable a **richer** set of expressions for describing data shapes. Instead of using arbitrary placeholders like `?` or `-1`, which are limited to basic dimensionality, symbolic dimensions can be **constrained** using complex arithmetic and relationships. For example, you can directly express how one dimension depends on another, or how it is scaled or tiled, in a much more natural way.
P: In contrast, using `?` or `-1` often leads to brittle code, where developers need to manually calculate or infer the actual size at runtime, introducing potential errors and making the code harder to reason about.
P: For example, you can easily express relationships between dimensions using symbolic variables:
C: c. Expressiveness and Flexibility
P: If you were using `?` or `-1`, expressing the same relationships would require convoluted workarounds or additional runtime logic.

=== UID:F20-S05
H: **d. Compile-Time Optimization**
P: One of the significant benefits of symbolic dimensions is that they can help with **compile-time optimizations**. Since Croqtile allows symbolic dimensions to be evaluated at compile time, it can potentially optimize memory layouts and kernel execution plans based on these dimensions. This can result in more efficient code generation and better performance compared to using `?` or `-1`, which may require runtime evaluation and dynamic memory allocation, hindering optimization opportunities.

=== UID:F21-S01
H: Introduction to Croqtile Value Numbering
P: In Croqtile, the value numbering process serves to facilitate Shape Inference by either directly inferring shapes known at compile time or generating shape computation expressions.
P: While many value numbering approaches focus on eliminating redundant computations (expressions that consistently produce the same value), Croqtile's value numbering aims to establish a progressive method for simplifying the Croqtile domain-specific construct 'mdspan'. In this context, the application of constant propagation, algebraic simplification, and other techniques becomes more essential.
P: In terms of the implementation method, Croqtile utilizes a simple **scope-based** value numbering approach, an extension of the local value numbering method, complemented by a lexical-scoped value number table[1]. Unlike other value numbering methods, Croqtile does not necessitate the construction of basic blocks, pruned-SSA, or a compiler backend, as it applies value numbering directly onto the Abstract Syntax Tree (AST). However, to facilitate this approach, certain restrictions must be imposed on Croqtile's syntax rules, particularly to prevent value mutation.

=== UID:F21-S02
H: Quick Example
P: The value numbering process assigns a value number to all entities containing a list of values, including for mdspan and ituple. For example,
C: A shape expression
P: The value numbering process would produce values including:
C: Its value-number graph
P: In this illustration, the number after "VN #" is the value number of an expression. I either represents a constant integer value (known at compile time), a symbolic value (like variable 'a', 'b'), or composition of other value numbers. Here, value number #0 and #3 represents a reference to constant value, where #1, and #2 represents reference to the integer-typed variable "b" and "c". #4 represents the value of an addition expression, which gives the sum of #2 and #3. That is the expression of 'c + 3' when dereferencing all the value numbers. And lastly, value number #5 represents the list of values defined by mdspan 'a' symbolically. With such a definition, you may translate the indexing operations, e.g. "a(0)", results in value number #0 and thus a constant of 1.

=== UID:F21-S03

=== UID:F21-S04
H: From Expressions to Value Numbers
P: In current implement, the value numbering process in a syntax-directed way. It traverse the Abstract Syntax Tree nodes and generate value numbers when the node contains an expression.
P: The method of generating value numbers is simple. For each expression, it takes the expression as the input, and returns corresponding value number.
P: More specifically, it:
L: Generates a signature from an expression. | If the signature has already existed, return the associated value number by looking up the value number table. | Or else, it associates the signature with a new value number in the value number table, and return it.
P: Notably, for simplicity, current signature is an string of expression. This can be improved by hashing later (like most value numbering process does).
P: In addition, to handle the symbol definition and symbol reference, it follows the below rules:
L: When an expression is to initialize a symbol (variable or partial type), the symbol is assign the same value number of the expression. More specifically, it associate the signature of the *scoped* symbol name with the value number in the value number table. | When an expression references a symbol, it looks up its corresponding signature in the value number table, and return it.
P: Note the symbol must use its *scoped name* for signature generation, since identical names are allowed to be defined in different scope.

=== UID:F21-S05
H: Simplifications
P: Every time the value numbering process generates a signature, it tries to simplify the expression. For example,
C: A constant expression
P: Given the simple integer expression, the value numbering process could generates signatures and associated value number as illustrated:
C: The simplified graph
P: Here, without optimizations, two constant values #0 and #1, and a new value number #2, which is an addition of #0 and #1 are generated. #2 has the signature of "+:#0:#1". However, considering that addition of constants can be evaluated at compile time, simplification could be performed. The below output showcases the optimized value numbering process:
C: The folded graph
P: The signature of #2 now represents a constant, which implies the algebraic simplification is applied successfully.

=== UID:F21-S06
H: From Value Numbers to Expression
P: After value numbering and simplification, each expression is associated with its simplified value number. In Croqtile, this is important step for the shape inference process of the mdspan type.
P: The *mdspan* type is a partial type that defines count of fundamental-typed elements. And in programming, together with a fundamental type, e.g, it decides the memory size an *spanned* object takes.
P: The shape detail of *mdspan* could be runtime decided. However, in many scenario, the shape is possible to be fully evaluated to fixed numbers at compile time. In such a scenario, static memory allocation can also be triggered at compile time to best utilize the system memory. Or else, the it requires to generate code to calculate the shapes.
P: Consequently, except the constant values, it is required to generate expressions back from value numbers.

=== UID:F21-S07

=== UID:F21-S08
H: Lexical Scopes and Restriction
P: In Croqtile, you applying the value numbering *GLOBALLY* but on top of AST. As it is a syntax-directed method without basic blocks and thus lack data flow analysis, it is impossible to inference a shape with varying dimensions.

=== UID:F21-S09
H: Limitations
P: More specifically, it has to set two limitations:
L: Any mdspan can not have element with bounded integer type, or produced from bounded ituple. | Simple integer type and ituple type can not be re-defined.

=== UID:F21-S10
H: Reference
P: [1] Briggs P, Cooper K D, Simpson L T. Value numbering[J]. Software: Practice and Experience, 1997, 27(6): 701-724.

=== UID:F05-S01
H: Parallelism: How Croqtile Represents Parallel Work
P: The previous two chapters kept things simple: `parallel i, j\` by [4, 8] created 32 instances, each handling one element, and you never asked where those instances actually ran. That was fine for element-wise addition and for understanding data movement --- the compiler picked sensible defaults.

=== UID:F05-S02
H: How You Think About Parallel Tasks
P: Before you touch any Croqtile syntax, let's talk about what "parallel" actually means.
P: Suppose you have eight independent tasks --- eight tiles to add, eight rows to process, whatever. You can describe all eight at once and call them "parallel." But that statement says nothing about how many actually run at the same time. On a single CPU core, they execute one after another. On a 4-core CPU, maybe four run simultaneously.
F: f05-fig1_virtual_parallelism_light.png
P: centerThe tasks are identical*center The point is that parallelism is a **virtual** concept. If you want to change the tiling, you also have to change the launch configuration. The logical and physical are tangled together. Croqtile untangles them.

=== UID:F05-S08
H: `: group-4` --- cooperative group
C: : group-4 --- cooperative group
P: 3.7 puts this to work with tensor-core operations.

=== UID:F05-S12
H: Matrix Multiply: Putting It Together
P: With `parallel`, `foreach`, `dma.copy`, `chunkat`, `#`, and space specifiers, a tiled matrix multiply is within reach.
P: The plan: multiply a `[128, 256]` matrix by a `[256, 256]` matrix to get a `[128, 256]` result.

=== UID:F15-S13
H: Control Structures
P: Croqtile differs from C++ in how it expresses parallelization and loops, but it follows C++ in using `if`-`else` blocks to handle branches inside a program.

=== UID:F30-S01
H: Sequentialism and Parallelism
P: Repetitive tasks are common in programming.
P: Currently, the most widely used parallelization paradigm is **Single Program, Multiple Data (SPMD)**. In this paradigm, code is written once but executed as multiple instances simultaneously. Each instance operates independently unless synchronization is explicitly implemented. Croqtile adopts this paradigm and abstracts parallelism using the **parallel-by** block.
P: Looping, on the other hand, is a more straightforward concept and is the most common method for writing sequential code. In Croqtile, programmers can construct loops using the **with-in + foreach** block. This structure supports *do-while* style iterations, which are particularly useful in scenarios involving multiple buffers, given Croqtile's primary focus on orchestrating data movement.
P: In the following sections, this section examines the details of constructing loops and implementing parallelism in Croqtile.

=== UID:F31-S01
P: A `parallel-by` block declares a set of independent instances; the space specifier says how those instances are grouped.

=== UID:F31-S02
H: Parallel Execution Block
P: In Croqtile, the code run in parallel are placed into the *parallel execution block*, which showcasing there are multiple instances of the same code that are executed simultaneously.

=== UID:F31-S07

=== UID:F31-S11
H: In short
P: A `parallel-by` block declares independent instances and a space specifier groups them; masking then decides which of those instances actually execute a statement.
P: Loops remain necessary for work that has to happen in order, and 2.7 introduces their Croqtile forms.

=== UID:F33-S01

=== UID:F33-S02
H: Parallelization and Masking
P: It is usually necessary to mask off some *parallel thread* in programming SPMD code. Programmers may invoke fixed number of parallel threads where the tasks can not evenly divided. Thus masking is necessary to close this gap: it makes only parts of the threads to work, while leaving the other threads wait the working threads..
P: In Croqtile code, the code may be implicitly masked, or explicit masked. Let us dive into the detail.

=== UID:F33-S03
H: Explicit Masking/Condition
P: Thread masking is a concept when programs have a full view of all instances of SPMD code. However, from the SPMD coding practise, which program is limited in single thread view, the thread masking appears as a boolean expression which typically is a comparison result related to thread identifiers.
P: In the examples that follow, Croqtile experimentally supports explicit masking of the `foreach` block, which mask off loop code as a whole. For example:
C: Explicit Masking/Condition

=== UID:F33-S04
H: Implicit Masking
P: Some code structures in Croqtile imply implicit masking. One typical situation is the multi-level `parallel-by` block. The below code showcases an example:
C: Implicit Masking
P: In this example, there exists two-level of parallelization. The code inside `parallel p by 6` block but outside `parallel q by 2` blocks is only for *parallel-level-0*. However, for target like , it could invoke 2 parallel threads in practise to execute either code inside *parallel-level-0* or *parallel-level-1*. Therefore, the *parallel-level-0*-only part is as if guarded with a C++ block `if (q == 0)`.
P: Conditional `foreach` Block Since It is possible to guard the `foreach` loop with conditions. One direct

=== UID:F33-S05
H: Experimental: Optional *Where-Binding* for `within`
P: You can append a *where-clause* to impose **Where-Binding** constraints among different bounded variables defined in a `with-in` statement. This allows different bounded variables to be treated as aliases for each other. Below is an example:
C: Experimental: Optional Where-Binding for within
P: Syntactically, the ``-> operation establishes the *where-binding* between two bounded variables. Within the `with-in` block, any reference to `n_p` refers to `n`, and vice versa.

=== UID:F34-S01
P: While Croqtile primarily targets SPMD (Single-Program-Multiple-Data) programming, it also supports MPMD (Multiple-Program-Multiple-Data) execution through thread masking. In this section, you demonstrate how to write divergent code using `inthreads` block in Croqtile.

=== UID:F34-S02

=== UID:F34-S03
H: Masking in SPMD
P: It is typically necessary to mask off some *parallel threads* in SPMD programming. For example, programmers often divide data into chunks for parallel or iterative processing. However, there may be a remainder that cannot be evenly distributed across all threads.
F: f34-masking-datachunk-dispatch.png
P: The figure aboves showcases a tiling of 13 data chunks, where 4 parallel threads are invoked for the processing of each chunk. In this case, 4 iterations are required to process all the data chunks. However, for the last iteration, only *thread-0* is required to work. You may observe the **thread mask** for each iteration are `1111`, except for the last iteration (`1000`). The thread mask is a combination of boolean bits, where `1` stands for **active**, and `0` stands for **inactive**. So in this case, only *thread-0* is set to be *active* to avoid unnecessary data chunk processing in the last iteration.

=== UID:F34-S04
H: Uniform and Divergence
P: Masking is an implementation-level concept. From the perspective of the SPMD programming, you typically claim the code that processes data chunks **diverges** because not all the thread instances execute the same code due to masking. And programmers should be aware of that their SPMD code is a combination of:
L: **Uniform Code**: *All threads* execute the same code path. | **Divergent Code**: Only the *masked threads* execute the *divergent* code path.
P: It is necessary to have the *divergent* code in SPMD code to handle different real-world requirements. And from a perspective of different programming paradigm, applying *divergent* code in SPMD programs can even mimic MPMD (Multiple Program Multiple Data) behavior. In MPMD, different threads execute different code. Therefore, *divergence* can extend SPMD to MPMD scope conceptually.
P: In Croqtile, you are able to create divergent code via `inthreads` block. Let us dive into the detail.

=== UID:F34-S05

=== UID:F34-S08
H: Masking for Multi-Levels
P: Thread masking can be applied at different parallel levels. The following example demonstrates this:
C: Masking for Multi-Levels
P: This code uses two levels of parallelization:
L: Outer level (`p`): Parallel count of 3 | Inner level (`q`): Parallel count of 4
P: The first `inthreads` predicate (`p ` 2 && q == 0) restricts execution to:
L: Outer parallel IDs: `0, 1` (excluding 2) | Inner parallel ID: `0` only
P: The thread mask (`011-0001`) reflects these constraints, where the two levels are separated by a hyphen (-).
P: The second predicate (`q == 1`) imposes no restriction on the outer level (`p`), allowing all outer IDs (`0, 1, 2`) while restricting the inner level to `q == 1`.

=== UID:F34-S10
H: In short
P: Masking is what allows divergent execution inside an otherwise uniform parallel region. Developers can use either explicit thread masking via `inthreads` blocks or implicit control through `parallel-by` constructs to restrict execution to specific threads.
P: While thread masking provides flexibility, it introduces thread asynchrony that complicates coordination. The next section will explore advanced techniques for managing these synchronization challenges.

=== UID:F05-S13
H: Scalar Matmul (No DMA)
P: The first version uses only `parallel` and `.at()`, before any tiling or data movement is added
C: Scalar Matmul (No DMA)
F: f05-fig5_scalar_matmul_light.png
P: centerThe [128, 256] output divided into a 16 $×$ 64 tile grid. Each (p, q) pair owns one tile.*center This is dense --- here is what each piece does. **Output shape from operand dimensions.** `s32 [lhs.`span(0), rhs.span(1)] output builds the output shape from the inputs: `lhs.span(0)` is 128 (rows of the left matrix) and `rhs.span(1)` is 256 (columns of the right matrix). **Multi-axis parallel.** `parallel p, q\` by [16, 64] declares two parallel indices of one parallel level with a comma. This creates a 16 $×$ 64 = 1024-way parallel grid. Each `(p, q)` pair owns a tile of the output. **Named tuple destructuring.** `foreach index = m, n, k\` in [128 / #p, 256 / #q, 256] introduces three nested loop indices --- `m`, `n`, `k` --- bound to a tuple called `index`. The trip counts are: itemize[leftmargin=1.6em] `128 /` #p = 128 / 16 = 8 rows per tile `256 /` #q = 256 / 64 = 4 columns per tile `256` for the full contraction dimension itemize **Composed global indices.** `p#m` composes the tile index with the within-tile offset. `p` selects which of the 16 row-tiles, `m` runs 0..7 within that tile, so `p#m` covers 0..127 across all tiles. **The arithmetic.** `output.`at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n) is the textbook dot product: for each output element, sum products along K.

=== UID:F05-S17
P: Pick global position `(row=37, col=50)` in the output. Working out which block and thread own it fixes the index arithmetic for the rest of the kernel.
P: Blocks partition 128 rows into 8 tiles of 16: `px = 37 /` 16 = 2, offset `qx = 37 % 16 = 5`. Columns partition 256 into 16 tiles of 16: `py = 50 /` 16 = 3, offset `qy = 50 % 16 = 2`. So block `(2, 3)`, thread `(5, 2)`.
P: For that thread, the K-loop runs 16 iterations. On iteration `tile_k = 0`, `dma.copy` loads `lhs` rows 32..47 and K columns 0..15 into shared `lhs_load`, and `rhs` K rows 0..15 and columns 48..63 into shared `rhs_load`. The inner `foreach k in [16]` accumulates `lhs_`load.data.at(5, k) * rhs_load.data.at(k, 2) for k = 0..15 into `output.`at(37, 50). Then `tile_k = 1` loads the next K-strip, and so on. After all 16 iterations, `output.`at(37, 50) holds the complete dot product --- same answer as the scalar reference.

=== UID:F04-S04
H: `chunkat`, `subspan`, and `view().from()`: Three Ways to Select a Tile
P: Croqtile gives you three common ways to describe the tile you want to move.

=== UID:F04-S13
H: `#`: Compose and Extent
P: Croqtile uses `#` in two related ways.

=== UID:F09-S06
H: Handling irregular access
P: The tiling primitives introduced so far --- `chunkat`, `subspan(.`..).at(...) --- assume tensors that divide evenly into tiles. This works for textbook GEMM where M, N, K are multiples of the tile dimensions. Production kernels are rarely so tidy. Expert batches start at dynamic offsets. Convolution windows overlap. The last K-tile is almost never an exact multiple of TILE_K. Some inputs arrive as flat buffers that need reshaping before MMA can consume them.
P: Croqtile provides four tools for these cases. Each one modifies how the compiler interprets a tensor's addressing without touching the pipeline structure --- DMA, TMA, MMA, events, swizzle all continue to work unchanged.

=== UID:F29-S05
H: Tile and Flow
P: It is possible to use the `chunkat` expression for either the source or the destination of a *DMA statement*. When `chunkat` is used only for the source, it typically divides the input or buffer into smaller chunks to move it to storage closer to the processor. In contrast, when `chunkat` is used only for the destination, it typically moves the buffer to compose part of the output. The following code showcases an example:
C: Tile and Flow
P: In this example, the statement applies `chunkat` to the destination, moving the data block in `shared` and "un-tiling" each block to the output. Note that the Croqtile compiler **requires the source data and destination data to have the same shape and element type** if not using *storage* as the destination *data expression*.
P: Using the `chunkat` expression for both the source and destination is rarely supported by any target platform at this time.

=== UID:F40-S07
H: Compose Two Bounded Variables for Tiling with `#`
P: In the example above, there is a need to tile the matrices in a manner that involves **two bounded variables**. This is where the **`#` operator** comes into play, allowing the program to compute the **Cartesian product** of two bounded variables (**not commutative**).
L: The code defines the tiling parameters with four bounded variables: `q`, `n`, `h`, and `w`, whose upper bounders are `6`, `N/6`, `H/16`, and `W/512`. | Suppose `N` is 18, `q` represents physical thread index. You want each thread to process data that is contiguous in the first dimension. That is, the first thread handles input[02][xxx], the second thread handles input[35][xxx], and so on. | `q # n` will result in a new bounded variable implicitly, whose upper bound is `N`: bound of `q` multiply bound of `n`. | For every data move, the stride of each dimension is 1, 3, 16, 512. In the last dimension, the index is `w`. In the first dimension, the index is `q * (bound of n) + n`.
L: The combination of `q` and `n` ensures that physical and virtual parallelism can work together efficiently. | This helps optimize **memory access patterns**, especially when working with large matrices, where efficient memory use is crucial. | Unlike higher-level loop scheduling techniques like in TVM, which abstract away the actual data movement, Croqtile explicitly links **virtual loops** to **physical DMA operations**. This results in more efficient data movement and computation.

=== UID:F09-S07
C: A window at a constant offset
P: The expression selects a `[16, 16]` slice starting at row 37, column 50. The origin can be any runtime value.
F: f09-fig4_view_from_light.png
P: The power of `view` / `from` is composability: once you carve out a window, all downstream operations --- `subspan`, `chunkat`, `dma.copy`, `tma.copy` --- apply to the window as if it were a standalone tensor. The pipeline stays unchanged; only the origin shifts:
C: A window at a runtime offset
P: In an MoE stack, each expert's token batch starts at a dynamic row `expert_offset`. The `view` / `from` call rewires the operand at the top of the pipeline; everything below --- DMA, staging, MMA loop --- runs the same code for every expert.

=== UID:F09-S08
P: Adding `.step(sM, sK)` overrides the stride: tiles are spaced `sM` rows and `sK` columns apart instead of being packed:
C: Strided tiles: .subspan, .step, and .at
F: f09-fig5_subspan_step_light.png
P: Tile `(0,0)` starts at `(0,0)`, but tile `(1,0)` starts at row 32 (not row 16), leaving a 16-row gap. Omitting `.step` uses a step equal to the tile size --- the packed case.

=== UID:F09-S09
H: Zero-padding: `.zfill`
P: Every tiled kernel faces the same edge case: what happens when M or K is not a multiple of the tile size? The last tile along that axis is partial --- some of its elements lie beyond the tensor boundary.
P: The brute-force fix is to add a special code path for the boundary tile. But this breaks the uniformity of the MMA loop and adds branches that the compiler cannot optimize away. `.zfill` solves this elegantly:
C: Zero-padding: .zfill
F: f09-fig6_zfill_light.png

=== UID:F09-S10
H: Layout reinterpretation: `span_as`
P: Some kernels load data as a flat 1D strip but need to feed it to MMA as a 2D matrix. A naive approach would copy the data into a reshaped buffer --- `span_as` avoids this by reinterpreting the existing buffer's shape in place:
C: A flat strip reinterpreted as a matrix
P: Element count is preserved; only the logical rank changes. No data is moved.
C: A movement statement using the reinterpretation
F: f09-fig7_span_as_light.png

=== UID:F09-S11
H: In short
T: | **Concept** | **Syntax** | **Role** | <ROW> | --- | --- | --- | <ROW> | Software DMA (Ch. 2, 6) | `dma.copy` / `dma.`copy.swiz<N> | Thread-cooperative tile transfer; | <ROW> | Swizzle | `.`swiz<N> on copy + `mma.`load.swiz<N> | layout conflict-free SMEM layout; same effect for DMA and TMA | <ROW> | Arbitrary windows | `view(M,N).`from(r,c) | Ragged or runtime-positioned slices | <ROW> | Strided tiling | `.`subspan().step().at() | Non-packed layouts, overlapping stencils | <ROW> | Partial tiles | `.zfill` | Zero-fill out-of-bounds elements | <ROW> | Shape reinterpretation | `span_`as([dims]) | Zero-copy reshape for staging buffers |

=== UID:F09-S12
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `tma.`copy.swiz<N> src => dst | TMA copy with swizzle mode `N` (0--3) | <ROW> | `dma.`copy.swiz<N> src => dst | DMA copy with swizzle mode `N` (0--3); same layout as TMA | <ROW> | `mma.`load.swiz<N> src | MMA operand load consistent with swizzle `N` | <ROW> | `.`view(M, N).from(r, c) | Arbitrary-offset `M x N` window | <ROW> | `.`subspan(M, K).step(sM, sK).at(i, j) | Strided tile selection | <ROW> | `.zfill` | Zero-fill out-of-bounds elements on copy source | <ROW> | `span_as(dims)` | Reinterpret linear storage as shaped tensor |
P: 3.6 takes the tile expressions above and uses them as the source and destination of movement statements. For the cases where a program needs plain C++, 2.13 collects the `__device__` functions, preprocessor guards and injected source text.

=== UID:F03-S07
H: What You Have So Far
P: The constructs introduced so far assemble into one skeleton: a `__co__` function with typed, shaped parameters and a return value; `parallel` and `.at()` for element-wise computation; `lhs.span` for deriving output shapes from inputs; and the host-side `make_spandata` / `.view()` API for creating and passing tensors.
P: 2.9 introduces `dma.copy` and `chunkat` to express block-level data movement.

=== UID:F04-S01
H: Data Movement: The Tile Movement Engine
P: 1.4 used scalar indexing: pick one position, read inputs, compute, write output.
P: Croqtile makes that explicit. Instead of treating data movement as a hidden side effect of indexing, it gives you a small family of tile-movement statements:
L: `dma.copy` moves a tile as-is. | `dma.`transp<...> permutes dimensions while moving it. | `dma.`pad<...> enlarges a tile and fills the border.
P: This part of the guide introduces that model, shows the core syntax, and gives the rules you need to write DMA and TMA code that is both correct and likely to lower well.
F: f04-fig1_element_vs_block_light.png
P: centerLeft: element-by-element thinking. Right: tile-by-tile movement into fast memory.*center

=== UID:F04-S10
H: Other Operations: `transp` and `pad`

=== UID:F04-S20
H: DMA vs TMA
P: Both engines move shaped tiles. The difference is how they do it and what constraints they require.
T: |  | **`dma`** | **`tma`** | <ROW> | --- | --- | --- | <ROW> | Common destination | `shared` | `shared` | <ROW> | Async support | Yes | Yes |
P: Example TMA use:
C: DMA vs TMA
P: The practical rule is simple:
L: Use `dma` when tile sizes are dynamic, irregular, or you want the portable path.

=== UID:F04-S21
H: The main TMA constraint
P: TMA tile shapes must be host-computable. Runtime tile indices are fine:
C: A host-computable tile shape
P: But a kernel-runtime value cannot define the tile shape itself:
C: A runtime tile index that is rejected
P: If the tile size truly depends on kernel-runtime state, use `dma.copy` instead.

=== UID:F09-S02
H: `tma.copy`: target tensor movement
P: The surface syntax mirrors `dma.copy`:
C: tma.copy: target tensor movement
P: Same source expression, same `=` destination form. The difference is **who does the work**:
F: f09-fig3_tma_descriptor_light.png
L: **DMA path.** Threads cooperate to cover the tile; each lane participates in address math and load issue. | **TMA path.** One descriptor-based operation describes the tensor slice;
F: f09-fig1_tma_vs_dma_light.png

=== UID:F15-S18
H: Async Operation: the DMA Statement
P: Except for parallel execution, Croqtile allows one fixed form of async operation: the DMA statement.
P: Conceptually, a DMA statement is executed asynchronously with the SPMD code. It works quite similar to CPU async thread, except its behavior is limited by the DMA configuration. (CPU allows to program the async thread as will)
P: The below code showcases one basic DMA statement.
C: An asynchronous DMA declaration
P: You might observe the assignment of 'f'. This is the handler for the asynchronous DMA entity. In Croqtile, you refer to it as the 'future' of the DMA operation. Programmers have the option to code the explicit synchronization statement 'wait' to pause the current thread until the 'future' arrives. Otherwise, the thread continues execution in parallel with the DMA operation.
P: At times, programmers may find it tedious to explicitly define temporal data. Croqtile offers an even simpler syntax:
C: The same declaration with storage
P: Here, you do not need to specify the exact target location where the DMA transfers data. Instead, you only specify the destination memory type. This is advantageous in many scenarios. And to retrieve the transferred data, you simply invoke the 'data' member function of 'future'. Besides, the dma operation is not marked with '.async', so the wait operation on 'f' is not needed and cannot be performed.
P: DMA operations entail intricate details that demand careful programming. Programmers should refer to the DMA manual to make informed decisions for their code. Nonetheless, Croqtile compiler provides plenty of static and runtime checks to assist programmers in avoiding potential errors in this aspect.

=== UID:F28-S02
H: Data Movement Statement
P: Croqtile aims to address multiple levels of data movement. Croqtile adopts this terminology and abstracts data movement as **DMA Operations**.

=== UID:F28-S05
H: Operation Type
P: Croqtile maps these functionalities at the software level as different **DMA Operation Types**. The supported operations include:
L: `dma.copy`: Copies flat memory directly. | `dma.pad`: Pads shaped data while transferring the data. | `dma.transp`: Transposes shaped data while transferring the data.
P: The configurations, such as padding and transposing details, are programmed as parameters of the DMA operation. For example:
C: Operation Type
P: Here, the **DMA configuration**s are enclosed by ``>. The configuration varies according to different operations. The detailed configuration syntax and limitations for a specific platform, such as , are listed below as an example:
P: Programmers should note that support for DMA types other than `dma.copy` varies by platform. For instance, on the platform, it could be either a mapping of TMA or an orchestration of load instructions from multiple threads.

=== UID:F39-S01
P: The data movement is essential for *tileflow programs*. The *DMA statement* has a form for each of those levels, from a single copy to a chained non-blocking transfer.

=== UID:F39-S02

=== UID:F39-S03

=== UID:F39-S04

=== UID:F39-S05
H: Full Non-Blocking DMA Mode (Chain Mode in Croqtile)
P: In Croqtile, it's possible to perform a full non-blocking DMA by chaining multiple asynchronous DMA operations and using event-based notifications with after. This enables complete non-blocking execution, where one DMA operation is triggered only after the completion of a prior one. Here's an example of such a setup:
C: Full Non-Blocking DMA Mode (Chain Mode in Croqtile)
P: In this example:
L: `out_store` is the asynchronous DMA operation that transfers data from l2_out to output.chunkat(m_tile, n_tile). | `out_store_s` is another DMA operation or event that must complete before out_store can proceed. | The after out_store_s syntax specifies that `out_store` should only start after the completion of `out_store_s`, which ensures that there is no blocking in the main thread.
P: In this case, neither `out_store` nor `out_store_s` will block the main program flow. The program continues executing while these DMA operations are handled in the background. The key difference here is that the completion of `out_store_s` triggers the start of `out_store`, creating an event-driven dependency between the two DMA operations. This model enables highly efficient and non-blocking memory transfers.

=== UID:F39-S06

=== UID:F39-S07

=== UID:F39-S08

=== UID:F05-S14
H: DMA Matmul: Tiles in shared storage
C: DMA Matmul: Tiles in shared storage
F: f05-fig6_dma_matmul_light.png
P: The scalar version indexed one element at a time. The tiled version first copies a tile into shared storage, then computes on that tile:
L: The outer `parallel` now uses `: block` with brace-form indices `px, py\`. An 8 $×$ 16 grid of blocks covers the output.
L: A `foreach` over `tile_k` walks the K dimension in 16 steps. Each step copies one strip of `lhs` and one strip of `rhs` into `shared` storage with `dma.`copy ... => shared.
L: An inner `parallel qx, qy\` by [16, 16] : thread creates 256 threads within each block. Each thread owns one output element within the block's tile.
L: The arithmetic reads from `lhs_load.data` and `rhs_load.data` --- the shared-memory copies --- instead of global `lhs` and `rhs`.
P: The composed indices work in two layers: `px#qx` composes block index `px` with thread index `qx` to form the global row; `py#qy` does the same for columns.

=== UID:F08-S02
H: Double buffering with `swap`
P: Give the K-loop **two** logical buffers. While the math drains buffer 0, DMA fills buffer 1 with the next tile. After the math step, **swap** the handles: what was "next" becomes "current," and the freed slot is ready for the following load.
P: Croqtile spells this with `dma.copy.async` (non-blocking copy), `dma.any` (a placeholder future), `swap` (exchange future handles), and a three-phase loop:
C: Double buffering with swap

=== UID:F08-S03
H: The three phases

=== UID:F08-S04
H: `swap`: names, not bytes

=== UID:F08-S06
H: `dma.any`: placeholder futures

=== UID:F38-S01
H: Multi-Buffering Optimization with Croqtile
P: To aggressively optimize the high-performance computation kernels, one effective method is to overlap computation with DMA (Direct Memory Access) data movement. This approach, known as **multi-buffering optimization**, typically requires programmers to manage multiple buffers.
P: In Croqtile, this method requires two key programming primitives: the 'dummy' future **dma.any**, and the **swap/rotate** function. Let's illustrate by an example:
C: Multi-Buffering Optimization with Croqtile
P: In croqtile function `ele_add`, there are two input parameters: 'lhs' and 'rhs', both double-buffered in local storage. You refer to these buffers as 'A' and 'B'. While buffer A is used for computation, buffer B is filled with data via a DMA operation. Once the kernel function completes computation on buffer A, the roles of A and B are swapped: A becomes the buffer for loading the next data chunk, and B becomes the buffer for computation. This process iterates until all data chunks are consumed.
P: The implementation divides data movements and computations into three stages, including:
L: the **prologue**(line 8-11). It (pre-)load the first chunk of data. In this example, buffer A is pre-loaded with data. | the **body**(line 12-20). It processes the data pre-loaded in last iteration or in prologue, and pre-load the data for next iteration. | and the **epilogue**(line 21-25). It processes the last data chunk only.
P: In the code, futures 'lfB' and 'rfB' are declared as dummies. These dummy futures serve as placeholders and are replaced by DMA statements at lines 13-14. The reason for declaring dummy futures is that the futures ('lfB' and 'rfB') of the DMA invoked in the body stage are used in the epilogue stage (lines 21-22). However, from a lexical scope perspective, defining futures inside the foreach-block (lines 12-20) would not extend their lifetime to their last uses. Therefore, it is necessary to declare 'lfB' and 'rfB' early.
P: The **swap** statements at lines 18-19 exchange futures to ensure the **wait** statement at line 15 functions correctly. Note that only futures with identical DMA operations can be swapped. Additionally, the iteration range of the *foreach* statement at line 12 is adjusted. `foreach y(1:)` means the loop over the *bounded variable* 'y' starts from the second value up to the upper bound. In line 17, the *chunkat* expression is also tuned. `y - 1` results in the current value of 'y' minus 1, setting up the body stage to work properly with the prologue stage. This adjustment is necessary because the iteration count must be reduced by 1, and the pre-load operation should fetch the 'next' chunk while computation is conducted on the 'current' chunk of data. Alternatively, you can use `foreach y(:-1)` and adjust the chunkat expression of the pre-load statements at lines 13-14 to achieve the same effect.
P: In line 25, the expression `y(-1)` retrieves the 'last value' of 'y' within its bound.
P: The **select** expressions in line 21-22 handle both odd and even upper-bound cases for the bounded variable 'y'. In either case, it obtains the future of the last chunk.
P: If the programmer want to enable more than two buffers, **rotate** statements is required to replace **swap**.

=== UID:F09-S05
H: Example: TMA in a pipelined matmul
P: The pipeline skeleton from 3.8.5 is unchanged: ring of stages, `wait` / `trigger` on events, MMA commit, consumer drains tiles while producer fills the next slot. Here the producer swaps `dma.copy` for `tma.`copy.swiz<3> and the consumer swaps `mma.load` for `mma.`load.swiz<3>:
C: Example: TMA in a pipelined matmul
P: Compared to the 3.8.5 `dma.copy` version, only the ingress and operand load lines change; events, staging indices, and commit stay the same. --- Choose TMA or DMA for stores according to your target.

=== UID:F04-S18
H: Partial Tiles, Guarding, and `.zfill`
P: Real kernels often end with a partial tile. The important distinction is whether the **source tile shape** and **destination tile shape** are the same.
P: For example:
C: Partial Tiles, Guarding, and .zfill
P: Here `lhs_load_s` might have shape `[group_`M, TILE_K], while the source view has shape `[ROWS_`M, TILE_K] and `ROWS_M` may be smaller than `group_M` on the last tile. That is a real shape mismatch, so `.zfill` means: copy the valid source rows and write zero into the remaining rows of the destination tile.
P: That is the right use for `.zfill`.
P: What happens when the shapes are the same:
L: If the source view and destination tile have the same shape, the compiler already guards runtime tails automatically. | In that case it generates masking so out-of-bounds elements are not read or written. | This is enough for cases like tiling a `[100]` tensor into `[64]` tiles: the last tile simply has 36 valid elements.
P: The `.zfill` form is **not** the general mechanism for ``any partial tile.'' It is specifically for cases where the destination tile is larger than the valid source region and the extra part of the destination must be filled with zeros.
P: Why `.zfill` matters:
L: If the consumer reads only the valid region, the compiler's automatic masking is usually enough. | If the consumer reads the whole destination tile, especially an MMA fragment, the invalid region must hold a known value. | `.zfill` makes that extra region zero.
P: Use `.zfill` when the valid source region is smaller than the destination tile and the consumer will treat the destination as a full dense tile. This is the common case for GEMM-style pipelines with dynamic row counts or padded edge tiles.
P: If you add `.zfill` when the source and destination shapes are already the same, the compiler can warn that the `.zfill` is redundant. Conversely, when the shapes differ, the compiler can also warn that `.zfill` is needed to avoid using undefined values in the uncovered region.

=== UID:F04-S19
H: `.`swiz<N>: Shared-Memory Layout for the Consumer
P: Typical GEMM staging code looks like this:
C: .swiz N : Shared-Memory Layout for the Consumer
P: The load-side and compute-side swizzles need to agree. If you stage a tile with `.`swiz<128>, the consumer should usually load it with the matching swizzle.
P: This part of the guide only introduces the idea. The MMA chapters go deeper into which swizzle values pair well with which matrix layouts.

=== UID:F06-S01
H: Tensor Contraction: the `mma` Syntax
P: 3.4's tiled matmul computes the inner contraction the way a CPU would: a `foreach k` loop where each iteration multiplies two scalars and accumulates into one output element. That works.
P: The job is **2D tensor contraction** --- the operation C += A $×$ B where A, B, and C are small, fixed-shape matrix tiles. It is the inner kernel of every GEMM, every convolution expressed as an im2col matmul, every attention head's QKT.
P: centerThe same mathematical operation --- C += A $×$ B on tile-shaped operands*center What makes this hard for programmers is not the math but the . Get the layout wrong --- say, load a column-major tile into a row-major fragment --- and the result is silently incorrect. figure[htbp] [width=0.86,height=0.42,keepaspectratio]f06-fig2_register_loading_light.png figure Croqtile's design sidesteps this complexity entirely. figure[htbp] [width=0.86,height=0.42,keepaspectratio]f06-fig3_mma_syntax_light.png figure centerThe four-step MMA syntax is an abstract interface*center

=== UID:F06-S03
H: Scaling the cooperation scope
P: The four-step syntax stays the same regardless of how many threads cooperate on a single contraction. What changes is the **cooperation scope** --- And that progression is the story of this section.

=== UID:F06-S07
H: Beyond dense FP16: what else the four steps express
P: The examples above use `mma.row.row` on dense FP16 tiles. The same four-step pattern extends to workloads the basic form cannot reach.
P: But it needs a **metadata operand** `me` that encodes which elements are nonzero:
C: A sparse contraction
P: The `.sp` suffix adds the metadata operand; everything else is the same fill-load-multiply-store rhythm. Any layout combination works: `mma.row.col.sp`, etc. You load `me` from a separate metadata tensor alongside A and B.
C: A scaled contraction
P: Each result element is scaled by `sc_a` and `sc_b` after the contraction --- no separate dequant kernel needed. Alternatively, scaling can be a **standalone statement** when the scale source differs from the standard fused path:
C: Another contraction layout
P: The standalone `mma.scale` appears in some MoE and mixed-precision kernels.
P: The ``N> must agree between `tma.`copy.swiz<N> and `mma.`load.swiz<N> --- a mismatch reads garbage. For output, `mma.`store.transp mc, dst writes the accumulator with rows and columns swapped, useful when the next stage expects column-major data.
P: The table below collects every variant for reference.

=== UID:F09-S03
H: Swizzle and layout conflicts
C: A swizzled copy
P: The MMA read path must use the same `swiz`N> so addresses match the staged layout:
C: A swizzled mma load
P: Swizzle is not a TMA-specific feature. In Croqtile, `dma.`copy.swiz<N> and `tma.`copy.swiz<N> produce identical shared-memory layouts. The difference is only in how the data gets there; the layout that arrives is the same in both cases.
F: f09-fig2_swizzle_light.png

=== UID:F06-S04
H: One group, one tile: the simplest MMA matmul
P: One cooperative group is written `: group` in Croqtile.
C: One group, one tile: the simplest MMA matmul
P: With all dimensions 16, extents are 1$×$1
P: 3.6 goes deeper on `subspan` versus `chunkat`.
B: `subspan(h, w)` declares a fixed-size rectangular view shape, and `.at(i, j)` selects which view instance you want from the parent tensor. In other words, `lhs.`subspan(TILE_M, TILE_K).at(block_m, iv_k) means "take the `[TILE_`M, TILE_K] tile at `(block_`m, iv_k)." Use `subspan` when you want an explicit tile shape; use `chunkat` when you want to slice by equal chunks of an existing dimension.
P: The four-step syntax reads linearly and the tile geometry is obvious.

=== UID:F06-S05
H: Widening the scope: cooperative group and mma
P: The four-step syntax does not change. The only thing that changes in Croqtile is the space specifier: `: group-4` instead of `: group`.
F: f06-fig4_sm86_vs_sm90_light.png
C: Widening the scope: cooperative group and mma
P: Fill, load, multiply, store --- the rhythm is identical. The differences are mechanical:
P: That is the whole point of the abstraction: the same four operations, the same layout contract, the same `chunkat` / `subspan` expressions. You think about *what* contraction to perform; the cooperation width is a deployment detail.

=== UID:F06-S06
H: Tiling further: two cooperative group sharing operands
P: 3.4 introduced `parallel p1 by 2 :` group-4 --- with MMA, both groups can share the same B tile while loading different rows of A.
C: Tiling further: two cooperative group sharing operands
P: The four-step syntax stays invariant; only the parallel decomposition of the tile changes.
T: | **Aspect** |  |  |  | <ROW> | --- | --- | --- | --- | <ROW> | Operand sharing | N/A | N/A | B tile shared, A rows split by `p1` | <ROW> | Threads | 32 | 128 | 256 | <ROW> | Accumulator | `mma.fill 0.0` | `mma.`fill.f16 0.0f | `mma.`fill.f32 0.0f |

=== UID:F06-S08
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `mc = mma.`fill 0.0 | Initialize accumulator tile to zero | <ROW> | `mma.`fill.f16 0.0f / `mma.`fill.f32 0.0f | Accumulator with explicit precision | <ROW> | `mma.`load.swiz<N> src | Load with swizzle mode (see *Ch 7*) | <ROW> | `mma.`row.row mc, ma, mb | C += A $×$ B (both row-major) | <ROW> | `mma.`row.col mc, ma, mb | C += A $×$ B (A row-major, B col-major) | <ROW> | `mma.`row.row.sp mc, ma, mb, me | Sparse MMA with metadata operand | <ROW> | `mma.`row.row.scale mc, ma, mb, sc_a, sc_b | Fused MMA + per-tile dequantization | <ROW> | `mma.`scale mc, sc_a, sc_b | Standalone post-MMA scaling | <ROW> | `mma.`store.transp mc, dst | Write accumulator transposed | <ROW> | `cdiv(a, b)` | Ceiling division | <ROW> | `_`_co__ void fn(...) | Kernel that writes results in-place |

=== UID:F06-S09
H: In short
T: | **Topic** | **Takeaway** | <ROW> | --- | --- | <ROW> | 2D tensor contraction | C += A $×$ B on tile-shaped operands --- the universal inner kernel | <ROW> | Four-step abstraction | **fill $→$ load $→$ multiply $→$ store**; the compiler handles fragment layouts for each target | <ROW> | Scaling cooperation | Syntax stays invariant | <ROW> | Layout contract | `mma.row.row`, `mma.row.col`, etc. | <ROW> | Sparse and quantized | `.sp` adds a metadata operand; `.scale` fuses per-tile dequantization | <ROW> | Swizzle and pipeline | `mma.`load.swiz<N> matches swizzled shared layouts; `mma.commit` fences pipeline stages |
P: The contraction is fast, but loads and compute still take turns --- 3.8 introduces **role specialization** so different thread groups can play different roles: one group loading data while another computes, overlapping memory and math in time.

=== UID:F07-S01
H: Control Flow: Divergence, Concurrency, and Coordination
P: Every programming language that targets a parallel machine must answer one question: **what happens when different threads need to do different things?**
P: On a CPU, the answer is trivial --- each core has its own program counter, so each thread runs whatever code it likes. --- It must **serialize** the divergent paths and mask inactive threads.
P: All threads evaluate the condition; threads where it is false are masked (deactivated); This is **predicated execution** --- simple, universal, and sometimes ruinously expensive.
P: Predicated execution is not enough. This is not a data-dependent branch --- it is two **structurally different programs** that happen to share an address space. Trying to express this with `if` means one program pauses while the other runs. There is no overlap, no pipeline.
P: Croqtile introduces two additional control-flow primitives to fill this gap:
L: **`inthreads.`async** --- **structured concurrent regions**: compile-time partitioning of threads into groups that run **different programs simultaneously**. The compiler generates separate instruction streams; | **`shared event` / `wait` / `trigger`** --- **inter-region signaling**: lightweight synchronization tokens that let concurrent regions communicate safely.

=== UID:F07-S03
H: Structured concurrent regions with `inthreads.`async
C: Structured concurrent regions with inthreads.async
P: The distinction from `if` is structural, not just performance:
T: |  | **`if` (predicated execution)** | **`inthreads.`async** (structured concurrency) | <ROW> | --- | --- | --- | <ROW> | **Resolution** | Runtime --- every thread evaluates the condition | Compile time --- thread assignment is fixed | <ROW> | **Instruction streams** | One program; divergent threads masked | Separate programs per region | <ROW> | **PL analogy** | `if`/`else` in any language | `async`/`spawn` in structured concurrency (Trio, Go goroutines, Cilk) |
P: The figure below shows the effect. --- The producer's DMA and the consumer's MMA overlap in time:
F: f07-fig1_role_comparison_light.png
P: centerEach waits for the other. Bottom: `inthreads.`async* partitions into two concurrent programs --- DMA and MMA overlap, roughly halving wall-clock time.center

=== UID:F07-S05
H: Inter-region signaling with events
P: When `inthreads.`async creates concurrent regions, those regions need a way to communicate. Croqtile provides **events**
C: Inter-region signaling with events
P: Events have two operations:
L: **`trigger name`** --- signal that a condition is met (e.g., "data is ready") | **`wait name`** --- block until the corresponding `trigger` fires
P: The producer calls `trigger full` after writing a tile to signal "data ready." The consumer calls `wait full` before reading, blocking until the signal arrives. Symmetrically, the consumer triggers `empty` after finishing its read (the buffer can be reused), and the producer waits on `empty` before writing the next tile.
P: This is a **credit-based bounded buffer** protocol --- `full` is the "data available" credit; `empty` is the "buffer free" credit.

=== UID:F08-S01
H: Synchronization in Practice: Pipelines, Buffers, and Events
P: 3.8 introduced three control-flow primitives: **`if`** for predicated execution, **`inthreads.`async** for structured concurrent regions, and **events** (`shared event` / `wait` / `trigger`) for inter-region signaling. This part of the guide puts them to work. You will see two progressively complex kernel patterns, each demonstrating how the primitives compose to solve real synchronization problems.
P: The first pattern --- **double buffering with `swap`** --- uses a single thread group that interleaves loading and computing within one program. No `inthreads.`async, no events: just two buffer handles and a rotation. The second pattern --- **the full 1P1C event pipeline** --- splits loading and computing into separate concurrent programs with `inthreads.`async and coordinates them with event arrays.
P: Both patterns solve the same underlying problem: **you cannot read a buffer while someone is writing to it**. They differ in how they structure the solution.

=== UID:F08-S09
H: Why you cannot "just overlap" without events
P: The `swap` pattern works because one thread group controls both loading and computing --- it knows the order. They cannot share a `swap` schedule; they need a signaling mechanism.
P: The picture below contrasts a strict load-then-compute staircase with double-buffered overlap: same logical work, less idle time.
F: f08-fig1_pipeline_timeline_light.png

=== UID:F34-S07
H: Make it Asynchronous
P: Similar to the `parallel-by` statements, it is possible to make the threads with/without executing the divergent code be asynchronous. The following code showcases a example that conduct MPMD execution in the SPMD code:
C: Make it Asynchronous
P: In this example, it divides the parallel threads evenly into two sub-groups, without forcing synchronization after each `inthreads` block. In this way, threads in different groups are executed in parallel, mimicking a MPMD executing. However, the `sync.shared` statement established a **synchronization point** for all threads. That makes sure code for both paths get executed after this *synchronization point*.
P: Only the outer `inthreads` can be declared as asynchronous. Programmers will get an compile-time error when using `inthreads.`async for any inner `inthreads` block.

=== UID:F35-S03
H: Event Instances
P: In Croqtile, events are associated with the storage specifier, either `global`, `shared`, and `local`. Such storage specifier specifies the event instances.
C: Event Instances
P: The comments in the above code explain the event instances. The mechanism is same to the spanned data declarations, where storage specifier is also required. From implementation perspective, event can be implement as the leveled boolean data within the specified storage. Therefore, it also takes the storage of this level. However, in concept, the event belongs to a specific level. Therefore, its instance count coresponds to its parallel level.

=== UID:F35-S04
H: DeadLocks
P: By default, the event is lock-free for high-performance synchoronization between different asynchronous blocks. That implies it is risky when multiple threads try to update the event, especially for `wait`.
P: For example,
C: A single-instance deadlock
P: In this code, two threads wait a single event `e`. However, since `wait` resets the event `e`, it is possible that `thread-0` resets the event before `thread-1` find the `e` has been triggered. As a result, `thread-1` is never unblocked and deadlocking happens.
C: A credit-count deadlock
P: In this code, you utilize an event array to guarantee each event is waited and reset by a single thread.

=== UID:F07-S04
H: The canonical pattern: 1 producer + 1 consumer
P: The most common use of `inthreads.`async is the **1P1C** (one producer, one consumer) split for matmul:
C: The canonical pattern: 1 producer 1 consumer
P: Sharing an address space is exactly what makes this dangerous. Without coordination, the consumer might read a buffer before the producer has finished writing it. This is where events come in.

=== UID:F07-S08
H: A 1P1C matmul skeleton
P: Event-based synchronization is left out of this skeleton; 3.8.5 adds the full pipeline protocol. Focus on the program structure:
C: A 1P1C matmul skeleton

=== UID:F07-S09
P: In the earlier kernels, the grid grew with the problem: roughly one block per output tile. For large matrices that means large launch counts, and the last wave of blocks often leaves SMs partially idle --- **tail underutilization**.
P: A **persistent kernel** fixes the launch size (often near the SM count) and lets each block iterate over multiple tiles:
C: Persistent scheduling and the if guard
F: f07-fig2_persistent_kernel_light.png

=== UID:F07-S10
H: Data-dependent vs persistent grids
T: | **Aspect** | **One block per tile** | **Persistent (`NUM_GROUPS` blocks)** | <ROW> | --- | --- | --- | <ROW> | Grid size | Grows with problem | Fixed | <ROW> | Tail utilization | Last wave may leave SMs idle | All SMs stay busy | <ROW> | Extra constructs | Minimal | `total_tiles`, `tile_`iter # block_id, `if` | <ROW> | Complexity | Lower | Higher |

=== UID:F07-S11
H: `parallel.async`: host-level concurrency
P: Everything above runs inside a kernel. Sometimes you need concurrency at the **host level**: launch a grid without blocking the CPU.
C: parallel.async: host-level concurrency
P: This is **host orchestration**, orthogonal to in-kernel control flow. It does not replace `inthreads.`async for thread partitioning or `if` for runtime predicates --- it decides *when* and *where* a grid runs relative to other grids.

=== UID:F31-S10
H: Asynchronous program launch and Stream Binding
P: By default, the host waits for a `parallel-by` kernel to complete before continuing. To launch kernels asynchronously, use `parallel.async`.
C: Asynchronous program launch and Stream Binding
P: 3.8.4 covers this in detail.

=== UID:F36-S01
H: Synchronization and Asynchronous `parallel-by`
P: Similar to `inthreads`, `parallel-by` can also be made asynchronous by appending `.async`:
C: An asynchronous launch
P: Without `.async`, the host waits for the kernel to finish before continuing:
C: The synchronous form
P: With `.async`, the host code continues immediately after launching the kernel. This is essential for overlapping computation with data transfers or launching independent kernels concurrently.

=== UID:F36-S02
P: In Croqtile, you can bind a `parallel-by` block to a specific stream using angle-bracket syntax:
C: Stream Binding
P: In this example:
L: **`stream s0, stream s1`**: Stream parameters passed from the host. | **`parallel.`async**s0>: Launches the kernel asynchronously on stream `s0`. | **`parallel.`async**s1>: Launches a second kernel on stream `s1`.

=== UID:F36-S03
H: Stream Binding without `.async`
P: Stream binding also works with synchronous `parallel-by`. In this case, the host launches the kernel on the specified stream and then waits for that stream to complete:
C: Stream Binding without .async

=== UID:F36-S04
H: Rules
L: Stream binding is only allowed on **block-level** parallel (`: block`). | The identifier inside ``...> must be a `stream`-typed variable. | Without stream binding, synchronous `parallel-by` uses device-wide
P: synchronization, and asynchronous `parallel-by` uses the default stream.

=== UID:F08-S10
H: The full 1P1C event pipeline
P: This kernel combines `inthreads.`async (from 3.8) with event arrays to build a complete multi-stage pipeline. The producer and consumer run as separate concurrent programs, coordinated entirely through `wait` / `trigger`:
C: The full 1P1C event pipeline

=== UID:F08-S11
H: Walking through the kernel

=== UID:F08-S12
H: Credit flow for one stage
P: The diagram matches the code: bootstrap grants empty credits; the producer waits on `empty`, fills, signals `full`; the consumer waits on `full`, computes, signals `empty`. When `iv_k` wraps modulo `STAGES`, the same physical stage re-enters the cycle.
F: f08-fig2_event_credit_flow_light.png

=== UID:F07-S14
H: In short
T: | **Concept** | **Primitive** | **When to use** | <ROW> | --- | --- | --- | <ROW> | Predicated execution | `if` | Data-dependent decisions (bounds, conditions) | <ROW> | Structured concurrency | `inthreads.`async | Compile-time thread partitioning (producer/consumer, heterogeneous roles) | <ROW> | Inter-region signaling | `shared event` / `wait` / `trigger` | Coordination between concurrent regions | <ROW> | Host concurrency | `parallel.async` / `stream s` | Multi-kernel overlap, non-blocking launch | <ROW> | Persistent scheduling | `if` + `foreach` + `#` | Fixed grid size, tile striping with padding guard |
P: The 1P1C skeleton above is incomplete: without `wait` / `trigger`, the consumer can read before the producer has finished writing. 3.8.5 adds the buffering and event protocols that make the pipeline safe, and walks through the resulting kernel stage by stage.

=== UID:F08-S15
H: In short
T: | **Pattern** | **Primitives used** | **When to use** | <ROW> | --- | --- | --- | <ROW> | `swap` double buffering | `dma.any`, `swap` | Single thread group interleaving load and compute |
P: Both patterns achieve the same goal --- overlapping memory and compute --- but at different levels of complexity. Start with `swap` for simpler kernels;
P: The same synchronization patterns from this part of the guide, with richer data movement primitives underneath.

=== UID:F10-S01
H: C++ Interop: Device Functions, Inline Code, and the Preprocessor
P: Every high-level language needs an escape hatch to the layer below. Python has `ctypes` and C extensions. Rust has `unsafe` blocks and `extern "C"`. Java has JNI.
P: Croqtile's interoperability story has three parts:
L: **`__device__` functions** --- The compiler passes them through unchanged. | **`__cpp__`** | **The preprocessor** --- `#define`, `#if`, `#ifdef`, `#error` for compile-time configuration, the same role `#define` plays in C/C++ codebases.
P: The preceding sections have built a stack of abstractions: tiles, parallelism, MMA, pipelines, events, TMA. That is the Croqtile you want to live in. This part of the guide is about the times you need to step outside.

=== UID:F10-S03
H: `__device__` vs ordinary C++ inline
P: Both `__device__` functions and ordinary C++ functions are passed through unchanged, but they land in different compilation paths --- and the distinction matters.
P: A `__device__` function compiles to . The `__co__` body calls it with the `call` keyword, and the generated `__global__` kernel invokes it directly on the device.
P: An ordinary C++ function (no `__device__` or `__co__` annotation) compiles to **CPU code**. It runs on the host --- setting up buffers, launching kernels, printing results. You cannot call a host function from inside a `__co__` body;
P: The boundary is strict: `__device__` lives on the device side of the `c++` split, host C++ lives on the host side. The `__co__` function is the bridge --- the compiler transforms it into a `__global__` kernel (device side) and generates a host-side wrapper that launches it.

=== UID:F23-S05
H: Difference with C++ pre-processing
P: Croqtile preprocessing runs much earlier than c++ preprocessing. The workflow of croqtile compilation is as shown below:
C: Difference with C pre-processing
P: The primary target of croqtile preprocessing is to make host/device macros work as a whole, but such a workflow makes it possible sometimes different. From an implementation perspective, Croqtile pre-processor only substitute/conditionally-compile code inside the tileflow function, while leaving other pre-processing to the C++ preprocessor. That could restrict Croqtile pre-processing in a limited scope.

=== UID:F23-S06
H: Pre-defined Macros
P: To mimic a target native compilation, croqtile preprocess also takes the builtin macros from the target. Consequently, these macros can be utilized inside tileflow functions as well as the host code.

=== UID:F24-S05

=== UID:F24-S06
H: Trigger C++ Template Instantiation
P: It is possible for Croqtile code to make function call to a template function. For example:
C: A templated device helper
P: In this case, it defines device function template named `matmul_kernel`, which takes three template parameters. As the caller specified the template function arguments, it triggers the instantiation of the function template, which results in a template function `matmul`16, 4, 72> for the call.
P: So in Croqtile, calling a template function is similar to those in C++. However, **the template argument passed must be able to be inferred as compile time constant value** by Croqtile compiler. Therefore, any runtime values can result in error.
C: A call that instantiates it
P: Croqtile compiler could inference values as much as possible at compile time. If the template argument can not be inferred, the compilation will abort and error will be emitted.

=== UID:F25-S08
H: More Questions
P: Like C++ functions, Croqtile functions can be defined anyway in the source code. One common raised question is about the **function signature**, as long as programmers may want to link the module with Croqtile function definition.
P: The answer is that Croqtile function follows the C++ calling convention, which makes it standard C++ function with C++ name mangling. **`extern "C"`** is not possible for a Croqtile function, since it utilize C++ template-based object as its parameter type (spanned data).
P: Some features are not supported yet:
L: Croqtile function declaration without definition. This allows calling Croqtile function from another module. | Croqtile function with `inline`. This allows the Croqtile function be used in headers.

=== UID:F26-S01
H: *Factor* Target Specific Details
P: In some target like *factor*, it requires the device function to follow C-linkage, where the device functions must be prefixed with `extern "C"`. Since template functions can not be applies to functions with C-linkage, Croqtile has some wrapper tricks to enable it. For example:
C: A cok block around device code
P: Programmers can turn `bar` into a function template, like following:
C: The block after factoring target-specific details
P: As the comment depicts, Croqtile compiler may generate a specialization version of function template `bar`3>, which enables the function template as well. But note that, the function template no longer requires `extern "C" decoration` as those for normal *factor* device functions.
P: As the implementation is different with other targets, you may utilize `-kt` to enable such functionality for *croqtile-factor* compilation.

=== UID:F10-S04
H: `__device__` functions: target alongside Croqtile
P: A custom reduction, a sorting network, a special math function --- Croqtile does not need to manage these;
P: A `__co__` function can call a `__device__` function using the `call` keyword:
C: device functions: target alongside Croqtile
P: This pattern --- Croqtile handles the parallelism, tiling, and memory orchestration; --- Is common in production kernels like MoE (mixture-of-experts) top-K, custom reductions, and specialized math.

=== UID:F10-S06
H: storage hints: `setmaxnreg`
C: storage hints: setmaxnreg

=== UID:F10-S13
H: In short
T: | **Syntax** | **What it does** | **Runs on** | <ROW> | --- | --- | --- | <ROW> | `#`define NAME value | Object-like macro; shared across `__co__` and host regions | Preprocessor | <ROW> | `#if` / `#ifdef` / `#error` | Conditional compilation and compile-time assertions | Preprocessor | <ROW> | `-DNAME=value` | Define macros from the command line for tile sweeps | Preprocessor |
P: 3.11 turns to the other side of the workflow: what to do when the kernel compiles but the output is wrong --- debugging, verbose modes and systematic narrowing. The two worked programs in 3.10 put the interoperability mechanisms into context.

=== UID:F09-S04
H: Why the restricted interface works: expressiveness vs performance
P: That flexibility has a cost.
P: Croqtile takes the opposite approach: it restricts the expressible patterns to those that are **guaranteed to be in the performance sweet spot**. There is no way to accidentally write a strided, uncoalesced global load or a layout conflict shared-memory layout --- the syntax simply does not allow it.
F: f09-fig8_expressiveness_light.png
P: Croqtile's `dma.copy` follows the same principle --- The trade is explicit: you give up the ability to write arbitrary data movement, and in return, every data movement you write is fast.

=== UID:F13-S02
H: Easy to Use
P: You work on tensors and tiles, not raw buffers and pointers.

=== UID:F13-S03
H: Compile-Time Safety
P: In development builds, 1,319 runtime assertions guard every transfer and memory access. Together they eliminate entire classes of bugs that would otherwise need `memcheck` and hours of printf debugging.

=== UID:F13-S04
H: Dynamic Shapes
P: First-class symbolic dimensions let you write a kernel once and run it on any shape without recompilation. The compiler derives packed-K, metadata columns, grid dimensions, and shared-memory size automatically. Static and runtime memory are unified --- no template metaprogramming, no boilerplate.

=== UID:F13-S05
H: Born for AI Tuning
P: Compact, structured syntax keeps entire kernels inside AI context windows (30--60 lines vs. Structured error messages and well-documented CLI arguments let autonomous agents compile, profile, and iterate without human intervention.

=== UID:F40-S01
H: Tiling Optimisation in Croqtile
P: Two complete programs are enough to see why the tiling constructs are shaped the way they are.

=== UID:F40-S02
H: Example 1: `matmul`
P: The first example is matrix multiplication on a device program. The code shows how Croqtile's tiling and memory management mechanisms work together to optimize matrix operations, including how DMA (Direct Memory Access) is leveraged for efficient data movement.

=== UID:F40-S03
H: Example Code: Matrix Multiplication with Tiling
C: Example Code: Matrix Multiplication with Tiling

=== UID:F40-S04
H: Memory Tiling with `chunkat`
P: In the matrix multiplication example, **memory tiling** is achieved using the `chunkat` method, which slices the input matrices `lhs` (left-hand side) and `rhs` (right-hand side) into smaller chunks to be processed in parallel.
L: The `lhs` and `rhs` matrices are tiled using the `chunkat` method. The chunks are defined by `m_tile`, `k_tile`, and `n_tile`, which represent different dimensions of the matrices involved in the multiplication. | The chunks for both matrices are loaded asynchronously into the (`lhs_load` and `rhs_load`) with the `dma.copy.async` operation. This operation loads a portion of the matrix from a source buffer into a staging buffer, which is a common optimization in large matrix operations to improve data locality.
L: The variables `m_tile`, `k_tile`, and `n_tile` are **bounded variables**. They are defined in the context of `parallel-by` and `with-in` constructs, which define the iteration space for the tiling. | These bounded variables allow the chunks to be computed in parallel, where each thread works on a different slice of the matrix. This parallelization enables efficient computation of matrix multiplications across multiple threads.

=== UID:F40-S05
H: Example 2: rgb2gray
P: The second example is the RGB to Grayscale conversion algorithm. It converts a color (RGB) image to a grayscale image using a weighted average method. It shows how physical parallelism works together with virtual parallelism.

=== UID:F40-S06
H: Example Code
C: Example Code

=== UID:F11-S01
H: Debug and Verbose: Printing, Assertions, and GDB
P: The universal debugging discipline is the same everywhere: **systematically narrow the search space**, from cheap checks to expensive ones. Start with static analysis and compile-time assertions (free, catches whole classes of bugs). Move to targeted runtime probes (cheap, localizes the problem). Resort to interactive debuggers only when the cheaper tools have narrowed the suspect list.
P: Croqtile provides tools at each level: **compile-time shape printing** (`print!`, `println!`) to verify tile dimensions without launching a kernel, **runtime assertions** (`assert`) for invariant checks, **guarded device `println`** for runtime inspection of specific threads, **debug RTTI** for `debugger`, and a **verbose mode** (`-v`) to see what the compiler is actually doing under the hood.

