=== UID:F14-S01
P: A `.co` file brings the application and its parallel work into one source file. Ordinary host C++ prepares inputs and calls a `__co__` function. That function describes the parallel work and can invoke device helpers for specialized computations. Understanding which compiler handles each part makes both integration and diagnostics easier.

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
P: The target determines whether host and device sources are compiled together or separately:
L: **Single Source Compilation Model**: The target compiler allows device and host programs to be in a single source file for target compilation. | **Separate Source Compilation Model**: Host and device code must be compiled separately.
P: The code shown above naturally supports the *Single Source Compilation Model*. However, to support the *Separate Source Compilation Model*, Croqtile requires wrapping the Device Program with the `__cok__` block, as shown below:
C: Transpilation and Compilation
P: The `__cok__` wrapper marks device code that Croqtile must separate from host code before invoking the target compilers. It is needed when the target uses separate host and device compilation. The source can still be maintained as one `.co` file; the wrapper tells the build process where to split it.

=== UID:F14-S07
H: Device Program: Parallel Computation
P: A device helper receives the data selected and staged by its caller. Its job is to compute on that data using the target language. The following addition helper illustrates this boundary: Croqtile arranges the chunks and transfers, while the helper operates on the pointers and element count passed to it.
P: The device helper from the earlier example is repeated below:
C: A device helper called from tileflow
P: For targets that only support the _Separated Source Programming Model_, the code may need to be wrapped within a `_`_cok__ block. The equivalent code is shown below:
C: The cok wrapper for separate sources
P: In this separate-source example, `extern "C"` replaces `__device__` because the target requires C linkage for device functions.
P: The helper runs within the parallel execution arranged by its caller. In this example, the Tileflow function performs the transfers and the helper computes on the selected chunk. Keeping those responsibilities separate lets the transfer structure change without rewriting the helper's arithmetic.

=== UID:F31-S08
H: Heterogeneity: Transpiling to program launch
P: For targets that separate host and device execution, `parallel-by` specifies both the parallel instance count and the boundary between host and device code:
L: Code **outside** the `parallel-by` block is transpiled into **host code**. | Code **inside** the block is transpiled into **device code**, as illustrated below:
C: Heterogeneity: Transpiling to program launch
P: The `parallel-by` statement therefore also marks a device-program launch. Croqtile represents this boundary through the same SPMD parallelism model used to describe the work.

=== UID:F12-S02
P: This reference covers every aspect of programming with Croqtile:
L: **Program structure**: How Croqtile programs are organized --- the `__co__` function, host-side APIs, and C++ interoperability. | **Shaped data**: Croqtile's type system for multi-dimensional tensors --- shapes, MDSpan, spanned data, symbolic dimensions, and dynamic shapes. | **Loop and parallelism**: SPMD parallelism, loop control, tiling, and iteration constructs. | **Data movement**: DMA statements, tiling with `chunkat`, TMA, swizzle, and advanced data movement patterns. | **C++ embeddings**: Input/output conventions, `__cpp__` inline blocks, macros, and preprocessor integration. | **Optimization patterns**: Tileflow optimizations, async DMA patterns, multi-buffering, and performance tuning techniques.

=== UID:F14-S09
H: In short
P: A Croqtile program consists of host code, Tileflow code, and device code. Compilation translates the Tileflow functions into target source. The call chain typically runs from the host to a Tileflow function, then to a device function.
P: Croqtile combines higher-level DMA operations with iteration and tiling, so a compact program can express the full data flow. The following sections explain the syntax and semantics of these constructs.

=== UID:F05-S11
H: Type System Aside: `mdspan` and `ituple`
P: Consider the information needed to select one tile: the tensor's shape, the number of pieces along each axis, and the coordinates of the chosen piece. Croqtile represents these explicitly with shapes, tuples, and bounded indices. They let a tile expression retain the relationships that would otherwise be spread across offset calculations.
P: A shaped declaration such as `s32 [128, 256] lhs` fixes both the element type and the rank. Here the extents are constants, so the compiler can evaluate them before execution. It can reject a three-coordinate access to this two-dimensional tensor and check compatible tile shapes. A declaration with symbolic dimensions keeps the same rank information even when some extents come from the caller.
P: The `span(i)` syntax extracts one dimension: `lhs.span(0)` is 128, `rhs.span(1)` is 256. The shorthand `lhs.span` copies the entire shape.
P: An `ituple` groups integer values. In `parallel px, py by [8, 16]`, the coordinates are also bounded: their current values identify one instance, while their bounds describe the grid. The compiler can simplify the bounds and derived expressions without knowing one fixed current index for all instances. Index calculations that depend on execution still take place at runtime.
P: Sections 2.4 and 2.5 introduce the declarations. Here the useful distinction is between information known from the program and information supplied by a particular call. Compile-time checks use the former; entry checks and runtime assertions cover constraints that require the latter.

=== UID:F15-S01

=== UID:F15-S02
P: Croqtile aims to simplify the everyday work of developing high-performance kernels, making it easier to express and optimize their computation and data movement.

=== UID:F15-S03
H: Embedding Croqtile within C++
P: Croqtile is a DSL embedded in C++. Its compiler translates Croqtile functions into C++ and implicitly includes `choreo.h` so the generated code can interoperate with ordinary C++. For example:
C: Embedding Croqtile within C++
P: The `__co__` prefix identifies the function body Croqtile translates. Ordinary C++ calls that function by its declared name. The host-facing interface uses shaped buffer wrappers so the call carries dimensions as well as an address.
P: In addition, certain Croqtile-specific types are introduced to ensure consistency between Croqtile function arguments and their respective callers. Considering that there are some important programming elements missing, you will defer the introduction of such details until the data types are explained.

=== UID:F15-S04
H: Variables and Types
P: Croqtile introduces four fundamental type categories: scalar-type, spanned-type, integer-tuple-type (ituple-type) and bounded-type. These types serve for different purposes.
L: **Scalar Type**. It is designed to fulfill the requirement of program control. It consists of *Integer Type* and *Boolean Type*. | **Spanned Type**. It represents the data type (normally the tensor) for computation. Apart from referencing the raw data, a *spanned type* also associates data with a shape represented by multi-dimensional ranges. The 'multi-dimensional ranges' is named 'mdspan'. It is useful for tiling, etc, which is introduced soon. | **Integer Tuple (I-Tuple) Type**. It represents a group of integer values. A common usage of *i-tuple* is to introduce the bound of multi-dimensional data. | **Bounded (Integer/ITuple) Type**. This is the special type that is used to simplify data (sub-zone) reference. And it works with loop construct to iteratively process data.
P: Scalar and `ituple` values resemble familiar constructs in general-purpose languages. Spanned and bounded types are specific to Croqtile's domain; the following sections explain their roles.

=== UID:F15-S06
H: Spanned Types
P: A spanned type combines a **fundamental element type** with a **multidimensional span (`mdspan`)**. Neither component alone is a complete type for data with storage; Croqtile calls them **partial types**.
P: Separating these components lets a program manipulate shapes independently of element types. This is useful for multidimensional machine learning data: tiling divides it into smaller blocks, and data movement transfers those blocks between storage levels. Shape calculations describe these operations.
P: The following examples define these partial types and combine them into complete types for multidimensional data declarations.

=== UID:F15-S07
H: The Partial Type: **mdspan**
P: An `mdspan` is a partial type: it describes a shape but cannot by itself declare a data buffer. Shape values can nevertheless be defined and manipulated independently. Tiling depends on multidimensional shape rather than element type, so this separation simplifies shape calculations.
P: The compiler also checks `mdspan` expressions to detect errors as early as possible, before execution when the necessary information is available.
P: In Croqtile, a mdspan is braced by '[' and ']'. The upper-bound of dimensions inside are separated by ','. That is,
C: An mdspan declaration
P: An `mdspan` can be declared explicitly, with an optional rank in angle brackets, or inferred from its initializer, as in the declaration of `d`. The types of `mds` and `d` depend on the values of `a` and `b`. If these values are supplied at runtime, some checks must also occur at runtime.
P: For a C++ programmer, an `mdspan` describes the ranges of a multidimensional array. In the example, `sp` has two dimensions, with indices from 0 to 6 and from 0 to 7, respectively.
P: When a mdspan is defined, it is possible to get the integer dimension value using operator '()' over the mdspan.
C: A shape literal
P: The expression `sp(0)` returns the first extent of `sp`. Extents are integer values and can participate in integer arithmetic.
P: You can therefore derive a new `mdspan` from an existing one:
C: A shape with one dimension changed
P: This makes tiling multidimensional shapes convenient. Because tiling is common in high-performance kernels, Croqtile provides a shorter form for the same calculation:
C: The same shape again
P: This version computes the same shape with less code, following Croqtile's aim of expressing common operations concisely.
P: A named `mdspan` must be initialized when declared and cannot be modified afterwards. To describe a different shape, derive a new name from the existing one. This preserves the relationship between shape expressions for the compiler's analysis.
P: An `mdspan` can also be defined using operations on whole tuples. The `ituple` discussion explains those forms.

=== UID:F15-S12
H: Bounded Types
P: A bounded integer carries a value and an exclusive upper bound. In `parallel p by 6`, its possible values are 0 through 5 and its upper bound is 6. A tile selection can therefore determine both which piece `p` identifies and how many pieces exist. The `parallel-by` and `with-in` constructs establish this association.
P: A bounded tuple extends the same idea to several dimensions. Each component has its own bound, and an `mdspan` describes the combined range. For example, a tuple bounded by `[4, 8]` can identify one of 32 positions. Section 2.7 shows how to declare and iterate over these indices.

=== UID:F15-S17

=== UID:F15-S22
H: In short
P: Croqtile expresses SPMD programs in a C++-style DSL embedded in C++. It reduces the low-level work of moving data between storage levels, particularly through DMA, and is therefore also described as a dataflow DSL. Its purpose is to support everyday kernel development by letting programmers focus on the organization of computation rather than repetitive implementation details.
P: Feedback from using these constructs helps guide further improvements to the language.

=== UID:F17-S03
H: Group of Ranges and Group of Integers
P: In Croqtile, *mdspan* can be used to represent dimensional shapes. However, as introduced, *mdspan* stands for **Multi-Dimensional Span**. The elements inside are *Dimensional Spans*, which imply ranges of values.
P: For example, an *mdspan* `[7, 14]` represents a group of two ranges `[0, 7), [0, 14)`. Therefore, an *mdspan* does not represent a list of integer values. However, grouping integer values is useful in certain scenarios:
L: When considering the delta of two mdspans, multiple integers are needed. In high-level semantics, the delta may reflect padding differences when adjusting shape dimensions. | When tiling a shape, the multiple tiling factors are integers, not ranges.
P: These situations motivate Croqtile to have a type for grouping integer values.

=== UID:F17-S08
H: Look-Ahead: Bounded I-Tuples
P: A bounded tuple attaches ranges to an ordinary integer tuple. For example, a position `1, 2` together with bounds `[4, 8]` identifies one position and also describes the space containing it. Loops use the bounds for iteration, and `chunkat` uses them to derive a partition.

=== UID:F18-S02
H: *Spanned*: Data and Buffers
P: A function's inputs and outputs are often called data, while its temporary storage is called a buffer. Croqtile represents both with spanned types: stored elements associated with a shape. The terminology describes how the storage is used, not a different type system.
P: The term spanned data is used for either case. An `mdspan` describes the shape; a spanned object combines that description with an element type and access to storage. Naming or deriving a shape alone does not create a copy of the data.

=== UID:F18-S07
H: Buffer Lifetime Management
P: The Croqtile compiler manages the lifetime of storage allocated inside a function. It attempts to reuse buffers whose lifetimes do not overlap, so programmers do not need to manage those allocations themselves.
P: Automatic allocation does not replace synchronization. If an asynchronous transfer or computation still uses a buffer, that use must finish before the storage is overwritten or released. In a pipeline, the waits and completion operations make these lifetimes explicit enough for the program to reuse the same slots safely.

=== UID:F31-S06
H: Bounded Variable
P: In a `parallel-by` block, the variables defined after the `parallel` keyword are referred to as **Bounded Variables**. Specifically:
L: If the variable is an *integer*, it is called a **bounded integer**. | If the variable is an *ituple*, it is called a **bounded ituple**.
P: This terminology indicates that the variable not only has a value but also has defined bounds. For example, in the statement `parallel p by 6`, `p` is a *bounded integer* with a bound `[0, 6)`, where `6` is the exclusive upper bound.
P: The concept of bounded variables will be explored further in later sections, where their usage will be detailed.

=== UID:F32-S09
H: Values of Bounded Variables
P: A bounded variable carries a range and a current value. For an index declared by `with-in`, the following rules describe how `foreach` supplies that current value:
L: **Current Value**: itemize[leftmargin=1.6em] | Within a `foreach` statement, the current value is determined by the loop iteration, as the bounded variable serves as the iteration variable. | Outside of a `foreach` statement, the bounded variable always has a value of zero.
L: **Upper-Bound Value**: This is specified in the `with-in` or `parallel-by` statements.
P: After a `foreach` loop, a `with-in` index has its outside-the-loop value again; it is not a C++ loop variable left at the terminating value. A parallel index follows a different rule: its value identifies the current parallel instance. The example below concerns the sequential `with-in` case.
C: Values of Bounded Variables

=== UID:F37-S01
H: Memory in Croqtile
P: Storage becomes part of the algorithm when different parallel instances need different views of a tile. The same `[M, N]` shape may describe a global input, a shared staging tile, or a private working buffer. The shape controls indexing; the storage specifier determines where the buffer belongs.

=== UID:F37-S02
H: Spanned Types in Croqtile
P: A spanned type combines an element type, such as `f32`, with a shape. The same shape can be reused with another element type, and a derived shape can describe a tile without changing what each element stores.
P: The spanned type by itself does not allocate storage for the data. Instead, it serves as a blueprint for defining how the data will be organized in memory, allowing for runtime flexibility and optimizations.

=== UID:F37-S03
H: Storage Qualifiers in Croqtile
P: The storage specifiers `global`, `shared`, and `local` distinguish device-wide storage, storage shared within a block, and thread-private storage on the targets used here. Section 2.6 gives declaration examples. Choose a location from the intended readers and writers of the buffer, then arrange transfers between locations as needed.

=== UID:F16-S02
H: First-Class Citizen: Shape
P: A raw pointer does not tell a caller whether its elements form a vector, a matrix, or a tile cut from a larger tensor. That missing information has to be supplied elsewhere whenever code computes an offset or checks a transfer. Croqtile keeps shape information alongside the data so declarations, tile selections, and movement statements can refer to the same dimensions.
P: In contrast, Croqtile enforces code safety and simplifies the programming of shaped data by requiring that **any data declared or used must be associated with a shape**. This motivates Croqtile to treat shape as a first-class citizen.

=== UID:F16-S05
H: Evaluation of *mdspan*
P: An mdspan with constant values is sufficient for many scenarios, as high-performance device kernels often require fine-tuning based on fixed input data shapes. In these cases, the `mdspan`s are evaluated at compile-time, meaning their values do not incur extra execution time or storage overhead.
P: When a dimension comes from an input, generated host code evaluates the required shape expressions on entry to the Croqtile function. Constant parts can still be simplified beforehand. The amount of runtime work depends on the remaining expressions and checks.

=== UID:F17-S07
H: Evaluation of Integers and I-Tuples
P: Integer and tuple expressions are evaluated at compile time when their operands are known. Expressions depending on runtime indices or input sizes remain computations in the generated program. Immutability helps the compiler reuse and simplify them; it does not make every value a constant.

=== UID:F19-S01
P: A kernel often needs to preserve its structure while accepting different input sizes. Anonymous dimensions leave a size unspecified; symbolic dimensions also give that size a name. The name matters when two inputs must agree or when an output dimension is derived from an input.

=== UID:F19-S02
H: Fixed and Dynamic Shape
P: Data with fixed shapes is commonly used in high-performance computing kernels, allowing for aggressive optimization with known dimensions. However, certain scenarios require handling shapes with runtime dimensions, known as dynamic shapes. This necessitates kernel generality to manage inputs of varying shapes, a feature highlighted by many machine learning frameworks.

=== UID:F19-S05
H: Improved Code Safety
P: The reason for improved code safety of using *symbolic dimension* is that more comprehensive code checks can be applied by Croqtile compiler. For example, consider an *anonymous dimension* version of the `Matmul` function:
C: A kernel with anonymous dimensions
P: Croqtile performs both compile-time and runtime checks to ensure safety:
L: **Compile-time Checks**: Croqtile verifies rank consistency. In this example, `lhs.span` has a rank of `2`, matching the rank of `tile_`m, tile_k, so there are no issues. | **Runtime Checks**: Croqtile generates code to validate dimensions at runtime. For instance, in the declaration of `tiled_lhs`, its shape must not have a dimension of `0`, which would result in an invalid zero-sized buffer. Croqtile's *transpilation* process generates the following *target host code*:
C: The transpiled signature
P: Here, the `croqtile_`abort function aborts program execution if the condition is not met. This early check helps Croqtile detect issues promptly.
P: For the *symbolic dimension* version, Croqtile compiler generates additional checks in the *target host code*:
C: The generated shape check
P: In addition to the checks performed in the *anonymous dimension* version, the *symbolic dimension* version verifies the consistency of dimensions between `lhs` and `rhs`, resulting in safer code.
P: Therefore, it is recommended that programmers use symbolic dimensions not only for ease of use but also to enhance code safety.

=== UID:F19-S06
H: Cost and Limitations
P: Dynamic dimensions leave less information available during compilation. Some constraints must therefore be checked at runtime. Croqtile places shape checks in generated host code where possible; application-specific requirements may still need assertions.
P: Anonymous and symbolic dimension syntax is restricted to spanned-data parameters. An internal buffer can derive its shape from an input, but it cannot introduce a new unspecified dimension with `?` or a new symbolic name. The following declaration is rejected:
C: Cost and Limitations

=== UID:F19-S07
H: In short
P: Anonymous dimensions express an unknown size. Symbolic dimensions express an unknown size together with a relationship the program can name. Reusing `K` in two operands records that their contraction dimensions must agree. Croqtile checks the parts it can establish statically and generates checks for relationships that depend on actual input dimensions.

=== UID:F20-S01

=== UID:F20-S02
H: **a. Improved Readability and Clarity**
P: Using symbolic dimensions like `M` and `N` explicitly in the shape definition makes the program **easier to read** and understand. Instead of seeing a `?` or `-1` and wondering what the dimension is supposed to represent, the reader can immediately grasp that `M` and `N` are variables that depend on runtime values. This improves **maintainability** and reduces the cognitive load for developers who are reading and modifying the code.
P: For example, two question marks identify two unknown extents; they do not state that those extents are equal. A repeated symbolic name records that equality directly in the function signature:
C: a. Improved Readability and Clarity
P: The anonymous form remains useful when the dimensions are independent. Choose names when the relationship between them is part of the operation.

=== UID:F20-S03
H: **b. Type Safety and Compile-Time Checking**
P: Symbolic dimensions let the compiler reason about expressions without knowing every numerical extent. It can check ranks and simplify relationships such as repeated uses of `M / 2`. This does not eliminate runtime validation: whether caller-supplied dimensions agree or satisfy a required constraint may only be known when the function is entered.
P: The expression below keeps the derived shape tied to the named input dimensions. That connection is available to the compiler and visible to the reader, instead of being maintained through separate size arguments:
C: b. Type Safety and Compile-Time Checking
P: This ensures type consistency and reduces the potential for errors during runtime.

=== UID:F20-S04
H: **c. Expressiveness and Flexibility**
P: A named dimension can appear in several shape expressions. For example, a buffer may use half of M along one axis and a padded N along another. These expressions show how the new shape depends on the input dimensions.
P: An anonymous dimension can still be read through the input's shape. A symbolic name is useful when that same value occurs in several expressions: changing one derivation then leaves its relationship to the other dimensions visible.
P: For example, you can easily express relationships between dimensions using symbolic variables:
C: c. Expressiveness and Flexibility
P: The benefit is that the dependency is written where the shape is declared. The program still has to satisfy the resulting size and divisibility requirements for each call.

=== UID:F20-S05
H: **d. Compile-Time Optimization**
P: Symbolic expressions give the compiler opportunities to reuse a dimension calculation and simplify expressions that share known relationships. Constant parts can be evaluated during compilation; parts depending on input sizes remain runtime calculations. A symbolic name preserves information for optimization, but it does not turn an unknown input size into a constant or guarantee a particular performance result.

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
P: A number after "VN #" identifies an expression's value. It may represent a constant integer, a symbolic value, or a combination of other value numbers. Here, #0 and #3 refer to constants; #1 and #2 refer to the integer variables `b` and `c`. Value #4 represents the sum of #2 and #3, which expands to `c + 3`. Value #5 represents the list of values in `mdspan a`. Indexing `a(0)` selects #0, the constant 1.

=== UID:F21-S03

=== UID:F21-S04
H: From Expressions to Value Numbers
P: The current value-numbering implementation follows the syntax tree. It traverses the abstract syntax tree and assigns a value number whenever a node contains an expression.
P: The method of generating value numbers is simple. For each expression, it takes the expression as the input, and returns corresponding value number.
P: More specifically, it:
L: Generates a signature from an expression. | If the signature already exists, return the associated value number by looking up the value number table. | Or else, it associates the signature with a new value number in the value number table, and return it.
P: For simplicity, the current implementation represents an expression signature as a string. A hash-based representation could replace it later, as in many value-numbering implementations.
P: In addition, to handle the symbol definition and symbol reference, it follows the following rules:
L: When an expression is to initialize a symbol (variable or partial type), the symbol is assigned the same value number of the expression. More specifically, it associates the signature of the *scoped* symbol name with the value number in the value number table. | When an expression references a symbol, it looks up its corresponding signature in the value number table, and return it.
P: Note the symbol must use its *scoped name* for signature generation, since identical names are allowed to be defined in different scopes.

=== UID:F21-S05
H: Simplifications
P: Every time the value numbering process generates a signature, it tries to simplify the expression. For example,
C: A constant expression
P: Given the simple integer expression, the value numbering process could generate signatures and associated value number as illustrated:
C: The simplified graph
P: Here, without optimizations, two constant values #0 and #1, and a new value number #2, which is an addition of #0 and #1 are generated. #2 has the signature of "+:#0:#1". However, considering that addition of constants can be evaluated at compile time, simplification could be performed. The following output showcases the optimized value numbering process:
C: The folded graph
P: The signature of #2 now represents a constant, which implies the algebraic simplification is applied successfully.

=== UID:F21-S06
H: From Value Numbers to Expression
P: After value numbering and simplification, each expression refers to its simplified value number. Croqtile uses these results to infer `mdspan` shapes.
P: An `mdspan` describes the number of elements in each dimension. Together with a fundamental element type, it determines the storage size of a spanned object.
P: Some dimensions are determined at runtime; others can be fully evaluated at compile time. Known dimensions allow storage requirements to be determined in advance. Otherwise, the compiler generates code to calculate the shape at runtime.
P: For nonconstant values, the compiler must reconstruct expressions from the value numbers.

=== UID:F21-S07

=== UID:F21-S08
H: Lexical Scopes and Restriction
P: Croqtile applies value numbering globally across the abstract syntax tree. This syntax-directed approach does not construct basic blocks or perform data-flow analysis, so it cannot infer shapes whose dimensions change through reassignment.
P: For a reader of generated code, the consequence is straightforward: a constant shape may disappear into constants, while a symbolic shape becomes a calculation in the generated program. Value numbering tries to preserve and simplify those relationships. It does not prove every mathematical identity, so an expression that seems equivalent by hand may still need a clearer formulation.

=== UID:F21-S09
H: Limitations
P: More specifically, it has to set two limitations:
L: Any mdspan cannot have element with bounded integer type, or produced from bounded ituple. | Simple integer type and ituple type cannot be re-defined.

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
F: The same eight tasks take eight, two, or one time steps when one, four, or eight tasks can run at once.
P: Parallelism describes which tasks can be independent. The execution resources determine how many of those tasks can make progress at once.
P: When tile counts and launch dimensions are written as the same set of constants, changing the tiling also forces a change to the resource mapping. Keeping the two descriptions separate makes those choices easier to revise.
P: Croqtile uses `parallel` for the independent work and space specifiers for its mapping. The work can therefore be read before deciding whether an axis belongs to blocks, threads, or cooperating groups.

=== UID:F05-S08
H: `: group-4` --- cooperative group
C: : group-4 --- cooperative group
P: 3.7 puts this to work with tensor-core operations.
P: Here `group-4` names a cooperation scope used by the matrix operations below. Each parallel index identifies a cooperating group, so an operation in the body is understood in that group's scope. It is not a request for every participating thread to compute and store an independent copy of the same matrix tile.

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
P: A sequential loop is needed when later work depends on earlier work, such as accumulating successive K slices. Croqtile separates the bounded range declared by `with-in` from the traversal performed by `foreach`. This also permits work before and after the traversal, as used in buffered pipelines.
P: The following sections explain how Croqtile constructs loops and expresses parallel work.

=== UID:F31-S01
P: A `parallel-by` block declares a set of independent instances; the space specifier says how those instances are grouped.

=== UID:F31-S02
H: Parallel Execution Block
P: A **parallel execution block** contains code executed by multiple instances. Each instance runs the same code with its own parallel index.

=== UID:F31-S07

=== UID:F31-S11
H: In short
P: A `parallel-by` block declares independent instances and a space specifier groups them; masking then decides which of those instances actually execute a statement.
P: Loops remain necessary for work that has to happen in order, and 2.7 introduces their Croqtile forms.

=== UID:F33-S01

=== UID:F33-S02
H: Parallelization and Masking
P: SPMD programs often launch a fixed number of parallel instances even when the work does not divide evenly among them. Masking limits execution to the instances with valid work and leaves the others inactive.
P: Croqtile supports both explicit masks and masking implied by the surrounding code structure.

=== UID:F33-S03
H: Explicit Masking/Condition
P: A thread mask describes which SPMD instances execute a region. Within the code for a single instance, it appears as a Boolean condition, usually a comparison involving a thread identifier.
P: In the examples that follow, Croqtile experimentally supports explicit masking of the `foreach` block, which masks off loop code as a whole. For example:
C: Explicit Masking/Condition

=== UID:F33-S04
H: Implicit Masking
P: Some code structures in Croqtile imply implicit masking. One typical situation is the multi-level `parallel-by` block. For example:
C: Implicit Masking
P: This example has two parallel levels. Code inside `parallel p by 6` but outside `parallel q by 2` belongs only to the outer level. If the target launches two inner instances for that region, the outer-only code behaves as though guarded by `if (q == 0)`.
P: A condition can also guard a `foreach` loop, controlling execution of the loop as a whole.

=== UID:F33-S05
H: Experimental: Optional *Where-Binding* for `within`
P: You can append a *where-clause* to impose **Where-Binding** constraints among different bounded variables defined in a `with-in` statement. This allows different bounded variables to be treated as aliases for each other. Below is an example:
C: Experimental: Optional Where-Binding for within
P: Syntactically, the ``-> operation establishes the *where-binding* between two bounded variables. Within the `with-in` block, any reference to `n_p` refers to `n`, and vice versa.

=== UID:F34-S01
P: Croqtile primarily uses SPMD (Single Program, Multiple Data), but thread masking also allows different instances to run different code, expressing MPMD (Multiple Program, Multiple Data) behavior. The `inthreads` block selects these divergent regions.

=== UID:F34-S02

=== UID:F34-S03
H: Masking in SPMD
P: It is typically necessary to mask off some *parallel threads* in SPMD programming. For example, programmers often divide data into chunks for parallel or iterative processing. However, there may be a remainder that cannot be evenly distributed across all threads.
F: f34-masking-datachunk-dispatch.png
P: The figure distributes 13 data chunks across four parallel instances in four iterations. All instances are active for the first three iterations, giving mask `1111`. Only instance 0 has work in the final iteration, giving mask `0001` when the least significant bit represents instance 0. A `1` denotes an active instance and a `0` an inactive one, preventing unnecessary processing of nonexistent chunks.

=== UID:F34-S04
H: Uniform and Divergence
P: From the SPMD programmer's perspective, code **diverges** when masking causes different instances to follow different paths. An SPMD program can therefore contain:
L: **Uniform Code**: *All threads* execute the same code path. | **Divergent Code**: Only the *masked threads* execute the *divergent* code path.
P: It is necessary to have the *divergent* code in SPMD code to handle different real-world requirements. And from a perspective of different programming paradigm, applying *divergent* code in SPMD programs can even mimic MPMD (Multiple Program Multiple Data) behavior. In MPMD, different threads execute different code. Therefore, *divergence* can extend SPMD to MPMD scope conceptually.
P: Use an `inthreads` block to select the instances that execute a divergent region.

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
P: Masking identifies who executes a region. Whether the other instances wait at its end depends on the synchronization form. Section 3.8 develops asynchronous regions, where the program must express data dependencies with events rather than rely on a join after each region.

=== UID:F05-S13
H: Scalar Matmul (No DMA)
P: The first version uses `parallel` to divide the output and `.at()` to read individual elements. It does not yet use explicit data movement.
C: Scalar Matmul (No DMA)
F: The [128, 256] output has a [16, 64] tile grid. Each tile has shape [8, 4]; only the first few tiles are drawn.
P: The tile dimensions follow from the parallel bounds:
P: The output declaration derives 128 rows from `lhs.span(0)` and 256 columns from `rhs.span(1)`. The contraction dimension appears in both inputs but does not become an output dimension.
P: The declaration `parallel p, q by [16, 64]` gives one parallel level two coordinate axes. Its 1,024 index pairs partition the output into tiles of eight rows and four columns.
P: The loop binds `m`, `n`, and `k` as components of `index`. The first two visit positions inside the output tile; the third traverses the contraction dimension. Their trip counts follow directly from the output partition:
L: `128 /` #p = 128 / 16 = 8 rows per tile | `256 /` #q = 256 / 64 = 4 columns per tile | `256` for the full contraction dimension
P: The composed row `p#m` has value `p * 8 + m`. As p visits the 16 row tiles and m visits the eight rows within each tile, it covers rows 0 through 127. The column expression `q#n` works the same way with four columns per tile.
P: For each output position, the body sums `lhs.at(p#m, k) * rhs.at(k, q#n)` over K. All contributions go to the same output element. The accumulator must start at zero before this sum; the abbreviated listing leaves that initialization to be supplied.

=== UID:F05-S17
P: Trace output position `(37, 50)` using the block-and-thread decomposition of the DMA matmul in Section 3.6.1. That example uses an 8 by 16 block grid, with each block handling a 16 by 16 output tile. This differs from the 16 by 64 logical grid in the scalar example immediately above; both cover the same output shape.
P: Blocks partition 128 rows into 8 tiles of 16: `px = 37 /` 16 = 2, offset `qx = 37 % 16 = 5`. Columns partition 256 into 16 tiles of 16: `py = 50 /` 16 = 3, offset `qy = 50 % 16 = 2`. So block `(2, 3)`, thread `(5, 2)`.
P: For that thread, the K-loop runs 16 iterations. On iteration `tile_k = 0`, `dma.copy` loads `lhs` rows 32..47 and K columns 0..15 into shared `lhs_load`, and `rhs` K rows 0..15 and columns 48..63 into shared `rhs_load`. The inner `foreach k in [16]` accumulates `lhs_`load.data.at(5, k) * rhs_load.data.at(k, 2) for k = 0..15 into `output.`at(37, 50). Then `tile_k = 1` loads the next K-strip, and so on. After all 16 iterations, `output.`at(37, 50) holds the complete dot product --- same answer as the scalar reference.

=== UID:F04-S04
H: `chunkat`, `subspan`, and `view().from()`: Three Ways to Select a Tile
P: A tile expression answers two questions: how large is the region, and where does it begin? With `chunkat`, bounded indices provide the partition counts and select a piece. With `subspan(...).at(...)`, the tile shape is explicit and the coordinates count tiles. With `view(...).from(...)`, both the shape and the starting element coordinates are explicit. The distinction matters whenever a window starts between regular tile boundaries.

=== UID:F04-S13
H: `#`: Compose and Extent
P: The unary form `#p` retrieves an index's upper bound. The binary form `p # i` combines an outer index with an inner one: its value is `p * #i + i`, and its bound is `#p * #i`. For eight tiles containing 16 elements each, tile 2 and inner element 5 identify element 37 in a range of 128 elements.

=== UID:F09-S06
H: Handling irregular access
P: Regular tiling works well when the dimensions divide evenly and the origin follows a fixed grid. Other kernels need a different description: expert batches can start at runtime offsets, convolution windows can overlap, and a final tile can contain fewer valid elements. A flat input may also need a multidimensional view before it can be consumed.
P: The following operations adjust the selected region or its shape while leaving the surrounding algorithm recognizable. The resulting view still has to satisfy the chosen transfer operation's requirements, including shape, layout, and any target-specific alignment constraints.

=== UID:F29-S05
H: Tile and Flow
P: It is possible to use the `chunkat` expression for either the source or the destination of a *DMA statement*. When `chunkat` is used only for the source, it typically divides the input or buffer into smaller chunks to move it to storage closer to the processor. In contrast, when `chunkat` is used only for the destination, it typically moves the buffer to compose part of the output. For example:
C: Tile and Flow
P: In this example, the statement applies `chunkat` to the destination, moving the data block in `shared` and "un-tiling" each block to the output. Note that the Croqtile compiler **requires the source data and destination data to have the same shape and element type** if not using *storage* as the destination *data expression*.
P: Using the `chunkat` expression for both the source and destination is rarely supported by any target platform at this time.

=== UID:F40-S07
H: Compose Two Bounded Variables for Tiling with `#`
P: The RGB-to-grayscale example in Section 3.10.2 combines a worker index q with a sequential image index n. The expression `q # n` assigns consecutive images to each worker. For N = 18 and six workers, the inner bound is 3, so worker 1 processes images 3, 4, and 5. The details below use that same partition.
P: With six workers, q has bound 6 and n has bound N / 6. Their composition has bound N and value `q * (N / 6) + n`. The remaining indices h and w choose spatial tiles; the channel dimension stays whole. Thus the worker assignment and the local traversal jointly identify the source tile and its output position.
P: Changing the operand order changes the assignment. For N = 18, `n # q` would give worker 1 the images 1, 7, and 13 as n advances, instead of the consecutive images 3, 4, and 5 selected by `q # n`. The operator makes this scheduling choice explicit in the data expression; its performance depends on how the resulting accesses behave on the target.

=== UID:F09-S07
P: A mixture-of-experts kernel often packs different numbers of tokens for different experts. Each expert's data begins at an offset known only at runtime, so a fixed tile grid may not describe the required slice. A window gives that slice its own origin before the inner tiling begins.
P: The expression `view(M, N).from(row, col)` describes an M by N rectangle beginning at the supplied element coordinates. The selection need not begin on a tile-grid boundary. A transfer using that view must still satisfy its own alignment and shape requirements:
C: A window at a constant offset
P: The expression selects a `[16, 16]` slice starting at row 37, column 50. The origin can be any runtime value.
F: The small example uses a [4, 4] window from (2, 3); the same rule selects the [16, 16] window from (37, 50) in the code.
P: The power of `view` / `from` is composability: once you carve out a window, all downstream operations --- `subspan`, `chunkat`, `dma.copy`, `tma.copy` --- apply to the window as if it were a standalone tensor. The pipeline stays unchanged; only the origin shifts:
C: A window at a runtime offset
P: In an MoE stack, each expert's token batch starts at a dynamic row `expert_offset`. The `view` / `from` call rewires the operand at the top of the pipeline; everything below --- DMA, staging, MMA loop --- runs the same code for every expert.
P: Use `chunkat` when the partition counts already describe the desired region. Use `view` and `from` when the origin is an explicit element offset, then apply any regular tiling within that window. This keeps the irregular outer selection separate from the repeated inner work.

=== UID:F09-S08
P: A `subspan(M, K).at(i, j)` expression uses tile coordinates. Without an explicit step, its element origin is `(i * M, j * K)`. Specifying a step changes the distance between successive tile origins while leaving the selected tile extent unchanged.
P: Adding `.step(sM, sK)` overrides the stride: tiles are spaced `sM` rows and `sK` columns apart instead of being packed:
C: Strided tiles: .subspan, .step, and .at
F: The small diagram contrasts packed [3, 3] tiles with step(5, 5). In the code, [16, 16] tiles with step(32, 32) leave 16-element gaps.
P: Tile `(0,0)` starts at `(0,0)`, but tile `(1,0)` starts at row 32 (not row 16), leaving a 16-row gap. Omitting `.step` uses a step equal to the tile size --- the packed case.
P: A step smaller than the tile extent creates overlapping windows; a larger step leaves gaps. For example, a 16-row window advanced by eight rows shares half its rows with the next window. The step controls the origin sequence, so it should not be confused with a stride between elements inside one tile.

=== UID:F09-S09
H: Zero-padding: `.zfill`
P: Every tiled kernel faces the same edge case: what happens when M or K is not a multiple of the tile size? The last tile along that axis is partial --- some of its elements lie beyond the tensor boundary.
P: A separate code path for the boundary tile adds branches and complicates a uniform MMA loop. The `.zfill` modifier handles the missing source elements during transfer:
C: Zero-padding: .zfill
P: The `.zfill` modifier supplies zero values where the destination tile extends beyond the valid source region. In a matrix product, those zeros add no contribution to the sum. The consumer can therefore process a full tile with a known value in every position. This establishes the required numerical behavior; the cost of handling the boundary still depends on the target and transfer path.
F: A [4, 4] boundary tile from a [6, 8] source keeps its valid rows and fills the two out-of-range rows with zeros.

=== UID:F09-S10
H: Layout reinterpretation: `span_as`
P: Some kernels load data as a flat 1D strip but need to feed it to MMA as a 2D matrix. A naive approach would copy the data into a reshaped buffer --- `span_as` avoids this by reinterpreting the existing buffer's shape in place:
C: A flat strip reinterpreted as a matrix
P: Element count is preserved; only the logical rank changes. No data is moved.
C: A movement statement using the reinterpretation
P: The product `rows * cols` must equal the number of elements in the underlying region. A reshape preserves linear element order, so it cannot transpose a matrix or remove padding between rows. When the dimensions are constants, the compiler can check the element count before execution; dimensions derived from inputs must satisfy the same invariant for the actual call.
F: The 16-element buffer becomes a [4, 4] view with the same linear order. The 1D strip is displayed on two lines for readability.

=== UID:F09-S11
H: In short
T: | **Concept** | **Syntax** | **Role** | <ROW> | --- | --- | --- | <ROW> | Software DMA | `dma.copy` / `dma.`copy.swiz<N> | Thread-cooperative tile transfer; | <ROW> | Swizzle | `.`swiz<N> on copy + `mma.`load.swiz<N> | Shared-memory remapping; matching modes on copy and load | <ROW> | Arbitrary windows | `view(M,N).`from(r,c) | Ragged or runtime-positioned slices | <ROW> | Strided tiling | `.`subspan().step().at() | Non-packed layouts, overlapping stencils | <ROW> | Partial tiles | `.zfill` | Zero-fill out-of-bounds elements | <ROW> | Shape reinterpretation | `span_`as([dims]) | Zero-copy reshape for staging buffers |

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
P: 1.3 used scalar indexing: pick one position, read inputs, compute, write output.
P: A tiled implementation instead selects a region, stages it in the storage used by the computation, and reuses it there. Croqtile expresses the transfer directly, including transformations that belong to the movement itself:
L: `dma.copy` moves a tile as-is. | `dma.`transp<...> permutes dimensions while moving it. | `dma.`pad<...> enlarges a tile and fills the border.
P: This part of the guide introduces that model, shows the core syntax, and gives the rules you need to write DMA and TMA code that is both correct and likely to lower well.
F: Scalar indexing reads individual elements; explicit tile movement stages a selected tile into shared storage.

=== UID:F04-S10
H: Other Operations: `transp` and `pad`
P: A plain copy preserves the source's shape and element order. Use `transp` when the destination needs a different dimension order, or `pad` when it needs extra positions filled with a specified value. Treat the destination shape as part of the operation: later tile selection and computation must use the transformed shape, not the original one.

=== UID:F04-S20
H: DMA vs TMA
P: DMA and TMA use similar source and destination expressions, but their implementations impose different constraints. The DMA path uses cooperating threads to move the tile. TMA describes a tensor transfer to an engine through a descriptor, so the compiler must be able to construct that description before the kernel uses it.
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
F: The descriptor supplies tensor layout information, the source expression supplies tile coordinates, and the transfer writes the shared destination.
L: **DMA path.** Threads cooperate to cover the tile; each lane participates in address math and load issue. | **TMA path.** A descriptor-based operation describes the tensor slice and delegates address generation to the transfer engine.
P: The descriptor carries the tensor geometry needed by the transfer engine. The compiler builds it from the function's shape and layout information. The program still waits for tile readiness and protects buffer reuse; changing the transfer engine does not remove either dependency.
F: DMA distributes transfer work across threads; TMA delegates tensor address generation to a descriptor-based transfer engine.

=== UID:F15-S18
H: Async Operation: the DMA Statement
P: Alongside parallel execution, Croqtile provides asynchronous DMA statements for data movement.
P: An asynchronous DMA operation runs alongside the SPMD code. Like an asynchronous CPU thread, it allows other work to continue, but its behavior is restricted to the configured data transfer.
P: The following code showcases one basic DMA statement.
C: An asynchronous DMA declaration
P: The assignment names `f` as the **future** of the asynchronous DMA operation. Use `wait f` to pause the current instance until the transfer completes. Until then, the instance can continue executing alongside the transfer.
P: Explicit temporary-buffer declarations can be repetitive. Croqtile also allows the following shorter form:
C: The same declaration with storage
P: Here, you do not need to specify the exact target location where the DMA transfers data. Instead, you only specify the destination memory type. This is advantageous in many scenarios. And to retrieve the transferred data, you simply invoke the 'data' member function of 'future'. Besides, the dma operation is not marked with '.async', so the wait operation on 'f' is not needed and cannot be performed.
P: The shorter declaration leaves allocation to the compiler, but the data dependency is unchanged. Identify the destination through the future, wait if the transfer is asynchronous, and check that the next operation expects the resulting shape and storage. The advanced patterns below preserve these requirements while allowing several transfers to be in flight.

=== UID:F28-S02
H: Data Movement Statement
P: Croqtile aims to address multiple levels of data movement. Croqtile adopts this terminology and abstracts data movement as **DMA Operations**.

=== UID:F28-S05
H: Operation Type
P: Croqtile maps these functionalities at the software level as different **DMA Operation Types**. The supported operations include:
L: `dma.copy`: Copies flat memory directly. | `dma.pad`: Pads shaped data while transferring the data. | `dma.transp`: Transposes shaped data while transferring the data.
P: The configurations, such as padding and transposing details, are programmed as parameters of the DMA operation. For example:
C: Operation Type
P: DMA configurations appear in angle brackets, and their contents depend on the operation. The supported syntax and constraints also depend on the target platform.
P: Support for operations other than `dma.copy` varies by platform. A transfer may map to TMA or to a sequence of load instructions issued by cooperating threads.

=== UID:F39-S01
P: The data movement is essential for *tileflow programs*. The *DMA statement* has a form for each of those levels, from a single copy to a chained non-blocking transfer.

=== UID:F39-S02

=== UID:F39-S03

=== UID:F39-S04

=== UID:F39-S05
H: Full Non-Blocking DMA Mode (Chain Mode in Croqtile)
P: A transfer that uses the result of another transfer must start after its predecessor completes. The `after` clause records that dependency without making the issuing code wait at that point. The example chains a copy from a staging buffer to an earlier operation:
C: Full Non-Blocking DMA Mode (Chain Mode in Croqtile)
P: In this example:
L: `out_store` is the asynchronous DMA operation that transfers data from l2_out to output.chunkat(m_tile, n_tile). | `out_store_s` is another DMA operation or event that must complete before out_store can proceed. | The after out_store_s syntax specifies that `out_store` should only start after the completion of `out_store_s`, which ensures that there is no blocking in the main thread.
P: The issuing code may continue while the dependency is pending. This does not make the destination immediately safe to read, and it does not release the source for reuse. A later consumer still needs the completion of the final transfer. Read a chain from source to destination and check each handoff, including the last one back to ordinary computation.

=== UID:F39-S06

=== UID:F39-S07

=== UID:F39-S08

=== UID:F05-S14
H: DMA Matmul: Tiles in shared storage
C: DMA Matmul: Tiles in Shared Storage
F: Each block stages [16, 16] operand tiles for one K slice; 256 threads accumulate the block’s [16, 16] output tile.
P: The scalar version indexed one element at a time. The tiled version first copies a tile into shared storage, then computes on that tile:
L: The outer `parallel` now uses `: block` with brace-form indices `px, py\`. An 8 $×$ 16 grid of blocks covers the output. | A `foreach` over `tile_k` walks the K dimension in 16 steps. Each step copies one strip of `lhs` and one strip of `rhs` into `shared` storage with `dma.`copy ... => shared. | An inner `parallel qx, qy\` by [16, 16] : thread creates 256 threads within each block. Each thread owns one output element within the block's tile. | The arithmetic reads from `lhs_load.data` and `rhs_load.data` --- the shared-memory copies --- instead of global `lhs` and `rhs`.
P: The composed indices work in two layers: `px#qx` composes block index `px` with thread index `qx` to form the global row; `py#qy` does the same for columns.
P: The block grid `[8, 16]` gives each block 16 rows and 16 columns of output. Its `[16, 16]` threads assign one output element to each thread. Sixteen K tiles, each containing 16 elements along K, cover the full contraction dimension of 256.
P: The reuse occurs inside the block. A loaded left-hand tile supplies several output columns, and a loaded right-hand tile supplies several output rows. The K loop visits every slice once and adds its contribution to the same output positions. As in the scalar sketch, initialize the output to zero before accumulating. Staging changes the source of the inner-loop reads, while the dot product remains the same.

=== UID:F08-S02
H: Double buffering with `swap`
P: Give the K-loop **two** logical buffers. While the math drains buffer 0, DMA fills buffer 1 with the next tile. After the math step, **swap** the handles: what was "next" becomes "current," and the freed slot is ready for the following load.
P: Croqtile spells this with `dma.copy.async` (non-blocking copy), `dma.any` (a placeholder future), `swap` (exchange future handles), and a three-phase loop:
C: Double buffering with swap

=== UID:F08-S03
H: The three phases
P: The prologue issues loads for tile 0 into `lf0` and `rf0`. No previous tile exists to compute on yet, so these first transfers establish the state needed by the loop.
P: In steady state, the loop issues the next loads into `lf1` and `rf1`, waits for the current loads, and computes from `lf0` and `rf0`. The two pairs refer to different buffers, so the next transfer can overlap the current computation. The next loads need not finish first. After the current computation, `swap` makes the next pair current for the following iteration.
P: The epilogue consumes the final loaded tile after the loop has stopped issuing transfers. It still requires that tile's load to be complete before reading it. The listing above illustrates handle rotation but omits a final wait and output initialization; a runnable accumulation must supply both, just as it does for the steady-state reads.

=== UID:F08-S04
H: `swap`: names, not bytes
P: The operation `swap(lf0, lf1)` exchanges the handles that name the current and next buffers; it does not copy their elements or wait for a transfer. For three buffers, `rotate(f0, f1, f2)` cycles the handles. In either case, follow the handle after the exchange to identify which transfer must complete before a read.

=== UID:F08-S06
H: `dma.any`: placeholder futures
P: A `dma.any` declaration reserves a future name before there is a transfer to assign to it. This lets both handles be declared in the scope containing the loop and its epilogue. The placeholder contains no ready tile: a real transfer must replace it before its data is consumed.

=== UID:F38-S01
H: Multi-Buffering Optimization with Croqtile
P: Overlapping computation with DMA data movement can improve kernel performance. This **multi-buffering** pattern requires multiple buffers so one can be filled while another is used for computation.
P: In Croqtile, this method requires two key programming primitives: the 'dummy' future **dma.any**, and the **swap/rotate** function. Let's illustrate by an example:
C: Multi-Buffering Optimization with Croqtile
P: The addition kernel keeps two local buffer sets for each input. The A handles name the current operands, while the B handles receive the next loads. After computing and storing the current result, the swaps advance those roles. The two inputs travel together through this schedule, so each addition uses matching left and right chunks.
P: The implementation divides data movements and computations into three stages, including:
L: the **prologue**(line 8-11). It preloads the first chunk of data. In this example, buffer A is pre-loaded with data. | the **body**(line 12-20). It processes the data preloaded in the previous iteration or in prologue, and preloads the data for next iteration. | and the **epilogue**(line 21-25). It processes the last data chunk only.
P: The placeholders `lfB` and `rfB` are declared before the inner loop so their names remain available in the epilogue. The loop replaces them with real transfer results before using their data. Declaring those names only inside the loop would put them out of scope when the final chunk is consumed.
P: The **swap** statements at lines 18-19 exchange futures to ensure the **wait** statement at line 15 functions correctly. Note that only futures with identical DMA operations can be swapped. Additionally, the iteration range of the *foreach* statement at line 12 is adjusted. `foreach y(1:)` means the loop over the *bounded variable* 'y' starts from the second value up to the upper bound. In line 17, the *chunkat* expression is also tuned. `y - 1` results in the current value of 'y' minus 1, setting up the body stage to work properly with the prologue stage. This adjustment is necessary because the iteration count must be reduced by 1, and the pre-load operation should fetch the 'next' chunk while computation is conducted on the 'current' chunk of data. Alternatively, you can use `foreach y(:-1)` and adjust the chunkat expression of the pre-load statements at lines 13-14 to achieve the same effect.
P: In line 25, the expression `y(-1)` retrieves the 'last value' of 'y' within its bound.
P: The **select** expressions in line 21-22 handle both odd and even upper-bound cases for the bounded variable 'y'. In either case, it obtains the future of the last chunk.
P: To use more than two buffers, replace **swap** with **rotate**.

=== UID:F09-S05
H: Example: TMA in a pipelined matmul
P: This example uses a producer to fill a ring of shared buffers and a consumer to accumulate matrix products from them. The producer waits for an empty slot, fills it with TMA, and signals that it is full. The consumer waits for that signal, loads operands with the matching swizzle, and releases the slot only after its matrix operations finish using it. Section 3.8.5 develops the event protocol in detail.
C: Example: TMA in a pipelined matmul
P: Changing the transfer path does not change which region owns a slot or when it can be reused. That separation is useful when comparing DMA and TMA: preserve the event ordering, then check the descriptor, tile-shape, and layout requirements of the selected path. Choose the store operation supported by the target.

=== UID:F06-S01
H: Tensor Contraction: the `mma` Syntax
P: 3.4's tiled matmul computes the inner contraction the way a CPU would: a `foreach k` loop where each iteration multiplies two scalars and accumulates into one output element. That works.
P: The job is **2D tensor contraction** --- the operation C += A $×$ B where A, B, and C are small, fixed-shape matrix tiles. It is the inner kernel of every GEMM, every convolution expressed as an im2col matmul, every attention head's QKT.
P: The difficulty is mapping matrix elements to the distributed operand fragments. Get the layout wrong --- say, load a column-major tile into a row-major fragment --- and the result is silently incorrect.
F: Operand fragments are distributed across cooperating lanes. The color pattern is schematic, not an exact instruction-level register map.
P: Croqtile exposes the contraction through `fill`, `load`, multiply-accumulate, and `store`. The compiler handles the distribution of fragment elements across the chosen cooperation scope. The programmer still chooses compatible tile shapes and states the operand layouts; the abstraction depends on those choices being correct.
F: Initialize the accumulator, load both operands, accumulate, and store; repeat the load and accumulation steps over K slices.

=== UID:F06-S03
H: Scaling the cooperation scope
P: The four-step syntax stays the same as the number of cooperating threads changes. The following examples vary the **cooperation scope** while preserving those operations.

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

=== UID:F09-S03
H: Swizzle and layout conflicts
P: A swizzle changes how elements of a staged tile are arranged in shared storage. The logical matrix shape stays the same, but the reader must use the matching address mapping. Croqtile exposes this choice on both DMA and TMA copies:
C: A swizzled copy
P: The MMA read path must use the same `swiz`N> so addresses match the staged layout:
C: A swizzled mma load
P: Swizzle is not a TMA-specific feature. In Croqtile, `dma.`copy.swiz<N> and `tma.`copy.swiz<N> produce identical shared-memory layouts. The difference is only in how the data gets there; the layout that arrives is the same in both cases.
P: The template argument selects a target-supported swizzle mode. The examples use numbered modes, including mode 0 for the identity mapping. Valid modes and compatible tile extents depend on the target; use the same documented convention on the transfer and MMA load.
P: The copy and `mma.load.swiz<N>` must agree on the layout. Loading a swizzled tile with an incompatible mode interprets the same bytes at different matrix positions, producing incorrect values. Treat this pairing as a correctness requirement when changing either operation.
F: Eight-bank schematic: different addresses in one bank serialize, while remapping spreads accesses. Copy and MMA load must use matching swizzle.

=== UID:F06-S07
H: Beyond dense FP16: what else the four steps express
P: The examples above use `mma.row.row` on dense FP16 tiles. The same four-step pattern extends to workloads the basic form cannot reach.
P: A sparse contraction also needs a **metadata operand** `me` that identifies the nonzero elements:
C: A sparse contraction
P: The `.sp` suffix adds the metadata operand; everything else is the same fill-load-multiply-store rhythm. Any layout combination works: `mma.row.col.sp`, etc. You load `me` from a separate metadata tensor alongside A and B.
P: For quantized operands with per-tile scales, the contraction also needs the scale values associated with the selected A and B tiles. The `.scale` form expresses this scaling alongside the matrix operation:
C: A scaled contraction
P: Each result element is scaled by `sc_a` and `sc_b` after the contraction --- no separate dequant kernel needed. Alternatively, scaling can be a **standalone statement** when the scale source differs from the standard fused path:
C: Another contraction layout
P: The standalone `mma.scale` appears in some MoE and mixed-precision kernels.
P: The ``N> must agree between `tma.`copy.swiz<N> and `mma.`load.swiz<N> --- a mismatch reads garbage. For output, `mma.`store.transp mc, dst writes the accumulator with rows and columns swapped, useful when the next stage expects column-major data.
P: The table below collects every variant for reference.

=== UID:F06-S04
H: One group, one tile: the simplest MMA matmul
P: The smallest example assigns one output tile to a cooperative group, written `: group`. It accepts left and right operands plus an output buffer supplied by the caller. Follow one output tile through the code: initialize its accumulator, visit every K slice, store the accumulated tile in shared storage, and copy it to its output position.
C: One group, one tile: the simplest MMA matmul
P: The function returns `void` because it writes into the `output` parameter. The caller supplies that buffer and reads it after the operation completes. Inside the function, `output_s` is the staging buffer used between `mma.store` and the final copy.
P: The ceiling division `cdiv(M, TILE_M)` counts the tiles along M, including a possible partial tile. The pair `(block_m, block_n)` identifies the output tile assigned to the current block. The operand selections use the same output coordinates with the current K-slice coordinate.
P: When the matrix dimensions and the tile dimensions are all 16, the block grid is 1 by 1. Larger matrices produce more output tiles. The `MMA_M`, `MMA_N`, and `MMA_K` names in the listing must be defined consistently with the intended fragment geometry before compiling it.
P: Section 3.5 explains how `subspan` and `chunkat` select tiles.
B: `subspan(h, w)` declares a fixed-size rectangular view shape, and `.at(i, j)` selects which view instance you want from the parent tensor. In other words, `lhs.`subspan(TILE_M, TILE_K).at(block_m, iv_k) means "take the `[TILE_`M, TILE_K] tile at `(block_`m, iv_k)." Use `subspan` when you want an explicit tile shape; use `chunkat` when you want to slice by equal chunks of an existing dimension.
P: The accumulator spans the entire K loop. Initializing it inside that loop would discard earlier contributions, while storing inside the loop would expose only a partial result. Keeping initialization before the loop and the store after it makes the output tile's lifetime visible.

=== UID:F06-S05
H: Widening the scope: cooperative group and mma
P: A wider cooperating group uses `: group-4` while retaining the same fill, load, multiply, and store operations. The example also changes tile dimensions and staging buffers to suit that scope. The shared syntax makes the two versions comparable, but a change in cooperation width still requires compatible tile geometry and target support.
F: The : group and : group-4 scopes use different cooperation widths while preserving the fill, load, multiply, and store interface.
C: Widening the scope: cooperative group and mma
P: The sequence remains fill, load, multiply, and store; the space specifier determines the cooperation scope.
P: The fill suffix chooses accumulator precision. This listing uses `mma.fill.f16 0.0f`, so the initial single-precision literal is converted for an FP16 accumulator. An FP32 accumulator is a separate choice, expressed with `.f32`, and provides a wider range and greater precision for partial sums. The operand type alone should not be used to infer what an explicitly annotated accumulator stores.
P: The expression `chunkat(_, iv_warp)` leaves the row dimension whole and subdivides only K. The underscore means that no partition is applied to that dimension.
P: The underscore in the operand selection leaves one dimension intact while the loop divides K into MMA slices. Each slice contributes to the same accumulator. This lets the outer staging tile be larger than the fragment consumed by one matrix operation without changing the output tile being accumulated.

=== UID:F06-S06
H: Tiling further: two cooperative groups sharing operands
P: 3.4 introduced `parallel p1 by 2 :` group-4 --- with MMA, both groups can share the same B tile while loading different rows of A.
C: Tiling further: two cooperative groups sharing operands
P: With `TILE_M = 128` and a 64-row group tile, `p1` selects either the upper or lower half of the block's output rows. The groups load different rows of A and share `rhs_load_s`, since both need the same B columns and K slice.
P: The four-step syntax stays invariant; only the parallel decomposition of the tile changes.
T: | **Aspect** | One group | One group-4 | Two group-4 groups | <ROW> | --- | --- | --- | --- | <ROW> | Operand sharing | N/A | N/A | B tile shared, A rows split by `p1` | <ROW> | Threads | 32 | 128 | 256 | <ROW> | Accumulator | `mma.fill 0.0` | `mma.`fill.f16 0.0f | `mma.`fill.f32 0.0f |
P: The two groups must store disjoint output row ranges. They may read the same right-hand tile because those reads do not modify it, but each group needs its own accumulator for its assigned rows. This separates operand reuse from output ownership: sharing an input does not imply sharing a partial sum.

=== UID:F06-S08
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `mc = mma.`fill 0.0 | Initialize accumulator tile to zero | <ROW> | `mma.`fill.f16 0.0f / `mma.`fill.f32 0.0f | Accumulator with explicit precision | <ROW> | `mma.`load.swiz<N> src | Load with swizzle mode (Section 3.7) | <ROW> | `mma.`row.row mc, ma, mb | C += A $×$ B (both row-major) | <ROW> | `mma.`row.col mc, ma, mb | C += A $×$ B (A row-major, B col-major) | <ROW> | `mma.`row.row.sp mc, ma, mb, me | Sparse MMA with metadata operand | <ROW> | `mma.`row.row.scale mc, ma, mb, sc_a, sc_b | Fused MMA + per-tile dequantization | <ROW> | `mma.`scale mc, sc_a, sc_b | Standalone post-MMA scaling | <ROW> | `mma.`store.transp mc, dst | Write accumulator transposed | <ROW> | `cdiv(a, b)` | Ceiling division | <ROW> | `_`_co__ void fn(...) | Kernel that writes results in-place |
P: Read the modifiers together with the data path. A swizzled load must match the layout produced by the copy; a sparse operation needs metadata for the same operand tile; scaling values must belong to the tile being accumulated. The short statement names do not remove these relationships.

=== UID:F06-S09
H: In short
T: | **Topic** | **Takeaway** | <ROW> | --- | --- | <ROW> | 2D tensor contraction | C += A $×$ B on tile-shaped operands --- the universal inner kernel | <ROW> | Four-step abstraction | **fill $→$ load $→$ multiply $→$ store**; the compiler handles fragment layouts for each target | <ROW> | Scaling cooperation | Syntax stays invariant | <ROW> | Layout contract | `mma.row.row`, `mma.row.col`, etc. | <ROW> | Sparse and quantized | `.sp` adds a metadata operand; `.scale` fuses per-tile dequantization | <ROW> | Swizzle and pipeline | `mma.`load.swiz<N> matches swizzled shared layouts; `mma.commit` fences pipeline stages |
P: The contraction is fast, but loads and compute still take turns --- 3.8 introduces **role specialization** so different thread groups can play different roles: one group loading data while another computes, overlapping memory and math in time.

=== UID:F07-S01
H: Control Flow: Divergence, Concurrency, and Coordination
P: Every programming language that targets a parallel machine must answer one question: **what happens when different threads need to do different things?**
P: On a CPU, each core has its own program counter. When threads share an instruction stream, however, divergent paths execute in sequence while inactive threads are masked.
P: All threads evaluate the condition, and those for which it is false are masked. This **predicated execution** is simple and widely used, but divergent paths can be expensive.
P: A tiled matrix multiplication also has work with different roles: one group can load the next operands while another computes with the current operands. The groups need to progress independently between well-defined handoffs. A runtime condition describes which branch an instance takes; a pipeline also needs a way to express those separate roles and their dependencies.
P: Croqtile introduces two additional control-flow primitives to fill this gap:
L: **`inthreads.`async** --- **structured concurrent regions**: compile-time partitioning of threads into groups that run **different programs simultaneously**. The compiler generates separate instruction streams; | **`shared event` / `wait` / `trigger`** --- **inter-region signaling**: lightweight synchronization tokens that let concurrent regions communicate safely.

=== UID:F07-S03
H: Structured concurrent regions with `inthreads.`async
P: The form `inthreads.async` assigns different thread groups to separately progressing regions. Its predicate identifies the group responsible for each region. This is useful for a producer and consumer whose roles are fixed even though their progress depends on which buffers are ready.
C: Structured concurrent regions with inthreads.async
P: The distinction from `if` is structural, not just performance:
T: |  | **`if` (predicated execution)** | **`inthreads.`async** (structured concurrency) | <ROW> | --- | --- | --- | <ROW> | **Resolution** | Runtime --- every thread evaluates the condition | Compile time --- thread assignment is fixed | <ROW> | **Instruction streams** | One program; divergent threads masked | Separate programs per region | <ROW> | **PL analogy** | `if`/`else` in any language | `async`/`spawn` in structured concurrency (Trio, Go goroutines, Cilk) |
P: The regions are lexically scoped, so the compiler can identify their participants from the program structure. The code does not create an unbounded number of tasks at runtime. Each region operates within the declared parallel scope.
P: Without `.async`, the region rejoins the surrounding execution before later code proceeds. Adding `.async` allows other regions to advance independently. Any data handoff between them then needs an explicit synchronization protocol.
P: The figure below shows the effect. --- The producer's DMA and the consumer's MMA overlap in time:
F: Four load/compute pairs are shown in both schedules. Concurrent producer and consumer regions overlap work while preserving dependencies.

=== UID:F07-S05
H: Inter-region signaling with events
P: When `inthreads.`async creates concurrent regions, those regions need a way to communicate. Croqtile provides **events**.
C: Inter-region signaling with events
P: Events have two operations:
L: **`trigger name`** --- signal that a condition is met (e.g., "data is ready") | **`wait name`** --- block until the corresponding `trigger` fires
P: The producer calls `trigger full` after writing a tile to signal "data ready." The consumer calls `wait full` before reading, blocking until the signal arrives. Symmetrically, the consumer triggers `empty` after finishing its read (the buffer can be reused), and the producer waits on `empty` before writing the next tile.
P: This is a **credit-based bounded buffer** protocol --- `full` is the "data available" credit; `empty` is the "buffer free" credit.

=== UID:F08-S01
H: Synchronization in Practice: Pipelines, Buffers, and Events
P: A buffer pipeline has two ordering requirements. A consumer must wait until the producer finishes filling its slot, and the producer must wait until the consumer finishes using that slot before overwriting it. A fast load does not remove the second requirement, and a fast consumer does not remove the first.
P: The double-buffering pattern in Section 3.6.2 uses one group to issue transfers, wait for current inputs, and rotate handles. A producer-consumer pipeline assigns those roles to separate asynchronous regions. Events then carry the readiness and reuse signals that the single-group version expressed through its own instruction order.
P: Both patterns solve the same underlying problem: **you cannot read a buffer while someone is writing to it**. They differ in how they structure the solution.

=== UID:F08-S09
H: Why you cannot "just overlap" without events
P: The `swap` pattern uses one thread group to control both loading and computation in a known order. Separate producer and consumer groups need a signaling mechanism to coordinate their independent schedules.
P: The picture below contrasts a strict load-then-compute staircase with double-buffered overlap: same logical work, less idle time.
F: The same four K tiles run sequentially or with two-buffer overlap. Timings are schematic and include pipeline startup and drain.

=== UID:F34-S07
H: Make it Asynchronous
P: Like `parallel-by`, `inthreads` can execute asynchronously. This lets instances running a divergent region proceed independently of those outside it, expressing MPMD behavior within an SPMD program:
C: Make it Asynchronous
P: The example divides the parallel instances into two groups without synchronizing after each `inthreads` block, allowing the groups to execute concurrently. The subsequent `sync.shared` establishes a synchronization point that both paths must reach before execution continues.
P: Only the outer `inthreads` can be declared as asynchronous. Programmers will get a compile-time error when using `inthreads.`async for any inner `inthreads` block.

=== UID:F35-S03
H: Event Instances
P: In Croqtile, events are associated with the storage specifier, either `global`, `shared`, and `local`. Such storage specifier specifies the event instances.
C: Event Instances
P: The comments identify the event instances. As with spanned-data declarations, the storage specifier determines the level at which an event is stored and shared. An implementation may represent events with Boolean state at that storage level; the number of instances follows the corresponding parallel scope.

=== UID:F35-S04
H: Deadlocks
P: Events are lock-free by default to support efficient synchronization between asynchronous blocks. Multiple instances updating the same event require care, particularly when using `wait`.
P: For example,
C: A single-instance deadlock
P: Both threads wait on the same event `e`. Because `wait` resets the event, `thread-0` may reset it before `thread-1` observes the trigger. In that case, `thread-1` remains blocked and the program deadlocks.
C: A credit-count deadlock
P: In this code, you utilize an event array to guarantee each event is waited and reset by a single thread.

=== UID:F07-S04
H: The canonical pattern: 1 producer + 1 consumer
P: The most common use of `inthreads.`async is the **1P1C** (one producer, one consumer) split for matmul:
C: The canonical pattern: 1 producer 1 consumer
P: The declaration `parallel p1 by 2 : group-4` creates two cooperating groups in the same surrounding block. The value of `p1` selects a role; it does not identify a tile along K.
P: The region for `p1 == 0` is the producer. It owns the writes that fill the shared staging buffers. The other group can proceed with computation while those writes target a different, available slot.
P: The region for `p1 == 1` is the consumer. It reads completed tiles and updates its accumulator. The two regions share staging storage, so the event protocol must identify when a slot changes from writable to readable and back again.
P: Because both regions share a buffer, their reads and writes must be coordinated. Without synchronization, the consumer could read before the producer finishes writing. Events express the required ordering.

=== UID:F07-S08
H: A 1P1C matmul skeleton
P: Event-based synchronization is left out of this skeleton; 3.8.5 adds the full pipeline protocol. Focus on the program structure:
C: A 1P1C matmul skeleton
P: The producer loop visits `cdiv(K, TILE_K)` input slices. For each slice it fills the left and right staging buffers. Both operands must be ready before the consumer starts the corresponding contraction.
P: The consumer initializes one accumulator for its output tile, then loads and multiplies the operand fragments for each K slice. It writes the output after the last contribution. Because the two loops run independently, matching loop counts alone does not guarantee that the consumer sees the intended slice.
P: The skeleton omits the handoff between these loops. Before use as a working pipeline, add a readiness signal after loading and a reuse signal after consumption. Section 3.8.5 supplies both with per-slot events.

=== UID:F07-S09
P: In the earlier kernels, the grid grew with the problem: roughly one block per output tile. For large matrices that means large launch counts, and the last wave of blocks often leaves SMs partially idle --- **tail underutilization**.
P: A **persistent kernel** fixes the launch size (often near the SM count) and lets each block iterate over multiple tiles:
C: Persistent scheduling and the if guard
P: The declaration fixes the number of workers at `NUM_GROUPS`. Work size changes the number of loop iterations rather than the number of launched blocks.
P: The expression `tile_iter # block_id` has value `tile_iter * NUM_GROUPS + block_id`. It assigns the next consecutive set of tiles to the workers on each iteration.
P: The condition `tile_id < total_tiles` excludes indices beyond the final valid tile. All work for that tile belongs inside the guard, including any producer-consumer protocol whose participants must agree whether the tile exists.
F: Four fixed blocks process ten tiles in three iterations; the final two padded tile IDs are skipped by the if guard.
P: For four workers and ten output tiles, the composed tile indices are 0--3 on the first iteration, 4--7 on the second, and 8--11 on the third. The guard accepts only 8 and 9 in the last iteration. This covers each valid tile once, but it does not guarantee that every worker stays busy until the end; tile cost and the remaining work still determine utilization.

=== UID:F07-S10
H: Data-dependent vs persistent grids
T: | **Aspect** | **One block per tile** | **Persistent (`NUM_GROUPS` blocks)** | <ROW> | --- | --- | --- | <ROW> | Grid size | Grows with problem | Fixed | <ROW> | Tail utilization | Last wave may leave SMs idle | Workers can process further tiles; a tail may remain | <ROW> | Extra constructs | Minimal | `total_tiles`, `tile_`iter # block_id, `if` | <ROW> | Complexity | Lower | Higher |

=== UID:F07-S11
H: `parallel.async`: host-level concurrency
P: Everything above runs inside a kernel. Sometimes you need concurrency at the **host level**: launch a grid without blocking the CPU.
C: parallel.async: host-level concurrency
P: An asynchronous launch returns control to the host after submitting the kernel. Completion happens later. Stream binding selects which stream receives the work; without an explicit binding, the default stream is used.
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
L: Stream binding is only allowed on **block-level** parallel (`: block`). | The identifier inside ``...> must be a `stream`-typed variable. | Without stream binding, synchronous `parallel-by` uses device-wide synchronization.
P: Without stream binding, asynchronous `parallel-by` uses the default stream.
P: A stream selects an ordering context for launches. It does not make two kernels independent if the second reads data written by the first. When using separate streams, keep that dependency explicit and retain the buffers until their queued users finish. Host-side return from an asynchronous launch only confirms that execution was submitted.

=== UID:F08-S10
H: The full 1P1C event pipeline
P: This kernel combines `inthreads.`async (from 3.8) with event arrays to build a complete multi-stage pipeline. The producer and consumer run as separate concurrent programs, coordinated entirely through `wait` / `trigger`:
C: The full 1P1C event pipeline

=== UID:F08-S11
H: Walking through the kernel
P: The stage index is `iv_k % STAGES`. With two slots, tiles 0 and 2 use slot 0, while tiles 1 and 3 use slot 1. These are different logical tiles sharing physical storage at different times; the event pair belongs to the slot, not to one permanent tile.
P: The producer waits on `empty[stage]` before writing either operand into that slot. Both copies in this example are synchronous, so triggering `full[stage]` after them means both operands are ready. Replacing the copies with asynchronous operations would require completing those operations before sending the same signal.
P: The consumer first triggers every `empty` event. All events begin unset, so this bootstrap grants the producer permission to fill each slot once. No `full` event is triggered here because no input has been loaded yet.
P: The consumer waits on `full[stage]`, loads the operand fragments, and accumulates the slice. It then performs `mma.commit` before triggering `empty[stage]`. That order matters: issuing a matrix operation is not sufficient evidence that it has stopped reading the shared operands.
P: The completion point protects buffer reuse. If the producer overwrites a slot while MMA still reads it, the consumer can mix operands from two K slices even though the tile shapes are correct. Checking the `full` signal alone cannot detect this mistake; the return of the `empty` credit is equally necessary.

=== UID:F08-S12
H: Credit flow for one stage
P: The diagram matches the code: bootstrap grants empty credits; the producer waits on `empty`, fills, signals `full`; the consumer waits on `full`, computes, signals `empty`. When `iv_k` wraps modulo `STAGES`, the same physical stage re-enters the cycle.
F: An empty credit permits filling a stage; a full credit permits reading it. The consumer releases the stage only after computation completes.

=== UID:F07-S14
H: In short
T: | **Concept** | **Primitive** | **When to use** | <ROW> | --- | --- | --- | <ROW> | Predicated execution | `if` | Data-dependent decisions (bounds, conditions) | <ROW> | Structured concurrency | `inthreads.`async | Compile-time thread partitioning (producer/consumer, heterogeneous roles) | <ROW> | Inter-region signaling | `shared event` / `wait` / `trigger` | Coordination between concurrent regions | <ROW> | Host concurrency | `parallel.async` / `stream s` | Multi-kernel overlap, non-blocking launch | <ROW> | Persistent scheduling | `if` + `foreach` + `#` | Fixed grid size, tile striping with padding guard |
P: The completed pipeline adds two dependencies to the earlier skeleton: input readiness before computation, and completion of operand use before slot reuse. When adapting it, trace one slot through a full cycle and then through its next reuse. This exposes missing signals and mismatched iteration counts more clearly than looking at either loop alone.

=== UID:F08-S15
H: In short
T: | **Pattern** | **Primitives used** | **When to use** | <ROW> | --- | --- | --- | <ROW> | `swap` double buffering | `dma.any`, `swap` | Single thread group interleaving load and compute | <ROW> | Event pipeline | `inthreads.async`, events | Separate producer and consumer groups |
P: Handle rotation is enough when one group controls both the transfer schedule and the computation. Separate producer and consumer regions need events because neither can infer the other's progress from its own instruction order. More buffering can give them room to overlap, but it cannot repair a missing dependency.
P: The same synchronization patterns can be combined with the richer data-movement primitives described in this guide.

=== UID:F10-S01
H: C++ Interop: Device Functions, Inline Code, and the Preprocessor
P: A kernel may need a reduction, a sorting routine, or an instruction that Croqtile does not express directly. Existing device C++ can supply that computation while Croqtile continues to arrange its inputs, outputs, and parallel scope. The interface needs to preserve the assumptions on both sides of the call.
P: Croqtile's interoperability story has three parts:
L: **`__device__` functions** --- The compiler passes them through unchanged. | **`__cpp__`** | **The preprocessor** --- `#define`, `#if`, `#ifdef`, `#error` for compile-time configuration, the same role `#define` plays in C/C++ codebases.
P: The preceding sections introduced tiles, parallelism, MMA, pipelines, events, and TMA. C++ interoperability extends these abstractions when a kernel needs additional device functions or target-specific code.

=== UID:F10-S03
H: `__device__` vs ordinary C++ inline
P: Both `__device__` functions and ordinary C++ functions are passed through unchanged, but they land in different compilation paths --- and the distinction matters.
P: A `__device__` function compiles to device code. The `__co__` body invokes it with `call`, and the generated `__global__` kernel calls it directly on the device.
P: An ordinary C++ function (no `__device__` or `__co__` annotation) compiles to **CPU code**. It runs on the host --- setting up buffers, launching kernels, printing results. You cannot call a host function from inside a `__co__` body.
P: The boundary is strict: `__device__` lives on the device side of the `c++` split, host C++ lives on the host side. The `__co__` function is the bridge --- the compiler transforms it into a `__global__` kernel (device side) and generates a host-side wrapper that launches it.

=== UID:F23-S05
H: Difference with C++ pre-processing
P: Croqtile preprocessing runs much earlier than c++ preprocessing. The workflow of croqtile compilation is as shown below:
C: Differences from C++ Preprocessing
P: Croqtile preprocessing coordinates macros across the host and device portions of a program. Its implementation substitutes macros and handles conditional compilation within Tileflow functions, while leaving other preprocessing to the C++ preprocessor. This division can produce differences from ordinary C++ preprocessing.

=== UID:F23-S06
H: Pre-defined Macros
P: Croqtile also imports the target's built-in macros to match native compilation. These macros can be used in both Tileflow functions and host code.

=== UID:F24-S05

=== UID:F24-S06
H: Trigger C++ Template Instantiation
P: It is possible for Croqtile code to make function call to a template function. For example:
C: A templated device helper
P: The call supplies three constant template arguments to `matmul_kernel`. The target compiler instantiates the corresponding specialization, just as it would for an ordinary C++ template call.
P: Template arguments must be evaluable as constants during Croqtile compilation. An input-dependent value cannot select a C++ specialization at runtime:
C: A call that instantiates it
P: The Croqtile compiler attempts to evaluate template arguments at compile time. If it cannot determine a required constant, compilation stops with an error.

=== UID:F25-S08
H: More Questions
P: Like C++ functions, Croqtile functions can be defined throughout a source file. When linking a module that contains one, its **function signature** determines how it is called.
P: A Croqtile function follows the C++ calling convention and uses C++ name mangling. It does not support `extern "C"`, because its spanned-data parameters use C++ template-based types.
P: Some features are not supported yet:
L: Croqtile function declaration without definition. This allows calling Croqtile function from another module. | Croqtile function with `inline`. This allows the Croqtile function to be used in headers.

=== UID:F10-S04
H: `__device__` functions: target alongside Croqtile
P: A custom reduction, sorting network, or specialized math operation can be implemented as a device helper.
P: A `__co__` function can call a `__device__` function using the `call` keyword:
C: device functions: target alongside Croqtile
P: The helper receives raw pointers to element storage. Its interface must therefore state any sizes or layout assumptions that it needs; the pointer alone does not preserve the Croqtile shape.
P: The `SHFL_XOR` helper is ordinary device code passed to the target compiler. Its use inside the computation belongs to the helper's implementation; Croqtile does not translate that body into tile operations.
P: The `call group_topk<8>(...)` statement supplies the helper's compile-time configuration and ordinary arguments. Croqtile converts spanned-data arguments to element pointers, and the target compiler validates the resulting C++ call.
P: The `inthreads (tid == 0)` region selects one thread for the enclosed work and rejoins afterwards. It has no `.async` modifier, so it does not introduce a separately progressing producer or consumer.
P: In this pattern, Croqtile handles parallelism, tiling, and data movement, while the device helper performs the specialized computation. Production kernels use it for MoE top-K, custom reductions, and specialized math.

=== UID:F10-S06
H: storage hints: `setmaxnreg`
C: storage hints: setmaxnreg
P: The injected strings place target-specific register hints at the start of the producer and consumer regions, before their long-running loops. The two roles can have different resource needs. Croqtile preserves the instruction text; the target compiler and device determine whether the hint is valid. Keep this code close to the region whose behavior it changes, and verify the result with the target tools.

=== UID:F10-S13
H: In short
T: | **Syntax** | **What it does** | **Runs on** | <ROW> | --- | --- | --- | <ROW> | `#`define NAME value | Object-like macro; shared across `__co__` and host regions | Preprocessor | <ROW> | `#if` / `#ifdef` / `#error` | Conditional compilation and compile-time assertions | Preprocessor | <ROW> | `-DNAME=value` | Define macros from the command line for tile sweeps | Preprocessor | <ROW> | `call helper(...)` | Invoke a device helper | Device | <ROW> | `__cpp__("...")` | Insert target C++ at the current scope | Target-dependent |
P: 3.11 turns to the other side of the workflow: what to do when the kernel compiles but the output is wrong --- debugging, verbose modes and systematic narrowing. The two worked programs in 3.10 put the interoperability mechanisms into context.

=== UID:F09-S04
H: Why the restricted interface works: expressiveness vs performance
P: Writing every load and address calculation separately gives a programmer many choices, but it also leaves the compiler to recover which operations form one tile transfer. Croqtile asks the program to describe that region explicitly. The restriction is useful when the desired movement fits a regular shape, because its geometry is available before individual transfers are generated.
P: Croqtile makes tile geometry and layout explicit through structured movement operations. This gives the compiler the information it needs to lower a transfer efficiently and avoids repeating per-element address choreography.
F: Structured tile operations expose geometry and layout for efficient lowering; performance still depends on the chosen tiles and execution configuration.
P: The trade is explicit: structured operations constrain how a transfer is described. They support efficient lowering, while actual performance still depends on tile sizes, alignment, layouts, and synchronization.

=== UID:F13-S02
H: Easy to Use
P: A tile selection can use the input shape and bounded indices directly. For example, changing the number of chunks changes the derived chunk size without requiring a second set of manually calculated offsets. The programmer still chooses the partition; the language keeps the consequences of that choice connected.

=== UID:F13-S03
H: Compile-Time Safety
P: Shape and rank checks catch incompatible uses before execution when enough information is available. Runtime checks cover constraints that depend on actual inputs, and assertions can state additional application requirements. Together they help localize failures, but they do not replace checking output values or the synchronization protocol.

=== UID:F13-S04
H: Dynamic Shapes
P: Symbolic dimensions let one source program express a family of input shapes. Output sizes and iteration counts can be derived from the named dimensions instead of repeated as independent constants. Each call must still satisfy the kernel's tiling, layout, and target constraints; symbolic shapes do not imply that every shape is supported equally well.

=== UID:F13-S05
H: Born for AI Tuning
P: Compact source gives an automated tuning process room to retain the kernel, compiler diagnostics, and measurement results together. Explicit tile sizes and cooperation scopes also provide concrete choices to vary. Each candidate still needs correctness checks and measurement on the intended workload; a successful compilation only establishes part of that evidence.

=== UID:F40-S01
H: Tiling Optimisation in Croqtile
P: The following examples use the same tile-selection machinery for different computations. Matrix multiplication carries partial sums across K slices; RGB-to-grayscale conversion processes each selected image region independently. Comparing their data paths shows which parts of the structure come from tiling and which come from the computation itself.

=== UID:F40-S02
H: Example 1: `matmul`
P: The first example is matrix multiplication on a device program. The code shows how Croqtile's tiling and memory management mechanisms work together to optimize matrix operations, including how DMA (Direct Memory Access) is leveraged for efficient data movement.

=== UID:F40-S03
H: Example Code: Matrix Multiplication with Tiling
C: Example Code: Matrix Multiplication with Tiling
P: The outer staging buffers hold full operands in this configuration because the outer tile bounds are all 1. The inner bounds `[4, 4, 8]` divide M, K, and N: a left tile is `[128, 256]`, a right tile is `[256, 128]`, and their product contributes to a `[128, 128]` output tile. The `after` clauses order the inner input copies after the outer copies. This listing is a structural illustration: helper definitions, complete waits before operand use and buffer reuse, and output initialization must be supplied and checked before treating it as a runnable kernel.

=== UID:F40-S04
H: Memory Tiling with `chunkat`
P: In the matrix multiplication example, **memory tiling** is achieved using the `chunkat` method, which slices the input matrices `lhs` (left-hand side) and `rhs` (right-hand side) into smaller chunks to be processed in parallel.
L: The `lhs` and `rhs` matrices are tiled using the `chunkat` method. The chunks are defined by `m_tile`, `k_tile`, and `n_tile`, which represent different dimensions of the matrices involved in the multiplication. | The chunks for both matrices are loaded asynchronously into shared storage (`lhs_load` and `rhs_load`) with the `dma.copy.async` operation. This operation loads a portion of the matrix from a source buffer into a staging buffer, which is a common optimization in large matrix operations to improve data locality.
L: The variables `m_tile`, `k_tile`, and `n_tile` are **bounded variables**. They are defined in the context of `parallel-by` and `with-in` constructs, which define the iteration space for the tiling. | These bounded variables allow the chunks to be computed in parallel, where each thread works on a different slice of the matrix. This parallelization enables efficient computation of matrix multiplications across multiple threads.

=== UID:F40-S05
H: Example 2: rgb2gray
P: The second example is the RGB to Grayscale conversion algorithm. It converts a color (RGB) image to a grayscale image using a weighted average method. It shows how physical parallelism works together with virtual parallelism.

=== UID:F40-S06
H: Example Code
C: Example Code
P: The input shape is `[N, 3, H, W]`, with the three color channels in the second dimension. The underscore in `chunkat(q#n, _, h, w)` keeps those channels together while selecting one image and a spatial region. Removing that channel dimension from the loaded shape gives the shape of `out_L1`; the helper writes one grayscale value for each spatial position.
P: The composition `q # n` assigns a consecutive group of images to each of the six parallel instances. The bounds assume that N divides by 6, H by 16, and W by 512 for this regular partition. Within those assumptions, each input tile contains all three channels for a 16 by 512 region, and the final copy writes its grayscale result to the matching output tile. Other image sizes need an explicit boundary policy.

=== UID:F11-S01
H: Debug and Verbose: Printing, Assertions, and GDB
P: Debugging works best by **systematically narrowing the search**, starting with inexpensive checks. Static analysis and compile-time assertions catch errors without runtime overhead. Targeted runtime checks then help locate the problem. Use an interactive debugger once these checks have narrowed the possible causes.
P: Croqtile provides tools at each level: **compile-time shape printing** (`print!`, `println!`) to verify tile dimensions without launching a kernel, **runtime assertions** (`assert`) for invariant checks, **guarded device `println`** for runtime inspection of specific threads, **debug RTTI** for `debugger`, and a **verbose mode** (`-v`) to see what the compiler is actually doing under the hood.
P: Start from a small input whose expected result is easy to compute. For a wrong matrix element, recover its tile and local coordinates, check the selected input slices, and compare one partial sum before examining the full output. If the program hangs, trace one event slot instead: record its initial state, each trigger, and the instance expected to consume it. This distinguishes an indexing error from a missing handoff.
P: When the failure occurs during compilation, identify the stage before changing the kernel. Preprocessed output reveals macro expansion, generated source reveals Croqtile's translation, and verbose output identifies the target compiler command. Once the issue is fixed, rerun the reference comparison with diagnostic output disabled before measuring performance. Sections 1.4 and 2.14 give the corresponding commands and inspection syntax.

