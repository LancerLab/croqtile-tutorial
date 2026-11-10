=== UID:F03-S02
P: At its simplest a program has two parts. A `__co__` function describes computation on shaped tensors, and the compiler translates that function into target code. The host program is standard C++ that prepares the data, calls the function and checks the result. Both live in a single `.co` file, and the `__co__` prefix is all the compiler needs to tell them apart.

=== UID:F14-S03
H: Host, Device and Tileflow
P: Both of those parts are present here, but the ordinary C++ side is worth splitting once more: a `.co` file may also carry a `__device__` helper. The listing is written from the inside out --- a device helper, the `__co__` function that calls it, and the host `main()` that calls the `__co__` function.
C: The three regions of one source file
P: Execution starts at the bottom of the listing, in the host program. Written in standard C++ and run on the CPU, it is the entry point of the module and the caller of the Tileflow program --- the `__co__` functions --- and it manages the overall workflow of the heterogeneous application. A typical host body prepares the input data, invokes a Croqtile function, handles its return value and advances to the next step.
P: The Tileflow program called from there is the part Croqtile adds. Built from `__co__` functions, it is the core of a `.co` program: it orchestrates data movement between host and device and between storage levels within a single device. A typical Tileflow body moves data into a buffer at the appropriate storage level, calls a device function to compute on it, and moves the result back to the host. It is the only region the compiler translates.
P: Device functions sit at the other end of that call chain. A `__device__` function defines computation-intensive operations that run on the target device; the prefix marks it as device code. Like the host program, it passes through Croqtile compilation unchanged. The compiler leaves both sides of ordinary C++ alone; only the `__co__` function between them is rewritten.

=== UID:F03-S04
H: The Croqtile Function
P: Start with the signature, which declares more than a C++ function prototype would: every parameter and the result carry a shape.
C: A kernel signature
P: The `__co__` prefix marks a Croqtile function. The declaration `s32 [4, 8]` combines signed 32-bit elements with a two-dimensional shape: four rows and eight columns. Both inputs and the result follow the form `type [shape] name`. These declarations give the compiler the information needed to check ranks and reason about index bounds. Constraints that depend on caller-supplied dimensions may also require runtime checks.
P: Croqtile supports these element types: `s32` (signed 32-bit int), `f16` (half-precision float), `bf16` (brain float), `f32` (single-precision float), and `f8_e4m3` / `f8_e5m2` (8-bit floats). In the examples that follow, `s32` is the simplest to reason about.
C: Allocating the output
P: This allocates the output tensor. The expression `lhs.span` copies the full shape from `lhs`, so `output` automatically has shape `[4, 8]`. If you later change the input shape, the output adjusts --- a pattern that makes Croqtile functions easier to generalize.
C: The parallel body
P: The statement `parallel i, j by [4, 8]` declares 32 independent instances, one for each element position. An instance with `(i, j) = (1, 3)` reads that position from both inputs and writes the sum to the same position in `output`. No two instances write the same element, and none needs a result produced by another instance. That independence is what permits parallel execution.
P: Without a space specifier, the compiler chooses how to map the instances to execution resources. The statement establishes their independence; it does not require one physical thread per element or all instances to run at once.
C: Returning the result
P: A `__co__` function returns its result tensor. The return type in the signature (`s32 [4, 8]`) must match what you actually return. The caller on the host side receives a `croq:`:spanned_data --- an owning buffer with shape metadata that you can index into with `[i][j]`.

=== UID:F08-S08
H: `_`_co__ auto return type
P: Use `auto` when the result shape is easiest to describe inside the function. In `__co__ auto matmul(...)`, the compiler infers the return type from the buffer named by `return output`. The result still has an element type and a shape; `auto` saves repeating that information in the signature. An explicit return type is useful when you want the signature itself to state the required output shape.
P: The return convention is the same for element-wise arithmetic, scalar matrix multiplication, and MMA kernels that return a buffer. It does not depend on how the function computes the elements.

=== UID:F03-S05
H: The Host Program
P: The host supplies both storage and shape information. In the addition example, it first creates two owning buffers, fills them, and passes views to the kernel. Keeping ownership on the host lets the same input buffers remain available for checking the result after the call.
C: Creating the shaped inputs
P: The call `croq::make_spandata<T>(dims...)` allocates an owning host buffer. The template parameter selects the element type, and the arguments supply its dimensions. The calls to `fill_random` then initialize its elements within the specified range.
C: Calling the kernel
P: Calling a `__co__` function from C++ looks like a normal function call. The `.view()` method produces a non-owning `spanned_view` from an owning `spanned_data` --- this is how you pass host tensors into a Croqtile function without transferring ownership. The return value `res` is a `croq:`:spanned_data that owns its buffer.
P: The rest of `main` verifies the result by iterating over every element and asserting equality. The host program handles setup and verification; the computation takes place in the `__co__` function.

=== UID:F05-S16
H: Host Code
P: The host side uses the same `make_spandata` / `.view()` pattern used by the host code above:
C: Host Code
P: The loop after the call checks every element against a plain C++ reference computation, so an incorrect result is reported before the program prints anything.

=== UID:F15-S21
H: Parameter Passing between Croqtile and C++ Function
P: Passing arguments between Croqtile and C++ requires the `croqtile.h` header. Combine a raw C++ pointer with its dimensions to construct Croqtile spanned data, as shown below.
C: Parameter Passing between Croqtile and C Function
P: The `make_spanview` utility template wraps the data without copying it. In the Croqtile function `foo`, the corresponding parameter `d` has type `f32 mdspan`2>. An implicit conversion allows `d` to be passed as the first argument to the C++ function `bar`; `|d|` supplies its total element count.
P: Scalar values can also pass between C++ and Croqtile functions. An `ituple`, however, is used only inside a Croqtile function.

=== UID:F25-S01

=== UID:F25-S02

=== UID:F25-S03
H: Input Parameters
P: A Croqtile function accepts only two kinds of input parameter, the *spanned data* and the *integer*.
P: Similar to any C++ program, the parameter may either be named or unnamed:
C: Input Parameters
P: The second parameter is unnamed, so the function body cannot refer to it. The declaration is still valid.

=== UID:F25-S04
H: C++ Host: Data Argument with Shape
P: Croqtile spanned data carries a shape in addition to its data pointer. Before passing a C/C++ pointer or array to a Croqtile function, wrap it with the required shape information.
P: One direct method is to use the Croqtile API provided in `croqtile.h`. A program does not have to include `croqtile.h` explicitly in a Croqtile compilation environment; the compiler includes it for every Croqtile program.
C: Passing a shaped buffer
P: The `croqtile:`:make_spanned function template takes the rank as a template argument and the typed pointer and shape list as function arguments. It returns a `croqtile:`:spanned_view that wraps the pointer and dimension list, making them a valid Croqtile argument. Its simplified definition is:
C: The host-side buffer wrapper
P: Dimensions do not have to be constants; runtime values work as well:
C: A call built from runtime dimensions
P: Here, `input` has a runtime shape. On entry to the Tileflow program, Croqtile checks that `M == 7` and `N == 16`. **If either condition fails, the program terminates immediately**, because the function's shape assumptions do not hold.
P: Spanned-data inputs are **references to the original data or buffers**. Calling a Croqtile function does not copy the input data, even though the parameter syntax may resemble a declaration with value semantics.

=== UID:F25-S05

=== UID:F25-S06
H: Return Type and Type Deduction
P: Similar to the input, *spanned data* and *integer* are valid output types. The *void* type is also allowed if nothing is returned.
P: In addition to the explicit return type annotation, Croqtile supports type inference on returns. For example:
C: Return Type and Type Deduction
P: Here, `auto` is inferred as `f32 [7, 16]` from `result`. This is useful when the output shape is derived inside the function: changing that derivation does not require a matching edit to the return annotation. Use an explicit type when a fixed return shape is part of the interface you want to enforce.

=== UID:F25-S07
H: C++ Host: Shape Return
P: The C++ host code that receives the output of Croqtile function actually receives a buffer. The return has value semantics: the caller receives the buffer itself, not a reference to an object allocated inside the called function. For example:
C: Returning a shaped buffer
P: The example returns `output` with value semantics, but the implementation avoids copying the buffer. The returned C++ type is `croqtile:`:spanned_data, whose simplified definition is:
C: The host-side result wrapper
P: Unlike `croqtile:`:spanned_view, `croqtile:`:spanned_data owns the buffer it points to. **Returning it does not copy the buffer**; the implementation uses return value optimization.
P: Use member functions such as `.rank()` and `.shape()` to query the returned buffer's shape. The host can then work with the result without depending on the implementation of the Croqtile function.

=== UID:F04-S16
H: `span` and `span(i)`
P: The member `tensor.span` returns the whole shape, while `tensor.span(i)` returns the extent of dimension i. For matrix multiplication, the output takes its row count from the left input and its column count from the right input:
C: span and span(i)
P: The example builds a result whose dimensions come from two different operands. Reusing those dimensions keeps the output declaration tied to its inputs. The same approach is useful for a staging buffer that must match the shape of a moved tile.
F: For a [4, 8] tensor, .span returns [4, 8], while .span(0) and .span(1) return 4 and 8.

=== UID:F16-S01
P: A shape in Croqtile is a value a program can name and compute with, not just an annotation on a declaration. The examples below define one, derive another from it, and leave a dimension open for the caller to fill in.

=== UID:F16-S03
H: Defining Shapes with `mdspan`
P: The keyword `mdspan` stands for multidimensional span. It describes the extents of a shape without allocating its elements. For example, a shape `[7, 8]` has rank 2 and describes 56 positions; choosing `f32` or `s32` for those positions is a separate decision.
P: You can explicitly define `mdspan` variables as follows:
C: Declaring a shape
P: The colon introduces the shape initializer, and square brackets enclose its dimensions. Thus `s0` has seven rows and eight columns, while `s1` has one dimension of size 3. A named shape can be reused in later buffer declarations.
P: You can specify the **rank** by placing ``rank> after `mdspan`. The compiler checks that this rank matches the initialization expression and reports a compilation error if they differ:
C: A rank mismatch
P: Because the initializer identifies a shape, the compiler can infer the `mdspan` type when the keyword is omitted:
C: Omitting the mdspan keyword
P: Here `s3` has rank 3 and dimensions 7, 8, and 9. Every named shape requires an initializer, so the inferred form is often sufficient.

=== UID:F16-S04
H: Deriving *mdspan*s
P: New shapes often derive from existing ones. **Tiling** or **blocking** divides dimensions, while **padding** increases selected dimensions by a specified amount.
P: Shape arithmetic expresses these derivations directly:
C: The original shape
P: The first derived shape divides dimension 0 by 2 and dimension 1 by 4, then adds a dimension of size 1. The expression determines the new dimension order. It changes the shape description; it does not move any elements.
P: In Croqtile, the definition of `new-shape0` is equivalent to:
C: A derived shape
P: This form lists the dimensions explicitly inside `[]`. The **element-of** operation `()` retrieves a dimension from the existing shape. Both forms produce the same result; the earlier form is syntactic sugar for the complete initialization expression.
P: The second derived shape adds 2 to the original dimension 1 and divides dimension 0 by 16, then places them in that order.
P: Furthermore, you may use the `mdspan` as a whole for derivations:
C: Rebinding a shape
P: Arithmetic on a whole shape applies to each dimension. Adding 1 to `[128, 64]` therefore gives `[129, 65]`; it does not append a dimension. The two expressions below spell out that same calculation:
P: The expression `new-`shape1 : shape + 1;
P: is equivalent to
P: The expression `new-`shape1 : shape [(0) + 1, (1) + 1];
P: The derived definition of `new-shape3` demonstrates that using an *mdspan* variable in an *mdspan initialization expression* results in **concatenation** behavior. Thus, the declaration
P: The expression `new-`shape3 : [shape, 6];
P: is equivalent to
P: The expression `new-`shape3 : shape [(0), (1), 6];
P: The shorter form is syntactic sugar for the complete shape definition.

=== UID:F19-S03
H: *mdspan* with **Anonymous Dimension**
P: Croqtile supports shape dimensions with unknown values, similar to many high-level machine learning languages. The simplest method is to use **Anonymous Dimension** in `mdspan`:
C: A kernel with an unknown dimension
P: Here, the question mark `?` represents a compile-time unknown value. Any positive integer could be assumed here. However, the value must be provided at runtime, as the following example shows:
C: The caller supplying that dimension
P: The first dimension of `input` comes from the spanned input passed to `foo`; `bar` obtains the value from its caller at runtime. For example, it may come from a configuration file or a shape calculation in a machine learning framework.
P: An anonymous dimension is sufficient when only its size matters within one input. It does not give two inputs a shared name for a dimension that must agree. Matrix multiplication needs precisely that relationship: the left operand's column count must equal the right operand's row count.

=== UID:F19-S04
H: Step Further: **Symbolic Dimension**
P: In Croqtile, **symbolic dimensions** give names to dimensions whose values are unknown at compile time. For example:
C: Step Further: Symbolic Dimension
P: Each input dimension has a symbolic name: `M`, `K`, or `N`. Unlike anonymous dimensions, these names make relationships between shapes explicit, improving readability and maintenance. They also give the compiler more information for consistency checks.

=== UID:F15-S05
H: Scalar Types
P: Croqtile provides **integer** and **Boolean** scalar types. Its integer type is a signed 32-bit value, like C++ `int` or `int32_t`, with range $-2^31$ to $2^31-1$. The following example uses it in variable and function declarations.
C: Scalar Types
P: Integers support arithmetic, shifts, and logical operations with C++ syntax. Croqtile does not provide unsigned or 8-, 16-, or 64-bit scalar control integers: a signed 32-bit integer is sufficient for the program-control tasks this type serves.
P: The Boolean type corresponds to C++ `bool`. Boolean operations and conversions between Boolean and integer values follow C++ rules.

=== UID:F15-S10
H: I-Tuple Types
P: An integer tuple is an ordered collection of integers, commonly used as a multidimensional index.
P: To define an i-tuple, simply enclose its elements within '' and '' braces. That is,
C: I-Tuple Types

=== UID:F15-S11
H: Operations over *mdspan* and *i-tuple*
P: Croqtile supports operations between `ituple` and `mdspan` values. The following example applies fixed tiling factors to a shape:
C: Operations over mdspan and i-tuple
P: In Croqtile, mdspan can be defined with such **Tuple-Span Operations**. The supported operations include:
L: *mdspan* $/$ *i-tuple* | *mdspan* $+$ *i-tuple* | *mdspan* $%$ *i-tuple* | *mdspan* $*$ *i-tuple* | *mdspan* $-$ *i-tuple*
P: Each operation applies to corresponding dimensions. Tuple-span notation collects those calculations into one expression, so the relationship between a shape and its tiling factors remains visible.

=== UID:F17-S01
P: Integers in a kernel control loops and index arrays, and an *i-tuple* groups several of them into one index. Both are declared with an initializer and both are checked against the shape they are used with.

=== UID:F17-S02
H: Integers for Program Control
P: Control integers describe loop ranges, indices, and related calculations. Croqtile uses a signed 32-bit `int` for these tasks. Numerical data stored in a tensor uses the separate fundamental element types described in Section 2.6.
P: In Croqtile, an **integer**-typed variable is defined similar to C/C++:
C: An initialized integer
P: Croqtile can also infer the type from the *initialization expression*, so an integer variable can be declared in another style:
C: Assignment to an integer
P: Similar to C/C++, you can define multiple integers in a single line:
C: Several integers at once
P: In Croqtile, the *mdspan*-typed shape cannot be used as the function parameter. However, integers can be passed from the host to the Tileflow program, and from the Tileflow program to the device:
C: An integer parameter
P: Unlike C/C++, Croqtile does not allow the declaration of integers without initialization or the reassignment of an integer. For example:
C: A declaration without initialization
P: This enforces that **integers must be initialized at declaration and cannot be reassigned**. In some contexts, these integers are referred to as "**immutable**" or "**constant**".

=== UID:F17-S04

=== UID:F17-S05
H: Defining I-Tuples
P: An integer tuple, written `ituple`, groups several integer values in a fixed order. Its initializer uses braces:
C: An i-tuple literal
P: Since Croqtile can infer types from the *initialization expression*, programmers can often omit the `ituple` keyword. However, without explicit type annotation, an *ituple* variable definition might look similar to an *mdspan* definition if you are not yet familiar with Croqtile. To distinguish them:
L: The *initialization expression* of an *ituple* follows an assignment operation `=` (like integers), whereas an *mdspan* is initialized after `:`; | The *initialization expression* of an *ituple* is enclosed by `\`, not `[]` as for *mdspan*.
P: Similar to `mdspan`, you may enforce rank check for *ituple*s at compile time:
C: An inconsistent rank

=== UID:F17-S06
H: Operations on I-Tuples
P: Operations on *i-tuple*s are similar to those on *mdspan*. You can either use the *element-of* operation `()` to retrieve the element values or use *ituple* as a whole:
C: Assigning a new tuple
P: Similar to *integer* and *mdspan*, an *ituple* variable must be initialized and cannot be reassigned.
P: In practice, *ituple*s are often used with *mdspan*s. For example:
C: Indexing an mdspan with an i-tuple
P: Here, dividing an `mdspan` shape by an `ituple` of tiling factors produces `tiled_shape`. Adding an anonymous `ituple` to the original shape produces `padded_shape`. The shape and tuple must have matching ranks; otherwise, the compiler reports an error.
C: An invalid shape expression

=== UID:F17-S09
H: In short
P: Use an integer for one control value and an `ituple` for an ordered group of values. A shape such as `[7, 14]` describes ranges, while a tuple such as `1, 2` can describe a position or a pair of tiling factors. The matching ranks let shape arithmetic apply each factor to the corresponding dimension.
P: Both integers and ordinary tuples must be initialized and cannot be reassigned. This restriction concerns the names used for control and shape calculations. A buffer element is different: the addition example writes to `output.at(i, j)`, and a matrix multiplication updates its accumulator. Keeping these roles distinct avoids treating an immutable index as though it were a mutable accumulator.

=== UID:F15-S08
H: Fully-Typing
P: A buffer declaration needs both an element type and a shape. The shape determines how many positions the buffer contains, while the fundamental type determines what each position stores. The examples below reuse a named shape for half-precision data and use a literal shape for single-precision data.
C: Complete Types
P: The basic element types include:
L: Unsigned Integers: *u8/u16/u32* | Signed Integers: *s8/s16/s32* | Floating-points: *f16/bf16/f32*
P: The types `s32` and `int` serve different purposes in Croqtile. The fundamental element type `s32` must be combined with a shape to declare a fully typed buffer.

=== UID:F15-S09
H: The Storage Qualifier
P: A tile may be shared by cooperating threads or used privately by one thread. Its element type and shape do not express that distinction; a storage specifier does.
P: The three storage specifiers used in these examples are:
L: **global**, | **shared**, | and **local**.
P: For example:
C: The Storage Qualifier
P: Choose the storage specifier to match the code that uses the buffer. Defaults depend on the execution context and target; use an explicit specifier when a transfer or a cooperating group requires a particular storage location.

=== UID:F18-S01
P: A Croqtile Tileflow program describes data movement, so buffer declarations are fundamental. Each declaration specifies an element type, a shape, and, when needed, a storage location.

=== UID:F18-S03
H: Defining a *Spanned Data*
P: Defining a *spanned data* requires both the element type and its shape. For example,
C: A spanned-data declaration
P: This defines a buffer with a shape of `[32, 16]`, where the element type is `f32`, which is the *IEEE-754* single-precision float. It is also possible to use a named `mdspan` to define the buffer:
C: A standalone shape
P: This example uses the named `mdspan` `shape` to declare the two-dimensional buffer `data0` with 32-bit integer elements. Place the shape inside `[]`; element access, arithmetic, and concatenation can also appear in the shape expression.
P: In Croqtile terminology, the element type is referred to as the **fundamental type**. Croqtile supports multiple fundamental types, including:
L: **Unsigned Integers**: `u8`, `u16`, `u32` | **Signed Integers**: `s8`, `s16`, `s32` | **Floating Points**: `f16`, `bf16`, `f32`
P: The prefixes `u` and `s` mean unsigned and signed, and the numeric suffix gives the bit width. The type `f16` is IEEE 754 half precision; `bf16` is the 16-bit brain floating-point format commonly used in machine learning.
P: The element type `s32` and the control type `int` have different roles, even though both describe 32-bit integers. An `s32` declaration needs a shape. The following standalone declaration is rejected:
C: A rejected fundamental type
P: This results in a compile-time failure. Consequently, `s32` cannot be used for program control like `int`, and type conversion between them is not possible.

=== UID:F18-S04
P: In practical code, some buffer declarations must specify a **storage specifier**, like so:
C: Storage Specifier
P: Since Croqtile handles storage in a heterogeneous context, a buffer definition without a storage specifier defaults to the storage type of the host program, i.e., CPU memory. Other storage specifiers are defined by the target.
L: *global*: Refers to the device's global storage. | *shared*: Refers to the device block's shared storage | *local*: Refers to the thread-private storage.
P: Other targets may have different definitions. Furthermore, buffers with different storage types have limitations in their declarations. You will explore this in later chapters.

=== UID:F18-S05
H: Initialization
P: It is possible to initialize a buffer at the declaration site. For example:
C: Initialization
P: The braces after the buffer name contain one initial value, applied to every element. This form fills the whole buffer; it is not an element-by-element initializer list.

=== UID:F18-S06
H: Declaring the Parameters and Return Values
P: Spanned data can appear in the parameter and return declarations of a Croqtile function:
C: An explicit return shape
P: Parameter and return declarations resemble variable declarations, but they do not allow operations on an existing `mdspan`, storage specifiers, or initializers.
P: One useful built-in member function for the *spanned* parameters is `.span`, which provides the associated *mdspan* of the *spanned data*. The following code demonstrates how to use it for a buffer declaration:
C: A deduced return shape

=== UID:F18-S08
H: In short
P: A declaration determines the element type and shape, an initializer fills the buffer, and a storage specifier selects its location. The compiler manages the buffer's lifetime. Dynamic shapes extend these declarations to dimensions supplied at runtime.

=== UID:F37-S04
H: 1. `global`
P: A `global` declaration places the buffer in the target's global storage. It is useful for data that must remain accessible beyond one thread or one cooperating block. Declaring the storage does not initialize its contents or copy data into it:
C: 1. global

=== UID:F37-S05
H: 2. `shared`
P: A `shared` buffer provides a common staging area for threads in a block. In a tiled matrix multiplication, a block can load an operand tile once and let several threads read it. The reads must follow completion of the load, and the next load must wait until those reads finish.
C: 2. shared

=== UID:F37-S06
H: 3. `local`
P: It is particularly useful when each thread requires a small amount of data, such as scalar values or small arrays that do not need to be shared with other threads. Example:
C: 3. local

=== UID:F37-S07
H: Default Behavior and Usage
P: Omitting the specifier leaves storage selection to the target and the surrounding execution context. This is convenient for introductory code, but a name without a specifier does not by itself tell you which threads can share its contents. When following a data path, check the declaration together with the parallel region that contains it.
C: A declaration without a specifier
P: If you want to place data in a different memory space (shared or local), you need to specify the appropriate storage qualifier.
C: Choosing local storage explicitly

=== UID:F37-S08
H: Example Code: Using Storage Qualifiers
P: The following sketch combines the storage declarations:
C: Example Code: Using Storage Qualifiers
P: The declarations illustrate three storage locations. The copy lines sketch the direction of movement between them. To turn this sketch into a working transfer, select source and destination views with compatible shapes: the full arrays declared here have different sizes. A storage specifier chooses a location; it does not make differently shaped buffers interchangeable.

=== UID:F05-S03
H: Croqtile's Two-Layer Parallelism Model
P: Croqtile gives you two separate knobs:
L: **`parallel`** --- declares that iterations are independent and *may* run concurrently. This is a logical statement. | **Space specifiers** (`: block`, `: thread`, `: group`, `: group-4`) --- tell the compiler where each parallel axis maps. This is a physical statement.
P: There is also `foreach`, which is the opposite of `parallel`: iterations run **sequentially**, in order. Use `foreach` when later iterations depend on earlier ones --- an accumulation loop, a K-dimension reduction, a pipeline that loads tile *k* before computing with it.
C: Croqtile's Two-Layer Parallelism Model
F: Space specifiers map parallel axes to blocks and threads; foreach loops remain sequential within that structure.
P: If you omit the space specifier, as in `parallel p by 8`, the compiler chooses a default mapping. The introductory examples use this form. Explicit specifiers let you control the mapping when performance requires it.
P: Both `parallel` and `foreach` can use multidimensional indices. Once the outer and inner ranges are established, the compose operator `#` connects their coordinates to positions in the full tensor. The bounds are part of that calculation, which is why changing the iteration geometry also changes tile selection.

=== UID:F05-S05
H: `: block` --- parallel blocks
C: : block --- parallel blocks
P: This declares an 8 by 16 grid, giving 128 blocks. Each block must be able to complete its assigned output tile independently; the example does not provide a communication or synchronization protocol between blocks.
P: Use `: block` for the outermost tiling of your output. If your output is `[128, 256]` and each block handles a `[16, 16]` tile, you need `128/`16 x 256/16 = 8 x 16 = 128 blocks.

=== UID:F05-S06
H: `: thread` Threads Within a Block
C: : thread --- Threads Within a Block
P: This creates 16 $×$ 16 = 256 threads within the enclosing block.
P: Use `: thread` for the finest-grain parallelism --- the per-element work within a tile.

=== UID:F05-S09
H: Nesting Specifiers
P: Real kernels nest these levels. A common pattern:
C: Nesting Specifiers
P: The outer `parallel` creates the block grid. The inner `parallel` creates threads within each block. The braces `px, py\` introduce multi-dimensional indices --- shorthand for two nested `parallel` lines.

=== UID:F08-S05
H: `with tile_`k in 16
P: The statement `with tile_k in 16` gives `tile_k` the range from 0 through 15, with upper bound `#tile_k = 16`. It does not run the body 16 times. A later `foreach tile_k` performs that iteration. Separating the binding from the loop lets a pipeline select tile 0 before the loop and then iterate over the remaining tiles.

=== UID:F08-S07
H: `foreach tile_`k(1:): sliced iteration
P: The range expression `(1:)` starts iteration at index 1 and keeps the original exclusive upper bound. With 16 tiles, it visits 1 through 15. This is useful after a prologue has loaded tile 0: the loop issues each remaining load once, while computation consumes the previously loaded tile. The final loaded tile is handled after the loop.

=== UID:F15-S14
H: Parallel Region: the 'parallel-by' Block
P: Systems such as CPUs can use asynchronous threads for parallel execution. Croqtile expresses parallel work with the Single Program, Multiple Data (SPMD) model.
P: The syntax of constructing a parallel region is quite different. It employs the C-style bracket and encloses the code for parallel execution within the 'parallel-by' block.
C: Parallel Region: the parallel-by Block
P: The `parallel` and `by` keywords declare six execution instances. Each runs the same SPMD code with a different value of `p`. You can think of `p` as the index of a six-iteration loop whose iterations are independent and may execute in any order.
P: The declaration also associates `p` with the range `[0, 6)`, making it a **bounded integer**. Operations such as `chunkat` use both its current value and its bound.

=== UID:F15-S15
H: The 'with-in' Block and 'where' Clause
P: A `with-in` statement binds indices to a range without launching parallel instances. For a two-dimensional range, it can bind one tuple:
C: Indexing a with-in block
P: Here, `index` is an `ituple` with two elements. You can also name the elements individually:
C: A tuple of indices
P: The tuple and its components can also be named together:
C: Indices derived from a tuple
P: You name 'index' as a **bounded ituple** in such scenarios.
P: A `with-in` statement creates bounded indices but does not introduce parallel execution. Its body executes sequentially unless it contains another construct that introduces parallel work.
P: In addition, programmers could append a 'where' clause. For example,
C: Nested bounded groups
P: The relation `n `-> n_p requires the two indices to have the same value in every iteration. Within the `with-in` block, either name can be used wherever the other is required. Such bindings are useful when expressing AI kernels.

=== UID:F15-S16
H: The 'foreach' Block
P: Once the *bounded-ituple* is defined by the 'with-in' clause, programmers can loop over the bounded-ituples/bounded-integers. In Croqtile, this is simple.
C: The foreach Block
P: Each iteration gives the bounded variable a current value for the duration of the loop body. The bound stays fixed. Use that current value to select data, and use the bound when deriving how many pieces the data is divided into. Declaring a parallel index instead would assign one value to each independent instance rather than visit the values in order.

=== UID:F31-S03
H: Basic Syntax: `parallel-by`
P: The basic syntax for a `parallel-by` block is as follows:
C: A named parallel index
P: The braces enclose the body executed by each parallel instance. In this example:
L: **`parallel`**: This keyword initiates the *parallel execution block*. | **`p`**: It is the **parallel variable**. It contains an integer value identifying the current *parallel thread*. Each *parallel thread* runs the same code but with a different value of *parallel variable* `p`. | **`by 6`**: It indicates that there are 6 *parallel thread*s. Thus, the *parallel variable* `p` ranges from `0` to `5` across different threads.
P: In some scenarios, programmers may not require an explicit *parallel variable* for a simple parallel construct. It is possible to omit it:
C: An unnamed parallel index
P: This creates two execution instances. Without a named index, the code cannot use that index to select different data for each instance, as SPMD programs commonly need to do.

=== UID:F31-S04
H: Multiple-Level Parallelism
P: In Croqtile, you can define multiple levels of parallelism. Here's an example:
C: One parallel level
P: This code defines two levels of parallelism with values `2` and `12`, respectively. This means a total of `2 x 12 = 24` parallel threads are invoked for the task. Each thread is identified by a unique combination of `p` and `q`. In Croqtile, you refer to the level of `p` as *parallel-level-0* and the level of `q` as *parallel-level-1*.
P: The nesting determines which instances belong to the same enclosing region. Shared data and synchronization must follow that structure. Put cooperating work at a level whose instances can use the required storage and synchronization operations.
P: In Croqtile syntax, you can define multiple-level parallelism in a single line:
C: Two parallel levels
P: This comma-separated form expresses the same two parallel levels as the nested version, provided there is no intervening code between those levels.

=== UID:F31-S05
H: Sub-Level Parallelism
P: Some targets allow a single parallel level to be divided into up to three coordinate axes. Croqtile expresses these axes using `ituple` and `mdspan`:
C: Sub-Level Parallelism
P: In this example, two levels of parallelism are defined. The first level (*level-0*) is subdivided into three sub-levels, annotated by the `px, py, pz\` `ituple`. The second level (*level-1*) is subdivided into two sub-levels, annotated by the `qx, qy\` `ituple`, or `t_index` as a whole. For each sub-level, the parallelization count corresponds to the respective value inside the `mdspan`.
P: The first element inside the `ituple` represents the **least significant parallel variable**. Croqtile adopts this ordering to align with developers' conventions, rather than for internal consistency.

=== UID:F32-S01
P: A `with-in` statement declares a *bounded variable*; a `foreach` statement iterates over its range. Together they describe a loop without spelling out its bounds in C++.

=== UID:F32-S02
H: Loops in Croqtile
P: In C++, loops are typically constructed using `for`, `while`, and `do-while` blocks. These loops include conditions for termination and usually involve variable manipulations related to loop control. Programmers have the freedom to define exit conditions, loop stepping, and other aspects, allowing for flexible loop structures.
P: Croqtile uses bounded ranges for these loops. The range is defined once and then visited by `foreach`:
C: A foreach loop
P: Here, `index` is a *bounded variable* (introduced in the previous section). If `index` has an upper bound of `6`, the above code is equivalent to:
C: The equivalent C++ loop
P: This form suits tile traversal: the loop visits the valid coordinates in a declared range, and the same bounds can be used to derive each tile's shape.
P: In Croqtile, *bounded variables* defined within a `parallel-by` statement are immutable and cannot be used in `foreach` statements. Instead, programmers should use the `with-in` statement to define the *bounded variables* suitable for loops.

=== UID:F32-S03

=== UID:F32-S04
H: Defining *Bounded Variable*s:
P: A `with-in` statement allows you to define either a *bounded integer* or a *bounded ituple*. Syntactically, it resembles the `parallel-by` statement but only defines *bounded variable*s without invoking multiple instances for execution. Below are examples:
C: A bounded variable
P: Here, `x` represents a bounded integer with an upper bound of `128`. Both `y` and `index` represent bounded ituples whose upper bounds are `512` and `10, 10` respectively. Similar to the `parallel-by` statement, you can name the elements of the ituple for clarity or declare both the bounded ituple and the associated bounded integers, as shown below:
C: A tuple of bounded variables
P: Like the `parallel-by` statement, multiple `with-in` declarations can be combined using a comma-separated syntax, as demonstrated below:
C: A variable derived from a tuple
P: In this way, two bounded ituples are defined. Note that either `index` or `idx` can only be referenced within the following `with-in` block.

=== UID:F32-S05

=== UID:F32-S06
P: The `with-in` statement defines *bounded variable*s, which can then be iterated over using the `foreach` statement. The basic syntax is as follows:
C: Indexing one shape
P: In this example, the `foreach` block iterates 6 times, with the value of the **iteration variable** `index` ranging from `0` to `5` incrementally.
P: It is also possible to iterate over *bounded ituple*s. For example:
C: A tuple of indices
P: This code defines a *bounded ituple* `index`. Iterating over `index` is equivalent to a nested loop structure:
C: The generated C++ loop
P: The left-to-right order of indices in a bounded tuple determines the outer-to-inner nesting of the loops. Keep this ordering in mind when the order of iteration matters.
P: The same outer-to-inner ordering applies to a comma-separated list of bounded variables:
C: Several bounded groups
P: In this code, two bounded variables are defined. The following `foreach` block is equivalent to a multi-level nested loop:
C: The generated inner loop

=== UID:F32-S07
H: Syntactic Sugar
C: A foreach loop
P: The code above is equivalent to the following code:
C: The with-in form it expands to

=== UID:F32-S08
H: Deriving the Loop From a Bounded Integer
P: In certain scenarios, such as pipelining data movement, it may be necessary to modify loop iterations. In Croqtile, this can be achieved by deriving a loop from a *bounded integer* within the `foreach` statement. For example:
C: A tuple of bounded variables
P: In this case, the **Range Expression** `y(1::)` is used to derive the loop. This results in equivalent C/C++ code:
C: The derived C++ loop
P: The `y` loop starts at `1`. A range expression consists of a bounded variable followed by three colon-separated integer fields in parentheses:
C: The bounded-variable syntax
P: This derives a loop from the `bounded-`variable, where the `bounded-`variable serves as the *iteration variable* of the loop. Specifically:
L: The initial value of the *iteration variable* is the `lower-offset` plus the lower bound of the `bounded-`variable. Since all bounded integers have a lower bound of `0` in Croqtile, the `lower-offset` sets the initial value of the loop. | The loop terminates when the *iteration variable* is equal to or greater than the upper bound of the `bounded-`variable plus `upper-offset`. Negative values are typically used for `upper-offset`. | The *iteration variable* increments by `stride` at the end of each iteration.
P: Thus, `y(1:-1:2)` results in a loop like `for (y = 0 + 1; y ` 17 - 1; y += 2) in the example above. If any field of the *range expression* is not specified, it results in default values: `0` for `lower-offset` and `upper-offset`, and `1` for `stride`.
P: Note that *range expression*s only apply to the *bounded variable*s. *Range expression* over the *bounded ituple* triggers an error at compile time.

=== UID:F32-S10
H: In short
P: Bounded variables define iteration ranges, and range expressions adjust the starting point, ending point, and stride. Reading these loops requires distinguishing a variable's current value from its upper bound.
P: Deriving a loop this way is useful for multi-buffered data movement, where the loop may skip a tile loaded before iteration begins. Section 3.6.2 develops this pattern.

=== UID:F04-S05
H: `chunkat(...)`: equal partitioning
C: chunkat(...): equal partitioning
P: The `chunkat` expression divides each dimension according to the corresponding index bound. With input shape `[64, 128]` and `parallel tr, tc by [4, 8]`, it selects a `[16, 16]` tile. The current values of `tr` and `tc` select which tile, while their bounds determine its size.
P: That makes `chunkat` the natural tool when your tiling follows the parallel decomposition directly.
F: chunkat(1, 3) selects rows 16–31 and columns 48–63 from a [64, 128] tensor partitioned into [4, 8] tiles.

=== UID:F04-S06
H: `subspan(.`..).at(...): explicit tile extents on a tile grid
C: subspan(...).at(...): explicit tile extents on a tile grid
P: The `subspan` expression supplies the tile extents explicitly, and `.at(...)` supplies tile coordinates. For a `[16, 16]` tile, coordinate `(2, 3)` begins at element `(32, 48)`. This form is useful when the tile size is a fixed algorithmic choice rather than derived from the number of parallel instances.

=== UID:F04-S07
H: `view(.`..).from(...): explicit tile extents with explicit element offsets
C: view(...).from(...): explicit tile extents with explicit element offsets
P: The form `view(...).from(...)` gives a window an explicit shape and an origin measured in elements. A `[16, 16]` window starting at `(37, 50)` keeps that exact origin; it does not round it to the nearest tile boundary.
P: This is the right tool when the tile origin is already available as element offsets rather than tile indices, or when the region is not most naturally described as an equal chunk. It is especially useful for dynamic and irregular access patterns where `chunkat(...)` would be awkward or impossible.
P: You can think of the three forms like this:
L: Use `chunkat(...)` when the surrounding parallel decomposition already defines an equal tiling. | Use `subspan(.`..).at(...) when you want an explicit tile shape and tile coordinates. | Use `view(.`..).from(...) when you want an explicit tile shape and element-space starting offsets.

=== UID:F04-S14
H: Compose: `outer # inner`
C: Compose: outer inner
P: This combines a tile coordinate with an in-tile coordinate to produce a coordinate in the full tensor.
P: For an outer tile index `tr` and an inner index `i`, `tr # i` has value `tr * #i + i`. If a tile has 16 rows, tile 2 and inner row 5 identify global row 37. Its bound is the product of the two bounds. Operand order matters because the inner bound supplies the stride.
F: With 16 elements per tile, tile = 2 and i = 3 compose to global position 2 × 16 + 3 = 35.

=== UID:F04-S15
H: Extent: `#name`
C: Extent: name
P: The notation `#tile` means the extent of the `tile` axis: how many tiles exist along that dimension. If `parallel tile by 8`, then `#tile` is `8`.
F: Prefix #tile gives the axis extent, 8; infix tile # i composes a tile coordinate and an in-tile offset.

=== UID:F15-S19
H: Bounded-ituple/integer and 'chunkat' Operation
P: A `chunkat` expression selects a view of existing spanned data. It uses each bounded argument twice: the upper bound determines the number of chunks, and the current value selects one chunk. Selection alone does not move the elements; the surrounding DMA statement performs the transfer.
C: Bounded-ituple/integer and chunkat Operation
P: The example above shows one typical use of `chunkat`. Here the spanned data has type `f32 [6, 10, 100]`. The operation takes two parameters --- the *integer* `p` and the *ituple* `index` --- and divides the data into $6 × 10 × 10$ pieces, one for each combination of their ranges. Each piece holds 10 consecutive elements along the least significant dimension and is copied into local storage.

=== UID:F29-S01
P: A *chunkat expression* names one tile of a larger buffer in a DMA statement, so the movement code never has to compute an offset by hand.

=== UID:F29-S02
H: *Ubound* Operation
P: A bounded variable declared by `parallel-by` or `with-in` is an integer or tuple with associated bounds. A `foreach` loop uses its current value as the index and its upper bound to determine the iteration range.
P: In Croqtile, a program can use the **ubound operation** to explicitly retrieve the upper bound of a bounded variable. *Ubound* is a unary operation that uses the prefix symbol `#` before a bounded variable. For example:
C: Ubound Operation
P: In this code, the local buffer `a` is shaped as `[32, 16]`, since `#size` evaluates to the upper-bound of `size`. However, the declaration of `b` uses the *current value* of `size`, resulting in an incorrect shape definition.

=== UID:F29-S03
H: Use `chunkat` Expression to Tile Data
P: A transfer can use a whole buffer, but a tiled algorithm usually needs only the portion assigned to the current instance. Put `chunkat` in the source expression to describe that portion directly:
C: Use chunkat Expression to Tile Data
P: In this example, the built-in member function `.chunkat` is used to create a tile or block for the source expression of the DMA statement. Each tiled block is called a **chunk** in Croqtile. The `.chunkat(...)` data expression can be considered a simpler form of `.`chunk(...).at(...), where the semantics of each part are illustrated in the figure below:
F: f29-chunkat.png
P: Considering the expression `input.`chunkat(p), which is equivalent to `input.`chunk(#p).at(p):
L: `input` is the *spanned data* to be tiled. | `#p` indicates the **tiling factor**. It also specifies the total **chunk count** for each dimension to process. In this case, the 1D input is tiled by `6`, resulting in `6` chunks to be processed. | `.chunk(#p)` determines the **chunk shape**. The chunk is a 1D data segment tiled from `input`, with its size being the data *mdspan* divided by the *upper-bound* of the bounded variable. In this example, the chunk shape is calculated as `[input.`span / #p] = [6]. | `.at(p)` determines the **chunk index**. Here, `p` ranges from `[0, 6)`, allowing parallel threads to process different chunks of `input`. Specifically, parallel thread `1` processes a chunk with elements `[6, 12)` from `input`, while parallel thread `5` processes a chunk with elements `[30, 36)`, and so on.
P: In many scenarios, it is unnecessary to use separate `.chunk` and `.at` expressions, as `.chunkat` takes bounded variables as parameters. The upper-bounds of these variables act as the *tiling factors*, and their current values act as the *chunk index*.

=== UID:F29-S04
H: Use Multiple Bounded Variables
P: The example below uses several bounded variables at once:
C: Use Multiple Bounded Variables
P: In this example, there are two bounded variables: `p` is the *bounded integer* defined by the `parallel-by` statement, while `index` is the *bounded ituple* with two integers. The `chunkat` expression concatenates them to form a *bounded ituple* with upper-bounds `#`p, #index and current values `p, index\`. Thus, the shape of each chunk is calculated as `[36, 14, 8] /` #p, #index = [36, 14, 8] / 6, 7, 4 = [6, 2, 2].
P: In addition, since `p` is the bounded variable for parallelization, there are `6` threads working in parallel, each handling a large chunk of `[6, 14, 8]`. Within a single parallel thread, it iterates `7x4` times, each time moving a `[6, 2, 2]` small chunk to a compiler-allocated `shared` storage.
P: The same bounds now describe both the iteration space and the selected data. If the partition changes, the compiler derives the new chunk dimensions from those bounds. To review the transfer, check the input shape, divide by the bounds, and then multiply the chunk coordinates by the resulting chunk extents to recover the element origin.

=== UID:F29-S06
H: In short
P: The `chunkat` expression divides a buffer into smaller chunks for a DMA statement, and the shape of each chunk follows from the bound used to select it. You explored how `chunkat` can be applied to either the source or destination of a DMA statement to manage data movement between memory and processors.
P: Section 3.6.1 combines these constructs in a complete Tileflow program.

=== UID:F04-S02
H: The DMA Statement Family
P: All of Croqtile's tile-movement operations use the same basic form:
C: The general DMA form
P: The important pieces are:
T: | **Part** | **Meaning** | <ROW> | --- | --- | <ROW> | `future` | Optional handle to the transfer result. Use `future.data` to access the moved tile and `future.span` to query its shape. | <ROW> | `operation` | `copy`, `transp`...>, or `pad`...>. | <ROW> | `modifiers` | Optional flags such as `.async`, `.zfill`, or `.`swiz<N>. | <ROW> | `source =` destination | The source view and the destination memory or destination view. |
P: Minimal examples:
C: A concrete copy
P: The key shift from 1.3 is that these statements move shaped regions, not individual elements.

=== UID:F04-S03
H: First Example: Tiled Addition Through shared storage
C: Tiled Addition Through Shared Storage
P: The arithmetic is still element-wise addition. What changed is the data path:
L: `chunkat(tr, tc)` selects one logical tile from each input. | `dma.`copy ... => shared stages that tile into block-visible fast memory. | `tr # i` and `tc # j` map tile-local coordinates back to global output coordinates.
P: This is the core Croqtile pattern: select a tile, move it, compute on the moved tile.
F: For [64, 128] inputs and a [4, 8] tile grid, tile (1, 3) stages [16, 16] operands before element-wise addition.

=== UID:F04-S08
H: Futures, `.data`, and `.span`
P: The result of a DMA or TMA statement is a future-like handle:
C: A future from a copy
P: That handle gives you two important things:
L: `f.data`: the moved tile, as a spanned tensor. | `f.span`: the shape of that tile.
P: Example:
C: Reading data and span
P: Even synchronous copies are worth naming when you want to pass the moved tile to later code.
F: The copy result exposes the destination tensor through .data and its shape through .span; asynchronous reads require completion first.

=== UID:F04-S09
H: Synchronous and Asynchronous Copies
P: By default, copies are synchronous:
C: A synchronous copy
P: Add `.async` when you want the transfer to proceed in the background:
C: An asynchronous copy
P: The same pattern applies to TMA:
C: An asynchronous tma copy
P: A future identifies the destination before an asynchronous transfer necessarily finishes. Access through `f.data` does not mean the elements are ready. Place `wait f` before the first operation that reads those elements, and keep the source and destination valid while the transfer is in flight.
P: Later chapters use async copies to overlap load and compute. In this part of the guide, the important part is the contract: `.async` creates an in-flight transfer, and `wait` makes it safe to consume.

=== UID:F04-S11
H: `dma.`transp<...>
C: dma.transp ...
P: This permutes dimensions during the movement itself. You do not first copy and then transpose. The transfer produces data in the transposed layout.

=== UID:F04-S12
H: `dma.`pad<...>
C: dma.pad ...
P: Padding creates a larger destination tile. The template arguments describe low padding, high padding, internal stride-style padding, and the fill value. Use `pad` when you want explicit border values, not just out-of-bounds protection.

=== UID:F14-S08
H: Tileflow Program: Orchestrating the Data Movement
P: The *Tileflow Program* consists of *Croqtile functions*. As described earlier, it manages the movement of data between the host and the target device, ensuring that data is copied correctly across different storage locations.
P: For convenience, let's revisit the code:
C: The tileflow program
P: The `__co__` function accepts `lhs` and `rhs`, both with shape `[6, 17, 128]` and element type `s32` (signed 32-bit integer). The output has the same shape and element type.
P: The `parallel p by 6 .`.. block indicates that the enclosed code runs in parallel. Specifically, six instances of the code are executed concurrently. This implies a transition in the execution environment from the host to the device.
P: Inside the `parallel-by` block, `with-in` binds `index` to the bounds `[17, 4]`. This two-dimensional bounded tuple supplies the indices for the `foreach` loop.
P: The `foreach index .`.. statement is equivalent to the following C code:
C: The equivalent C++ loop
P: Within the `foreach` block, the `dma.copy` statement describes how data movement occurs. For example, consider the statement `lhs_`load = dma.copy lhs.chunkat(p, index) => local;:
L: The symbol `lhs_load` is called **future** of the DMA operation, which contains information about the DMA destination. | `dma.copy` invokes a direct DMA data transfer without transforming the shape of the data. The expression on the left-hand side of `=` represents the DMA source, while the right-hand side represents the destination. | In this case, the destination is specified as a `local` buffer, which is automatically allocated by the Croqtile compiler. | The source expression `lhs.`chunkat(p, index) selects a chunk of `lhs`. The bounds of `p` and `index`, namely `[6, 17, 4]`, determine the partition. Dividing the input shape `[6, 17, 128]` by these bounds gives chunk shape `[1, 1, 32]`. The current values select the chunk: instance `p = 1` and iteration `index = 16, 2\` select chunk `(1, 16, 2)`, whose element origin is `(1, 16, 64)`.
P: The following figure shows how the bounds determine the chunk shape and the indices select a chunk.
F: Dividing shape [6, 17, 128] by bounds [6, 17, 4] gives chunks of shape [1, 1, 32]. Chunk (1, 16, 2) starts at element (1, 16, 64).
P: With the DMA statement, different chunks of data tiled from `lhs` are moved iteratively and in parallel from the host to the device's `local` storage. Similarly, the DMA statement manages `rhs` by moving it in small chunks to `local` storage for processing.
P: The declaration `local s32 [lhs_`load.span] l1_out; creates a buffer for each parallel instance, with the same shape as the loaded chunk. The subsequent `call` invokes the device function `kernel`, which writes its result to that buffer. A final DMA statement copies the result back to the host output, completing the iteration.
P: In this code, each parallel thread runs `17x4` iterations, with each iteration handling a `1x1x32`-sized chunk of data. The *croqtile program* terminates when all `6` parallel threads have completed their iterations. It then returns the output buffer to its caller, the host program.

=== UID:F28-S01
P: The DMA statement describes data movement and can name its result with a future. Its syntax specifies the operation, source, and destination.

=== UID:F28-S03
P: A DMA statement can execute asynchronously, allowing subsequent code to run while the transfer is in progress. An asynchronous transfer requires explicit synchronization before its result is used.
P: To support these features, the syntax is organized as follows:
C: Basic Syntax
P: The name on the left of `=` denotes a future. It identifies the transfer result and, for an asynchronous operation, provides the handle used by `wait`. A synchronous transfer may omit the name when later code does not need a handle to its destination.
P: The right-hand side of the **DMA operation** includes the **operation type** (`dma-op`), the **data expression** for the operation source (`src-expr`), and the *data expression* for the destination (`dst-expr`). The source and destination are separated by the symbol `=`, which indicates the direction of data flow.

=== UID:F28-S04
H: Synchronous and Asynchronous
P: An asynchronous copy is non-blocking: subsequent code can execute while the transfer is pending. A synchronous copy completes before execution proceeds past the statement. These terms describe when the issuing code can continue, not whether it may safely read an unfinished transfer's destination.
P: In Croqtile, the `.async` suffix in a **DMA operation** indicates that the operation is asynchronous. For example:
C: Synchronous and Asynchronous
P: In this example, the DMA operation that produces the future `f0` is asynchronous, allowing the program to continue executing while the data transfer occurs in the background. The DMA operation that produces `f1` is synchronous, meaning the program will pause until the data is fully transferred from `data1` to `shared` storage. Meanwhile, the data copy of `data0` might still be in progress.
P: The statement `wait f0;` waits until the transfer from `data0` to `shared` storage is complete, synchronizing subsequent execution with that transfer. Ensure that every asynchronous result is awaited before it is used.

=== UID:F28-S06
H: Data Expression
P: The data expressions used so far have been simple, either a defined **spanned data** or a **storage location**. The **storage location** is used solely for the **destination buffer** declaration, as it requires the Croqtile compiler to allocate memory for it. For example:
C: Data Expression
P: The second DMA statement names `shared` as its destination, so the compiler allocates the buffer with the inferred shape. It is equivalent to the first statement's explicit buffer declaration. Naming a storage location is convenient when the compiler can infer the destination shape.
P: The same source and destination positions can contain tile expressions. For example, `input.chunkat(p)` restricts the copy to the portion selected by `p`; an explicit destination view restricts where the result is written. Section 2.8 explains how those views are formed.

=== UID:F28-S07
H: Future and Wait
P: Both *Sync-DMA* and *Async-DMA* operations can be used to define a *future*. As seen in earlier examples, a *future* acts as a handle for asynchronous operations and is typically used as a parameter of the `wait` statement to synchronize an *Async-DMA*.
P: Croqtile allows the `wait` statement to take multiple *future*s. However, using a *future* from a **Sync-DMA** in a `wait` statement will result in an error. For example:
C: Two asynchronous copies
P: In Croqtile, another use of a *future* is to retrieve the destination buffer of the associated DMA statement. This is also the reason why **Sync-DMA** is allowed to define a future. Croqtile provides two built-in member functions for *future* variables:
L: `.span` to retrieve the **mdspan** of the DMA destination buffer. | `.data` to retrieve the reference of the DMA destination buffer (spanned data).
P: For example:
C: Waiting on both futures
P: Here, `.span` retrieves the shape of the destination buffer associated with `f0`, and `.data` retrieves the destination data associated with `f1`. Applying the element-count operation to `f1`'s destination shape returns its total number of elements.

=== UID:F28-S08
H: In short
P: DMA statements support synchronous and asynchronous transfers and several movement operations. Futures name the results, expose destination buffers, and provide handles for waiting on asynchronous work.
P: Section 2.8 explains how data expressions select tiles or blocks within a buffer.

=== UID:F34-S06
H: Parallel Perspective of Tileflow Programs
P: In Croqtile, a *divergent* code region is generated with the keyword `inthreads`, followed by a comparison expression.
P: Let us have a look at the following example:
C: Parallel Perspective of Tileflow Programs
P: You can examine Croqtile tileflow programs from the perspective of parallelism. It consists of
L: *Sequential Code* | *SPMD-Styled Parallel Code* enclosed in `parallel-by` block, which consists of: itemize[leftmargin=1.6em] | *Uniform Code* | *Divergent Code* enclosed in `inthreads` block.
P: In this example, `p `= 2 initiates the *divergent code block*, where from the implementation perspective, it creates a thread mask of `000111`, which allows only thread `0`, `1` and `2` to execute the code inside. And thread `3`, `4` and `5` will skip this path. **All threads are forced synchronized after the `inthreads` block**. In this case, it implies thread `3`, `4`, and `5` waits for the completion of the others before stepping forward.
P: An `inthreads` block resembles a C++ conditional, but it selects participants in an SPMD region. Its placement and predicate must therefore refer to that parallel structure:
L: Can only appear in the SPMD code region, which is enclosed in the `parallel-by` block. | Can only have a comparison result related to thread identifiers.
P: Croqtile compiler has the checks for both the comparison expression and `inthreads`-block placement.

=== UID:F34-S09
H: Implicit Masking
P: In Croqtile, code between multiple `parallel-by` levels can be considered as a type of implicit masking. For example:
C: Implicit Masking
P: In this example, there exists two levels of parallelization. The code inside `parallel p by 6` block but outside `parallel q by 2` blocks is conceptually for the outer parallelization only. This means, only one thread from the sub-level parallelization execute the code, which is equivalent to `inthreads (q == 0) .`.. . Programmers must be aware there is only single thread instance to execute the code part. Only the code inside the inner `parallel-by` is executed by all threads.

=== UID:F35-S01
P: An asynchronous region can advance independently, but a dependency on another region still needs to be expressed. An event marks that dependency: one region triggers it after completing the required work, and another waits before using the result. The event carries readiness information; the data remains in the buffer shared by the regions.

=== UID:F35-S02
H: Event Basics
P: The meaning of an event varies between programming languages. In Croqtile, an **event** coordinates asynchronous code blocks at different execution levels.
P: A Croqtile event has two states, **SET** and **UNSET**. A newly declared event starts in the UNSET state. For example:
C: Declaring a shared event
P: A `shared` event is declared inside a `parallel-by` block and used by its parallel instances. Events can also use `global` or `local` storage. The target platform determines the constraints on their declarations and references.
P: In addition, you may declare an event array following C-style syntax:
C: Declaring an event array
P: A comma-separated expression can declare both an event and an event array on one line.
P: The `trigger` statement changes an event from **UNSET** to **SET**:
C: Triggering an event array
P: The comma-separated statement triggers every event in array `e` and the single event `e1`.
P: The `wait` statement blocks the current instance while the event is **UNSET**. Execution resumes once the event becomes **SET**.
P: Triggering an event is what unblocks the code waiting on a *wait statement*. For example,
C: Waiting on an event
P: In this example, the triggering instance sets `e`, allowing the waiting instance to continue. Both then reach `sync.shared`. The event establishes a dependency between the two asynchronous regions: the trigger must occur before the wait can finish.
P: In Croqtile semantics, **an event is always auto-reset by its `wait` statement**. That means when a `wait` statement is unblocked, the events that are waited are changed to *UNSET* state.

=== UID:F07-S06
H: Event arrays for multi-stage pipelines
P: Consider a producer that loads tiles and a consumer that computes on them. Before reading a slot, the consumer needs a signal that its data is ready; before overwriting it, the producer needs a signal that the previous read has finished. Call these signals `full` and `empty`. A pipeline with several reusable slots gives each slot its own pair of events:
C: Event arrays for multi-stage pipelines
P: Each physical buffer slot gets its own `full`/`empty` pair. The ring index `stage = iv_`k % STAGES maps the unbounded K iteration to a fixed number of physical slots. With four stages, the producer can run several tiles ahead before blocking on `wait empty`.

=== UID:F07-S07
H: Bootstrap protocol
P: The consumer must **seed** the `empty` credits before the K-loop starts:
C: Bootstrap protocol
P: Without this bootstrap, the producer's first `wait empty[0]` blocks forever --- a deadlock, not a mysterious MMA bug. This is a common pitfall: every bounded buffer protocol requires initial credits.
P: 3.8.5 develops the full double-buffered and multi-stage pipeline kernels that put these primitives to work. The examples there compose `inthreads.`async, events, `swap`, and `mma.commit` into complete, runnable matmul pipelines.

=== UID:F35-S05
H: In short
P: Events order asynchronous `inthreads` regions. A `trigger` sets an event or event array to **SET**. A `wait` blocks until the event is triggered, then resets it to **UNSET**.
P: Pair event triggers and waits carefully. Incorrect ordering can cause deadlocks that are difficult to diagnose.

=== UID:F07-S02
H: Predicated execution with `if`
P: Croqtile's `if` behaves like its C counterpart:
C: Predicated execution with if
P: All threads in scope evaluate the condition. Threads where it is false skip the body. Threads on the skipped side sit idle while the taken side runs, then vice versa. This is the price of runtime flexibility.
P: Use `if` for runtime decisions such as bounds checks and conditional accumulation. The condition may depend on the current tile coordinate or input values. Keep it uniform across a cooperating group when the body contains an operation that requires the group to participate together.
P: Whenever possible, keep `if` conditions uniform across the execution group: in this example, all 32 threads should agree.

=== UID:F07-S12
H: Early return from `parallel`
P: Use `yield` to return early from a `parallel` block. It generates a `return;` instruction that ends device execution for the current instance.
C: Early return from parallel
B: The `yield` keyword currently will **not** do any compile-time nor runtime checks. Be careful about synchronization safety.
P: Check an early exit against the remaining synchronization in the region. An instance that leaves before triggering an event can strand its consumer; an instance that skips a required collective operation can strand its peers. A tail guard is easiest to reason about when all instances responsible for the same tile make the same decision.

=== UID:F06-S02
P: Every contraction kernel follows the same rhythm, and the spelling makes each step visible:
L: **`mma.fill 0.0`** | **`mma.load`** | **`mma.`row.**row mc, ma, mb --- issue the contraction: **C += A $×$ B** into `mc`. | **`mma.`store mc, dst**
P: You loop steps 2--3 over K (loading the next K-slice each iteration, accumulating into the same `mc`), then run step 4 once to flush the completed output tile. You do not declare per-lane layouts; the compiler derives them from the target and your layout choice (`row.row` here).
P: The `.row.row` suffix is a **layout contract** --- Both operands are row-major. The full set of layout combinations is `row.row`, `row.col`, `col.row`, and `col.col`; in practice, `row.row` and `row.col` cover most kernels. Choosing the wrong variant produces incorrect results.
P: For two K slices, initialize `mc` once, load the first pair of operands and accumulate, then load the second pair and accumulate into the same `mc`. Only then store the output. The load handles describe operand fragments; the accumulator holds the partial result across iterations. This distinction explains why fill and store sit outside the repeated load-and-multiply sequence.

=== UID:F10-S05
H: `__cpp__`: verbatim C++ injection
P: The `__cpp__` form inserts a string as C++ source at the point where it appears in a Croqtile function. Use it for a short target operation that needs to sit inside a particular parallel or asynchronous region. The inserted text must be valid in that scope, with all referenced names visible to the target compiler. Croqtile passes the contents through without parsing or rewriting them.
L: **`_`_**cpp__("...") --- Ordinary string; best for short one-liners. | **`_`_**cpp__(R"(...)") --- Raw string literal; use for `asm volatile` where escaping every `"` would be painful.

=== UID:F10-S07
H: Macros inside `__cpp__` strings
P: A common mistake: assuming the preprocessor expands macros inside string literals.
C: A macro inside a cpp string
P: This fails --- the preprocessor does not expand inside strings. Use numeric literals inside `__cpp__` and `#if` / `#error` outside to enforce consistency:
C: Its expanded form

=== UID:F10-S08
P: Croqtile's preprocessor runs before the main compiler pass. Macros expand in both `__co__` regions and host C++ in the same `.co` file, so one definition keeps tile geometry and host-side checks aligned.
T: | **Directive** | **Role** | <ROW> | --- | --- | <ROW> | `#`define NAME value | Object-like macro: textual replacement | <ROW> | `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion | <ROW> | `#ifdef` / `#ifndef` | Shorthand for whether a macro is defined | <ROW> | `#`error message | Force a compile-time failure with a message |
P: Function-like macros (`#`define MAX(a, b) ...) are not supported. Use `constexpr` helpers in ordinary C++.

=== UID:F10-S09
H: Tile geometry as macros
P: Production matmul sources centralize tile dimensions at the top:
C: Tile geometry as macros
P: The same names appear in `parallel`, shared-memory declarations, `foreach` bounds, and host-side verification.

=== UID:F10-S10
H: Compile-time assertions with `#error`
C: Compile-time assertions with error
P: The preprocessor evaluates this condition before the Croqtile function is parsed. An incompatible set of constants therefore stops the build with the supplied message. Use this for relationships among configuration macros; a property of input data supplied at runtime belongs in a runtime assertion instead.

=== UID:F10-S11
H: Conditional code paths
C: Conditional code paths
P: The preprocessor keeps one branch and discards the other before Croqtile parses the `__co__` body. Command-line defines: `croq kernel.`co -DMATMUL_TILE_K=128.

=== UID:F15-S20
H: Calling C++ Functions
P: Croqtile functions are not allowed to call another Croqtile function. However, inside a Croqtile function, it is normal to have function calls to C++ kernels.
C: Calling C++ Functions
P: The example uses the Croqtile `call` keyword to invoke the existing C++ function `bar`.

=== UID:F23-S01
P: A shared configuration constant often appears in both the host allocation and the kernel declaration. Defining it once as an object-like macro keeps those uses together. The examples below show how replacement text and conditional compilation affect the source before Croqtile checks its shapes and types.

=== UID:F23-S02
H: Object-Like Macros
P: Croqtile preprocessor supports C/C++ **object-like macros** to connect both host code and tileflow code. The *object-like macro* performs pure text replacement and accepts no parameters. The following code gives an example:
C: An object-like macro
P: In the code snippet, the inputs of croqtile function `matmul` are not dynamically shaped. Instead, Croqtile pre-processor substitutes `M`, `N`, `K` with values `256`, `32`, `64` ahead of Croqtile compilation, which produces tileflow function with statically shaped inputs. In this way, programmers make different code consistent and easy to manipulate.
P: The preprocessor currently supports only macros without parameters, so code like:
C: A macro with parameters
P: would not work.

=== UID:F23-S03
H: Comments
P: Croqtile supports both C-style block comments (`/*...*/`) and line comments (`//...`) through its preprocessor.

=== UID:F23-S04
H: Conditional Compilation
P: Croqtile supports conditional compilation with directives including `#if`, `#ifdef`, `#ifndef`, `#else`, and `#endif`:
C: Conditional Compilation
P: The same preprocessing conditions can control both host C++ and Croqtile code.
P: The preprocessor is still being extended, and full C preprocessing support is not planned: a C feature is picked up only when a program actually needs it.

=== UID:F24-S01
P: Croqtile uses device functions to perform target computations. The `call` statement specifies where to invoke a device function and which arguments to pass.

=== UID:F24-S02
H: Call Device Functions
P: To invoke a device function from a Croqtile function, use a statement beginning with `call`.

=== UID:F24-S03
H: Basic Syntax of Call Statements
P: The general syntax for a `call` statement in Croqtile is:
C: The general call syntax
P: After `call`, write the function name and, optionally, comma-separated template arguments in angle brackets. Ordinary arguments follow in parentheses, separated by commas as in C/C++.
P: The target compiler checks that the called function exists and that its signature accepts the generated arguments. Croqtile does not perform these C++ checks during transpilation. A call error may therefore mention generated source; compare it with the helper declaration and the original call site.
P: In the following example, `call bar();` in `mou` is illegal code since it tries to call device function in a code location which is assumed to be host code area.
C: An illegal call site

=== UID:F24-S04
H: The Convention: Allowed Argument Types
P: In current implementation, the argument must be either:
L: *Spanned Data* type, or | Scalar *Integer* type, or | Scalar *Floating-Point* type.
P: Here, the scalar *integer* type only includes `int`, whereas the scalar *floating-point type* includes `float`, `double`, `half`, `bfp16`, and `half8`. However, since the target-platform differs in floating-point support, the *floating-point type* parameter also varies for different target platforms.
P: Croqtile does not perform arithmetic on these floating-point scalar values in Tileflow code. Instead, the Tileflow program passes them to device functions for computation.
P: For example:
C: The Convention: Allowed Argument Types
P: Here the spanned-data arguments become pointers to their element types. The device helper receives those pointers without shape metadata; pass any required dimensions separately. The declared pointer element type must match the data supplied by the call.
P: For element formats without a native target type, use the corresponding Croqtile C++ type, such as `croqtile::f16` or `croqtile::bf16`, where supported.

=== UID:F24-S07
H: In short
P: A `call` statement invokes a device function, with or without template arguments. Care is needed since no check is applied at transpilation time, which may result in more confusing error reporting since the check is applied at target compile time with generated functions.

=== UID:F11-S02
H: `print!` and `println!`: compile-time inspection
P: The bang variants (`print!`, `println!`) run at **compile time**, not at runtime. They print to the compiler's output and are useful for inspecting shapes, extents, and type information without launching a kernel:
C: A compile-time shape print
P: When you compile this file, the compiler emits:
C: The printed shape
P: String literals are concatenated (`"wor" "ld!"` becomes `"world!"`).

=== UID:F11-S03
H: `print` and `println`: runtime device output
C: A runtime inspect kernel
P: Each call takes a comma-separated mix of string literals and expressions. The output order across threads is **nondeterministic**.
P: To inspect one execution path, guard the print with a condition on its index:
C: A print inside a parallel block
P: Without the guard you get one line per thread --- thousands of lines for a large grid.

=== UID:F11-S04
H: `assert`: runtime invariant checks
P: Croqtile's `assert` builtin checks an invariant at runtime and aborts the kernel with a message if it fails:
C: assert: runtime invariant checks
P: On the device, this compiles to a `printf` of the message followed by an abort. Use assertions to catch index-out-of-range, null pointer, and other "this should never happen" conditions early.

=== UID:F04-S25
H: Checklist
P: The syntax collected here means the following:
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `dma.`copy.async ... | Start a non-blocking DMA transfer. | <ROW> | `wait f` | Wait until future `f` is ready to consume. | <ROW> | `future.data` | Access the moved tile. | <ROW> | `future.span` | Access the moved tile's shape. | <ROW> | `dma.`transp<...> | Permute dimensions during transfer. | <ROW> | `dma.`pad<...> | Pad during transfer with an explicit fill value. | <ROW> | `.zfill` | Zero-fill out-of-bounds positions in partial tiles. | <ROW> | `.`swiz<N> | Choose a shared-memory swizzle layout. | <ROW> | `chunkat(...)` | Select one equal tile from a regular tiling. | <ROW> | `subspan(.`..).at(...) | Select a tile by explicit tile extents and tile coordinates. | <ROW> | `view(.`..).from(...) | Select a tile by explicit extents and explicit element offsets. | <ROW> | `tile # i` | Compose tile and in-tile coordinates into a global coordinate. | <ROW> | `#tile` | Ask for the extent of the `tile` axis. |
P: When reading a movement statement, first identify the selected region and its destination, then check when the destination becomes ready. These questions remain the same in a pipeline; the extra buffers simply allow different tiles to be at different points in that sequence.

=== UID:F05-S18
H: Summary of New Syntax
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `parallel p by N` | N-way parallelism, index `p` runs 0..N-1 concurrently | <ROW> | `parallel px, py\` by [M, N] | Multi-dimensional parallel level (`p` and `q` correspond to one parallel level) | <ROW> | `parallel p by A, q by B` | Comma-separated parallel levels (`p` and `q` correspond to different parallel levels) | <ROW> | `parallel q by N :` thread | Map to threads within a block | <ROW> | `foreach index = m, n, k\` in [a, b, c] | Named tuple destructuring in a `foreach` | <ROW> | `lhs.span(0)` | Extract one dimension of a tensor's shape | <ROW> | `p # m` | Compose outer tile index `p` with inner offset `m` |
P: Section 3.7 replaces scalar arithmetic in the inner loop with matrix contraction operations.

=== UID:F07-S13
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `if (expr) .`.. | Predicated execution --- runtime conditional, divergent threads masked | <ROW> | `inthreads.`async (condition) | Structured concurrent region --- compile-time thread partitioning | <ROW> | `shared event name[N]` | Declare N synchronization tokens | <ROW> | `trigger name` | Signal that a condition is met | <ROW> | `wait name` | Block until the corresponding `trigger` fires | <ROW> | `tile_`id = tile_iter # block_id | Compose indices for tile striping | <ROW> | `int total_`tiles = expr | Local integer variable | <ROW> | `yield` | Early return from `parallel` block |

=== UID:F08-S14
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `dma.`copy.async src => dst | Non-blocking copy (returns immediately) | <ROW> | `dma.any` | Placeholder future (no transfer in flight yet) | <ROW> | `swap(f0, f1)` | Exchange two future handles without copying data | <ROW> | `rotate(f0, f1, f2)` | Cycle three future handles | <ROW> | `with tile_`k in N ... | Scoped tile axis binding with extent N | <ROW> | `foreach tile_`k(1:) | Iterate starting from index 1 | <ROW> | `_`_co__ auto fn(...) | Return type inferred from `return` statement |

