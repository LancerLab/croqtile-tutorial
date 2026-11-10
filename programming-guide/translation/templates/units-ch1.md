=== UID:F01-S01
P: Much of the development effort in a high-performance kernel goes into organising the computation. Beyond implementing the arithmetic, the programmer has to arrange data storage, divide parallel work and coordinate execution. These choices interact, so a change can require updates in several places. Different input sizes or target machines call for another round of validation and tuning. Implementing, maintaining and revising these arrangements accounts for much of the cost.

=== UID:F01-S02
P: Croqtile is a C++ embedded DSL for high-performance kernel programming, designed to make this work easier while preserving control over performance. It provides a direct way to describe the organisation of computation, data movement and coordination in a compact program. The programmer chooses how the kernel works; the compiler derives related information, checks consistency and generates the underlying code. This reduces the repetitive work involved in expressing and maintaining those choices.

=== UID:F01-S03
P: The resulting programs are easier to read as a whole and revise as requirements change. They can accommodate different input sizes, and the abstractions introduce no additional runtime overhead. Compact source and useful compiler feedback also make the same process accessible to AI-assisted development and tuning.

=== UID:F12-S01


=== UID:F13-S01


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
P: The `__co__` prefix marks this as a Croqtile function. Unlike regular C++ functions, the signature carries full shape information: `s32 [4, 8]` means "a 2D tensor of shape 4 $×$ 8 with element type `s32` (signed 32-bit integer)". Both parameters and the return type follow this convention --- `type [shape] name`. The compiler uses these shapes to verify that every indexing operation is in-bounds at compile time.
P: Croqtile supports these element types: `s32` (signed 32-bit int), `f16` (half-precision float), `bf16` (brain float), `f32` (single-precision float), and `f8_e4m3` / `f8_e5m2` (8-bit floats). In the examples that follow, `s32` is the simplest to reason about.
C: Allocating the output
P: This allocates the output tensor. The expression `lhs.span` copies the full shape from `lhs`, so `output` automatically has shape `[4, 8]`. If you later change the input shape, the output adjusts --- a pattern that makes Croqtile functions easier to generalize.
C: The parallel body
P: What `parallel` leaves unsaid is how those instances map onto the machine: without an annotation the compiler decides. That mapping is deliberately absent from the language here --- the statement describes independent work, not the resources that run it.
C: Returning the result
P: A `__co__` function returns its result tensor. The return type in the signature (`s32 [4, 8]`) must match what you actually return. The caller on the host side receives a `croq:`:spanned_data --- an owning buffer with shape metadata that you can index into with `[i][j]`.

=== UID:F08-S08
H: `_`_co__ auto return type
P: centerUsing `s32` with scalar accumulation is a simplified style that isolates the `swap` mechanism (3.6.2*); the same return.type deduction applies to the FP16/MMA kernels in Section 3.7.center

=== UID:F03-S05
H: The Host Program
P: The host side is plain C++ with two Croqtile API calls: one creates a shaped buffer from raw storage, the other hands a view of it to the kernel.
C: Creating the shaped inputs
C: Calling the kernel
P: Calling a `__co__` function from C++ looks like a normal function call. The `.view()` method produces a non-owning `spanned_view` from an owning `spanned_data` --- this is how you pass host tensors into a Croqtile function without transferring ownership. The return value `res` is a `croq:`:spanned_data that owns its buffer.
P: The rest of `main` is ordinary verification: loop over every element and assert equality. The host program is deliberately boring --- the interesting work happens in the `__co__` function.

=== UID:F05-S16
H: Host Code
P: The host side uses the same `make_spandata` / `.view()` pattern used by the host code above:
C: Host Code
P: The loop after the call checks every element against a plain C++ reference computation, so an incorrect result is reported before the program prints anything.

=== UID:F15-S21
H: Parameters Passing between Croqtile and C++ Function
P: Passing arguments between Croqtile and C++ requires the `croqtile.h` header. Normally, programmers combine the raw C++ pointer and associated dimensions info to construct Croqtile spanned data. The code below demonstrates how it works.
C: Parameters Passing between Croqtile and C Function
P: In the example, you make use of croqtile utility function (template) 'make_spanview' to wrap the data, which does not invoke copy. And in Croqtile function 'foo', the parameter 'f32 mdspan<2>' is the corresponding entity. Because the data is implicitly cast, you can simply use 'd' as the first parameter of C++ function 'bar'. And operation '|d|' obtains the total size of the spanned data 'd'. These are used for calling C++ function 'bar'.
P: Similarly, *Scalar Type* data can also be passed from/to Croqtile function. Nevertheless, ituple is only used inside Croqtile function.

=== UID:F25-S01

=== UID:F25-S02

=== UID:F25-S03
H: Input Parameters
P: A Croqtile function accepts only two kinds of input parameter, the *spanned data* and the *integer*.
P: Similar to any C++ program, the parameter may either be named or unnamed:
C: Input Parameters
P: In this example, the second parameter is unnamed, which makes it unable to be reference in the Croqtile function. However, this is still the valid code.

=== UID:F25-S04
H: C++ Host: Data Argument with Shape
P: One obvious gap between C/C++ pointers/arrays and Croqtile *spanned data* is that *spanned data* is shaped. Therefore, to call a Croqtile function, it is necessary to convert pointers or arrays.
P: One direct method is to use the Croqtile API provided in `croqtile.h`. A program does not have to include `croqtile.h` explicitly in a Croqtile compilation environment; the compiler includes it for every Croqtile program.
C: Passing a shaped buffer
P: In the above example, you make use of `croqtile:`:make_spanned function template, providing the rank as template parameter, and typed pointer together with shape list as the function parameters. In this way, a shape is attached with the pointer, which makes a valid Croqtile function parameter. As you may figure out from `croqtile.h`, it results in `croqtile:`:spanned_view object, which simply wraps the C/C++ pointer/array and the array-based list. Its simplified definition is:
C: The host-side buffer wrapper
P: Dimensions do not have to be constants; runtime values work as well:
C: A call built from runtime dimensions
P: In this example, the `input` shape is runtime variable. But do not worry if wrong parameter is provided. Croqtile will check if `M == 7` and `N == 16` when entering into the tileflow program. **If not, the program will terminate immediately** since all the assumptions about shapes in the Croqtile function does not hold.
P: One important notice for the *spanned data* inputs is that they are **references to the input data/buffers**. That implies no data copy happens at invoking Croqtile function, though somebody may think from the language perspective it look like copy-semantics.

=== UID:F25-S05

=== UID:F25-S06
H: Return Type and Type Deduction
P: Similar to the input, *spanned data* and *integer* are valid output types. The *void* type is also allowed if nothing is returned.
P: In addition to the explicit return type annotation, Croqtile support type inference on returns. For example:
C: Return Type and Type Deduction
P: In this code, the return type is annotated as `auto`, which is the same keyword as C++. Croqtile is capable of inferring return type as `f32 [7, 16]`. Therefore, it is recommended to use `auto` whenever possible.

=== UID:F25-S07
H: C++ Host: Shape Return
P: The C++ host code that receives the output of Croqtile function actually receives a buffer. The return has value semantics: the caller receives the buffer itself, not a reference to an object allocated inside the called function. For example:
C: Returning a shaped buffer
P: In this example, the `output` is semantically copied to the caller. However, in Croqtile's implementation, no copy happens. The returned type is `croqtile:`:spanned_data, where its simplified definition is as following:
C: The host-side result wrapper
P: In contrast to `croqtile:`:spanned_view, `croqtile:`:spanned_data owns the buffer it points to. Thus there is **no copy actually happens when returning from a Croqtile function**. In compiler terminology, this is called the *Return Value Optimization*.
P: You may use member function like `.rank()`, `.shape()` to query shape related information of Croqtile return. It lets the host work with the result without knowing the details of the Croqtile function that produced it.

=== UID:F04-S16
H: `span` and `span(i)`
C: span and span(i)
P: Deriving one shape from another comes up constantly in data movement code, because output shapes often come from different inputs, and moved tiles frequently need to preserve or recompute shape information.
F: f04-fig_span_light.png
P: centerUse `span` for the whole shape, `span(i)` for a single axis.*center

=== UID:F16-S01
P: A shape in Croqtile is a value a program can name and compute with, not just an annotation on a declaration. The examples below define one, derive another from it, and leave a dimension open for the caller to fill in.

=== UID:F16-S03
H: Defining Shapes with `mdspan`
P: In Croqtile, shapes can be defined with the `mdspan` keyword, which stands for **Multi-Dimensional Span**. This keyword represents multi-dimensional data for computations.
P: You can explicitly define `mdspan` variables as follows:
C: Declaring a shape
P: In this example, the leading keyword `mdspan` indicates the declaration of a *mdspan* variable, followed by the user-provided variable name. Each *mdspan* variable is initialized with an **initialization expression**, which consists of comma-separated integer values enclosed by `[]`. The symbol `:`, which immediately follows the variable name, introduces the *initialization expression*. Therefore, `s0` is defined as a 2D shape with `7` rows and `8` columns, while `s1` is a 1D shape with a dimension of `3`.
P: It is possible to optionally specify the **rank** of an *mdspan* variable by placing ``> after the `mdspan` keyword. If the rank is explicitly specified, it informs the Croqtile compiler to check for rank consistency. If the rank value differs from the corresponding *initialization expression*, it triggers a failure at compilation. Here is an example:
C: A rank mismatch
P: In addition to explicit mdspan declarations, the Croqtile compiler can infer the type of `mdspan` variable from its *initialization expression*, eliminating the need for the explicit `mdspan` keyword:
C: Omitting the mdspan keyword
P: In this code, `s3` is an `mdspan` of rank `3` with dimensions `7, 8, 9`. Since Croqtile **requires mdspan to always be initialized within declarations**, the type inference version is preferred in programming practice.

=== UID:F16-S04
H: Deriving *mdspan*s
P: In practical coding, it is common to derive a new shape from an existing one. For example, you might want to perform data **tiling** or **blocking**, which requires dividing the dimensions of a shape. Alternatively, you might want to **pad** specific dimensions, which involves adding delta to the shape dimensions.
P: In Croqtile, such shape derivations can be easily accomplished using arithmetic operations on *mdspan*. The following code showcases an example:
C: The original shape
P: In this example, the `new-shape0` is derived from `shape`, with dimension 0 divided by `2`, dimension 1 divided by `4`. This corresponds to *tiling* operation in high-level semantics. Additionally, the code adds a new dimension to `new-shape`. In high-level semantics, this operation is often referred to as *reshaping*.
P: In Croqtile, the definition of `new-shape0` is equivalent to:
C: A derived shape
P: Here, the initial `shape` is explicitly listed element-wise rather than specified outside `[]`. But similar to the prior version, The **element-of** operation, which is annotated as `()`, is used on top of existing shape to retrieve dimension values. Obviously, this approach requires more code but yields the same result. Thus, the prior version can be considered *syntactic sugar* for the complete *initialization expression* of the new shape.
P: In the code example, `new-shape1` is also derived from `shape`, it pads dimension 1 by `2` and swaps the dimensions in the derived shape.
P: Furthermore, you may use the `mdspan` as a whole for derivations:
C: Rebinding a shape
P: Arithmetic operations on an *mdspan* variable are applied dimensionally. Thus, the statement
P: The expression `new-`shape1 : shape + 1;
P: is equivalent to
P: The expression `new-`shape1 : shape [(0) + 1, (1) + 1];
P: The derived definition of `new-shape3` demonstrates that using an *mdspan* variable in an *mdspan initialization expression* results in **concatenation** behavior. Thus, the declaration
P: The expression `new-`shape3 : [shape, 6];
P: is equivalent to
P: The expression `new-`shape3 : shape [(0), (1), 6];
P: The prior version can be as well deemed as a *syntactical sugar* of a complete definition.

=== UID:F19-S03
H: *mdspan* with **Anonymous Dimension**
P: Croqtile supports shape dimensions with unknown values, similar to many high-level machine learning languages. The simplest method is to use **Anonymous Dimension** in `mdspan`:
C: A kernel with an unknown dimension
P: Here, the question mark `?` represents a compile-time unknown value. Any positive integer could be assumed here. However, the value must be provided at runtime, as the following example shows:
C: The caller supplying that dimension
P: In this example, the first dimension of `input` is provided by `foo`'s *spanned* input, where the value is retrieved from `bar`'s caller at execution. You may assume the value be determined by reading a configuration file, through shape derivation in a machine learning framework, or etc..
P: The *anonymous dimension* can work properly for this case. However, in certain scenarios, it is not sufficient.

=== UID:F19-S04
H: Step Further: **Symbolic Dimension**
P: Another approach in Croqtile to support dynamic shape is to use **Symbolic Dimension**. *Symbolic dimension*s are named but as well unknown at compile-time. Here's an example:
C: Step Further: Symbolic Dimension
P: In the code, each dimension is given a symbolic name (`M, K, N` in this case) for the *spanned* inputs. Compared to *anonymous dimension*, *symbolic dimension* approach clearly describes the relationship between shapes. The code looks intuitive and is easy to maintain. Furthermore, it is possible to improve the code safety with the additional information provided. Let us dive deeper for this.

=== UID:F15-S05
H: Scalar Types
P: Scalar types in Croqtile include the *Integer Type* and *Boolean Type*. The *Integer Type* is similar to C++ type 'int' or 'int32_t'. It is an signed value which takes 32-bits, ranging from $-2^31$ to $2^31-1$. The below code illustrates its usage for defining the data and function declaration.
C: Scalar Types
P: Operations like integer arithmetics, shift and logical operation are all supported. And the syntax is identical to C++ builtin operations. In Croqtile, you neither provide equivalence of unsigned scalar integers, nor equivalence of 8-bits, 16-bits, 64-bits scalar integers. This design is simply motivated by the fact: these types are not essential for program control purposes. And normally a 32-bits signed integer can fulfill such work.
P: The *Boolean Type* in Croqtile is similar to C++ type 'bool'. The operations on top of Boolean, and conversion between Integers, are identical to these of C++.

=== UID:F15-S10
H: I-Tuple Types
P: An integer tuple is an unordered set of integers. As described, it is normally used as a (subscription) index.
P: To define an i-tuple, simply enclose its elements within '' and '' braces. That is,
C: I-Tuple Types

=== UID:F15-S11
H: Operations over *mdspan* and *i-tuple*
P: In Croqtile, you allow special operation over *i-tuple* and *mdspan*. Below is an example to apply a fixed tiling over an mdspan:
C: Operations over mdspan and i-tuple
P: In Croqtile, mdspan can be defined with such **Tuple-Span Operations**. The supported operations includes:
L: *mdspan* $/$ *i-tuple* | *mdspan* $+$ *i-tuple* | *mdspan* $%$ *i-tuple* | *mdspan* $*$ *i-tuple* | *mdspan* $-$ *i-tuple*
P: Essentially, these operations can be achieved through mdspan *dimension-wise* definition. However, *tuple-span operations* aid programmers in writing more readable code. This is also the objective that Croqtile aims to achieve.

=== UID:F17-S01
P: Integers in a kernel control loops and index arrays, and an *i-tuple* groups several of them into one index. Both are declared with an initializer and both are checked against the shape they are used with.

=== UID:F17-S02
H: Integers for Program Control
P: In high-performance computation kernels, integers are typically used for program control rather than computation. In Croqtile *tileflow programs*, integers are used exclusively for loop control and array indexing. Given the limited value ranges in these scenarios, Croqtile has provided a single integer type --- the *32-bit signed integer* --- to simplify the design for this domain-specific use.
P: In Croqtile, an **integer**-typed variable is defined similar to C/C++:
C: An initialized integer
P: Croqtile can also infer the type from the *initialization expression*, so an integer variable can be declared in another style:
C: Assignment to an integer
P: Similar to C/C++, you can define multiple integers in a single line:
C: Several integers at once
P: In Croqtile, the *mdspan*-typed shape can not be used as the function parameter. However, integers can be passed from the host to the Tileflow program, and from the Tileflow program to the device:
C: An integer parameter
P: Unlike C/C++, Croqtile does not allow the declaration of integers without initialization or the reassignment of an integer. For example:
C: A declaration without initialization
P: This enforces that **integers must be initialized at declaration and cannot be reassigned**. In some contexts, these integers are referred to as "**immutable**" or "**constant**".

=== UID:F17-S04

=== UID:F17-S05
H: Defining I-Tuples
P: In Croqtile, multiple integers can be grouped into an **integer-tuple**, or **i-tuple** (or **ituple**). The keyword `ituple` can be used to define an *i-tuple*. The below code showcases the usage:
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
P: In this code, a mdspan-typed shape is divided by an ituple-typed tiling factor to derive a `tiled_shape`. Additionally, a `padded_shape` is derived from the addition of the initial `shape` and an anonymous *ituple*. Note that when a *mdspan* is added by an *ituple*, their rank must be consistent. Or else, an error will occur.
C: An invalid shape expression

=== UID:F17-S09
H: In short
P: Integers and i-tuples are the two scalar-shaped values a kernel declares. The *Integer*s follow C/C++ syntax. *i-tuples* follow a syntax similar to *mdspan*, but they can operate on *mdspan* to derive new values.
P: Both types **require an _initialization expression_ and cannot be reassigned**, making them appear as **constant** or **immutable** values. In fact, all variables defined in Croqtile tileflow programs are immutable, except for those of specific types. Therefore, **the assignment operator `=` is used for variable initialization in most cases**, which programs must be aware of.

=== UID:F15-S08
H: Fully-Typing
P: A *mdspan* can not be applied alone to define the data for computation. In Croqtile function, a data definition must be fully-typed, which consists of a fundamental type and a *mdspan*. The below code showcases how it works.
C: Fully-Typing
P: There are fundamental types that Croqtile has supported, including:
L: Unsigned Integers: *u8/u16/u32* | Signed Integers: *s8/s16/s32* | Floating-points: *f16/bf16/f32*
P: Note 's32' and 'int' are different in Croqtile. 's32' is a fundamental type, which can not be applied for a fully typing.

=== UID:F15-S09
H: The Storage Qualifier
P: A spanned-typed data in Croqtile is usually large.
P: In croqtile, the language defines three storage qualifiers to annotate the data being defined, including:
L: **global**, | **shared**, | and **local**.
P: The below code showcases the usage.
C: The Storage Qualifier
P: By default, when no storage qualifier appears, the data defined is considered as from the *global* storage.

=== UID:F18-S01
P: Croqtile *tileflow program* describes how to move data around. Therefore, declaring or defining the data or buffer are fundamental. The declarations that follow fix an element type, a shape and, where it matters, a storage specifier.

=== UID:F18-S03
H: Defining a *Spanned Data*
P: Defining a *spanned data* requires both the element type and its shape. For example,
C: A spanned-data declaration
P: This defines a buffer with a shape of `[32, 16]`, where the element type is `f32`, which is the *IEEE-754* single-precision float. It is also possible to use a named `mdspan` to define the buffer:
C: A standalone shape
P: Here the previous example used the named *mdspan* `shape` to define a 2D buffer named `data0`, where the elements are 32-bit integers. Note that the named *mdspan* `shape` is placed inside `[]`. Additionally, it is allowed to use *element-of*, arithmetics, or concatenation on *mdspan*s to specify the buffer's shape.
P: In Croqtile terminology, the element type is referred to as the **fundamental type**. Croqtile supports multiple fundamental types, including:
L: **Unsigned Integers**: `u8`, `u16`, `u32` | **Signed Integers**: `s8`, `s16`, `s32` | **Floating Points**: `f16`, `bf16`, `f32`
P: The prefixes `u` and `s` stand for "unsigned" and "signed", respectively, while the number suffix indicates the bit width of the type. `f16` refers to the *IEEE-754* standard *half-float*, and `bf16` refers to the 16-bit *binary float*, commonly used in machine learning scenario.
P: Note `s32` is different from `int`. Although they occupy the same amount of storage, in Croqtile, `s32` can not be used alone to define programming entities. For example:
C: A rejected fundamental type
P: This results in a compile-time failure. Consequently, `s32` cannot be used for program control like `int`, and type conversion between them is not possible.

=== UID:F18-S04
P: In practical code, some buffer declarations must specify a **storage specifier**, like so:
C: Storage Specifier
P: Since Croqtile handles storage in a heterogenous context, a buffer definition without a storage specifier defaults to the storage type of the host program, i.e., CPU memory. Other storage specifiers are defined by the target.
L: *global*: Refers to the device's global storage. | *shared*: Refers to the device block's shared storage | *local*: Refers to the thread-private storage.
P: Other targets may have different definitions. Furthermore, buffers with different storage types have limitations in their declarations. You will explore this in later chapters.

=== UID:F18-S05
H: Initialization
P: It is possible to initialize a buffer at the declaration site. For example:
C: Initialization
P: The syntax for the *initialization expression* of a *spanned data* is straightforward: it encloses an initial value inside brackets following the variable name. However, its functionality is limited --- it always set all elements to a fixed value.

=== UID:F18-S06
H: Declaring the Parameters and Return Values
P: A *spanned data* is passed between host and tileflow programs. Therefore, a croqtile function can have *spanned data*s as its parameters. The following code showcases an example:
C: An explicit return shape
P: The syntax is similar to variable definitions, except that operations on an existing *mdspan* is not possible. Additionally, no storage specifier is allowed or initialization is allowed.
P: One useful built-in member function for the *spanned* parameters is `.span`, which provides the associated *mdspan* of the *spanned data*. The following code demonstrates how to use it for a buffer declaration:
C: A deduced return shape

=== UID:F18-S08
H: In short
P: A declaration fixes the element type and shape, an initializer can fill the buffer at that point, a storage specifier says where it lives, and the compiler manages its lifetime from there. This enables us to step further to the next essential topic: the dynamic shape support.

=== UID:F37-S04
H: 1. `global`
L: The default memory space for data in Croqtile, if no storage qualifier is specified.
C: 1. global

=== UID:F37-S05
H: 2. `shared`
P: A `shared` buffer follows the same declaration syntax:
C: 2. shared

=== UID:F37-S06
H: 3. `local`
P: It is particularly useful when each thread requires a small amount of data, such as scalar values or small arrays that do not need to be shared with other threads. Example:
C: 3. local

=== UID:F37-S07
H: Default Behavior and Usage
P: This default behavior simplifies memory management for users who do not need to explicitly define where their data should be stored, but it also means that accessing such data could incur higher latency, especially when working with large arrays or parallel processing units.
C: A declaration without a specifier
P: If you want to place data in a different memory space (shared or local), you need to specify the appropriate storage qualifier.
C: Choosing local storage explicitly

=== UID:F37-S08
H: Example Code: Using Storage Qualifiers
P: To better understand how storage qualifiers work, consider the following code example:
C: Example Code: Using Storage Qualifiers
P: In this code:

=== UID:F05-S03
H: Croqtile's Two-Layer Parallelism Model
P: Croqtile gives you two separate knobs:
L: **`parallel`** --- declares that iterations are independent and *may* run concurrently. This is a logical statement. | **Space specifiers** (`: block`, `: thread`, `: group`, `: group-4`) --- tell the compiler each parallel axis maps to. This is a physical statement.
P: There is also `foreach`, which is the opposite of `parallel`: iterations run **sequentially**, in order. Use `foreach` when later iterations depend on earlier ones --- an accumulation loop, a K-dimension reduction, a pipeline that loads tile *k* before computing with it.
C: Croqtile s Two-Layer Parallelism Model
F: f05-fig2_logical_vs_physical_light.png
P: centerLeft: the logical nesting the programmer writes (parallel = concurrent, foreach = sequential). Space specifiers bridge the two.*center If you omit the space specifier --- just `parallel p by 8` --- the compiler picks a default mapping. That is what Chapters 1 and 2 did, and it worked fine for simple cases. Explicit specifiers are how you take control when performance matters. Both `parallel` and `foreach` support multi-dimensional indices (`a, b\` syntax), the compose operator (`#`), and the extent operator (`#p`). Everything from Chapters 1 and 2 still applies.

=== UID:F05-S05
H: `: block` --- parallel blocks
C: : block --- parallel blocks
P: This creates an 8 $×$ 16 = 128 block grid. Blocks cannot communicate with each other during execution --- they are truly independent.
P: Use `: block` for the outermost tiling of your output. If your output is `[128, 256]` and each block handles a `[16, 16]` tile, you need `128/`16 x 256/16 = 8 x 16 = 128 blocks.

=== UID:F05-S06
H: `: thread` --- Threads Within a Block
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
P: Opens a scoped region and binds `tile_k` as a tile axis with extent 16. Inside the block, `tile_k` is the chunk index for `chunkat` along K, and `#tile_k` is 16.

=== UID:F08-S07
H: `foreach tile_`k(1:): sliced iteration
P: The slice `(1:)` means tile indices `1, 2, ...` through the end. Tile 0 was loaded in the prologue.

=== UID:F15-S14
H: Parallel Region: the 'parallel-by' Block
P: In systems like CPU, it allows of asynchronous thread to realize the parallel execution. However, in Croqtile, it employs the Single Instruction Multiple Data (SPMD) model as it way to realize parallelization.
P: The syntax of constructing a parallel region is quite different. It employs the C-style bracket and encloses the code for parallel execution within the 'parallel-by' block.
C: Parallel Region: the parallel-by Block
P: The above code snippet illustrates the method to create a parallel region with Croqtile keyword 'parallel' and 'by'. Here, it assume there are 6 execution threads. Each of the thread execute the same SPMD code inside but with a different 'p' value. Alternatively, if you are more familiar with sequential C/C++ programming, you may consider 'p' as the iteration variable of a loop with 6 iterations. (But any iteration may go first to be executed!)
P: Despite parallelism, there is one more implication of 'parallel-by'. In the statement, 'p' is an integer associated with its bound [0, 6). You name 'p' as a **bounded integer** instead of a simple integer. In some special operations like 'chunkat' (explain later), it requires the *bounded-integer* to work properly since the bound is essential for its computation.

=== UID:F15-S15
H: The 'with-in' Block and 'where' Clause
P: Similar to 'parallel-by', 'with-in' statement can also bind *i-tuples* to a *mdspan*. The below code shows an example.
C: Indexing a with-in block
P: Here, 'index' is a *i-tuple* with 2 elements. Sometimes programmers prefer that the 2 elements being named. This is possible by using the below syntax.
C: A tuple of indices
P: Or even to name both the *i-tuple* and its elements.
C: Indices derived from a tuple
P: You name 'index' as a **bounded ituple** in such scenarios.
P: While 'with-in' may seem like 'parallel-by', it operates differently. One significant difference is 'with-in' statement does not have implication for parallelism. The code block inside 'with-in' statement is sequentially executed. It does nothing more than creating the *bounded-ituple*.
P: In addition, programmers could append a 'where' clause. For example,
C: Nested bounded groups
P: The code snippet requires 'n' and 'n_p' to have an identical value in all iterations. Thus inside the 'with-in' block, it is possible to replace'n' whenever 'n_p' is required, or the opposite. Such a facility is useful to program many AI kernels. Programmers should use operation '<->' to establish such relations.

=== UID:F15-S16
H: The 'foreach' Block
P: Once the *bounded-ituple* is defined by the 'with-in' clause, programmers can loop over the bounded-ituples/bounded-integers. In Croqtile, this is simple.
C: The foreach Block

=== UID:F31-S03
H: Basic Syntax: `parallel-by`
P: The basic syntax for a `parallel-by` block is as follows:
C: A named parallel index
P: You name the code inside `\` as the SPMD-style **parallel execution block**, and the instances of the code as the **parallel thread**. In this example:
L: **`parallel`**: This keyword initiates the *parallel execution block*. | **`p`**: It is the **parallel variable**. It contains an integer value identifying the current *parallel thread*. Each *parallel thread* runs the same code but with a different value of *parallel variable* `p`. | **`by 6`**: It indicates that there are totally 6 *parallel thread*s. Thus, the *parallel variable* `p` ranges from `0` to `5` across different threads.
P: In some scenarios, programers may not require an explicit *parallel variable* for a simple parallel construct. It is possible to omit it:
C: An unnamed parallel index
P: This invokes two parallel threads to execute. However, it is not possible for the two threads to work on different data, which is normally required by SPMD programs for a data-parallel processing purpose.

=== UID:F31-S04
H: Multiple-Level Parallelism
P: In Croqtile, you can define multiple levels of parallelism. Here's an example:
C: One parallel level
P: This code defines two levels of parallelism with values `2` and `12`, respectively. This means a total of `2 x 12 = 24` parallel threads are invoked for the task. Each thread is identified by a unique combination of `p` and `q`. In Croqtile, you refer to the level of `p` as *parallel-level-0* and the level of `q` as *parallel-level-1*.
P: One key difference observable by programmers is the **synchronization cost**. This hierarchical design encourages programmers to place the most divergent code (that requires synchronization) within the **block** level to enhance overall performance.
P: In Croqtile syntax, you can define multiple-level parallelism in a single line:
C: Two parallel levels
P: This achieves the same parallelization as the previous example. It uses the **comma-separated** *parallel-by* expression, assuming there is no code in the between two parallel levels.

=== UID:F31-S05
H: Sub-Level Parallelism
P: Targets like allow a single parallelism level to be divided into up to three sub-levels. In Croqtile, `ituple` and `mdspan` are used for this syntax:
C: Sub-Level Parallelism
P: In this example, two levels of parallelism are defined. The first level (*level-0*) is subdivided into three sub-levels, annotated by the `px, py, pz\` `ituple`. The second level (*level-1*) is subdivided into two sub-levels, annotated by the `qx, qy\` `ituple`, or `t_index` as a whole. For each sub-level, the parallelization count corresponds to the respective value inside the `mdspan`.
P: The first element inside the `ituple` represents the **least significant parallel variable**. Croqtile adopts this ordering to align with developers' conventions, rather than for internal consistency.

=== UID:F32-S01
P: A `with-in` statement declares a *bounded variable*; a `foreach` statement iterates over its range. Together they describe a loop without spelling out its bounds in C++.

=== UID:F32-S02
H: Loops in Croqtile
P: In C++, loops are typically constructed using `for`, `while`, and `do-while` blocks. These loops include conditions for termination and usually involve variable manipulations related to loop control. Programmers have the freedom to define exit conditions, loop stepping, and other aspects, allowing for flexible loop structures.
P: In contrast, Croqtile, being a domain-specific language for data movement, supports a more constrained loop structure. The current loop construction in Croqtile is as follows:
C: A foreach loop
P: Here, `index` is a *bounded variable* (introduced in the previous section). If `index` has an upper bound of `6`, the above code is equivalent to:
C: The equivalent C++ loop
P: While *bounded variables* may seem restrictive, they are sufficient for data movement tasks, as you typically won't move data with a negative or out-of-bound index.
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
P: Notice the ordering of the loop nesting: the **left-to-right** order of the *bounded variable*s within a *bounded ituple* corresponds to the **outer-to-inner** nesting of the loops. This ordering is essential for correct code behavior constructing sometimes.
P: Furthermore, This rule also applies to the *comma-separated bounded variable list* that follows the `foreach` statement:
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
P: As seen in the code, the `y`-loop starts at `1`. The *range expression* consists of a *bounded variable*, followed by braced enclosing three colon-separated integer values, like the below form:
C: The bounded-variable syntax
P: This derives a loop from the `bounded-`variable, where the `bounded-`variable serves as the *iteration variable* of the loop. Specifically:
L: The initial value of the *iteration variable* is the `lower-offset` plus the lower bound of the `bounded-`variable. Since all bounded integers have a lower bound of `0` in Croqtile, the `lower-offset` sets the initial value of the loop. | The loop terminates when the *iteration variable* is equal to or greater than the upper bound of the `bounded-`variable plus `upper-offset`. Negative values are typically used for `upper-offset`. | The *iteration variable* increments by `stride` at the end of each iteration.
P: Thus, `y(1:-1:2)` results in a loop like `for (y = 0 + 1; y ` 17 - 1; y += 2) in the example above. If any field of the *range expression* is not specified, it results in default values: `0` for `lower-offset` and `upper-offset`, and `1` for `stride`.
P: Note that *range expression*s only apply to the *bounded variable*s. *Range expression* over the *bounded ituple* triggers an error at compile time.

=== UID:F32-S10
H: In short
P: Bounded variables and the loops derived from them are what data movement code is built from. You introduced the syntax and behavior of these constructs, including how to apply range operations to modify loop iterations and the importance of understanding the current and upper-bound values of bounded variables.
P: The loop deriving part is important for implementing multi-buffering data movement, which is essential for building high-performance kernels and will be introduced in optimization chapters later.

=== UID:F04-S05
H: `chunkat(...)`: equal partitioning
C: chunkat(...): equal partitioning
P: That makes `chunkat` the natural tool when your tiling follows the parallel decomposition directly.
F: f04-fig3_chunkat_light.png
P: center`chunkat(1, 3)` selects one equal tile from a regular tiling grid.*center

=== UID:F04-S06
H: `subspan(.`..).at(...): explicit tile extents on a tile grid
C: subspan(...).at(...): explicit tile extents on a tile grid

=== UID:F04-S07
H: `view(.`..).from(...): explicit tile extents with explicit element offsets
C: view(...).from(...): explicit tile extents with explicit element offsets
P: This is the right tool when the tile origin is already available as element offsets rather than tile indices, or when the region is not most naturally described as an equal chunk. It is especially useful for dynamic and irregular access patterns where `chunkat(...)` would be awkward or impossible.
P: You can think of the three forms like this:
L: Use `chunkat(...)` when the surrounding parallel decomposition already defines an equal tiling. | Use `subspan(.`..).at(...) when you want an explicit tile shape and tile coordinates. | Use `view(.`..).from(...) when you want an explicit tile shape and element-space starting offsets.

=== UID:F04-S14
H: Compose: `outer # inner`
C: Compose: outer inner
P: This combines a tile coordinate with an in-tile coordinate to produce a coordinate in the full tensor.
P: Read `tr # i` as "row `i` inside tile `tr`."
F: f04-fig_compose_light.png
P: centerCompose a tile index with an in-tile offset to recover a global coordinate.*center

=== UID:F04-S15
H: Extent: `#name`
C: Extent: name
P: The notation `#tile` means the extent of the `tile` axis: how many tiles exist along that dimension. If `parallel tile by 8`, then `#tile` is `8`.
F: f04-fig_extent_light.png
P: centerPrefix `#` asks for an extent. Infix `#` composes coordinates.*center

=== UID:F15-S19
H: Bounded-ituple/integer and 'chunkat' Operation
P: 'chunkat' is an operation performed on spanned data. It creates a new *mdspan* over the existing data. Thus in certain systems, it is referred to as 'subview'. However, as 'chunkat' accepts bounded-ituple and bounded-integer as parameters, it is named differently in Croqtile.
C: Bounded-ituple/integer and chunkat Operation
P: The example above shows one typical use of `chunkat`. Here the spanned data has type `f32 [6, 10, 100]`. The operation takes two parameters --- the *integer* `p` and the *ituple* `index` --- and divides the data into $6 × 10 × 10$ pieces, one for each combination of their ranges. Each piece holds 10 consecutive elements along the least significant dimension and is copied into local storage.

=== UID:F29-S01
P: A *chunkat expression* names one tile of a larger buffer in a DMA statement, so the movement code never has to compute an offset by hand.

=== UID:F29-S02
H: *Ubound* Operation
P: In the previous chapter, you learned that a *bounded variable* defined by a `parallel-by` or `with-in` statement is either an *integer* or an *ituple* with one or more bounds. You also learned how to use the *current value* and *upper-bound value* of the *bounded variable* to construct loops in `foreach` blocks, where the *upper-bound* is implicitly used.
P: In Croqtile, a program can use the **ubound operation** to explicitly retrieve the upper bound of a bounded variable. *Ubound* is a unary operation that uses the prefix symbol `#` before a bounded variable. For example:
C: Ubound Operation
P: In this code, the local buffer `a` is shaped as `[32, 16]`, since `#size` evaluates to the upper-bound of `size`. However, the declaration of `b` uses the *current value* of `size`, resulting in an incorrect shape definition.

=== UID:F29-S03
H: Use `chunkat` Expression to Tile Data
P: In all the examples from the previous section, the *data expressions* of the *DMA statement* directly utilized the *spanned* data or *storage* annotation. However, programmers can use the **chunkat** expression to enable tiling behavior in practice. For example:
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
P: With this approach, you can apply tiling to data movement, where each iteration in each parallel thread selects only a part of the data/buffer for transfer. The Croqtile compiler uses information about the *spanned* data and the *bounded variables* to transpile the DMA statement into C++ subscripting, DMA configurations, etc., simplifying the trivial and error-prone work that C++ programmers typically face. This makes learning the Croqtile syntax worthwhile.

=== UID:F29-S06
H: In short
P: The `chunkat` expression divides a buffer into smaller chunks for a DMA statement, and the shape of each chunk follows from the bound used to select it. You explored how `chunkat` can be applied to either the source or destination of a DMA statement to manage data movement between memory and processors.
P: Now, you may try to write a full tileflow program. 3.6.1 puts the pieces together into a complete tileflow program.

=== UID:F04-S02
H: The DMA Statement Family
P: All of Croqtile's tile-movement operations use the same basic form:
C: The general DMA form
P: The important pieces are:
T: | **Part** | **Meaning** | <ROW> | --- | --- | <ROW> | `future` | Optional handle to the transfer result. Use `future.data` to access the moved tile and `future.span` to query its shape. | <ROW> | `operation` | `copy`, `transp`...>, or `pad`...>. | <ROW> | `modifiers` | Optional flags such as `.async`, `.zfill`, or `.`swiz<N>. | <ROW> | `source =` destination | The source view and the destination memory or destination view. |
P: Minimal examples:
C: A concrete copy
P: The key shift from 1.4 is that these statements move shaped regions, not individual elements.

=== UID:F04-S03
H: First Example: Tiled Addition Through shared storage
C: First Example: Tiled Addition Through shared storage
P: The arithmetic is still element-wise addition. What changed is the data path:
L: `chunkat(tr, tc)` selects one logical tile from each input. | `dma.`copy ... => shared stages that tile into block-visible fast memory. | `tr # i` and `tc # j` map tile-local coordinates back to global output coordinates.
P: This is the core Croqtile pattern: select a tile, move it, compute on the moved tile.
F: f04-fig2_tiled_add_light.png
P: centerLoad a tile, compute on the staged data, then write results back using global coordinates.*center

=== UID:F04-S08
H: Futures, `.data`, and `.span`
P: The result of a DMA or TMA statement is a future-like handle:
C: A future from a copy
P: That handle gives you two important things:
L: `f.data`: the moved tile, as a spanned tensor. | `f.span`: the shape of that tile.
P: Example:
C: Reading data and span
P: Even synchronous copies are worth naming when you want to pass the moved tile to later code.
F: f04-fig_future_data_light.png
P: centerThe transfer result is a handle. Read the copied tile through `.data` and its shape through `.span`.*center

=== UID:F04-S09
H: Synchronous and Asynchronous Copies
P: By default, copies are synchronous:
C: A synchronous copy
P: Add `.async` when you want the transfer to proceed in the background:
C: An asynchronous copy
P: The same pattern applies to TMA:
C: An asynchronous tma copy
P: Rule: if you read `f.data` from an async transfer, wait first.
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
P: In this code, the `__co__`-prefixed croqtile function accepts two inputs `lhs` and `rhs`, both with the shape `[6, 17, 128]` and the element type of `s32` (signed 32-bit integer). And the output is defined to have the same shape and type as the inputs.
P: The `parallel p by 6 .`.. block indicates that the enclosed code runs in parallel. Specifically, six instances of the code are executed concurrently. This implies a transition in the execution environment from the host to the device.
P: Inside the `parallel-by` block, a `with-in` block binds the symbol `index` to two values, `17` and `4`. In Croqtile, `index` is referred to as a `bounded-ituple` with two `bounded` values, which can be used in `foreach` statements. (You will explain `bounded` types in later chapters.)
P: The `foreach index .`.. statement is equivalent to the following C code:
C: The equivalent C++ loop
P: Within the `foreach` block, the `dma.copy` statement describes how data movement occurs. For example, consider the statement `lhs_`load = dma.copy lhs.chunkat(p, index) => local;:
L: The symbol `lhs_load` is called **future** of the DMA operation, which contains information about the DMA destination. | `dma.copy` invokes a direct DMA data transfer without transforming the shape of the data. The expression on the left-hand side of `=` represents the DMA source, while the right-hand side represents the destination. | In this case, the destination is specified as a `local` buffer, which is automatically allocated by the Croqtile compiler. | The source expression `lhs.`chunkat(p, index) is referred to as the **chunkat expression** in Croqtile. Here, `p` and `index` tiling factors for the `lhs`. Given that `lhs` has a shape of `[6, 17, 128]` and the upper bounds of `p` and `index` are `6`, `17`, and `4`, the data chunk size is `1x1x32` (`6/`6, 17/17, 128/4). In each iteration, a single data chunk is used as the source, with the exact chunk determined by the current values of `p` and `index`. For example, in parallel thread `1` and the iteration `16, 2\`, the chunk's offset is set to `1, 16, 2\`.
P: This is illustrated in the below figure:
F: f14-chunkat-6-17-4.drawio.png
P: With the DMA statement, different chunks of data tiled from `lhs` are moved iteratively and in parallel from the host to the device's `local` storage. Similarly, the DMA statement manages `rhs` by moving it in small chunks to `local` storage for processing.
P: Next, the statement `local s32 [lhs_`load.span] l1_out; defines a per-parallel-thread buffer. Note here it takes its shape from the expression `lhs_load.span`, which represents the tiled block. The buffer is used to store the output data in the subsequent `call` statement. The `call` statement invoke the *device function* named `kernel` to perform computations. Once the computation is complete, another DMA statement moves the output data from the `local` buffer back to host. This complete one iteration.
P: In this code, each parallel thread runs `17x4` iterations, with each iteration handling a `1x1x32`-sized chunk of data. The *croqtile program* terminates when all `6` parallel threads have completed their iterations. It then returns the output buffer to its caller, the host program.

=== UID:F28-S01
P: The data movement is abstracted as the most complicated statement in Croqtile. This section will introduce you the basic *DMA statement* structure and the future variable they produce.

=== UID:F28-S03
P: A DMA statement in Croqtile can be an asynchronous entity, meaning that the code following an **asynchronous DMA** executes in parallel with the DMA statement. For a *asynchronous DMA*, explicit synchronization is required before using the DMA result.
P: To support these features, the syntax is organized as follows:
C: Basic Syntax
P: The statement defines a **future**-typed variable, which appears on the left side of the operator `=`. In Croqtile, **a *future* represents both the handle of the asynchronous execution instance and the DMA result**. Therefore, the definition of *future* can be ignored for *synchronized DMA* statements.
P: The right-hand side of the **DMA operation** includes the **operation type** (`dma-op`), the **data expression** for the operation source (`src-expr`), and the *data expression* for the destination (`dst-expr`). The source and destination are separated by the symbol `=`, which indicates the direction of data flow.

=== UID:F28-S04
H: Synchronous and Asynchronous
P: In Croqtile, the `.async` suffix in a **DMA operation** indicates that the operation is asynchronous. The following code provides an example:
C: Synchronous and Asynchronous
P: In this example, the DMA operation that produces the future `f0` is asynchronous, allowing the program to continue executing while the data transfer occurs in the background. The DMA operation that produces `f1` is synchronous, meaning the program will pause until the data is fully transferred from `data1` to `shared` storage. Meanwhile, the data copy of `data0` might still be in progress.
P: Executing the statement `wait f0;` causes the program to check if `data0` has been fully transferred to `shared` storage. This synchronizes the main execution flow with the asynchronous DMA of `f0`. Therefore programmers must make sure there is no "dangling future* left in your code.

=== UID:F28-S06
H: Data Expression
P: The data expressions used so far have been simple, either a defined **spanned data** or a **storage location**. The **storage location** is used solely for the **destination buffer** declaration, as it requires the Croqtile compiler to allocate memory for it. The following code provides an example:
C: Data Expression
P: In this example, the destination of the second DMA operation is specified as the storage location `shared`. This requires the compiler to allocate storage accordingly, making it equivalent to the first DMA operation. In practice, you recommend programmers use storage locations as destinations, as Croqtile can deduce the destination shape from the **DMA statement**.
P: In practice, simple **data expressions** are rare. The next section introduces the **ChunkAt** expression to demonstrate how to implement tiling in data expressions.

=== UID:F28-S07
H: Future and Wait
P: Both *Sync-DMA* and *Async-DMA* operations can be used to define a *future*. As seen in earlier examples, a *future* acts as a handle for asynchronous operations and is typically used as a parameter of the `wait` statement to synchronize an *Async-DMA*.
P: Croqtile allows the `wait` statement to take multiple *future*s. However, using a *future* from a **Sync-DMA** in a `wait` statement will result in an error. The following code provides an example:
C: Two asynchronous copies
P: In Croqtile, another use of a *future* is to retrieve the destination buffer of the associated DMA statement. This is also the reason why **Sync-DMA** is allowed to define a future. Croqtile provides two built-in member functions for *future* variables:
L: `.span` to retrieve the **mdspan** of the DMA destination buffer. | `.data` to retrieve the reference of the DMA destination buffer (spanned data).
P: For example:
C: Waiting on both futures
P: In this example, you use `.span` built-in member function to retrieve the shape of `f0`'s destination buffer, and `.data` to retrieve the *spanned* data of `f1`'s destination. In addition, the **ElementCount** operation on the *mdspan* of the destination buffer of `f1`. The **ElementCount** operation returns the number of elements in an **mdspan**, which is useful in many scenarios.

=== UID:F28-S08
H: In short
P: You explored both synchronous and asynchronous operations, and saw how Croqtile supports advanced data handling with various *DMA Operation Types*. Finally, you discovered how futures help manage async operations and retrieve buffers, ensuring smooth data flow and synchronization.
P: In the next section, you will step further to the *Data Expression* to see how to make data tiling/blocking happen.

=== UID:F07-S06
H: Event arrays for multi-stage pipelines
P: For pipelines with multiple buffered stages, declare event arrays:
C: Event arrays for multi-stage pipelines
P: Each physical buffer slot gets its own `full`/`empty` pair. The ring index `stage = iv_`k % STAGES maps the unbounded K iteration to a fixed number of physical slots. With four stages, the producer can run several tiles ahead before blocking on `wait empty`.

=== UID:F07-S07
H: Bootstrap protocol
P: The consumer must **seed** the `empty` credits before the K-loop starts:
C: Bootstrap protocol
P: Without this bootstrap, the producer's first `wait empty[0]` blocks forever --- a deadlock, not a mysterious MMA bug. This is a common pitfall: every bounded buffer protocol requires initial credits.
P: 3.8.5 develops the full double-buffered and multi-stage pipeline kernels that put these primitives to work. The examples there compose `inthreads.`async, events, `swap`, and `mma.commit` into complete, runnable matmul pipelines.

=== UID:F34-S06
H: Parallel Perspective of Tileflow Programs
P: In Croqtile, a *divergent* code region is generated with the keyword `inthreads`, where a compare expression followed.
P: Let us have a look at the following example:
C: Parallel Perspective of Tileflow Programs
P: You may now re-examine Croqtile tileflow programs from the perspective of parallelism. It consists of
L: *Sequential Code* | *SPMD-Styled Parallel Code* enclosed in `parallel-by` block, which consists of: itemize[leftmargin=1.6em] | *Uniform Code* | *Divergent Code* enclosed in `inthreads` block.
P: In this example, `p `= 2 initiates the *divergent code block*, where from the implementation perspective, it creates a thread mask of `000111`, which allows only thread `0`, `1` and `2` to execute the code inside. And thread `3`, `4` and `5` will skip this path. **All threads are forced synchronized after the `inthreads` block**. In this case, it implies thread `3`, `4`, and `5` waits for the completion of the others before stepping forward.
P: In Croqtile, the `inthreads` block looks like a `if` conditional block in C/C++. However, `inthreads` explicit claims the code divergence, which is a concept in SPMD program, rather than the conditional block for sequential code. Therefore, the `inthreads` block:
L: Can only appear in the SPMD code region, which is enclosed in the `parallel-by` block. | Can only have a comparison result related to thread identifiers.
P: Croqtile compiler has the checks for both the comparison expression and `inthreads`-block placement.

=== UID:F34-S09
H: Implicit Masking
P: In Croqtile, code between multiple `parallel-by` levels can be considered as a type of implicit masking. The below code showcases an example:
C: Implicit Masking
P: In this example, there exists two levels of parallelization. The code inside `parallel p by 6` block but outside `parallel q by 2` blocks is conceptually for the outer parallelization only. This means, only one thread from the sub-level parallelization execute the code, which is equivalent to `inthreads (q == 0) .`.. . Programmers must be aware there is only single thread instance to execute the code part. Only the code inside the inner `parallel-by` is executed by all threads.

=== UID:F35-S01
P: As `inthreads` introduce MPMD programming pattern in Croqtile, it is necessary to synchornize between different asynchronous code. You can use event in Croqtile to achieve that.

=== UID:F35-S02
H: Event Basics
P: There are different definitions of `event` in different programming language. In Croqtile, the **Event** is used to handle communication and interaction between asynchronous code blocks at different level.
P: An event in Croqtile is either *SET* or *UNSET*. When declaring an event, it is initialized with the *UNSET* state. For example, the following code initialize a `shared` event:
C: Declaring a shared event
P: A `shared` event can only be declared inside a `parallel-by` block and used by parallel threads. Similarly, `global` and `local` events can be defined as well. There is normally restrictions syntactially for either the declarations and value references according to the target platforms.
P: In addition, you may declare an event array following C-style syntax:
C: Declaring an event array
P: A comma-separated expression can declare both an event and an event array on one line.
P: In Croqtile, the `trigger` statements changes the `event` from *UNSET* to *SET* state:
C: Triggering an event array
P: The `trigger` statement triggers each event in an event array `e`, as well as the single event `e1`, as showed in the comma-separated list.
P: Similarly, the `wait` statement can wait for an event in a blocking mannar. The wait statement blocks currentthreads for execution if the event is *UNSET*, until the event state is changed to *SET*.
P: Triggering an event is what unblocks the code waiting on a *wait statement*. For example,
C: Waiting on an event
P: You have observed similar code in last section, except for that there are `wait` and `trigger` statements. In this code snippet, once the event `e` is triggered by the *trigger-thread*, the *wait-thread* is no longer blocking. As a consequent, the two asynchronous threads followed an order to reach the synchronization point defined by `sync.shared`, where thread `0` completes its `inthreads` block before thread `1`. From another perspective, the events here **chains the asynchronous `inthreads`**. This is useful in many scenarios.
P: In Croqtile semantics, **an event is always auto-reset by its `wait` statement**. That means when a `wait` statement is unblocked, the events that are waited are changed to *UNSET* state.

=== UID:F35-S05
H: In short
P: Events let asynchronous `inthreads` blocks agree on an order. The `trigger` statements can set events and event arrays to *SET* state, while `wait` statements block execution, and reset events to *UNSET* until they are triggered.
P: Programmers must be careful to use Croqtile events, or else it is possible to result in deadlocks, which is hard to debug.

=== UID:F07-S02
H: Predicated execution with `if`
P: Croqtile's `if` behaves like its C counterpart:
C: Predicated execution with if
P: All threads in scope evaluate the condition. Threads where it is false skip the body. --- Threads on the skipped side sit idle while the taken side runs, then vice versa. This is the price of runtime flexibility.
P: The practical rule: keep `if` conditions (all 32 threads agree) whenever possible.

=== UID:F07-S12
H: Early return from `parallel`
P: Sometimes you need to early return from a `parallel` block. The `yield` keyword generates a `return;` instruction that leaves the device code for that instance.
C: Early return from parallel
B: The `yield` keyword currently will **not** do any compile-time nor runtime checks. Be careful about synchronization safety.

=== UID:F06-S02
P: Every contraction kernel follows the same rhythm, and the spelling makes each step visible:
L: **`mma.fill 0.0`** | **`mma.load`** | **`mma.`row.**row mc, ma, mb --- issue the contraction: **C += A $×$ B** into `mc`. | **`mma.`store mc, dst**
P: You loop steps 2--3 over K (loading the next K-slice each iteration, accumulating into the same `mc`), then run step 4 once to flush the completed output tile. --- You do not declare per-lane layouts; the compiler derives them from the target and your layout choice (`row.row` here).
P: The `.row.row` suffix is a **layout contract** --- Both operands are row-major. The full set of layout combinations is `row.row`, `row.col`, `col.row`, and `col.col`; in practice, `row.row` and `row.col` cover most kernels. Choosing the wrong variant is a correctness bug, not a performance hint

=== UID:F10-S05
H: `__cpp__`: verbatim C++ injection
P: Whatever you place there must be valid at the splice point. The Croqtile compiler does not parse or rewrite the contents.
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

=== UID:F10-S11
H: Conditional code paths
C: Conditional code paths
P: The preprocessor keeps one branch and discards the other before Croqtile parses the `__co__` body. Command-line defines: `croqtile kernel.`co -DMATMUL_TILE_K=128.

=== UID:F15-S20
H: Function Calls and Call Croqtile Function
P: Croqtile functions are not allowed to call another Croqtile function. However, inside a Croqtile function, it is normal to have function calls to C++ kernels.
C: Function Calls and Call Croqtile Function
P: In the above example, program calls the existing C++ function 'bar' using Croqtile keyword 'call', which is intuitive.

=== UID:F23-S01
P: Macros have long been a fundamental feature in C/C++ programming. To enable seamless manipulation of both C++ and *tileflow* programs, macro expansion is essential. This section demonstrates Croqtile's macro processing capabilities.

=== UID:F23-S02
H: Object-Like Macros
P: Croqtile preprocessor supports C/C++ **object-like macros** to connect both host code and tileflow code. The *object-like macro* performs pure text replacement and accepts no parameters. The below code gives an example:
C: An object-like macro
P: In the code snippet, the inputs of croqtile function `matmul` are not dynamically shaped. Instead, Croqtile pre-processor substitutes `M`, `N`, `K` with values `256`, `32`, `64` ahead of Croqtile compilation, which produces tileflow function with statically shaped inputs. In this way, programmers makes different code consistent and easy to manipulate.
P: The preprocessor currently supports only macros without parameters, so code like:
C: A macro with parameters
P: would not work.

=== UID:F23-S03
H: Comments
P: Croqtile support C-style comments, either `/*...*/` or `//...`, leveraging the capability croqtile preprocessor has provided.

=== UID:F23-S04
H: Conditional Compilation
P: Croqtile also support conditional compilation which C/C++ programs used a lot. This includes `#if`/`#ifdef`/`#ifndef`/`#else`/`#endif`. The below code snippet showcases the usage:
C: Conditional Compilation
P: In this way, it also make host and croqtile code be controlled within the same preprocess method.
P: The preprocessor is still being extended, and full C preprocessing support is not planned: a C feature is picked up only when a program actually needs it.

=== UID:F24-S01
P: In current implementation of Croqtile functions, it invokes *device function* to implement the specified target computations. This section will learn the detail about the *call statement*s in Croqtile.

=== UID:F24-S02
H: Call Device Functions
P: To call a device function in the Croqtile function, it requires the call statement explicitly led by `call`.

=== UID:F24-S03
H: Basic Syntax of Call Statements
P: The general syntax for a `call` statement in Croqtile is:
C: The general call syntax
P: Here, the keyword `call` is followed by a function name `func-name`, an optional ``> enclosed template arguments, which are comma-separated. The `arguments` are also comma-separated and listed inside `()` like normal C/C++ functions.
P: In the following example, `call bar();` in `mou` is illegal code since it tries to call device function in a code location which is assumed to be host code area.
C: An illegal call site

=== UID:F24-S04
H: The Convention: Allowed Argument Types
P: In current implementation, the argument must be either:
L: *Spanned Data* type, or | Scalar *Integer* type, or | Scalar *Floating-Point* type.
P: Here, the scalar *integer* type only includes `int`, whereas the scalar *floating-point type* includes `float`, `double`, `half`, `bfp16`, and `half8`. However, since the target-platform differs in floating-point support, the *floating-point type* parameter also varies for different target platforms.
P: In addition, **operations on floating-point are not supported** by Croqtile till now, considering the floating is not useful for tileflow programs. Instead, they are typically values for device computation, which the *tileflow program*s pass such values directly to device kernels.
P: The below code showcases an example:
C: The Convention: Allowed Argument Types
P: In this example, it passes different data types as arguments from Croqtile function to the device functions, which matches the device parameters exactly. Note that in the device function, the **corresponding parameter type of the *spanned data* argument is simply the pointer of its _element type_**, where the shape information is dropped. For example, the `foo`'s argument `p` is `float*`, which corresponds the *spanned data* argument `input.data`. In Croqtile, you names the parameter as a **decayed** pointer of the *spanned data*. If using other form like `float p[]`, it will trigger failure in target compilation stage.
P: For some types like `f16`, and `bf16`, there may not be native target support of such types, it is possible to utilize `croqtile::f16` and `croqtile::bf16` to handle such types.

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
P: Each call takes a comma-separated mix of string literals and expressions. The output order across threads is **nondeterministic**
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
P: The next chapter builds on this by overlapping movement with compute. Once you can think of a kernel as a loop that repeatedly stages tiles and computes on staged tiles, double buffering and pipelining become natural extensions rather than new ideas.

=== UID:F05-S18
H: Summary of New Syntax
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `parallel p by N` | N-way parallelism, index `p` runs 0..N-1 concurrently | <ROW> | `parallel px, py\` by [M, N] | Multi-dimensional parallel level <br> (`p` and `q` correspond to one parallel level) | <ROW> | `parallel p by A, q by B` | Comma-separated parallel levels <br> (`p` and `q` correspond to different parallel levels) | <ROW> | `parallel q by N :` thread | Map to threads within a block | <ROW> | `foreach index = m, n, k\` in [a, b, c] | Named tuple destructuring in a `foreach` | <ROW> | `lhs.span(0)` | Extract one dimension of a tensor's shape | <ROW> | `p # m` | Compose outer tile index `p` with inner offset `m` |
P: 3.7 replaces the scalar `.at()` arithmetic in the inner loop with contraction operations

=== UID:F07-S13
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `if (expr) .`.. | Predicated execution --- runtime conditional, divergent threads masked | <ROW> | `inthreads.`async (condition) | Structured concurrent region --- compile-time thread partitioning | <ROW> | `shared event name[N]` | Declare N synchronization tokens | <ROW> | `trigger name` | Signal that a condition is met | <ROW> | `wait name` | Block until the corresponding `trigger` fires | <ROW> | `tile_`id = tile_iter # block_id | Compose indices for tile striping | <ROW> | `int total_`tiles = expr | Local integer variable | <ROW> | `yield` | Early return from `parallel` block |

=== UID:F08-S14
H: Syntax at a glance
T: | **Syntax** | **Meaning** | <ROW> | --- | --- | <ROW> | `dma.`copy.async src => dst | Non-blocking copy (returns immediately) | <ROW> | `dma.any` | Placeholder future (no transfer in flight yet) | <ROW> | `swap(f0, f1)` | Exchange two future handles without copying data | <ROW> | `rotate(f0, f1, f2)` | Cycle three future handles | <ROW> | `with tile_`k in N ... | Scoped tile axis binding with extent N | <ROW> | `foreach tile_`k(1:) | Iterate starting from index 1 | <ROW> | `_`_co__ auto fn(...) | Return type inferred from `return` statement |

