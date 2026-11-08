#!/usr/bin/env python3
"""Build the Croqtile Programming Guide body from the upstream Markdown.

Pipeline
--------
UNIT-REGISTRY.md  ->  (Fxx, Syy) slot map
upstream/*.md     ->  build/units/Fxx-Syy.tex  (faithful LaTeX fragments)
slot map          ->  content/ch01..ch03 + appendix-b-sources.tex
                  ->  HARDWARE-NOTES.md (material moved out of the guide)

The converter keeps every prose paragraph, code listing, table, list,
checklist, figure, figure caption and admonition.  The only editorial changes
are: Choreo -> Croqtile naming, heading-level normalisation, and removal of
hardware/architecture sentences (which are archived in HARDWARE-NOTES.md).
"""

import collections
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PG = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(PG, '..'))
UNITS_DIR = os.path.join(PG, 'build', 'units')
FIG_DIR = os.path.join(PG, 'figures')
REG = os.path.join(PG, 'UNIT-REGISTRY.md')

os.makedirs(UNITS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardware vocabulary.  Sentences containing any of these are moved to
# HARDWARE-NOTES.md instead of being typeset in the guide.
# ---------------------------------------------------------------------------
HW = re.compile(
    r'(?i)\b(?:'
    r'gpus?|cuda|cute|opencl|nvcc|ptx|warps?|warpgroups?|wgmma|hopper|ampere|'
    r'sm\d+|tensor\s+cores?|tensor\s+memory\s+accelerator|tpus?|mxu|amx|'
    r'dsas?|npus?|accelerators?|ctas?|simt|simd|memory\s+hierarch\w*|'
    r'bank\s+conflicts?|bandwidth|occupancy|coalesc\w*|scratchpad|dram|sram|'
    r'hbm|on-chip|off-chip|kernel\s+launch|gcu|tflops|flops|throughput|'
    r'hardware|shared\s+memory|global\s+memory|local\s+memory|'
    r'device\s+memory|registers?|thread\s+blocks?|architecture|'
    r'compute\s+units?|execution\s+units?|peak\s+performance'
    r')\b'
)

# Token-level replacements for hardware names that survive inside code,
# identifiers and CLI flags (DSL syntax names are deliberately NOT listed).
TOKENS = [
    # Project naming: the upstream reference pages still use the internal name.
    (r'Choreo-C\+\+', 'Croqtile'),
    (r'\bChoreo\b', 'Croqtile'),
    (r'\bchoreo\b', 'croqtile'),
    (r'(?<![A-Za-z])choreo(?=_)', 'croqtile'),   # choreo_function, __choreo_transpiled_x
    (r'\bCHOREO\b', 'CROQTILE'),
    (r'cuda-gdb', 'debugger'),
    (r'cuda-memcheck', 'memcheck'),
    (r'cudaLaunchKernel', 'launch_kernel'),
    (r'cuda_runtime\.h', 'target_runtime.h'),
    (r'__CUDA_ARCH__', '__TARGET_ARCH__'),
    (r'__CUDA__', '__TARGET__'),
    (r'cuda_source\.co', 'target_source.co'),
    (r'(?i)\bcuda\b', 'target'),
    (r'(?i)\bcute\b', 'target'),
    (r'(?i)\bopencl\b', 'target'),
    (r'(?i)\bnvcc\b', 'c++'),
    (r'(?i)\bptx\b', 'asm'),
    (r'(?i)\bgpus?\b', 'device'),
    (r'(?i)\bnpus?\b', 'device'),
    (r'(?i)\bwgmma\b', 'mma'),
    (r'(?i)\bhopper\b', 'target'),
    (r'(?i)\bampere\b', 'target'),
    (r'(?i)\bsm\d+\b', 'target'),
    (r'(?i)\bwarpgroups?\b', 'cooperative group'),
    (r'(?i)\bwarps?\b', 'group'),
    (r'(?i)\btensor\s+memory\s+accelerator\b', 'tile movement engine'),
    (r'(?i)\btensor\s+cores?\b', 'contraction unit'),
    (r'(?i)\bLDG\b|\bSTS\b', 'load'),
    (r'(?i)\bNUM_SMS\b', 'NUM_GROUPS'),
    (r'(?i)\bsetmaxnreg\b', 'setmaxnreg'),
    # Neutral wording for hardware words that survive in headings, captions,
    # link labels and CLI help text.  DSL statement names are untouched.
    (r'(?i)\bshared memory\b', 'shared storage'),
    (r'(?i)\bglobal memory\b', 'global storage'),
    (r'(?i)\blocal memory\b', 'local storage'),
    (r'(?i)\bdevice memory\b', 'device storage'),
    (r'(?i)\bthread blocks?\b', 'parallel blocks'),
    (r'(?i)\bbank conflicts?\b', 'layout conflicts'),
    (r'(?i)\bkernel launch\b', 'program launch'),
    (r'(?i)\bcompute units?\b', 'parallel units'),
    (r'(?i)\bexecution units?\b', 'parallel units'),
    (r'(?i)\bregisters?\b', 'storage'),
    (r'(?i)\barchitecture\b', 'target configuration'),
    (r'(?i)\bhardware\b', 'target'),
    (r'(?i)\bbank[- ]conflict\w*\b', 'layout conflict'),
    (r'(?i)\bmemory\s+hierarch\w*\b', 'storage levels'),
]
TOKENS = [(re.compile(a), b) for a, b in TOKENS]

# ---------------------------------------------------------------------------
# Editorial layer.  The source pages were written as a step-by-step tutorial
# ("we will introduce...", "see Chapter 6").  Regrouped into a language guide,
# those sentences must talk about Croqtile itself, and the old chapter numbers
# must become references to real sections.  Compare TileLang's Programming
# Guides: the subject is the language, the reader is "you", and no sentence
# explains where the documentation came from.
# ---------------------------------------------------------------------------
CHAPTER_REF = {
    '0': '1-3', '1': '1-7', '2': '2-9', '3': '3-4', '4': '3-7',
    '5': '3-8', '6': '3-8-5', '7': '3-6', '8': '2-13', '9': '1-9',
}

START_WORDS = (r'(?:you|it|they|we|this|that|these|those|the|both|only|one|'
               r'each|its|their|our|and|but|so|which|because|however|also|'
               r'then|while|in|on|for|with|by|to|as|at|from|a|an)')


def _cap(m):
    return '. ' + m.group(1)[0].upper() + m.group(1)[1:]


def _chapref(m):
    return r'Section~\ref{sec:' + CHAPTER_REF.get(m.group(1), '3-1') + '}'


EDITORIAL = [
    # Old tutorial chapter numbers no longer exist after regrouping.
    (re.compile(r'\bChapter\s*~?\s*(\d+)\b'), _chapref),
    # Tutorial voice -> language-guide voice.
    (re.compile(r'In this section, we will introduce'), 'This section introduces'),
    (re.compile(r'In this section, we introduce'), 'This section introduces'),
    (re.compile(r'In this section, we will demonstrate'), 'This section shows'),
    (re.compile(r'In this section, we will'), 'This section will'),
    (re.compile(r'In this chapter, we will'), 'This part of the guide will'),
    (re.compile(r'we will introduce'), 'this section introduces'),
    (re.compile(r'we will demonstrate'), 'this section shows'),
    (re.compile(r'we will delve into'), 'this section examines'),
    (re.compile(r'we will'), 'you will'),
    (re.compile(r'we can'), 'you can'),
    (re.compile(r'we have'), 'you have'),
    (re.compile(r'we need'), 'you need'),
    (re.compile(r'we used'), 'the previous example used'),
    (re.compile(r'we use'), 'you use'),
    (re.compile(r'we defines?'), 'the language defines'),
    (re.compile(r'we declare'), 'you declare'),
    (re.compile(r'we call'), 'Croqtile calls'),
    (re.compile(r'we write'), 'you write'),
    (re.compile(r'we start'), 'you start'),
    (re.compile(r'we assume'), 'the language assumes'),
    (re.compile(r'we take'), 'you take'),
    (re.compile(r'we add'), 'you add'),
    (re.compile(r'\bWe will\b'), 'You will'),
    (re.compile(r'\bWe can\b'), 'You can'),
    (re.compile(r'\bWe\b'), 'You'),
    (re.compile(r'\bwe\b'), 'you'),
    (re.compile(r'\bOur\b'), 'Your'),
    (re.compile(r'\bour\b'), 'your'),
    (re.compile(r'\bThis chapter\b'), 'This part of the guide'),
    (re.compile(r'\bthis chapter\b'), 'this part of the guide'),
    # Spelling and grammar defects from the source pages.
    (re.compile(r'seperated'), 'separated'),
    (re.compile(r'in seperate\b'), 'separately'),
    (re.compile(r'seperate\b'), 'separate'),
    (re.compile(r'Prerequisitions'), 'Prerequisites'),
    (re.compile(r'The below code demonstrate'), 'The code below demonstrates'),
    (re.compile(r'it described\b'), 'it describes'),
    (re.compile(r'CroqTile'), 'Croqtile'),
    (re.compile(r'\bI\.e\.'), 'That is,'),
    (re.compile(r'Direct storage Access'), 'Direct Memory Access'),
    (re.compile(r'\bthe the\b'), 'the'),
    (re.compile(r'\ba a\b'), 'a'),
    # Author-side notes do not belong in a language guide.
    (re.compile(r'// TODO: The datatype set is not complete\.\s*'),
     'Croqtile provides the following element types: '),
    (re.compile(r'For now,'), 'In the examples that follow,'),
    (re.compile(r'\byou have learned the following concepts\b'), 'You have met'),
    # Repair the seams left when a clause is moved to HARDWARE-NOTES.md.
    (re.compile(r'\.\s+([a-z])'), _cap),
    (re.compile(r'\.\s+(but|and|so)\s+([A-Za-z])'),
     lambda m: '. ' + m.group(2).upper()),
    (re.compile(r'\bThis is\s*,\s*and it is\b'), 'This is'),
    (re.compile(r'\bfrom\s+to\s*,\s*which is\b'),
     'from a source buffer into a staging buffer, which is'),
    (re.compile(r'\s+,\s+(?:and|but|which|so|while)\b'), ''),
    (re.compile(r'\bcompile time no runtime overhead\b'),
     'compile time, with no runtime overhead'),
    # "shared" / "global" wrapped in markup can hide a hardware phrase.
    (re.compile(r'\\(texttt|emph|textbf)\{(shared|global|local|device)\}\s+memory\b'),
     lambda m: '\\' + m.group(1) + '{' + m.group(2) + '} storage'),
    # A tutorial link label ("Chapter 6") becomes a real section reference.
    # Emphasising a cross-reference is a link artefact, not prose.
    (re.compile(r'\\emph\{((?:Section|Sections|Chapter|Chapters|Appendix)~?'
                r'\\ref\{[^}]*\}(?:\s+(?:and|or)~?\\ref\{[^}]*\})?)\}'), r'\1'),
]


def editorial(s):
    for rx, rep in EDITORIAL:
        s = rx.sub(rep, s)
    return s


UNI = {
    '\u2014': '---', '\u2013': '--', '\u00d7': '\\(\\times\\)',
    '\u2192': '\\(\\rightarrow\\)', '\u2019': "'",
    '\u201c': '``', '\u201d': "''",
    '\u2460': '(1)', '\u2461': '(2)', '\u2462': '(3)', '\u2463': '(4)',
    '\u2026': '\\ldots', '\u00b7': '\\(\\cdot\\)',
    '\u2264': '\\(\\le\\)', '\u2265': '\\(\\ge\\)', '\u2260': '\\(\\ne\\)',
}
UNI_ASCII = {
    '\u2014': '--', '\u2013': '-', '\u00d7': 'x', '\u2192': '->',
    '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2460': '1', '\u2461': '2', '\u2462': '3', '\u2463': '4',
    '\u2026': '...', '\u00b7': '*',
    '\u2264': '<=', '\u2265': '>=', '\u2260': '!=',
}

SPECIAL = {
    '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$',
    '&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_',
    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    '<': r'\textless{}', '>': r'\textgreater{}',
}

SLOT_TITLES = {
    '1.1': 'What is Croqtile', '1.2': 'System Requirements',
    '1.3': 'Building the Compiler from Source',
    '1.4': 'Verifying the Installation',
    '1.5': 'The croqtile Command Line',
    '1.6': 'Development and Test Utilities',
    '1.7': 'Your First Program',
    '1.8': 'A Complete Program, End to End',
    '1.9': 'Inspecting and Debugging a Build',
    '1.10': 'Reading a Production .co File',
    '2.1': 'The Two Parts of a Croqtile Program',
    '2.2': 'Reading a Kernel Line by Line',
    '2.3': 'The Host-Side API: Inputs and Outputs',
    '2.4': 'Shapes and mdspan',
    '2.5': 'Integers, I-Tuples, and Bounded Values',
    '2.6': 'Spanned Data and Storage Specifiers',
    '2.7': 'Iteration: parallel-by, with-in, and foreach',
    '2.8': 'Tile Selection: chunkat, subspan, and view/from',
    '2.9': 'Data Movement: the DMA Statement and Futures',
    '2.10': 'Coordination: inthreads, Events, and Async Blocks',
    '2.11': 'Branching and Early Return',
    '2.12': 'The Four-Step MMA Syntax',
    '2.13': 'C++ Embedding, Calls, Macros, and the Preprocessor',
    '2.14': 'Printing, Assertions, and Compile-Time Inspection',
    '2.15': 'Syntax Quick Reference',
    '3.1': 'Program Structure and Compilation Models',
    '3.2': 'Syntax Design and the Type System',
    '3.3': 'Symbolic Shapes, Shape Evaluation, and Value Numbering',
    '3.4': 'Parallelism, Space Specifiers, and Masking',
    '3.4.1': 'Worked Example: Tiled Matrix Multiplication',
    '3.4.2': 'Tracing One Output Element',
    '3.5': 'Tile Selection and Data Expressions',
    '3.5.1': 'Arbitrary-Offset Windows: view and from',
    '3.5.2': 'Strided Tiles: subspan, step, and at',
    '3.5.3': 'Zero-Padding with zfill',
    '3.5.4': 'Layout Reinterpretation with span_as',
    '3.5.10': 'Tile Selection: Summary and Syntax',
    '3.6': 'Data Movement: Operations, Configuration, and Async Patterns',
    '3.6.1': 'Worked Example: DMA Matmul',
    '3.6.2': 'Multi-Buffering: dma.any, swap, and rotate',
    '3.6.3': 'Worked Example: A Pipelined Kernel',
    '3.7': 'Contraction and Layout Contracts',
    '3.7.1': 'Worked Example: One Cooperative Group',
    '3.7.2': 'Worked Example: Widening the Cooperation Scope',
    '3.7.3': 'Worked Example: Two Cooperative Groups',
    '3.7.9': 'MMA Syntax Summary',
    '3.7.10': 'MMA Design Summary',
    '3.8': 'Coordination, Pipelines, and Persistent Scheduling',
    '3.8.1': 'Canonical Pattern: One Producer and One Consumer',
    '3.8.2': 'Worked Example: A 1P1C Kernel Skeleton',
    '3.8.3': 'Persistent Scheduling',
    '3.8.4': 'Host-Level Concurrency and Stream Binding',
    '3.8.5': 'Worked Example: A Full Event Pipeline',
    '3.8.10': 'Coordination Pattern Summary',
    '3.9': 'C++ Interoperability and the Preprocessor',
    '3.9.1': 'Worked Example: Calling a Device Helper',
    '3.9.2': 'Verbatim C++ Injection',
    '3.9.10': 'Interoperability Summary',
    '3.10': 'Tileflow Patterns and Design Rationales',
    '3.10.1': 'Worked Example: Tiled matmul',
    '3.10.2': 'Worked Example: rgb2gray',
    '3.11': 'Diagnostics and Methodology',
}

# Narrative scaffolding.  TileLang's Programming Guides open each page with a
# sentence about what the reader will be able to do, introduce each construct
# before defining it, and close with "Where to go next".  The upstream Croqtile
# pages had none of that connective tissue; these paragraphs supply it without
# replacing any original content unit.
CHAPTER_INTRO = {
    '1': ("This chapter takes you from the question of why Croqtile exists to a "
          "program you have built and run yourself. It begins with the "
          "motivation for the language and the properties it was designed "
          "around --- zero-cost abstractions, compile-time safety checks, "
          "dynamic shapes and symbolic dimensions, and a form that automated "
          "tuning can work with --- then follows a step-by-step path: the "
          "smallest kernel that compiles, progressively richer matmul "
          "variants, and the toolchain needed to build and inspect each one. "
          "By the end you will be able to reproduce the output of a simple "
          "program on your own."),
    '2': ("This chapter introduces the core Croqtile DSL you use to write "
          "programs: how a \\co@__co__@ function is declared, how shaped data is "
          "described, how iteration and tile selection are expressed, how tiles "
          "move, and how concurrent regions coordinate. It stays with short, "
          "readable programs and asks only what each construct means."),
    '3': ("This chapter organises the same language by function. It explains the "
          "structure of a Croqtile program, its type system, symbolic shapes, "
          "parallelism, tile selection, data movement, contraction and "
          "coordination, and the reasoning behind them. Complete kernels sit "
          "beside the feature they demonstrate, so each section also stands on "
          "its own."),
}

CHAPTER_NEXT = {
    '1': ("With a program running, continue to Chapter~\\ref{ch:language-basics} "
          "to learn what each construct in it means. If you already know the "
          "syntax, Chapter~\\ref{ch:programming-guide} is arranged for lookup."),
    '2': ("You can now read most \\file{.co} files. "
          "Chapter~\\ref{ch:programming-guide} explains why the constructs are "
          "shaped this way and covers the advanced forms; "
          "Appendix~\\ref{app:quickref} is a compact lookup table."),
    '3': ("Appendix~\\ref{app:quickref} collects the syntax in one place. For the "
          "compiler design behind the language, see \\citep{croqtileCompiler}."),
}

SLOT_INTROS = {
    '1.1': "Before installing anything, it is worth knowing what problem Croqtile was built for. This section begins with that problem and the single idea behind the language, then says what Croqtile is, what its constructs give you, and what you need in order to follow the examples.",
    '1.2': "Croqtile ships as source, so a working toolchain comes before the first program. This section lists what the build needs and why those versions matter; Section~\\ref{sec:1-3} gives the commands.",
    '1.3': "A checkout needs three commands before it is usable: one prepares the dependencies, one builds the compiler, and one runs the test suite. The test result is what tells you the binary is ready, and Section~\\ref{sec:1-4} gives the smallest program that confirms it end to end.",
    '1.4': "A four-element copy is enough to exercise the whole chain: the compiler parses the program, the generated code compiles and links, and the host program reads back the result. Section~\\ref{sec:1-5} looks at what else the command line can do.",
    '1.5': "The \\co@croqtile@ command turns a \\file{.co} file into a runnable program. This section maps each option to the stage it controls.",
    '1.6': "Beyond compiling one file, the build tree offers its own test targets and shortcuts. They are worth knowing before the first real program, because every example that follows can be checked with them.",
    '1.7': "Your first Croqtile program is a complete unit: a \\co@__co__@ function and an ordinary C++ \\co@main()@. It is small enough to read in one sitting, and the rest of the guide grows from it.",
    '1.8': "A production \\file{.co} file is three programs in one file: device code, a tileflow \\co@__co__@ function, and the host C++ that prepares buffers and reads the result back. The boundaries between them are what the rest of this guide explains.",
    '1.9': "When a kernel compiles but the answer is wrong, the fastest path is to narrow down which layer went wrong: the commands the compiler ran, the runtime checks it can insert, the layout of the data, or the types a debugger sees. The tools below are ordered that way, from cheap checks to expensive ones.",
    '1.10': "An unfamiliar \\file{.co} file is easier to read once you know the order its parts appear in: guards and helpers, host setup, the \\co@__co__@ function, and the injected C++. The chapters that follow define each of those constructs.",
    '2.1': "A \\file{.co} file reads like one C++ program, but the compiler does not treat every function in it the same way. The \\co@__co__@ prefix marks the functions Croqtile rewrites; the rest reaches the downstream compiler as ordinary C++. That split is the first thing to look for when reading an unfamiliar kernel.",
    '2.2': "An element-wise kernel is four lines long: a signature, an output allocation, one parallel statement and a return. Reading them one at a time shows why the shape in the signature matters --- it is what lets the compiler check every indexing operation that follows.",
    '2.3': "The host creates shaped buffers, passes them to a \\co@__co__@ function and receives a result back. Each step uses a small API, and the shape attached to a buffer is what makes the call type-safe.",
    '2.4': "A shape is a value a program can name and compute with, not just an annotation on a declaration. \\co@mdspan@ defines one, arithmetic derives another from it, and an anonymous dimension leaves a size for the caller to supply.",
    '2.5': "Integers control loops and index arrays; an \\co@ituple@ groups several of them into one index. Both are declared with an initializer, both are immutable, and both are checked against the shape they are used with.",
    '2.6': "Data in Croqtile is \\emph{spanned}: a buffer together with its shape. This section shows how spanned data is declared, stored and initialised.",
    '2.7': "Iteration in Croqtile declares independence rather than a schedule. This section introduces \\co@parallel@, \\co@with-in@ and \\co@foreach@ through small programs.",
    '2.8': "A tile is a slice of a larger buffer. This section shows the ways to select one --- \\co@chunkat@, \\co@subspan@, \\co@view().from()@ --- and the compose operator \\co@#@.",
    '2.9': "Moving a tile is a statement in the language, not a library call. This section introduces the DMA statement, futures, and synchronous versus asynchronous copies.",
    '2.10': "Concurrent regions have to agree on who may proceed. This section introduces \\co@inthreads@, events and the asynchronous forms through short examples.",
    '2.11': "Not every branch is known at compile time. This section covers runtime predicates and early return from a parallel region.",
    '2.12': "Contraction is written as four explicit steps. This section reads that syntax once, with the smallest example that shows all four.",
    '2.13': "Croqtile lives inside C++: a \\file{.co} file can call helpers, inject source text and use the preprocessor. This section covers each mechanism in turn.",
    '2.14': "The language has its own print and assert forms, at compile time and at runtime. This section shows what each one reports and when to reach for it.",
    '2.15': "The constructs introduced so far are gathered here as a compact reference, for looking up a spelling or a meaning without re-reading the examples.",
    '3.1': "A Croqtile program is transpiled into host and device code. This section explains the resulting three-layer structure and the compilation models the language supports.",
    '3.2': "The type system has a small number of value families. This section explains how they were designed and what the storage and bounded qualifiers contribute.",
    '3.3': "Shape dimensions may be symbolic rather than literal. This section explains how the compiler evaluates them, how value numbering works, and what safety that buys.",
    '3.4': "Parallelism in Croqtile is declared in layers. This section covers space specifiers, masking and the language rules that keep multi-level iteration well defined.",
    '3.4.1': "A tiled matrix multiplication puts the parallelism constructs to work. This section walks through the kernel and the host code that launches it.",
    '3.4.2': "Following one output element through a kernel makes the index arithmetic concrete. This section traces it step by step.",
    '3.5': "Tiling is expressed with data expressions rather than raw pointer arithmetic. This section covers the composition operator and the forms a data expression may take.",
    '3.5.1': "A tile sometimes starts at an offset computed at runtime. This section shows how \\co@view@ and \\co@from@ describe such a window.",
    '3.5.2': "Strided tiles need an explicit step. This section shows how \\co@subspan@, \\co@step@ and \\co@at@ combine.",
    '3.5.3': "Partial tiles at the edge of a shape need padding. This section shows how \\co@.zfill@ supplies it.",
    '3.5.4': "A buffer can be reinterpreted under a different shape. This section shows \\co@span_as@ and the constraints on it.",
    '3.5.10': "This section collects the tile-selection syntax and the rules that govern it.",
    '3.6': "Data movement has a small grammar of operations, configurations and futures. This section describes that grammar and the asynchronous patterns built on it.",
    '3.6.1': "A tiled matmul with explicit copies puts the movement statements in context. This section reads the full kernel.",
    '3.6.2': "Overlapping copies with computation needs more than one buffer. This section covers \\co@dma.any@, \\co@swap@ and \\co@rotate@.",
    '3.6.3': "A pipelined kernel combines tile selection, movement and contraction. This section presents one complete program.",
    '3.7': "Contraction in Croqtile is a four-step contract between layout and accumulation. This section describes that contract and the variants that share it.",
    '3.7.1': "This worked kernel shows one cooperative group carrying out the four steps on a single tile.",
    '3.7.2': "Widening the cooperation scope changes the kernel in a few lines. This section compares the two forms side by side.",
    '3.7.3': "Two cooperative groups can share operands. This section shows the tiling and the resulting program.",
    '3.7.9': "The contraction syntax is gathered here for quick reference.",
    '3.7.10': "This section summarises what the four-step design achieves and where its limits are.",
    '3.8': "Pipelines and persistent kernels are built from concurrent regions and events. This section explains the coordination primitives and the protocols they implement.",
    '3.8.1': "The smallest useful pipeline has one producer and one consumer. This section establishes that pattern before adding stages.",
    '3.8.2': "A 1P1C kernel skeleton shows the producer and consumer regions in code. This section reads it in full.",
    '3.8.3': "A persistent kernel decides its own work rather than being launched once per tile. This section covers the guard and the scheduling loop.",
    '3.8.4': "Concurrency can also be expressed on the host side. This section covers \\co@parallel.async@ and stream binding.",
    '3.8.5': "A full event pipeline combines the coordination primitives described above. This section walks through its stages and credit flow.",
    '3.8.10': "The coordination patterns and their syntax are collected here.",
    '3.9': "Croqtile does not replace C++; it sits inside it. This section covers the boundary between the two, including calls, injected source and template instantiation.",
    '3.9.1': "A device helper called from a \\co@__co__@ function shows the argument conventions in practice.",
    '3.9.2': "Verbatim C++ injection lets a program carry target-specific snippets. This section shows the forms it takes.",
    '3.9.10': "The interoperability mechanisms and their rules are collected here.",
    '3.10': "Tileflow patterns and the design decisions behind the language are easiest to see in complete programs. This section presents two worked examples and the reasoning they illustrate.",
    '3.10.1': "A tiled matmul shows how bounded variables, \\co@chunkat@ and movement statements compose into one program.",
    '3.10.2': "An rgb2gray kernel shows the same composition operator on a different shape.",
    '3.11': "Diagnostics are part of using a language well. This section collects the methodology behind compile-time inspection, verbose builds and runtime checks.",
}

def caption_names(names):
    """Build a transform that names a unit's listing captions in order.

    Upstream pages reuse one section title as the caption of every code block
    in that section.  A unit-level transform keeps the rewrite local to one
    content unit, so refining one section never touches another.
    """
    ordered = tuple(names)

    def transform(frag):
        it = iter(ordered)

        def sub(m):
            try:
                return 'caption={%s}' % next(it)
            except StopIteration:
                return m.group(0)

        return re.sub(r'caption=\{[^}]*\}', sub, frag)

    return transform


# A very small number of source units are meta-narrative about the old
# tutorial (its chapter list).  They are replaced, not summarised.
UNIT_OVERRIDES = [
    ('F11-S10',
     r'(?s)You have (?:met|learned the following concepts):.*?\\end\{itemize\}',
     "The tools above answer three questions in order: what the compiler "
     "believed at compile time, what the program did at runtime, and which "
     "flag exposes each. Start with \\texttt{print!} when a shape or a type is "
     "surprising, move to \\texttt{assert} when a value is surprising, and only "
     "then turn on a verbose build."),
    ('F11-S10',
     r'The next step is to open a benchmark kernel.*?\Z',
     "Chapter~\\ref{ch:programming-guide} takes the same ideas further: it "
     "explains why the constructs are defined this way, and "
     "Appendix~\\ref{app:quickref} lets you look up any spelling."),
    # Tutorial navigation links pointed at chapters that no longer exist.
    ('F01-S01',
     r'For detailed syntax design and language reference, see the \\emph\{Coding Reference\}\.',
     "For detailed syntax design, see Chapter~\\ref{ch:programming-guide}."),
    ('F03-S07',
     r'The \\emph\{next chapter\} introduces',
     "Section~\\ref{sec:2-9} introduces"),
    ('F05-S11',
     r'For the full details, see the \\emph\{mdspan reference\} and the \\emph\{ituple reference\}\.',
     "For the full details, see Sections~\\ref{sec:2-4} and~\\ref{sec:2-5}."),
    ('F06-S09',
     r'The \\emph\{next chapter\} introduces',
     "Section~\\ref{sec:3-8} introduces"),
    ('F09-S12',
     r'(?s)The \\emph\{next chapter\} steps past pure Croqtile into.*?to the metal\.',
     "Section~\\ref{sec:2-13} covers C++ interop: \\texttt{\\_\\_device\\_\\_} "
     "functions, preprocessor guards and injected source text for the cases "
     "where a program needs plain C++."),
    ('F10-S13',
     r'(?s)The \\emph\{next chapter\} turns to the other side.*?systematic narrowing\.',
     "Section~\\ref{sec:1-9} turns to the other side of the workflow: what to "
     "do when the kernel compiles but the output is wrong --- debugging, "
     "verbose modes and systematic narrowing."),
    ('F31-S10',
     r'This is covered in detail in the \\emph\{Asynchronous parallel-by\} section\.',
     "Section~\\ref{sec:3-8-4} covers this in detail."),
    # Round 15: 1.1 keeps the Triton-style arc (observable phenomenon, one
    # premise, one-sentence definition, concrete comparison, honest difficulty)
    # but carries the substance of the repo's own Part I tutorial and Part III
    # coding reference: the four language properties, the matmul-based running
    # example, and the construct inventory.
    ('F01-S01',
     r'Welcome to the Croqtile tutorial\.',
     "The programs at the centre of modern workloads are often short. A matrix "
     "multiplication, a normalization step, an attention block: each is a few "
     "lines of mathematics. Yet turning those lines into a program that runs "
     "well has become a specialist's job, and one that has to be done again "
     "whenever a shape, a dataflow or a machine changes. The reason is rarely "
     "the arithmetic. It is that the same small computation also has to say "
     "where its data lives, how it is divided into tiles, in what order those "
     "tiles become available, and how concurrent regions agree on progress. "
     "In most settings those decisions are scattered across library calls, "
     "build options and code a reader has to infer, so changing one of them "
     "quietly changes several others. A single family of matrix "
     "multiplications already produces a long series of variants, each with "
     "its own tiling, movement and coordination; keeping them consistent is "
     "where most of the effort goes."),
    ('F01-S01',
     r'(?s)Each chapter introduces.*?\Z',
     "Croqtile begins with a single premise: if those decisions were part of "
     "the language itself, a person could state them once, in one place, and "
     "let the compiler check them together before producing the program that "
     "runs. Croqtile is a C++ embedded DSL for high-performance kernel "
     "programming, designed from the ground up around four properties: an "
     "easy-to-use syntax whose abstractions cost nothing at run time, "
     "comprehensive compile-time safety checks, first-class dynamic shapes "
     "and symbolic dimensions, and a form that AI tuning tools can work with. "
     "A program states the shapes, iterations, tiles, movement and "
     "coordination it needs; the compiler turns that description into "
     "runnable code. The examples in this guide grow by one idea at a time, "
     "so a later listing usually differs from an earlier one only by the "
     "construct that section introduces."),
    ('F01-S02',
     r'(?s)\\begin\{enumerate\}.*?\\end\{enumerate\}',
     "Consider what happens when a computation needs a different tile size. "
     "In a program where tiling, copying and coordination are written "
     "separately, that one change reaches into the loop bounds, the address "
     "arithmetic, the staging buffers and the coordination points. In Croqtile "
     "the iteration declares the tile, the tile selects the data, and a "
     "movement statement says when it is ready; the change stays local, and "
     "the compiler rejects the combinations that do not fit. That is the key "
     "benefit of making tiles part of the language: the person writing the "
     "program keeps the decisions, while the compiler takes on the bookkeeping "
     "that used to tie them together. It also means a complete kernel is short "
     "enough to read in one sitting, which is how the examples here are built."
     "\n\n"
     "The vocabulary follows the way a kernel is put together. A "
     "\\co@__co__@ function describes the parallel work and sits beside the "
     "ordinary C++ that prepares buffers through the host API. Shaped types "
     "built on MDSpan and spanned data carry symbolic dimensions that need "
     "not be fixed until the program runs. Iteration constructs declare "
     "independent regions and loop over tiles, DMA statements and "
     "\\texttt{chunkat} select and move those tiles, and events order the "
     "regions that share them. Where plain C++ is needed again, "
     "\\co@__cpp__@ blocks, macros and the preprocessor are available in the "
     "same file. Appendix~\\ref{app:quickref} collects the exact spelling of "
     "each form, and the chapters that follow put them to work in complete "
     "programs."),
    ('F01-S03',
     r'\\item Basic C\+\+ knowledge \(functions, pointers, arrays\)',
     "\\item Basic C++ (functions, pointers and arrays) and the idea of a "
     "shaped tensor\n"
     "\\item Some familiarity with parallel programs: independent work items, "
     "tiles, and coordination between concurrent regions"),
    ('F01-S03', r'A working Croqtile compiler',
     'A machine on which you can build the compiler'),
    ('F12-S01',
     r'(?s)Welcome to the Croqtile coding reference\..*?\Z',
     "Of course, putting those decisions in the language does not make writing "
     "a kernel trivial; it moves the difficulty into the compiler, where it "
     "can be discharged once instead of rediscovered by every program. The "
     "compile-time safety checks have to decide shapes and indices before "
     "anything runs. Symbolic dimensions have to survive until the program "
     "runs, so one source serves many input sizes. The abstractions have to "
     "stay zero-cost, and the constructs have to stay few enough that a person "
     "can hold all of them in mind. Those constraints shape almost every rule "
     "in the following chapters; giving you a sense of how the compiler meets "
     "them is one of the purposes of this guide."),
    ('F10-S01', r'So far the tutorial has built', 'The preceding sections have built'),
    # An upstream page whose entire body was target-platform material would
    # otherwise leave a bare heading behind.  Give it a transition instead.
    ('F13-S01', r'\A\s*\Z',
     "From here the path is step by step. The chapters that follow start with "
     "the smallest kernel that compiles and evolve a single running example "
     "through progressively richer matmul variants, so each new idea appears "
     "in a concrete, working program and every major construct is met before "
     "the end. The next step is a practical one: build the compiler so that "
     "each of those programs can be run rather than only read. "
     "Section~\\ref{sec:1-2} lists what you need, and "
     "Section~\\ref{sec:1-3} walks through the build."),
    # Section 1.2: Part I lists the toolchain in a table, Part III repeats it
    # as a bare list.  Keep the table, turn the duplicate into the missing
    # explanation of the version floor, and point the fallback at the real
    # setup section (Triton/TileLang installation pages hand off this way).
    ('F02-S01',
     r'(?s)Before writing your first kernel.*?\Z',
     "Croqtile is built from source rather than installed as a prebuilt "
     "package. The compiler is itself a C++ program whose parser is generated "
     "from a grammar, so the first thing to check is the toolchain that "
     "builds it. The table below gives the versions that are known to work: a "
     "C++17-capable compiler and two parser-generator tools."),
    # Fold the table's one-line lead-in into the paragraph above it so the
    # table never leaves a dangling sentence at the foot of a page.
    ('F02-S02', r'You need a C\+\+17-capable compiler and two parser-generator tools:', ''),
    # This table has no caption or label, so a float would strand it on the
    # next page between two halves of the same explanation.
    ('F02-S02', r'\\begin\{table\}\[htbp\]', r'\begin{center}'),
    ('F02-S02', r'\\end\{table\}', r'\end{center}'),
    ('F02-S02',
     r'(?s)Most Linux distributions.*\Z',
     "Most Linux distributions ship Flex but an older Bison. If the Bison on "
     "your system is below 3.8, the setup step described in "
     "Section~\\ref{sec:1-3} fetches a compatible version on its own, so the "
     "distribution package does not have to be replaced by hand."),
    ('F22-S03',
     r'(?s)\\begin\{itemize\}.*?\\end\{itemize\}',
     "The version floor is not arbitrary. GCC 8.1 and Clang 6 are the first "
     "releases with complete C++17 support, and Bison 3.8 and Flex 2.6.4 are "
     "the versions whose generated code supports the C++ features the "
     "compiler uses. An older Bison is the usual obstacle, and it is the one "
     "the setup step resolves for you."),
    # Section 1.3: Triton's "From Source" and TileLang's "Building from
    # Source" name each command block and move straight on to verification.
    # The upstream page gave all three listings the same caption.
    ('F02-S03', r'caption=\{Build from Source\}\]\ngit clone',
     'caption={Clone and prepare}]\ngit clone'),
    ('F02-S03', r'caption=\{Build from Source\}\]\nmake\n',
     'caption={Build the compiler}]\nmake\n'),
    ('F02-S03', r'caption=\{Build from Source\}\]\nmake test\n',
     'caption={Run the test suite}]\nmake test\n'),
    ('F02-S03', r'Then run the test suite to confirm everything works:',
     'Then run the test suite; it is the check that the freshly built compiler is sound:'),
    ('F02-S03', r'Add it to your \\texttt\{PATH\} or invoke it by its full path\.',
     "Add it to your \\texttt{PATH} or invoke it by its full path. "
     "Section~\\ref{sec:1-4} compiles a minimal program to confirm the "
     "installation end to end."),
    # Section 1.4: Triton and TileLang both say what a verification command
    # proves, then name the expected output.  The upstream page dropped the
    # reader straight into a program with no sentence about its purpose.
    ('F02-S04',
     r'Create a minimal \\texttt\{\.co\} file to confirm the compiler runs:',
     "The smallest useful check is a kernel that copies four integers. Save it "
     "as \\file{verify.co}:"),
    ('F02-S04', r'caption=\{Verify the Installation\}\]\n__co__',
     'caption={A minimal Croqtile program}]\n__co__'),
    ('F02-S04', r'caption=\{Verify the Installation\}\]\ncroqtile verify\.co',
     'caption={Compile and run the check}]\ncroqtile verify.co'),
    ('F02-S04',
     r'You should see \\texttt\{OK\}\. If you do, the compiler, linker, and runtime are all working\.',
     "You should see \\texttt{OK}. If you do, the compiler, linker, and "
     "runtime are all working. Section~\\ref{sec:1-5} lists the options that "
     "change what the command produces."),
    # Section 1.5: the Part III page opened with "However" because its first
    # sentence --- the source-to-source translation model --- was archived
    # with the vendor names.  Restore the concept in neutral wording, repair
    # the fused/broken sentences, and give the three overlapping units
    # distinct roles and distinct listing captions.
    ('F22-S05', r'(?s)However, since Croqtile integrates.*?properly configured\.',
     "Croqtile is a source-to-source translator: it turns a \\file{.co} "
     "program into C++ source that a downstream compiler consumes. With that "
     "compiler in place the chain behaves like an end-to-end compiler, so the "
     "command line can offer the usual output forms."),
    ('F22-S05',
     r'Therefore, Once Croqtile is built, developers can compile Croqtile programs into various output forms, including:',
     "Once the compiler is built, a program can be lowered into several output forms:"),
    ('F22-S05', r'(?s)Species except for .*?support and limitations\.',
     "All but the first and last are the familiar forms from a C++ toolchain. "
     "Which of them a build can produce depends on what the downstream "
     "compiler supports."),
    ('F22-S05',
     r'The usage of the Croqtile compiler is similar to that of \\texttt\{gcc\} or \\texttt\{clang\}\. For example:',
     "The command itself has the usual shape:"),
    ('F22-S05', r'without performing the .target compilation\*\.',
     'without performing the downstream compilation.'),
    ('F22-S05', r'Here is a example output of', 'Here is an example of the output of'),
    ('F22-S05', r'caption=\{Compile Croqtile Program\}\]\ncroqtile your_program\.co',
     'caption={Default invocation}]\ncroqtile your_program.co'),
    ('F22-S05', r'caption=\{Compile Croqtile Program\}\]\ncroqtile -t target',
     'caption={Emitting lowered source}]\ncroqtile -t target'),
    ('F22-S05', r'caption=\{Compile Croqtile Program\}\]\n// --help',
     'caption={Full option list}]\n// --help'),
    ('F03-S06',
     r'Croqtile files use the \\texttt\{\.co\} extension\. The compiler works like \\texttt\{gcc\} or \\texttt\{clang\}:',
     "For a first program the same command is enough. Files use the "
     "\\file{.co} extension, and the default invocation produces "
     "\\texttt{a.out}:"),
    ('F03-S06', r'caption=\{Build and Compile\}\]\ncroqtile ele_add\.co',
     'caption={Compiling a first program}]\ncroqtile ele_add.co'),
    ('F03-S06', r'caption=\{Build and Compile\}\]\ncroqtile --help',
     'caption={Listing the options}]\ncroqtile --help'),
    # Section 1.6: the Part III "Environment" page interleaves six command
    # blocks with sentence fragments, and both units list the same shortcuts.
    # Keep every command, rewrite the connective prose, and name each listing.
    ('F22-S04', r'(?s)To streamline the configuration.*\Z',
     "The build tree has its own set of shortcuts, and \\texttt{make help} is "
     "the index to them:\n\n"
     "\\begin{lstlisting}[language={},caption={Listing Makefile targets}]\n"
     "make help\n"
     "\\end{lstlisting}\n\n"
     "A fresh checkout is prepared with \\texttt{make setup}. It fetches the "
     "tools the build and test steps need --- Flex, Bison 3.8 or newer, "
     "FileCheck and the GoogleTest sources --- so the result does not depend "
     "on which versions the system already has:\n\n"
     "\\begin{lstlisting}[language={},caption={Preparing a fresh checkout}]\n"
     "make setup\n"
     "\\end{lstlisting}\n\n"
     "Once the environment is configured, the compiler is built with:\n\n"
     "\\begin{lstlisting}[language={},caption={Building the configured tree}]\n"
     "make\n"
     "\\end{lstlisting}\n\n"
     "Croqtile is still in development, so the unit tests are the check that "
     "the binary is sound. A failing run stops and reports the error rather "
     "than leaving a build that only breaks later:\n\n"
     "\\begin{lstlisting}[language={},caption={Running the unit tests}]\n"
     "make test\n"
     "\\end{lstlisting}\n\n"
     "The sample operators can be tested as a whole family, or narrowed to a "
     "single one. The first command runs every sample:\n\n"
     "\\begin{lstlisting}[language={},caption={All sample operator tests}]\n"
     "make sample-test\n"
     "\\end{lstlisting}\n\n"
     "The second selects one operator --- \\texttt{add}, \\texttt{mul}, "
     "\\texttt{relu}, \\texttt{sigmoid}, \\texttt{softmax} or \\texttt{tanh}:\n\n"
     "\\begin{lstlisting}[language={},caption={One sample operator}]\n"
     "make sample-test-operator OPERATOR=add\n"
     "\\end{lstlisting}"),
    ('F02-S06', r'(?s)The Makefile includes shortcuts.*\Z',
     "When you change Croqtile itself, three shortcuts are enough to narrow a "
     "failure from the whole suite down to one operator family:\n\n"
     "\\begin{lstlisting}[language={},caption={Everyday development shortcuts}]\n"
     "make help                              # list all available targets\n"
     "make sample-test                       # run all sample operator tests\n"
     "make sample-test-operator OPERATOR=add # test a specific operator\n"
     "\\end{lstlisting}\n\n"
     "They are the commands to reach for when a build passes but an operator "
     "does not, or when a change to the compiler needs checking against the "
     "samples.\n\n"
     "With the compiler installed and verified, you are ready to write your "
     "first real Croqtile program in Section~\\ref{sec:1-7}."),
    # Section 1.7: a first program should say what it deliberately leaves out
    # and what skeleton it shows, then hand the reader on.  Upstream used a
    # tutorial slogan, a casual "no fancy thread hierarchy", two identical
    # captions and a vague "closer look".
    ('F03-S01', r'(?s)You are about to write.*\Z',
     "The shortest program that shows the shape of the language is an "
     "element-wise addition of two small matrices. It has no tiling, no "
     "explicit data movement and no coordination between regions; those are "
     "the constructs every later example adds. What it does have is the full "
     "skeleton of a Croqtile program: a \\co@__co__@ function that describes "
     "the parallel work, and an ordinary C++ \\co@main()@ that prepares the "
     "inputs, calls it and checks the result. Every program in the guide is "
     "built from these two halves."),
    ('F03-S03',
     r'caption=\{A Complete Example: Element-Wise Addition\}\]\n__co__',
     'caption={A complete element-wise addition}]\n__co__'),
    ('F03-S03',
     r'caption=\{A Complete Example: Element-Wise Addition\}\]\ncroqtile ele_add\.co',
     'caption={Compile and run the example}]\ncroqtile ele_add.co'),
    ('F03-S03',
     r'You should see \\texttt\{Test Passed\}\. Now for a closer look at each part\.',
     "You should see \\texttt{Test Passed}. The chapters that follow take this "
     "same program apart, one construct at a time."),
    # Section 1.8: both Part III openings were placeholders ("Below is a
    # complete  code example", "As introduced ... here is the code again")
    # that did not say what the listing demonstrates or why it is repeated.
    ('F14-S05',
     r'Below is a complete  code example that performs element-wise addition on two arrays of the same size and element type:',
     "Here is a complete example that adds two arrays of the same shape and "
     "element type. Its three parts are labelled in the listing: a device "
     "helper, the tileflow function and the host program that calls it:"),
    ('F14-S06', r'(?s)As introduced, the .*?here is the code again:',
     "The host program is where the data is created and where the result is "
     "checked. It is shown again here with the surrounding calls so that the "
     "boundary between ordinary C++ and the tileflow function is visible:"),
    # Section 1.9: the imported "Debugging tip" arrived with no bridge, and
    # the workflow page used one caption for both of its listings.
    ('F08-S13', r'If something looks wrong after editing a pipeline, verify',
     "Before reaching for the compiler's inspection flags, look at the "
     "program itself. If something looks wrong after editing a pipeline, verify"),
    ('F11-S09', r'caption=\{Putting it together\}\]\n#ifdef DEBUG_PRINT',
     'caption={Guarding debug prints}]\n#ifdef DEBUG_PRINT'),
    ('F11-S09', r'caption=\{Putting it together\}\]\ncroqtile kernel\.co',
     'caption={Debug and production builds}]\ncroqtile kernel.co'),
    # Section 1.10: the only source content was a bare five-item checklist
    # whose first item had no explanation.  Keep all five regions, but write
    # them in the order a reader actually meets them in the file.
    ('F10-S12', r'(?s)When you open a benchmark kernel.*\Z',
     "A benchmark kernel looks dense until you know the order in which its "
     "parts appear. Read it top down. The file opens with macros and "
     "\\texttt{\\#error} guards that fix the conditions the rest of it is "
     "compiled under. Below them sit \\texttt{\\_\\_device\\_\\_} functions "
     "--- helper algorithms such as top-K, reductions and shuffle wrappers "
     "that Croqtile passes through unchanged. Then comes the host setup, "
     "ordinary C++ that creates buffers, chooses the launch configuration "
     "and takes timings. The centre of the file is the "
     "\\texttt{\\_\\_co\\_\\_} function, where the orchestration lives: "
     "\\texttt{parallel}, \\texttt{foreach}, TMA/MMA, "
     "\\texttt{inthreads.async} and events, each region mapping back to one "
     "of the chapters you have just read. Last are the "
     "\\texttt{\\_\\_cpp\\_\\_} islands, usually only a handful of lines. "
     "Most production files follow this order, so the same five checks work "
     "every time."),
    # Markdown read "6 * 10 * 10" as emphasis, and the source grammar was off.
    ('F15-S19',
     r'(?s)The above example showcases.*?\Z',
     "The example above shows one typical use of \\texttt{chunkat}. Here the "
     "spanned data has type \\texttt{f32 [6, 10, 100]}. The operation takes "
     "two parameters --- the \\emph{integer} \\texttt{p} and the "
     "\\emph{ituple} \\texttt{index} --- and divides the data into "
     "\\(6 \\times 10 \\times 10\\) pieces, one for each combination of "
     "their ranges. Each piece holds 10 consecutive elements along the least "
     "significant dimension and is copied into local storage."),
    # Sentences whose second half was archived as target-platform material.
    ('F07-S08',
     r'(?s)Event-based synchronization is omitted.*?Focus on the program structure:',
     "Event-based synchronization is left out of this skeleton; "
     "Section~\\ref{sec:3-8-5} adds the full pipeline protocol. "
     "Focus on the program structure:"),
    ('F07-S14',
     r'(?s)The 1P1C skeleton above is incomplete.*?\Z',
     "The 1P1C skeleton above is incomplete: without \\texttt{wait} / "
     "\\texttt{trigger}, the consumer can read before the producer has "
     "finished writing. Section~\\ref{sec:3-8-5} adds the buffering and "
     "event protocols that make the pipeline safe, and walks through the "
     "resulting kernel stage by stage."),
    ('F09-S03',
     r'The difference is only in how the data gets there[^\n]*?loads vs\.',
     "The difference is only in how the data gets there; the layout that "
     "arrives is the same in both cases."),
    ('F09-S03', r'the pairing it is a correctness invariant',
     'the pairing --- it is a correctness invariant'),
    # Round 25 (2.1): the tutorial speaks of two parts, the coding reference
    # of host / device / tileflow regions.  Reconcile the two accounts, put
    # back the translation step that was archived with a platform name, and
    # repair the debris that names left behind.
    ('F03-S02',
     r'(?s)Every Croqtile program has exactly two parts:.*\Z',
     "At its simplest a program has two parts. A \\texttt{\\_\\_co\\_\\_} "
     "function describes computation on shaped tensors, and the compiler "
     "translates that function into target code. The host program is "
     "standard C++ that prepares the data, calls the function and checks "
     "the result. Both live in a single \\file{.co} file, and the "
     "\\texttt{\\_\\_co\\_\\_} prefix is all the compiler needs to tell "
     "them apart."),
    ('F14-S03',
     r'caption=\{Host, Device and Tileflow\}',
     'caption={The three regions of one source file}'),
    ('F14-S03',
     r'The code below showcases a \\emph\{Croqtile\} program targeting :',
     "Both of those parts are present here, but the ordinary C++ side is "
     "worth splitting once more: a \\file{.co} file may also carry a "
     "\\texttt{\\_\\_device\\_\\_} helper. The listing is written from the "
     "inside out --- a device helper, the \\texttt{\\_\\_co\\_\\_} function "
     "that calls it, and the host \\texttt{main()} that calls the "
     "\\texttt{\\_\\_co\\_\\_} function."),
    ('F14-S03',
     r"(?s)Let's briefly review each part:.*\Z",
     "Execution starts at the bottom of the listing, in the host program. "
     "Written in standard C++ and run on the CPU, it is the entry point of "
     "the module and the caller of the Tileflow program --- the "
     "\\texttt{\\_\\_co\\_\\_} functions --- and it manages the overall "
     "workflow of the heterogeneous application. A typical host body prepares "
     "the input data, invokes a Croqtile function, handles its return value "
     "and advances to the next step.\n\n"
     "The Tileflow program called from there is the part Croqtile adds. Built "
     "from \\texttt{\\_\\_co\\_\\_} functions, it is the core of a "
     "\\file{.co} program: it orchestrates data movement between host and "
     "device and between storage levels within a single device. A typical "
     "Tileflow body moves data into a buffer at the appropriate storage "
     "level, calls a device function to compute on it, and moves the result "
     "back to the host. It is the only region the compiler translates.\n\n"
     "Device functions sit at the other end of that call chain. A "
     "\\texttt{\\_\\_device\\_\\_} function defines computation-intensive "
     "operations that run on the target device; the prefix marks it as device "
     "code. Like the host program, it passes through Croqtile compilation "
     "unchanged. The compiler leaves both sides of ordinary C++ alone; only "
     "the \\texttt{\\_\\_co\\_\\_} function between them is rewritten."),
    # Round 26 (2.2): one line, one purpose; four captions that used to be
    # the same section title; two sentences left duplicated or dangling by
    # the source pages.
    ('F03-S04', None, caption_names([
        'A kernel signature', 'Allocating the output',
        'The parallel body', 'Returning the result'])),
    ('F03-S04',
     r'This is the heart of the program\. Here it is piece by piece\.',
     "Start with the signature, which declares more than a C++ function "
     "prototype would: every parameter and the result carry a shape."),
    ('F03-S04',
     r'Croqtile provides the following element types: Croqtile supports these element types:',
     'Croqtile supports these element types:'),
    ('F03-S04',
     r'(?s)The short answer is that .*?at this level\.',
     "What \\texttt{parallel} leaves unsaid is how those instances map onto "
     "the machine: without an annotation the compiler decides. That mapping "
     "is deliberately absent from the language here --- the statement "
     "describes independent work, not the resources that run it."),
    ('F08-S08',
     r'(?s)\\begin\{center\}\\small\\emph\{.*?\\end\{center\}',
     "\\begin{center}\\small\\emph{Using \\texttt{s32} with scalar "
     "accumulation is a simplified style that isolates the \\texttt{swap} "
     "mechanism (Section~\\ref{sec:3-6-2}); the same return-type deduction "
     "applies to the FP16/MMA kernels in Section~\\ref{sec:3-7}.}"
     "\\end{center}"),
    # Round 27 (2.3): host buffers keep their shape across the call.
    ('F03-S05', None, caption_names([
        'Creating the shaped inputs', 'Calling the kernel'])),
    ('F25-S04', None, caption_names([
        'Passing a shaped buffer', 'The host-side buffer wrapper',
        'A call built from runtime dimensions'])),
    ('F25-S07', None, caption_names([
        'Returning a shaped buffer', 'The host-side result wrapper'])),
    ('F03-S05',
     r'The host is plain C\+\+ with a few Croqtile API calls:',
     "The host side is plain C++ with two Croqtile API calls: one creates a "
     "shaped buffer from raw storage, the other hands a view of it to the "
     "kernel."),
    ('F05-S16', r'from earlier chapters:', 'used by the host code above:'),
    ('F05-S16',
     r'Nothing new here\. The host program stays boring so the Croqtile function stays the star\.',
     "The loop after the call checks every element against a plain C++ "
     "reference computation, so an incorrect result is reported before the "
     "program prints anything."),
    ('F15-S21',
     r'Passing arguments to Croqtile or opposite requires inclusion of Croqtile header file: croqtile\.h\.',
     "Passing arguments between Croqtile and C++ requires the "
     "\\texttt{croqtile.h} header."),
    ('F25-S03',
     r'For a Croqtile function, it only accepts two types of input parameters: the \\emph\{spanned data\}, and the \\emph\{integer\}\.',
     "A Croqtile function accepts only two kinds of input parameter, the "
     "\\emph{spanned data} and the \\emph{integer}."),
    ('F25-S04',
     r'Though programs does not necessary to explicitly include \\texttt\{croqtile\.h\} in Croqtile compilation environment, it is included for every Croqtile programs\.',
     "A program does not have to include \\texttt{croqtile.h} explicitly in a "
     "Croqtile compilation environment; the compiler includes it for every "
     "Croqtile program."),
    ('F25-S04',
     r'To make it easy to understand, you list simplified definition is like:',
     'Its simplified definition is:'),
    ('F25-S04',
     r'It is also possible to have variable as dimensions to construct the shaped data, like following:',
     "Dimensions do not have to be constants; runtime values work as well:"),
    ('F25-S07',
     r'Semantics, the return must be copied than reference, or else it returns a reference to an object allocated in the called function\.',
     "The return has value semantics: the caller receives the buffer itself, "
     "not a reference to an object allocated inside the called function."),
    ('F25-S07', r'In contrary to', 'In contrast to'),
    ('F25-S07',
     r'no copy actually happen when returning',
     'no copy actually happens when returning'),
    ('F25-S07', r"In compiler's term,", 'In compiler terminology,'),
    ('F25-S07',
     r'This allows you able to work without knowing the too much detail about Croqtile function\.',
     "It lets the host work with the result without knowing the details of "
     "the Croqtile function that produced it."),
    # Round 28 (2.4): a shape is a value; name each shape example.
    ('F16-S03', None, caption_names([
        'Declaring a shape', 'A rank mismatch',
        'Omitting the mdspan keyword'])),
    ('F16-S04', None, caption_names([
        'The original shape', 'A derived shape', 'Rebinding a shape'])),
    ('F19-S03', None, caption_names([
        'A kernel with an unknown dimension',
        'The caller supplying that dimension'])),
    ('F16-S01',
     r'In Croqtile, shape is a first-class citizen\. In this section, you will learn how to program shapes in Croqtile code\.',
     "A shape in Croqtile is a value a program can name and compute with, "
     "not just an annotation on a declaration. The examples below define "
     "one, derive another from it, and leave a dimension open for the "
     "caller to fill in."),
    ('F04-S16',
     r'This comes up constantly in data movement code because',
     'Deriving one shape from another comes up constantly in data movement code, because'),
    ('F16-S04',
     r'Note that arithmetic operations on an \\emph\{mdspan\} variable',
     'Arithmetic operations on an \\emph{mdspan} variable'),
    # Round 29 (2.5): integers and i-tuples.
    ('F17-S02', None, caption_names([
        'An initialized integer', 'Assignment to an integer',
        'Several integers at once', 'An integer parameter',
        'A declaration without initialization'])),
    ('F17-S05', None, caption_names([
        'An i-tuple literal', 'An inconsistent rank'])),
    ('F17-S06', None, caption_names([
        'Assigning a new tuple', 'Indexing an mdspan with an i-tuple',
        'An invalid shape expression'])),
    ('F17-S01',
     r'In this section, you will learn how to define \\emph\{integer\}s and integer-tuples \(\\emph\{i-tuple\}s\) in Croqtile code and understand their usage\.',
     "Integers in a kernel control loops and index arrays, and an "
     "\\emph{i-tuple} groups several of them into one index. Both are "
     "declared with an initializer and both are checked against the shape "
     "they are used with."),
    ('F17-S02',
     r'However, since Croqtile can infer the type from the \\emph\{initialization expression\}, you can also define an integer variable in another style:',
     "Croqtile can also infer the type from the \\emph{initialization "
     "expression}, so an integer variable can be declared in another style:"),
    ('F17-S06',
     r'Note that, similar to \\emph\{integer\} and \\emph\{mdspan\}, \*\*an \\emph\{ituple\} variable must be initialized and cannot be reassigned\*\*\.',
     "Similar to \\emph{integer} and \\emph{mdspan}, an \\emph{ituple} "
     "variable must be initialized and cannot be reassigned."),
    ('F17-S09',
     r'In this section, you learned how to define \\emph\{integer\}s and \\emph\{i-tuple\}s in Croqtile\.',
     "Integers and i-tuples are the two scalar-shaped values a kernel "
     "declares."),
    ('F15-S05',
     r'As described, Scalar types in Croqtile includes',
     'Scalar types in Croqtile include'),
    # Generic fallbacks, applied to every unit after the specific ones.
    ('*', r'\\emph\{next chapter\}', 'the next section'),
    ('*', r'\\emph\{previous chapter\}', 'the previous section'),
    ('*', r'\bturns The\b', 'turns. The'),
    ('*', r'\s+,\s*,\s+', ', '),
    ('*', r'\s+,\s+\.', '.'),
]

CHAPTERS = [
    ('1', 'Getting Started', 'content/ch01-getting-started.tex', 'ch:getting-started'),
    ('2', 'Language Basics', 'content/ch02-language-basics.tex', 'ch:language-basics'),
    ('3', 'Programming Guide', 'content/ch03-programming-guide.tex', 'ch:programming-guide'),
]

removed = collections.defaultdict(list)   # uid -> [(kind, raw text)]
figures_copied = {}
warnings = []


def norm_unicode(s, ascii_only=False):
    table = UNI_ASCII if ascii_only else UNI
    for k, v in table.items():
        s = s.replace(k, v)
    s = ''.join(ch if ord(ch) < 128 else '?' for ch in s)
    return s


def apply_tokens(s):
    for pat, rep in TOKENS:
        s = pat.sub(rep, s)
    return s


def esc(s):
    return ''.join(SPECIAL.get(ch, ch) for ch in s)


def code_tt(s):
    """Inline code: breakable \\texttt, safe in tabulars, captions and bold."""
    s = norm_unicode(s, ascii_only=True)
    s = apply_tokens(s)
    s = esc(s)
    if len(s) > 14:
        s = re.sub(r'([._#\-/:<>@])', r'\1\\allowbreak{}', s)
    return r'\texttt{' + s + '}'


SEP_RE = re.compile(r'-{1,2}|[\u2014\u2013]|;')


def scrub(raw, uid, kind):
    """Split plain text into clauses; archive and drop hardware clauses.

    Dashes and semicolons are kept as joining tokens.  Discarding them used to
    fuse two independent clauses into one ungrammatical sentence, for example
    "the compiler does not enforce the pairing it is a correctness invariant".
    """
    chunks = re.split(r'((?<=[.;!?])\s+|\s+-{1,2}\s+|\s+[\u2014\u2013]\s+|\s+;\s+)', raw)
    parts = [c.strip() for c in chunks if c.strip()]
    keep, drop, after_drop = [], [], []
    pending = False
    for p in parts:
        sep = bool(SEP_RE.fullmatch(p))
        if not sep and HW.search(p):
            drop.append(p)
            pending = True
            continue
        keep.append(p)
        after_drop.append(pending)
        if not sep:
            pending = False
    if drop:
        removed[uid].append((kind, ' '.join(drop)))
    buf, joiner = [], None
    for p, was_after_drop in zip(keep, after_drop):
        if SEP_RE.fullmatch(p):
            if buf:
                joiner = ';' if p == ';' else '---'
            continue
        if joiner:
            if joiner == ';':
                buf[-1] += ';'
            else:
                buf.append(joiner)
            joiner = None
        if (was_after_drop or (drop and not buf)) and p and p[0].islower():
            p = p[0].upper() + p[1:]
        buf.append(p)
    return ' '.join(buf)


def inline(raw, uid, kind='prose', allow_scrub=True, depth=0):
    """Convert one Markdown inline string into LaTeX."""
    store = {}

    def hold(value):
        key = 'QZ%03dQZ' % len(store)
        store[key] = value
        return key

    # Protect math, inline code, links, bold and italic before scrubbing.
    raw = re.sub(r'\$\$?[^$]+\$\$?', lambda m: hold(m.group(0)), raw)
    raw = re.sub(r'`([^`]+)`', lambda m: hold(('CODE', m.group(1))), raw)
    raw = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                 lambda m: hold(('LINK', m.group(1), m.group(2))), raw)
    raw = re.sub(r'\*\*\*([^*]+)\*\*\*', lambda m: hold(('BI', m.group(1))), raw)
    raw = re.sub(r'\*\*([^*]+)\*\*', lambda m: hold(('B', m.group(1))), raw)
    raw = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', lambda m: hold(('I', m.group(1))), raw)

    if allow_scrub:
        raw = scrub(raw, uid, kind)
    out = norm_unicode(esc(raw))

    # Restore protected spans.
    for key, val in store.items():
        if isinstance(val, str):                       # math
            repl = norm_unicode(val)
        elif val[0] == 'CODE':
            repl = code_tt(val[1])
        elif val[0] in ('B', 'I', 'BI'):
            # Code/link/math spans were protected before emphasis, so a bold
            # span may itself contain QZ placeholders.  Restore them to raw
            # Markdown first, then let a nested conversion handle the markup.
            inner_src = val[1]
            if depth < 3:
                for k2, v2 in store.items():
                    if k2 == key:
                        continue
                    if isinstance(v2, str):
                        rawmd = v2
                    elif v2[0] == 'CODE':
                        rawmd = '`' + v2[1] + '`'
                    elif v2[0] == 'LINK':
                        rawmd = '[' + v2[1] + '](' + v2[2] + ')'
                    else:
                        mk = {'B': '**', 'I': '*', 'BI': '***'}[v2[0]]
                        rawmd = mk + v2[1] + mk
                    inner_src = inner_src.replace(k2, rawmd)
                inner = inline(inner_src, uid, kind, allow_scrub=True,
                               depth=depth + 1)
            else:
                inner = norm_unicode(esc(inner_src))
            if not inner:
                repl = ''
            elif val[0] == 'B':
                repl = r'\textbf{' + inner + '}'
            elif val[0] == 'I':
                repl = r'\emph{' + inner + '}'
            else:
                repl = r'\textbf{\emph{' + inner + '}}'
        else:                                          # LINK
            label = norm_unicode(esc(scrub(val[1], uid, kind)))
            url = val[2]
            if re.match(r'https?://', url):
                repl = r'\href{' + esc(url) + '}{' + label + '}'
            else:
                repl = r'\emph{' + label + '}'
        out = out.replace(key, repl)
    return editorial(apply_tokens(out)).strip()


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------
def parse_registry():
    files, rows = [], {}
    for line in open(REG, encoding='utf-8'):
        m = re.match(r'^### F(\d+) `(.+)`', line)
        if m:
            files.append((int(m.group(1)), m.group(2)))
            continue
        m = re.match(r'^\| `(F(\d+)-S(\d+))` \|', line)
        if not m:
            continue
        cells = [c.strip() for c in re.split(r'(?<!\\)\|', line.strip().strip('|'))]
        if len(cells) < 5:
            continue
        rows[(int(m.group(2)), int(m.group(3)))] = {
            'uid': 'F%02d-S%02d' % (int(m.group(2)), int(m.group(3))),
            'heading': cells[1], 'cls': cells[3], 'slot': cells[4],
            'note': cells[5] if len(cells) > 5 else '',
        }
    return files, rows


# ---------------------------------------------------------------------------
# Section splitting (mirrors the registry's section boundaries)
# ---------------------------------------------------------------------------
def split_sections(path):
    lines = open(path, encoding='utf-8').read().replace('\r', '').split('\n')
    sections, pre = [], []
    cur_head, cur_body, in_fence = None, [], False
    for line in lines:
        if in_fence:
            cur_body.append(line)
            if re.match(r'^\s*```', line):
                in_fence = False
            continue
        if re.match(r'^\s*```', line):
            in_fence = True
            cur_body.append(line)
            continue
        hm = re.match(r'^(#{1,4})\s+(.*)$', line)
        if hm:
            if cur_head is None:
                pre = cur_body
            else:
                sections.append((cur_head, cur_body))
            cur_head = (len(hm.group(1)), hm.group(2).strip())
            cur_body = []
        else:
            cur_body.append(line)
    if cur_head is None:
        return [(None, pre + cur_body)]
    sections.append((cur_head, cur_body))
    if pre:
        sections[0] = (sections[0][0], pre + sections[0][1])
    return sections


# ---------------------------------------------------------------------------
# Figure resolution
# ---------------------------------------------------------------------------
def resolve_image(src, md_path, fid):
    src = src.split('#')[0].strip()
    md_dir = os.path.dirname(os.path.join(ROOT, md_path))
    rel = src[3:] if src.startswith('../') else src
    candidates = [
        os.path.join(md_dir, src),
        os.path.join(ROOT, 'docs', rel),
        os.path.join(ROOT, rel),
        os.path.join(md_dir, rel),
    ]
    for c in candidates:
        if os.path.isfile(c):
            base = os.path.basename(c)
            name = 'f%02d-%s' % (fid, re.sub(r'[^A-Za-z0-9._-]', '-', base))
            dest = os.path.join(FIG_DIR, name)
            if not os.path.isfile(dest):
                shutil.copyfile(c, dest)
            figures_copied[name] = c
            return name
    warnings.append('missing image %s (from %s)' % (src, md_path))
    return None


def safe_caption(s, uid):
    s = re.sub(r'`([^`]+)`', r'\1', s)
    s = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', s)
    s = apply_tokens(norm_unicode(s))
    s = re.sub(r'[^A-Za-z0-9 ,.:/()\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:110] or 'Code listing'


def intro_tex(s):
    """Render a hand-written narrative sentence for use inside a macro argument."""
    s = re.sub(r'\\co@([^@]+)@',
               lambda m: r'\texttt{' + esc(norm_unicode(m.group(1), ascii_only=True)) + '}',
               s)
    return s


JUNK_PARA = re.compile(r'^[A-Za-z]\s*\.?$|^[\-.,;:~]+$')


def clean_fragment(tex):
    """Remove debris left behind when a clause was archived as hardware text."""
    tex = re.sub(r'\\(?:emph|textbf|texttt|textit)\{\s*\}', '', tex)
    tex = re.sub(r'(?m)^\\item\s*$', '', tex)
    hold = {}

    def stash(m):
        key = 'QZC%04dQZ' % len(hold)
        hold[key] = m.group(0)
        return key

    tex = re.sub(r'(?s)\\begin\{(\w+)\}.*?\\end\{\1\}', stash, tex)
    paras = []
    for para in re.split(r'\n\s*\n', tex):
        t = para.strip()
        if not t:
            continue
        if (len(t) > 40 or t.startswith(
                ('\\begin', '\\end', '\\paragraph', '\\subsection',
                 '\\noindent', '\\includegraphics', '\\item', '\\centering',
                 '\\caption', '\\label', 'QZC'))):
            paras.append(t)
            continue
        plain = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{[^}]*\}', '', t)
        plain = re.sub(r'---|--|\\[a-zA-Z]+', '', plain).strip()
        if not plain or JUNK_PARA.match(plain):
            continue
        paras.append(t)
    out = '\n\n'.join(paras)
    for key, val in hold.items():
        out = out.replace(key, val)
    return out


GENERIC_HEADINGS = {
    'overview': '', 'what you will find here': '', 'basic syntax': '',
    'introduction': '',
    'quick summary': r'\paragraph*{In short}',
    'summary': r'\paragraph*{In short}',
    'chapter summary': r'\paragraph*{In short}',
    'new syntax': r'\paragraph*{Syntax at a glance}',
    'syntax summary': r'\paragraph*{Syntax at a glance}',
    # Titles that name the old tutorial rather than a language construct.
    'chapters': '', 'croqtile tutorial': '', 'croqtile coding reference': '',
    'what is good in croqtile?': '',
    # Section 1.2 already carries a requirements table; the coding-reference
    # page repeated it under a bare "Prerequisitions" heading.
    'prerequisitions': '',
    # Section 1.3 repeats its slot title as a subsection.
    'build from source': '',
    # Section 1.4 repeats its slot title as a subsection.
    'verify the installation': '',
    # Section 1.7 opens with the old tutorial's chapter slogan.
    'hello croqtile: from zero to running kernel': '',
}


def norm_head(s):
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    return re.sub(r'[^a-z0-9]', '', s.lower())


def heading_for(title, uid, slot):
    """Normalise a source section title to one level under the slot heading."""
    plain = re.sub(r'`([^`]+)`', r'\1', title)
    plain = re.sub(r'\*+', '', plain).strip()
    low = plain.lower().rstrip('.:')
    # "Ch 0: ..." numbers belong to the old tutorial, not to this guide.
    if re.match(r'^Ch(?:apter)?\s*\d+\s*[:.]', plain, re.I):
        return ''
    if low in GENERIC_HEADINGS:
        return GENERIC_HEADINGS[low]
    # A source page title that merely repeats the section title adds noise
    # right after the narrative introduction; let the section title stand.
    a, b = norm_head(plain), norm_head(SLOT_TITLES.get(slot, ''))
    if a and b and (a == b or (min(len(a), len(b)) >= 10 and (a in b or b in a))):
        return ''
    conv = inline(title, uid, 'heading', allow_scrub=False)
    return '\\subsection*{' + conv + '}'


# ---------------------------------------------------------------------------
# Block conversion
# ---------------------------------------------------------------------------
LIST_RE = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.*)$')
IMAGE_RE = re.compile(r'^\s*!\[([^\]]*)\]\(([^)]+)\)')
ITALIC_RE = re.compile(r'^\s*\*([^*].*?)\*\s*$')


def convert_table(lines, uid):
    rows = []
    for line in lines:
        cells = [c.strip() for c in re.split(r'(?<!\\)\|', line.strip().strip('|'))]
        if all(re.fullmatch(r':?-{2,}:?', c.replace(' ', '')) for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ''
    ncol = max(len(r) for r in rows)
    rows = [r + [''] * (ncol - len(r)) for r in rows]
    out = ['\\begin{table}[htbp]', '\\centering\\small',
           '\\begin{tabularx}{\\linewidth}{@{}' + 'l' + 'X' * (ncol - 1) + '@{}}',
           '\\toprule']
    for i, row in enumerate(rows):
        conv = []
        for cell in row:
            c = inline(cell, uid, 'table', allow_scrub=True)
            conv.append(c)
        if any(not c.strip() for c in conv) and i > 0:
            removed[uid].append(('table row', ' | '.join(row)))
            continue
        if i == 0:
            conv = [r'\textbf{' + c + '}' if c else c for c in conv]
        out.append(' & '.join(conv) + r' \\')
        if i == 0:
            out.append('\\midrule')
    out += ['\\bottomrule', '\\end{tabularx}', '\\end{table}', '']
    return '\n'.join(out)


def convert_list(lines, uid):
    items = []
    for line in lines:
        m = LIST_RE.match(line)
        if not m:
            continue
        text = m.group(3)
        text = re.sub(r'^\[ \]\s*', r'$\\square$ ', text)
        text = re.sub(r'^\[[xX]\]\s*', r'$\\boxtimes$ ', text)
        conv = inline(text, uid, 'list')
        if conv.strip():
            items.append((len(m.group(1)), bool(re.match(r'\d+\.', m.group(2))), conv))
    if not items:
        return ''
    out, stack = [], []
    for indent, ordered, conv in items:
        level = 1 if indent >= 2 else 0
        env = 'enumerate' if ordered else 'itemize'
        while len(stack) > level + 1:
            out.append('\\end{' + stack.pop() + '}')
        if len(stack) <= level:
            out.append('\\begin{' + env + '}[leftmargin=1.6em]')
            stack.append(env)
        elif stack[-1] != env:
            out.append('\\end{' + stack.pop() + '}')
            out.append('\\begin{' + env + '}[leftmargin=1.6em]')
            stack.append(env)
        out.append('\\item ' + conv)
    while stack:
        out.append('\\end{' + stack.pop() + '}')
    out.append('')
    return '\n'.join(out)


def convert_body(lines, uid, md_path, fid, section_title):
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code -------------------------------------------------------
        fm = re.match(r'^\s*```\s*(\S*)', line)
        if fm:
            lang = fm.group(1).lower()
            i += 1
            code = []
            while i < n and not re.match(r'^\s*```', lines[i]):
                code.append(lines[i].replace('\r', ''))
                i += 1
            i += 1
            clean = []
            for cl in code:
                cl = norm_unicode(cl, ascii_only=True)
                # Strip hardware explanations from comments, keep the code.
                if '//' in cl:
                    head, comment = cl.split('//', 1)
                    if HW.search(comment):
                        removed[uid].append(('code comment', comment.strip()))
                        cl = head.rstrip()
                    else:
                        cl = head + '//' + apply_tokens(comment)
                elif cl.lstrip().startswith('#') and HW.search(cl):
                    removed[uid].append(('code comment', cl.strip()))
                    continue
                cl = apply_tokens(cl)
                if cl.strip():
                    clean.append(cl.rstrip())
            if clean:
                cap = safe_caption(section_title, uid)
                lstlang = 'croqtile' if lang in ('choreo', 'co', '') else '{}'
                out.append('\\begin{lstlisting}[language=' + lstlang +
                           ',caption={' + cap + '}]')
                out.extend(clean)
                out.append('\\end{lstlisting}')
                out.append('')
            continue

        # HTML comments / include directives --------------------------------
        if stripped.startswith('<!--'):
            while i < n and '-->' not in lines[i]:
                i += 1
            i += 1
            continue
        if stripped.startswith('!INCLUDE_IF_EXISTS'):
            i += 1
            continue
        if re.fullmatch(r'-{3,}|\*{3,}|_{3,}', stripped):
            i += 1
            continue

        # <details> video blocks --------------------------------------------
        if stripped.startswith('<details>'):
            # An embedded animation carries no prose: the <summary> is only a
            # label, and pointing at the source site is editorial metadata.
            while i < n and '</details>' not in lines[i]:
                i += 1
            i += 1
            continue

        # Admonitions -------------------------------------------------------
        am = re.match(r'^!!!\s+(note|warning|tip|info)\s*(?:"([^"]*)")?\s*$', stripped)
        if am:
            kind = am.group(1).lower()
            title = am.group(2) or ('Common pitfall' if kind == 'warning' else 'Key point')
            env = 'pitfall' if kind == 'warning' else 'keypoint'
            i += 1
            body = []
            while i < n and (not lines[i].strip() or lines[i].startswith((' ', '\t'))):
                if lines[i].strip():
                    body.append(lines[i].strip())
                i += 1
            out.append('\\begin{' + env + '}[' + norm_unicode(esc(title)) + ']')
            for b in body:
                c = inline(b, uid, 'admonition')
                if c:
                    out.append(c + '\n')
            out.append('\\end{' + env + '}')
            out.append('')
            continue

        # Tables ------------------------------------------------------------
        if stripped.startswith('|') and stripped.endswith('|'):
            tbl = []
            while i < n and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                tbl.append(lines[i])
                i += 1
            out.append(convert_table(tbl, uid))
            continue

        # Lists -------------------------------------------------------------
        if LIST_RE.match(line):
            lst = []
            while i < n and LIST_RE.match(lines[i]):
                lst.append(lines[i])
                i += 1
            out.append(convert_list(lst, uid))
            continue

        # Figures -----------------------------------------------------------
        im = IMAGE_RE.match(line)
        if im:
            imgs = []
            while i < n:
                m2 = IMAGE_RE.match(lines[i])
                if not m2:
                    break
                imgs.append(m2)
                i += 1
            for m2 in imgs:
                src = m2.group(2)
                if '#only-dark' in src:
                    continue
                name = resolve_image(src, md_path, fid)
                if name:
                    out.append('\\begin{figure}[htbp]\\centering')
                    out.append('\\includegraphics[width=0.86\\linewidth,'
                               'height=0.42\\textheight,keepaspectratio]{' + name + '}')
                    out.append('\\end{figure}')
            # Italic caption directly below the figure block.
            while i < n and not lines[i].strip():
                i += 1
            if i < n and ITALIC_RE.match(lines[i]):
                cap = inline(ITALIC_RE.match(lines[i]).group(1), uid, 'figure caption')
                if cap.strip():
                    out.append('\\begin{center}\\small\\emph{' + cap + '}\\end{center}')
                    i += 1
            out.append('')
            continue

        # Standalone italic caption -----------------------------------------
        im2 = ITALIC_RE.match(line)
        if im2:
            cap = inline(im2.group(1), uid, 'figure caption')
            if cap.strip():
                out.append('\\begin{center}\\small\\emph{' + cap + '}\\end{center}')
                out.append('')
            i += 1
            continue

        # Level-5 heading used as a bold lead-in ----------------------------
        hm5 = re.match(r'^#{5,6}\s+(.*)$', stripped)
        if hm5:
            t = inline(hm5.group(1), uid, 'heading', allow_scrub=False)
            out.append('\\noindent\\textbf{' + t + '}\\par')
            out.append('')
            i += 1
            continue

        # Blockquote --------------------------------------------------------
        if stripped.startswith('>'):
            quote = []
            while i < n and lines[i].strip().startswith('>'):
                quote.append(lines[i].strip().lstrip('>').strip())
                i += 1
            c = inline(' '.join(quote), uid, 'quote')
            if c.strip():
                out.append('\\begin{quote}' + c + '\\end{quote}')
                out.append('')
            continue

        # **Note:** callout --------------------------------------------------
        if re.match(r'^\*\*Note:\*\*', stripped):
            c = inline(re.sub(r'^\*\*Note:\*\*\s*', '', stripped), uid, 'note')
            out.append('\\begin{keypoint}[Note]')
            out.append(c)
            out.append('\\end{keypoint}')
            out.append('')
            i += 1
            continue

        # Ordinary paragraph ------------------------------------------------
        para = [stripped]
        i += 1
        while i < n and lines[i].strip():
            s = lines[i].strip()
            if (re.match(r'^\s*```', lines[i]) or LIST_RE.match(lines[i])
                    or IMAGE_RE.match(lines[i]) or ITALIC_RE.match(lines[i])
                    or s.startswith(('|', '>', '<!--', '!!!', '!INCLUDE_IF_EXISTS',
                                     '<details>', '#', '-', '*'))
                    or re.fullmatch(r'-{3,}|\*{3,}|_{3,}', s)):
                break
            para.append(s)
            i += 1
        c = inline(' '.join(para), uid, 'prose')
        if c.strip():
            out.append(c)
            out.append('')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    files, rows = parse_registry()
    slot_units = collections.defaultdict(list)
    manifest = []

    for fid, rel in files:
        path = os.path.join(ROOT, rel)
        sections = split_sections(path)
        expected = sum(1 for k in rows if k[0] == fid)
        if len(sections) != expected:
            raise SystemExit('section mismatch %s: %d vs %d' % (rel, len(sections), expected))
        for sid, (head, body) in enumerate(sections, 1):
            info = rows[(fid, sid)]
            uid, slot = info['uid'], info['slot']
            title = head[1] if head else rel
            source_heading = ('#' * (head[0] if head else 2)) + ' ' + title

            if slot == '占位':
                manifest.append((uid, rel, slot, 'placeholder', ''))
                continue
            if slot == 'X':
                removed[uid].append(('whole section',
                                    '#' * (head[0] if head else 2) + ' ' + title + '\n' +
                                    '\n'.join(body)))
                manifest.append((uid, rel, slot, 'moved', ''))
                continue

            frag = convert_body(body, uid, rel, fid, title)
            for ouid, pat, repl in UNIT_OVERRIDES:
                if ouid in (uid, '*'):
                    if callable(repl):
                        frag = repl(frag)
                    else:
                        frag = re.sub(pat, lambda m, t=repl: t, frag)
            frag = clean_fragment(frag)
            # An empty registered group heading contributes no heading at all;
            # otherwise the PDF shows a title with nothing beneath it.
            heading_tex = heading_for(title, uid, slot) if frag.strip() else ''
            text = (heading_tex + '\n\n' if heading_tex else '') + frag
            frag_path = os.path.join(UNITS_DIR, uid + '.tex')
            open(frag_path, 'w', encoding='utf-8').write(text + '\n')
            slot_units[slot].append(uid)
            manifest.append((uid, rel, slot, 'kept', title))

    # Chapter files ---------------------------------------------------------
    for num, title, out_name, label in CHAPTERS:
        slots = sorted([s for s in slot_units if s.split('.')[0] == num],
                       key=lambda s: tuple(int(x) for x in s.split('.')))
        buf = ['%% Generated by tools/build_guide.py from UNIT-REGISTRY.md.',
               '%% Do not edit by hand: edit the source Markdown or the registry.',
               '\\chapter{' + title + '}', '\\label{' + label + '}', '',
               '\\guidechapterintro{' + intro_tex(CHAPTER_INTRO[num]) + '}', '']
        for slot in slots:
            level = 'section' if len(slot.split('.')) == 2 else 'subsection'
            buf.append('\\' + level + '{' + esc(SLOT_TITLES.get(slot, slot)) + '}')
            buf.append('\\label{sec:' + slot.replace('.', '-') + '}')
            buf.append('')
            if slot in SLOT_INTROS:
                buf.append('\\guideslotintro{' + intro_tex(SLOT_INTROS[slot]) + '}')
                buf.append('')
            for uid in slot_units[slot]:
                buf.append('\\input{build/units/' + uid + '}')
                # Two units on consecutive lines would be typeset as one
                # paragraph.  The blank line gives each its own paragraph.
                buf.append('')
            buf.append('')
        buf += ['\\guidechapternext{' + intro_tex(CHAPTER_NEXT[num]) + '}', '']
        open(os.path.join(PG, out_name), 'w', encoding='utf-8').write('\n'.join(buf) + '\n')

    # Engineering source map -------------------------------------------------
    # This is deliberately NOT part of the PDF.  A reader of the language guide
    # has no use for a list of which source page a paragraph came from; it lives
    # with the build artefacts instead.
    smap = ['# Source material map (engineering document, not part of the guide)', '',
            'Generated from `UNIT-REGISTRY.md`. The reader-facing guide contains',
            'no provenance metadata; this file exists so editors can trace every',
            'content unit back to its source.',
            '', '| Unit | Source file | Slot | Status |', '|---|---|---|---|']
    for uid, rel, slot, status, _ in manifest:
        smap.append('| `%s` | `%s` | %s | %s |' % (uid, rel, slot, status))
    smap += ['', 'Units marked `moved` are archived verbatim in '
             '`HARDWARE-NOTES.md`; `placeholder` units are empty or TODO pages.',
             '']
    open(os.path.join(PG, 'build', 'SOURCE-MAP.md'), 'w',
         encoding='utf-8').write('\n'.join(smap))

    # Hardware notes --------------------------------------------------------
    hw = ['# Material moved out of the Programming Guide', '',
          'These are the exact upstream passages removed while assembling the',
          'language-only guide, keyed by the content-unit IDs in',
          '`UNIT-REGISTRY.md`. Nothing here is rewritten: it is the raw source',
          'text, kept so that the hardware/architecture material can become its',
          'own document.', '']
    for uid in sorted(removed):
        if not removed[uid]:
            continue
        hw.append('## ' + uid)
        hw.append('')
        for kind, text in removed[uid]:
            hw.append('- **' + kind + '**:')
            hw.append('')
            for ln in text.split('\n'):
                hw.append('    ' + ln)
            hw.append('')
    open(os.path.join(PG, 'HARDWARE-NOTES.md'), 'w', encoding='utf-8').write('\n'.join(hw) + '\n')

    # Report ----------------------------------------------------------------
    print('files          :', len(files))
    print('units          :', len(manifest),
          '(kept %d, moved %d, placeholder %d)' % (
              sum(1 for m in manifest if m[3] == 'kept'),
              sum(1 for m in manifest if m[3] == 'moved'),
              sum(1 for m in manifest if m[3] == 'placeholder')))
    print('slots          :', len(slot_units))
    print('figures copied :', len(figures_copied))
    print('removed chunks :', sum(len(v) for v in removed.values()))
    if warnings:
        print('warnings       :', len(warnings))
        for w in warnings[:15]:
            print('   ', w)
    open(os.path.join(PG, 'build', 'manifest.tsv'), 'w', encoding='utf-8').write(
        'uid\tsource\tslot\tstatus\theading\n' +
        ''.join('\t'.join(m) + '\n' for m in manifest))


if __name__ == '__main__':
    main()
