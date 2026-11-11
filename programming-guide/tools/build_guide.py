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
import filecmp

from figure_editorial import polish_figures
from prose_editorial import polish_prose
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PG = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(PG, '..'))
UNITS_DIR = os.path.join(PG, 'build', 'units')
FIG_DIR = os.path.join(PG, 'figures')
REG = os.path.join(PG, 'UNIT-REGISTRY.md')

# These introductory interop units repeat material covered in chapters 2 and 3.
# Keep their upstream sources and source-map entries, but omit them from the book.
OMITTED_UNITS = {
    'F10-S02', 'F10-S12', 'F14-S05', 'F14-S06',
    # GPU-backend-only features, not part of the portable language:
    'F10-S06',  # setmaxnreg register hints
    'F09-S02',  # tma.copy
    'F09-S05',  # TMA in a pipelined matmul
    'F09-S09',  # .zfill demonstrated through a TMA copy
    'F04-S20',  # DMA vs TMA
    'F04-S21',  # the main TMA constraint
    'F06-S05',  # cooperative group and mma (wgmma)
    'F06-S06',  # two cooperative groups sharing operands
    'F05-S08',  # : group-4 cooperative group level
    'F10-S07',  # macros lesson taught through a setmaxnreg string
}

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
    '1.1': 'What is Croqtile',
    '1.2': 'Setting Up: Requirements, Build, and Verification',
    '1.3': 'The Toolchain: Command Line and Build Utilities',
    '1.4': 'Programs with C++ Integration',
    '1.5': 'Inspecting and Debugging a Build',
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
    '1': ("This chapter takes you from installing Croqtile to a program you "
          "have built and run yourself. After the language's motivation, "
          "set up and verify the compiler, run a complete element-wise "
          "addition example, and use the command-line options to inspect "
          "its build. Finally, learn the tools for checking compilation "
          "commands, runtime values, and incorrect results."),
    '2': ("You will learn to read and write Croqtile programs by working "
          "through short examples. Start with the structure of a complete "
          "program and the roles of its functions, then follow how shapes, "
          "iteration, tile selection, data movement and coordination are "
          "expressed. Each example adds a construct and explains its role, "
          "so you can build up a working understanding of the syntax."),
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

# Section-level lead-ins were removed: they repeated the section headings
# and created cross-references only needed because the sections were split too finely.
SLOT_INTROS = {}

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
def strip_gpu_backend(text):
    """Replace GPU-backend-only spellings with their portable equivalents.

    tma.copy is the GPU tensor-memory engine; dma.copy is the same source
    operation on the other backends. group-4 is a GPU warpgroup level; the
    portable form of the same role split is a thread level.
    """
    text = text.replace('tma.copy', 'dma.copy')
    text = text.replace('tma.\\allowbreak{}copy', 'dma.\\allowbreak{}copy')
    text = text.replace('group-4', 'thread')
    text = text.replace('Group-4', 'Thread')
    return text


def _rename_runtime(text):
    """Spell the runtime the way it is actually shipped.

    The library is `choreo`, aliased as `croq`; the guide uses the short alias.
    `croqtile` appears as a header name, as a namespace written either plainly
    or with LaTeX line-break hints, and inside the assertion helper name. The
    assertion helper is a free function, so it carries no namespace qualifier.
    """
    for old, new in (
            ('croqtile::croqtile_assert', 'croq_assert'),
            ('croq::croq_assert', 'croq_assert'),
            ('choreo::choreo_assert', 'croq_assert'),
            ('choreo_assert', 'croq_assert'),
            ('croqtile_assert', 'croq_assert'),
            ('croqtile.h', 'choreo.h'),
            ('make\\_\\allowbreak{}spanned', 'make\\_\\allowbreak{}spanview'),
            ('make_spanned', 'make_spanview'),
            ('croqtile:', 'croq:'),
            ('choreo:', 'croq:'),
            ('croq::span_view<2, croq::s32>',
             'croq::spanned_view<croq::s32, 2>'),
    ):
        text = text.replace(old, new)
    return text


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
    ('F10-S01', r'So far the tutorial has built', 'The preceding sections have built'),
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
     "your system is below 3.8, the setup step below fetches a compatible version on its own, so the "
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
     "The next step compiles a minimal program to confirm the "
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
    ('F02-S04', r'caption=\{Verify the Installation\}\]\ncroq verify\.co',
     'caption={Compile and run the check}]\ncroq verify.co'),
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
    ('F22-S05', r'caption=\{Compile Croqtile Program\}\]\ncroq your_program\.co',
     'caption={Default invocation}]\ncroq your_program.co'),
    ('F22-S05', r'caption=\{Compile Croqtile Program\}\]\ncroq -t target',
     'caption={Emitting lowered source}]\ncroq -t target'),
    ('F22-S05', r'caption=\{Compile Croqtile Program\}\]\n// --help',
     'caption={Full option list}]\n// --help'),
    ('F03-S06',
     r'Croqtile files use the \\texttt\{\.co\} extension\. The compiler works like \\texttt\{gcc\} or \\texttt\{clang\}:',
     "For a first program the same command is enough. Files use the "
     "\\file{.co} extension, and the default invocation produces "
     "\\texttt{a.out}:"),
    ('F03-S06', r'caption=\{Build and Compile\}\]\ncroq ele_add\.co',
     'caption={Compiling a first program}]\ncroq ele_add.co'),
    ('F03-S06', r'caption=\{Build and Compile\}\]\ncroq --help',
     'caption={Listing the options}]\ncroq --help'),
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
     r'caption=\{A Complete Example: Element-Wise Addition\}\]\ncroq ele_add\.co',
     'caption={Compile and run the example}]\ncroq ele_add.co'),
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
    ('F11-S09', r'caption=\{Putting it together\}\]\ncroq kernel\.co',
     'caption={Debug and production builds}]\ncroq kernel.co'),
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
        'A kernel signature (fragment)', 'Allocating the output',
        'The parallel body', 'Returning the result'])),
    ('F15-S05', None, caption_names(['Scalar Types (fragment)'])),
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
    # Round 30 (2.6): spanned data and storage specifiers.
    ('F18-S03', None, caption_names([
        'A spanned-data declaration', 'A standalone shape',
        'A rejected fundamental type'])),
    ('F18-S06', None, caption_names([
        'An explicit return shape', 'A deduced return shape'])),
    ('F37-S07', None, caption_names([
        'A declaration without a specifier',
        'Choosing local storage explicitly'])),
    ('F18-S01',
     r'In this section, you will learn related syntax\.',
     "The declarations that follow fix an element type, a shape and, where "
     "it matters, a storage specifier."),
    ('F18-S08',
     r'This section covered defining and managing data/buffers in Croqtile tileflow programs, including declaration syntax, initialization, storage specifiers, and efficient buffer lifetime management by the compiler\.',
     "A declaration fixes the element type and shape, an initializer can "
     "fill the buffer at that point, a storage specifier says where it "
     "lives, and the compiler manages its lifetime from there."),
    ('F37-S05', r'(?m)^Example:$',
     'A \\texttt{shared} buffer follows the same declaration syntax:'),
    # Round 31 (2.7): iteration constructs, one name per example.
    ('F15-S15', None, caption_names([
        'Indexing a with-in block', 'A tuple of indices',
        'Indices derived from a tuple', 'Nested bounded groups'])),
    ('F31-S03', None, caption_names([
        'A named parallel index', 'An unnamed parallel index'])),
    ('F31-S04', None, caption_names([
        'One parallel level', 'Two parallel levels'])),
    ('F32-S02', None, caption_names([
        'A foreach loop', 'The equivalent C++ loop'])),
    ('F32-S04', None, caption_names([
        'A bounded variable', 'A tuple of bounded variables',
        'A variable derived from a tuple'])),
    ('F32-S06', None, caption_names([
        'Indexing one shape', 'A tuple of indices',
        'The generated C++ loop', 'Several bounded groups',
        'The generated inner loop'])),
    ('F32-S07', None, caption_names([
        'A foreach loop', 'The with-in form it expands to'])),
    ('F32-S08', None, caption_names([
        'A tuple of bounded variables', 'The derived C++ loop',
        'The bounded-variable syntax'])),
    ('F05-S03', r'And there is \\texttt\{foreach\}', 'There is also \\texttt{foreach}'),
    ('F15-S14', r'However, the syntax of constructing a parallel region',
     'The syntax of constructing a parallel region'),
    ('F31-S05', r'Note that the first element', 'The first element'),
    ('F32-S01',
     r'In this section, you will learn about constructing \\emph\{bounded variable\}s using the \\texttt\{with-in\} statement and creating loops with the \\texttt\{foreach\} statement\.',
     "A \\texttt{with-in} statement declares a \\emph{bounded variable}; a "
     "\\texttt{foreach} statement iterates over its range. Together they "
     "describe a loop without spelling out its bounds in C++."),
    ('F32-S10',
     r'In this section, you explain the use of \\texttt\{with-in\} and \\texttt\{foreach\} statements in Croqtile to define and iterate over \\emph\{bounded variable\}s, which are essential for data movement tasks\.',
     "Bounded variables and the loops derived from them are what data "
     "movement code is built from."),
    # Round 32 (2.8): tile selection.
    ('F04-S15', r'(?m)^Here \\texttt', 'The notation \\texttt'),
    ('F29-S01',
     r'This section introduces the \\emph\{chunkat expression\} for the DMA statement, which is the core concept of Croqtile that simplifies data movement coding\.',
     "A \\emph{chunkat expression} names one tile of a larger buffer in a "
     "DMA statement, so the movement code never has to compute an offset "
     "by hand."),
    ('F29-S02', r'However, in Croqtile, you can use the',
     'In Croqtile, a program can use the'),
    ('F29-S04',
     r'The following code showcases an example that utilizes multiple bounded variables:',
     'The example below uses several bounded variables at once:'),
    ('F29-S06',
     r'In this section, you learned about using the \\texttt\{chunkat\} expression in Croqtile for tiling data in DMA statements, which allows dividing data into smaller chunks for efficient processing\.',
     "The \\texttt{chunkat} expression divides a buffer into smaller chunks "
     "for a DMA statement, and the shape of each chunk follows from the "
     "bound used to select it."),
    ('F29-S06', r'Wish you good luck!',
     'Section~\\ref{sec:3-6-1} puts the pieces together into a complete '
     'tileflow program.'),
    # Round 33 (2.9): DMA statements and futures.
    ('F04-S02', None, caption_names([
        'The general DMA form', 'A concrete copy'])),
    ('F04-S08', None, caption_names([
        'A future from a copy', 'Reading data and span'])),
    ('F04-S09', None, caption_names([
        'A synchronous copy', 'An asynchronous copy',
        'An asynchronous tma copy'])),
    ('F14-S08', None, caption_names([
        'The tileflow program', 'The equivalent C++ loop'])),
    ('F28-S07', None, caption_names([
        'Two asynchronous copies', 'Waiting on both futures'])),
    ('F28-S06',
     r'So far, you have only used simple data expressions, either',
     'The data expressions used so far have been simple, either'),
    ('F28-S06',
     r'However, the use of simple \\textbf\{data expressions\} is rare in practice\. In the next section, this section introduces',
     'In practice, simple \\textbf{data expressions} are rare. The next section introduces'),
    # Round 34 (2.10): events and coordination.
    ('F35-S02', None, caption_names([
        'Declaring a shared event', 'Declaring an event array',
        'Triggering an event array', 'Waiting on an event'])),
    ('F35-S02',
     r'Note that since it is a \\texttt\{shared\} event, it can only be declared inside \\texttt\{parallel-by\} block and used for parallel threads\.',
     "A \\texttt{shared} event can only be declared inside a "
     "\\texttt{parallel-by} block and used by parallel threads."),
    ('F35-S02',
     r'Note it is allowed to use comma-separated expression to define either event and event array in a single line\.',
     "A comma-separated expression can declare both an event and an event "
     "array on one line."),
    ('F35-S02',
     r'Therefore, triggering an event is used to unblocking code for the \\emph\{wait statement\}\.',
     "Triggering an event is what unblocks the code waiting on a "
     "\\emph{wait statement}."),
    ('F35-S02', r'Note that, in Croqtile semantics,', 'In Croqtile semantics,'),
    ('F35-S05',
     r'In this section, you learnt .event. that can ochestrate asynchronous \\texttt\{inthreads\} blocks\.',
     "Events let asynchronous \\texttt{inthreads} blocks agree on an order."),
    # Round 35 (2.11): branching and early return keep their source prose.
    ('F07-S12',
     r'The \\texttt\{yield\} keyword will simply generate a \\texttt\{return;\} instruction to get out of device code\.',
     "The \\texttt{yield} keyword generates a \\texttt{return;} instruction "
     "that leaves the device code for that instance."),
    # Round 36 (2.12): the four contraction steps.
    ('F06-S02',
     r'Every tensor-contraction kernel follows the same rhythm:',
     "Every contraction kernel follows the same rhythm, and the spelling "
     "makes each step visible:"),
    # Round 37 (2.13): C++ embedding, calls, macros and the preprocessor.
    ('F10-S07', None, caption_names([
        'A macro inside a cpp string', 'Its expanded form'])),
    ('F23-S02', None, caption_names([
        'An object-like macro', 'A macro with parameters'])),
    ('F24-S03', None, caption_names([
        'The general call syntax', 'An illegal call site'])),
    ('F23-S02',
     r'Note that, till now Croqtile does only support macros with parameters\. It means code like:',
     "The preprocessor currently supports only macros without parameters, "
     "so code like:"),
    ('F23-S04',
     r'Note that, the capability of croqtile preprocessor is still enhancing\. But it is likely that you would not implement full C preprocessing support\. Croqtile would not pick up existing C features unless people find it is necessary\.',
     "The preprocessor is still being extended, and full C preprocessing "
     "support is not planned: a C feature is picked up only when a program "
     "actually needs it."),
    ('F24-S03',
     r'Note that \\textbf\{Croqtile transpilation process would not apply any check',
     '\\textbf{Croqtile transpilation process would not apply any check'),
    ('F24-S03', r'Therefore, in the following example,', 'In the following example,'),
    ('F24-S07',
     r'In this section, you learned the syntax to invoke device functions, including normal ones and those triggers function template instantiations\.',
     "A \\texttt{call} statement invokes a device function, with or without "
     "template arguments."),
    ('F24-S07', r'Programmers must code in cautious since', 'Care is needed since'),
    # Round 38 (2.14): compile-time and runtime inspection.
    ('F11-S02', None, caption_names([
        'A compile-time shape print', 'The printed shape'])),
    ('F11-S03', None, caption_names([
        'A runtime inspect kernel', 'A print inside a parallel block'])),
    # Round 39 (2.15): quick reference.
    ('F04-S25',
     r'By the end of this part of the guide, the new pieces of syntax should mean the following:',
     'The syntax collected here means the following:'),
    ('F05-S18', r'The next chapter replaces the scalar',
     'Section~\\ref{sec:3-7} replaces the scalar'),
    ('F05-S18', r'\\textbf\{tensor-core\} operations', 'contraction operations'),
    # Round 40 (3.1): program structure and compilation models.
    ('F14-S07', None, caption_names([
        'A device helper called from tileflow',
        'The cok wrapper for separate sources'])),
    ('F14-S01',
     r'This section introduces the fundamental structure of \\emph\{Croqtile\} programs and their associated terminology\.',
     "A \\file{.co} program has a fixed structure: host code, device code "
     "and a tileflow function, each treated differently by the compiler."),
    ('F14-S04', r'Thus, the Croqtile compiler functions as a end-to-end compiler',
     'The Croqtile compiler functions as an end-to-end compiler'),
    # Round 41 (3.2): syntax design and the type system.
    ('F15-S07', None, caption_names([
        'An mdspan declaration', 'A shape literal',
        'A shape with one dimension changed', 'The same shape again'])),
    ('F37-S02',
     r'However, the spanned type by itself does not allocate storage or memory for the data\.',
     'The spanned type by itself does not allocate storage for the data.'),
    # Round 42 (3.3): symbolic shapes and value numbering.
    ('F19-S05', None, caption_names([
        'A kernel with anonymous dimensions',
        'The transpiled signature', 'The generated shape check'])),
    ('F21-S02', None, caption_names([
        'A shape expression', 'Its value-number graph'])),
    ('F21-S05', None, caption_names([
        'A constant expression', 'The simplified graph',
        'The folded graph'])),
    ('F16-S05', r'So far, you have seen mdspan with constant values, which is sufficient',
     'An mdspan with constant values is sufficient'),
    ('F16-S05', r'However, in some scenarios, a runtime shape',
     'In some scenarios, a runtime shape'),
    ('F19-S07',
     r'In this section, you discussed \\emph\{anonymous dimension\} and \\emph\{symbolic dimension\} in Croqtile, which support dynamic shapes required in certain scenarios\.',
     "Anonymous and symbolic dimensions are the two ways Croqtile "
     "represents a shape that is not fully known when the program is "
     "written."),
    # Round 43 (3.4): restore the if-else concept archived with a platform
    # clause, and remove the tutorial bookends around masking.
    ('F15-S13',
     r'Croqtile has significant difference with C\+\+ on parallelization, loop, etc\.',
     "Croqtile differs from C++ in how it expresses parallelization and "
     "loops, but it follows C++ in using \\texttt{if}-\\texttt{else} blocks "
     "to handle branches inside a program."),
    ('F31-S01',
     r'This section will guide you through using the \\texttt\{parallel-by\} block to implement parallel code in Croqtile\.',
     "A \\texttt{parallel-by} block declares a set of independent "
     "instances; the space specifier says how those instances are grouped."),
    ('F31-S11',
     r'In this section, you explored[^.]*\.',
     "A \\texttt{parallel-by} block declares independent instances and a "
     "space specifier groups them; masking then decides which of those "
     "instances actually execute a statement."),
    ('F31-S11',
     r'However, while parallelism enhances efficiency, loops remain essential\. In the next section, this section introduces how to build loops in Croqtile\.',
     "Loops remain necessary for work that has to happen in order, and "
     "Section~\\ref{sec:2-7} introduces their Croqtile forms."),
    ('F33-S04',
     r'There are some code structures in Croqtile that implies implicit masking\.',
     'Some code structures in Croqtile imply implicit masking.'),
    ('F34-S10',
     r'In this section, you examined the motivation for thread masking[^.]*\.',
     "Masking is what allows divergent execution inside an otherwise "
     "uniform parallel region."),
    # Round 44 (3.4.1): the worked example keeps its code.
    ('F05-S13',
     r'Start with just \\texttt\{parallel\} and \\texttt\{\.at\(\)\}',
     "The first version uses only \\texttt{parallel} and \\texttt{.at()}, "
     "before any tiling or data movement is added"),
    # Round 45 (3.4.2): trace one output element.
    ('F05-S17', r'Which block and thread own it\?',
     'Working out which block and thread own it fixes the index arithmetic '
     'for the rest of the kernel.'),
    # Round 46 (3.5): data expressions.
    ('F29-S05', r'However, using the', 'Using the'),
    # Round 47 (3.5.1): arbitrary-offset windows.
    ('F09-S07', None, caption_names([
        'A window at a constant offset', 'A window at a runtime offset'])),
    ('F09-S07',
     r'This is a \\texttt\{\[16, 16\]\} slice starting at',
     'The expression selects a \\texttt{[16, 16]} slice starting at'),
    # Round 48 (3.5.2): strided tiles. The source prose is already continuous.
    # Round 49 (3.5.3): zero padding. The source prose is already continuous.
    # Round 50 (3.5.4): layout reinterpretation.
    ('F09-S10', None, caption_names([
        'A flat strip reinterpreted as a matrix',
        'A movement statement using the reinterpretation'])),
    # Round 51 (3.5.10): summary and hand-off.
    ('F09-S12',
     r'(?s)Section~\\ref\{sec:2-13\} covers C\+\+ interop:.*\Z',
     "Section~\\ref{sec:3-6} takes the tile expressions above and uses them "
     "as the source and destination of movement statements. For the cases "
     "where a program needs plain C++, Section~\\ref{sec:2-13} collects the "
     "\\texttt{\\_\\_device\\_\\_} functions, preprocessor guards and "
     "injected source text."),
    # Round 52 (3.6): DMA grammar and async patterns.
    ('F04-S21', None, caption_names([
        'A host-computable tile shape',
        'A runtime tile index that is rejected'])),
    ('F15-S18', None, caption_names([
        'An asynchronous DMA declaration',
        'The same declaration with storage'])),
    ('F03-S07',
     r'The skeleton of every Croqtile program is now in your hands:',
     'The constructs introduced so far assemble into one skeleton:'),
    ('F39-S01',
     r'In this section, you will learn more advanced syntax of \\emph\{DMA statements\}[^.]*\.',
     "The \\emph{DMA statement} has a form for each of those levels, from a "
     "single copy to a chained non-blocking transfer."),
    # Round 53 (3.6.1): DMA matmul.
    ('F05-S14', r'What changed from the scalar version:',
     "The scalar version indexed one element at a time. The tiled version "
     "first copies a tile into shared storage, then computes on that tile:"),
    # Round 54 (3.6.2): multi-buffering.
    ('F38-S01', r'Note that the \\textbf\{select\} expressions',
     'The \\textbf{select} expressions'),
    # Round 55 (3.6.3): pipelined kernel. The source prose is continuous.
    # Round 56 (3.7): contraction variants and layout contracts.
    ('F06-S07', None, caption_names([
        'A sparse contraction', 'A scaled contraction',
        'Another contraction layout'])),
    ('F09-S03', None, caption_names([
        'A swizzled copy', 'A swizzled mma load'])),
    ('F04-S18', r'So \\texttt\{\.zfill\} is', 'The \\texttt{.zfill} form is'),
    # Round 57 (3.7.1): one cooperative group.
    ('F06-S04', r'In Croqtile, that corresponds to \\texttt\{: group\}\.',
     'One cooperative group is written \\texttt{: group} in Croqtile.'),
    # Round 58 (3.7.2): widening the cooperation scope.
    ('F06-S05', r'But the four-step syntax', 'The four-step syntax'),
    # Round 59 (3.7.3): two cooperative groups.
    ('F06-S06', r'--- With MMA', '--- with MMA'),
    # Round 60 (3.7.9): MMA syntax summary. The reference table is unchanged.
    # Round 61 (3.7.10): design summary. The source prose is continuous.
    # Round 62 (3.8): coordination primitives.
    ('F35-S04', None, caption_names([
        'A single-instance deadlock', 'A credit-count deadlock'])),
    ('F07-S01', r'But predicated execution', 'Predicated execution'),
    ('F34-S07', r'Note that, only the', 'Only the'),
    # Round 63 (3.8.1): one producer, one consumer.
    ('F07-S04', r'But sharing an address space', 'Sharing an address space'),
    # Round 64 (3.8.2): the 1P1C skeleton already has a repaired hand-off.
    # Round 65 (3.8.3): persistent scheduling.
    ('F07-S09', r'In Chapters 3--4, the grid grew',
     'In the earlier kernels, the grid grew'),
    # Round 66 (3.8.4): host-level concurrency.
    ('F36-S01', None, caption_names([
        'An asynchronous launch', 'The synchronous form'])),
    # Round 67 (3.8.5): full event pipeline. The source prose is continuous.
    # Round 68 (3.8.10): coordination summary. The source table is unchanged.
    # Round 69 (3.9): C++ interoperability boundary.
    ('F24-S06', None, caption_names([
        'A templated device helper', 'A call that instantiates it'])),
    ('F26-S01', None, caption_names([
        'A cok block around device code',
        'The block after factoring target-specific details'])),
    ('F25-S08', r'However, there are some features yet to support:',
     'Some features are not supported yet:'),
    ('F25-S08', r'Note that \\textbf', '\\textbf'),
    ('F23-S05',
     r'One important notices about croqtile preprocessing is that it is triggered much earlier than',
     'Croqtile preprocessing runs much earlier than'),
    # Round 70 (3.9.1): calling a device helper. Source prose is continuous.
    # Round 71 (3.9.2): verbatim C++ injection.
    # Round 72 (3.9.10): interoperability summary.
    ('F10-S13',
     r'(?s)Section~\\ref\{sec:1-9\} turns to the other side.*\Z',
     "Section~\\ref{sec:3-11} turns to the other side of the workflow: what "
     "to do when the kernel compiles but the output is wrong --- debugging, "
     "verbose modes and systematic narrowing. The two worked programs in "
     "Section~\\ref{sec:3-10} put the interoperability mechanisms into "
     "context."),
    # Round 73 (3.10): tileflow patterns.
    ('F40-S01',
     r'This section will extend your discussion on tiling optimization in Croqtile by examining some practical examples\.',
     "Two complete programs are enough to see why the tiling constructs are "
     "shaped the way they are."),
    ('F09-S04', r'This flexibility is a double-edged sword\.',
     'That flexibility has a cost.'),
    # Round 74 (3.10.1): tiled matmul worked example.
    ('F40-S02', r'This example demonstrates how', 'The code shows how'),
    # Round 75 (3.10.2): rgb2gray worked example.
    ('F40-S05', r'The algorithm converts a color \(RGB\) image',
     'It converts a color (RGB) image'),
    ('F40-S05',
     r'This example shows how physical parallelism work together with virtual parallelism\.',
     "It shows how physical parallelism works together with virtual "
     "parallelism."),
    # High-level purpose and development benefits for the opening section.
    ('F01-S01',
     '(?s)\\A.*\\Z',
     'Much of the development effort in a high-performance kernel goes into organising the '
     'computation. Beyond implementing the arithmetic, the programmer has to arrange data '
     'storage, divide parallel work and coordinate execution. These choices interact, so a '
     'change can require updates in several places. Different input sizes or target '
     'machines call for another round of validation and tuning. Implementing, maintaining '
     'and revising these arrangements accounts for much of the cost.'),
    ('F01-S02',
     '(?s)\\A.*\\Z',
     'Croqtile is a C++ embedded DSL for high-performance kernel programming, designed to '
     'make this work easier while preserving control over performance. It provides a '
     'direct way to describe the organisation of computation, data movement and '
     'coordination in a compact program. The programmer chooses how the kernel works; the '
     'compiler derives related information, checks consistency and generates the '
     'underlying code. This reduces the repetitive work involved in expressing and '
     'maintaining those choices.'),
    ('F01-S03',
     '(?s)\\A.*\\Z',
     'The resulting programs are easier to read as a whole and revise as requirements '
     'change. They can accommodate different input sizes, and the abstractions introduce '
     'no additional runtime overhead. Compact source and useful compiler feedback also '
     'make the same process accessible to AI-assisted development and tuning.'),
    ('F12-S01', '(?s)\\A.*\\Z', ''),
    ('F13-S01', '(?s)\\A.*\\Z', ''),
    # Standalone inline-code paragraphs need a prose prefix for translation.
    ('F08-S07', '(?m)^\\\\texttt\\{\\(1:', 'The slice \\texttt{(1:'),
    ('F16-S04', '(?m)^\\\\texttt\\{new', 'The expression \\texttt{new'),
    # Round 76 (3.11): diagnostics methodology. Source prose is continuous.
    # Round 77: code correctness. Upstream examples use syntax this toolchain
    # rejects (swizzle modes, mma.commit) or forms the language rejects
    # (reading .data from a buffer, a bare return type).
    ('F09-S03', r'swiz<3>', 'swiz<128>'),
    ('F09-S03', r'swiz\\textless\{\}3\\textgreater\{\}', r'swiz\textless{}128\textgreater{}'),
    ('F09-S03',
     r'\\textbf\{Swizzle levels\.\}.*?granularity\.',
     r'\textbf{Swizzle width.} The template argument is the swizzle width in bytes, and the examples use \texttt{swiz\textless{}128\textgreater{}}. A wider swizzle defeats wider conflict patterns but requires tile extents that line up with that width.'),
    ('F09-S05', r'swiz<3>', 'swiz<128>'),
    ('F09-S09', r'swiz<3>', 'swiz<128>'),
    ('F08-S10', r'(?m)^[ \t]*mma\.commit;[ \t]*\n', ''),
    ('F09-S05', r'(?m)^[ \t]*mma\.commit;[ \t]*\n', ''),
    ('F08-S11', r', \\texttt\{mma\.commit\}, and', ' and'),
    ('F08-S11',
     r'\\texttt\{mma\.commit\} is the fence.*?overwrites the buffer\.',
     "The consumer signals \\co@empty[stage]@ only after the tile's matrix operations have completed; releasing a stage early lets the producer overwrite operands the MMA is still reading."),
    ('F06-S09', r'; \\texttt\{mma\.commit\} fences pipeline stages', ''),
    ('F07-S07', r'\\texttt\{swap\}, and \\texttt\{mma\.commit\} into', r'\texttt{swap} into'),
    ('F10-S04',
     r'score = shared_scores\.data\.span_as\(\|shared_scores\.data\.span\|\)\.at\(gid # tid\);',
     'flat_scores = shared_scores.span_as([|shared_scores.span|]);\n          score = flat_scores.at(gid # tid);'),
    ('F10-S11', r'__co__ foo\(\) \{', '__co__ auto foo() {'),
    ('F23-S04', r'__co__ foo\(\) \{', '__co__ auto foo() {'),
    # Round 78: drop GPU-backend-only vocabulary that the tma.copy / group-4
    # rewrite leaves behind in prose.
    ('F28-S05', r'A transfer may map to TMA or to a sequence',
     'A transfer may map to a dedicated copy engine or to a sequence'),
    ('F09-S03', r'Swizzle is not a TMA-specific feature\..*?same in both cases\.',
     r'Swizzle is a property of the copy, not of a particular engine: \texttt{dma.\allowbreak{}copy.\allowbreak{}swiz\textless{}N\textgreater{}} produces the same shared-memory layout regardless of how the data gets there.'),
    ('F04-S01', r'write DMA and TMA code', 'write DMA code'),
    ('F09-S12', r'(?m)^\\texttt\{dma.*?& TMA copy with swizzle mode.*?\n', ''),
    ('F09-S12', r'DMA copy with swizzle mode \\texttt\{N\} \(0--3\); same layout as TMA',
     r'Copy with swizzle width \\texttt{N} in bytes'),
    ('F04-S08', r'The result of a DMA or TMA statement', 'The result of a DMA statement'),
    ('F04-S09', r'The same pattern applies to TMA:', 'The same pattern applies to asynchronous copies:'),
    ('F04-S09', r'An asynchronous tma copy', 'An asynchronous copy'),
    # Round 79: `block` is a parallel-level keyword, so it cannot name a
    # variable in the chunkat example.
    ('F04-S02', r'chunkat\(block\)', 'chunkat(blk)'),
    # Round 80: make the listings self-contained and logically correct.
    # A declared span starts uninitialized, so accumulating into `output`
    # with `+=` needs an explicit clearing step. The matmul listings relied
    # on prose to say so; the code now does it.
    ('F05-S13',
     re.escape('  parallel {p, q} by [16, 64] {\n    foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]\n      output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n);\n  }'),
     '  parallel {p, q} by [16, 64] {\n    foreach {m, n} in [128 / #p, 256 / #q]\n      output.at(p#m, q#n) = 0;\n    foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]\n      output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n);\n  }'),
    ('F05-S14',
     re.escape('  parallel {px, py} by [8, 16] : block {\n    foreach {tile_k} in [16] {'),
     '  parallel {px, py} by [8, 16] : block {\n    parallel {qx, qy} by [16, 16] : thread\n      output.at(px#qx, py#qy) = 0;\n    foreach {tile_k} in [16] {'),
    ('F08-S02',
     re.escape('    parallel {qx, qy} by [16, 16] : thread {\n    with tile_k in 16 {'),
     '    parallel {qx, qy} by [16, 16] : thread {\n    output.at(px#qx, py#qy) = 0;\n    with tile_k in 16 {'),
    # A void kernel with an output parameter never delivers its result: the
    # caller receives only what the function returns.
    ('F07-S08',
     re.escape('__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {'),
     '__co__ auto matmul(global f16 [M, K] lhs, global f16 [N, K] rhs) {\n  f16 [M, N] output;'),
    ('F07-S09',
     re.escape('__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {'),
     '__co__ auto matmul(global f16 [M, K] lhs, global f16 [N, K] rhs) {\n  f16 [M, N] output;'),
    ('F08-S10',
     re.escape('__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {'),
     '__co__ auto matmul(global f16 [M, K] lhs, global f16 [N, K] rhs) {\n  f16 [M, N] output;'),
    ('F07-S08', r'\n\}\n\\end\{lstlisting\}', '\n  return output;\n}\n\\end{lstlisting}'),
    ('F07-S09', r'\n\}\n\\end\{lstlisting\}', '\n  return output;\n}\n\\end{lstlisting}'),
    ('F08-S10', r'\n\}\n\\end\{lstlisting\}', '\n  return output;\n}\n\\end{lstlisting}'),
    # The MMA example uses MMA_M / MMA_N / MMA_K without defining them.
    ('F06-S04', re.escape('#define WARP_K 16\n'),
     '#define WARP_K 16\n#define MMA_M 16\n#define MMA_N 16\n#define MMA_K 16\n'),
    # `M` is a symbolic extent; read it from the operand instead of naming it.
    ('F24-S04', re.escape('call bar(input.data, M, buffer.span(0));'),
     'call bar(input.data, input.span(0), buffer.span(0));'),
    # The runtime header and namespace are `choreo` (aliased as `croq`).
    ('F15-S21', re.escape('#include "croqtile.h"'), '#include "choreo.h"'),
    ('F19-S05', re.escape('int tile_m = 32, tile_m = 8, tile_k = 16;'),
     'int tile_m = 32, tile_n = 8, tile_k = 16;'),
    # `out_store_s` is assigned inside the inner loop but read after it; give
    # it a placeholder before the loop, the same way the double-buffer example
    # seeds its spare futures with `dma.any`.
    ('F40-S03',
     re.escape('          parallel q by 1 {\n            with index = {m_tile_s, k_tile_s, n_tile_s} in [4, 4, 8] {'),
     '          out_store_s = dma.any;\n          parallel q by 1 {\n            with index = {m_tile_s, k_tile_s, n_tile_s} in [4, 4, 8] {'),
    # The operands must be resident before the dot product consumes them.
    ('F40-S03',
     re.escape('                call dot_general_fp32(lhs_load_s.data, rhs_load_s.data,'),
     '                wait lhs_load_s, rhs_load_s;\n                call dot_general_fp32(lhs_load_s.data, rhs_load_s.data,'),
    # The final store is awaited before the function returns the buffer it fills.
    ('F40-S03',
     re.escape('          out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;'),
     '          out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;\n          wait out_store;'),
    # Round 81: the runtime library is spelled `choreo` (with `croq` as an
    # alias). The guide used `croqtile` for the header, the namespace, the
    # assertion helper and a factory name that does not exist in that spelling.
    ('*', '', _rename_runtime),
    # Round 82: residual code defects the runtime pass turned up.
    # F23-S02 declares the left operand `[M, N]` but passes an `[M, K]` buffer.
    ('F23-S02', re.escape('__co__ auto matmul(f32 [M, N] lhs, f32 [N, K] rhs) { /*...*/ }'),
     '__co__ auto matmul(f32 [M, K] lhs, f32 [N, K] rhs) { /*...*/ }'),
    # F24-S06 computed a [16, 4] tile per block, wrote it nowhere, and looped
    # over the row blocks only, so just the diagonal of the tile grid was
    # covered. Index both axes, write the tile back, and mark the movement
    # block-uniform so the thread-level DMA rule does not apply.
    ('F24-S06', re.escape('  parallel p by 6 {\n    shared s32 [output.span / #p] buffer;   // shape: [16, 4]\n    lhs_load = dma.copy lhs.chunkat(p, _) => shared;\n    rhs_load = dma.copy rhs.chunkat(_, p) => shared;\n    call matmul_kernel<buffer.span(0), buffer.span(1), 72>(lhs_load.data, rhs_load.data, buffer);\n  }'), '  parallel {p, q} by [6, 6] : block {\n    shared s32 [output.span / {#p, #q}] buffer;   // shape: [16, 4]\n    lhs_load = dma.copy lhs.chunkat(p, _) => shared;\n    rhs_load = dma.copy rhs.chunkat(_, q) => shared;\n    call matmul_kernel<buffer.span(0), buffer.span(1), 72>(lhs_load.data, rhs_load.data, buffer);\n    dma.copy buffer => output.chunkat(p, q);\n  }'),
    # F19-S05 quotes compiler output, so it keeps the compiler's spelling:
    # `choreo::`, const-reference parameters, and `runtime_check`.
    ('F19-S05', re.escape('croq::spanned_view<croq::s32, 2> lhs'),
     'const choreo::spanned_view<choreo::s32, 2> & lhs'),
    ('F19-S05', re.escape('croq::spanned_view<croq::s32, 2> rhs'),
     'const choreo::spanned_view<choreo::s32, 2> & rhs'),
    ('F19-S05', re.escape('croq_assert('), 'choreo::runtime_check('),
    # keep the quoted listing aligned after the longer parameter type
    ('F19-S05', re.escape('\n                                const choreo::spanned_view'),
     '\n                                  const choreo::spanned_view'),
    ('F19-S05', re.escape('\n                "the 0 dimension'),
     '\n                        "the 0 dimension'),
    ('F19-S05', re.escape('\n                "dimension \'K\''),
     '\n                        "dimension \'K\''),
    ('F19-S05', re.escape('\n                "with a dimension value'),
     '\n                        "with a dimension value'),
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
            # Refresh changed upstream assets so rebuilds cannot retain stale figures.
            if not os.path.isfile(dest) or not filecmp.cmp(c, dest, shallow=False):
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
    if uid in {'F01-S03', 'F02-S02', 'F03-S06', 'F10-S12', 'F22-S05'}:
        return ''  # Its former prerequisites list is now continuous opening prose.
    if uid == 'F02-S05':
        return r'\subsection*{Compiler Options}'
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
                    out.append('\\includegraphics[width=\\linewidth,'
                               'height=0.56\\textheight,keepaspectratio]{' + name + '}')
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
            frag = strip_gpu_backend(frag)
            if uid in OMITTED_UNITS:
                manifest.append((uid, rel, slot, 'omitted', title))
                continue
            for ouid, pat, repl in UNIT_OVERRIDES:
                if ouid in (uid, '*'):
                    if callable(repl):
                        frag = repl(frag)
                    else:
                        frag = re.sub(pat, lambda m, t=repl: t, frag)
            first_chapter_refs = {
                'F02-S04': {'sec:1-5': 'sec:1-3'},
                'F02-S06': {'sec:1-7': 'sec:1-4'},
                'F04-S01': {'sec:1-7': 'sec:1-3'},
                'F04-S02': {'sec:1-7': 'sec:1-3'},
                'F10-S02': {'sec:1-9': 'sec:1-5'},
            }
            for old_ref, new_ref in first_chapter_refs.get(uid, {}).items():
                frag = frag.replace(old_ref, new_ref)
            editorial_path = os.path.join(PG, 'editorial', 'getting-started', uid + '.md')
            if os.path.isfile(editorial_path):
                with open(editorial_path, encoding='utf-8') as editorial_file:
                    editorial_lines = editorial_file.read().splitlines()
                frag = convert_body(editorial_lines, uid, rel, fid, title)
                if uid == 'F03-S06':
                    frag = frag.replace(r'\end{figure}',
                                        r'\caption{Compiling and running ele\_add, followed by compiler help}' + '\n' + r'\end{figure}')
                captions = {'F22-S05': ['Inspecting intermediate output and compiler help'],
                            'F02-S06': ['Testing sample operators']}
                if uid in captions:
                    frag = caption_names(captions[uid])(frag)
            frag = polish_figures(frag, uid, esc)
            frag = clean_fragment(frag)
            # An empty registered group heading contributes no heading at all;
            # otherwise the PDF shows a title with nothing beneath it.
            heading_tex = heading_for(title, uid, slot) if frag.strip() else ''
            text = (heading_tex + '\n\n' if heading_tex else '') + frag
            text = polish_prose(text, uid)
            text = strip_gpu_backend(text)
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
            chapter_order = {
                '1.2': ['F02-S01', 'F02-S02', 'F22-S01', 'F22-S03',
                        'F02-S03', 'F22-S02', 'F02-S04'],
                '1.3': ['F03-S01', 'F03-S03', 'F03-S06', 'F02-S05',
                        'F22-S05', 'F02-S06', 'F22-S04'],
                '1.5': ['F11-S05', 'F11-S06', 'F11-S08', 'F11-S09',
                        'F11-S07', 'F08-S13', 'F11-S10'],
                '2.10': ['F34-S06', 'F34-S09', 'F35-S01', 'F35-S02',
                         'F07-S06', 'F07-S07', 'F35-S05'],
                '3.1': ['F14-S01', 'F14-S02', 'F14-S04', 'F14-S07',
                        'F31-S08', 'F12-S02', 'F14-S09'],
                '3.7': ['F06-S01', 'F06-S03', 'F04-S18', 'F04-S19',
                        'F09-S03', 'F06-S07'],
            }
            ordered_units = chapter_order.get(slot, slot_units[slot])
            if set(ordered_units) != set(slot_units[slot]):
                raise ValueError('Chapter order differs from registry: ' + slot)
            for uid in ordered_units:
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
             '`HARDWARE-NOTES.md`; `placeholder` units are empty or TODO pages. '
             'Units marked `omitted` repeat introductory interop material '
             'covered in chapters 2 and 3; their upstream sources remain available.',
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
          '(kept %d, moved %d, placeholder %d, omitted %d)' % (
              sum(1 for m in manifest if m[3] == 'kept'),
              sum(1 for m in manifest if m[3] == 'moved'),
              sum(1 for m in manifest if m[3] == 'placeholder'),
              sum(1 for m in manifest if m[3] == 'omitted')))
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
