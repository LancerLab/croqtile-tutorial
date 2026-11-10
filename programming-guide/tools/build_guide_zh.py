#!/usr/bin/env python3
"""Build the Chinese edition of the Croqtile Programming Guide.

The English edition is the source of truth.  Code listings, tables and figures
are extracted from ``build/units/*.tex`` and ``content/*.tex`` verbatim; the
translation files supply only prose, headings, captions and table wording.

Workflow
--------
    python3 tools/build_guide_zh.py templates   # write translation/templates/*.md
    # ... translate every TAG: line into translation/zh/*.md ...
    python3 tools/build_guide_zh.py check       # coverage / terminology check
    python3 tools/build_guide_zh.py build       # write MD + HTML + PDF

A translation file has one ``=== KEY`` block per content key, followed by one
line per token.  The tag sequence must match the English template exactly::

    H  heading                 P  prose paragraph      I  lead-in
    N  where-to-go-next        T  complete Markdown table
    L  list items, " | " separated                 B  callout text
    C  code-listing caption    F  figure caption

Only the core domain terms listed in ``CORE_TERMS`` are written as
``中文(English)`` at their first occurrence; every other annotation is
stripped by ``normalize``. See ``TRANSLATION-GLOSSARY.md`` section 5.
"""
import argparse
import csv
import html as html_mod
import json
import re
import sys
from pathlib import Path

PG = Path(__file__).resolve().parents[1]
TPL = PG / 'translation' / 'templates'
ZH = PG / 'translation' / 'zh'
OUT_MD = PG / 'CROQTILE-GUIDE-ZH.md'
OUT_HTML = PG / 'CROQTILE-GUIDE-ZH.html'
OUT_PDF = PG / 'output' / 'croqtile-programming-guide-zh.pdf'
CH1_EDITORIAL = PG / 'translation' / 'zh' / 'chapter1-editorial.json'
FORBIDDEN = ['内核', '瓦片', '零成本抽象', '核心程序', '协做组', '宿主端',
             '编译期', '鳄霸', '本节将介绍', '本章将介绍']
# 术语双语格式固定为「中文(English)」：中文译名后紧跟半角括号与英文原文。
# 全角形式「中文（English）」视为格式错误，由 check() 报告。
FULLWIDTH_BILINGUAL = re.compile(
    r'[\u4e00-\u9fff]{1,12}（[A-Za-z][A-Za-z0-9 ,.\-_+]{0,60}）')
# 首次出现标注：中文译名 + 半角括号内的英文术语。
BILINGUAL = re.compile(
    r'([\u4e00-\u9fff]+)\(([A-Za-z0-9][A-Za-z0-9 ,.\-_+/<>]*)\)')

# 只有这些核心领域术语在首次出现时保留英文原文。通用词（性能问题、模型、程序、机器、
# 函数、变量、类型、编译、线程等）中文译名直观且无歧义，附英文只会打断阅读，一律不标注。
CORE_TERMS = {
    'kernel', 'tile', 'ituple', 'integer-tuple', 'future',
    'mpmd', 'single program multiple data, spmd', 'embedded dsl',
    'symbolic dimension', 'value numbering', 'rank',
    'storage specifier', 'data movement', 'contraction',
    'cooperative group', 'zero-cost', 'multi-buffering', 'masking',
}

CHAPTER_FILES = {'1': 'ch01-getting-started.tex',
                 '2': 'ch02-language-basics.tex',
                 '3': 'ch03-programming-guide.tex'}
CHAPTER_TITLES = {'1': 'Getting Started', '2': 'Language Basics',
                  '3': 'Programming Guide'}

# ---------------------------------------------------------------------------
# LaTeX -> readable inline markup (English templates only)
# ---------------------------------------------------------------------------

SPECIAL = {
    r'\_': '_', r'\#': '#', r'\&': '&', r'\%': '%', r'\{': '{', r'\}': '}',
    r'\textless{}': '<', r'\textgreater{}': '>', r'\allowbreak{}': '',
    r'\(': '$', r'\)': '$', r'\rightarrow': '→', r'\times': '×',
}


def unescape(s):
    for a, b in [('\\_', '_'), ('\\#', '#'), ('\\&', '&'), ('\\%', '%'),
                 ('\\{', '{'), ('\\}', '}'), ('\\allowbreak{}', ''),
                 ('\\textless{}', '<'), ('\\textgreater{}', '>')]:
        s = s.replace(a, b)
    return s


def tex_inline(s):
    """Render generated TeX as readable text with `code` marks."""
    s = re.sub(r'\\co@([^@]+)@', lambda m: '`' + unescape(m.group(1)) + '`', s)
    for cmd in ('texttt', 'file'):
        s = re.sub(r'\\' + cmd + r'\{([^}]*)\}',
                   lambda m: '`' + unescape(m.group(1)) + '`', s)
    for cmd, mark in (('emph', '*'), ('textbf', '**'), ('textit', '*')):
        s = re.sub(r'\\' + cmd + r'\{([^}]*)\}',
                   lambda m: mark + unescape(m.group(1)) + mark, s)
    s = re.sub(r'\\citep\{[^}]*\}', '[参考文献]', s)
    s = re.sub(r'(?:Sections|Section|Chapters|Chapter|Appendix)~\\ref\{(?:sec|ch|app):([^}]*)\}',
               lambda m: m.group(1).replace('-', '.'), s)
    s = re.sub(r'\\ref\{(?:sec|ch|app):([^}]*)\}',
               lambda m: m.group(1).replace('-', '.'), s)
    s = s.replace('~', ' ')
    for a, b in SPECIAL.items():
        s = s.replace(a, b)
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[{}]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


# ---------------------------------------------------------------------------
# Tokenise generated TeX
# ---------------------------------------------------------------------------

ENV_NAMES = ('lstlisting', 'table', 'figure', 'itemize', 'enumerate',
             'center', 'keypoint', 'pitfall', 'tabularx')
BEGIN = re.compile(r'\\begin\{(' + '|'.join(ENV_NAMES) + r')\}')
HEADING = re.compile(r'\\(chapter|section|subsection|paragraph)\*?\{(.*)\}\s*$')


def split_tokens(text, keep_leading_inline=False):
    """Return [(tag, payload), ...] for one generated TeX fragment."""
    lines = text.split('\n')
    tokens, prose = [], []
    i = 0

    def flush():
        if prose:
            p = ' '.join(x.strip() for x in prose if x.strip()).strip()
            if p and not p.startswith(('\\centering', '\\small', '\\rule')):
                tokens.append(('P', tex_inline(p)))
            prose.clear()

    while i < len(lines):
        stripped = lines[i].strip()
        m = HEADING.match(stripped)
        if m:
            flush()
            tokens.append(('H', tex_inline(m.group(2))))
            i += 1
            continue
        mb = BEGIN.match(stripped)
        if mb:
            flush()
            name = mb.group(1)
            depth = 1
            buf = [lines[i]]
            i += 1
            while i < len(lines) and depth:
                l = lines[i]
                if l.strip().startswith('\\begin{' + name + '}'):
                    depth += 1
                if l.strip().startswith('\\end{' + name + '}'):
                    depth -= 1
                buf.append(l)
                i += 1
            tokens.append(classify_env(name, '\n'.join(buf)))
            continue
        if not stripped:
            flush()
        elif stripped.startswith(('\\', '%')) and not prose and not stripped.startswith(
                (r'\texttt{', r'\textbf{', r'\emph{', r'\textit{', r'\co@', r'\file{')):
            pass
        else:
            prose.append(lines[i])
        i += 1
    flush()
    return [t for t in tokens if t[1] != '' or t[0] in ('C', 'F')]


def classify_env(name, block):
    if name == 'lstlisting':
        cap = re.search(r'caption=\{([^}]*)\}', block)
        code = re.sub(r'(?s)^.*?\\begin\{lstlisting\}(?:\[[^\]]*\])?\r?\n', '', block)
        code = re.sub(r'(?s)\\end\{lstlisting\}.*$', '', code)
        lang = re.search(r'\blanguage=(\{[^}]*\}|[A-Za-z]+)', block)
        language = lang.group(1).strip('{}') if lang else ''
        return ('C', {'caption': tex_inline(cap.group(1)) if cap else '',
                      'code': code.rstrip(), 'language': language})
    if name == 'figure':
        img = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', block)
        cap = re.search(r'\\caption\{([^}]*)\}', block)
        return ('F', {'path': img.group(1) if img else '',
                      'caption': tex_inline(cap.group(1)) if cap else ''})
    if name in ('table', 'center', 'tabularx'):
        m = re.search(r'(?s)\\begin\{tabularx\}.*?\r?\n(.*?)\\end\{tabularx\}', block)
        if not m:
            return ('P', tex_inline(block))
        body = re.sub(r'(?m)^\s*\\[a-zA-Z]+(?:\[[^\]]*\])?\s*$', '', m.group(1))
        body = re.sub(r'\\(?:toprule|midrule|bottomrule|hline|endhead|endfirsthead)\b', '', body)
        rows = []
        for raw in re.split(r'\\\\', body):
            raw = raw.strip()
            if not raw:
                continue
            cells = [tex_inline(x).replace('|', '\\|') for x in raw.split('&')]
            if cells and any(cells):
                rows.append(cells)
        return ('T', rows)
    if name in ('itemize', 'enumerate'):
        items = [tex_inline(m.group(1)) for m in
                 re.finditer(r'\\item\s+(.*?)(?=\n\\item|\n\\end|\Z)', block, re.S)]
        return ('L', items)
    body = re.sub(r'(?s)^.*?\\begin\{' + name + r'\}(?:\[[^\]]*\])?', '', block)
    body = re.sub(r'(?s)\\end\{' + name + r'\}.*$', '', body)
    return ('B', tex_inline(body))


def template_line(tag, payload):
    if tag == 'C':
        return 'C: ' + payload['caption']
    if tag == 'F':
        return 'F: ' + (payload['caption'] or payload['path'])
    if tag == 'T':
        lines = []
        for j, row in enumerate(payload):
            lines.append('| ' + ' | '.join(row) + ' |')
            if j == 0:
                lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')
        return 'T: ' + ' <ROW> '.join(lines)
    if tag == 'L':
        return 'L: ' + ' | '.join(payload)
    return tag + ': ' + str(payload)


def lines_of(text, keep_leading_inline=False):
    return [template_line(t, p) for t, p in split_tokens(text, keep_leading_inline)]


# ---------------------------------------------------------------------------
# Chinese edition follows the three English chapters.
# ---------------------------------------------------------------------------
ZH_CHAPTER_ORIGIN = {'1': '1', '2': '2', '3': '3'}
ZH_CHAPTER_TITLES = CHAPTER_TITLES.copy()


def zh_remap(slot):
    return slot


def zh_origin(slot):
    return slot


# ---------------------------------------------------------------------------
# Document skeleton
# ---------------------------------------------------------------------------

def load_manifest():
    with open(PG / 'build/manifest.tsv', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['status'] == 'kept']
    by_slot = {}
    for r in rows:
        slot = zh_remap(r['slot'])
        if slot:
            by_slot.setdefault(slot, []).append(r['uid'])
    slots = sorted(by_slot, key=lambda s: tuple(int(x) for x in s.split('.')))
    # Read order and visible numbering from the English chapters, not registry IDs.
    layout = english_layout()
    for slot, (_, _, uids) in layout.items():
        if set(uids) != set(by_slot[slot]):
            raise ValueError('English chapter and manifest differ: ' + slot)
        by_slot[slot] = uids
    slots = list(layout)
    return by_slot, slots


def chapter_texts():
    return {n: (PG / 'content' / fn).read_text(encoding='utf-8')
            for n, fn in CHAPTER_FILES.items()}


def english_layout():
    """Map stable source labels to the numbers actually typeset by LaTeX."""
    layout = {}
    for chapter, text in chapter_texts().items():
        section = subsection = 0
        pattern = r'\\(section|subsection)\{([^\n]*)\}\s*\\label\{sec:([^}]+)\}\s*(.*?)(?=\n\\(?:section|subsection)\{|\Z)'
        for m in re.finditer(pattern, text, re.S):
            if m[1] == 'section':
                section += 1
                subsection = 0
                number = f'{chapter}.{section}'
            else:
                subsection += 1
                number = f'{chapter}.{section}.{subsection}'
            slot = m[3].replace('-', '.')
            uids = re.findall(r'\\input\{build/units/([^}]+)\}', m[4])
            layout[slot] = (number, tex_inline(m[2]), uids)
    return layout


def display_references(text, layout):
    # Only prose references outside inline code use source-label numbering.
    def replace(part):
        return re.sub(r'(?<![\w.])([123]\.\d+(?:\.\d+)?)(?![\w.])',
                      lambda m: layout[m[1]][0] if m[1] in layout else m[0], part)
    return ''.join(part if i % 2 else replace(part)
                   for i, part in enumerate(re.split(r'(`[^`]*`)', text)))


def structure_keys():
    """Ordered (key, template lines) for everything outside the units."""
    out = []
    front = (PG / 'content/frontmatter.tex').read_text(encoding='utf-8')
    front = re.sub(r'(?s)\\begin\{titlepage\}.*?\\end\{titlepage\}', '', front)
    # The Chinese edition presents content only: drop the section that explains
    # the book's own typography (fonts, callout boxes, listing conventions).
    front_lines = []
    skip = False
    for line in lines_of(front):
        if line.startswith('H: '):
            skip = line == 'H: Conventions'
        if not skip:
            front_lines.append(line)
    out.append(('FRONT', front_lines))

    texts = chapter_texts()
    for n, origin in ZH_CHAPTER_ORIGIN.items():
        text = texts[origin]
        ct = re.search(r'\\chapter\{([^}]*)\}', text)
        ci = re.search(r'\\guidechapterintro\{(.*?)\}\n', text, re.S)
        cn = re.search(r'\\guidechapternext\{(.*?)\}\n', text, re.S)
        lines = []
        if ct:
            lines.append('H: ' + tex_inline(ct.group(1)))
        if ci:
            lines.append('I: ' + tex_inline(ci.group(1)))
        if cn:
            lines.append('N: ' + tex_inline(cn.group(1)))
        out.append(('CH' + n, lines))
        out.append(('CHTITLE' + n, ['H: ' + ZH_CHAPTER_TITLES[n]]))

    _, slots = load_manifest()
    for slot in slots:
        origin = zh_origin(slot)
        lines = []
        if origin:
            text = texts[origin[0]]
            m = re.search(r'\\(?:section|subsection)\{([^}]*)\}\s*\\label\{sec:' +
                          origin.replace('.', '-') + r'\}\s*(.*?)(?=\n\\(?:section|subsection)\{|\Z)',
                          text, re.S)
            if m:
                lines.append('H: ' + tex_inline(m.group(1)))
        out.append(('SEC:' + slot, lines))

    out.append(('APP', lines_of((PG / 'content/appendix-a-quickref.tex').read_text(encoding='utf-8'))))
    out.append(('BACK', lines_of((PG / 'content/backmatter.tex').read_text(encoding='utf-8'))))
    out.append(('GLOSS', [
        'H: Glossary',
        'P: English term and the Chinese rendering fixed for this edition.',
        'T: | English | 中文 | <ROW> | --- | --- | <ROW> | kernel | 核函数 |',
    ]))
    return out


def unit_keys(by_slot, slots, chapter):
    for slot in slots:
        if slot.split('.')[0] != chapter:
            continue
        for uid in by_slot[slot]:
            text = (PG / 'build/units' / (uid + '.tex')).read_text(encoding='utf-8')
            yield 'UID:' + uid, lines_of(text, keep_leading_inline=chapter == '1')


def skeleton_order():
    by_slot, slots = load_manifest()
    order = ['FRONT']
    for n in ZH_CHAPTER_ORIGIN:
        order.append('CHTITLE' + n)
        order.append('CH' + n)
        for slot in [s for s in slots if s.split('.')[0] == n]:
            order.append('SEC:' + slot)
            order += ['UID:' + u for u in by_slot[slot]]
    return order + ['APP', 'BACK', 'GLOSS']


def dump(path, keys):
    with open(path, 'w', encoding='utf-8') as f:
        for key, lines in keys:
            f.write('=== ' + key + '\n')
            for line in lines:
                f.write(line + '\n')
            f.write('\n')


def write_templates():
    TPL.mkdir(parents=True, exist_ok=True)
    ZH.mkdir(parents=True, exist_ok=True)
    by_slot, slots = load_manifest()
    dump(TPL / 'structure.md', structure_keys())
    for n in ZH_CHAPTER_ORIGIN:
        dump(TPL / ('units-ch' + n + '.md'), unit_keys(by_slot, slots, n))
    print('templates written to', TPL)
    for p in sorted(TPL.glob('*.md')):
        ls = p.read_text(encoding='utf-8').splitlines()
        print('  %-16s %5d lines, %4d keys' %
              (p.name, len(ls), sum(1 for x in ls if x.startswith('=== '))))


def load_blocks(directory):
    data = {}
    for p in sorted(directory.glob('*.md')):
        key, lines = None, []
        for raw in p.read_text(encoding='utf-8').splitlines():
            if raw.startswith('=== '):
                if key:
                    data[key] = lines
                key, lines = raw[4:].strip(), []
            elif raw.strip():
                lines.append(raw.strip())
        if key:
            data[key] = lines
    return data


def normalize():
    """Keep 中文(English) only for core terms, once per book."""
    files = {p: p.read_text(encoding='utf-8').splitlines()
             for p in sorted(ZH.glob('*.md'))}
    index = {}
    for path, lines in files.items():
        key = None
        for i, raw in enumerate(lines):
            if raw.startswith('=== '):
                key = raw[4:].strip()
                index[key] = (path, [])
            elif raw.strip() and key:
                index[key][1].append(i)
    seen, changed = set(), 0
    for key in skeleton_order():
        if key not in index:
            continue
        path, idxs = index[key]
        for i in idxs:
            line = files[path][i]

            def repl(m):
                en = m.group(2).strip().lower()
                if en not in CORE_TERMS:
                    return m.group(1)
                if en in seen:
                    return m.group(1)
                seen.add(en)
                return m.group(0)

            new = BILINGUAL.sub(repl, line)
            if new != line:
                files[path][i] = new
                changed += 1
    for path, lines in files.items():
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('normalized lines:', changed, 'core terms kept:', len(seen))
    return 0


def check():
    tpl, zh = load_blocks(TPL), load_blocks(ZH)
    missing = [k for k in skeleton_order() if k not in zh and tpl.get(k)]
    mismatch = []
    for k, lines in tpl.items():
        if k not in zh:
            continue
        a = [x.split(':', 1)[0] for x in lines]
        b = [x.split(':', 1)[0] for x in zh[k]]
        if a != b:
            mismatch.append((k, a, b))
    bad = []
    for k, lines in zh.items():
        for line in lines:
            for w in FORBIDDEN:
                if w in line:
                    bad.append((k, w, line[:70]))
            if FULLWIDTH_BILINGUAL.search(line):
                bad.append((k, '全角术语括号（应为半角）', line[:70]))
    print('template keys :', len(tpl))
    print('translated    :', len(zh))
    print('missing keys  :', len(missing), missing[:12])
    print('tag mismatches:', len(mismatch))
    for k, a, b in mismatch[:8]:
        print('   ', k, 'template', a, 'zh', b)
    print('forbidden hits:', len(bad))
    for x in bad[:8]:
        print('   ', x)
    return 1 if (missing or mismatch or bad) else 0


# ---------------------------------------------------------------------------
# Build Markdown / HTML / PDF
# ---------------------------------------------------------------------------

# Code listings: read the keyword table straight from the English edition's
# listings setup so the two editions cannot drift apart.
def code_keywords():
    sty = (PG / 'croqguide.sty').read_text(encoding='utf-8')
    m = re.search(r'\\lstdefinelanguage\{croqtile\}\{%(.*?)\n\}', sty, re.S)
    blob = re.search(r'morekeywords=\{(.*?)\}', m.group(1), re.S).group(1)
    return set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', blob.replace('%\n', '')))


CODE_TOKEN = re.compile(
    r'(?P<comment>//[^\n]*|/\*.*?\*/)|'
    r'(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
    r'(?P<word>[A-Za-z_][A-Za-z0-9_]*)',
    re.S)


def render_code_html(raw, keywords, keyword_set):
    raw = raw.rstrip('\r\n')  # fenced_code appends a trailing newline
    out = ['<pre class="code-hl"><code>']
    line = []

    def flush():
        out.append('<span class="ln">' + ''.join(line) + '</span>')
        line.clear()

    def emit(text, cls=None):
        for i, part in enumerate(text.split('\n')):
            if i:
                flush()
            if part:
                esc = html_mod.escape(part, quote=False)
                line.append('<span class="%s">%s</span>' % (cls, esc) if cls else esc)

    if not raw.strip():
        out.append('</code></pre>')
        return ''.join(out)
    pos = 0
    for m in CODE_TOKEN.finditer(raw):
        emit(raw[pos:m.start()])
        kind, val = m.lastgroup, m.group()
        if kind == 'word':
            emit(val, 'k' if keywords and val in keyword_set else None)
        elif kind == 'comment':
            emit(val, 'c')
        else:
            emit(val, 's')
        pos = m.end()
    emit(raw[pos:])
    flush()
    out.append('</code></pre>')
    return ''.join(out)


def highlight_listings(body):
    kws = code_keywords()
    return re.sub(
        r'<pre><code class="language-(co|text)">(.*?)</code></pre>',
        lambda m: render_code_html(html_mod.unescape(m.group(2)),
                                   m.group(1) == 'co', kws),
        body, flags=re.S)


CSS = r'''
@page { size: A4; margin: 2.3cm 2.1cm 2.0cm 2.1cm;
  @top-right { content: "Croqtile 编程指南"; font-family: "Noto Sans CJK SC";
               font-size: 8pt; color: #6b7280; }
  @bottom-center { content: counter(page); font-family: "Noto Serif CJK SC"; font-size: 9pt; } }
@page :first { @top-right { content: ""; } @bottom-center { content: ""; } }
body { font-family: "Noto Serif CJK SC", serif; font-size: 10.5pt;
       line-height: 1.72; color: #1f2933; text-align: left; }
h1 { font-family: "Noto Sans CJK SC", sans-serif; font-size: 20pt; color: #137a5b;
     border-bottom: 2px solid #137a5b; padding-bottom: 6px; margin-top: 26px;
     break-before: page; }
h2 { font-family: "Noto Sans CJK SC", sans-serif; font-size: 14.5pt; color: #1f2933;
     margin-top: 20px; border-left: 4px solid #137a5b; padding-left: 8px; }
h3 { font-family: "Noto Sans CJK SC", sans-serif; font-size: 12pt; color: #137a5b;
     margin-top: 15px; }
h4 { font-family: "Noto Sans CJK SC", sans-serif; font-size: 11pt; color: #1f2933;
     margin-top: 13px; }
p { margin: 0 0 0.55em 0; orphans: 2; widows: 2; }
.cap { font-family: "Noto Sans CJK SC", sans-serif; font-size: 9pt; color: #43515c;
       margin: 12px 0 4px 0; }
pre.code-hl { counter-reset: codeline; background: #f6f8fa;
            border: 1px solid #d9dee3; border-radius: 3px; padding: 8px 10px;
            margin: 0 0 4px 0; font-family: "Noto Sans Mono CJK SC", monospace;
            font-size: 8.4pt; line-height: 1.42; white-space: pre-wrap;
            overflow-wrap: anywhere; color: #1f2933;
            box-decoration-break: clone; }
pre.code-hl code { background: none; padding: 0; font-family: inherit;
                   font-size: inherit; color: inherit; }
pre.code-hl .ln { display: block; position: relative; padding-left: 2.9em; }
pre.code-hl .ln::before { counter-increment: codeline; content: counter(codeline);
                          position: absolute; left: 0; width: 2.2em; text-align: right;
                          color: #b3bcc4; font-size: 7.3pt; }
pre.code-hl .k { color: #137a5b; font-weight: 700; }
pre.code-hl .c { color: #5b6770; font-style: italic; }
pre.code-hl .s { color: #a96400; }
.listing-cap { font-family: "Noto Sans CJK SC", sans-serif; font-size: 9pt;
               color: #43515c; text-align: center; margin: 4px 0 13px 0; }
code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 0.9em;
       background: #f1f3f5; padding: 0 2px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0 12px 0; }
th { border-top: 1.5px solid #1f2933; border-bottom: 1px solid #1f2933;
     font-family: "Noto Sans CJK SC"; text-align: left; padding: 5px 6px; }
td { border-bottom: 1px solid #d9dee3; padding: 5px 6px; vertical-align: top; }
img { width: 100%; max-width: 100%; display: block; margin: 10px auto; }
.guide-figure { break-inside: avoid; }
.guide-figure .cap { text-align: center; margin-top: 5px; }
.guide-figure.wide img { width: 100%; max-width: 100%; }
ul, ol { margin: 4px 0 10px 0; padding-left: 1.5em; }
blockquote { margin: 10px 0; padding: 8px 12px; background: #eef4f8;
             border-left: 3px solid #4d7ea8; color: #34495e; }
.titlepage { text-align: center; break-after: page; padding-top: 4.5cm; }
.titlepage .big { font-family: "Noto Sans CJK SC"; font-size: 30pt; color: #1f2933; }
.titlepage .sub { font-family: "Noto Sans CJK SC"; font-size: 20pt; color: #137a5b;
                  margin-top: 8px; }
.toc-heading { font-family: "Noto Sans CJK SC", sans-serif; font-size: 19pt;
               color: #137a5b; letter-spacing: .35em; text-align: center;
               margin: 4px 0 16px 0; padding-bottom: 12px;
               border-bottom: 1.5px solid #137a5b; }
.toc { margin: 0; }
.toc a { display: block; text-decoration: none;
         color: #1f2933; line-height: 1.5; break-inside: avoid; }
.toc-l2::after, .toc-l3::after {
  content: leader(' · ') target-counter(attr(href), page);
  font-family: "Noto Serif CJK SC", serif; font-size: 9pt; color: #9aa6ae; }
.toc code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: .88em;
            background: none; padding: 0; color: #43515c; }
.toc .tn { display: inline-block; font-family: "Noto Sans CJK SC", sans-serif;
           color: #8a97a0; }
.toc-l1 { position: relative; padding-right: 2.8em;
          font-family: "Noto Sans CJK SC", sans-serif; font-size: 12.5pt;
          font-weight: 700; color: #137a5b; margin: 12px 0 5px 0;
          padding-top: 3px; padding-bottom: 5px; padding-left: 9px;
          border-left: 3px solid #137a5b; border-bottom: .6px solid #d6e6df;
          break-after: avoid; }
.toc-l1 .tn { width: 4.4em; color: #137a5b; }
.toc-l1::after { content: target-counter(attr(href), page);
                 position: absolute; right: 0; top: 0;
                 font-family: "Noto Serif CJK SC", serif; font-size: 9.5pt;
                 color: #137a5b; font-weight: 700; }
.toc-l2 { font-size: 10pt; padding-left: 1.4em; margin: 1px 0; }
.toc-l2 .tn { width: 3.4em; }
.toc-l3 { font-size: 9.2pt; color: #5a6872; padding-left: 3.2em; margin: 1px 0; }
.toc-l3 .tn { width: 4.2em; color: #9aa6ae; }
.toc-l3::after { color: #9aa6ae; }
.page-break { break-before: page; }
'''


def render_simple(lines, md, heading_level):
    for line in lines:
        tag, _, text = line.partition(':')
        text = text.strip()
        if tag == 'H':
            md.append(heading_level + ' ' + text)
        elif tag == 'I':
            md.append(text)
        elif tag == 'N':
            md.append(text)
        elif tag == 'T':
            md += text.split(' <ROW> ')
        elif tag == 'L':
            md += ['- ' + x.strip() for x in text.split(' | ') if x.strip()]
        elif tag == 'B':
            md.append('> ' + text)
        elif tag == 'C':
            md.append('<div class="cap">' + text + '</div>')
        else:
            md.append(text)
        md.append('')


def build():
    normalize()
    if check():
        print('check failed; aborting build')
        return 1
    zh = load_blocks(ZH)
    editorial = json.loads(CH1_EDITORIAL.read_text(encoding='utf-8'))
    order = skeleton_order()
    layout = english_layout()
    first_chapter_units = {uid for slot, (_, _, uids) in layout.items()
                           if slot.startswith('1.') for uid in uids}
    zh = {key: [display_references(line, layout) for line in lines]
          for key, lines in zh.items()}
    listing_no = figure_no = table_no = 0
    md = []

    def emit_figure(path, caption, compact=False):
        nonlocal figure_no
        if not (PG / 'figures' / path).is_file():
            raise FileNotFoundError(path)
        figure_no += 1
        cls = 'guide-figure syntax' if compact else 'guide-figure'
        if path == 'f03-compile_and_run_zh.png':
            cls += ' wide'
        md.extend(['<div class="%s">' % cls,
                   '<img src="figures/%s" alt="%s">' %
                   (html_mod.escape(path, quote=True), html_mod.escape(caption, quote=True)),
                   '<div class="cap">图 %d：%s</div>' % (figure_no, html_mod.escape(caption)),
                   '</div>', ''])

    def emit_editorial(items):
        nonlocal listing_no
        for item in items:
            kind = item['kind']
            if kind == 'prose':
                md.extend([display_references(item['text'], layout), ''])
            elif kind == 'figure':
                emit_figure(item['path'], item['caption'], item.get('compact', True))
            elif kind == 'code':
                source = PG / 'build/units' / (item['source_uid'] + '.tex')
                code = next(p['code'] for t, p in split_tokens(source.read_text()) if t == 'C')
                if item.get('kernel_only'):
                    code = code.split('\nint main()', 1)[0].rstrip()
                listing_no += 1
                md.extend(['```co', code, '```', '',
                           '<div class="listing-cap">清单 %d：%s</div>' %
                           (listing_no, item['caption']), ''])
            else:
                raise ValueError('Unknown chapter 1 editorial item: ' + kind)

    md += ['<div class="titlepage">',
           '<div class="big">Croqtile</div>',
           '<div class="sub">编程指南</div>',
           '</div>', '']

    # Table of contents: chapter / section / subsection rows with page numbers.
    # English annotations and inline-code marks belong to the body only.
    def toc_text(s):
        s = re.sub(r'([\u4e00-\u9fff]+)\([^()]*\)', r'\1', s)
        s = (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;'))
        return re.sub(r'`([^`]+)`', r'<code>\1</code>', s)

    def toc_row(cls, anchor, num, title):
        return ('<a class="%s" href="#%s"><span class="tn">%s</span>'
                '<span class="tt">%s</span></a>' %
                (cls, anchor, toc_text(num), toc_text(title)))

    toc = []
    for key in order:
        if key.startswith('CHTITLE'):
            n = key[-1]
            toc.append(toc_row('toc-l1', 'ch' + n, '第 %s 章' % n,
                               zh[key][0][3:].strip()))
        elif key.startswith('SEC:'):
            slot = key[4:]
            head = next((x[3:].strip() for x in zh[key] if x.startswith('H:')), '')
            cls = 'toc-l2' if len(slot.split('.')) == 2 else 'toc-l3'
            toc.append(toc_row(cls, 'sec-' + slot.replace('.', '-'), layout[slot][0], head))
    md += ['<div class="toc-heading">目录</div>', '<div class="toc">']
    md += toc
    md += ['</div>', '<div class="page-break"></div>', '']

    pending_next = None
    for key in order:
        if pending_next and (key.startswith('CHTITLE') or key == 'APP'):
            md += [pending_next, '']
            pending_next = None
        lines = zh.get(key, [])
        if key.startswith('CHTITLE'):
            n = key[-1]
            md.append('# 第 %s 章 %s <a id="ch%s"></a>' %
                       (n, lines[0][3:].strip() if lines else '', n))
            md.append('')
            continue
        if key.startswith('CH') and not key.startswith('CHTITLE'):
            for line in lines:
                tag, _, text = line.partition(':')
                text = text.strip()
                if tag == 'I':
                    md += [text, '']
                elif tag == 'N':
                    pending_next = text
            continue
        if key.startswith('SEC:'):
            slot = key[4:]
            level = '##' if len(slot.split('.')) == 2 else '###'
            for line in lines:
                tag, _, text = line.partition(':')
                text = text.strip()
                if tag == 'H':
                    md.append('%s %s %s <a id="sec-%s"></a>' %
                                  (level, layout[slot][0], text, slot.replace('.', '-')))
                elif tag == 'I':
                    md.append(text)
                else:
                    md.append(text)
                md.append('')
            emit_editorial(editorial['sections'].get(slot, []))
            continue
        if key.startswith('UID:'):
            uid = key[4:]
            text = (PG / 'build/units' / (uid + '.tex')).read_text(encoding='utf-8')
            tokens = split_tokens(text, keep_leading_inline=uid in first_chapter_units)
            codes = [p for t, p in tokens if t == 'C']
            figs = [p for t, p in tokens if t == 'F']
            tables = [p for t, p in tokens if t == 'T']
            ci = fi = ti = 0
            hooks = editorial['units'].get(uid, {})
            emit_editorial(hooks.get('before', []))
            occurrences = {}
            for line in lines:
                tag, _, body = line.partition(':')
                body = body.strip()
                if tag == 'H':
                    md += ['#### ' + body, '']
                elif tag == 'P':
                    md += [body, '']
                elif tag == 'I':
                    md += [body, '']
                elif tag == 'N':
                    md += [body, '']
                elif tag == 'B':
                    md += ['> ' + body, '']
                elif tag == 'L':
                    md += ['- ' + x.strip() for x in body.split(' | ') if x.strip()]
                    md.append('')
                elif tag == 'T':
                    table_no += 1
                    md += body.split(' <ROW> ') + ['']
                elif tag == 'C':
                    listing_no += 1
                    payload = codes[ci] if ci < len(codes) else {'code': ''}
                    ci += 1
                    lang = payload.get('language', '')
                    fence = 'co' if lang == 'croqtile' else 'text'
                    md += ['```' + fence, payload.get('code', ''), '```', '',
                           '<div class="listing-cap">清单 %d：%s</div>' %
                           (listing_no, body), '']
                elif tag == 'F':
                    payload = figs[fi] if fi < len(figs) else {'path': ''}
                    fi += 1
                    replacement = editorial['figure_replacements'].get(uid + ':' + str(fi))
                    if replacement:
                        emit_figure(replacement['path'], body, replacement.get('compact', False))
                    else:
                        emit_figure(payload['path'], body)
                occurrences[tag] = occurrences.get(tag, 0) + 1
                emit_editorial(hooks.get('after_' + tag + str(occurrences[tag]), []))
            emit_editorial(hooks.get('after', []))
            continue
        render_simple(lines, md, '##')

    text = '\n'.join(md) + '\n'
    OUT_MD.write_text(text, encoding='utf-8')

    import markdown
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    body = highlight_listings(body)
    doc = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
           '<title>Croqtile 编程指南</title><style>' + CSS + '</style>'
           '</head><body>' + body + '</body></html>')
    OUT_HTML.write_text(doc, encoding='utf-8')

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    from weasyprint import HTML
    HTML(string=doc, base_url=str(PG)).write_pdf(str(OUT_PDF))
    print('written:', OUT_MD.name, OUT_HTML.name, OUT_PDF)
    print('listings:', listing_no, 'figures:', figure_no, 'tables:', table_no)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['templates', 'normalize', 'check', 'build'])
    a = ap.parse_args()
    if a.action == 'templates':
        write_templates()
        return 0
    if a.action == 'normalize':
        return normalize()
    if a.action == 'check':
        return check()
    return build()


if __name__ == '__main__':
    sys.exit(main())
