#!/usr/bin/env python3
"""Build the Chinese edition as LaTeX, reusing the English unit fragments.

The English ``build/units/*.tex`` fragments already carry the listings,
figures and tables.  Only the prose, headings and captions are replaced, from
``translation/zh/*.md``.  The result is a tree of LaTeX fragments that the
normal pdfLaTeX/XeLaTeX pipeline can typeset, so the Chinese edition gets the
same layout and the same figure/listing numbering as the English one.

Usage
-----
    python3 tools/build_guide_zh_tex.py unit F03-S03   # write one fragment
    python3 tools/build_guide_zh_tex.py all            # write every fragment
    python3 tools/build_guide_zh_tex.py chapters       # content-zh/ch0X-*.tex
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

PG = Path(__file__).resolve().parents[1]
TOOLS = PG / 'tools'


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        'build_guide_zh', TOOLS / 'build_guide_zh.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_builder()

FIGURE_REPLACEMENTS = {}
_ch1 = PG / 'translation/zh/chapter1-editorial.json'
if _ch1.is_file():
    import json as _json
    FIGURE_REPLACEMENTS = _json.loads(
        _ch1.read_text(encoding='utf-8')).get('figure_replacements', {})
OUT = PG / 'build-zh' / 'units'
CONTENT_OUT = PG / 'content-zh'


# ---------------------------------------------------------------------------
# Text conversion
# ---------------------------------------------------------------------------

ESC = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
       ('#', r'\#'), ('_', r'\_'), ('~', r'\textasciitilde{}'),
       ('^', r'\textasciicircum{}'), ('<', r'\textless{}'),
       ('>', r'\textgreater{}')]


def esc(s):
    for a, b in ESC:
        s = s.replace(a, b)
    return s


def texify(s):
    """Markdown marks in the translation -> LaTeX."""
    out = []
    for part in re.split(r'(`[^`]*`)', s):
        if len(part) > 1 and part.startswith('`') and part.endswith('`'):
            out.append(r'\co@' + part[1:-1] + r'@')
            continue
        p = esc(part)
        p = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\\emph{\1}', p)
        out.append(p)
    return ''.join(out)


def texify_heading(s):
    """Headings are moving arguments: no verbatim, escape the specials."""
    out = []
    for part in re.split(r'(`[^`]*`)', s):
        if len(part) > 1 and part.startswith('`') and part.endswith('`'):
            out.append(r'\texttt{' + esc(part[1:-1]) + '}')
            continue
        p = esc(part)
        p = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\\emph{\1}', p)
        out.append(p)
    return ''.join(out)


# ---------------------------------------------------------------------------
# Fragment emission
# ---------------------------------------------------------------------------

def emit_table(ztext):
    rows = [[c.strip() for c in r.split(' | ')] for r in ztext.split(' <ROW> ')]
    rows = [r for r in rows if r and not all(re.fullmatch(r'-{3,}', c) for c in r)]
    ncol = max(len(r) for r in rows)
    spec = '@{}l' + 'X' * (ncol - 1) + '@{}'
    head, body = rows[0], rows[1:]
    lines = ['\\begin{table}[htbp]\\centering\\small',
             '\\begin{tabularx}{\\linewidth}{' + spec + '}', '\\toprule',
             ' & '.join(texify(c) for c in head) + r' \\', '\\midrule']
    for r in body:
        r = r + [''] * (ncol - len(r))
        lines.append(' & '.join(texify(c) for c in r) + r' \\')
    lines += ['\\bottomrule', '\\end{tabularx}', '\\end{table}']
    return '\n'.join(lines)


def figure_path(uid, index, path):
    key = '%s:%d' % (uid, index)
    rep = FIGURE_REPLACEMENTS.get(key)
    return rep['path'] if rep else path


def emit_figure(payload, caption, path=None):
    path = path or payload['path']
    return ('\\begin{figure}[htbp]\\centering\n'
            '\\includegraphics[width=\\linewidth,height=0.56\\textheight,'
            'keepaspectratio]{' + path + '}\n'
            '\\caption{' + texify(caption) + '}\n'
            '\\end{figure}')


def emit_code(payload, caption):
    lang = payload.get('language') or 'croqtile'
    return ('\\begin{lstlisting}[language=' + lang + ',caption={' +
            texify(caption) + '}]\n' + payload['code'] +
            '\n\\end{lstlisting}')


def emit_unit(uid, tokens, zlines):
    if len(tokens) != len(zlines):
        raise ValueError('%s: %d tokens, %d translation lines' %
                         (uid, len(tokens), len(zlines)))
    out = []
    fig_no = 0
    for (tag, payload), zline in zip(tokens, zlines):
        ztag, _, ztext = zline.partition(':')
        ztext = ztext.strip()
        if ztag != tag:
            raise ValueError('%s: tag mismatch %s vs %s' % (uid, ztag, tag))
        if tag == 'H':
            out.append('\\subsection*{' + texify_heading(ztext) + '}')
        elif tag in ('P', 'I', 'N'):
            out.append(texify(ztext))
        elif tag == 'B':
            out.append('\\begin{keypoint}\n' + texify(ztext) + '\n\\end{keypoint}')
        elif tag == 'C':
            out.append(emit_code(payload, ztext))
        elif tag == 'F':
            fig_no += 1
            out.append(emit_figure(payload, ztext, figure_path(uid, fig_no, payload['path'])))
        elif tag == 'T':
            out.append(emit_table(ztext))
        elif tag == 'L':
            items = [x.strip() for x in ztext.split(' | ') if x.strip()]
            out.append('\\begin{itemize}[leftmargin=1.6em]\n' +
                       '\n'.join('\\item ' + texify(x) for x in items) +
                       '\n\\end{itemize}')
        else:
            raise ValueError('%s: unknown tag %r' % (uid, tag))
    return '\n\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_unit(uid, zh):
    src = PG / 'build/units' / (uid + '.tex')
    tokens = B.split_tokens(src.read_text(encoding='utf-8'))
    key = 'UID:' + uid
    if key not in zh:
        raise KeyError('missing translation for ' + uid)
    text = emit_unit(uid, tokens, zh[key])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (uid + '.tex')).write_text(text, encoding='utf-8')
    return len(tokens)


def all_units():
    zh = B.load_blocks(B.ZH)
    uids = sorted(p.stem for p in (PG / 'build/units').glob('*.tex'))
    ok = fail = 0
    for uid in uids:
        try:
            write_unit(uid, zh)
            ok += 1
        except Exception as e:  # noqa: BLE001 - report and continue
            print('FAIL %-10s %s' % (uid, e))
            fail += 1
    print('units: %d ok, %d failed' % (ok, fail))
    return 1 if fail else 0


def zh_structure():
    """Parse translation/zh/structure.md into {key: [lines]}."""
    return B.load_blocks(B.ZH)


def block_text(lines, tag):
    for line in lines:
        t, _, text = line.partition(':')
        if t == tag:
            return text.strip()
    return None


def chapters():
    zh = zh_structure()
    CONTENT_OUT.mkdir(parents=True, exist_ok=True)
    files = {'1': 'ch01-getting-started', '2': 'ch02-language-basics',
             '3': 'ch03-programming-guide'}
    for num, stem in files.items():
        src = (PG / 'content' / (stem + '.tex')).read_text(encoding='utf-8')
        title = block_text(zh.get('CHTITLE' + num, []), 'H')
        intro = block_text(zh.get('CH' + num, []), 'I')
        out = []
        sec_index = 0
        for line in src.split('\n'):
            if line.startswith('\\chapter{') and title:
                out.append('\\chapter{' + title + '}')
                out.append('\\guidechaptermark{' + title + '}')
                continue
            if line.startswith('\\guidechapterintro{') and intro:
                out.append('\\guidechapterintro{' + texify_heading(intro) + '}')
                continue
            if line.startswith('\\section{'):
                sec_index += 1
                key = 'SEC:%s.%d' % (num, sec_index)
                t = block_text(zh.get(key, []), 'H')
                if t is None:
                    print('  missing', key, '-> keeping English title')
                else:
                    out.append('\\section{' + texify_heading(t) + '}')
                    continue
            out.append(line.replace('\\input{build/units/', '\\input{build-zh/units/'))
        (CONTENT_OUT / (stem + '.tex')).write_text('\n'.join(out), encoding='utf-8')
        print('wrote content-zh/' + stem + '.tex')


def frontmatter():
    zh = zh_structure()
    lines = zh.get('FRONT', [])
    body = []
    for line in lines:
        tag, _, text = line.partition(':')
        text = text.strip()
        if tag == 'H':
            body.append('\\chapter*{' + texify_heading(text) + '}')
            body.append('\\addcontentsline{toc}{chapter}{' + texify_heading(text) + '}')
        elif tag in ('P', 'I', 'N'):
            body.append(texify(text))
        else:
            raise ValueError('frontmatter tag ' + tag)
    title = ['\\frontmatter', '\\begin{titlepage}', '\\centering',
             '\\vspace*{1.4cm}',
             '{\\sffamily\\bfseries\\fontsize{36}{40}\\selectfont\\color{croqink} Croqtile\\par}',
             '\\vspace{0.1cm}',
             '{\\sffamily\\bfseries\\fontsize{26}{30}\\selectfont\\color{croqgreen} 编程指南\\par}',
             '\\vspace{0.8cm}',
             '{\\Large\\itshape 面向高性能核函数的 Croqtile 语言导读\\par}',
             '\\vspace{1.1cm}', '\\rule{0.72\\linewidth}{0.6pt}\\par', '\\vspace{0.9cm}',
             '{\\large 版本 \\guideversion{} --- \\guidereleasedate\\par}',
             '\\vfill', '{\\small\\color{croqgray} LancerLab\\par}', '\\vspace{1.0cm}',
             '\\end{titlepage}', '\\thispagestyle{empty}', '\\cleardoublepage', '']
    CONTENT_OUT.mkdir(parents=True, exist_ok=True)
    (CONTENT_OUT / 'frontmatter.tex').write_text(
        '\n'.join(title + ['\n\n'.join(body)]) + '\n', encoding='utf-8')
    print('wrote content-zh/frontmatter.tex')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['unit', 'all', 'chapters'])
    ap.add_argument('uid', nargs='?')
    args = ap.parse_args()
    if args.mode == 'unit':
        zh = B.load_blocks(B.ZH)
        n = write_unit(args.uid, zh)
        print('%s: %d blocks written' % (args.uid, n))
        return 0
    if args.mode == 'all':
        return all_units()
    chapters()
    frontmatter()
    return 0


if __name__ == '__main__':
    sys.exit(main())
