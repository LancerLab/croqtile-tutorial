#!/usr/bin/env python3
"""Review loop driver for the Croqtile Programming Guide.

    python3 tools/build_guide.py
    python3 tools/review.py --round N      # review the current build
    ... fix issues ...
    python3 tools/build_guide.py
    python3 tools/review.py --round N+1 --compare

The point of the loop is that every round produces a comparable score.  A
round is only "done" when the total is lower than the previous one and no
new category regresses.  Two consecutive zero rounds mean convergence.
Only files that actually reach the PDF are scanned;
engineering documents (README, UNIT-REGISTRY, HARDWARE-NOTES) are excluded.
"""

import argparse
import collections
import glob
import importlib.util
import json
import os
import re
import sys

PG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PG)

spec = importlib.util.spec_from_file_location('bg', os.path.join(PG, 'tools', 'build_guide.py'))
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

FILES = (sorted(glob.glob('build/units/*.tex'))
         + ['content/frontmatter.tex', 'content/backmatter.tex',
            'content/appendix-a-quickref.tex', 'content/ch01-getting-started.tex',
            'content/ch02-language-basics.tex', 'content/ch03-programming-guide.tex'])
if os.path.isfile('content/appendix-b-sources.tex'):
    FILES.append('content/appendix-b-sources.tex')

CHECKS = {
    # 1. Editorial metadata: nothing about how the book was assembled belongs
    #    in a language guide aimed at a Croqtile programmer.
    'meta': [
        r'upstream', r'reorganis', r'\bdecomposed\b', r'regroup', r'content unit',
        r'Source Material Map', r'source map', r'companion document',
        r'pdfLaTeX', r'BibTeX', r'Typeset with', r'docs_dir', r'mkdocs',
        r'\.src\.md', r'empty stubs?', r'unfinished drafts?', r'// TODO',
        r'Animated version: available', r'registered but', r'stub',
        r'this guide is a reorgan', r'Appendix~\\ref\{app:sourcemap\}',
    ],
    # 2. Point of view: write about Croqtile, not about "we" or the old
    #    tutorial's chapter numbering.
    'pov': [
        r'\bWe\b', r'\bwe\b', r'\bOur\b', r'\bour\b',
        r'\bChapter\s+\d+\b', r'this chapter', r'In this section, we',
        r'In this chapter, we', r'you have learned the following',
    ],
    # 3. Dozens of identical "Overview"/"Quick Summary" headings are not a
    #    narrative; they are artefacts of concatenating source pages.
    'generic_headings': [
        r'\\subsection\*\{Overview\}', r'\\subsection\*\{Quick Summary\}',
        r'\\subsection\*\{Summary\}', r'\\subsection\*\{Chapter summary\}',
        r'\\subsection\*\{New syntax\}', r'\\subsection\*\{What You Will Find Here\}',
    ],
    # 4. Spelling / grammar defects carried over from the source pages.
    'typos': [
        r'seperat', r'Prerequisitions', r'The below code demonstrate',
        r'it described', r'CroqTile', r'\bI\.e\.', r'Direct storage Access',
        r'\boccured\b', r'\brecieve\b', r'\bparamter\b', r'\bcompatable\b',
        r'\blenght\b', r'\bwidht\b', r'\bexplict\b', r'\balot\b',
        r'\bthe the\b', r'\ba a\b', r'\bwe defines\b',
    ],
    # 5. Sentence fragments left behind when a clause is removed.
    'broken': [
        r'\.\s+(?:but|and|so|which|because|however|also|then|while)\b',
        r'\.\s+[a-z]{1,4}\s+[a-z]', r'\s+,\s+(?:and|but|which|so|while)\b',
        r'\bis\s+,\s', r'\.\s+\.', r'\bcompile time no runtime overhead\b',
        r'\bthe the\b', r'\ba a\b', r'\s+,\s*,\s+',
    ],
    # Tutorial navigation links that lost their target after regrouping.
    'dangling_refs': [
        r'\\emph\{next chapter\}', r'\\emph\{previous chapter\}',
        r'\\emph\{[^}]*reference\}', r'\\emph\{Coding Reference\}',
        r'\\emph\{Asynchronous parallel-by\}',
    ],
    # Source headings that merely repeat the section title.
    'dup_headings': [],
    # Identical long sentences across units: a guide should say a thing once.
    'repeated_sentences': [],
    # Round 8: structure and arithmetic left over from the source pages.
    'stale_titles': [r'\\subsection\*\{[^}]*Ch(?:apter)?\s*\d+\s*[:.]'],
    'inline_number_emph': [r'\\emph\{\s*\d+\s*\}'],
    'heading_only': [],     # a heading with no body at all
    'merged_paragraphs': [],  # adjacent \\inputs with no blank line between
    'preface_positioning': [],  # the reader-facing language positioning
    # 6. Target-platform vocabulary, stray conversion placeholders.
    'hardware': None,      # uses build_guide.HW
    'placeholders': [r'QZ\d{3}QZ'],
    # 7. Round 4: prose quality.  These are not vocabulary problems; they are
    #    the debris left when a clause was archived, plus tutorial-era voice
    #    that survived the earlier point-of-view pass.
    'empty_markup': [
        r'\\(?:emph|textbf|texttt|textit)\{\s*\}',
        r'\\href\{[^}]*\}\{\s*\}',
        r'\\item\s*$',
        r'^\s*[A-Z]\s*\.\s*$',          # "A ." left after a caption was dropped
        r'^\s*---\s*$',
    ],
    'emph_ref': [r'\\emph\{[^}]*\\ref\{'],
    'tutorial_voice': [
        r'Welcome to the', r'Each chapter introduces', r'\bSo far the tutorial\b',
        r'coding reference', r'\\subsection\*\{Croqtile Tutorial\}',
        r'\\subsection\*\{Croqtile Coding Reference\}',
    ],
    'fragments': [
        r'\bvs\.\s*$',                   # clause cut off after "vs."
        r'\bprotocols\s*$',              # clause cut off before its dash
        r'\bomitted\s+(?:Section|\\emph)',
        r'\bpairing\s+it\s+is\b',
        r'\bturns\s+The\b',
    ],
}

# A few categories are line-anchored, so they need re.MULTILINE.
FLAGS = {'meta': re.I, 'empty_markup': re.M, 'fragments': re.M | re.I}

# Narrative scaffolding expected in the generated chapter files.
NARRATIVE = {
    'chapter_intro': (r'\\guidechapterintro\{', 3),
    'slot_intro': (r'\\guideslotintro\{', 62),
    'where_next': (r'\\guidechapternext\{', 3),
}


def strip_listings(s):
    """Blank out verbatim listings so code samples do not create false hits."""
    return re.sub(r'(?s)\\begin\{lstlisting\}.*?\\end\{lstlisting\}',
                  lambda m: re.sub(r'[^\n]', ' ', m.group(0)), s)


def scan():
    text = {}
    for p in FILES:
        text[p] = strip_listings(open(p, encoding='utf-8').read())
    result = {}
    for cat, pats in CHECKS.items():
        hits = []
        if pats is None:
            for p, s in text.items():
                for m in bg.HW.finditer(s):
                    line = s.count('\n', 0, m.start()) + 1
                    hits.append((p, line, s[max(0, m.start() - 45):m.end() + 45].replace('\n', ' ')))
        else:
            for pat in pats:
                rx = re.compile(pat, FLAGS.get(cat, 0))
                for p, s in text.items():
                    for m in rx.finditer(s):
                        line = s.count('\n', 0, m.start()) + 1
                        hits.append((p, line, s[max(0, m.start() - 45):m.end() + 45].replace('\n', ' ')))
        result[cat] = hits

    # A source-page heading that repeats its section title is redundant.
    title_keys = {re.sub(r'[^a-z0-9]', '', t.lower()): t
                  for t in bg.SLOT_TITLES.values()}
    dupes = []
    for p in sorted(glob.glob('build/units/*.tex')):
        first = open(p, encoding='utf-8').readline().strip()
        m = re.match(r'\\subsection\*\{(.+)\}', first)
        if not m:
            continue
        key = re.sub(r'[^a-z0-9]', '',
                     re.sub(r'\\[a-zA-Z]+', '', m.group(1)).lower())
        if key and key in title_keys:
            dupes.append((p, 1, m.group(1)[:90]))
    result['dup_headings'] = dupes

    # Identical long sentences across different units are a sign that the
    # opening pages were rewritten into the same introduction several times.
    seen = collections.Counter()
    origin = {}
    for p, s in text.items():
        for para in re.split(r'\n\s*\n', s):
            if para.strip().startswith('\\'):
                continue
            for sent in re.split(r'(?<=[.!?])\s+', para.strip()):
                t = re.sub(r'\s+', ' ', sent).strip()
                if len(t) < 55 or t.startswith(('\\', '&', '%')):
                    continue
                seen[t] += 1
                pos = s.find(t[:40])
                origin.setdefault(t, (p, s.count('\n', 0, max(pos, 0)) + 1))
    repeated = []
    for sentence, n in seen.items():
        if n > 1:
            p, line = origin[sentence]
            repeated.append((p, line, 'x%d %s' % (n, sentence[:90])))
    result['repeated_sentences'] = repeated

    # Two source pages are registered as genuinely empty group headings.
    empty_ok = {'build/units/F25-S01.tex', 'build/units/F33-S01.tex'}
    bare = []
    for p in sorted(glob.glob('build/units/*.tex')):
        s = open(p, encoding='utf-8').read()
        if not re.search(r'\\(?:subsection|paragraph)\*\{', s):
            continue
        body = re.sub(r'\\(?:subsection|paragraph)\*\{[^}]*\}\s*', '', s).strip()
        if not body and p not in empty_ok:
            bare.append((p, 1, 'heading with no body'))
    result['heading_only'] = bare

    # Units are separate prose blocks; joining two \inputs on consecutive
    # lines makes LaTeX typeset them as one paragraph.
    merged = []
    for p in FILES:
        if not p.startswith('content/ch0'):
            continue
        lines = open(p, encoding='utf-8').read().split('\n')
        for i, line in enumerate(lines[:-1]):
            if line.startswith('\\input') and lines[i + 1].startswith('\\input'):
                merged.append((p, i + 1, line.strip() + '  +  ' + lines[i + 1].strip()))
    result['merged_paragraphs'] = merged

    # The Preface must argue from the background trend, to human productivity,
    # and only then to agent-readiness.  Protect both the concepts and their
    # order so a future rewrite cannot collapse back into an AI-only pitch.
    preface = text.get('content/frontmatter.tex', '')
    concepts = ('heterogeneous', 'author productivity', 'human productivity',
                'agent-ready')
    gaps = [('content/frontmatter.tex', 1, 'missing ' + kw)
            for kw in concepts if kw not in preface]
    order = [preface.find(k) for k in ('author productivity',
                                       'human productivity', 'agent-ready')]
    if -1 not in order and not all(a < b for a, b in zip(order, order[1:])):
        gaps.append(('content/frontmatter.tex', 1, 'motivation out of order'))

    # The argument must read as prose: several natural paragraphs rather than
    # one wall of text, and no label-style lists of isolated words.
    opening = preface.split('\\section*')[0]
    paras = [p for p in re.split(r'\n\s*\n', opening)
             if p.strip() and not p.strip().startswith(('%', '\\'))]
    if len(paras) < 4:
        gaps.append(('content/frontmatter.tex', 1, 'opening not segmented'))
    for env in ('description', 'enumerate', 'itemize'):
        if '\\begin{' + env + '}' in preface:
            gaps.append(('content/frontmatter.tex', 1, 'fragmented list: ' + env))
    if re.search(r'\\textbf\{[A-Z][^}]*:\}', preface):
        gaps.append(('content/frontmatter.tex', 1, 'label-style fragments'))
    result['preface_positioning'] = gaps

    chapter_text = ''.join(text[p] for p in FILES if p.startswith('content/ch0'))
    for key, (pat, expected) in NARRATIVE.items():
        result.setdefault('narrative', {})[key] = (len(re.findall(pat, chapter_text)), expected)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, default=0)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--samples', type=int, default=2)
    args = ap.parse_args()

    res = scan()
    counts = {k: len(v) for k, v in res.items() if k != 'narrative'}
    narrative = res.get('narrative', {})
    missing = sum(max(0, expected - have) for have, expected in narrative.values())
    counts['narrative_gap'] = missing
    total = sum(counts.values())

    print('=== Review round %d ===' % args.round)
    for k in CHECKS:
        print('%-18s %4d' % (k, counts.get(k, 0)))
        for p, line, ctx in res.get(k, [])[:args.samples]:
            print('    %s:%d  ...%s...' % (p, line, ctx.strip()))
    print('%-18s %4d' % ('narrative_gap', missing))
    for key, (have, expected) in narrative.items():
        print('    %-16s %d/%d' % (key, have, expected))
    print('%-18s %4d' % ('TOTAL', total))

    out = os.path.join(PG, 'build', 'review-round%d.json' % args.round)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'round': args.round, 'counts': counts, 'total': total,
               'narrative': {k: v[0] for k, v in narrative.items()}},
              open(out, 'w', encoding='utf-8'), indent=2)

    prev = os.path.join(PG, 'build', 'review-round%d.json' % (args.round - 1))
    if args.compare and os.path.isfile(prev):
        old = json.load(open(prev, encoding='utf-8'))
        print('\n--- vs round %d (total %d -> %d) ---' % (old['round'], old['total'], total))
        for k in sorted(set(old['counts']) | set(counts)):
            a, b = old['counts'].get(k, 0), counts.get(k, 0)
            flag = ' OK' if b <= a else ' REGRESSION'
            print('  %-18s %4d -> %4d%s' % (k, a, b, flag))
        if total == 0 and old['total'] == 0:
            print('  STABLE AT ZERO (loop converged)')
        elif total >= old['total']:
            print('  ROUND DID NOT IMPROVE')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
