#!/usr/bin/env python3
"""Coverage check for one guide section (slot).

For every content unit in a slot, compare the upstream Part I (tutorial/) and
Part III (documentation/) Markdown with the generated LaTeX fragments and
report anything that looks like lost information:

  * clauses archived in HARDWARE-NOTES.md that contain no target term at all
    (a plausible accidental removal);
  * source sentences whose content words are largely absent from the output;
  * code blocks, tables and images that did not survive the conversion.

Usage:
    python3 tools/check_coverage.py 1.2
    python3 tools/check_coverage.py --all
"""
import argparse
import csv
import importlib.util
import os
import re
from pathlib import Path

PG = Path(__file__).resolve().parents[1]
ROOT = PG.parent
_spec = importlib.util.spec_from_file_location('bg', PG / 'tools' / 'build_guide.py')
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)

STOP = set('''a an and are as at be been but by can could do does for from had has
have if in into is it its of on or our that the their then there these this
those to was we were what when where which while will with would you your also
not no such than they them one two each other more most some any all both very
just only same so because about over under again further once here how'''.split())


def words(s):
    s = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', s)
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = re.sub(r'\\[a-zA-Z@]+\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z@]+', ' ', s)
    s = re.sub(r'[^A-Za-z0-9_ ]', ' ', s)
    return [w for w in s.lower().split() if len(w) > 2 and w not in STOP]


def source_sentences(body):
    """Plain prose sentences that carry no target-platform vocabulary."""
    out, buf, fence = [], [], False
    for line in body:
        if re.match(r'^\s*```', line):
            fence = not fence
            continue
        if fence or not line.strip():
            continue
        if line.lstrip().startswith(('#', '|', '<', '!', '-', '*', '>', ':')):
            continue
        buf.append(line.strip())
    for sent in re.split(r'(?<=[.!?])\s+', ' '.join(buf)):
        sent = sent.strip()
        if len(words(sent)) >= 7 and not bg.HW.search(sent):
            out.append(sent)
    return out


def archived_without_target_terms():
    p = PG / 'HARDWARE-NOTES.md'
    if not p.is_file():
        return {}
    chunks, uid = {}, None
    for line in p.read_text(encoding='utf-8').split('\n'):
        m = re.match(r'^## (F\d+-S\d+)', line)
        if m:
            uid = m.group(1)
            chunks.setdefault(uid, [])
            continue
        if uid and line.startswith('    ') and line.strip():
            text = line.strip()
            if not bg.HW.search(text):
                chunks[uid].append(text)
    return {k: v for k, v in chunks.items() if v}


def report(slot, units, sections_by_uid, archived):
    gen_paths = [PG / 'build/units' / (u['uid'] + '.tex') for u in units]
    gen = ' '.join(p.read_text(encoding='utf-8') for p in gen_paths if p.is_file())
    gen_words = set(words(gen))
    print('=== slot %s : %d units ===' % (slot, len(units)))
    src_code = src_table = src_image = 0
    missing = []
    for u in units:
        raw = sections_by_uid.get(u['uid'], '')
        src_code += len(re.findall(r'^\s*```', raw, re.M)) // 2
        src_table += len(re.findall(r'^\s*\|.*\|\s*$', raw, re.M))
        src_image += len(re.findall(r'!\[', raw))
        for sent in source_sentences(raw.split('\n')):
            w = set(words(sent))
            if not w:
                continue
            cover = sum(x in gen_words for x in w) / len(w)
            if cover < 0.6:
                missing.append((u['uid'], cover, sent[:110]))
    print('source   : %d code blocks, %d table rows, %d images' %
          (src_code, src_table, src_image))
    print('generated: %d listings, %d tables, %d figures' %
          (gen.count(r'\begin{lstlisting}'), gen.count(r'\begin{table}'),
           gen.count(r'\begin{figure}')))
    false_removals = [(u['uid'], t) for u in units for t in archived.get(u['uid'], [])]
    print('archived clauses with no target term: %d' % len(false_removals))
    for uid, text in false_removals[:8]:
        print('  ! %s  %s' % (uid, text[:100]))
    print('source sentences with weak coverage : %d' % len(missing))
    for uid, cover, sent in missing[:12]:
        print('  ? %s (%.2f) %s' % (uid, cover, sent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slot', nargs='?')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    files, rows = bg.parse_registry()
    manifest = list(csv.DictReader(open(PG / 'build/manifest.tsv',
                                        encoding='utf-8'), delimiter='\t'))
    sections_by_uid = {}
    for fid, rel in files:
        for sid, (head, body) in enumerate(bg.split_sections(os.path.join(ROOT, rel)), 1):
            sections_by_uid['F%02d-S%02d' % (fid, sid)] = '\n'.join(body)
    archived = archived_without_target_terms()
    slots = sorted({r['slot'] for r in manifest
                    if r['slot'] not in ('X', '占位') and r['status'] == 'kept'},
                   key=lambda s: tuple(int(x) for x in s.split('.')))
    targets = slots if args.all else [args.slot]
    for slot in targets:
        units = [r for r in manifest if r['slot'] == slot and r['status'] == 'kept']
        if not units:
            print('=== slot %s : no kept units ===' % slot)
            continue
        report(slot, units, sections_by_uid, archived)


if __name__ == '__main__':
    main()
