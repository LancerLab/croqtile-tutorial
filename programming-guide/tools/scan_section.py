#!/usr/bin/env python3
"""Compact quality report for one section, used by the refinement loop.

Combines the source-coverage check with a short prose scan: openings that
look like dangling placeholders, near-empty paragraphs, duplicated listing
captions and headings repeated from the slot title.

Usage: python3 tools/scan_section.py 2.1
"""
import argparse
import csv
import importlib.util
import os
import re
from pathlib import Path

PG = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cov = load('cov', PG / 'tools' / 'check_coverage.py')
bg = load('bg', PG / 'tools' / 'build_guide.py')

BAD_OPEN = re.compile(
    r'^(?:However|Therefore|As introduced|Below is|Note that|In this section|'
    r'Species|Thus|Hence|So|And|But|Also|Then|Finally|Lastly)\b', re.I)
SHORT = re.compile(r'^[A-Za-z]{1,5}[ .!?,;:]*$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slot')
    args = ap.parse_args()
    manifest = list(csv.DictReader(open(PG / 'build/manifest.tsv',
                                        encoding='utf-8'), delimiter='\t'))
    units = [r for r in manifest if r['slot'] == args.slot and r['status'] == 'kept']
    files, rows = bg.parse_registry()
    by_uid = {}
    for fid, rel in files:
        for sid, (head, body) in enumerate(
                bg.split_sections(os.path.join(cov.ROOT, rel)), 1):
            by_uid['F%02d-S%02d' % (fid, sid)] = '\n'.join(body)
    cov.report(args.slot, units, by_uid, cov.archived_without_target_terms())

    print('--- prose scan ---')
    captions = {}
    for u in units:
        p = PG / 'build/units' / (u['uid'] + '.tex')
        s = p.read_text(encoding='utf-8') if p.is_file() else ''
        paras = [x.strip() for x in re.split(r'\n\s*\n', s) if x.strip()]
        prose = [x for x in paras if not x.startswith('\\')]
        first = prose[0][:130].replace('\n', ' ') if prose else '(no prose)'
        print('%s [%d paras] %s' % (u['uid'], len(prose), first))
        for x in prose:
            if BAD_OPEN.match(x) or SHORT.match(x):
                print('    ! %s' % x[:80].replace('\n', ' '))
        for c in re.findall(r'caption=\{([^}]*)\}', s):
            captions.setdefault(c, []).append(u['uid'])
        heads = re.findall(r'\\(?:sub)?section\*\{([^}]*)\}', s)
        if heads:
            print('    heading: %s' % heads[0][:70])
    dupc = {k: v for k, v in captions.items() if len(v) > 1}
    print('duplicate captions: %d' % len(dupc))
    for k, v in list(dupc.items())[:8]:
        print('    %s -> %s' % (k[:60], ','.join(sorted(set(v)))))


if __name__ == '__main__':
    main()
