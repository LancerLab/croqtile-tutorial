#!/usr/bin/env python3
"""Render the guide's Manim figures on white, crop margins, and sync assets.

Run from any directory: python3 programming-guide/tools/render_figures.py
Requires Manim and Pillow. Dark site variants are independent of this workflow.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
PG = ROOT / 'programming-guide'


def inventory():
    seen = set()
    for unit in csv.DictReader((PG / 'build/manifest.tsv').open(), delimiter='\t'):
        if unit['status'] != 'kept' or unit['uid'] == 'F11-S09':
            continue
        tex = (PG / 'build/units' / (unit['uid'] + '.tex')).read_text()
        for asset in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex):
            if asset in seen or not asset.endswith('_light.png'):
                continue
            seen.add(asset)
            base = re.sub(r'^f\d+-', '', asset).removesuffix('_light.png')
            scripts = [p for p in (ROOT / 'figures').glob('*.py') if p.stem.endswith('_' + base)]
            if base == 'compile_and_run':
                scripts = [ROOT / 'figures/guide_compile_and_run.py']
            elif base == 'chunkat_3d':
                scripts = [ROOT / 'figures/guide_chunkat_3d.py']
            elif 'sm86_vs_sm90' in base:
                scripts = [ROOT / 'figures/ch04_fig2_sm86_vs_sm90.py']
            if len(scripts) != 1:
                raise ValueError(f'Expected one Manim source for {asset}: {scripts}')
            script = scripts[0]
            scene = re.search(r'^class (\w+)\(', script.read_text(), re.M)[1]
            destinations = list((ROOT / 'docs/assets/images').rglob(base + '_light.png'))
            if base == 'chunkat_3d':
                destinations = [ROOT / 'documentation/assets/figures/chunkat_3d_light.png']
            if len(destinations) != 1:
                raise ValueError(f'Expected one tutorial asset for {asset}')
            yield asset, script, scene, destinations[0]


def render(item, temp):
    asset, script, scene, destination = item
    work = temp / script.stem
    work.mkdir()
    log = work / 'render.log'
    env = dict(os.environ, MANIM_THEME='light')
    with log.open('w') as out:
        subprocess.run(['python3', '-m', 'manim', '-s', '-r', '2560,1440', '--disable_caching',
                        '--media_dir', str(work), str(script), scene],
                       cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    images = list((work / 'images').rglob(scene + '*.png'))
    if len(images) != 1:
        raise ValueError(f'Unexpected render output for {scene}: {images}')
    image = Image.open(images[0]).convert('RGB')
    bbox = ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox()
    if not bbox:
        raise ValueError(f'Empty scene: {scene}')
    # A fixed canvas preserves identical type sizes and margins at full text width.
    # Per-image cropping made otherwise identical labels change size in the book.
    if bbox[0] < 24 or bbox[1] < 24 or bbox[2] > image.width-24 or bbox[3] > image.height-24:
        raise ValueError(f'Content too close to canvas edge in {scene}: {bbox}')
    image.save(destination)
    shutil.copyfile(destination, PG / 'figures' / asset)
    if asset == 'f03-compile_and_run_light.png':
        shutil.copyfile(destination, PG / 'figures/f03-compile_and_run_zh.png')
    return asset, image.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', help='Render only matching asset names or scene names')
    args = parser.parse_args()
    temp = Path(tempfile.mkdtemp(prefix='croq-white-figures-'))
    print('Render logs:', temp, flush=True)
    items = list(inventory())
    if args.only:
        items = [item for item in items if any(key in item[0] or key == item[2] for key in args.only)]
        if not items:
            parser.error('No figures match --only')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for asset, size in pool.map(lambda item: render(item, temp), items):
            print(asset, size, flush=True)
    print(f'Rendered and synchronized {len(items)} figures.', flush=True)


if __name__ == '__main__':
    main()
