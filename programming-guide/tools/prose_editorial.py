"""Apply reviewed copy edits and narrative expansions without changing listings.

The exact original text is retained in the editorial file so upstream changes
cannot silently apply a correction to the wrong passage. Order is significant:
the initial copy edits precede the section-by-section narrative pass.
"""
import json
from pathlib import Path
import re

EDITS = json.loads((Path(__file__).resolve().parents[1] /
                    'editorial/prose-polish.json').read_text())
NARRATIVE = json.loads((Path(__file__).resolve().parents[1] /
                        'editorial/narrative-polish.json').read_text())
LISTING = re.compile(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', re.S)


def polish_prose(text, uid):
    # The opening line contains the caption, which is also subject to copy edits.
    before = [block.split('\n', 1)[1] for block in LISTING.findall(text)]
    for stage in (EDITS, NARRATIVE):
        # Narrative edits are reviewed against generated units, whose writer
        # adds a final newline after this function returns.
        if stage is NARRATIVE:
            text += '\n'
        for edit in stage.get(uid, []):
            old, new = edit['before'], edit['after']
            if text.count(old) != 1:
                raise ValueError(f'{uid}: copy edit no longer matches exactly: {old[:100]!r}')
            text = text.replace(old, new, 1)
    if [block.split('\n', 1)[1] for block in LISTING.findall(text)] != before:
        raise ValueError(f'{uid}: copy edit changed a code listing')
    return text.rstrip('\n')
