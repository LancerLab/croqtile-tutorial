"""Publication captions and corrections shared by the English guide build."""
import json
from pathlib import Path
import re

PG = Path(__file__).resolve().parents[1]
CAPTIONS = json.loads((PG / 'editorial/figure-captions.json').read_text())


def polish_figures(fragment, uid, escape):
    fragment = fragment.replace('f14-chunkat-6-17-4.drawio.png',
                                'f14-chunkat_3d_light.png')
    if uid == 'F11-S09':
        fragment = re.sub(r'(?s)\\begin\{figure\}.*?\\end\{figure\}\s*', '', fragment)
    # Keep the caption inside the figure so it travels with the image.
    pattern = r'(?s)(\\begin\{figure\}.*?\\includegraphics[^\n]*\{([^}]+)\}.*?)(\\end\{figure\})(\s*\\begin\{center\}\\small\\emph\{.*?\}\\end\{center\})?'
    def replace(match):
        path = match[2]
        if path not in CAPTIONS:
            return match[0]
        body = re.sub(r'(?s)\\caption\{.*?\}\s*', '', match[1])
        return body + '\\caption{' + escape(CAPTIONS[path]['en']) + '}\n' + match[3]
    fragment = re.sub(pattern, replace, fragment)
    if uid == 'F06-S01':
        fragment = re.sub(r'(?s)\\begin\{center\}.*?\\end\{center\}\s*', '', fragment)
        fragment = fragment.replace('What makes this hard for programmers is not the math but the .',
                                    'The difficulty is mapping matrix elements to the distributed operand fragments.')
    if uid == 'F09-S02':
        fragment = fragment.replace('One descriptor-based operation describes the tensor slice;',
                                    'A descriptor-based operation describes the tensor slice and delegates address generation to the transfer engine.')
    if uid == 'F09-S04':
        fragment = re.sub(r'(?s)Croqtile takes the opposite approach:.*?(?=\n\n\\begin\{figure\})',
                         'Croqtile makes tile geometry and layout explicit through structured movement operations. '
                         'This gives the compiler the information it needs to lower a transfer efficiently and avoids repeating per-element address choreography.', fragment)
        fragment = re.sub(r"(?s)Croqtile's \\texttt\{dma.copy\} follows.*?\Z",
                         "The trade is explicit: structured operations constrain how a transfer is described. "
                         "They support efficient lowering, while actual performance still depends on tile sizes, alignment, layouts, and synchronization.", fragment)
    return fragment
