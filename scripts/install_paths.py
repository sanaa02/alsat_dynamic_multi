#!/usr/bin/env python3
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIRS     = ['core', 'evaluation', 'models', 'training', 'wrappers']
MARKER      = '# ---- ALSAT path-setup'

PREAMBLE = (
    '\n# ---- ALSAT path-setup --------------------------------------------\n'
    'import os as _os, sys as _sys\n'
    "_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))\n"
    'import path_setup  # noqa\n'
    '# -------------------------------------------------------------------\n\n'
)

def _find_insert(lines):
    i = 0
    if lines and lines[0].startswith('#!'):
        i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip()[:3] in ('"""', "'''"):
        d = lines[i].strip()[:3]
        rest = lines[i].strip()[3:]
        if rest.endswith(d):
            return i + 1
        i += 1
        while i < len(lines) and d not in lines[i]:
            i += 1
        return i + 1
    return i

patched = skipped = 0
for subdir in SUBDIRS:
    sub_path = os.path.join(SCRIPTS_DIR, subdir)
    if not os.path.isdir(sub_path):
        print(f'  [skip] {subdir}/ not found'); continue
    for fname in sorted(os.listdir(sub_path)):
        if not fname.endswith('.py') or fname.startswith('__'): continue
        fpath = os.path.join(sub_path, fname)
        content = open(fpath, encoding='utf-8').read()
        if MARKER in content: skipped += 1; continue
        lines = content.split('\n')
        idx   = _find_insert(lines)
        new   = lines[:idx] + PREAMBLE.split('\n') + lines[idx:]
        open(fpath, 'w', encoding='utf-8').write('\n'.join(new))
        print(f'  OK  {subdir}/{fname}'); patched += 1

print(f'\n  {patched} files patched, {skipped} already done.')
print('  Run any script from any directory -- imports will work.')
