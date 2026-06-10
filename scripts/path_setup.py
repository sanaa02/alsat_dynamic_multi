#!/usr/bin/env python3
import os, sys
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_SUBDIRS = ['core', 'evaluation', 'models', 'training', 'wrappers']

def setup():
    for p in [_HERE] + [os.path.join(_HERE, s) for s in _SUBDIRS]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

def root_path(*parts):
    return os.path.join(_ROOT, *parts)

def scripts_path(*parts):
    return os.path.join(_HERE, *parts)

setup()


# ── Apply bsk_rl patches at import time ──────────────────────────────────
# Must run before any bsk_rl environment is created.
# Fixes: (1) eclipse search infinite loop for SSO perpetual illumination
#        (2) basePowerDraw warning from Basilisk dynamics model
import logging as _lg
_lg.basicConfig(level=_lg.WARNING)  # ensure logging is configured first
try:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    # import bsk_patches as _bsk_patches
    # _bsk_patches.apply_all()
except Exception as _e:
    _lg.getLogger(__name__).warning(f"bsk_patches not available: {_e}")
