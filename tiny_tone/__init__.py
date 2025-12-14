"""TinyToneNet-150k: tiny Mandarin tone classifier package.

This folder is self-contained so it can be dropped into existing repos.
"""

import os

# macOS/conda setups sometimes crash on torch import with:
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
# This opt-in workaround keeps the process alive. Override by setting TONE_KMP_DUPLICATE_LIB_OK=0.
if os.environ.get("TONE_KMP_DUPLICATE_LIB_OK", "1") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .model import TinyToneNet150k

__all__ = ["TinyToneNet150k"]
