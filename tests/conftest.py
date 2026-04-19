# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src/`` is on sys.path when running from a fresh checkout that
# hasn't been ``pip install -e .``'d yet.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
