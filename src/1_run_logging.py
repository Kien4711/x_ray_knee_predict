# Pipeline step 1: logging setup (runs first for any CLI command).
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_knee_oa_logging(
    log_file: Path,
    *,
    level: int = logging.INFO,
    also_console: bool = True,
) -> logging.Logger:
    """Attach file (+ optional console) handlers to the ``knee_oa`` logger tree."""
    log_file = log_file.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("knee_oa")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if also_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    return root
