"""Local logging setup: a shared logger that writes to console and to a file.

Detailed per-step *metrics* are handled separately by the JSONL callback in
``callbacks.py``; this module is about human-readable run logs.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_LOGGER_NAME = "nemotron_finetune"
_CONFIGURED = False


def setup_logging(output_dir: str | Path, level: str = "INFO") -> logging.Logger:
    """Configure the package logger to log to stdout and ``<output_dir>/logs/train.log``."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    # Avoid duplicate handlers if called twice in the same process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not _CONFIGURED and not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)
    return logger


def write_json(path: str | Path, obj: Any) -> None:
    """Write a JSON document (used for run metadata / environment snapshots)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
