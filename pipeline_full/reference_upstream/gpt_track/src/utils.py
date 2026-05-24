"""Small utilities shared across the pipeline.

Logging, seeding, timestamping, safe IO. Everything else lives in topic-specific
modules.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_LOG_INITIALIZED = False


def get_logger(name: str = "annita") -> logging.Logger:
    """Return a configured logger. Idempotent across calls."""
    global _LOG_INITIALIZED
    logger = logging.getLogger(name)
    if not _LOG_INITIALIZED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Squelch noisy third-party warnings that don't affect correctness.
        import warnings as _w
        _w.filterwarnings("ignore", category=UserWarning)
        _w.filterwarnings("ignore", category=FutureWarning)
        _w.filterwarnings("ignore", category=RuntimeWarning)
        _LOG_INITIALIZED = True
    return logger


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy and PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp() -> str:
    """Return a YYYYMMDD_HHMM timestamp suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def visit_index(col: str) -> int | None:
    """Return the integer visit index encoded in the column name, or None.

    Uses the ``_v<int>`` suffix convention (e.g. ``ast_v3`` -> 3, ``Age_v12`` -> 12).
    """
    m = re.search(r"_v(\d+)$", col)
    return int(m.group(1)) if m else None


def base_name(col: str) -> str | None:
    """Strip a ``_v<int>`` suffix and return the biomarker stem (or None)."""
    m = re.match(r"(.+)_v\d+$", col)
    return m.group(1) if m else None


def write_json(path: Path, obj: Any) -> None:
    """Dump ``obj`` to JSON, creating the parent dir as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class StopWatch:
    """Tiny context manager that logs wall time of a block."""

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger or get_logger()
        self._t0 = 0.0

    def __enter__(self) -> "StopWatch":
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.time() - self._t0
        self.logger.info("%s took %.2fs", self.label, dt)


def chunked(seq: Iterable, n: int) -> Iterable[list]:
    """Yield successive n-sized chunks from ``seq``."""
    bucket: list = []
    for x in seq:
        bucket.append(x)
        if len(bucket) == n:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
