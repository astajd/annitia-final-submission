"""Run the Phase 2 feature-set sweep.

For every Phase 2 config (`experiments/configs/phase2_*.yaml`) we run the
existing :func:`src.run_experiment.run_config` to produce per-fold CV metrics
and OOF/test predictions. Submissions are *not* written here; we generate the
4 named candidates separately in :mod:`src.build_phase2_submissions`.

Usage::

    python -m src.run_phase2_sweep
"""
from __future__ import annotations

from pathlib import Path

from . import config as cfg
from .run_experiment import run_config
from .utils import get_logger

_LOG = get_logger(__name__)


def main() -> None:
    cfg.ensure_dirs()
    configs = sorted(cfg.EXPERIMENT_CONFIGS.glob("phase2_*.yaml"))
    if not configs:
        _LOG.warning("no phase2_*.yaml configs found")
        return
    for p in configs:
        try:
            run_config(p)
        except Exception as e:  # noqa: BLE001
            _LOG.error("config %s failed: %s", p.name, e)


if __name__ == "__main__":
    main()
