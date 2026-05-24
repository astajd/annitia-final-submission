"""Re-export :func:`metrics.cindex` to avoid circular imports inside model modules."""
from .metrics import cindex

__all__ = ["cindex"]
