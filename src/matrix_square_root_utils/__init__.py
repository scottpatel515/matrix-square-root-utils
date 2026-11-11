"""Matrix square root utilities.

Public API:
    sqrtm(A, tol=1e-12, max_iter=50) -> list[list[float]]
    DenmanBeaversError
"""

from .core import DenmanBeaversError, sqrtm

__all__ = ["sqrtm", "DenmanBeaversError"]
