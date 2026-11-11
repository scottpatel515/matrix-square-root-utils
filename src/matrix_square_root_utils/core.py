"""Principal matrix square root via the Denman-Beavers iteration.

This module implements the scaled Denman-Beavers iteration for small dense
matrices. The iteration computes both the square root Y and its inverse Z:

    Y_{k+1} = 0.5 * (Y_k + Z_k^{-1})
    Z_{k+1} = 0.5 * (Z_k + Y_k^{-1})

starting from Y_0 = A and Z_0 = I. The iteration converges for matrices whose
eigenvalues lie in the open right half-plane. A simple scaling by 1/||A||_inf
is applied once at the start to improve the region of convergence; this is a
cheap heuristic, not a full balancing scheme.

Only the standard library is used. Matrices are represented as lists of lists
of floats (row-major).
"""

from __future__ import annotations

import math
from typing import List, Sequence

Matrix = List[List[float]]


class DenmanBeaversError(Exception):
    """Raised when the Denman-Beavers iteration cannot produce a result."""


def _identity(n: int) -> Matrix:
    """Return the n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> Matrix:
    """Multiply two square matrices of the same size."""
    n = len(A)
    C: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C


def _add(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> Matrix:
    """Add two matrices of the same size."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _scale(A: Sequence[Sequence[float]], s: float) -> Matrix:
    """Scale a matrix by a scalar."""
    n = len(A)
    return [[A[i][j] * s for j in range(n)] for i in range(n)]


def _norm_inf(A: Sequence[Sequence[float]]) -> float:
    """Return the infinity norm (maximum absolute row sum) of A."""
    best = 0.0
    for row in A:
        s = 0.0
        for v in row:
            s += abs(v)
        if s > best:
            best = s
    return best


def _inverse(A: Sequence[Sequence[float]], tol: float) -> Matrix:
    """Return the inverse of A using Gauss-Jordan elimination with partial pivoting.

    Raises DenmanBeaversError if A is singular to working precision.
    """
    n = len(A)
    aug: Matrix = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        # Find pivot.
        pivot_row = -1
        pivot_val = 0.0
        for r in range(col, n):
            v = abs(aug[r][col])
            if v > pivot_val:
                pivot_val = v
                pivot_row = r
        if pivot_row == -1 or pivot_val < tol:
            raise DenmanBeaversError("matrix is singular to working precision")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        inv_pivot = 1.0 / aug[col][col]
        for j in range(2 * n):
            aug[col][j] *= inv_pivot
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            row_r = aug[r]
            row_c = aug[col]
            for j in range(2 * n):
                row_r[j] -= factor * row_c[j]
    return [row[n:] for row in aug]


def _max_abs_diff(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> float:
    """Return max |A[i][j] - B[i][j]|."""
    n = len(A)
    best = 0.0
    for i in range(n):
        Ai = A[i]
        Bi = B[i]
        for j in range(n):
            d = abs(Ai[j] - Bi[j])
            if d > best:
                best = d
    return best


def _validate_square(A: Sequence[Sequence[float]]) -> int:
    """Check that A is a non-empty square matrix of floats. Return its size."""
    if not isinstance(A, (list, tuple)):
        raise TypeError("matrix must be a list of rows")
    n = len(A)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    for row in A:
        if not isinstance(row, (list, tuple)):
            raise TypeError("each row must be a list")
        if len(row) != n:
            raise ValueError("matrix must be square")
        for v in row:
            if not isinstance(v, (int, float)):
                raise TypeError("matrix entries must be numbers")
    return n


def sqrtm(A: Sequence[Sequence[float]], tol: float = 1e-12, max_iter: int = 50) -> Matrix:
    """Compute the principal square root of a small dense matrix.

    The principal square root is the unique square root whose eigenvalues lie
    in the open right half-plane (when it exists). This implementation uses the
    scaled Denman-Beavers iteration, which converges for matrices with no
    eigenvalues on the closed negative real axis.

    Args:
        A: A square matrix given as a sequence of rows (lists of numbers).
        tol: Convergence tolerance on the maximum absolute change of Y.
        max_iter: Maximum number of iterations before giving up.

    Returns:
        The principal square root Y as a list of lists of floats.

    Raises:
        ValueError: If A is empty or not square.
        TypeError: If A is not a sequence of sequences of numbers.
        DenmanBeaversError: If the iteration fails to converge, a singular
            iterate is encountered, or the matrix appears to have no
            principal square root in the supported class.

    Notes:
        The initial scaling Y_0 = A / ||A||_inf is a cheap heuristic that
        widens the convergence region. It is not a substitute for a proper
        Schur-based method, but it is enough for the small, well-conditioned
        matrices this library targets.
    """
    n = _validate_square(A)
    A_mat: Matrix = [[float(v) for v in row] for row in A]

    norm = _norm_inf(A_mat)
    if norm == 0.0:
        # The zero matrix is its own square root.
        return [[0.0] * n for _ in range(n)]

    Y = _scale(A_mat, 1.0 / norm)
    Z = _identity(n)

    for _ in range(max_iter):
        try:
            Z_inv = _inverse(Z, tol)
            Y_inv = _inverse(Y, tol)
        except DenmanBeaversError:
            raise
        Y_next = _scale(_add(Y, Z_inv), 0.5)
        Z_next = _scale(_add(Z, Y_inv), 0.5)
        if _max_abs_diff(Y_next, Y) < tol:
            # Undo the initial scaling: if B = A / s, then sqrt(A) = sqrt(s) * sqrt(B).
            return _scale(Y_next, math.sqrt(norm))
        Y, Z = Y_next, Z_next

    raise DenmanBeaversError(
        "Denman-Beavers iteration did not converge within max_iter=%d" % max_iter
    )
