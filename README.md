# matrix-square-root-utils

Computes the principal square root of a small dense matrix using the scaled
Denman-Beavers iteration.

## Usage

```python
from matrix_square_root_utils import sqrtm, DenmanBeaversError

A = [[5.0, 4.0], [4.0, 5.0]]
R = sqrtm(A)
print(R)
# [[2.9999999999999996, 1.9999999999999998],
#  [1.9999999999999998, 2.9999999999999996]]

try:
    sqrtm([[-1.0, 0.0], [0.0, -1.0]])
except DenmanBeaversError as exc:
    print("no principal square root:", exc)
```

The public API is exactly two names:

- `sqrtm(A, tol=1e-12, max_iter=50)` returns a list of lists of floats.
- `DenmanBeaversError` is raised when the iteration cannot produce a result.

## Why this exists

Most numerical libraries that offer a matrix square root pull in BLAS, LAPACK,
or a large scientific stack. For a 2x2 or 3x3 matrix embedded in a small
tool, that is a heavy dependency for a single operation. This library trades
robustness on large or ill-conditioned matrices for zero dependencies: it uses
only the standard library and a fixed iteration count.

The trade-off is explicit. The Denman-Beavers iteration is simple and
self-contained, but it converges slowly (or not at all) when the matrix has
eigenvalues near the negative real axis, and it requires a matrix inverse at
every step. For small, well-conditioned inputs it is more than adequate; for
anything large, use a Schur-based method from a real linear algebra library.

## The awkward edge

The iteration only converges for matrices whose eigenvalues lie in the open
right half-plane. A matrix like `[[-1, 0], [0, -1]]` has a square root, but not
a principal one in that class, and the iteration will fail. This library raises
`DenmanBeaversError` in that case rather than returning a complex result. If
you need square roots of matrices with negative real eigenvalues, this is not
the library for you.

A second edge: the initial scaling by `1 / ||A||_inf` is a heuristic. It widens
the convergence region but does not guarantee convergence. The `max_iter` and
`tol` arguments are there so callers can tune or detect failure.
