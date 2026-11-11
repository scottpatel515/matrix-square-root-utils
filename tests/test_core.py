import math
import unittest

from matrix_square_root_utils import DenmanBeaversError, sqrtm


def _matmul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _max_abs_diff(A, B):
    n = len(A)
    return max(abs(A[i][j] - B[i][j]) for i in range(n) for j in range(n))


class TestSqrtm(unittest.TestCase):
    def assertMatrixAlmostEqual(self, got, want, tol=1e-8):
        self.assertEqual(len(got), len(want))
        for i in range(len(got)):
            self.assertEqual(len(got[i]), len(want[i]))
        diff = _max_abs_diff(got, want)
        self.assertLess(diff, tol, "matrices differ by %g" % diff)

    def test_identity(self):
        A = [[1.0, 0.0], [0.0, 1.0]]
        R = sqrtm(A)
        self.assertMatrixAlmostEqual(R, A)

    def test_zero_matrix(self):
        A = [[0.0, 0.0], [0.0, 0.0]]
        R = sqrtm(A)
        self.assertMatrixAlmostEqual(R, A)

    def test_diagonal(self):
        A = [[4.0, 0.0], [0.0, 9.0]]
        R = sqrtm(A)
        self.assertMatrixAlmostEqual(R, [[2.0, 0.0], [0.0, 3.0]])

    def test_scalar_multiple_of_identity(self):
        A = [[16.0, 0.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 16.0]]
        R = sqrtm(A)
        self.assertMatrixAlmostEqual(R, [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])

    def test_rotation_90_degrees(self):
        # A = [[0, -1], [1, 0]] has square root (1/sqrt(2)) * [[1, -1], [1, 1]].
        A = [[0.0, -1.0], [1.0, 0.0]]
        R = sqrtm(A)
        s = 1.0 / math.sqrt(2.0)
        want = [[s, -s], [s, s]]
        self.assertMatrixAlmostEqual(R, want, tol=1e-8)

    def test_symmetric_positive_definite(self):
        A = [[5.0, 4.0], [4.0, 5.0]]
        R = sqrtm(A)
        # R^2 should equal A.
        self.assertMatrixAlmostEqual(_matmul(R, R), A, tol=1e-8)
        # R should be symmetric for a symmetric A.
        self.assertAlmostEqual(R[0][1], R[1][0], places=8)

    def test_non_symmetric_with_positive_eigenvalues(self):
        A = [[3.0, 1.0], [0.0, 2.0]]
        R = sqrtm(A)
        self.assertMatrixAlmostEqual(_matmul(R, R), A, tol=1e-8)

    def test_negative_identity_raises(self):
        A = [[-1.0, 0.0], [0.0, -1.0]]
        with self.assertRaises(DenmanBeaversError):
            sqrtm(A)

    def test_empty_matrix_raises(self):
        with self.assertRaises(ValueError):
            sqrtm([])

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            sqrtm([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            sqrtm([[1.0, "a"], [3.0, 4.0]])

    def test_accepts_integers(self):
        A = [[4, 0], [0, 4]]
        R = sqrtm(A)
        self.assertMatrixAlmostEqual(R, [[2.0, 0.0], [0.0, 2.0]])

    def test_non_convergence_raises(self):
        A = [[4.0, 0.0], [0.0, 4.0]]
        with self.assertRaises(DenmanBeaversError):
            sqrtm(A, tol=0.0, max_iter=0)

    def test_singular_raises(self):
        A = [[0.0, 1.0], [0.0, 0.0]]
        with self.assertRaises(DenmanBeaversError):
            sqrtm(A)


if __name__ == "__main__":
    unittest.main()
