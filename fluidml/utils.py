"""Utilities for the tbrf package.

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis

"""

# Built-in packages

# Third party packages
import numpy as np

# Local packages


__all__ = [
    "enforce_realizability",
    "make_realizable",
    "make_realizable2",
]


def make_realizable(b: np.ndarray) -> np.ndarray:
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn:
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.
    :param b: the predicted anisotropy tensor (num_points X 9 array)

    Parameters
    ----------
    b : np.ndarray[shape=(n, 9)]
        Anisotropy tensor on which to enforce realizability.

    Returns
    -------
    b_realizable : np.ndarray[shape=(n, 9)]

    """
    b = b.copy()
    n, _ = b.shape
    A = np.zeros([3, 3])
    for i in range(n):
        # Scales all on-diags to retain zero trace
        if np.min(b[i, [0, 4, 8]]) < -1.0 / 3.0:
            b[i, [0, 4, 8]] *= -1.0 / (3.0 * np.min(b[i, [0, 4, 8]]))
        if 2.0 * np.abs(b[i, 1]) > b[i, 0] + b[i, 4] + 2.0 / 3.0:
            b[i, 1] = (b[i, 0] + b[i, 4] + 2.0 / 3.0) * 0.5 * np.sign(b[i, 1])
            b[i, 3] = (b[i, 0] + b[i, 4] + 2.0 / 3.0) * 0.5 * np.sign(b[i, 1])
        if 2.0 * np.abs(b[i, 5]) > b[i, 4] + b[i, 8] + 2.0 / 3.0:
            b[i, 5] = (b[i, 4] + b[i, 8] + 2.0 / 3.0) * 0.5 * np.sign(b[i, 5])
            b[i, 7] = (b[i, 4] + b[i, 8] + 2.0 / 3.0) * 0.5 * np.sign(b[i, 5])
        if 2.0 * np.abs(b[i, 2]) > b[i, 0] + b[i, 8] + 2.0 / 3.0:
            b[i, 2] = (b[i, 0] + b[i, 8] + 2.0 / 3.0) * 0.5 * np.sign(b[i, 2])
            b[i, 6] = (b[i, 0] + b[i, 8] + 2.0 / 3.0) * 0.5 * np.sign(b[i, 2])

        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = b[i, 0]
        A[1, 1] = b[i, 4]
        A[2, 2] = b[i, 8]
        A[0, 1] = b[i, 1]
        A[1, 0] = b[i, 1]
        A[1, 2] = b[i, 5]
        A[2, 1] = b[i, 5]
        A[0, 2] = b[i, 2]
        A[2, 0] = b[i, 2]
        evalues, evectors = np.linalg.eig(A)
        if (
            np.max(evalues)
            < (3.0 * np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1]) / 2.0
        ):
            evalues = (
                evalues
                * (3.0 * np.abs(np.sort(evalues)[1]) - np.sort(evalues)[1])
                / (2.0 * np.max(evalues))
            )
            A = np.dot(
                np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors)
            )
            for j in range(3):
                b[i, j] = A[j, j]
            b[i, 1] = A[0, 1]
            b[i, 5] = A[1, 2]
            b[i, 2] = A[0, 2]
            b[i, 3] = A[0, 1]
            b[i, 7] = A[1, 2]
            b[i, 6] = A[0, 2]
        if np.max(evalues) > 1.0 / 3.0 - np.sort(evalues)[1]:
            evalues = (
                evalues * (1.0 / 3.0 - np.sort(evalues)[1]) / np.max(evalues)
            )
            A = np.dot(
                np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors)
            )
            for j in range(3):
                b[i, j] = A[j, j]
            b[i, 1] = A[0, 1]
            b[i, 5] = A[1, 2]
            b[i, 2] = A[0, 2]
            b[i, 3] = A[0, 1]
            b[i, 7] = A[1, 2]
            b[i, 6] = A[0, 2]

    return b


def make_realizable2(b: np.ndarray) -> np.ndarray:
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn:
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.
    :param b: the predicted anisotropy tensor (num_points X 9 array)

    Parameters
    ----------
    b : np.ndarray[shape=(n, 9)]
        Anisotropy tensor on which to enforce realizability.

    Returns
    -------
    b_realizable : np.ndarray[shape=(n, 9)]

    """
    b = b.copy().reshape(-1, 3, 3)
    n = b.shape[0]
    A = np.zeros([3, 3])
    for i in range(n):
        # Scales all on-diags to retain zero trace
        min_diag = np.min(b[i, [0, 1, 2], [0, 1, 2]])
        if min_diag < -1 / 3:
            b[i, [0, 1, 2], [0, 1, 2]] *= -1 / (3 * min_diag)
        if 2 * np.abs(b[i, 0, 1]) > b[i, 0, 0] + b[i, 1, 1] + 2 / 3:
            b[i, 0, 1] = (
                (b[i, 0, 0] + b[i, 1, 1] + 2 / 3) * 0.5 * np.sign(b[i, 0, 1])
            )
            b[i, 1, 0] = (
                (b[i, 0, 0] + b[i, 1, 1] + 2 / 3) * 0.5 * np.sign(b[i, 0, 1])
            )
        if 2 * np.abs(b[i, 1, 2]) > b[i, 1, 1] + b[i, 2, 2] + 2 / 3:
            b[i, 1, 2] = (
                (b[i, 1, 1] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 1, 2])
            )
            b[i, 2, 1] = (
                (b[i, 1, 1] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 1, 2])
            )
        if 2 * np.abs(b[i, 0, 2]) > b[i, 0, 0] + b[i, 2, 2] + 2 / 3:
            b[i, 0, 2] = (
                (b[i, 0, 0] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 0, 2])
            )
            b[i, 2, 0] = (
                (b[i, 0, 0] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 0, 2])
            )

        # Enforce positive semidefinite by pushing evals to non-negative
        A[0, 0] = b[i, 0, 0]
        A[1, 1] = b[i, 1, 1]
        A[2, 2] = b[i, 2, 2]
        A[0, 1] = b[i, 0, 1]
        A[1, 0] = b[i, 0, 1]
        A[1, 2] = b[i, 1, 2]
        A[2, 1] = b[i, 1, 2]
        A[0, 2] = b[i, 0, 2]
        A[2, 0] = b[i, 0, 2]
        evals, evecs = np.linalg.eig(A)
        evals_sorted = np.sort(evals)
        evals_max = np.max(evals)
        if evals_max < (3 * np.abs(evals_sorted[1]) - evals_sorted[1]) / 2:
            evals = (
                evals
                * (3 * np.abs(evals_sorted[1]) - evals_sorted[1])
                / (2 * evals_max)
            )
            A = evecs @ np.diag(evals) @ np.linalg.inv(evecs)
            for j in range(3):
                b[i, 0, j] = A[j, j]
            b[i, 0, 1] = A[0, 1]
            b[i, 1, 2] = A[1, 2]
            b[i, 0, 2] = A[0, 2]
            b[i, 1, 0] = A[0, 1]
            b[i, 2, 1] = A[1, 2]
            b[i, 2, 0] = A[0, 2]
        evals_sorted = np.sort(evals)
        evals_max = np.max(evals)
        if evals_max > 1 / 3 - evals_sorted[1]:
            evals = evals * (1 / 3 - evals_sorted[1]) / evals_max
            A = evecs @ np.diag(evals) @ np.linalg.inv(evecs)
            for j in range(3):
                b[i, 0, j] = A[j, j]
            b[i, 0, 1] = A[0, 1]
            b[i, 1, 2] = A[1, 2]
            b[i, 0, 2] = A[0, 2]
            b[i, 1, 0] = A[0, 1]
            b[i, 2, 1] = A[1, 2]
            b[i, 2, 0] = A[0, 2]
    b = b.reshape(-1, 9)

    return b


def enforce_realizability(bhat: np.ndarray) -> np.ndarray:
    r"""Enforce the realizibility of the anisotropy tensors. Each tensor
    must follow:
    - $\forall i = 1...3 \frac{-1}{3} \leq b_{ii} \geq
    \frac{2}{3}$
    - $\forall i, j = 1...3, i \neq j \frac{-1}{2} \leq b_{ij} \geq
    \frac{1}{2}$

    Parameters
    ----------
    bhat : np.ndarray[shape=(n, 3, 3)]

    Results
    -------
    b : np.ndarray[shape=(n, 3, 3)]

    Notes
    -----
    Adapted from Ling et al. (2016), see https://github.com/tbnn/tbnn:
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability by
    shifting values within acceptable ranges for `A[i, i] > -1/3` and
    `2|A[i, j]| < A[i, i] + A[j, j] + 2/3`. Then, if eigenvalues
    negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called
    iteratively to get convergence to a realizable state.

    """
    b = bhat.copy()
    n = b.shape[0]

    for i in range(n):
        # Scales all on-diags to retain zero trace
        min_diag = np.min(b[i, [0, 1, 2], [0, 1, 2]])
        if min_diag < -1 / 3:
            b[i, [0, 1, 2], [0, 1, 2]] *= -1 / (3 * min_diag)
        if 2 * np.abs(b[i, 0, 1]) > b[i, 0, 0] + b[i, 1, 1] + 2 / 3:
            b[i, 0, 1] = (
                (b[i, 0, 0] + b[i, 1, 1] + 2 / 3) * 0.5 * np.sign(b[i, 0, 1])
            )
            b[i, 1, 0] = (
                (b[i, 0, 0] + b[i, 1, 1] + 2 / 3) * 0.5 * np.sign(b[i, 0, 1])
            )
        if 2 * np.abs(b[i, 1, 2]) > b[i, 1, 1] + b[i, 2, 2] + 2 / 3:
            b[i, 1, 2] = (
                (b[i, 1, 1] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 1, 2])
            )
            b[i, 2, 1] = (
                (b[i, 1, 1] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 1, 2])
            )
        if 2 * np.abs(b[i, 0, 2]) > b[i, 0, 0] + b[i, 2, 2] + 2 / 3:
            b[i, 0, 2] = (
                (b[i, 0, 0] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 0, 2])
            )
            b[i, 2, 0] = (
                (b[i, 0, 0] + b[i, 2, 2] + 2 / 3) * 0.5 * np.sign(b[i, 0, 2])
            )

        # Enforce positive semidefinite by pushing evals to non-negative
        A = np.zeros([3, 3])
        A[0, 0] = b[i, 0, 0]
        A[1, 1] = b[i, 1, 1]
        A[2, 2] = b[i, 2, 2]
        A[0, 1] = b[i, 0, 1]
        A[1, 0] = b[i, 0, 1]
        A[1, 2] = b[i, 1, 2]
        A[2, 1] = b[i, 1, 2]
        A[0, 2] = b[i, 0, 2]
        A[2, 0] = b[i, 0, 2]
        evals, evecs = np.linalg.eig(A)
        evals_sorted = np.sort(evals)
        evals_max = np.max(evals)
        if evals_max < (3 * np.abs(evals_sorted[1]) - evals_sorted[1]) / 2:
            evals = (
                evals
                * (3 * np.abs(evals_sorted[1]) - evals_sorted[1])
                / (2 * evals_max)
            )
            A = evecs @ np.diag(evals) @ np.linalg.inv(evecs)
            for j in range(3):
                # Equivalent to `b[i, 0, 0] = A[0, 0]` as `b[i, 0, 1]`
                # and `b[i, 0, 2]` get overwritten below
                b[i, 0, j] = A[j, j]
            b[i, 0, 1] = A[0, 1]
            b[i, 1, 2] = A[1, 2]
            b[i, 0, 2] = A[0, 2]
            b[i, 1, 0] = A[0, 1]
            b[i, 2, 1] = A[1, 2]
            b[i, 2, 0] = A[0, 2]
        evals_sorted = np.sort(evals)
        evals_max = np.max(evals)
        if evals_max > 1 / 3 - evals_sorted[1]:
            evals = evals * (1 / 3 - evals_sorted[1]) / evals_max
            A = evecs @ np.diag(evals) @ np.linalg.inv(evecs)
            for j in range(3):
                # Equivalent to `b[i, 0, 0] = A[0, 0]` as `b[i, 0, 1]`
                # and `b[i, 0, 2]` get overwritten below
                b[i, 0, j] = A[j, j]
            b[i, 0, 1] = A[0, 1]
            b[i, 1, 2] = A[1, 2]
            b[i, 0, 2] = A[0, 2]
            b[i, 1, 0] = A[0, 1]
            b[i, 2, 1] = A[1, 2]
            b[i, 2, 0] = A[0, 2]

    return b


if __name__ == "__main__":
    pass
