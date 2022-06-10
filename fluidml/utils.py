#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-01-29 16:30:32
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-10 13:45:37

"""Utilities for the tbrf package.

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis
    - `s` is the number of TBDTs in the TBRF

"""

# Built-in packages

# Third party packages
import numpy as np
from numpy.random import default_rng

# Local packages

__all__ = [
    "jsonify",
    "random_sampling",
    "get_SR",
    "get_Ak",
    "get_TB10",
    "get_invariants_FS1",
    "get_invariants_FS2",
    "get_invariants_FS3",
]


def jsonify(x):
    """Convert a variable of numpy type such as np.int, np.float, or
    np.ndarray into a serializable type.

    Parameters
    ----------
    x : Any
        The variable to convert.

    Returns
    -------
    y : Any
        The converted variable.

    """
    if isinstance(x, np.int64):
        y = int(x)
    elif isinstance(x, np.float64):
        y = float(x)
    elif isinstance(x, np.ndarray):
        y = jsonify(x.tolist())
    elif isinstance(x, list):
        y = [jsonify(e) for e in x]
    elif isinstance(x, dict):
        y = {jsonify(k): jsonify(v) for k, v in x.items()}
    else:
        y = x
    return y

def random_sampling(
    *args: np.ndarray,
    size: int | float = 1.0,
    replace: bool = False,
    seed: int | None = None,
) -> list:
    """Take a random sample of each argument according to `frac` and
    `replace`.

    Parameters
    ----------
    *args : np.ndarray
        The list of argument to sample where each argument is an
        `np.ndarray` with shape `(n, m)` where `n` is the number of
        points and `m` the number of features.
    size : int | float
        The number of samples or fraction of the total sample size.
    replace : bool
        Whether to replace while sampling.
    seed : int | None

    Returns
    -------
    args_sampled : list[np.ndarray]
        The samples from the initial `args`.

    """
    n = len(args[0])
    if isinstance(size, float):
        size = round(size * n)

    rng = default_rng(seed)
    idx = rng.choice(n, size, replace=replace)
    args_sampled = [a[idx] for a in args]

    return args_sampled

def get_SR(
    gradU: np.ndarray, scale_factors: np.ndarray | None = None
) -> tuple:
    """Compute mean strain rate and mean rotation rate tensors.

    Parameters
    ----------
    gradU : np.ndarray
        Gradient of the velocity U with shape `(n, 3, 3)`.
    scale_factors : np.ndarray
        Scale factors with shape `(n,)`, recommanded to use `k/epsilon`.
        Default is None.

    Returns
    -------
    S : np.ndarray
        Mean strain rate tensor with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensor with shape `(n, 3, 3)`.

    """
    n = len(gradU)
    if scale_factors is None:
        scale_factors = np.ones(n)

    S = np.zeros([n, 3, 3])
    R = np.zeros([n, 3, 3])
    for i in range(n):
        S[i] = 0.5 * (gradU[i] + gradU[i].T) * scale_factors[i]
        R[i] = 0.5 * (gradU[i] - gradU[i].T) * scale_factors[i]

    return S, R


def get_Ak(
    gradk: np.ndarray,
    scale_factors: np.ndarray | None = None,
) -> np.ndarray:
    """Compute turbulent kinetic energy gradient antisymmetric tensor.

    Parameters
    ----------
    gradk : np.ndarray
        Gradient of the turbulent kinetic energy with shape `(n, 3)`.
    scale_factors : np.ndarray
        Scale factors with shape `(n,)`, recommanded to use `k/epsilon`.
        Default is None.

    Returns
    -------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensor with shape
        `(n, 3, 3)`.

    """
    n = len(gradk)
    if scale_factors is None:
        scale_factors = np.ones(n)

    Ak = np.zeros([n, 3, 3])
    for i in range(n):
        Ak[i] = -np.cross(np.eye(3), gradk[i]) * scale_factors[i]

    return Ak


def get_TB10(
    S: np.ndarray, R: np.ndarray, scale_factors: np.ndarray | None = None
) -> np.ndarray:
    """Compute tensor basis composed of 10 tensors.

    Parameters
    ----------
    S : np.ndarray
        Mean strain rate tensor with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensor with shape `(n, 3, 3)`.
    scale_factors : np.ndarray
        Scale factors with shape `(10,)`, recommanded to use `k/epsilon`.
        # ? Or use `[10] + 3*[100] + 2*[1_000] + 4*[10_0000]`
        Default is None.

    Returns
    -------
    T : np.ndarray
        Tensor basis with shape `(n, 10, 3, 3)`.

    """
    n, _, _ = S.shape
    if scale_factors is None:
        scale_factors = np.ones(10)

    T = np.zeros([n, 10, 3, 3])
    for i in range(n):
        T[i, 0] = S[i]
        T[i, 1] = S[i] @ R[i] - R[i] @ S[i]
        T[i, 2] = S[i] @ S[i] - (1 / 3) * np.eye(3) * np.trace(S[i] @ S[i])
        T[i, 3] = R[i] @ R[i] - (1 / 3) * np.eye(3) * np.trace(R[i] @ R[i])
        T[i, 4] = R[i] @ (S[i] @ S[i]) - S[i] @ (S[i] @ R[i])
        T[i, 5] = (
            R[i] @ (R[i] @ S[i])
            + S[i] @ (R[i] @ R[i])
            - (2 / 3) * np.eye(3) * np.trace(S[i] @ (R[i] @ R[i]))
        )
        T[i, 6] = R[i] @ (S[i] @ (R[i] @ R[i])) - R[i] @ (
            R[i] @ (S[i] @ R[i])
        )
        T[i, 7] = S[i] @ (R[i] @ (S[i] @ S[i])) - S[i] @ (
            S[i] @ (R[i] @ S[i])
        )
        T[i, 8] = (
            R[i] @ (R[i] @ (S[i] @ S[i]))
            + S[i] @ (S[i] @ (R[i] @ R[i]))
            - (2 / 3) * np.eye(3) * np.trace(S[i] @ (S[i] @ (R[i] @ R[i])))
        )
        T[i, 9] = R[i] @ (S[i] @ (S[i] @ (R[i] @ R[i]))) - R[i] @ (
            R[i] @ (S[i] @ (S[i] @ R[i]))
        )

        for j in range(10):
            T[i, j] *= scale_factors[j]

    return T


def get_invariants_FS1(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute invariants set FS1.

    Parameters
    ----------
    S : np.ndarray
        Mean strain rate tensor with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensor with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants set FS1 with shape `(n, 6)`.

    """
    n, _, _ = S.shape

    inv = np.zeros([n, 6])
    for i in range(n):
        inv[i, 0] = np.trace(S[i] @ S[i])
        inv[i, 1] = np.trace(R[i] @ R[i])
        inv[i, 2] = np.trace(S[i] @ S[i] @ S[i])
        inv[i, 3] = np.trace(R[i] @ (R[i] @ S[i]))
        inv[i, 4] = np.trace(R[i] @ (R[i] @ (S[i] @ S[i])))
        inv[i, 5] = np.trace(R[i] @ (R[i] @ (S[i] @ (R[i] @ (S[i] @ S[i])))))

    return inv


def get_invariants_FS2(
    S: np.ndarray, R: np.ndarray, Ak: np.ndarray
) -> np.ndarray:
    """Compute invariants set FS2 + FS2_extra.
    FS2_extra is ${A_k^2 R S, A_k^2 R S^2, A^2 S R S^2}$.

    Parameters
    ----------
    S : np.ndarray
        Mean strain rate tensor with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensor with shape `(n, 3, 3)`.
    Ak : np.ndarray | None
        Turbulent kinetic energy antisymmetric tensor with shape
        `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants set FS2 + FS2_extra with shape `(n, 13)`.
        FS2_extra is ${A_k^2 R S, A_k^2 R S^2, A^2 S R S^2}$.

    """
    n, _, _ = S.shape

    inv = np.zeros([n, 13])
    for i in range(n):
        inv[i, 0] = np.trace(Ak[i] @ Ak[i])
        inv[i, 1] = np.trace(Ak[i] @ (Ak[i] @ S[i]))
        inv[i, 2] = np.trace(Ak[i] @ (Ak[i] @ (S[i] @ S[i])))
        inv[i, 3] = np.trace(
            Ak[i] @ (Ak[i] @ (S[i] @ (Ak[i] @ (S[i] @ S[i]))))
        )
        inv[i, 4] = np.trace(R[i] @ Ak[i])
        inv[i, 5] = np.trace(R[i] @ (Ak[i] @ S[i]))
        inv[i, 6] = np.trace(R[i] @ (Ak[i] @ (S[i] @ S[i])))
        inv[i, 7] = np.trace(R[i] @ (R[i] @ (Ak[i] @ S[i])))
        inv[i, 8] = np.trace(Ak[i] @ (Ak[i] @ (R[i] @ S[i])))  # FS2_extra_1
        inv[i, 9] = np.trace(R[i] @ (R[i] @ (Ak[i] @ (S[i] @ S[i]))))  # FS2_9
        inv[i, 10] = np.trace(
            Ak[i] @ (Ak[i] @ (R[i] @ (S[i] @ S[i])))
        )  # FS2_extra_2
        inv[i, 11] = np.trace(
            R[i] @ (R[i] @ (S[i] @ (Ak[i] @ (S[i] @ S[i]))))
        )  # FS2_10
        inv[i, 12] = np.trace(
            Ak[i] @ (Ak[i] @ (S[i] @ (R[i] @ (S[i] @ S[i]))))
        )  # FS2_extra_3

    return inv


def get_invariants_FS3(data: dict) -> np.ndarray:
    """Compute invariants set FS3.

    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
            S : np.ndarray
                Symmetry tensor with shape `(n, 3, 3)`.
            R : np.ndarray
                Rotation tensor with shape `(n, 3, 3)`.
            k : np.ndarray | None
                Kinetic energy with shape `(n,)`.
            epsilon : np.ndarray | None
                Epsilon with shape `(n,)`.
            U : np.ndarray
                Velocity with shape `(n, 3)`.
            d : np.ndarray
                Distance to the wall with shape `(n,)`.
            gradp : np.ndarray
                Gradient of the pressure with shape `(n, 3)`.
            gradk : np.ndarray
                Gradient of the turbulent kinetic energy with shape
                `(n, 3)`.
            tau : np.ndarray
                Reynolds stress tensor with shape `(n, 3, 3)`.
            gradU : np.ndarray
                Gradient of the velocity U with shape `(n, 3, 3)`.
            gradU2 : np.ndarray
                Gradient of the velocity U squared with shape
                `(n, 3, 3)`.
            nu : float
                Viscosity.

    Returns
    -------
    inv : np.ndarray
        Invariants set FS3 with shape `(n, 9)`.
    """
    S, R = data["S"], data["R"]
    k, epsilon, tau, nu, d, U = (
        data["k"],
        data["epsilon"],
        data["tau"],
        data["nu"][0],
        data["d"],
        data["U"],
    )
    gradp, gradk, gradU, gradU2 = (
        data["gradp"],
        data["gradk"],
        data["gradU"],
        data["gradU2"],
    )

    n, _, _ = S.shape
    raw = np.zeros([n, 9])
    norm = np.zeros([n, 9])
    inv = np.zeros([n, 9])
    for i in range(n):

        raw[i, 0] = 0.5 * (
            np.linalg.norm(R[i]) ** 2 - np.linalg.norm(S[i]) ** 2
        )
        raw[i, 1] = k[i]
        raw[i, 2] = 1.0  # Ignore, `wang_inv[i, 2]` defined at the end
        raw[i, 3] = np.sum(U[i] * gradp[i])
        raw[i, 4] = k[i] / epsilon[i]
        raw[i, 5] = np.sqrt(np.sum(gradp[i] ** 2))
        raw[i, 6] = np.abs(np.einsum("i,j,ij", U[i], U[i], gradU[i]))  # FS3_9
        raw[i, 7] = np.sum(U[i] * gradk[i])  # FS3_7
        raw[i, 8] = np.linalg.norm(tau[i])  # FS3_8

        norm[i, 0] = np.linalg.norm(S[i]) ** 2
        norm[i, 1] = 0.5 * (U[i, 0] ** 2 + U[i, 1] ** 2 + U[i, 2] ** 2)
        norm[i, 2] = 1.0  # Ignore, `wang_inv[i, 2]` defined at the end
        norm[i, 3] = np.sqrt(
            np.sum(
                np.vstack([U[i], U[i], U[i]]).T ** 2
                * np.vstack([gradp[i], gradp[i], gradp[i]]) ** 2
            )
        )
        norm[i, 4] = 1 / np.linalg.norm(S[i])
        norm[i, 5] = 0.5 * np.sum([gradU2[j, j] for j in range(3)])
        norm[i, 6] = np.sqrt(
            np.einsum(
                "l,l,i,ij,k,kj", U[i], U[i], U[i], gradU[i], U[i], gradU[i]
            )
        )
        norm[i, 7] = np.abs(np.sum(tau[i] * S[i]))
        norm[i, 8] = k[i]

        inv[i] = raw[i] / (np.abs(raw[i]) + np.abs(norm[i]))
        inv[i, 2] = min(np.sqrt(k[i]) * d[i] / (50 * nu), 2.0)

    return inv


if __name__ == "__main__":
    pass
