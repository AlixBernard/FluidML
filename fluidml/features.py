"""Functions helping to compute features.

Glossary:
    - `n` is the total number of samples
    - `m` is the number of tensors in the tensor basis

"""

# Built-in packages

# Third party packages
import numpy as np

# Local packages


__all__ = [
    "get_k",
    "get_S",
    "get_R",
    "get_Ak",
    "get_Ap",
    "get_TB10",
    "get_invariants_FS1",
    "get_invariants_FS2",
    "get_invariants_FS3",
    "get_tau_BM",
    "get_inv1to2",
    "get_inv3to5",
    "get_inv6to14",
    "get_inv15to17",
    "get_inv18to41",
    "get_inv42",
    "get_inv43to47",
    "get_inv1to47",
]

I_3 = np.identity(3)


def get_k(tau: np.ndarray) -> np.ndarray:
    """Compute tubulent kinetic energy scalars.

    Parameters
    ----------
    tau : np.ndarray[shape=(n, 3, 3)]
        Reynolds stress tensors.

    Returns
    -------
    k : np.ndarray[shape=(n)]
        Turbulent kinetic energy scalars.

    """
    k = 0.5 * np.einsum("...jj", tau)
    return k


def get_S(gradU: np.ndarray) -> np.ndarray:
    """Compute mean strain rate tensors.

    Parameters
    ----------
    gradU : np.ndarray[shape=(n, 3, 3)]
        Velocity gradients.

    Returns
    -------
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.

    """
    S = 0.5 * (gradU + np.einsum("...ji", gradU))
    return S


def get_R(gradU: np.ndarray) -> np.ndarray:
    """Compute mean rotation rate tensors.

    Parameters
    ----------
    gradU : np.ndarray[shape=(n, 3, 3)]
        Velocity gradients.

    Returns
    -------
    R : np.ndarray[shape=(n, 3, 3)]
        Mean rotation rate tensors.

    """
    R = 0.5 * (gradU - np.einsum("...ji", gradU))
    return R


def get_Ak(gradk: np.ndarray) -> np.ndarray:
    r"""Compute turbulent kinetic energy gradient antisymmetric tensors.

    Parameters
    ----------
    gradk : np.ndarray[shape=(n, 3)]
        Turbulent kinetic energy gradients.

    Returns
    -------
    Ak : np.ndarray[shape=(n, 3, 3)]
        Turbulent kinetic energy gradient antisymmetric tensors, i.e.
        $A_k = -I \cross \nabla k$.

    Notes
    -----
    Here `eijk` denotes the Levi-Civita symbol, its use greatly reduce
    the computational time.

    """
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    u, v = gradk, I_3
    Ak = np.einsum("iuk,vk->uvi", np.einsum("ijk,uj->iuk", eijk, u), v)

    return Ak


def get_Ap(gradp: np.ndarray) -> np.ndarray:
    """Compute pressure gradient antisymmetric tensors.

    Parameters
    ----------
    gradp : np.ndarray[shape=(n, 3)]
        Pressure gradients.

    Returns
    -------
    Ap : np.ndarray[shape=(n, 3, 3)]
        Pressure gradient antisymmetric tensors.

    Notes
    -----
    Here `eijk` denotes the Levi-Civita symbol, its use greatly reduce
    the computational time.

    """
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    u, v = gradp, I_3
    Ap = np.einsum("iuk,vk->uvi", np.einsum("ijk,uj->iuk", eijk, u), v)

    return Ap


def get_TB10(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute tensor bases composed of 10 tensors to compute the
    Reynolds stress anisotropy tensor.

    Parameters
    ----------
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.
    R : np.ndarray[shape=(n, 3, 3)]
        Mean rotation rate tensors.

    Returns
    -------
    T : np.ndarray[shape=(n, 10, 3, 3)]
        Tensor bases.

    Notes
    -----
    The basis definition is taken from Pope (1975).
    The normalized versions of `S` and `R` should be used, also denoted
    respectively `Shat` and `Rhat`.

    """
    n = len(S)
    n_I_3 = np.array([I_3 for _ in range(n)])

    T = np.zeros([n, 10, 3, 3])
    T[:, 0] = S
    T[:, 1] = S @ R - R @ S
    T[:, 2] = S @ S - (1 / 3) * np.einsum(
        "...ij,...->...ij", n_I_3, np.trace(S @ S, axis1=-1, axis2=-2)
    )
    T[:, 3] = R @ R - (1 / 3) * np.einsum(
        "...ij,...->...ij", n_I_3, np.trace(R @ R, axis1=-1, axis2=-2)
    )
    T[:, 4] = R @ S @ S - S @ S @ R
    T[:, 5] = (
        R @ R @ S
        + S @ R @ R
        - (2 / 3)
        * np.einsum(
            "...ij,...->...ij", n_I_3, np.trace(S @ R @ R, axis1=-1, axis2=-2)
        )
    )
    T[:, 6] = R @ S @ R @ R - R @ R @ S @ R
    T[:, 7] = S @ R @ S @ S - S @ S @ R @ S
    T[:, 8] = (
        R @ R @ S @ S
        + S @ S @ R @ R
        - (2 / 3)
        * np.einsum(
            "...ij,...->...ij",
            n_I_3,
            np.trace(S @ S @ R @ R, axis1=-1, axis2=-2),
        )
    )
    T[:, 9] = R @ S @ S @ R @ R - R @ R @ S @ S @ R

    return T


def get_invariants_FS1(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute invariants set FS1.

    Parameters
    ----------
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.
    R : np.ndarray[shape=(n, 3, 3)]
        Mean rotation rate tensors.

    Returns
    -------
    inv : np.ndarray[shape=(n, 6)]
        Invariants set FS1.

    """
    n, _, _ = S.shape

    inv = np.zeros([n, 6])
    for i in range(n):
        inv[i, 0] = np.trace(S[i] @ S[i])
        inv[i, 1] = np.trace(R[i] @ R[i])
        inv[i, 2] = np.trace(S[i] @ S[i] @ S[i])
        inv[i, 3] = np.trace(R[i] @ R[i] @ S[i])
        inv[i, 4] = np.trace(R[i] @ R[i] @ S[i] @ S[i])
        inv[i, 5] = np.trace(R[i] @ R[i] @ S[i] @ R[i] @ S[i] @ S[i])

    return inv


def get_invariants_FS2(
    S: np.ndarray, R: np.ndarray, Ak: np.ndarray
) -> np.ndarray:
    """Compute invariants set FS2 + FS2_extra.
    FS2_extra is ${A_k^2 R S, A_k^2 R S^2, A^2 S R S^2}$.

    Parameters
    ----------
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.
    R : np.ndarray[shape=(n, 3, 3)]
        Mean rotation rate tensors.
    Ak : np.ndarray | None[shape=(n, 3, 3)]
        Turbulent kinetic energy antisymmetric tensors.

    Returns
    -------
    inv : np.ndarray[shape=(n, 13)]
        Invariants set FS2 + FS2_extra.
        FS2_extra is ${A_k^2 R S, A_k^2 R S^2, A^2 S R S^2}$.

    """
    n = len(S)

    inv = np.zeros([n, 13])
    for i in range(n):
        inv[i, 0] = np.trace(Ak[i] @ Ak[i])
        inv[i, 1] = np.trace(Ak[i] @ Ak[i] @ S[i])
        inv[i, 2] = np.trace(Ak[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 3] = np.trace(Ak[i] @ Ak[i] @ S[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 4] = np.trace(R[i] @ Ak[i])
        inv[i, 5] = np.trace(R[i] @ Ak[i] @ S[i])
        inv[i, 6] = np.trace(R[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 7] = np.trace(R[i] @ R[i] @ Ak[i] @ S[i])
        inv[i, 8] = np.trace(Ak[i] @ Ak[i] @ R[i] @ S[i])  # FS2_extra_1
        inv[i, 9] = np.trace(R[i] @ R[i] @ Ak[i] @ S[i] @ S[i])  # FS2_9
        inv[i, 10] = np.trace(
            Ak[i] @ Ak[i] @ R[i] @ S[i] @ S[i]
        )  # FS2_extra_2
        inv[i, 11] = np.trace(
            R[i] @ R[i] @ S[i] @ Ak[i] @ S[i] @ S[i]
        )  # FS2_10
        inv[i, 12] = np.trace(
            Ak[i] @ Ak[i] @ S[i] @ R[i] @ S[i] @ S[i]
        )  # FS2_extra_3

    return inv


def get_invariants_FS3(data: dict) -> np.ndarray:
    """Compute invariants set FS3.

    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
            epsilon : np.ndarray[shape=(n,)] | None
            gradk : np.ndarray[shape=(n, 3)]
                Gradient of the turbulent kinetic energy.
            gradp : np.ndarray[shape=(n, 3)]
                Pressure gradients.
            gradU : np.ndarray[shape=(n, 3, 3)]
                Velocity gradients.
            gradU2 : np.ndarray[shape=(n, 3, 3)]
                Square of the velocity gradients.
            k : np.ndarray[shape=(n,)] | None
                Kinetic energy.
            nu : float
                Viscosity.
            R : np.ndarray[shape=(n, 3, 3)]
                Mean rotation rate tensors.
            S : np.ndarray[shape=(n, 3, 3)]
                Mean strain rate tensors.
            tau : np.ndarray[shape=(n, 3, 3)]
                Reynolds stress tensors.
            U : np.ndarray[shape=(n, 3)]
                Velocity vectors.
            wallDistance : np.ndarray[shape=(n,)]
                Distances to the wall.

    Returns
    -------
    inv : np.ndarray[shape=(n, 9)]
        Invariants set FS3.

    """
    epsilon = data["epsilon"]
    gradk = data["gradk"]
    gradp = data["gradp"]
    gradU = data["gradU"]
    gradU2 = data["gradU2"]
    k = data["k"]
    nu = data["nu"]
    S = data["S"]
    R = data["R"]
    Shat = data["Shat"]
    Rhat = data["Rhat"]
    tau = data["tau"]
    U = data["U"]
    wallDistance = data["wallDistance"]

    n = len(S)
    raw = np.zeros((n, 9))
    norm = np.zeros((n, 9))
    div = np.ones((n, 9))
    non_zero_div = np.ones((n, 9))
    inv = np.zeros((n, 9))
    for i in range(n):
        raw[i, 0] = 0.5 * (
            np.linalg.norm(Rhat[i]) ** 2 - np.linalg.norm(Shat[i]) ** 2
        )
        raw[i, 1] = k[i]
        raw[i, 2] = 1.0  # Ignore, `wang_inv[i, 2]` defined at the end
        raw[i, 3] = np.sum(U[i] * gradp[i])
        raw[i, 4] = k[i] / epsilon[i]
        raw[i, 5] = np.sqrt(np.sum(gradp[i] ** 2))
        raw[i, 6] = np.abs(np.einsum("i,j,ij", U[i], U[i], gradU[i]))  # FS3_9
        raw[i, 7] = np.sum(U[i] * gradk[i])  # FS3_7
        raw[i, 8] = np.linalg.norm(tau[i])  # FS3_8

        norm[i, 0] = np.linalg.norm(Shat[i]) ** 2
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

        div[i] = np.abs(raw[i]) + np.abs(norm[i])
        non_zero_div[i] = np.where(div[i] == 0, 1e-30, div[i])
        inv[i] = raw[i] / non_zero_div[i]
        inv[i, 2] = min(np.sqrt(k[i]) * wallDistance[i] / (50 * nu[i]), 2.0)

    return inv


def get_b_BM(k: np.ndarray, nut: np.ndarray, S: np.ndarray) -> np.ndarray:
    r"""Compute normalized Reynolds-stress anisotropic tensors from the
    Boussinesq model: $b = - \frac{\nu_t}{k} S$.

    Parameters
    ----------
    k : np.ndarray[shape=(n,)]
        Turbulent kinetic energies.
    nut : np.ndarray[shape=(n,)]
        Turbulent viscosities.
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.

    Returns
    -------
    b_BM : np.ndarray[shape=(n, 3, 3)]
        Normalized Reynolds-stress anisotropic tensors from the
        Boussinesq model.

    """
    b_BM = np.einsum("i,ijk->ijk", -nut / k, S)
    return b_BM


def get_tau_BM(k: np.ndarray, nut: np.ndarray, S: np.ndarray) -> np.ndarray:
    r"""Compute Reynolds-stress tensors from the Boussinesq model:
    $\tau = \frac{2}{3} k I - 2 \nu_t S$.

    Parameters
    ----------
    k : np.ndarray[shape=(n,)]
        Turbulent kinetic energies.
    nut : np.ndarray[shape=(n,)]
        Turbulent viscosities.
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.

    Returns
    -------
    tau_BM : np.ndarray[shape=(n, 3, 3)]
        Reynolds-stress tensors from the Boussinesq model.

    """
    tau_BM = (2 / 3) * np.einsum("i,jk->ijk", k, I_3) - 2 * np.einsum(
        "i,ijk->ijk", nut, S
    )
    return tau_BM


def get_inv1to2(S: np.ndarray) -> np.ndarray:
    """Compute the invariants 1 to 2 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    S : np.ndarray
        Mean strain rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n, 2)`.

    """
    n = len(S)

    inv = np.zeros([n, 2])
    for i in range(n):
        inv[i, 0] = np.trace(S[i] @ S[i])
        inv[i, 1] = np.trace(S[i] @ S[i] @ S[i])

    return inv


def get_inv3to5(Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute the invariants 3 to 5 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensors with shape
        `(n, 3, 3)`.
    Ap : np.ndarray
        Pressure gradient antisymmetric tensors with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n, 3)`.

    """
    n = len(Ak)

    inv = np.zeros([n, 3])
    for i in range(n):
        inv[i, 0] = np.trace(R[i] @ R[i])
        inv[i, 1] = np.trace(Ap[i] @ Ap[i])
        inv[i, 2] = np.trace(Ak[i] @ Ak[i])

    return inv


def get_inv6to14(
    Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Compute the invariants 6 to 14 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensors with shape
        `(n, 3, 3)`.
    Ap : np.ndarray
        Pressure gradient antisymmetric tensors with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensors with shape `(n, 3, 3)`.
    S : np.ndarray
        Mean strain rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n, 9)`.

    """
    n = len(Ak)

    inv = np.zeros([n, 9])
    for i in range(n):
        inv[i, 0] = np.trace(R[i] @ R[i] @ S[i])
        inv[i, 1] = np.trace(R[i] @ R[i] @ S[i] @ S[i])
        inv[i, 2] = np.trace(R[i] @ R[i] @ S[i] @ R[i] @ S[i] @ S[i])
        inv[i, 3] = np.trace(Ap[i] @ Ap[i] @ S[i])
        inv[i, 4] = np.trace(Ap[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 5] = np.trace(Ap[i] @ Ap[i] @ S[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 6] = np.trace(Ak[i] @ Ak[i] @ S[i])
        inv[i, 7] = np.trace(Ak[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 8] = np.trace(Ak[i] @ Ak[i] @ S[i] @ Ak[i] @ S[i] @ S[i])

    return inv


def get_inv15to17(Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute the invariants 15 to 17 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensors with shape
        `(n, 3, 3)`.
    Ap : np.ndarray
        Pressure gradient antisymmetric tensors with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n, 3)`.

    """
    n = len(Ak)

    inv = np.zeros([n, 3])
    for i in range(n):
        inv[i, 0] = np.trace(R[i] @ Ap[i])
        inv[i, 1] = np.trace(Ap[i] @ Ak[i])
        inv[i, 2] = np.trace(R[i] @ Ak[i])

    return inv


def get_inv18to41(
    Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Compute the invariants 18 to 41 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensors with shape
        `(n, 3, 3)`.
    Ap : np.ndarray
        Pressure gradient antisymmetric tensors with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensors with shape `(n, 3, 3)`.
    S : np.ndarray
        Mean strain rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n, 24)`.

    """
    n = len(Ak)

    inv = np.zeros([n, 24])
    for i in range(n):
        inv[i, 0] = np.trace(R[i] @ Ap[i] @ S[i])
        inv[i, 1] = np.trace(R[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 2] = np.trace(R[i] @ R[i] @ Ap[i] @ S[i])
        inv[i, 3] = np.trace(Ap[i] @ Ap[i] @ R[i] @ S[i])
        inv[i, 4] = np.trace(R[i] @ R[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 5] = np.trace(Ap[i] @ Ap[i] @ R[i] @ S[i] @ S[i])
        inv[i, 6] = np.trace(R[i] @ R[i] @ S[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 7] = np.trace(Ap[i] @ Ap[i] @ S[i] @ R[i] @ S[i] @ S[i])

        inv[i, 8] = np.trace(R[i] @ Ak[i] @ S[i])
        inv[i, 9] = np.trace(R[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 10] = np.trace(R[i] @ R[i] @ Ak[i] @ S[i])
        inv[i, 11] = np.trace(Ak[i] @ Ak[i] @ R[i] @ S[i])
        inv[i, 12] = np.trace(R[i] @ R[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 13] = np.trace(Ak[i] @ Ak[i] @ R[i] @ S[i] @ S[i])
        inv[i, 14] = np.trace(R[i] @ R[i] @ S[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 15] = np.trace(Ak[i] @ Ak[i] @ S[i] @ R[i] @ S[i] @ S[i])

        inv[i, 16] = np.trace(Ap[i] @ Ak[i] @ S[i])
        inv[i, 17] = np.trace(Ap[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 18] = np.trace(Ap[i] @ Ap[i] @ Ak[i] @ S[i])
        inv[i, 19] = np.trace(Ak[i] @ Ak[i] @ Ap[i] @ S[i])
        inv[i, 20] = np.trace(Ap[i] @ Ap[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 21] = np.trace(Ak[i] @ Ak[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 22] = np.trace(Ap[i] @ Ap[i] @ S[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 23] = np.trace(Ak[i] @ Ak[i] @ S[i] @ Ap[i] @ S[i] @ S[i])

    return inv


def get_inv42(Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute the invariant 42 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensors with shape
        `(n, 3, 3)`.
    Ap : np.ndarray
        Pressure gradient antisymmetric tensors with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensors with shape `(n, 3, 3)`.
    S : np.ndarray
        Mean strain rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n,)`.

    """
    n = len(Ak)

    inv = np.zeros([n])
    for i in range(n):
        inv[i] = np.trace(R[i] @ Ap[i] @ Ak[i])

    return inv


def get_inv43to47(
    Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Compute the invariants 43 to 47 from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray
        Turbulent kinetic energy antisymmetric tensors with shape
        `(n, 3, 3)`.
    Ap : np.ndarray
        Pressure gradient antisymmetric tensors with shape `(n, 3, 3)`.
    R : np.ndarray
        Mean rotation rate tensors with shape `(n, 3, 3)`.
    S : np.ndarray
        Mean strain rate tensors with shape `(n, 3, 3)`.

    Returns
    -------
    inv : np.ndarray
        Invariants with shape `(n, 5)`.

    """
    n = len(Ak)

    inv = np.zeros([n, 5])
    for i in range(n):
        inv[i, 0] = np.trace(R[i] @ Ap[i] @ Ak[i] @ S[i])
        inv[i, 1] = np.trace(R[i] @ Ak[i] @ Ap[i] @ S[i])
        inv[i, 2] = np.trace(R[i] @ Ap[i] @ Ak[i] @ S[i] @ S[i])
        inv[i, 3] = np.trace(R[i] @ Ak[i] @ Ap[i] @ S[i] @ S[i])
        inv[i, 4] = np.trace(R[i] @ Ap[i] @ S[i] @ Ak[i] @ S[i] @ S[i])

    return inv


def get_inv1to47(
    Ak: np.ndarray, Ap: np.ndarray, R: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Compute the 47 invariants from Wu et al. (2018). To get the
    normalized invariants the input should be normalized before.

    Parameters
    ----------
    Ak : np.ndarray[shape=(n, 3, 3)]
        Turbulent kinetic energy antisymmetric tensors.
    Ap : np.ndarray[shape=(n, 3, 3)]
        Pressure gradient antisymmetric tensors.
    R : np.ndarray[shape=(n, 3, 3)]
        Mean rotation rate tensors.
    S : np.ndarray[shape=(n, 3, 3)]
        Mean strain rate tensors.

    Returns
    -------
    inv : np.ndarray[shape=(n, 47)]
        Invariants.

    """
    inv = np.hstack(
        [
            get_inv1to2(S),
            get_inv3to5(Ak, Ap, R),
            get_inv6to14(Ak, Ap, R, S),
            get_inv15to17(Ak, Ap, R),
            get_inv18to41(Ak, Ap, R, S),
            get_inv42(Ak, Ap, R).reshape(-1, 1),
            get_inv43to47(Ak, Ap, R, S),
        ]
    )

    return inv


if __name__ == "__main__":
    pass
