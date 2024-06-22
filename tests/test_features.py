"""Tests of functions from the module `fluidml.features`."""

# Built-in packages

# Third party packages
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# Local packages
from fluidml.features import (
    get_Ak,
    get_Ap,
    get_b_BM,
    get_k,
    get_R,
    get_S,
    get_tau_BM,
    get_TB10,
)

__all__ = [
    "test_get_k",
    "test_get_S",
    "test_get_R",
    "test_get_Ak",
    "test_get_Ap",
    "test_get_TB10",
    "test_get_b_BM",
    "test_get_tau_BM",
]


@pytest.fixture
def data1() -> dict[str, np.ndarray]:
    n = 5
    data = {
        "k": np.arange(n) - 5 / 3,
        "nut": np.arange(n) ** 2 - 0.8,
    }
    data |= {
        "gradU": ((np.arange(n * 3 * 3).reshape(-1, 3, 3) - 12) / 10) ** 3,
        "gradk": ((np.arange(n * 3).reshape(-1, 3) - 30) / 10) ** 2,
        "gradp": (np.arange(n * 3).reshape(-1, 3) / 10) ** 0.5 - 0.5,
    }
    data |= {
        "S": np.array([0.5 * (gU + gU.T) for gU in data["gradU"]]),
        "R": np.array([0.5 * (gU - gU.T) for gU in data["gradU"]]),
        "Ak": np.array([-np.cross(np.eye(3), gk) for gk in data["gradk"]]),
        "Ap": np.array([-np.cross(np.eye(3), gp) for gp in data["gradp"]]),
    }
    data["tb10"] = np.array(
        [
            [
                [s],
                [s @ r - r @ s],
                [s @ s - (1 / 3) * np.eye(3) * np.trace(s @ s)],
                [r @ r - (1 / 3) * np.eye(3) * np.trace(r @ r)],
                [r @ (s @ s) - s @ (s @ r)],
                [
                    (
                        r @ (r @ s)
                        + s @ (r @ r)
                        - (2 / 3) * np.eye(3) * np.trace(s @ (r @ r))
                    )
                ],
                [r @ (s @ (r @ r)) - r @ (r @ (s @ r))],
                [s @ (r @ (s @ s)) - s @ (s @ (r @ s))],
                [
                    (
                        r @ (r @ (s @ s))
                        + s @ (s @ (r @ r))
                        - (2 / 3) * np.eye(3) * np.trace(s @ (s @ (r @ r)))
                    )
                ],
                [r @ (s @ (s @ (r @ r))) - r @ (r @ (s @ (s @ r)))],
            ]
            for s, r in zip(data["S"], data["R"])
        ]
    ).reshape(-1, 10, 3, 3)
    data |= {
        "b_BM": np.array(
            [
                -(nut / k) * s
                for k, nut, s in zip(data["k"], data["nut"], data["S"])
            ]
        ),
        "tau_BM": np.array(
            [
                (2 / 3) * k * np.eye(3) - 2 * nut * s
                for k, nut, s in zip(data["k"], data["nut"], data["S"])
            ]
        ),
    }

    return data


def test_get_k(data1):
    assert_array_almost_equal(
        get_k(data1["tau_BM"]),
        0.5 * np.array([np.trace(x) for x in data1["tau_BM"]]),
    )


def test_get_S(data1):
    assert_array_almost_equal(get_S(data1["gradU"]), data1["S"])


def test_get_R(data1):
    assert_array_almost_equal(get_R(data1["gradU"]), data1["R"])


def test_get_Ak(data1):
    assert_array_almost_equal(get_Ak(data1["gradk"]), data1["Ak"])


def test_get_Ap(data1):
    assert_array_almost_equal(get_Ap(data1["gradp"]), data1["Ap"])


def test_get_TB10(data1):
    assert_array_almost_equal(get_TB10(data1["S"], data1["R"]), data1["tb10"])


def test_get_b_BM(data1):
    assert_array_almost_equal(
        get_b_BM(data1["k"], data1["nut"], data1["S"]), data1["b_BM"]
    )


def test_get_tau_BM(data1):
    assert_array_almost_equal(
        get_tau_BM(data1["k"], data1["nut"], data1["S"]), data1["tau_BM"]
    )
