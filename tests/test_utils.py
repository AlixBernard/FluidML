#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-01-05 14:24:02
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-27 14:37:23

"""Tests of functions from the module tbrf.src.my_helpers

"""

# Built-in packages

# Third party packages
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pathlib import Path

from fluidml.utils import get_S, get_R, get_Ak, get_Ap, get_TB10

# Local packages

__all__ = [
    "test_get_S",
    "test_get_R",
    "test_get_Ak",
    "test_get_Ap",
    "test_get_TB10",
    "test_get_invariants_FS1",
    "test_get_invariants_FS2",
    "test_get_invariants_FS3",
]


@pytest.fixture
def data1() -> dict[str, np.ndarray]:
    n = 5
    data = {
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

    return data


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


def test_get_invariants_FS1():
    ...


def test_get_invariants_FS2():
    ...


def test_get_invariants_FS3():
    ...
