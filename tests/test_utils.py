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

from fluidml.utils import (
    # Tested
    enforce_realizability,
    get_Ak,
    get_Ap,
    get_b_BM,
    get_k,
    get_R,
    get_S,
    get_tau_BM,
    get_TB10,
    # Not tested
    make_realizable,  # Used to test `enforce_realizability`
)

# Local packages

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


@pytest.fixture
def bhat1() -> np.ndarray:
    n = 5
    x = np.arange(n * 3 * 3).reshape(n, 3, 3) ** 0.2 - 3
    return x**2 - x - 2


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


def test_enforce_realizability(bhat1):
    bhat1_rlz = enforce_realizability(bhat1)
    bhat1_rlz_make = make_realizable(bhat1.reshape(-1, 9)).reshape(-1, 3, 3)
    for b1, b2 in zip(bhat1_rlz, bhat1_rlz_make):
        print(repr(b1), "\n", repr(b2), end="\n=============\n", sep="")
    assert_array_almost_equal(bhat1_rlz, bhat1_rlz_make)
    # expected_bhat1_rlz = np.array(
    #     [
    #         [-0.74, -0.29, -0.24, -0.29, 2.50, -0.16, -0.24, -0.16, 1.68],
    #         [1.54, 1.41, 1.30, 1.19, 1.09, 1.00, 0.92, 0.84, 0.76],
    #         [0.69, 0.63, 0.57, 0.51, 0.45, 0.39, 0.34, 0.29, 0.25],
    #         [0.20, 0.16, 0.11, 0.07, 0.03, 0.0, -0.03, -0.07, -0.10],
    #         [-0.14, -0.17, -0.20, -0.23, -0.26, -0.29, -0.32, -0.35, -0.37],
    #     ]
    # )
    # expected_bhat1_rlz = np.array(
    #     [
    #         [
    #             -0.7418186529052948,
    #             -0.29672746116211784,
    #             -0.24321410458313758,
    #             -0.29672746116211784,
    #             2.504545751181989,
    #             -0.1665788137598562,
    #             -0.24321410458313758,
    #             -0.1665788137598562,
    #             1.687380744421283,
    #         ],
    #         [
    #             1.5453056678731736,
    #             1.4176340842817856,
    #             1.3017387718636266,
    #             1.1956572704206465,
    #             1.097883869545413,
    #             1.007237334521089,
    #             0.9227734459504253,
    #             0.8437252468750582,
    #             0.7694610667715134,
    #         ],
    #         [
    #             0.6994543173844132,
    #             0.6332612999156724,
    #             0.5705045961574253,
    #             0.5108604344676202,
    #             0.454048938594954,
    #             0.399826502633287,
    #             0.34797975780337875,
    #             0.2983207473773275,
    #             0.25068302996976444,
    #         ],
    #         [
    #             0.20491850432421188,
    #             0.16089480066710937,
    #             0.11849312123693867,
    #             0.07760644007244988,
    #             0.03813799249909078,
    #             0.0,
    #             -0.03688741230321835,
    #             -0.07259713951840174,
    #             -0.10719588936152702,
    #         ],
    #         [
    #             -0.14074486492505978,
    #             -0.17330035557581258,
    #             -0.20491425045544354,
    #             -0.23563448646079177,
    #             -0.2655054405047128,
    #             -0.29456827418687626,
    #             -0.32286123765106023,
    #             -0.35041993830424456,
    #             -0.3772775791716598,
    #         ],
    #     ]
    # )
    # assert_array_almost_equal(bhat1_rlz, expected_bhat1_rlz)
