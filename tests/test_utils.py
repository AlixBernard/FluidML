"""Tests of functions from the module `fluidml.utils`."""

# Built-in packages

# Third party packages
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# Local packages
from fluidml.utils import enforce_realizability, make_realizable


__all__ = [
    "test_enforce_realizability",
]


@pytest.fixture
def bhat1() -> np.ndarray:
    n = 5
    x = np.arange(n * 3 * 3).reshape(n, 3, 3) ** 0.2 - 3
    return x**2 - x - 2


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
