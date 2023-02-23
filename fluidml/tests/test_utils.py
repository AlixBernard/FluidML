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
from pathlib import Path

from fluidml import utils

# Local packages

__all__ = [
    "test_get_SR",
    "test_get_Ak",
    "test_get_invariants_FS1",
    "test_get_invariants_FS2",
    "test_get_invariants_FS3",
]


FIXTURES_PATH = Path(__file__).parents[1] / "TBRF_new/fluidml/fixtures"
D = 3
X = np.arange(D**2).reshape(1, D, D)
SCALE_FACTORS = np.array([D])

def test_get_SR():
    S, R = utils.get_SR(X)
    assert np.all(S[0] == 0.5 * (X[0] + X[0].T))
    assert np.all(R[0] == 0.5 * (X[0] - X[0].T))
    
    S, R = utils.get_SR(X, scale_factors=SCALE_FACTORS)
    assert np.all(S[0] == 0.5 * (X[0] + X[0].T) * D)
    assert np.all(R[0] == 0.5 * (X[0] - X[0].T) * D)

def test_get_Ak():
    Ak = utils.get_Ak(X, scale_factors=SCALE_FACTORS)
    assert np.all(Ak[0] == -np.cross(np.eye(D), X[0]) * SCALE_FACTORS[0])

def test_get_invariants_FS1():
    ...

def test_get_invariants_FS2():
    ...

def test_get_invariants_FS3():
    ...
