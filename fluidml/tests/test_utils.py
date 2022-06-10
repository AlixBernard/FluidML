#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-01-05 14:24:02
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-10 13:42:52

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


FICTURES_PATH = Path.home() / "TBRF_new/fluidml/fixtures"
N = 3
X = np.arange(N**2).reshape(1, N, N)
SCALE_FACTORS = np.array([N])

def test_get_SR():
    S, R = utils.get_SR(X)
    assert np.all(S[0] == 0.5 * (X[0] + X[0].T))
    assert np.all(R[0] == 0.5 * (X[0] - X[0].T))
    
    S, R = utils.get_SR(X, scale_factors=SCALE_FACTORS)
    assert np.all(S[0] == 0.5 * (X[0] + X[0].T) * N)
    assert np.all(R[0] == 0.5 * (X[0] - X[0].T) * N)

def test_get_Ak():
    Ak = utils.get_Ak(X, scale_factors=SCALE_FACTORS)
    assert np.all(Ak[0] == -np.cross(np.eye(N), X[0]) * SCALE_FACTORS[0])

def test_get_invariants_FS1():
    ...

def test_get_invariants_FS2():
    ...

def test_get_invariants_FS3():
    ...
