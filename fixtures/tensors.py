#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-27 09:22:06
# @Last modified by: AlixBernard
# @Last modified time: 2022-07-07 12:10:27

""" Description. """

# Built-in packages

# Third party packages
import numpy as np

# Local packages

__all__ = [
    "graU_1",
    "gradU_2",
    "S_1",
    "S_2",
    "R_1",
    "R_2",
    "Ak_1",
    "Ak_2",
    "TB10_1",
    "TB10_2",
]


gradU_1 = np.arange(3).reshape(-1, 3)
gradU_2 = (np.arange(3).reshape(-1, 3) - 1) * 2

S_1 = np.arange(9).reshape(-1, 3, 3)
S_2 = np.arange(9).reshape(-1, 3, 3) - 3.5

R_1 = np.arange(9).reshape(-1, 3, 3)
R_2 = np.arange(9).reshape(-1, 3, 3) - 3.5

Ak_1 = np.arange(9).reshape(-1, 3, 3)
Ak_2 = np.arange(9).reshape(-1, 3, 3) - 3.5

TB10_1 = np.arange(9).reshape(-1, 3, 3)
TB10_2 = np.arange(9).reshape(-1, 3, 3) - 3.5


if __name__ == "__main__":
    pass
