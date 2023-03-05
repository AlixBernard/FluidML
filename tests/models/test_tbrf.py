import pytest
from pathlib import Path

import numpy as np

from fluidml.models import TBDT, TBRF


@pytest.fixture
def seed1():
    return 47


@pytest.fixture
def features():
    n, p = 5, 4
    return np.arange(n * p).reshape(n, p) - 5.5


@pytest.fixture
def targets():
    n = 5
    return np.arange(n * 9).reshape(n, 9) - 2.7


@pytest.fixture
def tb():
    n, m = 5, 2
    return np.arange(n * m * 9).reshape(n, m, 9) - 3.2


@pytest.fixture
def tbrf1():
    tbdt_kwargs = {
        "max_depth": 1,
        "max_features": "sqrt",
        "gamma": 1e-3,
        "optim_threshold": -1,
    }
    tbrf_kwargs = {
        "name": "TBRF-1",
        "n_estimators": 2,
        "bootstrap": True,
        "max_samples": None,
        "tbdt_kwargs": tbdt_kwargs,
    }
    return TBRF(**tbrf_kwargs)


@pytest.fixture
def tbrf1_as_dict():
    tbrf_dict = {
        "name": "TBRF-1",
        "n_estimators": 2,
        "bootstrap": True,
        "max_samples": None,
        "tbdt_kwargs": {
            "max_depth": 1,
            "max_features": "sqrt",
            "gamma": 0.001,
            "optim_threshold": -1,
        },
        "trees": [
            {
                "name": "TBRF-1_TBDT-1",
                "nodes": {
                    "R": {
                        "tag": "R",
                        "data": {
                            "split_i": 3,
                            "split_v": 3.5,
                            "g": [0.440899843950395, 0.06655163817786985],
                            "n_samples": 5,
                            "RMSE": 8.595475475347694,
                        },
                    },
                    "R1": {
                        "tag": "R1",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [0.5061079087239555, 0.008860387786607164],
                            "n_samples": 3,
                            "RMSE": 6.607018588629073,
                        },
                    },
                    "R0": {
                        "tag": "R0",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [1.9440298074400952, -0.9441627598264528],
                            "n_samples": 2,
                            "RMSE": 0.0014642560046348896,
                        },
                    },
                },
                "max_depth": 1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "gamma": 0.001,
                "optim_threshold": -1,
            },
            {
                "name": "TBRF-1_TBDT-2",
                "nodes": {
                    "R": {
                        "tag": "R",
                        "data": {
                            "split_i": 0,
                            "split_v": 8.5,
                            "g": [0.5326875959492602, -0.014969567392243118],
                            "n_samples": 5,
                            "RMSE": 8.505406145860315,
                        },
                    },
                    "R1": {
                        "tag": "R1",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [4.932820265944352, -3.9340978715479604],
                            "n_samples": 2,
                            "RMSE": 0.014003172078988718,
                        },
                    },
                    "R0": {
                        "tag": "R0",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [0.6720814279079177, -0.1297156363871484],
                            "n_samples": 3,
                            "RMSE": 6.417666303834766,
                        },
                    },
                },
                "max_depth": 1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "gamma": 0.001,
                "optim_threshold": -1,
            },
        ],
    }
    return tbrf_dict


class TestTBRF:
    def test_to_dict(self, tbrf1, features, targets, tb, tbrf1_as_dict, seed1):
        tbrf1.fit(features, targets, tb, seed=seed1)
        import json

        with open("tmp1-tbrf.json", "w") as file:
            json.dump(tbrf1.to_dict(), file, indent=4)
        with open("tmp2-tbrf.json", "w") as file:
            json.dump(tbrf1_as_dict, file, indent=4)
        assert tbrf1.to_dict() == tbrf1_as_dict

    def test_from_dict(
        self, tbrf1, features, targets, tb, tbrf1_as_dict, seed1
    ):
        tbrf1.fit(features, targets, tb, seed=seed1)
        tbrf2 = TBRF.from_dict(tbrf1_as_dict)
        for tbdt1, tbdt2 in zip(tbrf1.trees, tbrf2.trees):
            assert tbrf2 == tbrf1
