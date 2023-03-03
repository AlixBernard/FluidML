import pytest
from pathlib import Path

import numpy as np

from fluidml.models import TBDT, TBRF


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
        "random_state": 7,
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
        "random_state": 7,
        "_n_rng_calls": 2,
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
                            "split_i": 0,
                            "split_v": 8.5,
                            "g": [0.42212070866550117, 0.08259707734490784],
                            "n_samples": 5,
                            "RMSE": 8.619297685864288,
                        },
                    },
                    "R1": {
                        "tag": "R1",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [4.932820265944352, -3.9340978715479604],
                            "n_samples": 1,
                            "RMSE": 0.009901737935175048,
                        },
                    },
                    "R0": {
                        "tag": "R0",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [0.4305613913475792, 0.07795430138623327],
                            "n_samples": 4,
                            "RMSE": 7.679718166179676,
                        },
                    },
                },
                "max_depth": 1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "gamma": 0.001,
                "optim_threshold": -1,
                "random_state": 944904,
                "_n_rng_calls": 2,
            },
            {
                "name": "TBRF-1_TBDT-2",
                "nodes": {
                    "R": {
                        "tag": "R",
                        "data": {
                            "split_i": 1,
                            "split_v": 7.5,
                            "g": [0.4266641833926959, 0.077322227959428],
                            "n_samples": 5,
                            "RMSE": 8.625658825933307,
                        },
                    },
                    "R1": {
                        "tag": "R1",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [4.932820266100461, -3.934097871686912],
                            "n_samples": 3,
                            "RMSE": 0.01715031295666816,
                        },
                    },
                    "R0": {
                        "tag": "R0",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [0.4311362986261399, 0.07894395445619434],
                            "n_samples": 2,
                            "RMSE": 5.4217304702076285,
                        },
                    },
                },
                "max_depth": 1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "gamma": 0.001,
                "optim_threshold": -1,
                "random_state": 625095,
                "_n_rng_calls": 2,
            },
        ],
    }
    return tbrf_dict


class TestTBRF:
    def test_to_dict(self, tbrf1, features, targets, tb, tbrf1_as_dict):
        tbrf1.fit(features, targets, tb)
        import json

        with open("tmp1-tbrf.json", "w") as file:
            json.dump(tbrf1.to_dict(), file, indent=4)
        with open("tmp2-tbrf.json", "w") as file:
            json.dump(tbrf1_as_dict, file, indent=4)
        assert tbrf1.to_dict() == tbrf1_as_dict

    def test_from_dict(self, tbrf1, features, targets, tb, tbrf1_as_dict):
        tbrf1.fit(features, targets, tb)
        print(">>>>>>>>>>>>>>>>>>")
        tbrf2 = TBRF.from_dict(tbrf1_as_dict)
        tbrf1._rng, tbrf2._rng = None, None
        for tbdt1, tbdt2 in zip(tbrf1.trees, tbrf2.trees):
            tbdt1._rng, tbdt2._rng = None, None
        assert tbrf2 == tbrf1
