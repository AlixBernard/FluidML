import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from fluidml.models import TBRF


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
def b_prediction1():
    return np.array(
        [
            [-7.30, -6.52, -5.75, -4.98, -4.21, -3.44, -2.67, -1.90, -1.13],
            [6.58, 7.35, 8.12, 8.89, 9.66, 10.43, 11.20, 11.97, 12.74],
            [16.79, 17.32, 17.85, 18.38, 18.91, 19.43, 19.96, 20.49, 21.02],
            [26.31, 26.84, 27.36, 27.89, 28.42, 28.95, 29.48, 30.01, 30.54],
            [34.40, 35.16, 35.92, 36.67, 37.43, 38.19, 38.94, 39.70, 40.46],
        ]
    )


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
                "max_depth": 1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "gamma": 0.001,
                "optim_threshold": -1,
                "nodes": {
                    "R": {
                        "tag": "R",
                        "data": {
                            "split_i": 3,
                            "split_v": 3.5,
                            "g": [0.440899843950395, 0.06655163817786985],
                            "n_samples": 5,
                            "RMSE": 1.2813378307873173,
                        },
                    },
                    "R0": {
                        "tag": "R0",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [1.9440298074400952, -0.9441627598264528],
                            "n_samples": 2,
                            "RMSE": 0.0003451284500901504,
                        },
                    },
                    "R1": {
                        "tag": "R1",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [0.5061079087239555, 0.008860387786607164],
                            "n_samples": 3,
                            "RMSE": 1.2715213202286189,
                        },
                    },
                },
            },
            {
                "name": "TBRF-1_TBDT-2",
                "max_depth": 1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "gamma": 0.001,
                "optim_threshold": -1,
                "nodes": {
                    "R": {
                        "tag": "R",
                        "data": {
                            "split_i": 0,
                            "split_v": 8.5,
                            "g": [0.5326875959492602, -0.014969567392243118],
                            "n_samples": 5,
                            "RMSE": 1.2679110878925437,
                        },
                    },
                    "R0": {
                        "tag": "R0",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [0.6720814279079177, -0.1297156363871484],
                            "n_samples": 3,
                            "RMSE": 1.2350804560293975,
                        },
                    },
                    "R1": {
                        "tag": "R1",
                        "data": {
                            "split_i": None,
                            "split_v": None,
                            "g": [4.932820265944352, -3.9340978715479604],
                            "n_samples": 2,
                            "RMSE": 0.003300579311725016,
                        },
                    },
                },
            },
        ],
    }
    return tbrf_dict


class TestTBRF:
    def test_to_dict(self, tbrf1, features, targets, tb, tbrf1_as_dict, seed1):
        tbrf1.fit(features, targets, tb, seed=seed1)
        import json

        with open("test_tmp1-tbrf.json", "w") as file:
            json.dump(tbrf1.to_dict(), file, indent=4)
        with open("test_tmp2-tbrf.json", "w") as file:
            json.dump(tbrf1_as_dict, file, indent=4)
        assert tbrf1.to_dict() == tbrf1_as_dict

    def test_from_dict(
        self, tbrf1, features, targets, tb, tbrf1_as_dict, seed1
    ):
        tbrf1.fit(features, targets, tb, seed=seed1)
        tbrf2 = TBRF.from_dict(tbrf1_as_dict)
        assert tbrf2 == tbrf1

    def test_predict(self, tbrf1, features, targets, tb, seed1, b_prediction1):
        tbrf1.fit(features, targets, tb, seed=seed1)
        g_trees, b_trees, b = tbrf1.predict(features, tb)
        import json

        with open("test_preds-tbrf.txt", "w") as file:
            file.write(
                json.dumps(
                    {
                        "g_trees": [g.tolist() for g in g_trees],
                        "b_trees": [b.tolist() for b in b_trees],
                        "b": b.tolist(),
                    },
                    indent=4,
                )
            )
        assert_array_almost_equal(b, b_prediction1, decimal=2)
