import pytest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal

from fluidml.models import TBDT


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
def g_prediction1():
    return np.array(
        [
            [0.94442465, 0.05555829],
            [0.94442465, 0.05555829],
            [0.55545823, -0.01743836],
            [0.55545823, -0.01743836],
            [0.55545823, -0.01743836],
        ]
    )


@pytest.fixture
def b_prediction1():
    return np.array(
        [
            [-2.69, -1.69, -0.69, 0.30, 1.30, 2.29, 3.29, 4.29, 5.29],
            [15.29, 16.29, 17.29, 18.29, 19.29, 20.29, 21.29, 22.29, 23.29],
            [17.49, 18.02, 18.56, 19.10, 19.64, 20.18, 20.71, 21.25, 21.79],
            [27.17, 27.71, 28.25, 28.78, 29.32, 29.86, 30.40, 30.94, 31.47],
            [36.85, 37.39, 37.93, 38.47, 39.01, 39.54, 40.08, 40.62, 41.16],
        ]
    )


@pytest.fixture
def tbdt1():
    kwargs = {
        "name": "TBDT-1",
        "max_depth": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "gamma": 1e-3,
        "optim_threshold": -1,
    }
    return TBDT(**kwargs)


@pytest.fixture
def tbdt1_as_dict():
    return {
        "name": "TBDT-1",
        "nodes": {
            "R": {
                "tag": "R",
                "data": {
                    "split_i": 0,
                    "split_v": -3.5,
                    "g": [0.42590877160095203, 0.07918256518898324],
                    "n_samples": 5,
                    "RMSE": 8.616046224895372,
                },
            },
            "R1": {
                "tag": "R1",
                "data": {
                    "split_i": 1,
                    "split_v": 1.5,
                    "g": [0.4492982041554332, 0.05879830420926233],
                    "n_samples": 4,
                    "RMSE": 7.68299030900193,
                },
            },
            "R11": {
                "tag": "R11",
                "data": {
                    "split_i": 3,
                    "split_v": 3.5,
                    "g": [0.47617462083569206, 0.038794309017542904],
                    "n_samples": 3,
                    "RMSE": 6.607018587777909,
                },
            },
            "R111": {
                "tag": "R111",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.5554582251667012, -0.017438358337960452],
                    "n_samples": 2,
                    "RMSE": 5.264848639164375,
                },
            },
            "R110": {
                "tag": "R110",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
            "R10": {
                "tag": "R10",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
            "R0": {
                "tag": "R0",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
        },
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "gamma": 0.001,
        "optim_threshold": -1,
    }


@pytest.fixture
def tbdt1_as_graphviz():
    return r"""digraph tree {
	label="TBDT-1";

	"R" [label="split feat idx: 0\nvalue: -3.500e+00\nnb samples: 5\nRMSE: 8.616e+00", shape=rectangle];
	"R0" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"R1" [label="split feat idx: 1\nvalue: 1.500e+00\nnb samples: 4\nRMSE: 7.683e+00", shape=rectangle];
	"R10" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"R11" [label="split feat idx: 3\nvalue: 3.500e+00\nnb samples: 3\nRMSE: 6.607e+00", shape=rectangle];
	"R110" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"R111" [label="split feat idx: None\nvalue: None\nnb samples: 2\nRMSE: 5.265e+00", shape=rectangle];

	"R" -> "R1";
	"R" -> "R0";
	"R1" -> "R11";
	"R1" -> "R10";
	"R11" -> "R111";
	"R11" -> "R110";
}"""


class TestTBDT:
    def test_to_dict(self, tbdt1, features, targets, tb, tbdt1_as_dict, seed1):
        tbdt1.fit(features, targets, tb, seed=seed1)
        import json

        with open("tmp1.json", "w") as file:
            json.dump(tbdt1.to_dict(), file, indent=4)
        with open("tmp2.json", "w") as file:
            json.dump(tbdt1_as_dict, file, indent=4)
        assert tbdt1.to_dict() == tbdt1_as_dict

    def test_from_dict(
        self, tbdt1, features, targets, tb, tbdt1_as_dict, seed1
    ):
        tbdt1.fit(features, targets, tb, seed=seed1)
        tbdt2 = TBDT.from_dict(tbdt1_as_dict)
        assert tbdt2 == tbdt1

    def test_save_to_json(self, tbdt1, features, targets, tb, seed1):
        tbdt1.fit(features, targets, tb, seed=seed1)
        file_path = Path(__file__).parent / "test_tbdt1.json"
        tbdt1.save_to_json(file_path)
        tbdt2 = TBDT.load_from_json(file_path)
        file_path.unlink()
        assert tbdt1 == tbdt2

    def test_load_from_json(self, tbdt1, features, targets, tb, seed1):
        tbdt1.fit(features, targets, tb, seed=seed1)
        file_path = Path(__file__).parent / "test_tbdt1.json"
        tbdt1.save_to_json(file_path)
        tbdt2 = TBDT.load_from_json(file_path)
        file_path.unlink()
        assert tbdt1 == tbdt2

    def test_to_graphviz(
        self, tbdt1, features, targets, tb, tbdt1_as_graphviz, seed1
    ):
        tbdt1.fit(features, targets, tb, seed=seed1)
        with open("tmp.dot", "w") as file:
            file.write(tbdt1.to_graphviz())
        assert tbdt1.to_graphviz() == tbdt1_as_graphviz

    def test_predict(
        self, tbdt1, features, targets, tb, seed1, g_prediction1, b_prediction1
    ):
        tbdt1.fit(features, targets, tb, seed=seed1)
        g, b = tbdt1.predict(features, tb)
        assert_array_almost_equal(g, g_prediction1)
        assert_array_almost_equal(b, b_prediction1, decimal=2)
