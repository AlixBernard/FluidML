import pytest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal

from fluidml.models import TBDT, fit_tensor


@pytest.fixture
def seed1():
    return 47


@pytest.fixture
def gamma1():
    return 4e-1


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
def TT(tb, gamma1):
    n, m = 5, 2
    TT = np.zeros([n, m, m])
    for i in range(n):
        TT[i] = tb[i] @ tb[i].T + gamma1 * np.eye(m)
    return TT


@pytest.fixture
def Ty(targets, tb):
    n, m = 5, 2
    Ty = np.zeros([n, m])
    for i in range(n):
        Ty[i] = tb[i] @ targets[i]
    return Ty


@pytest.fixture
def g_prediction1():
    return np.array(
        [
            [0.9444246516991186, 0.055558291050967164],
            [1.9440298074400952, -0.9441627598264528],
            [2.942493937670776, -1.9428759081457725],
            [0.7074901246994713, -0.1694760869379522],
            [0.7074901246994713, -0.1694760869379522],
        ]
    )


@pytest.fixture
def b_prediction1():
    return np.array(
        [
            [-2.69, -1.69, -0.69, 0.30, 1.30, 2.29, 3.29, 4.29, 5.296],
            [6.30, 7.30, 8.30, 9.30, 10.30, 11.29, 12.29, 13.29, 14.297],
            [15.30, 16.30, 17.30, 18.30, 19.30, 20.29, 21.29, 22.29, 23.295],
            [25.80, 26.34, 26.88, 27.41, 27.95, 28.49, 29.03, 29.57, 30.107],
            [35.49, 36.02, 36.56, 37.10, 37.64, 38.18, 38.71, 39.25, 39.791],
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
        "max_depth": 3,
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
                    "split_v": -3.5,
                    "g": [0.42590877160095203, 0.07918256518898324],
                    "n_samples": 5,
                    "RMSE": 1.2844043370764329,
                },
            },
            "R0": {
                "tag": "R0",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 4.5388174132844476e-05,
                },
            },
            "R1": {
                "tag": "R1",
                "data": {
                    "split_i": 1,
                    "split_v": 1.5,
                    "g": [0.4492982041554332, 0.05879830420926233],
                    "n_samples": 4,
                    "RMSE": 1.280498384833655,
                },
            },
            "R10": {
                "tag": "R10",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [1.9440298074400952, -0.9441627598264528],
                    "n_samples": 1,
                    "RMSE": 0.0003451284500901504,
                },
            },
            "R11": {
                "tag": "R11",
                "data": {
                    "split_i": 3,
                    "split_v": 7.5,
                    "g": [0.5061079087239555, 0.008860387786607164],
                    "n_samples": 3,
                    "RMSE": 1.2715213202286189,
                },
            },
            "R110": {
                "tag": "R110",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [2.942493937670776, -1.9428759081457725],
                    "n_samples": 1,
                    "RMSE": 0.0009880860159300995,
                },
            },
            "R111": {
                "tag": "R111",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.7074901246994713, -0.1694760869379522],
                    "n_samples": 2,
                    "RMSE": 1.2409367273320815,
                },
            },
        },
    }


@pytest.fixture
def tbdt1_as_graphviz():
    return r"""digraph tree {
	label="TBDT-1";

	"R" [label="split feat idx: 0\nvalue: -3.500e+00\nnb samples: 5\nRMSE: 1.284e+00", shape=rectangle];
	"R0" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 4.539e-05", shape=rectangle];
	"R1" [label="split feat idx: 1\nvalue: 1.500e+00\nnb samples: 4\nRMSE: 1.280e+00", shape=rectangle];
	"R10" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 3.451e-04", shape=rectangle];
	"R11" [label="split feat idx: 3\nvalue: 7.500e+00\nnb samples: 3\nRMSE: 1.272e+00", shape=rectangle];
	"R110" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 9.881e-04", shape=rectangle];
	"R111" [label="split feat idx: None\nvalue: None\nnb samples: 2\nRMSE: 1.241e+00", shape=rectangle];

	"R" -> "R0";
	"R" -> "R1";
	"R1" -> "R10";
	"R1" -> "R11";
	"R11" -> "R110";
	"R11" -> "R111";
}"""


def test_fit_tensor(TT, Ty, tb, targets):
    ghat, bhat = fit_tensor(TT, Ty, tb, targets)
    expected_ghat = np.array([0.42508692, 0.07988004])
    expected_bhat = np.array(
        [
            [-0.89, -0.39, 0.11, 0.61, 1.12, 1.62, 2.13, 2.63, 3.14],
            [8.19, 8.69, 9.20, 9.70, 10.21, 10.71, 11.22, 11.72, 12.23],
            [17.28, 17.78, 18.29, 18.79, 19.30, 19.80, 20.31, 20.81, 21.32],
            [26.37, 26.87, 27.38, 27.88, 28.39, 28.89, 29.40, 29.90, 30.41],
            [35.46, 35.96, 36.47, 36.97, 37.48, 37.98, 38.49, 38.99, 39.50],
        ]
    )
    assert_array_almost_equal(ghat, expected_ghat)
    assert_array_almost_equal(bhat, expected_bhat, decimal=2)


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
        try:
            tbdt1.save_to_json(file_path)
            tbdt2 = TBDT.load_from_json(file_path)
        except Exception as err:
            print(
                "The follwing error prevented removing the temporarly save"
                f"file {file_path}:\n\t{err}"
            )
        finally:
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
        with open("test_tmp.dot", "w") as file:
            file.write(tbdt1.to_graphviz())
        assert tbdt1.to_graphviz() == tbdt1_as_graphviz

    def test_predict(
        self, tbdt1, features, targets, tb, seed1, g_prediction1, b_prediction1
    ):
        tbdt1.fit(features, targets, tb, seed=seed1)
        g, b = tbdt1.predict(features, tb)
        import json

        with open("test_preds.txt", "w") as file:
            file.write(
                json.dumps({"g": g.tolist(), "b": b.tolist()}, indent=4)
            )
        assert_array_almost_equal(g, g_prediction1)
        assert_array_almost_equal(b, b_prediction1, decimal=2)
