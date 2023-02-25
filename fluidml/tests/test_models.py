import pytest

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
def tbdt1():
    kwargs = {
        "name": "TBDT-1",
        "max_depth": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "gamma": 1e-3,
        "optim_threshold": -1,
        "random_state": 42,
    }
    return TBDT(**kwargs)


@pytest.fixture
def tbdt1_as_dict():
    return {
        "name": "TBDT-1",
        "max_depth": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "gamma": 0.001,
        "optim_threshold": -1,
        "random_state": 42,
        "nodes": {
            "0": {
                "tag": "0",
                "data": {
                    "split_i": 0,
                    "split_v": -3.5,
                    "g": [0.42590877160095203, 0.07918256518898324],
                    "n_samples": 5,
                    "RMSE": 8.616046224895372,
                },
            },
            "01": {
                "tag": "01",
                "data": {
                    "split_i": 1,
                    "split_v": 1.5,
                    "g": [0.4492982041554332, 0.05879830420926233],
                    "n_samples": 4,
                    "RMSE": 7.68299030900193,
                },
            },
            "011": {
                "tag": "011",
                "data": {
                    "split_i": 2,
                    "split_v": 2.5,
                    "g": [0.47617462083569206, 0.038794309017542904],
                    "n_samples": 3,
                    "RMSE": 6.607018587777909,
                },
            },
            "0111": {
                "tag": "0111",
                "data": {
                    "split_i": 0,
                    "split_v": 0.5,
                    "g": [0.5554582251667012, -0.017438358337960452],
                    "n_samples": 2,
                    "RMSE": 5.264848639164375,
                },
            },
            "01111": {
                "tag": "01111",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [1.9440298074400952, -0.9441627598264528],
                    "n_samples": 1,
                    "RMSE": 0.001035385350270451,
                },
            },
            "01110": {
                "tag": "01110",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
            "0110": {
                "tag": "0110",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
            "010": {
                "tag": "010",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
            "00": {
                "tag": "00",
                "data": {
                    "split_i": None,
                    "split_v": None,
                    "g": [0.9444246516991186, 0.055558291050967164],
                    "n_samples": 1,
                    "RMSE": 0.00013616452239853343,
                },
            },
        },
    }


@pytest.fixture
def tbdt1_as_graphviz():
    return r"""digraph tree {
	label="TBDT-1";

	"0" [label="split feat idx: 0\nvalue: -3.500e+00\nnb samples: 5\nRMSE: 8.616e+00", shape=rectangle];
	"00" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"01" [label="split feat idx: 1\nvalue: 1.500e+00\nnb samples: 4\nRMSE: 7.683e+00", shape=rectangle];
	"010" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"011" [label="split feat idx: 2\nvalue: 2.500e+00\nnb samples: 3\nRMSE: 6.607e+00", shape=rectangle];
	"0110" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"0111" [label="split feat idx: 0\nvalue: 5.000e-01\nnb samples: 2\nRMSE: 5.265e+00", shape=rectangle];
	"01110" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.362e-04", shape=rectangle];
	"01111" [label="split feat idx: None\nvalue: None\nnb samples: 1\nRMSE: 1.035e-03", shape=rectangle];

	"0" -> "01";
	"0" -> "00";
	"01" -> "011";
	"01" -> "010";
	"011" -> "0111";
	"011" -> "0110";
	"0111" -> "01111";
	"0111" -> "01110";
}"""


def test_TBDT_to_dict(tbdt1, features, targets, tb, tbdt1_as_dict):
    tbdt1.fit(features, targets, tb)
    import json

    with open("tmp1.json", "w") as file:
        json.dump(tbdt1.to_dict(), file, indent=4)
    with open("tmp2.json", "w") as file:
        json.dump(tbdt1_as_dict, file, indent=4)
    assert tbdt1.to_dict() == tbdt1_as_dict


def test_TBDT_to_graphviz(tbdt1, features, targets, tb, tbdt1_as_graphviz):
    tbdt1.fit(features, targets, tb)
    with open("tmp.dot", "w") as file:
        file.write(tbdt1.to_graphviz())
    assert tbdt1.to_graphviz() == tbdt1_as_graphviz
