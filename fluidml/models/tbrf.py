"""Class for the Tensor Basis Random Forest (TBRF).

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis
    - `s` is the number of TBDTs in the TBRF

"""

__all__ = ["TBRF"]


import json
import logging
import multiprocessing as mp
from time import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng


from fluidml.models import Tree, TBDT


class TBRF:
    """Tensor Basis Random Forest.

    Attributes
    ----------
    name : str, default='TBRF'
        Name of the forest used as its string representation.
    n_estimators : int, default=10
        Number of trees to build in the forest.
    max_features : int or float or str or None, default='sqrt'
        Number of features to consider when looking for the best split:
            - if int then consider `max_features`
            - if float then consider `ceil(max_features * m)`
            - if 'sqrt' then consider `ceil(srqt(m))`
            - if 'log2' then consider `ceil(log2(m))`
            - if None then consider `m`
        where `m` is the total number of features.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If
        False, the whole dataset is used to build each tree.
    max_samples : int or float or None, default=None
        If bootstrap is True, the number of samples to draw from x to
        to train each tree:
            - if None then draw `n` samples
            - if int then draw `max_samples` samples
            - if float then draw `round(max_samples * n)` samples
        where `n` is the total number of sample.
    tbdt_kwargs : dict or None, default=None
        Keyword arguments for the TBDTs.
    logger : logging.Logger, default=None
        Logger to output details.

    Methods
    -------
    to_json
    from_json
    to_graphviz
    fit
    predict

    """

    # TODO: add OOB score

    def __init__(
        self,
        name: str = "TBRF",
        n_estimators: int = 10,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        logger: logging.Logger | None = None,
        tbdt_kwargs: dict | None = None,
    ) -> None:
        self.name = name
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self._logger = logger
        self.tbdt_kwargs = tbdt_kwargs if tbdt_kwargs is not None else {}

        self.trees = [
            TBDT(
                name=f"{self.name}_TBDT-{i+1}",
                **self.tbdt_kwargs,
            )
            for i in range(self.n_estimators)
        ]
        self._log(logging.INFO, f"Initialized {self.n_estimators} TBDTs")

    def __len__(self) -> int:
        return len(self.trees)

    def __str__(self) -> str:
        s = f"{self.name}"
        return s

    def __repr__(self) -> str:
        attrs2skip = ["logger"]

        str_attrs = []
        for k, v in sorted(self.__dict__.items()):
            if k in attrs2skip:
                continue

            if k == "trees":
                s = f"{k}: {[tbdt.name for tbdt in self.trees]!r}"
            else:
                s = f"{k}: {v!r}"
            str_attrs.append(s)

        obj_repr = f"TBRF({', '.join(str_attrs)})"
        return obj_repr

    def __eq__(self, tbrf) -> bool:
        attrs2skip = ["trees", "_logger"]
        for k in self.__dict__:
            if k in attrs2skip:
                continue
            if self.__dict__[k] != tbrf.__dict__[k]:
                return False

        if len(self.trees) != len(tbrf.trees):
            return False
        for tree1, tree2 in zip(self.trees, tbrf.trees):
            if tree1 != tree2:
                return False
        return True

    def _log(self, level: int, message: str, *args, **kwargs) -> None:
        if self._logger is not None:
            self._logger.log(level, message, *args, **kwargs)

    def _timer_func(func):
        def wrap_func(self, *args, **kwargs):
            t1 = time()
            result = func(self, *args, **kwargs)
            t2 = time()
            self._log(
                logging.DEBUG,
                f"Method '{self.name}.{func.__name__}()' executed in "
                f"{(t2-t1):.2f}s",
            )
            return result

        return wrap_func

    def _get_n_samples(self, n: int) -> int:
        """Compute the number of samples to use from `x` based on
        `self.max_samples`.

        Parameters
        ----------
        n : int
            Number of samples available to draw from.

        Returns
        -------
        n_samples : int
            Number of samples to draw.

        Raises
        ------
        TypeError
            If the attribute `max_sample` is neither an int, float, or
            None.

        """
        if self.max_samples is None:
            n_samples = n
        elif isinstance(self.max_samples, int):
            n_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            n_samples = round(self.max_samples * n)
        else:
            raise TypeError(
                f"The {self.max_samples} is not recognized"
                f" for the attribute `max_samples`"
            )
        return n_samples

    def to_dict(self) -> dict:
        """Returns the TBRF as its dict representation."""
        attrs2skip = ["_logger"]
        d = {}
        for k, v in self.__dict__.items():
            if k in attrs2skip:
                continue
            if k == "trees":
                d[k] = [tbdt.to_dict() for tbdt in v]
                continue
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, tbrf_dict: dict):
        """Create a TBRF from its dict representation.

        Parameters
        ----------
        tbrf_dict : dict
            The dict representation of the TBDT to create.

        """
        tbrf_kwargs = {
            k: v for k, v in tbrf_dict.items() if k not in ["trees"]
        }
        tbrf = TBRF(**tbrf_kwargs)
        trees = []
        for tbdt_dict in tbrf_dict["trees"]:
            tbdt = TBDT.from_dict(tbdt_dict)
            trees.append(tbdt)
        tbrf.trees = trees
        return tbrf

    def to_json(self, dir_path: Path):
        """Save the TBRF as a directory containing the JSON files of its
        attributes and the TBDTs' attributes. The TBRF JSON file only
        has the list of names of the TBDTs in its field "trees".

        Parameters
        ----------
        dir_path : Path
            Path to the directory where to save the TBRF's and TBDTs'
            files, the directory will be deleted and recreated.

        """
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            self._log(logging.INFO, f"Created the folder: '{dir_path}'")

        for tbdt in self.trees:
            tbdt_filename = f"{tbdt}.json"
            tbdt_path = dir_path / tbdt_filename
            tbdt.to_json(tbdt_path)

        attrs2skip = ["_logger"]
        json_attrs = {}
        for k, v in self.__dict__.items():
            if k in attrs2skip:
                continue
            elif k == "trees":
                json_attrs[k] = [tree.name for tree in v]
            else:
                json_attrs[k] = v
        del json_attrs["tbdt_kwargs"]["_logger"]

        tbrf_path = dir_path / f"{self}.json"
        with open(tbrf_path, "w") as file:
            json.dump(json_attrs, file, indent=4)

        self._log(logging.INFO, f"Saved '{self}' as: '{tbrf_path}'")

    def from_json(self, dir_path: Path):
        """Load the TBRF as a folder containing the JSON files of its
        attributes and the TBDTs' attributes. The TBRF JSON file only
        has the list of names of the TBDTs in its field "trees".

        Parameters
        ----------
        dir_path : Path
                Path to the folder containing the TBRF's and TBDTs' JSON
                files.

        """
        tbrf_path = dir_path / f"{str(self)}.json"
        with open(tbrf_path, "r") as file:
            json_attrs = json.load(file)

        tbdt_names = []
        for k, v in json_attrs:
            if k == "trees":
                tbdt_names.extend(v)
            self.__setattr__(k, v)
        self._log(logging.INFO, f"Loaded '{self}' from: '{tbrf_path}'")

        for name in tbdt_names:
            tbdt_filename = f"{name}.json"
            tbdt_path = dir_path / tbdt_filename
            self.trees[k] = TBDT().load(tbdt_path)

    def to_graphviz(
        self,
        dir_path: str | Path,
        shape: str = "rectangle",
        graph: str = "diagraph",
    ) -> None:
        """Export each tree of the random forest as a graphviz dot file.

        Parameters
        ----------
        dir_path : str or Path
            Directory to which save the tree.
        shape : {'rectangle', 'circle'}, default='rectangle'
            Shape of the nodes.
        graph : str, default='diagraph'
            Type of graph.

        """
        for tree in self.trees:
            tree.to_graphviz(dir_path, shape=shape, graph=graph)

        self._log(
            logging.INFO, f"Exported '{self.name}' to graphviz in '{dir_path}'"
        )

    @_timer_func
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tb: np.ndarray,
        n_jobs: int = 1,
        seed: int | None = None,
    ) -> dict:
        """Create the TBRF given input features `x`, true response `y`,
        and tensor basis `tb`.

        Parameters
        ----------
        x : np.ndarray
            Input features with shape `(n, p)`.
        y : np.ndarray
            Anisotropy tensors `b` with shape `(n, 9)` on which to fit
            the tree.
        tb : np.ndarray
            Tensor bases with shape `(n, m, 9)`.
        n_jobs : int, default=1
            The number of jobs to run in parallel, -1 means using all
            processors.
        seed : int | None

        """
        self._log(logging.INFO, f"Fitting all trees of '{self.name}'")

        rng = default_rng(seed)
        tbdt_seeds = [rng.integers(int(1e1)) for _ in range(self.n_estimators)]
        print(tbdt_seeds)
        jobs = (n_jobs,) if n_jobs != -1 else ()
        with mp.Pool(*jobs) as pool:
            res = [
                pool.apply_async(self._fit_tree, (i, x, y, tb, seed))
                for i, seed in enumerate(tbdt_seeds)
            ]
            self.trees = [r.get() for r in res]

        self._log(logging.INFO, f"Fitted all trees of '{self.name}'")

    def _fit_tree(
        self,
        i_tree: int,
        x: np.ndarray,
        y: np.ndarray,
        tb: np.ndarray,
        seed: int | None = None,
    ) -> list[Tree]:
        """Fit the specified tree."""
        print(f"{i_tree=}, {seed=}")
        rng = default_rng(seed)
        n = len(x)
        n_samples = self._get_n_samples(n)
        tbdt = self.trees[i_tree]
        idx_sampled = (
            rng.choice(n, size=n_samples, replace=True)
            if self.bootstrap
            else np.arange(n)
        )
        print(idx_sampled)

        x_sampled = x[idx_sampled]
        y_sampled = y[idx_sampled]
        tb_sampled = tb[idx_sampled]
        tbdt.fit(x_sampled, y_sampled, tb_sampled, rng.integers(int(1e9)))

        return self.trees[i_tree]

    @_timer_func
    def predict(
        self,
        x: np.ndarray,
        tb: np.ndarray,
        method: str = "mean",
        n_jobs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tensor Basis Random Forest predictions given input features
        `x_test` and tensor basis `tb_test`, make predictions for the
        anisotropy tensor `b` using its fitted trees.

        Parameters
        ----------
        x : np.ndarray
            Input features with shape `(n, p)`.
        tb : np.ndarray
            Tensor bases with shape `(n, m, 9)`.
        method : {'mean', 'median'}
            How to compute the TBRF prediction from all the TBDT
            predictions, possible values are 'mean' and 'median'.
        n_jobs : int, default=1
            The number of jobs to run in parallel, -1 means using all
            processors.

        Returns
        -------
        bhat : np.ndarray
            Anisotropy tensors with shape `(n, 9)`.
        b : np.ndarray
            Anisotropy tensors for each TBDT in the TBRF with shape
            `(s, n, 9)`.
        g : np.ndarray
            Tensor basis coefficients for each TBDT in the TBRF with
            shape `(s, n, 10)`.

        Raises
        ------
        ValueError
            If the parameter `method` is not a valid value.

        """
        n, m, _ = tb.shape

        # Initialize predictions
        b_trees = np.zeros([len(self), n, 9])
        g_trees = np.zeros([len(self), n, m])

        jobs = (n_jobs,) if n_jobs != -1 else ()
        with mp.Pool(*jobs) as pool:
            res = [
                pool.apply_async(self._predict_tree, (i, x, tb))
                for i in range(len(self))
            ]
            data = [r.get() for r in res]

        # Go through the TBDTs of the TBRF to make predictions
        for i in range(len(self)):
            g_trees[i], b_trees[i] = data[i]

        if method == "mean":
            b = np.mean(b_trees, axis=0)
        elif method == "median":
            b = np.median(b_trees, axis=0)
        else:
            raise ValueError(
                f"The `method` attribute must be 'mean' or 'median'"
            )

        self._log(logging.INFO, "Predicted the anysotropy tensor 'b'")

        return g_trees, b_trees, b

    def _predict_tree(
        self, i_tree: int, x: np.ndarray, tb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict from the tree specified."""
        g, b = self.trees[i_tree].predict(x, tb)
        return g, b
