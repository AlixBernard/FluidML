"""Class for the Tensor Basis Random Forest (TBRF).

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis
    - `d` is the number of components of the vector/tensor to predict
    - `s` is the number of TBDTs in the TBRF

"""

__all__ = ["PREDICTION_METHODS", "TBRF"]


import functools
import logging
import more_itertools as mit
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.random import default_rng

from fluidml.models import TBDT

PREDICTION_METHODS = {
    "mean": lambda x: np.mean(x, axis=0),
    "median": lambda x: np.median(x, axis=0),
}


def _log(
    level: int,
    message: str,
    logger: logging.Logger | None = None,
    *args,
    **kwargs,
) -> None:
    if logger is not None:
        logger.log(level, message, *args, **kwargs)


def _timer_debug_log(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        logger = kwargs.get("logger")
        try:
            class_name = f"{args[0].__class__.__name__}"
        except (IndexError, AttributeError):
            class_name = None
        prefix = f"{class_name}." if class_name is not None else ""
        _log(
            logging.DEBUG,
            f"'{prefix}{func.__name__}' executed in {(t2-t1):.2f}s",
            logger,
        )
        return result

    return wrap_func


class TBRF:
    """Tensor Basis Random Forest.

    Attributes
    ----------
    name : str, default='TBRF'
        Name of the forest used as its string representation.
    n_estimators : int, default=10
        Number of trees to build in the forest.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If
        False, the whole dataset is used to build each tree.
    max_samples : int | float | None, default=None
        If bootstrap is True, the number of samples to draw from x to
        to train each tree:
            - if None then draw `n` samples
            - if int then draw `max_samples` samples
            - if float then draw `round(max_samples * n)` samples
        where `n` is the total number of sample.
    tbdt_kwargs : dict | None, default=None
        Keyword arguments for the TBDTs.
    trees : list[Tree]
        The TBDTs of the TBRF.

    Methods
    -------
    to_dict
    from_dict
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
        tbdt_kwargs: dict | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.tbdt_kwargs = tbdt_kwargs if tbdt_kwargs is not None else {}

        padding = f"0>{len(str(self.n_estimators))}"
        self.trees = [
            TBDT(
                name=f"{self.name}_TBDT-{i+1:{padding}}",
                **self.tbdt_kwargs,
            )
            for i in range(self.n_estimators)
        ]
        _log(logging.INFO, f"Initialized {self.n_estimators} TBDTs", logger)

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
        attrs2skip = ["trees"]
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
            If the attribute `max_sample` is neither an int, a float,
            nor a None.

        """
        if self.max_samples is None:
            n_samples = n
        elif isinstance(self.max_samples, int):
            n_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            n_samples = round(self.max_samples * n)
        else:
            raise TypeError(
                f"The value {self.max_samples} is not recognized"
                f" for the attribute `max_samples`"
            )
        return n_samples

    def to_dict(self) -> dict:
        """Returns the TBRF as its dict representation."""
        d = {k: v for k, v in self.__dict__.items() if k != "trees"}
        d["trees"] = [tbdt.to_dict() for tbdt in self.trees]
        return d

    @classmethod
    def from_dict(cls, tbrf_dict: dict):
        """Create a TBRF from its dict representation.

        Parameters
        ----------
        tbrf_dict : dict
            The dict representation of the TBRF to create.

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

    def to_graphviz(
        self,
        dir_path: Path | str,
        shape: str = "rectangle",
        graph: str = "diagraph",
    ) -> None:
        """Export each tree of the random forest as a graphviz dot file.

        Parameters
        ----------
        dir_path : Path | str
            Directory to which save the tree.
        shape : {'rectangle', 'circle'}, default='rectangle'
            Shape of the nodes.
        graph : str, default='diagraph'
            Type of graph.

        """
        for tree in self.trees:
            tree.to_graphviz(
                Path(dir_path) / f"{tree.name}.dot", shape=shape, graph=graph
            )

    @_timer_debug_log
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tb: np.ndarray,
        n_jobs: int | None = None,
        seed: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create the TBRF given input features `x`, true response `y`,
        and tensor bases `tb`.

        Parameters
        ----------
        x : np.ndarray[shape=(n, p)]
            Input features.
        y : np.ndarray[shape=(n, d)]
            Anisotropy tensors $b_{ij}$ on which to fit the tree.
        tb : np.ndarray[shape=(n, m, d)]
            Tensor bases.
        n_jobs : int, default=None
            The number of jobs to run in parallel, None means using all
            processors.
        seed : int | None

        """
        _log(logging.INFO, f"Fit {self.name}", logger)
        t_start = perf_counter()

        rng = default_rng(seed)
        seeds = [rng.integers(int(1e1)) for _ in range(self.n_estimators)]
        with mp.Pool(processes=n_jobs) as pool:
            res = [
                pool.apply_async(self._fit_tree, (i, x, y, tb, seed, logger))
                for i, seed in enumerate(seeds)
            ]
            self.trees = [r.get() for r in res]

        t_end = perf_counter()
        t_delta = t_end - t_start
        _log(logging.INFO, f"Fitted {self.name} in {t_delta:.3f}s", logger)

    def _fit_tree(
        self,
        i_tree: int,
        x: np.ndarray,
        y: np.ndarray,
        tb: np.ndarray,
        seed: int | None = None,
        logger: logging.Logger | None = None,
    ) -> TBDT:
        """Fit the specified tree."""
        rng = default_rng(seed)
        n = len(x)
        n_samples = self._get_n_samples(n)
        tbdt = self.trees[i_tree]
        idx_sampled = (
            rng.choice(n, size=n_samples, replace=True)
            if self.bootstrap
            else np.arange(n)
        )
        tbdt_seed = rng.integers(int(1e9))

        x_sampled = x[idx_sampled]
        y_sampled = y[idx_sampled]
        tb_sampled = tb[idx_sampled]
        tbdt.fit(
            x_sampled, y_sampled, tb_sampled, seed=tbdt_seed, logger=logger
        )

        return self.trees[i_tree]

    @_timer_debug_log
    def predict(
        self,
        x: np.ndarray,
        tb: np.ndarray,
        method: str = "mean",
        n_jobs: int | None = None,
        logger: logging.Logger | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tensor Basis Random Forest predictions given input features
        `x_test` and tensor bases `tb_test`, make predictions for the
        anisotropy tensor `b` using its fitted trees.

        Parameters
        ----------
        x : np.ndarray[shape=(n, p)]
            Input features.
        tb : np.ndarray[shape=(n, m, d)]
            Tensor bases.
        method : str {'mean', 'median'}
            How to compute the TBRF prediction from all the TBDT
            predictions, possible values are 'mean' and 'median'.
        n_jobs : int, default=None
            The number of jobs to run in parallel, None means using all
            processors. Using 1 may be faster.

        Returns
        -------
        g_trees : np.ndarray[shape=(s, n, m)]
            Tensor bases coefficients for each TBDT in the TBRF.
        y_trees : np.ndarray[shape=(s, n, d)]
            Anisotropy tensors $b_{ij}$ for each TBDT in the TBRF.
        y : np.ndarray[shape=(n, d)]
            Anisotropy tensors.

        Raises
        ------
        ValueError
            If the parameter `method` is not a valid value.

        """
        _log(logging.INFO, f"Predict via {self.name}", logger)
        t_start = perf_counter()

        n, m, d = tb.shape

        # Initialize predictions
        g_trees = np.zeros([len(self), n, m])
        y_trees = np.zeros([len(self), n, d])

        with mp.Pool(processes=n_jobs) as pool:
            n_chunks = n_jobs if n_jobs is not None else os.cpu_count()
            res = [
                pool.apply_async(
                    self._predict_chunk_trees,
                    (list(chunk_trees), x, tb, logger),
                )
                for chunk_trees in mit.divide(n_chunks, self.trees)
            ]
            data = [r.get() for r in res]

        i0, i1 = 0, 0
        for g_chunk, y_chunk in data:
            i0 = i1
            i1 += len(g_chunk)
            g_trees[i0:i1], y_trees[i0:i1] = g_chunk, y_chunk

        try:
            y = PREDICTION_METHODS[method](y_trees)
        except KeyError:
            raise ValueError(
                "The `method` attribute must be one of "
                f"{set(PREDICTION_METHODS)}"
            )

        t_end = perf_counter()
        t_delta = t_end - t_start
        _log(
            logging.INFO,
            f"Predicted via {self.name} in {t_delta:>.3f}s",
            logger,
        )

        return g_trees, y_trees, y

    def _predict_chunk_trees(
        self,
        chunk_trees: list[TBDT],
        x: np.ndarray,
        tb: np.ndarray,
        logger: logging.Logger | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict using the chunk of chunk_TBDTs specified."""
        chunk_size = len(chunk_trees)
        n, m, d = tb.shape
        y_chunk_trees = np.zeros([chunk_size, n, d])
        g_chunk_trees = np.zeros([chunk_size, n, m])

        for i, tree in enumerate(chunk_trees):
            g_chunk_trees[i], y_chunk_trees[i] = tree.predict(x, tb, logger)

        return g_chunk_trees, y_chunk_trees
