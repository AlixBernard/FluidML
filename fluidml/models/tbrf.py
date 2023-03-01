"""Classes for the Tensor Basis Decision Tree (TBDT) and the Tensor
Basis Random Forest (TBRF).

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis
    - `s` is the number of TBDTs in the TBRF

"""

__all__ = ["TBRF"]


import json
import logging
import time
from pathlib import Path

import numpy as np


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
    random_state : int or None, default=None
        Controls both the randomness of the bootstrapping of the samples
        used when building trees (if `bootstrap == True`) and the
        sampling of the features to consider when looking for the best
        split at each node (if `max_features < m`).
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
        random_state: int | None = None,
        logger: logging.Logger | None = None,
        tbdt_kwargs: dict | None = None,
    ) -> None:
        self.name = name
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self._rng = default_rng(random_state)
        self._logger = logger
        self.tbdt_kwargs = tbdt_kwargs if tbdt_kwargs is not None else {}

        self.trees = [
            TBDT(
                name=f"{self.name}_TBDT-{i}",
                random_state=self._rng.choice(1000),
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

        attrs2skip = ["logger", "rng"]
        json_attrs = {}
        for k, v in self.__dict__.items():
            if k in attrs2skip:
                continue
            elif k == "trees":
                json_attrs[k] = [tree.name for tree in v]
            else:
                json_attrs[k] = v
        del json_attrs["tbdt_kwargs"]["logger"]

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
        self, x: np.ndarray, y: np.ndarray, tb: np.ndarray, n_jobs: int = 1
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

        """
        self._log(logging.INFO, f"Fitting all trees of '{self.name}'")

        jobs = (n_jobs,) if n_jobs != -1 else ()
        with mp.Pool(*jobs) as pool:
            res = [
                pool.apply_async(self._fit_tree, (i, x, y, tb))
                for i in range(len(self))
            ]
            self.trees = [r.get() for r in res]

        self._log(logging.INFO, f"Fitted all trees of '{self.name}'")

    def _fit_tree(
        self, i_tree: int, x: np.ndarray, y: np.ndarray, tb: np.ndarray
    ) -> list[Tree]:
        """Fit the specified tree."""
        n = len(x)
        n_samples = self._get_n_samples(n)
        rng = self.trees[i_tree]._rng
        if self.bootstrap:
            idx_sampled = rng.choice(n, size=n_samples, replace=True)
        else:
            idx_sampled = np.arange(n)

        x_sampled = x[idx_sampled]
        y_sampled = y[idx_sampled]
        tb_sampled = tb[idx_sampled]
        self.trees[i_tree].fit(x_sampled, y_sampled, tb_sampled)

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
