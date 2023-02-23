#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-03-24 15:58:14
# @Last modified by: AlixBernard
# @Last modified time: 2022-07-13 16:19:30

"""Classes for the Tensor Basis Decision Tree (TBDT) and the Tensor
Basis Random Forest (TBRF).

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis
    - `s` is the number of TBDTs in the TBRF

"""

__all__ = [
    "fit_tensor",
    "obj_func_J",
    "find_Jmin_sorted",
    "find_Jmin_opt",
    "TBDT",
    "TBRF",
]


# Built-in packages
import json
import logging
import multiprocessing as mp
from collections import OrderedDict
from time import time
from functools import partial
from pathlib import Path

# Third party packages
import numpy as np
from scipy import optimize as opt
from numpy.random import default_rng
from treelib import Tree, Node

# Local packages
from fluidml import utils


def fit_tensor(
    TT: np.ndarray,
    Ty: np.ndarray,
    tb: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Makes a least square fit on training data `y`, by using the
    preconstructed matrices $T^t T$ and $T^t y$.
    Used in the `create_split()` method. The least squares fit is
    done with respect to scalar coefficients $g$ in the tensor basis
    series $b = \sum_{m=1}^{10} g^{(m)} T_{(m)}$.

    Parameters
    ----------
    TT : np.ndarray
        Preconstructed matrix $T^t T$ with shape `(n, m, 9)`.
    Ty : np.ndarray
        Preconstructed matrix $T^t*f$.
    tb : np.ndarray
        Tensor Basis for each points with shape `(n, m, 9)` where
        `n` is the number of points and `d` is the number of tensors
        in the tensor basis.
    y : np.ndarray
        Anisotropy tensor `b` (target) on which to fit the tree with
        shape `(n, 9)`.

    Returns
    -------
    ghat : np.ndarray
        Optimum value for the tensor basis coefficients with shape
        `(m,)`.
    bhat : np.ndarray
        Anysotropy tensor with shape `(n, 9)`.
    diff : np.ndarray
        The difference between `bhat` and the target `y` with shape
        `(n, 9)`.

    """
    n, m, _ = TT.shape
    lhs = TT.sum(axis=0)
    rhs = Ty.sum(axis=0)

    # Solve Eq. 3.25
    ghat, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    bhat = np.zeros([n, 9])
    for i in range(m):
        bhat += ghat[i] * tb[:, i]
    diff = y - bhat

    return ghat, bhat, diff


def obj_func_J(
    y_sorted: np.ndarray,
    tb_sorted: np.ndarray,
    TT_sorted: np.ndarray,
    Ty_sorted: np.ndarray,
    i_float: float | None = None,
) -> tuple[float, dict]:
    """Objective function which minimize the RMSE difference w.r.t. the
    target `y`.

    Parameters
    ----------
    y_sorted : np.ndarray
        Sorted output features.
    tb_sorted : np.ndarray
        Sorted tensor basess.
    TT_sorted : np.ndarray
        Sorted preconstructed matrices $transpose(T)*T$.
    Ty_sorted : np.ndarray
        Sorted preconstructed matrices $transpose(T)*f$.
    i_float : float or None
        If not None, index which value will be turned to an int to use
        when splitting the data.

    Returns
    -------
    J : float
        The value of the cost function.
    extra : dict
        Dictionarry containing the following extra data: g_l, g_r,
        diff_l, diff_r, diff.

    """
    if i_float is None:
        g, _, diff = fit_tensor(TT_sorted, Ty_sorted, tb_sorted, y_sorted)
        extra = {"g": g, "diff": diff}
    else:
        i = int(i_float)
        g_l, _, diff_l = fit_tensor(
            TT_sorted[:i], Ty_sorted[:i], tb_sorted[:i], y_sorted[:i]
        )
        g_r, _, diff_r = fit_tensor(
            TT_sorted[i:], Ty_sorted[i:], tb_sorted[i:], y_sorted[i:]
        )

        diff = np.vstack([diff_l, diff_r])
        extra = {
            "g_l": g_l,
            "g_r": g_r,
            "diff_l": diff_l,
            "diff_r": diff_r,
            "diff": diff,
        }
    J = np.mean(diff**2)
    return J, extra


def find_Jmin_sorted(
    split_i: int,
    x: np.ndarray,
    y: np.ndarray,
    tb: np.ndarray,
    TT: np.ndarray,
    Ty: np.ndarray,
) -> dict:
    """Find optimum splitting point for the feature with index
    `feat_i`. Data is pre-sorted to save computational costs($n log(n)$
    instead of $n^2$).

    Parameters
    ----------
    feat_i : int
        Index of the feature on which to find the optimum splitting
        point.
    x : np.ndarray
        Input features with shape `(n, p)`.
    y : np.ndarray
        Anisotropy tensor `b` (target) on which to fit the tree with
        shape `(n, 9)`.
    tb : np.ndarray
        Tensor basess with shape `(n, m, 9)`.
    TT : np.ndarray
        Preconstructed matrix $transpose(T)*T$.
    Ty : np.ndarray
        Preconstructed matrix $transpose(T)*f$.

    Returns
    -------
    results : dict
        Same as `best_res` in higher method.

    """
    n, p = x.shape
    asort = np.argsort(x[:, split_i])

    obs_identical = True if np.all(x == x[0]) else False

    best_J = 1e12
    for i in range(1, n):
        J, extra = obj_func_J(
            y[asort], tb[asort], TT[asort], Ty[asort], i_float=i
        )
        if J < best_J:
            best_i, best_J, best_extra = i, J, extra

    results = {
        "J": best_J,
        "split_i": split_i,
        "split_v": float(
            0.5 * (x[asort][best_i - 1, split_i] + x[asort][best_i, split_i])
        ),
        "idx_l": asort[:best_i],
        "idx_r": asort[best_i:],
        "g_l": best_extra["g_l"],
        "g_r": best_extra["g_r"],
        "MSE_l": np.mean(best_extra["diff_l"] ** 2),
        "MSE_r": np.mean(best_extra["diff_r"] ** 2),
        "n_l": best_i,
        "n_r": n - best_i,
    }

    return results


def find_Jmin_opt(
    idx: int,
    x: np.ndarray,
    y: np.ndarray,
    tb: np.ndarray,
    TT: np.ndarray,
    Ty: np.ndarray,
) -> dict:
    """Find optimum splitting point by using an optimization routine.

    Parameters
    ----------
    idx : int
        Index of the feature on which to find the optimum splitting
        point.
    x : np.ndarray
        Input features with shape `(n, p)`.
    y : np.ndarray
        Anisotropy tensor `b` (target) on which to fit the tree with
        shape `(n, 9)`.
    tb : np.ndarray
        Tensor basis with shape `(n, m, 9)`.
    TT : np.ndarray
        Preconstructed matrix $transpose(T)*T$.
    Ty : np.ndarray
        Preconstructed matrix $transpose(T)*f$.

    Returns
    -------
    results : dict
        Same as `best_res` in higher method.

    """
    n, p = x.shape
    asort = np.argsort(x[idx, :])
    asort_back = np.argsort(asort)

    x_sorted = x[asort]
    y_sorted = y[asort]
    tb_sorted = tb[asort]
    TT_sorted = TT[asort]
    Ty_sorted = Ty[asort]

    obs_identical = True if np.all(x == x[0]) else False

    res = opt.minimize_scalar(
        obj_func_J,
        method="brent",
        tol=None,
        args=(y_sorted, tb_sorted, TT_sorted, Ty_sorted),
        bounds=(1, n - 1),
        options={"xtol": 1e-8, "maxiter": 200},
    )
    i_split = int(res.x)

    # TODO: in case optimization algorithm does not work it
    # - returns 0, needs further testing
    if i_split == 0:
        i_split = 1

    # Find all relevant parameters for the minimum which was found
    # ? Maybe this can be improved as it is redundant
    J, extra = obj_func_J(
        y_sorted, tb_sorted, TT_sorted, Ty_sorted, i_float=i_split
    )
    i_l_sorted = np.zeros(n, dtype=bool)
    i_l_sorted[:i_split] = True
    i_r_sorted = ~i_l_sorted

    results = {
        "J": J,
        "split_i": idx,
        "split_v": (
            0.5 * (x_sorted[i_split - 1, idx] + x_sorted[i_split, idx])
        ),
        "i_l": i_l_sorted[asort_back],
        "i_r": i_r_sorted[asort_back],
        "g_l": extra["g_l"],
        "g_r": extra["g_r"],
        "MSE_l": np.mean(extra["diff_l"] ** 2),
        "MSE_r": np.mean(extra["diff_r"] ** 2),
        "n_l": i_split,
        "n_r": n - i_split,
    }

    if obs_identical:
        # Right and left splits are made equal. This leads to
        # termination of the branch later on in `self.fit()`
        results["g_l"] = extra["g_r"]
        results["i_l"] = i_r_sorted[asort_back]
        results["n_l"] = 0
        results["MSE_l"] = 0

    return results


class TBDT:
    """Tensor Basis Decision Tree.

    Attributes
    ----------
    name : str, default='TBDT'
        Name of the tree used as its string representation.
    max_depth : int, default=400
        The maximum depth of the tree.
    min_sample_leaf : int
        Minimum number of samples required to be a leaf node.
    max_features : int or float or str or None, default='sqrt'
        Number of features to consider when looking for the best split:
            - if int then consider `max_features`
            - if float then consider `ceil(max_features * p)`
            - if 'sqrt' then consider `ceil(srqt(p))`
            - if 'log2' then consider `ceil(log2(p))`
            - if None then consider `p`
        where `p` is the total number of features. If the considered
        number of feature is not at least 1 then an error is raised.
    gamma : float, default=1.0
        The regularization parameter gamma, 1.0 means no regularization.
    optim_threshold : int, default=1_000
        Threshold for which if the number of points is below, brute
        force will be used and optim otherwise, if it is -1 then
        optimization is disabled.
    random_state : int or None, default=None
        Controls randomness when sampling the features.
    logger : logging.Logger or None, default=None
        Logger to output details.

    Methods
    -------
    to_json
    from_json
    to_graphviz
    create_split
    fit
    predict

    """

    def __init__(
        self,
        name: str = "TBDT",
        max_depth: int = 400,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        gamma: float = 1e0,
        optim_threshold: int = 1_000,
        random_state: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self.tree = Tree()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.gamma = gamma
        self.optim_threshold = optim_threshold
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.logger = logger

    def __str__(self) -> str:
        s = f"{self.name}"
        return s

    def __repr__(self) -> str:
        attrs2skip = ["logger"]

        str_attrs = []
        for k, v in sorted(self.__dict__.items()):
            if k not in attrs2skip:
                str_attrs.append(f"{k}: {v!r}")

        obj_repr = f"TBDT({', '.join(str_attrs)})"
        return obj_repr

    def _log(self, level: int, message: str, *args, **kwargs) -> None:
        if self.logger is not None:
            self.logger.log(level, message, *args, **kwargs)

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

    def _get_n_feats(self, p: int) -> int:
        """Compute the number of features to consider to perform each
        split (cf. attribute `max_features`).

        Parameters
        ----------
        p : int
            Total number of features available.

        Returns
        -------
        n_feats : int

        Raises
        ------
        RuntimeError
            If the value of the attribute `max_features` is not one of
            {'sqrt', 'log2'}, an int between 1 and `p`, or a float
            between 0 and 1.


        """
        if self.max_features is None:
            n_split_feats = p
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return int(np.ceil(np.sqrt(p)))
            elif self.max_features == "log2":
                return int(np.ceil(np.log2(p)))
        elif isinstance(self.max_features, int):
            if 1 <= self.max_features <= p:
                return self.max_features
        elif isinstance(self.max_features, float):
            if 0.0 < self.max_features <= 1.0:
                if round(self.max_features * p) >= 1:
                    return int(np.ceil((self.max_features * p)))
        raise RuntimeError(
            f"The attribute `max_features` (={self.max_features}) has an "
            f"incorrect value."
        )

    def to_json(self, path: Path | str) -> None:
        """Save the TBDT as a JSON file containing its attributes.

        Parameters
        ----------
        path : Path | str

        """
        attrs2skip = ["logger", "rng"]
        json_attrs = {}
        for k, v in self.__dict__.items():
            if k in attrs2skip:
                continue
            elif k == "tree":
                v_dict = OrderedDict(v.to_dict(with_data=True, sort=True))
                if "children" in v_dict:
                    v_dict.move_to_end("children")
                json_attrs["tree"] = v_dict
            else:
                json_attrs[k] = v
        json_attrs = OrderedDict(json_attrs)
        json_attrs.move_to_end("tree")

        with open(path, "w") as file:
            json.dump(json_attrs, file, indent=2)

        self._log(logging.INFO, f"Saved '{self}' as: '{path}'")

    def from_json(self, path: Path) -> None:
        """Load the TBDT from a JSON file containing its attributes.

        Parameters
        ----------
        path : Path

        """
        with open(path, "r") as file:
            json_attrs = json.load(file)
        for k, v in json_attrs:
            if k == "tree":
                self.tree = self._load_tree(v)
            else:
                self.__setattr__(k, v)
        self.rng = default_rng(self.random_state)

        self._log(logging.INFO, f"Loaded '{self}' from: '{path}'")

    def _load_tree(self, tree_dict: dict) -> None:
        subtree = tree_dict
        id_, node_dict = subtree.items()[0]
        data = node_dict["data"]
        nodes2add = [(Node(identifier=id_, tag=id_, data=data), None)]
        while nodes2add:
            node, parent = nodes2add.pop()
            self.tree.add_node(node, parent=parent)

            if "children" in node_dict:
                for subtree in node_dict["children"]:
                    id_, node_dict = subtree.items
                    data = node_dict["data"]
                    nodes2add.append(
                        (Node(identifier=id_, tag=id_, data=data), node)
                    )

    def to_graphviz(
        self, dir_path: Path, shape="rectangle", graph="diagraph"
    ) -> None:
        """Export the tree to the graphviz dot format.

        Parameters
        ----------
        dir_path : str or Path
            Directory to which save the tree.
        shape : {'rectangle', 'circle'}, default='rectangle'
            Shape of the nodes.
        graph : str, default='diagraph'
            Type of graph.

        """
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        self.tree.to_graphviz(Path(dir_path) / f"{self.name}.dot")

        self._log(
            logging.INFO,
            f"Exported '{self.name}' to graphviz as: "
            f"'{dir_path / f'{self.name}.dot'}'",
        )

    def create_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tb: np.ndarray,
        TT: np.ndarray,
        Ty: np.ndarray,
    ) -> dict:
        r"""Creates a split at a node for given input features `x`,
        training output `y`, tensor basis `tb`, and the preconstruced
        matrices `TT` and `Ty`.

        Parameters
        ----------
        x : np.ndarray
            Input features with shape `(n, p)`.
        y : np.ndarray
            Anisotropy tensors `b` (target) on which to fit the tree,
            with shape `(n, 9)`.
        tb : np.ndarray
            Tensor bases with shape `(n, m, 9)`.
        TT : np.ndarray
            Preconstructed matrix $transpose(T)*T$.
        Ty : np.ndarray
            Preconstructed matrix $transpose(T)*f$.

        Returns
        -------
        best_res : dict
            J : np.float64, split_i : list, split_v : list, i_l : list,
            i_r : list, g_l : np.ndarray, g_r : np.ndarray,
            MSE_l : np.float64, MSE_r : np.float64, n_l : int, n_r : int

        Notes
        -----
        The preconstructed matrices are ``$T^t T$ and $T^t y$``.

        """
        # Select from the available features a subset of features to
        # decide splitting from
        n, p = x.shape
        n_feats = self._get_n_feats(p)

        random_feats = self.rng.choice(p, size=n_feats, replace=False)
        x = x[:, random_feats]

        # If enabled, use optimization instead of brute force
        if 0 <= self.optim_threshold <= n and False:
            # ! DEACTIVATED
            # - Reason: problem with opt.minimize_scalar
            partial_find_Jmin = partial(
                find_Jmin_opt, x=x, y=y, tb=tb, TT=TT, Ty=Ty
            )
        else:
            partial_find_Jmin = partial(
                find_Jmin_sorted, x=x, y=y, tb=tb, TT=TT, Ty=Ty
            )

        # Go through each splitting feature to select optimum splitting
        # point, and save the relevant data in lists
        res_li = {}
        for i in range(n_feats):
            results = partial_find_Jmin(i)
            for k in results:
                if k not in res_li:
                    res_li[k] = []
                res_li[k].append(results[k])

        # Find best splitting fitness found for all splitting features,
        # and return relevant parameters
        i_best = res_li["J"].index(min(res_li["J"]))
        best_res = {k: v[i_best] for k, v in res_li.items()}
        chosen_split_i = int(random_feats[i_best])
        best_res["split_i"] = chosen_split_i

        return best_res

    @_timer_func
    def fit(self, x: np.ndarray, y: np.ndarray, tb: np.ndarray) -> dict:
        """Fit the TBDT.

        Parameters
        ----------
        x : np.ndarray
            Input features with shape `(n, p)`.
        y : np.ndarray
            Anisotropy tensors `b` with shape `(n, 9)` on which to fit
            the TBDT.
        tb : np.ndarray
            Tensor bases with shape `(n, m, 9)`.

        Returns
        -------
        fitted_params : dict
            Dictionary containing lists of node paths, tensor basis
            coefficients `g`, splitting variables and values, `bhat`,
            fitted values for `bhat`. This dictionary is also assigned
            to the TBDT.
            Keys: node_paths, g, split_i, split_v, N_data, n_data,
            MSE, n

        """
        self._log(logging.INFO, f"Fitting '{self.name}'")

        n, m, _ = tb.shape
        # Preconstruct the N_obs matrices for the lhs and rhs terms in
        # the least squares problem
        TT = np.zeros([n, m, m])
        Ty = np.zeros([n, m])
        for i in range(n):
            TT[i] = tb[i] @ tb[i].T + self.gamma * np.eye(m)  # Eq. 3.25 TT
            Ty[i] = tb[i] @ y[i]  # Eq. 3.25 Tb

        # Tree construction
        nodes2add = [(Node(identifier="0"), None, np.arange(n))]
        while nodes2add:
            node, parent, idx = nodes2add.pop()
            g, b, diff = fit_tensor(TT[idx], Ty[idx], tb[idx], y[idx])
            rmse = np.sqrt(np.sum(diff**2))
            n_samples = len(idx)

            if len(idx) <= self.min_samples_leaf:
                # TODO: Integrate with the last part of the while loop
                split_i = None
                split_v = None
                node.tag = node.identifier
                node.data = {
                    "split_i": split_i,
                    "split_v": split_v,
                    "g": list(map(float, g)),
                    "n_samples": n_samples,
                    "RMSE": rmse,
                    "display": (
                        f"feat idx: {split_i}\n"
                        f"feat val: {split_v}\n"
                        f"nb samples: {n_samples:,}\n"
                        f"RMSE: {rmse:.5e}"
                    ),
                }
                self.tree.add_node(node, parent=parent)

                self._log(
                    logging.DEBUG,
                    f"Fitted node '{node.identifier:<35}', "
                    f"RMSE={rmse:.5e}, n_samples={n_samples:>6,}",
                )

                continue

            res = self.create_split(x[idx], y[idx], tb[idx], TT[idx], Ty[idx])
            split_i = res["split_i"]
            split_v = res["split_v"]
            idx_l, idx_r = res["idx_l"], res["idx_r"]

            if len(idx_l) == 0 or len(idx_r) == 0:
                split_i = None
                split_v = None
            else:
                node_l = Node(identifier=f"{node.identifier}0")
                node_r = Node(identifier=f"{node.identifier}1")
                nodes2add.append((node_l, node, idx_l))
                nodes2add.append((node_r, node, idx_r))

            node.tag = node.identifier
            node.data = {
                "split_i": split_i,
                "split_v": split_v,
                "g": list(map(float, g)),
                "display": (
                    f"feat idx: {split_i}\n"
                    f"feat val: {split_v:.5e}\n"
                    f"nb samples: {n_samples:,}\n"
                    f"RMSE: {rmse:.5e}"
                ),
            }
            self.tree.add_node(node, parent=parent)

            self._log(
                logging.DEBUG,
                f"Fitted node '{node.identifier:<35}', "
                f"RMSE={rmse:.5e}, n_samples={n_samples:>6,}",
            )

        self._log(logging.INFO, f"Fitted '{self.name}'")

    @_timer_func
    def predict(
        self, x: np.ndarray, tb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the tensor basis coefficients `g` and use them to
        compute the anisotropy tensor, given the input features `x` and
        the tensor basis `tb`.

        Parameters
        ----------
        x : np.ndarray
            Input features with shape `(n, p)`.
        tb : np.ndarray
            Tensor basess with shape `(n, m, 9)`.

        Returns
        -------
        bhat : np.ndarray
            Anisotropy tensors with shape `(n, 9)`.
        ghat : np.ndarray
            Tensor basis coefficients with shape `(n, m)`.

        """
        n, m, _ = tb.shape
        bhat = np.zeros([n, 9])
        ghat = np.zeros([n, m])

        for i in range(n):
            node = self.tree.get_node("0")
            split_i = node.data["split_i"]
            split_v = node.data["split_v"]
            g = node.data["g"]

            while split_i is not None:
                if x[i, split_i] <= split_i:
                    node = self.tree.get_node(f"{node.identifier}0")
                else:
                    node = self.tree.get_node(f"{node.identifier}1")
                split_i = node.data["split_i"]
                split_v = node.data["split_v"]
                g = node.data["g"]

            ghat[i] = g
            for j in range(m):
                bhat[i] += g[j] * tb[i, j]

        return ghat, bhat


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
        self.rng = default_rng(random_state)
        self.logger = logger
        self.tbdt_kwargs = tbdt_kwargs if tbdt_kwargs is not None else {}

        self.trees = [
            TBDT(
                name=f"{self.name}_TBDT-{i}",
                random_state=self.rng.choice(1000),
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
        if self.logger is not None:
            self.logger.log(level, message, *args, **kwargs)

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
        rng = self.trees[i_tree].rng
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


if __name__ == "__main__":
    pass
