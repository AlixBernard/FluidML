#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-03-24 15:58:14
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-13 01:08:56

"""Classes for the Tensor Basis Decision Tree (TBDT) and the Tensor
Basis Random Forest (TBRF).

Glossary:
    - `n` is the total number of samples
    - `p` is the total number of features
    - `m` is the number of tensors in the tensor basis
    - `s` is the number of TBDTs in the TBRF

"""

# Built-in packages
import json
import logging
import multiprocessing as mp
from time import time
from functools import partial
from pathlib import Path

# Third party packages
import numpy as np
from scipy import optimize as opt
from numpy.random import default_rng

# Local packages
from fluidml import utils

__all__ = [
    "fit_tensor",
    "obj_func_J",
    "find_Jmin_sorted",
    "find_Jmin_opt",
    "TBDT",
    "TBRF",
]


def fit_tensor(
    TT: np.ndarray,
    Ty: np.ndarray,
    tb: np.ndarray,
    y: np.ndarray,
) -> tuple:
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
) -> float:
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
    i_float : float | None
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
    feat_i: int,
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
    asort = np.argsort(x[:, feat_i])
    asort_back = np.argsort(asort)

    x_sorted = x[asort]
    y_sorted = y[asort]
    tb_sorted = tb[asort]
    TT_sorted = TT[asort]
    Ty_sorted = Ty[asort]

    # Flag which activates when all features are the same
    # e.g. due to feature capping
    obs_identical = True if np.all(x == x[0]) else False

    results = {"J": 1e10}  # Initial cost set really large

    if obs_identical:
        # Exception: all observations are equal
        # Terminate further splitting of the node
        J, extra = obj_func_J(y_sorted, tb_sorted, TT_sorted, Ty_sorted)
        i_r_sorted = np.ones(n, dtype=bool)

        results["J"] = J
        results["split_var"] = feat_i
        results["split_val"] = 0.5 * (
            x_sorted[0, feat_i] + x_sorted[1, feat_i]
        )
        results["i_l"] = i_r_sorted
        results["i_r"] = i_r_sorted
        results["g_l"] = extra["g"]
        results["g_r"] = extra["g"]
        results["MSE_l"] = 0
        results["MSE_r"] = J
        results["n_l"] = 0
        results["n_r"] = n

    else:
        for i in range(1, n):
            J_tmp, extra = obj_func_J(
                y_sorted, tb_sorted, TT_sorted, Ty_sorted, i_float=i
            )

            if J_tmp < results["J"]:
                i_l_sorted = np.zeros(n, dtype=bool)
                i_l_sorted[:i] = True
                i_r_sorted = ~i_l_sorted

                results["J"] = J_tmp
                results["split_var"] = feat_i
                results["split_val"] = 0.5 * (
                    x_sorted[i - 1, feat_i] + x_sorted[i, feat_i]
                )
                results["i_l"] = i_l_sorted[asort_back]
                results["i_r"] = i_r_sorted[asort_back]
                results["g_l"] = extra["g_l"]
                results["g_r"] = extra["g_r"]
                results["MSE_l"] = np.mean(extra["diff_l"] ** 2)
                results["MSE_r"] = np.mean(extra["diff_r"] ** 2)
                results["n_l"] = i
                results["n_r"] = n - i

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
        "split_var": idx,
        "split_val": (
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
    name : str
        Name of the tree used as its string representation.
        Default is "TBDT".
    max_depth : int
        The maximum depth of the tree.
        Default is 400.
    min_sample_leaf : int
        Minimum number of samples required to be a leaf node.
    max_features : int | float | str | None
        Number of features to consider when looking for the best split:
            - if int then consider `max_features`
            - if float then consider `round(max_features * m)`
            - if 'sqrt' then consider `srqt(m)`
            - if 'log2' then consider `log2(m)`
            - if None then consider `m`
        where `m` is the total number of features.
        Default is 'sqrt'.
    gamma : float
        The regularization parameter gamma.
        Default is 1., ie. no regularization.
    optim_threshold : int
        Threshold for which if the number of points is below, brute
        force will be used and optim otherwise, if it is -1 then
        optimization is disabled.
        Default is 1_000
    random_state : int | None
        Controls randomness when sampling the features.
    logger : logging.logger
        Logger to output details.
        Default is None.

    Methods
    -------
    create_split
    fit
    get_node_indices
    predict
    read
    save

    """

    def __init__(
        self,
        name: str = "TBDT",
        max_depth: int = 400,
        min_samples_leaf: int = 1,
        max_features: "int | float | str | None" = "sqrt",
        gamma: float = 1e0,
        optim_threshold: int = 1_000,
        random_state: "int | None" = None,
        logger: "logging.logger | None" = None,
    ):
        self.name = name
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.gamma = gamma
        self.optim_threshold = optim_threshold
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.logger = logger

    def __str__(self):
        s = f"{self.name}"
        return s

    def __repr__(self):
        attrs2skip = ["logger"]

        str_attrs = []
        for k, v in sorted(self.__dict__.items()):
            if k not in attrs2skip:
                str_attrs.append(f"{k}: {v!r}")

        obj_repr = f"TBDT({', '.join(str_attrs)})"
        return obj_repr

    def _timer_func(func):
        def wrap_func(self, *args, **kwargs):
            t1 = time()
            result = func(self, *args, **kwargs)
            t2 = time()
            self.logger.debug(
                f"Function {func.__name__!r} executed in {(t2-t1):.4f}s"
            )
            return result

        return wrap_func

    def _get_n_feats(self, m: int) -> int:
        """Compute the number of features to consider to perform each
        split (cf. attribute `max_features`).

        Parameters
        ----------
        p : int
            Total number of features available.

        Returns
        -------
        n_feats : int

        """
        if self.max_features is None:
            n_split_feats = p
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                n_split_feats = int(np.ceil(np.sqrt(m)))
            elif self.max_features == "log2":
                n_split_feats = int(np.ceil(np.log2(m)))
            else:
                raise ValueError(
                    f"The attribute `max_features` must be 'sqrt' or "
                    f"'log2' when it is a str"
                )
        elif isinstance(self.max_features, int):
            if 0 <= self.max_features <= m:
                n_split_feats = self.max_features
            else:
                raise ValueError(
                    f"The attribute `max_features` must respect"
                    f" 0 <= `max_features` <= {m} when it is an int"
                )
        elif isinstance(self.max_features, float):
            if 0.0 <= self.max_features <= 1.0:
                n_split_feats = round(self.max_features * m)
            else:
                raise ValueError(
                    f"The attribute `max_features` must respect"
                    f" 0 <= `max_features` <= 1 when it is a float"
                )
        else:
            raise TypeError(
                f"The attribute `max_features` is of incorrect type"
            )

        if n_split_feats == 0:
            raise ValueError(f"Not enough features selected for splitting")

        return n_split_feats

    def save(self, path: Path):
        """Save the TBDT as a JSON file containing its attributes.

        Parameters
        ----------
        path : Path

        """
        attrs2skip = ["logger", "rng"]
        json_attrs = {}
        for k, v in self.__dict__.items():
            if k in attrs2skip:
                continue
            # ! json_attrs[k] = utils.jsonify(v)
        try:
            with open(path, "w") as file:
                json.dump(json_attrs, file, indent=4)
        except TypeError as exc:
            self.logger.error(
                f"Could not save '{self}' due to:\n\tTypeError: {exc}"
            )
            return
        if self.logger is not None:
            self.logger.info(f"Saved '{self}' as: '{path}'")

    def load(self, path: Path):
        """Load the TBDT from a JSON file containing its attributes.

        Parameters
        ----------
        path : Path

        """
        with open(path, "r") as file:
            json_attrs = json.load(file)
        for k, v in json_attrs:
            self.__setattr__(k, v)
        if self.logger is not None:
            self.logger.info(f"Loaded '{self}' from: '{path}'")

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
        matrices for $T^t T$ and $T^t y$.

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
            J : np.float64
                The value of the cost function when fitting the tensors
                for the feature `idx`.
            split_var : list
                Feature index.
            split_val : list
                The value making the split.
            i_l : list
                Indices of the points in the left child.
            i_r : list
                Indices of the points in the right child.
            g_l : np.ndarray
                Optimum value for the tensor basis coefficients of the
                left childwith shape `(m,)`.
            g_r : np.ndarray
                Optimum value for the tensor basis coefficients of the
                right child with shape
                `(m,)`.
            MSE_l : np.float64
                Mean Square Error in the left child node.
            MSE_r : np.float64
                Mean Square Error in the right child node.
            n_l : int
                Number of points in the left child node.
            n_r : int
                Number of points in the right child node.

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
        chosen_split_var = random_feats[i_best]
        best_res["split_var"] = chosen_split_var

        return best_res

    def get_node_indices(self, node: np.ndarray, node_path: np.ndarray):
        """Obtain the train data indices which are binned in a node
        referenced by `node_path`. A node path is a list of 0 and 1,
        denoting left split and right split respectively.

        Parameters
        ----------
        node : np.ndarray[int]
            ???
        node_path : np.ndarray[in]
            The path to the node ordered from the root of the tree,
            where 0 and 1 denotes left split and right split
            respectively.

        Returns
        -------
        indices : np.ndarray[bool]
            Indices as booleans indicating which training samples are
            active at `node`.

        """
        if len(node_path.shape) == 1:
            indices = np.array(node_path == node)

        else:
            indices = np.zeros(node_path.shape[1], dtype=bool)
            for i in range(node_path.shape[1]):
                # Check if the node path for a given training data point
                # corresponds to the given node
                if np.all(node == node_path[:, i]):
                    indices[i] = True

        return indices

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
            Keys: node_paths, g, split_var, split_val, N_data, n_data,
            MSE, n

        """
        if self.logger is not None:
            self.logger.info(f"Fitting '{self.name}'")

        n, m, _ = tb.shape
        # Preconstruct the N_obs matrices for the lhs and rhs terms in
        # the least squares problem
        TT = np.zeros([n, m, m])
        Ty = np.zeros([n, m])
        for i in range(n):
            TT[i] = tb[i] @ tb[i].T + self.gamma * np.eye(m)  # Eq. 3.25 TT
            Ty[i] = tb[i] @ y[i]  # Eq. 3.25 Tb

        # Create fitted_params dict, which contains the nodes and
        # corresponding values which are necessary for predictions later
        # node_paths : list
        #   All nodes which are created. 0 indicates a left path, 1 a
        #   right path. 2 indicates the feature ended in a terminal node
        # g : list
        #   All least squares coefficients at each node
        # split_var : list
        #   The variable used for splitting
        # split_val : list
        #   The value ""
        # N_data:       total amount of datapoints used for training
        # n_data:       amount of data points in each node
        # ? n

        fitted_params = {
            "node_paths": [],
            "g": [],
            "split_var": [],
            "split_val": [],
            "N_data": n,
            "n_data": [],
            "MSE": [],
            "n": [],
        }

        # Queue which contains the child nodes which need to be resolved
        # in the next `i`-th iteration
        tmp_queue = []  # Temporary variable which stores child nodes
        # `queue` is set to `tmp_queue` at next iteration (ie. child
        # nodes are now current nodes)
        queue = []

        for i in range(self.max_depth):

            # TODO: merge creation of root node (i=0) and further nodes
            if i == 0:
                if self.logger is not None:
                    self.logger.debug(
                        f"Building '{self}', level {i:>3}, "
                        f"amount of nodes: {1:>4}"
                    )

                g, _, diff = fit_tensor(TT, Ty, tb, y)
                fitted_params["MSE"].append(mse := np.mean(diff**2))
                fitted_params["n"].append(n)

                # root node: initialization and first split
                res = self.create_split(x, y, tb, TT, Ty)

                node_path = np.array(res["i_r"] * 1)

                # Add all necessary information to fitted_params
                fitted_params["g"].extend([res["g_l"], res["g_r"]])
                fitted_params["MSE"].extend([res["MSE_l"], res["MSE_r"]])
                fitted_params["n"].extend([res["n_l"], res["n_r"]])
                fitted_params["split_var"].append(res["split_var"])
                fitted_params["split_val"].append(res["split_val"])
                fitted_params["n_data"].append(n)

                # Check left child node
                # Min samples leaf reached, no more splitting possible
                if res["i_l"].sum() == self.min_samples_leaf:
                    pass

                # Empty node should not happen, print error and abort
                elif res["i_l"].sum() == 0:
                    if self.logger is not None:
                        self.logger.warning("Error: indices left empty")
                    break

                # Left/Right bin indices are the same, can happen when
                # input features are equal with only two features to
                # choose from
                elif all(res["i_l"] == res["i_r"]):
                    pass

                # Not terminal node, add to queue for further splitting
                else:
                    tmp_queue.append([0])
                    fitted_params["node_paths"].append([0])

                # Check right child node
                if res["i_r"].sum() == self.min_samples_leaf:
                    pass
                elif res["i_r"].sum() == 0:
                    if self.logger is not None:
                        self.logger.warning("Error: indices right empty")
                    break
                elif all(res["i_r"] == res["i_l"]):
                    pass
                else:
                    tmp_queue.append([1])
                    fitted_params["node_paths"].append([1])

            else:
                if self.logger is not None and queue:
                    self.logger.debug(
                        f"Building '{self}', level {i:>3}, "
                        f"amount of nodes: {len(queue):>4}"
                    )

                # New node_path variables for each node will be added to
                # `tmp_node_path`. After going though each node,
                # node_path will be set to `tmp_node_path`. A 2 in
                # `node_path` indicates the data is fully split
                tmp_node_path = np.vstack(
                    [
                        node_path,
                        2 * np.ones([1, fitted_params["N_data"]]),
                    ]
                )

                # Go through nodes
                for j in range(len(queue)):

                    # Get current node
                    node = np.array(queue[j])

                    # Get an array of booleans with training data points
                    # corresponding to the current node. Maybe a nicer
                    # solution can be found for this
                    indices = self.get_node_indices(node, node_path)

                    # Split data into left and right bin
                    res = self.create_split(
                        x[indices],
                        y[indices],
                        tb[indices],
                        TT[indices],
                        Ty[indices],
                    )

                    # Add left and right split to fitted_params
                    fitted_params["g"].extend([res["g_l"], res["g_r"]])
                    fitted_params["MSE"].extend([res["MSE_l"], res["MSE_r"]])
                    fitted_params["n"].extend([res["n_l"], res["n_r"]])
                    fitted_params["split_var"].append(res["split_var"])
                    fitted_params["split_val"].append(res["split_val"])
                    fitted_params["n_data"].append(n)

                    # Check whether the left and right splits are
                    # terminal nodes, and add child nodes to queue
                    # One datapoint -> no more splitting possible:
                    left_node_n = x[indices][res["i_l"]].shape[0]
                    if left_node_n <= self.min_samples_leaf:
                        pass
                    # Left/Right bin indices are the same, can happen
                    # when input features are equal
                    elif all(res["i_l"] == res["i_r"]):
                        pass
                    # Empty node should not happen, just in case for
                    # debugging:
                    elif left_node_n == 0:
                        if self.logger is not None:
                            self.logger.warning("Error: indices left empty")
                        break
                    # Otherwise, create child node and add to queue and
                    # fitted_params
                    else:
                        tmp_queue.append(queue[j] + [0])
                        fitted_params["node_paths"].append(queue[j] + [0])

                    right_node_n = x[indices][res["i_r"]].shape[0]
                    if right_node_n <= self.min_samples_leaf:
                        pass
                    elif all(res["i_r"] == res["i_l"]):
                        pass
                    elif right_node_n == 0:
                        if self.logger is not None:
                            self.logger.warning("Error: indices right empty")
                        break
                    else:
                        tmp_queue.append(queue[j] + [1])
                        fitted_params["node_paths"].append(queue[j] + [1])

                    tmp_node_path[i, indices] = res["i_r"]

                # Update the node_paths of all the training variables
                node_path = tmp_node_path

            # Add child nodes to current queue
            queue = tmp_queue
            tmp_queue = []

        self.fitted_params = fitted_params

        if self.logger is not None:
            self.logger.info(f"Fitted '{self.name}'")

    @_timer_func
    def predict(self, x: np.ndarray, tb: np.ndarray) -> tuple:
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
            Tensor basis coefficients with shape `(n, m,)`.

        """
        n, m, _ = tb.shape
        bhat = np.zeros([n, 9])
        ghat = np.zeros([n, m])

        params = self.fitted_params
        for i in range(n):
            node_path = []
            # Start at root, add 0 or 1 to input feature node_path
            # depending on left/right split
            if x[i, params["split_var"][0]] <= params["split_val"][0]:
                node_path.append(0)
            else:
                node_path.append(1)

            # While the given node_path is present in `params`,
            # keep binning the data
            while node_path in params["node_paths"]:
                # Return the index of the node in `params`:
                i_current_node = params["node_paths"].index(node_path)

                if (
                    x[i, params["split_var"][i_current_node + 1]]
                    <= params["split_val"][i_current_node + 1]
                ):
                    node_path.append(0)
                else:
                    node_path.append(1)

            # Remove last element of the split, and save the last split
            # for prediction
            last_split = node_path[-1]
            node_path = node_path[:-1]

            # Get index prediction giving corresponding values for `g`
            i_pred = params["node_paths"].index(node_path)

            if last_split == 0:
                g = params["g"][2 * (i_pred + 1)]
            else:
                g = params["g"][2 * (i_pred + 1) + 1]

            tmp_b = np.zeros(9)
            for j in range(m):
                tmp_b += g[j] * tb[i, j]
                ghat[i, j] = g[j]

            bhat[i] = tmp_b

        return ghat, bhat


class TBRF:
    """Tensor Basis Random Forest.

    Attributes
    ----------
    name : str
        Name of the forest used as its string representation.
        Default is "TBRF".
    n_estimators : int
        Number of trees to build in the forest.
    max_features : int | float | str | None
        Number of features to consider when looking for the best split:
            - if int then consider `max_features`
            - if float then consider `round(max_features * m)`
            - if 'sqrt' then consider `ceil(srqt(m))`
            - if 'log2' then consider `ceil(log2(m))`
            - if None then consider `m`
        where `m` is the total number of features.
        Default is 'sqrt'.
    bootstrap : bool
        Whether bootstrap samples are used when building trees. If
        False, the whole dataset is used to build each tree.
        Default is True.
    max_samples : int | float | None
        If bootstrap is True, the number of samples to draw from x to
        to train each tree:
            - if None then draw `n` samples
            - if int then draw `max_samples` samples
            - if float then draw `round(max_samples * n)` samples
        where `n` is the total number of sample.
        Default is None.
    n_jobs : int | None
        The number of jobs to run in parallel. The methods `fit` and
        `predict` are parallelized over the trees. None means 1.
        -1 means using all processors.
        Default is None.
    random_state : int | None
        Controls both the randomness of the bootstrapping of the samples
        used when building trees (if bootstrap=True) and the sampling of
        the features to consider when looking for the best split at each
        node (if max_features < m).
        Default is None.
    tbdt_kwargs : dict | None
        Keyword arguments for the TBDTs.
    logger : logging.logger
        Logger to output details.
        Default is None.

    Methods
    -------
    fit
    predict

    """

    # TODO: add oob score?

    def __init__(
        self,
        name: str = "TBRF",
        n_estimators: int = 10,
        bootstrap: bool = True,
        max_samples: "int | float | None" = None,
        n_jobs: "int | None" = None,
        random_state: "int | None" = None,
        logger: "logging.logger | None" = None,
        tbdt_kwargs: "dict | None" = None,
    ):
        self.name = name
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.logger = logger
        self.tbdt_kwargs = tbdt_kwargs if tbdt_kwargs is not None else {}

        self.trees = [
            TBDT(
                name=f"TBDT_{i}",
                random_state=self.rng.choice(1000),
                **self.tbdt_kwargs,
            )
            for i in range(self.n_estimators)
        ]
        if self.logger is not None:
            self.logger.info(f"Initialized {self.n_estimators} TBDTs")

    def __len__(self):
        return len(self.trees)

    def __str__(self):
        s = f"{self.name}"
        return s

    def __repr__(self):
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

    def _timer_func(func):
        def wrap_func(self, *args, **kwargs):
            t1 = time()
            result = func(self, *args, **kwargs)
            t2 = time()
            self.logger.debug(
                f"Function {func.__name__!r} executed in {(t2-t1):.4f}s"
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

        """
        if self.max_samples is None:
            n_samples = n
        elif isinstance(self.max_samples, int):
            n_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            n_samples = round(self.max_samples * n)
        else:
            raise ValueError(
                f"The {self.max_samples} is not recognized"
                f" for the attribute `max_samples`"
            )
        return n_samples

    def save(self, dir_path: Path):
        """Save the TBRF as a folder containing the JSON files of its
        attributes and the TBDTs' attributes. The TBRF JSON file only
        has the list of names of the TBDTs in its field "trees".

        Parameters
        ----------
        dir_path : Path
                Path to the folder where to save the TBRF's and TBDTs'
                files, the folder will be deleted and recreated.

        """
        if not dir_path.exists():
            dir_path.mkdir()
            if self.logger is not None:
                self.logger.info(f"Created the folder: '{dir_path}'")

        for tbdt in self.trees:
            tbdt_filename = f"{tbdt}.json"
            tbdt_path = dir_path / tbdt_filename
            tbdt.save(tbdt_path)

        json_attrs = self.__dict__.copy()
        json_attrs["trees"] = [tbdt.name for tbdt in self.trees]
        # ! json_attrs = utils.replace_types_in_dict_list(
        # -     json_attrs,
        # -     logging.Logger,
        # -     lambda log: str(log),
        # -     recursive_dict=True,
        # -     recursive_list=True,
        # - )
        del json_attrs["logger"]
        del json_attrs["rng"]

        tbrf_filename = f"{self}.json"
        tbrf_path = dir_path / tbrf_filename
        with open(tbrf_path, "w") as file:
            # json.dump(json_attrs, file, indent=4)
            file.write("TBRF.save disabled")
        if self.logger is not None:
            self.logger.info(f"Saved '{self}' as: '{tbrf_path}'")

    def load(self, dir_path: Path):
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
        if self.logger is not None:
            self.logger.info(f"Loaded '{self}' from: '{tbrf_path}'")

        for name in tbdt_names:
            tbdt_filename = f"{name}.json"
            tbdt_path = dir_path / tbdt_filename
            self.trees[k] = TBDT().load(tbdt_path)

    @_timer_func
    def fit(self, x: np.ndarray, y: np.ndarray, tb: np.ndarray) -> dict:
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

        """
        if self.logger is not None:
            self.logger.info(f"Fitting all trees of '{self.name}'")

        jobs = (self.n_jobs,) if self.n_jobs != -1 else ()
        with mp.Pool(*jobs) as pool:
            res = [
                pool.apply_async(self._fit_tree, (i, x, y, tb))
                for i in range(len(self))
            ]
            self.trees = [r.get() for r in res]

        if self.logger is not None:
            self.logger.info(f"Fitted all trees of '{self.name}'")

    def _fit_tree(
        self, i_tree: int, x: np.ndarray, y: np.ndarray, tb: np.ndarray
    ) -> None:
        """Fit the specified tree."""
        n = len(x)
        n_samples = self._get_n_samples(n)
        rng = self.trees[i_tree].rng
        if self.bootstrap:
            idx_sampled = rng.choice(n, size=n_sampled, replace=True)
        else:
            idx_sampled = np.arange(n)

        x_sampled = x[idx_sampled]
        y_sampled = y[idx_sampled]
        tb_sampled = tb[idx_sampled]
        self.trees[i_tree].fit(x_sampled, y_sampled, tb_sampled)
        
        return self.trees[i_tree]

    @_timer_func
    def predict(self, x: np.ndarray, tb: np.ndarray, method: str = "mean"):
        """Tensor Basis Random Forest predictions given input features
        `x_test` and tensor basis `tb_test`, make predictions for the
        anisotropy tensor `b` using its fitted trees.

        Parameters
        ----------
        x : np.ndarray
            Input features with shape `(n, p)`.
        tb : np.ndarray
            Tensor bases with shape `(n, m, 9)`.
        method : str
            How to compute the TBRF prediction from all the TBDT
            predictions, possible values are 'mean' and 'median'.

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

        """
        n, m, _ = tb.shape

        # Initialize predictions
        b_trees = np.zeros([len(self), n, 9])
        g_trees = np.zeros([len(self), n, m])

        jobs = (self.n_jobs,) if self.n_jobs != -1 else ()
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

        if self.logger is not None:
            self.logger.info("Predicted the anysotropy tensor 'b'")

        return g_trees, b_trees, b

    def _predict_tree(self, i_tree: int, x: np.ndarray, tb: np.ndarray):
        """Predict from the tree specified."""
        g, b = self.trees[i_tree].predict(x, tb)
        return g, b


if __name__ == "__main__":
    pass
