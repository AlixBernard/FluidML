"""Classes for the Tensor Basis Decision Tree (TBDT).

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
]


# Built-in packages
import json
import logging
from collections import deque, OrderedDict
from time import perf_counter
from functools import partial
from pathlib import Path

# Third party packages
import numpy as np
from scipy import optimize as opt
from numpy.random import default_rng

# Local packages
from fluidml.models import Tree, Node


def _log(
    level: int,
    message: str,
    logger: logging.Logger | None = None,
    *args,
    **kwargs,
) -> None:
    if logger is not None:
        logger.log(level, message, *args, **kwargs)


def fit_tensor(
    TT: np.ndarray,
    Ty: np.ndarray,
    tb: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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

    """
    n, m, _ = TT.shape
    lhs = TT.sum(axis=0)
    rhs = Ty.sum(axis=0)

    # Solve Eq. 3.25
    ghat, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    bhat = np.einsum("j,ijk->ik", ghat, tb)

    return ghat, bhat


def cost_J(y: np.ndarray, bhat: np.ndarray) -> tuple[float, dict]:
    """Objective function which minimize the MSE difference w.r.t. the
    target `y`.

    Parameters
    ----------
    y : np.ndarray
        The target.
    bhat : np.ndarray
        The target obtained.

    Returns
    -------
    J : float
        The value of the cost function.

    """
    diff = y - bhat
    J = np.mean(diff**2)
    return J


def obj_func_J(
    y_sorted: np.ndarray,
    tb_sorted: np.ndarray,
    TT_sorted: np.ndarray,
    Ty_sorted: np.ndarray,
    i: int | None = None,
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
    i : int or None
        If not None, index to use when splitting the data.

    Returns
    -------
    J : float
        The value of the cost function.
    extra : dict
        Dictionarry containing the following extra data: g_l, g_r,
        diff_l, diff_r, diff.

    """
    if i is None:
        ghat, bhat = fit_tensor(TT_sorted, Ty_sorted, tb_sorted, y_sorted)
        diff = y_sorted - bhat
        extra = {"g": ghat, "diff": diff}
    else:
        g_l, b_l = fit_tensor(
            TT_sorted[:i], Ty_sorted[:i], tb_sorted[:i], y_sorted[:i]
        )
        diff_l = y_sorted[:i] - b_l
        g_r, b_r = fit_tensor(
            TT_sorted[i:], Ty_sorted[i:], tb_sorted[i:], y_sorted[i:]
        )
        diff_r = y_sorted[i:] - b_r

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
    split_feat_i: int,
    x: np.ndarray,
    y: np.ndarray,
    tb: np.ndarray,
    TT: np.ndarray,
    Ty: np.ndarray,
) -> tuple[dict[str, int | float]]:
    """Find optimum splitting point for the feature with index
    `feat_i`. Data is pre-sorted to save computational costs($n log(n)$
    instead of $n^2$).

    Parameters
    ----------
    split_feat_i : int
        Index of the feature on which to find the optimum splitting
        point.
    x : np.ndarray
        Input features with shape `(n, p)`.
    y : np.ndarray
        Anisotropy tensor `b` (target) on which to fit the tree with
        shape `(n, 9)`.
    tb : np.ndarray
        Tensor bases with shape `(n, m, 9)`.
    TT : np.ndarray
        Preconstructed matrix $transpose(T)*T$.
    Ty : np.ndarray
        Preconstructed matrix $transpose(T)*f$.

    Returns
    -------
    split_data : dict[str, int | float]
        Data containing the value of the cost function J with the split,
        the index of the feature on which the split is made, and the
        value determining the split.
    left_data : dict[str, int | float]
        Data containing the indices, the value of `ghat`, the mean
        square error, and the number of the sample that are present in
        the left node from the split.
    right_data : dict[str, int | float]
        Data containing the indices, the value of `ghat`, the mean
        square error, and the number of the sample that are present in
        the right node from the split.


    """
    n = len(x)
    asort = np.argsort(x[:, split_feat_i])
    TT_sorted = TT[asort]
    Ty_sorted = Ty[asort]
    tb_sorted = tb[asort]
    y_sorted = y[asort]

    best_J = 1e12
    for i in range(1, n):
        ghat_sorted_l, bhat_sorted_l = fit_tensor(
            TT_sorted[:i], Ty_sorted[:i], tb_sorted[:i], y_sorted[:i]
        )
        diff_sorted_l = y_sorted[:i] - bhat_sorted_l
        ghat_sorted_r, bhat_sorted_r = fit_tensor(
            TT_sorted[i:], Ty_sorted[i:], tb_sorted[i:], y_sorted[i:]
        )
        diff_sorted_r = y_sorted[i:] - bhat_sorted_r
        diff_sorted = np.vstack([diff_sorted_l, diff_sorted_r])
        J = np.mean(diff_sorted**2)
        if J < best_J:
            best_i, best_J = i, J
            best_ghat_l, best_ghat_r = ghat_sorted_l, ghat_sorted_r
            best_diff_l, best_diff_r = diff_sorted_l, diff_sorted_r

    split_data = {
        "J": best_J,
        "split_i": split_feat_i,
        "split_v": 0.5 * x[asort][best_i - 1 : best_i + 1, split_feat_i].sum(),
    }
    left_data = {
        "idx": asort[:best_i],
        "ghat": best_ghat_l,
        "MSE": np.mean(best_diff_l**2),
        "n": best_i,
    }
    right_data = {
        "idx": asort[best_i:],
        "ghat": best_ghat_r,
        "MSE": np.mean(best_diff_r**2),
        "n": n - best_i,
    }

    return split_data, left_data, right_data


def find_Jmin_opt(
    split_feat_i: int,
    x: np.ndarray,
    y: np.ndarray,
    tb: np.ndarray,
    TT: np.ndarray,
    Ty: np.ndarray,
) -> dict[str, int | float | np.ndarray]:
    """Find optimum splitting point by using an optimization routine.

    Parameters
    ----------
    split_feat_i : int
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
    results : dict[str, int | float | None]
        Same as `best_res` in higher method.

    """
    n = len(x)
    asort = np.argsort(x[:, split_feat_i])
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
    best_i = int(res.x)

    # TODO: in case optimization algorithm does not work it
    # - returns 0, needs further testing
    if best_i == 0:
        best_i = 1

    # Find all relevant parameters for the minimum which was found
    # ? Maybe this can be improved as it is redundant
    J, extra = obj_func_J(y_sorted, tb_sorted, TT_sorted, Ty_sorted, i=best_i)
    i_l_sorted = np.zeros(n, dtype=bool)
    i_l_sorted[:best_i] = True
    i_r_sorted = ~i_l_sorted

    results = {
        "J": J,
        "split_i": split_feat_i,
        "split_v": 0.5 * x_sorted[best_i - 1 : best_i + 1, split_feat_i].sum(),
        "i_l": i_l_sorted[asort_back],
        "i_r": i_r_sorted[asort_back],
        "g_l": extra["g_l"],
        "g_r": extra["g_r"],
        "MSE_l": np.mean(extra["diff_l"] ** 2),
        "MSE_r": np.mean(extra["diff_r"] ** 2),
        "n_l": best_i,
        "n_r": n - best_i,
    }

    if obs_identical:
        # Right and left splits are made equal. This leads to
        # termination of the branch later on in `self.fit()`
        results["g_l"] = extra["g_r"]
        results["i_l"] = i_r_sorted[asort_back]
        results["n_l"] = 0
        results["MSE_l"] = 0

    return results


def create_split(
    x: np.ndarray,
    y: np.ndarray,
    tb: np.ndarray,
    TT: np.ndarray,
    Ty: np.ndarray,
    feats_idx: np.ndarray,
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
    feats_idx : np.ndarray
        Indices of the features chosen to create the split from.

    Returns
    -------
    split_data : dict[str, int | float]
        Data containing the value of the cost function J with the split,
        the index of the feature on which the split is made, and the
        value determining the split.
    left_data : dict[str, int | float]
        Data containing the indices, the value of `ghat`, the mean
        square error, and the number of the sample that are present in
        the left node from the split.
    right_data : dict[str, int | float]
        Data containing the indices, the value of `ghat`, the mean
        square error, and the number of the sample that are present in
        the right node from the split.

    Notes
    -----
    The preconstructed matrices are ``$T^t T$ and $T^t y$``.

    """
    n_feats = len(feats_idx)
    x = x[:, feats_idx]

    # # If enabled, use optimization instead of brute force
    # if 0 <= self.optim_threshold <= n:
    #     # ! DEACTIVATED
    #     # - Reason: problem with opt.minimize_scalar
    #     partial_find_Jmin = partial(
    #         find_Jmin_opt, x=x, y=y, tb=tb, TT=TT, Ty=Ty
    #     )
    # else:
    partial_find_Jmin = partial(
        find_Jmin_sorted, x=x, y=y, tb=tb, TT=TT, Ty=Ty
    )

    # Go through each splitting feature to select optimum splitting
    # point, and save the relevant data in lists
    best_J = 1e12
    for i in range(n_feats):
        split_data, left_data, right_data = partial_find_Jmin(i)
        if split_data["J"] < best_J:
            best_split_data = split_data
            best_left_data = left_data
            best_right_data = right_data
            best_i = i
            best_J = split_data["J"]
    best_split_data["split_i"] = int(feats_idx[best_i])  # Cast np.int to int

    return best_split_data, best_left_data, best_right_data


class TBDT:
    """Tensor Basis Decision Tree.

    Attributes
    ----------
    name : str, default='TBDT'
        Name of the tree used as its string representation.
    max_depth : int, default=400
        The maximum depth of the tree.
    min_samples_split : int
        Minimum number of samples required to consider to split a node.
    min_samples_leaf : int
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

    Methods
    -------
    to_dict
    from_dict
    save_to_json
    load_from_json
    to_graphviz
    fit
    predict

    """

    def __init__(
        self,
        name: str = "TBDT",
        max_depth: int = 400,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        gamma: float = 1e0,
        optim_threshold: int = 1_000,
    ) -> None:
        self.name = name
        self.tree = Tree()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.gamma = gamma
        self.optim_threshold = optim_threshold

    def __str__(self) -> str:
        s = f"{self.name}"
        return s

    def __repr__(self) -> str:
        str_attrs = [f"{k}: {v!r}" for k, v in sorted(self.__dict__.items())]
        obj_repr = f"TBDT({', '.join(str_attrs)})"
        return obj_repr

    def __eq__(self, tbdt2) -> bool:
        attrs2skip = ["tree"]
        for k, v in self.__dict__.items():
            if k in attrs2skip:
                continue
            if v != tbdt2.__dict__[k]:
                return False
        if self.tree != tbdt2.tree:
            return False
        return True

    def _timer_func(func):
        def wrap_func(self, *args, **kwargs):
            t1 = perf_counter()
            result = func(self, *args, **kwargs)
            t2 = perf_counter()
            logger = kwargs.get("logger")
            _log(
                logging.DEBUG,
                f"Method {self.name}.{func.__name__} executed in "
                f"{(t2-t1):.2f}s",
                logger,
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
        int
            Number of features to consider for each split.

        Raises
        ------
        RuntimeError
            If the value of the attribute `max_features` is not one of
            {'sqrt', 'log2'}, an int between 1 and `p`, or a float
            between 0 and 1.


        """
        FEAT_SUBSET_SIZES = {
            None: lambda x: x,
            "sqrt": lambda x: int(np.ceil(np.sqrt(x))),
            "log": lambda x: int(np.ceil(np.log(x))),
            "log2": lambda x: int(np.ceil(np.log2(x))),
            "log10": lambda x: int(np.ceil(np.log10(x))),
        }
        if isinstance(self.max_features, int):
            if 1 <= self.max_features <= p:
                return self.max_features
        if isinstance(self.max_features, float):
            if 0.0 < self.max_features <= 1.0:
                if round(self.max_features * p) >= 1:
                    return int(np.ceil((self.max_features * p)))
        try:
            return FEAT_SUBSET_SIZES[self.max_features](p)
        except IndexError:
            pass
        raise ValueError(
            f"The attribute `max_features` (={self.max_features}) has an "
            f"incorrect value."
        )

    def to_dict(self) -> dict:
        """Returns the TBDT as its dict representation."""
        d = {k: v for k, v in self.__dict__.items() if k != "tree"}
        d["nodes"] = {
            node.identifier: {"tag": node.tag, "data": node.data}
            for node in self.tree.all_nodes()
        }
        return d

    @classmethod
    def from_dict(cls, tbdt_dict: dict):
        """Create a `TBDT` from its dict representation.

        Parameters
        ----------
        tbdt_dict : dict
            The dict representation of the TBDT to create.

        """
        tbdt_kwargs = {
            k: v for k, v in tbdt_dict.items() if k not in ["nodes"]
        }
        tbdt = TBDT(**tbdt_kwargs)
        nodes2add = []
        for k, v in tbdt_dict["nodes"].items():
            identifier, tag, data = k, v["tag"], v["data"]
            node = Node(identifier, tag, data=data)
            parent = identifier[:-1] if len(identifier) > 1 else None
            nodes2add.append((node, parent))
        nodes2add.sort(key=lambda node_tuple: len(node_tuple[0].identifier))
        for node, parent in nodes2add:
            tbdt.tree.add_node(node, parent=parent)
        return tbdt

    def save_to_json(self, path: Path) -> None:
        """Save the TBDT as a JSON file containing its dict representation.

        Parameters
        ----------
        path : Path

        """
        tbdt_dict = OrderedDict(self.to_dict())
        tbdt_dict.move_to_end("nodes")

        with open(path, "w") as file:
            json.dump(tbdt_dict, file, indent=4)

    @classmethod
    def load_from_json(cls, path: Path) -> None:
        """Load the TBDT from a JSON file containing its dict representation.

        Parameters
        ----------
        path : Path

        """
        with open(path, "r") as file:
            tbdt_dict = json.load(file)
        return cls.from_dict(tbdt_dict)

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
        self, dir_path: Path | None = None, shape="rectangle", graph="digraph"
    ) -> str:
        """Export the tree to the graphviz dot format, returning it as
        a str. If `dir_path` is specified, save it in this directory.

        Parameters
        ----------
        dir_path : str or Path
            Directory to which save the tree.
        shape : {'rectangle', 'circle'}, default='rectangle'
            Shape of the nodes.
        graph : str, default='digraph'
            Type of graph.

        Returns
        -------
        dot_str : str
            String of the graph in the dot format.

        """

        def node_label(node: Node) -> str:
            data = node.data
            split_i, split_v = data["split_i"], data["split_v"]
            split_i_str = f"{split_i:9<}" if split_i is not None else "None"
            split_v_str = f"{split_v:1.3e}" if split_v is not None else "None"
            label = (
                f"split feat idx: {split_i_str}\\n"
                f"value: {split_v_str}\\n"
                f"nb samples: {data['n_samples']}\\n"
                f"RMSE: {data['RMSE']:1.3e}"
            )
            return label

        nodes, connections = [], []
        if self.tree.nodes:
            for n in self.tree.expand_tree(mode=self.tree.WIDTH):
                nid = self.tree[n].identifier
                state = f'"{nid}" [label="{node_label(self.tree[n])}", shape={shape}]'
                nodes.append(state)
                for c in self.tree.children(nid):
                    cid = c.identifier
                    connections.append(f'"{nid}" -> "{cid}"')

        ret, tab = "\n", "\t"
        dot_str = (
            f"{graph} tree {{\n"
            f'\tlabel="{self.name}";\n'
            f"{ret if len(nodes) > 0 else ''}"
            f"{f';{ret}'.join([f'{tab}{node}' for node in nodes])};\n"
            f"{ret if len(connections) > 0 else ''}"
            f"{f';{ret}'.join([f'{tab}{conn}' for conn in connections])};\n"
            f"}}"
        )

        if dir_path is not None:
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
            with open(dir_path / f"{self.name}.dot", "w") as file:
                file.write(dot_str)

        return dot_str

    @_timer_func
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tb: np.ndarray,
        seed: int | None = None,
        logger: logging.Logger | None = None,
    ):
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
        seed : int | None
            Random state to use to create the random number generator.

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
        _log(logging.DEBUG, f"Fitting {self.name}", logger)
        t_start = perf_counter()

        rng = default_rng(seed)
        n, p = x.shape
        n, m, _ = tb.shape
        # Preconstruct the N_obs matrices for the lhs and rhs terms in
        # the least squares problem
        TT = np.zeros([n, m, m])
        Ty = np.zeros([n, m])
        for i in range(n):
            TT[i] = tb[i] @ tb[i].T + self.gamma * np.eye(m)  # Eq. 3.25 TT
            Ty[i] = tb[i] @ y[i]  # Eq. 3.25 Tb

        # Tree construction
        idx = np.arange(n)
        ghat, bhat = fit_tensor(TT[idx], Ty[idx], tb[idx], y[idx])
        diff = y[idx] - bhat
        rmse = np.sqrt(np.mean(diff**2))
        nodes2add: deque[tuple(Node, Node | None, np.ndarray)] = deque(
            [(Node(identifier="R"), None, np.arange(n), ghat, rmse)]
        )
        while nodes2add:
            node, parent, idx, ghat, rmse = nodes2add.popleft()
            n_samples = len(idx)

            split_i, split_v = None, None
            has_min_samples_split = len(idx) >= self.min_samples_split
            has_reached_max_depth = len(node.identifier) - 1 >= self.max_depth
            if has_min_samples_split and not has_reached_max_depth:
                n_feats = self._get_n_feats(p)
                feats_idx = rng.choice(p, size=n_feats, replace=False)
                split_data, left_data, right_data = create_split(
                    x[idx], y[idx], tb[idx], TT[idx], Ty[idx], feats_idx
                )
                have_min_samples_leaf = (
                    len(left_data["idx"]) >= self.min_samples_leaf,
                    len(right_data["idx"]) >= self.min_samples_leaf,
                )
                if all(have_min_samples_leaf):
                    split_i = split_data["split_i"]
                    split_v = split_data["split_v"]
                    node_l = Node(identifier=f"{node.identifier}0")
                    node_r = Node(identifier=f"{node.identifier}1")
                    idx_l = idx[left_data["idx"]]
                    idx_r = idx[right_data["idx"]]
                    ghat_l, ghat_r = left_data["ghat"], right_data["ghat"]
                    rmse_l = np.sqrt(left_data["MSE"])
                    rmse_r = np.sqrt(right_data["MSE"])
                    nodes2add.append((node_l, node, idx_l, ghat_l, rmse_l))
                    nodes2add.append((node_r, node, idx_r, ghat_r, rmse_r))

            node.tag = node.identifier
            node.data = {
                "split_i": split_i,
                "split_v": split_v,
                "g": list(map(float, ghat)),
                "n_samples": n_samples,
                "RMSE": rmse,
            }
            self.tree.add_node(node, parent=parent)

            _log(
                logging.DEBUG,
                f"Fitted node {node.identifier:<35}, "
                f"RMSE={rmse:.5e}, n_samples={n_samples:>6,}",
                logger,
            )

        t_end = perf_counter()
        t_delta = t_end - t_start
        _log(logging.INFO, f"Fitted {self.name} in {t_delta: >9.3f}s", logger)

    @_timer_func
    def predict(
        self,
        x: np.ndarray,
        tb: np.ndarray,
        logger: logging.Logger | None = None,
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
        ghat : np.ndarray
            Tensor basis coefficients with shape `(n, m)`.
        bhat : np.ndarray
            Anisotropy tensors with shape `(n, 9)`.

        """
        n, m, _ = tb.shape
        bhat = np.zeros([n, 9])
        ghat = np.zeros([n, m])

        for sample_i in range(n):
            node = self.tree.get_node("R")
            split_i = node.data["split_i"]
            split_v = node.data["split_v"]
            g = node.data["g"]

            while split_i is not None:
                if x[sample_i, split_i] <= split_v:
                    node = self.tree.get_node(f"{node.identifier}0")
                else:
                    node = self.tree.get_node(f"{node.identifier}1")
                split_i = node.data["split_i"]
                split_v = node.data["split_v"]
                g = node.data["g"]

            ghat[sample_i] = g
            for j in range(m):
                bhat[sample_i] += g[j] * tb[sample_i, j]

        return ghat, bhat
