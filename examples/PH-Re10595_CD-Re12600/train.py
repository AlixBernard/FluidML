#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-04 21:10:54
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-09 02:49:57

""" Train a TBRF with SquareDuct as training cases. """

# Built-in packages
import logging
import multiprocessing as mp
from pathlib import Path

# Third party packages
import fluidml
import numpy as np
from fluidml.tensor_basis import TBDT, TBRF
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.metrics import mean_squared_error

# Local packages
import config as cfg
import utils


def preprocess_data_LF(
    data_LF: dict,
    scale_SR: bool = False,
    scale_Ak: bool = False,
    scale_TB: bool = False,
    select_feats: list = [],
) -> dict:
    """Preprocess the low fidelity data.

    Parameters
    ----------
    data_LF : dict
        Low fidelity data containing:
            gradU : np.ndarray
                The gradient of the velocity U as a 2-D with shape
                `(n, 9)` where `n` is the number of points.
    scale_SR, scale_Ak, scale_TB : bool
        Whether to scale the symmetry tensor and rotation tensor, or the
        tensor basis.
        Default is False.
    select_feats : list
        Indices of the invariants to select as features.
        |FS1| + |FS2| + |FS3| = 6 + 13 + 9 = 28

    Returns
    -------
    data : dict
        Preprocessed data containing:
            **data_LF : dict
                All keys and values from the initial data.
            gradU : np.ndarray
                Velocity gradient reshaped to `(n, 3, 3)`.
            S : np.ndarray
                Mean strain rate tensor with shape `(n, 3, 3)`.
            R : np.ndarray
                Mean rotation rate tensor with shape `(n, 3, 3)`.
            Ak : np.ndarray
                Turbulent kinetic energy antisymmetric tensor with shape
                `(n, 3, 3)`.
            invariants : np.ndarray
                FS1 + FS2 + FS2_extra + FS3 wioth shape `(n, 28)`.
            features : np.ndarray
                The features selected from the invariants with shape
                `(n, p)` where `p=len(select_feats)`.

    """
    data = data_LF.copy()
    data["gradU"] = data["gradU"].reshape(-1, 3, 3)

    scale_SR_factors = data["k"] / data["epsilon"] if scale_SR else None
    scale_Ak_factors = data["k"] / data["epsilon"] if scale_Ak else None
    scale_TB_factors = (
        [10, 100, 100, 100, 1_000, 1_000, 10_000, 10_000, 10_000, 10_000]
        if scale_TB
        else None
    )

    data["S"], data["R"] = fluidml.utils.get_SR(
        data["gradU"], scale_factors=scale_SR_factors
    )
    data["Ak"] = fluidml.utils.get_Ak(
        data["gradk"],
        scale_factors=scale_Ak_factors,
    )
    data["TB"] = fluidml.utils.get_TB10(
        data["S"], data["R"], scale_factors=scale_TB_factors
    )
    data["tb"] = data["TB"].reshape(-1, 10, 9)
    data["invariants"] = np.hstack(
        [
            fluidml.utils.get_invariants_FS1(data["S"], data["R"]),
            fluidml.utils.get_invariants_FS2(data["S"], data["R"], data["Ak"]),
            fluidml.utils.get_invariants_FS3(data),
        ]
    )

    data["features"] = data["invariants"][:, select_feats]

    for i in range(data["invariants"].shape[1]):
        std = data["invariants"][:, i].std()
        if std == 0:
            print(f"WARNING: invariant {i} has std={std}")

    return data


def reshape_bump(arr: np.ndarray, end_shape: tuple = (9,)) -> np.ndarray:
    arr_reshaped = np.zeros((175, 412, *end_shape))
    nx = [
        0,
        40,
        40 + 12,
        40 + 12 + 200,
        40 + 12 + 200 + 40,
        40 + 12 + 200 + 40 + 120,
    ]
    for i in range(len(nx) - 1):
        x0, x1 = nx[i], nx[i + 1]
        arr_reshaped[:, x0:x1] = arr[x0 * 175 : x1 * 175].reshape(
            175, x1 - x0, *end_shape
        )[::-1, :]
    return arr_reshaped.reshape(175 * 412, *end_shape)


def grid_shape(path: Path) -> tuple[int, int]:
    if str(path) in [
        f"PeriodicHills/Re{re}_kOmega_140"
        for re in [700, 1400, 2800, 5600, 10595]
    ]:
        n1, n2 = 150, 140
    elif str(path) in [
        f"SquareDuct/Re{re}_kOmega_50"
        for re in [1800, 2000, 2200, 2400, 2600, 2900, 3200, 3500]
    ]:
        n1, n2 = 50, 50
    elif str(path) in ["ConvDivChannel/Re12600_kOmega_100"]:
        n1, n2 = 100, 140
    elif str(path) in ["BackwardFacingStep/Re5100_kOmega_90"]:
        raise ValueError(f"BFS structure unknown for now")
    else:
        raise ValueError(f"The structure of '{path.name}'' is not recognized")
    return n1, n2


def main() -> None:
    logger = utils.setup_log(
        tag="TBRF train",
        level=logging.INFO,
        stdout=True,
        filename=f"{Path(__file__).parent}/train.log",
    )
    cfg.tbrf_kwargs["logger"] = logger
    cfg.tbdt_kwargs["logger"] = logger
    logger.info(cfg.config_log_text)

    ########################## TRAIN ###################################

    # Load and preprocess the low fidelity train data
    logger.info(f"Load and preprocess low fidelity train data")
    datasets_LF = []
    for path in cfg.train_data_paths:
        raw_data_LF = {}
        for field in cfg.required_fields:
            filename = f"{cfg.fields_filenames[field]}.npy"
            raw_data_LF[field] = np.load(path / filename)
        datasets_LF.append(
            preprocess_data_LF(
                raw_data_LF,
                scale_SR=cfg.scale_SR,
                scale_Ak=cfg.scale_Ak,
                scale_TB=cfg.scale_TB,
                select_feats=cfg.select_feats,
            )
        )

    # Load the high fidelity train data
    logger.info(f"Load high fidelity train data")
    datasets_HF = []
    for path in cfg.train_data_paths:
        raw_data_HF = {}
        for field in ["b_HF"]:
            filename = f"{field}.npy"
            raw_data_HF[field] = np.load(path / filename)
            raw_data_HF["target"] = raw_data_HF["b_HF"].copy().reshape(-1, 9)
        datasets_HF.append(raw_data_HF)

    # Assemble train data into an `np.ndarray`
    x_train = np.vstack([data_LF["features"] for data_LF in datasets_LF])
    y_train = np.vstack([data_HF["target"] for data_HF in datasets_HF])
    tb_train = np.vstack([data_LF["tb"] for data_LF in datasets_LF])

    # Normalize
    x_train, x_mu, x_std = utils.normalize_features(x_train)

    # Remove outliers
    outliers_idx = utils.get_outliers_idx(x_train, 5)
    x_train = np.delete(x_train, outliers_idx, axis=0)
    y_train = np.delete(y_train, outliers_idx, axis=0)
    tb_train = np.delete(tb_train, outliers_idx, axis=0)

    # Training
    tbrf = TBRF(**cfg.tbrf_kwargs)
    tbrf.fit(x_train, y_train, tb_train)
    if not cfg.trees_path.exists():
        cfg.trees_path.mkdir(parents=True)
    tbrf.save(cfg.trees_path / cfg.tbrf_kwargs["name"])

    # Compute and log MSE
    _, _, y_pred = tbrf.predict(x_train, tb_train)
    mse_train = mean_squared_error(y_train, y_pred)
    logger.info(f"MSE on train: {mse_train:e}")

    ########################### TEST ###################################

    figs = []
    for test_i, path in enumerate(cfg.test_data_paths):
        # Load and preprocess the low fidelity test data
        logger.info(f"Load and preprocess low fidelity test data: '{path}'")
        raw_data_LF = {}
        for field in cfg.required_fields:
            filename = f"{cfg.fields_filenames[field]}.npy"
            raw_data_LF[field] = np.load(path / filename)
        data_LF = preprocess_data_LF(
            raw_data_LF, select_feats=cfg.select_feats
        )

        # Load the high fidelity test data
        logger.info(f"Load and preprocess high fidelity test data:  '{path}'")
        raw_data_HF = {}
        for field in ["b_HF"]:
            filename = f"{field}.npy"
            raw_data_HF[field] = np.load(path / filename)
            raw_data_HF["target"] = raw_data_HF["b_HF"].reshape(-1, 9)
            b_true = raw_data_HF["target"]
        data_HF = raw_data_HF

        # Assemble test data into an array
        x_test = data_LF["features"]
        y_test = data_HF["target"]
        tb_test = data_LF["tb"]

        # Normalize
        x_test, _, _ = utils.normalize_features(x_test, x_mu, x_std)

        # Predict
        _, _, b_pred = tbrf.predict(x_test, tb_test)
        b_pred = np.clip(b_pred, b_true.min(), b_true.max())  # ! Clip `b_pred`
        np.save(path / f"b_pred_{tbrf.name}.npy", b_pred)
        mse_test_pred = mean_squared_error(y_test, b_pred)

        if np.any(
            b_pred[:, [1, 2, 5]] != b_pred[:, [3, 6, 7]]
        ):  # Check symmetry
            print(f"Non symmetry of `b_pred` detected")

        # Reshape as grid
        subpath = path.relative_to(path.parent.parent)
        if str(subpath) in [f"Bump/h{h}" for h in [20, 26, 31, 38, 42]]:
            b_pred_reshaped = reshape_bump(b_pred, end_shape=(9,))
            b_true_reshaped = reshape_bump(b_true, end_shape=(9,))
            n1, n2 = 175, 412
        else:
            n1, n2 = grid_shape(subpath)
        b_pred_reshaped = b_pred.reshape(n1, n2, 9)
        b_true_reshaped = b_true.reshape(n1, n2, 9)

        # Gaussian filter
        b_gaussianFilter_reshaped = np.zeros([n1, n2, 9])
        for i in range(9):
            b_gaussianFilter_reshaped[:, :, i] = gaussian_filter(
                b_pred_reshaped[:, :, i], **cfg.gaussianFilter_kwargs
            )
        b_gaussianFilter = b_gaussianFilter_reshaped.reshape(n1 * n2, 9)
        np.save(path / f"b_gaussianFilter_{tbrf.name}.npy", b_pred)
        mse_test_gaussianFilter = mean_squared_error(y_test, b_gaussianFilter)

        # median filter
        b_medianFilter_reshaped = np.zeros([n1, n2, 9])
        for i in range(9):
            b_medianFilter_reshaped[:, :, i] = median_filter(
                b_pred_reshaped[:, :, i], **cfg.medianFilter_kwargs
            )
        b_medianFilter = b_medianFilter_reshaped.reshape(n1 * n2, 9)
        np.save(path / f"b_medianFilter_{tbrf.name}.npy", b_pred)
        mse_test_medianFilter = mean_squared_error(y_test, b_medianFilter)
        logger.info("Save")

        logger.info(
            f"MSE on test case '{path}':"
            f"\n    pred: {mse_test_pred:>30.5e}"
            f"\n    pred-gaussianFilter: {mse_test_gaussianFilter:>15.5e}"
            f"\n    pred-medianFilter: {mse_test_medianFilter:>17.5e}"
        )


if __name__ == "__main__":
    main()
