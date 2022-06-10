#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-07 21:13:51
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-08 02:56:43

""" Description. """

# Built-in packages
import logging
import sys
from pathlib import Path

# Third party packages
import numpy as np
from numpy.random import default_rng
from scipy.interpolate import griddata

# Local packages

__all__ = []


def setup_log(
    tag: str = "NA",
    level: int = logging.DEBUG,
    stdout: bool = True,
    filename: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(tag)
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    if stdout:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def normalize_features(
    x: np.ndarray,
    mu: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray]:
    """Normalize the features `x` where `x` is an array with shape
    `(n, p)`. If `mu` and/or `std` are provided they must be an array
    with shape `(p,)` otherwise they will be computed from `x` for each
    feature.

    Parameters
    ----------
    x : np.ndarray
        Features with shape `(n, p)`.
    mu : np.ndarray | None
        Mean for each feature to use in the normalizing process, if
        provided then must have shape `(p,)`.
    std : Standard deviation for each feature to use in the normilizing
        process, if provided then must have shape `(p,)`.

    Returns
    -------
    x_norm : np.ndarray
        The normalized features.
    mu : np.ndarray
        The feature means used in the normilizing process with shape
        `(p,)`.
    std : np.ndarray
        The feature standard deviations used in the normilizing process
        with shape `(p,)`.

    """
    if mu is None:
        mu = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)

    x_norm = np.zeros(x.shape)
    m = x.shape[1]  # Number of features
    for i in range(m):
        if std[i] != 0:
            x_norm[:, i] = (x[:, i] - mu[i]) / std[i]
        else:
            print(f"std[{i}] = {std[i]}")
            x_norm[:, i] = np.full(x[:, i].shape, np.inf)
            # ? WTF???
            # - should replace `x_norm` by 0 or 1, not np.inf

    return x_norm, mu, std

def get_outliers_idx(
    x: np.ndarray, cap: np.ndarray | float = 1.0
) -> np.ndarray:
    """Find and returns the index of all points that have a feature with
    an absolute value greater than than `cap`. The input `x` must be an
    array with shape `(n, p)`, while `cap` must be an array with shape
    `(p,)`.

    Parameters
    ----------
    x : np.ndarray
        Features with shape `(n, p)`
    cap : float | np.ndarray
        Cap values as a float or an array with shape `(p,)`.

    Returns
    -------
    outliers_idx : np.ndarray
        Capped array `x` as an array of shape `(n2, p)` where `n2` is
        the number of points after removing outliers (`d` < `n2`).

    """
    n, p = x.shape
    if not isinstance(cap, np.ndarray):
        cap *= np.ones(
            [
                p,
            ]
        )

    feat_masks = np.ones(x.shape, dtype=bool)
    for i in range(p):
        feat_masks[:, i] = np.abs(x[:, i]) < cap[i]

    mask = np.array([arr.all() for arr in feat_masks])
    outliers_idx = np.arange(n)[mask]

    return outliers_idx

def remove_outliers(x: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """For each feature, remove the outliers that have a absolute value
    greater than `cap`. `x` must be `np.ndarray` with shape `(n, m)`
    where `n`, is the number of points while `cap` must be an
    `np.ndarray` with shape `(m,)`.

    Parameters
    ----------
    x : np.ndarray
        Features with shape `(n, p)`.
    cap : np.ndarray
        Cap values array with shape `(p,)`.

    Returns
    -------
    x_capped : np.ndarray
        Capped array `x` of shape `(n2, p)` where `n2` is the number of
        points after removing outliers (`d` < `n2`).

    """
    feat_masks = np.ones(x.shape, dtype=bool)
    m = x.shape[1]  # Number of features
    for i in range(m):
        feat_masks[:, i] = np.abs(x[:, i]) > cap[i]

    mask = np.array([arr.all() for arr in feat_masks])
    x_capped = x[~mask]

    return x_capped

def interp_on_grid(
    points: np.ndarray,
    values: np.ndarray,
    ngrid: tuple[int, int],
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    method: str = "cubic",
) -> None:
    """Interpolate the values from points onto a grid and plot them.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of `values`, 2-D ndarray with shape `(n, 2)` where
        `n` is the number of points.
    values : np.ndarray
        Values to interpolate.
    ngrid : tuple[int, int]
        Size of the grid in the x axis and y axis.
    xlim, ylim : tuple[float, float] | None
        Limits of the grid into the x axis and y axis, if None will be
        detected from `points`.
        Default is None.
    method : str
        Method to use when interpolating.
        Default is 'cubic'.

    Returns
    -------
    grid : np.ndarray
        The grid on which the data has been interpolated.
    interp_values : np.ndarray
        The interpolated values.

    """
    ngrid_x, ngrid_y = ngrid
    points_x, points_y = points[:, 0], points[:, 1]
    if xlim is None:
        x_min, x_max = points_x.min(), points_x.max()
    if xlim is None:
        y_min, y_max = points_y.min(), points.max()

    xi = np.linspace(x_min, x_max, ngrid_x)
    yi = np.linspace(y_min, y_max, ngrid_y)
    grid = np.meshgrid(xi, yi)

    interp_values = griddata(points, values, grid, method=method)

    return grid, interp_values

def main() -> None:
	pass


if __name__ == "__main__":
	main()
