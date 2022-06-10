#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-06 18:47:17
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-08 05:42:05

"""
Plot the anisotropy tensor for the test cases:
    - `b_true`
    - `b_pred`
    - `b_gaussianFilter`
    - `b_medianFilter`

"""

# Built-in packages
from pathlib import Path

# Third party packages
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate

import utils

# Local packages
import config as cfg


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
    dpi = 600
    path = Path.home() / "FluidML/data/train_data/ConvDivChannel/Re12600_kOmega_100"
    logger.info(f"Load data from '{path}'")
    b_true = np.load(path / f"b_HF.npy")
    b_pred = np.load(path / f"b_pred_{tbrf_name}.npy")
    b_gaussianFilter = np.load(path / f"b_gaussianFilter_{tbrf_name}.npy")
    b_medianFilter = np.load(path / f"b_medianFilter_{tbrf_name}.npy")

    # Clip for same colorbar for `b`-true, -pred, and -pred_filtered
    _, m = b_pred.shape
    for i in range(m):
        b_true_i_min, b_true_i_max = b_true[:, i].min(), b_true[:, i].max()
        b_pred[:, i] = np.clip(b_pred[:, i], b_true_i_min, b_true_i_max)
        b_gaussianFilter[:, i] = np.clip(
            b_gaussianFilter[:, i], b_true_i_min, b_true_i_max
        )
        b_medianFilter[:, i] = np.clip(
            b_medianFilter[:, i], b_true_i_min, b_true_i_max
        )

    # Reshape as grid
    subpath = path.relative_to(path.parent.parent)
    if str(subpath) in [f"Bump/h{h}" for h in [20, 26, 31, 38, 42]]:
        b_pred_reshaped = reshape_bump(b_pred, end_shape=(9,))
        b_gaussianFilter_reshaped = reshape_bump(
            b_gaussianFilter, end_shape=(9,)
        )
        b_medianFilter_reshaped = reshape_bump(
            b_medianFilter, end_shape=(9,)
        )
        b_true_reshaped = reshape_bump(b_true, end_shape=(9,))
        n1, n2 = 175, 412
    else:
        n1, n2 = grid_shape(subpath)
        b_pred_reshaped = b_pred.reshape(n1, n2, 9)
        b_true_reshaped = b_true.reshape(n1, n2, 9)

    # Plots
    plot_path = cfg.pictures_path / f"{plot_folder_names[test_i]}"
    if not plot_path.exists():
        plot_path.mkdir(parents=True)
    fmt = "%.1e"
    idxs = [0, 4, 8, 1, 2, 5]  # 11, 22, 33, 12, 13, 23

    if "SquareDuct" in str(path):
        for b, title, filename in plot_data:
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(title)
            for i in range(len(idxs)):
                ax = axs[i // 3, i % 3]
                ax.set_title(
                    f"$b_{{ {idxs[i]//3 + 1}, {idxs[i]%3 + 1} }}$"
                )
                pcm = ax.pcolormesh(
                    b_pred_reshaped[:, :, idxs[i]], cmap="jet"
                )
                ax.axis("scaled")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(pcm, cax=cax, format=fmt)
            fig.tight_layout()
            fig.savefig(plot_path / filename, dpi=dpi)
            logger.info(
                f"Saved plot of `b`-{title.lower().replace(' ', '-')} as"
                f" '{plot_path / filename}"
            )
    else:
        # data coordinates and values
        ngridx = 1_000
        ngridy = 1_000
        x = np.load(path / f"Cx.npy")
        y = np.load(path / f"Cy.npy")

        # target grid to interpolate to
        xi = np.linspace(x.min(), x.max(), ngridx)
        yi = np.linspace(y.min(), y.max(), ngridy)
        xi, yi = np.meshgrid(xi, yi)

        n_levels = 20
        levels = []
        for i in range(len(idxs)):
            b_true_i = b_true[:, idxs[i]]
            levels.append(
                np.linspace(b_true_i.min(), b_true_i.max(), n_levels)
            )

        for b, title, filename in plot_data:
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(title)
            for i in range(len(idxs)):
                b_i = np.clip(b[:, i, levels[i].min(), levels[i].max())
                zi = interpolate.griddata(
                    (x, y), b_i, (xi, yi), method="linear"
                )

                ax = axs[i // 3, i % 3]
                ax.set_title(
                    f"$b_{{ {idxs[i]//3 + 1}, {idxs[i]%3 + 1} }}$"
                )
                pcm = ax.contourf(xi, yi, zi, levels[i], cmap="jet")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(pcm, cax=cax, format=fmt)
            fig.tight_layout()
            fig.savefig(plot_path / filename, dpi=dpi)
            logger.info(
                f"Saved plot of `b`-{title.lower().replace(' ', '-')} as"
                f" '{plot_path / filename}"
            )


if __name__ == "__main__":
    main()
