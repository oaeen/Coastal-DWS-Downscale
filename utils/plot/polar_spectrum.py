import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator as RGI

sys.path.append(os.getcwd())

from utils.metrics.spectra_info import generate_extend_dirs, generate_freq


def interpolate_and_extend_spectra(
    spec,
    base_freq=0.0339,
    freq_new_num=72,
    dir_new_num=72,
    cdip_buyo=False,
):

    freq_n, dir_n = spec.shape
    spec = np.concatenate((spec, spec[:, 0:1]), axis=1)
    freq_ori = generate_freq(base=base_freq, cdip_buyo=cdip_buyo)
    dirs_ori = generate_extend_dirs(n=dir_n + 1)

    interpolator = RGI((dirs_ori, freq_ori), spec.T, method="cubic", bounds_error=False)

    freq_new = generate_freq(new_n=freq_new_num, base=base_freq, cdip_buyo=cdip_buyo)
    dirs_new = generate_extend_dirs(dir_new_num + 1)
    meshf, meshd = np.meshgrid(freq_new, dirs_new)
    spec = interpolator((meshd, meshf)).T

    return meshf, meshd, spec


def plot_spec(
    spec,
    filename,
    save_dir,
    spec_freq_lim=0.4,
    vmax=1.2,
    base_freq=0.0339,
    colorbar_label=r"Spectral Density (m$^{2}$s)",
    contour=True,
    cdip_buyo=False,
):
    meshf, meshd, spec_extend = interpolate_and_extend_spectra(
        spec, base_freq=base_freq, cdip_buyo=cdip_buyo
    )

    plt.rcdefaults()
    # matplotlib.rc('font', family='fantasy')
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", color="0.25")
    matplotlib.rc("ytick", labelsize=20)
    matplotlib.rc("axes", lw=1.6)

    _ = plt.figure(None, (8.5, 7))
    ax1 = plt.subplot(111, polar=True)
    plt.rc("grid", color="0.3", linewidth=0.8, linestyle="dotted")
    _ = plt.pcolormesh(
        meshd,
        meshf,
        spec_extend.transpose(),
        vmin=0,
        vmax=vmax,
        shading="gouraud",
        cmap="CMRmap_r",
    )
    cb = plt.colorbar(pad=0.10)
    cb.set_label(colorbar_label, fontsize=22)
    cb.ax.tick_params(labelsize=24)
    cb.locator = ticker.MultipleLocator(0.2)  # Set colorbar ticks interval
    cb.update_ticks()

    if contour:
        contour_label = np.array([0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.9, 1.2, 1.5, 1.9])
        _ = plt.contour(
            meshd,
            meshf,
            spec_extend.transpose(),
            contour_label,
            linewidths=1,
            colors="0.4",
        )

    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.set_ylim(0, spec_freq_lim)
    ax1.set_rgrids(
        radii=np.linspace(0.1, spec_freq_lim, int(spec_freq_lim / 0.1)),
        labels=[f"{i:.1f}Hz" for i in np.arange(0.1, spec_freq_lim + 0.1, 0.1)],
        angle=158,
    )
    ax1.xaxis.set_tick_params(pad=11)

    plt.grid(True)

    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%m%d-%H%M%S")

    plt.savefig(f"{save_dir}/{filename}_{time_stamp}.png", dpi=600, bbox_inches="tight")
    plt.close("all")


def plot_spec_list(
    spec_list,
    filename_list,
    save_dir,
    filename_appendix="",
    base_freq=0.0339,
    cdip_buyo=False,
):
    for spec, filename in zip(spec_list, filename_list):
        plot_spec(
            spec,
            filename=filename + filename_appendix,
            save_dir=save_dir,
            base_freq=base_freq,
            cdip_buyo=cdip_buyo,
        )
