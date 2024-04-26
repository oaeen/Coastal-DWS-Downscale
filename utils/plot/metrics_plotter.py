import os
import sys
import time
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr

sys.path.append(os.getcwd())

from config import Config
from utils.metrics.perf_recorder import *
from utils.metrics.spec_metrics import Spec_Metrics
from utils.metrics.spectra_info import *
from utils.plot.polar_spectrum import plot_spec

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_metrics(y_predict_unscale, y_true_unscale, config=Config()):
    _sample, freq, direction = y_predict_unscale.shape
    metrics = Spec_Metrics(
        freq=freq, direction=direction, cdip_buyo=config.is_y_cdip_buyo()
    )

    (
        swh_pre,
        mwd_pre,
        mwp_minus1_pre,
        mwp1_pre,
        mwp2_pre,
    ) = metrics.integral_predict_spec_parameters(y_predict_unscale)
    (
        swh_true,
        mwd_true,
        mwp_minus1_true,
        mwp1_true,
        mwp2_true,
    ) = metrics.integral_predict_spec_parameters(y_true_unscale)

    mwd_pre[np.where((mwd_true - mwd_pre) > 180)[0]] = (
        mwd_pre[np.where((mwd_true - mwd_pre) > 180)[0]] + 360
    )
    mwd_true[np.where((mwd_true - mwd_pre) < -180)[0]] = (
        mwd_true[np.where((mwd_true - mwd_pre) < -180)[0]] + 360
    )
    plot_hist2d(y_true_unscale, y_predict_unscale, get_spec_desc(), config=config)
    plot_hist2d(swh_true, swh_pre, get_swh_desc(), config=config)
    plot_hist2d(mwd_true, mwd_pre, get_mwd_desc(), config=config)
    plot_hist2d(mwp_minus1_true, mwp_minus1_pre, get_mwp_minus1_desc(), config=config)
    plot_hist2d(mwp1_true, mwp1_pre, get_mwp1_desc(), config=config)
    plot_hist2d(mwp2_true, mwp2_pre, get_mwp2_desc(), config=config)


def plot_hist2d(y_true, y_predict, data_description, config=Config()):
    y_true = y_true.flatten()
    y_predict = y_predict.flatten()

    y_true = np.nan_to_num(y_true)
    y_predict = np.nan_to_num(y_predict)

    data_type = data_description["data_type"]
    max_value = data_description["max_value"]
    vmax = data_description["vmax"]
    xlabel_text = data_description["xlabel_text"]
    ylabel_text = data_description["ylabel_text"]
    unit_text = data_description["unit_text"]

    if data_type == "swh" and config.y_location == "CDIP067":
        """special case for CDIP067"""
        max_value = 8

    metrics = Spec_Metrics()
    rmse, bias, corrcoef = metrics.evaluate_predict_spec_loss(
        y_true, y_predict, data_type
    )
    bias = f"{bias:.2g}" if np.abs(bias) < 1 else f"{bias:.2f}"
    upper_left_text = (
        f"RMSE = {rmse:.2f} {unit_text}\nBias = {bias} {unit_text}\nR = {corrcoef:.2f}"
    )

    plt.rcdefaults()
    plt.clf()
    plt.figure(figsize=(6, 6))

    fig, ax = plt.subplots()

    plt.hist2d(
        y_true,
        y_predict,
        bins=120,
        cmap=plt.cm.viridis,
        norm=mcolors.LogNorm(vmax=vmax),
        range=[[0, max_value], [0, max_value]],
    )
    plt.plot([0, max_value], [0, max_value])
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    cb = plt.colorbar()
    cb.set_label(r"Data density", fontsize=13)
    cb.ax.tick_params(labelsize=14)  # Increase colorbar ticks font size
    cb.update_ticks()

    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.text(0.05, 0.80, upper_left_text, transform=ax.transAxes, fontsize=14)
    plt.xlabel(xlabel_text, fontsize=13)
    plt.ylabel(ylabel_text, fontsize=13)
    evaluate_save_dir = config.get_evaluate_figure_save_dir()
    filename = f"{evaluate_save_dir}/hist2d_{data_type}_{datetime.now().strftime('%m%d-%H%M%S')}.png"
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close("all")
    logger.success(f"savefig: {filename}")
