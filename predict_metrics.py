import os
import sys
import time

import numpy as np
import torch
from loguru import logger

sys.path.append(os.getcwd())

from config import Config
from models.model import load_model
from utils.data_loaders.helper import get_dataloader
from utils.metrics.perf_recorder import *
from utils.plot.polar_spectrum import plot_spec, plot_spec_list
from utils.preprocess.extract.spec_fetcher import *
from utils.preprocess.prepare.scale_spec import inverse_scale_spec2d, load_scaler
from utils.metrics.spectra_info import correct_spec_direction


@torch.no_grad()
def predict(model, dataloader, config=Config()):
    model.eval()

    all_predict = []
    all_targets = []

    for spec_inputs, wind_inputs, targets in dataloader:
        spec_inputs = spec_inputs.to(config.device)
        wind_inputs = wind_inputs.to(config.device)
        targets = targets.to(config.device)
        outputs = model(spec_inputs, wind_inputs)

        all_predict.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_predict = np.concatenate(all_predict, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_predict, all_targets


@torch.no_grad()
def predict_without_wind(model, dataloader, config=Config()):
    model.eval()

    all_predict = []
    all_targets = []

    for spec_inputs, targets in dataloader:
        spec_inputs = spec_inputs.to(config.device)
        targets = targets.to(config.device)
        outputs = model(spec_inputs)

        all_predict.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_predict = np.concatenate(all_predict, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_predict, all_targets


def process_predict_spec2d(model=None, config=Config()):
    predict_start_time = time.time()
    if model is None:
        model, _ = load_model(config)

    _train_dataloader, test_dataloader = get_dataloader(False, config)

    if config.input_wind:
        y_predict, y_true = predict(model, test_dataloader, config=config)
    else:
        y_predict, y_true = predict_without_wind(model, test_dataloader, config=config)
    predict_end_time = time.time()

    logger.success(f"predict running time: {predict_end_time - predict_start_time}")

    y_scaler = load_scaler(config.get_y_data_dir())
    y_predict = inverse_scale_spec2d(y_predict, y_scaler, config.y_spec_scale_bins)
    y_true = inverse_scale_spec2d(y_true, y_scaler, config.y_spec_scale_bins)

    y_predict = correct_spec_direction(y_predict, config.is_y_cdip_buyo())
    y_true = correct_spec_direction(y_true, config.is_y_cdip_buyo())

    return y_predict, y_true


def plot_output_spec_samples(y_predict, y_true, config=Config()):
    # specific for CDIP028, CDIP045, CDIP067;
    target_year = 2020

    if config.y_location in ["CDIP093", "CDIP107"] and config.is_y_cdip_buyo():
        target_year = 2015

    samples_save_dir = config.get_samples_figure_save_dir()

    month_indices = [1, 4, 7, 10]
    y_predict_avg, y_predict_samples = get_specific_spec_samples(
        y_predict, target_year, month_indices, config
    )
    y_true_avg, y_true_samples = get_specific_spec_samples(
        y_true, target_year, month_indices, config
    )

    plot_spec(
        y_predict_avg,
        f"predict_avg",
        samples_save_dir,
        cdip_buyo=config.is_y_cdip_buyo(),
    )
    plot_spec(
        y_true_avg,
        f"true_avg",
        samples_save_dir,
        cdip_buyo=config.is_y_cdip_buyo(),
    )

    logger.debug(f"y_predict_samples.shape: {y_predict_samples.shape}")
    logger.debug(f"y_true_samples.shape: {y_true_samples.shape}")

    filename_list = [f"month{idx:02d}_day01_year{target_year}" for idx in month_indices]

    plot_spec_list(
        y_predict_samples,
        filename_list,
        samples_save_dir,
        filename_appendix=f"_predict",
        cdip_buyo=config.is_y_cdip_buyo(),
    )
    plot_spec_list(
        y_true_samples,
        filename_list,
        samples_save_dir,
        filename_appendix=f"_true",
        cdip_buyo=config.is_y_cdip_buyo(),
    )


def plot_input_spec_samples(target_year=None, input_pos="W1205N330"):
    """
    For CDIP093 & CDIP107: select year 2015 IOWAGA data
    """

    config = Config()
    config.X_location = f"input_{input_pos}"
    config.comment = f"{input_pos}_spec_input_year{target_year}"
    samples_save_dir = (
        f"{os.getcwd()}/results/samples_input/{config.X_location}/{config.comment}"
    )
    Path(samples_save_dir).mkdir(parents=True, exist_ok=True)

    X_input_spec_path = (
        f"{config.processed_data_dir}/IOWAGA/extract/{input_pos}/{target_year}.npy"
    )
    X_input_spec = np.load(X_input_spec_path)
    logger.success(f"load X input from {X_input_spec_path}")
    X_input_spec = correct_spec_direction(X_input_spec, is_cdip_buyo=False)

    month_indices = [1, 4, 7, 10]

    start_time = datetime(target_year, 1, 1, 0, 0, 0)
    iowaga_time_list = [
        start_time + timedelta(hours=3 * idx) for idx in range(X_input_spec.shape[0])
    ]

    X_input_avg = np.mean(X_input_spec, axis=0)
    X_input_samples = []

    for idx, time in enumerate(iowaga_time_list):
        if (
            time.month in month_indices
            and time.day == 1
            and np.abs((time.hour + time.minute / 60) - 12) <= 1
        ): # 12:00
            X_input_samples.append(X_input_spec[idx])

    X_input_samples = np.asarray(X_input_samples)

    logger.debug(f"X_input_avg.shape: {X_input_avg.shape}")
    logger.debug(f"X_input_samples.shape: {X_input_samples.shape}")

    plot_spec(
        X_input_avg,
        f"Avg_X_spec_{input_pos}",
        samples_save_dir,
    )
    filename_list = [
        f"month{idx:02d}_day01_year{target_year}_X_{input_pos}" for idx in month_indices
    ]
    plot_spec_list(
        X_input_samples,
        filename_list,
        samples_save_dir,
        filename_appendix=f"_input",
    )


if __name__ == "__main__":
    config = Config()
    plot_input_spec_samples(target_year=2015)
    plot_input_spec_samples(target_year=2020)
