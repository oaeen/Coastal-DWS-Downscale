import os
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.getcwd())


def scale_spec2d(spec, spec_scale_bins="freq"):
    """
    :param spec: origin spec data
    :param spec_scale_bins: scale data by "freq_and_dir" or "freq"
    :return: scaled spec data, scaler_dict
    """

    scaler_dict = {}

    _, freq, direction = spec.shape

    spec_train = spec[: int(len(spec) * 0.8)]
    spec_test = spec[int(len(spec) * 0.8) :]

    logger.debug(f"spec_train.shape: {spec_train.shape}")
    logger.debug(f"spec_test.shape: {spec_test.shape}")

    if spec_scale_bins == None:
        logger.warning("not scale, return original data")
        return spec_train, spec_test, scaler_dict

    spec_train_scaled = np.zeros_like(spec_train)
    spec_test_scaled = np.zeros_like(spec_test)

    for freq_idx in range(freq):
        scaler_obj = MinMaxScaler()

        if spec_scale_bins == "freq_and_dir":
            scaler_obj = scaler_obj.fit(spec_train[:, freq_idx, :])
            spec_train_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_train[:, freq_idx, :]
            )
            spec_test_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_test[:, freq_idx, :]
            )
            scaler_dict[freq_idx] = scaler_obj
        if spec_scale_bins == "freq":
            scaler_obj = scaler_obj.fit(
                spec_train[:, freq_idx, :].flatten().reshape(-1, 1)
            )
            spec_train_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_train[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
            spec_test_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_test[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
            scaler_dict[freq_idx] = scaler_obj

    return spec_train_scaled, spec_test_scaled, scaler_dict


def inverse_scale_spec2d(spec_scaled, scaler_dict, spec_scale_bins):
    _, freq, direction = spec_scaled.shape

    if spec_scale_bins == None:
        logger.warning("not inverse, return original data")
        return spec_scaled

    spec_unscaled = np.zeros_like(spec_scaled)
    logger.info(f"inverse_scale_spec2d: {spec_scale_bins}")
    if spec_scale_bins == "freq_and_dir":
        for freq_idx in range(freq):
            scaler_obj = scaler_dict[freq_idx]
            spec_unscaled[:, freq_idx, :] = scaler_obj.inverse_transform(
                spec_scaled[:, freq_idx, :]
            )
    else:
        for freq_idx in range(freq):
            scaler_obj = scaler_dict[freq_idx]
            spec_unscaled[:, freq_idx, :] = scaler_obj.inverse_transform(
                spec_scaled[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
    return spec_unscaled


def load_scaler(file_dir):
    scaler_path = f"{file_dir}/scaler.pkl"
    scaler = pickle.load(open(scaler_path, "rb"))
    logger.success(f"load scaler from {scaler_path}")
    return scaler
