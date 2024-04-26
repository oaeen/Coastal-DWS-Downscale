import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.append(os.getcwd())
from config import Config
from utils.preprocess.extract.load_data import load_spec, load_wind
from utils.preprocess.extract.spec_matcher import Spec_Matcher
from utils.preprocess.prepare.scale_spec import *


def prepare_iowaga_spec(spec_path, save_dir, spec_scale_bins):
    if os.path.isfile(spec_path):
        logger.success(f"load spec from path: {spec_path}")
        spec = np.load(spec_path)
    else:
        logger.success(f"load spec from dir: {spec_path}")
        spec = load_spec(spec_path)

    logger.success(f"load spec data from {spec_path}")
    spec = np.nan_to_num(spec)

    spec_train, spec_test, spec_scaler_dict = scale_spec2d(spec, spec_scale_bins)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_dir}/train.npy", spec_train.astype(np.float32))
    np.save(f"{save_dir}/test.npy", spec_test.astype(np.float32))
    pickle.dump(spec_scaler_dict, open(f"{save_dir}/scaler.pkl", "wb"))
    logger.success(f"save spec & scaler to {save_dir}")


def prepare_iowaga_wind(wind_path, save_dir):
    if os.path.isfile(wind_path):
        logger.success(f"load wind from path: {wind_path}")
        wind = np.load(wind_path)
    else:
        logger.success(f"load wind from dir: {wind_path}")
        wind = load_wind(wind_path)

    logger.success(f"load wind data from {wind_path}")

    wind = np.nan_to_num(wind)

    wind_train = wind[: int(len(wind) * 0.8)]
    wind_test = wind[int(len(wind) * 0.8) :]

    # apply MinMaxScale
    wind_speed_max, wind_speed_min = np.max(wind_train[:, 0]), np.min(wind_train[:, 0])
    wind_speed_diff = wind_speed_max - wind_speed_min
    wind_train[:, 0] = (wind_train[:, 0] - wind_speed_min) / wind_speed_diff
    wind_test[:, 0] = (wind_test[:, 0] - wind_speed_min) / wind_speed_diff

    logger.debug(f"wind_train.shape: {wind_train.shape}")
    logger.debug(f"wind_test.shape: {wind_test.shape}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_dir}/train.npy", wind_train.astype(np.float32))
    np.save(f"{save_dir}/test.npy", wind_test.astype(np.float32))
    logger.success(f"save wind to {save_dir}")


def prepare_filter_iowaga_wind_by_cdip_time(target_station, config=Config()):
    """
    filter the IOWAGA wind data by the time of the CDIP data
    """

    logger.info(f"start filter {target_station} iowaga data")

    iowaga_wind_dir = (
        f"{config.processed_data_dir}/IOWAGA/extract_wind/{target_station}"
    )

    logger.debug(f"clip_index_station: {target_station}")
    _, iowaga_index_list = Spec_Matcher(target_station).get_filter_data_index()

    print(f"len(iowaga_index_list): {len(iowaga_index_list)}")
    input_iowaga_wind_list = load_wind(iowaga_wind_dir, False)
    logger.debug(f"iowaga_wind.shape: {input_iowaga_wind_list.shape}")
    input_iowaga_wind_list = input_iowaga_wind_list[iowaga_index_list]
    logger.debug(f"iowaga_wind.shape after filter: {input_iowaga_wind_list.shape}")

    wind = np.nan_to_num(input_iowaga_wind_list)

    wind_train = wind[: int(len(wind) * 0.8)]
    wind_test = wind[int(len(wind) * 0.8) :]

    wind_speed_max, wind_speed_min = np.max(wind_train[:, 0]), np.min(wind_train[:, 0])
    wind_speed_diff = wind_speed_max - wind_speed_min
    wind_train[:, 0] = (wind_train[:, 0] - wind_speed_min) / wind_speed_diff
    wind_test[:, 0] = (wind_test[:, 0] - wind_speed_min) / wind_speed_diff

    logger.debug(f"wind_train.shape: {wind_train.shape}")
    logger.debug(f"wind_test.shape: {wind_test.shape}")

    wind_save_dir = (
        f"{config.processed_data_dir}/CDIP/input/wind_input/{target_station}"
    )
    Path(wind_save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{wind_save_dir}/train.npy", wind_train.astype(np.float32))
    np.save(f"{wind_save_dir}/test.npy", wind_test.astype(np.float32))

    logger.success(
        f"save filter iowaga {target_station} wind with CDIP data time success"
    )


def prepare_iowaga_wind_script():
    target_locations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    for location in target_locations:
        wind_dir = f"{config.processed_data_dir}/IOWAGA/extract_wind/{location}"
        save_dir = f"{config.processed_data_dir}/IOWAGA/input/wind_input/{location}"
        prepare_iowaga_wind(wind_dir, save_dir)


def prepare_iowaga_spec_input_and_output_script():
    # input
    X_locations = [
        "W1215N335",
        "W1210N330",
        "W1205N330",
        "W1205N325",
        "W1200N320",
        "W1195N320",
        "W1190N320",
    ]

    config.X_data_desc = "spec_input"
    for location in X_locations:
        config.X_location = location
        save_dir = config.get_X_data_dir()
        spec_dir = f"{config.processed_data_dir}/IOWAGA/extract/{location}"
        prepare_iowaga_spec(spec_dir, save_dir, config.X_spec_scale_bins)

    # output
    y_locations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    config.y_data_source = "IOWAGA"
    config.y_data_desc = "spec_output"

    for location in y_locations:
        config.y_location = location
        spec_dir = (
            f"{config.processed_data_dir}/{config.y_data_source}/extract/{location}"
        )
        save_dir = config.get_y_data_dir()
        prepare_iowaga_spec(spec_dir, save_dir, config.y_spec_scale_bins)


def prepare_interpolate_iowaga_spec_and_wind_input_for_cdip_output_script():
    # for CDIP output
    target_stations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    for target_station in target_stations:
        prepare_filter_iowaga_wind_by_cdip_time(target_station, config=config)

    input_stations = [
        "W1215N335",
        "W1210N330",
        "W1205N330",
        "W1205N325",
        "W1200N320",
        "W1195N320",
        "W1190N320",
    ]
    target_stations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    for input_station in input_stations:
        for target_station in target_stations:
            spec_path = f"{config.processed_data_dir}/IOWAGA/filter/interpolated_input_dir36/{input_station}_to_{target_station}.npy"
            save_dir = f"{config.processed_data_dir}/IOWAGA/input/interpolated_spec_input_dir36_minmax_{config.X_spec_scale_bins}/{input_station}_to_{target_station}"
            prepare_iowaga_spec(spec_path, save_dir, config.X_spec_scale_bins)


if __name__ == "__main__":
    config = Config()
    # prepare_iowaga_wind_script()
    # prepare_iowaga_spec_input_and_output_script()
    # prepare_interp_iowaga_spec_and_wind_input_script()
