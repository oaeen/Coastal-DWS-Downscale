import os
import sys
import time
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import torch
from loguru import logger

sys.path.append(os.getcwd())

from config import Config
from utils.preprocess.extract.spec_matcher import Spec_Matcher


def get_cdip_spec_time_list(config=Config()):
    """
    get the time list of the CDIP test data
    """
    station = config.y_location
    filename = f"{config.raw_data_dir}/CDIP/{station}/{station[-3:]}p1_historic.nc"
    nc = netCDF4.Dataset(filename)
    cdip_time_list = nc.variables["waveTime"][:]
    nc.close()

    (
        filter_cdip_index_list,
        _filter_iowaga_index_list,
    ) = Spec_Matcher(station, config).get_filter_data_index()

    filter_cdip_time_list = cdip_time_list[filter_cdip_index_list]

    train_len = int(0.8 * len(filter_cdip_time_list))
    filter_cdip_time_list = filter_cdip_time_list[train_len:]
    filter_cdip_time_list = [
        datetime.utcfromtimestamp(t) for t in filter_cdip_time_list
    ]
    return filter_cdip_time_list


def get_iowaga_spec_time_list(spec, config=Config()):
    """
    get the time list of the IOWAGA data
    test data: 29 years * 0.2 = 5.8 years,
    from 2017 to 2021, 5 years, 14608 samples
    5 * 365 * 8 + 8(2020 is leap year) = 14608
    """
    test_start_idx = 14608

    start_time = datetime(2017, 1, 1, 0, 0, 0)
    iowaga_time_list = [
        start_time + timedelta(hours=3 * idx) for idx in range(test_start_idx)
    ]
    return iowaga_time_list


def get_target_year_spec(spec, config=Config()):
    """
    get the target year data of the spec
    """
    target_year = 2020

    if config.y_location in ["CDIP093", "CDIP107"] and config.is_y_cdip_buyo():
        target_year = 2015

    if config.is_y_cdip_buyo():
        logger.info(f"GET CDIP Target Spec of {target_year}")
        time_list = get_cdip_spec_time_list(config)
        time_list = time_list[config.spec_window_size - 1 :]
    else:
        logger.info(f"GET IOWAGA Target Spec of {target_year}")
        time_list = get_iowaga_spec_time_list(config)
        test_start_idx = 14608
        spec = spec[-test_start_idx:]

    spec_target_year = []
    for idx, time in enumerate(time_list):
        if time.year == target_year:
            spec_target_year.append(spec[idx])

    spec_target_year = np.asarray(spec_target_year)
    logger.info(f"spec_target_year{target_year}.shape: {spec_target_year.shape}")

    return spec_target_year


def get_specific_spec_samples(spec, target_year, month_indices, config=Config()):

    if config.is_y_cdip_buyo():
        logger.info(f"PLOT CDIP DATA SPEC SAMPLES")
        time_list = get_cdip_spec_time_list(config)
        time_list = time_list[config.spec_window_size - 1 :]
    else:
        logger.info(f"PLOT IOWAGA DATA SPEC SAMPLES")
        time_list = get_iowaga_spec_time_list(config)
        test_start_idx = 14608
        spec = spec[-test_start_idx:]

    spec_target_year = []
    spec_samples = []
    for idx, time in enumerate(time_list):
        if time.year == target_year:
            spec_target_year.append(spec[idx])
            # Find Sample Close to Month 1/4/7/10 Day1 12:00
            if (
                time.month in month_indices
                and time.day == 1
                and np.abs((time.hour + time.minute / 60) - 12) <= 1
            ):  # 指定月份1日的12点左右
                logger.debug(
                    f"Find {config.y_data_source}: {config.y_location} #{time} Sample"
                )
                spec_samples.append(spec[idx])

    spec_avg = np.mean(spec_target_year, axis=0)
    spec_samples = np.array(spec_samples)
    return spec_avg, spec_samples
