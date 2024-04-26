import calendar
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

sys.path.append(os.getcwd())

from config import Config


class Spec_Matcher:
    """
    According to the CDIP wave spectrum time data, get the corresponding IOWAGA time index
    """

    def __init__(self, station, config=Config()):
        self.station = station
        self.index_save_dir = f"{config.processed_data_dir}/CDIP/index"
        self.cdip_index_path = f"{self.index_save_dir}/{station}_index.npy"
        self.iowaga_index_path = f"{self.index_save_dir}/IOWAGA_{station}_index.npy"
        self.time_diff_path = f"{self.index_save_dir}/time_diff_{station}.npy"
        self.filename = (
            f"{config.raw_data_dir}/CDIP/{station}/{station[-3:]}p1_historic.nc"
        )
        Path(self.index_save_dir).mkdir(parents=True, exist_ok=True)

    def get_clip_cdip_time(self, cdip_time_list, time_delta=timedelta(hours=3)):
        """
        According to the input CDIP wave spectrum time data,
        obtain the corresponding IOWAGA time period whose time period matches
        The time period for clipping is from 0 o'clock on the next day
        when the file starts to 21 o'clock on the day before the last day when the file ends.
        According to the clipped time period,
        obtain the data at 0, 3, 6, 9, 12, 15, 18, and 21 hours every day within the time period.
        """
        start_utc_time = datetime.utcfromtimestamp(cdip_time_list[0]).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        end_utc_time = datetime.utcfromtimestamp(cdip_time_list[-1]).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        logger.info(f"clip start time: {start_utc_time}, clip end time: {end_utc_time}")

        iowaga_utctime_list = [
            start_utc_time + i * time_delta
            for i in range((end_utc_time - start_utc_time).days * 8)
        ]
        iowaga_unixtime_list = [
            calendar.timegm(dt.timetuple()) for dt in iowaga_utctime_list
        ]
        return iowaga_unixtime_list

    def get_filter_data_index(self, max_tolerance_diff_time=900):
        """
        Based on the obtained CDIP target time period,
        find the index of the most recent time point in the original data,
        and filter out data with a time difference greater than 30 minutes.
        """

        if os.path.exists(self.cdip_index_path) and os.path.exists(
            self.iowaga_index_path
        ):
            logger.success(f"load filter index from exist file")
        else:
            logger.warning(
                f"filter index {self.cdip_index_path} and {self.iowaga_index_path} not exist, start filter data index"
            )

            logger.info(
                f"start filter {self.station} data diff time < {max_tolerance_diff_time}s"
            )
            nc = netCDF4.Dataset(self.filename)
            cdip_unixtime_list = nc.variables["waveTime"][:]
            nc.close()
            iowaga_unixtime_list = self.get_clip_cdip_time(cdip_unixtime_list)
            self.process_filter_data_index(
                cdip_unixtime_list, iowaga_unixtime_list, max_tolerance_diff_time
            )

        filter_cdip_index_list = np.load(self.cdip_index_path)
        filter_iowaga_index_list = np.load(self.iowaga_index_path)
        return filter_cdip_index_list, filter_iowaga_index_list

    def process_filter_data_index(
        self, cdip_unixtime_list, iowaga_unixtime_list, max_tolerance_diff_time=900
    ):
        filter_cdip_index_list = []
        filter_iowaga_unixtime_list = []
        time_diff_list = []
        for iowaga_unixtime in tqdm(iowaga_unixtime_list, desc="filter data index"):
            cdip_time_index = (np.abs(cdip_unixtime_list - iowaga_unixtime)).argmin()
            diff_time = abs(cdip_unixtime_list[cdip_time_index] - iowaga_unixtime)
            if diff_time > max_tolerance_diff_time:
                continue

            filter_cdip_index_list.append(cdip_time_index)
            filter_iowaga_unixtime_list.append(iowaga_unixtime)
            time_diff_list.append(diff_time)

        time_diff_list = np.array(time_diff_list)
        logger.info(f"filter num: {len(iowaga_unixtime_list)-len(time_diff_list)}")

        filter_iowaga_index_list = self.iowaga_unixtime_2_index(
            filter_iowaga_unixtime_list
        )
        np.save(self.cdip_index_path, filter_cdip_index_list)
        np.save(self.iowaga_index_path, filter_iowaga_index_list)
        np.save(self.time_diff_path, time_diff_list)
        logger.success(f"save {self.station} filter index success")

    def iowaga_unixtime_2_index(
        self, unixtime_list, start_time=datetime(1993, 1, 1, 0, 0, 0)
    ):
        """
        Convert the unixtime of IOWAGA that matches the CDIP measured data after filtering to the index in iowaga.
        """
        seconds_per_day = 24 * 60 * 60
        utctime_list = [datetime.utcfromtimestamp(t) for t in unixtime_list]

        time_difference_list = [
            (utctime - start_time).total_seconds() / seconds_per_day
            for utctime in utctime_list
        ]

        sample_period_time = 0.125  # The sampling period is 0.125 days, 3 hours
        iowaga_index_list = [
            int(time_diff / sample_period_time) for time_diff in time_difference_list
        ]
        return iowaga_index_list

    def remove_discontinuous_data(self, station):
        filter_cdip_index_list, filter_iowaga_index_list = self.get_filter_data_index(
            station
        )
        spec_window_size = 3
        iowaga_sample_index_list = [
            filter_iowaga_index_list[i : i + spec_window_size]
            for i in range(len(filter_iowaga_index_list) - spec_window_size)
        ]
        logger.info(
            f"before remove discontinuous data, sample num: {len(iowaga_sample_index_list)}"
        )
        # remove discontinuous data
        for i, sample in enumerate(iowaga_sample_index_list):
            if (sample[-1] - sample[0]) > spec_window_size:
                iowaga_sample_index_list[i] = None

        iowaga_sample_index_list = [
            sample for sample in iowaga_sample_index_list if sample is not None
        ]
        logger.info(
            f"after remove discontinuous data, sample num: {len(iowaga_sample_index_list)}"
        )
