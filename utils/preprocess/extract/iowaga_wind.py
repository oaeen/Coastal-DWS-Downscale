import glob
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from netCDF4 import Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())

from config import Config


def extract_wind_data(id, data_dir, save_dir, START_YEAR=1993, END_YEAR=2022):
    """
    :param id: buyo id
    :param data_dir: directory of the data
    :param save_dir: directory to save the extracted data
    :param START_YEAR: the start year of the data
    :param END_YEAR: the end year of the data

    read the netcdf file from START_YEAR to END_YEAR,
    extract the wind data, apply MinMaxScaler, and save it as npy file
    """
    save_dir = f"{save_dir}/{id}"
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="extracting..."):
        if os.path.exists(f"{save_dir}/{year}.npy"):
            logger.success(f"{year}.npy exists")
            continue

        logger.info(f"\nProcessing year {year}...")
        year_folder = os.path.join(data_dir, str(year))
        file_list = glob.glob(os.path.join(year_folder, f"*{id}*"))
        file_list.sort()

        # dilute data
        wind_speed_list = []
        wind_dir_cos_list = []
        wind_dir_sin_list = []
        for i, file in enumerate(tqdm(file_list, desc=f"Reading {year}...")):
            logger.info(f"\nProcessing {year}#{i+1}...")
            logger.info(f"Reading {file}...")
            nc_file = Dataset(file)
            wind_speed = nc_file.variables["wnd"]
            wind_speed = wind_speed[:, 0]
            wind_speed = np.nan_to_num(wind_speed)
            wind_speed = wind_speed.filled(0)

            wind_dir = nc_file.variables["wnddir"]
            wind_dir = wind_dir[:, 0]
            wind_dir = np.nan_to_num(wind_dir)
            wind_dir = wind_dir.filled(0)
            wind_dir = wind_dir * np.pi / 180

            wind_dir_cos = np.cos(wind_dir)
            wind_dir_sin = np.sin(wind_dir)

            wind_speed_list.append(wind_speed)
            wind_dir_cos_list.append(wind_dir_cos)
            wind_dir_sin_list.append(wind_dir_sin)

        wind_speed = np.concatenate(wind_speed_list, axis=0)
        logger.info(
            f"year: {year}: max: {np.max(wind_speed)}, min: {np.min(wind_speed)}"
        )

        wind_dir_cos = np.concatenate(wind_dir_cos_list, axis=0)
        wind_dir_sin = np.concatenate(wind_dir_sin_list, axis=0)
        wind_info = np.stack((wind_speed, wind_dir_cos, wind_dir_sin), axis=0)

        wind_info = np.swapaxes(wind_info, 0, 1)

        logger.debug(f"Shape of {year} is {wind_info.shape}")

        np.save(f"{save_dir}/{year}.npy", wind_info)
        logger.success(f"Save {year}.npy successfully!")


if __name__ == "__main__":
    config = Config()

    save_dir = f"{config.processed_data_dir}/IOWAGA/extract_wind"
    data_dir = f"{config.raw_data_dir}/IOWAGA/SPEC_AT_BUOYS"
    for buyo_id in ["CDIP028", "CDIP045", "46219", "CDIP093", "CDIP107"]:
        extract_wind_data(buyo_id, data_dir, save_dir)
