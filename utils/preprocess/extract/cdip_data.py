import os
from pathlib import Path

import numpy as np
from loguru import logger
from netCDF4 import Dataset

import sys

sys.path.append(os.getcwd())

from config import Config


def extract_cdip_data(station, var_name, save_dir, config=Config()):
    """
    :param station: buoy id
    :param var_name: name of the variable to extract
    :param save_dir: directory to save the extracted data
    :param END_YEAR: the end year of the data

    read the netcdf file from START_YEAR to END_YEAR,
    extract the data corresponding to the buoy id, and save it as npy file
    """
    data_path = f"{config.raw_data_dir}/CDIP/{station}/{station[-3:]}p1_historic.nc"

    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)
    logger.info(f"\nProcessing station {station}...")

    logger.info(f"Reading file {data_path}...")
    nc_file = Dataset(data_path)

    var_data = nc_file.variables[var_name]
    var_data = np.nan_to_num(var_data)  # Convert NaN value to 0

    logger.debug(f"before filter, {station}:{var_name}: {var_data.shape}")
    # filter cdip data to match the time of iowaga data
    var_data = filter_cdip_data(station, var_data)
    logger.debug(f"after filter, {station}:{var_name}: {var_data.shape}")

    logger.debug(f"Shape of {station}:{var_name} is {var_data.shape}")
    np.save(f"{save_dir}/{station}.npy", var_data)
    logger.success(f"Save {station}:{var_name} successfully!")


def filter_cdip_data(station, data_list):
    """
    :param station: buoy id
    :param data_list: data list of the cdip buoy
    :param index_list: index which get from the iowaga data
    """
    index_list = np.load(f"{config.processed_data_dir}/CDIP/index/{station}_index.npy")
    data_list = np.array(data_list)
    data_list = data_list[index_list]
    return data_list


if __name__ == "__main__":
    config = Config()

    stations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    var_names = [
        "waveHs",
        "waveEnergyDensity",
        "waveA1Value",
        "waveA2Value",
        "waveB1Value",
        "waveB2Value",
    ]
    for station in stations:
        for var_name in var_names:
            save_dir = f"{config.processed_data_dir}/CDIP/filter/{var_name}"
            extract_cdip_data(
                station, var_name=var_name, save_dir=save_dir, config=config
            )
