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
from utils.preprocess.extract.load_data import load_spec
from utils.preprocess.extract.spec_matcher import Spec_Matcher


def extract_wave_data(id, data_dir, save_dir, START_YEAR=1993, END_YEAR=2022):
    """
    :param id: buyo id
    :param data_dir: directory of the data
    :param save_dir: directory to save the extracted data
    :param START_YEAR: the start year of the data
    :param END_YEAR: the end year of the data

    read the netcdf file from START_YEAR to END_YEAR,
    extract the data corresponding to the buoy id, and save it as npy file
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

        spec_list = []
        for i, file in enumerate(tqdm(file_list, desc=f"Reading {year}...")):
            logger.info(f"\nProcessing {year}#{i+1}...")
            logger.info(f"Reading {file}...")
            spec = Dataset(file).variables["efth"]
            spec = spec[:, 0, :, :]
            spec = np.nan_to_num(spec)  # Convert masked value to 0

            spec_list.append(spec)
        spec = np.concatenate(spec_list, axis=0)
        logger.debug(f"Shape of {year} is {spec.shape}")
        np.save(f"{save_dir}/{year}.npy", spec)
        logger.success(f"Save {year}.npy successfully!")


def get_filter_input_iowaga_spec(iowaga_position, target_station, config=Config()):
    """
    :param iowaga_position: the position of the iowaga data
    :param target_station: the station of the CDIP data
    :param config: the config object

    load the iowaga data from the file if it exists,
    otherwise, filter the iowaga data to match cdip data time and save it as a file
    """
    iowaga_spec_save_path = f"{config.processed_data_dir}/IOWAGA/filter/input/{iowaga_position}_to_{target_station}.npy"
    if os.path.exists(iowaga_spec_save_path):
        logger.success(f"load data from exist file {iowaga_spec_save_path}")
    else:
        logger.info(f"start filter {iowaga_position} iowaga data")
        Path(iowaga_spec_save_path).parent.mkdir(parents=True, exist_ok=True)
        iowaga_spec_dir = (
            f"{config.processed_data_dir}/IOWAGA/extract/{iowaga_position}"
        )

        print(f"clip_index_station: {target_station}")
        _, iowaga_index_list = Spec_Matcher(target_station).get_filter_data_index()
        input_iowaga_spec_list = load_spec(iowaga_spec_dir)
        logger.debug(f"iowaga_spec.shape: {input_iowaga_spec_list.shape}")
        input_iowaga_spec_list = input_iowaga_spec_list[iowaga_index_list]
        logger.debug(f"iowaga_spec.shape after filter: {input_iowaga_spec_list.shape}")

        np.save(f"{iowaga_spec_save_path}", input_iowaga_spec_list)
        logger.success(f"save iowaga {iowaga_position} data success")
    return np.load(iowaga_spec_save_path)


if __name__ == "__main__":
    config = Config()

    # extract iowaga target data
    save_dir = f"{config.processed_data_dir}/IOWAGA/extract"
    data_dir = f"{config.raw_data_dir}/IOWAGA/SPEC_AT_BUOYS"
    for buyo_id in ["CDIP028", "CDIP045", "46219", "CDIP093", "CDIP107"]:
        extract_wave_data(buyo_id, data_dir, save_dir)

    # extract iowaga input data
    data_dir = f"{config.raw_data_dir}/IOWAGA/SPEC_NW139TO100"
    for loc in [
        "W1215N335",
        "W1210N330",
        "W1205N330",
        "W1205N325",
        "W1200N320",
        "W1195N320",
        "W1190N320",
    ]:
        extract_wave_data(loc, data_dir, save_dir)

    # filter iowaga input data to match cdip data time
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
            get_filter_input_iowaga_spec(input_station, target_station, config=config)
