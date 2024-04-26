import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.append(os.getcwd())
from config import Config
from utils.preprocess.prepare.scale_spec import *


def prepare_cdip_data(config=Config()):
    data_path = f"{config.processed_data_dir}/CDIP/filter/{config.y_data_desc}/{config.y_location}.npy"
    save_dir = config.get_y_data_dir()
    spec = np.load(data_path)

    spec_train, spec_test, scaler = scale_spec2d(
        spec, spec_scale_bins=config.y_spec_scale_bins
    )

    np.save(f"{save_dir}/train.npy", spec_train.astype(np.float32))
    np.save(f"{save_dir}/test.npy", spec_test.astype(np.float32))
    pickle.dump(scaler, open(f"{save_dir}/scaler.pkl", "wb"))
    logger.success(f"save spec & scaler to {save_dir}")


if __name__ == "__main__":
    config = Config()
    config.y_data_source = "CDIP"

    y_locations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]

    spec_var_names = ["MEM_dir36_smooth_freq_dir_time"]
    for location in y_locations:
        config.y_location = location
        for spec_var in spec_var_names:
            config.y_data_desc = spec_var
            prepare_cdip_data(config)
