import glob
import os
import sys

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.append(os.getcwd())


def load_spec(spec_dir):
    pattern = f"{spec_dir}/*.npy"
    logger.info(f"pattern: {pattern}")
    file_list = glob.glob(pattern)
    file_list.sort()
    y = []

    for idx, file in tqdm(enumerate(file_list), desc=f"load {pattern}..."):
        data = np.load(file)
        y.append(data)

    y = np.concatenate(y, axis=0)
    logger.debug(f"y.shape: {y.shape}")
    return y.astype(np.float32)


def load_wind(filepath):
    pattern = f"{filepath}/*.npy"
    logger.info(f"pattern: {pattern}")
    file_list = glob.glob(pattern)
    file_list.sort()
    y = []
    for idx, file in tqdm(enumerate(file_list), desc=f"load {pattern}..."):
        data = np.load(file)
        y.append(data)

    y = np.concatenate(y, axis=0)

    logger.debug(f"wind.shape: {np.asarray(y).shape}")
    return y.astype(np.float32)
