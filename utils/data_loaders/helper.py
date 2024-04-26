import os
import sys

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from config import Config
from utils.data_loaders.seq_dataset import Seq_2In_Dataset, Seq_Dataset


def get_dataloader(shuffle=True, config=Config()):
    if config.input_wind:
        logger.info(f"load with wind data")
        train, test = load_data_with_wind(shuffle, config=config)
    else:
        logger.info(f"load without wind data")
        train, test = load_data_no_wind(shuffle, config=config)

    return train, test


def load_X_spec(config=Config()):
    if config.input_point_num == 7:
        X_locations = [
            "W1215N335",
            "W1210N330",
            "W1205N330",
            "W1205N325",
            "W1200N320",
            "W1195N320",
            "W1190N320",
        ]
    elif config.input_point_num == 1:
        X_locations = [config.X_location]
    else:
        raise ValueError("input_point_num should be 1 or 7")

    if config.y_data_source == "CDIP":
        X_locations = [
            f"{X_location}_to_{config.y_location}" for X_location in X_locations
        ]

    X_train = []
    X_test = []

    for location in X_locations:
        X_train_loc = np.load(f"{config.get_X_data_dir(location)}/train.npy")
        X_test_loc = np.load(f"{config.get_X_data_dir(location)}/test.npy")

        logger.success(f"load X data from {config.get_X_data_dir(location)}")

        X_train.append(X_train_loc)
        X_test.append(X_test_loc)

    X_train = np.swapaxes(X_train, 0, 1)
    X_test = np.swapaxes(X_test, 0, 1)

    logger.debug(f"X_spec_train: {X_train.shape}, X_spec_test: {X_test.shape}")
    return X_train, X_test


def load_y_spec(config=Config()):
    y_train = np.load(f"{config.get_y_data_dir()}/train.npy")
    y_test = np.load(f"{config.get_y_data_dir()}/test.npy")
    logger.success(f"load y data from {config.get_y_data_dir()}")
    logger.debug(f"y_spec_train: {y_train.shape}, y_spec_test: {y_test.shape} ")

    return y_train, y_test


def load_IOWAGA_wind(config=Config()):
    wind_dir = f"{config.processed_data_dir}/{config.y_data_source}/input/wind_input/{config.y_location}"
    logger.success(f"load IOWAGA wind data from {wind_dir}")

    X_wind_train = np.load(f"{wind_dir}/train.npy")
    X_wind_test = np.load(f"{wind_dir}/test.npy")

    logger.debug(
        f"X_wind_train: {X_wind_train.shape}, X_wind_test: {X_wind_test.shape} "
    )
    return X_wind_train, X_wind_test


def load_data_no_wind(shuffle=True, config=Config()):
    X_train, X_test = load_X_spec(config)
    y_train, y_test = load_y_spec(config)
    train_dataset = Seq_Dataset(X_train, y_train, config.spec_window_size)
    test_dataset = Seq_Dataset(X_test, y_test, config.spec_window_size)
    logger.debug(
        f"train/test dataset samples: {len(train_dataset)}/{len(test_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=shuffle
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=shuffle
    )

    return train_dataloader, test_dataloader


def load_data_with_wind(shuffle=True, config=Config()):
    X_train, X_test = load_X_spec(config)
    X_wind_train, X_wind_test = load_IOWAGA_wind(config=config)
    y_train, y_test = load_y_spec(config)

    train_dataset = Seq_2In_Dataset(
        X_train, X_wind_train, y_train, config.spec_window_size
    )
    test_dataset = Seq_2In_Dataset(X_test, X_wind_test, y_test, config.spec_window_size)

    logger.debug(
        f"train/test dataset samples: {len(train_dataset)}/{len(test_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=shuffle
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=shuffle
    )

    return train_dataloader, test_dataloader
