from datetime import datetime

import torch.nn as nn
from loguru import logger

from config import Config
from predict_metrics import (
    plot_output_spec_samples,
    process_predict_spec2d,
)
from train import build_prediction_model
from utils.plot.metrics_plotter import plot_metrics
from utils.preprocess.extract.spec_fetcher import *
from models.model import load_model


def run_for_train():
    for location in ["CDIP028"]:  # , "CDIP045", "CDIP067", "CDIP093", "CDIP107"

        logger.info(f"current buyo: {location}")
        run_config(in_num=7, in_wind=True, window_size=3, y_location=location)
        run_config(
            in_num=7,
            in_wind=False,
            window_size=3,
            y_location=location,
            y_data_source="CDIP",
        )

        # run_config(in_num=7, in_wind=False, window_size=1, y_location=location)
        # run_config(in_num=1, in_wind=True, window_size=1, y_location=location)
        # run_config(in_num=1, in_wind=False, window_size=3, y_location=location)
        # run_config(in_num=1, in_wind=False, window_size=1, y_location=location)


def run_for_test():
    run_config(
        in_num=7,
        in_wind=True,
        window_size=3,
        y_location="CDIP028",
        train=False,
        eval=True,
    )


def run_config(
    in_num,
    in_wind,
    window_size,
    y_location,
    y_data_source="IOWAGA",
    train=True,
    eval=True,
):
    config = Config()
    config.y_location = y_location
    config.input_point_num = in_num
    config.input_wind = in_wind
    config.spec_window_size = window_size
    config.set_dataset_exp_param(y_data_source)
    config.set_input_channels()
    config.set_model()

    config.set_comment()

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logger.add(f"{config.get_log_dir()}/{time}.log")
    logger.info(f"config.comment: {config.comment}")

    if train:
        build_prediction_model(config)
    if eval:
        y_predict, y_true = process_predict_spec2d(config=config)
        plot_output_spec_samples(y_predict, y_true, config=config)
        plot_metrics(y_predict, y_true, config)

    logger.remove(file_handler)


if __name__ == "__main__":
    # run_for_train()
    run_for_test()
