import os
import re
from pathlib import Path

import torch
import torch.nn as nn

from models.lion import Lion


class Config:
    def __init__(self):
        self.X_location = "W1205N330"  # W1205N330 or offshore
        self.X_data_desc = "spec_input"  # spec_input or interpolated_spec_input_dir36

        self.y_location = "CDIP028"
        self.y_data_source = "IOWAGA"  # IOWAGA or CDIP
        self.y_data_desc = "spec_output"  # spec_output or MEM36

        self.input_point_num = 1
        self.input_wind = False
        self.spec_window_size = 1

        self.input_channels = None

        self.X_spec_scale_bins = "freq"  # "freq_and_dir" or "freq" or "none"
        self.y_spec_scale_bins = "freq_and_dir"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = 100
        self.batch_size = 128
        self.patience = 10
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2

        self.criterion = nn.MSELoss()  # nn.MSELoss() and nn.L1Loss()
        self.optimizer = Lion  # Lion or Adam torch.optim.Adam
        self.output_size = 1

        self.network_type = "UNet"
        self.module_filename = f"models.{self.network_type}"

        self.run_id = None
        self.comment = None
        # Path & Dir
        self.raw_data_dir = "/mnt/e/data/raw"
        self.processed_data_dir = "/mnt/e/data/processed"

    def set_comment(self):
        wind_desc = "Wind" if self.input_wind else ""
        X_location = "Offshore" if self.input_point_num == 7 else "W1205N330"

        self.comment = f"{self.y_data_source}{self.y_location[-3:]}_{X_location}{self.spec_window_size}_{wind_desc}"

    def set_input_channels(self):
        input_wind = 1 if self.input_wind else 0
        self.input_channels = self.input_point_num * self.spec_window_size + input_wind

    def set_dataset_exp_param(self, y_data_source):
        self.y_data_source = y_data_source
        if self.is_y_cdip_buyo():
            self.X_data_desc = "interpolated_spec_input_dir36"
            self.y_data_desc = "MEM36"
        else:
            self.X_data_desc = "spec_input"
            self.y_data_desc = "spec_output"

    def set_model(self):
        if self.input_wind:
            self.network_type = f"UNetW_{self.y_data_source}"
            self.module_filename = f"models.{self.network_type}"
        else:
            self.network_type = f"UNet"
            self.module_filename = f"models.{self.network_type}"

    def is_y_cdip_buyo(self):
        return self.y_data_source != "IOWAGA"

    def get_X_data_dir(self, X_location=None):
        if X_location is None:
            X_location = self.X_location

        input_data_dir = f"{self.processed_data_dir}/IOWAGA/input/{self.X_data_desc}_minmax_{self.X_spec_scale_bins}/{X_location}"
        Path(input_data_dir).mkdir(parents=True, exist_ok=True)
        return input_data_dir

    def get_y_data_dir(self):
        target_data_dir = f"{self.processed_data_dir}/{self.y_data_source}/output/{self.y_data_desc}_minmax_{self.y_spec_scale_bins}/{self.y_location}"
        Path(target_data_dir).mkdir(parents=True, exist_ok=True)
        return target_data_dir

    def get_log_dir(self):
        log_dir = f"{os.getcwd()}/results/logs/{self.y_location}/{self.comment}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_model_dir(self):
        model_dir = (
            f"{os.getcwd()}/results/checkpoints/{self.y_location}/{self.comment}"
        )
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_samples_figure_save_dir(self):
        samples_save_dir = (
            f"{os.getcwd()}/results/samples_output/{self.y_location}/{self.comment}"
        )
        Path(samples_save_dir).mkdir(parents=True, exist_ok=True)
        return samples_save_dir

    def get_seasons_figure_save_dir(self):
        seasons_save_dir = (
            f"{os.getcwd()}/results/seasons/{self.y_location}/{self.comment}"
        )
        Path(seasons_save_dir).mkdir(parents=True, exist_ok=True)
        return seasons_save_dir

    def get_evaluate_figure_save_dir(self):
        evaluate_save_dir = (
            f"{os.getcwd()}/results/evaluate/{self.y_location}/{self.comment}"
        )
        Path(evaluate_save_dir).mkdir(parents=True, exist_ok=True)
        return evaluate_save_dir
