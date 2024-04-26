import numpy as np
import torch


class Seq_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data, spec_window_size):
        self.input_data = input_data
        self.target_data = target_data
        self.spec_window_size = spec_window_size
        self.sample_size = (len(input_data) - spec_window_size) + 1

    def __getitem__(self, idx):
        """
        Get a single sample with shape (window_size, longitude, latitude, feature)

        :param idx: The index of the sample to retrieve
        :return: A tuple containing the input and target data for the sample
        """
        end_idx = idx + self.spec_window_size
        spec = self.input_data[idx:end_idx]
        # spec 形状为 (window_size, channels, height, width)
        # 将其转换为 (window_size*channels, height, width)
        spec = spec.reshape(-1, spec.shape[2], spec.shape[3])
        target = self.target_data[end_idx - 1]
        return spec, target

    def __len__(self):
        """
        Calculate the number of samples in the dataset
        """
        return self.sample_size


class Seq_2In_Dataset(torch.utils.data.Dataset):

    def __init__(self, spec_input_data, wind_input_data, target_data, spec_window_size):
        # assert len(spec_data) == len(
        #     target_data
        # ), "Input data and target data must have the same length."
        self.input_data = spec_input_data
        self.wind_input_data = wind_input_data
        self.target_data = target_data
        self.spec_window_size = spec_window_size
        self.sample_size = (len(spec_input_data) - spec_window_size) + 1

    def __getitem__(self, idx):
        """
        Get a single sample with shape (window_size, longitude, latitude, feature)

        :param idx: The index of the sample to retrieve
        :return: A tuple containing the input and target data for the sample
        """
        end_idx = idx + self.spec_window_size
        spec = self.input_data[idx:end_idx]
        # trans to (window_size*channels, height, width)
        spec = spec.reshape(-1, spec.shape[2], spec.shape[3])
        wind = self.wind_input_data[end_idx - 1]
        target = self.target_data[end_idx - 1]

        return spec, wind, target

    def __len__(self):
        """
        Calculate the number of samples in the dataset
        """
        return self.sample_size
