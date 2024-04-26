import importlib
import os
import platform
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from packaging import version

import sys

sys.path.append(os.getcwd())

from config import Config


def get_model_info(config=Config()):
    module = importlib.import_module(config.module_filename)
    Net = getattr(module, config.network_type)

    model = Net(
        input_channels=config.input_channels,
        output_size=config.output_size,
    ).to(config.device)

    return model


def load_model(config=Config()):
    """
    load model from checkpoint
    """
    model = get_model_info(config)
    start_epoch = 0
    checkpoint_model_path = f"{config.get_model_dir()}/best_model.pt"
    if os.path.exists(checkpoint_model_path):
        checkpoint = torch.load(checkpoint_model_path)
        model.load_state_dict(checkpoint)
        logger.success(f"Loaded model from checkpoint: {checkpoint_model_path}")
    else:
        logger.warning(
            f"Model file {checkpoint_model_path} does not exist, training from scratch!"
        )

    return model, start_epoch


def save_model(model, config=Config()):
    """
    save model to checkpoint
    """

    torch.save(model.state_dict(), f"{config.get_model_dir()}/best_model.pt")
