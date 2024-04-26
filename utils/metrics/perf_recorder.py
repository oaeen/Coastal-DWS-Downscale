import os
import shutil
import sys

import pandas as pd
from loguru import logger

sys.path.append(os.getcwd())

from config import Config


def report_perf(
    writer,
    current_epoch,
    run_time,
    train_loss,
    val_loss_dict,
    best_val_loss=None,
    best_val_loss_epoch=None,
    TOTAL_EPOCHS=1000,
):
    """
    Report performance metrics
    """
    logger.info(
        f"Epoch {current_epoch + 1}/{TOTAL_EPOCHS} | Time: {run_time:.2f}s | Best: {best_val_loss:.7f} in epoch {best_val_loss_epoch+1}"
    )
    val_mse = val_loss_dict["MSE"]
    val_R = val_loss_dict["R"]
    logger.info(
        f"Train MSE: {train_loss:.7f} | Val: MSE: {val_mse:.7f} | R: {val_R:.4f}"
    )

    writer.add_scalar("Loss/train", train_loss, current_epoch)
    writer.add_scalar("Loss/val", val_mse, current_epoch)
    writer.add_scalar("R/val", val_R, current_epoch)


def backup_model_and_config(config=Config()):
    log_dir = config.get_log_dir()
    model_file = f"{os.getcwd()}/models/{config.network_type}.py"
    shutil.copy(model_file, log_dir)

    config_dict = config.__dict__
    df = pd.DataFrame(config_dict.items(), columns=["name", "value"])
    df.to_csv(f"{log_dir}/config.csv", index=False)
