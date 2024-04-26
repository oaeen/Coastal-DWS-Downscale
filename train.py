import gc
import profile
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torchinfo
from loguru import logger
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models.model import load_model, save_model
from utils.data_loaders.helper import get_dataloader
from utils.metrics.perf_recorder import *


def train(model, dataloader, optimizer, criterion, config=Config()):
    model.train()
    train_loss = 0
    scaler = amp.GradScaler()

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs, targets = (inputs.to(config.device), targets.to(config.device))

        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    return train_loss / len(dataloader)


@torch.no_grad()
def test(model, dataloader, criterion, config=Config()):
    model.eval()
    val_loss = 0
    all_outputs = []
    all_targets = []
    val_loss_dict = {}

    for inputs, targets in dataloader:
        inputs, targets = (inputs.to(config.device), targets.to(config.device))

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_loss_dict["MSE"] = val_loss / len(dataloader)
    val_loss_dict["R"], _ = pearsonr(all_targets.flatten(), all_outputs.flatten())

    return val_loss_dict


def train_with_wind(model, dataloader, optimizer, criterion, config=Config()):
    model.train()
    train_loss = 0
    scaler = amp.GradScaler()  # 创建梯度缩放器

    for spec_inputs, wind_inputs, targets in dataloader:
        optimizer.zero_grad()
        spec_inputs, wind_inputs, targets = (
            spec_inputs.to(config.device),
            wind_inputs.to(config.device),
            targets.to(config.device),
        )

        with amp.autocast():
            outputs = model(spec_inputs, wind_inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    return train_loss / len(dataloader)


@torch.no_grad()
def test_with_wind(model, dataloader, criterion, config=Config()):
    model.eval()
    val_loss = 0
    all_outputs = []
    all_targets = []
    val_loss_dict = {}

    for spec_inputs, wind_inputs, targets in dataloader:
        spec_inputs, wind_inputs, targets = (
            spec_inputs.to(config.device),
            wind_inputs.to(config.device),
            targets.to(config.device),
        )

        outputs = model(spec_inputs, wind_inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

        all_outputs.append(outputs.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_loss_dict["MSE"] = val_loss / len(dataloader)
    val_loss_dict["R"], _ = pearsonr(all_targets.flatten(), all_outputs.flatten())

    return val_loss_dict


def build_prediction_model(config=Config()):
    logger.info("Training model...")

    train_dataloader, test_dataloader = get_dataloader(True, config)
    model, start_epoch = load_model(config)
    backup_model_and_config(config)

    writer = SummaryWriter(log_dir=config.get_log_dir())

    criterion = config.criterion
    optimizer = config.optimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, verbose=True, min_lr=1e-7
    )

    best_loss = float("inf")
    best_loss_epoch = -1
    no_improvement_count = 0

    if config.input_wind:
        train_func = train_with_wind
        test_func = test_with_wind
    else:
        train_func = train
        test_func = test

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()

        train_loss = train_func(model, train_dataloader, optimizer, criterion, config)
        val_loss_dict = test_func(model, test_dataloader, criterion, config)

        t1 = time.time()
        val_loss = val_loss_dict["MSE"]
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_loss_epoch = epoch
            no_improvement_count = 0
            save_model(model, config)
            logger.success(f"improved, model saved.")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} epochs.")
            if no_improvement_count >= config.patience:
                logger.success("Early stopping triggered, stopping the training.")
                break

        report_perf(
            writer=writer,
            current_epoch=epoch,
            run_time=t1 - t0,
            train_loss=train_loss,
            val_loss_dict=val_loss_dict,
            best_val_loss=best_loss,
            best_val_loss_epoch=best_loss_epoch,
            TOTAL_EPOCHS=config.epochs,
        )

    torch.cuda.empty_cache()
    gc.collect()
    logger.success("Training completed.")


if __name__ == "__main__":
    config = Config()
    build_prediction_model(config)
