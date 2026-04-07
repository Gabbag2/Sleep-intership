import os
import sys
import json
from datetime import datetime

import click
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils import *
from models.models import SleepEventLSTMClassifier
from models.dataset import SleepEventClassificationDataset as Dataset
from models.dataset import sleep_event_finetune_full_collate_fn as collate_fn


def masked_cross_entropy_loss(outputs, y_data, mask):
    """
    outputs: (B, seq_len, num_classes)
    y_data:  (B, seq_len)
    mask:    (B, seq_len)
    """
    B, seq_len, num_classes = outputs.shape

    outputs = outputs.reshape(B * seq_len, num_classes)
    y_data = y_data.reshape(B * seq_len).long()
    mask = mask.reshape(B * seq_len)

    class_weights = {
        0: 1,
        1: 4,
        2: 2,
        3: 4,
        4: 3
    }

    weights_tensor = torch.ones(num_classes, device=outputs.device)
    for cls, weight in class_weights.items():
        weights_tensor[cls] = weight

    loss = F.cross_entropy(outputs, y_data, weight=weights_tensor, reduction="none")

    valid_mask = (mask == 0).float()
    loss = loss * valid_mask

    denom = valid_mask.sum()
    if denom == 0:
        return loss.sum()

    loss = loss.sum() / denom
    return loss


@click.command("finetune_sleep_staging")
@click.option(
    "--config_path",
    type=str,
    default=os.path.normpath(os.path.join(ROOT_DIR, "configs", "config_finetune_sleep_events.yaml"))
)
@click.option(
    "--channel_groups_path",
    type=str,
    default=os.path.normpath(os.path.join(ROOT_DIR, "configs", "channel_groups.json"))
)
@click.option("--checkpoint_path", type=str, default=None)
@click.option(
    "--split_path",
    type=str,
    default=os.path.normpath(os.path.join(ROOT_DIR, "configs", "dataset_split.json"))
)
@click.option("--train_split", type=str, default="train")
def finetune_sleep_staging(config_path, channel_groups_path, checkpoint_path, split_path, train_split):
    config = load_config(config_path)
    channel_groups = load_config(channel_groups_path)

    if split_path:
        config["split_path"] = split_path

    channel_like = config["channel_like"]
    channel_like_string = "_".join(channel_like)

    if checkpoint_path:
        output = checkpoint_path
        config = load_data(os.path.join(output, "config.json"))
    else:
        output = os.path.join(
            config["model_path"],
            f"{config['model']}_custom_scoring_{channel_like_string}"
        )
        os.makedirs(output, exist_ok=True)

    os.makedirs(os.path.join(output, "plots"), exist_ok=True)

    logger.info(f"Model path: {output}")
    logger.info(f"Split Path: {config['split_path']}")
    logger.add(os.path.join(output, "training.log"), rotation="10 MB")
    logger.info("Loaded configuration file")
    logger.info(f"Batch Size: {config['batch_size']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_params = config["model_params"]
    model_class = getattr(sys.modules[__name__], config["model"])
    logger.info(f"Model Class: {config['model']}")
    model = model_class(**model_params).to(device)
    model_name = type(model).__name__

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    logger.info(f"Model initialized: {model_name}")
    total_layers, total_params = count_parameters(model)
    logger.info(f"Trainable parameters: {total_params / 1e6:.2f} million")
    logger.info(f"Number of layers: {total_layers}")

    logger.info("Loading Data...")

    train_dataset = Dataset(config, channel_groups, split=train_split)
    val_dataset = Dataset(config, channel_groups, split="validation")

    num_workers = config.get("num_workers", 4)
    batch_size = config.get("batch_size", 1)

    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    logger.info("Data Loaded!")

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if checkpoint_path:
        ckpt_file = os.path.join(output, "checkpoint.pth")
        if os.path.isfile(ckpt_file):
            logger.info(f"Loading checkpoint '{ckpt_file}'")
            checkpoint = torch.load(ckpt_file, map_location=device)
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))

            if isinstance(model, nn.DataParallel):
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                new_sd = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
                model.load_state_dict(new_sd, strict=False)

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        else:
            logger.info("Checkpoint path provided but checkpoint.pth not found. Starting from scratch.")

    if config.get("use_wandb", False):
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(project="PSG-fm", name=f"run_at_{current_timestamp}", config=config)

    num_epochs = config.get("epochs", 8)
    save_iter = config.get("save_iter", 100)
    log_interval = config.get("log_interval", 10)
    eval_iter = config.get("eval_iter", 50)

    history_path = os.path.join(output, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            full_history = json.load(f)
    else:
        full_history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for i, (x_data, y_data, padded_matrix, hdf5_path_list) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        ):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            padded_matrix = padded_matrix.to(device)

            outputs, mask = model(x_data, padded_matrix)
            loss = masked_cross_entropy_loss(outputs, y_data, mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / (i + 1)
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}"
                )
                if config.get("use_wandb", False):
                    wandb.log({
                        "Train Loss": avg_loss,
                        "Step": (epoch * len(train_loader)) + i + 1
                    })

            if (i + 1) % save_iter == 0:
                checkpoint_file = os.path.join(output, "checkpoint.pth")
                avg_loss = running_loss / (i + 1)

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "best_val_loss": best_val_loss
                }, checkpoint_file)

                save_data(config, os.path.join(output, "config.json"))
                logger.info(f"Checkpoint saved at {checkpoint_file}")

            if (i + 1) % eval_iter == 0:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for x_val, y_val, padded_val, _ in tqdm(
                        val_loader,
                        desc=f"Validation Epoch {epoch + 1}/{num_epochs}"
                    ):
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                        padded_val = padded_val.to(device)

                        outputs_val, mask_val = model(x_val, padded_val)
                        loss_val = masked_cross_entropy_loss(outputs_val, y_val, mask_val)
                        val_loss += loss_val.item()

                val_loss /= len(val_loader)
                logger.info(
                    f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}], "
                    f"Iteration [{i + 1}]: {val_loss:.4f}"
                )

                if config.get("use_wandb", False):
                    wandb.log({
                        "Validation Loss": val_loss,
                        "Step": (epoch * len(train_loader)) + i + 1
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(output, "best.pth")
                    torch.save(model.state_dict(), best_model_path)
                    save_data(config, os.path.join(output, "config.json"))
                    logger.info(f"Best model saved at {best_model_path}")

                model.train()

        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

        if config.get("use_wandb", False):
            wandb.log({
                "Epoch": epoch + 1,
                "Loss": epoch_loss
            })

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val, padded_val, _ in tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch + 1}/{num_epochs}"
            ):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                padded_val = padded_val.to(device)

                outputs_val, mask_val = model(x_val, padded_val)
                loss_val = masked_cross_entropy_loss(outputs_val, y_val, mask_val)
                val_loss += loss_val.item()

        val_loss /= len(val_loader)
        logger.info(f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {val_loss:.4f}")

        if config.get("use_wandb", False):
            wandb.log({
                "Validation Loss": val_loss,
                "Epoch": epoch + 1
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            save_data(config, os.path.join(output, "config.json"))
            logger.info(f"Best model saved at {best_model_path}")

        scheduler.step(val_loss)

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss
        }
        full_history.append(history_entry)

        with open(history_path, "w") as f:
            json.dump(full_history, f, indent=4)

        logger.info(f"History updated: {history_path}")

        model.train()

    logger.info("Training finished.")


if __name__ == "__main__":
    finetune_sleep_staging()