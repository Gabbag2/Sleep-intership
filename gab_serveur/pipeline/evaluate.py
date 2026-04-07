import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, 
                              classification_report,
                              precision_recall_fscore_support)
from torch.utils.data import DataLoader
import click
from loguru import logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils import *
from models.models import SleepEventLSTMClassifier
from models.dataset import SleepEventClassificationDataset as Dataset
from models.dataset import sleep_event_finetune_full_collate_fn as collate_fn

STAGE_NAMES = ['Wake', 'Stage1', 'Stage2', 'Stage3', 'REM']

def plot_loss_curves(history_path, save_dir):
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs     = [h['epoch']      for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss']   for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss,   label='Val Loss',   marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Loss curves saved: {path}")

def plot_confusion_matrix(all_labels, all_preds, save_dir):
    cm         = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                vmin=0, vmax=100, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Recall %)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")

    return cm, cm_percent

def plot_per_class_metrics(all_labels, all_preds, save_dir):
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0,1,2,3,4], zero_division=0
    )

    x = np.arange(len(STAGE_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x,         recall,    width, label='Recall')
    ax.bar(x + width, f1,        width, label='F1')

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Per-class Metrics on Validation Set')
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()

    path = os.path.join(save_dir, 'per_class_metrics.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Per-class metrics saved: {path}")

    # Log textuel
    logger.info("\n" + classification_report(
        all_labels, all_preds,
        labels=[0,1,2,3,4],
        target_names=STAGE_NAMES,
        digits=3, zero_division=0
    ))

@click.command()
@click.option("--model_dir", type=str, required=True,
              help="Dossier contenant best.pth et config.json")
@click.option("--config_path", type=str,
              default=os.path.join(ROOT_DIR, 'configs', 'config_finetune_sleep_events.yaml'))
@click.option("--channel_groups_path", type=str,
              default=os.path.join(ROOT_DIR, 'configs', 'channel_groups.json'))

def evaluate(model_dir, config_path, channel_groups_path):

    save_dir = os.path.join(model_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    config         = load_config(config_path)
    channel_groups = load_config(channel_groups_path)

    history_path = os.path.join(model_dir, 'history.json')
    if os.path.exists(history_path):
        plot_loss_curves(history_path, save_dir)
    else:
        logger.warning("history.json introuvable, pas de courbes de loss")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_params = config['model_params']
    model = SleepEventLSTMClassifier(**model_params).to(device)

    best_path = os.path.join(model_dir, 'best.pth')
    state_dict = torch.load(best_path, map_location=device)
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    logger.info(f"Modèle chargé depuis {best_path}")

    val_dataset = Dataset(config, channel_groups, split="validation")
    val_loader  = DataLoader(val_dataset, batch_size=2,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=collate_fn)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x_data, y_data, padded_matrix, _ in val_loader:
            x_data       = x_data.to(device)
            y_data       = y_data.to(device)
            padded_matrix = padded_matrix.to(device)

            outputs, mask = model(x_data, padded_matrix)
            preds = torch.argmax(outputs, dim=-1)

            valid = (mask == 0).cpu().numpy().flatten()
            all_preds.extend(preds.cpu().numpy().flatten()[valid])
            all_labels.extend(y_data.cpu().numpy().flatten()[valid])

    plot_confusion_matrix(all_labels, all_preds, save_dir)
    plot_per_class_metrics(all_labels, all_preds, save_dir)

    logger.info(f"\nTous les plots sont dans : {save_dir}")

if __name__ == "__main__":
    evaluate()