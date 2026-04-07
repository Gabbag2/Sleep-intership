from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, f1_score, confusion_matrix
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
import os
import glob
import pyedflib
import yaml
import json
import h5py
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import multiprocessing
from loguru import logger
import warnings
from scipy.signal import butter, filtfilt, resample
import mne
sys.path.append("..")
sys.path.append("../sleepfm")
"""from preprocessing.preprocessing import EDFToHDF5Converter"""
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer, SleepEventLSTMClassifier, DiagnosisFinetuneFullLSTMCOXPHWithDemo
from utils import load_config, load_data, save_data, count_parameters
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
device = torch.device("cuda")
print(device)


CHANNEL_GROUPS = {
    "EEG": ["Fp1", "C3", "O1", "C4"],       
    "EOG": ["EOG G", "EOG D"],
    "EMG": ["EMG 1", "EMG 2"],          
    "EKG": ["ECG"],
    "REF": ["A2"],   
    "RESP": ["Thermistance", "Flow",]                   
}
CHANNEL_OI = sorted({ch for grp in CHANNEL_GROUPS.values() for ch in grp})

CATEGORY_MAPPING = {
    "EEG": "BAS",
    "EOG": "BAS",
    "REF": "BAS",
    "ECG": "EKG",
    "EMG": "EMG",
    "RESP": "RESP",
    "EKG": "EKG"
}
CHANNEL_GROUPS_FOR_MODEL = {'BAS': [], 'EKG': [], 'RESP': [], 'EMG': []}

for category, channels in CHANNEL_GROUPS.items():
    model_category = CATEGORY_MAPPING.get(category, 'BAS')
    CHANNEL_GROUPS_FOR_MODEL[model_category].extend(channels)
print(CHANNEL_GROUPS_FOR_MODEL)

model_path = "../sleepfm/checkpoints/model_base"
channel_groups_path = "../sleepfm/configs/channel_groups.json"
config_path = os.path.join(model_path, "config.json")

config = load_config(config_path)
modality_types = config["modality_types"]
in_channels = config["in_channels"]
patch_size = config["patch_size"]
embed_dim = config["embed_dim"]
num_heads = config["num_heads"]
num_layers = config["num_layers"]
pooling_head = config["pooling_head"]
dropout = 0.0

print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Version PyTorch: {torch.__version__}")
model_class = getattr(sys.modules[__name__], config['model'])
model = model_class(in_channels, patch_size, embed_dim, num_heads, num_layers, pooling_head=pooling_head, dropout=dropout)

device = torch.device("cuda")
print(device)
if device.type == "cuda":
    model = torch.nn.DataParallel(model)

model.to(device)
total_layers, total_params = count_parameters(model)
print(f'Trainable parameters: {total_params / 1e6:.2f} million')
print(f'Number of layers: {total_layers}')

checkpoint = torch.load(os.path.join(model_path, "best.pt"))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

from glob import glob

hdf5_paths = "C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\sleep_fm\\sleepfm-clinical\\notebooks\\train_data"
hdf5_files = glob(os.path.join(hdf5_paths, "*.hdf5"))

dataset = SetTransformerDataset(config, CHANNEL_GROUPS_FOR_MODEL , hdf5_paths=hdf5_files, split="test")
dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=16, 
                                            num_workers=1, 
                                            shuffle=False, 
                                            collate_fn=collate_fn)

output = os.path.join(hdf5_paths, "train_emb")
output_5min_agg = os.path.join(hdf5_paths, "train_5min_agg_emb")
os.makedirs(output, exist_ok=True)
os.makedirs(output_5min_agg, exist_ok=True)

with torch.no_grad():
    with tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
            (bas, resp, ekg, emg) = batch_data
            (mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list

            bas = bas.to(device, dtype=torch.float)
            resp = resp.to(device, dtype=torch.float)
            ekg = ekg.to(device, dtype=torch.float)
            emg = emg.to(device, dtype=torch.float)

            mask_bas = mask_bas.to(device, dtype=torch.bool)
            mask_resp = mask_resp.to(device, dtype=torch.bool)
            mask_ekg = mask_ekg.to(device, dtype=torch.bool)
            mask_emg = mask_emg.to(device, dtype=torch.bool)

            embeddings = [
                model(bas, mask_bas),
                model(resp, mask_resp),
                model(ekg, mask_ekg),
                model(emg, mask_emg),
            ]
            # Model gives two kinds of embeddings. Granular 5 second-level embeddings and aggregated 5 minute-level embeddings. We save both of them below. 

            embeddings_new = [e[0].unsqueeze(1) for e in embeddings]

            for i in range(len(file_paths)):
                file_path = file_paths[i]
                chunk_start = chunk_starts[i]
                subject_id = os.path.basename(file_path).split('.')[0]
                output_path = os.path.join(output_5min_agg, f"{subject_id}.hdf5")

                with h5py.File(output_path, 'a') as hdf5_file:
                    for modality_idx, modality_type in enumerate(config["modality_types"]):
                        if modality_type in hdf5_file:
                            dset = hdf5_file[modality_type]
                            chunk_start_correct = chunk_start // (embed_dim * 5 * 60)
                            chunk_end = chunk_start_correct + embeddings_new[modality_idx][i].shape[0]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + embeddings_new[modality_idx][i].shape[1:])
                            dset[chunk_start_correct:chunk_end] = embeddings_new[modality_idx][i].cpu().numpy()
                        else:
                            hdf5_file.create_dataset(modality_type, data=embeddings_new[modality_idx][i].cpu().numpy(), chunks=(embed_dim,) + embeddings_new[modality_idx][i].shape[1:], maxshape=(None,) + embeddings_new[modality_idx][i].shape[1:])

            embeddings_new = [e[1] for e in embeddings]

            for i in range(len(file_paths)):
                file_path = file_paths[i]
                chunk_start = chunk_starts[i]
                subject_id = os.path.basename(file_path).split('.')[0]
                output_path = os.path.join(output, f"{subject_id}.hdf5")

                with h5py.File(output_path, 'a') as hdf5_file:
                    for modality_idx, modality_type in enumerate(config["modality_types"]):
                        if modality_type in hdf5_file:
                            dset = hdf5_file[modality_type]
                            chunk_start_correct = chunk_start // (embed_dim * 5)
                            chunk_end = chunk_start_correct + embeddings_new[modality_idx][i].shape[0]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + embeddings_new[modality_idx][i].shape[1:])
                            dset[chunk_start_correct:chunk_end] = embeddings_new[modality_idx][i].cpu().numpy()
                        else:
                            hdf5_file.create_dataset(modality_type, data=embeddings_new[modality_idx][i].cpu().numpy(), chunks=(embed_dim,) + embeddings_new[modality_idx][i].shape[1:], maxshape=(None,) + embeddings_new[modality_idx][i].shape[1:])
            pbar.update()
            
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        B, S, E = x.shape
        return x + self.encoding[:, :S, :]

class AttentionPooling(nn.Module):
    """Pooling spatial avec attention"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x, mask=None):
        # x: (B, C, E), mask: (B, C)
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, E)
        
        # Concat query with input
        x = torch.cat([query, x], dim=1)  # (B, 1+C, E)
        
        if mask is not None:
            # Add False for query position
            query_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([query_mask, mask], dim=1)
        
        x = self.transformer_layer(x, src_key_padding_mask=mask)
        return x[:, 0, :]  # Return query output

class MultiScaleTemporalBlock(nn.Module):
    """
    Capture des patterns temporels à plusieurs échelles:
    - 5-10s: Micro-éveils, mouvements oculaires rapides
    - 30s-1min: Transitions entre stades
    - 2-5min: Cycles de sommeil
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Échelle courte (local): Conv1D avec petit kernel
        self.short_scale = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Échelle moyenne: Conv1D avec kernel moyen
        self.medium_scale = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Échelle longue: Attention
        self.long_scale = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion des échelles
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # x: (B, S, E)
        residual = x
        
        # Short scale
        x_short = self.short_scale(x.transpose(1, 2)).transpose(1, 2)
        
        # Medium scale
        x_medium = self.medium_scale(x.transpose(1, 2)).transpose(1, 2)
        
        # Long scale avec attention
        if mask is not None:
            x_long, _ = self.long_scale(x, x, x, key_padding_mask=mask)
        else:
            x_long, _ = self.long_scale(x, x, x)
        
        # Fusion
        x_fused = torch.cat([x_short, x_medium, x_long], dim=-1)
        x_out = self.fusion(x_fused)
        
        # Residual connection
        x_out = self.norm(x_out + residual)
        
        return x_out

class SleepStagingModel(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=5,
        num_classes=5,
        pooling_head=4,
        dropout=0.1,
        max_seq_length=2160,
    ):
        super().__init__()
        
        if max_seq_length is None:
            max_seq_length = 20000
            
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale blocks
        self.temporal_blocks = nn.ModuleList([
            MultiScaleTemporalBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x, mask):
        B, C, S, E = x.shape
        
        # Spatial pooling
        x = rearrange(x, 'b c s e -> (b s) c e')
        mask_spatial = rearrange(mask[:, :, 0].unsqueeze(1).expand(-1, S, -1), 'b s c -> (b s) c').bool()
        x = self.spatial_pooling(x, mask_spatial)
        x = x.view(B, S, E)
        
        # Temporal modeling
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        
        mask_temporal = mask[:, 0, :].bool()
        for block in self.temporal_blocks:
            x = block(x, mask_temporal)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, mask_temporal
    
def load_data(file_path):
    with open(file_path, 'r') as f:
        if file_path.endswith('.yaml'):
            return yaml.safe_load(f)

config_path = "test.yaml"
config = load_data(config_path)
print(" Config chargée depuis test.yaml")
print(f" Model: {config.get('model', 'N/A')}")
print(f" Model params: {config.get('model_params', {})}")

sleep_staging_model = SleepStagingModel(**config['model_params'])
sleep_staging_model = sleep_staging_model.to(device)
sleep_staging_model = nn.DataParallel(sleep_staging_model)

print(f"\n Modèle créé : {type(sleep_staging_model).__name__}")
print(f"Nombre de paramètres : {sum(p.numel() for p in sleep_staging_model.parameters()):,}")

# ========== OPTIONNEL: CHARGER UN CHECKPOINT ==========
# Si vous avez un checkpoint pré-entraîné à charger
checkpoint_path = "C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\sleep_fm\\sleepfm-clinical\\notebooks\\save_training\\checkpoints\\model_sleep_staging\\best.pth"

if os.path.exists(checkpoint_path):
    print(f"\n Chargement du checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extraire le state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Retirer le préfixe "module." si nécessaire
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    # Charger (strict=False car votre architecture peut différer)
    sleep_staging_model.load_state_dict(new_state_dict, strict=False)
    print(" Checkpoint chargé")
else:
    print(f"\n  Aucun checkpoint trouvé à {checkpoint_path}")
    print("   Le modèle sera entraîné depuis zéro")
    
class SleepEventClassificationDataset(Dataset):
    def __init__(
        self,
        config,
        channel_groups,
        hdf5_paths,
        label_files,
        split="train",
    ):
        self.config = config
        self.max_channels = self.config["max_channels"]
        self.context = int(self.config["context"])
        self.channel_like = self.config["channel_like"]

        self.max_seq_len = config["model_params"]["max_seq_length"]

        # --- Build label lookup: {study_id: label_csv_path} ---
        # study_id = filename without extension, e.g. "SSC_12345"
        labels_dict = {
            os.path.basename(p).rsplit(".", 1)[0]: p
            for p in label_files
            if os.path.exists(p)
        }

        # --- Filter to HDF5s that exist and have a matching label file ---
        hdf5_paths = [p for p in hdf5_paths if os.path.exists(p)]
        hdf5_paths = [
            p for p in hdf5_paths
            if os.path.basename(p).rsplit(".", 1)[0] in labels_dict
        ]

        if config.get("max_files"):
            hdf5_paths = hdf5_paths[: config["max_files"]]

        self.hdf5_paths = hdf5_paths
        self.labels_dict = labels_dict

        # --- Build index map ---
        # Each item is (hdf5_path, label_path, start_index)
        
        if self.context == -1:
            self.index_map = [
                (p, labels_dict[os.path.basename(p).rsplit(".", 1)[0]], -1)
                for p in self.hdf5_paths
            ]
        else:
            self.index_map = []
            loop = tqdm(self.hdf5_paths, total=len(self.hdf5_paths), desc=f"Indexing {split} data")
            for hdf5_file_path in loop:
                file_prefix = os.path.basename(hdf5_file_path).rsplit(".", 1)[0]
                label_path = labels_dict[file_prefix]

                with h5py.File(hdf5_file_path, "r") as hf:
                    dset_names = list(hf.keys())
                    if len(dset_names) == 0:
                        continue

                    # Use first dataset to define length (same as your original behavior)
                    first_name = dset_names[0]
                    dataset_length = hf[first_name].shape[0]

                for i in range(0, dataset_length, self.context):
                    self.index_map.append((hdf5_file_path, label_path, i))

        # If you have logger, keep; otherwise you can remove these.
        # logger.info(f"Number of files in {split} set: {len(self.hdf5_paths)}")
        # logger.info(f"Number of files to be processed in {split} set: {len(self.index_map)}")

        self.total_len = len(self.index_map)

    def __len__(self):
        return self.total_len

    def get_index_map(self):
        return self.index_map

    def __getitem__(self, idx):
        hdf5_path, label_path, start_index = self.index_map[idx]

        
        labels_df = pd.read_csv(label_path)
        labels_df["StageNumber"] = labels_df["StageNumber"].replace(-1, 0)

        y_data = labels_df["StageNumber"].to_numpy()
        if self.context != -1:
            y_data = y_data[start_index : start_index + self.context]

        x_data = []
        with h5py.File(hdf5_path, "r") as hf:
            dset_names = list(hf.keys())
            for dataset_name in dset_names:
                if dataset_name in self.channel_like:
                    if self.context == -1:
                        x_data.append(hf[dataset_name][:])
                    else:
                        x_data.append(hf[dataset_name][start_index : start_index + self.context])

        if not x_data:
            # Skip this data point if x_data is empty
            return self.__getitem__((idx + 1) % self.total_len)

        x_data = np.array(x_data)  # (C, T, F) assuming each channel returns (T, F)
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)

        min_length = min(x_data.shape[1], len(y_data))
        x_data = x_data[:, :min_length, :]
        y_data = y_data[:min_length]

        return x_data, y_data, self.max_channels, self.max_seq_len, hdf5_path
    
def sleep_event_finetune_full_collate_fn(batch):
    x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

    num_channels = max(max_channels_list)

    max_seq_len_temp = max([item.size(1) for item in x_data])
    # Determine the max sequence length for padding
    if max_seq_len_list[0] is None:
        max_seq_len = max_seq_len_temp
    else:
        max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])

    padded_x_data = []
    padded_y_data = []
    padded_mask = []

    for x_item, y_item in zip(x_data, y_data):

        # first non-zero index of y_data
        #print(y_item.shape)
        tgt_sleep_no_sleep = np.where(y_item > 0, 1, 0)
        moving_avg_tgt_sleep_no_sleep = np.convolve(tgt_sleep_no_sleep, np.ones(1080)/1080, mode='valid')
        try:
            first_non_zero_index = np.where(moving_avg_tgt_sleep_no_sleep > 0.5)[0][0]
        except IndexError:
            first_non_zero_index = 0



        #non_zero_indices = (y_item != 0).nonzero(as_tuple=True)[0]
        #first_non_zero_index = non_zero_indices[0].item() - 20
        if first_non_zero_index < 0:
            first_non_zero_index = 0

        #first_non_zero_index = 0

        #print(f"First non-zero index of y_data: {first_non_zero_index}")
        # Get the shape of x_item
        c, s, e = x_item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len + first_non_zero_index)  # Ensure the sequence length doesn't exceed max_seq_len

        # Create a padded tensor and a mask tensor for x_data
        padded_x_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        #print(f"Shape of x_item: {x_item[:c, first_non_zero_index:s, :e].shape}")
        padded_x_item[:c, :s-first_non_zero_index, :e] = x_item[:c, first_non_zero_index:s, :e]
        mask[:c, :s-first_non_zero_index] = 0  # 0 for real data, 1 for padding

        # Pad y_data with zeros to match max_seq_len
        padded_y_item = torch.zeros(max_seq_len)
        padded_y_item[:s-first_non_zero_index] = y_item[first_non_zero_index:s]

        # Append padded items to lists
        padded_x_data.append(padded_x_item)
        padded_y_data.append(padded_y_item)
        padded_mask.append(mask)

    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    y_data = torch.stack(padded_y_data)
    padded_mask = torch.stack(padded_mask)

    '''
    for y_data_mini in y_data:
        unique_labels = torch.unique(y_data_mini)
        print(f"Unique labels in batch: {unique_labels}")
    '''

    return x_data, y_data, padded_mask, hdf5_path_list

def sleep_collate_easy(batch):
     # Unpack batch
    x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

    # Déterminer les dimensions max
    num_channels = max(max_channels_list)
    max_seq_len_temp = max([item.size(1) for item in x_data])
    
    if max_seq_len_list[0] is None:
        max_seq_len = max_seq_len_temp
    else:
        max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])

    padded_x_data = []
    padded_y_data = []
    padded_mask = []

    # Padding pour chaque élément du batch
    for x_item, y_item in zip(x_data, y_data):
        
        c, s, e = x_item.size()  # (canaux, séquence, features)
        c = min(c, num_channels)
        s = min(s, max_seq_len)

        # Créer tenseurs paddés
        padded_x_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))
        padded_y_item = torch.zeros(max_seq_len)

        # Copier les données réelles
        padded_x_item[:c, :s, :e] = x_item[:c, :s, :e]
        mask[:c, :s] = 0  # 0 = vraies données, 1 = padding
        padded_y_item[:s] = y_item[:s]

        # Ajouter aux listes
        padded_x_data.append(padded_x_item)
        padded_y_data.append(padded_y_item)
        padded_mask.append(mask)

    # Empiler en batch
    x_data = torch.stack(padded_x_data)      # (batch, channels, seq, features)
    y_data = torch.stack(padded_y_data)      # (batch, seq)
    padded_mask = torch.stack(padded_mask)   # (batch, channels, seq)

    return x_data, y_data, padded_mask, hdf5_path_list

from glob import glob
base_data_path = "C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\sleep_fm\\sleepfm-clinical\\notebooks\\train_data\\train_emb"
base_save_path = "C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\sleep_fm\\sleepfm-clinical\\notebooks\\save_training"

# Créer les dossiers de sauvegarde
checkpoint_dir = os.path.join(base_save_path, "checkpoints")
log_dir = os.path.join(base_save_path, "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Sauvegarder la config
with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

print(f" Checkpoints: {checkpoint_dir}")
print(f" Logs: {log_dir}")

# Trouver tous les fichiers HDF5
hdf5_files = glob(os.path.join(base_data_path, "*.hdf5"))

file_pairs = []

for hdf5_path in hdf5_files:
    # Extraire le nom de base (ex: C1_012_PSG1)
    hdf5_basename = os.path.basename(hdf5_path).replace('.hdf5', '')

    csv_pattern = f"{hdf5_basename}.csv"
    csv_path = os.path.join(base_data_path, csv_pattern)
    
    if os.path.exists(csv_path):
        file_pairs.append((hdf5_path, csv_path))
    else:
        print(f"CSV manquant pour: {hdf5_basename}")
        print(f"Cherché: {csv_pattern}")

print(f"\n Total de paires HDF5-CSV: {len(file_pairs)}")

train_pairs = file_pairs[: int(len(file_pairs)*0.8)]  
val_pairs = file_pairs[int(len(file_pairs)*0.8):len(file_pairs)]   
    

# Séparer les chemins
train_hdf5 = [pair[0] for pair in train_pairs]
train_labels = [pair[1] for pair in train_pairs]

val_hdf5 = [pair[0] for pair in val_pairs]
val_labels = [pair[1] for pair in val_pairs]

# CRÉER LES DATASETS

train_dataset = SleepEventClassificationDataset(
    config, 
    CHANNEL_GROUPS_FOR_MODEL, 
    split="train", 
    hdf5_paths=train_hdf5, 
    label_files=train_labels
)

val_dataset = SleepEventClassificationDataset(
    config, 
    CHANNEL_GROUPS_FOR_MODEL, 
    split="val", 
    hdf5_paths=val_hdf5, 
    label_files=val_labels
)

print(f"\n📊 Datasets créés:")
print(f"   Train: {len(train_dataset)} échantillons")
print(f"   Val:   {len(val_dataset)} échantillons")

if len(train_dataset) == 0:
    print("\n❌ ERREUR: Le dataset train est vide!")
    print("   Vérifiez que les noms de fichiers HDF5 et CSV correspondent")
    # Afficher les noms exacts
    print("\n🔍 Vérification des noms:")
    for hdf5 in train_hdf5:
        basename = os.path.basename(hdf5).replace('.hdf5', '')
        print(f"   HDF5 basename: '{basename}'")
    for csv in train_labels:
        basename = os.path.basename(csv).replace('PSG4_Hypnogram_Export_', '').replace('.csv', '')
        print(f"   CSV basename: '{basename}'")

#  DATALOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0,  
    collate_fn=sleep_event_finetune_full_collate_fn,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=0,
    collate_fn=sleep_event_finetune_full_collate_fn,
    pin_memory=True if torch.cuda.is_available() else False
)

# MODÈLE

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, S, C) - prédictions
            targets: (B, S) - labels
            mask: (B, S) - masque (True = ignorer, False = garder)
        """
        B, S, C = logits.shape
        
        # Flatten
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()
        mask_flat = mask.reshape(-1)
        
        # Calculer la loss pour tous les éléments
        loss = self.ce_loss(logits_flat, targets_flat)
        
        # ⚠️ CORRECTION : Convertir le masque booléen en float
        # Si mask = True → ignorer (multiplier par 0)
        # Si mask = False → garder (multiplier par 1)
        if mask_flat.dtype == torch.bool:
            # Inverser : True → 0.0, False → 1.0
            mask_flat = (~mask_flat).float()
        else:
            # Si déjà float : 1 = ignorer, 0 = garder
            mask_flat = 1.0 - mask_flat
        
        # Appliquer le masque
        loss = loss * mask_flat
        
        # Moyenne sur les éléments valides
        num_valid = mask_flat.sum()
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum()
        
        return loss

criterion = MaskedCrossEntropyLoss(class_weights=None).to(device)

optimizer = optim.AdamW(
    sleep_staging_model.parameters(),
    lr=config['lr'],
    weight_decay=config.get('weight_decay', 1e-5)
)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    accumulation_steps = config.get('accumulation_steps', 1)
    gradient_clip = config.get('gradient_clip', 1.0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    optimizer.zero_grad()

    for batch_idx, (x_batch, y_batch, mask_batch, paths) in enumerate(pbar):

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        outputs = model(x_batch, mask_batch)
        if isinstance(outputs, tuple):
            if len(outputs) == 3:
                logits, _, mask_temporal = outputs
            else:
                logits, mask_temporal = outputs
        else:
            logits = outputs
            mask_temporal = mask_batch[:, 0, :]
        loss = criterion(logits, y_batch, mask_temporal)
        loss = loss / accumulation_steps
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            mask_flat = mask_temporal.cpu().numpy().flatten()
            preds_flat = preds.cpu().numpy().flatten()
            labels_flat = y_batch.cpu().numpy().flatten()
            
            valid_idx = mask_flat == 0
            all_preds.extend(preds_flat[valid_idx])
            all_labels.extend(labels_flat[valid_idx])
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1_macro, f1_weighted


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for x_batch, y_batch, mask_batch, paths in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            
            outputs = model(x_batch, mask_batch)
            
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    logits, _, mask_temporal = outputs
                else:
                    logits, mask_temporal = outputs
            else:
                logits = outputs
                mask_temporal = mask_batch[:, 0, :]
            
            loss = criterion(logits, y_batch, mask_temporal)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask_flat = mask_temporal.cpu().numpy().flatten()
            preds_flat = preds.cpu().numpy().flatten()
            labels_flat = y_batch.cpu().numpy().flatten()
            
            valid_idx = mask_flat == 0
            all_preds.extend(preds_flat[valid_idx])
            all_labels.extend(labels_flat[valid_idx])
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels



best_val_loss = float('inf')
best_val_f1 = 0.0
patience_counter = 0
patience = config.get('patience', 10)

history = {
    'train_loss': [], 'train_acc': [], 'train_f1_macro': [], 'train_f1_weighted': [],
    'val_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': [],
    'learning_rates': []
}

num_epochs = config['epochs']

for epoch in range(num_epochs):
    print(f" Epoch {epoch+1}/{num_epochs}")
    
    # Train
    train_loss, train_acc, train_f1_macro, train_f1_weighted = train_one_epoch(
        sleep_staging_model, train_loader, criterion, optimizer, device, epoch
    )
    print("entrainement2")
    # Validate
    val_loss, val_acc, val_f1_macro, val_f1_weighted, val_preds, val_labels = validate(
        sleep_staging_model, val_loader, criterion, device, epoch
    )
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # History
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1_macro'].append(train_f1_macro)
    history['train_f1_weighted'].append(train_f1_weighted)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_f1_weighted'].append(val_f1_weighted)
    history['learning_rates'].append(current_lr)
    
    # Print
    print(f"\n Résultats:")
    print(f"   Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1-Macro: {train_f1_macro:.4f}")
    print(f"   Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1-Macro: {val_f1_macro:.4f}")
    print(f"   LR: {current_lr:.2e}")
    
    # Best model
    is_best = val_f1_macro > best_val_f1
    
    if is_best:
        best_val_loss = val_loss
        best_val_f1 = val_f1_macro
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': sleep_staging_model.module.state_dict() if isinstance(sleep_staging_model, nn.DataParallel) else sleep_staging_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1_macro': val_f1_macro,
            'config': config,
            'history': history
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
        print(f"\n Meilleur modèle sauvegardé (F1: {val_f1_macro:.4f})")
        
        print(f"\n Rapport:")
        print(classification_report(
            val_labels, val_preds,
            target_names=['Wake', 'Stade 1', 'Stade 2', 'Stade 3', 'REM'],
            digits=4
        ))
    else:
        patience_counter += 1
        print(f"\n Patience: {patience_counter}/{patience}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n Early stopping")
        break

print("\n ENTRAÎNEMENT TERMINÉ")
print(f" Meilleur F1-Macro: {best_val_f1:.4f}")