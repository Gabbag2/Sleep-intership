import numpy as np
import pandas as pd
import os

STAGE_TO_NUMBER = {
    'W': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 3,
    'R': 4
}

STAGE_NAMES = {
    0: "Wake",
    1: "Stage 1",
    2: "Stage 2",
    3: "Stage 3",
    4: "REM"
}

txt_file = 'C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\gab_serveur\\PSG4_Hypnogram_Export_HI_002_PSG1.txt'
csv_file = 'C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\gab_serveur\\PSG4_Hypnogram_Export_HI_002_PSG1.csv'

stages_30s = np.loadtxt(txt_file, dtype=str)
stages_5s = np.repeat(stages_30s, 6)
n_epochs = len(stages_5s)

starts = np.arange(0, n_epochs * 5, 5)
stops = starts + 5

df = pd.DataFrame({
    'Start': starts,
    'Stop': stops,
    'StageName': [STAGE_NAMES.get(STAGE_TO_NUMBER.get(s, -1), "Unknown") for s in stages_5s],
    'StageNumber': [STAGE_TO_NUMBER.get(s, -1) for s in stages_5s],
    'EmbeddingNumber': np.arange(n_epochs)
})

df.to_csv(csv_file, index=False)
print(f"Done: {len(df)} epochs -> {csv_file}")