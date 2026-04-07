import os
import glob
import pyedflib
import h5py
import numpy as np
import pandas as pd
import datetime
from scipy.signal import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from loguru import logger
import argparse
import warnings
from scipy.signal import butter, filtfilt
import mne
import time 

class EDFToHDF5Converter:
    def __init__(self, root_dir, target_dir, resample_rate=512, num_threads=1, num_files=-1, channels=None):
        self.resample_rate = resample_rate 
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_threads = num_threads
        self.num_files = num_files
        self.channels = channels
        self.file_locations = self.get_files() 
        # self.scorers = ['ES','LS','MS']
        self.flow_events = {'Central Apnea': 1, 'Mixed Apnea': 2, 'Obstructive Apnea': 3, 'Hypopnea': 4, 'RERA': 5}
        self.plm_events = {'P-Marker': 1, 'I-Marker': 2, 'LM Body position': 3, 'LM Resp': 4}
        self.arousal_events = {'Autonomic arousals': 1, 'Respiratory Arousal': 2}
        self.sleep_stages = {'Wake': 0, 'Rem': 1, 'N1': 2, 'N2': 3, 'N3': 4, 'Artifact': 5}


    def get_files(self):
        # Search for all '.edf' files within each subdirectory of the root directory
        file_paths = glob.glob(os.path.join(self.root_dir, '**/*.[eE][dD][fF]'), recursive=True)
        file_names = [os.path.basename(path) for path in file_paths]
        return file_paths, file_names
    
    def extract_start_time(self, file_path):
        with open(file_path, 'r') as file:
            lines = [next(file) for _ in range(5)]
            
        for line in lines:
            if line.startswith("Start Time:"):
                return line.split(": ", 1)[1].strip()
        return "Start Time not found"

    def create_signal_from_events(self, df, total_seconds, event_type = None):
        valid_types = {'flow', 'plm', 'arousal','stages'}
        if event_type not in valid_types:
            raise ValueError("event_type must be 'flow', 'plm', or 'arousal'")
        
        total_samples = int(total_seconds * self.resample_rate)
        # make initial array of zeros with length total_sec
        event_array = np.zeros(total_samples)

        # go through each event and mark the corresponding seconds in the array
        for _, row in df.iterrows():
            event_start = int(row['sec_from_start'] * self.resample_rate)
            event_stop = int(row['sec_from_start'] * self.resample_rate + row['dur'] * self.resample_rate)
            if event_type == 'flow':
                event_code = self.flow_events.get(row['event_type'], 0)
            elif event_type == 'plm':
                event_code = self.plm_events.get(row['event_type'], 0)
            elif event_type == 'arousal':
                event_code = self.arousal_events.get(row['event_type'], 0)
            elif event_type == 'stages':
                event_code = self.sleep_stages.get(row['event_type'])
            
            event_array[event_start:event_stop] = event_code
        
        return event_array

    def make_event_dataframe(self, folder, event_type = None):
        valid_types = {'flow', 'plm', 'arousal','stages'}
        if event_type not in valid_types:
            raise ValueError("event_type must be 'flow', 'plm', or 'arousal'")
        if event_type == 'flow':
            flow_file = os.path.join(self.root_dir,folder,'Flow Events.txt')
            df = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
        elif event_type == 'plm':
            flow_file = os.path.join(self.root_dir,folder,'PLM Events.txt')
            df = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
        elif event_type == 'arousal':
            flow_file = os.path.join(self.root_dir,folder,'Autonomic arousals.txt')
            dfAutonomic = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
            flow_file = os.path.join(self.root_dir,folder,'Classification arousals.txt')
            dfClassification = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
            df = pd.concat([dfAutonomic,dfClassification], ignore_index=True)
        elif event_type == 'stages':
            flow_file1 = os.path.join(self.root_dir,folder,'Flow Events.txt')
            start_time = self.extract_start_time(file_path = flow_file1)
            start_time = datetime.datetime.strptime(start_time, "%m/%d/%Y %I:%M:%S %p")
            flow_file = os.path.join(self.root_dir,folder,'Sleep profile.txt')
            df = pd.read_csv(flow_file, delimiter=';',skiprows=7, names=['start', 'event_type']) 
            df['start'] = pd.to_datetime(df['start'], format='%H:%M:%S,%f').dt.time
            df['sec_from_start'] = df['start'].apply(lambda x: (datetime.datetime.combine(datetime.date(1,1,1),x) - datetime.datetime.combine(datetime.date(1,1,1),start_time.time())).total_seconds())
            df['dur'] = 30
            df['event_type'] = df['event_type'].str.strip()
            if df['sec_from_start'].iloc[0] < 0:
                df['dur'][0] = 30 + df['sec_from_start'].iloc[0]
                df['sec_from_start'].iloc[0] = 0
            df.loc[df.sec_from_start < 0, 'sec_from_start'] += 24*60*60
            return df

        # df = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
        if len(df.values) != 0:
            df[['start', 'stop']] = df['start-stop'].str.split('-', expand=True)

            start_time = self.extract_start_time(file_path = flow_file)
            start_time = datetime.datetime.strptime(start_time, "%m/%d/%Y %I:%M:%S %p")

            df['start'] = pd.to_datetime(df['start'], format='%H:%M:%S,%f').dt.time
            df['stop'] = pd.to_datetime(df['stop'], format='%H:%M:%S,%f').dt.time
            df['sec_from_start'] = df['start'].apply(lambda x: (datetime.datetime.combine(datetime.date(1,1,1),x) - datetime.datetime.combine(datetime.date(1,1,1),start_time.time())).total_seconds())
            df['sec_from_stop'] = df['stop'].apply(lambda x: (datetime.datetime.combine(datetime.date(1,1,1),x) - datetime.datetime.combine(datetime.date(1,1,1),start_time.time())).total_seconds())

            df['duration'] = pd.to_numeric(df['duration'])

            df = df[['start', 'stop', 'duration', 'event_type', 'sec_from_start','sec_from_stop']]

            df.loc[df.sec_from_start < 0, 'sec_from_start'] += 24*60*60
            df.loc[df.sec_from_stop < 0, 'sec_from_stop'] += 24*60*60
            df['dur'] = df['sec_from_stop'] - df['sec_from_start']
        else:
            df  = pd.DataFrame(columns = ['start', 'stop', 'duration', 'event_type', 'sec_from_start','sec_from_stop', 'dur'])

        return df

    def convert_events(self, folder,total_seconds,event_type):
        df_events = self.make_event_dataframe(folder, event_type=event_type)
        event_array = self.create_signal_from_events(df=df_events, total_seconds=total_seconds, event_type=event_type)
        return event_array

    def read_edf_old(self, file_path):
        logger.info('reading edf')
        with pyedflib.EdfReader(file_path) as edf:
            signals = [edf.readSignal(i) for i in range(edf.signals_in_file)]
            sample_rates = np.array([edf.getSampleFrequency(i) for i in range(edf.signals_in_file)])
            channel_names = np.array([edf.getLabel(i) for i in range(edf.signals_in_file)])
            # annotations = []#edf.readAnnotations()
        return signals, sample_rates, channel_names
    
    def read_edf(self, file_path):
        logger.info(f'Reading EDF: {os.path.basename(file_path)}')
        
        try:
            # Charger avec filtrage de canaux si spécifié
            if self.channels is not None and len(self.channels) > 0:
                # Vérifier d'abord quels canaux existent
                logger.info(f'Checking available channels...')
                raw_temp = mne.io.read_raw_edf(file_path,include=self.channels, preload=False, verbose=False)
                
                available_channels = [ch for ch in self.channels if ch in raw_temp.ch_names]
                missing_channels = set(self.channels) - set(available_channels)
                
                if missing_channels:
                    logger.warning(f"⚠️  Canaux manquants: {missing_channels}")
                
                if not available_channels:
                    raise ValueError(f"❌ Aucun des canaux spécifiés n'a été trouvé dans {file_path}\n"
                                f"   Canaux demandés: {self.channels}\n"
                                f"   Canaux disponibles: {raw_temp.ch_names}")
                
                logger.info(f'✅ Chargement de {len(available_channels)}/{len(self.channels)} canaux')
                raw = mne.io.read_raw_edf(file_path, include=available_channels, preload=True, verbose=False)
            else:
                # Charger tous les canaux si aucun filtre spécifié
                logger.info('Loading all channels (no filter specified)')
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Extraire les données
            signals = [raw.get_data(picks=[ch_name])[0] for ch_name in raw.ch_names]
            sample_rates = np.array([raw.info['sfreq'] for _ in raw.ch_names])
            channel_names = np.array([CHANNEL_RENAME_MAP.get(ch, ch) for ch in raw.ch_names])
            print(channel_names)
            logger.info(f'✅ Loaded {len(channel_names)} channels, {len(signals[0])} samples at {sample_rates[0]} Hz')
            
            return signals, sample_rates, channel_names
            
        except Exception as e:
            logger.error(f"❌ Error reading {file_path}: {str(e)}")
            raise

    def resample_signals_old(self, signals, sample_rates):
        # Vectorization and broadcasting could be applied within the resample function itself
        logger.info('resampling signals')
        resampled_signals = [resample(signal, int(len(signal) * self.resample_rate / rate))
                             for signal, rate in zip(signals, sample_rates)]
        standardized_signals = [(signal - np.mean(signal)) / np.std(signal) for signal in resampled_signals]
        # add signal names as input and filter the spo2 preprocessing as scaled 0 to 1
        return np.stack(standardized_signals)# Using np.stack for proper array dimensions


    def safe_standardize(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std == 0:
            standardized_signal = (signal - mean)
        else:
            standardized_signal = (signal - mean) / std
        
        return standardized_signal
        
    def filter_signal(self, signal, sample_rate):
        print("Filtering signal")
        nyquist_freq = sample_rate / 2
        cutoff = min(self.resample_rate / 2, nyquist_freq)
        normalized_cutoff = cutoff / nyquist_freq
        b, a = butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def resample_signals(self, signals, sample_rates):
        logger.info('resampling signals')
        resampled_signals = []
        for signal, rate in zip(signals, sample_rates):
            # Calculate the duration of the signal
            duration = len(signal) / rate
            
            # Original time points
            original_time_points = np.linspace(0, duration, num=len(signal), endpoint=False)
            
            # New sample rate and new time points
            new_sample_count = int(duration * self.resample_rate)
            new_time_points = np.linspace(0, duration, num=new_sample_count, endpoint=False)

            #filter signal
            if rate > self.resample_rate:
                signal = self.filter_signal(signal, rate)
            
            # Linear interpolation
            resampled_signal = np.interp(new_time_points, original_time_points, signal)
            
            # Standardize the resampled signal (optional, based on your need)
            # standardized_signal = (resampled_signal - np.mean(resampled_signal)) / np.std(resampled_signal)
            standardized_signal = self.safe_standardize(resampled_signal)
            
            if np.isnan(standardized_signal).any():
                logger.info('Found NaN in the resampled signal.')
                # Handle the NaN case here (e.g., skip or fix the signal)
                continue

            resampled_signals.append(standardized_signal)

        
        return np.stack(resampled_signals)  # Stack for a consistent output format

    def save_to_hdf5(self, signals, channel_names, annotation_signals, annotation_names, file_path):
        logger.info('saving hdf5')
        samples_per_chunk = 5 * 60 * self.resample_rate
        with h5py.File(file_path, 'w') as hdf:
            for signal, name in zip(signals, channel_names):
                dataset_name = self._get_unique_name(hdf, name)
                hdf.create_dataset(dataset_name, data=signal,
                                   dtype='float16', chunks=(samples_per_chunk,), compression="gzip")

            for annot_signal, annot_name in zip(annotation_signals, annotation_names):
                hdf.create_dataset(annot_name, data=annot_signal)

    def _get_unique_name(self, hdf, base_name):
        # Helper method to ensure dataset names are unique
        i = 1
        unique_name = base_name
        while unique_name in hdf:
            unique_name = f"{base_name}_{i}"
            i += 1
        return unique_name

    def get_annotations(self, total_seconds, folder):
        # folder_path = 'C:/Users/45223/OneDrive - Danmarks Tekniske Universitet/PhD/Oliver_triple_scored/triple_scored_studies1/CSA009/ES'
        flow_events = self.convert_events(folder = folder, total_seconds = total_seconds, event_type = 'flow')
        plm_events = self.convert_events( folder = folder, total_seconds = total_seconds, event_type = 'plm')
        arousal_events = self.convert_events( folder = folder, total_seconds = total_seconds, event_type = 'arousal')
        sleep_stages = self.convert_events( folder = folder, total_seconds = total_seconds, event_type = 'stages')

        return flow_events, plm_events, arousal_events, sleep_stages
    
    def convert(self, edf_path, hdf5_path):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        total_duration_seconds = len(signals[0])/sample_rates[0]
        event_signals = []
        event_signal_names = []
        self.save_to_hdf5(resampled_signals, channel_names,event_signals,event_signal_names, hdf5_path)

    def convert_multiprocessing(self, args):
        edf_files = args

        for edf_file in tqdm(edf_files, desc="Converting EDF files"):

            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir, edf_file.split('/')[-1].replace(replace_str, '.hdf5'))

            if os.path.exists(hdf5_file):
                logger.info(f"File already processed: {hdf5_file}")
                continue
            try:
                self.convert(edf_file, hdf5_file)
            except Exception as e:
                warnings.warn(f"Warning: Could not process the file {edf_file}. Error: {str(e)}")
                continue
        return [1]

    def convert_all(self):
        edf_files, edf_names = self.get_files() 
        # folders = self.get_folders()
        for edf_file in tqdm(edf_files, desc="Converting EDF files"):
            # edf_files = [os.path.join(folder, f) for f in os.listdir(os.path.join(self.root_dir,folder)) if f.endswith('.edf')]
            # edf_file = os.path.join(self.root_dir,edf_files[0])
            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir,edf_file.split('/')[-1].replace(replace_str, '.hdf5'))
            # logger.info(edf_file)
            # logger.info(hdf5_file)

            try:
                self.convert(edf_file, hdf5_file)
            except Exception as e:
                warnings.warn(f"Warning: Could not process the file {edf_file}. Error: {str(e)}")
                continue

    def convert_all_multiprocessing(self):
        edf_files, edf_names = self.get_files() 

        if self.num_files != -1:
            edf_files = edf_files[:self.num_files]

        edf_files_chunks = np.array_split(edf_files, self.num_threads)
        tasks = [(edf_files_chunk) for edf_files_chunk in edf_files_chunks]
        with multiprocessing.Pool(self.num_threads) as pool:
            preprocessed_results = [y for x in pool.imap_unordered(self.convert_multiprocessing, tasks) for y in x]

    def convert_with_annot(self, edf_path, hdf5_path, folder):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        total_duration_seconds = len(signals[0])/sample_rates[0]
        event_signals = []
        event_signal_names = []
        for scorer in self.scorers:
            scorer_folder = os.path.join(folder, scorer) 
            flow_events, plm_events, arousal_events, sleep_stages = self.get_annotations(total_seconds = total_duration_seconds, folder = scorer_folder)
            event_signals.extend([flow_events, plm_events, arousal_events, sleep_stages])
            event_signal_names.extend(['flow_events'+scorer, 'plm_events'+scorer, 'arousal_events'+scorer, 'sleep_stages'+scorer])
        self.save_to_hdf5(resampled_signals, channel_names,event_signals,event_signal_names, hdf5_path)

    def convert_all_with_annot(self):
        edf_files, edf_names = self.get_files() 
        folders = self.get_folders()
        for folder in tqdm(folders, desc="Converting EDF files"):
            edf_files = [os.path.join(folder, f) for f in os.listdir(os.path.join(self.root_dir,folder)) if f.lower().endswith('.edf')]
            edf_file = os.path.join(self.root_dir,edf_files[0])
            # header = self.extract_end_time(edf_file)
            # hdf5_file = os.path.join(self.root_dir, edf_file.replace('.edf', '.hdf5'))
            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir,edf_file.split('\\')[-1].replace(replace_str, '.hdf5'))

            # logger.info(hdf5_file)
            self.convert_with_annot(edf_file, hdf5_file, folder)
    def plot_results(self, resampled_signals, channel_names):
        print("plotting resampled_signals")
        num_signals = len(resampled_signals)
        fig, axs = plt.subplots(num_signals, 1, figsize=(15, 3*num_signals), sharex=True)
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        for i, (signal, name) in enumerate(zip(resampled_signals, channel_names)):
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            axs[i].plot(signal_chunk)
            axs[i].set_title(name)
            axs[i].set_ylabel('Amplitude')
        
        axs[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.show()

    def plot_first_results(self, resampled_signals, channel_names):
        print("plotting resampled_signals")
        num_signals = len(resampled_signals)
        fig = plt.figure(figsize=(15, 3))
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        for i, (signal, name) in enumerate(zip(resampled_signals, channel_names)):
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            plt.plot(signal_chunk)
            plt.title(name)
            plt.ylabel('Amplitude')
            break
        
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.show()

    def process_and_plot_single_file(self, edf_path):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        #self.plot_results(resampled_signals, channel_names)
        self.plot_first_results(resampled_signals, channel_names)
       
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

CHANNEL_RENAME_MAP = {
    # EOG
    "EOG G": "E1",
    "EOG D": "E2",
    "EOG(L)": "E1",
    "EOG(R)": "E2",
    "LOC": "E1",
    "ROC": "E2",

    # EMG
    "EMG 1": "EMG1",
    "EMG 2": "EMG2",
    "Chin 1": "EMG1",
    "Chin 2": "EMG2",

    # RESP
    "Thermistance": "Therm",
    "Thermistor": "Therm",
    "Flow": "Airflow",
    "AIRFLOW": "Airflow",

    # ECG
    "ECG": "ECG",
}

for category, channels in CHANNEL_GROUPS.items():
    model_category = CATEGORY_MAPPING.get(category, 'BAS')
    CHANNEL_GROUPS_FOR_MODEL[model_category].extend(channels)
    

def edf_to_hdf5(input_folder, output_folder, CHANNEL_OI=CHANNEL_OI):
    # Trouver tous les EDF
    edf_files = glob.glob(os.path.join(input_folder, "**/*.edf"), recursive=True)
    edf_files += glob.glob(os.path.join(input_folder, "**/*.EDF"), recursive=True)
    
    print(f"\n📁 {len(edf_files)} fichiers EDF trouvés\n")
    
    converter = EDFToHDF5Converter(
        root_dir=input_folder,
        target_dir=output_folder,
        resample_rate=128,
        channels=CHANNEL_OI
    )
    
    # Convertir avec barre de progression
    success = 0
    errors = 0
    skipped = 0
    
    for edf_path in tqdm(edf_files, desc="🔄 Conversion", unit="fichier"):
        basename = os.path.basename(edf_path).replace('.edf', '').replace('.EDF', '')
        hdf5_path = os.path.join(output_folder, f"{basename}.hdf5")
        
        # Skip si existe déjà
        if os.path.exists(hdf5_path):
            skipped += 1
            tqdm.write(f"⏭️  {basename} (déjà converti)")
            continue
        
        try:
            start = time.time()
            converter.convert(edf_path, hdf5_path)
            duration = time.time() - start
            success += 1
            tqdm.write(f"✅ {basename} ({duration:.1f}s)")
        except Exception as e:
            errors += 1
            tqdm.write(f"❌ {basename}: {str(e)[:50]}")
        """os.remove(edf_path)"""
    # Résumé
    print(f"\n{'='*50}")
    print(f"✅ Réussis:  {success}")
    print(f"⏭️  Ignorés:  {skipped}")
    print(f"❌ Erreurs:  {errors}")
    print(f"{'='*50}")

def txt_to_csv(input_folder, output_folder, force_overwrite=False):
    """
    Convertit les fichiers TXT (hypnogrammes) en CSV avec epochs de 5s
    
    Args:
        input_folder: Dossier contenant les fichiers TXT
        output_folder: Dossier où sauvegarder les CSV
        force_overwrite: Si True, reconvertir même si le fichier existe déjà
    """
    os.makedirs(output_folder, exist_ok=True)
    
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
    
    # Chercher les fichiers TXT
    txt_files = glob.glob(os.path.join(input_folder, "**/*.txt"), recursive=True)
    txt_files += glob.glob(os.path.join(input_folder, "**/*.TXT"), recursive=True)
    
    print(f"\n📁 {len(txt_files)} fichiers TXT trouvés")
    print(f"📂 Dossier de sortie: {output_folder}\n")
    
    if len(txt_files) == 0:
        logger.warning("⚠️  Aucun fichier TXT trouvé!")
        return
    
    # Compteurs
    success = 0
    errors = 0
    skipped = 0
    
    for txt_file in tqdm(txt_files, desc="🔄 Conversion TXT→CSV", unit="fichier"):
        try:
            # Nom du fichier original
            original_basename = os.path.basename(txt_file).replace('.txt', '').replace('.TXT', '')
            
            # Enlever le préfixe "PSG4_Hypnogram_Export_"
            if original_basename.startswith("PSG4_Hypnogram_Export_"):
                new_basename = original_basename.replace("PSG4_Hypnogram_Export_", "")
            else:
                new_basename = original_basename
            
            # Chemin du CSV de sortie
            csv_file = os.path.join(output_folder, f"{new_basename}.csv")
            
            # ✅ VÉRIFIER SI LE FICHIER EXISTE DÉJÀ
            if os.path.exists(csv_file) and not force_overwrite:
                skipped += 1
                tqdm.write(f"⏭️  {new_basename}.csv (déjà existe)")
                continue
            
            # Charger les stages (30s par ligne)
            stages_30s = np.loadtxt(txt_file, dtype=str)
            
            # Répéter chaque stage 6 fois (pour 6 epochs de 5s)
            stages_5s = np.repeat(stages_30s, 6)
            n_epochs = len(stages_5s)
            
            # Créer les timestamps
            starts = np.arange(0, n_epochs * 5, 5)
            stops = starts + 5
            
            # Créer le DataFrame
            df = pd.DataFrame({
                'Start': starts,
                'Stop': stops,
                'StageName': [STAGE_NAMES.get(STAGE_TO_NUMBER.get(s, -1), "Unknown") for s in stages_5s],
                'StageNumber': [STAGE_TO_NUMBER.get(s, -1) for s in stages_5s],
                'EmbeddingNumber': np.arange(n_epochs)
            })
            
            # Sauvegarder
            df.to_csv(csv_file, index=False)
            
            # Vérifier que le fichier a bien été créé
            if os.path.exists(csv_file):
                success += 1
                tqdm.write(f"✅ {new_basename}.csv ({len(df)} epochs)")
            else:
                raise Exception("Le fichier CSV n'a pas été créé")
            
        except Exception as e:
            errors += 1
            tqdm.write(f"❌ {os.path.basename(txt_file)}: {str(e)[:80]}")
            
            # Supprimer le fichier partiel s'il existe
            if 'csv_file' in locals() and os.path.exists(csv_file):
                os.remove(csv_file)
    
    print(f"✅ Convertis avec succès:  {success}")
    print(f"⏭️  Fichiers ignorés:       {skipped}")
    print(f"❌ Erreurs:                {errors}")
    print(f"📂 Dossier de sortie:      {output_folder}")


"""
# ========== UTILISATION ==========
input_folder = "C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\sleep_fm\\sleepfm-clinical\\notebooks\\train_data_more"
output_folder = "C:\\Users\\gabri\\Desktop\\stage_sommeil\\algo\\sleep_fm\\sleepfm-clinical\\notebooks\\train_data_more"

txt_to_csv(input_folder, output_folder)
edf_to_hdf5(edf_folder, output_folder)"""



