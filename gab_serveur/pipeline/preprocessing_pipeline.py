import os
import sys
import click
from loguru import logger

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from preprocessing.preprocessing import edf_to_hdf5, txt_to_csv

@click.command("run_preprocessing")
@click.option("--input_folder", type=str, 
              default=os.path.join(ROOT_DIR, 'data'),
              help="Folder containing EDF files")
@click.option("--output_folder", type=str, 
              default=os.path.join(ROOT_DIR, 'data', 'preprocessed'),
              help="Folder to save HDF5 files")

def run_preprocessing(input_folder, output_folder):
    """
    Convert all EDF files in input_folder to HDF5 format
    """
    logger.info(f"📂 Input folder: {input_folder}")
    logger.info(f"💾 Output folder: {output_folder}")
    
    # Vérifier que le dossier input existe
    if not os.path.exists(input_folder):
        logger.error(f"❌ Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Créer le dossier output si nécessaire
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"✅ Output folder ready")
    
    # Lancer la conversion
    logger.info(f"🚀 Starting conversion...")
    CHANNEL_GROUPS = {
    "EEG": ["Fp1", "C3", "O1", "C4"],       
    "EOG": ["EOG G", "EOG D"],
    "EMG": ["EMG 1", "EMG 2"],          
    "EKG": ["ECG"],
    "REF": ["A2"],   
    "RESP": ["Thermistance", "Flow",]                   
}
    CHANNEL_OI = sorted({ch for grp in CHANNEL_GROUPS.values() for ch in grp})
    try:
        edf_to_hdf5(input_folder, output_folder, CHANNEL_OI)
        txt_to_csv(input_folder, output_folder)
        logger.info(f"✅ Conversion completed successfully!")
        logger.info(f"📁 HDF5 files saved in: {output_folder}")
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        sys.exit(1)
    

if __name__ == '__main__':
    run_preprocessing()