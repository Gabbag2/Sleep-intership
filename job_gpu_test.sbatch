#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/test_gpu-%j.out
#SBATCH --error=logs/test_gpu-%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load miniforge/24.11.3-0
eval "$(conda shell.bash hook)"
conda activate sleepfm_env

python -c "import torch; print('GPU disponible:', torch.cuda.is_available()); print('Nombre de GPUs:', torch.cuda.device_count()); print('Nom du GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun')"