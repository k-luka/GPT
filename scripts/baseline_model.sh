#!/bin/bash
 
#SBATCH --job-name=LM_muon
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=124:00:00
#SBATCH --output=output/logs/LM_muon.log
#SBATCH --constraint=el9
hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
module load cuda/12.8.1

conda activate LLM

python pretrain.py