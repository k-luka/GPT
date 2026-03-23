#!/bin/bash

#SBATCH --job-name=muon_exp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=124:00:00
#SBATCH --output=output/logs/muon_experiments.log
#SBATCH --constraint=el9
hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
module load cuda/12.8.1

conda activate LLM

EXPERIMENTS=(
    "exp_0_adamw"
    "exp_1_muon_current"
    "exp_2_muon_no_wd"
    "exp_3_muon_lr100"
    "exp_4_muon_lr150"
    "exp_5_split_qkv"
    "exp_6_split_all"
    "exp_7_best"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "============================================"
    echo "Starting experiment: ${exp}"
    echo "Time: $(date)"
    echo "============================================"
    curl -s "https://api.day.app/R6cKMT5xvoxAxjKkzWruZP/${exp}-started" > /dev/null

    python pretrain.py \
        --config-path=config/experiments \
        --config-name="${exp}"

    echo "============================================"
    echo "Finished experiment: ${exp}"
    echo "Time: $(date)"
    echo "============================================"
    curl -s "https://api.day.app/R6cKMT5xvoxAxjKkzWruZP/${exp}-completed" > /dev/null
    echo ""
done

echo "All experiments completed at $(date)"
