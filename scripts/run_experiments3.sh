#!/bin/bash

#SBATCH --job-name=nano_extra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=124:00:00
#SBATCH --output=output/logs/nano_extra_experiments.log
#SBATCH --constraint=el9
hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
module load cuda/12.8.1

conda activate LLM

EXPERIMENTS=(
    "muon_baseline"
    "depth_wider"
    "depth_deeper"
    "gqa_2"
    "gqa_4"
    "wsd_schedule"
    "no_qk_norm"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "============================================"
    echo "Starting experiment: ${exp}"
    echo "Time: $(date)"
    echo "============================================"
    curl -s "https://api.day.app/R6cKMT5xvoxAxjKkzWruZP/${exp}-started" > /dev/null

    python pretrain.py \
        --config-path=config/experiments3 \
        --config-name="${exp}"

    echo "============================================"
    echo "Finished experiment: ${exp}"
    echo "Time: $(date)"
    echo "============================================"
    curl -s "https://api.day.app/R6cKMT5xvoxAxjKkzWruZP/${exp}-completed" > /dev/null
    echo ""
done

echo "All experiments completed at $(date)"
