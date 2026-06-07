#!/bin/bash
#SBATCH --job-name=longeval_rerank
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=${PROJECT}/logs/rerank_%j.out
#SBATCH --error=${PROJECT}/logs/rerank_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_rerank.sh
PROJECT=/path/to/Ceng596_Project

echo "Node      : $(hostname)"
echo "Start time: $(date)"
nvidia-smi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

export HF_HOME=$PROJECT/huggingface
export IR_DATASETS_HOME=/path/to/ir_datasets

mkdir -p "$PROJECT/logs"

python "$PROJECT/eval_rerank.py"

echo "End time: $(date)"
