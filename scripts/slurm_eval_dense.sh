#!/bin/bash
#SBATCH --job-name=eval_dense
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=${PROJECT}/logs/eval_dense_%j.out
#SBATCH --error=${PROJECT}/logs/eval_dense_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_eval_dense.sh
PROJECT=/path/to/Ceng596_Project

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

export HF_HOME=$PROJECT/huggingface
python $PROJECT/eval_dense.py
