#!/bin/bash
#SBATCH --job-name=bm25_ft_all
#SBATCH --partition=palamut-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=${PROJECT}/logs/bm25_ft_all_%j.out
#SBATCH --error=${PROJECT}/logs/bm25_ft_all_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_bm25_fulltext_all.sh
# Runs BM25 on fulltext corpus (snapshot-1) and abstract corpus (snapshots 2 & 3).
PROJECT=/path/to/Ceng596_Project

echo "Node      : $(hostname)"
echo "Start time: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

python $PROJECT/bm25_fulltext_all.py

echo "End time: $(date)"
