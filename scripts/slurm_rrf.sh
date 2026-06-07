#!/bin/bash
#SBATCH --job-name=longeval_rrf
#SBATCH --partition=palamut-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=${PROJECT}/logs/rrf_%j.out
#SBATCH --error=${PROJECT}/logs/rrf_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_rrf.sh
# Requires BM25 and Dense runs to exist first.
PROJECT=/path/to/Ceng596_Project

echo "Node      : $(hostname)"
echo "Start time: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

mkdir -p "$PROJECT/logs" "$PROJECT/output/rrf"

python $PROJECT/longeval-code/clef26/scientific-retrieval/baseline-pyterrier-rrf/rrf.py \
    --bm25   $PROJECT/output/bm25 \
    --dense  $PROJECT/output/dense \
    --output $PROJECT/output/rrf

echo "End time: $(date)"
