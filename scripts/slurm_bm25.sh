#!/bin/bash
#SBATCH --job-name=longeval_bm25
#SBATCH --partition=palamut-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=${PROJECT}/logs/bm25_%j.out
#SBATCH --error=${PROJECT}/logs/bm25_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_bm25.sh
PROJECT=/path/to/Ceng596_Project

echo "Node      : $(hostname)"
echo "Start time: $(date)"

BASELINE=$PROJECT/longeval-code/clef26/scientific-retrieval/baseline-pyterrier/baseline.py
INDEX_DIR=$PROJECT/indexes/bm25
OUTPUT_DIR=$PROJECT/output/bm25

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

mkdir -p "$INDEX_DIR" "$OUTPUT_DIR" "$PROJECT/logs"

export IR_DATASETS_HOME=$PROJECT/ir_datasets
mkdir -p "$IR_DATASETS_HOME/longeval-sci-2026"
ln -sf "$PROJECT/longeval_sci_training_2026_abstract.zip" \
    "$IR_DATASETS_HOME/longeval-sci-2026/longeval_sci_training_2026_abstract.zip" 2>/dev/null || true

echo "=== Running BM25 baseline ==="
python "$BASELINE" \
    --dataset longeval-sci-2026/clef-2026/sci \
    --index "$INDEX_DIR" \
    --output "$OUTPUT_DIR"

echo "End time: $(date)"
