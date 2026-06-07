#!/bin/bash
#SBATCH --job-name=longeval_tfidf
#SBATCH --partition=palamut-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --output=${PROJECT}/logs/tfidf_%j.out
#SBATCH --error=${PROJECT}/logs/tfidf_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_tfidf.sh
PROJECT=/path/to/Ceng596_Project

echo "Node      : $(hostname)"
echo "Start time: $(date)"

BASELINE=$PROJECT/longeval-code/clef26/scientific-retrieval/baseline-pyterrier-tfidf/baseline_tfidf.py
INDEX_DIR=$PROJECT/indexes/bm25   # reuses BM25 index (same preprocessing pipeline)
OUTPUT_DIR=$PROJECT/output/tfidf

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

mkdir -p "$OUTPUT_DIR" "$PROJECT/logs"

export HF_HOME=$PROJECT/huggingface
export IR_DATASETS_HOME=$PROJECT/ir_datasets
mkdir -p "$IR_DATASETS_HOME/longeval-sci-2026"
ln -sf "$PROJECT/longeval_sci_training_2026_abstract.zip" \
    "$IR_DATASETS_HOME/longeval-sci-2026/longeval_sci_training_2026_abstract.zip" 2>/dev/null || true

echo "=== Running TF-IDF (SMART lnc.ltc) baseline ==="
python "$BASELINE" \
    --dataset longeval-sci-2026/clef-2026/sci \
    --index "$INDEX_DIR" \
    --output "$OUTPUT_DIR"

echo "End time: $(date)"
