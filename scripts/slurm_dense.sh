#!/bin/bash
#SBATCH --job-name=longeval_dense
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=${PROJECT}/logs/dense_%j.out
#SBATCH --error=${PROJECT}/logs/dense_%j.err

# Usage: Edit PROJECT to point to your repo root, then: sbatch scripts/slurm_dense.sh
# Requires a GPU node (Qwen3-Embedding-4B encoding).
PROJECT=/path/to/Ceng596_Project

echo "Node      : $(hostname)"
echo "Start time: $(date)"
nvidia-smi

BASELINE=$PROJECT/longeval-code/clef26/scientific-retrieval/baseline-pyterrier-dense/baseline_st.py
INDEX_DIR=$PROJECT/indexes/dense
OUTPUT_DIR=$PROJECT/output/dense

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

mkdir -p "$INDEX_DIR" "$OUTPUT_DIR" "$PROJECT/logs"

export HF_HOME=$PROJECT/huggingface
export IR_DATASETS_HOME=$PROJECT/ir_datasets
mkdir -p "$IR_DATASETS_HOME/longeval-sci-2026"
ln -sf "$PROJECT/longeval_sci_training_2026_abstract.zip" \
    "$IR_DATASETS_HOME/longeval-sci-2026/longeval_sci_training_2026_abstract.zip" 2>/dev/null || true

echo "=== Running dense baseline (Qwen3-Embedding-4B via sentence-transformers) ==="
python "$BASELINE" \
    --dataset longeval-sci-2026/clef-2026/sci \
    --index "$INDEX_DIR" \
    --output "$OUTPUT_DIR"

echo "End time: $(date)"
