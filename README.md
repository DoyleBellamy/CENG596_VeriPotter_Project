# LongEval 2026 – Scientific Retrieval (CENG 596 Project)

CLEF 2026 LongEval Task 1: Scientific Document Retrieval. This project builds and evaluates a multi-stage retrieval pipeline over ~870k scientific papers from the [CORE](https://core.ac.uk) collection, studying how retrieval quality changes across three temporal snapshots.

**Team VeriPotter** — Middle East Technical University, Department of Computer Engineering

---

## Overview

The challenge provides a corpus of scientific papers across 3 snapshots (training snapshot + 2 test snapshots). The goal is to retrieve the most relevant papers for 381 test queries, and measure how performance degrades over time (temporal robustness).

We implemented four retrieval methods:

| Method | Description | nDCG@10 (training) |
|---|---|---|
| TF-IDF | SMART lnc.ltc weighting | 0.034 |
| BM25 | PyTerrier / Terrier 5.11, default k1=1.2 b=0.75 | 0.345 |
| Dense | Qwen3-Embedding-4B + FlexIndex (pyterrier-dr) | 0.380 |
| RRF | Reciprocal Rank Fusion of BM25 + Dense, k=60 | 0.379 |
| BM25-Fulltext | BM25 on full paper text (40% coverage) | 0.365 |

All scores are nDCG@10 on 100 training queries against snapshot-1.

---

## Repository Structure

```
.
├── demo/
│   ├── app.py                  # Flask web app — type a query, get BM25 results
│   └── templates/index.html    # Search UI
│
├── scripts/                    # Anonymized SLURM job scripts (edit PROJECT= before use)
│   ├── setup_env.sh            # Create conda environment
│   ├── slurm_bm25.sh           # Run BM25 baseline
│   ├── slurm_tfidf.sh          # Run TF-IDF baseline
│   ├── slurm_dense.sh          # Run Dense baseline (GPU required)
│   ├── slurm_rrf.sh            # Run RRF fusion
│   ├── slurm_eval_dense.sh     # Evaluate dense retrieval
│   └── slurm_bm25_fulltext_all.sh  # BM25 on fulltext corpus, all 3 snapshots
│
├── eval_rrf.py                 # Evaluate BM25 + Dense + RRF on training queries
├── eval_dense.py               # Evaluate dense retrieval on training queries
├── eval_fulltext.py            # Compare abstract vs fulltext BM25
├── bm25_fulltext.py            # BM25 indexing + retrieval on fulltext corpus (snapshot-1)
├── bm25_fulltext_all.py        # BM25 fulltext across all 3 snapshots (for submission)
│
├── longeval-code/              # Git submodule — official baseline code from organizers
│   └── clef26/scientific-retrieval/
│       ├── baseline-pyterrier/         BM25 baseline
│       ├── baseline-pyterrier-tfidf/   TF-IDF (SMART lnc.ltc) baseline
│       ├── baseline-pyterrier-dense/   Dense baseline (Qwen3-Embedding-4B)
│       └── baseline-pyterrier-rrf/     RRF fusion (k=60)
│
└── ir-datasets-longeval/       # Git submodule — dataset loader for LongEval
```

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/longeval-2026.git
cd longeval-2026
```

Or if you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Create the conda environment

```bash
bash scripts/setup_env.sh
conda activate longeval
```

This installs: PyTerrier 0.13, pyterrier-dr, sentence-transformers, Flask, tira, and the ir-datasets-longeval package.

### 3. Set your paths

All scripts use two environment variables. Set them before running:

```bash
export PROJECT=/path/to/this/repo
export IR_DATASETS_HOME=/path/to/ir_datasets   # where LongEval dataset files live
```

---

## Running the Baselines

All baselines require the LongEval dataset files. Download them from the [TIRA platform](https://www.tira.io/task-overview/longeval-2026).

### BM25 (CPU, ~30 min)
```bash
sbatch scripts/slurm_bm25.sh
# Output: output/bm25/snapshot-{1,2,3}/run.txt.gz
```

### TF-IDF (CPU, ~2 hours)
```bash
sbatch scripts/slurm_tfidf.sh
# Output: output/tfidf/snapshot-{1,2,3}/run.txt.gz
```

### Dense — Qwen3-Embedding-4B (GPU required, ~3.5h per snapshot)
```bash
sbatch scripts/slurm_dense.sh
# Output: output/dense/snapshot-{1,2,3}/run.txt.gz
```

### RRF Fusion (CPU, ~5 min — requires BM25 + Dense runs first)
```bash
sbatch scripts/slurm_rrf.sh
# Output: output/rrf/snapshot-{1,2,3}/run.txt.gz
```

### BM25 Fulltext (CPU, ~1.5 hours)
```bash
sbatch scripts/slurm_bm25_fulltext_all.sh
# Output: output/bm25_fulltext/snapshot-{1,2,3}/run.txt.gz
```

---

## Evaluation

Evaluation uses the 100 training queries and qrels provided by the organizers. Requires BM25 + Dense indexes and runs to exist first.

```bash
# Evaluate BM25 + Dense + RRF together
conda activate longeval
python eval_rrf.py

# Evaluate dense only (GPU required)
sbatch scripts/slurm_eval_dense.sh

# Compare abstract vs fulltext BM25
python eval_fulltext.py
```

---

## Demo Web App

A Flask app that lets you type a query and get the top-N BM25 results (title, authors, abstract snippet, score). Requires the BM25 index and abstract corpus JSONL files to be present locally.

```bash
conda activate longeval
python demo/app.py
# Running on http://0.0.0.0:5000
```

If running on a remote server, access via SSH tunnel:
```bash
ssh -L 5000:SERVER_HOSTNAME:5000 YOUR_USERNAME@CLUSTER_ADDRESS
# then open http://localhost:5000 in your browser
```

---

## How It Works

### Corpus
- ~870,000 scientific papers from the CORE collection
- Each document: title + abstract (fulltext available for ~40% of papers)
- 3 temporal snapshots: training (March 2026), test-1 (June–August 2026), test-2 (September–November 2026)

### BM25 Pipeline
1. Documents indexed with PyTerrier's `IterDictIndexer` (text = title + abstract)
2. Term pipeline: Stopword removal → Porter stemming
3. At query time: same tokenization applied to query text
4. BM25 scoring with Terrier defaults (k1=1.2, b=0.75)
5. Top-1000 documents per query written to TREC-format run file

### Dense Pipeline
1. Documents encoded with `Qwen/Qwen3-Embedding-4B` (max 512 tokens, asymmetric encoding)
2. Embeddings stored in FlexIndex (pyterrier-dr)
3. Query encoded with `prompt_name="query"`, nearest neighbours retrieved via dot product

### RRF Fusion
Combines BM25 and Dense rankings:
```
RRF_score(doc) = 1/(60 + rank_bm25) + 1/(60 + rank_dense)
```

---

## Requirements

- Python 3.10
- Java 21 (for PyTerrier / Terrier 5.11)
- CUDA GPU (for Dense and RRF baselines)
- ~60GB disk space for indexes
- ~2GB RAM for BM25 index + doc store

See `scripts/setup_env.sh` for the full dependency list.
