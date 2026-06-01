#!/usr/bin/env python3
"""Evaluate dense retrieval (Qwen3-Embedding-4B) on 100 training queries."""
import os
from pathlib import Path
import pandas as pd, pytrec_eval
import pyterrier as pt
from pyterrier_dr import FlexIndex
from sentence_transformers import SentenceTransformer
import warnings; warnings.filterwarnings('ignore')

pt.java.init()

PROJECT    = Path(__file__).parent
IR_DATA    = Path(os.environ.get('IR_DATASETS_HOME', PROJECT / 'ir_datasets')) / 'longeval-sci-2026'
QRELS_RAW  = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-raw.txt'
QRELS_DCTR = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-dctr.txt'
QUERIES    = IR_DATA / 'task1_longeval_adhoc-queries-snapshot-train.tsv'
FLEX_PATH  = PROJECT / 'indexes/dense/snapshot-1/my_index.flex'

def load_qrels_dict(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def evaluate(run_dict, qrels_dict):
    measures = ['ndcg_cut_10','ndcg_cut_20','map','recip_rank']
    ev = pytrec_eval.RelevanceEvaluator(qrels_dict, measures)
    results = ev.evaluate(run_dict)
    agg = {m: 0.0 for m in measures}
    for r in results.values():
        for m in measures: agg[m] += r.get(m, 0.0)
    return {m: agg[m]/len(results) for m in measures}

qrels_raw  = load_qrels_dict(QRELS_RAW)
qrels_dctr = load_qrels_dict(QRELS_DCTR)
topics = pd.read_csv(QUERIES, sep='\t', header=None, names=['qid','query'])
print(f'Loaded {len(topics)} training queries', flush=True)

model = SentenceTransformer('Qwen/Qwen3-Embedding-4B', trust_remote_code=True)
model.max_seq_length = 512
print('Model loaded', flush=True)

query_vecs = model.encode(topics['query'].tolist(), batch_size=32,
    show_progress_bar=True, normalize_embeddings=True, prompt_name='query').astype('float32')
topics['query_vec'] = list(query_vecs)
print('Queries encoded', flush=True)

index = FlexIndex(FLEX_PATH)
retriever = index.retriever(num_results=1000)
run = retriever(topics)
print(f'Retrieval done: {len(run)} rows', flush=True)

dense_dict = {}
for _, row in run.iterrows():
    dense_dict.setdefault(str(row['qid']), {})[str(row['docno'])] = float(row['score'])

for qname, qrels in [('Raw', qrels_raw), ('DCTR', qrels_dctr)]:
    d = evaluate(dense_dict, qrels)
    print(f'Dense ({qname}): nDCG@10={d["ndcg_cut_10"]:.4f}  nDCG@20={d["ndcg_cut_20"]:.4f}  MAP={d["map"]:.4f}  MRR={d["recip_rank"]:.4f}')
