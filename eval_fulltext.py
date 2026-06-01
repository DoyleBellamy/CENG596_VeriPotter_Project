#!/usr/bin/env python3
"""Compare BM25 abstract vs BM25 fulltext on 100 training queries."""
import os, gzip
from pathlib import Path
import pytrec_eval

PROJECT       = Path(__file__).parent
IR_DATA       = Path(os.environ.get('IR_DATASETS_HOME', PROJECT / 'ir_datasets')) / 'longeval-sci-2026'
QRELS_RAW     = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-raw.txt'
QRELS_DCTR    = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-dctr.txt'
ABSTRACT_RUN  = PROJECT / 'output/bm25_abstract_train/snapshot-1/run.txt.gz'
FULLTEXT_RUN  = PROJECT / 'output/bm25_fulltext/snapshot-1/run.txt.gz'

# Known abstract BM25 results from eval_rrf.py run (logged previously)
ABSTRACT_KNOWN = {
    'raw':  {'ndcg_cut_10': 0.3453, 'ndcg_cut_20': 0.3296, 'map': 0.1889, 'recip_rank': 0.5565},
    'dctr': {'ndcg_cut_10': 0.3182, 'ndcg_cut_20': 0.3062, 'map': 0.1742, 'recip_rank': 0.5155},
}

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def load_run(path):
    run = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            parts = line.strip().split()
            qid, docid, score = parts[0], parts[2], float(parts[4])
            run.setdefault(qid, {})[docid] = score
    return run

def evaluate(run, qrels):
    measures = ['ndcg_cut_10', 'ndcg_cut_20', 'map', 'recip_rank']
    ev = pytrec_eval.RelevanceEvaluator(qrels, measures)
    results = ev.evaluate(run)
    agg = {m: 0.0 for m in measures}
    for r in results.values():
        for m in measures:
            agg[m] += r.get(m, 0.0)
    return {m: agg[m]/len(results) for m in measures}

qrels_raw  = load_qrels(QRELS_RAW)
qrels_dctr = load_qrels(QRELS_DCTR)

ft_run = load_run(FULLTEXT_RUN)
print(f'Fulltext run: {len(ft_run)} queries, overlap with qrels: {len(set(ft_run) & set(qrels_raw))}')

ft_raw  = evaluate(ft_run, qrels_raw)
ft_dctr = evaluate(ft_run, qrels_dctr)

ab = ABSTRACT_KNOWN

print()
print(f'{"Model":<25} {"nDCG@10":>9} {"nDCG@20":>9} {"MAP":>8} {"MRR":>8}  (Raw qrels)')
print('-' * 65)
print(f'{"BM25-Abstract":<25} {ab["raw"]["ndcg_cut_10"]:>9.4f} {ab["raw"]["ndcg_cut_20"]:>9.4f} {ab["raw"]["map"]:>8.4f} {ab["raw"]["recip_rank"]:>8.4f}')
print(f'{"BM25-Fulltext":<25} {ft_raw["ndcg_cut_10"]:>9.4f} {ft_raw["ndcg_cut_20"]:>9.4f} {ft_raw["map"]:>8.4f} {ft_raw["recip_rank"]:>8.4f}')
delta_raw = ft_raw["ndcg_cut_10"] - ab["raw"]["ndcg_cut_10"]
print(f'{"Delta":>25} {delta_raw:>+9.4f}')

print()
print(f'{"Model":<25} {"nDCG@10":>9} {"nDCG@20":>9} {"MAP":>8} {"MRR":>8}  (DCTR qrels)')
print('-' * 65)
print(f'{"BM25-Abstract":<25} {ab["dctr"]["ndcg_cut_10"]:>9.4f} {ab["dctr"]["ndcg_cut_20"]:>9.4f} {ab["dctr"]["map"]:>8.4f} {ab["dctr"]["recip_rank"]:>8.4f}')
print(f'{"BM25-Fulltext":<25} {ft_dctr["ndcg_cut_10"]:>9.4f} {ft_dctr["ndcg_cut_20"]:>9.4f} {ft_dctr["map"]:>8.4f} {ft_dctr["recip_rank"]:>8.4f}')
delta_dctr = ft_dctr["ndcg_cut_10"] - ab["dctr"]["ndcg_cut_10"]
print(f'{"Delta":>25} {delta_dctr:>+9.4f}')
