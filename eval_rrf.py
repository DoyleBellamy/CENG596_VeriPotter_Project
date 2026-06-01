#!/usr/bin/env python3
"""Evaluate BM25, Dense, and RRF on 100 training queries."""
import os
from pathlib import Path
import pandas as pd, pytrec_eval
import pyterrier as pt
from pyterrier_dr import FlexIndex
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

pt.java.init()

PROJECT    = Path(__file__).parent
IR_DATA    = Path(os.environ.get('IR_DATASETS_HOME', PROJECT / 'ir_datasets')) / 'longeval-sci-2026'
QRELS_RAW  = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-raw.txt'
QRELS_DCTR = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-dctr.txt'
QUERIES    = IR_DATA / 'task1_longeval_adhoc-queries-snapshot-train.tsv'
IDX        = PROJECT / 'indexes/bm25/snapshot-1'
FLEX_PATH  = PROJECT / 'indexes/dense/snapshot-1/my_index.flex'
K = 60

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

# BM25
index = pt.IndexFactory.of(IDX)
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()
topics_tok = topics.copy()
topics_tok['query'] = topics_tok['query'].apply(lambda q: ' '.join(tokeniser.getTokens(q)))
bm25_run = pt.terrier.Retriever(index, wmodel='BM25', num_results=1000)(topics_tok)
bm25_ranked = {}
for qid, grp in bm25_run.groupby('qid'):
    grp = grp.sort_values('score', ascending=False).reset_index(drop=True)
    bm25_ranked[qid] = {row['docno']: int(idx+1) for idx, row in grp.iterrows()}
print('BM25 done', flush=True)

# Dense
model = SentenceTransformer('Qwen/Qwen3-Embedding-4B', trust_remote_code=True)
model.max_seq_length = 512
query_vecs = model.encode(topics['query'].tolist(), batch_size=32,
    show_progress_bar=True, normalize_embeddings=True, prompt_name='query').astype('float32')
topics['query_vec'] = list(query_vecs)
flex = FlexIndex(FLEX_PATH)
dense_run = flex.retriever(num_results=1000)(topics)
dense_ranked = {}
for qid, grp in dense_run.groupby('qid'):
    grp = grp.sort_values('score', ascending=False).reset_index(drop=True)
    dense_ranked[qid] = {row['docno']: int(idx+1) for idx, row in grp.iterrows()}
print('Dense done', flush=True)

# RRF
all_qids = set(bm25_ranked) | set(dense_ranked)
rrf_dict = {}
for qid in all_qids:
    scores = defaultdict(float)
    for ranked in [bm25_ranked, dense_ranked]:
        if qid not in ranked: continue
        for docno, rank in ranked[qid].items():
            scores[docno] += 1.0 / (K + rank)
    rrf_dict[qid] = dict(sorted(scores.items(), key=lambda x: -x[1])[:1000])
print('RRF done', flush=True)

bm25_dict  = {qid: {d: 1.0/(K+r) for d,r in ranked.items()} for qid, ranked in bm25_ranked.items()}
dense_dict = {qid: {d: 1.0/(K+r) for d,r in ranked.items()} for qid, ranked in dense_ranked.items()}

print(f"\n{'Model':<10} {'nDCG@10':>9} {'nDCG@20':>9} {'MAP':>8} {'MRR':>8}  (Raw qrels)")
print("-" * 55)
for name, rd in [('BM25', bm25_dict), ('Dense', dense_dict), ('RRF', rrf_dict)]:
    r = evaluate(rd, qrels_raw)
    print(f"{name:<10} {r['ndcg_cut_10']:>9.4f} {r['ndcg_cut_20']:>9.4f} {r['map']:>8.4f} {r['recip_rank']:>8.4f}")

print(f"\n{'Model':<10} {'nDCG@10':>9} {'nDCG@20':>9} {'MAP':>8} {'MRR':>8}  (DCTR qrels)")
print("-" * 55)
for name, rd in [('BM25', bm25_dict), ('Dense', dense_dict), ('RRF', rrf_dict)]:
    r = evaluate(rd, qrels_dctr)
    print(f"{name:<10} {r['ndcg_cut_10']:>9.4f} {r['ndcg_cut_20']:>9.4f} {r['map']:>8.4f} {r['recip_rank']:>8.4f}")
