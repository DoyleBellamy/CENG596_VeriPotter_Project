#!/usr/bin/env python3
"""Evaluate BM25, Dense, RRF, and CrossEncoder reranking on 100 training queries."""
import os, json
from pathlib import Path
from collections import defaultdict
import pandas as pd, pytrec_eval
import pyterrier as pt
from pyterrier_dr import FlexIndex
from sentence_transformers import SentenceTransformer, CrossEncoder
import warnings; warnings.filterwarnings('ignore')

pt.java.init()

PROJECT    = Path(__file__).parent
IR_DATA    = Path(os.environ.get('IR_DATASETS_HOME', PROJECT / 'ir_datasets')) / 'longeval-sci-2026'
QRELS_RAW  = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-raw.txt'
QRELS_DCTR = IR_DATA / 'task1_longeval_adhoc-qrels-snapshot-train-dctr.txt'
QUERIES    = IR_DATA / 'task1_longeval_adhoc-queries-snapshot-train.tsv'
IDX        = PROJECT / 'indexes/bm25/snapshot-1'
FLEX_PATH  = PROJECT / 'indexes/dense/snapshot-1/my_index.flex'
DOC_DIR    = (PROJECT / 'longeval_sci_training_2026_abstract/data/processed'
              '/doc_collection_09032026_abstract_2/snapshot-1'
              '/longeval_sci_training_2026_abstract/documents')
K          = 60
CE_DEPTH   = 100  # rerank top-100 from RRF


def load_qrels_dict(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def evaluate(run_dict, qrels_dict):
    measures = ['ndcg_cut_10', 'ndcg_cut_20', 'map', 'recip_rank']
    ev = pytrec_eval.RelevanceEvaluator(qrels_dict, measures)
    results = ev.evaluate(run_dict)
    agg = {m: 0.0 for m in measures}
    for r in results.values():
        for m in measures:
            agg[m] += r.get(m, 0.0)
    return {m: agg[m] / len(results) for m in measures}


# ── Load qrels and queries ────────────────────────────────────────────────────
qrels_raw  = load_qrels_dict(QRELS_RAW)
qrels_dctr = load_qrels_dict(QRELS_DCTR)
topics = pd.read_csv(QUERIES, sep='\t', header=None, names=['qid', 'query'])
print(f'Loaded {len(topics)} training queries', flush=True)
qid_to_query = dict(zip(topics['qid'].astype(str), topics['query']))

# ── Build doc store from JSONL ────────────────────────────────────────────────
print('Building doc store...', flush=True)
doc_store = {}
for fpath in sorted(DOC_DIR.glob('*.jsonl')):
    with open(fpath) as fh:
        for line in fh:
            doc = json.loads(line)
            title    = doc.get('title', '') or ''
            abstract = doc.get('abstract', '') or ''
            doc_store[str(doc['id'])] = (title + ' ' + abstract).strip()
print(f'Loaded {len(doc_store)} docs', flush=True)

# ── BM25 ──────────────────────────────────────────────────────────────────────
index = pt.IndexFactory.of(str(IDX))
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()
topics_tok = topics.copy()
topics_tok['query'] = topics_tok['query'].apply(lambda q: ' '.join(tokeniser.getTokens(q)))
bm25_run = pt.terrier.Retriever(index, wmodel='BM25', num_results=1000)(topics_tok)
bm25_ranked = {}
for qid, grp in bm25_run.groupby('qid'):
    grp = grp.sort_values('score', ascending=False).reset_index(drop=True)
    bm25_ranked[qid] = {row['docno']: int(idx + 1) for idx, row in grp.iterrows()}
print('BM25 done', flush=True)

# ── Dense ─────────────────────────────────────────────────────────────────────
model = SentenceTransformer('Qwen/Qwen3-Embedding-4B', trust_remote_code=True)
model.max_seq_length = 512
query_vecs = model.encode(topics['query'].tolist(), batch_size=32,
    show_progress_bar=True, normalize_embeddings=True, prompt_name='query').astype('float32')
topics['query_vec'] = list(query_vecs)
flex = FlexIndex(str(FLEX_PATH))
dense_run = flex.retriever(num_results=1000)(topics)
dense_ranked = {}
for qid, grp in dense_run.groupby('qid'):
    grp = grp.sort_values('score', ascending=False).reset_index(drop=True)
    dense_ranked[qid] = {row['docno']: int(idx + 1) for idx, row in grp.iterrows()}
print('Dense done', flush=True)

# ── RRF ───────────────────────────────────────────────────────────────────────
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

# ── CrossEncoder reranking ────────────────────────────────────────────────────
print(f'Loading CrossEncoder...', flush=True)
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

ce_dict = {}
for i, (qid, ranked_docs) in enumerate(rrf_dict.items()):
    query_text = qid_to_query.get(str(qid), '')
    top_docs   = sorted(ranked_docs.items(), key=lambda x: -x[1])[:CE_DEPTH]
    pairs      = [(query_text, doc_store.get(str(docno), '')) for docno, _ in top_docs]
    scores     = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
    ce_dict[qid] = {docno: float(s) for (docno, _), s in zip(top_docs, scores)}
    if (i + 1) % 10 == 0:
        print(f'  Reranked {i+1}/{len(rrf_dict)} queries', flush=True)
print('CrossEncoder reranking done', flush=True)

# ── Evaluate all models ───────────────────────────────────────────────────────
bm25_dict  = {qid: {d: 1.0/(K+r) for d, r in ranked.items()} for qid, ranked in bm25_ranked.items()}
dense_dict = {qid: {d: 1.0/(K+r) for d, r in ranked.items()} for qid, ranked in dense_ranked.items()}

for qrels_name, qrels in [('Raw', qrels_raw), ('DCTR', qrels_dctr)]:
    print(f"\n{'Model':<20} {'nDCG@10':>9} {'nDCG@20':>9} {'MAP':>8} {'MRR':>8}  ({qrels_name} qrels)")
    print('-' * 65)
    for name, rd in [('BM25', bm25_dict), ('Dense', dense_dict),
                     ('RRF', rrf_dict), ('CrossEncoder', ce_dict)]:
        r = evaluate(rd, qrels)
        print(f"{name:<20} {r['ndcg_cut_10']:>9.4f} {r['ndcg_cut_20']:>9.4f} "
              f"{r['map']:>8.4f} {r['recip_rank']:>8.4f}")
