#!/usr/bin/env python3
"""BM25 retrieval on fulltext corpus (snapshot-1 only, for comparison with abstract BM25)."""
import os, json, gzip
from pathlib import Path
import pandas as pd
import pyterrier as pt

pt.java.init()

PROJECT      = Path(__file__).parent
FULLTEXT_DIR = PROJECT / 'longeval_sci_training_2026_fulltext'
INDEX_DIR    = PROJECT / 'indexes' / 'bm25_fulltext' / 'snapshot-1'
OUTPUT_DIR   = PROJECT / 'output' / 'bm25_fulltext' / 'snapshot-1'
IR_DATA      = Path(os.environ.get('IR_DATASETS_HOME', PROJECT / 'ir_datasets')) / 'longeval-sci-2026'
QUERIES      = IR_DATA / 'task1_longeval_adhoc-queries-snapshot-train.tsv'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Find snapshot-1 document files ──────────────────────────────────────────
# Unzipped fulltext should mirror abstract structure; find snapshot-1 JSONL files
candidates = list(FULLTEXT_DIR.rglob('*snapshot-1*/*.jsonl'))
if not candidates:
    candidates = list(FULLTEXT_DIR.rglob('*.jsonl'))

# Filter to snapshot-1 only
snap1_files = [f for f in candidates if 'snapshot-1' in str(f)]
if not snap1_files:
    snap1_files = candidates  # fallback: use all if no snapshot subdir found

print(f"Found {len(snap1_files)} JSONL file(s) for snapshot-1:", flush=True)
for f in sorted(snap1_files):
    print(f"  {f}", flush=True)

# Peek at first doc to see field names
first_file = sorted(snap1_files)[0]
with open(first_file) as fh:
    first_doc = json.loads(fh.readline())
print(f"\nFirst doc keys: {list(first_doc.keys())}", flush=True)
print(f"Sample title  : {str(first_doc.get('title',''))[:80]}", flush=True)
print(f"Sample text snippet: {str(first_doc.get('fulltext', first_doc.get('text', first_doc.get('body', ''))))[:200]}", flush=True)

# ── Document generator ───────────────────────────────────────────────────────
def doc_iter(files):
    for fpath in sorted(files):
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                docno = str(doc.get('id', doc.get('docno', '')))
                title = doc.get('title', '') or ''
                # field is 'fullText' (capital T); fall back to abstract if missing
                body = (doc.get('fullText') or doc.get('fulltext') or
                        doc.get('body') or doc.get('abstract') or '')
                text = (title + ' ' + body).strip()
                if docno and text:
                    yield {'docno': docno, 'text': text}

# ── Build index ──────────────────────────────────────────────────────────────
if True:  # always reindex (fixed fullText field name)
    print("\nBuilding BM25 fulltext index for snapshot-1...", flush=True)
    indexer = pt.IterDictIndexer(
        str(INDEX_DIR), overwrite=True,
        meta={'docno': 100, 'text': 4096},  # store less text meta to save RAM
    )
    indexer.index(doc_iter(snap1_files))
    print("Indexing complete.", flush=True)
else:
    print("\nIndex already exists, skipping indexing.", flush=True)

index = pt.IndexFactory.of(str(INDEX_DIR))
stats = index.getCollectionStatistics()
print(f"Index stats: {stats.getNumberOfDocuments()} docs, "
      f"{stats.getNumberOfUniqueTerms()} terms, "
      f"{stats.getNumberOfTokens()} tokens", flush=True)

# ── Retrieve on training queries ─────────────────────────────────────────────
print("\nRunning BM25 retrieval on training queries...", flush=True)
topics = pd.read_csv(QUERIES, sep='\t', header=None, names=['qid', 'query'])
print(f"Loaded {len(topics)} queries", flush=True)

tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()
topics['query'] = topics['query'].apply(lambda q: ' '.join(tokeniser.getTokens(q)))

bm25 = pt.terrier.Retriever(index, wmodel='BM25', num_results=1000)
run = bm25(topics)

out_file = OUTPUT_DIR / 'run.txt.gz'
with gzip.open(out_file, 'wt') as f:
    for _, row in run.iterrows():
        f.write(f"{row['qid']} Q0 {row['docno']} {int(row['rank'])} {row['score']:.6f} bm25-fulltext\n")

print(f"Written {len(run)} results to {out_file}", flush=True)
print("Done.", flush=True)
