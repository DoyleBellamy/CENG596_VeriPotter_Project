#!/usr/bin/env python3
"""
BM25 retrieval for official submission across all 3 snapshots.
  - Snapshot-1: fulltext corpus (fullText field, fall back to abstract)
  - Snapshot-2: abstract corpus (no fulltext available for test set)
  - Snapshot-3: abstract corpus (no fulltext available for test set)
All runs use the 381 official test queries.
"""
import os, json, gzip
from pathlib import Path
import pandas as pd
import pyterrier as pt

pt.java.init()

PROJECT      = Path(__file__).parent
IR_DATA      = Path(os.environ.get('IR_DATASETS_HOME', PROJECT / 'ir_datasets')) / 'longeval-sci-2026'
TEST_QUERIES = IR_DATA / 'longeval_adhoc-queries-snapshot-test.tsv'

SNAPSHOTS = {
    'snapshot-1': {
        'doc_dir': PROJECT / 'longeval_sci_training_2026_fulltext/data/processed/doc_collection_parallel_09032026_parallel_2/snapshot-1/longeval_sci_training_2026_fulltext/documents',
        'index_dir': PROJECT / 'indexes/bm25_fulltext/snapshot-1',
        'output_dir': PROJECT / 'output/bm25_fulltext/snapshot-1',
        'text_field': 'fullText',   # capital T, falls back to abstract
    },
    'snapshot-2': {
        'doc_dir': IR_DATA / 'longeval_sci_06_08_2026_documents/data/processed/doc_collection_09032026_abstract_2/snapshot-2/longeval_sci_test-06-08_2026_abstract/documents',
        'index_dir': PROJECT / 'indexes/bm25_fulltext/snapshot-2',
        'output_dir': PROJECT / 'output/bm25_fulltext/snapshot-2',
        'text_field': 'abstract',
    },
    'snapshot-3': {
        'doc_dir': IR_DATA / 'longeval_sci_09_11_2026_documents/data/processed/doc_collection_09032026_abstract_2/snapshot-3/longeval_sci_test-09-11_2026_abstract/documents',
        'index_dir': PROJECT / 'indexes/bm25_fulltext/snapshot-3',
        'output_dir': PROJECT / 'output/bm25_fulltext/snapshot-3',
        'text_field': 'abstract',
    },
}

tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()

topics = pd.read_csv(TEST_QUERIES, sep='\t', header=None, names=['qid', 'query'])
print(f'Loaded {len(topics)} test queries', flush=True)
topics['query'] = topics['query'].apply(lambda q: ' '.join(tokeniser.getTokens(q)))


def doc_iter(doc_dir, text_field):
    for fpath in sorted(doc_dir.glob('*.jsonl')):
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                docno = str(doc.get('id', doc.get('docno', '')))
                title = doc.get('title', '') or ''
                body = doc.get(text_field) or doc.get('abstract') or ''
                text = (title + ' ' + body).strip()
                if docno and text:
                    yield {'docno': docno, 'text': text}


for snap, cfg in SNAPSHOTS.items():
    print(f'\n=== {snap} ===', flush=True)
    cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    cfg['index_dir'].mkdir(parents=True, exist_ok=True)

    out_file = cfg['output_dir'] / 'run.txt.gz'
    if out_file.exists():
        print(f'  Output already exists, skipping.', flush=True)
        continue

    # Build index if needed
    if not (cfg['index_dir'] / 'data.properties').exists():
        print(f'  Indexing {snap} (field={cfg["text_field"]})...', flush=True)
        indexer = pt.IterDictIndexer(
            str(cfg['index_dir']), overwrite=True,
            meta={'docno': 100, 'text': 4096},
        )
        indexer.index(doc_iter(cfg['doc_dir'], cfg['text_field']))
        stats = pt.IndexFactory.of(str(cfg['index_dir'])).getCollectionStatistics()
        print(f'  Indexed: {stats.getNumberOfDocuments()} docs, '
              f'{stats.getNumberOfUniqueTerms()} terms, '
              f'{stats.getNumberOfTokens()} tokens', flush=True)
    else:
        print(f'  Index already exists, skipping indexing.', flush=True)

    # Retrieve
    print(f'  Retrieving...', flush=True)
    index = pt.IndexFactory.of(str(cfg['index_dir']))
    bm25 = pt.terrier.Retriever(index, wmodel='BM25', num_results=1000)
    run = bm25(topics)

    with gzip.open(out_file, 'wt') as f:
        for _, row in run.iterrows():
            f.write(f"{row['qid']} Q0 {row['docno']} {int(row['rank'])} {row['score']:.6f} bm25-fulltext\n")

    print(f'  Written {len(run)} results to {out_file}', flush=True)

print('\nAll snapshots done.', flush=True)
