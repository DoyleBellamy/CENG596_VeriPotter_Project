#!/usr/bin/env python3
"""BM25 paper search demo — LongEval 2026 snapshot-1 (abstract corpus)."""
import json
from pathlib import Path

import pandas as pd
import pyterrier as pt
from flask import Flask, render_template, request

PROJECT   = Path(__file__).parent.parent
INDEX_DIR = PROJECT / 'indexes/bm25/snapshot-1'
DOC_DIR   = (PROJECT / 'longeval_sci_training_2026_abstract/data/processed'
             '/doc_collection_09032026_abstract_2/snapshot-1'
             '/longeval_sci_training_2026_abstract/documents')

app = Flask(__name__)

# Loaded once on first request
_loaded = False
doc_store = {}
bm25 = None
tokeniser = None


def _load():
    global _loaded, doc_store, bm25, tokeniser
    if _loaded:
        return
    print('Initialising PyTerrier...', flush=True)
    pt.java.init()

    print('Loading doc store...', flush=True)
    for fpath in sorted(DOC_DIR.glob('*.jsonl')):
        with open(fpath) as fh:
            for line in fh:
                doc = json.loads(line)
                authors = [a.get('name', '') for a in doc.get('authors', [])]
                doc_store[str(doc['id'])] = {
                    'title':    doc.get('title', '') or '',
                    'abstract': doc.get('abstract', '') or '',
                    'authors':  ', '.join(authors[:5]),
                }
    print(f'Loaded {len(doc_store)} docs.', flush=True)

    index = pt.IndexFactory.of(str(INDEX_DIR))
    bm25  = pt.terrier.Retriever(index, wmodel='BM25', num_results=1000)
    tokeniser = pt.java.autoclass(
        'org.terrier.indexing.tokenisation.Tokeniser'
    ).getTokeniser()
    _loaded = True
    print('Ready.', flush=True)


@app.before_request
def ensure_loaded():
    _load()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    n     = min(int(request.args.get('n', 10)), 100)

    if not query:
        return render_template('index.html', error='Please enter a query.')

    query_tok = ' '.join(tokeniser.getTokens(query))
    topics    = pd.DataFrame([{'qid': '1', 'query': query_tok}])
    run       = bm25(topics).head(n)

    results = []
    for _, row in run.iterrows():
        meta = doc_store.get(str(row['docno']), {})
        results.append({
            'rank':     int(row['rank']) + 1,
            'score':    round(float(row['score']), 3),
            'docno':    row['docno'],
            'title':    meta.get('title', '(no title)'),
            'authors':  meta.get('authors', ''),
            'abstract': meta.get('abstract', '')[:300],
        })

    return render_template('index.html', query=query, n=n, results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
