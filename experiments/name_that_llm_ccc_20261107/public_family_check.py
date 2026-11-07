"""One fixed public-family comparator; exploratory, registered after first lockbox view.

TFIDF char(1,5) + word(1,2) + 15 standardized style features, concatenated
and fitted with balanced LR C=2. It is not the contestant's exact 60-feature code.
This comparator cannot change the original CCC gate choice.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from run_killtest import load_data, style_features, clf, score, SEED

start = time.time()
d = load_data(Path(sys.argv[1] if len(sys.argv) > 1 else 'D:/name-that-llm.zip'))
g = np.random.default_rng(SEED).permutation(sorted(d.prompt_id.unique()))
parts = {k: d[d.prompt_id.isin(v)].reset_index(drop=True) for k, v in
         [('fit', g[45:]), ('dev', g[25:45]), ('lockbox', g[:25])]}
blocks = {k: [] for k in parts}
for analyzer, gram, limit in [('char', (1, 5), 250000), ('word', (1, 2), 100000)]:
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=gram, min_df=2,
                       max_features=limit, sublinear_tf=True, dtype=np.float32,
                       token_pattern=r'(?u)\b\w+\b' if analyzer == 'word' else None)
    blocks['fit'].append(v.fit_transform(parts['fit'].text))
    for k in ('dev', 'lockbox'):
        blocks[k].append(v.transform(parts[k].text))
styles = {k: style_features(v.text) for k, v in parts.items()}
scale = StandardScaler().fit(styles['fit'])
for k in parts:
    blocks[k].append(csr_matrix(scale.transform(styles[k]) * .1))
x = {k: hstack(v).tocsr() for k, v in blocks.items()}
m = clf().fit(x['fit'], parts['fit'].model)
classes = sorted(d.model.unique())
result = {'name': 'B0_PUBLIC_FAMILY', 'scope': 'exploratory comparator after first lockbox opening; no tuning',
          'definition': 'concatenated char1-5 + word1-2 TFIDF + 15 style features; balanced LR C=2',
          'not_exact_contestant_reproduction': True,
          'balanced_accuracy': {k: score(parts[k].model, m.predict(x[k]), classes) for k in ('dev', 'lockbox')},
          'seconds': round(time.time() - start, 2)}
out = Path(__file__).resolve().parent / 'public_family_results.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
