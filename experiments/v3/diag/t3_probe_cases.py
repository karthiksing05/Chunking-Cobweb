"""Case study for S3 (chunk representations): nearest neighbours of a few chunk
types under the context and the content representation (cosine), from the
vectors t3_chunk_probe.py saved. Phrase types (NLTK) label the neighbours for the
reader only.

Usage: t3_probe_cases.py <tag> <seed> <chunk type> [<chunk type> ...]
"""
import json
import sys

import numpy as np

tag, seed, queries = sys.argv[1], sys.argv[2], sys.argv[3:]
base = "experiments/v3/results/probe/"
d = json.load(open(f"{base}chunks_{tag}_s{seed}.json", encoding="utf-8"))
keys, labels = d["keys"], d["labels"]
for rep in ("context", "content"):
    X = np.load(f"{base}X_{rep}_{tag}_s{seed}.npy").astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    for q in queries:
        if q not in keys:
            continue
        i = keys.index(q)
        sim = X @ X[i]
        sim[i] = -2
        nn = np.argsort(-sim)[:5]
        print(json.dumps({"rep": rep, "query": q, "type": labels[i],
                          "neighbours": [[keys[j], labels[j], round(float(sim[j]), 3)] for j in nn],
                          "same type": int(sum(labels[j] == labels[i] for j in nn))}, ensure_ascii=False))
