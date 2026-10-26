"""Representational probe of the chunks memory uses (TinyStories): every committed
chunk type (word string chosen in an analysis, 2+ occurrences) gets two learned
representations, read from the current trees and averaged over its instances:

  content   the band concepts on its content leaf's path (what it is made of)
  context   its type name: the parent concepts of its sampled context leaves (where
            chunks of its content concept stand)

Each is projected to 2-D (PCA of the L2-normalised concept vectors) and plotted,
coloured by a rough phrase type from NLTK tags in the sentence (NP / VP / PP / S /
other; a diagnostic only, never seen by the learner). Quantitative: 5-NN phrase-type
purity in each space (cosine) against the same with labels shuffled.

Usage: t3_chunk_probe.py <cfg json> <seed> <n> <tag>
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
import nltk  # noqa: E402
from trellis2.stories import heldout_path, load_corpus  # noqa: E402
from trellis2.trellis3 import FILL, Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
L = Trellis3(seed=seed, **cfg)
phrase_tags = defaultdict(Counter)
for s in corpus[:n]:
    tree = L.learn(s)
    tags = [t for _, t in nltk.pos_tag(s)]
    for (i, j) in tree.brackets():
        if j - i >= 2:
            phrase_tags[tuple(s[i:j])][tuple(tags[i:j])] += 1


def phrase_type(tags):
    t = [x[:2] for x in tags]
    if t[0] in ("IN", "TO"):
        return "PP"
    if t[0] == "VB" or (t[0] in ("MD", "RB") and "VB" in t):
        return "VP"
    nounish = ("NN", "PR", "DT", "CD", "JJ", "CC", "PO")
    if all(x in nounish for x in t) and any(x in ("NN", "PR") for x in t):
        return "NP"
    if any(x in ("NN", "PR") for x in t[:2]) and "VB" in t:
        return "S"
    return "other"


# hierarchical vectors: a concept stands for every concept on its path (root left out),
# so two chunks are as similar as the ancestry their concepts share
reps = {"content": defaultdict(Counter), "context": defaultdict(Counter)}
count = Counter()
L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
ctx_ix, cnt_ix = L.concept_index()


def add_path(vec, tree, node, w, pre):
    for a in tree.ancestors(node)[::-1][1:]:
        vec[pre + str(a.id)] += w


for e in L.records:
    if e.token is None and e.cleaf is not None and any(w == "ctx" and a == FILL for w, a, _, _ in e.pas):
        key = L._type_key(e)
        count[key] += 1
        add_path(reps["content"][key], L.cnt, e.cleaf, 1.0, "n")
        bag = L._res_bag(("c", e.cleaf.id)) or {}
        for v, w in bag.items():
            if v.startswith("c") and int(v[1:]) in ctx_ix:
                add_path(reps["context"][key], L.ctx, ctx_ix[int(v[1:])], w, "x")
keys = [k for k, c in count.most_common(600) if c >= 2 and k in phrase_tags]
labels = [phrase_type(phrase_tags[k].most_common(1)[0][0]) for k in keys]
os.makedirs("experiments/v3/results/probe", exist_ok=True)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

colors = {"NP": "#2b6cb0", "VP": "#c05621", "PP": "#2f855a", "S": "#6b46c1", "other": "#a0aec0"}
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
out = {"tag": tag, "seed": seed, "n": n, "chunk types": len(keys), "phrase types": Counter(labels)}
rng = np.random.default_rng(seed)
for ax, rep in zip(axes, ("content", "context")):
    vocab = sorted({v for k in keys for v in reps[rep][k]})
    col = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(keys), len(vocab)))
    for r, k in enumerate(keys):
        for v, w in reps[rep][k].items():
            X[r, col[v]] = w
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    np.save(f"experiments/v3/results/probe/X_{rep}_{tag}_s{seed}.npy", X)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :5]
    lab = np.array(labels)
    pur = float(np.mean(lab[nn] == lab[:, None]))
    shuf = lab.copy()
    rng.shuffle(shuf)
    base = float(np.mean(shuf[nn] == shuf[:, None]))
    out[f"{rep}: 5-NN phrase-type purity / shuffled"] = [round(pur, 3), round(base, 3)]
    Xc = X - X.mean(0)
    U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    Y = U[:, :2] * sv[:2]
    out[f"{rep}: variance in 2 PCs"] = round(float((sv[:2] ** 2).sum() / (sv ** 2).sum()), 3)
    for t in colors:
        m = lab == t
        ax.scatter(Y[m, 0], Y[m, 1], s=12, c=colors[t], label=f"{t} ({int(m.sum())})", alpha=0.75, edgecolors="none")
    for r in range(min(45, len(keys))):
        ax.annotate(" ".join(keys[r]), Y[r], fontsize=6.5, alpha=0.85)
    ax.set_title(f"{rep} representation: 5-NN purity {pur:.2f} (shuffled {base:.2f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"TRELLIS chunk types on TinyStories ({n} sentences, seed {seed}, {tag}); colour = NLTK phrase type (diagnostic)", fontsize=11)
fig.tight_layout()
png = f"experiments/v3/results/probe/chunks_{tag}_s{seed}.png"
fig.savefig(png, dpi=150)
out["png"] = png
out["examples by type"] = {t: [" ".join(k) for k, l in zip(keys, labels) if l == t][:12] for t in colors}
json.dump({"keys": [" ".join(k) for k in keys], "labels": labels, **out},
          open(f"experiments/v3/results/probe/chunks_{tag}_s{seed}.json", "w", encoding="utf-8"))
print(json.dumps(out))
