"""2-D maps of the chunk probe's saved vectors (run in the `parse` env: scikit-learn):
t-SNE (cosine) of each representation, coloured by the diagnostic phrase type.
Usage: probe_plot.py <tag> <seed>"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

tag, seed = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"experiments/v3/results/probe/chunks_{tag}_s{seed}.json", encoding="utf-8"))
keys, labels = d["keys"], np.array(d["labels"])
colors = {"NP": "#2b6cb0", "VP": "#c05621", "PP": "#2f855a", "S": "#6b46c1", "other": "#a0aec0"}
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, rep in zip(axes, ("content", "context")):
    X = np.load(f"experiments/v3/results/probe/X_{rep}_{tag}_s{seed}.npy")
    keep = np.linalg.norm(X, axis=1) > 0
    Y = TSNE(2, metric="cosine", init="pca", perplexity=min(30, max(5, keep.sum() // 4)), random_state=seed).fit_transform(X[keep])
    lab = labels[keep]
    ks = [k for k, m in zip(keys, keep) if m]
    for t, c in colors.items():
        m = lab == t
        ax.scatter(Y[m, 0], Y[m, 1], s=14, c=c, label=f"{t} ({int(m.sum())})", alpha=0.8, edgecolors="none")
    for r in range(min(60, len(ks))):
        ax.annotate(ks[r], Y[r], fontsize=6.5, alpha=0.85)
    pur = d.get(f"{rep}: 5-NN phrase-type purity / shuffled")
    ax.set_title(f"{rep} representation (t-SNE, cosine); 5-NN purity {pur[0]:.2f} vs shuffled {pur[1]:.2f}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"TRELLIS chunk types, TinyStories {d['n']} sentences, seed {seed} ({tag}); colour = NLTK phrase type (diagnostic only)", fontsize=11)
fig.tight_layout()
out = f"experiments/v3/results/probe/chunks_tsne_{tag}_s{seed}.png"
fig.savefig(out, dpi=150)
print(out)
