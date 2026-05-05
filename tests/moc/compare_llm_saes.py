"""
compare_llm_saes.py
====================
Load the SAE latents produced by train_llm_saes.py and evaluate them on:

  1. Summary table  – mean L0, dead-neuron %, linear-probe accuracy (POS),
                       KNN@5 accuracy.  Saved as gpt_acts_output/summary.csv.
  2. KNN curves     – accuracy vs k for k in {1,3,5,10,20,50}.
  3. Linear probe   – per-class (per POS tag) bar chart.
  4. UMAP scatter   – 2D projections coloured by POS tag.
  5. t-SNE scatter  – same.
  6. Feature-activation histograms – distribution of activation magnitudes.
  7. Dead-neuron bar chart          – % dead neurons per method.

All plots saved to gpt_acts_output/plots/.

Usage:
    python tests/moc/compare_llm_saes.py
"""

import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from umap import UMAP
from sklearn.manifold import TSNE

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(HERE, "gpt_acts_output")
ACT_DIR   = os.path.join(OUT_DIR, "acts")
MODEL_DIR = os.path.join(OUT_DIR, "models")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load arrays ───────────────────────────────────────────────────────────────

print("Loading arrays …")
pos_train = np.load(os.path.join(ACT_DIR, "pos_train.npy"))
pos_test  = np.load(os.path.join(ACT_DIR, "pos_test.npy"))

with open(os.path.join(ACT_DIR, "pos_vocab.json")) as f:
    pos_vocab = json.load(f)  # {tag: idx}
IDX_TO_POS = {v: k for k, v in pos_vocab.items()}
POS_LABELS = [IDX_TO_POS[i] for i in range(len(IDX_TO_POS))]

# Convert int16 indices to strings for display
y_train = np.array([IDX_TO_POS[i] for i in pos_train])
y_test  = np.array([IDX_TO_POS[i] for i in pos_test])

def load_z(tag):
    tr = np.load(os.path.join(MODEL_DIR, f"Z_{tag}_train.npy"))
    te = np.load(os.path.join(MODEL_DIR, f"Z_{tag}_test.npy"))
    return tr.astype(np.float32), te.astype(np.float32)

print("  Loading SAE latents …")
Z_l1sae_tr,   Z_l1sae_te   = load_z("l1sae")
Z_topksae_tr, Z_topksae_te = load_z("topksae")
Z_jumprelu_tr,Z_jumprelu_te= load_z("jumprelu")
Z_cobweb_tr,  Z_cobweb_te  = load_z("cobweb")

with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)

D_SAE  = meta["d_sae"]
TOP_K  = meta["top_k"]
L1_LAM = meta["l1_lam"]
print(f"  D_SAE={D_SAE}  TOP_K={TOP_K}")

# ── Helpers ───────────────────────────────────────────────────────────────────

KNN_KS = [1, 3, 5, 10, 20, 50]


def repr_stats(Z):
    """Return (mean_l0, dead_pct): avg non-zeros per row; % features always 0."""
    nz = (Z != 0)
    return nz.sum(axis=1).mean(), (~nz.any(axis=0)).mean() * 100


def linear_probe(Z_tr, y_tr, Z_te, y_te):
    """Logistic regression (L2, max_iter=1000). Returns (overall, per_class dict)."""
    le  = LabelEncoder().fit(y_tr)
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(Z_tr, le.transform(y_tr))
    y_pred = le.inverse_transform(clf.predict(Z_te))
    overall = (y_pred == y_te).mean()
    per_class = {}
    for tag in le.classes_:
        mask = y_te == tag
        if mask.sum() > 0:
            per_class[tag] = (y_pred[mask] == y_te[mask]).mean()
    return overall, per_class


def knn_curves(Z_tr, y_tr, Z_te, y_te, ks=KNN_KS):
    return [
        KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(Z_tr, y_tr).score(Z_te, y_te)
        for k in ks
    ]


# ── Methods registry ──────────────────────────────────────────────────────────

METHODS = [
    (f"L1-SAE (d={D_SAE}, λ={L1_LAM})",       Z_l1sae_tr,    Z_l1sae_te,    "o-", "#4878d0"),
    (f"TopK-SAE (d={D_SAE}, k={TOP_K})",        Z_topksae_tr,  Z_topksae_te,  "s-", "#ee854a"),
    (f"JumpReLU-SAE (d={D_SAE}, k≈{TOP_K})",   Z_jumprelu_tr, Z_jumprelu_te, "^-", "#6acc65"),
    (f"Cobweb-TopK (d={D_SAE}, k={TOP_K})",     Z_cobweb_tr,   Z_cobweb_te,   "D-", "#d65f5f"),
]

# ── Evaluate ──────────────────────────────────────────────────────────────────

print("\nEvaluating …")
results = []
knn_accs_all = []
for name, Z_tr, Z_te, marker, color in METHODS:
    print(f"  {name} …", flush=True)
    avg_l0, dead_pct = repr_stats(Z_tr)
    overall, per_cls = linear_probe(Z_tr, y_train, Z_te, y_test)
    knns              = knn_curves(Z_tr, y_train, Z_te, y_test)
    knn5              = knns[KNN_KS.index(5)] * 100
    print(f"    lin={overall*100:.1f}%  knn@5={knn5:.1f}%  "
          f"avg_l0={avg_l0:.1f}  dead={dead_pct:.1f}%")
    results.append({
        "method":        name,
        "lin_probe_pct": round(overall * 100, 2),
        "knn5_pct":      round(knn5, 2),
        "avg_l0":        round(float(avg_l0), 2),
        "dead_pct":      round(float(dead_pct), 2),
    })
    knn_accs_all.append((knns, per_cls))

# ── Save summary CSV ──────────────────────────────────────────────────────────

csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "lin_probe_pct", "knn5_pct", "avg_l0", "dead_pct"])
    w.writeheader()
    w.writerows(results)
print(f"\nSummary saved → {csv_path}")

# ── Print table ───────────────────────────────────────────────────────────────

print(f"\n  {'Method':<50} {'Lin.Probe':>10} {'KNN@5':>7} {'Avg L0':>8} {'Dead%':>7}")
print(f"  {'-'*86}")
for r in results:
    print(f"  {r['method']:<50} {r['lin_probe_pct']:>9.1f}% {r['knn5_pct']:>6.1f}% "
          f"{r['avg_l0']:>8.1f} {r['dead_pct']:>6.1f}%")

# ── Plot helpers ──────────────────────────────────────────────────────────────

CMAP_POS = plt.get_cmap("tab20")
pos_color = {tag: CMAP_POS(i / len(POS_LABELS)) for i, tag in enumerate(POS_LABELS)}

# ── 1. KNN curves ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
for (name, _, _, marker, color), (knns, _) in zip(METHODS, knn_accs_all):
    ax.plot(KNN_KS, [a * 100 for a in knns], marker, label=name, color=color)
ax.set_xlabel("k  (number of neighbours)")
ax.set_ylabel("Test Accuracy %")
ax.set_title("KNN Test Accuracy vs k  —  GPT-2 Last-Layer SAEs")
ax.set_xticks(KNN_KS)
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "knn_vs_k.png"), dpi=120)
plt.close()
print("Saved knn_vs_k.png")

# ── 2. Linear probe per-class bar chart ───────────────────────────────────────

n_m = len(METHODS)
tags_sorted = sorted(POS_LABELS)
x = np.arange(len(tags_sorted))
w = 0.8 / n_m
offsets = [(i - (n_m - 1) / 2) * w for i in range(n_m)]

fig, ax = plt.subplots(figsize=(max(14, len(tags_sorted) * 1.2), 5))
for i, ((name, _, _, _, color), (_, per_cls)) in enumerate(zip(METHODS, knn_accs_all)):
    vals = [per_cls.get(t, 0.0) * 100 for t in tags_sorted]
    ax.bar(x + offsets[i], vals, w, label=name, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(tags_sorted, rotation=45, ha="right")
ax.set_ylabel("Test Accuracy %")
ax.set_title("Linear Probe — Per-POS-tag Accuracy")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "linear_probe_per_pos.png"), dpi=120)
plt.close()
print("Saved linear_probe_per_pos.png")

# ── 3. Dead-neuron bar chart ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
names_short = [r["method"].split(" (")[0] for r in results]
dead_vals   = [r["dead_pct"] for r in results]
colors      = [c for _, _, _, _, c in METHODS]
bars = ax.bar(names_short, dead_vals, color=colors, alpha=0.85, edgecolor="black")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_ylabel("Dead neurons %")
ax.set_title("Dead Neurons per SAE Variant")
ax.set_ylim(0, max(dead_vals) * 1.25 + 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "dead_neurons.png"), dpi=120)
plt.close()
print("Saved dead_neurons.png")

# ── 4. Feature-activation magnitude histograms ───────────────────────────────

fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 4), sharey=True)
for ax, (name, Z_tr, _, _, color) in zip(axes, METHODS):
    acts_flat = Z_tr[Z_tr != 0].ravel()
    if len(acts_flat) > 200_000:
        acts_flat = np.random.default_rng(42).choice(acts_flat, 200_000, replace=False)
    ax.hist(acts_flat, bins=60, color=color, alpha=0.85, edgecolor="none")
    ax.set_title(name.split(" (")[0], fontsize=9)
    ax.set_xlabel("Activation magnitude")
    ax.set_yscale("log")
axes[0].set_ylabel("Count (log scale)")
fig.suptitle("Activation Magnitude Distributions (non-zero only)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "act_histograms.png"), dpi=120)
plt.close()
print("Saved act_histograms.png")

# ── 5. UMAP scatter ───────────────────────────────────────────────────────────

# Sub-sample for speed — UMAP is O(N log N)
UMAP_N = min(5_000, len(Z_l1sae_tr))
rng    = np.random.default_rng(42)
_idx   = rng.choice(len(Z_l1sae_tr), UMAP_N, replace=False)
y_sub  = y_train[_idx]

print(f"Computing UMAP projections (N={UMAP_N}) …")
_umap = UMAP(n_components=2, random_state=42)

fig, axes = plt.subplots(1, n_m, figsize=(6 * n_m, 5))
fig.suptitle("UMAP Projections by POS tag", fontsize=12)
for ax, (name, Z_tr, _, _, _) in zip(axes, METHODS):
    Z2 = _umap.fit_transform(Z_tr[_idx])
    for tag in POS_LABELS:
        m = y_sub == tag
        if m.sum() > 0:
            ax.scatter(Z2[m, 0], Z2[m, 1], s=2, alpha=0.4,
                       color=pos_color[tag], label=tag)
    ax.set_title(name.split(" (")[0], fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
handles = [plt.Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=pos_color[t], markersize=7, label=t)
           for t in POS_LABELS]
fig.legend(handles=handles, title="POS", loc="center right",
           bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=2)
plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig(os.path.join(PLOT_DIR, "scatter_umap.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved scatter_umap.png")

# ── 6. t-SNE scatter ──────────────────────────────────────────────────────────

TSNE_N = min(3_000, len(Z_l1sae_tr))
_idx_t = rng.choice(len(Z_l1sae_tr), TSNE_N, replace=False)
y_tsub = y_train[_idx_t]

print(f"Computing t-SNE projections (N={TSNE_N}) …")
_tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)

fig, axes = plt.subplots(1, n_m, figsize=(6 * n_m, 5))
fig.suptitle("t-SNE Projections by POS tag", fontsize=12)
for ax, (name, Z_tr, _, _, _) in zip(axes, METHODS):
    Z2t = _tsne.fit_transform(Z_tr[_idx_t])
    for tag in POS_LABELS:
        m = y_tsub == tag
        if m.sum() > 0:
            ax.scatter(Z2t[m, 0], Z2t[m, 1], s=2, alpha=0.4,
                       color=pos_color[tag], label=tag)
    ax.set_title(name.split(" (")[0], fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
fig.legend(handles=handles, title="POS", loc="center right",
           bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=2)
plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig(os.path.join(PLOT_DIR, "scatter_tsne.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved scatter_tsne.png")

# ── 7. L0 vs linear-probe scatter ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
for (name, _, _, _, color), r in zip(METHODS, results):
    ax.scatter(r["avg_l0"], r["lin_probe_pct"], s=120, color=color,
               edgecolors="black", zorder=3, label=name)
    ax.annotate(name.split(" (")[0], (r["avg_l0"], r["lin_probe_pct"]),
                textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("Average L0 norm")
ax.set_ylabel("Linear Probe Accuracy %")
ax.set_title("Sparsity vs. Linear Separability")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "l0_vs_linear.png"), dpi=120)
plt.close()
print("Saved l0_vs_linear.png")

print(f"\nAll plots saved to {PLOT_DIR}")
