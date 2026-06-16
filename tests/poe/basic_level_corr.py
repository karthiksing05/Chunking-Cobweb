"""Basic-level (closed-form expected-PMI) analysis for the PIXEL-CORRELATION hierarchy.
================================================================================
Adaptation of the image-based basic-level module to the PoE *primitives* setting.  Instead of
training a Continuous Cobweb tree on images, this loads the saved pixel cross-correlation matrix
``R`` (``<dataset>_output/primitives/R.npy``, written by the per-dataset poe scripts) and fits a
tree over its ROWS — i.e. one instance per pixel, that pixel's correlation profile against every
other pixel.  The "concepts" are then groups of pixels that co-vary; ``get_basic`` finds the
basic-level group for each pixel's branch.

Every concept is a diagonal Gaussian, so the expected pointwise mutual information against the
root has a closed form (``CobwebContinuousNode.expected_pmi``):

    EPMI(c) = E_{x|c} [ log p_c(x) - log p_root(x) ]

Walking a leaf to the root and taking the arg-max EPMI gives that branch's *basic level*
(``get_basic``).  NOTE vs the image module: the correlation tree has ``num_labels = 0`` (pixels
carry no class), so ``include_labels`` is False and the ``alpha`` knob is INERT — only
``prior_var`` shapes the basic level.  Empirically (see the slider) ``prior_var`` must be SMALL
(~0.01–1, the scale of correlations); large values flatten every Gaussian and collapse EPMI to 0.

Three static figures (written to ``<dataset>_output/primitives/basic_level/``) + an interactive
2-slider EPMI-by-depth explorer.  The correlation is taken over the SOFT donation weights and node
thumbnails show the TRUE continuous mean correlation profile (coolwarm), not a 0/1 mask.

  1. ``epmi_by_depth.png``       -- mean EPMI per tree depth, marking the peak depth (the average
     basic level) and the mean ``get_basic`` depth.
  2. ``dense_concept_tree.png``  -- node-link diagram of per-node mean correlation-profile
     thumbnails (coolwarm), y = node support.
  3. ``basic_level_regions.png`` -- the distinct basic-level nodes, each shown as its mean
     correlation profile (coolwarm, true value), ordered by how many pixels resolve to it.

Usage:
    python basic_level_corr.py [dataset] [--prior-var 0.3] [--alpha 10] [--top-k 24] [--interactive]
    (dataset in {colormnist, mnist, fashionmnist, multimnist}; default colormnist)
"""
from __future__ import annotations

import argparse, math, os
from collections import defaultdict

import numpy as np
from cobweb.cobweb_continuous import CobwebContinuousTree

HERE = os.path.dirname(os.path.abspath(__file__))
_E = np.zeros(0, np.float32)
RNG = np.random.default_rng(0)

# canvas shape per dataset (for laying correlation vectors back out in pixel space)
SHAPES = {"colormnist": (32, 32), "mnist": (32, 32), "fashionmnist": (32, 32),
          "multimnist": (28, 112)}

# Parameters that DEFINE the basic level.  prior_var small (correlation scale); alpha inert here.
BASIC_PRIOR_VAR = 0.3
BASIC_ALPHA = 10.0
INCLUDE_LABELS = False                       # correlation tree has num_labels=0

PRIOR_VAR_LOG10_MIN, PRIOR_VAR_LOG10_MAX = -3.0, 6.0
ALPHA_LOG10_MIN, ALPHA_LOG10_MAX = -3.0, 6.0


# --------------------------------------------------------------------------- #
# Load correlation data + build the correlation tree
# --------------------------------------------------------------------------- #
def load_R(dataset):
    path = os.path.join(HERE, f"{dataset}_output", "primitives", "R.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run tests/poe/{dataset}_poe.py first to write R.npy.")
    R = np.nan_to_num(np.load(path), nan=0.0).astype(np.float32)
    return R, SHAPES.get(dataset, (int(round(math.sqrt(R.shape[0]))),) * 2)


def build_tree(R):
    NP_ = R.shape[0]
    tree = CobwebContinuousTree(size=NP_, covar_from=1, num_labels=0)
    for i in RNG.permutation(NP_): tree.ifit(np.ascontiguousarray(R[i]), _E)
    return tree


# --------------------------------------------------------------------------- #
# Traversal helpers
# --------------------------------------------------------------------------- #
def collect_nodes(tree):
    out = []
    def _rec(n, d):
        out.append((d, n))
        for c in n.children: _rec(c, d + 1)
    _rec(tree.root, 0)
    return out


def node_depth(n):
    d = 0; p = n.parent
    while p is not None: d += 1; p = p.parent
    return d


def fp(n):
    """Content fingerprint — Cobweb returns fresh wrappers per access, so identity is unstable."""
    return (int(n.count), hash(np.asarray(n.mean, np.float32).tobytes()))


# --------------------------------------------------------------------------- #
# EPMI computations
# --------------------------------------------------------------------------- #
def mean_epmi_by_depth(nodes, root, prior_var, alpha):
    sums, counts = defaultdict(float), defaultdict(int)
    for d, n in nodes:
        if n is root: continue
        e = n.expected_pmi(prior_var, alpha, INCLUDE_LABELS)
        sums[d] += e; counts[d] += 1
    xs = sorted(counts)
    return xs, [sums[d] / counts[d] for d in xs], [counts[d] for d in xs]


def pixel_basics(R, tree, prior_var, alpha):
    """Per PIXEL (row of R): route to leaf, walk to basic-level node.  Returns
    (depths[NP], labels[NP] over distinct basic nodes, by_fp dict fp->{node,pixels})."""
    NP_ = R.shape[0]
    depths = np.empty(NP_, int); labels = np.empty(NP_, int)
    by_fp = {}                                                # fp -> {node, pixels, label}
    for p in range(NP_):
        b = tree.get_leaf(np.ascontiguousarray(R[p]), _E).get_basic(False, prior_var, alpha, INCLUDE_LABELS)
        f = fp(b)
        if f not in by_fp:
            by_fp[f] = {"node": b, "pixels": [], "label": len(by_fp)}
        by_fp[f]["pixels"].append(p)
        depths[p] = node_depth(b); labels[p] = by_fp[f]["label"]
    return depths, labels, by_fp


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def corr_img(vec, shape):
    """Map a correlation vector in [-1,1] to a coolwarm RGB image — the TRUE continuous
    correlation value (no 0/1 binding)."""
    import matplotlib.pyplot as plt
    return plt.get_cmap("coolwarm")(np.clip((np.asarray(vec, np.float32).reshape(shape) + 1) / 2, 0, 1))[..., :3]


# --------------------------------------------------------------------------- #
# Figure 1 — mean EPMI by depth
# --------------------------------------------------------------------------- #
def plot_epmi_by_depth(nodes, depths_px, root, prior_var, alpha, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, ns = mean_epmi_by_depth(nodes, root, prior_var, alpha)
    peak = xs[int(np.argmax(ys))] if xs else 0
    mean_basic = float(np.mean(depths_px)) if len(depths_px) else 0.0
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(xs, ys, color="#1f77b4", lw=2, marker="o", ms=7, zorder=3, label="mean EPMI")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"{y:.2f}\n(n={n})", (x, y), textcoords="offset points",
                    xytext=(0, 9), fontsize=7, color="#1f77b4", ha="center")
    ax.axvline(peak, color="#d62728", ls="--", alpha=0.7, label=f"peak depth = {peak} (basic level)")
    ax.axvline(mean_basic, color="#2ca02c", ls=":", alpha=0.8,
               label=f"mean get_basic depth = {mean_basic:.2f}")
    ax.set_xlabel("hierarchy depth (root = 0, deeper = more specific)", fontsize=11)
    ax.set_ylabel(r"mean  $E_{x|c}[\,\log p_c(x) - \log p_{root}(x)\,]$", fontsize=11)
    ax.set_title(f"{title} correlation hierarchy — mean expected-PMI by depth (pixels only)\n"
                 f"prior_var = {prior_var:g}   alpha = {alpha:g} (inert: no labels)", fontsize=13)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9); fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig); return path


# --------------------------------------------------------------------------- #
# Figure 2 — node-link concept hierarchy of correlation-profile thumbnails
# --------------------------------------------------------------------------- #
def plot_count_tree(tree, shape, path, title, max_nodes=130, fig_h=6.5, cell_in=0.42):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    recs, keep = [], []
    def _rec(n, parent):
        i = len(recs); keep.append(n)
        recs.append({"parent": parent, "count": float(n.count),
                     "mean": np.asarray(n.mean, np.float32).copy(), "kids": []})
        for c in n.children: recs[i]["kids"].append(_rec(c, i))
        return i
    _rec(tree.root, -1)
    counts_desc = sorted((r["count"] for r in recs), reverse=True)
    thr = counts_desc[max_nodes - 1] if len(counts_desc) > max_nodes else 1.0
    kept = {i for i, r in enumerate(recs) if r["count"] >= thr}
    for r in recs: r["kk"] = [c for c in r["kids"] if c in kept]
    xpos, ctr = {}, [0]
    def _ax(i):
        kk = recs[i]["kk"]
        if not kk: xpos[i] = float(ctr[0]); ctr[0] += 1
        else:
            for c in kk: _ax(c)
            xpos[i] = sum(xpos[c] for c in kk) / len(kk)
    _ax(0)
    nL = max(ctr[0], 1); root_c = recs[0]["count"]
    yv = lambda c: math.log10(max(c, 1.0))
    dpi = 150; fig_w = max(14.0, nL * cell_in)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    zoom = max(0.07, min(0.8, 0.9 * (fig_w * dpi / (nL + 1)) / shape[1]))
    for i in kept:
        for c in recs[i]["kk"]:
            ax.plot([xpos[i], xpos[c]], [yv(recs[i]["count"]), yv(recs[c]["count"])],
                    color="0.8", lw=0.3, zorder=1)
    for i in kept:
        im = OffsetImage(corr_img(recs[i]["mean"], shape), zoom=zoom); im.image.axes = ax
        ax.add_artist(AnnotationBbox(im, (xpos[i], yv(recs[i]["count"])), frameon=True, pad=0.0,
                                     zorder=3, bboxprops=dict(edgecolor="0.55", linewidth=0.25)))
    ax.set_xlim(-1, nL); ax.set_ylim(yv(thr) - 0.05, yv(root_c) * 1.03)
    ax.set_xticks([])
    ticks = sorted({int(thr), 5, 10, 25, 50, 100, 200, 400, int(root_c)})
    ticks = [c for c in ticks if thr <= c <= root_c]
    ax.set_yticks([yv(c) for c in ticks]); ax.set_yticklabels([str(c) for c in ticks], fontsize=8)
    ax.set_ylabel("node support (pixel count)", fontsize=10)
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
    ax.set_title(f"{title} correlation hierarchy — concept thumbnails = mean correlation profile "
                 f"(coolwarm, true value); top {len(kept)} by support", fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); return path


# --------------------------------------------------------------------------- #
# Figure 3 — basic-level nodes: correlation prototype (large) + spatial footprint (inset)
# --------------------------------------------------------------------------- #
def plot_basic_level_regions(by_fp, NP_, shape, path, title, prior_var, alpha, top_k=24, ncols=6):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(by_fp.values(), key=lambda r: len(r["pixels"]), reverse=True)[:top_k]
    k = len(recs)
    if not k: return None
    ncols = min(ncols, k); nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.3 * nrows + 0.6), squeeze=False)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols); ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
        if idx >= k: ax.axis("off"); continue
        rec = recs[idx]; node = rec["node"]; pix = rec["pixels"]
        ax.imshow(corr_img(node.mean, shape))                        # node's TRUE mean correlation profile
        ax.text(0.03, 0.97, f"{len(pix)}px\nd{node_depth(node)} n={int(node.count)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))
    fig.suptitle(f"{title} correlation hierarchy — basic-level nodes (mean correlation profile, coolwarm true value)\n"
                 f"prior_var={prior_var:g}, alpha={alpha:g} (inert); "
                 f"top {k} of {len(by_fp)} basic nodes by pixel coverage (NP={NP_})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=140); plt.close(fig); return path


# --------------------------------------------------------------------------- #
# Interactive viewer (prior_var + alpha sliders)
# --------------------------------------------------------------------------- #
def interactive(nodes, R, tree, root, init_pv, init_alpha):
    import matplotlib
    if matplotlib.get_backend().lower() == "agg":
        for bk in ("MacOSX", "QtAgg", "TkAgg"):
            try: matplotlib.use(bk, force=True); break
            except Exception: continue
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    fig = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[16, 1, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0]); ax_pv = fig.add_subplot(gs[1]); ax_al = fig.add_subplot(gs[2])
    s_pv = Slider(ax_pv, r"$\log_{10}(\mathrm{prior\_var})$", PRIOR_VAR_LOG10_MIN,
                  PRIOR_VAR_LOG10_MAX, valinit=float(np.log10(init_pv)), valstep=0.01)
    s_al = Slider(ax_al, r"$\log_{10}(\alpha)$ [inert: no labels]", ALPHA_LOG10_MIN,
                  ALPHA_LOG10_MAX, valinit=float(np.log10(init_alpha)), valstep=0.01)
    leaves_px = [tree.get_leaf(np.ascontiguousarray(R[p]), _E) for p in range(R.shape[0])]
    def draw(pv, al):
        ax.clear()
        xs, ys, ns = mean_epmi_by_depth(nodes, root, pv, al)
        ax.plot(xs, ys, color="#1f77b4", lw=2, marker="o", ms=7, zorder=3, label="mean EPMI")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{y:.2f}\n(n={n})", (x, y), textcoords="offset points",
                        xytext=(0, 9), fontsize=7, color="#1f77b4", ha="center")
        peak = xs[int(np.argmax(ys))] if xs else 0
        ax.axvline(peak, color="#d62728", ls="--", alpha=0.7, label=f"peak depth = {peak}")
        basics = [lf.get_basic(False, pv, al, INCLUDE_LABELS) for lf in leaves_px]
        bd = [node_depth(b) for b in basics]; nreg = len({fp(b) for b in basics})
        ax.axvline(np.mean(bd), color="#2ca02c", ls=":", alpha=0.8,
                   label=f"mean get_basic depth = {np.mean(bd):.2f}  ({nreg} basic regions)")
        ax.set_xlabel("hierarchy depth (root = 0, deeper = more specific)", fontsize=11)
        ax.set_ylabel(r"mean  $E_{x|c}[\,\log p_c(x) - \log p_{root}(x)\,]$", fontsize=11)
        ax.set_title(f"correlation hierarchy EPMI by depth   prior_var={pv:.4g}   alpha={al:.4g}",
                     fontsize=12)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4); ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9); fig.canvas.draw_idle()
    s_pv.on_changed(lambda _v: draw(10 ** s_pv.val, 10 ** s_al.val))
    s_al.on_changed(lambda _v: draw(10 ** s_pv.val, 10 ** s_al.val))
    draw(10 ** s_pv.valinit, 10 ** s_al.valinit)
    plt.show()


# --------------------------------------------------------------------------- #
def run(dataset, prior_var, alpha, top_k, interactive_mode):
    R, shape = load_R(dataset)
    NP_ = R.shape[0]
    print(f"[{dataset}] R {R.shape}, canvas {shape}; building correlation tree …")
    tree = build_tree(R)
    nodes = collect_nodes(tree)
    print(f"[{dataset}] tree: {len(nodes)} nodes, max depth {max(d for d, _ in nodes)}")
    print(f"[{dataset}] resolving basic level per pixel (prior_var={prior_var:g}, alpha={alpha:g}) …")
    depths_px, labels, by_fp = pixel_basics(R, tree, prior_var, alpha)
    print(f"[{dataset}] {len(by_fp)} distinct basic-level nodes over {NP_} pixels; "
          f"mean basic depth {depths_px.mean():.2f}")
    outdir = os.path.join(HERE, f"{dataset}_output", "primitives", "basic_level")
    os.makedirs(outdir, exist_ok=True)
    p1 = plot_epmi_by_depth(nodes, depths_px, tree.root, prior_var, alpha,
                            os.path.join(outdir, "epmi_by_depth.png"), dataset)
    print("  wrote", p1)
    p2 = plot_count_tree(tree, shape,
                         os.path.join(outdir, "dense_concept_tree.png"), dataset)
    print("  wrote", p2)
    p3 = plot_basic_level_regions(by_fp, NP_, shape,
                                  os.path.join(outdir, "basic_level_regions.png"),
                                  dataset, prior_var, alpha, top_k=top_k)
    print("  wrote", p3)
    if interactive_mode:
        print(f"[{dataset}] opening interactive EPMI-by-depth explorer (close window to exit) …")
        interactive(nodes, R, tree, tree.root, prior_var, alpha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Basic-level EPMI analysis of the PoE pixel-correlation hierarchy.")
    ap.add_argument("dataset", nargs="?", default="colormnist",
                    choices=list(SHAPES), help="which dataset's R.npy to analyze")
    ap.add_argument("--prior-var", type=float, default=BASIC_PRIOR_VAR)
    ap.add_argument("--alpha", type=float, default=BASIC_ALPHA)
    ap.add_argument("--top-k", type=int, default=24)
    ap.add_argument("--interactive", action="store_true", help="open the prior_var/alpha slider explorer")
    a = ap.parse_args()
    run(a.dataset, a.prior_var, a.alpha, a.top_k, a.interactive)
