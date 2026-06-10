"""
ColorMNIST — Test-Time Compositional Generalization with Cobweb (Product-of-Experts)
====================================================================================
Cobweb version of Wang, Gupta, Zhu & MacLellan, "Test-Time Compositional Generalization
in Diffusion Models via Concept Discovery" (2026).  The paper turns a pretrained DDPM
into a hierarchy of Gaussian "concept" modes, picks a few that cover an OOD query, and
multiplies them into a Product-of-Experts.  ColorMNIST in the paper is a 32x32 pixel
DDPM (no autoencoder), so we do the same thing on RAW PIXELS with Cobweb as the hierarchy.

Three steps:

  1. CONCEPTS = Cobweb nodes.  Each node n is a diagonal Gaussian N(m_n, σ²_n) with
     m_n = node.mean and σ²_{n,r} = sum_sq_r/count + prior_var.  Shallow nodes are broad
     concepts (backgrounds), deep nodes are specific ones (a digit shape) — the paper's
     coarse-to-fine mode hierarchy, for free.

  2. SELECT concepts with the paper's greedy argmax (Eq. 9) over a candidate POOL gathered from
     the tree.  For each concept, best-first expand the tree by the coverage gain (Eqs. 8-9)
     Δ(n) = Σ_r max(ℓ_{n,r}(x_q) − cur_r, 0) to a pool of ~3% of the nodes, then take the
     global Δ-maximizer.  Folding the pick into cur RE-ROUTES the next concept's pool toward the
     pixels still unexplained — so the background and the digit are found on different branches
     (the Cobweb analog of the paper's modes from different noise levels).  HOW MANY concepts:
     fixed K, or the leftover-image cutoff — stop once μ_T explains ≥99% of the query.

  3. COMPOSE with the paper's product of Gaussians (Eqs. 7 & 10) at temperature τ=0.1:
         w_n(r) = softmax_{n∈S}( ℓ_{n,r}(x_q)/τ )            (per-pixel concept weights)
         μ_T[r] = (Σ_n w_n(r)·m_{n,r}/σ²_{n,r}) / (Σ_n w_n(r)/σ²_{n,r})
     Low τ → each pixel comes mostly from its single best concept.  We have no diffusion
     sampler, so the generated image is this PoE mean μ_T.

Benchmark (paper §4.1): 32x32 RGB, 10 digits × 4 fg × 4 bg = 160 slots, 120 seen / 40
held-out OOD.  Baselines: Top-1 / Top-3 nearest seen-class retrieval.  Metrics: FID, CLIP,
k-NN P/R/F1 vs Faithfulness (queries) and Generalization sets.
"""

import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch, torchvision, torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from cobweb.cobweb_continuous import CobwebContinuousTree

RNG = np.random.default_rng(0)
HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "colormnist_output"); os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(HERE, "mnist_output", "data")            # reuse downloaded MNIST
PRIOR_VAR = 0.05854983152                                        # CobwebContinuousTree default

# ── Hyperparameters ───────────────────────────────────────────────────────────
TAU      = 0.1     # per-coordinate softmax temperature (Eq. 10); low → sharp per-pixel ownership
K_MAX    = 6       # safety cap on concepts per query
EXPLAIN_FRAC = 0.99 # cutoff: keep adding concepts until the composed μ_T explains this fraction of
                    # the query's content (R² vs the query mean) — i.e. until little leftover remains

# ── Palettes (paper §4.1, Table 3) ────────────────────────────────────────────
FG_COLORS = {"yellow":(0.93,0.90,0.20), "green":(0.30,0.80,0.35), "cyan":(0.30,0.82,0.88), "pink":(0.95,0.52,0.78)}
BG_COLORS = {"deepred":(0.42,0.06,0.06), "navy":(0.05,0.10,0.38), "purple":(0.26,0.05,0.36), "brown":(0.30,0.19,0.06)}
FG_NAMES, BG_NAMES = list(FG_COLORS), list(BG_COLORS)
IMG = 32; D = IMG * IMG * 3

# ── Render ColorMNIST ─────────────────────────────────────────────────────────
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
_imgs, _lbls = next(iter(torch.utils.data.DataLoader(trainset, batch_size=20000, shuffle=False)))
GRAY, GLAB = _imgs.squeeze(1).numpy(), _lbls.numpy()
_by_digit = {d: np.where(GLAB == d)[0] for d in range(10)}

def render(g28, fg, bg):
    """Pad a 28×28 gray digit to 32×32 and colorize: pixel = bg + intensity·(fg−bg)."""
    c = np.zeros((IMG, IMG), np.float32); c[2:30, 2:30] = g28
    fg = np.asarray(FG_COLORS[fg], np.float32); bg = np.asarray(BG_COLORS[bg], np.float32)
    return (bg[None, None, :] + c[:, :, None] * (fg - bg)[None, None, :]).reshape(-1)

# ── Compositional 120-seen / 40-OOD split (verified) ──────────────────────────
ALL_SLOTS = [(d, f, b) for d in range(10) for f in FG_NAMES for b in BG_NAMES]
def make_split(n_ood=40, tries=500):
    for _ in range(tries):
        ood = [ALL_SLOTS[int(i)] for i in RNG.permutation(len(ALL_SLOTS))[:n_ood]]; oset = set(ood)
        seen = [s for s in ALL_SLOTS if s not in oset]
        if len({s[0] for s in seen}) < 10 or len({s[1] for s in seen}) < 4 or len({s[2] for s in seen}) < 4:
            continue
        # every OOD slot is a NOVEL combination whose factors are individually seen
        if all(any(s[2] == b for s in seen) and any(s[0] == d and s[1] == f for s in seen)
               and any(s[1] == f and s[2] == b for s in seen) for (d, f, b) in ood):
            return seen, ood
    raise RuntimeError("no valid split")
SEEN_SLOTS, OOD_SLOTS = make_split()
print(f"Split: {len(SEEN_SLOTS)} seen / {len(OOD_SLOTS)} OOD")

N_PER_SEEN, N_QUERY, N_GEN = 80, 16, 16     # per-class generated/faithful and generalization sizes
_cur = {d: 0 for d in range(10)}
def take(d, k): i = _by_digit[d][_cur[d]:_cur[d] + k]; _cur[d] += k; return i

X_train, dig_tr, fg_tr, bg_tr, slot_tr, _tr_gi = [], [], [], [], [], []
for (d, f, b) in SEEN_SLOTS:
    for gi in take(d, N_PER_SEEN):
        X_train.append(render(GRAY[gi], f, b)); _tr_gi.append(int(gi))
        dig_tr.append(d); fg_tr.append(FG_NAMES.index(f)); bg_tr.append(BG_NAMES.index(b)); slot_tr.append((d, f, b))
X_train = np.asarray(X_train, np.float32); dig_tr = np.asarray(dig_tr); fg_tr = np.asarray(fg_tr); bg_tr = np.asarray(bg_tr)
ood_queries, ood_genset, _ood_gi = {}, {}, []
for (d, f, b) in OOD_SLOTS:
    qi = take(d, N_QUERY + N_GEN); _ood_gi.extend(int(g) for g in qi)
    ims = np.asarray([render(GRAY[g], f, b) for g in qi], np.float32)
    ood_queries[(d, f, b)] = ims[:N_QUERY]; ood_genset[(d, f, b)] = ims[N_QUERY:]
print(f"  train {X_train.shape}; OOD {len(OOD_SLOTS)}×{N_QUERY} queries + {N_GEN} gen imgs")

# verify the split is a genuine compositional OOD split (no slot/combination/exemplar leakage)
_seen, _ood = set(SEEN_SLOTS), set(OOD_SLOTS)
assert len(_seen) == 120 and len(_ood) == 40 and _seen.isdisjoint(_ood)
assert set(zip(dig_tr.tolist(), [FG_NAMES[i] for i in fg_tr], [BG_NAMES[i] for i in bg_tr])).isdisjoint(_ood)
assert set(_tr_gi).isdisjoint(set(_ood_gi))
print("  ✓ verified: 120/40 disjoint, no combination or exemplar leakage")

# ── Build the Cobweb tree on raw pixels (= the paper's density-mode hierarchy) ──
print("Building Cobweb tree …")
tree = CobwebContinuousTree(size=D, covar_from=1, num_labels=0); _empty = np.zeros(0, np.float32)
for c, i in enumerate(RNG.permutation(len(X_train))):
    tree.ifit(X_train[i], _empty)
    if (c + 1) % 2000 == 0: print(f"  {c+1}/{len(X_train)}")
print("  built.")

def by_depth(root):
    bd, q = {}, [(root, 0)]
    while q:
        n, d = q.pop(0); bd.setdefault(d, []).append(n)
        for ch in n.children: q.append((ch, d + 1))
    return bd
BD = by_depth(tree.root); print("  nodes/depth:", {d: len(v) for d, v in sorted(BD.items())})
TREE_NODES = sum(len(v) for v in BD.values())
BFS_BUDGET = int(0.03 * TREE_NODES)   # best-first candidate-pool size per concept (3% of tree)
print(f"  tree nodes: {TREE_NODES}  (BFS pool budget = {BFS_BUDGET})")

def _node_depth(n):
    d = 0; p = n.parent
    while p is not None: d += 1; p = p.parent
    return d

# ── A Cobweb node as a clean-space diagonal-Gaussian concept ───────────────────
def node_feats(node, x):
    """(mean m_n, per-dim variance σ²_{n,r}, per-dim log-lik ℓ_{n,r}(x))."""
    m = np.asarray(node.mean, np.float32)
    v = np.asarray(node.sum_sq, np.float32) / np.float32(node.count) + np.float32(PRIOR_VAR)
    ll = -0.5 * np.log(2.0 * np.pi * v) - 0.5 * (x - m) ** 2 / v
    return m, v, ll

def marginal_gain(ll, cur):
    """Submodular coverage gain Δ = Σ_r max(ℓ_{n,r} − cur_r, 0) — credit only where this
    concept beats the running coverage cur (paper Eqs. 8-9).  cur=None → Δ = Σ_r ℓ (singleton)."""
    return float(ll.sum()) if cur is None else float(np.maximum(ll - cur, 0.0).sum())

def absorb(cur, ll):
    """Fold a picked concept into the running coverage: cur_r ← max(cur_r, ℓ_{n,r})."""
    return ll.copy() if cur is None else np.maximum(cur, ll)

def _poe_mu(M, V, L, tau):
    """The PoE mean μ_T (Eqs. 7,10) for stacked concept feats (k,D)."""
    if M.shape[0] == 1: return M[0]
    w = np.exp((L - L.max(0, keepdims=True)) / tau); w /= w.sum(0, keepdims=True)
    prec = (w / V).sum(0)
    return (w * M / V).sum(0) / np.maximum(prec, 1e-12)

# ══ (1) SELECT concepts by a best-first candidate pool (Wang et al. §3.2 on the tree) ══
def select_concepts(x, record=None, fixed_k=None, explain_frac=EXPLAIN_FRAC, max_nodes=BFS_BUDGET, tau=TAU):
    """For each concept (Wang's Eq. 9 argmax over a candidate pool): best-first expand the tree by
    coverage gain  Δ(n) = Σ_r max(ℓ_{n,r}(x_q) − cur_r, 0) (Eqs. 8-9) — pop the max-Δ node, push
    its children — up to a pool of `max_nodes` (~3% of the tree), then take the global Δ-maximizer
    in that pool.  Folding the pick into cur RE-ROUTES the next concept's pool toward the pixels
    still unexplained, so the background and the digit are found on different branches.
    HOW MANY: `fixed_k`, else the leftover-image cutoff — stop once the composed μ_T explains
    ≥ `explain_frac` of the query (R² vs its mean).  `record`: appends nodes visited."""
    import heapq
    cache = {}
    def feats(n):
        e = cache.get(id(n))
        if e is None: e = node_feats(n, x); cache[id(n)] = e
        return e
    cur = None
    def gain(n): return marginal_gain(feats(n)[2], cur)           # Δ given current coverage (residual)
    chosen, gains, picked, Ms, Vs, Ls, explored = [], [], set(), [], [], [], 0
    sstot = float(((x - x.mean()) ** 2).sum()) + 1e-12
    for _ in range(fixed_k or K_MAX):
        pq = [(-gain(tree.root), 0, tree.root)]; tie = 1           # best-first pool, max-Δ first
        best_n, best_g, pop = None, 0.0, 0
        while pq and pop < max_nodes:
            _, _, n = heapq.heappop(pq); pop += 1
            if n is not tree.root and id(n) not in picked:
                g = gain(n)
                if g > best_g: best_n, best_g = n, g
            for c in n.children:
                heapq.heappush(pq, (-gain(c), tie, c)); tie += 1
        explored += pop
        if best_n is None or best_g <= 1e-6: break                # leftover already covered
        picked.add(id(best_n)); m, v, l = feats(best_n)
        chosen.append(best_n); gains.append(best_g); Ms.append(m); Vs.append(v); Ls.append(l)
        cur = absorb(cur, l)                                      # subtract it → re-route next pool
        if fixed_k is None:                                       # leftover-image cutoff
            mu = _poe_mu(np.stack(Ms), np.stack(Vs), np.stack(Ls), tau)
            if 1.0 - float(((x - mu) ** 2).sum()) / sstot >= explain_frac: break
    if record is not None: record.append(explored)
    return [(p, *feats(p), g) for p, g in zip(chosen, gains)]      # (node, mean, var, ll, Δ-gain)

# ══ (2) COMPOSE — per-pixel product of Gaussians (paper Eqs. 7, 10) ══
def poe_compose(x, selector, tau=TAU):
    """Select concepts, then compose them as the paper's PoE.  Returns (μ_T, nodes in recovery
    order, per-pixel weights w, per-concept coverage gain Δ from selection)."""
    sel = selector(x)
    if not sel:
        return x.copy(), [], np.zeros((0, D), np.float32), np.zeros(0, np.float32)
    nodes = [s[0] for s in sel]; gains = np.asarray([s[4] for s in sel], np.float32)
    M = np.stack([s[1] for s in sel]); V = np.stack([s[2] for s in sel]); L = np.stack([s[3] for s in sel])
    if len(sel) == 1:
        return M[0].astype(np.float32), nodes, np.ones((1, D), np.float32), gains
    w = np.exp((L - L.max(0, keepdims=True)) / tau); w /= w.sum(0, keepdims=True)   # Eq. 10
    return _poe_mu(M, V, L, tau).astype(np.float32), nodes, w, gains                # Eq. 7

# ════════════════════════════════════════════════════════════════════════════════
# PRIMITIVE DISCOVERY — treat every PoE donation heatmap (90% variant) as its own image,
# fit ONE Cobweb tree over them, and visualize the hierarchy's top-4-level mean heatmaps.
# Each node's mean is a recurring "primitive" pixel-blob the composition donates.
# ════════════════════════════════════════════════════════════════════════════════
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
PRIM = os.path.join(OUT_DIR, "primitives"); os.makedirs(PRIM, exist_ok=True)
NP = IMG * IMG   # 1024 spatial pixels

print("Gathering PoE donation heatmaps over all OOD queries (90% cutoff) …")
sel90 = lambda y: select_concepts(y, explain_frac=0.90)
MAPS = []                                                    # one (NP,) heatmap per composed concept
for s in OOD_SLOTS:
    for x in ood_queries[s]:
        mu, nodes, w, gains = poe_compose(x, sel90)
        for k in range(len(w)):
            MAPS.append(w[k].reshape(IMG, IMG, 3).mean(2).reshape(NP))   # channel-avg donation heatmap
MAPS = np.ascontiguousarray(MAPS, np.float32)
print(f"  {len(MAPS)} donation heatmaps")

_mag = plt.get_cmap("magma")
SUB = os.path.join(PRIM, "subtrees"); os.makedirs(SUB, exist_ok=True)

# ── Pixel cross-correlation → regions ──────────────────────────────────────────
# Correlate every pixel's donation profile (its value across all heatmaps) with every
# other pixel, then group pixels into regions with Cobweb.  Each region = a set of pixels
# that the PoE tends to donate together — a spatial "primitive".
print("Pixel cross-correlation regions …")
def _nodes_at_depth(root, depth):
    cur = [root]
    for _ in range(depth):
        nxt = [c for n in cur for c in n.children]
        if not nxt: break
        cur = nxt
    return cur
def cobweb_pixel_regions(R, depth):
    t = CobwebContinuousTree(size=R.shape[1], covar_from=1, num_labels=0)
    for i in RNG.permutation(len(R)): t.ifit(np.ascontiguousarray(R[i]), _empty)
    nodes = _nodes_at_depth(t.root, depth)
    if len(nodes) < 2: nodes = list(t.root.children)
    if len(nodes) < 2: return np.zeros(len(R), int), 1
    lab = np.array([int(np.argmax([nd.log_prob(np.ascontiguousarray(r), _empty) for nd in nodes])) for r in R])
    uniq = sorted(set(lab.tolist())); remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[l] for l in lab]), len(uniq)

R = np.corrcoef(MAPS.T.astype(np.float64))                   # (NP,NP) pixel cross-correlation
R = np.nan_to_num(R, nan=0.0).astype(np.float32)             # constant pixels → 0 correlation
from matplotlib.colors import ListedColormap
_others = [plt.get_cmap("tab20")(i) for i in range(20) if i not in (0, 1)] + list(plt.get_cmap("tab20b").colors)
_PAL = ListedColormap([(0.23, 0.43, 0.69)] + _others)        # index 0 = background blue, rest distinct
def _bg_first(lab):                                          # relabel so the corner (background) cluster = 0
    bg = lab[0]; remap = {bg: 0}; nxt = 1
    for l in sorted(set(lab.tolist())):
        if l != bg: remap[l] = nxt; nxt += 1
    return np.array([remap[l] for l in lab])
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
order = None
for ax, depth in [(axes[0], 1), (axes[1], 2), (axes[2], 3), (axes[3], 4)]:
    lab, nc = cobweb_pixel_regions(R, depth); lab = _bg_first(lab)
    if depth == 2: order = np.argsort(lab)                   # reorder corr matrix by depth-2 regions (next panel)
    ax.imshow(lab.reshape(IMG, IMG), cmap=_PAL, vmin=0, vmax=len(_PAL.colors) - 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Cobweb depth-{depth}\n{nc} regions", fontsize=10)
fig.suptitle("PoE pixel primitives (90%) — pixels grouped by donation cross-correlation (a region = pixels donated together)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(os.path.join(PRIM, "pixel_correlation_regions.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  pixel-correlation regions → {os.path.join(PRIM, 'pixel_correlation_regions.png')}")

# the cross-correlation matrix itself, reordered by region for visible block structure
fig, ax = plt.subplots(figsize=(5, 4.6))
im = ax.imshow(R[np.ix_(order, order)], cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Pixel donation cross-correlation\n(pixels ordered by region)", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
plt.tight_layout(); plt.savefig(os.path.join(PRIM, "pixel_correlation_matrix.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  cross-correlation matrix → {os.path.join(PRIM, 'pixel_correlation_matrix.png')}")

# ── hierarchy of the cross-correlation: each node = the region of pixels routed to it ──
print("Building Cobweb hierarchy over pixel cross-correlation …")
rtree = CobwebContinuousTree(size=NP, covar_from=1, num_labels=0)
for i in RNG.permutation(NP): rtree.ifit(np.ascontiguousarray(R[i]), _empty)

def render_region_hierarchy(root, out_path, title, max_depth=4, top_children=6, root_idxs=None):
    _disp = {}
    def dch(n, d):
        if d >= max_depth: return []
        if id(n) not in _disp: _disp[id(n)] = sorted(n.children, key=lambda c: c.count, reverse=True)[:top_children]
        return _disp[id(n)]
    member = {}; deepest = []                                # id(node) -> list of pixel indices routed to it
    def assign_members(n, d, idxs):
        member[id(n)] = idxs; ch = dch(n, d)
        if not ch or not idxs: return
        lp = np.array([[c.log_prob(np.ascontiguousarray(R[p]), _empty) for c in ch] for p in idxs])
        best = lp.argmax(1)
        for ci, c in enumerate(ch): assign_members(c, d + 1, [idxs[j] for j in range(len(idxs)) if best[j] == ci])
    assign_members(root, 0, list(range(NP)) if root_idxs is None else list(root_idxs))
    def span(n, d):
        ch = dch(n, d); return 1 if not ch else sum(span(c, d + 1) for c in ch)
    pos = {}
    def assign(n, d, xl):
        sp = span(n, d); pos[id(n)] = (xl + sp / 2.0, d); cur = xl
        if d == max_depth: deepest.append((n, member.get(id(n), [])))
        for c in dch(n, d): assign(c, d + 1, cur); cur += span(c, d + 1)
        return sp
    tw = assign(root, 0, 0.0)
    fig, ax = plt.subplots(figsize=(max(12, tw * 1.6), (max_depth + 1) * 2.6))
    ax.set_xlim(-0.6, tw + 0.6); ax.set_ylim(-0.8, max_depth + 0.8); ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=12)
    def de(n, d):
        x0, _ = pos[id(n)]
        for c in dch(n, d):
            xc, dc = pos[id(c)]; ax.plot([x0, xc], [d + 0.34, dc - 0.34], color="gray", lw=0.8, zorder=0); de(c, d + 1)
    de(root, 0)
    def dn(n, d):
        x0, _ = pos[id(n)]; mask = np.zeros(NP, np.float32); mask[member.get(id(n), [])] = 1.0
        img = _mag(mask.reshape(IMG, IMG))[..., :3]
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=1.8), (x0, d), frameon=True, bboxprops=dict(edgecolor="black", lw=1.2)))
        ax.text(x0, d + 0.42, f"{len(member.get(id(n), []))}px", ha="center", va="top", fontsize=6)
        for c in dch(n, d): dn(c, d + 1)
    dn(root, 0)
    plt.tight_layout(); plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()
    return deepest

level = render_region_hierarchy(rtree.root, os.path.join(PRIM, "correlation_hierarchy.png"),
                        "PoE pixel-correlation hierarchy (90%) — each node = the region of pixels routed to it "
                        "(bright = pixels in the region), top 5 levels", max_depth=4)
print(f"  correlation hierarchy → {os.path.join(PRIM, 'correlation_hierarchy.png')}")

# subtrees: 3 levels deep rooted at each leaf region of the correlation hierarchy
_nsub = 0
for i, (node, idxs) in enumerate(sorted(level, key=lambda t: -len(t[1]))):
    if not node.children or not idxs: continue
    render_region_hierarchy(node, os.path.join(SUB, f"subtree_{i:02d}.png"),
                            f"Pixel-correlation subtree (3 levels) — root region {len(idxs)}px", max_depth=3, root_idxs=idxs)
    _nsub += 1
print(f"  {_nsub} subtrees → {SUB}/")
print("Done.")

