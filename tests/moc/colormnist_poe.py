"""
ColorMNIST — Test-Time Compositional Generalization with Cobweb-PoE
===================================================================
Replication, in the Cobweb framework, of the ColorMNIST experiment from
Wang, Gupta, Zhu & MacLellan, "Test-Time Compositional Generalization in
Diffusion Models via Concept Discovery" (2026).

The paper repurposes a pretrained diffusion model as a *hierarchy of density
modes*: for a single out-of-distribution (OOD) query depicting a held-out
combination of primitive factors, it discovers reusable concept prototypes,
greedily selects relevant ones with a submodular coverage objective, and
composes their local Gaussians into a Product-of-Experts (PoE) teacher q_T.

Here Cobweb *is* that hierarchy of Gaussian density modes — every node n is a
diagonal-Gaussian expert q_n(x)=N(m_n, Σ_n) with m_n=node.mean,
σ²_{n,r}=sum_sq_r/count + prior_var (verified to match node.log_prob).  We
therefore run the paper's concept-discovery + PoE composition *directly on the
Cobweb tree*, with no diffusion model, mode-ascent, or LoRA.

Benchmark (matches the paper):
    digit identity (10) × digit color (4) × background color (4) = 160 slots
    120 slots SEEN by Cobweb during training, 40 held out as OOD.
For each OOD query (a held-out (digit, fg, bg) combination, whose individual
factors are seen in *other* combinations) the question is whether composing
discovered concepts recovers the unseen combination better than retrieving /
compositing the nearest seen prototypes.

Why this fixes the "weird concepts" seen on plain MNIST: ~80% of MNIST pixels
are zero background, so a thin low-variance prototype "covers" the background
and wins selection regardless of the query.  In ColorMNIST the background is a
solid *color* filling most pixels, so the dominant concept captures the
background-color factor and the next captures the digit-shape+color factor — a
genuine compositional decomposition.

Metrics: the diffusion-specific FID/CLIP/precision-recall (which need a trained
sampler + LoRA) are adapted to what Cobweb's analytic teacher exposes — the PoE
mean μ_T (Eq. 7) is the generated image, scored by (i) attribute classifiers
trained on seen data [does the composition produce the right held-out (digit,
fg, bg)?] and (ii) feature distance to a *faithfulness* set (the query) vs a
*generalization* set (other held-out images of the same OOD class).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from cobweb.cobweb_continuous import CobwebContinuousTree

RNG = np.random.default_rng(0)

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "colormnist_output")
DATA_DIR = os.path.join(HERE, "mnist_output", "data")   # reuse already-downloaded MNIST
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Palettes (RGB in [0,1]) ───────────────────────────────────────────────────
# 4 muted high-contrast digit colors and 4 dark background colors (paper §4.1).
FG_COLORS = {
    "yellow": (0.93, 0.90, 0.20),
    "green":  (0.30, 0.80, 0.35),
    "cyan":   (0.30, 0.82, 0.88),
    "pink":   (0.95, 0.52, 0.78),
}
BG_COLORS = {
    "deepred": (0.42, 0.06, 0.06),
    "navy":    (0.05, 0.10, 0.38),
    "purple":  (0.26, 0.05, 0.36),
    "brown":   (0.30, 0.19, 0.06),
}
FG_NAMES = list(FG_COLORS)
BG_NAMES = list(BG_COLORS)
IMG = 32                      # 32×32 RGB
D   = IMG * IMG * 3           # 3072-dim flat vector (HWC)

# ── Load grayscale MNIST and render ColorMNIST ─────────────────────────────────

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)


def mnist_arrays(dataset, n):
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    imgs, lbls = next(iter(loader))
    return imgs.squeeze(1).numpy(), lbls.numpy()       # (n,28,28) in [0,1], (n,)


GRAY, GLAB = mnist_arrays(trainset, 20_000)
# index of grayscale exemplars per digit
_by_digit = {dd: np.where(GLAB == dd)[0] for dd in range(10)}


def render(gray28, fg, bg):
    """Pad a 28×28 gray digit to 32×32 and colorize: pixel = bg + intensity·(fg−bg)."""
    canvas = np.zeros((IMG, IMG), dtype=np.float32)
    canvas[2:30, 2:30] = gray28
    fg = np.asarray(FG_COLORS[fg], dtype=np.float32)
    bg = np.asarray(BG_COLORS[bg], dtype=np.float32)
    img = bg[None, None, :] + canvas[:, :, None] * (fg - bg)[None, None, :]
    return img.reshape(-1)                              # (3072,) HWC


# ── Compositional slots and the 120/40 seen/OOD split ─────────────────────────
ALL_SLOTS = [(dd, f, b) for dd in range(10) for f in FG_NAMES for b in BG_NAMES]   # 160
assert len(ALL_SLOTS) == 160

# Hold out 40 slots such that every factor value still appears in the seen split
# and each held-out slot's factors are individually seen in other combinations.
def make_split(n_ood=40, tries=500):
    for _ in range(tries):
        ood_idx = RNG.permutation(len(ALL_SLOTS))[:n_ood]
        ood_slots = [ALL_SLOTS[int(i)] for i in ood_idx]
        ood_set = set(ood_slots)
        seen_slots = [s for s in ALL_SLOTS if s not in ood_set]
        digits = {s[0] for s in seen_slots}; fgs = {s[1] for s in seen_slots}; bgs = {s[2] for s in seen_slots}
        if len(digits) < 10 or len(fgs) < 4 or len(bgs) < 4:
            continue
        # each OOD slot's factors must be recoverable from *other* seen combinations:
        # some seen slot shares its bg, some shares its (digit,fg), some shares its (fg,bg)
        if all(any(s[2] == b for s in seen_slots)
               and any(s[0] == dd and s[1] == f for s in seen_slots)
               and any(s[1] == f and s[2] == b for s in seen_slots)
               for (dd, f, b) in ood_slots):
            return seen_slots, ood_slots
    raise RuntimeError("could not find a valid split")


SEEN_SLOTS, OOD_SLOTS = make_split()
print(f"Split: {len(SEEN_SLOTS)} seen slots, {len(OOD_SLOTS)} OOD slots")

# ── Build train / query / generalization image sets ───────────────────────────
N_PER_SEEN  = 80     # training images per seen slot
N_QUERY     = 12     # OOD query images per held-out slot (drive concept discovery)
N_GENSET    = 25     # other held-out images per OOD slot (generalization reference)

print("Rendering ColorMNIST …")
_cursor = {dd: 0 for dd in range(10)}
def take(dd, k):
    idx = _by_digit[dd][_cursor[dd]:_cursor[dd] + k]
    _cursor[dd] += k
    return idx

X_train, dig_tr, fg_tr, bg_tr = [], [], [], []
_train_gi = []                                              # grayscale exemplar indices used in training
for (dd, f, b) in SEEN_SLOTS:
    for gi in take(dd, N_PER_SEEN):
        X_train.append(render(GRAY[gi], f, b)); _train_gi.append(int(gi))
        dig_tr.append(dd); fg_tr.append(FG_NAMES.index(f)); bg_tr.append(BG_NAMES.index(b))
X_train = np.asarray(X_train, dtype=np.float32)
dig_tr = np.asarray(dig_tr); fg_tr = np.asarray(fg_tr); bg_tr = np.asarray(bg_tr)
print(f"  train images: {X_train.shape}")

# OOD queries + per-slot generalization sets
ood_queries, ood_genset = {}, {}
_ood_gi = []                                                # grayscale exemplar indices used for OOD
for (dd, f, b) in OOD_SLOTS:
    qi = take(dd, N_QUERY + N_GENSET)
    _ood_gi.extend(int(g) for g in qi)
    qimgs = np.asarray([render(GRAY[g], f, b) for g in qi], dtype=np.float32)
    ood_queries[(dd, f, b)] = qimgs[:N_QUERY]
    ood_genset[(dd, f, b)]  = qimgs[N_QUERY:]
print(f"  OOD: {len(OOD_SLOTS)} slots × {N_QUERY} queries + {N_GENSET} generalization imgs")

# ── Verify the split is a genuine *compositional* OOD split ───────────────────
# We require, and now assert, that:
#   (1) seen/OOD slot sets are disjoint and sized 120/40 (no slot leakage);
#   (2) no (digit,fg,bg) TRIPLE in the training images coincides with any OOD slot
#       (Cobweb literally never sees an OOD combination);
#   (3) every OOD slot is a NOVEL COMBINATION of factors that ARE individually seen
#       — the OOD digit, fg-color and bg-color each appear in training, just never
#       together — which is exactly what makes composition (not memorisation) the
#       only way to recover it (paper §4.1: held-out compositions of seen primitives).
print("Verifying compositional OOD split …")
_seen_set, _ood_set = set(SEEN_SLOTS), set(OOD_SLOTS)
assert len(_seen_set) == 120 and len(_ood_set) == 40
assert _seen_set.isdisjoint(_ood_set), "seen/OOD slot leakage!"
_train_triples = set(zip(dig_tr.tolist(),
                         [FG_NAMES[i] for i in fg_tr],
                         [BG_NAMES[i] for i in bg_tr]))
assert _train_triples == _seen_set, "training images contain a non-seen combination!"
assert _train_triples.isdisjoint(_ood_set), "an OOD combination leaked into training!"
_seen_digits = {s[0] for s in SEEN_SLOTS}; _seen_fg = {s[1] for s in SEEN_SLOTS}; _seen_bg = {s[2] for s in SEEN_SLOTS}
_novel = 0
for (dd, f, b) in OOD_SLOTS:
    assert (dd, f, b) not in _train_triples                       # combination unseen
    assert dd in _seen_digits and f in _seen_fg and b in _seen_bg  # each factor seen
    assert (dd, f, b) not in _seen_set                             # truly held out
    _novel += 1
print(f"  ✓ 120 seen / 40 OOD disjoint; all {_novel} OOD slots are novel combinations "
      f"of seen factors (digits seen: {len(_seen_digits)}/10, fg: {len(_seen_fg)}/4, bg: {len(_seen_bg)}/4)")
print(f"  ✓ Cobweb training set contains exactly the 120 seen combinations, none of the 40 OOD")
#   (4) no grayscale MNIST exemplar is shared between train and OOD — so OOD digit
#       *shapes* are unseen instances too (only the digit identity is shared).
assert len(set(_train_gi)) == len(_train_gi), "a grayscale exemplar was reused within training!"
assert set(_train_gi).isdisjoint(set(_ood_gi)), "a grayscale exemplar is shared between train and OOD!"
print(f"  ✓ {len(_train_gi)} train + {len(_ood_gi)} OOD grayscale exemplars, all distinct (no shape leakage)")

# ── Train Cobweb on the seen images (unsupervised, images only) ───────────────
PRIOR_VAR = 0.05854983152
print("Building Cobweb tree …")
tree = CobwebContinuousTree(size=D, covar_from=1, num_labels=0)
_empty = np.zeros(0, dtype=np.float32)
_order = RNG.permutation(len(X_train))
for c, i in enumerate(_order):
    tree.ifit(X_train[i], _empty)
    if (c + 1) % 2000 == 0:
        print(f"  {c+1}/{len(X_train)} inserted")
print("  tree built.")

# ── PoE prototype pool: Cobweb nodes across depths (hierarchy of density modes) ─
POE_DEPTHS, POE_MIN_COUNT, POE_CAP = (1, 2, 3, 4, 5, 6), 15, 400


def by_depth(root):
    bd, q = {}, [(root, 0)]
    while q:
        node, d = q.pop(0)
        bd.setdefault(d, []).append(node)
        for ch in node.children:
            q.append((ch, d + 1))
    return bd


BD = by_depth(tree.root)
print("  nodes/depth:", {d: len(v) for d, v in sorted(BD.items())})
cand = [(d, n) for d in POE_DEPTHS for n in BD.get(d, []) if n.count >= POE_MIN_COUNT]
cand.sort(key=lambda dn: dn[1].count, reverse=True)
cand = cand[:POE_CAP]
poe_depths = np.asarray([d for d, _ in cand])
poe_nodes  = [n for _, n in cand]
P = len(poe_nodes)
print(f"  prototype pool: {P} nodes; per-depth {[int((poe_depths==d).sum()) for d in POE_DEPTHS]}")

proto_mean = np.stack([np.asarray(n.mean,   dtype=np.float32) for n in poe_nodes])
proto_var  = np.stack([np.asarray(n.sum_sq, dtype=np.float32) / np.float32(n.count)
                       for n in poe_nodes]) + np.float32(PRIOR_VAR)
proto_logc = (-0.5 * np.log(2.0 * np.pi * proto_var)).astype(np.float32)
proto_iv   = (0.5 / proto_var).astype(np.float32)

POE_K   = 6      # concept prototypes composed per query
POE_TAU = 1.0


def poe_compose(x, k=POE_K, tau=POE_TAU):
    """Concept discovery + Product-of-Experts composition for ONE query x,
    exactly as in the paper (§3.1-3.2, Eqs. 8-10 for selection, Eq. 7 for PoE).

    Selection (greedy submodular coverage F(S)=Σ_r max_{n∈S} ℓ_{n,r}):
      S_1 = {argmax_n F({n})}                       # best singleton (Eq. 8)
      S_{i+1} = S_i ∪ {argmax_n ∆_n(S_i)}           # max marginal gain (Eq. 9)
    PoE composition with per-dim weights w_n(r)=softmax_{n∈S}(ℓ_{n,r}/τ) (Eq. 10):
      Σ_T⁻¹ = Σ_n Diag(w_n) Σ_n⁻¹,  μ_T = Σ_T Σ_n Diag(w_n) Σ_n⁻¹ m_n   (Eq. 7)

    Returns (mu_T, var_T, sel, gains): PoE mean image, its diagonal variance, the
    selected prototype indices (selection order), and their coverage values
    (gains[0] = singleton coverage F({n_1}); gains[i>0] = marginal gain ∆_n)."""
    L = proto_logc - proto_iv * (x[None, :] - proto_mean) ** 2        # (P, d) per-dim ℓ_{n,r}
    sing = L.sum(axis=1)                                              # F({n}) for every prototype
    j0 = int(sing.argmax())                                          # best singleton initialises S_1
    chosen = np.zeros(P, dtype=bool); chosen[j0] = True
    sel, gains = [j0], [float(sing[j0])]
    cur = L[j0].copy()                                               # running per-dim coverage
    for _ in range(k - 1):
        g = np.maximum(L - cur[None, :], 0.0).sum(axis=1)            # marginal gain ∆_n (Eq. 9)
        g[chosen] = -np.inf
        j = int(g.argmax())
        sel.append(j); gains.append(float(g[j])); chosen[j] = True
        cur = np.maximum(cur, L[j])
    sel = np.asarray(sel)
    # per-dim PoE weights softmax(ℓ/τ) over the selected set (Eq. 10)
    Lsel = L[sel] / tau
    w = np.exp(Lsel - Lsel.max(axis=0, keepdims=True))
    w /= w.sum(axis=0, keepdims=True)                                  # (k, d)
    iv_sel = proto_iv[sel]
    prec = (w * iv_sel).sum(axis=0)                                    # Σ w·Σ⁻¹  (PoE precision, Eq. 7)
    mu   = (w * iv_sel * proto_mean[sel]).sum(axis=0) / prec           # μ_T
    var  = 1.0 / np.maximum(prec, 1e-6)
    return mu.astype(np.float32), var.astype(np.float32), sel, np.asarray(gains)


def teacher_top(x, k):
    """Top-k seen-prototype baseline: rank prototypes by singleton log-lik F({n})
    = Σ_r ℓ_{n,r}(x), compose the best k via softmax(−loss) PoE (paper §4.2).
    k=1 → nearest seen prototype; k=3 → top-3 composition."""
    sing = (proto_logc - proto_iv * (x[None, :] - proto_mean) ** 2).sum(axis=1)   # (P,)
    sel = np.argsort(-sing)[:k]
    if k == 1:
        return proto_mean[sel[0]].copy(), sel
    w = np.exp(sing[sel] - sing[sel].max()); w /= w.sum()              # weights ∝ likelihood
    iv_sel = proto_iv[sel]
    prec = (w[:, None] * iv_sel).sum(axis=0)
    mu = (w[:, None] * iv_sel * proto_mean[sel]).sum(axis=0) / prec
    return mu.astype(np.float32), sel


# ── PoE-Search: guided best-first walk through the tree, accruing evidence ─────
# Instead of a pre-collected flat pool (poe_compose), this mirrors tree.predict's
# best-first expansion but uses the *marginal coverage gain* as the search
# heuristic, and stops adaptively once a candidate adds little new evidence:
#   frontier = children of already-accrued nodes (so the search descends the tree)
#   at each step accrue argmax_n ∆_n = Σ_r max(ℓ_{n,r} − cov_r, 0), expand its kids
#   stop when the best marginal gain < PSEARCH_EVID · (first gain), or max reached
# The accrued nodes are then merged by the same per-dim PoE (Eq. 7/10).  Unlike the
# fixed-depth pool, this can follow a promising branch to ANY depth and uses an
# adaptive, query-dependent number of concepts.
PSEARCH_MAX  = 16     # cap on accrued concepts
PSEARCH_EVID = 0.05   # stop when next gain < this fraction of the first gain
_node_depth_cache = {}

def _node_depth(n):
    d = _node_depth_cache.get(id(n))
    if d is None:
        d, p = 0, n.parent
        while p is not None:
            d += 1; p = p.parent
        _node_depth_cache[id(n)] = d
    return d

def poe_search(x, max_nodes=PSEARCH_MAX, evid_frac=PSEARCH_EVID, tau=POE_TAU):
    """Best-first marginal-gain search over the live tree (à la tree.predict);
    returns (mu_T, accrued_nodes, gains, depths)."""
    cache = {}
    def ell(node):
        e = cache.get(id(node))
        if e is None:
            mean = np.asarray(node.mean, dtype=np.float32)
            var  = np.asarray(node.sum_sq, dtype=np.float32) / np.float32(node.count) + np.float32(PRIOR_VAR)
            e = (-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mean) ** 2 / var, var, mean)
            cache[id(node)] = e
        return e
    root = tree.root
    cov = ell(root)[0].copy()                       # seed coverage with the root expert
    frontier = list(root.children)
    sel, gains, first = [], [], None
    while frontier and len(sel) < max_nodes:
        gs = [float(np.maximum(ell(n)[0] - cov, 0.0).sum()) for n in frontier]
        bi = int(np.argmax(gs)); best, gain = frontier[bi], gs[bi]
        if first is None:
            first = gain if gain > 0 else 1.0
        if sel and gain < evid_frac * first:        # enough evidence accrued → stop
            break
        sel.append(best); gains.append(gain)
        cov = np.maximum(cov, ell(best)[0])
        frontier.pop(bi); frontier.extend(best.children)   # expand chosen node into the tree
    if not sel:                                     # degenerate fallback
        sel = [max(root.children, key=lambda n: float(ell(n)[0].sum()))]; gains = [0.0]
    # merge accrued experts via per-dim PoE (Eq. 7/10)
    Lsel  = np.stack([ell(n)[0] for n in sel]) / tau
    w     = np.exp(Lsel - Lsel.max(axis=0, keepdims=True)); w /= w.sum(axis=0, keepdims=True)
    prec  = np.stack([1.0 / ell(n)[1] for n in sel])
    means = np.stack([ell(n)[2] for n in sel])
    denom = (w * prec).sum(axis=0)
    mu    = (w * prec * means).sum(axis=0) / denom
    return mu.astype(np.float32), sel, gains, [_node_depth(n) for n in sel]


# ── Attribute classifiers (trained on seen images) ────────────────────────────
# Adapted metric: do generated images carry the correct held-out (digit, fg, bg)?
print("Training attribute classifiers on seen images …")
clf_digit = LinearSVC(max_iter=3000).fit(X_train, dig_tr)
clf_fg    = KNeighborsClassifier(n_neighbors=5).fit(X_train, fg_tr)
clf_bg    = KNeighborsClassifier(n_neighbors=5).fit(X_train, bg_tr)
print(f"  train attr acc — digit {clf_digit.score(X_train, dig_tr):.3f} "
      f"fg {clf_fg.score(X_train, fg_tr):.3f} bg {clf_bg.score(X_train, bg_tr):.3f}")


# ── Evaluate every method on every OOD query ──────────────────────────────────
# Our method (PoE) vs the paper's prototype baselines (§4.2). Every teacher here is
# built from discovered/seen concept prototypes — there is no query-only teacher.
#   Top-1:      nearest seen prototype (highest singleton log-lik) — pure retrieval.
#   Top-3:      PoE of the 3 best singletons — multi-concept composition baseline.
#   PoE:        greedy submodular selection of K=6 pool prototypes + per-dim PoE.
#   PoE-Search: best-first marginal-gain walk through the live tree, adaptive stop.
#   Query-only: single-query Gaussian teacher N(x_q, σ²I) — its mean is the query
#               itself (memorisation baseline; faithful but cannot generalise).
METHODS = ["PoE", "PoE-Search", "Top-1", "Top-3", "Query-only"]


def generate(method, x):
    if method == "PoE":        return poe_compose(x)[0]
    if method == "PoE-Search": return poe_search(x)[0]
    if method == "Top-1":      return teacher_top(x, 1)[0]
    if method == "Top-3":      return teacher_top(x, 3)[0]
    if method == "Query-only": return x.copy()
    raise ValueError(method)


print("Evaluating compositional generalization on OOD queries …")
# generate every (method, query) image, then batch-classify attributes
gen_imgs = {m: [] for m in METHODS}
true_dig, true_fg, true_bg, faith, gen_dist = [], [], [], {m: [] for m in METHODS}, {m: [] for m in METHODS}
poe_sel_depths, search_depths, search_ncounts = [], [], []   # depth/size logs for analysis
for (dd, f, b) in OOD_SLOTS:
    fg_i, bg_i = FG_NAMES.index(f), BG_NAMES.index(b)
    gcen = ood_genset[(dd, f, b)].mean(axis=0)            # generalization-set centroid
    for x in ood_queries[(dd, f, b)]:
        true_dig.append(dd); true_fg.append(fg_i); true_bg.append(bg_i)
        poe_mu, _, poe_sel_x, _ = poe_compose(x)          # compute PoE once, reuse + log depths
        poe_sel_depths.extend(int(poe_depths[j]) for j in poe_sel_x)
        s_mu, s_sel, _, s_depths = poe_search(x)          # compute PoE-Search once, reuse + log
        search_depths.extend(s_depths); search_ncounts.append(len(s_sel))
        precomputed = {"PoE": poe_mu, "PoE-Search": s_mu}
        for m in METHODS:
            g = precomputed.get(m) if m in precomputed else generate(m, x)
            gen_imgs[m].append(g)
            faith[m].append(float(np.linalg.norm(g - x)))
            gen_dist[m].append(float(np.linalg.norm(g - gcen)))
true_dig = np.asarray(true_dig); true_fg = np.asarray(true_fg); true_bg = np.asarray(true_bg)
poe_sel_depths = np.asarray(poe_sel_depths); search_depths = np.asarray(search_depths)

print(f"\n  {'Method':<12} {'Digit':>7} {'Fg':>7} {'Bg':>7} {'Joint':>7} {'Faith↓':>8} {'Gen↓':>8}")
print("  " + "-" * 62)
rows = []
for m in METHODS:
    G = np.asarray(gen_imgs[m], dtype=np.float32)
    pd, pf, pb = clf_digit.predict(G), clf_fg.predict(G), clf_bg.predict(G)
    ok = (pd == true_dig); okf = (pf == true_fg); okb = (pb == true_bg)
    r = dict(method=m,
             digit=100 * ok.mean(), fg=100 * okf.mean(), bg=100 * okb.mean(),
             joint=100 * (ok & okf & okb).mean(),
             faith=float(np.mean(faith[m])), gen=float(np.mean(gen_dist[m])))
    rows.append(r)
    print(f"  {m:<12} {r['digit']:>6.1f}% {r['fg']:>6.1f}% {r['bg']:>6.1f}% "
          f"{r['joint']:>6.1f}% {r['faith']:>8.2f} {r['gen']:>8.2f}")

with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=["method", "digit", "fg", "bg", "joint", "faith", "gen"])
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
print(f"  summary → {os.path.join(OUT_DIR, 'summary.csv')}")

# ── Concept depth vs. tree depth ──────────────────────────────────────────────
# How deep in the hierarchy are the concepts PoE actually merges, vs how deep the
# tree goes?  The candidate pool is drawn from depths POE_DEPTHS (shallow, well-
# supported modes); PoE then merges a few of those per query.  The full tree is
# much deeper (its leaves are near-singletons), but those fine modes are never
# sampled — composition uses coarse/mid-level concepts.
TREE_MAX_DEPTH = max(BD.keys())
print("\nConcept depth vs. tree depth:")
print(f"  tree max depth                 : {TREE_MAX_DEPTH}")
print(f"  candidate pool depths          : {int(poe_depths.min())}–{int(poe_depths.max())} "
      f"(per-depth counts {[int((poe_depths==d).sum()) for d in POE_DEPTHS]})")
print(f"  merged-concept depth (PoE pool): min {int(poe_sel_depths.min())}, "
      f"median {int(np.median(poe_sel_depths))}, mean {poe_sel_depths.mean():.2f}, "
      f"max {int(poe_sel_depths.max())}")
print(f"  merged-concept depth (PoE-Srch): min {int(search_depths.min())}, "
      f"median {int(np.median(search_depths))}, mean {search_depths.mean():.2f}, "
      f"max {int(search_depths.max())}")
_sc = np.asarray(search_ncounts)
print(f"  PoE-Search accrued #concepts   : min {_sc.min()}, median {int(np.median(_sc))}, "
      f"mean {_sc.mean():.2f}, max {_sc.max()}  (adaptive; PoE pool fixed at K={POE_K})")

fig, ax = plt.subplots(figsize=(10, 4.4))
bins = np.arange(0, TREE_MAX_DEPTH + 2) - 0.5
ax.hist(poe_depths, bins=bins, alpha=0.45, label=f"candidate pool ({P})", color="#999999",
        weights=np.full(len(poe_depths), 1.0), density=True)
ax.hist(poe_sel_depths, bins=bins, alpha=0.6, label="merged by PoE (pool, K=6)", color="#e377c2", density=True)
ax.hist(search_depths, bins=bins, alpha=0.6, label="merged by PoE-Search (adaptive)", color="#1f9e54", density=True)
ax.axvline(TREE_MAX_DEPTH, color="red", ls="--", lw=1.5, label=f"tree max depth = {TREE_MAX_DEPTH}")
ax.set_xlabel("node depth in hierarchy (0 = root)"); ax.set_ylabel("density of selections")
ax.set_title("Depth of concepts sampled for composition vs. tree depth\n"
             "(pool PoE is capped at depths ≤6; tree-guided PoE-Search follows branches deeper)")
ax.set_xticks(range(0, TREE_MAX_DEPTH + 1)); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "concept_depths.png"), dpi=130)
plt.close()
print(f"  concept-depth figure → {os.path.join(OUT_DIR, 'concept_depths.png')}")

# ── Shared drawing helper + the OOD slots shown across all galleries ──────────
def show(ax, vec, title=None):
    ax.imshow(np.clip(vec.reshape(IMG, IMG, 3), 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


N_SHOW = min(10, len(OOD_SLOTS))
show_slots = [OOD_SLOTS[i] for i in np.linspace(0, len(OOD_SLOTS) - 1, N_SHOW).astype(int)]

# (1) PoE concepts graphic: query · PoE μ_T · the K discovered concepts (depth·Δ).
NCc = 2 + POE_K
fig, axes = plt.subplots(N_SHOW, NCc, figsize=(NCc * 1.45, N_SHOW * 1.55))
fig.suptitle(
    "PoE on ColorMNIST — greedy submodular pool composition\n"
    f"query · PoE μ_T · top-{POE_K} discovered concepts (mean-image; d = tree depth, Δ = coverage gain)",
    fontsize=10)
for row, (dd, f, b) in enumerate(show_slots):
    x = ood_queries[(dd, f, b)][0]
    mu, var, sel, gains = poe_compose(x)
    order = np.argsort(-gains)
    show(axes[row, 0], x); axes[row, 0].set_ylabel(f"{dd}/{f}/{b}", fontsize=6)
    if row == 0: axes[row, 0].set_title("query", fontsize=7)
    show(axes[row, 1], mu, "PoE μ_T" if row == 0 else None)
    for c in range(POE_K):
        j = int(sel[order[c]]); dep = int(poe_depths[j])
        show(axes[row, 2 + c], proto_mean[j], f"d{dep}·Δ{gains[order[c]]:.0f}")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "poe_default_concepts.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  PoE concepts graphic → {os.path.join(OUT_DIR, 'poe_default_concepts.png')}")

# ── PoE-Search graphic: query · μ_T · the top-6 nodes the search collected ─────
# Dedicated view of the tree-guided variant.  For each OOD query we show the
# PoE-Search reconstruction and the top-6 accrued nodes (by marginal gain), each
# drawn as its mean-image and labelled with its tree depth d and gain Δ — making
# explicit which concepts the best-first walk pulled in and how deep they sit.
PSEARCH_SHOW = 6
NCp = 2 + PSEARCH_SHOW
fig, axes = plt.subplots(N_SHOW, NCp, figsize=(NCp * 1.45, N_SHOW * 1.55))
fig.suptitle(
    "PoE-Search on ColorMNIST — guided best-first walk through the tree\n"
    f"query · PoE-Search μ_T · top-{PSEARCH_SHOW} accrued nodes (mean-image; d = tree depth, Δ = marginal gain)",
    fontsize=10)
for row, (dd, f, b) in enumerate(show_slots):
    x = ood_queries[(dd, f, b)][0]
    s_mu, s_sel, s_gains, s_depths = poe_search(x)
    order = np.argsort(-np.asarray(s_gains))
    show(axes[row, 0], x); axes[row, 0].set_ylabel(f"{dd}/{f}/{b}", fontsize=6)
    if row == 0: axes[row, 0].set_title("query", fontsize=7)
    show(axes[row, 1], s_mu, f"PoE-Search μ_T\n({len(s_sel)} nodes accrued)" if row == 0 else None)
    for c in range(PSEARCH_SHOW):
        ax = axes[row, 2 + c]
        if c < len(order):
            j = order[c]
            show(ax, np.asarray(s_sel[j].mean, dtype=np.float32),
                 f"d{s_depths[j]}·Δ{s_gains[j]:.0f}")
        else:
            ax.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "poe_search_concepts.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  PoE-Search graphic → {os.path.join(OUT_DIR, 'poe_search_concepts.png')}")

# ── Final method comparison: every teacher's reconstruction side by side ──────
# query · PoE μ_T · PoE-Search μ_T · Top-1 · Top-3 · Query-only.
_cmp_cols = ["query", "PoE μ_T", "PoE-Search μ_T", "Top-1", "Top-3", "Query-only"]
fig, axes = plt.subplots(N_SHOW, len(_cmp_cols), figsize=(len(_cmp_cols) * 1.55, N_SHOW * 1.55))
fig.suptitle("ColorMNIST OOD composition — method comparison\n"
             "held-out (digit, fg, bg) query reconstructed by each teacher", fontsize=11)
for row, (dd, f, b) in enumerate(show_slots):
    x = ood_queries[(dd, f, b)][0]
    imgs = [x, poe_compose(x)[0], poe_search(x)[0],
            teacher_top(x, 1)[0], teacher_top(x, 3)[0], x.copy()]
    for col, (img, name) in enumerate(zip(imgs, _cmp_cols)):
        show(axes[row, col], img, name if row == 0 else None)
    axes[row, 0].set_ylabel(f"{dd}/{f}/{b}", fontsize=6)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "colormnist_methods.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  method comparison → {os.path.join(OUT_DIR, 'colormnist_methods.png')}")

# ── Bar chart of attribute / joint accuracy by method ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
groups = ["digit", "fg", "bg", "joint"]
xpos = np.arange(len(groups)); width = 0.8 / len(METHODS)
for i, m in enumerate(METHODS):
    r = rows[i]
    ax.bar(xpos + (i - (len(METHODS) - 1) / 2) * width, [r[g] for g in groups], width, label=m)
ax.set_xticks(xpos); ax.set_xticklabels(["digit", "fg-color", "bg-color", "joint (all 3)"])
ax.set_ylabel("OOD accuracy %"); ax.set_ylim(0, 105)
ax.set_title("ColorMNIST OOD compositional accuracy — generated image attributes")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "colormnist_accuracy.png"), dpi=130)
plt.close()
print(f"  accuracy chart → {os.path.join(OUT_DIR, 'colormnist_accuracy.png')}")

# ── Concept hierarchy: mean image at each node, top-down ──────────────────────
# Main view: expand HIER_DEPTH levels below the ROOT, showing the HIER_TOP_CH
# highest-count children of each node.  Each node is drawn as its Gaussian
# mean-image m_n (the "concept" it represents); border colour = the node's
# dominant ground-truth digit, n = its instance count.  We then save a folder of
# SUBTREES, each rooted at one of the depth-3 nodes (the bottom of the main view)
# and expanded SUB_DEPTH further levels, so the deeper structure is legible.
HIER_DEPTH  = 3      # main view depth below the root
HIER_TOP_CH = 6      # children shown per node
SUB_DEPTH   = 3      # depth of each level-3-rooted subtree
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

_cmap = plt.get_cmap("tab10")

def greedy_label_counts_full(root, X, ylab):
    """Greedy-descend each instance to a leaf (best child by log_prob over ALL
    children); tally its digit at every node on the path. Gives per-node digit
    distributions for the whole tree, reused by every subtree figure."""
    counts = {}
    for x, lab in zip(X, ylab):
        n = root
        while True:
            counts.setdefault(id(n), np.zeros(10, np.int64))[int(lab)] += 1
            if not n.children:
                break
            n = max(n.children, key=lambda c: c.log_prob(x, _empty))
    return counts

print("Tallying per-node digit distributions (full greedy descent) …")
NODE_COUNTS = greedy_label_counts_full(tree.root, X_train, dig_tr)


def render_hierarchy(root_node, max_depth, out_path, title, top_children=HIER_TOP_CH):
    """Render `root_node` and `max_depth` levels of its highest-count children as a
    mean-image tree. Returns the displayed nodes at the deepest level."""
    def disp_children(n, d):
        return sorted(n.children, key=lambda c: c.count, reverse=True)[:top_children] if d < max_depth else []

    def span(n, d):
        ch = disp_children(n, d)
        return 1 if not ch else sum(span(c, d + 1) for c in ch)

    pos, deepest = {}, []
    def assign(n, d, x_left):
        sp = span(n, d); pos[id(n)] = (x_left + sp / 2.0, d)
        if d == max_depth:
            deepest.append(n)
        cur = x_left
        for c in disp_children(n, d):
            assign(c, d + 1, cur); cur += span(c, d + 1)
        return sp
    total_w = assign(root_node, 0, 0.0)

    # adaptive sizing: width tracks the tree (capped); thumbnail zoom tracks spacing
    _SCALE, _MAXW = 1.45, 150.0
    fig_w = min(max(12, total_w * _SCALE), _MAXW)
    zoom  = 1.7 * (fig_w / total_w) / _SCALE
    fig, ax = plt.subplots(figsize=(fig_w, (max_depth + 1) * 2.6))
    ax.set_xlim(-0.6, total_w + 0.6); ax.set_ylim(-0.8, max_depth + 0.8)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title(title, fontsize=13)

    def draw_edges(n, d):
        x, _ = pos[id(n)]
        for c in disp_children(n, d):
            xc, dc = pos[id(c)]
            ax.plot([x, xc], [d + 0.34, dc - 0.34], color="gray", lw=0.8, zorder=0)
            draw_edges(c, d + 1)
    draw_edges(root_node, 0)

    def draw_nodes(n, d):
        x, _ = pos[id(n)]
        img = np.clip(np.asarray(n.mean, dtype=np.float32).reshape(IMG, IMG, 3), 0, 1)
        cnt = NODE_COUNTS.get(id(n))
        if cnt is not None and cnt.sum() > 0:
            dom = int(cnt.argmax()); pct = 100 * cnt[dom] / cnt.sum()
            color, label = _cmap(dom), f"n={int(n.count)}\ndigit {dom} · {pct:.0f}%"
        else:
            color, label = "black", f"n={int(n.count)}"
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (x, d), frameon=True,
                                     bboxprops=dict(edgecolor=color, lw=2.5)))
        ax.text(x, d + 0.40, label, ha="center", va="top", fontsize=5)
        for c in disp_children(n, d):
            draw_nodes(c, d + 1)
    draw_nodes(root_node, 0)

    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=_cmap(i), label=str(i)) for i in range(10)],
              title="dominant digit", loc="lower center", ncol=10, fontsize=8,
              title_fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return deepest


print(f"Rendering main concept hierarchy (depth {HIER_DEPTH}, top {HIER_TOP_CH} children/node) …")
level3_nodes = render_hierarchy(
    tree.root, HIER_DEPTH, os.path.join(OUT_DIR, "colormnist_hierarchy.png"),
    f"ColorMNIST — Cobweb concept hierarchy (mean image per node, "
    f"depth {HIER_DEPTH}, top {HIER_TOP_CH} children/node)")
print(f"  hierarchy → {os.path.join(OUT_DIR, 'colormnist_hierarchy.png')}")

# Subtrees: one figure per level-3 node, expanded SUB_DEPTH further levels.
SUB_DIR = os.path.join(OUT_DIR, "subtrees")
os.makedirs(SUB_DIR, exist_ok=True)
print(f"Rendering {len(level3_nodes)} level-3 subtrees (depth {SUB_DEPTH} each) → {SUB_DIR}/ …")
_n_sub = 0
for i, node in enumerate(level3_nodes):
    if not node.children:
        continue                                   # nothing to expand below a leaf
    cnt = NODE_COUNTS.get(id(node))
    dom = int(cnt.argmax()) if cnt is not None and cnt.sum() > 0 else -1
    title = (f"ColorMNIST — subtree rooted at a level-3 node "
             f"(n={int(node.count)}, dominant digit {dom}; expanded {SUB_DEPTH} levels)")
    render_hierarchy(node, SUB_DEPTH, os.path.join(SUB_DIR, f"subtree_{i:02d}_digit{dom}.png"), title)
    _n_sub += 1
print(f"  saved {_n_sub} subtree figures → {SUB_DIR}/")
print("Done.")
