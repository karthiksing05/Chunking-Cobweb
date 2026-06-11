"""
MNIST (plain grayscale) — Cobweb + Product-of-Experts, validation-set evaluation
================================================================================
Same Cobweb-PoE pipeline as colormnist_poe.py, but on REGULAR grayscale MNIST (no
colourization) and WITHOUT the compositional OOD split — we just train on MNIST train
and evaluate on the validation (test) set, grouped by digit class.  The main point of
interest here is the PRIMITIVE-DISCOVERY analysis (pixels the PoE donates together).

  1. CONCEPTS = Cobweb nodes.  Each node n is a diagonal Gaussian N(m_n, σ²_n) over the
     32×32 grayscale image (gray replicated to 3 channels so the same code + Inception/CLIP
     feature extractors run unchanged).  Shallow nodes are broad concepts, deep nodes specific.

  2. SELECT concepts with the paper's greedy argmax (Eq. 9) over a candidate POOL gathered from
     the tree: best-first expand by coverage gain Δ(n)=Σ_r max(ℓ_{n,r}(x_q)−cur_r,0) to ~3% of
     the nodes, take the global Δ-maximizer, fold into cur, re-route.  HOW MANY: fixed K, or the
     leftover-image cutoff — stop once μ_T explains ≥99% of the query.

  3. COMPOSE with the per-pixel product of Gaussians (Eqs. 7 & 10) at τ=0.1; the generated
     image is the PoE mean μ_T (no diffusion sampler).

Eval: train Cobweb on a balanced subset of MNIST train; queries come from the MNIST test set.
Each of the 10 digit classes is scored against two reference sets — Faithfulness (its query
images) and Generalization (other held-out test images of the same digit) — with FID, CLIP,
k-NN P/R/F1, plus a digit-classifier accuracy.  Baselines: Top-1 / Top-3 nearest class-mean.
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
OUT_DIR  = os.path.join(HERE, "mnist_output"); os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"); os.makedirs(DATA_DIR, exist_ok=True)  # shared root-level data/
PRIOR_VAR = 0.05854983152                                        # CobwebContinuousTree default

# ── Hyperparameters ───────────────────────────────────────────────────────────
TAU      = 0.1     # per-coordinate softmax temperature (Eq. 10); low → sharp per-pixel ownership
K_MAX    = 6       # safety cap on concepts per query
EXPLAIN_FRAC = 0.99 # cutoff: keep adding concepts until the composed μ_T explains this fraction of
                    # the query's content (R² vs the query mean) — i.e. until little leftover remains

IMG = 32; D = IMG * IMG * 3

# ── Render plain grayscale MNIST (gray replicated to 3 channels) ────────────────
def render(g28):
    """Pad a 28×28 gray digit to 32×32 and replicate to 3 channels (no colour)."""
    c = np.zeros((IMG, IMG), np.float32); c[2:30, 2:30] = g28
    return np.repeat(c[:, :, None], 3, axis=2).reshape(-1)

transform = transforms.ToTensor()
_tr = torchvision.datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
_te = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
def _load(ds, n):
    im, lb = next(iter(torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)))
    return im.squeeze(1).numpy(), lb.numpy()
GR_TR, LB_TR = _load(_tr, 60000)
GR_TE, LB_TE = _load(_te, 10000)
_tr_by = {d: np.where(LB_TR == d)[0] for d in range(10)}
_te_by = {d: np.where(LB_TE == d)[0] for d in range(10)}

# ── No OOD split: train on MNIST train, evaluate on MNIST test, grouped by digit ─
DIGITS = list(range(10))
N_PER_DIGIT, N_QUERY, N_GEN = 960, 50, 16    # train per class; test queries; generalization refs
OOD_SLOTS = DIGITS                            # "classes" = the 10 digits (name kept for shared code)

X_train, dig_tr = [], []
for d in DIGITS:
    for gi in _tr_by[d][:N_PER_DIGIT]:
        X_train.append(render(GR_TR[gi])); dig_tr.append(d)
X_train = np.asarray(X_train, np.float32); dig_tr = np.asarray(dig_tr)

ood_queries, ood_genset = {}, {}
for d in DIGITS:
    qi = _te_by[d][:N_QUERY + N_GEN]
    ims = np.asarray([render(GR_TE[g]) for g in qi], np.float32)
    ood_queries[d] = ims[:N_QUERY]; ood_genset[d] = ims[N_QUERY:]
print(f"Train {X_train.shape} ({N_PER_DIGIT}/digit); validation {len(DIGITS)}×{N_QUERY} queries + {N_GEN} gen imgs")

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

# ══ (2′) SAMPLE — Cobweb-native generative prior (scheme A: descendant-leaf resampling) ══
# The paper draws clean varied samples by routing q_T's score through a diffusion model whose
# reverse process re-projects onto the image manifold.  We have no diffusion model — but Cobweb
# IS a hierarchical generative prior: its leaves are tight Gaussians on clusters of real training
# instances, so a leaf sample is clean and on-manifold.  We keep the coverage-selected concept
# frontier, then replace each selected node's mean with a SAMPLED descendant leaf and recompose —
# so each sample is a different clean handwriting exemplar of the query's digit, on-manifold.
# (Image analog of grammar subtree-exchange replay.)
def _sample_leaf(node, rng):
    """Count-weighted walk down node's subtree to a leaf (Cobweb's own generative process)."""
    n = node
    while n.children:
        ch = list(n.children); cnt = np.array([float(c.count) for c in ch], np.float64)
        n = ch[int(rng.choice(len(ch), p=cnt / cnt.sum()))]
    return n

def poe_sample(x, selector, tau=TAU, rng=RNG):
    """Like poe_compose, but each selected concept donates a SAMPLED descendant leaf instead of its
    mean — a clean, on-manifold draw from the composed distribution.  Returns (sample, leaves)."""
    sel = selector(x)
    if not sel: return x.copy(), []
    leaves = [_sample_leaf(s[0], rng) for s in sel]
    feats = [node_feats(l, x) for l in leaves]                  # (mean, var, per-dim ℓ) of each leaf
    M = np.stack([f[0] for f in feats]); V = np.stack([f[1] for f in feats]); L = np.stack([f[2] for f in feats])
    return _poe_mu(M, V, L, tau).astype(np.float32), leaves     # _poe_mu handles the k=1 case

# ── Baselines: nearest class retrieval (paper §4.2 Top-k) ──────────────────────
# C = the 10 digit classes, each a diagonal Gaussian.  Top-1 = nearest class mean;
# Top-3 = the per-pixel PoE of the 3 nearest classes.
CLASS_MEAN = np.stack([X_train[dig_tr == d].mean(0) for d in DIGITS]).astype(np.float32)
CLASS_VAR  = np.stack([X_train[dig_tr == d].var(0)  for d in DIGITS]).astype(np.float32) + np.float32(PRIOR_VAR)
_CLOGC = (-0.5 * np.log(2.0 * np.pi * CLASS_VAR)).astype(np.float32)

def teacher_topk(x, k):
    L = _CLOGC - 0.5 * (x[None, :] - CLASS_MEAN) ** 2 / CLASS_VAR          # (10, D) per-dim ℓ
    sel = np.argsort(-L.sum(1))[:k]
    if k == 1: return CLASS_MEAN[sel[0]].copy()
    Ls, M, V = L[sel], CLASS_MEAN[sel], CLASS_VAR[sel]
    w = np.exp((Ls - Ls.max(0, keepdims=True)) / TAU); w /= w.sum(0, keepdims=True)
    prec = (w / V).sum(0)
    return ((w * M / V).sum(0) / np.maximum(prec, 1e-12)).astype(np.float32)

_ef = lambda f: (lambda y: select_concepts(y, explain_frac=f))  # leftover-image cutoff at fraction f
_k3 = lambda y: select_concepts(y, fixed_k=3)                   # paper's fixed K=3
METHODS = {
    "PoE 99%":   lambda x: poe_compose(x, _ef(0.99))[0],        # leftover cutoff, explain ≥ 99% (mean μ_T)
    "PoE 90%":   lambda x: poe_compose(x, _ef(0.90))[0],        # leftover cutoff, explain ≥ 90%
    "PoE K=3":   lambda x: poe_compose(x, _k3)[0],             # fixed K=3 (paper's choice)
    "PoE-S 99%": lambda x: poe_sample(x, _ef(0.99))[0],        # scheme A: one clean leaf-resampled draw
    "Top-1":     lambda x: teacher_topk(x, 1),
    "Top-3":     lambda x: teacher_topk(x, 3),
}

# ══ METRICS — exactly as in Wang et al. §4.2 ══════════════════════════════════
from scipy.spatial.distance import cdist
from torchvision.models import inception_v3, Inception_V3_Weights

print("Loading Inception-V3 + CLIP feature extractors …")
_incep = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1); _incep.fc = torch.nn.Identity(); _incep.eval()
_IMm = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); _IMs = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def _chw(Xf): return torch.tensor(Xf.reshape(-1, IMG, IMG, 3), dtype=torch.float32).permute(0, 3, 1, 2)
@torch.no_grad()
def incep_feats(Xf, bs=64):
    o = []
    for i in range(0, len(Xf), bs):
        t = F.interpolate(_chw(Xf[i:i+bs]), size=299, mode="bilinear", align_corners=False)
        o.append(_incep((t - _IMm) / _IMs).numpy())
    return np.concatenate(o, 0)
try:
    import open_clip
    _clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai"); _clip.eval()
    _CLm = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    _CLs = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    @torch.no_grad()
    def clip_feats(Xf, bs=64):
        o = []
        for i in range(0, len(Xf), bs):
            t = F.interpolate(_chw(Xf[i:i+bs]), size=224, mode="bicubic", align_corners=False)
            fe = _clip.encode_image((t - _CLm) / _CLs).numpy(); o.append(fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-8))
        return np.concatenate(o, 0)
    HAVE_CLIP = True
except Exception as e:
    print("  CLIP unavailable:", e); HAVE_CLIP = False

def fid(fr, fg):
    """Exact 2048-d Fréchet Inception Distance via a low-rank (n×n) eigendecomposition."""
    mu1, mu2 = fr.mean(0), fg.mean(0); n, m = len(fr), len(fg)
    Mr, Mg = fr - mu1, fg - mu2
    trC1, trC2 = (Mr ** 2).sum() / (n - 1), (Mg ** 2).sum() / (m - 1)
    K = Mr @ Mg.T
    ev = np.linalg.eigvalsh(K @ K.T) / ((n - 1) * (m - 1))
    tr_sqrt = np.sqrt(np.clip(ev, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + trC1 + trC2 - 2 * tr_sqrt)
def prec_recall(real, gen, k=3):                             # Kynkäänniemi k-NN P/R
    def kth(Z): Dm = cdist(Z, Z); Dm.sort(1); return Dm[:, min(k, len(Z) - 1)]
    rr, rg = kth(real), kth(gen)
    P = float((cdist(gen, real) <= rr[None, :]).any(1).mean())
    R = float((cdist(real, gen) <= rg[None, :]).any(1).mean())
    return P, R
def f1(p, r): return 0.0 if p + r == 0 else 2 * p * r / (p + r)

# digit classifier (kept only for an interpretable JOINT-accuracy context column)
clf_d = LinearSVC(max_iter=3000).fit(X_train, dig_tr)

print("Computing per-class reference features …")
refFi = {s: incep_feats(ood_queries[s]) for s in OOD_SLOTS}
refGi = {s: incep_feats(ood_genset[s])  for s in OOD_SLOTS}
if HAVE_CLIP:
    refFc = {s: clip_feats(ood_queries[s]) for s in OOD_SLOTS}
    refGc = {s: clip_feats(ood_genset[s])  for s in OOD_SLOTS}

print("Evaluating each method (paper metrics, per digit class) …")
def evaluate(fn):
    acc = {k: [] for k in ["fidF","fidG","clF","clG","pF","rF","f1F","pG","rG","f1G","joint"]}
    for d in OOD_SLOTS:
        G = np.asarray([fn(x) for x in ood_queries[d]], np.float32); gi = incep_feats(G)
        acc["fidF"].append(fid(refFi[d], gi)); acc["fidG"].append(fid(refGi[d], gi))
        pF, rF = prec_recall(refFi[d], gi); pG, rG = prec_recall(refGi[d], gi)
        acc["pF"].append(pF); acc["rF"].append(rF); acc["f1F"].append(f1(pF, rF))
        acc["pG"].append(pG); acc["rG"].append(rG); acc["f1G"].append(f1(pG, rG))
        if HAVE_CLIP:
            gc = clip_feats(G); acc["clF"].append(float((gc @ refFc[d].T).mean())); acc["clG"].append(float((gc @ refGc[d].T).mean()))
        acc["joint"].append(100 * (clf_d.predict(G) == d).mean())          # digit-recovery accuracy
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for k, v in acc.items() if v}

RES = {m: evaluate(fn) for m, fn in METHODS.items()}
def cell(mv): return f"{mv[0]:.1f}±{mv[1]:.1f}"
print(f"\n  Paper metrics — mean±SE over {len(OOD_SLOTS)} OOD classes.   [F=Faithfulness, G=Generalization]")
print(f"  {'Method':<16}{'FID-F↓':>11}{'FID-G↓':>11}{'CLIP-F↑':>10}{'CLIP-G↑':>10}{'F1-F↑':>9}{'F1-G↑':>9}{'joint↑':>8}")
print("  " + "-" * 84)
for m, R in RES.items():
    clF = cell(R['clF']) if HAVE_CLIP else "  n/a"; clG = cell(R['clG']) if HAVE_CLIP else "  n/a"
    print(f"  {m:<16}{cell(R['fidF']):>11}{cell(R['fidG']):>11}{clF:>10}{clG:>10}"
          f"{cell(R['f1F']):>9}{cell(R['f1G']):>9}{R['joint'][0]:>7.1f}%")
with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fcsv:
    cols = ["method","fidF","fidG","clF","clG","pF","rF","f1F","pG","rG","f1G","joint"]
    w = csv.writer(fcsv); w.writerow([c + "_mean" for c in cols])
    for m, R in RES.items(): w.writerow([m] + [round(R[c][0], 3) if c in R else "" for c in cols[1:]])
print(f"  summary → {os.path.join(OUT_DIR, 'summary.csv')}")

# ── Per-variant search stats over all OOD queries: K, depth, nodes visited ──
print("\n  Best-first pool selection — per-query stats by variant:")
_VARIANTS = [("99%", dict(explain_frac=0.99), "#4878d0"), ("90%", dict(explain_frac=0.90), "#6acc65"),
             ("K=3", dict(fixed_k=3), "#956cb4")]
_K = {n: [] for n, _, _ in _VARIANTS}; _EXP = {n: [] for n, _, _ in _VARIANTS}; _DEPTH = {n: [] for n, _, _ in _VARIANTS}
_LEAF = {n: [] for n, _, _ in _VARIANTS}                          # per selected concept: 1 if leaf (no children)
for s in OOD_SLOTS:
    for x in ood_queries[s]:
        for name, kw, _ in _VARIANTS:
            nodes = [t[0] for t in select_concepts(x, record=_EXP[name], **kw)]
            _K[name].append(len(nodes)); _DEPTH[name].extend(_node_depth(n) for n in nodes)
            _LEAF[name].extend(0 if n.children else 1 for n in nodes)
_nq = len(_K["99%"])
for name, _, _ in _VARIANTS:
    leaf_frac = 100 * np.mean(_LEAF[name]) if _LEAF[name] else 0.0
    print(f"    {name:<4} K mean {np.mean(_K[name]):.2f}  depth mean {np.mean(_DEPTH[name]):.2f}  "
          f"nodes mean {np.mean(_EXP[name]):.0f}  leaves {leaf_frac:.0f}% / internal {100-leaf_frac:.0f}%")

# depth histogram (default 99% cutoff), highest → lowest depth
_d99 = _DEPTH["99%"]; _dmin, _dmax = int(min(_d99)), int(max(_d99)); _bins = np.arange(_dmin, _dmax + 2) - 0.5
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.hist(_d99, bins=_bins, color="#4878d0", alpha=0.85)
ax.axvline(np.mean(_d99), color="k", ls="--", lw=1, label=f"mean {np.mean(_d99):.1f}")
ax.set_xticks(range(_dmin, _dmax + 1)); ax.invert_xaxis()
ax.set_xlabel("Cobweb tree depth of selected concept  (deeper/leaf ← → shallower/root)")
ax.set_ylabel(f"count over {_nq} OOD queries × selected concepts")
ax.set_title(f"Depth of selected concepts (99% cutoff, all {len(OOD_SLOTS)}×{N_QUERY} validation queries)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "concept_depths.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"    depth histogram → {os.path.join(OUT_DIR, 'concept_depths.png')}")

# concept-type bar chart: mean # of selected concepts that are internal (have children) vs leaves
_names = [n for n, _, _ in _VARIANTS]; _x = np.arange(len(_names))
_internal = [np.sum([1 - v for v in _LEAF[n]]) / _nq for n in _names]   # mean count per query
_leaves   = [np.sum(_LEAF[n]) / _nq for n in _names]
fig, ax = plt.subplots(figsize=(7, 4.4))
b1 = ax.bar(_x, _internal, 0.6, color="#4878d0", label="internal (has children)")
b2 = ax.bar(_x, _leaves, 0.6, bottom=_internal, color="#ee854a", label="leaf (no children)")
for i in range(len(_names)):
    if _internal[i] > 0.05: ax.text(_x[i], _internal[i]/2, f"{_internal[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
    if _leaves[i]   > 0.05: ax.text(_x[i], _internal[i] + _leaves[i]/2, f"{_leaves[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
ax.set_xticks(_x); ax.set_xticklabels(_names); ax.set_xlabel("variant")
ax.set_ylabel(f"mean # selected concepts per query (over {_nq} OOD queries)")
ax.set_title("Selected concepts — internal nodes vs leaves, by variant")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "concept_types.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"    concept-type bar chart → {os.path.join(OUT_DIR, 'concept_types.png')}")

# nodes-visited bar chart: mean nodes per query, by variant
_names = [n for n, _, _ in _VARIANTS]; _cols = [c for _, _, c in _VARIANTS]
_means = [np.mean(_EXP[n]) for n in _names]
fig, ax = plt.subplots(figsize=(7, 4.4))
bars = ax.bar(_names, _means, color=_cols)
for b, m in zip(bars, _means): ax.text(b.get_x() + b.get_width()/2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel(f"mean tree nodes visited per query (over {_nq} OOD queries)")
ax.set_xlabel("variant")
ax.set_title("Search effort — average nodes visited per query, by variant")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "nodes_explored.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"    nodes-visited histogram (all variants) → {os.path.join(OUT_DIR, 'nodes_explored.png')}")

# ── Metrics bar chart — paper metrics by method, Faithfulness vs Generalization ─
_panels = [("FID ↓", "fidF", "fidG"), ("CLIP ↑", "clF", "clG"), ("Precision ↑", "pF", "pG"),
           ("Recall ↑", "rF", "rG"), ("F1 ↑", "f1F", "f1G")]
_panels = [p for p in _panels if p[1] in RES[next(iter(RES))]]
_mlabels = list(RES); _x = np.arange(len(_mlabels)); _w = 0.38
fig, axes = plt.subplots(1, len(_panels), figsize=(len(_panels) * 3.0, 4.2))
for ax, (title, kf, kg) in zip(axes, _panels):
    mf = [RES[m][kf][0] for m in _mlabels]; sf = [RES[m][kf][1] for m in _mlabels]
    mg = [RES[m][kg][0] for m in _mlabels]; sg = [RES[m][kg][1] for m in _mlabels]
    ax.bar(_x - _w/2, mf, _w, yerr=sf, capsize=2, color="#4878d0", label="Faithfulness")
    ax.bar(_x + _w/2, mg, _w, yerr=sg, capsize=2, color="#ee854a", label="Generalization")
    ax.set_title(title, fontsize=10); ax.set_xticks(_x)
    ax.set_xticklabels([m.replace(" ", "\n", 1) for m in _mlabels], rotation=30, ha="right", fontsize=6)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=6)
fig.suptitle("MNIST validation — metrics by method (mean ± SE over 10 digit classes)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUT_DIR, "metrics.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  metrics chart → {os.path.join(OUT_DIR, 'metrics.png')}")

# ── Figures ───────────────────────────────────────────────────────────────────
def show(ax, vec, title=None, ylabel=None):
    ax.imshow(np.clip(vec.reshape(IMG, IMG, 3), 0, 1)); ax.set_xticks([]); ax.set_yticks([])
    if title:  ax.set_title(title, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=6)
N_SHOW = min(10, len(OOD_SLOTS)); show_slots = [OOD_SLOTS[i] for i in np.linspace(0, len(OOD_SLOTS)-1, N_SHOW).astype(int)]

# (1) method comparison
cols = list(METHODS)
fig, axes = plt.subplots(N_SHOW, 1 + len(cols), figsize=((1+len(cols)) * 1.6, N_SHOW * 1.6))
fig.suptitle("MNIST validation composition — held-out digit query reconstructed by each method", fontsize=11)
for r, d in enumerate(show_slots):
    x = ood_queries[d][0]
    show(axes[r, 0], x, "query" if r == 0 else None, f"digit {d}")
    for c, m in enumerate(cols):
        show(axes[r, 1 + c], METHODS[m](x), m if r == 0 else None)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(OUT_DIR, "methods.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  methods figure → {os.path.join(OUT_DIR, 'methods.png')}")

# (1b) Cobweb-prior SAMPLING (scheme A): query · μ_T · N clean leaf-resampled draws of the SAME
#      digit.  μ_T is the deterministic PoE mean; each sample swaps in a different descendant
#      leaf per concept, so variety = handwriting style while the digit stays fixed.
N_SAMP = 6
fig, axes = plt.subplots(N_SHOW, 2 + N_SAMP, figsize=((2 + N_SAMP) * 1.6, N_SHOW * 1.55), squeeze=False)
fig.suptitle(f"MNIST — Cobweb-prior samples (scheme A): query · μ_T · {N_SAMP} clean leaf-resampled draws", fontsize=11)
for r, d in enumerate(show_slots):
    x = ood_queries[d][0]
    show(axes[r, 0], x, "query" if r == 0 else None, f"digit {d}")
    show(axes[r, 1], poe_compose(x, _ef(0.99))[0], "μ_T (mean)" if r == 0 else None)
    for c in range(N_SAMP):
        show(axes[r, 2 + c], poe_sample(x, _ef(0.99))[0], f"sample {c+1}" if r == 0 else None)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(OUT_DIR, "samples.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  samples figure → {os.path.join(OUT_DIR, 'samples.png')}")

# (2) concept decomposition: query · μ_T · the concepts that compose it, in RECOVERY ORDER
#     (concept 1 = first picked).  %=each concept's share of the total coverage gain Σ Δ —
#     the contribution to the marginal that the selection heuristic actually used.  Columns
#     are sized to the most concepts any shown query uses (no wasted whitespace).
def _share(gains):
    g = np.asarray(gains, np.float32); s = g.sum()
    return g / s if len(g) and s > 0 else np.zeros(len(g), np.float32)

def _gather(selector):
    rows = []
    for d in show_slots:
        x = ood_queries[d][0]; mu, nodes, w, gains = poe_compose(x, selector)
        rows.append((x, d, mu, nodes, w, gains))
    ncol = max((len(r[3]) for r in rows), default=1)
    return rows, ncol

def render_concepts(selector, out_path, title):
    rows, ncol = _gather(selector)
    fig, axes = plt.subplots(N_SHOW, 2 + ncol, figsize=((2 + ncol) * 1.6, N_SHOW * 1.55), squeeze=False)
    fig.suptitle(title, fontsize=10)
    for r, (x, d, mu, nodes, w, gains) in enumerate(rows):
        contrib = _share(gains)
        show(axes[r, 0], x, "query" if r == 0 else None, f"digit {d}")
        show(axes[r, 1], mu, "μ_T" if r == 0 else None)
        for c in range(ncol):
            ax = axes[r, 2 + c]; ax.axis("off")
            if c < len(nodes):
                nd = nodes[c]
                show(ax, np.asarray(nd.mean, np.float32), f"d{_node_depth(nd)}·{100*contrib[c]:.0f}%")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()

# (2b) donation heatmaps: per concept, the per-coordinate composition weight w_n(r) it donates
#      (reshaped to 32×32, averaged over channels); %label = share of coverage gain ΣΔ.
def render_heatmaps(selector, out_path, title):
    rows, ncol = _gather(selector)
    fig, axes = plt.subplots(N_SHOW, 2 + ncol, figsize=((2 + ncol) * 1.6, N_SHOW * 1.55), squeeze=False)
    fig.suptitle(title, fontsize=10); im = None
    for r, (x, d, mu, nodes, w, gains) in enumerate(rows):
        contrib = _share(gains)
        show(axes[r, 0], x, "query" if r == 0 else None, f"digit {d}")
        show(axes[r, 1], mu, "μ_T" if r == 0 else None)
        for c in range(ncol):
            ax = axes[r, 2 + c]; ax.set_xticks([]); ax.set_yticks([])
            if c < len(nodes):
                heat = w[c].reshape(IMG, IMG, 3).mean(2)
                im = ax.imshow(heat, cmap="magma", vmin=0.0, vmax=1.0)
                ax.set_xlabel(f"d{_node_depth(nodes[c])}·{100*contrib[c]:.0f}%", fontsize=6)
                if r == 0: ax.set_title(f"concept {c+1}", fontsize=8)
            else:
                ax.axis("off")
    if im is not None: fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="per-coord weight w_n(r)")
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()

# concepts + heatmaps per PoE variant, saved to subdirectories
CDIR = os.path.join(OUT_DIR, "concepts"); HDIR = os.path.join(OUT_DIR, "heatmaps")
os.makedirs(CDIR, exist_ok=True); os.makedirs(HDIR, exist_ok=True)
for _suf, _sel, _lab in [("99", _ef(0.99), "explain ≥ 99%"), ("90", _ef(0.90), "explain ≥ 90%"),
                          ("k3", _k3, "fixed K=3")]:
    render_concepts(_sel, os.path.join(CDIR, f"concepts_{_suf}.png"),
                    f"PoE decomposition ({_lab}) — query · μ_T · concepts in recovery order (d=tree depth, %=share of coverage gain ΣΔ)")
    render_heatmaps(_sel, os.path.join(HDIR, f"heatmaps_{_suf}.png"),
                    f"Concept donation heatmaps ({_lab}) — query · μ_T · per-coord weight w_n(r) (Eq. 10, τ={TAU}); %=share of coverage gain ΣΔ")
print(f"  concepts → {CDIR}/  ·  heatmaps → {HDIR}/  (variants: 99, 90, k3)")

# (3) concept hierarchy: mean image at each node, top-down — plus level-3 subtrees
HIER_TOP = 6; _cmap = plt.get_cmap("tab10")
def greedy_counts(root, Xs, ys, maxd):
    counts = {}
    for xx, lab in zip(Xs, ys):
        n = root
        for dd in range(maxd + 1):
            counts.setdefault(id(n), np.zeros(10, np.int64))[int(lab)] += 1
            if not n.children or dd == maxd: break
            n = max(n.children, key=lambda c: c.log_prob(xx, _empty))
    return counts
NODE_COUNTS = greedy_counts(tree.root, X_train, dig_tr, 6)

def render_hierarchy(root_node, out_path, title, max_depth=3, top_children=HIER_TOP):
    def dch(n, d): return sorted(n.children, key=lambda c: c.count, reverse=True)[:top_children] if d < max_depth else []
    def span(n, d):
        ch = dch(n, d); return 1 if not ch else sum(span(c, d + 1) for c in ch)
    pos = {}; deepest = []
    def assign(n, d, xl):
        sp = span(n, d); pos[id(n)] = (xl + sp / 2.0, d)
        if d == max_depth: deepest.append(n)
        cur = xl
        for c in dch(n, d): assign(c, d + 1, cur); cur += span(c, d + 1)
        return sp
    tw = assign(root_node, 0, 0.0)
    fig, ax = plt.subplots(figsize=(max(12, tw * 1.5), (max_depth + 1) * 2.6))
    ax.set_xlim(-0.6, tw + 0.6); ax.set_ylim(-0.8, max_depth + 0.8); ax.invert_yaxis(); ax.axis("off")
    ax.set_title(title, fontsize=12)
    def de(n, d):
        x0, _ = pos[id(n)]
        for c in dch(n, d):
            xc, dc = pos[id(c)]
            ax.plot([x0, xc], [d + 0.34, dc - 0.34], color="gray", lw=0.8, zorder=0); de(c, d + 1)
    de(root_node, 0)
    def dn(n, d):
        x0, _ = pos[id(n)]; img = np.clip(np.asarray(n.mean, np.float32).reshape(IMG, IMG, 3), 0, 1)
        cnt = NODE_COUNTS.get(id(n))
        if cnt is not None and cnt.sum() > 0:
            dom = int(cnt.argmax()); color = _cmap(dom); lab = f"n={int(n.count)}\ndigit {dom}·{100*cnt[dom]/cnt.sum():.0f}%"
        else:
            color, lab = "black", f"n={int(n.count)}"
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=1.7), (x0, d), frameon=True, bboxprops=dict(edgecolor=color, lw=2.5)))
        ax.text(x0, d + 0.40, lab, ha="center", va="top", fontsize=5)
        for c in dch(n, d): dn(c, d + 1)
    dn(root_node, 0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=_cmap(i), label=str(i)) for i in range(10)],
              title="dominant digit", loc="lower center", ncol=10, fontsize=8, title_fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(); plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()
    return deepest

level3 = render_hierarchy(tree.root, os.path.join(OUT_DIR, "hierarchy.png"),
                          "Cobweb concept hierarchy — mean image per node (top-down, depth 3, top 6 children)", 3)
print(f"  hierarchy figure → {os.path.join(OUT_DIR, 'hierarchy.png')}")
SUB_DIR = os.path.join(OUT_DIR, "subtrees"); os.makedirs(SUB_DIR, exist_ok=True)
_nsub = 0
for i, node in enumerate(level3):
    if not node.children: continue
    cnt = NODE_COUNTS.get(id(node)); dom = int(cnt.argmax()) if cnt is not None and cnt.sum() > 0 else -1
    render_hierarchy(node, os.path.join(SUB_DIR, f"subtree_{i:02d}_digit{dom}.png"),
                     f"Level-3 subtree (3 levels deep) — root n={int(node.count)}, dominant digit {dom}", 3)
    _nsub += 1
print(f"  {_nsub} level-3 subtrees → {SUB_DIR}/")

# ════════════════════════════════════════════════════════════════════════════════
# PRIMITIVE DISCOVERY — pixels the PoE donates together (90% variant).  Gather every
# composed concept's donation heatmap, correlate pixels by donation profile, and group
# them with Cobweb to reveal reusable spatial primitives.
# ════════════════════════════════════════════════════════════════════════════════
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
print(f"  {_nsub} primitive subtrees → {SUB}/")
print("Done.")
