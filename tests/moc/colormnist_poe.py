"""
ColorMNIST — Test-Time Compositional Generalization with Cobweb (Product-of-Experts)
====================================================================================
Cobweb adaptation of Wang, Gupta, Zhu & MacLellan, "Test-Time Compositional
Generalization in Diffusion Models via Concept Discovery" (2026).

The paper repurposes a pretrained diffusion model as a hierarchy of density modes:
for an out-of-distribution query it discovers reusable concept prototypes, greedily
selects relevant ones with a submodular coverage objective, and composes their local
Gaussians into a Product-of-Experts (PoE).  Cobweb *is* such a hierarchy of Gaussian
density modes — every node n is a diagonal-Gaussian expert  q_n(x)=N(m_n, Σ_n) with
m_n=node.mean, σ²_{n,r}=sum_sq_r/count + prior_var (verified to match node.log_prob) —
so we run the paper's discovery + composition directly on the Cobweb tree, in raw
pixel space, with no diffusion model.

THE METHOD — two steps.  Concepts = Cobweb nodes (each a diagonal-Gaussian expert).
The per-pixel expert log-likelihood is  ℓ_{n,r}(x) = log N(x_r; m_{n,r}, σ²_{n,r}).

  (1) HOW WE SELECT THE CONCEPTS TO COMPOSE — TOP-DOWN discovery, then submodular pick
        (a) DISCOVER candidates by descending the tree best-first (paper's modes-of-the-
            marginal at multiple scales): priority = node posterior φ(n)=Σ_r ℓ_{n,r}(x)+log P(n);
            expand a node only while a child raises φ, so each branch stops at its natural
            granularity.  No static pool — candidates are gathered per query.  [search_candidates]
        (b) PICK K by greedy submodular coverage F(S)=Σ_r max_{n∈S} ℓ_{n,r}(x) (paper Eqs. 8-9):
            best singleton, then max marginal gain, until |S|=K(=6).            [submodular_select]
        → the few concepts that, together, best explain the query (e.g. a background-colour
          concept + a digit-shape+colour concept).

  (2) HOW WE COMPOSE THEM — per-dim Product-of-Experts at the HARD limit (paper Eq. 10, τ→0)
        the paper weights each concept per pixel  w_n(r) = softmax_{n∈S}(ℓ_{n,r}(x)/τ);
        we take τ→0, which is per-pixel ownership:
            μ_T[r] = m_{ argmax_{n∈S} ℓ_{n,r}(x) , r }
        i.e. every pixel is copied from the SINGLE selected concept that best explains the
        query at that pixel.                                   [code: poe_compose]

The hard limit is the crux: soft averaging (τ=1) blends every concept on every pixel
→ blurry, wrong (joint ≈ 27%); hard per-pixel ownership routes background pixels to the
background-colour concept and the stroke to the digit-shape+colour concept → a sharp,
correct held-out (digit, fg, bg) (joint ≈ 48%).

Benchmark (paper §4.1): 32×32 RGB, 10 digits × 4 fg-colours × 4 bg-colours = 160
slots, 120 SEEN by Cobweb / 40 held out as OOD (verified compositional split).
Metrics (no diffusion sampler → adapted): attribute classifiers (digit/fg/bg) on the
composed image → per-factor + JOINT accuracy ["did we produce the held-out combo?"],
plus a generalization distance to other held-out class images (guards against the
degenerate "just reconstruct the query" solution — which K=6 abstraction prevents).
"""

import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch, torchvision, torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from cobweb.cobweb_continuous import CobwebContinuousTree

RNG = np.random.default_rng(0)
HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "colormnist_output"); os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(HERE, "mnist_output", "data")            # reuse downloaded MNIST
PRIOR_VAR = 0.05854983152                                        # CobwebContinuousTree default

# ── Palettes (paper §4.1) ─────────────────────────────────────────────────────
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

N_PER_SEEN, N_QUERY, N_GEN = 80, 20, 20     # per-class generated/faithful and generalization sizes
_cur = {d: 0 for d in range(10)}
def take(d, k): i = _by_digit[d][_cur[d]:_cur[d] + k]; _cur[d] += k; return i

X_train, dig_tr, fg_tr, bg_tr, _tr_gi = [], [], [], [], []
for (d, f, b) in SEEN_SLOTS:
    for gi in take(d, N_PER_SEEN):
        X_train.append(render(GRAY[gi], f, b)); _tr_gi.append(int(gi))
        dig_tr.append(d); fg_tr.append(FG_NAMES.index(f)); bg_tr.append(BG_NAMES.index(b))
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

# ── Build the Cobweb tree on raw pixels ───────────────────────────────────────
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

# ── Concept prototypes = Cobweb nodes (depths 1-6, well-supported) ────────────
POE_DEPTHS, MIN_COUNT, POOL_CAP, POE_K = (1, 2, 3, 4, 5, 6), 15, 400, 12
_cand = sorted([(d, n) for d in POE_DEPTHS for n in BD.get(d, []) if n.count >= MIN_COUNT],
               key=lambda dn: dn[1].count, reverse=True)[:POOL_CAP]
poe_nodes  = [n for _, n in _cand]; poe_depths = np.asarray([d for d, _ in _cand]); P = len(poe_nodes)
proto_mean = np.stack([np.asarray(n.mean,   np.float32) for n in poe_nodes])
proto_var  = np.stack([np.asarray(n.sum_sq, np.float32) / np.float32(n.count) for n in poe_nodes]) + np.float32(PRIOR_VAR)
proto_logc = (-0.5 * np.log(2.0 * np.pi * proto_var)).astype(np.float32)
proto_iv   = (0.5 / proto_var).astype(np.float32)
print(f"  {P} concept prototypes")

import heapq
SEARCH_NCAND, SEARCH_MAXPOP = 250, 1500   # wide enough for best-first to reach cross-colour concepts
_logN = np.log(float(tree.root.count))
def _node_depth(n):
    d = 0; p = n.parent
    while p is not None: d += 1; p = p.parent
    return d
def node_feats(node, x):
    """A node as a diagonal-Gaussian expert: (mean, per-dim ℓ, ½·precision, posterior φ)."""
    m = np.asarray(node.mean, np.float32)
    v = np.asarray(node.sum_sq, np.float32) / np.float32(node.count) + np.float32(PRIOR_VAR)
    ll = -0.5 * np.log(2.0 * np.pi * v) - 0.5 * (x - m) ** 2 / v
    return m, ll, (0.5 / v).astype(np.float32), float(ll.sum()) + np.log(float(node.count)) - _logN

# ══ (1) SELECT — discover candidate concepts TOP-DOWN, then submodular-pick K ════
def search_candidates(x, n_cand=SEARCH_NCAND, max_pop=SEARCH_MAXPOP, min_count=MIN_COUNT):
    """Best-first DESCENT of the Cobweb tree — the analog of the paper's discovery of
    density modes at multiple abstraction scales.  Priority = node posterior
    φ(n)=Σ_r ℓ_{n,r}(x)+log P(n); a node's children are expanded only while a child
    raises φ, so each branch stops at its natural granularity (φ-peak).  Collect the
    well-supported nodes visited as the per-query candidate concepts (no static pool)."""
    cache = {}
    def f(n):
        e = cache.get(id(n))
        if e is None: e = node_feats(n, x); cache[id(n)] = e
        return e
    pq = [(-f(tree.root)[3], 0, tree.root)]; tie = 1; C = []; pop = 0
    while pq and pop < max_pop and len(C) < n_cand:
        _, _, node = heapq.heappop(pq); pop += 1
        if node is not tree.root and node.count >= min_count: C.append(node)
        kids = node.children
        if kids and max(f(c)[3] for c in kids) > f(node)[3]:        # descend while φ still rises
            for c in kids: heapq.heappush(pq, (-f(c)[3], tie, c)); tie += 1
    if not C: C = [max(tree.root.children, key=lambda c: f(c)[3])]
    M = np.stack([f(n)[0] for n in C]); L = np.stack([f(n)[1] for n in C]); IV = np.stack([f(n)[2] for n in C])
    return C, M, L, IV

def submodular_select(L, k):
    """Greedy submodular coverage F(S)=Σ_r max_{n∈S} ℓ_{n,r} (paper Eqs. 8-9):
    best singleton, then max marginal gain.  Returns indices into the candidate set."""
    sing = L.sum(1); j0 = int(sing.argmax()); chosen, picked, cur = [j0], {j0}, L[j0].copy()
    for _ in range(min(k, L.shape[0]) - 1):
        g = np.maximum(L - cur[None, :], 0.0).sum(1)
        for j in picked: g[j] = -np.inf
        j = int(g.argmax())
        if g[j] <= 0: break
        chosen.append(j); picked.add(j); cur = np.maximum(cur, L[j])
    return chosen

# ══ (2) COMPOSE — per-dim Product-of-Experts at the hard limit τ→0 (paper Eq. 10) ══
def poe_compose(x, k=POE_K, tau=0.0, n_cand=SEARCH_NCAND, max_pop=SEARCH_MAXPOP, use_all=False):
    """(1) discover candidate concepts top-down; (2) submodular-pick K (or use_all);
    (3) compose by per-dim PoE weights w_n(r)=softmax(ℓ_{n,r}/τ).  tau→0 (default) =
    per-pixel ownership: each pixel from the single selected concept that best explains
    the query there.  Returns (mu_T, selected node objects, per-pixel weights w (K,d) —
    one-hot at τ→0; w.mean(1) is each concept's share of the composition)."""
    C, M, L, IV = search_candidates(x, n_cand, max_pop)
    chosen = list(range(len(C))) if use_all else submodular_select(L, k)
    nodes = [C[j] for j in chosen]; Ms = M[chosen]; Ls = L[chosen]; IVs = IV[chosen]
    if tau <= 0:                                                        # τ→0: hard per-pixel ownership
        own = Ls.argmax(0)                                             # which selected concept owns each pixel
        w = np.zeros_like(Ls); w[own, np.arange(D)] = 1.0              # one-hot ownership map
        return Ms[own, np.arange(D)].astype(np.float32), nodes, w
    w = np.exp((Ls - Ls.max(0, keepdims=True)) / tau); w /= w.sum(0, keepdims=True)
    mu = (w * IVs * Ms).sum(0) / (w * IVs).sum(0)
    return mu.astype(np.float32), nodes, w

def teacher_topk(x, k):
    """Nearest-seen-prototype baseline (paper §4.2): rank the depth-1-6 pool by
    singleton log-lik.  (The retrieval baseline scans a flat pool; our method navigates.)"""
    sing = (proto_logc - proto_iv * (x[None, :] - proto_mean) ** 2).sum(1)
    sel = np.argsort(-sing)[:k]
    if k == 1: return proto_mean[sel[0]].copy()
    w = np.exp(sing[sel] - sing[sel].max()); w /= w.sum(); iv = proto_iv[sel]
    return ((w[:, None] * iv * proto_mean[sel]).sum(0) / (w[:, None] * iv).sum(0)).astype(np.float32)

# Temperature (T) sweep of the per-dim PoE composition.  Selection (top-down discovery
# + submodular pick) is IDENTICAL across all T; only the composition changes:
#   T = 0  → per-pixel ownership (hard limit, the method);  larger T → softer averaging.
TAU_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0]
METHODS = {f"T = {t:g}": (lambda x, t=t: poe_compose(x, tau=t)[0]) for t in TAU_SWEEP}
METHODS["Top-1 retrieval"] = lambda x: teacher_topk(x, 1)                                    # nearest-seen-prototype baseline
METHODS["Query-only"]      = lambda x: np.clip(x + 0.15 * RNG.standard_normal(D).astype(np.float32), 0, 1)  # N(x_q, σ²I) sample (paper §4.2)

# ══ METRICS — exactly as in Wang et al. §4.2 ══════════════════════════════════
# For each OOD class we score the generated images against two reference sets —
# FAITHFULNESS (the query images) and GENERALIZATION (other held-out images of the
# class) — with FID, CLIP image-image cosine, and k-NN (k=3) Precision/Recall/F1 in
# Inception-V3 feature space.  Reported as mean ± SE over the 40 OOD classes.
import torch.nn.functional as F
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
    """Exact 2048-d Fréchet Inception Distance via a low-rank (n×n) eigendecomposition
    (avoids the 2048² matrix sqrt; valid for small per-class N)."""
    mu1, mu2 = fr.mean(0), fg.mean(0); n, m = len(fr), len(fg)
    Mr, Mg = fr - mu1, fg - mu2
    trC1, trC2 = (Mr ** 2).sum() / (n - 1), (Mg ** 2).sum() / (m - 1)
    K = Mr @ Mg.T                                            # (n, m)
    ev = np.linalg.eigvalsh(K @ K.T) / ((n - 1) * (m - 1))   # nonzero eigenvalues of C1·C2
    tr_sqrt = np.sqrt(np.clip(ev, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + trC1 + trC2 - 2 * tr_sqrt)
def prec_recall(real, gen, k=3):                             # Kynkäänniemi k-NN P/R
    def kth(Z): D = cdist(Z, Z); D.sort(1); return D[:, min(k, len(Z) - 1)]
    rr, rg = kth(real), kth(gen)
    P = float((cdist(gen, real) <= rr[None, :]).any(1).mean())
    R = float((cdist(real, gen) <= rg[None, :]).any(1).mean())
    return P, R
def f1(p, r): return 0.0 if p + r == 0 else 2 * p * r / (p + r)

# attribute classifiers (kept only for an interpretable JOINT-accuracy context column)
clf_d = LinearSVC(max_iter=3000).fit(X_train, dig_tr); clf_f = KNeighborsClassifier(5).fit(X_train, fg_tr); clf_b = KNeighborsClassifier(5).fit(X_train, bg_tr)

print("Computing per-class reference features …")
refFi = {s: incep_feats(ood_queries[s]) for s in OOD_SLOTS}
refGi = {s: incep_feats(ood_genset[s])  for s in OOD_SLOTS}
if HAVE_CLIP:
    refFc = {s: clip_feats(ood_queries[s]) for s in OOD_SLOTS}
    refGc = {s: clip_feats(ood_genset[s])  for s in OOD_SLOTS}

print("Evaluating each method (paper metrics, per OOD class) …")
def evaluate(fn):
    acc = {k: [] for k in ["fidF","fidG","clF","clG","pF","rF","f1F","pG","rG","f1G","joint"]}
    for s in OOD_SLOTS:
        d, f, b = s; fi, bi = FG_NAMES.index(f), BG_NAMES.index(b)
        G = np.asarray([fn(x) for x in ood_queries[s]], np.float32); gi = incep_feats(G)
        acc["fidF"].append(fid(refFi[s], gi)); acc["fidG"].append(fid(refGi[s], gi))
        pF, rF = prec_recall(refFi[s], gi); pG, rG = prec_recall(refGi[s], gi)
        acc["pF"].append(pF); acc["rF"].append(rF); acc["f1F"].append(f1(pF, rF))
        acc["pG"].append(pG); acc["rG"].append(rG); acc["f1G"].append(f1(pG, rG))
        if HAVE_CLIP:
            gc = clip_feats(G); acc["clF"].append(float((gc @ refFc[s].T).mean())); acc["clG"].append(float((gc @ refGc[s].T).mean()))
        acc["joint"].append(100 * ((clf_d.predict(G) == d) & (clf_f.predict(G) == fi) & (clf_b.predict(G) == bi)).mean())
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for k, v in acc.items() if v}

RES = {m: evaluate(fn) for m, fn in METHODS.items()}
def cell(mv): return f"{mv[0]:.1f}±{mv[1]:.1f}"
print(f"\n  Paper metrics — mean±SE over {len(OOD_SLOTS)} OOD classes.   [F=Faithfulness, G=Generalization]")
print(f"  {'Method':<20}{'FID-F↓':>11}{'FID-G↓':>11}{'CLIP-F↑':>10}{'CLIP-G↑':>10}{'F1-F↑':>9}{'F1-G↑':>9}{'joint↑':>8}")
print("  " + "-" * 88)
for m, R in RES.items():
    clF = cell(R['clF']) if HAVE_CLIP else "  n/a"; clG = cell(R['clG']) if HAVE_CLIP else "  n/a"
    print(f"  {m:<20}{cell(R['fidF']):>11}{cell(R['fidG']):>11}{clF:>10}{clG:>10}"
          f"{cell(R['f1F']):>9}{cell(R['f1G']):>9}{R['joint'][0]:>7.1f}%")
with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fcsv:
    cols = ["method","fidF","fidG","clF","clG","pF","rF","f1F","pG","rG","f1G","joint"]
    w = csv.writer(fcsv); w.writerow([c + ("_mean") for c in cols])
    for m, R in RES.items(): w.writerow([m] + [round(R[c][0], 3) if c in R else "" for c in cols[1:]])
print(f"  summary → {os.path.join(OUT_DIR, 'summary.csv')}")

# ── Metrics bar chart — paper metrics by method, Faithfulness vs Generalization ─
# One panel per metric (FID↓, CLIP↑, Precision↑, Recall↑, F1↑); grouped bars per
# method with SE error bars; FID is on its own axis (≫1), the rest are in [0,1].
_panels = [("FID ↓", "fidF", "fidG"), ("CLIP ↑", "clF", "clG"), ("Precision ↑", "pF", "pG"),
           ("Recall ↑", "rF", "rG"), ("F1 ↑", "f1F", "f1G")]
_panels = [p for p in _panels if p[1] in RES[next(iter(RES))]]      # drop CLIP if unavailable
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
fig.suptitle("ColorMNIST OOD — Wang et al. metrics by method (mean ± SE over 40 OOD classes)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUT_DIR, "metrics.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  metrics chart → {os.path.join(OUT_DIR, 'metrics.png')}")

# ── Temperature (T) sweep — metrics vs composition temperature ────────────────
_sw = [f"T = {t:g}" for t in TAU_SWEEP]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
for key, lab, c in [("f1F", "F1 Faithful", "#4878d0"), ("f1G", "F1 General", "#ee854a"), ("joint", "joint/100", "#6acc65")]:
    sc = 100.0 if key == "joint" else 1.0
    ys = [RES[n][key][0] / sc for n in _sw]; es = [RES[n][key][1] / sc for n in _sw]
    axL.errorbar(TAU_SWEEP, ys, yerr=es, marker="o", capsize=2, label=lab, color=c)
axL.set_xlabel("T (PoE composition temperature)"); axL.set_ylabel("score ↑"); axL.set_title("F1 / joint vs T")
axL.legend(fontsize=8); axL.grid(alpha=0.3)
for key, lab, c in [("fidF", "FID Faithful", "#4878d0"), ("fidG", "FID General", "#ee854a")]:
    ys = [RES[n][key][0] for n in _sw]; es = [RES[n][key][1] for n in _sw]
    axR.errorbar(TAU_SWEEP, ys, yerr=es, marker="o", capsize=2, label=lab, color=c)
axR.set_xlabel("T (PoE composition temperature)"); axR.set_ylabel("FID ↓"); axR.set_title("FID vs T")
axR.legend(fontsize=8); axR.grid(alpha=0.3)
fig.suptitle("ColorMNIST OOD — PoE temperature sweep  (T=0 = per-pixel ownership → larger T = softer averaging)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(os.path.join(OUT_DIR, "tau_sweep.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  τ-sweep chart → {os.path.join(OUT_DIR, 'tau_sweep.png')}")

# ── Figures ───────────────────────────────────────────────────────────────────
def show(ax, vec, title=None, ylabel=None):
    ax.imshow(np.clip(vec.reshape(IMG, IMG, 3), 0, 1)); ax.set_xticks([]); ax.set_yticks([])
    if title:  ax.set_title(title, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=6)
N_SHOW = min(10, len(OOD_SLOTS)); show_slots = [OOD_SLOTS[i] for i in np.linspace(0, len(OOD_SLOTS)-1, N_SHOW).astype(int)]

# (1) method comparison
cols = list(METHODS)
fig, axes = plt.subplots(N_SHOW, 1 + len(cols), figsize=((1+len(cols)) * 1.6, N_SHOW * 1.6))
fig.suptitle("ColorMNIST OOD composition — held-out (digit, fg, bg) query reconstructed by each method", fontsize=11)
for r, (d, f, b) in enumerate(show_slots):
    x = ood_queries[(d, f, b)][0]
    show(axes[r, 0], x, "query" if r == 0 else None, f"{d}/{f}/{b}")
    for c, m in enumerate(cols):
        show(axes[r, 1 + c], METHODS[m](x), m if r == 0 else None)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(OUT_DIR, "methods.png"), dpi=130, bbox_inches="tight"); plt.close()
print(f"  methods figure → {os.path.join(OUT_DIR, 'methods.png')}")

# (2) concept decomposition, PER METHOD (per T): query · μ_T · the concepts that
#     compose it, ordered by each concept's share (pixels owned at T=0; mean weight at T>0).
SHOW_CONCEPTS = min(POE_K, 8)
def render_concepts(tau, out_path, title):
    fig, axes = plt.subplots(N_SHOW, 2 + SHOW_CONCEPTS, figsize=((2 + SHOW_CONCEPTS) * 1.5, N_SHOW * 1.55))
    fig.suptitle(title, fontsize=10)
    for r, (d, f, b) in enumerate(show_slots):
        x = ood_queries[(d, f, b)][0]; mu, nodes, w = poe_compose(x, tau=tau)
        contrib = w.mean(1)                                            # each concept's share of the composition
        show(axes[r, 0], x, "query" if r == 0 else None, f"{d}/{f}/{b}")
        show(axes[r, 1], mu, "μ_T" if r == 0 else None)
        order = np.argsort(-contrib)
        for c in range(SHOW_CONCEPTS):
            ax = axes[r, 2 + c]; ax.axis("off")
            if c < len(order):
                i = int(order[c]); nd = nodes[i]
                show(ax, np.asarray(nd.mean, np.float32), f"d{_node_depth(nd)}·{100*contrib[i]:.0f}%")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()

render_concepts(0.0, os.path.join(OUT_DIR, "concepts.png"),
                f"Per-pixel PoE decomposition (T=0) — query · μ_T · top {SHOW_CONCEPTS} of K={POE_K} concepts "
                f"(d=tree depth, %=pixels owned)")
print(f"  concepts figure → {os.path.join(OUT_DIR, 'concepts.png')}")
CONC_DIR = os.path.join(OUT_DIR, "concepts"); os.makedirs(CONC_DIR, exist_ok=True)
for t in TAU_SWEEP:                                                    # concepts for each composition idea (T)
    render_concepts(t, os.path.join(CONC_DIR, f"concepts_T{t:g}.png"),
                    f"PoE concept decomposition at T={t:g} — query · μ_T · top {SHOW_CONCEPTS} concepts "
                    f"(% = concept share: pixels owned at T=0, mean weight at T>0)")
print(f"  per-T concept figures → {CONC_DIR}/")

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
NODE_COUNTS = greedy_counts(tree.root, X_train, dig_tr, 6)              # to depth 6 (covers subtrees)

def render_hierarchy(root_node, out_path, title, max_depth=3, top_children=HIER_TOP):
    """Render root_node + max_depth levels of its top-count children as a mean-image tree.
    Returns the nodes at the deepest displayed level."""
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
# level-3 subtrees: 3 levels deep, rooted at each leaf of the depth-3 hierarchy
SUB_DIR = os.path.join(OUT_DIR, "subtrees"); os.makedirs(SUB_DIR, exist_ok=True)
_nsub = 0
for i, node in enumerate(level3):
    if not node.children: continue
    cnt = NODE_COUNTS.get(id(node)); dom = int(cnt.argmax()) if cnt is not None and cnt.sum() > 0 else -1
    render_hierarchy(node, os.path.join(SUB_DIR, f"subtree_{i:02d}_digit{dom}.png"),
                     f"Level-3 subtree (3 levels deep) — root n={int(node.count)}, dominant digit {dom}", 3)
    _nsub += 1
print(f"  {_nsub} level-3 subtrees → {SUB_DIR}/")
print("Done.")
