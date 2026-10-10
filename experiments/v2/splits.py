"""Split moves: refine a category by its composition context (domain-neutral).

After the representation cut and ``merge_symbols`` (stage A of
``compile_grammar``), each symbol may be split in two. The move:

* units: the symbol's elements grouped by their own composition key (a
  primitive's token; a composite's pair of part symbols), so that every
  occurrence of a type goes to the same side;
* profile of a unit: how often its elements occur with each composition
  context: the sibling's composition key and side, and the parent's
  symbol and side (the categories it combines with). The element's own
  attributes never enter the profile;
* proposal: the sign of the leading principal direction of the units'
  (normalised, count-weighted) profiles, then 2-means from it (heuristic);
* ranking: the exact plain-PCFG code on the split symbols (cheap);
* acceptance: the best few candidates are refitted exactly (the composition
  cut searched again and the tables refitted, as Phase A2 did) and a split
  is kept only if the full code shrinks. Repeated while a split pays.

Nothing here knows about the domain: grammatical number enters only the
evaluation.

``install_splits(heads=...)`` substitutes ``trellis2.model.compile_grammar``
(and, with heads, what ``heads.install`` substitutes) in this process only.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import heads as H
import trellis2.grammar as _grammar_mod
import trellis2.model as _model_mod
from trellis2.grammar import _compact
from trellis2.mdl import dm_code

CONFIG = {"heads": False, "candidates": 3, "max_splits": 40, "min_units": 2, "log": True, "components": 3,
          # Joint (coordinated) splits: each single candidate seeds a split
          # propagated through the composition graph.
          "joint": False, "joint_rounds": 8, "joint_separation": 0.5, "joint_minority": 0.05,
          "joint_max_seeds": 30}
_ORIGINAL_COMPILE = _grammar_mod.compile_grammar


def element_keys(mem, s: np.ndarray, V: int, tok_index: Dict[str, int]) -> np.ndarray:
    """Each element's composition key under symbols ``s``: its token for a
    primitive, (left symbol, right symbol) for a composite, as one integer."""
    K = int(s.max()) + 1
    prim = np.array([k == mem.PRIMITIVE for k in mem.kind])
    left, right = np.array(mem.left), np.array(mem.right)
    tok = np.array([tok_index.get(t, -1) if t is not None else -1 for t in mem.token])
    key = np.where(prim, tok, V + s[np.maximum(left, 0)] * K + s[np.maximum(right, 0)])
    return key.astype(np.int64)


def plain_code(mem, s: np.ndarray, key: np.ndarray, V: int, alpha: float) -> float:
    """Plain-PCFG code (nats) of the elements under per-element symbols ``s``:
    each symbol's row of outcomes (keys), and the domain's top level."""
    K = int(s.max()) + 1
    w = np.asarray(mem.weight, dtype=float)
    return dm_code(s, key, w, V + K * K, alpha) + mem.top_level_nats(s, K, alpha)


def contexts_of(mem, s: np.ndarray, key: np.ndarray) -> List[List[Tuple]]:
    """Each element's composition context: (sibling key, side) and (parent
    symbol, side)."""
    parent = np.asarray(mem.parent)
    left, right = np.asarray(mem.left), np.asarray(mem.right)
    out = []
    for e in range(len(s)):
        p = parent[e]
        if p < 0:
            out.append([("root",)])
            continue
        is_left = left[p] == e
        sib = right[p] if is_left else left[p]
        out.append([("sib", int(key[sib]), int(is_left)), ("par", int(s[p]), int(is_left))])
    return out


def propose(A: int, s: np.ndarray, key: np.ndarray, ctx, w) -> List[np.ndarray]:
    """Candidate 2-way splits of symbol A's elements (boolean masks over all
    elements): the sign of each of the leading principal directions of the
    units' context profiles (count-weighted and unweighted), each refined by
    2-means."""
    els = np.flatnonzero(s == A)
    if els.size < 4:
        return []
    units, unit_of = np.unique(key[els], return_inverse=True)
    if units.size < CONFIG["min_units"]:
        return []
    feat: Dict[Tuple, int] = {}
    rows, cols, vals = [], [], []
    for u, e in zip(unit_of, els):
        for f in ctx[e]:
            j = feat.setdefault(f, len(feat))
            rows.append(u)
            cols.append(j)
            vals.append(w[e])
    if len(feat) < 2:
        return []
    X = np.zeros((units.size, len(feat)))
    np.add.at(X, (rows, cols), vals)
    n = X.sum(axis=1)
    P = X / np.maximum(n[:, None], 1e-12)
    labels = []
    for wt in (np.sqrt(n), np.ones_like(n)):
        mu = (wt[:, None] ** 2 * P).sum(0) / (wt ** 2).sum()
        try:
            _, _, vt = np.linalg.svd((P - mu) * wt[:, None], full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        for v in vt[:CONFIG["components"]]:
            lab = (P - mu) @ v > 0
            if lab.all() or (~lab).all():
                continue
            for _ in range(20):                     # 2-means on the profiles, count-weighted
                c1 = (n[lab, None] * P[lab]).sum(0) / n[lab].sum()
                c0 = (n[~lab, None] * P[~lab]).sum(0) / n[~lab].sum()
                new = ((P - c1) ** 2).sum(1) < ((P - c0) ** 2).sum(1)
                if new.all() or (~new).all() or np.array_equal(new, lab):
                    break
                lab = new
            if lab.all() or (~lab).all():
                continue
            canon = lab if lab[0] else ~lab
            if not any(np.array_equal(canon, x) for x in labels):
                labels.append(canon)
    masks = []
    for lab in labels:
        mask = np.zeros(len(s), dtype=bool)
        mask[els[lab[unit_of]]] = True
        masks.append(mask)
    return masks


class Graph:
    """The composition graph of the stored elements: each element's parent,
    sibling (and side) and children."""

    def __init__(self, mem):
        self.parent = np.asarray(mem.parent, dtype=np.int64)
        self.left = np.asarray(mem.left, dtype=np.int64)
        self.right = np.asarray(mem.right, dtype=np.int64)
        self.prim = np.array([k == mem.PRIMITIVE for k in mem.kind])
        p = self.parent
        has = p >= 0
        pp = np.where(has, p, 0)
        self.is_left = has & (self.left[pp] == np.arange(len(p)))
        self.sib = np.where(has, np.where(self.is_left, self.right[pp], self.left[pp]), -1)
        # (relation, neighbour) pairs: parent; sibling on the right (I am
        # left); sibling on the left (I am right); left child; right child.
        self.rel = [np.where(has, p, -1),
                    np.where(self.is_left, self.sib, -1),
                    np.where(has & ~self.is_left, self.sib, -1),
                    np.where(self.prim, -1, self.left),
                    np.where(self.prim, -1, self.right)]


def _two_means(P: np.ndarray, n: np.ndarray) -> Optional[np.ndarray]:
    wt = np.sqrt(n)
    mu = (wt[:, None] ** 2 * P).sum(0) / max((wt ** 2).sum(), 1e-12)
    try:
        _, _, vt = np.linalg.svd((P - mu) * wt[:, None], full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    lab = (P - mu) @ vt[0] > 0
    for _ in range(20):
        if lab.all() or (~lab).all():
            return None
        c1 = (n[lab, None] * P[lab]).sum(0) / n[lab].sum()
        c0 = (n[~lab, None] * P[~lab]).sum(0) / n[~lab].sum()
        new = ((P - c1) ** 2).sum(1) < ((P - c0) ** 2).sum(1)
        if np.array_equal(new, lab):
            break
        lab = new
    if lab.all() or (~lab).all():
        return None
    c1 = (n[lab, None] * P[lab]).sum(0) / n[lab].sum()
    c0 = (n[~lab, None] * P[~lab]).sum(0) / n[~lab].sum()
    minority = min(n[lab].sum(), n[~lab].sum()) / n.sum()
    if np.abs(c1 - c0).sum() < CONFIG["joint_separation"] or minority < CONFIG["joint_minority"]:
        return None
    return lab


def propagate(A: int, seed_mask: np.ndarray, s: np.ndarray, key: np.ndarray, graph: Graph) -> Optional[np.ndarray]:
    """A coordinated split seeded by splitting symbol A by ``seed_mask``: every
    other symbol is split by 2-means on its units' profiles of neighbour
    labels (parent, siblings, children), iterated to a fixed point. Returns
    the new symbols (or None if nothing beyond the seed splits)."""
    E = len(s)
    z = np.full(E, -1, dtype=np.int64)
    inA = s == A
    z[inA] = seed_mask[inA].astype(np.int64)
    K = int(s.max()) + 1
    for _ in range(CONFIG["joint_rounds"]):
        F = np.zeros((E, 10))
        for r, nb in enumerate(graph.rel):
            ok = (nb >= 0)
            ok[ok] = z[nb[ok]] >= 0
            idx = np.flatnonzero(ok)
            np.add.at(F, (idx, 2 * r + z[nb[idx]]), 1.0)
        zl = np.where(graph.prim, -2, z[np.maximum(graph.left, 0)])
        zr = np.where(graph.prim, -2, z[np.maximum(graph.right, 0)])
        unit_rows = np.stack([s, key, zl, zr], axis=1)
        _, unit = np.unique(unit_rows, axis=0, return_inverse=True)
        unit = unit.reshape(-1)
        new = z.copy()
        for B in range(K):
            if B == A:
                continue
            eb = np.flatnonzero(s == B)
            if eb.size < 4:
                continue
            ub, inv = np.unique(unit[eb], return_inverse=True)
            X = np.zeros((ub.size, 10))
            np.add.at(X, inv, F[eb])
            n = X.sum(1)
            has = n > 0
            if has.sum() < 2:
                new[eb] = -1
                continue
            lab = _two_means(X[has] / n[has, None], n[has])
            if lab is None:
                new[eb] = -1
                continue
            ul = np.full(ub.size, -1)
            ul[np.flatnonzero(has)] = lab.astype(np.int64)
            new[eb] = ul[inv]
        if np.array_equal(new, z):
            break
        z = new
    out = s.copy()
    nxt = K
    split_any = False
    for B in range(K):
        m = s == B
        if B == A:
            out[m & seed_mask] = nxt
            nxt += 1
            continue
        if (z[m] == 1).any() and (z[m] == 0).any():
            out[m & (z == 1)] = nxt
            nxt += 1
            split_any = True
    return out if split_any else None


def prune(s_old: np.ndarray, s_new: np.ndarray, mem, V: int, tok_index, alpha: float,
          keep: Optional[int] = None) -> np.ndarray:
    """Merge back, one at a time, each part a joint split created (except the
    seed's, ``keep``) while that lowers the exact plain-PCFG code: the
    propagation may split symbols whose context does not depend on the seed's
    distinction, and such splits only cost bits."""
    s_new = s_new.copy()
    while True:
        key = element_keys(mem, s_new, V, tok_index)
        base = plain_code(mem, s_new, key, V, alpha)
        best = None
        created = [int(x) for x in np.unique(s_new[s_new != s_old])]
        for x in created:
            m = s_new == x
            orig = int(np.bincount(s_old[m]).argmax())
            if keep is not None and orig == keep:
                continue
            trial = np.where(m, orig, s_new)
            _, trial = np.unique(trial, return_inverse=True)
            trial = trial.reshape(-1)
            d = plain_code(mem, trial, element_keys(mem, trial, V, tok_index), V, alpha) - base
            if d < -1e-9 and (best is None or d < best[0]):
                best = (d, trial)
        if best is None:
            return s_new
        s_new = best[1]


def split_symbols(mem, s: np.ndarray, alpha: float, seed: int, stage_b, log=None):
    """Greedy split moves while the full code shrinks. ``stage_b(s)`` refits
    and returns a grammar whose info['total bits'] is the full code."""
    from trellis2.grammar import UNK
    vocab = mem.vocabulary() + [UNK]
    tok_index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    w = np.asarray(mem.weight, dtype=float)
    _, s = _compact(np.asarray(s, dtype=np.int64))
    g = stage_b(s)
    best_bits = g.info["total bits"]
    history = [{"step": -1, "unsplit stage B total bits": best_bits}]
    rejected = set()
    graph = Graph(mem) if CONFIG["joint"] else None
    for step in range(CONFIG["max_splits"]):
        t_step = time.time()
        key = element_keys(mem, s, V, tok_index)
        base_plain = plain_code(mem, s, key, V, alpha)
        ctx = contexts_of(mem, s, key)
        K = int(s.max()) + 1
        cands = []
        for A in range(K):
            for mask in propose(A, s, key, ctx, w):
                sig = (A, int(mask.sum()), int(np.flatnonzero(mask)[:1].sum()))
                if sig in rejected:
                    continue
                s2 = np.where(mask, K, s)
                key2 = element_keys(mem, s2, V, tok_index)
                d = plain_code(mem, s2, key2, V, alpha) - base_plain
                cands.append((d, A, s2, sig, "single", mask))
        if graph is not None:
            # Joint splits seeded by the best single candidates (plain code).
            seeds = sorted(cands, key=lambda x: x[0])[:CONFIG["joint_max_seeds"]]
            for d, A, _, sig, _, mask in seeds:
                if sig + ("joint",) in rejected:
                    continue
                sj = propagate(A, mask, s, key, graph)
                if sj is not None and CONFIG.get("prune", True):
                    sj = prune(s, sj, mem, V, tok_index, alpha, keep=A)
                    if np.array_equal(sj, s) or len(np.unique(s[sj != s])) < 2:
                        sj = None
                if sj is not None:
                    keyj = element_keys(mem, sj, V, tok_index)
                    dj = plain_code(mem, sj, keyj, V, alpha) - base_plain
                    cands.append((dj, A, sj, sig + ("joint",), "joint", None))
        cands = [c[:5] for c in cands]
        cands.sort(key=lambda x: x[0])
        # The best few of each kind are scored exactly (joint splits' plain
        # codes can crowd out every single split).
        k = CONFIG["candidates"]
        cands = sorted([c for c in cands if c[4] == "single"][:k] + [c for c in cands if c[4] == "joint"][:k],
                       key=lambda x: x[0])
        propose_seconds = time.time() - t_step
        accepted = False
        for d, A, s2, sig, kind in cands:
            mask = s2 != s
            g2 = stage_b(s2)
            delta = g2.info["total bits"] - best_bits
            history.append({"step": step, "symbol": A, "kind": kind, "elements moved": int(mask.sum()),
                            "symbols split": int(len(np.unique(s[mask]))),
                            "plain delta (bits)": d / np.log(2), "full delta (bits)": delta,
                            "proposal seconds": propose_seconds, "step seconds": time.time() - t_step})
            if log:
                log(history[-1])
            if delta < -1e-6:
                def kinds(m):
                    seen = []
                    for e in np.flatnonzero(m):
                        d_ = mem.describe(int(e))
                        if d_ not in seen:
                            seen.append(d_)
                        if len(seen) == 6:
                            break
                    return seen
                history[-1]["moved"] = kinds(mask)
                history[-1]["kept"] = kinds((s == A) & ~mask)
                history[-1]["split symbols"] = [kinds((s == B) & mask)[:4] for B in np.unique(s[mask])][:12]
                if log:
                    log({"accepted": kind, "symbols split": history[-1]["symbols split"],
                     "moved": history[-1]["moved"], "from": history[-1]["kept"]})
                s, g, best_bits, accepted = s2, g2, g2.info["total bits"], True
                break
            rejected.add(sig)
        if not accepted:
            break
    return s, g, history


def compile_grammar_split(mem, rtree, leaves, alpha: float = 0.001, seed: int = 0, search: bool = True,
                          merge: bool = True, **kw):
    """``compile_grammar``, then split moves on its symbols (exact full code),
    with or without heads (``CONFIG['heads']``)."""
    t0 = time.time()
    g0 = _ORIGINAL_COMPILE(mem, rtree, leaves, alpha=alpha, seed=seed, search=search, merge=merge, **kw)
    if CONFIG["heads"]:
        stage_b = lambda s: H.compile_with_symbols_heads(mem, s, alpha=alpha, seed=seed, search=search,  # noqa: E731
                                                         side=H.CONFIG["side"], betas=H.CONFIG["betas"],
                                                         passes=H.CONFIG["passes"])
    else:
        stage_b = lambda s: H.compile_with_symbols_heads(mem, s, alpha=alpha, seed=seed, search=search,  # noqa: E731
                                                         betas=(), passes=1)
    s, g, history = split_symbols(mem, g0.elem_symbol, alpha, seed, stage_b,
                                  log=(lambda h: print("    split", h, flush=True)) if CONFIG["log"] else None)
    # Fields stage A gives (per element; the representation nodes per
    # symbol are those of the unsplit symbol).
    g.elem_fine, g.elem_evidence, g.elem_depth = g0.elem_fine, g0.elem_evidence, g0.elem_depth
    parent_sym = {}
    for e, (a, b) in enumerate(zip(g.elem_symbol, g0.elem_symbol)):
        parent_sym.setdefault(int(a), int(b))
    g.symbol_nodes = [g0.symbol_nodes[parent_sym[k]] if parent_sym.get(k, -1) < len(g0.symbol_nodes) else []
                      for k in range(g.K)]
    yields = [defaultdict(float) for _ in range(g.K)]
    for e in range(len(g.elem_symbol)):
        yields[int(g.elem_symbol[e])][mem.describe(e)] += mem.weight[e]
    from collections import Counter
    g.symbol_yields = [Counter(y) for y in yields]
    g.info["unsplit total bits"] = g0.info["total bits"]
    g.info["unsplit stage B total bits"] = history[0]["unsplit stage B total bits"]
    g.info["splits accepted"] = sum(1 for h in history[1:] if h["full delta (bits)"] < -1e-6)
    g.info["split history"] = history
    g.info["chunk types"] = len({(int(g.elem_symbol[e]), int(g.elem_symbol[mem.left[e]]), int(g.elem_symbol[mem.right[e]]))
                                 for e in range(len(mem)) if mem.kind[e] != mem.PRIMITIVE})
    g.info["split seconds"] = time.time() - t0
    return g


def install_splits(heads: bool = False, representation_heads: bool = False) -> List[str]:
    """Substitute ``trellis2.model.compile_grammar`` (and, with heads, the
    names ``heads.install`` substitutes) in this process."""
    names = []
    CONFIG["heads"] = heads
    if heads or representation_heads:
        names += H.install(representation=representation_heads, composition=True, side="learned")
    else:
        # The charts must still dispatch on HeadGrammar (stage B returns one).
        names += H.install(representation=False, composition=True, side="learned")
    _model_mod.compile_grammar = compile_grammar_split
    names = [n for n in names if n != "trellis2.model.compile_grammar"] + \
        ["trellis2.model.compile_grammar (splits.compile_grammar_split)"]
    return names


# --------------------------------------------------------------------------- #
# Final-round refinement: splits only on the final grammar

def refine(model, joint: bool, alpha: Optional[float] = None, seed: Optional[int] = None):
    """Split moves applied once to a consolidated model's grammar (the rounds
    before it are the baseline's): the model's grammar is replaced by the
    refined one if the full code shrinks (split_symbols only accepts such)."""
    g0 = model.grammar
    mem = model.memory
    alpha = model.alpha if alpha is None else alpha
    seed = model.seed if seed is None else seed
    CONFIG["joint"] = joint
    stage_b = lambda s: H.compile_with_symbols_heads(mem, s, alpha=alpha, seed=seed, search=True,  # noqa: E731
                                                     betas=(), passes=1)
    t0 = time.time()
    s, g, history = split_symbols(mem, g0.elem_symbol, alpha, seed, stage_b,
                                  log=(lambda h: print("    split", h, flush=True)) if CONFIG["log"] else None)
    accepted = sum(1 for h in history[1:] if h.get("full delta (bits)", 0) < -1e-6)
    info = {"refine: baseline total bits": g0.info["total bits"],
            "refine: stage B on unsplit symbols": history[0]["unsplit stage B total bits"],
            "refine: splits accepted": accepted,
            "refine: joint accepted": sum(1 for h in history[1:] if h.get("kind") == "joint"
                                          and h.get("full delta (bits)", 0) < -1e-6),
            "refine: seconds": time.time() - t0}
    if accepted:
        g.symbol_yields = [defaultdict(float) for _ in range(g.K)]
        from collections import Counter
        ys = [Counter() for _ in range(g.K)]
        for e in range(len(g.elem_symbol)):
            ys[int(g.elem_symbol[e])][mem.describe(e)] += mem.weight[e]
        g.symbol_yields = ys
        g.elem_fine, g.elem_evidence, g.elem_depth = g0.elem_fine, g0.elem_evidence, g0.elem_depth
        g.info["chunk types"] = len({(int(g.elem_symbol[e]), int(g.elem_symbol[mem.left[e]]),
                                      int(g.elem_symbol[mem.right[e]]))
                                     for e in range(len(mem)) if mem.kind[e] != mem.PRIMITIVE})
        g.info.update(info)
        model._grammar = g
    else:
        g0.info.update(info)
    return model.grammar


def install_final(joint: bool, where: str = "consolidate") -> List[str]:
    """Final-round refinement in this process: after ``Trellis2.consolidate``
    (models learned from given analyses) or after ``UnsupervisedLearner.sleep``
    (the night's final model; the night's own consolidations stay baseline)."""
    names = H.install(representation=False, composition=True, side="learned")   # charts dispatch only
    import trellis2.model as M
    M.compile_grammar = H._ORIGINAL_COMPILE if False else _ORIGINAL_COMPILE
    names = [n for n in names if n != "trellis2.model.compile_grammar"]
    if where == "consolidate":
        orig = M.Trellis2.consolidate

        def consolidate(self, init_labels=None):
            orig(self, init_labels)
            return refine(self, joint)
        M.Trellis2.consolidate = consolidate
        names.append("trellis2.model.Trellis2.consolidate (+ splits.refine)")
    else:
        import trellis2.unsupervised as U
        orig = U.UnsupervisedLearner.sleep

        def sleep(self):
            orig(self)
            refine(self.model, joint)
            self.analyses = self._in_categories(self.model, self.trees)
            return self.model.grammar
        U.UnsupervisedLearner.sleep = sleep
        names.append("trellis2.unsupervised.UnsupervisedLearner.sleep (+ splits.refine on the final model)")
    return names
