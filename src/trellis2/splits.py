"""Split moves: refining categories by description length, at the end of a
consolidation (the second coordinated move, beside the joint chunk move).

The structure search can only merge categories, and the representation cut
groups elements by how they behave alone; a distinction that pays only when
several categories make it together (grammatical number: determiners, nouns,
noun phrases and verbs at once) is out of reach of either. A split move
refines the grammar's symbols after the cut, scored by the exact full code:

* single split: one symbol's elements in two groups. Units are the symbol's
  elements grouped by their own composition key (a primitive's token; a
  composite's two part symbols and relation), so every occurrence of a type
  stays together; a unit's profile is its composition context: the
  sibling's composition key and side, and the parent's symbol, relation and
  side (the categories it combines with, never its own attributes).
  Candidate groups: the signs of the leading principal directions of the
  profiles (count-weighted and not), each refined by 2-means.
* joint split: a single split propagated through the composition graph.
  Every other symbol's units are profiled by the labels of their neighbours
  (parent, siblings, children) and split by 2-means, to a fixed point; the
  parts the propagation creates are then merged back one at a time while
  that shortens the exact plain code (pruning the splits that only cost).

Candidates are ranked by the exact plain-PCFG code (cheap); the best few of
each kind are scored by the full code (``grammar.compile_with_symbols``: the
composition hierarchy, its cut and the tables, exactly as ``compile_grammar``
builds them) and a split is kept only if the full code shrinks; repeated
while one pays. ``refine`` applies the moves once, to a consolidated
grammar: earlier rounds are never changed (refining every round lets the
rounds diverge, which cost characters' generation). Nothing here knows the
domain.

Interface (for reuse as a local move, e.g. by an incremental learner):

* ``refine(mem, grammar, alpha, seed, search=True, config=None, log=None)
  -> Grammar``: the whole move. ``mem`` is any domain's ``Memory`` holding
  the recorded elements; ``grammar`` a grammar compiled from it (its
  ``elem_symbol``: one symbol per element). Returns a new grammar whose
  ``info["total bits"]`` is strictly shorter, or ``grammar`` itself
  (``info["splits: accepted"] == 0``). The new grammar's info holds
  ``splits: accepted / joint accepted / history / seconds``.
* ``split_symbols(mem, symbols, score, alpha, config, log) -> (symbols,
  grammar, history)``: the greedy loop on any scoring function
  ``score(symbols) -> Grammar`` (``info["total bits"]`` is the code to
  shorten); ``refine`` passes ``grammar.compile_with_symbols``.
* ``grammar.compile_with_symbols(mem, symbols, alpha, seed, search)``: the
  exact full code (and grammar) of any symbol assignment: stage B of
  ``compile_grammar`` (the composition hierarchy, its cut, the tables), so
  for the symbols ``compile_grammar`` chose it gives that grammar's code
  exactly. Symbols are renumbered in increasing order.
* ``_Context(mem)``: the elements as arrays; ``.keys(s)`` the composition
  keys, ``.plain_code(s)`` the exact plain-PCFG code (nats) used to rank
  candidates. ``propose(ctx, A, s, key, config)`` single-split masks for
  symbol A; ``propagate(ctx, A, mask, s, key, config)`` a joint split (new
  symbols numbered after the old) or None; ``prune(ctx, s_old, s_new,
  alpha, keep)`` the propagated parts that pay.
* ``DEFAULT_CONFIG``: the caps (candidates scored exactly per step and kind,
  seeds of joint splits, rounds, ...); every entry can be overridden by
  ``config``. Cost: about one exact refit (``compile_with_symbols``) per
  candidate scored, i.e. up to ``2 * candidates`` per accepted split.

Options: ``Trellis2(joint_splits=True)`` refines after ``consolidate``;
``UnsupervisedLearner(joint_splits=True)`` (the default) refines the
night's chosen grammar at its end.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grammar import UNK, Grammar, _compact, compile_with_symbols
from .mdl import dm_code

DEFAULT_CONFIG = {
    "candidates": 5,         # scored exactly per step, of each kind (single, joint)
    "max_splits": 40,
    "min_units": 2,
    "components": 3,         # principal directions tried per symbol
    "joint": True,
    "joint_max_seeds": 40,   # single candidates (best plain code) that seed a joint split
                             # (40/5 against 20/3: agree +4 to +9 points grammatical at 2-3x the cost;
                             # 60/8 adds nothing)
    "joint_rounds": 8,
    "joint_separation": 0.5,  # L1 distance between the 2-means centroids
    "joint_minority": 0.05,
}


class _Context:
    """The memory's elements as arrays, for one symbol assignment at a time."""

    def __init__(self, mem):
        self.mem = mem
        vocab = mem.vocabulary() + [UNK]
        self.tok_index = {t: i for i, t in enumerate(vocab)}
        self.V = len(vocab)
        self.prim = np.array([k == mem.PRIMITIVE for k in mem.kind])
        self.tok = np.array([self.tok_index.get(t, -1) if t is not None else -1 for t in mem.token], dtype=np.int64)
        self.left = np.asarray(mem.left, dtype=np.int64)
        self.right = np.asarray(mem.right, dtype=np.int64)
        self.parent = np.asarray(mem.parent, dtype=np.int64)
        relations = getattr(mem, "relations", None)
        rel_index = {r: i for i, r in enumerate(relations or [])}
        self.rel = np.array([rel_index.get(r, 0) for r in mem.relation], dtype=np.int64)
        self.n_rel = max(len(rel_index), 1)
        self.w = np.asarray(mem.weight, dtype=float)
        has = self.parent >= 0
        pp = np.where(has, self.parent, 0)
        self.is_left = has & (self.left[pp] == np.arange(len(pp)))
        self.sib = np.where(has, np.where(self.is_left, self.right[pp], self.left[pp]), -1)
        # (relation, neighbour): parent; sibling on the right (I am left);
        # sibling on the left (I am right); left part; right part.
        self.neighbours = [np.where(has, self.parent, -1),
                           np.where(self.is_left, self.sib, -1),
                           np.where(has & ~self.is_left, self.sib, -1),
                           np.where(self.prim, -1, self.left),
                           np.where(self.prim, -1, self.right)]

    def keys(self, s: np.ndarray) -> np.ndarray:
        """Each element's composition key under symbols ``s``, as an integer:
        its token, or V + (left symbol, right symbol, relation)."""
        K = int(s.max()) + 1
        l, r = s[np.maximum(self.left, 0)], s[np.maximum(self.right, 0)]
        return np.where(self.prim, self.tok, self.V + (l * K + r) * self.n_rel + self.rel).astype(np.int64)

    def plain_code(self, s: np.ndarray, key: Optional[np.ndarray] = None, alpha: float = 0.001) -> float:
        """The plain-PCFG code (nats) of the elements under symbols ``s``: one
        Dirichlet-multinomial row of outcomes per symbol, and the domain's
        top level (the code ``compile_grammar`` cuts the representation
        hierarchy by)."""
        K = int(s.max()) + 1
        key = self.keys(s) if key is None else key
        return dm_code(s, key, self.w, self.V + K * K * self.n_rel, alpha) + self.mem.top_level_nats(s, K, alpha)


def _two_means(P: np.ndarray, n: np.ndarray, start: np.ndarray) -> Optional[np.ndarray]:
    lab = start
    for _ in range(20):
        if lab.all() or (~lab).all():
            return None
        c1 = (n[lab, None] * P[lab]).sum(0) / n[lab].sum()
        c0 = (n[~lab, None] * P[~lab]).sum(0) / n[~lab].sum()
        new = ((P - c1) ** 2).sum(1) < ((P - c0) ** 2).sum(1)
        if np.array_equal(new, lab):
            break
        lab = new
    return None if lab.all() or (~lab).all() else lab


def _leading(P: np.ndarray, wt: np.ndarray, k: int) -> List[np.ndarray]:
    mu = (wt[:, None] ** 2 * P).sum(0) / max((wt ** 2).sum(), 1e-12)
    try:
        _, _, vt = np.linalg.svd((P - mu) * wt[:, None], full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    return [(P - mu) @ v > 0 for v in vt[:k]]


def propose(ctx: _Context, A: int, s: np.ndarray, key: np.ndarray, config: dict) -> List[np.ndarray]:
    """Candidate 2-way splits of symbol A (boolean masks over all elements)."""
    els = np.flatnonzero(s == A)
    if els.size < 4:
        return []
    units, unit_of = np.unique(key[els], return_inverse=True)
    if units.size < config["min_units"]:
        return []
    feats: Dict[Tuple, int] = {}
    rows, cols = [], []
    for u, e in zip(unit_of, els):
        p = ctx.parent[e]
        if p < 0:
            fs = [("root",)]
        else:
            side = int(ctx.is_left[e])
            fs = [("sib", int(key[ctx.sib[e]]), side), ("par", int(s[p]), int(ctx.rel[p]), side)]
        for f in fs:
            rows.append(u)
            cols.append(feats.setdefault(f, len(feats)))
    if len(feats) < 2:
        return []
    X = np.zeros((units.size, len(feats)))
    np.add.at(X, (rows, cols), ctx.w[els][np.repeat(np.arange(len(els)), [1 if ctx.parent[e] < 0 else 2 for e in els])])
    n = X.sum(1)
    P = X / np.maximum(n[:, None], 1e-12)
    labels = []
    for wt in (np.sqrt(n), np.ones_like(n)):
        for start in _leading(P, wt, config["components"]):
            lab = _two_means(P, n, start)
            if lab is None:
                continue
            canon = lab if lab[0] else ~lab
            if not any(np.array_equal(canon, x) for x in labels):
                labels.append(canon)
    out = []
    for lab in labels:
        mask = np.zeros(len(s), dtype=bool)
        mask[els[lab[unit_of]]] = True
        out.append(mask)
    return out


def propagate(ctx: _Context, A: int, seed_mask: np.ndarray, s: np.ndarray, key: np.ndarray,
              config: dict) -> Optional[np.ndarray]:
    """A coordinated split seeded by splitting symbol A by ``seed_mask``:
    every other symbol split by 2-means on its units' profiles of neighbour
    labels, iterated to a fixed point. Returns the new symbols (new ones
    numbered after the old), or None if nothing beyond the seed splits."""
    E = len(s)
    z = np.full(E, -1, dtype=np.int64)
    inA = s == A
    z[inA] = seed_mask[inA].astype(np.int64)
    K = int(s.max()) + 1
    for _ in range(config["joint_rounds"]):
        F = np.zeros((E, 2 * len(ctx.neighbours)))
        for r, nb in enumerate(ctx.neighbours):
            ok = nb >= 0
            ok[ok] = z[nb[ok]] >= 0
            idx = np.flatnonzero(ok)
            np.add.at(F, (idx, 2 * r + z[nb[idx]]), ctx.w[idx])
        zl = np.where(ctx.prim, -2, z[np.maximum(ctx.left, 0)])
        zr = np.where(ctx.prim, -2, z[np.maximum(ctx.right, 0)])
        _, unit = np.unique(np.stack([s, key, zl, zr], axis=1), axis=0, return_inverse=True)
        unit = unit.reshape(-1)
        new = z.copy()
        for B in range(K):
            if B == A:
                continue
            eb = np.flatnonzero(s == B)
            if eb.size < 4:
                continue
            ub, inv = np.unique(unit[eb], return_inverse=True)
            X = np.zeros((ub.size, F.shape[1]))
            np.add.at(X, inv, F[eb])
            n = X.sum(1)
            has = n > 0
            new[eb] = -1
            if has.sum() < 2:
                continue
            P, nn = X[has] / n[has, None], n[has]
            starts = _leading(P, np.sqrt(nn), 1)
            lab = _two_means(P, nn, starts[0]) if starts else None
            if lab is None:
                continue
            c1 = (nn[lab, None] * P[lab]).sum(0) / nn[lab].sum()
            c0 = (nn[~lab, None] * P[~lab]).sum(0) / nn[~lab].sum()
            if (np.abs(c1 - c0).sum() < config["joint_separation"]
                    or min(nn[lab].sum(), nn[~lab].sum()) / nn.sum() < config["joint_minority"]):
                continue
            ul = np.full(ub.size, -1)
            ul[np.flatnonzero(has)] = lab.astype(np.int64)
            new[eb] = ul[inv]
        if np.array_equal(new, z):
            break
        z = new
    out = s.copy()
    nxt, split_any = K, False
    for B in range(K):
        m = s == B
        if B == A:
            out[m & seed_mask] = nxt
            nxt += 1
        elif (z[m] == 1).any() and (z[m] == 0).any():
            out[m & (z == 1)] = nxt
            nxt += 1
            split_any = True
    return out if split_any else None


def prune(ctx: _Context, s_old: np.ndarray, s_new: np.ndarray, alpha: float, keep: Optional[int] = None) -> np.ndarray:
    """Merge back, one at a time, each part a joint split created (except the
    seed's) while that shortens the exact plain code."""
    K0 = int(s_old.max()) + 1
    s_new = s_new.copy()
    while True:
        base = ctx.plain_code(s_new, alpha=alpha)
        best = None
        for x in [int(v) for v in np.unique(s_new[s_new >= K0])]:
            m = s_new == x
            orig = int(np.bincount(s_old[m]).argmax())
            if keep is not None and orig == keep:
                continue
            trial = np.where(m, orig, s_new)
            trial = np.where(trial > x, trial - 1, trial)      # keep new symbols contiguous
            d = ctx.plain_code(trial, alpha=alpha) - base
            if d < -1e-9 and (best is None or d < best[0]):
                best = (d, trial)
        if best is None:
            return s_new
        s_new = best[1]


def split_symbols(mem, symbols: np.ndarray, score, alpha: float = 0.001, config: Optional[dict] = None,
                  log=None):
    """Greedy split moves while the full code shrinks. ``score(s)`` returns the
    grammar for symbols ``s`` (its info['total bits'] is the full code).
    Returns (symbols, grammar, history)."""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    ctx = _Context(mem)
    _, s = _compact(np.asarray(symbols, dtype=np.int64))
    g = score(s)
    best_bits = g.info["total bits"]
    history: List[dict] = [{"step": -1, "total bits": best_bits}]
    rejected = set()
    for step in range(cfg["max_splits"]):
        t0 = time.time()
        key = ctx.keys(s)
        base = ctx.plain_code(s, key, alpha)
        K = int(s.max()) + 1
        single = []
        for A in range(K):
            for mask in propose(ctx, A, s, key, cfg):
                sig = (A, int(mask.sum()), int(np.flatnonzero(mask)[0]))
                if sig in rejected:
                    continue
                s2 = np.where(mask, K, s)
                single.append((ctx.plain_code(s2, alpha=alpha) - base, A, s2, sig, "single", mask))
        single.sort(key=lambda c: c[0])
        joint = []
        if cfg["joint"]:
            for d, A, _, sig, _, mask in single[:cfg["joint_max_seeds"]]:
                if sig + ("joint",) in rejected:
                    continue
                sj = propagate(ctx, A, mask, s, key, cfg)
                if sj is not None:
                    sj = prune(ctx, s, sj, alpha, keep=A)
                    if len(np.unique(s[sj != s])) < 2:
                        continue
                    joint.append((ctx.plain_code(sj, alpha=alpha) - base, A, sj, sig + ("joint",), "joint", None))
            joint.sort(key=lambda c: c[0])
        cands = sorted(single[:cfg["candidates"]] + joint[:cfg["candidates"]], key=lambda c: c[0])
        accepted = False
        for d, A, s2, sig, kind, _ in cands:
            g2 = score(s2)
            delta = g2.info["total bits"] - best_bits
            moved = s2 != s
            history.append({"step": step, "kind": kind, "symbol": int(A), "symbols split": int(len(np.unique(s[moved]))),
                            "elements moved": int(moved.sum()), "plain delta (bits)": float(d / np.log(2)),
                            "full delta (bits)": float(delta), "seconds": time.time() - t0})
            if log:
                log(history[-1])
            if delta < -1e-6:
                s, g, best_bits, accepted = s2, g2, g2.info["total bits"], True
                break
            rejected.add(sig)
        if not accepted:
            break
    return s, g, history


def refine(mem, grammar: Grammar, alpha: float = 0.001, seed: int = 0, search: bool = True,
           config: Optional[dict] = None, log=None) -> Grammar:
    """Split moves applied once to a consolidated grammar (its symbols are
    the starting point; the rounds that made it are not changed). Returns the
    refined grammar if a split shortened the full code, else ``grammar``."""
    t0 = time.time()
    s, g, history = split_symbols(
        mem, grammar.elem_symbol, lambda x: compile_with_symbols(mem, x, alpha=alpha, seed=seed, search=search),
        alpha=alpha, config=config, log=log)
    accepted = [h for h in history[1:] if h["full delta (bits)"] < -1e-6]
    summary = {"splits: accepted": len(accepted),
               "splits: joint accepted": sum(h["kind"] == "joint" for h in accepted),
               "splits: unsplit total bits": grammar.info["total bits"],
               "splits: unsplit total bits (recomputed)": history[0]["total bits"],
               "splits: seconds": time.time() - t0}
    if not accepted or g.info["total bits"] >= grammar.info["total bits"] - 1e-6:
        grammar.info.update(summary)
        grammar.info["splits: accepted"] = 0
        return grammar
    # Representation-side fields come from the unsplit grammar: each new
    # symbol keeps the nodes of the symbol it was split from.
    origin: Dict[int, int] = {}
    for new, old in zip(g.elem_symbol.tolist(), grammar.elem_symbol.tolist()):
        origin.setdefault(int(new), int(old))
    g.symbol_nodes = [grammar.symbol_nodes[origin[k]] if origin.get(k, -1) < len(grammar.symbol_nodes) else []
                      for k in range(g.K)]
    g.elem_fine, g.elem_evidence, g.elem_depth = grammar.elem_fine, grammar.elem_evidence, grammar.elem_depth
    yields = [Counter() for _ in range(g.K)]
    for e in range(len(g.elem_symbol)):
        yields[int(g.elem_symbol[e])][mem.describe(e)] += mem.weight[e]
    g.symbol_yields = yields
    g.info.update({k: v for k, v in grammar.info.items() if k.startswith("R ")})
    g.info.update(summary)
    g.info["splits: history"] = history
    return g
