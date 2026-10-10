"""Phase A of heads: why does grammatical number merge?

For a run of ``run_complex.py`` (grammar, size, seed) and either the night's
grammar (refitted from its stored analyses, ``grammar_night.pkl``) or the
gold-tree grammar (``Trellis2`` on the gold trees):

* A1 (representation): in the representation hierarchy of the final
  consolidation, how well can a cut separate singular from plural elements?
  For elements whose span is a gold constituent of a number-bearing category
  (nominal: Nb, NP; verbal: Vt, VP, CoVP, RC, Vc; determiners: Det), the
  purity by number of the grammar's chosen categories, against the best
  purity of any cut through the tree with the same number of nodes holding
  those elements (and twice as many, and the leaves).
* A2 (code): the chosen categories with every category that mixes number
  split into a singular and a plural part (oracle: an element's number is the
  gold number of the smallest gold constituent containing its span; elements
  without a number go to their category's majority part), the composition
  cut searched again and the tables refitted, with the code of
  ``compile_grammar``'s stage B (re-implemented here; it reproduces the
  unsplit grammar's code exactly). Reported: total bits split vs unsplit, and
  1,000 samples from each: grammatical and number-blind grammatical.

Nothing under ``src/`` is changed: ``trellis2.model.compile_grammar`` is
wrapped in this process only, to keep the representation tree it was given.

Usage:
    python experiments/v2/number_oracle.py --grammar agree --train 2000 --seed 13 --side night
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import trellis2.model as _model_mod  # noqa: E402
from trellis2 import Trellis2  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.grammar import (CONTEXT_WEIGHTS, COMPOSITION_ATTRS, LN2, UNK, Grammar, TreeIndex,  # noqa: E402
                              _compact, _composition_instance, _Elements, _factored_code, _pair_rows,
                              evidence_cut, search_cut)
from trellis2.cobweb import CobwebTree  # noqa: E402
from trellis2.mdl import elias_delta_bits, ml_row_nats, split_bits  # noqa: E402
from trellis2.memory import Memory  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402
import complex_grammars as CG  # noqa: E402

RESULTS = os.path.join(HERE, "results", "complex")
OUT = os.path.join(HERE, "results", "heads", "phase_a")

# Keep the representation tree and leaves each grammar was read off.
_compile = _model_mod.compile_grammar


def _compile_keep(mem, rtree, leaves, **kw):
    g = _compile(mem, rtree, leaves, **kw)
    g._rtree, g._leaves = rtree, leaves
    return g


_model_mod.compile_grammar = _compile_keep

GROUPS = {"nominal": ("Nb", "NP"), "verbal": ("Vt", "VP", "CoVP", "RC", "Vc"), "determiner": ("Det",)}


# --------------------------------------------------------------------------- #
# Models

def refit(grammar: str, n: int, seed: int, side: str):
    """(memory, grammar) of the night's final consolidation (refitted from its
    stored analyses) or of the gold-tree grammar; and the gold examples in
    memory order."""
    train_all, _ = CG.load(grammar)
    order = np.random.default_rng(seed).permutation(len(train_all))
    train = [train_all[i] for i in order[:n]]
    if side == "night":
        with open(os.path.join(RESULTS, f"{grammar}_{n}_s{seed}", "grammar_night.pkl"), "rb") as f:
            saved = pickle.load(f)
        assert saved["train"] == [e.sentence for e in train]
        learner = UnsupervisedLearner(seed=seed)
        for e in train:
            learner.observe(e.tokens)
        model = learner.fit(saved["trees"])
    else:
        model = Trellis2(seed=seed)
        for e in train:
            model.learn(e.tokens, e.tree)
        model.consolidate()
    return model.memory, model.grammar, train


# --------------------------------------------------------------------------- #
# Gold number of every element

def strip(label: str) -> str:
    return label[:-2] if label.endswith(("_s", "_p")) else label


def number_of(label: str) -> Optional[str]:
    return label[-1] if label.endswith(("_s", "_p")) else None


def element_gold(mem: Memory, train, cfg: CFG):
    """Per element: (gold category if its span is a gold constituent else None,
    gold number of the smallest gold constituent containing its span)."""
    gold_cat, gold_num = [], []
    labels = []
    for e in train:
        lab = cfg.gold_labels(e.tokens, e.tree)
        spans = sorted(lab, key=lambda s: s[1] - s[0])
        labels.append((lab, spans))
    for sid, (i, j) in zip(mem.experience_of, mem.span):
        lab, spans = labels[sid]
        gold_cat.append(lab.get((i, j)))
        cover = next(s for s in spans if s[0] <= i and j <= s[1])
        gold_num.append(number_of(lab[cover]))
    return gold_cat, gold_num


# --------------------------------------------------------------------------- #
# A1: purity by number of cuts through the representation hierarchy

def best_cut_purity(index: TreeIndex, leaf_s: np.ndarray, leaf_p: np.ndarray, budget: int) -> np.ndarray:
    """f[k] = the most elements classified correctly by number by a cut that
    uses at most k nodes holding elements of the group (nodes holding none are
    free), k = 0..budget."""
    n = len(index.nodes)
    s = np.zeros(n)
    p = np.zeros(n)
    s[index.node_of_leaf] = leaf_s
    p[index.node_of_leaf] = leaf_p
    for i in reversed(range(n)):            # children come after parents
        for c in index.children[i]:
            s[i] += s[c]
            p[i] += p[c]
    NEG = -1e18
    f: Dict[int, np.ndarray] = {}
    kk = np.arange(budget + 1)
    sums = kk[:, None] + kk[None, :]
    for i in reversed(range(n)):
        if s[i] + p[i] == 0:
            continue
        own = max(s[i], p[i])
        g = np.full(budget + 1, NEG)
        g[0] = 0.0
        for c in index.children[i]:
            if s[c] + p[c] == 0:
                continue
            fc = f.pop(c)
            m = g[:, None] + fc[None, :]
            out = np.full(2 * budget + 1, NEG)
            np.maximum.at(out, sums.ravel(), m.ravel())
            g = np.maximum.accumulate(out[:budget + 1])
        h = np.maximum(g, np.where(kk >= 1, own, NEG))
        h[0] = NEG
        f[i] = np.maximum.accumulate(h)
    return f[0]


def phase_a1(mem, g, gold_cat, gold_num) -> dict:
    index = TreeIndex(g._rtree)
    leafpos = np.array([index.leafpos_of_id[l.id] for l in g._leaves])
    out = {}
    sym = g.elem_symbol
    for group, cats in GROUPS.items():
        els = [e for e, (c, nn) in enumerate(zip(gold_cat, gold_num))
               if c is not None and strip(c) in cats and nn is not None]
        if not els:
            continue
        w = np.array([mem.weight[e] for e in els])
        is_s = np.array([gold_num[e] == "s" for e in els])
        total = float(w.sum())
        # The chosen categories.
        by = defaultdict(lambda: [0.0, 0.0])
        for e, ww, ss in zip(els, w, is_s):
            by[int(sym[e])][0 if ss else 1] += ww
        chosen = sum(max(v) for v in by.values()) / total
        mixed = {f"S{k}": {"singular": v[0], "plural": v[1]} for k, v in by.items()
                 if min(v) >= 0.05 * sum(v)}
        k0 = len(by)
        leaf_s = np.zeros(index.n_leaves)
        leaf_p = np.zeros(index.n_leaves)
        np.add.at(leaf_s, leafpos[els], w * is_s)
        np.add.at(leaf_p, leafpos[els], w * ~is_s)
        budget = max(4 * k0, 8)
        f = best_cut_purity(index, leaf_s, leaf_p, budget)
        leaves_pure = float(np.maximum(leaf_s, leaf_p).sum()) / total
        need95 = next((k for k in range(1, budget + 1) if f[k] >= 0.95 * total), None)
        out[group] = {"elements": total, "singular share": float((w * is_s).sum()) / total,
                      "categories holding them": k0, "chosen categories: purity by number": chosen,
                      "categories mixing number (minority >= 5%)": mixed,
                      f"best cut, {k0} nodes": float(f[k0]) / total,
                      f"best cut, {2 * k0} nodes": float(f[min(2 * k0, budget)]) / total,
                      "leaves (finest cut)": leaves_pure,
                      "fewest nodes for 95% purity": need95 if need95 else f"> {budget}"}
    out["tree nodes"] = len(index.nodes)
    out["tree leaves"] = index.n_leaves
    return out


# --------------------------------------------------------------------------- #
# A2: stage B of compile_grammar on given symbols

def elements_of(mem: Memory, tok_index) -> _Elements:
    """``grammar._elements`` without the representation leaves (stage B does
    not read them)."""
    values = mem.contexts()
    pairs = None
    if values is not None and values and isinstance(values[0], tuple):
        pairs, values = values, [x[-1] for x in values]
    contexts = sorted(set(values)) if values is not None else None
    ctx_index = {x: i for i, x in enumerate(contexts or [])}
    contexts2 = sorted(set(pairs)) if pairs is not None else None
    pair_index = {x: i for i, x in enumerate(contexts2 or [])}
    return _Elements(
        leafpos=np.zeros(len(mem), dtype=np.int64),
        prim=np.array([k == Memory.PRIMITIVE for k in mem.kind]),
        tok=np.array([tok_index[t] if t is not None else -1 for t in mem.token], dtype=np.int64),
        left=np.array(mem.left, dtype=np.int64), right=np.array(mem.right, dtype=np.int64),
        root=np.array(mem.is_root), w=np.array(mem.weight, dtype=float),
        rel=np.zeros(len(mem), dtype=np.int64), n_rel=1,
        ctx=None if values is None else np.array([ctx_index[x] for x in values], dtype=np.int64),
        contexts=contexts,
        ctx2=None if pairs is None else np.array([pair_index[x] for x in pairs], dtype=np.int64),
        contexts2=contexts2)


def compile_with_symbols(mem: Memory, s: np.ndarray, alpha: float = 0.001, seed: int = 0) -> Grammar:
    """``compile_grammar`` stage B and the tables, for the given symbol of every
    element (sequences only; same code, same composition-cut search)."""
    vocab = mem.vocabulary() + [UNK]
    tok_index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    el = elements_of(mem, tok_index)
    _, s = _compact(np.asarray(s, dtype=np.int64))
    K = int(s.max()) + 1
    info: Dict[str, float] = {}
    keys = [("P", vocab[el.tok[e]]) if el.prim[e] else ("C", int(s[el.left[e]]), int(s[el.right[e]]))
            for e in range(len(s))]
    weight_of: Dict[tuple, float] = defaultdict(float)
    for key, w in zip(keys, el.w):
        weight_of[key] += w
    C = CobwebTree(COMPOSITION_ATTRS, seed=seed)
    leaf_of_key = {}
    for key in sorted(weight_of, key=lambda k: (-weight_of[k], str(k))):
        leaf_of_key[key] = C.ifit(_composition_instance(key), weight_of[key])
    cindex = TreeIndex(C)
    cleafpos = np.array([cindex.leafpos_of_id[leaf_of_key[k].id] for k in keys], dtype=np.int64)
    cstarts = {"evidence": evidence_cut(cindex, alpha=0.5), "leaves": [int(i) for i in cindex.node_of_leaf]}
    best = None
    for weight in [None] + ([CONTEXT_WEIGHTS[1]] if el.ctx is not None else []):
        cost = _factored_code(el, s, K, V, alpha, cleafpos, weight)
        cut, _, _, _ = search_cut(cindex, cost, cstarts)
        options = [weight] if weight is None else list(CONTEXT_WEIGHTS)
        if weight is not None and el.ctx2 is not None:
            options += [(b1, b2) for b1 in CONTEXT_WEIGHTS for b2 in CONTEXT_WEIGHTS]
        for w_ in options:
            value = _factored_code(el, s, K, V, alpha, cleafpos, w_)(cindex.assign(cut))
            if best is None or value < best[0] - 1e-9:
                best = (value, w_, cut)
    _, weight, ccut = best
    ccost = _factored_code(el, s, K, V, alpha, cleafpos, weight)
    two = isinstance(weight, tuple)
    offered2 = el.ctx2 is not None
    if weight is None:
        el.ctx = None
    if not two:
        el.ctx2 = None
    beta1 = weight[0] if two else weight
    _, c = _compact(cindex.assign(ccut)[cleafpos])
    M = int(c.max()) + 1
    w = el.w
    prim, comp = el.prim, ~el.prim
    n_U = np.zeros((K, M))
    np.add.at(n_U, (s, c), w)
    if el.ctx is not None:
        n_Uc = np.zeros((K, len(el.contexts), M))
        np.add.at(n_Uc, (s, el.ctx, c), w)
    if el.ctx2 is not None:
        cell, row_of = np.unique(s * len(el.contexts2) + el.ctx2, return_inverse=True)
        n_Uc2 = np.zeros((len(cell), M))
        np.add.at(n_Uc2, (row_of.reshape(-1), c), w)
    n_prim = np.bincount(c[prim], weights=w[prim], minlength=M)
    n_comp = np.bincount(c[comp], weights=w[comp], minlength=M)
    n_E = np.zeros((M, V))
    np.add.at(n_E, (c[prim], el.tok[prim]), w[prim])
    n_L = np.zeros((M, K))
    n_R = np.zeros((M, K))
    np.add.at(n_L, (c[comp], s[el.left[comp]]), w[comp])
    np.add.at(n_R, (c[comp], s[el.right[comp]]), w[comp])

    def normalize(n):
        n = n + alpha
        return n / n.sum(axis=-1, keepdims=True)

    top_fields, top_counts = mem.top_level_tables(s, K, alpha)
    total_code = mem.top_level_nats(s, K, alpha) + mem.layout_nats(alpha) + ccost(cindex.assign(ccut))
    info["bits (factored grammar)"] = total_code / LN2
    info["structure bits"] = elias_delta_bits(K) + elias_delta_bits(M)
    if el.contexts is not None:
        n_options = 1 + len(CONTEXT_WEIGHTS) + (len(CONTEXT_WEIGHTS) ** 2 if offered2 else 0)
        info["structure bits"] += float(np.log2(n_options))
    info["total bits"] = info["bits (factored grammar)"] + info["structure bits"]
    info["symbols"], info["rule classes"] = K, M
    info["read context weight"] = 0.0 if weight is None else (weight[0] if two else weight)
    pk = (n_prim + alpha) / (n_prim + n_comp + 2 * alpha)
    return Grammar(vocab=vocab, U=normalize(n_U), pk=pk, **top_fields, Lt=normalize(n_L), Rt=normalize(n_R),
                   E=normalize(n_E), alpha=alpha, info=info, elem_symbol=s, elem_rule=c,
                   contexts=None if el.ctx is None else el.contexts,
                   Uc=(None if el.ctx is None else
                       (n_Uc + beta1 * normalize(n_U)[:, None, :]) / (n_Uc.sum(axis=2, keepdims=True) + beta1)),
                   Uc2=None if el.ctx2 is None else _pair_rows(n_Uc2, cell, el, n_Uc, normalize(n_U), beta1, weight[1]),
                   fresh_pieces=False)


def oracle_split(sym: np.ndarray, gold_num, minority: float = 0.05):
    """Every category whose definite-number elements are mixed (minority share
    >= ``minority``) split into singular and plural parts; elements without a
    number go to the category's majority part."""
    tally = defaultdict(Counter)
    for k, nn in zip(sym, gold_num):
        if nn is not None:
            tally[int(k)][nn] += 1
    split = {k for k, t in tally.items() if min(t["s"], t["p"]) >= minority * (t["s"] + t["p"])}
    major = {k: ("s" if t["s"] >= t["p"] else "p") for k, t in tally.items()}
    new = []
    for k, nn in zip(sym, gold_num):
        k = int(k)
        new.append((k, (nn or major[k])) if k in split else (k, ""))
    ids: Dict[tuple, int] = {}
    return np.array([ids.setdefault(x, len(ids)) for x in new]), sorted(split)


def gen_rates(g: Grammar, cfg: CFG, blind: CFG, train_set, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        x = g.sample(rng, max_len=45)
        if x is not None:
            samples.append(x[0])
    gram = [cfg.recognizes(t) for t in samples]
    bl = [blind.recognizes(t) for t in samples]
    nov = [tuple(t) not in train_set for t in samples]
    return {"grammatical": float(np.mean(gram)), "grammatical ignoring number": float(np.mean(bl)),
            "novel": float(np.mean(nov)), "grammatical and novel": float(np.mean([a and b for a, b in zip(gram, nov)])),
            "examples": [" ".join(t) + (" ✓" if a else (" (agreement)" if b else " ✗"))
                         for t, a, b in zip(samples[:12], gram, bl)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--side", default="night", choices=("night", "gold"))
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cfg, blind = CFG(CG.build(args.grammar)), CFG(CG.number_blind(args.grammar))
    mem, g, train = refit(args.grammar, args.train, args.seed, args.side)
    res = {"grammar": args.grammar, "train": args.train, "seed": args.seed, "side": args.side,
           "refit total bits": g.info["total bits"], "refit seconds": time.time() - t0}
    if args.side == "night":
        with open(os.path.join(RESULTS, f"{args.grammar}_{args.train}_s{args.seed}", "results.json")) as f:
            res["recorded total bits"] = json.load(f)["unsupervised"]["total_bits"]
    print(f"refit: {g.info['total bits']:.1f} bits ({res.get('recorded total bits')} recorded), "
          f"{g.K} symbols, {time.time() - t0:.0f}s", flush=True)
    gold_cat, gold_num = element_gold(mem, train, cfg)
    res["A1"] = phase_a1(mem, g, gold_cat, gold_num)
    print(json.dumps(res["A1"], indent=1), flush=True)
    train_set = {tuple(e.tokens) for e in train}
    same = compile_with_symbols(mem, g.elem_symbol, seed=args.seed)
    res["A2"] = {"unsplit (stage B re-run)": {"total bits": same.info["total bits"], "symbols": same.K,
                                              "rule classes": same.M},
                 "unsplit, chosen grammar": {"total bits": g.info["total bits"], "symbols": g.K, "rule classes": g.M}}
    split, which = oracle_split(g.elem_symbol, gold_num)
    sg = compile_with_symbols(mem, split, seed=args.seed)
    res["A2"]["split"] = {"categories split": len(which), "total bits": sg.info["total bits"],
                          "symbols": sg.K, "rule classes": sg.M}
    res["A2"]["split minus unsplit (bits)"] = sg.info["total bits"] - same.info["total bits"]
    print(json.dumps(res["A2"], indent=1), flush=True)
    res["A2"]["generation, unsplit"] = gen_rates(g, cfg, blind, train_set, args.n_gen, args.seed)
    res["A2"]["generation, split"] = gen_rates(sg, cfg, blind, train_set, args.n_gen, args.seed)
    for k in ("generation, unsplit", "generation, split"):
        print(k, {a: b for a, b in res["A2"][k].items() if a != "examples"}, flush=True)
    res["seconds"] = time.time() - t0
    with open(os.path.join(args.out, f"{args.grammar}_{args.train}_s{args.seed}_{args.side}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
