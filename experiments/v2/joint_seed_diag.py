"""Diagnostic for the joint split move: is the incomplete number split on
gold trees a seed problem or a propagation problem?

For a grammar refitted as ``number_oracle.py`` does (night's or gold-tree),
each symbol holding number-bearing elements is used as the seed with its
oracle number split (evaluation only: number never enters the learner);
the split is propagated by ``splits.propagate`` and scored by the exact full
code. Reported per seed: how many symbols the propagation splits, the number
purity of the result (nominal / verbal / determiner), and the full-code
change, next to the oracle split of every category (Phase A2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import number_oracle as NO  # noqa: E402
import complex_grammars as CG  # noqa: E402
import splits as SP  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.grammar import UNK  # noqa: E402


def purity(sym, gold_cat, gold_num):
    out = {}
    for group, cats in NO.GROUPS.items():
        by = {}
        for e, (c, nn) in enumerate(zip(gold_cat, gold_num)):
            if c is not None and NO.strip(c) in cats and nn is not None:
                by.setdefault(int(sym[e]), Counter())[nn] += 1
        tot = sum(sum(v.values()) for v in by.values())
        out[group] = round(sum(max(v.values()) for v in by.values()) / tot, 3) if tot else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--side", default="gold")
    ap.add_argument("--heads", action="store_true",
                    help="also score the pruned splits with head-conditioned rules (heads.compile_with_symbols_heads)")
    ap.add_argument("--pickle", default=None)
    args = ap.parse_args()
    cfg = CFG(CG.build(args.grammar))
    if args.pickle:
        NO.RESULTS = os.path.dirname(os.path.dirname(args.pickle))
    mem, g, train = NO.refit(args.grammar, args.train, args.seed, args.side)
    import heads as H
    hb = (lambda x: H.compile_with_symbols_heads(mem, x, seed=args.seed).info["total bits"]) if args.heads else None
    gold_cat, gold_num = NO.element_gold(mem, train, cfg)
    s = g.elem_symbol.copy()
    base = NO.compile_with_symbols(mem, s, seed=args.seed).info["total bits"]
    vocab = mem.vocabulary() + [UNK]
    tok_index = {t: i for i, t in enumerate(vocab)}
    key = SP.element_keys(mem, s, len(vocab), tok_index)
    graph = SP.Graph(mem)
    res = {"grammar": args.grammar, "side": args.side, "unsplit bits": base, "unsplit purity": purity(s, gold_cat, gold_num)}
    split_all, _ = NO.oracle_split(s, gold_num)
    res["oracle all: delta"] = NO.compile_with_symbols(mem, split_all, seed=args.seed).info["total bits"] - base
    res["oracle all: purity"] = purity(split_all, gold_cat, gold_num)
    if hb:
        res["heads: unsplit bits"] = hb(s)
        res["heads: oracle all: delta"] = hb(split_all) - res["heads: unsplit bits"]
    print(json.dumps(res), flush=True)
    res["seeds"] = []
    gn = np.array([x == "p" for x in gold_num])
    for A in range(int(s.max()) + 1):
        m = s == A
        t = Counter(gold_num[e] for e in np.flatnonzero(m) if gold_num[e])
        if min(t.get("s", 0), t.get("p", 0)) < 0.05 * sum(t.values()) or sum(t.values()) < 20:
            continue
        mask = m & gn
        out = SP.propagate(A, mask, s, key, graph)
        row = {"seed symbol": A, "yields": [y for y, _ in g.symbol_yields[A].most_common(4)]}
        if out is None:
            row["propagated"] = "nothing beyond the seed"
            out = np.where(mask, s.max() + 1, s)
        row["symbols split"] = int(len(np.unique(s[out != s])))
        row["purity"] = purity(out, gold_cat, gold_num)
        row["delta bits"] = NO.compile_with_symbols(mem, out, seed=args.seed).info["total bits"] - base
        pr = SP.prune(s, out, mem, len(vocab), tok_index, 0.001, keep=A)
        row["pruned: symbols split"] = int(len(np.unique(s[pr != s])))
        row["pruned: purity"] = purity(pr, gold_cat, gold_num)
        row["pruned: delta bits"] = NO.compile_with_symbols(mem, pr, seed=args.seed).info["total bits"] - base
        if hb:
            row["heads: pruned: delta bits"] = hb(pr) - res["heads: unsplit bits"]
        res["seeds"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    path = os.path.join(HERE, "results", "heads", "phase_a",
                        f"joint_seed_{args.grammar}_{args.train}_s{args.seed}_{args.side}{'_heads' if hb else ''}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
