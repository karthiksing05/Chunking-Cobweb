"""Diagnostic: does the oracle number split pay one category at a time?

For a grammar (night's or gold-tree, as ``number_oracle.py``), the oracle
number split of Phase A2 applied to each mixing category alone, and to all of
them at once; each scored by the exact full code (stage B refitted). If the
singles do not pay but the joint split does, a greedy one-split-at-a-time
search cannot reach it (evaluation only: number never enters a learner).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import number_oracle as NO  # noqa: E402
import complex_grammars as CG  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--side", default="gold")
    args = ap.parse_args()
    cfg = CFG(CG.build(args.grammar))
    mem, g, train = NO.refit(args.grammar, args.train, args.seed, args.side)
    _, gold_num = NO.element_gold(mem, train, cfg)
    base = NO.compile_with_symbols(mem, g.elem_symbol, seed=args.seed).info["total bits"]
    split_all, which = NO.oracle_split(g.elem_symbol, gold_num)
    res = {"grammar": args.grammar, "train": args.train, "side": args.side, "unsplit bits": base,
           "all split: delta": NO.compile_with_symbols(mem, split_all, seed=args.seed).info["total bits"] - base,
           "categories split": len(which), "singles": []}
    print("all:", res["all split: delta"], flush=True)
    for k in which:
        only = np.array([(s_ if s_ != k else (s_ if nn in (None, "s") else len(set(g.elem_symbol.tolist()))))
                         for s_, nn in zip(g.elem_symbol.tolist(), gold_num)])
        # Elements without a number stay in the category's original part.
        d = NO.compile_with_symbols(mem, only, seed=args.seed).info["total bits"] - base
        n_s = sum(1 for s_, nn in zip(g.elem_symbol, gold_num) if s_ == k and nn == "s")
        n_p = sum(1 for s_, nn in zip(g.elem_symbol, gold_num) if s_ == k and nn == "p")
        res["singles"].append({"category": int(k), "singular": n_s, "plural": n_p, "delta bits": d,
                               "yields": [y for y, _ in g.symbol_yields[k].most_common(5)]})
        print(res["singles"][-1], flush=True)
    out = os.path.join(HERE, "results", "heads", "phase_a")
    with open(os.path.join(out, f"singles_{args.grammar}_{args.train}_s{args.seed}_{args.side}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
