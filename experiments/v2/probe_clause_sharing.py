"""Idea 3 probe: do embedded clauses share a category with whole experiences?

Gold trees of a complex grammar, the supervised consolidation, with and without
``sentence_parent`` (the sentence recorded as the parent of a whole tree's top
parts, as an embedded clause records its own parent). For every element whose
gold label is the start symbol, the grammar's category is read off, and the
share of embedded clauses (inside a complement or a coordination) whose
category is one a root clause takes is reported, with the usual evaluation.
The gold labels are a diagnostic only; nothing in the learner sees them.

    python experiments/v2/probe_clause_sharing.py --grammar deep --train 2000 --seed 13
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), HERE, ROOT]

import numpy as np  # noqa: E402

import complex_grammars as CG  # noqa: E402
import run_complex as RC  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.model import Trellis2  # noqa: E402

OUT = os.path.join(HERE, "results", "clause_sharing")


def clause_categories(model, grammar, cfg, train):
    """(root clause categories, embedded clause categories) over the training elements."""
    mem = model.memory
    sym = np.asarray(grammar.elem_symbol)
    gold = [cfg.gold_labels(e.tokens, e.tree) for e in train]
    root, embedded = Counter(), Counter()
    for e in range(len(mem.kind)):
        s = mem.experience_of[e]
        span = mem.span[e]
        if gold[s].get(span) != cfg.start:
            continue
        if span == (0, len(train[s].tokens)):
            root[int(sym[e])] += 1
        else:
            embedded[int(sym[e])] += 1
    return root, embedded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="deep", choices=CG.GRAMMARS)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--sentence-parent", action="store_true")
    ap.add_argument("--spine-depth", type=int, default=2)
    args = ap.parse_args()

    full, test = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(full))
    train = [full[i] for i in order[:args.train]]
    test = test[:args.heldout]
    cfg = CFG(CG.build(args.grammar))
    train_set = {" ".join(e.tokens) for e in train}

    t0 = time.time()
    model = Trellis2(seed=args.seed, sentence_parent=args.sentence_parent, spine_depth=args.spine_depth)
    for e in train:
        model.learn(e.tokens, e.tree)
    g = model.consolidate()
    secs = time.time() - t0
    root, emb = clause_categories(model, g, cfg, train)
    shared = sum(c for k, c in emb.items() if k in root)
    res = {"grammar": args.grammar, "train": args.train, "seed": args.seed,
           "sentence_parent": args.sentence_parent, "spine_depth": args.spine_depth, "seconds": secs,
           "total_bits": g.info["total bits"], "symbols": g.K, "rule_classes": g.M,
           "root clause categories": dict(root.most_common()),
           "embedded clause categories": dict(emb.most_common()),
           "embedded clauses in a root category": shared / max(sum(emb.values()), 1)}
    res.update(RC.evaluate(model, test, cfg, train_set, args.seed, args.n_gen))
    res.pop("samples", None)
    os.makedirs(OUT, exist_ok=True)
    name = (f"{args.grammar}_{args.train}_s{args.seed}_{'sp' if args.sentence_parent else 'base'}"
            + ("" if args.spine_depth == 2 else f"_d{args.spine_depth}") + ".json")
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "sample_examples"}, indent=1, default=str))


if __name__ == "__main__":
    main()
