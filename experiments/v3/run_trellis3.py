"""TRELLIS v3 (trellis2.trellis3): v1's hierarchies with chunk context, donated
representations and inside-outside parsing, on the paper's synthetic corpora.

Reports representation diagnostics first: how purely the context hierarchy's
concepts group words by gold part of speech, and the content hierarchy's
concepts group gold constituents by phrase label; then held-out bracket
omission of the best-expected analysis (side diagnostic) against right-branching.

Usage: python experiments/v3/run_trellis3.py --conditions small,med --seeds 13 --config '{}' --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, Tree, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG, BracketTally  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402


def purity(pairs):
    groups = defaultdict(Counter)
    for g, lab in pairs:
        groups[g][lab] += 1
    n = sum(sum(c.values()) for c in groups.values())
    return (sum(c.most_common(1)[0][1] for c in groups.values()) / n if n else 0.0), len(groups)


def run(cond, seed, cfg):
    t0 = time.time()
    train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    g = CFG(target_grammar(cond))
    L = Trellis3(seed=seed, **cfg)
    for e in train:
        L.learn(e.tokens)
    learn_s = time.time() - t0
    te, rb = BracketTally(), BracketTally()
    phrase_pairs = {d: [] for d in (1, 2, 3)}
    word_test_pairs = {d: [] for d in (1, 2, 3)}
    for e in test:
        r = L.parse(e.tokens)
        te.add(e.tree, r["tree"])
        n = len(e.tokens)
        rb.add(e.tree, Tree(n, {(i, n): i + 1 for i in range(n - 1)}, {}, [(0, n)]))
        lab = g.gold_labels(e.tokens, e.tree)
        gold = e.tree.brackets()
        for w in r["words"]:
            ids = L._ctx_ids(w)
            for d in (1, 2, 3):
                word_test_pairs[d].append((ids[d - 1], lab.get(w.span, "?")))
        for key in r["chosen"]:
            span = (key[0], key[2])
            if span in gold:
                ids = L._cnt_ids(r["comps"][key])
                for d in (1, 2, 3):
                    phrase_pairs[d].append((ids[d - 1], lab.get(span, "?")))
    base_word = Counter(l for _, l in word_test_pairs[1])
    base_phrase = Counter(l for _, l in phrase_pairs[1])
    row = {"condition": cond, "seed": seed, "config": cfg,
           "held-out omission": te.omission, "right-branching omission": rb.omission,
           "word POS purity by depth": {d: purity(word_test_pairs[d]) for d in (1, 2, 3)},
           "word POS base rate": base_word.most_common(1)[0][1] / max(sum(base_word.values()), 1),
           "phrase purity by depth (gold constituents found)": {d: purity(phrase_pairs[d]) for d in (1, 2, 3)},
           "phrase base rate": base_phrase.most_common(1)[0][1] / max(sum(base_phrase.values()), 1) if base_phrase else 0,
           "parses": [L.parse(e.tokens)["tree"].to_string(e.tokens) for e in test[:6]],
           "gold": [e.tree.to_string(e.tokens) for e in test[:6]],
           "stats": dict(L.stats), "learn seconds": learn_s, "seconds": time.time() - t0}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="small,med,large,term_low,term_med,term_high")
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.loads(args.config)
    rows = []
    for cond in args.conditions.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = run(cond, seed, cfg)
            rows.append(r)
            wp, pp = r["word POS purity by depth"], r["phrase purity by depth (gold constituents found)"]
            print(f"[{cond} s{seed}] omission {r['held-out omission']:.3f} (RB {r['right-branching omission']:.3f}) | "
                  f"word POS purity d1/d2/d3 {wp[1][0]:.2f}/{wp[2][0]:.2f}/{wp[3][0]:.2f} (base {r['word POS base rate']:.2f}) | "
                  f"phrase purity {pp[1][0]:.2f}/{pp[2][0]:.2f}/{pp[3][0]:.2f} (base {r['phrase base rate']:.2f}, "
                  f"{pp[2][1]} concepts) | {r['seconds']:.0f}s", flush=True)
            for p, gd in zip(r["parses"][:3], r["gold"][:3]):
                print("    ", p, "   gold", gd, flush=True)
            with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
