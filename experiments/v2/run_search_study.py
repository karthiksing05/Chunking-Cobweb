"""How well does each search setting minimize the description length, and does
a shorter code mean a better grammar?

For each condition (v1 split, 320 training sentences) the structure search is
run from each of the last 12 partitions on the word-class merge path, with
greedy search (beam 1), the default beam (4, patience 3) and a wide beam (16,
patience 5). Each run records the plain-PCFG code it reaches and the
generation commission of the maximum-likelihood grammar read off its
analyses (1,000 samples checked by the target grammar). The gold trees,
labelled with gold categories, give a reference code.

Usage:
    python experiments/v2/run_search_study.py --out experiments/v2/results/search
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.mdl_search import _nodes, chunk_and_merge, code_bits, word_classes  # noqa: E402

SETTINGS = [(1, 0), (4, 3), (16, 5)]
LEVELS = 12


def ml_commission(analyses, cfg, n=1000, seed=0, max_len=40) -> float:
    """Share of sentences sampled from the maximum-likelihood plain PCFG of the
    analyses (top level: category distribution and stop probability) that the
    target grammar rejects. Over-long samples are redrawn."""
    rows, start = defaultdict(Counter), Counter()
    tops = 0
    for sentence in analyses:
        tops += len(sentence)
        for top in sentence:
            start[top[0]] += 1
            for lab, body in _nodes(top):
                rows[lab][("w", body) if isinstance(body, str) else ("p", body[0][0], body[1][0])] += 1
    p_stop = len(analyses) / tops
    rng = np.random.default_rng(seed)
    table = {lab: (list(r), np.array(list(r.values()), float) / sum(r.values()))
             for lab, r in rows.items()}
    start_items = list(start)
    start_p = np.array([start[k] for k in start_items], float) / sum(start.values())

    def expand(lab, out):
        stack = [lab]
        while stack:
            items, p = table[stack.pop()]
            o = items[rng.choice(len(items), p=p)]
            if o[0] == "w":
                out.append(o[1])
                if len(out) > max_len:
                    return False
            else:
                stack.extend([o[2], o[1]])
        return True

    rejected = drawn = 0
    while drawn < n:
        out, ok = [], True
        while ok:
            ok = expand(start_items[rng.choice(len(start_items), p=start_p)], out)
            if not ok or rng.random() < p_stop:
                break
        if not ok:
            continue
        drawn += 1
        rejected += not cfg.recognizes(out)
    return rejected / n


def gold_code(train, cfg, n_tokens, alpha) -> float:
    def node(tokens, tree, labels, span):
        i, j = span
        if j - i == 1:
            return (labels[span], tokens[i])
        k = tree.split[span]
        return (labels[span], (node(tokens, tree, labels, (i, k)),
                               node(tokens, tree, labels, (k, j))))
    gold = [[node(e.tokens, e.tree, cfg.gold_labels(e.tokens, e.tree), (0, len(e.tokens)))]
            for e in train]
    return code_bits(gold, n_tokens, alpha)


def run_condition(condition: str, seed: int, data_root: str, alpha: float = 0.001) -> dict:
    t0 = time.time()
    examples = load_corpus(os.path.join(data_root, CONDITIONS[condition]))
    train, _ = v1_split(examples, seed)
    cfg = CFG(target_grammar(condition))
    sentences = [e.tokens for e in train]
    n_tokens = len({w for s in sentences for w in s}) + 1
    path = word_classes(sentences, alpha)
    runs = []
    for cls in path[-LEVELS:]:
        flat = [[(("w", cls[w]), w) for w in s] for s in sentences]
        for beam, patience in SETTINGS:
            t1 = time.time()
            analyses, bits = chunk_and_merge(flat, n_tokens, alpha, beam=beam, patience=patience)
            runs.append({"word_classes": len(set(cls.values())), "final_partition": cls is path[-1],
                         "beam": beam, "patience": patience, "bits": bits,
                         "commission": ml_commission(analyses, cfg, seed=seed),
                         "seconds": time.time() - t1})
    out = {"condition": condition, "seed": seed, "gold_bits": gold_code(train, cfg, n_tokens, alpha),
           "flat_bits": code_bits([[(("w", path[-1][w]), w) for w in s] for s in sentences],
                                  n_tokens, alpha),
           "runs": runs, "seconds": time.time() - t0}
    print(f"[{condition}] {len(runs)} runs in {out['seconds']:.0f}s", flush=True)
    return out


def summarise(results) -> str:
    lines = ["Plain-PCFG code of the training sentences (bits) and the commission of the "
             "maximum-likelihood grammar read off the analyses.", "",
             "| Condition | Gold trees | Greedy, final word classes | Greedy, best of 12 starts | "
             "Beam 4, best of 12 starts | Beam 16, best of 12 starts | Spearman(code, commission) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        runs = r["runs"]

        def best(beam):
            x = min((u for u in runs if u["beam"] == beam), key=lambda u: u["bits"])
            return f"{x['bits']:,.0f} ({100 * x['commission']:.1f}%)"
        final = next(u for u in runs if u["final_partition"] and u["beam"] == 1)
        rho = spearmanr([u["bits"] for u in runs], [u["commission"] for u in runs]).correlation
        lines.append(f"| {r['condition']} | {r['gold_bits']:,.0f} | {final['bits']:,.0f} "
                     f"({100 * final['commission']:.1f}%) | {best(1)} | {best(4)} | {best(16)} | "
                     f"{rho:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "search"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_condition, args.conditions.split(","),
                                [args.seed] * 6, [args.data_root] * 6))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    table = summarise(results)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
