"""TRELLIS v2 on the paper's six synthetic conditions.

Same corpora, seeds and splits as v1 (seeded shuffle, 320 train / first 40
test sentences). At each checkpoint the model consolidates and is scored on:

* parse omission / commission: 1 - bracket recall / precision on held-out
  sentences (minimum-Bayes-risk parse);
* generation commission: share of sampled sentences the target grammar
  rejects; novelty: share of samples not in the training set;
* code length: bits per training derivation; held-out bits per sentence.

Usage:
    python experiments/v2/run_synthetic.py --out experiments/v2/results/main
    python experiments/v2/run_synthetic.py --conditions med,large --seeds 13 \
        --checkpoints 320 --set spine_depth=3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import Trellis2, load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG, BracketTally  # noqa: E402

SEEDS = [13, 17, 7, 42, 100]
CHECKPOINTS = [10, 20, 40, 80, 160, 320]
V1_ENDPOINTS = {  # 20-seed endpoints at n=300 (reports/..., from the shipped CSVs)
    "small": (0.0004, 0.000), "med": (0.033, 0.011), "large": (0.062, 0.016),
    "term_low": (0.057, 0.062), "term_med": (0.070, 0.047), "term_high": (0.097, 0.034),
}


def run_one(condition: str, seed: int, checkpoints, params: dict, n_gen: int,
            data_root: str) -> list:
    examples = load_corpus(os.path.join(data_root, CONDITIONS[condition]))
    train, test = v1_split(examples, seed)
    cfg = CFG(target_grammar(condition))
    model = Trellis2(seed=seed, **params)
    rows = []
    seen = set()
    for t, ex in enumerate(train, 1):
        model.learn(ex.tokens, ex.tree)
        seen.add(ex.sentence)
        if t not in checkpoints:
            continue
        t0 = time.time()
        g = model.consolidate()
        tally, test_bits = BracketTally(), 0.0
        for e in test:
            chart = model.chart(e.tokens)
            tally.add(e.tree, chart.mbr_tree())
            test_bits += -chart.log_prob / np.log(2)
        samples, rejected = model.generate(n_gen, np.random.default_rng(seed))
        gram = [cfg.recognizes(toks) for toks, _ in samples]
        novel = [" ".join(toks) not in seen for toks, _ in samples]
        rows.append({
            "condition": condition, "seed": seed, "n_train": t,
            "omission": tally.omission, "parse_commission": tally.commission,
            "exact_match": tally.exact_match,
            "gen_commission": 1.0 - float(np.mean(gram)),
            "novelty": float(np.mean(novel)),
            "grammatical_novel": float(np.mean([a and b for a, b in zip(gram, novel)])),
            "gen_rejected_long": rejected,
            "bits_per_train_sentence": g.info["bits per sentence"],
            "bits_per_test_sentence": test_bits / max(len(test), 1),
            "symbols": g.K, "rules": g.M,
            "rounds": g.info.get("consolidation rounds", 1),
            "seconds": time.time() - t0,
        })
        print(f"[{condition} s{seed} n={t}] omission {tally.omission:.3f} "
              f"gen-commission {rows[-1]['gen_commission']:.3f} novelty {rows[-1]['novelty']:.2f} "
              f"symbols {g.K} rules {g.M} ({rows[-1]['seconds']:.0f}s)", flush=True)
    return rows


def summarise(rows: list, n_final: int) -> str:
    lines = ["| Condition | Omission (v2) | Omission (v1) | Gen. commission (v2) | "
             "Gen. commission (v1) | Novelty (v2) | Symbols | Rule classes |",
             "|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond and r["n_train"] == n_final]
        if not rs:
            continue
        def ms(key):
            v = np.array([r[key] for r in rs])
            return f"{100 * v.mean():.1f}% ± {100 * v.std():.1f}"
        v1o, v1c = V1_ENDPOINTS[cond]
        lines.append(f"| {cond} | {ms('omission')} | {100 * v1o:.1f}% | {ms('gen_commission')} | "
                     f"{100 * v1c:.1f}% | {ms('novelty')} | "
                     f"{np.mean([r['symbols'] for r in rs]):.1f} | "
                     f"{np.mean([r['rules'] for r in rs]):.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--checkpoints", default=",".join(map(str, CHECKPOINTS)))
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "main"))
    ap.add_argument("--set", action="append", default=[],
                    help="model parameter, e.g. --set spine_depth=2 --set alpha=0.01")
    args = ap.parse_args()

    params = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        params[k] = float(v) if "." in v or "e" in v else int(v)
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    os.makedirs(args.out, exist_ok=True)

    jobs = [(c, s) for c in conditions for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c, s, checkpoints, params, args.n_gen, args.data_root)
                   for c, s in jobs]
        for f in futures:
            rows.extend(f.result())
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"params": params, "rows": rows}, f, indent=1)
    table = summarise(rows, max(checkpoints))
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"Parameters: {params or 'defaults'}\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
