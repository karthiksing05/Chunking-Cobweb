"""Split moves on the paper's six synthetic corpora, from gold trees.

For each condition and seed (v1 splits: 320 training, 40 held-out), the
supervised model (``Trellis2`` on the gold trees) with the current compile
(``baseline``) or with split moves (``splits``, ``splits.py``; no heads);
evaluated as ``run_unsupervised.py`` evaluates (code, held-out bits,
generation commission and novelty, bracket omission/commission).

Usage:
    python experiments/v2/run_splits_paper.py --conditions small,med --seeds 13,17 --variant splits
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="small,med,large,term_low,term_med,term_high")
    ap.add_argument("--seeds", default="13,17")
    ap.add_argument("--variant", choices=("baseline", "splits", "final_single", "final_joint"), default="splits")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "heads", "paper"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    subs = []
    if args.variant.startswith("final"):
        import splits as SP
        SP.CONFIG["log"] = False
        subs = SP.install_final(joint=args.variant == "final_joint", where="consolidate")
    elif args.variant == "splits":
        import splits as SP
        SP.CONFIG["log"] = False
        subs = SP.install_splits(heads=False, representation_heads=False)
    from trellis2 import load_corpus, v1_split
    from trellis2 import model as M
    from trellis2.data import CONDITIONS, default_data_root, target_grammar
    from trellis2.evaluation import CFG
    import run_unsupervised as RU
    for cond in args.conditions.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            t0 = time.time()
            examples = load_corpus(os.path.join(default_data_root(), CONDITIONS[cond]))
            train, test = v1_split(examples, seed)
            cfg = CFG(target_grammar(cond))
            model = M.Trellis2(seed=seed)
            for e in train:
                model.learn(e.tokens, e.tree)
            g = model.consolidate()
            res = RU.evaluate(model, g, test, cfg, {e.sentence for e in train}, seed, args.n_gen)
            res.update(condition=cond, seed=seed, variant=args.variant, substituted=subs,
                       seconds=time.time() - t0, splits_accepted=g.info.get("splits accepted"),
                       unsplit_total_bits=g.info.get("unsplit total bits"),
                       unsplit_stage_b_total_bits=g.info.get("unsplit stage B total bits"),
                       refine={k: v for k, v in g.info.items() if k.startswith("refine")})
            if args.variant.startswith("final"):
                res["splits_accepted"] = g.info.get("refine: splits accepted")
            print(f"[{cond} s{seed} {args.variant}] {res['total_bits']:.1f} bits, K={res['symbols']}, "
                  f"test {res['test_bits_per_sentence']:.2f}, gen-commission {res['gen_commission']:.3f}, "
                  f"novelty {res['novelty']:.3f}, omission {res['omission']:.3f}, splits {res['splits_accepted']} "
                  f"({res['seconds']:.0f}s)", flush=True)
            with open(os.path.join(args.out, f"{cond}_s{seed}_{args.variant}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
