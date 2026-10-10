"""Heads on Chinese characters: ``run_characters.run`` with ``heads.install``.

Modes as in ``run_characters.py``: ``supervised`` (operators as tokens, the
IDS structures given) and ``unsupervised`` (token sequences alone, one
night). Variants: ``composition`` (heads in the composition hierarchy and
the grammar) and ``both`` (and in the representation hierarchy). The
relational mode is not supported yet (``CharacterMemory.sample`` and
``log_prob`` draw the two parts independently).

Usage:
    python experiments/v2/run_heads_characters.py --mode supervised --variant both
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import heads as H  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner as _UL  # noqa: E402

WORKERS = [1]


class ParallelLearner(_UL):
    """The night's learner with ``--workers`` processes (same result)."""

    def __init__(self, *a, **k):
        k.setdefault("workers", WORKERS[0])
        super().__init__(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("supervised", "unsupervised"), default="supervised")
    ap.add_argument("--variant", choices=("composition", "both", "control", "splits", "splits_heads", "splits_joint",
                                              "final_single", "final_joint"), default="both")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "heads", "characters"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.variant.startswith("final"):
        import splits as SP
        subs = SP.install_final(joint=args.variant == "final_joint",
                                where="consolidate" if args.mode == "supervised" else "sleep")
    elif args.variant.startswith("splits"):
        import splits as SP
        SP.CONFIG["joint"] = args.variant == "splits_joint"
        subs = SP.install_splits(heads=args.variant == "splits_heads",
                                 representation_heads=args.variant == "splits_heads")
    else:
        subs = H.install(representation=args.variant in ("both", "control"), composition=True, side="learned",
                         representation_sides=args.variant != "control")
    import trellis2.unsupervised as U
    import run_characters as RCH
    if args.workers > 1:
        # The night's learner, with its independent searches and consolidations in parallel.
        WORKERS[0] = args.workers
        RCH.UnsupervisedLearner = ParallelLearner
        subs.append("run_characters.UnsupervisedLearner (workers)")
    from trellis2.characters import default_ids_path
    out = RCH.run(args.mode, args.seed, args.train, args.test, args.n_gen, default_ids_path())
    out["variant"], out["substituted"] = args.variant, subs
    g_info = None
    with open(os.path.join(args.out, f"{args.mode}_{args.variant}_s{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, default=str), flush=True)


if __name__ == "__main__":
    main()
