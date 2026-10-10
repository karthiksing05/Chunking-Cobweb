"""``run_complex.py``'s unsupervised night with the final-round split
refinement (``splits.install_final(where="sleep")``): the night is the
baseline's, and its final model's symbols are refined by single (and joint)
split moves. Results: ``experiments/v2/results/heads/nights/``.

Usage:
    python experiments/v2/run_complex_final.py --grammar agree --train 2000 --seed 13 --joint --workers 6
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import splits as SP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "heads", "nights"))
    a = ap.parse_args()
    subs = SP.install_final(joint=a.joint, where="sleep")
    print("substituted:", subs, flush=True)
    import run_complex as RC
    args = argparse.Namespace(grammar=a.grammar, train=a.train, seed=a.seed, heldout=300, n_gen=1000, prompt_n=5,
                              workers=a.workers, no_supervised=False, no_prompts=True,
                              out=os.path.join(a.out, "joint" if a.joint else "single"))
    RC.run(args)


if __name__ == "__main__":
    main()
