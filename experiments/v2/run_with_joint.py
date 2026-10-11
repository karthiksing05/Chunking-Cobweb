"""Run another experiment script with a joint-chunk placement installed in
this process (``joint_chunks.install``), without editing it: the script's
``UnsupervisedLearner`` is substituted by the placement's learner class, and
``--out-const`` redirects a script whose output directory is a module
constant (``OUT``).

    python experiments/v2/run_with_joint.py --variant search --script run_unsupervised -- \
        --seeds 13,17 --out experiments/v2/results/joint_chunks/unsupervised_search
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "src"), HERE]

import joint_chunks as JC  # noqa: E402
import trellis2.unsupervised as U  # noqa: E402


def main():
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("baseline", "search", "final"))
    ap.add_argument("--script", required=True)
    ap.add_argument("--out-const", default=None)
    ap.add_argument("--config", default="{}", help="JSON overrides of joint_chunks.CONFIG")
    args = ap.parse_args(own)
    subs = []
    if args.variant != "baseline":
        subs = JC.install(args.variant, json.loads(args.config))
    mod = importlib.import_module(args.script)
    if hasattr(mod, "UnsupervisedLearner"):
        mod.UnsupervisedLearner = JC.LEARNER_CLASS(U.UnsupervisedLearner)
        subs.append(f"{args.script}.UnsupervisedLearner")
    if args.out_const:
        os.makedirs(args.out_const, exist_ok=True)
        mod.OUT = args.out_const
    print(f"variant {args.variant}; substituted {subs}; config {JC.CONFIG}", flush=True)
    sys.argv = [mod.__file__] + rest
    mod.main()


if __name__ == "__main__":
    main()
