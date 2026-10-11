"""Joint chunk moves on Chinese characters from sequences alone (and checks
elsewhere): the driver.

Subcommands:

* ``night``: one unsupervised night on characters (``run_characters.py``'s
  split: 2,000 learned / 500 held out), with the joint chunk move
  (``joint_chunks.py``) inside the search (``--variant search``), as
  final-round refinement (``--variant final``), or not at all
  (``--variant baseline``). Saves the night's state (search results, final
  analyses and trees) as ``state_<variant>_s<seed>.pkl`` and the evaluation
  of ``run_characters.evaluate`` and ``run_characters_memory.chunk_report``
  as ``night_<variant>_s<seed>.json``.
* ``diag``: Step 1, the oracle diagnostic on a saved baseline state (see
  ``joint_chunks.py``).

Usage:
    python experiments/v2/run_joint_chunks.py night --variant baseline --seed 13 --workers 8
    python experiments/v2/run_joint_chunks.py diag --seed 13 --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import trellis2.unsupervised as U  # noqa: E402
from trellis2.characters import default_ids_path, load_characters, parse_prefix, placements  # noqa: E402

import joint_chunks as JC  # noqa: E402
import run_characters as RCH  # noqa: E402
import run_characters_memory as RCM  # noqa: E402

LN2 = math.log(2)
OUT = os.path.join(HERE, "results", "joint_chunks")


class RecordingLearner(U.UnsupervisedLearner):
    """The night's learner, keeping the structure searches' results."""

    searched = None

    def _map(self, fn, jobs):
        out = super()._map(fn, jobs)
        if fn is U._search or getattr(fn, "is_search", False):
            self.searched = out
        return out


def characters(seed: int, n_train: int, n_test: int, path: str):
    chars, _ = load_characters(path, seed=seed)
    return chars, chars[:n_train], chars[n_train:n_train + n_test]


def evaluate_learner(learner, chars, train, test, seed: int, n_gen: int, path: str) -> dict:
    """``run_characters.evaluate`` (held-out bits, brackets, generation) and
    ``run_characters_memory.chunk_report`` (chunk validity, recall)."""
    real = {parse_prefix(c.tokens): c.char for c in chars}
    attested = set()
    for c in chars:
        placements(parse_prefix(c.tokens), attested)
    out = RCH.evaluate(learner.chart, learner.generate, train, test, real, attested, n_gen, seed)
    g = learner.model.grammar
    plain, _ = RCM.database_components(path)
    out["validity"] = RCM.chunk_report(learner.model.memory, g, "unsupervised",
                                       [c.tokens for c in train], plain)
    out.update(total_bits=g.info["total bits"], model_bits=g.info.get("model bits"),
               data_bits=g.info.get("data bits"), symbols=g.K, rule_classes=g.M,
               chunk_types=g.info["chunk types"])
    out["whole trees (train analyses)"] = float(np.mean([len(t.roots) == 1 for t in learner.trees]))
    out["gold-exact (train analyses)"] = float(np.mean(
        [t.brackets() == c.tree.brackets() and len(t.roots) == 1 for t, c in zip(learner.trees, train)]))
    return out


def run_night(args):
    os.makedirs(args.out, exist_ok=True)
    chars, train, test = characters(args.seed, args.train, args.test, args.path)
    names = JC.install(args.variant, JC_CONFIG_FROM_ARGS(args)) if args.variant != "baseline" else []
    if args.from_state:
        assert args.variant == "final"
        names = ["refine_final on the saved baseline night"]
        learner = RecordingLearner(seed=args.seed, workers=args.workers)
    else:
        learner = JC.LEARNER_CLASS(RecordingLearner)(seed=args.seed, workers=args.workers)
    for c in train:
        learner.observe(c.tokens)
    t0 = time.time()
    if args.from_state:
        # Final-round refinement of a saved night: its final model is the
        # consolidation of its final analyses (refitted here, checked).
        with open(os.path.join(args.out, f"state_baseline_s{args.seed}.pkl"), "rb") as f:
            saved = pickle.load(f)
        learner.trees = saved["trees"]
        learner.model = learner.fit(learner.trees)
        learner.analyses = learner._in_categories(learner.model, learner.trees)
        learner.history = list(saved["history"])
        learner.searched = saved["searched"]
        refit = learner.model.grammar.info["total bits"]
        print(f"refit of the saved night: {refit:.3f} bits (night {saved['total bits']:.3f})", flush=True)
        if args.variant == "final":
            learner.joint_log = JC.refine_final(learner)
    else:
        learner.sleep()
    seconds = time.time() - t0
    print(f"[night {args.variant} s{args.seed}] {seconds:.0f}s, total "
          f"{learner.model.grammar.info['total bits']:.1f} bits", flush=True)
    state = {"seed": args.seed, "variant": args.variant, "analyses": learner.analyses,
             "trees": learner.trees, "searched": learner.searched, "history": learner.history,
             "n_tokens": len({w for s in learner.experiences for w in s}) + 1,
             "total bits": learner.model.grammar.info["total bits"],
             "joint": getattr(learner, "joint_log", None)}
    tag = args.variant + ("_from_state" if args.from_state else "")
    with open(os.path.join(args.out, f"state_{tag}_s{args.seed}.pkl"), "wb") as f:
        pickle.dump(state, f)
    out = evaluate_learner(learner, chars, train, test, args.seed, args.n_gen, args.path)
    out.update(variant=args.variant, seed=args.seed, train=args.train, test=args.test,
               seconds=seconds, substituted=names, history=learner.history,
               joint=getattr(learner, "joint_log", None), config=dict(JC.CONFIG))
    with open(os.path.join(args.out, f"night_{tag}_s{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: out[k] for k in ("omission", "commission", "test_bits_per_character", "generation",
                                          "total_bits", "symbols", "chunk_types")}, default=str), flush=True)
    print("validity", out["validity"]["type modal class"], out["validity"]["recall"]["50"], flush=True)


def JC_CONFIG_FROM_ARGS(args) -> dict:
    return {k: v for k, v in {"max_leaves": args.max_leaves, "min_count": args.min_count,
                               "top_windows": args.top_windows}.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("night", "diag"))
    ap.add_argument("--variant", default="baseline", choices=("baseline", "search", "final"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-leaves", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=None)
    ap.add_argument("--top-windows", type=int, default=None)
    ap.add_argument("--from-state", action="store_true",
                    help="night: start from the saved baseline night (final-round refinement only)")
    ap.add_argument("--top", type=int, default=20, help="diag: components scored one by one")
    ap.add_argument("--path", default=default_ids_path())
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.cmd == "night":
        run_night(args)
    else:
        JC.run_diagnostic(args)


if __name__ == "__main__":
    main()
