"""Profile one unsupervised night (cProfile, one process) to find the hot loops.

Two representative nights:

* ``synthetic``: the night of ``run_unsupervised.py`` on one of the paper's
  corpora (v1 split, 320 sentences), default night (sampling included);
* ``stories``: the night of ``run_stories.py`` (TinyStories, 3–5 words over
  100 words by default, ``read_words=2``), at ``--train`` sentences.

``workers=1``, so every search and consolidation runs in this process and the
profile sees all of it. The raw profile goes to ``<out>/<name>.prof``; the
summary (``<name>.json`` and ``<name>.md``) groups the time:

* *self time by group* (``cobweb_cu``'s ``ifit`` and ``nodes`` are wrapped
  in Python functions, which call straight through, so that cProfile can
  see them; attribute reads on its nodes stay in their callers' time): each function's own time, where time spent in a
  library or built-in function (numpy, scipy, math, builtins) is charged to
  the repository function that called it (split in proportion to the
  callers' shares), so "chart" includes the numpy work the chart loops do;
  ``cobweb_cu`` calls are a group of their own;
* *cumulative time* of the entry points of each group (``Chart.__init__``,
  ``Chart.viterbi_tree``, ``_State.scored_moves``, ``compile_grammar``, ...).

Usage:
    python experiments/v2/profile_night.py --domain synthetic --condition large --out experiments/v2/results/profile
    python experiments/v2/profile_night.py --domain stories --train 1000 --out experiments/v2/results/profile
    python experiments/v2/profile_night.py --analyse experiments/v2/results/profile/large_s13.prof
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402
import trellis2.grammar as _grammar_mod  # noqa: E402
import trellis2.memory as _memory_mod  # noqa: E402
from trellis2.cobweb import CobwebTree as _CobwebTree  # noqa: E402


class _ProfiledCobwebTree(_CobwebTree):
    """cobweb_cu's methods are nanobind functions, which cProfile does not
    see as calls of their own; these Python wrappers make the time spent in
    the compiled Cobweb visible (same results: they only call through)."""

    def ifit(self, *a, **k):
        return _cobweb_ifit(self, *a, **k)

    def nodes(self, *a, **k):
        return _cobweb_nodes(self, *a, **k)


def _cobweb_ifit(tree, *a, **k):
    return _CobwebTree.ifit(tree, *a, **k)


def _cobweb_nodes(tree, *a, **k):
    return _CobwebTree.nodes(tree, *a, **k)


# This process only: the night's modules build their trees from the wrapper.
_grammar_mod.CobwebTree = _ProfiledCobwebTree
_memory_mod.CobwebTree = _ProfiledCobwebTree


def night_synthetic(condition: str, seed: int):
    examples = load_corpus(os.path.join(default_data_root(), CONDITIONS[condition]))
    train, _ = v1_split(examples, seed)
    learner = UnsupervisedLearner(seed=seed, workers=1)
    for e in train:
        learner.observe(e.tokens)
    return learner


def night_complex(grammar: str, train_n: int, seed: int):
    """As ``run_complex.py``: the seed permutes the 2,000 training sentences."""
    sys.path.insert(0, HERE)
    import complex_grammars as CG
    train_all, _ = CG.load(grammar)
    order = np.random.default_rng(seed).permutation(len(train_all))
    learner = UnsupervisedLearner(seed=seed, workers=1)
    for i in order[:train_n]:
        learner.observe(train_all[i].tokens)
    return learner


def night_stories(train_n: int, seed: int, vocab: int, max_len: int):
    from trellis2.stories import read_sentences, simple_sentences
    corpus = read_sentences()
    simple = simple_sentences(corpus, vocab, 3, max_len)
    order = np.random.default_rng(seed).permutation(len(simple))
    train = [simple[i] for i in order[500:500 + train_n]]   # as run_stories.py (500 held out)
    learner = UnsupervisedLearner(seed=seed, workers=1, read_words=2, previous_word=True)
    for s in train:
        learner.observe(s)
    return learner


# --------------------------------------------------------------------------- #
# Grouping

def _is_repo(fn) -> bool:
    path = fn[0]
    return "/trellis2/" in path or "/experiments/v2/" in path


def _is_cobweb(fn) -> bool:
    return (fn[0] == "~" and "obweb" in fn[2]) or (
        os.path.basename(fn[0]) == "profile_night.py" and fn[2].startswith("_cobweb_"))


def group_of(fn) -> str:
    path, _, name = fn
    if _is_cobweb(fn):
        return "Cobweb calls (cobweb_cu)"
    base = os.path.basename(path)
    if base == "chart.py":
        return "chart passes (chart.py)"
    if base == "mdl_search.py":
        if name in ("word_classes", "class_bigram_bits"):
            return "word classes (mdl_search.word_classes)"
        return "structure search (mdl_search: moves, scoring, apply)"
    if base in ("grammar.py", "mdl.py"):
        return "grammar read-out: cuts, code lengths (grammar.py, mdl.py)"
    if base == "memory.py":
        return "memory: instances, replay driver (memory.py)"
    return "rest"


def owners(stats: dict) -> dict:
    """For every function, how its self time is split among repository
    functions (or cobweb_cu): {owner: share}."""
    memo: dict = {}

    def own(fn, depth=0):
        if fn in memo:
            return memo[fn]
        if _is_repo(fn) or _is_cobweb(fn) or depth > 50:
            memo[fn] = {fn: 1.0}
            return memo[fn]
        memo[fn] = {fn: 1.0}             # cycle guard
        callers = stats[fn][4]
        weights = {c: v[2] + 1e-12 for c, v in callers.items()}   # caller's share of fn's self time
        total = sum(weights.values())
        out: dict = defaultdict(float)
        if total <= 0 or not callers:
            out[fn] = 1.0
        else:
            for c, w in weights.items():
                for o, s in own(c, depth + 1).items():
                    out[o] += s * w / total
        memo[fn] = dict(out)
        return memo[fn]

    return {fn: own(fn) for fn in stats}


def analyse(prof_path: str, wall: float | None = None) -> dict:
    st = pstats.Stats(prof_path)
    stats = st.stats
    total = sum(v[2] for v in stats.values())
    own = owners(stats)
    by_func: dict = defaultdict(float)
    for fn, v in stats.items():
        for o, s in own[fn].items():
            by_func[o] += v[2] * s
    by_group: dict = defaultdict(float)
    for fn, t in by_func.items():
        by_group[group_of(fn)] += t
    funcs = sorted(by_func.items(), key=lambda x: -x[1])[:40]

    def label(fn):
        return f"{os.path.basename(fn[0])}:{fn[1]}({fn[2]})"

    cum = sorted(((label(fn), v[3], v[1]) for fn, v in stats.items() if _is_repo(fn) or _is_cobweb(fn)),
                 key=lambda x: -x[1])[:45]
    return {"profile": prof_path, "total profiled seconds": total, "wall seconds (unprofiled)": wall,
            "self time by group (library time charged to the calling repo function)":
                {g: t for g, t in sorted(by_group.items(), key=lambda x: -x[1])},
            "self time by function (top 40)": [(label(fn), t) for fn, t in funcs],
            "cumulative time (repo and cobweb_cu functions, top 45)": [(n, t, c) for n, t, c in cum]}


def write_summary(summary: dict, md_path: str):
    total = summary["total profiled seconds"]
    lines = [f"Profile `{os.path.basename(summary['profile'])}`: {total:.0f} s profiled"
             + (f", {summary['wall seconds (unprofiled)']:.0f} s wall without the profiler" if summary.get("wall seconds (unprofiled)") else ""),
             "", "| Group (self time, library calls charged to the caller) | Seconds | Share |", "|---|---|---|"]
    for g, t in summary["self time by group (library time charged to the calling repo function)"].items():
        lines.append(f"| {g} | {t:.1f} | {t / total:.1%} |")
    lines += ["", "| Function (self + library time) | Seconds | Share |", "|---|---|---|"]
    for n, t in summary["self time by function (top 40)"]:
        lines.append(f"| `{n}` | {t:.1f} | {t / total:.1%} |")
    lines += ["", "| Function (cumulative) | Seconds | Calls |", "|---|---|---|"]
    for n, t, c in summary["cumulative time (repo and cobweb_cu functions, top 45)"]:
        lines.append(f"| `{n}` | {t:.1f} | {c} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("synthetic", "stories", "complex"), default="synthetic")
    ap.add_argument("--condition", default="large")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--grammar", default="deep", help="complex domain: agree, relative or deep")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--vocab", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--unprofiled", action="store_true", help="also time the same night without the profiler")
    ap.add_argument("--analyse", default=None, help="only summarise an existing .prof")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "profile"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.analyse:
        s = analyse(args.analyse)
        write_summary(s, args.analyse[:-5] + ".md")
        return
    if args.domain == "synthetic":
        name = f"{args.condition}_s{args.seed}"
        make = lambda: night_synthetic(args.condition, args.seed)   # noqa: E731
    elif args.domain == "complex":
        name = f"complex_{args.grammar}_{args.train}_s{args.seed}"
        make = lambda: night_complex(args.grammar, args.train, args.seed)   # noqa: E731
    else:
        name = f"stories_{args.train}_v{args.vocab}_l{args.max_len}_s{args.seed}"
        make = lambda: night_stories(args.train, args.seed, args.vocab, args.max_len)   # noqa: E731
    wall = None
    if args.unprofiled:
        learner = make()
        t0 = time.time()
        learner.sleep()
        wall = time.time() - t0
        print(f"unprofiled night: {wall:.0f} s, {learner.model.grammar.info['total bits']:.1f} bits", flush=True)
    learner = make()
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    g = learner.sleep()
    prof.disable()
    profiled_wall = time.time() - t0
    prof_path = os.path.join(args.out, name + ".prof")
    prof.dump_stats(prof_path)
    print(f"profiled night: {profiled_wall:.0f} s, {g.info['total bits']:.1f} bits", flush=True)
    s = analyse(prof_path, wall)
    s["profiled wall seconds"] = profiled_wall
    s["night history"] = [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in h.items()}
                          for h in learner.history]
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(s, f, indent=1, default=str)
    write_summary(s, os.path.join(args.out, name + ".md"))


if __name__ == "__main__":
    main()
