"""The continuously incremental learner (no night) beyond the paper's corpora,
on the night's own setups and measures:

* ``stories``: TinyStories as ``run_stories.py`` (3-5 words over the 100 most
  frequent words; 500 held out, the next ``--train`` learned): held-out bits
  per sentence, and generated sentences real / new (its own sentences, i.e.
  whole trees, and all samples);
* ``complex``: ``complex_grammars.py`` as ``run_complex.py`` (``--grammar
  agree``, 2,000 training sentences, 300 held out): grammatical, novel,
  agreement errors, held-out bits, bracket omission and commission;
* ``characters``: Chinese characters from token sequences as
  ``run_joint_chunks.py`` / ``run_characters.py`` (2,000 learned, 500 held
  out): well formed, positions attested, real, novel, bracket omission and
  commission, held-out bits per character.

The learner sees each experience once, in order; at checkpoints it is
evaluated. Usage:

    python experiments/v2/run_incremental_domains.py stories --train 2500 --seed 13 \
        --out experiments/v2/results/incremental_learner/stories_2500
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

from trellis2.incremental import HedgedBeam, IncrementalBeam, IncrementalLearner  # noqa: E402

LN2 = math.log(2)


def make_learner(args):
    cfg = json.loads(args.config)
    if args.beam_width > 1:
        return HedgedBeam(seed=args.seed, width=args.beam_width, config=cfg)
    if args.beam:
        return IncrementalBeam(seed=args.seed, configs=[dict(cfg, **json.loads(c)) for c in args.beam])
    return IncrementalLearner(seed=args.seed, config=cfg)


def stream(learner, items, checkpoints, evaluate, log):
    curve, spent = [], 0.0
    for k, x in enumerate(items, 1):
        t0 = time.time()
        learner.observe(x)
        spent += time.time() - t0
        if k in checkpoints or k == len(items):
            t1 = time.time()
            point = evaluate(k)
            point.update(experiences=k, learn_seconds=spent, eval_seconds=time.time() - t1,
                         total_bits=learner.grammar.info["total bits"], symbols=learner.grammar.K,
                         rule_classes=learner.grammar.M, chunk_types=learner.grammar.info["chunk types"],
                         stats=dict(learner.stats))
            curve.append(point)
            log(point)
    return curve


def stories(args):
    import run_stories as RS
    from trellis2.stories import read_sentences, simple_sentences
    corpus = read_sentences()
    corpus_set = {tuple(s) for s in corpus}
    corpus_grams = {tuple(s[i:i + n]) for s in corpus for n in (2, 3) for i in range(len(s) - n + 1)}
    simple = simple_sentences(corpus, args.vocab, 3, args.max_len)
    order = np.random.default_rng(args.seed).permutation(len(simple))
    test = [simple[i] for i in order[:args.test]]
    train = [simple[i] for i in order[args.test:args.test + args.train]]
    learner = make_learner(args)

    def evaluate(k):
        held = sum(-learner.chart(s).log_prob / LN2 for s in test) / len(test)
        rng = np.random.default_rng(args.seed)
        seen = {tuple(s) for s in train[:k]}
        out = {"held-out bits per sentence": held}
        for name, whole in (("own sentences (whole trees)", True), ("all samples", False)):
            gen, _ = learner.generate(args.n_gen, rng, max_len=20, whole_only=whole)
            out[name] = RS.generation_measures([t for t, _ in gen], seen, corpus_set, corpus_grams,
                                               (3, args.max_len))
        tops = [len(r) for r in learner.roots]
        out["whole trees (train analyses)"] = float(np.mean([t == 1 for t in tops]))
        return out

    def log(p):
        print(f"[stories {p['experiences']}] {p['total_bits']:.0f} bits K={p['symbols']} M={p['rule_classes']} "
              f"held-out {p['held-out bits per sentence']:.2f}, real (own) "
              f"{p['own sentences (whole trees)']['real']:.3f}, real (all) {p['all samples']['real']:.3f}, "
              f"whole {p['whole trees (train analyses)']:.2f} ({p['learn_seconds']:.0f}s)", flush=True)
    curve = stream(learner, train, set(args.checkpoints), evaluate, log)
    models = {k: RS.ngram_model(train, k) for k in (1, 2, 3)}
    return {"domain": "stories", "train": len(train), "test": len(test), "curve": curve,
            "baselines": {f"{k}-gram": RS.ngram_bits(models[k], test, k) for k in (1, 2, 3)}}, learner


def complex_(args):
    import complex_grammars as CG
    import run_complex as RC
    from trellis2.evaluation import CFG
    train_all, held = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(train_all))
    train = [train_all[i] for i in order[:args.train]]
    test = held[:args.heldout]
    cfg = CFG(CG.build(args.grammar))
    RC.BLIND["cfg"] = CFG(CG.number_blind(args.grammar))
    learner = make_learner(args)

    def evaluate(k):
        seen = {tuple(e.tokens) for e in train[:k]}
        out = RC.evaluate(learner, test, cfg, seen, args.seed, args.n_gen, train[:k], learner.stored_trees())
        out.pop("samples", None)
        return out

    def log(p):
        g = p["generation"]
        print(f"[{args.grammar} {p['experiences']}] {p['total_bits']:.0f} bits K={p['symbols']} "
              f"grammatical {g['grammatical']:.3f} (agreement only {g.get('agreement errors only', 0):.3f}), "
              f"test {p['test_bits_per_sentence']:.2f}, omission {p['omission']:.3f} ({p['learn_seconds']:.0f}s)",
              flush=True)
    curve = stream(learner, [e.tokens for e in train], set(args.checkpoints), evaluate, log)
    return {"domain": "complex", "grammar": args.grammar, "train": len(train), "curve": curve}, learner


def characters(args):
    import run_characters as RCH
    from trellis2.characters import default_ids_path, load_characters, parse_prefix, placements
    path = default_ids_path()
    chars, _ = load_characters(path, seed=args.seed)
    train, test = chars[:args.train], chars[args.train:args.train + args.test]
    real = {parse_prefix(c.tokens): c.char for c in chars}
    attested = set()
    for c in chars:
        placements(parse_prefix(c.tokens), attested)
    learner = make_learner(args)

    def evaluate(k):
        out = RCH.evaluate(learner.chart, learner.generate, train[:k], test, real, attested, args.n_gen, args.seed)
        trees = learner.stored_trees()
        out["whole trees (train analyses)"] = float(np.mean([len(t.roots) == 1 for t in trees]))
        out["gold-exact (train analyses)"] = float(np.mean(
            [t.brackets() == c.tree.brackets() and len(t.roots) == 1 for t, c in zip(trees, train)]))
        return out

    def log(p):
        g = p["generation"]
        print(f"[characters {p['experiences']}] {p['total_bits']:.0f} bits K={p['symbols']} "
              f"well formed {g['well formed']:.3f}, omission {p['omission']:.3f}, commission "
              f"{p['commission']:.3f}, test {p['test_bits_per_character']:.2f} ({p['learn_seconds']:.0f}s)", flush=True)
    curve = stream(learner, [c.tokens for c in train], set(args.checkpoints), evaluate, log)
    return {"domain": "characters", "train": len(train), "curve": curve}, learner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=("stories", "complex", "characters"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--checkpoints", default="100,250,500,1000,2000")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--beam", action="append", default=[],
                    help="a hypothesis of the beam (JSON config overrides); give several")
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--pickle", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.checkpoints = [int(c) for c in args.checkpoints.split(",") if c]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    res, learner = {"stories": stories, "complex": complex_, "characters": characters}[args.domain](args)
    res.update(seed=args.seed, config=json.loads(args.config), beam=args.beam, seconds=time.time() - t0,
               step_seconds=[h["seconds"] for h in learner.history],
               memory_elements_total=len(learner.memory), live_elements=sum(len(x) for x in learner.elems),
               stats=dict(learner.stats))
    name = f"{args.domain}_{args.grammar + '_' if args.domain == 'complex' else ''}{args.train}_s{args.seed}.json"
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    if args.pickle:
        sys.setrecursionlimit(100000)
        pk = os.path.join(ROOT, "logs", "inc_pickles", os.path.basename(os.path.normpath(args.out)))
        os.makedirs(pk, exist_ok=True)
        with open(os.path.join(pk, name.replace(".json", ".pkl")), "wb") as f:
            pickle.dump(learner, f)
    print(f"done in {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
