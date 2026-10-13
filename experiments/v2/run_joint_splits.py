"""Split moves (``trellis2.splits``) at the end of the night, across domains.

Refinement is the night's last step, so one night gives both arms: the
night (``src`` default: joint chunks on, ``joint_splits`` off) is evaluated,
then its chosen grammar is refined exactly as ``UnsupervisedLearner(
joint_splits=True)`` does (``splits.refine``) and evaluated again.

Domains: ``complex`` (``complex_grammars.py``, as ``run_complex.py``),
``paper`` (the paper's six corpora, as ``run_unsupervised.py``), ``stories``
(TinyStories as ``run_stories.py``), ``characters`` (sequences alone, as
``run_characters.py``, plus ``run_characters_memory.chunk_report``).

Usage:
    python experiments/v2/run_joint_splits.py --domain complex --grammar agree --train 2000 --seed 13 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from trellis2 import load_corpus, v1_split  # noqa: E402
from trellis2.splits import refine  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402

LN2 = math.log(2)
OUT = os.path.join(HERE, "results", "joint_splits")


def night(learner, items):
    t0 = time.time()
    for x in items:
        learner.observe(x)
    learner.sleep()
    return time.time() - t0


def refined(learner):
    """Apply the night's last step (split moves) to the learner's model."""
    t0 = time.time()
    model = learner.model
    g0 = model.grammar
    g = refine(model.memory, g0, alpha=learner.alpha, seed=learner.seed, search=model.search,
               config=learner.split_config)
    info = {k: v for k, v in g.info.items() if k.startswith("splits:") and k != "splits: history"}
    info["changed"] = g is not g0
    info["seconds"] = time.time() - t0
    info["history"] = [h for h in g.info.get("splits: history", []) if h.get("full delta (bits)", 0) < -1e-6]
    if g is not g0:
        model._grammar = g
        learner.analyses = learner._in_categories(model, learner.trees)
    return info


def grammar_info(g):
    return {"total_bits": g.info["total bits"], "symbols": g.K, "rule_classes": g.M, "chunk_types": g.info.get("chunk types")}


def run_complex(args, learner):
    import complex_grammars as CG
    import run_complex as RC
    from trellis2.evaluation import CFG
    train_all, held = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(train_all))
    train = [train_all[i] for i in order[:args.train]]
    test = held[:300]
    cfg = CFG(CG.build(args.grammar))
    RC.BLIND["cfg"] = CFG(CG.number_blind(args.grammar))
    train_set = {tuple(e.tokens) for e in train}
    res = {"night seconds": night(learner, [e.tokens for e in train])}

    def ev():
        out = grammar_info(learner.model.grammar)
        out.update(RC.evaluate(learner, test, cfg, train_set, args.seed, 1000, train, learner.trees))
        out.pop("samples", None)
        return out
    res["off"] = ev()
    res["splits"] = refined(learner)
    res["on"] = ev() if res["splits"]["changed"] else res["off"]
    return res


def run_paper(args, learner):
    import run_unsupervised as RU
    from trellis2.data import CONDITIONS, default_data_root, target_grammar
    from trellis2.evaluation import CFG
    examples = load_corpus(os.path.join(default_data_root(), CONDITIONS[args.grammar]))
    train, test = v1_split(examples, args.seed)
    cfg = CFG(target_grammar(args.grammar))
    sentences = {e.sentence for e in train}
    res = {"night seconds": night(learner, [e.tokens for e in train])}
    res["off"] = RU.evaluate(learner, learner.model.grammar, test, cfg, sentences, args.seed, 1000)
    res["splits"] = refined(learner)
    res["on"] = (RU.evaluate(learner, learner.model.grammar, test, cfg, sentences, args.seed, 1000)
                 if res["splits"]["changed"] else res["off"])
    return res


def run_stories(args, learner):
    import run_stories as RS
    from trellis2.stories import read_sentences, simple_sentences
    corpus = read_sentences()
    corpus_set = {tuple(s) for s in corpus}
    grams = {tuple(s[i:i + n]) for s in corpus for n in (2, 3) for i in range(len(s) - n + 1)}
    simple = simple_sentences(corpus, 100, 3, 5)
    order = np.random.default_rng(args.seed).permutation(len(simple))
    test = [simple[i] for i in order[:500]]
    train = [simple[i] for i in order[500:500 + args.train]]
    train_set = {tuple(s) for s in train}
    res = {"night seconds": night(learner, train)}
    models = {k: RS.ngram_model(train, k) for k in (1, 2, 3)}
    res["ngram held-out bits"] = {f"{k}-gram": RS.ngram_bits(models[k], test, k) for k in (1, 2, 3)}

    def ev():
        out = grammar_info(learner.model.grammar)
        out["held-out bits per sentence"] = sum(-learner.chart(s).log_prob / LN2 for s in test) / len(test)
        rng = np.random.default_rng(args.seed)
        for name, whole in (("own sentences (whole trees)", True), ("all samples", False)):
            gen, _ = learner.generate(1000, rng, max_len=20, whole_only=whole)
            out[name] = RS.generation_measures([t for t, _ in gen], train_set, corpus_set, grams, (3, 5))
        out["whole trees (train)"] = float(np.mean([len(t.roots) == 1 for t in learner.trees]))
        return out
    res["off"] = ev()
    res["splits"] = refined(learner)
    res["on"] = ev() if res["splits"]["changed"] else res["off"]
    return res


def run_characters(args, learner):
    import run_characters as RCH
    import run_characters_memory as RCM
    from trellis2.characters import default_ids_path, load_characters, parse_prefix, placements
    path = default_ids_path()
    chars, _ = load_characters(path, seed=args.seed)
    train, test = chars[:args.train], chars[args.train:args.train + 500]
    real = {parse_prefix(c.tokens): c.char for c in chars}
    attested = set()
    for c in chars:
        placements(parse_prefix(c.tokens), attested)
    plain, _ = RCM.database_components(path)
    res = {"night seconds": night(learner, [c.tokens for c in train])}

    def ev():
        out = grammar_info(learner.model.grammar)
        out.update(RCH.evaluate(learner.chart, learner.generate, train, test, real, attested, 1000, args.seed))
        out.pop("examples", None)
        rep = RCM.chunk_report(learner.model.memory, learner.model.grammar, "unsupervised",
                               [c.tokens for c in train], plain, recall=True)
        out["validity"] = {
            "chunk types": rep["chunk types"],
            "share of chunk types whose modal yield is an IDS component": rep["type modal class"]["component"],
            "specific reused units (purity >= 0.5 and a database component used as a proper part)":
                rep["share of types: purity >= 0.5 and a database component used as a proper part"],
            "strict top-50 component recall (modal yield of a chunk type with purity >= 0.5)":
                rep["recall"]["50"]["modal yield of a chunk type with purity >= 0.5"],
            "top-50 recall (modal yield of a chunk type)": rep["recall"]["50"]["modal yield of a chunk type"],
            "type modal class": rep["type modal class"], "mean purity": rep["mean purity"]}
        return out
    res["off"] = ev()
    res["splits"] = refined(learner)
    res["on"] = ev() if res["splits"]["changed"] else res["off"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("complex", "paper", "stories", "characters"), default="complex")
    ap.add_argument("--grammar", default="agree", help="complex grammar, or paper condition")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    kw = {"read_words": 2, "previous_word": True} if args.domain == "stories" else {}
    learner = UnsupervisedLearner(seed=args.seed, workers=args.workers, **kw)
    t0 = time.time()
    res = {"stories": run_stories, "paper": run_paper, "complex": run_complex,
           "characters": run_characters}[args.domain](args, learner)
    res.update(domain=args.domain, grammar=args.grammar, train=args.train, seed=args.seed,
               seconds=time.time() - t0, history=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                   for k, v in h.items()} for h in learner.history])
    name = f"{args.domain}_{args.grammar if args.domain in ('complex', 'paper') else ''}_{args.train}_s{args.seed}".replace("__", "_")
    with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str, ensure_ascii=False)
    off, on = res["off"], res["on"]
    print(f"[{name}] off {off.get('total_bits'):.1f} -> on {on.get('total_bits'):.1f} bits; splits {res['splits']}",
          flush=True)


if __name__ == "__main__":
    main()
