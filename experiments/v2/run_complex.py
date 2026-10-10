"""The unsupervised night on the complex synthetic grammars
(``complex_grammars.py``: agreement, centre embedding, nested relatives,
coordination, 6-30 words), where grammaticality is exact.

One run = one grammar, one training size, one seed. The seed permutes the
2,000 training sentences and the first ``--train`` are learned from sentences
alone (``UnsupervisedLearner``, the full night of ``run_unsupervised.py``,
sampling included). Reported:

* the night: total, model and data bits, symbols, chunk types, wall time per stage;
* the gold-tree grammar (``Trellis2`` on the gold trees of the same sentences):
  its code and the same measures, for comparison;
* held-out bits per sentence (summed over analyses), against word bigram and
  trigram models (add-0.1, as ``run_stories.py``);
* brackets: omission and commission (Langley & Stromsten) of the
  minimum-risk parse of held-out sentences and of the night's own (Viterbi)
  analyses of the training sentences against the gold trees; top-level pieces;
* generation, per 1,000 samples: grammatical (the target grammar accepts it),
  novel (not a training sentence), grammatical and novel;
* n-gram generation (orders 2-4, a temperature sweep) for the same three
  rates, and the n-gram's grammaticality at TRELLIS's novelty (interpolated);
* prompting as ``run_prompts_synthetic.py``: the first 1-4 words of held-out
  sentences completed by both grammars and by word bigram/trigram models.

Usage:
    python experiments/v2/run_complex.py --grammar agree --train 500 --seed 13 --workers 6
    python experiments/v2/run_complex.py --summarise
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from trellis2 import Trellis2  # noqa: E402
from trellis2.chart import Chart  # noqa: E402
from trellis2.evaluation import CFG, BracketTally  # noqa: E402
from trellis2.prompt import PromptChart  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402
import complex_grammars as CG  # noqa: E402
from run_prompts import ngram_complete, ngram_continuation_bits, ngram_counts  # noqa: E402

LN2 = math.log(2)
OUT = os.path.join(HERE, "results", "complex")


def crosses(a, b) -> bool:
    (i, j), (k, l) = a, b
    return i < k < j < l or k < i < l < j


# --------------------------------------------------------------------------- #
# n-grams

def ngram_bits(counts, order, vocab_size, test, alpha=0.1):
    total = 0.0
    for s in test:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            c = counts.get(tuple(seq[i - order + 1:i]), Counter())
            total -= math.log2((c[seq[i]] + alpha) / (sum(c.values()) + alpha * vocab_size))
    return total / len(test)


def ngram_sample(counts, order, rng, temperature=1.0, max_len=45):
    """A sentence from the maximum-likelihood n-gram, each next-word
    distribution raised to 1/temperature and renormalised."""
    out = []
    seq = ["<s>"] * (order - 1)
    while len(out) < max_len:
        c = counts[tuple(seq[len(seq) - order + 1:])]
        words = list(c)
        p = np.array([c[w] for w in words], float) ** (1.0 / temperature)
        w = words[rng.choice(len(words), p=p / p.sum())]
        if w == "</s>":
            break
        out.append(w)
        seq.append(w)
    return out


BLIND = {}   # the target grammar with number erased, per process (set in ``run``)


def rates(samples, cfg, train_set):
    gram = [cfg.recognizes(s) for s in samples]
    novel = [tuple(s) not in train_set for s in samples]
    out = {"grammatical": float(np.mean(gram)), "novel": float(np.mean(novel)),
           "grammatical and novel": float(np.mean([a and b for a, b in zip(gram, novel)])),
           "mean length": float(np.mean([len(s) for s in samples]))}
    if "cfg" in BLIND:
        # Number erased: accepted here but not by the target grammar = an
        # agreement error only; rejected here = a structural error.
        blind = [BLIND["cfg"].recognizes(s) for s in samples]
        out["grammatical ignoring number"] = float(np.mean(blind))
        out["agreement errors only"] = float(np.mean([b and not a for a, b in zip(gram, blind)]))
    return out


# --------------------------------------------------------------------------- #

def evaluate(model, test, cfg, train_set, seed, n_gen, train_examples=None, trees=None):
    """Held-out bits and brackets; generation rates; brackets of the
    training analyses (``trees``) against gold."""
    t0 = time.time()
    tally, noncross, total, bits, tops = BracketTally(), 0, 0, 0.0, []
    for e in test:
        chart = model.chart(e.tokens)
        tree = chart.mbr_tree()
        tally.add(e.tree, tree)
        bits += -chart.log_prob / LN2
        tops.append(len(chart.viterbi_tree().roots))
        gold = e.tree.brackets()
        for s in tree.brackets():
            total += 1
            noncross += not any(crosses(s, g) for g in gold)
    out = {"test_bits_per_sentence": bits / len(test), "test_top_level_chunks": float(np.mean(tops)),
           "test_whole_trees": float(np.mean([t == 1 for t in tops])),
           "omission": tally.omission, "parse_commission": tally.commission,
           "exact_trees": tally.exact_match, "non_crossing": noncross / max(total, 1)}
    t1 = time.time()
    samples, rejected = model.generate(n_gen, np.random.default_rng(seed), max_len=45)
    toks = [s for s, _ in samples]
    out["generation"] = rates(toks, cfg, train_set)
    out["generation"]["redrawn (over 45 words)"] = rejected
    out["generation"]["whole trees"] = float(np.mean([len(t.roots) == 1 for _, t in samples]))
    # The same rates among the samples derived as one tree.
    whole = [s for s, t in samples if len(t.roots) == 1]
    if whole:
        out["generation, whole trees only"] = rates(whole, cfg, train_set)
    out["samples"] = [" ".join(s) for s in toks]
    out["sample_examples"] = [t.to_string(s) + ("  ✓" if cfg.recognizes(s) else "  ✗")
                              + ("" if tuple(s) in train_set else "  (new)") for s, t in samples[:30]]
    if trees is not None and train_examples is not None:
        tt = BracketTally()
        for e, t in zip(train_examples, trees):
            tt.add(e.tree, t)
        out["train_omission"], out["train_commission"] = tt.omission, tt.commission
        out["train_whole_trees"] = float(np.mean([len(t.roots) == 1 for t in trees]))
        out["train_exact_trees"] = tt.exact_match
    out["eval_seconds"] = {"parse": t1 - t0, "generate": time.time() - t1}
    return out


def prompting(grammars, train, test, cfg, seed, n=5, n_sentences=40):
    """As ``run_prompts_synthetic.py``: prompts are the first 1-4 tokens of
    held-out sentences (at least one token left), each completed ``n`` times."""
    sentences = [list(e.tokens) for e in train]
    train_set = {tuple(s) for s in sentences}
    starts = {tuple(s[:k]) for s in sentences for k in range(1, 5)}
    vocab = sorted({w for s in sentences for w in s} | {"</s>"})
    ngrams = [(order, ngram_counts(sentences, order)) for order in (3, 2, 1)]
    prompts, seen_p = [], set()
    for e in test[:n_sentences]:
        for k in range(1, min(4, len(e.tokens) - 1) + 1):
            p = tuple(e.tokens[:k])
            if p not in seen_p:
                seen_p.add(p)
                prompts.append((p, list(e.tokens)))
    names = list(grammars) + ["word bigram", "word trigram"]
    rng = np.random.default_rng(seed)
    tally, bits, examples = defaultdict(Counter), defaultdict(list), []
    for p, sentence in prompts:
        group = "seen" if p in starts else "unseen"
        charts = {}
        for name, g in grammars.items():
            pc = PromptChart(g, list(p))
            if np.isfinite(pc.log_prefix):
                charts[name] = pc
                bits[(name, group)].append(-(Chart(g, sentence).log_prob - pc.log_prefix) / LN2)
        bits[("word bigram", group)].append(ngram_continuation_bits(ngrams[1][1], 2, len(vocab), p, sentence))
        bits[("word trigram", group)].append(ngram_continuation_bits(ngrams[0][1], 3, len(vocab), p, sentence))
        for name in names:
            for _ in range(n):
                if name in grammars:
                    if name not in charts:
                        tally[(name, group)]["impossible"] += 1
                        continue
                    c = None
                    while c is None:
                        c = charts[name].complete(rng, max_len=45)
                    tokens = c.tokens
                else:
                    tokens = ngram_complete(ngrams[1:] if name == "word bigram" else ngrams, list(p), rng, max_len=45)
                ok, new = cfg.recognizes(tokens), tuple(tokens) not in train_set
                t = tally[(name, group)]
                t["n"] += 1
                t["grammatical"] += ok
                t["novel"] += new
                t["novel and grammatical"] += ok and new
                if len(examples) < 15 and name == "TRELLIS v2, sentences alone" and group == "unseen":
                    examples.append(f"{' '.join(p)} → {' '.join(tokens)} {'✓' if ok else '✗'}")
    return {"prompts": dict(Counter("seen" if p in starts else "unseen" for p, _ in prompts)),
            "tally": {f"{k[0]}|{k[1]}": dict(v) for k, v in tally.items()},
            "bits": {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in bits.items() if v},
            "examples": examples}


def run(args) -> dict:
    t_start = time.time()
    train_all, held = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(train_all))
    train = [train_all[i] for i in order[:args.train]]
    test = held[:args.heldout]
    cfg = CFG(CG.build(args.grammar))
    BLIND["cfg"] = CFG(CG.number_blind(args.grammar))
    train_set = {tuple(e.tokens) for e in train}
    sentences = [list(e.tokens) for e in train]
    res = {"grammar": args.grammar, "train": len(train), "heldout": len(test), "seed": args.seed,
           "lengths": {"train mean": float(np.mean([len(e.tokens) for e in train])),
                       "train max": int(max(len(e.tokens) for e in train))}}
    out_dir = os.path.join(args.out, f"{args.grammar}_{args.train}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    def save():
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)

    # Baselines first (cheap).
    vocab_size = len({w for s in sentences for w in s} | {"</s>"})
    counts = {k: ngram_counts(sentences, k) for k in (1, 2, 3, 4)}
    res["ngram held-out bits per sentence"] = {f"{k}-gram": ngram_bits(counts[k], k, vocab_size, [e.tokens for e in test])
                                               for k in (1, 2, 3)}
    rng = np.random.default_rng(args.seed)
    sweep = {}
    for k in (2, 3, 4):
        for T in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
            smp = [ngram_sample(counts[k], k, rng, T) for _ in range(args.n_gen)]
            sweep[f"{k}-gram T={T}"] = dict(rates(smp, cfg, train_set), order=k, temperature=T)
    res["ngram generation"] = sweep
    save()

    # The night.
    t0 = time.time()
    learner = UnsupervisedLearner(seed=args.seed, workers=args.workers)
    for e in train:
        learner.observe(e.tokens)
    g = learner.sleep()
    night = time.time() - t0
    info = g.info
    res["night seconds"] = night
    res["unsupervised"] = {"total_bits": info["total bits"], "model_bits": info["model bits"],
                           "data_bits": info["data bits"], "symbols": g.K, "rule_classes": g.M,
                           "chunk_types": info["chunk types"]}
    res["history"] = [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in h.items()}
                      for h in learner.history]
    print(f"[{args.grammar} {args.train} s{args.seed}] night {night:.0f}s: {info['total bits']:.0f} bits, "
          f"{g.K} symbols, {info['chunk types']} chunk types", flush=True)
    save()
    res["unsupervised"].update(evaluate(learner, test, cfg, train_set, args.seed, args.n_gen, train, learner.trees))
    u = res["unsupervised"]
    print(f"  unsupervised: test {u['test_bits_per_sentence']:.1f} b/s, omission {u['omission']:.3f}, "
          f"commission {u['parse_commission']:.3f}, generation {u['generation']}", flush=True)
    res["unsupervised"]["held-out analyses"] = [learner.parse(e.tokens).to_string(e.tokens) for e in test[:15]]
    save()

    with open(os.path.join(out_dir, "grammar_night.pkl"), "wb") as f:
        pickle.dump({"grammar": dataclasses.replace(g, symbol_nodes=[], rule_nodes=[]),
                     "train": [e.sentence for e in train], "trees": learner.trees}, f)

    # The gold-tree grammar.
    if not args.no_supervised:
        t0 = time.time()
        sup = Trellis2(seed=args.seed)
        for e in train:
            sup.learn(e.tokens, e.tree)
        sg = sup.consolidate()
        res["supervised seconds"] = time.time() - t0
        with open(os.path.join(out_dir, "grammar_gold.pkl"), "wb") as f:
            pickle.dump({"grammar": dataclasses.replace(sg, symbol_nodes=[], rule_nodes=[])}, f)
        res["supervised"] = {"total_bits": sg.info["total bits"], "model_bits": sg.info["model bits"],
                             "data_bits": sg.info["data bits"], "symbols": sg.K, "rule_classes": sg.M,
                             "chunk_types": sg.info["chunk types"]}
        res["supervised"].update(evaluate(sup, test, cfg, train_set, args.seed, args.n_gen))
        s = res["supervised"]
        print(f"  gold trees: {s['total_bits']:.0f} bits, test {s['test_bits_per_sentence']:.1f} b/s, "
              f"generation {s['generation']}", flush=True)
        save()

    # Matched novelty: each n-gram's grammaticality where its novelty equals TRELLIS's.
    nov = res["unsupervised"]["generation"]["novel"]
    matched = {}
    for k in (2, 3, 4):
        pts = sorted((v["novel"], v["grammatical"], v["grammatical and novel"]) for v in sweep.values() if v["order"] == k)
        xs = [p[0] for p in pts]
        if xs[0] <= nov <= xs[-1]:
            matched[f"{k}-gram"] = {"grammatical": float(np.interp(nov, xs, [p[1] for p in pts])),
                                    "grammatical and novel": float(np.interp(nov, xs, [p[2] for p in pts]))}
        else:
            matched[f"{k}-gram"] = {"outside the sweep's novelty range": [xs[0], xs[-1]]}
    res["ngram at TRELLIS novelty"] = {"novelty": nov, **matched}
    save()

    # Prompting.
    if not args.no_prompts:
        t0 = time.time()
        grammars = {"TRELLIS v2, sentences alone": g}
        if not args.no_supervised:
            grammars["TRELLIS v2, gold trees"] = sg
        res["prompting"] = prompting(grammars, train, test, cfg, args.seed, n=args.prompt_n)
        res["prompting seconds"] = time.time() - t0
        save()
    res["seconds"] = time.time() - t_start
    save()
    print(f"  done in {res['seconds']:.0f}s -> {out_dir}", flush=True)
    return res


# --------------------------------------------------------------------------- #

def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def summarise(out: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "*", "results.json"))):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    order = {g: i for i, g in enumerate(CG.GRAMMARS)}
    rows.sort(key=lambda r: (order.get(r["grammar"], 9), r["train"], r["seed"]))
    L = ["Complex synthetic grammars, learned from sentences alone (full night). Generation per 1,000 samples "
         "(all samples, forests included); held-out = first 300 held-out sentences. "
         "Brackets: omission / commission against the gold trees (held-out minimum-risk parses; training: the night's analyses).", "",
         "| Grammar | Train | Seed | Night (s) | Train bits (night / gold trees) | Symbols / chunk types | "
         "Held-out bits/sent (night / gold trees / 2-gram / 3-gram) | Held-out brackets om. / com. | "
         "Train brackets om. / com. | Whole trees (train) | Gen. grammatical | Novel | Grammatical & novel | "
         "Gold-tree grammar: grammatical / novel / both | Ignoring number: night / gold trees |",
         "|" + "---|" * 15]
    for r in rows:
        u = r.get("unsupervised", {})
        s = r.get("supervised", {})
        ng = r.get("ngram held-out bits per sentence", {})
        gen = u.get("generation", {})
        sgen = s.get("generation", {})
        if "test_bits_per_sentence" not in u:
            L.append(f"| {r['grammar']} | {r['train']} | {r['seed']} | (running or failed) |" + " |" * 10)
            continue
        L.append(
            f"| {r['grammar']} | {r['train']} | {r['seed']} | {r.get('night seconds', 0):.0f} | "
            f"{u['total_bits']:.0f} / {s.get('total_bits', float('nan')):.0f} | {u['symbols']} / {u['chunk_types']} | "
            f"{u['test_bits_per_sentence']:.1f} / {s.get('test_bits_per_sentence', float('nan')):.1f} / "
            f"{ng.get('2-gram', float('nan')):.1f} / {ng.get('3-gram', float('nan')):.1f} | "
            f"{pct(u['omission'])} / {pct(u['parse_commission'])} | "
            f"{pct(u.get('train_omission'))} / {pct(u.get('train_commission'))} | {pct(u.get('train_whole_trees'))} | "
            f"{pct(gen.get('grammatical'))} | {pct(gen.get('novel'))} | {pct(gen.get('grammatical and novel'))} | "
            f"{pct(sgen.get('grammatical'))} / {pct(sgen.get('novel'))} / {pct(sgen.get('grammatical and novel'))} | "
            f"{pct(gen.get('grammatical ignoring number'))} / {pct(sgen.get('grammatical ignoring number'))} |")
    L += ["", "n-gram generation at TRELLIS's novelty (interpolated over a temperature sweep T = 0.5–2; * = nearest sweep point, within 1 point of novelty):", "",
          "| Grammar | Train | Seed | TRELLIS novelty | TRELLIS grammatical | 2-gram grammatical | 3-gram grammatical | 4-gram grammatical |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r.get("ngram at TRELLIS novelty")
        if not m:
            continue
        nov, cells = m["novelty"], []
        for k in (2, 3, 4):
            # Interpolated over the sweep; within 1 point of the sweep's
            # novelty range, the nearest sweep point (marked *).
            pts = sorted((v["novel"], v["grammatical"]) for v in r["ngram generation"].values() if v["order"] == k)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if xs[0] <= nov <= xs[-1]:
                cells.append(pct(float(np.interp(nov, xs, ys))))
            elif min(abs(nov - xs[0]), abs(nov - xs[-1])) <= 0.01:
                cells.append(pct(ys[0] if abs(nov - xs[0]) < abs(nov - xs[-1]) else ys[-1]) + "*")
            else:
                cells.append(f"novelty only {100 * xs[0]:.0f}–{100 * xs[-1]:.0f}%")
        L.append(f"| {r['grammar']} | {r['train']} | {r['seed']} | {pct(m['novelty'])} | "
                 f"{pct(r['unsupervised']['generation']['grammatical'])} | " + " | ".join(cells) + " |")
    L += ["", "n-gram generation at T = 1 (grammatical / novel / both):", "",
          "| Grammar | Train | Seed | 2-gram | 3-gram | 4-gram | Ignoring number: 2 / 3 / 4-gram |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        sw = r.get("ngram generation", {})
        cells = []
        for k in (2, 3, 4):
            v = sw.get(f"{k}-gram T=1.0")
            cells.append("–" if not v else f"{pct(v['grammatical'])} / {pct(v['novel'])} / {pct(v['grammatical and novel'])}")
        blind = " / ".join(pct(sw.get(f"{k}-gram T=1.0", {}).get("grammatical ignoring number")) for k in (2, 3, 4))
        L.append(f"| {r['grammar']} | {r['train']} | {r['seed']} | " + " | ".join(cells) + f" | {blind} |")
    models = ("TRELLIS v2, sentences alone", "TRELLIS v2, gold trees", "word bigram", "word trigram")
    L += ["", "Prompting (first 1–4 words of 40 held-out sentences, 5 completions each): grammatical / grammatical and novel, "
          "for prompts unseen (seen) as the beginning of a training sentence:", "",
          "| Grammar | Train | Seed | Prompts unseen / seen | " + " | ".join(models) + " |", "|---|---|---|---|" + "---|" * len(models)]
    for r in rows:
        p = r.get("prompting")
        if not p:
            continue
        cells = []
        for m in models:
            parts = []
            for grp in ("unseen", "seen"):
                t = p["tally"].get(f"{m}|{grp}", {})
                tot = t.get("n", 0) + t.get("impossible", 0)
                parts.append("–" if not tot else f"{100 * t.get('grammatical', 0) / tot:.0f}% / {100 * t.get('novel and grammatical', 0) / tot:.0f}%")
            cells.append(f"{parts[0]} ({parts[1]})")
        L.append(f"| {r['grammar']} | {r['train']} | {r['seed']} | {p['prompts'].get('unseen', 0)} / {p['prompts'].get('seen', 0)} | "
                 + " | ".join(cells) + " |")
    L += ["", "Wall times (s): night; evaluation; gold-tree grammar; prompting; total:", ""]
    for r in rows:
        u = r.get("unsupervised", {})
        ev = u.get("eval_seconds", {})
        L.append(f"- {r['grammar']} {r['train']} s{r['seed']}: night {r.get('night seconds', float('nan')):.0f}; "
                 f"eval {sum(ev.values()) if ev else float('nan'):.0f}; gold {r.get('supervised seconds', float('nan')):.0f}; "
                 f"prompts {r.get('prompting seconds', float('nan')):.0f}; total {r.get('seconds', float('nan')):.0f}")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default="agree", choices=CG.GRAMMARS)
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--prompt-n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-supervised", action="store_true")
    ap.add_argument("--no-prompts", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.summarise:
        summarise(args.out)
        return
    run(args)


if __name__ == "__main__":
    main()
