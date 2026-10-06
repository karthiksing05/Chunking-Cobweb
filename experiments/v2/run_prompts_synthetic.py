"""Prompting on the paper's synthetic corpora, where every completion can be
judged exactly by the target grammar.

For each condition and seed (the v1 splits: 320 training and 40 held-out
sentences), the prompts are the first 1–4 tokens of the held-out sentences
(at least one token is left to complete). A prompt is *seen* if some training
sentence begins with it. Each prompt is completed ``--n`` times by

* TRELLIS v2 learned from the gold trees (``Trellis2``),
* TRELLIS v2 learned from the sentences alone (``UnsupervisedLearner``),
* word bigram and trigram models trained on the same sentences (continuing
  from the prompt's last tokens, backing off to a shorter history when the
  model has not seen it).

Reported: completions the target grammar accepts (grammatical), completions
that are not training sentences (novel), and both: the coherent new
combinations. The grammars' codes of the held-out sentences' actual
continuations given their prompts are reported too.

Usage:
    python experiments/v2/run_prompts_synthetic.py --out experiments/v2/results/prompts_synthetic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from trellis2 import Trellis2, load_corpus, v1_split  # noqa: E402
from trellis2.chart import Chart  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root, target_grammar  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from trellis2.prompt import PromptChart  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402
from run_prompts import ngram_complete, ngram_continuation_bits, ngram_counts  # noqa: E402

LN2 = math.log(2)
MODELS = ("TRELLIS v2, gold trees", "TRELLIS v2, sentences alone", "word bigram", "word trigram")


def run_one(condition: str, seed: int, n: int, data_root: str) -> dict:
    examples = load_corpus(os.path.join(data_root, CONDITIONS[condition]))
    train, test = v1_split(examples, seed)
    cfg = CFG(target_grammar(condition))
    sup = Trellis2(seed=seed)
    for ex in train:
        sup.learn(ex.tokens, ex.tree)
    sup.consolidate()
    unsup = UnsupervisedLearner(seed=seed)
    for ex in train:
        unsup.observe(ex.tokens)
    unsup.sleep()
    grammars = {MODELS[0]: sup.grammar, MODELS[1]: unsup.grammar}
    sentences = [list(ex.tokens) for ex in train]
    train_set = {tuple(s) for s in sentences}
    starts = {tuple(s[:k]) for s in sentences for k in range(1, 5)}
    vocab = sorted({w for s in sentences for w in s} | {"</s>"})
    ngrams = [(order, ngram_counts(sentences, order)) for order in (3, 2, 1)]
    prompts = []
    for ex in test:
        for k in range(1, min(4, len(ex.tokens) - 1) + 1):
            p = tuple(ex.tokens[:k])
            if p not in {q for q, _ in prompts}:
                prompts.append((p, list(ex.tokens)))
    rng = np.random.default_rng(seed)
    tally = defaultdict(Counter)
    bits = defaultdict(list)
    examples_out = []
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
        for name in MODELS:
            for _ in range(n):
                if name in grammars:
                    if name not in charts:          # the grammar cannot begin a sentence so
                        tally[(name, group)]["impossible"] += 1
                        continue
                    c = None
                    while c is None:
                        c = charts[name].complete(rng, max_len=40)
                    tokens = c.tokens
                    if len(examples_out) < 12 and name == MODELS[1] and group == "unseen":
                        examples_out.append({"prompt": " ".join(p), "completion": " ".join(tokens),
                                             "grammatical": cfg.recognizes(tokens)})
                else:
                    tokens = ngram_complete(ngrams[1:] if name == "word bigram" else ngrams, list(p), rng, max_len=40)
                t = tally[(name, group)]
                ok, new = cfg.recognizes(tokens), tuple(tokens) not in train_set
                t["n"] += 1
                t["grammatical"] += ok
                t["novel"] += new
                t["novel and grammatical"] += ok and new
    return {"condition": condition, "seed": seed, "prompts": dict(Counter("seen" if p in starts else "unseen" for p, _ in prompts)),
            "tally": {f"{k[0]}|{k[1]}": dict(v) for k, v in tally.items()},
            "bits": {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in bits.items() if v},
            "examples": examples_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="13,17,7,42,100")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "prompts_synthetic"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(c, int(s)) for c in args.conditions.split(",") for s in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_one, *zip(*jobs), [args.n] * len(jobs), [args.data_root] * len(jobs)))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(runs, f, indent=1)
    lines = [f"Prompts: the first 1–4 tokens of the held-out sentences (v1 splits, seeds {args.seeds}), each completed "
             f"{args.n} times; completions judged by the target grammar. Rates summed over seeds.", ""]
    for group in ("seen", "unseen"):
        lines += [f"**Prompts {group} as the beginning of a training sentence**", "",
                  "| Condition | Prompts | " + " | ".join(f"{m}: grammatical / novel and grammatical" for m in MODELS) + " |",
                  "|---|---|" + "---|" * len(MODELS)]
        for cond in args.conditions.split(","):
            rs = [r for r in runs if r["condition"] == cond]
            cells = []
            for m in MODELS:
                t = Counter()
                for r in rs:
                    t.update(r["tally"].get(f"{m}|{group}", {}))
                total = t["n"] + t["impossible"]
                cells.append("–" if not total else
                             f"{t['grammatical'] / total:.1%} / {t['novel and grammatical'] / total:.1%}")
            n_prompts = sum(r["prompts"].get(group, 0) for r in rs)
            lines.append(f"| {cond} | {n_prompts} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Bits of the held-out sentences' actual continuations given their prompts (mean over prompts and seeds):", "",
              "| Condition | " + " | ".join(f"{m} (seen / unseen)" for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for cond in args.conditions.split(","):
        rs = [r for r in runs if r["condition"] == cond]
        cells = []
        for m in MODELS:
            vals = [tuple(r["bits"].get(f"{m}|{g}", float("nan")) for g in ("seen", "unseen")) for r in rs]
            a, b = np.nanmean([v[0] for v in vals]), np.nanmean([v[1] for v in vals])
            cells.append(f"{a:.1f} / {b:.1f}")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "Completions of unseen prompts by TRELLIS v2 from sentences alone (✓ grammatical):", ""]
    for r in runs[:6]:
        for ex in r["examples"][:3]:
            lines.append(f"    {r['condition']:9s} {ex['prompt']:24s} → {ex['completion']} {'✓' if ex['grammatical'] else '✗'}")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
