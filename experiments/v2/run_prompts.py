"""Prompting a grammar learned from simple English (``run_stories.py``):
complete the beginnings of held-out sentences.

The prompts are the first one, two or three words of the held-out sentences
of a ``run_stories.py`` run (its ``grammar.pkl``). A prompt is *seen* if some
training sentence begins with it and *unseen* otherwise: the unseen ones ask
the grammar to continue a beginning it never read. Each prompt is completed
by TRELLIS v2 (``Trellis2.complete``: a scaffolded parse of the prompt drawn
from the grammar's posterior, then its open chunks finished), at temperature
1 and 0.5, and by word bigram and trigram models trained on the same
sentences (continuing from the prompt's last words, backing off to a shorter
history the model has not seen). Reported for each: completed sentences
real (in TinyStories), with every word triple in TinyStories (edges
included), new (not a training sentence), and both new and with every triple
attested; and the code of the held-out sentence's actual continuation given
its prompt.

Usage:
    python experiments/v2/run_prompts.py --grammar experiments/v2/results/stories/grammar.pkl \\
        --out experiments/v2/results/prompts
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.chart import Chart  # noqa: E402
from trellis2.prompt import PromptChart  # noqa: E402
from trellis2.stories import read_sentences  # noqa: E402

LN2 = math.log(2)


def ngram_counts(train, order):
    counts = defaultdict(Counter)
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            counts[tuple(seq[i - order + 1:i])][seq[i]] += 1
    return counts


def ngram_complete(models, prompt, rng, max_len=20):
    """Continue ``prompt`` with the highest-order model whose history has
    been seen (``models``: counts by order, highest first)."""
    out = list(prompt)
    while len(out) < max_len:
        for order, counts in models:
            h = tuple((["<s>"] * (order - 1) + out)[len(out):]) if order > 1 else ()
            c = counts.get(h)
            if c:
                break
        words = list(c)
        w = words[rng.choice(len(words), p=np.array([c[x] for x in words], float) / sum(c.values()))]
        if w == "</s>":
            break
        out.append(w)
    return out


def ngram_continuation_bits(counts, order, vocab_size, prompt, sentence, alpha=0.1):
    """Bits of the sentence's continuation after its prompt (add-alpha)."""
    seq = ["<s>"] * (order - 1) + list(sentence) + ["</s>"]
    bits = 0.0
    for i in range(order - 1 + len(prompt), len(seq)):
        h = tuple(seq[i - order + 1:i])
        c = counts.get(h, Counter())
        bits -= math.log2((c[seq[i]] + alpha) / (sum(c.values()) + alpha * vocab_size))
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default=os.path.join(HERE, "results", "stories", "grammar.pkl"))
    ap.add_argument("--per-length", type=int, default=100, help="prompts of each length (1, 2, 3 words)")
    ap.add_argument("--n", type=int, default=5, help="completions per prompt and model")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "prompts"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = pickle.load(open(args.grammar, "rb"))
    g, train, test = d["grammar"], [list(s) for s in d["train"]], [list(s) for s in d["test"]]
    corpus = read_sentences()
    corpus_set = {tuple(s) for s in corpus}
    padded = [("<s>",) + tuple(s) + ("</s>",) for s in corpus]
    triples = {p[i:i + 3] for p in padded for i in range(len(p) - 2)}
    train_set = {tuple(s) for s in train}
    starts = {tuple(s[:k]) for s in train for k in (1, 2, 3)}
    lo, hi = min(map(len, train)), max(map(len, train))

    def every(s):
        p = ("<s>",) + tuple(s) + ("</s>",)
        return all(p[i:i + 3] in triples for i in range(len(p) - 2))

    prompts = []
    for k in (1, 2, 3):
        seen = []
        for s in test:
            if len(s) > k and tuple(s[:k]) not in {p for p, _ in seen}:
                seen.append((tuple(s[:k]), s))
        prompts += seen[:args.per_length]
    rng = np.random.default_rng(args.seed)
    vocab = sorted({w for s in train for w in s} | {"</s>"})
    models = [(order, ngram_counts(train, order)) for order in (3, 2, 1)]
    results = {"prompts": len(prompts), "completions per prompt": args.n, "rows": [], "examples": []}
    names = ("TRELLIS v2, temperature 1", "TRELLIS v2, temperature 0.5", "word bigram", "word trigram")
    tally = {(name, group): Counter() for name in names for group in ("seen", "unseen")}
    held = {(name, group): [] for name in ("TRELLIS v2", "word bigram", "word trigram") for group in ("seen", "unseen")}
    for prompt, sentence in prompts:
        group = "seen" if prompt in starts else "unseen"
        pc = PromptChart(g, list(prompt))
        if not np.isfinite(pc.log_prefix):
            continue
        # The actual continuation's code given the prompt.
        held[("TRELLIS v2", group)].append(-(Chart(g, sentence).log_prob - pc.log_prefix) / LN2)
        held[("word bigram", group)].append(ngram_continuation_bits(models[1][1], 2, len(vocab), prompt, sentence))
        held[("word trigram", group)].append(ngram_continuation_bits(models[0][1], 3, len(vocab), prompt, sentence))
        for name in names:
            for _ in range(args.n):
                if name.startswith("TRELLIS"):
                    c = None
                    while c is None:
                        c = pc.complete(rng, temperature=1.0 if name.endswith(" 1") else 0.5, max_len=20)
                    tokens = c.tokens
                    if len(results["examples"]) < 60 and name.endswith(" 1"):
                        results["examples"].append({"group": group, "prompt": " ".join(prompt),
                                                    "analysis": render(c), "log_prob": c.log_prob})
                else:
                    tokens = ngram_complete(models[1:] if name == "word bigram" else models, list(prompt), rng)
                t = tally[(name, group)]
                t["n"] += 1
                if lo <= len(tokens) <= hi:
                    t["in range"] += 1
                    new, ok = tuple(tokens) not in train_set, every(tokens)
                    t["real"] += tuple(tokens) in corpus_set
                    t["every triple"] += ok
                    t["new"] += new
                    t["new and every triple"] += new and ok
    for (name, group), t in tally.items():
        r = t["in range"] or 1
        results["rows"].append({"model": name, "prompts": group, "completions": t["n"],
                                "of the training length": t["in range"] / max(t["n"], 1),
                                "real": t["real"] / r, "every triple attested": t["every triple"] / r,
                                "new": t["new"] / r, "new and every triple attested": t["new and every triple"] / r})
    results["held-out continuation bits"] = {f"{name}, {group} prompts": float(np.mean(v)) for (name, group), v in held.items() if v}
    results["prompt counts"] = {group: sum(1 for p, _ in prompts if (p in starts) == (group == "seen")) for group in ("seen", "unseen")}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    lines = [f"Prompts: the first 1–3 words of held-out sentences ({results['prompt counts']['seen']} seen as the beginning "
             f"of a training sentence, {results['prompt counts']['unseen']} unseen), each completed {args.n} times. "
             f"Grammar: `{os.path.relpath(args.grammar, ROOT)}`.", "",
             "| Completions | Prompts | Of the training length | Real | Every word triple attested | New | New and every triple attested |",
             "|---|---|---|---|---|---|---|"]
    for row in results["rows"]:
        lines.append(f"| {row['model']} | {row['prompts']} | {row['of the training length']:.0%} | {row['real']:.1%} | "
                     f"{row['every triple attested']:.1%} | {row['new']:.1%} | {row['new and every triple attested']:.1%} |")
    lines += ["", "Bits of the held-out sentence's actual continuation, given its prompt:", "",
              "| Model | Seen prompts | Unseen prompts |", "|---|---|---|"]
    for name in ("TRELLIS v2", "word bigram", "word trigram"):
        v = results["held-out continuation bits"]
        lines.append(f"| {name} | {v.get(f'{name}, seen prompts', float('nan')):.2f} | {v.get(f'{name}, unseen prompts', float('nan')):.2f} |")
    lines += ["", "Completions (the prompt, then | ; the analysis as drawn, open chunks of the scaffold in ⟨ ⟩):", ""]
    for ex in results["examples"][:30]:
        lines.append(f"    {ex['group']:6s}  {ex['prompt']:18s} → {ex['analysis']}")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def render(c) -> str:
    """The completed analysis, brackets for chunks, ⟨ ⟩ around the scaffold's
    open chunks, and | where the prompt ends."""
    t, tokens, open_spans = c.tree, c.tokens, set(c.open_spans)

    def rec(i, j):
        if j - i == 1:
            body = tokens[i]
        else:
            k = t.split[(i, j)]
            body = f"{rec(i, k)} {rec(k, j)}"
        if j - i == 1 and (i, j) not in open_spans:
            return body
        return f"⟨{body}⟩" if (i, j) in open_spans else f"[{body}]"
    text = " · ".join(rec(i, j) for i, j in t.roots)
    words = text.split(" ")
    # Mark the end of the prompt after its last token.
    count, out = 0, []
    for w in words:
        out.append(w)
        count += sum(1 for _ in [w.strip("[]⟨⟩·")] if w.strip("[]⟨⟩·"))
        if count == c.prompt_length:
            out.append("|")
            count = -10 ** 9
    return " ".join(out)


if __name__ == "__main__":
    main()
