"""TRELLIS v2 on Chinese characters: a first domain beyond language.

Each character is the prefix sequence of its full IDS decomposition (see
``trellis2/characters.py``), for example 湖 = ⿰ 氵 ⿰ 古 月. Characters of
at most 11 tokens from the main CJK block are shuffled with the seed; the
first ``--train`` are learned and the next ``--test`` are held out.

* supervised: TRELLIS v2 learns the gold IDS structures as token sequences
  with their trees (operators are tokens);
* relational: TRELLIS v2 learns the same structures with each operator as
  the relation that joins two parts, and each part's slot in its description;
* unsupervised: TRELLIS v2 learns from the sequences alone (one night);
* unigram and bigram token models (add-1/2) as references for held-out bits.

Reported: held-out bits per character; bracket omission and commission
against the gold structure (single tokens and the whole character excluded);
and for generated characters the share that is well formed (a composed
character whose operators all have their parts), that places
every component only where real characters do, that is a real character
(memorized from training, or a held-out real character rediscovered), and
that is novel. Generated characters are drawn by composing component glyphs
in the boxes their operators define.

Usage:
    python experiments/v2/run_characters.py --out experiments/v2/results/characters
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import Trellis2  # noqa: E402
from trellis2.characters import (OPERATORS, CharacterMemory, canonical, default_ids_path,  # noqa: E402
                                 from_relational, load_characters, parse_prefix, placements,
                                 sample_structure, structure_log_prob, structure_tokens,
                                 to_relational)
from trellis2.treebank import evaluable  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402

LN2 = math.log(2)
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"   # components and IDS operators
SURFACE, INK, INK2, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#2a78d6", "#eb6834"


def ngram_bits(train, test, order: int, alpha: float = 0.5) -> float:
    vocab = {t for c in train for t in c.tokens} | {"</s>", "<unk>"}
    counts, ctx = Counter(), Counter()
    for c in train:
        seq = ["<s>"] + c.tokens + ["</s>"]
        for i in range(1, len(seq)):
            h = tuple(seq[max(0, i - order + 1):i]) if order > 1 else ()
            counts[(h, seq[i])] += 1
            ctx[h] += 1
    total = 0.0
    for c in test:
        seq = ["<s>"] + [t if t in vocab else "<unk>" for t in c.tokens] + ["</s>"]
        for i in range(1, len(seq)):
            h = tuple(seq[max(0, i - order + 1):i]) if order > 1 else ()
            total -= math.log2((counts[(h, seq[i])] + alpha) / (ctx[h] + alpha * len(vocab)))
    return total / len(test)


def tally_generated(structures, trained, real, attested, n_gen) -> Tuple[dict, dict]:
    """Shares of generated structures that are well formed, place every
    component where real characters do, are real (and held out), or novel."""
    tally = Counter()
    examples = {"novel": [], "rediscovered": []}
    for structure in structures:
        if not isinstance(structure, tuple):     # ill formed, or a lone component
            continue
        tally["well formed"] += 1
        if not placements(structure) <= attested:
            continue
        tally["positions attested"] += 1
        if structure in real:
            tally["real"] += 1
            if structure not in trained:
                tally["rediscovered"] += 1
                examples["rediscovered"].append(real[structure])
        else:
            tally["novel"] += 1
            examples["novel"].append(" ".join(structure_tokens(structure)))
    share = {k: tally[k] / n_gen for k in ("well formed", "positions attested", "real", "rediscovered", "novel")}
    return share, {k: v[:200] for k, v in examples.items()}


def evaluate(chart_of, generate, train, test, real, attested, n_gen, seed) -> dict:
    hit = gold = pred = 0
    bits = 0.0
    for c in test:
        chart = chart_of(c.tokens)
        n = len(c.tokens)
        g, p = evaluable(c.tree.brackets(), n), evaluable(chart.mbr_tree().brackets(), n)
        hit, gold, pred = hit + len(g & p), gold + len(g), pred + len(p)
        bits -= chart.log_prob / LN2
    samples, _ = generate(n_gen, np.random.default_rng(seed))
    trained = {parse_prefix(c.tokens) for c in train}
    share, examples = tally_generated([parse_prefix(t) for t, _ in samples], trained, real, attested, n_gen)
    return {"omission": 1 - hit / max(gold, 1), "commission": 1 - hit / max(pred, 1),
            "test_bits_per_character": bits / len(test), "generation": share, "examples": examples}


def evaluate_relational(g, train, test, real, attested, n_gen, seed) -> dict:
    """Held-out bits of the known structures (the inside pass over each tree);
    generated trees are checked in the same canonical form as the real ones."""
    bits = -sum(structure_log_prob(g, to_relational(parse_prefix(c.tokens))) for c in test) / LN2
    rng = np.random.default_rng(seed)
    structures = [from_relational(sample_structure(g, rng)) for _ in range(n_gen)]
    trained = {canonical(parse_prefix(c.tokens)) for c in train}
    share, examples = tally_generated(structures, trained, real, attested, n_gen)
    return {"omission": None, "commission": None, "test_bits_per_character": bits / len(test),
            "generation": share, "examples": examples}


def run(mode: str, seed: int, n_train: int, n_test: int, n_gen: int, path: str) -> dict:
    chars, _ = load_characters(path, seed=seed)
    train, test = chars[:n_train], chars[n_train:n_train + n_test]
    # Relational trees write three-part operators as two joins; real
    # characters are compared in the same form.
    canon = canonical if mode == "relational" else (lambda structure: structure)
    real = {canon(parse_prefix(c.tokens)): c.char for c in chars}
    attested = set()
    for c in chars:
        placements(canon(parse_prefix(c.tokens)), attested)
    t0 = time.time()
    if mode == "relational":
        memory = CharacterMemory()
        model = Trellis2(seed=seed, memory=memory)
        for c in train:
            memory.add_structure(to_relational(parse_prefix(c.tokens)))
        g = model.consolidate()
    elif mode == "supervised":
        model = Trellis2(seed=seed)
        for c in train:
            model.learn(c.tokens, c.tree)
        g = model.consolidate()
        chart_of, generate = model.chart, model.generate
    else:
        learner = UnsupervisedLearner(seed=seed)
        for c in train:
            learner.observe(c.tokens)
        g = learner.sleep()
        chart_of, generate = learner.chart, learner.generate
    seconds = time.time() - t0
    if mode == "relational":
        out = evaluate_relational(g, train, test, real, attested, n_gen, seed)
    else:
        out = evaluate(chart_of, generate, train, test, real, attested, n_gen, seed)
    out.update(mode=mode, seed=seed, train=n_train, test=n_test, seconds=seconds,
               total_bits=g.info["total bits"], symbols=g.K, rule_classes=g.M,
               chunk_types=g.info["chunk types"],
               baselines={"unigram": ngram_bits(train, test, 1), "bigram": ngram_bits(train, test, 2)})
    print(f"[{mode} s{seed}] {seconds:.0f}s, test {out['test_bits_per_character']:.1f} b/char, "
          f"generation {out['generation']}", flush=True)
    return out


# ---------------------------------------------------------------------- #
# Drawing characters from their structure
# ---------------------------------------------------------------------- #
_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        _FONT = FontProperties(fname=FONT_PATH)
    return _FONT


def _leaves(s) -> int:
    return 1 if isinstance(s, str) else sum(_leaves(p) for p in s[1:])


INNER = {"⿴": (0.22, 0.2, 0.78, 0.78), "⿵": (0.22, 0.0, 0.78, 0.68), "⿶": (0.22, 0.32, 0.78, 1.0),
         "⿷": (0.32, 0.2, 1.0, 0.8), "⿸": (0.36, 0.0, 1.0, 0.66), "⿹": (0.0, 0.0, 0.64, 0.66),
         "⿺": (0.36, 0.34, 1.0, 1.0)}


def draw_structure(ax, s, x0, y0, x1, y1, color=INK):
    if isinstance(s, str):
        path = TextPath((0, 0), s, size=1, prop=_font())
        bb = path.get_extents()
        if bb.width <= 0 or bb.height <= 0:
            return
        pad = 0.04 * min(x1 - x0, y1 - y0)
        sx, sy = (x1 - x0 - 2 * pad) / bb.width, (y1 - y0 - 2 * pad) / bb.height
        # Stretch a component to its box, but at most twice its own aspect
        # ratio, so that a single stroke stays a stroke; centre what is left.
        sx, sy = min(sx, 2 * sy), min(sy, 2 * sx)
        t = (Affine2D().translate(-bb.x0 - bb.width / 2, -bb.y0 - bb.height / 2).scale(sx, sy)
             .translate((x0 + x1) / 2, (y0 + y1) / 2))
        ax.add_patch(PathPatch(t.transform_path(path), facecolor=color, edgecolor="none"))
        return
    op, parts = s[0], s[1:]
    weights = np.array([_leaves(p) ** 0.6 for p in parts], float)
    cuts = np.concatenate([[0], np.cumsum(weights / weights.sum())])
    if op in "⿰⿲":
        for p, a, b in zip(parts, cuts, cuts[1:]):
            draw_structure(ax, p, x0 + a * (x1 - x0), y0, x0 + b * (x1 - x0), y1, color)
    elif op in "⿱⿳":
        for p, a, b in zip(parts, cuts, cuts[1:]):
            draw_structure(ax, p, x0, y1 - b * (y1 - y0), x1, y1 - a * (y1 - y0), color)
    elif op == "⿻":
        for p in parts:
            draw_structure(ax, p, x0, y0, x1, y1, color)
    else:
        outer, inner = parts
        draw_structure(ax, outer, x0, y0, x1, y1, color)
        a, b, c, d = INNER[op]
        w, h = x1 - x0, y1 - y0
        draw_structure(ax, inner, x0 + a * w, y0 + b * h, x0 + c * w, y0 + d * h, color)


def drawable(tokens) -> bool:
    from fontTools.ttLib import TTCollection
    global _CMAP
    if "_CMAP" not in globals():
        _CMAP = TTCollection(FONT_PATH).fonts[0].getBestCmap()
    return all(t in "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻" or ord(t) in _CMAP for t in tokens)


LABELS = {"relational": "TRELLIS v2 from IDS structures, operators as relations",
          "supervised": "TRELLIS v2 from IDS structures, operators as tokens",
          "unsupervised": "TRELLIS v2 from sequences alone"}
TITLES = {"relational": "From the IDS structures, operators as relations",
          "supervised": "From the IDS structures, operators as tokens",
          "unsupervised": "From the sequences alone (unsupervised)"}


def figure(results, out_dir: str) -> str:
    """Novel characters generated by each model, drawn from their structure."""
    rows = sorted([r for r in results if r["seed"] == results[0]["seed"]],
                  key=lambda r: list(LABELS).index(r["mode"]))
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.15 * len(rows) + 0.5), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        ax.set_axis_off()
        novel = [t.split() for t in r["examples"]["novel"] if drawable(t.split())][:16]
        for k, tokens in enumerate(novel):
            x = k * 1.25
            ax.add_patch(plt.Rectangle((x - 0.04, -0.04), 1.08, 1.08, facecolor="white",
                                       edgecolor="#e1e0d9", linewidth=0.8))
            draw_structure(ax, parse_prefix(tokens), x, 0, x + 1, 1)
            parts = [t for t in tokens if t not in OPERATORS]      # the font has no IDS operators
            label = "".join(parts) if len(parts) <= 6 else "".join(parts[:5]) + "…"
            ax.text(x + 0.5, -0.2, label, ha="center", va="top", fontsize=7.5,
                    color=INK2, fontproperties=_font())
        ax.set_xlim(-0.2, 16 * 1.25)
        ax.set_ylim(-0.55, 1.2)
        ax.set_aspect("equal")
        gen = r["generation"]
        ax.set_title(f"{TITLES[r['mode']]}"
                     f": {100 * gen['well formed']:.0f}% well formed, {100 * gen['positions attested']:.0f}% with every "
                     f"component in an attested position, {100 * gen['rediscovered']:.1f}% held-out real characters",
                     loc="left", fontsize=10, color=INK)
    fig.suptitle("Characters TRELLIS v2 invents: novel combinations of learned components in learned positions",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, hspace=0.25)
    path = os.path.join(out_dir, "generated_characters.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def summarise(results) -> str:
    lines = ["| Model | Held-out bits/character | Bracket omission | Bracket commission | Well formed | "
             "Components in attested positions | Real: rediscovered held-out | Novel | Symbols | Chunk types |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    r0 = results[0]
    lines.append(f"| unigram tokens | {r0['baselines']['unigram']:.1f} | – | – | – | – | – | – | – | – |")
    lines.append(f"| bigram tokens | {r0['baselines']['bigram']:.1f} | – | – | – | – | – | – | – | – |")
    for mode in LABELS:
        rs = [r for r in results if r["mode"] == mode]
        if not rs:
            continue
        m = lambda f: float(np.mean([f(r) for r in rs]))
        brackets = ("– | –" if rs[0]["omission"] is None else
                    f"{100 * m(lambda r: r['omission']):.1f}% | {100 * m(lambda r: r['commission']):.1f}%")
        lines.append(
            f"| {LABELS[mode]} | {m(lambda r: r['test_bits_per_character']):.1f} | {brackets} | "
            f"{100 * m(lambda r: r['generation']['well formed']):.1f}% | "
            f"{100 * m(lambda r: r['generation']['positions attested']):.1f}% | "
            f"{100 * m(lambda r: r['generation']['rediscovered']):.1f}% | "
            f"{100 * m(lambda r: r['generation']['novel']):.1f}% | {m(lambda r: r['symbols']):.0f} | "
            f"{m(lambda r: r['chunk_types']):.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--path", default=default_ids_path())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "characters"))
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from an existing results.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.figure_only:
        with open(os.path.join(args.out, "results.json")) as f:
            print(figure(json.load(f), args.out))
        return
    jobs = [(mode, int(s)) for s in args.seeds.split(",") for mode in ("unsupervised", "supervised", "relational")]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = [f.result() for f in [pool.submit(run, m, s, args.train, args.test, args.n_gen, args.path)
                                        for m, s in jobs]]
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    table = summarise(results)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"Chinese characters (IDS), {args.train} training / {args.test} held-out characters, "
                f"seeds {args.seeds}.\n\n{table}\n")
    print(table)
    print(figure(results, args.out))


if __name__ == "__main__":
    main()
