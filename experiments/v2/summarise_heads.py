"""Write experiments/v2/results/heads/summary.md: the agreement line of
Track 1 (Phase A, heads, splits, joint splits, final-round refinement,
nights, characters, the paper's corpora), tables read from the result files.

Usage:
    python experiments/v2/summarise_heads.py
"""
from __future__ import annotations

import glob
import json
import os

INTRO = """# Track 1: grammatical number, heads and split moves (2026-10-08 to 10-10)

Testbed: `complex_grammars.py` (agree, relative, deep; number agreement, centre embedding, coordination; 2,000
training + 500 held-out sentences each). Measures per 1,000 samples: grammatical (target CFG), number-blind
grammatical (the CFG with number erased: accepted there but not by the CFG = an agreement error only), novel; held-out
bits; bracket omission and commission. Number is used only to evaluate, never by a learner. All code is new
(`experiments/v2/{number_oracle,heads,splits,run_heads_probe,run_heads_characters,run_splits_paper,run_complex_final,
joint_seed_diag,number_split_singles}.py`, `tests/trellis2/test_heads.py`); `src/trellis2` is unchanged and the
experiments substitute names in-process (listed in each result's `substituted`).

## Findings (confirmed unless marked)

1. **Why number merges (Phase A).** No cut through the representation hierarchy separates singular from plural
   (only its leaves, i.e. token identity, do). Splitting every number-mixing category (oracle) shortens the code on
   agree (500 and 2,000; night and gold trees) and on relative from gold trees at 2,000, but lengthens it on relative
   at 500 and on the night's relative analyses at 2,000 (+395 bits).
2. **Heads (both hierarchies, domain-neutral, exact; 12 tests pass) do not make number learnable.** Description
   length never chooses head-conditioned rules (the categories reaching the composition side carry no number);
   representation-side heads help relative's structure (−4.3k bits against a control with blank head attributes)
   but not agreement, and cost characters 2 points of well-formedness.
3. **Greedy barrier.** The oracle number split pays only jointly: on agree 2,000 every noun/determiner/NP category
   split alone costs +190 to +2,550 bits while all together save 1.6–2.2k. A one-symbol split move therefore never
   finds number (it finds role splits instead: good for code and structure, number purity unchanged).
4. **Joint split move** (a seed split propagated through the composition graph by 2-means on neighbours' labels,
   pruned by the exact plain code, accepted by the exact full code) finds number by itself on the night's agree
   analyses: number purity 94–98% / 92–100% / 92% (nominal / verbal / determiner), agreement-only errors 50–54% →
   10–14%, grammatical 46–50% → 86–89%, code −1.5k to −2.2k bits, held-out −1.2 bits per sentence (both seeds,
   full nights with final-round refinement). The remaining errors match the non-constituent bottleneck: only 46%
   (agree) / 32% (relative) of the night's analyses join the subject with its VP, and the oracle split leaves
   0.3% agreement errors on gold trees but 19–33% on the night's analyses (supported, not proven).
5. **Relative**: on the night's analyses even the oracle number split costs bits (+395), and propagated number
   splits cost +128 to +1,900 (head-conditioned rescoring gives identical numbers), so the night's relative
   grammar gains little (seed 17: 29.6% → 34.1% grammatical; seed 13: no change). From gold trees the coordinated
   split pays (oracle −2,190; propagation −1,672 at best; the fixed move accepts 7 joint splits: grammatical 31.7% →
   51.4%, number purity 83% / 80% / 92%). What would make it pay on the night's analyses: analyses that join subject
   and VP (the structure search), and more data (oracle gain grows with size: gold +581 at 500 → −2,190 at 2,000;
   night +767 → +395) (suspicion from the trend, two sizes).
6. **Final-round refinement is the configuration**: splitting in every consolidation round lets the rounds diverge
   (term_low +0.7% code; joint splits every round cost characters 9.5 points of well-formedness at a shorter code);
   refining only the final grammar keeps the agree gain, is never worse than baseline on the paper's corpora, and
   leaves characters at 96.7% well formed.
7. **Description length vs generation on characters**: a shorter code went with worse generation twice
   (joint splits every round: −1.7% code, well formed 97.1% → 87.6%; splits + heads: −1.7% code, 94.8%).
8. **Deep (10–30 words)** stays at forests (0% whole trees) with or without splits: category expressiveness does not
   make whole trees pay there.

Speed: a final-round refinement takes 2–20 minutes at 2,000 sentences (agree), up to 53 minutes (relative gold,
fixed selection: 6 exact refits per step); `splits.CONFIG['joint_max_seeds']` and `['candidates']` cap it.
"""

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
H = os.path.join(R, "heads")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def num(x, d=0):
    return "–" if x is None else f"{x:,.{d}f}"


def phase_a():
    L = ["## Phase A: why number merges", "",
         "Seed 13. Night = the night's grammar refitted from its stored analyses (reproduces the recorded code exactly); "
         "gold = the gold-tree grammar. A2 re-runs compile_grammar's stage B on fixed symbols (reproduces each chosen "
         "grammar's code exactly) and splits every category that mixes number (oracle, evaluation only).", "",
         "| Run | Unsplit bits | Oracle split bits | Δ | Grammatical (number-blind), unsplit → split |", "|---|---|---|---|---|"]
    for g in ("agree", "relative"):
        for n in (500, 2000):
            for side in ("night", "gold"):
                r = load(os.path.join(H, "phase_a", f"{g}_{n}_s13_{side}.json"))
                if not r:
                    continue
                a = r["A2"]
                u, s = a["generation, unsplit"], a["generation, split"]
                L.append(f"| {g} {n} {side} | {num(a['unsplit (stage B re-run)']['total bits'])} | {num(a['split']['total bits'])} | "
                         f"{a['split minus unsplit (bits)']:+,.0f} | {pct(u['grammatical'])} ({pct(u['grammatical ignoring number'])}) → "
                         f"{pct(s['grammatical'])} ({pct(s['grammatical ignoring number'])}) |")
    L += ["", "A1: no cut through the representation hierarchy separates number: the best cut with as many nodes as the "
          "chosen categories reaches 58–80% purity by number (base rate 54–68%), and 95% purity needs more than 4× the "
          "nodes; only the leaves are pure (token identity).", ""]
    for side in ("gold", "night"):
        r = load(os.path.join(H, "phase_a", f"singles_agree_2000_s13_{side}.json"))
        if r:
            ds = [x["delta bits"] for x in r["singles"]]
            L.append(f"Greedy barrier (agree 2,000, {side}): the oracle split of all categories at once {r['all split: delta']:+,.0f} bits; "
                     f"each category alone {min(ds):+,.0f} to {max(ds):+,.0f} bits ({sum(d > 0 for d in ds)} of {len(ds)} cost bits).")
    return L + [""]


def probes():
    L = ["## Heads and splits on gold trees (2,000 sentences, seed 13)", "",
         "| Grammar | Variant | Code | K | Held-out bits | Omission | Grammatical | Number-blind | Agreement-only errors |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = [("probe", "baseline"), ("probe", "composition"), ("probe", "control"), ("probe", "both"), ("probe", "oracle"),
            ("probe", "splits"), ("probe_splits2", "splits"), ("probe", "splits_heads"), ("joint", "splits_joint"),
            ("final", "final_single"), ("final", "final_joint"), ("nights/refix", "final_joint")]
    names = {"baseline": "baseline", "composition": "composition heads", "control": "control (head attributes, no sides)",
             "both": "heads, both sides (learned)", "oracle": "oracle heads (upper bound)", "splits": "splits every round",
             "splits_heads": "splits + heads", "splits_joint": "joint splits every round",
             "final_single": "final-round single", "final_joint": "final-round joint"}
    for g in ("agree", "relative"):
        for d, v in rows:
            r = load(os.path.join(H, d, f"gold_{v}_{g}_2000_s13.json"))
            if not r:
                continue
            gen = r["generation"]
            label = names[v] + (" (v2 proposals)" if d == "probe_splits2" else "") + (" (pruned, fixed selection)" if d == "nights/refix" else "")
            L.append(f"| {g} | {label} | {num(r['total_bits'])} | {r['symbols']} | {r['test_bits_per_sentence']:.2f} | "
                     f"{pct(r['omission'])} | {pct(gen['grammatical'])} | {pct(gen.get('grammatical ignoring number'))} | "
                     f"{pct(gen.get('agreement errors only'))} |")
    return L + [""]


def night_refits():
    L = ["## Splits on the night's own analyses (2,000 sentences)", "",
         "| Grammar | Seed | Variant | Code | Held-out bits | Grammatical | Agreement-only errors | Number purity (nominal / verbal / det) |",
         "|---|---|---|---|---|---|---|---|"]
    for g in ("agree", "relative"):
        for d, v, s in (("joint", "baseline", 13), ("joint", "splits", 13), ("joint", "splits_joint", 13),
                        ("final", "final_single", 13), ("final", "final_joint", 13),
                        ("nights/refix", "final_joint", 13), ("nights/refix", "final_joint", 17)):
            r = load(os.path.join(H, d, f"night_{v}_{g}_2000_s{s}.json"))
            if not r:
                continue
            p = r.get("number purity of final categories", {})
            pur = " / ".join(f"{100 * p[k]['purity by number']:.0f}%" if k in p else "–" for k in ("nominal", "verbal", "determiner"))
            label = v + (" (pruned, fixed selection)" if d == "nights/refix" else "")
            L.append(f"| {g} | {s} | {label} | {num(r['total_bits'])} | {r['test_bits_per_sentence']:.2f} | "
                     f"{pct(r['generation']['grammatical'])} | {pct(r['generation'].get('agreement errors only'))} | {pur} |")
    return L + [""]


def nights():
    L = ["## Full unsupervised nights, final-round refinement (single + joint, heads off) against the current night", "",
         "| Grammar | Seed | Code (night / +refine) | K | Held-out bits/sent (night / +refine / 2-gram / 3-gram) | Brackets om. = com. | "
         "Whole trees (train) | Grammatical (night / +refine) | Number-blind | Agreement-only errors (night / +refine) | Novel |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("agree", "relative", "deep"):
        for s in (13, 17):
            b = load(os.path.join(R, "complex", f"{g}_2000_s{s}", "results.json"))
            n = load(os.path.join(H, "nights", "joint", f"{g}_2000_s{s}", "results.json"))
            if not b:
                continue
            bu = b["unsupervised"]
            nu = (n or {}).get("unsupervised", {})
            if "test_bits_per_sentence" not in nu:
                L.append(f"| {g} | {s} | {num(bu['total_bits'])} / (running) |" + " |" * 9)
                continue
            ng = b["ngram held-out bits per sentence"]
            L.append(f"| {g} | {s} | {num(bu['total_bits'])} / {num(nu['total_bits'])} | {bu['symbols']} / {nu['symbols']} | "
                     f"{bu['test_bits_per_sentence']:.2f} / {nu['test_bits_per_sentence']:.2f} / {ng['2-gram']:.1f} / {ng['3-gram']:.1f} | "
                     f"{pct(bu['omission'])} / {pct(nu['omission'])} | {pct(bu.get('train_whole_trees'))} / {pct(nu.get('train_whole_trees'))} | "
                     f"{pct(bu['generation']['grammatical'])} / {pct(nu['generation']['grammatical'])} | "
                     f"{pct(nu['generation'].get('grammatical ignoring number'))} | "
                     f"{pct(bu['generation'].get('agreement errors only'))} / {pct(nu['generation'].get('agreement errors only'))} | "
                     f"{pct(nu['generation']['novel'])} |")
    return L + [""]


def characters():
    L = ["## Characters (2,000, seed 13)", "",
         "| Mode | Variant | Code | K | Held-out bits/char | Omission | Well formed | Attested | Real / rediscovered | Novel |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    base = load(os.path.join(R, "characters", "results.json"))
    for r in base or []:
        if r["mode"] in ("supervised", "unsupervised"):
            gen = r["generation"]
            L.append(f"| {r['mode']} | baseline (results/characters) | {num(r['total_bits'])} | {r['symbols']} | "
                     f"{r['test_bits_per_character']:.2f} | {pct(r['omission'])} | {pct(gen['well formed'])} | "
                     f"{pct(gen['positions attested'])} | {pct(gen['real'])} / {pct(gen['rediscovered'])} | {pct(gen['novel'])} |")
    for path in sorted(glob.glob(os.path.join(H, "characters", "*.json"))):
        r = load(path)
        gen = r["generation"]
        L.append(f"| {r['mode']} | {r['variant']} | {num(r['total_bits'])} | {r['symbols']} | {r['test_bits_per_character']:.2f} | "
                 f"{pct(r['omission'])} | {pct(gen['well formed'])} | {pct(gen['positions attested'])} | "
                 f"{pct(gen['real'])} / {pct(gen['rediscovered'])} | {pct(gen['novel'])} |")
    return L + [""]


def paper():
    L = ["## The paper's six corpora, gold trees (code / generation commission / held-out bits)", "",
         "| Condition | Seed | Baseline | Splits every round | Final-round single | Final-round joint |", "|---|---|---|---|---|---|"]
    for c in ("small", "med", "large", "term_low", "term_med", "term_high"):
        for s in (13, 17):
            cells = []
            for v in ("baseline", "splits", "final_single", "final_joint"):
                r = load(os.path.join(H, "paper", f"{c}_s{s}_{v}.json"))
                cells.append("–" if not r else f"{r['total_bits']:,.0f} / {pct(r['gen_commission'])} / {r['test_bits_per_sentence']:.2f}")
            L.append(f"| {c} | {s} | " + " | ".join(cells) + " |")
    return L + [""]


def main():
    L = [INTRO] + phase_a() + probes() + night_refits() + nights() + characters() + paper()
    with open(os.path.join(H, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
