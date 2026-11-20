"""Learning-curve figures for the talk (matplotlib, light background, site palette).
Parse success vs training size: experiments/v3/results/story_parse_eval/r216_figure.tsv
Generation fair count vs training size: experiments/v3/results/story_gen/r212_seeds.md (8-seed means).
Usage: python3 docs/talk/build/fig_talk_curves.py   (writes PNGs next to this script in img/)"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)
BLUE, GREEN, ORANGE, GREY, INK = "#2a78d6", "#1aa57a", "#d9601a", "#8a8f98", "#22262e"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#6b7280", "axes.labelcolor": INK,
                     "xtick.color": "#4b5563", "ytick.color": "#4b5563", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})

rows = list(csv.DictReader(open(os.path.join(ROOT, "experiments/v3/results/story_parse_eval/r216_figure.tsv")), delimiter="\t"))
def series(score, cls):
    pts = sorted((int(r["n"]), float(r["p_succ_set_mean"]), float(r["p_succ_gram_mean"]))
                 for r in rows if r["score"] == score and r["set"] == cls)
    return pts
LINES = [("TRELLIS parse", "logZ", BLUE, "-", "o"), ("TRELLIS generator", "gen_lp_gate", BLUE, "--", "s"),
         ("LSTM", "LSTM", GREEN, "-", "^"), ("GPT", "GPT", ORANGE, "-", "D"), ("5-gram", "5-gram", GREY, ":", "v")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, (cls, title) in zip(axs, [("C3scr", "Scrambled sentences"), ("C2full", "Content-swapped sentences")]):
    for lab, sc, col, ls, mk in LINES:
        p = series(sc, cls)
        ax.plot([a for a, _, _ in p], [b for _, b, _ in p], color=col, ls=ls, marker=mk, lw=2.2, ms=6, label=lab)
    real = series("logZ", cls)
    ax.axhline(sum(c for _, _, c in real) / len(real), color="#9ca3af", lw=1, ls=(0, (1, 2)))
    ax.text(125, 0.96, "real sentences accepted", fontsize=10.5, color="#6b7280")
    ax.set_xscale("log"); ax.set_xticks([125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(["125", "250", "500", "1k", "2k", "4k"]); ax.minorticks_off()
    ax.set_title(title, loc="left", fontweight="bold", fontsize=14, color=INK)
    ax.set_xlabel("Training sentences"); ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_ylim(0, 1.03)
axs[0].set_ylabel("Share of broken sentences accepted\n(lower is better)")
axs[1].legend(frameon=False, fontsize=11, loc="lower left", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "parse_success_curve.png"), dpi=200); plt.close(fig)

# generation fair count, mean of 8 seeds (r212_seeds.md)
N = [250, 500, 1000]
G = {"TRELLIS": ([.2419, .2982, .3281], BLUE, "o"), "LSTM": ([.2323, .2990, .3346], GREEN, "^"),
     "GPT": ([.2068, .2738, .3282], ORANGE, "D")}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for k, (v, c, m) in G.items():
    ax.plot(N, v, color=c, marker=m, lw=2.4, ms=7, label=k)
    pass
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["250", "500", "1,000"]); ax.minorticks_off()
ax.set_xlim(220, 1500); ax.set_ylim(.18, .36); ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_xlabel("Training sentences"); ax.set_ylabel("New, grammatical, novel sentences\nper attempted draw (8 seeds)"); ax.legend(frameon=False, loc="lower right", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "generation_fair_count.png"), dpi=200); plt.close(fig)
