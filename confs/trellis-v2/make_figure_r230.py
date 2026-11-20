"""Figure for the r230 controlled-grammar study: three regimes on one dial (C6) and learning curves.
Reads experiments/v3/results/story_gen/r230/judge_p1*.json. Run on a compute node with matplotlib."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/coc/flash10/ksingara3/Chunking-Cobweb"
R = ROOT + "/experiments/v3/results/story_gen/r230/"
OUT = ROOT + "/confs/trellis-v2/paper/figures/controlled_regimes.png"
SEEDS = (13, 17)
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"


def load():
    """Return {(dirname, arm_index): (sem, syn, non)} over all judge files."""
    out = {}
    for f in ("judge_p1", "judge_p1b", "judge_p1c"):
        for k, v in json.load(open(R + f + ".json")).items():
            d = k.split("p1/")[1].split("/")[0]
            arm = int(k.rsplit("arm", 1)[1].split(".")[0])
            a = v["all"]
            out[(d, arm)] = (a["sem"], a["syn"], a["non"])
    return out


def c6(J, s, n, b):
    """C6 shares at n sentences, seed s, level b ('leaf' or an int), reach 5."""
    if n == 1000:
        if b == "leaf":
            return J[(f"func5_s{s}_n1000_c6", 4)]
        if b == 7:
            return J[(f"func5_s{s}_n1000_c6", 5)]
        return J[(f"func5_s{s}_n1000_c6_b{b}", 0)]
    return J[(f"func5_s{s}_n{n}_c6_" + ("leaf" if b == "leaf" else f"b{b}"), 0)]


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#555555", "axes.labelcolor": "#222222", "xtick.color": "#333333",
                         "ytick.color": "#333333", "figure.facecolor": "white", "axes.facecolor": "white"})
    J = load()
    levels = ["leaf", 7, 6, 5, 4, 3, 2, 0]
    names = ["Meaningful (R-sem)", "Syntax-only (R-syn)", "Nonsense (R-non)"]
    cols = [BLUE, ORANGE, GREEN]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    x = list(range(len(levels)))
    for j in range(3):
        m, lo, hi = [], [], []
        for b in levels:
            v = [c6(J, s, 1000, b)[j] for s in SEEDS]
            m.append(sum(v) / 2); lo.append(min(v)); hi.append(max(v))
        ax1.fill_between(x, lo, hi, color=cols[j], alpha=0.18, lw=0)
        ax1.plot(x, m, "-o", color=cols[j], ms=3.5, lw=1.6)
    ax1.set_xticks(x); ax1.set_xticklabels([str(b) if b != "leaf" else "leaf" for b in levels])
    ax1.set_xlabel("Level $b$ of the word taxonomy (bits)"); ax1.set_ylabel("Share of 1,600 generations")
    ax1.set_ylim(0, 1.0); ax1.set_xlim(-0.3, 8.9)
    ax1.text(7.2, 0.93, "R-non", color=GREEN, fontsize=8.5, va="center")
    ax1.text(7.2, 0.09, "R-sem", color=BLUE, fontsize=8.5, va="center")
    ax1.text(2.0, 0.76, "R-syn", color=ORANGE, fontsize=8.5, va="center", ha="center")
    ax1.set_title("One dial, n = 1,000 (band: two seeds)", fontsize=9)
    ns = [250, 500, 1000, 2000, 4000]
    spec = [("leaf", 0, BLUE, "R-sem at leaf"), (6, 1, ORANGE, "R-syn at $b=6$"), (0, 2, GREEN, "R-non at $b=0$")]
    for b, j, c, lab in spec:
        v = [[c6(J, s, n, b)[j] for s in SEEDS] for n in ns]
        m = [sum(t) / 2 for t in v]
        ax2.fill_between(ns, [min(t) for t in v], [max(t) for t in v], color=c, alpha=0.18, lw=0)
        ax2.plot(ns, m, "-o", color=c, ms=3.5, lw=1.6)
        ax2.text(4300, m[-1] + (0.0 if j != 2 else -0.0), lab, color=c, fontsize=8.5, va="center")
    ax2.set_xscale("log"); ax2.set_xticks(ns); ax2.set_xticklabels(["250", "500", "1k", "2k", "4k"])
    ax2.minorticks_off()
    ax2.set_xlim(220, 4000 * 2.6); ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Training sentences"); ax2.set_ylabel("Share of the target regime")
    ax2.set_title("Learning curves", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("saved", OUT)
