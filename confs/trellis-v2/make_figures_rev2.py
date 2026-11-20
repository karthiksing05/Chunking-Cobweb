"""Figures for rev 2 of the TRELLIS v2 paper: bits curve (r225), abstraction law (r214 Add. 7, r227), chess boards (r227).
Run on a compute node with the trellis env (matplotlib, python-chess)."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chess

ROOT = "/coc/flash10/ksingara3/Chunking-Cobweb"
OUT = ROOT + "/confs/trellis-v2/paper/figures/"
TSV = ROOT + "/experiments/v3/results/story_gen/r225_bits_figure.tsv"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#222222", "xtick.color": "#333333",
                     "ytick.color": "#333333", "figure.facecolor": "white", "axes.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#777777", "#CC79A7"

# ---------- Figure 1: bits per sentence against n ----------
rows = list(csv.DictReader(open(TSV), delimiter="\t"))
d = {}
for r in rows:
    d.setdefault(r["model"], {})[int(r["n"])] = (float(r["bits_per_sentence_mean"]), float(r["bits_per_sentence_sd_seeds"]))
ns = [250, 500, 1000, 2000, 4000]
spec = [("unigram", "word unigram", GREY, "-", 1.4),
        ("bpe", "BPE + unigram code", PURPLE, "-", 1.6),
        ("trellis", "TRELLIS", BLUE, "-", 2.6),
        ("trellis_noconcept", "TRELLIS, concepts off", BLUE, "--", 1.4),
        ("trellis_neither", "TRELLIS, both off", BLUE, ":", 1.6),
        ("kn3", "Kneser-Ney trigram", ORANGE, "-", 1.6),
        ("gpt", "GPT", GREEN, "-", 1.6),
        ("lstm", "LSTM", RED, "-", 1.6)]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ends = []
for key, lab, col, ls, lw in spec:
    y = [d[key][n][0] for n in ns]
    ax.plot(ns, y, ls, color=col, lw=lw, marker="o", ms=4, mfc="white" if ls == "--" else col, mec=col)
    ends.append([y[-1], lab, col])
ends.sort()
# spread end labels by at least 1.15 bits
pos = []
for y, lab, col in ends:
    p = y if not pos or y - pos[-1][0] >= 1.5 else pos[-1][0] + 1.5
    pos.append((p, lab, col, y))
for p, lab, col, y in pos:
    ax.annotate(lab, xy=(4000, y), xytext=(4350, p), color=col, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=2))
ax.set_xscale("log", base=2)
ax.set_xticks(ns); ax.set_xticklabels(["250", "500", "1,000", "2,000", "4,000"])
ax.minorticks_off()
ax.set_xlim(230, 4000 * 1.02)
ax.set_xlabel("training sentences (TinyStories)")
ax.set_ylabel("held-out code length (bits per sentence)")
ax.set_ylim(18, 44)
ax.grid(axis="y", color="#e3e3e3", lw=0.6)
fig.subplots_adjust(right=0.70, left=0.12, bottom=0.15, top=0.97)
fig.savefig(OUT + "language_bits_curve.png", dpi=220)
plt.close(fig)

# ---------- Figure 2: the abstraction law ----------
levels = ["leaf", "b=7", "b=6", "b=4", "root"]
runs = {  # R-sem, R-syn per level, from Table dial (r214 Addendum 7); leaf = generation temperature 1 (same as class levels)
    "1,000 s13": [(.558, .206), (.331, .255), (.197, .208), (.107, .140), (.034, .095)],
    "4,000 s13": [(.580, .193), (.293, .211), (.180, .161), (.087, .116), (.029, .082)],
    "4,000 s17": [(.541, .218), (.323, .265), (.201, .225), (.116, .157), (.033, .083)],
}
marks = {"1,000 s13": "o", "4,000 s13": "s", "4,000 s17": "^"}
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"wspace": 0.42})
a = axs[0]
x = range(len(levels))
for name, v in runs.items():
    syn = [s + y for s, y in v]; mean = [s for s, y in v]
    a.plot(x, [t / syn[0] for t in syn], "-", color=BLUE, marker=marks[name], ms=4, lw=1.3)
    a.plot(x, [t / mean[0] for t in mean], "--", color=ORANGE, marker=marks[name], ms=4, lw=1.3, mfc="white")
a.set_xticks(list(x)); a.set_xticklabels(levels, fontsize=8)
a.set_ylim(0, 1.05); a.set_ylabel("fraction of leaf rate retained")
a.set_title("Language: pass rates", fontsize=9)
a.text(1.55, 0.78, "syntax", color=BLUE, fontsize=8.5); a.text(0.5, 0.10, "meaning", color=ORANGE, fontsize=8.5)
a.grid(axis="y", color="#e3e3e3", lw=0.6)
b = axs[1]
for name, v in runs.items():
    b.plot(x, [y / (s + y) for s, y in v], "-", color=RED, marker=marks[name], ms=4, lw=1.3, label=name)
b.set_xticks(list(x)); b.set_xticklabels(levels, fontsize=8)
b.set_ylim(0, 0.85); b.set_ylabel("meaningless share of grammatical")
b.set_title("Language: share", fontsize=9)
b.legend(frameon=False, fontsize=7, loc="upper left", title="sentences, seed", title_fontsize=7)
b.grid(axis="y", color="#e3e3e3", lw=0.6)
c = axs[2]
kc = ["0", "1", "2", "3", "4", "6", "root"]
real = [.890, .790, .738, .718, .696, .696, .681]
legal = [.890 + .056, .790 + .139, .738 + .184, .718 + .206, .696 + .215, .696 + .216, .681 + .233]
c.plot(range(7), [r / .945 for r in real], "--", color=ORANGE, marker="o", ms=4, lw=1.3, mfc="white")
c.plot(range(7), legal, "-", color=BLUE, marker="o", ms=4, lw=1.3)
# where-rank dial (r227): kc 0, 2, 6, root at x = 0, 2, 5, 6
rx = [0, 2, 5, 6]
rreal = [.829, .480, .219, .193]; rleg = [.829 + .053, .480 + .091, .219 + .121, .193 + .137]
c.plot(rx, [r / .945 for r in rreal], "--", color=ORANGE, marker="s", ms=4, lw=1.3, mfc="white")
c.plot(rx, rleg, "-", color=BLUE, marker="s", ms=4, lw=1.3)
c.set_xticks(range(7)); c.set_xticklabels(kc, fontsize=8)
c.set_ylim(0, 1.05); c.set_xlabel("anchor abstraction level", fontsize=8)
c.set_ylabel("fraction of real board retained")
c.set_title("Chess: file (o), rank (s)", fontsize=9)
c.text(3.2, 0.58, "realism", color=ORANGE, fontsize=8.5); c.text(0.3, 0.99, "legality", color=BLUE, fontsize=8.5)
c.grid(axis="y", color="#e3e3e3", lw=0.6)
for ax_ in axs[:2]:
    ax_.set_xlabel("abstraction level", fontsize=8)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.88)
fig.savefig(OUT + "abstraction_law.png", dpi=220)
plt.close(fig)

# ---------- Figure 3: chess boards by regime ----------
boards = [
 ("Real", ["1r4k1/2pR1pp1/r1p1n1p1/p1B2P2/2P3P1/7P/1PP5/2K4R", "1r1q1rk1/Qpp1n1pp/2pb4/p7/2N1P3/2P5/PP3PPP/R1B2RK1", "1r4k1/3R1ppp/4nrpp/2B2pP1/P3P3/5P2/2PP4/3K1R2"]),
 ("Legal, unrealistic", ["1r1k4/pppnp3/1ppPn3/6q1/4PP2/2NPQ1P1/5R1P/4R1K1", "r1k2r2/ppp1pp2/3q1p2/6pp/8/1P1bn1PP/2PP1PB1/3R1KQR", "r3k1nr/ppp1p1bp/2p1b3/3p2p1/2PPPn2/1P3N2/1P1N1PPP/R1BK1B1R"]),
 ("Nonsense", ["3p1k2/3q1p2/5PpP/2pPPp2/3pk3/r1pp4/pbn5/8", "p4pk1/4pBp1/2B1P3/P3r3/1r1b4/2b1n3/1p2q3/2p5", "2PP1q2/2ppP3/4p2n/Rp1p2p1/p1prk2p/3Pp3/4Pp2/2PK4"]),
]
glyph = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
         "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
fig, axs = plt.subplots(3, 3, figsize=(5.6, 5.9))
for i, (name, fens) in enumerate(boards):
    for j, fen in enumerate(fens):
        ax = axs[i][j]
        bd = chess.Board(fen + " w - - 0 1")
        for f in range(8):
            for r in range(8):
                ax.add_patch(plt.Rectangle((f, r), 1, r and 1 or 1, color="#f0d9b5" if (f + r) % 2 == 1 else "#b58863"))
        for sq, pc in bd.piece_map().items():
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ch = glyph[pc.symbol()]
            ax.text(f + .5, r + .5, ch, ha="center", va="center", fontsize=15, family="DejaVu Sans",
                    color="#111111" if pc.color == chess.BLACK else "#ffffff",
                    path_effects=[] )
            if pc.color == chess.WHITE:  # outline so white pieces read on light squares
                import matplotlib.patheffects as pe
                ax.texts[-1].set_path_effects([pe.withStroke(linewidth=1.6, foreground="#111111")])
        ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
        if j == 0:
            ax.text(-0.15, 4, name, rotation=90, ha="right", va="center", fontsize=9, color="#222222")
fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
fig.savefig(OUT + "chess_regime_boards.png", dpi=200)
plt.close(fig)
print("done")

# ---------- Figure 4: fair count under judge2 (r224) ----------
gn = [250, 500, 1000, 4000]
fc = {"syntax": {"TRELLIS": [.3030, .3718, .4096, .4163], "LSTM": [.2877, .3731, .4200, .4729], "GPT": [.2828, .3720, .4284, .4716]},
      "syntax and meaning": {"TRELLIS": [.1890, .2304, .2584, .2220], "LSTM": [.1828, .2435, .2840, .2916], "GPT": [.1671, .2265, .2711, .3042]}}
cols = {"TRELLIS": BLUE, "LSTM": RED, "GPT": GREEN}
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"wspace": 0.28})
for ax, (title, series) in zip(axs, fc.items()):
    for name, y in series.items():
        ax.plot(gn[:3], y[:3], "-", color=cols[name], lw=2.2 if name == "TRELLIS" else 1.5, marker="o", ms=4, label=name)
        ax.plot(gn[2:], y[2:], ":", color=cols[name], lw=1.5)
        ax.plot(gn[3:], y[3:], "o", color=cols[name], ms=4.5, mfc="white")
    ax.set_xscale("log", base=2); ax.set_xticks(gn); ax.set_xticklabels(["250", "500", "1,000", "4,000"]); ax.minorticks_off()
    ax.set_xlabel("training sentences"); ax.set_title("Judge: " + title, fontsize=9)
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
axs[0].set_ylabel("fair count (per attempted draw)"); axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.90)
fig.savefig(OUT + "generation_fair_count_judge2.png", dpi=220)
plt.close(fig)
print("done4")
