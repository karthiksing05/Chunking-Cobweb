"""Round 221: tables from r221_gen_v1_s*.json and the whole-board property that separates R-legal from R-real boards.
Property analysis: kept legal and real boards of every cell (both seeds, all N) plus the real held-out and the shuffled-legal controls
(r220 judge): per board the logit contribution of each feature group of the unchanged judge (piece-square, kind counts, pawns per file,
material, piece count) and plain board statistics."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
import r220_judge as J
D = "experiments/v3/results/chess_regimes/"
R = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(D + "r221_gen_v1_s*.json"))]
Ns = [k for k in R[0] if k[1:].isdigit() and all(k in r for r in R)]
out = ["seeds " + ",".join(str(r["seed"]) for r in R) + "; TAU %.3f; boards per cell %d per seed" % (R[0]["tau"], R[0]["n"])]
cells = [k for k in R[0][Ns[0]] if k.startswith("counts")]


def mean(N, c, k):
    return sum(r[N][c][k] for r in R) / len(R)


for N in [n for n in ("N1000", "N4000") if n in Ns]:
    out.append(f"\n### {N}\n\n| cell | R-real | R-legal | R-non | drop/board | pieces |\n|---|---|---|---|---|---|")
    for c in cells:
        out.append(f"| {c} | {mean(N,c,'real'):.3f} | {mean(N,c,'legal'):.3f} | {mean(N,c,'non'):.3f} | {mean(N,c,'drop_per_board'):.1f} | {mean(N,c,'mean_pieces'):.1f} |")
last = Ns[-1]
best = {"real": max(cells, key=lambda c: mean(last, c, "real")), "legal": max(cells, key=lambda c: mean(last, c, "legal")),
        "non": max(cells, key=lambda c: mean(last, c, "non"))}
out.append(f"\n### learning curves (R-real / R-legal / R-non) for the cells best at {last} on each regime (chosen after the grid)\n")
out.append("| N | " + " | ".join(f"{k}: {v}" for k, v in best.items()) + " |\n|---|---|---|---|")
for N in Ns:
    out.append(f"| {N[1:]} | " + " | ".join(f"{mean(N,c,'real'):.3f} / {mean(N,c,'legal'):.3f} / {mean(N,c,'non'):.3f}" for c in best.values()) + " |")
out.append("\n### go rule at any cell and N: R-real >= .6: %s; R-legal >= .5: %s; R-non >= .6: %s" % tuple(
    "max %.3f (%s %s)" % max((mean(N, c, k), N, c) for N in Ns for c in cells) for k in ("real", "legal", "non")))
out.append("\n### illegal causes, N=%s, seed %d (share of boards)\n" % (last, R[0]["seed"]))
for c in cells:
    out.append(f"- {c}: {R[0][last][c]['illegal_causes']}")
# property analysis
Jd = J.Judge()
GR = {"piece-square": slice(0, 768), "kind counts": slice(768, 780), "pawns per file": slice(780, 796), "material": slice(796, 797), "n pieces": slice(797, 798)}


def props(pos):
    x = (J.feats(pos) - Jd.mu) / Jd.sd
    c = {g: float((x[s] * Jd.w[s]).sum()) for g, s in GR.items()}
    wp = sum(t == "wP" for t in pos.values()); bp = sum(t == "bP" for t in pos.values())
    mat = sum(J.VAL[t[1]] * (1 if t[0] == "w" else -1) for t in pos.values())
    kws = [s for s, t in pos.items() if t == "wK"]; kbs = [s for s, t in pos.items() if t == "bK"]
    return {**c, "logit": sum(c.values()) + Jd.b, "pieces": len(pos), "wP": wp, "bP": bp, "|material diff|": abs(mat),
            "queens": sum(t[1] == "Q" for t in pos.values()), "wK rank": kws[0][1] + 1 if kws else -1, "bK rank": kbs[0][1] + 1 if kbs else -1}


def collect(kind):
    fens = []
    for r in R:
        for N in Ns:
            for c in cells:
                fens += r[N][c]["kept"][kind]
    return [C for C in (__import__("trellis2.chess", fromlist=["x"]).parse_fen(f) for f in fens if f)]
sys.path.insert(0, os.path.join(J.ROOT, "src"))
from trellis2 import chess as CH
import random
def boards(kind):
    return [CH.parse_fen(f) for r in R for N in Ns for c in cells for f in r[N][c]["kept"][kind] if f]
groups = {"generated R-real": boards("real"), "generated R-legal": boards("legal"),
          "control real held-out": [J.real(i) for i in range(25459, 25959)]}
rr = random.Random(5)
groups["control shuffled legal"] = [J.shuffled(p, rr) for p in groups["control real held-out"]]
groups["control shuffled legal"] = [p for p in groups["control shuffled legal"] if J.legal(p)]
rows = {g: [props(p) for p in ps] for g, ps in groups.items()}
keys = list(next(iter(rows.values()))[0])
out.append("\n### whole-board property analysis (mean per group; judge-logit contributions by feature group)\n")
out.append("| property | " + " | ".join(f"{g} (n={len(rows[g])})" for g in rows) + " |\n|---|" + "---|" * len(rows))
for k in keys:
    out.append(f"| {k} | " + " | ".join(f"{np.mean([r[k] for r in rows[g]]):.2f}" for g in rows) + " |")
open(D + "r221_tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
