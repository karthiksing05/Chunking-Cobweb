"""Round 220: tables from r220_gen_v1_s*.json (two seeds): per-arm regime shares by N, mean of seeds; showcase boards."""
import glob, json, os, sys
D = "experiments/v3/results/chess_regimes/"
R = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(D + "r220_gen_v1_s*.json"))]
out = []
Ns = [k for k in R[0] if k[1:].isdigit() and all(k in r for r in R)]
arms = ["kc=0", "abs0", "kc=1", "kc=2", "kc=3", "kc=4", "kc=6", "kc=99", "list"]
out.append("seeds " + ",".join(str(r["seed"]) for r in R) + "; TAU %.3f; samples per cell %d each" % (R[0]["tau"], R[0]["n"]))
for N in Ns:
    out.append(f"\n### {N}\n\n| arm | R-real | R-legal | R-non | drop/board | pieces | seed spread (real/legal/non) |\n|---|---|---|---|---|---|---|")
    for a in arms:
        rows = [r[N][a] for r in R]
        m = lambda k: sum(x[k] for x in rows) / len(rows)
        sp = " / ".join("%.3f" % (max(x[k] for x in rows) - min(x[k] for x in rows)) for k in ("real", "legal", "non"))
        out.append(f"| {a} | {m('real'):.3f} | {m('legal'):.3f} | {m('non'):.3f} | {m('drop_per_board'):.2f} | {m('mean_pieces'):.1f} | {sp} |")
N = Ns[-1]
DG = json.load(open(D + "r220_gen_diag_s19.json", encoding="utf-8")); out.append("\n### illegal causes (diag run, seed 19, N=500, 300 samples; share of boards with the flag)\n")
for a in arms:
    out.append(f"- {a}: {DG[chr(78)+chr(53)+chr(48)+chr(48)][a][chr(105)+chr(108)+chr(108)+chr(101)+chr(103)+chr(97)+chr(108)+chr(95)+chr(99)+chr(97)+chr(117)+chr(115)+chr(101)+chr(115)]}")
out.append(f"\n### showcase boards ({N}, seed {R[0]['seed']}): FEN board field\n")
for a in ("kc=0", "kc=2", "kc=99", "list"):
    for reg in ("real", "legal", "non"):
        for f in R[0][N][a]["examples"][reg][:3]:
            out.append(f"- {a} [{reg}] {f}")
open(D + "r220_tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
