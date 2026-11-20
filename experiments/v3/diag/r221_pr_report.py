"""Round 221b: tables from r221_pr_v1_s*.json (parse then regenerate)."""
import glob, json
D = "experiments/v3/results/chess_regimes/"
R = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(D + "r221_pr_v1_s*.json"))]
Ns = [k for k in R[0] if k[1:].isdigit() and all(k in r for r in R)]
cells = [k for k in R[0][Ns[0]] if isinstance(R[0][Ns[0]][k], dict)]
m = lambda N, c, k: sum(r[N][c][k] for r in R) / len(R)
out = ["seeds " + ",".join(str(r["seed"]) for r in R) + "; TAU %.3f; boards per cell %d per seed (300 held-out positions x 2 draws)" % (R[0]["tau"], R[0]["n"])]
for N in [n for n in ("N1000", "N4000") if n in Ns]:
    out.append(f"\n### {N} (parsed chunks per position: {R[0][N]['parsed chunks per position']})\n\n| cell | R-real | R-legal | R-non | chunks kept |\n|---|---|---|---|---|")
    for c in cells:
        out.append(f"| {c} | {m(N,c,'real'):.3f} | {m(N,c,'legal'):.3f} | {m(N,c,'non'):.3f} | {m(N,c,'chunks kept (no draw fit)'):.3f} |")
last = Ns[-1]
gen = [c for c in cells if not c.startswith("identity")]
best = {"real": max(gen, key=lambda c: m(last, c, "real")), "legal": max(gen, key=lambda c: m(last, c, "legal")), "non": max(gen, key=lambda c: m(last, c, "non"))}
out.append(f"\n### learning curves (R-real / R-legal / R-non), cells best at {last} per regime (chosen after the grid)\n\n| N | " + " | ".join(f"{k}: {v}" for k, v in best.items()) + " |\n|---|---|---|---|")
for N in Ns:
    out.append(f"| {N[1:]} | " + " | ".join(f"{m(N,c,'real'):.3f} / {m(N,c,'legal'):.3f} / {m(N,c,'non'):.3f}" for c in best.values()) + " |")
out.append("\n### go rule at any generated cell and N: " + "; ".join("%s max %.3f (%s %s)" % ((k,) + max((m(N, c, k), N, c) for N in Ns for c in gen)) for k in ("real", "legal", "non")))
out.append(f"\n### examples ({last}, seed {R[0]['seed']})\n")
for c in ("keep kc=0", "keep kc=2", "keep kc=99", "resample kc=2", "resample kc=99"):
    for reg in ("real", "legal", "non"):
        for f in R[0][last][c]["examples"][reg][:2]:
            out.append(f"- {c} [{reg}] {f}")
out.append(f"\n### illegal causes ({last}, seed {R[0]['seed']})\n")
for c in cells:
    out.append(f"- {c}: {R[0][last][c]['illegal_causes']}")
open(D + "r221_pr_tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
