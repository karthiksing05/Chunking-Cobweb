"""Round 226: tables from r227_v1_s*.json."""
import glob, json
D = "experiments/v3/results/chess_regimes/"
R = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(D + "r227_v1_s*.json"))]
Ns = [k for k in R[0] if k[1:].isdigit() and all(k in r for r in R)]
cells = [k for k in R[0][Ns[0]] if isinstance(R[0][Ns[0]][k], dict)]
m = lambda N, c, k: sum(r[N][c][k] for r in R) / len(R)
out = ["seeds " + ",".join(str(r["seed"]) for r in R) + "; TAU %.3f; boards per cell %d per seed" % (R[0]["tau"], R[0]["n"])]
for N in [n for n in ("N1000", "N4000") if n in Ns]:
    out.append(f"\n### {N}\n\n| cell | R-real | R-legal | R-non | only-check illegal | units kept |\n|---|---|---|---|---|---|")
    for c in cells:
        out.append(f"| {c} | {m(N,c,'real'):.3f} | {m(N,c,'legal'):.3f} | {m(N,c,'non'):.3f} | {m(N,c,"illegal only by check (share of all boards)"):.3f} | {m(N,c,"units kept (no draw fit)"):.3f} |")
last = Ns[-1]
gen = [c for c in cells if c.startswith("where")]
best = {"real": max(gen, key=lambda c: m(last, c, "real")), "legal": max(gen, key=lambda c: m(last, c, "legal")), "non": max(gen, key=lambda c: m(last, c, "non"))}
out.append(f"\n### learning curves (R-real / R-legal / R-non), cells best at {last} per regime (chosen after the grid)\n\n| N | " + " | ".join(f"{k}: {v}" for k, v in best.items()) + " |\n|---|---|---|---|")
for N in Ns:
    out.append(f"| {N[1:]} | " + " | ".join(f"{m(N,c,'real'):.3f} / {m(N,c,'legal'):.3f} / {m(N,c,'non'):.3f}" for c in best.values()) + " |")
out.append("\n### go rule at any generated cell and N: " + "; ".join("%s max %.3f (%s %s)" % ((k,) + max((m(N, c, k), N, c) for N in Ns for c in gen)) for k in ("real", "legal", "non")))
out.append(f"\n### illegal causes, share of ALL boards ({last}, seed {R[0]['seed']})\n")
for c in cells:
    out.append(f"- {c}: {R[0][last][c]['illegal_causes_of_all_boards']}")
out.append(f"\n### showcase boards ({last}, seed {R[0]['seed']}): board field of the FEN, ranks 8 to 1\n")
for c in list(best.values()) + ["where-rank kc=99", "where-file kc=99"]:
    for reg in ("real", "legal", "non"):
        for f in R[0][last][c]["examples"][reg][:3]:
            out.append(f"- {c} [{reg}] {f}")
open(D + "r227_tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
