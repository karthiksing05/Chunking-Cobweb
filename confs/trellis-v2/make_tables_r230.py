"""Write LaTeX tables of the r230 controlled-grammar study to stdout (supplement section). Run on a compute node."""
import json
from make_figure_r230 import load, c6, R, SEEDS

J = load()
f3 = lambda x: f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"
def trip(t): return " / ".join(f3(v) for v in t)

# controls
C = json.load(open(R + "controls.json"))
print("%% TABLE controls")
for k, v in C.items():
    if not isinstance(v, dict) or 'sem' not in v: continue
    print(f"{k} & {f3(v['sem'])} & {f3(v['syn'])} & {f3(v['non'])} \\\\")

# purity, func5 in full
P = json.load(open(R + "purity.json"))
def pk(fs, s): return [k for k in P if k.endswith(f"wordtree_{fs}_s{s}_n1000.json")][0]
SYN = "syn purity (class mates, same syntactic category)"
SEM = "sem purity (content class mates of the same kind, same semantic type)"
print("%% TABLE purity func5")
lv = ["0", "2", "3", "4", "5", "6", "7", "leaf"]
for b in lv:
    a = [P[pk("func5", s)]["levels"][b] for s in SEEDS]
    cl = "-".join(str(x) for x in sorted({q["classes"] for q in a})) if a[0]["classes"] != a[1]["classes"] else str(a[0]["classes"])
    print(f"{'leaf' if b=='leaf' else b} & {cl} & {f3(a[0][SYN])} / {f3(a[1][SYN])} & {f3(a[0][SEM])} / {f3(a[1][SEM])} \\\\")
print("%% chance", P[pk("func5", 13)]["chance syn purity (all pairs)"], P[pk("func5", 13)]["chance sem purity (same kind pairs)"])
print("%% TABLE purity feature sets")
for fs in ["all", "func5", "func8", "func15", "func30"]:
    cells = []
    for b in ["5", "6", "7"]:
        a = [P[pk(fs, s)]["levels"][b] for s in SEEDS]
        cells.append(f"{(a[0][SYN]+a[1][SYN])/2:.2f} / {(a[0][SEM]+a[1][SEM])/2:.2f}")
    print(f"{fs} & " + " & ".join(cells) + " \\\\")

# dial C6
print("%% TABLE dial C6")
for lab, b in [("leaf, $W'=5$", "leaf"), ("$b=7$", 7), ("$b=6$", 6), ("$b=5$", 5), ("$b=4$", 4), ("$b=3$", 3), ("$b=2$", 2), ("$b=0$", 0)]:
    t = [c6(J, s, 1000, b) for s in SEEDS]
    print(f"{lab} & " + " & ".join(f"{f3(t[0][j])} / {f3(t[1][j])}" for j in range(3)) + " \\\\")
print("%% reach at leaf")
for w, arm in [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4)]:
    t = [J[(f"func5_s{s}_n1000_c6", arm)] for s in SEEDS]
    print(f"$W'={w}$ & " + " & ".join(f"{f3(t[0][j])} / {f3(t[1][j])}" for j in range(3)) + " \\\\")
print("%% TABLE dial C9")
for lab, b in [("leaf", "leaf"), ("7", "7"), ("6", "6"), ("5", "5"), ("4", "4"), ("3", "3"), ("0", "0")]:
    t = [[J[(f"func5_s{s}_n1000_c9b{b}", a)] for a in range(4)] for s in SEEDS]
    maxsyn = [max(x[1] for x in t[i]) for i in range(2)]
    print(f"{lab} & " + " & ".join(f"{f3(t[0][3][j])} / {f3(t[1][3][j])}" for j in range(3)) + f" & {f3(maxsyn[0])} / {f3(maxsyn[1])} \\\\")

# learning curves
print("%% TABLE curves")
for n in [250, 500, 1000, 2000, 4000]:
    for s in SEEDS:
        print(f"{n:,} & {s} & " + " & ".join(trip(c6(J, s, n, b)) .replace(" / ", "/") for b in ["leaf", 6, 0]) + " \\\\")

# examples: first three per regime, in file order
print("%% TABLE examples")
E = [json.loads(l) for l in open(R + "examples.jsonl")]
reg = [("sem", 0), ("syn", 1), ("non", 2)]
for si, s in enumerate(SEEDS):
    for r, ai in reg:
        ex = E[3 * si + ai]["examples"][r][:3]
        for i, (sent, why, _) in enumerate(ex):
            print(f"{s if i==0 and r=='sem' else ''} & {r if i==0 else ''} & {sent} & {why.replace('+', ' + ')} \\\\")
        
