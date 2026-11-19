"""Round 213 (chess): text diagrams of chunk decompositions of fixed boards, per seed, and cross-seed agreement of decompositions.

Reads results/chess_inspect/r213_chess_s<seed>.json (r213_chess.py). Usage (compute node, repo root):
  r213_chess_boards.py <board indices comma list> -> results/chess_inspect/r213_chess_boards.txt
"""
import json
import sys
from itertools import combinations

D = "experiments/v3/results/chess_inspect/"
SEEDS = [13, 17, 19, 23]
R = {s: json.load(open(f"{D}r213_chess_s{s}.json", encoding="utf-8")) for s in SEEDS}
FIN = str(R[13]["checkpoints"][-1])
idxs = [int(x) for x in sys.argv[1].split(",")]
SYM = {"K": "K", "Q": "Q", "R": "R", "B": "B", "N": "N", "P": "P"}
out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))


def sym(tok):
    c = SYM[tok[1]]
    return c if tok[0] == "w" else c.lower()


def key(u):
    return "|".join("%s:%s" % (sq, t) for sq, t in sorted(u["pieces"]))


def multi(b):
    return [u for u in b["units"] if u["size"] >= 2]


def diagram(b, units):
    lab = {}
    for i, u in enumerate(units):
        for sq, _ in u["pieces"]:
            lab[sq] = "ABCDEFGH"[i]
    rows = []
    for r in range(7, -1, -1):
        left = " ".join(sym(b["fen"]["abcdefgh"[f] + str(r + 1)]) if "abcdefgh"[f] + str(r + 1) in b["fen"] else "." for f in range(8))
        right = " ".join(lab.get("abcdefgh"[f] + str(r + 1), "o" if "abcdefgh"[f] + str(r + 1) in b["fen"] else ".") for f in range(8))
        rows.append("%d  %s     %d  %s" % (r + 1, left, r + 1, right))
    rows.append("   a b c d e f g h        a b c d e f g h")
    return rows


for bi in idxs:
    b0 = R[13]["boards"][FIN][bi]
    us = sorted(multi(b0), key=lambda u: (-u["size"], key(u)))
    P("\n### board %d (%s, %d pieces), seed 13 at %s positions: pieces (left; white upper case, black lower case) and chunk letters (right; o = piece in no multi-piece chunk)" % (bi, b0["kind"], len(b0["fen"]), FIN))
    for row in diagram(b0, us):
        P("   " + row)
    for i, u in enumerate(us):
        P("   %s = %s (%d pieces, stored chunk: %s)" % ("ABCDEFGH"[i], ", ".join("%s%s" % (sym(t).upper() if False else sym(t), sq) for sq, t in sorted(u["pieces"])), u["size"], u["stored"]))
    k13 = {key(u) for u in multi(b0)}
    for s in SEEDS[1:]:
        bs = R[s]["boards"][FIN][bi]
        ks = {key(u) for u in multi(bs)}
        P("   seed %d: %d multi-piece units; identical to a seed-13 unit: %d of %d; units: %s" % (
            s, len(ks), len(ks & k13), len(k13), "; ".join(sorted(ks))))
# cross-seed agreement over all boards
P("\n### cross-seed agreement of decompositions over all %d boards at %s positions" % (len(R[13]["boards"][FIN]), FIN))
rows = []
for a, b in combinations(SEEDS, 2):
    j, cov = [], []
    for ba, bb in zip(R[a]["boards"][FIN], R[b]["boards"][FIN]):
        ka, kb = {key(u) for u in multi(ba)}, {key(u) for u in multi(bb)}
        j.append(len(ka & kb) / max(len(ka | kb), 1))
        pa = {sq for u in multi(ba) for sq, _ in u["pieces"]}
        pb = {sq for u in multi(bb) for sq, _ in u["pieces"]}
        cov.append(len(pa & pb) / max(len(pa | pb), 1))
    rows.append((sum(j) / len(j), sum(cov) / len(cov)))
P("mean over the 6 seed pairs and boards: Jaccard of the multi-piece unit sets %.3f; Jaccard of the covered squares %.3f" % (
    sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows)))
# baseline: same count of units drawn at random from each seed's stored chunks that fit on the board? use pieces-set overlap with random 4-piece subsets
import random
rng = random.Random(213)
rb = []
for a, b in combinations(SEEDS, 2):
    for ba, bb in zip(R[a]["boards"][FIN], R[b]["boards"][FIN]):
        sq = sorted(ba["fen"])
        na, nb = sum(u["size"] for u in multi(ba)), sum(u["size"] for u in multi(bb))
        sa = set(rng.sample(sq, min(na, len(sq))))
        sb = set(rng.sample(sq, min(nb, len(sq))))
        rb.append(len(sa & sb) / max(len(sa | sb), 1))
P("random baseline for the covered-squares Jaccard (each seed covers as many squares as it does, chosen at random among the board's pieces): %.3f" % (sum(rb) / len(rb)))
nk = [len(multi(b)) for b in R[13]["boards"][FIN]]
cv = [sum(u["size"] for u in multi(b)) / len(b["fen"]) for b in R[13]["boards"][FIN]]
P("seed 13: multi-piece units per board %.2f, share of pieces in a multi-piece unit %.2f; stored-chunk share of multi-piece units %.2f" % (
    sum(nk) / len(nk), sum(cv) / len(cv),
    sum(u["stored"] for b in R[13]["boards"][FIN] for u in multi(b)) / max(sum(len(multi(b)) for b in R[13]["boards"][FIN]), 1)))
open(D + "r213_chess_boards.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
