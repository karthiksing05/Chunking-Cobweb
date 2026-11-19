"""Round 213 (language) case studies and chunk biographies from the r213_lang.py dumps.

Usage (compute node, repo root): r213_lang_cases.py <held-out sentence indices, comma list> <biography strings separated by |> [seed=13]
Writes experiments/v3/results/story_gen/r213/cases.txt
"""
import json
import sys
from collections import Counter

D = "experiments/v3/results/story_gen/r213/"
SEEDS = [13, 17, 19, 23]
idxs = [int(x) for x in sys.argv[1].split(",")]
bios = [x.strip() for x in sys.argv[2].split("|") if x.strip()]
S0 = int(sys.argv[3]) if len(sys.argv) > 3 else 13
R = {s: json.load(open(f"{D}lang_s{s}.json", encoding="utf-8")) for s in SEEDS}
held = [tuple(x) for x in R[S0]["held"]]
out = []


def P(*a):
    out.append(" ".join(str(x) for x in a))


def render(words, spans):
    sp = sorted({(i, j) for i, j in spans}, key=lambda t: (t[0], -t[1]))
    kept = []
    for a in sp:
        if all(not (a[0] < b[1] and b[0] < a[1]) or (a[0] >= b[0] and a[1] <= b[1]) or (b[0] >= a[0] and b[1] <= a[1]) for b in kept):
            kept.append(a)

    def rec(i, j):
        parts, k = [], i
        inside = [x for x in kept if i <= x[0] and x[1] <= j and x != (i, j)]
        tops = [x for x in inside if not any(y != x and y[0] <= x[0] and x[1] <= y[1] for y in inside)]
        for a in sorted(tops):
            while k < a[0]:
                parts.append(words[k])
                k += 1
            parts.append(rec(*a))
            k = a[1]
        while k < j:
            parts.append(words[k])
            k += 1
        return "[" + " ".join(parts) + "]" if (i, j) in kept else " ".join(parts)
    return rec(0, len(words))


def concept_of(s, cp, string):
    """Deepest concept on the string's content path holding >= 4 distinct member strings (its own concept group)."""
    C = R[s]["concept"][str(cp)]
    path = C["path"].get(string)
    if not path:
        return None
    best = None
    for nid, bits in path:
        nd = C["nodes"][str(nid)]
        if nd["members"] >= 4:
            best = (nid, bits, nd)
    return best


def occurrences(string):
    w = string.split()
    return [(h, i) for h, s in enumerate(held) for i in range(len(s) - len(w) + 1) if list(s[i:i + len(w)]) == w]


def brackets_of(s, cp, h):
    return {tuple(b) for b in R[s]["parses"][str(cp)][h]["brackets"]}


CP = "1000"
gold = R[S0]["gold"]
for h in idxs:
    w = held[h]
    P("\n### held-out sentence %d: %s" % (h, " ".join(w)))
    for s in SEEDS:
        tag = "seed %d%s" % (s, " (shown)" if s == S0 else "")
        P("  %-14s %s" % (tag, R[s]["parses"][CP][h]["tree"]))
    P("  NLTK reference %s" % render(w, [(a, b) for a, b, _ in gold[h]]))
    for i, j, post in sorted(R[S0]["parses"][CP][h]["chunks"], key=lambda t: (t[1] - t[0], t[0])):
        if j - i >= len(w):
            continue
        st = " ".join(w[i:j])
        agree = [s for s in SEEDS if s != S0 and (i, j) in brackets_of(s, CP, h)]
        train_cnt = len([x for x in R[S0]["pos"].get(st, []) if x < 1000])
        stored = st in set(R[S0]["stored"][CP])
        occ = occurrences(st)
        elsewhere = [(hh, ii) for hh, ii in occ if hh != h]
        per_seed = []
        for s in SEEDS:
            tot_s = sum(1 for hh, ii in occ if hh != h or True)
            b = sum(1 for hh, ii in occ if (ii, ii + (j - i)) in brackets_of(s, CP, hh))
            per_seed.append("%d:%d/%d" % (s, b, tot_s))
        gold_here = (i, j) in {(a, b) for a, b, _ in gold[h]}
        P("  chunk '%s' span %d-%d posterior %.2f | in NLTK reference: %s | same span bracketed by other seeds: %s | stored as a composite in memory: %s, learn-tree bracket count in the 1000 training sentences: %d"
          % (st, i, j, post, "yes" if gold_here else "no", ",".join(map(str, agree)) or "none", "yes" if stored else "no", train_cnt))
        P("      this string in the 400 held-out sentences: %d occurrences; bracketed as this span per seed %s" % (len(occ), " ".join(per_seed)))
        c = concept_of(S0, CP, st)
        pth = R[S0]["concept"][CP]["path"].get(st)
        if pth and len(pth) >= 2:
            pn = R[S0]["concept"][CP]["nodes"][str(pth[-2][0])]
            P("      nearest concept c%s (%.1f bits, %d member strings): %s" % (pth[-2][0], pth[-2][1], pn["members"], "; ".join(a for a, _ in pn["top"][:8] if a != st)))
        if c:
            nid, bits, nd = c
            P("      concept c%s (%.1f bits, %d member strings); members: %s" % (nid, bits, nd["members"], "; ".join(a for a, _ in nd["top"] if a != st)[:230]))
        else:
            P("      no content-concept record kept for this string")
# biographies
P("\n## chunk biographies (count = learn-tree bracket occurrences in the stream so far)")
for st in bios:
    P("\n### '%s'" % st)
    for s in SEEDS:
        pos = sorted(R[s]["pos"].get(st, []))
        tl = [sum(1 for x in pos if x < c) for c in (100, 250, 500, 750, 1000)]
        first = pos[0] if pos else None
        stored_at = [c for c in (250, 500, 1000) if st in set(R[s]["stored"][str(c)])]
        row = "  seed %d: first bracketed at sentence %s; count at 100/250/500/750/1000 sentences: %s; stored composite at checkpoints %s" % (s, first, tl, stored_at)
        P(row)
        for c in (250, 500, 1000):
            cc = concept_of(s, c, st)
            if cc:
                nid, bits, nd = cc
                P("      cp %d concept c%s (%.1f bits, %d members): %s" % (c, nid, bits, nd["members"], "; ".join(a for a, _ in nd["top"][:8] if a != st)))
            else:
                P("      cp %d: no concept record (string not stored/tracked at this checkpoint)" % c)
        ctx = [" ".join(R[s]["train"][x]) for x in pos[:3]]
        P("      first training sentences where it was a bracket: %s" % " | ".join(ctx))
open(D + "cases.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))

# top chunks with their concepts (seed S0, cp 1000)
out2 = ["\n## top chunks of seed %d at 1000 with the concept each falls under (nearest = parent of its leaf; wider = deepest concept with >= 4 member strings)" % S0]
cnt = Counter({k: sum(1 for x in v if x < 1000) for k, v in R[S0]["pos"].items()})
for st, c in sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[:30]:
    pth = R[S0]["concept"]["1000"]["path"].get(st)
    row = "%s (%d)" % (st, c)
    if pth and len(pth) >= 2:
        pn = R[S0]["concept"]["1000"]["nodes"][str(pth[-2][0])]
        row += " | nearest c%s %d members: %s" % (pth[-2][0], pn["members"], "; ".join(a for a, _ in pn["top"][:6] if a != st))
    cc = concept_of(S0, "1000", st)
    if cc:
        row += " | wider c%s (%.1f bits, %d members): %s" % (cc[0], cc[1], cc[2]["members"], "; ".join(a for a, _ in cc[2]["top"][:6] if a != st))
    out2.append(row)
open(D + "cases.txt", "a", encoding="utf-8").write("\n".join(out2) + "\n")
print("\n".join(out2))
