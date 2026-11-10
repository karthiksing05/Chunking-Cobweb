"""Appendix A19: reads the reference-generator logs and writes frag_refs.html.

Every number comes from the logs at generation time; a missing log shows an en dash.
Run from the repo root on a compute node.
"""
import json
import os
import re

SEEDS = (13, 17, 19, 23)
S2 = (13, 17)


def jlines(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f2(x):
    return "–" if x is None else f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.2f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def key(arm):
    return json.dumps(arm, sort_keys=True)


def seed_of(name):
    m = re.search(r"_s(\d+)_", name)
    return int(m.group(1)) if m else None


def judge(path, n=None):
    """{(arm key, seed): judge row} for the rows that carry a file name."""
    out = {}
    for r in jlines(path):
        if not r.get("file"):
            continue
        if n is not None and f"_n{n}_" not in r["file"]:
            continue
        out[(key(r["arm"]), seed_of(r["file"]))] = r
    return out


def genrows(paths):
    """{(arm key, seed): generation row with a cross share}."""
    out = {}
    for p in paths:
        for r in jlines(p):
            if "cross share" in r and "arm" in r:
                out[(key(r["arm"]), r["seed"])] = r
    return out


def td(x, cls=True):
    return f'<td class="num">{x}</td>' if cls else f"<td>{x}</td>"


def row(label, cells):
    return f"<tr><td>{label}</td>" + "".join(td(c) for c in cells) + "</tr>"


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>\n' + "\n".join(rows) + "\n</tbody></table></div>"


def ys(j, k, seeds):
    return [j[(k, s)]["grammatical novel per generation"] for s in seeds if (k, s) in j]


def ymean(j, k, seeds):
    return mean(ys(j, k, seeds))


def jm(j, k, seeds, field):
    return mean([j[(k, s)][field] for s in seeds if (k, s) in j])


# ---------------------------------------------------------------- data
TR = {}  # TRELLIS candidate by n: {seed: row}
for n, path in ((1000, "logs/v3-r104-judge-base.log"), (500, "logs/v3-r106-judge-n500.log"),
                (2000, "logs/v3-r106-judge-n2000.log"), (4000, "logs/v3-r106-judge-n4000.log")):
    TR[n] = {}
    for r in jlines(path):
        if r.get("file") and "arm0." in r["file"] and f"_n{n}_" in r["file"]:
            TR[n][seed_of(r["file"])] = r
REFJ = {n: judge(f"logs/v3-r109-judge-n{n}.log", n) for n in (500, 1000, 2000, 4000)}
REFG = {n: genrows([f"logs/v3-r109-base-n{n}-{s}.log" for s in SEEDS]) for n in (500, 1000, 2000, 4000)}
TRCROSS = {}
for s in SEEDS:
    recs = [r for r in jlines(f"logs/v3-r104-base-story-{s}.log") if "cross share" in r]
    if recs:
        TRCROSS[s] = recs[0]["cross share"]


def diags(n, seed):
    return {r["diag"]: r for r in jlines(f"logs/v3-r109-base-n{n}-{seed}.log") if "diag" in r}


def tr_trigram_stored(seed):
    d = diags(1000, seed).get(f"story_s{seed}_n1000_arm0.jsonl")
    if not d:
        return None
    a = d.get("3-grams all stored, new", {}).get("share of generations")
    b = d.get("3-grams all stored, stored sentence", {}).get("share of generations")
    return None if a is None or b is None else a + b


# ---------------------------------------------------------------- table 1
REFS = [("Bigram sampler", {"ngram": 2}), ("Trigram sampler", {"ngram": 3}), ("4-gram sampler", {"ngram": 4}),
        ("Trigram, topic weight 1", {"ngram": 3, "topic": 1}), ("Trigram, topic weight 2", {"ngram": 3, "topic": 2}),
        ("One-word swap, random sentence", {"swap": "random"}), ("One-word swap, retrieved sentence", {"swap": "retrieve"})]
t1 = []
trs = TR[1000]
t1.append(row("<b>TRELLIS candidate recipe</b>",
              [f2(mean([r["grammatical novel per generation"] for r in trs.values()])),
               " / ".join(f2(trs[s]["grammatical novel per generation"]) if s in trs else "–" for s in SEEDS),
               pct(mean([r["judged grammatical"] for r in trs.values()])), pct(mean([r["novel"] for r in trs.values()])),
               f2(mean(list(TRCROSS.values()))), pct(mean([tr_trigram_stored(s) for s in SEEDS]))]))
for label, arm in REFS:
    k = key(arm)
    j, g = REFJ[1000], REFG[1000]
    y = [j[(k, s)]["grammatical novel per generation"] if (k, s) in j else None for s in SEEDS]
    t1.append(row(label, [f2(mean(y)), " / ".join(f2(v) for v in y), pct(jm(j, k, SEEDS, "judged grammatical")),
                          pct(jm(j, k, SEEDS, "novel")), f2(mean([g[(k, s)]["cross share"] for s in SEEDS if (k, s) in g])),
                          pct(mean([g[(k, s)]["every trigram stored"] for s in SEEDS if (k, s) in g]))]))
T1 = table(["1,000 sentences", "Yield, mean", "Yield by seed (13 / 17 / 19 / 23)", "Grammatical", "New", "Cross share",
            "Every trigram stored"], t1)
ty = mean([r["grammatical novel per generation"] for r in trs.values()])
bests = [(label, ymean(REFJ[1000], key(arm), SEEDS)) for label, arm in REFS]
bests = [b for b in bests if b[1] is not None]
best = max(bests, key=lambda b: b[1]) if bests else (None, None)
tri = ymean(REFJ[1000], key({"ngram": 3}), SEEDS)
swr = ymean(REFJ[1000], key({"swap": "retrieve"}), SEEDS)
ratio = None if not ty or not best[1] else best[1] / ty

# ---------------------------------------------------------------- table 2
def purity(n, s):
    if n == 1000:
        p = f"experiments/v3/results/abl_audit/r104-base-{s}.json"
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)["arms"]["two"]["word purity"]["12"]["tag purity"]
    rows = jlines(f"logs/v3-r108-audit-n{n}-{s}.log")
    if not rows:
        return None
    w = rows[0].get("word purity", {}).get("12")
    return w.get("tag purity") if isinstance(w, dict) else w


t2 = []
for n in (500, 1000, 2000, 4000):
    t2.append(row(f"{n:,}", [
        " / ".join(f2(TR[n][s]["grammatical novel per generation"]) if s in TR[n] else "–" for s in S2),
        " / ".join(f2(v) for v in [ (REFJ[n].get((key({"ngram": 3}), s)) or {}).get("grammatical novel per generation") for s in S2]),
        " / ".join(f2(v) for v in [ (REFJ[n].get((key({"swap": "retrieve"}), s)) or {}).get("grammatical novel per generation") for s in S2]),
        " / ".join(f2(purity(n, s)) for s in S2)]))
T2 = table(["Sentences", "TRELLIS yield", "Trigram yield", "One-word swap (retrieved) yield", "TRELLIS word-concept tag purity at 12 bits"], t2)

# ---------------------------------------------------------------- table 3
t3 = []
DR = [("Every bigram stored", "2-grams all stored, new"), ("An unstored bigram", "2-grams not all stored, new"),
      ("Every trigram stored", "3-grams all stored, new"), ("An unstored trigram", "3-grams not all stored, new")]
for label, k in DR:
    cells = []
    for field in ("share of generations", "judged grammatical"):
        cells.append(" / ".join(f2((diags(1000, s).get(f"story_s{s}_n1000_arm0.jsonl", {}).get(k) or {}).get(field)) for s in S2))
    t3.append(row(label, cells))
T3 = table(["New sentences with", "Share of generations (13 / 17)", "Judged grammatical (13 / 17)"], t3)
held = {s: diags(1000, s).get("held-out real sentences", {}) for s in S2}
T3b = table(["Held-out real sentences", "Seed 13", "Seed 17"],
            [row(lab, [f2(held[s].get(k)) for s in S2]) for lab, k in
             (("Every bigram stored", "every bigram stored"), ("Every trigram stored", "every trigram stored"), ("Stored whole", "stored whole"))])

# ---------------------------------------------------------------- table 4 (round 110)
def label110(a):
    if "tswap" in a:
        unit = "word for word" if a["tswap"] == "word" else "word or chunk"
        if a.get("by") == "fills":
            return f"TRELLIS: {unit}, Cobweb's window prediction" + (f", {a['up']} level{'s' if a['up'] != 1 else ''} up" if a.get("up") else "")
        return f"TRELLIS: {unit}, concept {a['up']} level{'s' if a['up'] != 1 else ''} up"
    if a.get("unit") == "span":
        return "Reference: span of 1–3 words, two swaps" if a.get("n") == 2 else "Reference: span of 1–3 words"
    if a.get("frame") == "wide":
        return "Reference: wide frame (two neighbours each side)"
    if a.get("frame") == "bigram":
        return "Reference: bigram licence"
    if a.get("n") == 2:
        return "Reference: two swaps"
    return "Reference: one swap, retrieved sentence"


t4 = []
missing110 = []
for kind, gglob in (("ref", "ref"), ("tswap", "tswap")):
    # round 110 was judged twice: read whichever judge log has more arm rows
    cands = [f"logs/v3-r110-judge-{kind}.log", f"logs/v3-r110-{kind}-judge.log"]
    jpath = max(cands, key=lambda p: len([r for r in jlines(p) if r.get("file")]))
    rows = [r for r in jlines(jpath) if r.get("file")]
    if not rows:
        missing110.append(" or ".join(cands))
        continue
    gens = genrows([f"logs/v3-r110-{gglob}-{s}.log" for s in SEEDS])
    dg = {}
    for s in SEEDS:
        for p in (f"logs/v3-r110-diag-{kind}-{s}.log", f"logs/v3-r110-{kind}-diag-{s}.log"):
            for r in jlines(p):
                if "diag" in r and str(r["diag"]).endswith(".jsonl"):
                    dg[(r["diag"], s)] = r
    arms = []
    for r in sorted(rows, key=lambda r: r["file"]):
        if key(r["arm"]) not in [k for k, _ in arms]:
            arms.append((key(r["arm"]), r["arm"]))
    j = judge(jpath)
    fileof = {(key(r["arm"]), seed_of(r["file"])): r["file"] for r in rows}
    for k, a in arms:
        sd = [s for s in SEEDS if (k, s) in j]
        sh, gr = [], []
        for s in sd:
            d = dg.get((fileof[(k, s)], s), {})
            e = d.get("2-grams not all stored, new")
            if e:
                sh.append(e.get("share of generations"))
                gr.append(e.get("judged grammatical"))
        t4.append(row(label110(a), [
            f"{f2(ymean(j, k, SEEDS))}" + ("" if len(sd) == len(SEEDS) else f" ({len(sd)} seeds)"),
            pct(jm(j, k, SEEDS, "judged grammatical")), pct(jm(j, k, SEEDS, "novel")),
            f2(mean([gens[(k, s)]["cross share"] for s in SEEDS if (k, s) in gens])),
            pct(mean([gens[(k, s)]["every bigram stored"] for s in SEEDS if (k, s) in gens])),
            f"{f2(mean(sh))}; {f2(mean(gr))}" if sh else "–"]))
T4 = table(["Generator (1,000 sentences)", "Yield, mean", "Grammatical", "New", "Cross share", "Every bigram stored",
            "New with an unstored bigram: share of generations; judged grammatical"], t4) if t4 else \
    '<p class="small">–</p>'

# ---------------------------------------------------------------- table 5 (generalization step alone)
NS5 = (500, 1000, 2000, 4000)


def licvals(fmt, name, n, seeds, getter=lambda r: r.get("average precision")):
    """(values, seeds used) for one licence name over the seed logs that exist."""
    vals, used = [], []
    for sd in seeds:
        for r in jlines(fmt.format(n=n, s=sd)):
            if r.get("licence") == name:
                v = getter(r)
                if v is not None:
                    vals.append(v)
                    used.append(sd)
                break
    return vals, used


R118 = "logs/v3-r118-words-tok-exact-set-sides-n{n}-{s}.log"
R117 = "logs/v3-r117-words-tok-exact-set-n{n}-{s}.log"
R113 = "logs/v3-r113-lic-n{n}-{s}.log"
R115 = "logs/v3-r115-tree-n{n}-{s}.log"
R122 = "logs/v3-r122-inc-n{n}-{s}.log"
base_get = lambda r: r.get("base rate")
p10_get = lambda r: (r.get("precision at recall") or {}).get("0.1")

# (label, path format, licence name, seeds, getter, precision at recall is valid)
LIC_ROWS = [
    ("Base rate", R118, "setup", SEEDS, base_get, False),
    ("Product of the two words' frequencies", R118, "frequency of the two words", SEEDS, None, True),
    ("Cobweb's prediction from the candidate memory (the current learner)", R113, "Cobweb, both", S2, None, False),
    None,  # best of the recipe variants (round 114)
    ("A Cobweb tree over word pairs", R115, "pair tree, all levels", S2, None, True),
    ("One Cobweb tree over word types, read by token counts", R117, "word tree, back-off", S2, None, True),
    ("The same tree, read by types (each member word counts once)", R117, "word tree by types, sum", S2, None, True),
    ("Words with similar neighbours vote", R118, "similar words", SEEDS, None, True),
    ("Analogy count", R118, "analogy, count", SEEDS, None, True),
    ("<b>Two Cobweb trees over word types, one per side, read by types</b>", R118, "word trees by side, by types, sum", SEEDS, None, True),
    ("The same two trees, kept current one occurrence at a time", R122, "word trees by side, by types, sum", SEEDS, None, True),
]

# round 114: best of the recipe variants, 1,000 sentences, "Cobweb, both"
variant_means = {}
if os.path.isdir("logs"):
    for fn in sorted(os.listdir("logs")):
        m = re.match(r"^v3-r114-lic-(.+)-(13|17)\.log$", fn)
        if not m:
            continue
        v = [r.get("average precision") for r in jlines(f"logs/{fn}") if r.get("licence") == "Cobweb, both"]
        v = [x for x in v if x is not None]
        if v:
            variant_means.setdefault(m.group(1), []).append(v[0])
variant_means = {k: mean(v) for k, v in variant_means.items()}

t5 = []
seedinfo5 = []
for spec in LIC_ROWS:
    if spec is None:
        t5.append(row("The same, best of 11 recipe variants",
                      ["–", f2(max(variant_means.values())) if variant_means else "–", "–", "–", "–"]))
        continue
    label, fmt, name, seeds, getter, pvalid = spec
    cells, used1000 = [], []
    for n in NS5:
        vals, used = licvals(fmt, name, n, seeds, getter or (lambda r: r.get("average precision")))
        cells.append(f2(mean(vals)))
        if n == 1000:
            used1000 = used
    pv, _ = licvals(fmt, name, 1000, seeds, p10_get) if pvalid else ([], [])
    cells.append(f2(mean(pv)) if pv else "–")
    t5.append(row(label, cells))
    seedinfo5.append((re.sub(r"<[^>]+>", "", label), used1000))
T5 = table(["Licence (mean over the seeds present: 13, 17, 19, 23 or 13, 17 only; seeds at 1,000 listed below)",
            "Average precision, 500 sentences", "Average precision, 1,000", "Average precision, 2,000", "Average precision, 4,000",
            "Precision at 10% recall, 1,000"], t5)
if variant_means:
    vs = sorted(variant_means.values())
    T5v = (f'<p class="small">Over the {len(vs)} recipe variants that produced a score, the seed-mean average precision at 1,000 '
           f'sentences runs from {f2(vs[0])} to {f2(vs[-1])}.</p>')
else:
    T5v = '<p class="small">No recipe variant produced a score.</p>'
T5s = ('<p class="small">Seeds at 1,000 sentences: ' +
       "; ".join(f"{lab}: {', '.join(str(x) for x in u) or '–'}" for lab, u in seedinfo5) + ".</p>")
p5_intro = ("<p>A generator has to decide which word pairs it may produce that the stream never showed. This test isolates that decision. "
            "Every pair of stream words that never stood side by side is scored by a licence; a pair counts as right when the whole corpus "
            "has it at least twice. Average precision ranks the pairs by the licence; the base rate is the share of unstored pairs that are "
            "right. No generation and no judge are involved, so one run takes minutes.</p>")
p5_defs = ("<p>Analogy count: for an unstored pair (a, b), the number of stored pairs (a2, b2) such that (a, b2) and (a2, b) are also stored. "
           "Two trees by side: each word type is described by the set of words stored just after it in one tree and just before it in the other; "
           "a word weighs as its count; a new concept is scored by its own expected correct guesses (exact_alone). The licence for (a, b) is "
           "the share of the words grouped with a that b followed, summed over the concepts on a's path, combined with the same reading for b's side.</p>")

# ---------------------------------------------------------------- table 6 (one licensed substitution)
FRAME = {"vote": "Frame votes (between the two neighbours)", "vote-wide": "Frame votes (plus two neighbours each side)",
         "vote-all": "Frame votes (plus each neighbour alone)"}


def pick_vote(a):
    p = a.get("pick")
    if p == "best":
        return ", random slot, best candidate"
    if p == "slot":
        return ", slot drawn by ratio"
    if p == "slot-best":
        return ", best slot"
    return ", random slot, drawn by votes"


def lab119(a):
    base = {"tree": "An unstored pair forced; two-tree pair licence", "count": "An unstored pair forced; analogy count"}.get(a.get("frame"), str(a))
    return base + (", best candidate" if a.get("pick") == "best" else ", drawn by licence")


def lab120(a):
    return FRAME.get(a.get("frame"), str(a)) + pick_vote(a)


def lab123(a):
    unit = {"words": "TRELLIS memory, word units", "chunks": "TRELLIS memory, committed chunks as units"}.get(a.get("uswap"), str(a))
    places = {"wide": " (two neighbours each side)", "all": " (plus each neighbour alone)"}.get(a.get("places"), "")
    return unit + places + pick_vote(a) + (", conditioned" if a.get("cond") else "")


PAIR_REQ = "An unstored pair required; "


def pick_slot(a):
    p = a.get("pick")
    if p == "slot-best":
        return ", best slot"
    if p == "slot":
        return ", slot drawn by ratio"
    return pick_vote(a)


def lab125(a):
    fr = {"vote-new": "frame votes (between the two neighbours)", "vote-wide-new": "frame votes (plus two neighbours each side)",
          "vote-all-new": "frame votes (plus each neighbour alone)", "tree": "two-tree pair licence", "count": "analogy count"}
    return PAIR_REQ + fr.get(a.get("frame"), str(a)) + pick_slot(a)


def lab126(a):
    fr = {"vote-new-pair": PAIR_REQ + "frame votes times pair licence (between the two neighbours)",
          "vote-wide-new-pair": PAIR_REQ + "frame votes times pair licence (plus two neighbours each side)",
          "vote-wide-pair": "Frame votes times pair licence on new pairs (plus two neighbours each side)"}
    lab = fr.get(a.get("frame"), str(a)) + pick_slot(a)
    return f"<b>{lab}</b>" if a.get("frame") == "vote-wide-new-pair" else lab


REQUIRED_FRAMES = {"vote-new", "vote-wide-new", "vote-all-new", "tree", "count", "vote-new-pair", "vote-wide-new-pair"}


def armidx(name):
    m = re.search(r"arm(\d+)", name or "")
    return int(m.group(1)) if m else None


def rounds6(n):
    """Sources per round for sentence count n: (label fn, judge path, generation paths, diag paths)."""
    ss = SEEDS if n == 1000 else S2
    return [
        (lab119, f"logs/v3-r119-judge-n{n}.log", [f"logs/v3-r119-lic-n{n}-{x}.log" for x in ss], [], False),
        (lab120, f"logs/v3-r120-judge-n{n}.log", [f"logs/v3-r120-vote-n{n}-{x}.log" for x in ss],
         [f"logs/v3-r120-diag-n{n}-{x}.log" for x in ss], False),
        (lab120, f"logs/v3-r121-judge-n{n}.log", [f"logs/v3-r121-slot-n{n}-{x}.log" for x in ss],
         [f"logs/v3-r121-diag-n{n}-{x}.log" for x in ss], False),
        (lab125, f"logs/v3-r125-judge-n{n}.log", [f"logs/v3-r125-new-n{n}-{x}.log" for x in ss], [], False, True),
        (lab126, f"logs/v3-r126-judge-n{n}.log", [f"logs/v3-r126-pair-n{n}-{x}.log" for x in ss],
         [f"logs/v3-r126-diag-n{n}-{x}.log" for x in ss], False, True),
        (lab123, f"logs/v3-r123-judge-n{n}.log", [f"logs/v3-r123-uswap-n{n}-{x}.log" for x in ss],
         [f"logs/v3-r123-diag-n{n}-{x}.log" for x in ss], True),
    ]


def diagmap(paths):
    out = {}
    for p in paths:
        for r in jlines(p):
            if "diag" in r and str(r["diag"]).endswith(".jsonl"):
                out[r["diag"]] = r
    return out


def dgm(dm, files, field, k):
    return mean([(dm.get(f, {}).get(k) or {}).get(field) for f in files])


def row6(label, j, gens, dm, arm, full, withchunk, ordered, req=False):
    k = key(arm)
    isreq = bool(req) and arm.get("frame") in REQUIRED_FRAMES
    sd = [x for x in full if (k, x) in j]
    if not sd:
        return None
    files = [j[(k, x)]["file"] for x in sd]
    g = []
    for x in sd:
        r = gens.get((k, x))
        if r is None and ordered.get(x):  # fall back to the arm's position in the seed's generation log
            i = armidx(j[(k, x)]["file"])
            if i is not None and i < len(ordered[x]):
                r = ordered[x][i]
        if r is not None:
            g.append(r)
    ebs = [r.get("every bigram stored") for r in g]
    unst = None if not ebs or any(e is None for e in ebs) else 1 - mean(ebs)
    ys_ = [j[(k, x)]["grammatical novel per generation"] for x in sd]
    cells = [f2(mean(ys_)) + ("" if len(sd) == len(full) else f" ({len(sd)} seed{'s' if len(sd) != 1 else ''})"),
             pct(mean([j[(k, x)]["judged grammatical"] for x in sd])), pct(mean([j[(k, x)]["novel"] for x in sd])),
             f2(mean([r.get("cross share") for r in g])), pct(unst),
             (f2(mean([j[(k, x)]["judged grammatical"] for x in sd])) if isreq
              else f2(dgm(dm, files, "judged grammatical", "2-grams not all stored, new"))),
             "–" if isreq else f2(dgm(dm, files, "judged grammatical", "2-grams all stored, new")),
             f2(mean([r.get("chunk for chunk") for r in g])) if withchunk else "–"]
    return row(label, cells)


def table6(n):
    full = SEEDS if n == 1000 else S2
    rows = []
    # references
    j, gens = REFJ[n], REFG[n]
    ordd = {}
    dm = diagmap([f"logs/v3-r109-base-n{n}-{x}.log" for x in full])
    r_ = row6("Reference: word stored between the same two neighbours", j, gens, dm, {"swap": "retrieve"}, full, False, ordd)
    if r_:
        rows.append(r_)
    # TRELLIS candidate recipe
    trn = TR[n]
    sd = [x for x in full if x in trn]
    if sd:
        cr = [TRCROSS.get(x) for x in sd] if n == 1000 else []
        stored = [tr_trigram_bigram(x) for x in sd] if n == 1000 else []
        files = [trn[x]["file"] for x in sd]
        dmt = dm if n == 1000 else {}
        unst = None if not stored or any(v is None for v in stored) else 1 - mean(stored)
        rows.append(row("TRELLIS candidate recipe (chart generation)", [
            f2(mean([trn[x]["grammatical novel per generation"] for x in sd])) + ("" if len(sd) == len(full) else f" ({len(sd)} seeds)"),
            pct(mean([trn[x]["judged grammatical"] for x in sd])), pct(mean([trn[x]["novel"] for x in sd])),
            f2(mean(cr)) if cr and None not in cr else "–", pct(unst),
            f2(dgm(dmt, files, "judged grammatical", "2-grams not all stored, new")),
            f2(dgm(dmt, files, "judged grammatical", "2-grams all stored, new")), "–"]))
    for labfn, jp, gp, dp, wc, *rq in rounds6(n):
        jr = judge(jp, n)
        if not jr:
            continue
        gens = genrows(gp)
        ordd = {}
        for p in gp:
            rs = [r for r in jlines(p) if "cross share" in r and "arm" in r]
            if rs:
                ordd[rs[0]["seed"]] = rs
        dm = diagmap(dp)
        arms = {}
        for (k, x), r in sorted(jr.items(), key=lambda kv: (armidx(kv[1]["file"]) if armidx(kv[1]["file"]) is not None else 99, kv[0][1] or 0)):
            arms.setdefault(k, r["arm"])
        for k, a in arms.items():
            r_ = row6(labfn(a), jr, gens, dm, a, full, wc, ordd, bool(rq and rq[0]))
            if r_:
                rows.append(r_)
    head = [f"{n:,} sentences (seeds {' / '.join(str(x) for x in full)})", "Yield, mean", "Grammatical", "New", "Cross share",
            "Outputs with an unstored bigram", "New with an unstored bigram, judged grammatical",
            "New with every bigram stored, judged grammatical", "Chunk involved"]
    return table(head, rows) if rows else '<p class="small">–</p>'


def tr_trigram_bigram(seed):
    d = diags(1000, seed).get(f"story_s{seed}_n1000_arm0.jsonl")
    if not d:
        return None
    a = d.get("2-grams all stored, new", {}).get("share of generations")
    b = d.get("2-grams all stored, stored sentence", {}).get("share of generations")
    return None if a is None or b is None else a + b


T6 = {n: table6(n) for n in (1000, 4000)}
p6_intro = ("<p>Each generator below takes the stored sentence that followed the stored sentence most like the prompt and replaces one unit. "
            "They differ in what may replace it and how the slot is chosen.</p>")
p6_defs = ("<p>Frame votes: a candidate word is scored by how often the words grouped with it, in the two word-type trees, stood in the place "
           "around the slot; the candidate itself counts where it stood there. Best slot: every position offers its best candidate, weighed by "
           "that candidate's votes relative to the votes of the word standing there, and the highest ratio is taken. The references and rounds "
           "119 to 121 read only the stream; round 123 runs the same procedure on the learner's memory. "
           "Pair licence: the two-tree licence of Table 5 for a pair of neighbouring words. Frame votes times pair licence: a replacement that "
           "makes an unstored pair is scored by its frame votes multiplied by the pair licence of each unstored pair it makes, so it must be "
           "supported by both readings of the trees.</p>")

# ---------------------------------------------------------------- table 7 (the licence as a coder)
CODERS = ["bigram, backed off to shares", "bigram, backed off to shares and the two trees' licence",
          "bigram, backed off to shares and the analogy count"]
t7 = []
for n in (500, 1000, 2000, 4000):
    bits = {c: {} for c in CODERS}
    for sd in S2:
        for r in jlines(f"logs/v3-r124-code-n{n}-{sd}.log"):
            if r.get("coder") in bits and r.get("bits per sentence") is not None:
                bits[r["coder"]][sd] = r["bits per sentence"]
    cells = [" / ".join(f2(bits[c].get(sd)) if bits[c].get(sd) is None else f"{bits[c][sd]:.2f}" for sd in S2) for c in CODERS]
    sv = [bits[CODERS[0]][sd] - bits[CODERS[1]][sd] for sd in S2 if sd in bits[CODERS[0]] and sd in bits[CODERS[1]]]
    cells.append("–" if not sv else f"{mean(sv):.2f}")
    t7.append(row(f"{n:,}", cells))
T7 = table(["Sentences", "Bigram backed off to shares (bits per held-out sentence, seeds 13 / 17)",
            "Plus the two trees' licence", "Plus the analogy count", "Saving of the two-tree coder over the baseline (bits, seed mean)"], t7)
p7_intro = ("<p>The licence gives a distribution over the next word, so it can be used to code held-out sentences. The baseline is a bigram "
            "model backed off to word shares by Witten-Bell. The other two mix the licence's distribution over next words, half and half, "
            "into that back-off. The half is hand-set.</p>")

# ---------------------------------------------------------------- page
def s(x):
    return "–" if x is None else f2(x)


lead = (f"<p>Simple reference generators, run on the same stream, prompts and judge, produce more new grammatical sentences than TRELLIS. "
        f"At 1,000 sentences TRELLIS's candidate recipe yields {s(ty)} per generation; the trigram sampler yields {s(tri)}, "
        f"the one-word swap on a retrieved sentence {s(swr)}, and the best reference ({best[0] or '–'}) {s(best[1])}, "
        f"{'–' if ratio is None else f'{ratio:.1f}'} times TRELLIS. TRELLIS's errors come from word pairs it never saw next to each other. "
        f"This page reports the references, the split of TRELLIS's errors and what follows from it.</p>"
        "<p><b>What worked.</b> Two Cobweb trees over word types, read by counting each member word once, decide which unseen word pairs "
        "may be produced far better than the learner's context concepts do (Table 5). A generator that replaces one word of a retrieved "
        "stored sentence, and accepts a replacement only when two readings of those trees agree, produces sentences that all contain a "
        "never-seen word pair and are about as often grammatical as the best reference, which never leaves stored pairs (Table 6). The same "
        "trees shorten the code for held-out sentences (Table 7). None of this is inside the learner yet: it reads the stream, one word "
        "at a time, and the learner's chunks did not help as units.</p>")

p_how = ("<h3>How to read this page</h3>"
         "<p><b>Yield</b> is the number of judged-grammatical new sentences per generation. <b>Cross share</b> is the topic agreement with "
         "the previous sentence, defined as in <a href=\"#a-rationale\">A18</a>. A word sequence is <b>stored</b> when it occurs in the "
         "stream the learner saw. The references never see the learner; they read only the stream.</p>")

p1 = ("<p>A k-gram sampler draws each word from the counts of what followed the previous k−1 words in the stream. A topic weight raises "
      "the counts of words that occur in sentences near the prompt. The one-word swap takes a stored sentence and replaces one word with "
      "another word that was stored between the same two neighbours. In the random swap the sentence is drawn at random; in the retrieved "
      "swap it is the stored sentence that followed the stored sentence most like the prompt.</p>")

p2 = (f"<p>TRELLIS falls as data grows while the references stay level. The word concept that generation reads becomes less pure as memory "
      f"grows (word-concept tag purity at 12 bits: "
      f"{s(mean([purity(500, x) for x in S2]))} at 500 sentences, {s(mean([purity(1000, x) for x in S2]))} at 1,000, "
      f"{s(mean([purity(2000, x) for x in S2]))} at 2,000, {s(mean([purity(4000, x) for x in S2]))} at 4,000).</p>")

hb = [mean([held[x].get("every bigram stored") for x in S2]), mean([held[x].get("every trigram stored") for x in S2]),
      mean([held[x].get("stored whole") for x in S2])]
p3 = (f"<p>New sentences built only from stored pairs are mostly grammatical, and those with an unstored pair mostly are not. "
      f"Real held-out sentences contain an unstored bigram in {'–' if hb[0] is None else pct(1 - hb[0])} of cases and an unstored trigram in "
      f"{'–' if hb[1] is None else pct(1 - hb[1])}; {pct(hb[2])} are stored whole. A generator restricted to stored pairs cannot reach them.</p>")

p4 = ("<ul>"
      "<li><b>The stored frame decides grammaticality.</b> A replacement stored between the same two neighbours on each side gives the most "
      "grammatical output and the fewest new sentences; one neighbour on each side gives more new sentences at a lower rate; two swaps in "
      "a row, or a replacement licensed only by its two stored pairs, cost grammaticality.</li>"
      "<li><b>TRELLIS's context concepts are poor substitution classes.</b> Replacing a word with another member of its context concept is "
      "less often grammatical than every reference swap, and each level up the tree makes it worse. Cobweb's window prediction and "
      "chunk-sized units do no better.</li></ul>")

p5 = ("<ul>"
      "<li><b>The current learner generalizes about neighbours no better than word frequency.</b> Every recipe variant we could score "
      "ranks unstored pairs about as well as multiplying the two words' frequencies. This is the measured cause of the errors in Table 3.</li>"
      "<li><b>Three changes were each needed.</b> Describing word types by their neighbours, instead of occurrences by their windows, "
      "helps a little when the tree is read by token counts. Reading the same tree by types, each member word counting once, helps much "
      "more: one frequent collocation no longer stands for its whole group. Splitting the description into one tree per side reaches the "
      "analogy count. A tree over word pairs does worse than word frequency.</li>"
      "<li><b>It fits one-pass learning and grows with data.</b> Kept current one occurrence at a time, the two trees score the same as "
      "when built at the end. They pass the analogy count at 2,000 and 4,000 sentences.</li>"
      "<li><b>exact_alone is required.</b> Without it the word-type tree stays nearly flat (a mean path of about three concepts) and "
      "ranks pairs no better than word frequency.</li></ul>")

p6 = ("<ul>"
      "<li><b>Take the best candidate; do not draw.</b> Drawing a replacement in proportion to its votes is grammatical about a quarter "
      "of the time: most of the vote mass lies on many weakly supported words.</li>"
      "<li><b>Choosing the slot matters as much as choosing the word.</b> Letting every position offer its best replacement and taking "
      "the best offer beats the reference swap on yield. Most of that gain is reuse: few of its outputs contain an unstored pair.</li>"
      "<li><b>One reading of the trees is not enough to generalize.</b> When an unstored pair is required, frame votes alone or the pair "
      "licence alone are grammatical 39 to 52% of the time. The analogy count, which ranked pairs well in Table 5, is grammatical 10 to "
      "16% of the time: its best-scoring pairs are pairs of frequent words.</li>"
      "<li><b>Two readings that agree generalize as well as reuse.</b> Scoring a replacement by its frame votes times the pair licence of "
      "each unstored pair it makes gives 67 to 68% grammatical output at 1,000 sentences and 59 to 61% at 4,000, with every output new and "
      "containing a never-seen pair. The reference swap, which makes no unstored pair, is 69% and 67% grammatical. Yield is .67 to .68 "
      "against .57 at 1,000 sentences and .59 to .61 against .51 at 4,000. With the best slot and no requirement, the outputs that "
      "contain an unstored pair are at least as often grammatical as those that do not.</li>"
      "<li><b>The learner's chunks do not help as units.</b> Running the same procedure on the learner's memory with its committed chunks "
      "as units gives equal or lower yield than words alone (round 123 rows). Its parses are weak (A18), so its chunks are poor units "
      "to substitute.</li></ul>"
      "<p class=\"small\">The judge passes 86 to 91% of real held-out sentences, so about 68% is still short of real text. The topic "
      "comes from retrieval: these generators inherit the cross share of the stored sentence they start from.</p>")

p7 = ("<p>The licence shortens the code at every size on both seeds. The saving shrinks as the stream grows, since more of the pairs a "
      "held-out sentence needs are already stored. The two trees and the analogy count save about the same.</p>")

p8 = ("<h3>What follows</h3><ul>"
      "<li><b>For the learner.</b> Its context tree describes occurrences by their windows. The licence that works describes types by "
      "their sets of neighbours, one tree per side, and reads concepts by types. Moving this into the learner is the next step; "
      "<code>src/trellis2/type_trees.py</code> holds the mechanism.</li>"
      "<li><b>For generation.</b> Chart generation draws every word from smoothed distributions and pays for it in unstored pairs. "
      "Substitution on a retrieved analysis keeps what is stored and generalizes only where two readings agree. It has no cut, no "
      "smoothing weight and no topic weight.</li>"
      "<li><b>For chunks.</b> The frame votes are a substitution test: a span that stands where single words stand, and is voted for by "
      "their groups, is a unit. Using that test to find chunks has not been tried.</li></ul>"
      "<h3>What this does not settle</h3><ul>"
      "<li>Everything here replaces one word. Two or more substitutions, and units longer than a word, are open.</li>"
      "<li>Stories only. Characters and chess have not been run with these trees.</li>"
      "<li>The coder's half-and-half mix is hand-set.</li>"
      "<li>Rounds 110 and 119 to 123 are judged on fewer seeds where the tables say so.</li></ul>")

out = (lead + p_how + "<h3>Table 1. Reference generators at 1,000 sentences, 4 seeds</h3>" + T1 + p1 +
       "<h3>Table 2. With more data (seeds 13 / 17)</h3>" + T2 + p2 +
       "<h3>Table 3. Where TRELLIS's errors come from (1,000 sentences, seeds 13 / 17)</h3>" + T3 + T3b + p3 +
       "<h3>Table 4. One substitution at a time (round 110)</h3>" + T4 + p4 +
       "<h3>Table 5. The generalization step alone</h3>" + p5_intro + T5 + T5v + T5s + p5_defs + p5 +
       "<h3>Table 6. Generating by one licensed substitution</h3>" + p6_intro +
       "<h4>1,000 sentences</h4>" + T6[1000] + "<h4>4,000 sentences</h4>" + T6[4000] + p6_defs + p6 +
       "<h3>Table 7. The same licence as a coder</h3>" + p7_intro + T7 + p7 + p8)
with open("docs/site/frag_refs.html", "w", encoding="utf-8") as f:
    f.write(out)

# plain-text echo for the report
import re as _re
print(_re.sub(r"<[^>]+>", " | ", out.replace("</tr>", "\n")).replace("| |", "|"))
print("missing r110:", missing110)
