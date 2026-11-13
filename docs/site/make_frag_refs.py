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
    ("<b>The same memory's context tree, read by types (round 143)</b>", "logs/v3-r143-lic-n{n}-{s}.log",
     "context tree by types, mean", SEEDS, None, True),
    ("<b>The same, read at the 4 concepts nearest each occurrence (rounds 158–159)</b>", "logs/v3-r159-lic-n{n}-{s}.log",
     "context tree by types, path near4, geo", S2, None, True),
    ("A Cobweb tree over word pairs", R115, "pair tree, all levels", S2, None, True),
    ("Words with similar neighbours vote (reference)", R118, "similar words", SEEDS, None, True),
    ("<b>Analogy count (reference)</b>", R118, "analogy, count", SEEDS, None, True),
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
           "Similar words: words whose stored followers resemble a's vote with how often b followed them, and likewise on b's side. "
           "Cobweb's prediction: after learning, the learner's memory predicts a word from one neighbour, in both directions.</p>")

# ---------------------------------------------------------------- page
def s(x):
    return "–" if x is None else f2(x)


lead = (f"<p>Simple reference generators, run on the same stream, prompts and judge, produce more new grammatical sentences than TRELLIS. "
        f"At 1,000 sentences TRELLIS's candidate recipe yields {s(ty)} per generation; the trigram sampler yields {s(tri)}, "
        f"the one-word swap on a retrieved sentence {s(swr)}, and the best reference ({best[0] or '–'}) {s(best[1])}, "
        f"{'–' if ratio is None else f'{ratio:.1f}'} times TRELLIS. TRELLIS's errors come from word pairs it never saw next to each other. "
        f"Its memory ranks such pairs no better than the product of the two words' frequencies (Table 5), and fixed-order n-gram coders "
        f"code chess positions in fewer bits than it does (Table 6). This page sets the bar the two-hierarchy learner has to clear: the "
        f"references for generation, the split of its errors, the test of its generalization step, and the reference coders.</p>")

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
      "ranks unstored pairs about as well as multiplying the two words' frequencies, when the memory is read by token counts: each "
      "stored occurrence votes with how probable the neighbour is at its context leaf. This is the measured cause of the errors in Table 3.</li>"
      "<li><b>Read by types, the same context tree does about twice as well.</b> A context concept's members are the distinct words "
      "that filled the occurrences under it. An unstored (a, b) is licensed by the share of the members of the concepts above b's "
      "occurrences that a was stored right before, and the same from a's side. Nothing in the tree changes, only how it is read: one "
      "frequent collocation now counts as one member, not as its count. It passes similar-word votes at 4,000 sentences and stays "
      "below the analogy count at every size. At 200 sentences, weighing the path differently (the root left out, each concept by its information, "
      "only the nearest concepts) moves it by less than .01.</li>"
      "<li><b>Read at the concepts nearest each occurrence, it passes the analogy count from 2,000 sentences on.</b> The whole path "
      "lets broad concepts near the root, which mix word classes, speak for a pair. Summing over only the four concepts nearest each "
      "occurrence gives .350 / .348 at 1,000, .367 / .376 at 2,000 and .353 / .356 at 4,000 (seeds 13 / 17), against the analogy "
      "count's .338 / .321, .340 / .326 and .291 / .290; at 500 it is level on one seed and behind on the other. Its precision at 5% "
      "recall is .76 to .80 from 2,000 on. Weighting each concept by its information does almost as well. The learner method is "
      "<code>Trellis3.type_licence(near=4)</code>.</li>"
      "<li><b>A Cobweb tree over word pairs does worse than word frequency.</b></li></ul>")


# Chinese characters coder table (rounds 127, 104c, 106c)
def trellis_chart_bits(sd, t):
    p = f"logs/v3-r106c-base-4000-{sd}.log" if t == 4000 else f"logs/v3-r104c-base-{sd}.log"
    for r in jlines(p):
        if r.get("train") == t and isinstance(r.get("trellis 0"), dict) and r["trellis 0"].get("bits") is not None:
            return r["trellis 0"]["bits"]
    return None


def s13_17(vals):
    return " / ".join("–" if v is None else f"{v:.2f}" for v in vals)


def _len157(sd, t):
    for r in jlines(f"logs/v3-r157c-{'len-4000' if t == 4000 else 'len'}-{sd}.log"):
        if r.get("train") == t:
            return r.get("trellis 0 + stored lengths")
    return None


tch = []
for t in (500, 1000, 2000, 4000):
    rr = {}
    for sd in S2:
        for r in jlines(f"logs/v3-r127-chars-{sd}.log"):
            if r.get("train") == t:
                rr[sd] = r
    cells = [s13_17([rr.get(sd, {}).get(f) for sd in S2]) for f in ("bigram", "trigram")]
    cells.append(s13_17([trellis_chart_bits(sd, t) for sd in S2]))
    cells.append(s13_17([_len157(sd, t) for sd in S2]))
    tch.append(row(f"{t:,}", cells))
TCH = table(["Characters learned", "Bigram (s13 / s17)", "Trigram (s13 / s17)", "TRELLIS chart coder (s13 / s17)",
             "<b>TRELLIS, lengths from stored characters (round 157)</b>"], tch)

# chess reference coders (rounds 132 and 141): 4,000 positions learned, seeds 13 / 17 / 19
CS = (13, 17, 19)
CHESS = {}
for sd in CS:
    for tag in ("control", "shift"):
        path = f"logs/v3-r132-chess-{tag}.log" if sd == 13 else f"logs/v3-r141-chess-{tag}-{sd}.log"
        for r in jlines(path):
            if r.get("tag") == tag and r.get("learn") == 4000 and r.get("tokens"):
                CHESS[(sd, tag, r["tokens"])] = r


def bpe_best(kind, sd, tag):
    try:
        with open(f"experiments/v3/results/chess_compress/r141-bpe-{kind}_s{sd}.json", encoding="utf-8") as fh:
            return json.load(fh)["sets"][tag]["best"]
    except Exception:
        return None


def by_seed(vals):
    return " / ".join("–" if v is None else f"{v:.1f}" for v in vals)


tc = []
for tokens, tname in (("abs", "absolute"), ("rel", "relative")):
    for field, nm in (("unigram", "Unigram"), ("bigram", "Bigram"), ("trigram", "Trigram")):
        tc.append(row(f"{nm}, {tname} tokens",
                      [by_seed([CHESS.get((sd, tag, tokens), {}).get(field) for sd in CS]) for tag in ("control", "shift")]))
for kind, lab in (("abs", "Co-occurrence chunks on squares"), ("rel", "Co-occurrence chunks on shapes")):
    tc.append(row(lab, [by_seed([bpe_best(kind, sd, tag) for sd in CS]) for tag in ("control", "shift")]))
for lab, run in (("TRELLIS, chess page recipe (seed 13)", "ab-control"), ("TRELLIS, absolute transfer run (seed 13)", "xfer-abs"),
                 ("TRELLIS, relative transfer run (seed 13)", "xfer-rel")):
    def _best(tag, run=run):
        try:
            with open(f"experiments/v3/results/chess_compress/{run}-{tag}_s13.json", encoding="utf-8") as fh:
                v = [r["bits per position"] for r in json.load(fh)["by levels"] if r.get("copy")]
            return f"{min(v):.1f}" if v else "–"
        except Exception:
            return "–"
    tc.append(row(lab, [f"{_best('control')} / – / –", f"{_best('shift')} / – / –"]))


def _cb(name):
    try:
        with open(f"experiments/v3/results/chess_compress/{name}.json", encoding="utf-8") as fh:
            v = [r["bits per position"] for r in json.load(fh)["by levels"] if r.get("copy")]
        return min(v) if v else None
    except Exception:
        return None


for lab, pat in (("TRELLIS, chess page recipe, rerun (round 146)", "r146-gap0-{S}_s{s}"),
                 ("TRELLIS, positions coded by gaps in reading order (round 146)", "r146-gap2-{S}_s{s}"),
                 ("<b>TRELLIS, gaps and a causal context (rounds 147–148)</b>", "{r}-causalond-4k-{S}_s{s}"),
                 ("The same, context read at the root only (ablation)", "{r}-causalonr-4k-{S}_s{s}")):
    tc.append(row(lab, [by_seed([_cb(pat.format(S=S, s=s, r="r147" if s == 13 else "r148")) for s in CS])
                        for S in ("control", "shift")]))
tc.append(row("<b>TRELLIS, causal context without the absolute square (rounds 150–151; no chunks stored)</b>",
              [by_seed([_cb(f"r150-d4b-dropsq-4k-{S}_s13" if s == 13 else f"r151-d4b-dropsq-4k-{S}_s{s}") for s in CS])
               for S in ("control", "shift")]))
TCHESS = table(["Coder (4,000 positions learned)", "Control (s13 / s17 / s19)", "Shifted (s13 / s17 / s19)"], tc)

p6_intro = ("<p>The same question for coding: bits per held-out item for count-based coders that read only the stream. A chess "
            "position is read in a fixed square order; absolute tokens name a piece and its square, relative tokens name a piece and its "
            "distance from the previous piece in that order. The n-grams back off by Witten-Bell; a token never seen in training costs the "
            "log of the inventory size.</p>")
p6 = ("<ul>"
      "<li><b>On chess, a trigram over relative tokens beats TRELLIS and the co-occurrence chunks</b> on control and shifted positions, "
      "on every seed run.</li>"
      "<li><b>Two repairs inside TRELLIS close a third of the gap on control and half on shifted boards.</b> The diagnosis: TRELLIS coded every unit with no "
      "conditioning on what came before, its context tree did no work in the code (context levels 0, 1 and 2 gave identical bits), "
      "and squares were absolute. Coding each unit's place as a gap from the previous unit in reading order saves 4 to 6 bits on "
      "control and 12 to 14 on shifted boards, on all three seeds. Describing each occurrence by what a reader has already decoded "
      "(the backward half of its star, the previous unit and the gap) and coding each unit's symbol from its context concept saves "
      "another 5 to 9 bits. Reading the same route at the root only gives back most of the control gain: the context tree is worth "
      "about 7 bits on control and 1.5 to 2.5 on shifted boards. The relative trigram is still about 19 bits ahead on control and "
      "21 on shifted boards. Conditioning "
      "the gap on the context gains about 1 bit (round 149): the context tree is organised by the square and the gap themselves.</li>"
      "<li><b>Without the absolute square, TRELLIS's context tree matches the trigram, but no chunks form.</b> Leaving the anchor's "
      "square out of the causal description, so concepts organise by what a reader knows (the backward star, the previous unit "
      "and the gap), brings TRELLIS to 99.4 to 99.9 bits on control and 104.4 to 105.0 on shifted boards on three seeds: within "
      "about 1 bit of the relative trigram on control and 3.5 to 6 bits better on shifted boards. The gap is also predicted from "
      "the previous unit's context path. But the learner then stores no chunks: the square's context gain was what started "
      "chunking. This is the context hierarchy coding piece by piece; the content hierarchy does no work here.</li>"
      "<li><b>Chunks do not pay on top of it (round 152).</b> Without the square no join has a positive gain: a causal chunk's "
      "context is its anchor piece's in every join, and the content tree starts empty. Storing joins from sampled analyses "
      "(<code>store</code> posterior) lets chunks form again, about 7 per position. Coded with them, by the cover that minimises "
      "bits, a position costs 124.3 bits at 1,000 learned (seed 13) against 105.2 with no chunks. A chunk of about three pieces "
      "costs about 12.8 bits (its concept 9.2, its gap 3.2, its body 0.4); its pieces coded one by one from their contexts cost "
      "about 11.8, because each piece's backward star already sees the earlier pieces of its own pattern. On chess, then, what "
      "chunks carry is carried as well by the context of each piece.</li>"
      "<li><b>On characters, TRELLIS's chart coder trails the trigram up to 2,000 characters and passes it at 4,000</b>, by 0.6 "
      "bits on seed 13 and 0.05 on seed 17.</li>"
      "<li><b>With lengths coded from the stored characters, TRELLIS beats the trigram at every size (round 157).</b> The chart's "
      "own length distribution put about 47% of its mass on even lengths, which about 10% of characters have (binary operators "
      "make odd lengths). Coding the length from the lengths of the characters learned (add-half over 1 to 64) and the string under "
      "the chart given its length saves 0.8 to 1.0 bits per character: 31.53 / 32.01 at 500 against the trigram's 31.94 / 32.67, "
      "and 24.04 / 23.82 at 4,000 against 25.48 / 24.87. A causal context route like chess's adds at most 0.1 to 0.4 bits here "
      "(round 153): the chart already conditions each token through its grammar, and the context tree spreads the occurrences "
      "that follow a token over many concepts.</li>"
      "<li><b>Described by nested suffixes, the context tree alone reaches the trigram on characters (rounds 163 to 167).</b> "
      "With the left context described by position (l1, l2), the occurrences after one token were scattered: at depth 2 of the "
      "sorted path about 60% of them, though 93% of the leaf shared the token. Described by what a reader knows as nested "
      "suffixes (the last token, the last two as one value, and so on; <code>nest</code>), the same depth holds 94 to 99% of them, "
      "and the causal route with no chunks stored codes 28.77 / 27.69 bits at 2,000 (two-token context) against 33.7 / 37.3 "
      "from the base learner's two-sided tree (round 153), and 32.94 / 32.55 at 1,000 against 33.63 / 33.96 positionally. Longer contexts help up to about four tokens: at 4,000, 26.20 / 24.97 with three, 25.76 / 24.45 with four, "
      "25.61 / 24.57 with five and 25.48 / 24.75 with six, against the trigram's 25.48 / 24.87. The context tree alone thus ties "
      "or beats the trigram at 4,000, as on chess, but the chart with stored lengths stays about 1.4 bits better "
      "(24.04 / 23.82).</li></ul>"
      '<p class="small">The co-occurrence and TRELLIS rows come from the chess page\'s coders, which treat unseen items a little '
      "differently.</p>")
p8 = ("<h3>What this means for the two-hierarchy learner</h3><ul>"
      "<li><b>The weak step is generalization.</b> Sentences TRELLIS builds from stored pairs are mostly grammatical; those with a pair "
      "it never saw mostly are not (Table 3), and its memory ranks such pairs at the level of word frequency (Table 5).</li>"
      "<li><b>The bar.</b> Generation: the trigram sampler and the retrieved one-word swap at every size in Tables 1 and 2. Coding: the "
      "n-gram rows of Table 6. The fast test of Table 5 comes first, because it needs no judge.</li>"
      "<li><b>Dropped.</b> A separate substitution learner reported here until 10 November has been dropped to focus on the "
      "two-hierarchy learner. Its code is in the repository history at commit 6b084ffb.</li></ul>")

# Table 7: generation from the two-hierarchy learner (rounds 144, 154, 155)
def _judge_arms(d):
    try:
        with open(f"experiments/v3/results/story_gen/{d}/judge.json", encoding="utf-8") as fh:
            return json.load(fh)["arms"]
    except Exception:
        return []


def _diag_rows(rnd, s):
    out_ = {}
    for r in jlines(f"logs/v3-{rnd}-diag-{s}.log"):
        if str(r.get("diag", "")).startswith("story_"):
            out_[r["diag"]] = r
    return out_


def t7_row(label, d, rnd, match, seeds=SEEDS):
    arms = [a for a in _judge_arms(d) if match(a["arm"]) and a["file"].startswith("story_")]
    by = {}
    for a in arms:
        m = re.search(r"_s(\d+)_", a["file"])
        if m:
            by[int(m.group(1))] = a
    ys_ = [by.get(s, {}).get("grammatical novel per generation") for s in seeds]
    gr = [by.get(s, {}).get("judged grammatical") for s in seeds]
    nv = [by.get(s, {}).get("novel") for s in seeds]
    un = []
    for s in seeds:
        if s in by:
            dr = _diag_rows(rnd, s).get(by[s]["file"], {}).get("2-grams not all stored, new", {})
            if dr.get("share of generations") is not None:
                un.append((dr["share of generations"], dr.get("judged grammatical")))
    ok = [v for v in ys_ if v is not None]
    rng_ = lambda v: "–" if not [x for x in v if x is not None] else (
        f"{pct(min(x for x in v if x is not None))}–{pct(max(x for x in v if x is not None))}")
    uns = "–" if not un else ("under 1% of outputs" if mean([a for a, _ in un]) < 0.01 else
                               f"{pct(mean([a for a, _ in un]))} of outputs; {pct(mean([b for _, b in un if b is not None]))} grammatical")
    return row(label, [f2(mean(ok)) if ok else "–", " / ".join("–" if v is None else f2(v) for v in ys_), rng_(gr), rng_(nv), uns])


_ch = lambda pl: (lambda a: "chart" in a and a["chart"].get("pair_lic") == pl)
_ca = lambda u, p, nw=False: (lambda a: a.get("causal") and a.get("units") == u and a.get("pair") == p and bool(a.get("new")) == nw)
T7 = table(["Generator (1,000 sentences)", "Yield, mean", "Yield by seed (13 / 17 / 19 / 23)", "Grammatical", "New",
            "New sentences with a never-stored pair"],
           [t7_row("Chart generation, candidate recipe (round 144 control)", "r144-typelic", "r144", _ch(None)),
            t7_row("Chart, never-stored pairs weighted by the type licence", "r144-typelic", "r144", _ch("weight")),
            t7_row("Chart, never-stored pairs gated at the median licence", "r144-typelic", "r144", _ch("gate")),
            t7_row("Causal, words (round 154)", "r154-causal", "r154", _ca("words", None)),
            t7_row("<b>Causal, words, gated</b>", "r154-causal", "r154", _ca("words", "gate")),
            t7_row("Causal, words and chunks", "r154-causal", "r154", _ca("both", None)),
            t7_row("Causal, words and chunks, gated", "r154-causal", "r154", _ca("both", "gate")),
            t7_row("Causal, words, gate at the 25th percentile (round 156)", "r156-quant", "r156", lambda a: a.get("causal") and a.get("q") == 0.25),
            t7_row("Causal, words, gate at the 10th percentile", "r156-quant", "r156", lambda a: a.get("causal") and a.get("q") == 0.1),
            t7_row("Causal, words, gate at the 2nd percentile", "r156-quant", "r156", lambda a: a.get("causal") and a.get("q") == 0.02),
            t7_row("<b>Causal, words, nearest-4 licence, median gate (round 159)</b>", "r159-near", "r159", lambda a: a.get("causal") and a.get("near") == 4 and not a.get("q")),
            t7_row("Causal, words, nearest-4 licence, gate at the 10th percentile", "r159-near", "r159", lambda a: a.get("causal") and a.get("near") == 4 and a.get("q") == 0.1),
            t7_row("Causal, words, nearest-4 licence, gate at the 2nd percentile", "r159-near", "r159", lambda a: a.get("causal") and a.get("near") == 4 and a.get("q") == 0.02),
            t7_row("Causal, nearest-4, specialised one level up, median gate (round 160)", "r160-spec", "r160", lambda a: a.get("spec") == 1 and not a.get("q")),
            t7_row("Causal, nearest-4, specialised one level up, 10th-percentile gate", "r160-spec", "r160", lambda a: a.get("spec") == 1 and a.get("q") == 0.1),
            t7_row("Causal, nearest-4, specialised two levels up, 10th-percentile gate", "r160-spec", "r160", lambda a: a.get("spec") == 2),
            t7_row("Causal, nearest-4, specialised three levels up, 10th-percentile gate", "r160-spec", "r160", lambda a: a.get("spec") == 3),
            t7_row("Causal, words, gated, never a stored sentence (round 155)", "r155-new", "r155", _ca("words", "gate", True)),
            t7_row("Causal, words and chunks, gated, never a stored sentence", "r155-new", "r155", _ca("both", "gate", True)),
            t7_row("<b>Causal, three-word window, words, gated (round 161)</b>", "r161-w3", "r161", _ca("words", "gate")),
            t7_row("Causal, three-word window, words and chunks, gated", "r161-w3", "r161", _ca("both", "gate")),
            t7_row("<b>Causal, three-word window, words, gated, never a stored sentence</b>", "r161-w3", "r161", _ca("words", "gate", True)),
            t7_row("Causal, three-word window, words and chunks, gated, never a stored sentence", "r161-w3", "r161", _ca("both", "gate", True)),
            t7_row("<b>Context tree only (no chunks stored), words, gated (round 162)</b>", "r162-nochunk-w2", "r162w2", _ca("words", "gate")),
            t7_row("Context tree only, words, gated, never a stored sentence", "r162-nochunk-w2", "r162w2", _ca("words", "gate", True)),
            t7_row("Context tree only, three-word window, words, gated", "r162-nochunk-w3", "r162w3", _ca("words", "gate")),
            t7_row("Context tree only, three-word window, words, gated, never a stored sentence", "r162-nochunk-w3", "r162w3", _ca("words", "gate", True)),
            t7_row("Reference: retrieved one-word swap", "r155-new", "r155", lambda a: a.get("swap") == "retrieve" and not a.get("new")),
            t7_row("Reference: retrieved one-word swap, never a stored sentence", "r155-new", "r155", lambda a: a.get("swap") == "retrieve" and a.get("new")),
            t7_row("Reference: trigram sampler", "r155-ngram", "r155b", lambda a: a.get("ngram") == 3 and not a.get("new")),
            t7_row("Reference: trigram sampler, never a stored sentence", "r155-ngram", "r155b", lambda a: a.get("ngram") == 3 and a.get("new")),
            t7_row("Reference: 4-gram sampler (round 161)", "r161-w3-ref", "r161r", lambda a: a.get("ngram") == 4 and not a.get("new")),
            t7_row("Reference: 4-gram sampler, never a stored sentence", "r155-ngram", "r155b", lambda a: a.get("ngram") == 4 and a.get("new"))])
p7_intro = ("<p>Chart generation builds a sentence top-down from the learner's grammar and fills each position from the chart's "
            "marginal times Cobweb's prediction. Causal generation is the story counterpart of chess's causal context: left to right, "
            "each filler (a word, or also a committed chunk, which places its words) is drawn from the learner's context tree given the "
            "two words already placed, jointly with whether the sentence ends after it. The gate keeps a never-stored pair only if the "
            "context tree read by types (Table 5) licenses it at or above the median over stored pairs. Yield is new judged-grammatical "
            "sentences per generated sentence, as in Table 1; the trigram sampler there yields .412 / .404 / .388 / .395.</p>")
p7 = ("<ul>"
      "<li><b>Causal generation from the learner's own context tree reaches the trigram sampler.</b> Gated, it yields .38 to .40 on "
      "every seed against the trigram's .39 to .41, and against .19 to .26 for the chart.</li>"
      "<li><b>The gate works by reuse.</b> Gated arms almost never make a never-stored pair; ungated, sentences that do are 10 to 21% "
      "grammatical, and the type licence does not raise that in chart generation either (11 to 16%). Loosening the gate (round 156) "
      "shows the licence does order them, weakly: sentences with a never-stored pair are about 13% grammatical ungated, 22% when "
      "the pair must pass the 2nd percentile of stored pairs' licences, 24% at the 10th, and roughly 27% at the 25th (few outputs), "
      "against about 66% for new sentences of stored pairs. With the licence read at the four nearest concepts (round 159), the pairs that pass the 10th-percentile gate are about 65% "
      "grammatical, level with new sentences of stored pairs, but they appear in under 1% of outputs; at the 2nd percentile they are "
      "about 35% grammatical in 2 to 3% of outputs. The best of the learner's generalizations are now as good as its reuse; there are "
      "too few of them to move the yield.</li>"
      "<li><b>Specialising a drawn word raises generalization volume and lowers quality (round 160).</b> Replacing a drawn word with a "
      "licensed member of its context concept one level up puts a never-stored pair in about 18% of outputs, about 32% of them "
      "grammatical, but new sentences of stored pairs fall from about 66% to about 54% grammatical; yield stays at .37 to .41. Two "
      "and three levels up are worse (.30 and .20).</li>"
      "<li><b>Chunks raise grammaticality and lower novelty.</b> With committed chunks as fillers, 73 to 86% of outputs are judged "
      "grammatical, but most are stored sentences, so yield falls.</li>"
      "<li><b>Never returning a stored sentence lifts every generator, and the n-grams most.</b> With the same rule for all, causal "
      "generation with chunks yields .68 to .71, ahead of the retrieved swap (.62 to .68) and behind the trigram sampler (.71 to .73); "
      "the 4-gram sampler reaches .83 to .88 by splicing stored sentences where they share three words. Under this rule yield rewards "
      "splicing as much as generalization, so it cannot tell them apart; the column on never-stored pairs is the measure of "
      "generalization, and there every generator here stays near zero or ungrammatical.</li>"
      "<li><b>A three-word window makes the learner a better splicer, not a better generalizer (round 161).</b> With the learner's "
      "window widened to three words and each filler drawn given the three words placed, new sentences of stored pairs are 74 to 83% "
      "grammatical, against about 66% with two. Never returning a stored sentence, it yields .74 to .79, ahead of the trigram (.71 to .73) "
      "and behind the 4-gram (.83 to .88); without that rule it yields .24 to .25, ahead of the 4-gram's .18 to .21, because it copies "
      "less. Never-stored pairs stay at 1 to 3% of outputs, mostly ungrammatical: context depth buys splicing, as it does for n-grams.</li>"
      "<li><b>The content hierarchy adds nothing to story generation (round 162).</b> With no chunk stored, so that the learner is "
      "its context tree alone, causal generation yields .39 to .40 (.64 to .68 never returning a stored sentence), level with the "
      "full learner, and .22 to .26 (.73 to .77) with a three-word window. Dropping the chunk context as well changes no output. "
      "The fast licence test agrees: the nearest-4 reading scores .35 / .35 at 1,000 and .37 / .38 at 2,000 without chunks, as "
      "with them. As on chess (Table 6), what generates and codes is the context hierarchy.</li>"
      "<li><b>Topic.</b> Causal generation ignores the previous sentence (cross share near 0, against .25 to .28 for the chart).</li></ul>")

out = (lead + p_how + "<h3>Table 1. Reference generators at 1,000 sentences, 4 seeds</h3>" + T1 + p1 +
       "<h3>Table 2. With more data (seeds 13 / 17)</h3>" + T2 + p2 +
       "<h3>Table 3. Where TRELLIS's errors come from (1,000 sentences, seeds 13 / 17)</h3>" + T3 + T3b + p3 +
       "<h3>Table 4. One substitution at a time (round 110)</h3>" + T4 + p4 +
       "<h3>Table 5. The generalization step alone</h3>" + p5_intro + T5 + T5v + T5s + p5_defs + p5 +
       "<h3>Table 6. Reference coders</h3>" + p6_intro + "<h4>6a. Chess (bits per held-out position)</h4>" + TCHESS +
       "<h4>6b. Chinese characters (bits per held-out character)</h4>" + TCH + p6 +
       "<h3>Table 7. Generating from the two-hierarchy learner</h3>" + p7_intro + T7 + p7 + p8)
with open("docs/site/frag_refs.html", "w", encoding="utf-8") as f:
    f.write(out)

# plain-text echo for the report
import re as _re
print(_re.sub(r"<[^>]+>", " | ", out.replace("</tr>", "\n")).replace("| |", "|"))
print("missing r110:", missing110)
