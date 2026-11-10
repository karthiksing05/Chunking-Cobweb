"""Appendix A18: reads the ablation logs and writes frag_rationale.html.

Numbers come from the logs; only the older-recipe row is fixed, from rounds 91 and 99 to
102 (its logs used other formats). Run from the repo root on a compute node.
"""
import json
import os
import re

SEEDS = (13, 17, 19, 23)


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


def story(prefix, t, arm=0):
    """Judged yield, grammatical, new; cross share from the story logs."""
    rows = [r for r in jlines(f"logs/{prefix}-judge-{t}.log") if r.get("file") and f"arm{arm}" in r["file"]]
    rows.sort(key=lambda r: r["file"])
    cross = []
    for s in SEEDS:
        recs = [r for r in jlines(f"logs/{prefix}-{t}-story-{s}.log") if "cross share" in r]
        if len(recs) > arm:
            cross.append(recs[arm]["cross share"])
    return {"y": [r["grammatical novel per generation"] for r in rows],
            "g": mean([r["judged grammatical"] for r in rows]),
            "n": mean([r["novel"] for r in rows]),
            "cross": mean(cross)}


def chars(prefix, t):
    out = {}
    for s in (13, 17):
        for r in jlines(f"logs/{prefix}c-{t}-{s}.log"):
            if "trellis 0" in r:
                out[(r["train"], s)] = (r["trellis 0"]["bits"], r["learn s"])
    return out


def toy(prefix, t, kind):
    out = {}
    for s in (13, 17):
        for r in jlines(f"logs/{prefix}t-{t}-{kind}-{s}.log"):
            if r.get("experiences") == 320:
                out[(r["cond"], s)] = (r["grammatical"], r["grammatical and new"])
    return out


def f2(x):
    return "–" if x is None else f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.2f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


# the arms: label, log prefix, tag
ARMS = [
    ("<b>Candidate base</b>", "v3-r104", "base"),
    ("<code>word_cc</code> back on", "v3-r104", "word-cc-on"),
    ("Type names back on", "v3-r103", "wc"),
    ("Neighbour chunks back on, with type names", "v3-r103", "wc-nb"),
    ("<code>exact_alone</code> off", "v3-r104", "exact-off"),
    ("<code>exact_alone</code> on both trees", "v3-r104", "exact-both"),
    ("<code>exact_alone</code> on the context tree only", "v3-r106", "exact-ctx"),
    ("One shared tree", "v3-r104", "shared"),
    ("One joint tree", "v3-r104", "joint"),
    ("Content not donated to context", "v3-r104", "donate-ctx"),
    ("Parts' content not donated", "v3-r104", "donate-cnt"),
    ("Names by depth, no bands", "v3-r104", "naming-depth"),
    ("Store the best analysis only", "v3-r104", "store-best"),
    ("Restructure on every change", "v3-r104", "restructure"),
    ("No repair", "v3-r104", "repair"),
    ("Window 1", "v3-r104", "window1"),
    ("No story facet", "v3-r104", "no-story-facet"),
]

OLD_STORY = "<tr><td>Older recipe (A17)</td><td class=\"num\">.186</td><td class=\"num\">.198 / .185 / .150 / .209</td><td class=\"num\">59–60%</td><td class=\"num\">53–57%</td><td class=\"num\">–</td><td class=\"num\">27.79 / 27.46</td><td class=\"num\">–</td></tr>"
OLD_TOY = "<tr><td>Older recipe (A17)</td><td class=\"num\">.00 / .00</td><td class=\"num\">.02 / .01</td><td class=\"num\">–</td><td class=\"num\">.45 / .51; .31 / .28</td><td class=\"num\">.50 / .53; .34 / .36</td></tr>"

t1, t2 = [], []
base = None
for label, prefix, t in ARMS:
    st, ch = story(prefix, t), chars(prefix, t)
    y = mean(st["y"])
    if t == "base":
        base = y
    bits = " / ".join(f"{ch[(2000, s)][0]:.2f}" if (2000, s) in ch else "–" for s in (13, 17))
    mins = mean([ch[(2000, s)][1] / 60 for s in (13, 17) if (2000, s) in ch])
    t1.append(f"<tr><td>{label}</td><td class=\"num\">{f2(y)}</td><td class=\"num\">{' / '.join(f2(v) for v in st['y'])}</td>"
              f"<td class=\"num\">{pct(st['g'])}</td><td class=\"num\">{pct(st['n'])}</td><td class=\"num\">{f2(st['cross'])}</td>"
              f"<td class=\"num\">{bits}</td><td class=\"num\">{'–' if mins is None else f'{mins:.0f}'}</td></tr>")
    if t == "no-story-facet":
        continue
    ex, io = toy(prefix, t, "exp"), toy(prefix, t, "io")

    def cell(d, c, both=False):
        if (c, 13) not in d or (c, 17) not in d:
            return "–"
        a, b = d[(c, 13)], d[(c, 17)]
        if both:
            return f"{f2(a[0])} / {f2(b[0])}; {f2(a[1])} / {f2(b[1])}"
        return f"{f2(a[1])} / {f2(b[1])}"
    t2.append(f"<tr><td>{label}</td><td class=\"num\">{cell(ex, 'med')}</td><td class=\"num\">{cell(ex, 'large')}</td><td class=\"num\">{cell(ex, 'term_low')}</td>"
              f"<td class=\"num\">{cell(io, 'med', True)}</td><td class=\"num\">{cell(io, 'large', True)}</td></tr>")

# generation-only arms on one memory
GEN = ["Base", "Fill from the chart only", "Fill from Cobweb only", "λ 1", "λ 16", "Content cut 0 bits", "Content cut 8 bits",
       "Words cut 8 bits", "No topic", "No sentence-order following"]
rows = [r for r in jlines("logs/v3-r104-judge-gen.log") if r.get("file")]
t3 = []
cross = {s: [r["cross share"] for r in jlines(f"logs/v3-r104-gen-story-{s}.log") if "cross share" in r] for s in (13, 17)}
for i, name in enumerate(GEN):
    by = {int(re.search(r"_s(\d+)_", r["file"]).group(1)): r for r in rows if f"arm{i}." in r["file"] or r["file"].endswith(f"arm{i}.jsonl")}
    y = [by[s]["grammatical novel per generation"] for s in (13, 17) if s in by]
    g = mean([by[s]["judged grammatical"] for s in (13, 17) if s in by])
    n = mean([by[s]["novel"] for s in (13, 17) if s in by])
    c = mean([cross[s][i] for s in (13, 17) if len(cross[s]) > i])
    t3.append(f"<tr><td>{name}</td><td class=\"num\">{' / '.join(f2(v) for v in y)}</td><td class=\"num\">{pct(g)}</td><td class=\"num\">{pct(n)}</td><td class=\"num\">{f2(c)}</td></tr>")
wc = story("v3-r103", "wc", 1)
t3.append(f"<tr><td>Expansion instead of inside-outside (type names on, 4 seeds)</td><td class=\"num\">{' / '.join(f2(v) for v in wc['y'])}</td><td class=\"num\">{pct(wc['g'])}</td><td class=\"num\">{pct(wc['n'])}</td><td class=\"num\">{f2(wc['cross'])}</td></tr>")

# chess: one change from the chess recipe with exact_alone on (the arms' control)
CHESS = [("Control: chess recipe, <code>exact_alone</code> on", "v3-ab-chess-control"), ("<code>exact_alone</code> off", "v3-ab-chess-exact-off"),
         ("<code>exact_alone</code> on content only", "v3-ab-chess-exact-content"), ("Type names off", "v3-ab-chess-type-names"),
         ("Names by depth", "v3-ab-chess-naming-depth"), ("Restructure on every change", "v3-ab-chess-restructure"), ("No refresh", "v3-ab-chess-no-refresh"),
         ("Store every sampled analysis (posterior)", "v3-ab-chess-store-posterior"),
         ("Type names off, <code>exact_alone</code> off, no refresh", "v3-r104-chess-tn-noex-r0"),
         ("The same, posterior storage", "v3-r104-chess-tn-noex-post")]
t4 = []
for name, stem in CHESS:
    cells = {}
    for sc in ("control", "shift"):
        rs = [r for r in jlines(f"logs/{stem}-{sc}.log") if "tag" in r]
        if rs:
            r = rs[-1]
            cells[sc] = (min(x["bits per position"] for x in r["by levels"] if x.get("copy") is True), r["memory bits (distinct chunks as piece-square lists)"], r["learn s"])
    a, b = cells.get("control"), cells.get("shift")
    t4.append(f"<tr><td>{name}</td><td class=\"num\">{a[0] if a else '–'}</td><td class=\"num\">{b[0] if b else '–'}</td>"
              f"<td class=\"num\">{f'{a[1] / 1000:.0f}k' if a else '–'}</td><td class=\"num\">{f'{a[2] / 60:.0f}' if a else '–'}</td></tr>")

# audit: records, posterior, purity
AUD = [("Candidate base", "r104-base", "two"), ("<code>exact_alone</code> off", "r104-exact-off", "two"), ("One shared tree, on the base", "r104-base", "shared"),
       ("One joint tree, on the base", "r104-base", "joint")]
t5 = []
for name, stem, a in AUD:
    v = []
    for s in (13, 17):
        with open(f"experiments/v3/results/abl_audit/{stem}-{s}.json", encoding="utf-8") as f:
            v.append(json.load(f)["arms"][a])
    fmt = lambda fn: " / ".join(fn(x) for x in v)
    t5.append(f"<tr><td>{name}</td><td class=\"num\">{fmt(lambda x: f'{x['records']:,}')}</td><td class=\"num\">{fmt(lambda x: f2(x['mean chosen posterior']))}</td>"
              f"<td class=\"num\">{fmt(lambda x: f2(x['word purity']['12']['tag purity']))}</td><td class=\"num\">{fmt(lambda x: f2(x['chunk purity']['12']['tag-pattern purity']))}</td></tr>")

# slot test
SL = [("<code>word_cc</code> off, <code>exact_alone</code> on content", "base"), ("<code>word_cc</code> off, <code>exact_alone</code> off", "noex"),
      ("<code>word_cc</code> on, <code>exact_alone</code> on content", "wcon"), ("<code>word_cc</code> on, <code>exact_alone</code> off", "wcon-noex")]
t6 = []
for name, t in SL:
    r = {(x["cond"], s): x for s in (13, 17) for x in jlines(f"logs/v3-slot-{t}-{s}.log")}
    c = lambda cond, k: " / ".join(str(r[(cond, s)]["up 1"][k]) for s in (13, 17))
    t6.append(f"<tr><td>{name}</td><td class=\"num\">{c('med', 'mean distinct words')}</td><td class=\"num\">{c('med', 'share under a one-word concept')}</td>"
              f"<td class=\"num\">{c('large', 'mean distinct words')}</td><td class=\"num\">{c('large', 'share under a one-word concept')}</td></tr>")

# parsing against gold on the toy grammars
PG = ("small", "med", "large", "term_low", "term_med")
PA = [("Older recipe", "older"), ("<code>word_cc</code> on, type names on, no neighbour chunks", "cnt"), ("<code>word_cc</code> off, type names on", "wc"),
      ("<code>word_cc</code> on, type names off", "tn-wcon"), ("<b>Candidate base</b>", "base"), ("<code>exact_alone</code> off", "exact-off"),
      ("<code>exact_alone</code> on the context tree only", "exact-ctx")]
t7 = []
fixed = None
for name, t in PA:
    r = {(x["cond"], x["seed"]): x for s in (13, 17) for x in jlines(f"logs/v3-shape-{t}-{s}.log")}
    fixed = r
    cells = "".join(f"<td class=\"num\">{f2(r[(c, 13)]['omission'])} / {f2(r[(c, 17)]['omission'])}</td>" for c in PG)
    m = mean([r[(c, s)]["omission"] for c in PG for s in (13, 17)])
    t7.append(f"<tr><td>{name}</td>{cells}<td class=\"num\">{f2(m)}</td></tr>")
cells = "".join(f"<td class=\"num\">{f2(fixed[(c, 13)]['right-branching omission'])} / {f2(fixed[(c, 17)]['right-branching omission'])}</td>" for c in PG)
t7.append(f"<tr><td>A fixed right-branching parser</td>{cells}<td class=\"num\">{f2(mean([fixed[(c, s)]['right-branching omission'] for c in PG for s in (13, 17)]))}</td></tr>")

# the candidate with more or less data
t8 = []
for n in (500, 1000, 2000, 4000):
    t = "base" if n == 1000 else f"n{n}"
    pre = "v3-r104" if n == 1000 else "v3-r106"
    rows = {int(re.search(r"_s(\d+)_", r["file"]).group(1)): r for r in jlines(f"logs/{pre}-judge-{t}.log") if r.get("file") and "arm0" in r["file"]}
    def c(k):
        return " / ".join(f2(rows[s][k]) if s in rows else "–" for s in (13, 17))
    ch = chars("v3-r104", "base")
    ch.update(chars("v3-r106", "base-4000"))
    bits = " / ".join(f"{ch[(n, s)][0]:.2f}" if (n, s) in ch else "–" for s in (13, 17))
    tri = {}
    for s in (13, 17):
        for pth in (f"logs/v3-r104c-base-{s}.log", f"logs/v3-r106c-base-4000-{s}.log"):
            for r in jlines(pth):
                if "trigram" in r:
                    tri[(r["train"], s)] = r["trigram"]
    tb = " / ".join(f"{tri[(n, s)]:.2f}" if (n, s) in tri else "–" for s in (13, 17))
    t8.append(f"<tr><td>{n:,}</td><td class=\"num\">{c('grammatical novel per generation')}</td><td class=\"num\">{c('judged grammatical')}</td><td class=\"num\">{c('novel')}</td><td class=\"num\">{bits}</td><td class=\"num\">{tb}</td></tr>")

HTML = f"""
<p><b>Verdict so far.</b> Removing parts one at a time produced a simpler candidate recipe that generates and compresses better than the recipe the main pages describe, and learns faster. Four decisions held up with a measured reason, four are weakly supported, and two remain real conflicts between generation and compression. The candidate has not been adopted, and the main pages still describe the older recipe.</p>
<p><b>The candidate recipe.</b> Two trees; posterior storage; names in information bands; sampled restructuring; repair; window 2; <code>exact_alone</code> on the content tree; the story facet on stories. It drops <code>word_cc</code>, type names and neighbour chunks. With type names off, <code>name_level</code>, <code>name_mix_k</code>, <code>type_k</code>, <code>type_key</code> and <code>word_content</code> no longer do anything: the run without them reproduces the story yields, character bits and toy results of the run with them, to the last digit.</p>
<h2>How to read this page</h2>
<ul>
  <li>Each row changes one thing from the candidate base. Everything else stays fixed.</li>
  <li><b>Yield</b> is judged-grammatical new sentences per generation, 1,000 sentences learned, seeds 13 / 17 / 19 / 23.</li>
  <li><b>Cross share</b> measures whether a sentence shares the previous sentence's topic: mean PMI of content words across the pair, as a share of real text's over random words of the same tag. It varies widely between seeds (.03 to .40 on the base).</li>
  <li>Characters are bits per held-out character at 2,000 learned, seeds 13 / 17. Trigram gets 26.43 / 26.02. Toy grammars use 320 experiences and are scored by the grammar, which never teaches.</li>
  <li>Differences in yield under about .03 are within the spread between seeds, so they are not treated as effects.</li>
</ul>
<h2>Stories and characters</h2>
<div class="tablewrap"><table>
  <thead><tr><th>Arm</th><th class="num">Yield, mean</th><th class="num">Yield by seed</th><th class="num">Grammatical</th><th class="num">New</th><th class="num">Cross share</th><th class="num">Characters</th><th class="num">Learning, min</th></tr></thead>
  <tbody>{OLD_STORY}{''.join(t1)}</tbody>
</table></div>
<h2>The candidate with more data</h2>
<div class="tablewrap"><table>
  <thead><tr><th>Learned, s13 / s17</th><th class="num">Yield</th><th class="num">Grammatical</th><th class="num">New</th><th class="num">Characters, bits</th><th class="num">Trigram, bits</th></tr></thead>
  <tbody>{''.join(t8)}</tbody>
</table></div>
<ul>
  <li><b>Compression improves with data faster than trigram does.</b> At 4,000 the candidate beats trigram on both seeds (24.89 against 25.48, 24.82 against 24.87); at 2,000 it was 0.3–0.5 bits behind.</li>
  <li><b>Generation does not improve with data.</b> Yield is .26 / .25 at 500 and .21 / .17 at 2,000. Grammaticality stays near 60% throughout; the share of new sentences moves little. So the remaining grammar errors are not a shortage of data: they come from how generation reads memory. We have not yet measured why yield falls as memory grows.</li>
</ul>
<h2>Toy grammars</h2>
<div class="tablewrap"><table>
  <thead><tr><th>Arm</th><th class="num">Expansion, new: <code>med</code></th><th class="num"><code>large</code></th><th class="num"><code>term_low</code></th><th class="num">Inside-outside, grammatical; new: <code>med</code></th><th class="num"><code>large</code></th></tr></thead>
  <tbody>{OLD_TOY}{''.join(t2)}</tbody>
</table></div>
<p class="small">Expansion "new" is grammatical and new as a share of everything generated, s13 / s17. A dash means the run did not finish or was not run.</p>
<h2>Memory and parsing</h2>
<div class="tablewrap"><table>
  <thead><tr><th>1,000 sentences, s13 / s17</th><th class="num">Records</th><th class="num">Mean posterior of the chosen analysis</th><th class="num">Word tag purity, 12 bits</th><th class="num">Chunk tag-pattern purity, 12 bits</th></tr></thead>
  <tbody>{''.join(t5)}</tbody>
</table></div>
<p class="small">The older recipe's arm with <code>word_cc</code> on and type names on had word purity about .87 and chunk purity about .91.</p>

<h2>Parsing against gold trees</h2>
<div class="tablewrap"><table>
  <thead><tr><th>Omission at 320 experiences, s13 / s17</th><th class="num"><code>small</code></th><th class="num"><code>med</code></th><th class="num"><code>large</code></th><th class="num"><code>term_low</code></th><th class="num"><code>term_med</code></th><th class="num">Mean</th></tr></thead>
  <tbody>{''.join(t7)}</tbody>
</table></div>
<ul>
  <li><b>The older recipe parses worse than a fixed right-branching parser</b>, and it can collapse: on <code>med</code>, seed 13, it reads almost every sentence as one left-branching chain. The example on the Parsing explorer is one of these. Neighbour chunks cause it: turning them off alone brings <code>med</code> to .31 / .25 and <code>large</code> to .22 / .30 (60 test sentences), and <code>exact_alone</code> makes little further difference. Keeping them with <code>exact_alone</code> on content only gives .40 / .29 and .31 / .40.</li>
  <li><b>The candidate beats the fixed parser on average</b> (.28 against .33), by a modest margin: clearly on <code>large</code>, about level on the rest. No recipe here parses well.</li>
  <li><b><code>small</code> looks like a loss for the candidate, but the evidence for that is thin.</b> Every sentence has the shape Det N V Det N, and the only bracket missed is whether the verb joins the subject or the object. Word sequences give little to separate these in this grammar, and earlier recipes flip between .00 and .25 on the same data. So we read this as an ambiguity that recipes resolve arbitrarily rather than a capability <code>word_cc</code> provides. We have not tested that reading.</li>
</ul>

<h2>Parts removed</h2>
<h3><code>word_cc</code>: the word's identity in its context description</h3>
<p><b>Reason.</b> A chunk's content concept is donated into its context, so a word's own content, the word itself, was donated the same way.</p>
<p><b>What removal did.</b> Expansion's grammatical new output on the toy grammars rose from about .02 to about .8 on the older recipe, and yield on stories rose with novelty. On this base grammaticality falls by about 5 points on stories, cross share falls from .34 to .24, and characters lose 0.2–0.3 bits. Word purity falls from .87 to .77.</p>
<p><b>Why.</b> A generation slot draws from the stored members of the concept just above a word's leaf. With the word's identity in its description, that concept mostly holds only that word:</p>
<div class="tablewrap"><table>
  <thead><tr><th>Concept above a word's leaf, 320 experiences, s13 / s17</th><th class="num"><code>med</code>: distinct words</th><th class="num">Share of occurrences alone with their word</th><th class="num"><code>large</code>: distinct words</th><th class="num">Share alone</th></tr></thead>
  <tbody>{''.join(t6)}</tbody>
</table></div>
<p><b>Verdict.</b> Removed. The reason was an analogy, and it caused the copying. The losses are real: word grouping is coarser at 12 bits, and sentences stay on topic less. Parsing against gold trees is about the same overall (see the table above).</p>
<h3>Type names</h3>
<p><b>Reason.</b> Naming a part by where all instances of its type have stood lets substitutable types share names wherever they occur.</p>
<p><b>What removal did.</b> Yield rose from .190 to .224, on all four seeds, learning became about three times faster, and five parameters stopped mattering. Characters lose 0.3 bits on seed 13 at 2,000 and are unchanged on seed 17.</p>
<p><b>Verdict.</b> Removed. The stated reason does hold on chunk grouping: chunk purity falls from .91 to about .76 without type names. Better grouping did not turn into better generation or compression at this scale, and it costs most of the learning time. That may change on larger data, which we have not tested.</p>
<h3>Neighbour chunks</h3>
<p><b>Reason.</b> A context also names the units beside the element at its own level, so generation can read chunks as well as words.</p>
<p><b>What removal did.</b> On the older recipe, removal cost story yield (.186 to .163). Once <code>word_cc</code> is off, the yield difference disappears (.190 with or without, both with type names on) and characters are 0.16–0.35 bits worse with them.</p>
<p><b>Verdict.</b> Removed. Their earlier benefit on stories depended on <code>word_cc</code>.</p>

<h2>Parts with a measured reason</h2>
<h3>Names in information bands</h3>
<p><b>Reason.</b> Naming a path by the first concept reaching each information level gives names of matched resolution, however deep the tree has grown.</p>
<p><b>Verdict.</b> Holds, strongly. Naming by depth costs about 4 to 4.6 bits on characters, drops expansion on <code>large</code> by more than half and lowers yield to .186. Chess is indifferent. Depth names do raise inside-outside on <code>med</code> (51–56% grammatical against 46–47%) and parse with a higher posterior, so a confident parse alone is a poor guide here.</p>
<h3>Posterior storage</h3>
<p><b>Reason.</b> Storing each analysis in proportion to its probability keeps uncertainty that the best analysis throws away.</p>
<p><b>Verdict.</b> Holds for compression: storing only the best analysis costs 2.5–3.7 bits on characters. It has a cost. Best-only learns about eight times faster, stores about a quarter of the records, and gets inside-outside on toy grammars to 54–63% grammatical against 46–49%. Chess uses best-only: posterior storage there costs 25–38% more memory, up to seven times the learning time, and 1.4–3 bits on shifted boards.</p>
<h3>Sentence-order following and both fill sources</h3>
<p><b>Reason.</b> Reading a word from both the chart and the Cobweb window, in order, should combine structure and local fit.</p>
<p><b>Verdict.</b> Holds, on two seeds. See the generation table below: turning off following loses about .07 in yield, and either fill source alone loses .02 to .10. Content identity also matters: cutting content at 0 bits halves the yield.</p>
<h3>Story facet and the topic weight</h3>
<p><b>Reason.</b> A sentence should belong with the one before it.</p>
<p><b>Verdict.</b> Holds, at a price. Removal gives the highest yield of any arm (.287) because grammaticality rises to about 75%, but cross share falls from about .24 to about .10. Yield alone would have called this part harmful. Whether the topic is worth .06 in yield is a judgement, and we keep it because the goal includes generating sentences that belong together.</p>

<h2>Parts with weak support</h2>
<ul>
  <li><b>Both directions of donation.</b> Removing either lowers yield by .012 to .018, within seed spread. Removing content-to-context lowers expansion on <code>term_med</code>; removing parts' content costs about 0.1 bits on characters.</li>
  <li><b>Repair.</b> Without it, yield is .206 against .224, characters are about the same, toy output is if anything slightly better, and learning is about a third faster. On the older recipe it raised chunk purity from .81 to .91. It is a candidate for deletion if that purity turns out not to matter.</li>
  <li><b>Sampled restructuring.</b> Restructuring on every change gives .210, slightly better characters (26.58 / 26.40) and almost no extra time on characters at 2,000. Sampling saved up to two-thirds more on the slower recipe. On this base the choice is close.</li>
  <li><b>Two trees.</b> On the base, shared gives .206 and joint .215, within seed spread of .224, and shared is 0.1–0.2 bits better on characters. Joint is worse on characters and on inside-outside. Two trees still group chunks best (.76, against .63 shared and .51 joint), although shared parses with a higher posterior. The earlier verdict that shared is rejected (A1) came from the older recipe and no longer stands as firmly. Chess has not been tested with one tree.</li>
</ul>

<h2>Conflicts we have not resolved</h2>
<h3><code>exact_alone</code></h3>
<p><b>Reason.</b> Cobweb scored a new leaf as if a lone instance predicted all its values with certainty. For a bag-valued description that files nearly everything as a new leaf, giving content concepts with thousands of leaf children.</p>
<p><b>What removal does.</b> It improves characters to 26.09 / 26.05 at 2,000. Against trigram that is 0.34 bits better on seed 13 and 0.03 worse on seed 17, with the same pattern at 1,000 and a small win on both seeds at 500. It also improves chess on shifted boards from 153.6 to 147.2, with control unchanged. It lowers yield to .194 on all four seeds, and collapses chunk purity from .76 to .12. Learning takes about 2.4 times as long on characters and three times on chess.</p>
<p><b>On the context tree only.</b> Turning it on for context and off for content was meant to keep grouping while letting content stay flat. It does neither: yield falls to .174 on all four seeds, below every other arm; characters are mixed (27.04 / 26.33 at 2,000, against 26.69 / 26.52); learning takes about four times as long; expansion on toy grammars is weaker on <code>med</code> and <code>large</code> and level on <code>term_low</code>; and parsing omission rises to .31. So the setting must be on for the content tree, and the context tree does not need it.</p>
<p><b>Verdict.</b> The reason holds: the flat tree is real and it damages grouping. The surprise is that a flat content tree compresses better, plausibly because exact chunks behave more like a large memory of seen sequences, though we have not tested that. The same setting cannot currently serve generation and compression, and this is the largest obstacle to a single recipe.</p>
<h3>Window size</h3>
<p><b>Reason.</b> Two tokens each side give the context tree enough to separate uses of a word.</p>
<p><b>What removal does.</b> Window 1 gives the best yield that keeps the topic (.266, cross share about .31). Its characters are about as bad as names by depth, 3.6–4.3 bits worse. Expansion on <code>term_low</code> falls by half.</p>
<p><b>Verdict.</b> Unresolved. Generation prefers a narrower context than compression does. A window of two tokens with generation reading a coarser cut does not settle it: cutting words at 8 bits gives .247 and loses the topic.</p>

<h2>Generation settings, one memory</h2>
<div class="tablewrap"><table>
  <thead><tr><th>Changed alone, s13 / s17</th><th class="num">Yield</th><th class="num">Grammatical</th><th class="num">New</th><th class="num">Cross share</th></tr></thead>
  <tbody>{''.join(t3)}</tbody>
</table></div>
<p class="small">λ between 1 and 16 changes yield by at most .04, and only on seed 13. Inside-outside clearly beats expansion on stories, while on toy grammars expansion is far more grammatical. Inside-outside at 46–49% on <code>med</code> and <code>large</code> remains unexplained.</p>

<h2>Chess</h2>
<div class="tablewrap"><table>
  <thead><tr><th>Seed 13, 4,000 learned, offsets only, best level</th><th class="num">Control, bits</th><th class="num">Shift, bits</th><th class="num">Memory, bits</th><th class="num">Learning, min</th></tr></thead>
  <tbody>{''.join(t4)}</tbody>
</table></div>
<p>Only <code>exact_alone</code> moves shifted boards by more than about 2 bits; only the storage rule moves memory much. Chess gives little evidence for or against the other decisions. This is one seed, so differences of a bit or two are not meaningful.</p>
<p><b>A correction.</b> The 147.2 reported on the chess page came from a run with <code>exact_alone</code> off, while the recipe table says it is on. With it on, shifted boards cost 153.6. Today's code reproduces both numbers.</p>
<h2>What this does not settle</h2>
<ul>
  <li>Most generation arms and all chess arms have one or two seeds, and only the base was run with more data.</li>
  <li>Purity and posterior say how memory is organised. On stories we have no gold parse, so we cannot say what their fall costs in parsing quality there. Toy parsing was measured for six recipes, not for every arm above.</li>
  <li>The candidate has not been run on bAbI, nor with the chart settings re-tuned for it.</li>
</ul>
"""
with open("docs/site/frag_rationale.html", "w", encoding="utf-8") as f:
    f.write(HTML)
print("base yield", base, "bytes", len(HTML))
print("\n".join(t1 + t2 + t3 + t4 + t5 + t6))
