"""Round 201: dependency-distance diagnostic for TRELLIS story generation.

Hypothesis (missing state): the window learner generates causally from a fixed W-word window, so its
ungrammatical sentences should concentrate where a syntactic dependency spans more than W words, and
LSTM / GPT samples should not show the same jump.

Inputs (all existing): story_gen/<dir>/story_s<seed>_n<n>_arm<k>.jsonl (header line + one row per
generation: prompt, gen, novel) and the matching .scores.json (Qwen2.5-3B P(yes grammatical) per row).
A sentence is grammatical when P(yes) > 0.5 (same rule as r192_summarize.py). Only rows with novel=true
(generated sentence not a training sentence) are used.

Parser: spaCy en_core_web_sm (env `cobweb`; the `trellis` env has no spaCy).
Arc length = |token index - head index| over spaCy tokens, root and punctuation arcs excluded.

Run on a compute node:  python experiments/v3/diag/r201_distance.py <n> <out.md>   (n = 1000 or 4000)
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

SG = "/coc/flash10/ksingara3/Chunking-Cobweb/experiments/v3/results/story_gen"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{SG}/r201_distance_n{N}.md"
PREFIX = ("r184", "r188", "r190", "r191", "r192")  # r192a is the "both: pairs" variant; excluded
REFW = (3, 4, 5)
RNG = np.random.default_rng(0)
NBOOT = 1000

import spacy  # noqa: E402

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# ------------------------------------------------------------------ load
rows = []  # dicts
for d in sorted(glob.glob(f"{SG}/*-n{N}-*")):
    base = os.path.basename(d)
    if not base.startswith(PREFIX) or not os.path.isdir(d):
        continue
    m = re.search(r"-w(\d)$", base)
    for f in sorted(glob.glob(f"{d}/story_s*_n{N}_arm*.jsonl")):
        if f.endswith("_refs.jsonl"):
            continue
        sc = f[:-6] + ".scores.json"
        if not os.path.exists(sc):
            continue
        with open(f) as fh:
            lines = [json.loads(x) for x in fh]
        arm = lines[0]["arm"]
        body = lines[1:]
        scores = json.load(open(sc))
        if len(scores) != len(body):
            continue
        if m and arm.get("units") == "words" and "lm" not in arm and "ngram" not in arm:
            fam, W = "TRELLIS", int(m.group(1))
        elif arm.get("lm") in ("gpt", "lstm"):
            fam, W = arm["lm"].upper(), None
        else:
            continue
        seed = int(re.search(r"_s(\d+)_", f).group(1))
        temp = arm.get("temp", 1.0)
        for r, s in zip(body, scores):
            if not r.get("novel"):
                continue
            rows.append(dict(fam=fam, W=W, seed=seed, temp=temp, dir=base, gen=r["gen"], prompt=r["prompt"],
                             gram=float(s > 0.5), p=float(s),
                             arm=f"{fam}{'-w%d' % W if W else ''}@{temp}/{base.split('-')[0]}"))

print("loaded", len(rows), "novel sentences", flush=True)

# ------------------------------------------------------------------ parse (unique gens)
uniq = sorted({r["gen"] for r in rows})
parsed = {}
for doc, g in zip(nlp.pipe(uniq, batch_size=256), uniq):
    toks = [(t.i, t.head.i, t.dep_, t.tag_, t.pos_, t.text.lower(), t.is_punct) for t in doc]
    arcs = [(abs(i - h), dep, i, h) for i, h, dep, tag, pos, txt, pu in toks if i != h and not pu]
    sub_dis = []
    for i, h, dep, tag, pos, txt, pu in toks:
        if dep in ("nsubj", "nsubjpass") and i != h:
            ttag = toks[h][3]
            vtxt = toks[h][5]
            # aux "was/were/is/are/has/have" attached as verb or aux; use head verb tag when VBZ/VBP, else look at aux children
            sg = pl = None
            if tag in ("NN", "NNP") or txt in ("he", "she", "it"):
                sg = True
            elif tag in ("NNS", "NNPS") or txt in ("they", "we"):
                sg = False
            if sg is None:
                continue
            # find the finite verb form: head itself or an aux child
            cand = [toks[h]] + [toks[c.i] for c in doc[h].children if c.dep_ in ("aux", "auxpass", "cop")]
            for _, _, _, vtag, _, vt, _ in cand:
                vs = None
                if vtag == "VBZ" or vt in ("was",):
                    vs = True
                elif vtag == "VBP" or vt in ("were",):
                    vs = False
                if vs is not None and vs != sg:
                    sub_dis.append((abs(i - h), i, h))
                    break
    last_pos = toks[-1][4] if toks else ""
    parsed[g] = dict(n=len(toks), arcs=arcs, maxarc=max([a[0] for a in arcs], default=0),
                     longest=max(arcs, key=lambda a: a[0]) if arcs else None,
                     sub_dis=sub_dis, last_pos=last_pos, toks=[t[5] for t in toks])
print("parsed", len(uniq), flush=True)
for r in rows:
    p = parsed[r["gen"]]
    r.update(n=p["n"], maxarc=p["maxarc"], sub_dis=p["sub_dis"], last_pos=p["last_pos"])
    r["bin"] = min(max(p["maxarc"], 1), 6)
    for w in range(1, 8):
        r[f"gt{w}"] = float(p["maxarc"] > w)
    r["nlong"] = {w: sum(a[0] > w for a in p["arcs"]) for w in REFW}


# ------------------------------------------------------------------ helpers
def ci_mean(x, nb=NBOOT):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), (nb, len(x)))
    b = x[idx].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(t, n=None):
    m, lo, hi = t
    if np.isnan(m):
        return "-"
    s = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
    return s + (f" (n={n})" if n is not None else "")


def logit_fit(X, y, iters=30, ridge=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def design(rs, w, fam_is_trellis):
    L = np.array([r["n"] for r in rs], float)
    lo = np.array([r[f"gt{w}"] for r in rs])
    t = np.array([fam_is_trellis(r) for r in rs], float)
    # length as categorical-ish: linear + quadratic, centred
    Lc = (L - 6) / 3
    X = np.column_stack([np.ones(len(rs)), lo, t, lo * t, Lc, Lc ** 2])
    return X


def did_logit(rs, w, nb=200):
    y = np.array([r["gram"] for r in rs])
    isT = lambda r: r["fam"] == "TRELLIS"
    X = design(rs, w, isT)
    b = logit_fit(X, y)
    bs = []
    for _ in range(nb):
        i = RNG.integers(0, len(rs), len(rs))
        try:
            bs.append(logit_fit(X[i], y[i])[[1, 3]])
        except Exception:
            pass
    bs = np.array(bs)
    return b[1], b[3], np.percentile(bs[:, 0], [2.5, 97.5]), np.percentile(bs[:, 1], [2.5, 97.5])


def did_strat(rs, w, nb=500):
    """Length-standardised raw DiD: per length bin (n tokens 2..12+), (gram|short - gram|long) for TRELLIS minus neural,
    averaged with weights = pooled count of bins where all 4 cells have >=5 rows."""
    g = np.array([r["gram"] for r in rs])
    L = np.clip(np.array([r["n"] for r in rs]), 0, 12)
    lg = np.array([r[f"gt{w}"] for r in rs]).astype(int)
    tt = np.array([r["fam"] == "TRELLIS" for r in rs]).astype(int)
    cell = (L * 4 + tt * 2 + lg).astype(int)  # L in 0..12 -> 52 cells

    def one(ix):
        c = np.bincount(cell[ix], minlength=52).reshape(13, 4).astype(float)
        s = np.bincount(cell[ix], weights=g[ix], minlength=52).reshape(13, 4)
        tot = wt = 0.0
        for Lb in range(2, 13):
            if c[Lb].min() < 5:
                continue
            m = s[Lb] / c[Lb]  # order: N-short, N-long, T-short, T-long
            tot += ((m[3] - m[2]) - (m[1] - m[0])) * c[Lb].sum()
            wt += c[Lb].sum()
        return tot / wt if wt else np.nan
    est = one(np.arange(len(rs)))
    bs = []
    for _ in range(nb):
        bs.append(one(RNG.integers(0, len(rs), len(rs))))
    bs = np.array([b for b in bs if not np.isnan(b)])
    return est, np.percentile(bs, [2.5, 97.5])


def auc(score, y):
    score = np.asarray(score, float)
    y = np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ties
    order = allv.argsort()
    sv = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ------------------------------------------------------------------ report
out = []
P = out.append
P(f"# Round 201 dependency-distance diagnostic (n={N})\n")
P("Generated by `experiments/v3/diag/r201_distance.py`. Judged generations reused from story_gen dirs "
  f"({', '.join(PREFIX)}; n{N}); grammatical = Qwen2.5-3B P(yes) > 0.5; only `novel` sentences. "
  "Parser spaCy en_core_web_sm 3.8 (env `cobweb`). Arc length = |i - head| over spaCy tokens, root and punctuation excluded. "
  "CIs are 95% percentile bootstrap over sentences (sentence-level resampling; sentences share prompts and seeds, so CIs are somewhat optimistic).\n")

fams = ["TRELLIS", "LSTM", "GPT"]
cnt = defaultdict(int)
for r in rows:
    cnt[(r["fam"], r["W"])] += 1
P("## 0. Data\n")
P("| family | window | novel sentences | seeds | arms (dir@temp) |\n|---|---|---|---|---|")
for (fam, W), c in sorted(cnt.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
    sub = [r for r in rows if r["fam"] == fam and r["W"] == W]
    P(f"| {fam} | {W if W else '-'} | {c} | {sorted({r['seed'] for r in sub})} | {len({r['arm'] for r in sub})} arms |")
P("")

groups = [("TRELLIS w%d" % w, [r for r in rows if r["fam"] == "TRELLIS" and r["W"] == w]) for w in (2, 3, 4, 5)]
groups += [("LSTM", [r for r in rows if r["fam"] == "LSTM"]), ("GPT", [r for r in rows if r["fam"] == "GPT"])]
groups = [(k, v) for k, v in groups if v]

P("## 1. Share of sentences by maximum arc length bin and grammatical rate by bin\n")
P("| model | overall gram | " + " | ".join(f"arc {b}" if b < 6 else "arc 6+" for b in range(1, 7)) + " |\n|---|---|" + "---|" * 6)
for k, v in groups:
    cells = []
    for b in range(1, 7):
        sub = [r["gram"] for r in v if r["bin"] == b]
        cells.append(fmt(ci_mean(sub), len(sub)) if len(sub) >= 5 else f"- (n={len(sub)})")
    P(f"| {k} | {fmt(ci_mean([r['gram'] for r in v]))} | " + " | ".join(cells) + " |")
P("")
P("Share of sentences per max-arc bin:\n")
P("| model | " + " | ".join(f"{b}" if b < 6 else "6+" for b in range(1, 7)) + " | mean length (tokens) |\n|---|" + "---|" * 7)
for k, v in groups:
    P(f"| {k} | " + " | ".join(f"{np.mean([r['bin'] == b for r in v]):.3f}" for b in range(1, 7)) + f" | {np.mean([r['n'] for r in v]):.2f} |")
P("")

P("## 2. Grammatical rate: any arc > W vs none\n")
P("TRELLIS windows use their own W; LSTM/GPT use the same thresholds W = 3, 4, 5 as references. "
  "Penalty = gram(no arc > W) - gram(any arc > W).\n")
P("| model | W | share any arc > W | gram, no long arc | gram, any long arc | penalty |\n|---|---|---|---|---|---|")
pen = {}
for k, v in groups:
    ws = [v[0]["W"]] if v[0]["fam"] == "TRELLIS" else list(REFW)
    for w in ws:
        a = [r["gram"] for r in v if r[f"gt{w}"] == 0]
        b = [r["gram"] for r in v if r[f"gt{w}"] == 1]
        if len(a) < 5 or len(b) < 5:
            continue
        ia = RNG.integers(0, len(a), (NBOOT, len(a)))
        ib = RNG.integers(0, len(b), (NBOOT, len(b)))
        dd = np.asarray(a)[ia].mean(1) - np.asarray(b)[ib].mean(1)
        pen[(k, w)] = (np.mean(a) - np.mean(b), np.percentile(dd, 2.5), np.percentile(dd, 97.5))
        P(f"| {k} | {w} | {len(b) / len(v):.3f} | {fmt(ci_mean(a), len(a))} | {fmt(ci_mean(b), len(b))} | {fmt(pen[(k, w)])} |")
P("")

P("## 3. Difference-in-differences: TRELLIS long-arc penalty minus neural long-arc penalty\n")
P("Two estimators per threshold W. (a) Logistic fit grammatical ~ long_arc + is_TRELLIS + long_arc:is_TRELLIS + length + length^2 "
  "(length in spaCy tokens, centred); the interaction coefficient is the log-odds DiD (negative = TRELLIS penalised more). "
  "(b) Length-stratified raw DiD in probability: for each length 2..12+, (gram long - gram short)|TRELLIS minus the same for neural, "
  "averaged with cell-count weights; `-` bins with <5 rows in a cell dropped. TRELLIS rows = windows matching W when the family has that window "
  "(row 'W-matched'); also pooled over all TRELLIS windows.\n")
P("| comparison | W | long-arc coef (neural) [CI] | interaction (TRELLIS - neural), logit [CI] | stratified DiD prob [CI] | n |\n|---|---|---|---|---|---|")
neural = [r for r in rows if r["fam"] != "TRELLIS"]
verdict_rows = []
for label, sel in [("TRELLIS all windows vs LSTM+GPT", lambda w: [r for r in rows if r["fam"] == "TRELLIS"]),
                   ("TRELLIS w=W vs LSTM+GPT", lambda w: [r for r in rows if r["fam"] == "TRELLIS" and r["W"] == w]),
                   ("TRELLIS w=W vs LSTM", lambda w: [r for r in rows if r["fam"] == "TRELLIS" and r["W"] == w]),
                   ("TRELLIS w=W vs GPT", lambda w: [r for r in rows if r["fam"] == "TRELLIS" and r["W"] == w])]:
    for w in REFW:
        tr = sel(w)
        if "vs LSTM+GPT" in label:
            ne = neural
        elif "vs LSTM" in label:
            ne = [r for r in rows if r["fam"] == "LSTM"]
        else:
            ne = [r for r in rows if r["fam"] == "GPT"]
        if len(tr) < 50 or len(ne) < 50:
            continue
        rs = tr + ne
        c1, c3, ci1, ci3 = did_logit(rs, w)
        est, ci = did_strat(rs, w)
        verdict_rows.append((label, w, c3, ci3, est, ci))
        P(f"| {label} | {w} | {c1:.2f} [{ci1[0]:.2f},{ci1[1]:.2f}] | {c3:.2f} [{ci3[0]:.2f},{ci3[1]:.2f}] | {est:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {len(tr)}+{len(ne)} |")
P("")

# operating-point matched
P("### 3b. Matched operating point\n")
P("TRELLIS and neural arms sit at different temperatures and so at different overall grammaticality. Restricting to arms (family, dir, temp) "
  "whose overall grammatical rate lies in [0.60, 0.85] gives comparable operating points.\n")
armg = defaultdict(list)
for r in rows:
    armg[r["arm"]].append(r["gram"])
okarm = {a for a, v in armg.items() if 0.60 <= np.mean(v) <= 0.85}
rm = [r for r in rows if r["arm"] in okarm]
P("| family | rows in matched arms | overall gram |\n|---|---|---|")
for fam in fams:
    s = [r["gram"] for r in rm if r["fam"] == fam]
    P(f"| {fam} | {len(s)} | {np.mean(s):.3f} |" if s else f"| {fam} | 0 | - |")
P("")
P("| W | interaction logit [CI] | stratified DiD prob [CI] | penalty TRELLIS | penalty neural |\n|---|---|---|---|---|")
matched_rows = []
for w in REFW:
    tr = [r for r in rm if r["fam"] == "TRELLIS"]
    ne = [r for r in rm if r["fam"] != "TRELLIS"]
    if len(tr) < 50 or len(ne) < 50:
        continue
    c1, c3, ci1, ci3 = did_logit(tr + ne, w)
    est, ci = did_strat(tr + ne, w)
    pT = np.mean([r["gram"] for r in tr if r[f"gt{w}"] == 0]) - np.mean([r["gram"] for r in tr if r[f"gt{w}"] == 1])
    pN = np.mean([r["gram"] for r in ne if r[f"gt{w}"] == 0]) - np.mean([r["gram"] for r in ne if r[f"gt{w}"] == 1])
    matched_rows.append((w, c3, ci3, est, ci))
    P(f"| {w} | {c3:.2f} [{ci3[0]:.2f},{ci3[1]:.2f}] | {est:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {pT:.3f} | {pN:.3f} |")
P("")

# length-controlled table by length bin for w5
P("## 4. Length control: grammatical rate by sentence length x (arc > 3), TRELLIS vs neural\n")
P("| length (tokens) | TRELLIS short-arc | TRELLIS long-arc | neural short-arc | neural long-arc |\n|---|---|---|---|---|")
for L in range(3, 13):
    cs = []
    for fam in (True, False):
        for lg in (0, 1):
            xs = [r["gram"] for r in rows if min(r["n"], 12) == L and ((r["fam"] == "TRELLIS") == fam) and r["gt3"] == lg]
            cs.append(f"{np.mean(xs):.2f} (n={len(xs)})" if len(xs) >= 5 else f"- (n={len(xs)})")
    P(f"| {L if L < 12 else '12+'} | " + " | ".join(cs) + " |")
P("")

# ------------------------------------------------------------------ error localisation
P("## 5. Error localisation in ungrammatical TRELLIS sentences\n")
P("Proxy: nsubj/nsubjpass whose number (noun tag NN/NNS, pronouns he/she/it vs they/we) disagrees with its finite verb or aux "
  "(VBZ/was vs VBP/were). `long` = that nsubj-verb arc > W (the sentence's TRELLIS window).\n")
P("| model | sentences | share ungrammatical | ungram with any arc > W | ungram with number disagreement | ... on arc > W | gram with number disagreement | ... on arc > W |\n|---|---|---|---|---|---|---|---|")
for k, v in groups:
    if v[0]["fam"] == "TRELLIS":
        Wk = lambda r: r["W"]
    else:
        Wk = lambda r: 4
    ug = [r for r in v if r["gram"] == 0]
    gg = [r for r in v if r["gram"] == 1]
    def stats(S):
        anyl = sum(r[f"gt{Wk(r)}"] == 1 for r in S)
        dis = sum(len(r["sub_dis"]) > 0 for r in S)
        disl = sum(any(a > Wk(r) for a, _, _ in r["sub_dis"]) for r in S)
        return anyl, dis, disl
    a1, d1, l1 = stats(ug)
    a2, d2, l2 = stats(gg)
    P(f"| {k} | {len(v)} | {len(ug) / len(v):.3f} | {a1} ({a1 / max(len(ug), 1):.3f}) | {d1} ({d1 / max(len(ug), 1):.3f}) | {l1} | {d2} ({d2 / max(len(gg), 1):.3f}) | {l2} |")
P("\n(For LSTM/GPT, the `W` used for 'arc > W' columns is 4.)\n")

# longest arc dep type
P("### 5b. Grammatical rate by dependency type of the longest arc (TRELLIS vs neural)\n")
P("| longest-arc dep | TRELLIS n | TRELLIS gram | neural n | neural gram |\n|---|---|---|---|---|")
dd = defaultdict(lambda: {"T": [], "N": []})
for r in rows:
    lg = parsed[r["gen"]]["longest"]
    dep = lg[1] if lg else "none"
    if dep in ("nsubj", "nsubjpass", "dobj", "conj", "cc", "det", "prep", "pobj", "aux", "advmod", "acomp", "xcomp", "ccomp", "attr", "mark"):
        pass
    else:
        dep = "other"
    dd[dep]["T" if r["fam"] == "TRELLIS" else "N"].append(r["gram"])
for dep, v in sorted(dd.items(), key=lambda kv: -len(kv[1]["T"])):
    P(f"| {dep} | {len(v['T'])} | {np.mean(v['T']):.3f} | {len(v['N'])} | {np.mean(v['N']) if v['N'] else float('nan'):.3f} |")
P("")

# ------------------------------------------------------------------ predictors
P("## 6. What predicts TRELLIS ungrammaticality (single-feature AUC for ungrammatical, TRELLIS rows; neural rows for comparison)\n")
FUNC = {"DET", "ADP", "CCONJ", "AUX", "PART", "SCONJ"}
feats = {
    "length (tokens)": lambda r: r["n"],
    "max arc": lambda r: r["maxarc"],
    "any arc > own W (4 for neural)": lambda r: r[f"gt{r['W'] or 4}"],
    "number of arcs > own W": lambda r: r["nlong"][r["W"] if r["W"] in REFW else 4] if (r["W"] in REFW or r["W"] is None) else 0,
    "ends on function word (DET/ADP/CCONJ/AUX/PART/SCONJ)": lambda r: float(r["last_pos"] in FUNC),
    "repeated word": lambda r: float(len(set(parsed[r["gen"]]["toks"])) < len(parsed[r["gen"]]["toks"])),
    "number disagreement": lambda r: float(len(r["sub_dis"]) > 0),
    "has 'and'/conj arc": lambda r: float(any(a[1] in ("conj", "cc") for a in parsed[r["gen"]]["arcs"])),
}
P("| feature | AUC for ungrammatical (TRELLIS) | AUC (LSTM) | AUC (GPT) |\n|---|---|---|---|")
T = [r for r in rows if r["fam"] == "TRELLIS" and r["W"] in REFW]
Ls = [r for r in rows if r["fam"] == "LSTM"]
Gs = [r for r in rows if r["fam"] == "GPT"]
for name, f in feats.items():
    def A(S):
        if len(S) < 30:
            return float("nan")
        return auc([f(r) for r in S], [1 - r["gram"] for r in S])
    P(f"| {name} | {A(T):.3f} | {A(Ls):.3f} | {A(Gs):.3f} |")
P("\nAUC 0.5 = no information; >0.5 means the feature is higher in ungrammatical sentences.\n")

# bin share among ungrammatical vs grammatical (TRELLIS w5)
P("### 6b. Where TRELLIS ungrammatical sentences sit (w5, share by max-arc bin)\n")
for w in (3, 4, 5):
    S = [r for r in rows if r["fam"] == "TRELLIS" and r["W"] == w]
    if not S:
        continue
    P(f"- w{w}: ungram share by bin 1..6+: " + ", ".join(f"{np.mean([r['bin'] == b for r in S if r['gram'] == 0]):.2f}" for b in range(1, 7)) +
      "; gram share: " + ", ".join(f"{np.mean([r['bin'] == b for r in S if r['gram'] == 1]):.2f}" for b in range(1, 7)))
P("")

# ------------------------------------------------------------------ cases
P("## 7. Cases: ungrammatical TRELLIS sentences with a long arc (> W), preferring number-disagreement arcs\n")
cands = [r for r in rows if r["fam"] == "TRELLIS" and r["W"] in REFW and r["gram"] == 0 and r[f"gt{r['W']}"] == 1]
cands.sort(key=lambda r: (-(any(a > r["W"] for a, _, _ in r["sub_dis"])), r["gen"]))
seen = set()
k = 0
dis_c = [r for r in cands if any(a > r["W"] for a, _, _ in r["sub_dis"])]
other_c = [r for r in cands if r not in dis_c]
pick = []
for pool, quota in ((dis_c, 5), (other_c, 5)):
    idx = RNG.permutation(len(pool))
    c = 0
    for i in idx:
        r = pool[i]
        if r["gen"] in seen:
            continue
        seen.add(r["gen"])
        pick.append(r)
        c += 1
        if c >= quota:
            break
for r in pick:
    p = parsed[r["gen"]]
    arcs = sorted([a for a in p["arcs"] if a[0] > r["W"]], key=lambda a: -a[0])[:3]
    desc = "; ".join(f"{a[1]}: '{p['toks'][a[2]]}' -> '{p['toks'][a[3]]}' (len {a[0]})" for a in arcs)
    dis = ""
    if r["sub_dis"]:
        a, i, h = r["sub_dis"][0]
        dis = f" **[number disagreement '{p['toks'][i]}' ~ '{p['toks'][h]}', arc {a}]**"
    P(f"- \"{r['gen']}\" (w{r['W']}, seed {r['seed']}, temp {r['temp']}, Qwen P(yes)={r['p']:.3f}); arcs > W: {desc}{dis}")
P("")

with open(OUT, "w") as fh:
    fh.write("\n".join(out) + "\n")
print("wrote", OUT)
