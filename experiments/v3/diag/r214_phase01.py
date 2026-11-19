"""Round 214 (tracks 1 and 2): regime purities (r214_judge2 axes) and r204_fair novelty quantities for generation dumps.

  r214_phase01.py collect <out dir> <label>=<dump dir> [...]       writes <out>/sentences.txt (distinct new sentences of every arm
                                                                    and of every r211 set) -> judge with r214_judge2.py score
  r214_phase01.py report <out dir> <scores.tsv> <out.md> <label>=<dump dir> [...]

Regimes (design section 0), from the judge's own pass flags (thresholds chosen by r214_judge2.md): R-sem = syn_pass and sem_pass,
R-syn = syn_pass and not sem_pass, R-non = not syn_pass (its sem_pass share = "word salad on a topic" is reported). Purity = share of
the arm's distinct new sentences in the regime. Novelty as r204_fair.py: attempted draws A = 1600 (400 prompts x 4 draws), distinct new
= distinct generated strings that are not a stored training sentence, tri|d = share of distinct new with an unstored padded trigram,
fair_X = distinct new in regime X with an unstored trigram per attempted draw, repeats = new rows minus distinct, repcut = repeats
that are a proper contiguous span of a stored sentence. Env: repo root, env chunks, compute node.
"""
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier as sf  # noqa: E402

GD = "experiments/v3/results/story_gen"
A = 1600
TRAIN = {}


def train(seed, n=1000):
    if (seed, n) not in TRAIN:
        TRAIN[(seed, n)] = sf.Train(sf.train_sentences(seed, n))
    return TRAIN[(seed, n)]


def arm_label(arm):
    k = []
    if arm.get("wreach") is not None:
        k.append("W'=%s" % arm["wreach"])
    if arm.get("bits") is not None:
        k.append("b=%s" % arm["bits"])
    k.append("t=%g" % arm.get("temp", 1.0))
    if arm.get("pair"):
        k.append("gate")
    return " ".join(k)


def load_arms(specs):
    out = []
    for sp in specs:
        lab, d = sp.split("=", 1)
        for f in sorted(glob.glob(os.path.join(d, "*_arm*.jsonl"))):
            m = re.search(r"_s(\d+)_n(\d+)_arm(\d+)\.jsonl$", f)
            lines = [json.loads(l) for l in open(f, encoding="utf-8")]
            arm = lines[0]["arm"]
            out.append(dict(group=lab, seed=int(m.group(1)), n=int(m.group(2)), ai=int(m.group(3)), arm=arm,
                            label=arm_label(arm), rows=[r["gen"] for r in lines[1:]]))
    return out


def distinct_new(rows, T):
    seen, newrows, cut = {}, 0, 0
    for g in rows:
        s = tuple(g.split())
        if s in T.whole:
            continue
        newrows += 1
        if s in seen:
            cut += s in T.sub
            continue
        seen[s] = (any(x not in T.tri for x in sf.grams(s, 3)), s in T.sub)
    return seen, newrows, cut


def r211_sets():
    p = os.path.join(GD, "r211", "r211_sets.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def collect(out, specs):
    os.makedirs(out, exist_ok=True)
    S = set()
    for a in load_arms(specs):
        seen, _, _ = distinct_new(a["rows"], train(a["seed"], a["n"]))
        S |= {" ".join(s) for s in seen}
    n_arms = len(S)
    for e in r211_sets():
        S |= {" ".join(x[0]) if isinstance(x[0], list) else x[0] for x in e["sents"]}
    with open(os.path.join(out, "sentences.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(S)) + "\n")
    print("sentences", len(S), "from arms", n_arms)


def read_scores(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")]
    h = rows[0]
    ix = {k: i for i, k in enumerate(h)}
    out = {}
    for r in rows[1:]:
        out[r[0]] = (float(r[ix["syn_p"]]), float(r[ix["sem_p"]]), int(r[ix["syn_pass"]]), int(r[ix["sem_pass"]]))
    return out


def regime(sc):
    syn, sem = sc[2], sc[3]
    return "sem" if syn and sem else "syn" if syn else "non"


def f3(x):
    return "-" if x is None else "%.3f" % x


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def stats(seen, newrows, cut, J):
    d = len(seen)
    reg = {s: regime(J[" ".join(s)]) for s in seen}
    r = dict(d=d, dist=d / A, tri=mean([v[0] for v in seen.values()]), repeats=newrows - d, repcut=cut,
             salad=(sum(1 for s in seen if reg[s] == "non" and J[" ".join(s)][3]) / d) if d else None)
    for k in ("sem", "syn", "non"):
        ss = [s for s in seen if reg[s] == k]
        r[k] = len(ss) / d if d else None
        r["tri_" + k] = mean([seen[s][0] for s in ss])
        r["fair_" + k] = sum(seen[s][0] for s in ss) / A
    return r


def report(out, scores, out_md, specs):
    J = read_scores(scores)
    arms = load_arms(specs)
    rng = random.Random(214)
    L = ["| group | arm | seed | distinct new | dist/draw | R-sem | R-syn | R-non | (non with sem_pass) | tri\\|d | tri\\|d in sem / syn / non | fair sem / syn / non | repeats (cut) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    samples = []
    for a in sorted(arms, key=lambda a: (a["group"], a["seed"], a["ai"])):
        T = train(a["seed"], a["n"])
        seen, nr, cut = distinct_new(a["rows"], T)
        r = stats(seen, nr, cut, J)
        L.append("| %s | %s | %d | %d | %.3f | %s | %s | %s | %s | %s | %s / %s / %s | %.3f / %.3f / %.3f | %d (%d) |" % (
            a["group"], a["label"], a["seed"], r["d"], r["dist"], f3(r["sem"]), f3(r["syn"]), f3(r["non"]), f3(r["salad"]), f3(r["tri"]),
            f3(r["tri_sem"]), f3(r["tri_syn"]), f3(r["tri_non"]), r["fair_sem"], r["fair_syn"], r["fair_non"], r["repeats"], r["repcut"]))
        pool = sorted(seen)
        pick = rng.sample(pool, min(5, len(pool)))
        samples.append("**%s / %s**\n" % (a["group"], a["label"]) + "\n".join(
            "- %s  (syn_p %.2f, sem_p %.2f, %s)" % (" ".join(s), J[" ".join(s)][0], J[" ".join(s)][1], regime(J[" ".join(s)])) for s in pick))
    # r211 re-judge: mean over seeds per family and temperature
    by = defaultdict(list)
    for e in r211_sets():
        sents = [tuple((x[0] if isinstance(x[0], list) else x[0].split())) for x in e["sents"]]
        T = train(e["seed"], 1000)
        seen = {s: (any(g not in T.tri for g in sf.grams(s, 3)), s in T.sub) for s in set(sents) if s not in T.whole}
        if not seen:
            continue
        reg = {s: regime(J[" ".join(s)]) for s in seen}
        d = len(seen)
        by[(e["fam"], e["temp"])].append(dict(d=d, dist=d / e["A"], tri=mean([v[0] for v in seen.values()]),
                                              **{k: sum(1 for s in seen if reg[s] == k) / d for k in ("sem", "syn", "non")},
                                              fair_sem=sum(seen[s][0] for s in seen if reg[s] == "sem") / e["A"]))
    R = ["| family | temp | seeds | distinct/seed | dist/draw | tri\\|d | R-sem | R-syn | R-non | fair R-sem |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(by, key=lambda k: (str(k[0]), k[1])):
        v = by[k]
        m = lambda key: mean([x[key] for x in v])
        R.append("| %s | %s | %d | %.0f | %.3f | %s | %s | %s | %s | %.3f |" % (k[0], k[1], len(v), m("d"), m("dist"), f3(m("tri")),
                                                                               f3(m("sem")), f3(m("syn")), f3(m("non")), m("fair_sem")))
    open(out_md, "w", encoding="utf-8").write(
        "## Arms (seed 13, n=1000)\n\n" + "\n".join(L) + "\n\n## Samples (5 per arm, random, fixed seed 214)\n\n" + "\n\n".join(samples) +
        "\n\n## r211 dumps re-judged (mean over seeds present; distinct new only)\n\n" + "\n".join(R) + "\n")
    print("wrote", out_md)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3:])
    else:
        report(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
