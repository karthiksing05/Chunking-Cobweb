"""Round 215: control sets per seed, the r214a recipe (experiments/v3/diag/r214_judge2.py build, copied, not imported) on each seed's own
held-out stories (the first 1500 of Random(seed)-shuffled stories, never trained on by that seed's stream), so a seed's training stream never
holds its test stories. Seed 13 is compared with results/story_gen/r214/controls.json (the 214a sets) and the match is printed.
Adds Gcal: 400 further distinct held-out sentences (not in the 300 base) for the calibration half.
Usage (repo root, compute node, env trellis or chunks):  r215_sets.py <seed>
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402

SEED, N_CLASS = 214, 300
FUNC = set("""is was were are am be been being has have had do does did will would can could should may might must
not very so too then just also there here now all up out down off to of in on at for with from by as that this these those
it its he she they we you i me him her them us my your his their our one two some any no yes more most much many
said say says""".split())
EXCLUDE = set("anna ben bob dad lucy max mia mom mommy sam sara spot sue tim tom lily don't everyone heard i'm kind lot lots something sorry thank idea time day home share fun ground over pretty long other".split())
CONTENT_TAGS = {"NN", "NNS", "VB", "VBD", "JJ"}
VERB_FRAME = {w: "INTR" for w in "came flew laughed lived played ran smiled walked went happened go run play".split()}
VERB_FRAME.update({w: "TR" for w in "found liked loved opened picked saw took thanked made eat find see take make".split()})
FLIP = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has"}


def norm_tag(t):
    return {"VBP": "VB", "VB": "VB"}.get(t, t)


def build(seed):
    import nltk
    train, held_lm, test, vocab = C.stream(seed, 10)
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    sents, seen = [], set()
    for st in test:
        for s in st:
            if keep(s) and tuple(s) not in seen:
                seen.add(tuple(s))
                sents.append(s)
    rng = random.Random(SEED)
    rng.shuffle(sents)
    print("held-out kept distinct sentences", len(sents), flush=True)
    tags = [[t for _, t in nltk.pos_tag(s)] for s in sents]
    cnt = defaultdict(Counter)
    for s, t in zip(sents, tags):
        for w, x in zip(s, t):
            cnt[w][norm_tag(x)] += 1
    maj = {w: c.most_common(1)[0][0] for w, c in cnt.items()}
    pool, pkey = defaultdict(list), {}
    for w, x in maj.items():
        if w not in FUNC and w not in EXCLUDE and x in CONTENT_TAGS and cnt[w][x] / sum(cnt[w].values()) >= .6 and cnt[w][x] >= 3:
            if x in ("VB", "VBD"):
                if w not in VERB_FRAME:
                    continue
                k = f"{x}-{VERB_FRAME[w]}"
            else:
                k = x
            pool[k].append(w)
            pkey[w] = k
    pool = {k: sorted(v) for k, v in pool.items() if len(v) >= 4}
    pkey = {w: k for w, k in pkey.items() if k in pool}

    def slots(s, t):
        out = []
        for i, (w, x) in enumerate(zip(s, t)):
            if w not in pkey or maj.get(w) != norm_tag(x):
                continue
            if pkey[w].split("-")[0] in ("VB", "VBD"):
                nxt = s[i + 1] if i + 1 < len(s) else None
                if nxt == "to" or (pkey[w].endswith("TR") and not pkey[w].endswith("INTR") and nxt is None):
                    continue
            out.append(i)
        return out

    base, rest = [], []
    for s, t in zip(sents, tags):
        (base if len(slots(s, t)) >= 2 else rest).append((s, t))
    allbase = base
    base = base[:N_CLASS]
    cal = [s for s, _ in allbase[N_CLASS:]] + [s for s, _ in rest]
    sets = defaultdict(list)
    r = random.Random(SEED + 1)
    for s, t in base:
        sl = slots(s, t)
        sets["C1"].append(" ".join(s))
        full = list(s)
        for i in sl:
            full[i] = r.choice([w for w in pool[pkey[s[i]]] if w != s[i]])
        sets["C2full"].append(" ".join(full))
        one = list(s)
        i = r.choice(sl)
        one[i] = r.choice([w for w in pool[pkey[s[i]]] if w != s[i]])
        sets["C2one"].append(" ".join(one))
        sc = list(s)
        for _ in range(100):
            r.shuffle(sc)
            if sc != list(s):
                break
        sets["C3scr"].append(" ".join(sc))
        cand = [i for i in range(len(s) - 1) if norm_tag(t[i]) != norm_tag(t[i + 1]) and s[i] != s[i + 1]]
        ad = list(s)
        if cand:
            i = r.choice(cand)
            ad[i], ad[i + 1] = ad[i + 1], ad[i]
        sets["C3adj_all"].append(" ".join(ad))
    c4 = []
    for s in sents:
        if any(w in FLIP for w in s) and not ({"i", "you", "and"} & set(s)):
            j = [i for i, w in enumerate(s) if w in FLIP][0]
            e = list(s)
            e[j] = FLIP[s[j]]
            c4.append((" ".join(s), " ".join(e)))
        if len(c4) >= N_CLASS:
            break
    out = {k: v for k, v in sets.items() if k != "C3adj_all"}
    out["C3adj"] = [a for a, o in zip(sets["C3adj_all"], sets["C1"]) if a != o]
    out["C3adj_src"] = [o for a, o in zip(sets["C3adj_all"], sets["C1"]) if a != o]
    out["C4"] = [e for _, e in c4]
    out["C4_src"] = [o for o, _ in c4]
    # the neural LMs have no id for a word outside the 250-word vocabulary ("has" is outside): C4v keeps the pairs whose error is in vocabulary
    iv = [all(w in vocab for w in e.split()) for e in out["C4"]]
    out["C4v"] = [e for e, k in zip(out["C4"], iv) if k]
    out["C4v_src"] = [o for o, k in zip(out["C4_src"], iv) if k]
    cs = set(out["C1"]) | set(out["C4_src"])
    out["Gcal"] = [" ".join(s) for s in cal if " ".join(s) not in cs][:400]
    if seed == 13:
        old = json.load(open("experiments/v3/results/story_gen/r214/controls.json", encoding="utf-8"))
        same = {k: out[k] == old[k] for k in ("C1", "C2full", "C2one", "C3scr", "C4", "C4_src")}
        print("seed 13 vs r214 controls.json:", same, flush=True)
    os.makedirs(C.SETS, exist_ok=True)
    json.dump(out, open(os.path.join(C.SETS, "sets_s%d.json" % seed), "w", encoding="utf-8"))
    print({k: len(v) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    build(int(sys.argv[1]))
