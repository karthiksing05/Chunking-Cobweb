"""Round 230: a controlled language in which syntax and meaning are defined exactly, with its generator, its exact judge and the purity measure.

Syntax (CFG with agreement):   S -> NP VP;  NP -> Det (Adj) N;  VP -> Vi | Vt NP | Vpp Prep NP
  number: a + sg noun, some + pl noun, the + either; the verb agrees with the subject noun (sg: -s form, pl: base form);
  frame: each verb has one frame (Vi / Vt / Vpp) and Vpp verbs fix their preposition (to / in).
Semantics (selectional restrictions): nouns have a type (person, animal, food, place, object); verbs restrict subject / object / complement
  types; adjectives restrict the noun type (hungry: animate, ripe: food ...).
Judge (rule-based, defined before any generation):  R-non = the sentence does not parse (words not in the lexicon, order, agreement, frame, preposition);
  R-syn = it parses but some selectional restriction fails; R-sem = it parses and every restriction holds.

Commands (compute node):
  r230_grammar.py controls  <out.json>                  judge validation on sampled / swapped / shuffled sentences
  r230_grammar.py judge     <out.json> <dump dir>...    regime shares of every arm jsonl in the dump dirs
  r230_grammar.py purity    <out.json> <wordtree.json>...  syntactic / semantic purity of the word-tree classes at each level
  r230_grammar.py examples  <dump jsonl> <k> <seed>     k judged examples per regime
"""
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------------------------- lexicon
NOUNS = {
    "person": ["boy", "girl", "man", "woman", "teacher", "farmer", "baker", "doctor"],
    "animal": ["dog", "cat", "horse", "cow", "bird", "rabbit", "fox", "pig"],
    "food": ["apple", "cake", "cookie", "pie", "carrot", "egg", "pear", "lemon"],
    "place": ["house", "garden", "school", "market", "farm", "park", "shop", "forest"],
    "object": ["ball", "book", "hat", "box", "chair", "key", "cup", "coat"],
}
ANIMATE = {"person", "animal"}
ALL = set(NOUNS)
ADJ = {  # adjective: noun types it may modify
    "hungry": ANIMATE, "sleepy": ANIMATE, "happy": ANIMATE, "kind": {"person"}, "tall": {"person", "place"},
    "quiet": {"person", "animal", "place"}, "old": {"person", "animal", "place", "object"}, "ripe": {"food"},
    "sweet": {"food"}, "fresh": {"food"}, "red": {"food", "object"}, "broken": {"object"},
}
ADJ_ANY = {"big": ALL, "small": ALL}
ADJ.update(ADJ_ANY)
# verb: (sg form, frame, subject types, object / complement types, preposition)
VERBS = {
    "sleep": ("sleeps", "vi", ANIMATE, None, None), "run": ("runs", "vi", ANIMATE, None, None),
    "smile": ("smiles", "vi", {"person"}, None, None), "laugh": ("laughs", "vi", {"person"}, None, None),
    "break": ("breaks", "vi", {"object"}, None, None), "spoil": ("spoils", "vi", {"food"}, None, None),
    "eat": ("eats", "vt", ANIMATE, {"food"}, None), "cook": ("cooks", "vt", {"person"}, {"food"}, None),
    "bake": ("bakes", "vt", {"person"}, {"food"}, None), "buy": ("buys", "vt", {"person"}, {"food", "object"}, None),
    "carry": ("carries", "vt", {"person"}, {"object", "food"}, None), "chase": ("chases", "vt", ANIMATE, {"animal"}, None),
    "feed": ("feeds", "vt", {"person"}, {"animal"}, None), "help": ("helps", "vt", {"person"}, {"person"}, None),
    "visit": ("visits", "vt", {"person"}, {"place"}, None), "clean": ("cleans", "vt", {"person"}, {"place", "object"}, None),
    "fix": ("fixes", "vt", {"person"}, {"object"}, None), "hug": ("hugs", "vt", {"person"}, {"person", "animal"}, None),
    "go": ("goes", "vpp", ANIMATE, {"place"}, "to"), "walk": ("walks", "vpp", ANIMATE, {"place"}, "to"),
    "return": ("returns", "vpp", ANIMATE, {"place"}, "to"), "live": ("lives", "vpp", ANIMATE, {"place"}, "in"),
    "work": ("works", "vpp", {"person"}, {"place"}, "in"), "stay": ("stays", "vpp", ANIMATE, {"place"}, "in"),
}
DETS = {"a": "sg", "some": "pl", "the": "any"}
PREPS = ["to", "in"]

# word -> entry
LEX = {}
for t, ws in NOUNS.items():
    for w in ws:
        LEX[w] = ("n", t, "sg")
        LEX[w + "s"] = ("n", t, "pl")
for a, ty in ADJ.items():
    LEX[a] = ("adj", a, None)
for d, nm in DETS.items():
    LEX[d] = ("det", d, nm)
for p in PREPS:
    LEX[p] = ("prep", p, None)
for v, (sg, fr, _, _, prep) in VERBS.items():
    LEX[v] = ("v", v, "pl")
    LEX[sg] = ("v", v, "sg")
assert len(LEX) == 2 * sum(len(v) for v in NOUNS.values()) + len(ADJ) + len(DETS) + len(PREPS) + 2 * len(VERBS), "ambiguous word"
POS = {"n": "NN", "adj": "JJ", "det": "DT", "prep": "IN", "v": "VB"}
TAG = {w: POS[e[0]] for w, e in LEX.items()}


def syncat(w):
    """The syntactic category of a word (what a CFG with agreement distinguishes)."""
    e = LEX[w]
    if e[0] == "n":
        return ("n", e[2])
    if e[0] == "v":
        _, fr, _, _, prep = VERBS[e[1]]
        return ("v", fr, prep, e[2])
    if e[0] == "det":
        return ("det", e[2])
    return (e[0],)


def semtype(w):
    """The semantic type of a content word (what the selectional restrictions distinguish); None for determiners and prepositions."""
    e = LEX[w]
    if e[0] == "n":
        return ("n", e[1])
    if e[0] == "adj":
        return ("adj", tuple(sorted(ADJ[e[1]])))
    if e[0] == "v":
        _, fr, s, o, prep = VERBS[e[1]]
        return ("v", tuple(sorted(s)), tuple(sorted(o)) if o else None)
    return None


# ----------------------------------------------------------------------------------------------- sampler
def _zipf(items, rng, a=0.7):
    return rng.choices(items, weights=[(i + 1) ** -a for i in range(len(items))])[0]


def _noun(types, rng, plural=None):
    ty = rng.choice(sorted(types))
    w = _zipf(NOUNS[ty], rng)
    return w, ty


def _np(types, rng, num=None):
    pl = (rng.random() < 0.4) if num is None else (num == "pl")
    n, ty = _noun(types, rng)
    toks = []
    toks.append(rng.choice(["the", "some"]) if pl else rng.choice(["a", "the"]))
    if rng.random() < 0.3:
        toks.append(rng.choice(sorted(a for a, ts in ADJ.items() if ty in ts)))
    toks.append(n + "s" if pl else n)
    return toks, ("pl" if pl else "sg")


def sample_sentence(rng):
    """One sentence that satisfies the syntax and every selectional restriction."""
    fr = rng.choices(["vi", "vt", "vpp"], weights=[0.25, 0.5, 0.25])[0]
    v = _zipf([k for k, x in VERBS.items() if x[1] == fr], rng, 0.4)
    sg, _, subj, obj, prep = VERBS[v]
    sub, num = _np(subj, rng)
    out = sub + [sg if num == "sg" else v]
    if fr == "vt":
        out += _np(obj, rng)[0]
    elif fr == "vpp":
        out += [prep] + _np(obj, rng)[0]
    return out


# ----------------------------------------------------------------------------------------------- judge
def _parse_np(t, i):
    """(ok, noun entry, adj name or None, next index)."""
    if i >= len(t) or t[i] not in LEX or LEX[t[i]][0] != "det":
        return None
    det = LEX[t[i]]
    i += 1
    adj = None
    if i < len(t) and t[i] in LEX and LEX[t[i]][0] == "adj":
        adj = t[i]
        i += 1
    if i >= len(t) or t[i] not in LEX or LEX[t[i]][0] != "n":
        return None
    n = LEX[t[i]]
    if det[2] != "any" and det[2] != n[2]:
        return None                                    # a + pl noun, some + sg noun
    return n, adj, i + 1


def judge(toks):
    """('sem' | 'syn' | 'non', detail) for a token list."""
    t = list(toks)
    r = _parse_np(t, 0)
    if r is None:
        return "non", "subject NP"
    sn, sadj, i = r
    if i >= len(t) or t[i] not in LEX or LEX[t[i]][0] != "v":
        return "non", "verb"
    ve = LEX[t[i]]
    if ve[2] != sn[2]:
        return "non", "agreement"
    _, fr, subj, obj, prep = VERBS[ve[1]]
    i += 1
    nps = [(sn, sadj)]
    if fr == "vi":
        if i != len(t):
            return "non", "trailing"
    else:
        if fr == "vpp":
            if i >= len(t) or t[i] != prep:
                return "non", "preposition"
            i += 1
        r = _parse_np(t, i)
        if r is None:
            return "non", "object NP"
        on, oadj, i = r
        if i != len(t):
            return "non", "trailing"
        nps.append((on, oadj))
    # syntax holds: semantics
    why = []
    if sn[1] not in subj:
        why.append("subject")
    if fr != "vi" and nps[1][0][1] not in obj:
        why.append("object")
    for (n, a) in nps:
        if a is not None and n[1] not in ADJ[a]:
            why.append("adjective")
    return ("syn", "+".join(why)) if why else ("sem", "")


# ----------------------------------------------------------------------------------------------- corpus loader for t3_story.py
def load_corpus(seed, n, n_max=4000):
    """Everything t3_story.py's TinyStories block defines, from the grammar. Training sentences are the first n of one stream per seed
    (nested across n), all grammar-valid (syntax and semantics)."""
    rng = random.Random(1000 + seed)
    stream = [sample_sentence(rng) for _ in range(max(n, n_max))]
    train = [(s, False) for s in stream[:n]]
    seen = {tuple(s) for s, _ in train}
    r2 = random.Random(2000 + seed)
    held = [sample_sentence(r2) for _ in range(1500)]
    pairs = [(sample_sentence(r2), sample_sentence(r2)) for _ in range(400)]
    val = [(sample_sentence(r2), sample_sentence(r2)) for _ in range(200)]
    lm_held = [sample_sentence(r2) for _ in range(500)]
    all_sents = [sample_sentence(r2) for _ in range(5000)]
    vocab = set(LEX)
    gold_ct = defaultdict(Counter)
    patterns = set()
    for s in all_sents:
        patterns.add(tuple(TAG[w] for w in s))
        for w in s:
            gold_ct[w][TAG[w]] += 1
    gold = {w: c.most_common(1)[0][0] for w, c in gold_ct.items()}
    by_tag = defaultdict(Counter)
    for s in all_sents:
        for w in s:
            if gold.get(w) in {"NN", "VB", "JJ"}:
                by_tag[gold[w]][w] += 1
    return dict(all_sents=all_sents, vocab=vocab, keep=lambda s: True, train=train, seen=seen, val=val, lm_held=lm_held, pairs=pairs,
                held=held, corpus=all_sents, gold=gold, patterns=patterns, by_tag=by_tag, CONTENT={"NN", "VB", "JJ"})


def install_nltk_stub():
    import nltk
    nltk.pos_tag = lambda toks, *a, **k: [(w, TAG.get(w, "NN")) for w in toks]


# ----------------------------------------------------------------------------------------------- controls
def swap_same_category(s, rng, k):
    """Replace k of the content words (adjective, noun, verb) by a different word of the same syntactic category (so the sentence stays
    in the grammar's syntax), ignoring semantics."""
    s = list(s)
    pos = [i for i, w in enumerate(s) if LEX[w][0] in ("n", "adj", "v")]
    bycat = defaultdict(list)
    for w in LEX:
        bycat[syncat(w)].append(w)
    for i in rng.sample(pos, min(k, len(pos))):
        cands = [w for w in bycat[syncat(s[i])] if w != s[i]]
        s[i] = rng.choice(cands)
    return s


def controls(out):
    rng = random.Random(5)
    N = 2000
    res = {}
    base = [sample_sentence(rng) for _ in range(N)]

    def tab(name, sents):
        c = Counter(judge(s)[0] for s in sents)
        d = Counter(judge(s)[1] for s in sents if judge(s)[0] != "sem")
        res[name] = {"n": len(sents), **{r: round(c[r] / len(sents), 3) for r in ("sem", "syn", "non")}, "reasons": dict(d.most_common(6))}
        print(json.dumps({name: res[name]}), flush=True)

    tab("grammar samples", base)
    tab("same-category swap, 1 word", [swap_same_category(s, rng, 1) for s in base])
    tab("same-category swap, 2 words", [swap_same_category(s, rng, 2) for s in base])
    tab("same-category swap, 3 words", [swap_same_category(s, rng, 3) for s in base])
    sh = []
    for s in base:
        for _ in range(20):
            p = rng.sample(s, len(s))
            if p != s:
                break
        sh.append(p)
    tab("word shuffle", sh)
    tab("agreement broken (verb form flipped)", [[(_flip(w) if LEX[w][0] == "v" else w) for w in s] for s in base])
    tab("random words (same length)", [[rng.choice(sorted(LEX)) for _ in s] for s in base])
    res["lexicon"] = {"words": len(LEX), "nouns": sum(1 for e in LEX.values() if e[0] == "n"), "verbs": sum(1 for e in LEX.values() if e[0] == "v"),
                      "adjectives": len(ADJ), "determiners": len(DETS), "prepositions": len(PREPS)}
    res["lengths"] = dict(sorted(Counter(len(s) for s in base).items()))
    res["distinct sentences in 20000 samples"] = len({tuple(sample_sentence(rng)) for _ in range(20000)})
    print(json.dumps(res["lexicon"]), json.dumps(res["lengths"]), res["distinct sentences in 20000 samples"], flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)


def _flip(w):
    v = LEX[w][1]
    return VERBS[v][0] if LEX[w][2] == "pl" else v


# ----------------------------------------------------------------------------------------------- judging dumps
def read_arm(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    return rows[0]["arm"], rows[1:]


def judge_dirs(out, dirs):
    res = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*_arm*.jsonl"))):
            arm, rows = read_arm(p)
            if not rows:
                continue
            js = [judge(r["gen"].split()) for r in rows]
            c = Counter(j[0] for j in js)
            nv = [(r["novel"], j[0]) for r, j in zip(rows, js)]
            cn = Counter(j for nov, j in nv if nov)
            nn = sum(cn.values())
            sh = lambda cc, tot: {k: round(cc[k] / tot, 3) if tot else None for k in ("sem", "syn", "non")}
            res[p] = {"arm": arm, "n": len(rows), "all": sh(c, len(rows)), "novel share": round(nn / len(rows), 3),
                      "among novel": sh(cn, nn), "n novel": nn,
                      "syn reasons": dict(Counter(j[1] for j in js if j[0] == "syn").most_common(5)),
                      "non reasons": dict(Counter(j[1] for j in js if j[0] == "non").most_common(5))}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    for p, r in res.items():
        print(json.dumps({"file": p, **r}), flush=True)


# ----------------------------------------------------------------------------------------------- purity
def purity(out, files):
    res = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        names = d["names"]
        r = {}
        for lvl, nm in names.items():
            cls = defaultdict(list)
            for w, c in nm.items():
                if w in LEX:
                    cls[c].append(w)
            same_syn = tot = same_sem_all = 0
            ss_syn = ss_sem = 0                        # among pairs sharing the syntactic category: same semantic type
            sem_pairs = sem_same = 0                   # pairs of content words: same semantic type
            for ws in cls.values():
                for i in range(len(ws)):
                    for j in range(i + 1, len(ws)):
                        a, b = ws[i], ws[j]
                        tot += 1
                        sy = syncat(a) == syncat(b)
                        same_syn += sy
                        ta, tb = semtype(a), semtype(b)
                        if ta is not None and tb is not None and ta[0] == tb[0]:
                            sem_pairs += 1
                            sem_same += ta == tb
                        if sy and ta is not None and tb is not None:
                            ss_syn += 1
                            ss_sem += ta == tb
            res_l = {"classes": len(cls), "mean size": round(len(nm) / max(len(cls), 1), 1), "pairs": tot,
                     "syn purity (class mates, same syntactic category)": round(same_syn / tot, 3) if tot else None,
                     "sem purity (content class mates of the same kind, same semantic type)": round(sem_same / sem_pairs, 3) if sem_pairs else None,
                     "sem purity within syntactic category": round(ss_sem / ss_syn, 3) if ss_syn else None, "n_sem_pairs": sem_pairs}
            r[lvl] = res_l
        # chance levels over all word pairs
        ws = [w for w in d["names"][next(iter(d["names"]))] if w in LEX]
        n_ = tp = sp = ss = ssp = 0
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                tp += 1
                sy = syncat(a) == syncat(b)
                n_ += sy
                ta, tb = semtype(a), semtype(b)
                if ta is not None and tb is not None and ta[0] == tb[0]:
                    sp += 1
                    ss += ta == tb
                if sy and ta is not None and tb is not None:
                    ssp += 1
        res[f] = {"levels": r, "chance syn purity (all pairs)": round(n_ / tp, 3), "chance sem purity (same kind pairs)": round(ss / sp, 3) if sp else None,
                  "types": len(ws)}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    for f, v in res.items():
        print(f, "chance syn", v["chance syn purity (all pairs)"], "chance sem", v["chance sem purity (same kind pairs)"], flush=True)
        for lvl, x in v["levels"].items():
            print("  ", lvl, x["classes"], x["syn purity (class mates, same syntactic category)"], x["sem purity (content class mates of the same kind, same semantic type)"],
                  x["sem purity within syntactic category"], flush=True)


def examples(path, k, seed):
    arm, rows = read_arm(path)
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        j = judge(r["gen"].split())
        by[j[0]].append((r["gen"], j[1], r["novel"]))
    out = {}
    for reg in ("sem", "syn", "non"):
        nov = [x for x in by[reg] if x[2]]
        pool = nov if len(nov) >= k else by[reg]
        out[reg] = rng.sample(pool, min(k, len(pool)))
    print(json.dumps({"arm": arm, "examples": out}))




def table(path):
    """Compact dial table from a judge json: one line per arm."""
    res = json.load(open(path, encoding="utf-8"))
    print("dir | family | b | W' | n | sem | syn | non | novel share | novel sem | novel syn | novel non")
    for p, r in res.items():
        a = r["arm"]
        fam = "C6" if "wgate" in a or a.get("wemit") is not None else "C9"
        b = a.get("wemit") if fam == "C6" else a.get("cbits")
        x = (p.split("/")[-2], fam, str(b), a.get("wreach"), r["n"], r["all"]["sem"], r["all"]["syn"], r["all"]["non"],
             r["novel share"], r["among novel"]["sem"], r["among novel"]["syn"], r["among novel"]["non"])
        print(" | ".join(str(v) for v in x))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "controls":
        controls(sys.argv[2])
    elif cmd == "judge":
        judge_dirs(sys.argv[2], sys.argv[3:])
    elif cmd == "purity":
        purity(sys.argv[2], sys.argv[3:])
    elif cmd == "table":
        table(sys.argv[2])
    elif cmd == "examples":
        examples(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
