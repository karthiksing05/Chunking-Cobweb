"""Round 213 (language): chunk inventory, concepts and held-out parses of the current TRELLIS recipe, per seed, along the stream.

Recipe = the r204/r205 edge recipe's learner (cfg of logs/v3_r204_grid.sh with window 7), learning the seed's n=1000 stream
(t3_story.py's stream: seed-shuffled stories, 1500 test stories dropped, follows = previous sentence kept). Sentences that
equal a held-out sentence of the shared set (results/silver/heldout_s13.json) are skipped, as in t3_parse_consistency.py,
so the 400 held-out sentences are unseen by every seed. Parses use the same learned memory, default parse (L.parse, chosen tree).

Per seed it saves (experiments/v3/results/story_gen/r213/lang_s<seed>.json):
  pos      chunk string -> the stream indices where it was a bracket (size >= 2, not the whole sentence) of the learn() tree
  stored   word strings of every composite record at each checkpoint
  concept  at each checkpoint, for chunks seen >= 3 times: the content-concept path [(id, bits)], and per path node the
           member strings by weight (the concept's content)
  parses   at each checkpoint, for each held-out sentence: tree string, binary brackets, chosen chunks with posteriors
  gold     NLTK shallow-phrase brackets (story_parse_eval.py definition) of the held-out sentences
  bpe      word-level BPE merges (story_parse_eval.learn_bpe) on each checkpoint's prefix, budget = distinct bracketed strings

Usage (compute node, repo root): r213_lang.py <seed> <n> <checkpoints e.g. 250,500,1000>
"""
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
from trellis2.stories import WORD, default_tinystories_path, heldout_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402
from story_parse_eval import learn_bpe  # noqa: E402

seed, n = int(sys.argv[1]), int(sys.argv[2])
checks = [int(x) for x in sys.argv[3].split(",")]
CFG = json.loads('{"children":false,"spine":false,"heights":false,"naming":"bands","tree":"cu","refresh":"verify","refresh_budget":0,'
                 '"store":"posterior","restructure":"sample","type_names":false,"word_cc":false,"repair_path":true,'
                 '"repair_use":700,"exact_alone":"content","story_facet":true,"window":7}')
OUT = "experiments/v3/results/story_gen/r213"
os.makedirs(OUT, exist_ok=True)

held = [tuple(s) for s in json.load(open(heldout_path(13), encoding="utf-8"))]
held_set = set(held)
with open(default_tinystories_path(), encoding="utf-8") as f:
    raw = f.read()
stories = []
for st in raw.split("<|endoftext|>"):
    st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
    sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
    if sents:
        stories.append(sents)
vocab = {w for w, _ in Counter(w for s in read_sentences() for w in s).most_common(250)}
keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
random.Random(seed).shuffle(stories)
train = []
for st in stories[1500:]:
    for i, s in enumerate(st):
        if keep(s) and tuple(s) not in held_set:
            train.append((tuple(s), i > 0 and keep(st[i - 1])))
    if len(train) >= n:
        break
train = train[:n]
del raw, stories


def gold_brackets(sentences):
    import nltk
    chunker = nltk.RegexpParser(r"""
      NP: {<DT|PRP\$|POS|CD|PDT>*<JJ.*|RBS|RBR>*<NN.*>+}
          {<PRP|EX>}
      AP: {<RB.*>*<JJ.*>+}
      VG: {<MD|TO>?<RB.*>*<VB.*>+(<RB.*>*<VB.*>+)*<RP>?}
      PP: {<IN|TO><NP>}
      VP: {<VG><NP|PP|AP|RB.*>+}
    """)
    out = []
    for words in sentences:
        tagged = nltk.pos_tag(["I" if w == "i" else w for w in words])
        tree = chunker.parse(tagged)
        spans = defaultdict(set)

        def walk(t, i):
            if isinstance(t, nltk.Tree):
                j = i
                for c in t:
                    j = walk(c, j)
                if t.label() in ("NP", "PP", "VP", "AP", "VG"):
                    spans[(i, j)].add(t.label())
                return j
            return i + 1
        walk(tree, 0)
        out.append([[i, j, "/".join(sorted(v))] for (i, j), v in sorted(spans.items()) if 2 <= j - i < len(words)])
    return out


def words_of(e):
    if e.token is not None:
        return [e.token]
    return words_of(e.left) + words_of(e.right)


def path_of(node):
    out = [node]
    while out[-1].parent is not None:
        out.append(out[-1].parent)
    return out[::-1]


L = Trellis3(seed=seed, **CFG)
pos = defaultdict(list)
res = {"seed": seed, "n": n, "cfg": CFG, "checkpoints": checks, "held": [list(s) for s in held], "train": [list(s) for s, _ in train],
       "gold": gold_brackets(held), "pos": {}, "stored": {}, "concept": {}, "parses": {}, "bpe": {}, "info": {}}
t0, done = time.time(), 0
for cp in checks:
    for idx in range(done, cp):
        s, fol = train[idx]
        tree = L.learn(list(s), follows=fol)
        for (i, j) in tree.brackets():
            if j - i >= 2 and (j - i) < len(s):
                pos[" ".join(s[i:j])].append(idx)
    done = cp
    # held-out parses with the same memory
    L._quiet, pv = True, L._pv
    L._pv = None
    ps = []
    for s in held:
        r = L.parse(list(s))
        sp = defaultdict(float)
        for (a, k, b), p in r["post"].items():
            sp[(a, b)] += p
        ch = [[i, j, round(sp[(i, j)], 3)] for i, _, j in r["chosen"]]
        ps.append({"tree": r["tree"].to_string(list(s)), "brackets": sorted([list(b) for b in r["tree"].brackets()]), "chunks": ch})
    L._quiet, L._pv = False, pv
    res["parses"][str(cp)] = ps
    # composite strings and their content concepts
    under, nodes, bys = defaultdict(Counter), {}, defaultdict(list)
    tot = max(L.cnt.root.count, 1e-9)
    for e in L.records:
        if e.token is None and e.cleaf is not None and e.weight > 0:
            st = " ".join(words_of(e))
            for node in path_of(e.cleaf):
                under[node.id][st] += e.weight
                nodes[node.id] = node
            bys[st].append(e)
    bits = lambda nd: -math.log2(max(nd.count, 1e-9) / tot)
    stored = sorted({" ".join(words_of(e)) for e in L.records if e.token is None})
    cur = {k: [x for x in v if x < cp] for k, v in pos.items()}
    tracked = {k for k, v in cur.items() if len(v) >= 1} | {" ".join(s[i:j]) for s, p_ in zip(held, ps) for i, j, _ in p_["chunks"] if j - i < len(s)}
    conc, used = {}, {}
    for st in tracked:
        if st not in bys:
            continue
        e = max(bys[st], key=lambda r: r.weight)
        p = path_of(e.cleaf)
        conc[st] = [[nd.id, round(bits(nd), 2)] for nd in p[1:]]
        for nd in p[1:]:
            used[nd.id] = {"bits": round(bits(nd), 2), "members": len(under[nd.id]),
                           "top": [[a, round(w, 2)] for a, w in under[nd.id].most_common(14)]}
    res["stored"][str(cp)] = stored
    res["concept"][str(cp)] = {"path": conc, "nodes": used}
    res["info"][str(cp)] = {"records": len(L.records), "composite strings": len(stored), "bracketed strings": len(cur),
                            "learn s": round(time.time() - t0)}
    # BPE on the same prefix, budget = distinct bracketed strings of TRELLIS at this checkpoint
    merges = learn_bpe([list(s) for s, _ in train[:cp]], max(len(cur), 1))
    res["bpe"][str(cp)] = [" ".join(m[0] + m[1]) for m in merges]
    print(json.dumps({"seed": seed, "cp": cp, **res["info"][str(cp)]}), flush=True)
    res["pos"] = dict(pos)
    json.dump(res, open(f"{OUT}/lang_s{seed}.json", "w", encoding="utf-8"))
print("done", seed, round(time.time() - t0))
