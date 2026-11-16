"""How story chunking changes with learning: which phrases the chunks of TinyStories encode.

Trellis3 learns kept TinyStories sentences in story order (the stream, held-out stories
and ``follows`` of t3_story.py with FOLLOW "story"). At each checkpoint it records:

  window     every chunk committed since the previous checkpoint: a composite (2+ words) of
             the analysis memory chose for the sentence (``learn``'s tree) that was stored as
             a record; labelled with the phrase labels below
  chance     the same labels on size-matched spans: "random" (a random contiguous span of
             the same length from the same sentence) and "pos" (a span of the same length
             starting on a word of the same coarse tag as the chunk's first word, drawn from
             any window sentence of the same length), so a lift > 1 means memory prefers
             the label (whole sentence: lift 1 by construction)
  held-out   300 held-out sentences parsed (no learning): the chosen composites other than
             the whole sentence (as t3_chunks_dump.py and the site's frag_chunks list them);
             "recognised" = those whose span posterior (summed over splits) >= theta, i.e.
             most of the analyses inside-outside weighs contain the span
  top        the most frequent chunks committed in the window, with their labels
  cases      a few held-out sentences with their chosen chunks (and posteriors) per checkpoint

Gold (scoring only; the learner never sees it): nltk.pos_tag on the lowercased sentence ("i"
passed as "I"; names tag as NN, some tags are wrong, e.g. "that" as IN), then nltk.RegexpParser
with a shallow cascade, in this order:
  NP  noun phrase    determiners/possessives/numbers, adjectives, 1+ nouns (proper nouns
                     included); or a pronoun / EX
  AP  adjective      adverbs + 1+ adjectives not inside an NP ("very happy", "not sad")
  VG  verb group     optional MD/TO, adverbs, 1+ verbs (adverbs between verbs allowed),
                     optional particle/adverb ("did not like", "to play", "ran away")
  PP  prepositional  IN/TO + NP
  VP  verb phrase    VG + 1+ complements (NP, PP, AP or adverb)
Labels of a span (i, j) (not exclusive):
  NP / PP / VP / AP  exactly an NP / PP / VP / AP span (VP needs a complement, so never just a VG)
  verb group         exactly a VG span
  subject+verb       an NP span immediately followed by a VG span ("she was", "the dog ran")
  inside one phrase  strictly within one NP/PP/VP/AP/VG span and not exactly any phrase
  crosses a boundary partially overlaps some phrase span (neither inside nor containing it),
                     other than a subject+verb span (which always cuts its VP)
  whole sentence     the whole sentence

Usage: story_chunks.py <cfg json> <seed> <checkpoints, e.g. 250,500,1000,2000,4000> <tag> [n_cases theta]
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import nltk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402

cfg, seed = json.loads(sys.argv[1]), int(sys.argv[2])
checkpoints = [int(x) for x in sys.argv[3].split(",")]
tag = sys.argv[4]
n_cases = int(sys.argv[5]) if len(sys.argv) > 5 else 6
theta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
n = checkpoints[-1]
N_TEST_STORIES, n_test = 1500, 300

# the stream of t3_story.py (FOLLOW "story"), copied: importing it would run its measurements
with open(default_tinystories_path(), encoding="utf-8") as f:
    raw = f.read()
stories = []
for st in raw.split("<|endoftext|>"):
    st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
    sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
    if sents:
        stories.append(sents)
all_sents = read_sentences()
vocab = {w for w, _ in Counter(w for s in all_sents for w in s).most_common(250)}
keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
random.Random(seed).shuffle(stories)
test, train_st = stories[:N_TEST_STORIES], stories[N_TEST_STORIES:]
train = []
for st in train_st:
    for i, s in enumerate(st):
        if keep(s):
            train.append((s, any(keep(t) for t in st[:i])))
    if len(train) >= n:
        break
train = train[:n]
held = [s for st in test for s in st if keep(s)]
random.Random(seed + 3).shuffle(held)
held = held[:n_test]
del raw, all_sents

LABELS = ["NP", "PP", "VP", "AP", "verb group", "subject+verb", "inside one phrase", "crosses a boundary", "whole sentence"]
CHUNKER = nltk.RegexpParser(r"""
  NP: {<DT|PRP\$|POS|CD|PDT>*<JJ.*>*<NN.*>+}
      {<PRP|EX>}
  AP: {<RB.*>*<JJ.*>+}
  VG: {<MD|TO>?<RB.*>*<VB.*>+(<RB.*>*<VB.*>+)*<RP|RB.*>?}
  PP: {<IN|TO><NP>}
  VP: {<VG><NP|PP|AP|RB.*>+}
""")


def phrases(words):
    """Coarse tags and the phrase spans {(i, j): set of NP/PP/VP/VG} of a sentence."""
    tagged = nltk.pos_tag(["I" if w == "i" else w for w in words])     # lowercased "i" tags as NN otherwise
    tree = CHUNKER.parse(tagged)
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
    return [t[:2] for _, t in tagged], dict(spans)


def labels(i, j, n_words, spans):
    found = set()
    here = spans.get((i, j), set())
    for lab, name in (("NP", "NP"), ("PP", "PP"), ("VP", "VP"), ("AP", "AP"), ("VG", "verb group")):
        if lab in here:
            found.add(name)
    if any("NP" in spans.get((i, k), ()) and "VG" in spans.get((k, j), ()) for k in range(i + 1, j)):
        found.add("subject+verb")
    if not here and any(a <= i and j <= b and (a, b) != (i, j) for a, b in spans):
        found.add("inside one phrase")
    if "subject+verb" not in found and any(a < j and i < b and not (a <= i and j <= b) and not (i <= a and b <= j)
                                           for a, b in spans):
        found.add("crosses a boundary")
    if (i, j) == (0, n_words):
        found.add("whole sentence")
    return found


rng = random.Random(seed + 2)


def rate(items):
    """items: [(i, j, n_words, spans)] -> share carrying each label, count."""
    m = len(items)
    c = Counter(x for it in items for x in labels(*it))
    return {k: round(c[k] / m, 4) if m else None for k in LABELS}, m


lift = lambda a, b: {k: (round(a[k] / b[k], 3) if a[k] is not None and b[k] else None) for k in LABELS}
G = {}                                                   # sentence index -> (tags, spans), train and held-out
for idx, (s, _) in enumerate(train):
    G[("t", idx)] = phrases(s)
for idx, s in enumerate(held):
    G[("h", idx)] = phrases(s)

L = Trellis3(seed=seed, **cfg)
out = {"tag": tag, "seed": seed, "cfg": cfg, "theta": theta, "held-out sentences": len(held), "labels": LABELS,
       "checkpoints": []}
done, t0 = 0, time.time()
cases = {j: [] for j in range(min(n_cases, len(held)))}
for cp in checkpoints:
    window, ch_rand, sents_w = [], [], []
    for idx in range(done, cp):
        s, fol = train[idx]
        a0 = len(L.records)
        tree = L.learn(s, follows=fol)
        stored = {(e.span[0], e.split, e.span[1]) for e in L.records[a0:] if e.token is None}
        tags, spans = G[("t", idx)]
        sents_w.append(idx)
        for (i, j), k in tree.split.items():
            if (i, k, j) in stored:
                window.append((idx, i, j))
                a = rng.randrange(0, len(s) - (j - i) + 1)
                ch_rand.append((a, a + j - i, len(s), spans))
    done = cp
    # pos chance: same span length, start on a word of the same coarse tag, in any window
    # sentence of the same length (so the chunk's own sentence always qualifies)
    starts = defaultdict(list)
    for idx in sents_w:
        for a, t in enumerate(G[("t", idx)][0]):
            starts[(t, len(train[idx][0]))].append((idx, a))
    ch_pos = []
    for idx, i, j in window:
        t = G[("t", idx)][0][i]
        cand = [(x, a) for x, a in starts[(t, len(train[idx][0]))] if a + j - i <= len(train[x][0])]
        x, a = rng.choice(cand)
        ch_pos.append((a, a + j - i, len(train[x][0]), G[("t", x)][1]))
    win_items = [(i, j, len(train[idx][0]), G[("t", idx)][1]) for idx, i, j in window]
    # held-out parses
    L._quiet, pv = True, L._pv
    L._pv = None
    chosen_h, rec_h = [], []
    t1 = time.time()
    for h, s in enumerate(held):
        r = L.parse(s)
        sp = defaultdict(float)
        for (a, k, b), p in r["post"].items():
            sp[(a, b)] += p
        spans = G[("h", h)][1]
        ch = [(i, j, round(sp[(i, j)], 3)) for i, k, j in r["chosen"] if (i, j) != (0, len(s))]
        chosen_h += [(i, j, len(s), spans) for i, j, _ in ch]
        rec_h += [(i, j, len(s), spans) for i, j, p in ch if p >= theta]
        if h in cases:
            cases[h].append({"learned": cp, "chunks": [{"span": [i, j], "words": " ".join(s[i:j]), "posterior": p,
                                                        "recognised": p >= theta,
                                                        "labels": sorted(labels(i, j, len(s), spans))}
                                                       for i, j, p in sorted(ch, key=lambda t: (t[0], -t[1]))]})
    L._quiet, L._pv = False, pv
    rw, nw = rate(win_items)
    rr, _ = rate(ch_rand)
    rp, _ = rate(ch_pos)
    rc, nc = rate(chosen_h)
    rh, nh = rate(rec_h)
    top = Counter(tuple(train[idx][0][i:j]) for idx, i, j in window).most_common(20)
    first = {}
    for idx, i, j in window:
        first.setdefault(tuple(train[idx][0][i:j]), (i, j, len(train[idx][0]), G[("t", idx)][1]))
    row = {"learned": cp, "seconds": round(time.time() - t0), "held-out parse seconds": round(time.time() - t1),
           "records": len(L.records), "window chunks": nw, "window sentences": len(sents_w),
           "mean size": round(sum(j - i for _, i, j in window) / max(1, nw), 3),
           "size counts": dict(sorted(Counter(j - i for _, i, j in window).items())),
           "rate": rw, "chance random": rr, "chance pos": rp, "lift random": lift(rw, rr), "lift pos": lift(rw, rp),
           "held-out chosen per sentence": round(nc / len(held), 3), "held-out chosen rate": rc,
           "held-out recognised per sentence": round(nh / len(held), 3),
           "held-out mean size": round(sum(j - i for i, j, _, _ in rec_h) / max(1, nh), 3), "held-out rate": rh,
           "top": [{"words": " ".join(g), "count": c, "labels": sorted(labels(*first[g]))} for g, c in top]}
    out["checkpoints"].append(row)
    print(json.dumps({k: row[k] for k in ("learned", "seconds", "held-out parse seconds", "window chunks", "mean size",
                                          "held-out recognised per sentence")}), flush=True)
    print(json.dumps({"rate": rw, "lift random": row["lift random"], "lift pos": row["lift pos"]}), flush=True)
    print(json.dumps({"held-out rate": rh, "top": [(t["words"], t["count"]) for t in row["top"][:10]]}), flush=True)
out["cases"] = [{"sentence": " ".join(held[h]), "tags": G[("h", h)][0],
                 "phrases": sorted([[a, b, sorted(v)] for (a, b), v in G[("h", h)][1].items()]),
                 "by checkpoint": cases[h]} for h in cases]
d = os.path.join(ROOT, "experiments", "v3", "results", "story_chunks")
os.makedirs(d, exist_ok=True)
json.dump(out, open(os.path.join(d, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print("wrote", os.path.join(d, f"{tag}_s{seed}.json"))
