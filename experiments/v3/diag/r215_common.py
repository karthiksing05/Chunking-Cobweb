"""Round 215 shared helpers: the story stream (as diag/t3_story.py builds it), the control sets, AUC and operating points."""
import json
import os
import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
import story_frontier as sf  # noqa: E402

OUT = "experiments/v3/results/story_parse_eval"
SETS = os.path.join(OUT, "r215_sets")
RAW = os.path.join(OUT, "r215_raw")
SEEDS = (13, 17, 19, 23)
NS = (125, 250, 500, 1000, 2000, 4000)
# paper recipe (r204/r205/r212 generation jobs): S + D0 + story_facet + window; window per n = the r205 selection
# (250 w7, 500 w6, 1000 w7, 4000 w5); 125 and 2000 have no r205 selection: w7 (as 250) and w6 (between 1000 and 4000) are assumed.
WINDOW = {125: 7, 250: 7, 500: 6, 1000: 7, 2000: 6, 4000: 5}
CFG = {"children": False, "spine": False, "heights": False, "naming": "bands", "tree": "cu", "refresh": "verify", "refresh_budget": 0,
       "store": "posterior", "restructure": "sample", "type_names": False, "word_cc": False, "repair_path": True, "repair_use": 700,
       "exact_alone": "content", "story_facet": True}
# classes (key -> description); the grammatical reference of each is C1 (same sentences), except C4 whose own sources are C4_src
CLASSES = ["C3scr", "C3adj", "C4v", "C2full", "C2one"]
NAMES = {"C3scr": "full scramble", "C3adj": "one adjacent swap across POS", "C4v": "agreement error (was/were, is/are, has/have), error word in the 250-word vocabulary",
         "C2full": "all content words swapped within POS (syntactic, not semantic)",
         "C2one": "one content word swapped within POS (syntactic, not semantic)"}


def stream(seed, n, follow="story", n_test=1500):
    """(train [(sentence, follows)], lm_held, test_stories, vocab), as t3_story.py (FOLLOW=story, r212 jobs)."""
    sf.train_sentences(seed, 10)
    vocab, stories = sf._STREAM["vocab"], list(sf._STREAM["stories"])
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    random.Random(seed).shuffle(stories)
    test, train_st = stories[:n_test], stories[n_test:]
    train, k = [], 0
    for k, st in enumerate(train_st):
        for i, s in enumerate(st):
            if keep(s):
                train.append((s, (i > 0 and keep(st[i - 1])) if follow == "adjacent" else any(keep(t) for t in st[:i])))
        if len(train) >= n:
            break
    train = train[:n]
    lm_held = [s for st in train_st[k + 1:k + 400] for s in st if keep(s)]
    return train, lm_held, test, vocab


def load_sets(seed):
    return json.load(open(os.path.join(SETS, "sets_s%d.json" % seed), encoding="utf-8"))


def all_sentences(S):
    """name -> list of sentence strings, in the file's order."""
    return {k: S[k] for k in ("Gcal", "C1", "C2full", "C2one", "C3scr", "C3adj", "C3adj_src", "C4", "C4_src", "C4v", "C4v_src")}


def auc(pos, neg):
    """P(pos score > neg score) + .5 P(tie); pos = grammatical (should score higher)."""
    if not pos or not neg:
        return None
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    r, i, tot = 0.0, 0, 0.0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        mid = (i + j + 1) / 2.0           # average 1-based rank of the tie block
        r += mid * sum(1 for t in s[i:j] if t[1])
        i = j
    return (r - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def paired_auc(a, b):
    """share of items whose grammatical score exceeds the corrupted one (ties .5)."""
    return sum((x > y) + .5 * (x == y) for x, y in zip(a, b)) / len(a) if a else None


def threshold(cal, p=0.9):
    """score thr such that a share p of the calibration scores is >= thr (success = score >= thr)."""
    v = sorted(cal)
    k = int((1 - p) * len(v))              # the k lowest fail
    return v[k] if k < len(v) else v[-1]


def succ(xs, thr):
    return sum(x >= thr for x in xs) / len(xs) if xs else None
