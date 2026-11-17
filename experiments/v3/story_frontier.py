"""Grammaticality vs novelty depth (r184): for each judged story dump, over the NEW generations (not a stored
training sentence), the judged-grammatical share (P(yes) > 0.5, the judge's threshold) and how far each generation
departs from the training sentences:

  copy        longest span copied verbatim from one training sentence, as a fraction of the generation's length (mean)
  tri / bi    share with at least one unstored trigram / bigram (sentence padded with <s>, </s>)
  gram|tri    judged-grammatical share among new sentences with an unstored trigram (gram|bi: bigram)

Reads <dump dir>/*_s{seed}_n{n}_arm{i}.jsonl and the .scores.json beside each (t3_judge.py), and the judge's
real / scrambled calibration from <dump dir>/judge.json. The training sentences are rebuilt with t3_story.py's
stream lines (copied; novelty is cross-checked against each row's own "novel" flag).

Usage: story_frontier.py <out.jsonl> <dump dir> [<dump dir> ...]     (compute node: reads the TinyStories file)
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.stories import WORD, default_tinystories_path, read_sentences  # noqa: E402

_STREAM = {}


def train_sentences(seed, n, n_test_stories=1500):
    """The first n kept sentences of the seed's stream, as t3_story.py builds them."""
    if "stories" not in _STREAM:
        with open(default_tinystories_path(), encoding="utf-8") as f:
            raw = f.read()
        stories = []
        for st in raw.split("<|endoftext|>"):
            st = re.sub(r"\s+", " ", st.replace("’", "'").replace("‘", "'"))
            sents = [w for w in (WORD.findall(m.group(0).lower()) for m in re.finditer(r"[^.!?]+[.!?]", st)) if w]
            if sents:
                stories.append(sents)
        all_sents = read_sentences()
        _STREAM["stories"] = stories
        _STREAM["vocab"] = {w for w, _ in Counter(w for s in all_sents for w in s).most_common(250)}
    vocab = _STREAM["vocab"]
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    stories = list(_STREAM["stories"])
    random.Random(seed).shuffle(stories)
    train_st = stories[n_test_stories:]
    train = []
    for st in train_st:
        for s in st:
            if keep(s):
                train.append(tuple(s))
        if len(train) >= n:
            break
    return train[:n]


def grams(s, k):
    p = ("<s>",) + tuple(s) + ("</s>",)
    return [p[i:i + k] for i in range(len(p) - k + 1)]


class Train:
    def __init__(self, sents):
        self.whole = set(sents)
        self.sub = set()                      # every contiguous span of a training sentence
        for s in sents:
            for i in range(len(s)):
                for j in range(i + 1, len(s) + 1):
                    self.sub.add(s[i:j])
        self.tri = {g for s in sents for g in grams(s, 3)}
        self.bi = {g for s in sents for g in grams(s, 2)}

    def longest_copy(self, s):
        best = 0
        for i in range(len(s)):
            for j in range(len(s), i + best, -1):
                if s[i:j] in self.sub:
                    best = j - i
                    break
        return best


def mean(v):
    return sum(v) / len(v) if v else None


def r3(x):
    return None if x is None else round(x, 3)


def main(out, dirs):
    cache = {}
    rows_out = []
    for d in dirs:
        d = d.rstrip("/")
        try:
            calib = json.load(open(os.path.join(d, "judge.json"), encoding="utf-8")).get("references", {})
        except (OSError, ValueError):
            calib = {}
        for f in sorted(glob.glob(os.path.join(d, "*_arm*.jsonl"))):
            m = re.search(r"_s(\d+)_n(\d+)_arm(\d+)\.jsonl$", f)
            sc = f[:-len(".jsonl")] + ".scores.json"
            if not m or not os.path.exists(sc):
                continue
            seed, n, ai = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if (seed, n) not in cache:
                cache[(seed, n)] = Train(train_sentences(seed, n))
            T = cache[(seed, n)]
            lines = [json.loads(l) for l in open(f, encoding="utf-8")]
            arm, rows = lines[0]["arm"], lines[1:]
            ps = json.load(open(sc, encoding="utf-8"))
            assert len(ps) == len(rows), (f, len(ps), len(rows))
            new = []
            mismatch = 0
            for r, p in zip(rows, ps):
                s = tuple(r["gen"].split())
                nv = s not in T.whole
                mismatch += nv != r["novel"]
                if nv:
                    new.append((s, p > 0.5))
            cp = [T.longest_copy(s) / len(s) for s, _ in new]
            tri = [any(g not in T.tri for g in grams(s, 3)) for s, _ in new]
            bi = [any(g not in T.bi for g in grams(s, 2)) for s, _ in new]
            ok = [g for _, g in new]
            cal = calib.get(f"{os.path.basename(f).split('_arm')[0]}_refs.jsonl", {})
            rec = {"dir": os.path.basename(d), "file": os.path.basename(f), "seed": seed, "n": n,
                   "arm": json.dumps(arm, sort_keys=True), "generated": len(rows), "new": len(new),
                   "novel flag mismatches": mismatch,
                   "novel judged grammatical": r3(mean(ok)),
                   "copy frac": r3(mean(cp)), "unstored trigram": r3(mean(tri)), "unstored bigram": r3(mean(bi)),
                   "gram | unstored trigram": r3(mean([a for a, t in zip(ok, tri) if t])),
                   "gram | unstored bigram": r3(mean([a for a, t in zip(ok, bi) if t])),
                   "n tri": sum(tri), "n bi": sum(bi),
                   "calibration real": cal.get("real"), "calibration scrambled": cal.get("word order scrambled")}
            rows_out.append(rec)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    # summary: mean over seeds per (dir, arm, n)
    g = defaultdict(list)
    for r in rows_out:
        g[(r["dir"], r["arm"], r["n"])].append(r)
    cols = ["novel judged grammatical", "copy frac", "unstored trigram", "unstored bigram",
            "gram | unstored trigram", "gram | unstored bigram"]
    print("\n%-14s %-52s %5s %5s | %s" % ("dir", "arm", "n", "seeds", " ".join(c[:11].rjust(11) for c in cols)))
    for k in sorted(g):
        rs = g[k]
        vals = [mean([r[c] for r in rs if r[c] is not None]) for c in cols]
        print("%-14s %-52s %5d %5d | %s" % (k[0], k[1][:52], k[2], len(rs),
                                             " ".join(("%11.3f" % v if v is not None else "%11s" % "-") for v in vals)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
