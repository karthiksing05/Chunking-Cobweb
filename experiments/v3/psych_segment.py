"""TRELLIS simulation of statistical word-segmentation experiments (CogSci-style replication, 1:1 with the human procedures).

Procedures and human numbers: docs/SAN96_PROTOCOL.md (sections 1.1, 1.2, 2.3, 2.4) only.

Usage:
  psych_segment.py --exp f1|f2|f3|i1|i2 --cond <cond> --arm full|conceptsoff|tp --seeds 1,2,3 [--out DIR]
    f1 (Frank et al. 2010 Exp 1)  cond = words per sentence in 1,2,3,4,6,8,12,24          (600 tokens)
    f2 (Exp 2)                    cond = total tokens 48,100,300,600,900,1200             (4 words per sentence)
    f3 (Exp 3)                    cond = word types 3,4,5,6,9                             (4 words/sentence, 600 tokens)
    i1 (Saffran, Aslin & Newport 1996 Exp 1, words vs nonwords)   cond = A or B (Condition)
    i2 (SAN96 Exp 2, words vs part-words)                         cond = A or B
  arm tp = only the forward transitional-probability baseline (no learner; cheap).  Every run also stores the TP baseline.
  Each seed is one simulated participant (its own random lexicon / stream / test items); the learner seed is the same seed.
  Output: experiments/v3/results/psych_segment/<exp>_<cond>_<arm>_s<seed>.json (one file per participant).

One fixed learner (no per-experiment tuning), fixed a priori choices:
  * Learner = configuration C3 of the paper (posterior storage, window 7; cfg CFG below, the r218 / r225 generation-study learner),
    minus story_facet (it needs the previous sentence of a story; there is none here). Nothing else is dropped: the generator-only
    keys (temperature, pair gate, edge, causal, units, new) live in the generate_tree arm sets, not in the learner cfg, and are
    not used. The parser is the default parser (search "chart", readout "mbr").
  * Perceptual buffer W = 12 syllables. An utterance (a sentence; for infants the whole continuous stream) of at most W syllables
    is one experience with true boundaries at both ends. A longer utterance is cut into consecutive buffers of W syllables (the
    last may be shorter); a buffer edge that is a true pause is a boundary (BOS / EOS markers fill the window), a cut edge is
    NOT observed (those window slots are absent, as with boundary="missing", for that side only). Implemented in
    Trellis3Seg._surface (per-edge flags _le, _re); nothing in trellis3.py is modified.
  * Test score S(x) = total stored posterior weight of the composites whose token span equals x, summed over every experience
    and every split (Trellis3Seg.learn_buf adds each stored composite's weight to a dict keyed by its tokens). 2AFC choice by
    the Luce rule P(word) = S(w) / (S(w) + S(p)), .5 when both are 0; per participant the mean over pairs.
    Secondary score: recognition bits of x presented alone (true boundaries both sides, learning off):
    bits = _climb(cnt, _content(top)) + _climb(ctx, top.ctx); choice = higher bits, ties .5.
    Infants also: proportion better (S(w) - S(f)) / (S(w) + S(f)), mean over pairs; positive = words more familiar (humans: looking
    time is longer to the NOVEL foil, so a human .06 corresponds to a words-minus-foil familiarity difference of the SAME size
    only if familiarity is read as novelty preference; sign convention here is words more familiar = positive).
  * Arm conceptsoff = Trellis3R225(concepts_off=True, concepts_off_parse=True) (leaf-only reads for the parser's gain, as the
    paper's "concepts off (leaf reads)", C8 / Table S19 variant noconceptp: the parse that decides what is stored also reads
    leaves only), and the secondary score's _climb reads the leaf concept only (as experiments/v3/r228_cs_leaf.py).
  * Baseline TP (Frank et al. 2010): forward TP P(b|a) = count(a b) / count(a . ) with a sentence-boundary symbol # at both ends
    of every utterance (# counts in the denominators and as a successor), score of an item = product of its within-item
    syllable-to-syllable TPs (the # transitions are not multiplied in), Luce rule.

ASSUMPTIONS (explicit; also written to every output JSON as "assumption"):
  A1 Frank Exp 3: the doc gives no word lengths or syllable reuse. Lengths cycle 2,3,4 (3 types = 2,3,4; 6 types = 2,3,4,2,3,4,
     the same multiset as Exp 1's 2,2,3,3,4,4; 9 types = three of each). Syllables are sampled without replacement from the 18 while
     available; when the pool runs out it is refilled with the syllables not already in that word, so a syllable is reused only
     across words, never within a word. Exps 1 and 2 use lengths 2,2,3,3,4,4 with each of the 18 syllables once.
  A2 Token counts that do not divide evenly by the number of types (100 tokens / 6, 600 / 9 ...) give the first (tokens mod types)
     randomly chosen types one extra token. Word order is a random permutation with no adjacent repetition (also across
     sentence boundaries), cut into sentences of the condition's length.
  A3 Frank part-word = a proper end (s >= 1 syllables, fewer than the word's length) of one word + a proper beginning of a
     different word, same total syllable length as its word; it must occur inside some training sentence at least once. The
     number of pairs whose part-word never occurs (necessarily all of them with one-word sentences) is recorded as
     n_partword_unseen. Pair i uses word i mod n_types as target; the same part-word is not used in two consecutive pairs.
  A4 Infants: 4 words x 45 tokens in one stream (540 syllables), no adjacent repeat; true boundary only at stream start and end;
     buffers longer than W start at a random phase (first buffer 1..W syllables, rng seed*7919+1), so cuts do not fall on
     word onsets; the first pilot used fixed cuts at multiples of 12, which always fell on infant word boundaries (fixed 2026-11-21).
  A5 Infant tests: every word x foil pair (2 x 2 = 4 pairs per infant). Exp 1 foils = the other condition's words that are
     "nonwords" for this condition (syllables of this language in a novel order, TP 0); Exp 2 foils = two part-words drawn from
     the stream: one 2+1 (end 2 of word a, first 1 of word b) and one 1+2, both occurring in the stream, from different word pairs.
  A6 Simulated infants: 16 per experiment = 8 per condition (seeds 1..8 per condition).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from trellis2.trellis3 import BOS, EOS, LN2, log_prob  # noqa: E402
from trellis2.trellis3_r225 import Trellis3R225  # noqa: E402

SYLS = "ba bi da du ti tu ka ki la lu gi gu pa pi va vu zi zu".split()
W_BUF = 12
OUT = os.path.join("experiments", "v3", "results", "psych_segment")

# C3 learner (r218_gen.sh CFG / r225_launch.sh BASE) minus story_facet
CFG = {"children": False, "spine": False, "heights": False, "naming": "bands", "tree": "cu", "refresh": "verify",
       "refresh_budget": 0, "store": "posterior", "restructure": "sample", "type_names": False, "word_cc": False,
       "repair_path": True, "repair_use": 700, "exact_alone": "content", "window": 7}
DROPPED = {"story_facet": "needs the previous sentence of a story (learn(follows=...)); no sentence-to-sentence context here"}

ASSUMPTION = ("A1 Exp3 lengths cycle 2,3,4, syllables without replacement then reuse across words only; A2 uneven token counts "
              "spread randomly, no adjacent repeat; A3 part-words = proper end + proper beginning, must occur in a sentence "
              "(count of misses recorded); A4 buffers longer than W cut from a random phase (first buffer 1..W); "
              "A5 infant tests = all 2x2 word/foil pairs, Exp2 foils one 2+1 and one 1+2 part-word from the stream; "
              "A6 16 infants per experiment = 8 per condition. See the script docstring.")


# --------------------------------------------------------------------------------------------- learner
class Trellis3Seg(Trellis3R225):
    """Trellis3R225 + per-edge boundary observation + the weight of every stored composite by its token span.

    _le / _re: whether the left / right edge of the current experience is a true boundary. A cut edge leaves the window slots
    beyond it absent (not BOS / EOS), exactly as Trellis3._surface does for the other side. Defaults True (= Trellis3._surface).
    """
    _le = True
    _re = True

    def _surface(self, tokens, i, j) -> dict:
        x = {}
        for k in range(1, self.window + 1):
            if i - k >= 0:
                x[f"l{k}"] = tokens[i - k]
            elif self._le and self.boundary == "marker":
                x[f"l{k}"] = BOS
            if j + k - 1 < len(tokens):
                x[f"r{k}"] = tokens[j + k - 1]
            elif self._re and self.boundary == "marker":
                x[f"r{k}"] = EOS
        return x

    def learn_buf(self, tokens, left_edge=True, right_edge=True):
        self._le, self._re = left_edge, right_edge
        a0 = len(self.records)
        try:
            self.learn(tokens)
        finally:
            self._le = self._re = True
        span_w = self.__dict__.setdefault("span_w", defaultdict(float))
        for e in self.records[a0:]:
            if e.token is None:                       # composites (>= 2 tokens); words excluded
                span_w[tuple(tokens[e.span[0]:e.span[1]])] += e.weight

    def S(self, x):
        return self.__dict__.get("span_w", {}).get(tuple(x), 0.0)

    def recog_bits(self, x, leaf=False):
        """Bits of x presented alone (true boundaries both sides), learning off."""
        x = list(x)
        self._le = self._re = True
        self._quiet = True
        try:
            r = self.parse(x)
        finally:
            self._quiet = False
        top = [r["comps"][k] for k in r["chosen"] if (k[0], k[2]) == (0, len(x))]
        if not top:
            return 0.0
        e = top[0]
        if leaf:
            return self._climb_leaf(self.cnt, self._content(e)) + self._climb_leaf(self.ctx, e.ctx)
        return self._climb(self.cnt, self._content(e)) + self._climb(self.ctx, e.ctx)

    def _climb_leaf(self, tree, x):
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = tree.categorize(x)[-1]
        if node is root or node.count < self.support:
            return 0.0
        return (log_prob(node, x, root, self.alpha) - log_prob(root, x, root, self.alpha)) / LN2


def buffers(utt, W=W_BUF, rng=None):
    """[(tokens, left_edge_is_true, right_edge_is_true)] for one utterance.

    A longer utterance is cut every W syllables starting from a random phase: the first buffer has a length drawn
    uniformly from 1..W (rng given), so cuts do not line up with word onsets (fix of A4, 2026-11-21: with W = 12 and
    trisyllabic infant words, cuts at multiples of 12 always fell on word boundaries)."""
    n = len(utt)
    if n <= W:
        return [(utt, True, True)]
    out = []
    first = rng.randint(1, W) if rng is not None else W
    cuts = [0] + list(range(first, n, W)) + [n]
    for a, b in zip(cuts, cuts[1:]):
        out.append((utt[a:b], a == 0, b == n))
    return out


# --------------------------------------------------------------------------------------------- stimuli
def no_repeat_perm(counts, rng):
    """Random sequence with counts[t] copies of type t and no two equal neighbours."""
    while True:
        rem = dict(counts)
        seq, prev = [], None
        ok = True
        for _ in range(sum(counts.values())):
            cand = [t for t, c in rem.items() if c > 0 and t != prev]
            if not cand:
                ok = False
                break
            t = rng.choices(cand, weights=[rem[c] for c in cand])[0]
            seq.append(t)
            rem[t] -= 1
            prev = t
        if ok:
            return seq


def make_lexicon(n_types, rng, exp3):
    if exp3:
        lens = [(2, 3, 4)[i % 3] for i in range(n_types)]
    else:
        lens = [2, 2, 3, 3, 4, 4]
    pool = []
    words = []
    for L in lens:
        w = []
        while len(w) < L:
            if not pool:
                pool = [s for s in SYLS if s not in w]
                rng.shuffle(pool)
            cand = [s for s in pool if s not in w]
            if not cand:
                pool = [s for s in SYLS if s not in w]
                rng.shuffle(pool)
                continue
            s = cand[0]
            pool.remove(s)
            w.append(s)
        words.append(tuple(w))
    return words


def frank_stimuli(exp, cond, seed):
    rng = random.Random(f"frank-{exp}-{cond}-{seed}")
    n_types, tokens, wps = 6, 600, 4
    if exp == "f1":
        wps = int(cond)
    elif exp == "f2":
        tokens = int(cond)
    else:
        n_types = int(cond)
    words = make_lexicon(n_types, rng, exp == "f3")
    base, rem = divmod(tokens, n_types)
    extra = set(rng.sample(range(n_types), rem))
    counts = {i: base + (1 if i in extra else 0) for i in range(n_types)}
    order = no_repeat_perm(counts, rng)
    sents = []
    for a in range(0, len(order), wps):
        sents.append([s for t in order[a:a + wps] for s in words[t]])
    # test pairs
    ngr = {L: Counter() for L in (2, 3, 4)}
    for s in sents:
        for L in ngr:
            for a in range(len(s) - L + 1):
                ngr[L][tuple(s[a:a + L])] += 1
    wordset = set(words)
    cands = {}
    for L in (2, 3, 4):
        c = []
        for ia, a in enumerate(words):
            for ib, b in enumerate(words):
                if ia == ib:
                    continue
                for s in range(1, L):
                    if s < len(a) and L - s < len(b):
                        pw = a[len(a) - s:] + b[:L - s]
                        if pw not in wordset:
                            c.append(pw)
        cands[L] = c
    pairs, unseen, prev = [], 0, None
    for i in range(30):
        w = words[i % n_types]
        L = len(w)
        seen = [p for p in dict.fromkeys(cands[L]) if ngr[L][p] > 0 and p != prev]
        if seen:
            pw = rng.choice(seen)
        else:
            unseen += 1
            allc = [p for p in dict.fromkeys(cands[L]) if p != prev] or cands[L]
            pw = rng.choice(allc)
        prev = pw
        pairs.append((w, pw))
    return {"words": words, "utts": sents, "pairs": pairs, "n_partword_unseen": unseen, "utts_syl": sum(len(s) for s in sents)}


INF_WORDS = {"A": ["tu pi ro", "go la bu", "bi da ku", "pa do ti"], "B": ["da pi ku", "ti la do", "bu ro bi", "pa go tu"]}


def infant_stimuli(exp, cond, seed):
    rng = random.Random(f"inf-{exp}-{cond}-{seed}")
    words = [tuple(w.split()) for w in INF_WORDS[cond]]
    other = "B" if cond == "A" else "A"
    order = no_repeat_perm({i: 45 for i in range(4)}, rng)
    stream = [s for t in order for s in words[t]]
    if exp == "i1":
        foils = [tuple(w.split()) for w in INF_WORDS[other]][:2]
        # SAN96b: A tests tupiro, golabu vs dapiku, tilado; B the reverse (words dapiku, tilado vs nonwords tupiro, golabu).
        pairs = [(w, f) for w in words[:2] for f in foils]
    else:
        ngram = {3: Counter(tuple(stream[i:i + 3]) for i in range(len(stream) - 2))}
        wordset = set(words)
        c21, c12 = [], []
        for ia, a in enumerate(words):
            for ib, b in enumerate(words):
                if ia != ib:
                    p21, p12 = a[1:] + b[:1], a[2:] + b[:2]
                    if ngram[3][p21] > 0:
                        c21.append(((ia, ib), p21))
                    if ngram[3][p12] > 0:
                        c12.append(((ia, ib), p12))
        k21 = rng.choice(c21)
        k12 = rng.choice([c for c in c12 if c[0] != k21[0]])
        foils = [k21[1], k12[1]]
        assert not (set(foils) & wordset)
        pairs = [(w, f) for w in words[:2] for f in foils]
        # words tested: the first two words of the condition's list (as in SAN96b: two words, two foils)
    return {"words": words, "utts": [stream], "pairs": pairs, "n_partword_unseen": 0, "utts_syl": len(stream)}


# --------------------------------------------------------------------------------------------- scoring
class TP:
    def __init__(self, utts):
        self.c, self.t = Counter(), Counter()
        for u in utts:
            seq = ["#"] + list(u) + ["#"]
            for a, b in zip(seq, seq[1:]):
                self.c[(a, b)] += 1
                self.t[a] += 1

    def score(self, x):
        p = 1.0
        for a, b in zip(x, x[1:]):
            p *= self.c[(a, b)] / self.t[a] if self.t[a] else 0.0
        return p


def luce(sw, sf):
    return sw / (sw + sf) if sw + sf > 0 else 0.5


def prop_better(sw, sf):
    return (sw - sf) / (sw + sf) if sw + sf > 0 else 0.0


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def run_one(exp, cond, arm, seed, outdir):
    t0 = time.time()
    st = frank_stimuli(exp, cond, seed) if exp.startswith("f") else infant_stimuli(exp, cond, seed)
    pairs = st["pairs"]
    tp = TP(st["utts"])
    res = {"exp": exp, "cond": cond, "arm": arm, "seed": seed, "assumption": ASSUMPTION, "W": W_BUF,
           "cfg": CFG, "cfg_dropped": DROPPED, "lexicon": ["".join(w) for w in st["words"]],
           "n_pairs": len(pairs), "n_partword_unseen": st["n_partword_unseen"], "n_syllables": st["utts_syl"],
           "pairs": ["".join(w) + " | " + "".join(p) for w, p in pairs]}
    tpl = [luce(tp.score(w), tp.score(p)) for w, p in pairs]
    res["tp_luce_mean"], res["tp_luce_se"] = mean_se(tpl)
    res["tp_luce_pairs"] = tpl
    if arm != "tp":
        kw = {"concepts_off": True, "concepts_off_parse": True} if arm == "conceptsoff" else {}
        L = Trellis3Seg(seed=seed, **CFG, **kw)
        nbuf = 0
        tl = time.time()
        brng = random.Random(seed * 7919 + 1)          # buffer phase, independent of the stimulus rng
        for u in st["utts"]:
            for toks, le, re_ in buffers(list(u), rng=brng):
                L.learn_buf(toks, le, re_)
                nbuf += 1
        res["learn_s"] = time.time() - tl
        res["n_buffers"] = nbuf
        res["n_records"] = len(L.records)
        Sw = [L.S(w) for w, _ in pairs]
        Sp = [L.S(p) for _, p in pairs]
        lu = [luce(a, b) for a, b in zip(Sw, Sp)]
        res["S_word"], res["S_foil"] = Sw, Sp
        res["luce_pairs"] = lu
        res["luce_mean"], res["luce_se"] = mean_se(lu)
        res["prop_better_mean"], res["prop_better_se"] = mean_se([prop_better(a, b) for a, b in zip(Sw, Sp)])
        cache = {}

        def bits(x):
            if x not in cache:
                cache[x] = L.recog_bits(x, leaf=(arm == "conceptsoff"))
            return cache[x]
        bw = [bits(tuple(w)) for w, _ in pairs]
        bp = [bits(tuple(p)) for _, p in pairs]
        ch = [1.0 if a > b else 0.5 if a == b else 0.0 for a, b in zip(bw, bp)]
        res["bits_word"], res["bits_foil"] = bw, bp
        res["bits_choice_pairs"] = ch
        res["bits_choice_mean"], res["bits_choice_se"] = mean_se(ch)
    res["time_s"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"{exp}_{cond}_{arm}_s{seed}.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(res, f)
    keys = ["luce_mean", "bits_choice_mean", "prop_better_mean", "tp_luce_mean", "n_partword_unseen", "n_records", "learn_s", "time_s"]
    print(json.dumps({"exp": exp, "cond": cond, "arm": arm, "seed": seed, **{k: (round(res[k], 4) if isinstance(res.get(k), float) else res.get(k)) for k in keys}}), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["f1", "f2", "f3", "i1", "i2"])
    ap.add_argument("--cond", required=True)
    ap.add_argument("--arm", default="full", choices=["full", "conceptsoff", "tp"])
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    for s in [int(x) for x in a.seeds.split(",")]:
        run_one(a.exp, a.cond, a.arm, s, a.out)
