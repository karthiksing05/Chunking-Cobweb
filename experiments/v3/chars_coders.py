"""Coders on Chinese characters (S5), the counterpart of the chess coder study (S6).

Each coder learns the first n characters (IDS decompositions as token sequences, prefix
order) and codes held-out characters; bits per character, the model not charged. A
token no coder has seen costs log2 of the whole token inventory on top of its escape.

  trellis    the chart's own probability of the sequence: the start over composite
             concepts times the inside score (chart_gen.ChartGen.logZ), the grammar
             TRELLIS generates from, so compression reads the same machinery
  unigram    each token by its frequency, then an end mark (the component list)
  bigram     interpolated Witten-Bell n-gram with start/end marks
  trigram    the same, order 3
  bpe        pair-merge units learned on the training sequences, units coded by their
             frequency with an escape to single tokens, then an end mark
  xz         xz(train + one test character) - xz(train), averaged over the first 200

CAUSAL=1 (environment; off by default, and then the output is unchanged) adds the causal
route, the counterpart of chess_causal (chess_compress_gen.py): each token, then the end mark,
is coded left to right from P(token | causal context concept). The causal description of
position i is what a prefix-order reader has decoded just before it, the two tokens before
it, {l1, l2} (<s> before the start), the left half of the learner's own context description
(its surface window l2 l1 _ r1 r2), so this partial description is sorted in the learner's
own context tree (the right half and the own-content facet left out: a reader does not
know them yet); also {l1} alone, and {l1 .. lw} for a learner of window w > 2. Every
training token and every training end mark is counted at each concept on the path its own
causal description sorts along; P(v | X) backs off from the root, est "cobweb" as Cobweb
predicts (chess), (n_X(v) + P(v | parent)) / (n_X + 1), the root Laplace-smoothed (0.5),
or est "wb" (Witten-Bell, as the trigram), (n_X(v) + T_X P(v | parent)) / (n_X + T_X),
over the seen tokens, the end mark and an escape (an unseen token: the escape, then
log2 of the inventory, as every coder here). The concepts at depths 0..63 of the path (a
shorter path's leaf standing in) are experts mixed by Bayesian weights (each expert's
probability of every token coded before; test characters in order, so a decoder keeps the
same weights). The character's code is then the chart code and the causal route mixed per
character by KT win counts (Krichevsky-Trofimov, as chess mixes the existing code and its
causal route per unit). Rows added: "causal" (bits per character of each variant: tree =
the depth mixture, leaf = the context concept alone, root = the root alone, the ablation
in which the context tree does no work, and their KT mixtures with the chart) and
"causal diag" (where the bits go), plus "chart + trigram (KT), reference": the same
per-character mixture with the trigram, the ceiling of such a mixture (not a TRELLIS code).
DIAG=1 adds "chart diag": the chart's bits split over posterior-sampled derivations (start,
rules, emissions, and the ambiguity credit) and its own length distribution. Learner config
"seq_causal": true (trellis_causal.CausalTrellis) makes the learner's context tree itself
hold causal descriptions (the left window only, no own-content facet), as chess_causal does.

Usage: chars_coders.py <seed> <checkpoints, e.g. 250,500,1000> <out.json> [config json] [charts json]
"""
import json
import lzma
import math
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from trellis2.characters import default_ids_path, load_characters  # noqa: E402
from trellis2.trellis3 import Trellis3  # noqa: E402
from trellis2.chart_gen import ChartGen  # noqa: E402

seed, checkpoints, out = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
N_TEST = int(os.environ.get("N_TEST", 300))
CHARTS = [{"cut": 2, "lam": 4, "word_sym": "context", "ctx_cut": 2},
          {"cut": 3, "lam": 4, "word_sym": "context", "ctx_cut": 3},
          {"cut": 2, "lam": 4, "word_sym": "category"},
          {"cut": 3, "lam": 4, "word_sym": "token"}]
CHARTS = json.loads(sys.argv[5]) if len(sys.argv) > 5 else CHARTS   # optional chart list

chars, _ = load_characters(default_ids_path(), seed=seed)
test = [c.tokens for c in chars[-N_TEST:] if len(c.tokens) > 1]
inventory = {t for c in chars for t in c.tokens}
ESC = math.log2(len(inventory))
LN2 = math.log(2)


def unigram(train):
    n = Counter(t for s in train for t in s)
    tot, V = sum(n.values()) + len(train), len(n) + 2           # + the end mark, + escape
    def bits(s):
        b = 0.0
        for t in list(s) + ["</s>"]:
            c = len(train) if t == "</s>" else n.get(t, 0)
            b += -math.log2((c + 0.5) / (tot + 0.5 * V)) + (ESC if t != "</s>" and c == 0 else 0)
        return b
    return bits


def ngram(train, order):
    counts = [defaultdict(Counter) for _ in range(order)]      # context tuple -> next counts
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            for k in range(order):
                counts[k][tuple(seq[i - k:i])][seq[i]] += 1
    V = len(inventory) + 1
    def p(ctx, t):
        pr = 1.0 / V
        for k in range(order):                                  # Witten-Bell, low to high order
            c = counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
            if not c:
                continue
            tot, types = sum(c.values()), len(c)
            pr = (c.get(t, 0) + types * pr) / (tot + types)
        return pr
    seen = {t for s in train for t in s}
    def bits(s):
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        return sum(-math.log2(p(seq[i - order + 1:i], seq[i])) for i in range(order - 1, len(seq))) \
            + sum(ESC for t in s if t not in seen)
    return bits


def bpe(train, n_merges=2000, min_count=2):
    seqs = [tuple(s) for s in train]
    merges = []
    for _ in range(n_merges):
        pc = Counter()
        for s in seqs:
            for a, b in zip(s, s[1:]):
                pc[(a, b)] += 1
        if not pc:
            break
        (a, b), c = pc.most_common(1)[0]
        if c < min_count:
            break
        merges.append((a, b))
        ab = a + " " + b
        new = []
        for s in seqs:
            o, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    o.append(ab); i += 2
                else:
                    o.append(s[i]); i += 1
            new.append(tuple(o))
        seqs = new
    rank = {m: r for r, m in enumerate(merges)}
    units = Counter(u for s in seqs for u in s)
    base = Counter(t for s in train for t in s)
    tot, V = sum(units.values()) + len(seqs), len(units) + 2
    btot = sum(base.values())
    def encode(s):
        s = list(s)
        while len(s) > 1:
            best = min(((rank.get((s[i], s[i + 1]), math.inf), i) for i in range(len(s) - 1)))
            if best[0] == math.inf:
                break
            i = best[1]
            s[i:i + 2] = [s[i] + " " + s[i + 1]]
        return s
    def bits(s):
        b = -math.log2((len(seqs) + 0.5) / (tot + 0.5 * V))   # the end mark
        for u in encode(s):
            c = units.get(u, 0)
            if c:
                b += -math.log2((c + 0.5) / (tot + 0.5 * V))
            else:                                               # escape, then the single token
                b += -math.log2(0.5 / (tot + 0.5 * V))
                b += -math.log2((base.get(u, 0) + 0.5) / (btot + 0.5 * (len(base) + 1)))
                b += ESC if u not in base else 0
        return b
    return bits, len(merges)


def xz_bits(train, items):
    blob = "\n".join(" ".join(s) for s in train).encode("utf-8")
    z0 = len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))
    return [8 * (len(lzma.compress(blob + b"\n" + " ".join(s).encode("utf-8"),
                                   preset=9 | lzma.PRESET_EXTREME)) - z0) for s in items]


def chart_bits(CG, seen, s):
    ins = CG.inside(list(s))
    z = CG.logZ(ins, len(s))
    if z == -math.inf:
        return None
    return -z / LN2 + sum(ESC for t in s if t not in seen)


CAUSAL = int(os.environ.get("CAUSAL", 0))
DIAG = int(os.environ.get("DIAG", 0))
LENGTH = int(os.environ.get("LENGTH", 0))   # code the length from stored lengths (see the chart loop)
MIX = os.environ.get("MIX")      # JSON learner config of a causal reader (r169; see mix_rows)
if CAUSAL or DIAG or MIX:
    import random
    import numpy as np
    from trellis2.characters import OPERATORS, token_slots   # diagnostics only: never read by a coder
    from trellis2.trellis3 import BOS, EOS
    NOVEL, DEPTHS, CALPHA = "<esc>", range(64), 0.5


def kt_mix(wins, ls):
    """-log2 of the mixture of routes with code lengths ``ls``, weighted by each route's share
    of the characters it coded best so far (KT); the winner's count goes up afterwards."""
    m = min(ls)
    n = sum(wins)
    out = m - math.log2(sum((w + 0.5) / (n + 0.5 * len(ls)) * 2.0 ** (m - l) for w, l in zip(wins, ls)))
    wins[min(range(len(ls)), key=lambda k: ls[k])] += 1
    return out


def bayes_mix(cum, ls):
    """-log2 of the Bayesian mixture of experts with code lengths ``ls`` (list), each weighted by
    its probability of everything it coded before (``cum``: past bits, updated afterwards)."""
    m0 = min(cum)
    m1 = min(c + l for c, l in zip(cum, ls))
    num = sum(2.0 ** (m1 - c - l) for c, l in zip(cum, ls))
    den = sum(2.0 ** (m0 - c) for c in cum)
    for k, l in enumerate(ls):
        cum[k] += l
    return m1 - math.log2(num) - (m0 - math.log2(den))


class CausalCoder:
    """CAUSAL: P(token | causal context concept) in the learner's own context tree (module doc)."""

    def __init__(self, L, train, seen, est="cobweb", width=2):
        self.L, self.seen, self.paths, self.est, self.width = L, seen, {}, est, width
        self.V = len(seen) + 2                              # the seen tokens, the end mark, the escape
        self.cx = {}
        for s in train:
            for i, t in enumerate(list(s) + [EOS]):
                for node in self.path(s, i):
                    self.cx.setdefault(node.id, Counter())[t] += 1
        self.n = {k: sum(d.values()) for k, d in self.cx.items()}
        self.cum = [0.0] * len(DEPTHS)                       # Bayesian depth weights (past bits per depth)
        self.ACC = Counter()

    def path(self, s, i):
        """The root-first path of position i's causal description {l1, l2}, sorted in the
        learner's context tree (quietly: the sort marks nothing for repair)."""
        key = tuple(s[i - k] if i >= k else BOS for k in range(1, self.width + 1))
        if key not in self.paths:
            quiet, self.L._quiet = self.L._quiet, True
            try:
                if getattr(self.L, "nest", False):     # nested suffixes (trellis_causal, r164)
                    from trellis2.trellis_causal import suffixes
                    q = suffixes(list(key), self.width)
                else:
                    q = {f"l{k + 1}": v for k, v in enumerate(key)}
                self.paths[key] = list(self.L.ctx.categorize(q))
            finally:
                self.L._quiet = quiet
        return self.paths[key]

    def probs(self, path, v):
        """P(v | X) at every concept on the path, root first, backed off from the root. est
        "cobweb" (chess): (n_X(v) + P(v | parent)) / (n_X + 1), the root Laplace (0.5); "wb"
        (Witten-Bell, the trigram's and PPM's escape): (n_X(v) + T_X P(v | parent)) / (n_X + T_X),
        T_X the number of distinct symbols counted at X, the root's parent uniform."""
        p, out = None, []
        for node in path:
            d = self.cx.get(node.id, {})
            c, n = d.get(v, 0), self.n.get(node.id, 0)
            if self.est == "wb":
                if p is None:
                    p = 1.0 / self.V
                p = (c + len(d) * p) / (n + len(d)) if n else p
            else:
                p = (c + CALPHA) / (n + CALPHA * self.V) if p is None else (c + p) / (n + 1.0)
            out.append(p)
        return out

    def code(self, s):
        """Bits of one character for each variant (tree: Bayesian depth mixture; leaf; root)."""
        out = Counter()
        slots = token_slots(s)
        for i, t in enumerate(list(s) + [EOS]):
            path = self.path(s, i)
            v = t if (t == EOS or t in self.seen) else NOVEL
            ps = self.probs(path, v)
            ls = [-math.log2(ps[min(d, len(ps) - 1)]) for d in DEPTHS]
            esc = ESC if v == NOVEL else 0.0
            b = {"tree": bayes_mix(self.cum, ls) + esc, "leaf": -math.log2(ps[-1]) + esc,
                 "root": -math.log2(ps[0]) + esc}
            for k, x in b.items():
                out[k] += x
            kind = "end" if t == EOS else ("operator" if t in OPERATORS else
                                           ("component (whole)" if slots[i] is None else "component"))
            self.ACC[f"tree bits, {kind}"] += b["tree"]
            self.ACC[f"root bits, {kind}"] += b["root"]
            self.ACC[f"count, {kind}"] += 1
            self.ACC["path depth (sum)"] += len(path) - 1
            self.ACC["count, positions"] += 1
            for name, node in (("depth 2", path[min(2, len(path) - 1)]), ("depth 6", path[min(6, len(path) - 1)]),
                               ("end", path[-1])):         # does the sort keep the previous token? (l1's share)
                l1, q = node.av.get("l1", {}), s[i - 1] if i else BOS
                tot, all_q = sum(l1.values()), path[0].av.get("l1", {}).get(q, 0.0)
                self.ACC[f"l1 share at the path's {name} (sum)"] += l1.get(q, 0.0) / tot if tot else 0.0
                # and the share of every stored occurrence after the same token that the concept holds
                self.ACC[f"l1 recall at the path's {name} (sum)"] += l1.get(q, 0.0) / all_q if all_q else 0.0
        return out


MIXREAD = os.environ.get("MIXREAD")   # "fills" (r173): the causal reader is Cobweb's prediction of the missing filler
_fp = {}


def fills_prob(LC, s, i, v, V, eps=0.01, only=None):
    """MIXREAD=fills (r173): P(next word = v | left context) as Cobweb predicts a missing attribute from a
    partial description: the learner's _gap_fills over its stored occurrences given only the left suffixes
    (what a reader has decoded), marginalizing over everything it cannot see (own content, right side).
    Word fillers, normalized; a uniform floor eps over the V seen tokens + escape. The same call story
    generation draws from (t3_story causal), so one reading of one learner serves both domains."""
    from trellis2.trellis_causal import suffixes
    left = tuple(s[i - k] if i - k >= 0 else BOS for k in range(1, LC.window + 1))
    key = (left, only)
    if key not in _fp:
        quiet, LC._quiet = LC._quiet, True
        q = suffixes(list(left), LC.window)
        if only is not None:               # r176: one suffix depth alone (nested suffixes are redundant evidence)
            a = "l1" if only == 1 else f"l1{only}"
            q = {a: q[a]}
        try:
            m = LC._gap_fills(q)
        finally:
            LC._quiet = quiet
        d = {f[1]: x for f, x in m.items() if f[0] == "w" and x > 0}
        z = sum(d.values())
        _fp[key] = {k: x / z for k, x in d.items()} if z > 0 else {}
    return (1 - eps) * _fp[key].get(v, 0.0) + eps / V


def mix_rows(CG, LC, train, seen, test, chart_len):
    """MIX (r169): one left-to-right code per character from two predictions of each next token, both
    learned: the chart's, P(t_i | t_<i, n) = Z(t_<=i, free rest) / Z(t_<i, free rest), which carries the
    grammar state (which part of a structure the next token completes), and the causal context tree's
    (LC, a seq_causal nest learner, Witten-Bell along its sorted path, the end mark excluded since the
    length is coded first), which carries local context. The length is coded from stored lengths (as
    LENGTH); each token by a Bayesian mixture over linear blends w * chart + (1 - w) * causal, w in
    0, .1, .. 1, weighted by each blend's probability of every token coded before (a decoder keeps the
    same weights). Rows: chart alone, causal alone, the mixture, on the characters the chart codes."""
    _fp.clear()
    cumd = [0.0] * LC.window if MIXREAD == "depths" else None
    lc = Counter(len(s) for s in train)
    p_len = lambda m: (lc.get(m, 0) + 0.5) / (len(train) + 0.5 * 64)
    C = CausalCoder(LC, train, seen, "wb", LC.window)
    W = [k / 10 for k in range(11)]
    cum = [0.0] * len(W)
    tot, n_ok, best = Counter(), 0, Counter()
    for s, cb in zip(test, chart_len):
        if cb is None:
            continue
        n = len(s)
        zprev = CG.logZ(CG.inside([None] * n), n)
        lb = -math.log2(p_len(n))
        b = Counter({"chart": lb, "causal": lb, "mixture": lb})
        steps = []
        for i, t in enumerate(s):
            zi = CG.logZ(CG.inside(list(s[:i + 1]) + [None] * (n - i - 1)), n)
            if zi == -math.inf or zprev == -math.inf:
                steps = None
                break
            pch, zprev = math.exp(zi - zprev), zi
            v = t if t in seen else NOVEL
            if MIXREAD == "fills":
                pc = fills_prob(LC, s, i, v, len(seen) + 1)
            elif MIXREAD == "depths":            # r176: each suffix depth read alone, Bayesian mixture over depths
                pc = 2.0 ** -bayes_mix(cumd, [-math.log2(fills_prob(LC, s, i, v, len(seen) + 1, only=k)) for k in range(1, LC.window + 1)])
            else:
                path = C.path(s, i)
                pc = C.probs(path, v)[-1] / max(1e-12, 1.0 - C.probs(path, EOS)[-1])
            steps.append((pch, min(pc, 1.0), ESC if v == NOVEL else 0.0))
        if steps is None:
            continue
        for pch, pc, esc in steps:
            b["chart"] += -math.log2(pch) + esc
            b["causal"] += -math.log2(pc) + esc
            b["mixture"] += bayes_mix(cum, [-math.log2(w * pch + (1 - w) * pc) for w in W]) + esc
        tot.update(b)
        n_ok += 1
    out = {k: v / n_ok for k, v in tot.items()} if n_ok else {}
    out["coded"] = n_ok / len(test)
    out["best blend (chart weight)"] = W[min(range(len(W)), key=lambda k: cum[k])]
    return out


def ngram_steps(train, order):
    """The trigram reference's bits per position (as ngram), for the breakdown only."""
    counts = [defaultdict(Counter) for _ in range(order)]
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        for i in range(order - 1, len(seq)):
            for k in range(order):
                counts[k][tuple(seq[i - k:i])][seq[i]] += 1
    V = len(inventory) + 1
    seen = {t for s in train for t in s}

    def steps(s):
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        out = []
        for i in range(order - 1, len(seq)):
            ctx, t, pr = seq[i - order + 1:i], seq[i], 1.0 / V
            for k in range(order):
                c = counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
                if c:
                    tot, types = sum(c.values()), len(c)
                    pr = (c.get(t, 0) + types * pr) / (tot + types)
            out.append(-math.log2(pr) + (ESC if t != "</s>" and t not in seen else 0.0))
        return out
    return steps


def causal_rows(L, train, seen, chart, test):
    """CAUSAL: the causal route's variants, their KT mixtures with the chart code, and where
    the bits go (by token kind, against the trigram's bits at the same positions)."""
    widths = sorted({L.window, 2, 1}, reverse=True)          # the left window, as wide as the learner's
    coders = {f"{est} l{w}": CausalCoder(L, train, seen, est, w) for est in ("cobweb", "wb") for w in widths}
    CC = coders[f"wb l{L.window}"]
    tri = ngram_steps(train, 3)
    keys = [f"{c} {k}" for c in coders for k in ("tree", "leaf", "root")]
    wins = {k: [0, 0] for k in keys}
    tot, tri_acc, n_ok = Counter(), Counter(), 0
    for s, cb in zip(test, chart):
        b = {f"{c} {k}": x for c, M in coders.items() for k, x in M.code(s).items()}
        for k in keys:
            tot[f"causal {k}"] += b[k]
            if cb is not None:
                tot[f"chart + causal {k} (KT)"] += kt_mix(wins[k], [cb, b[k]])
        if cb is not None:
            tot["chart"] += cb
            n_ok += 1
        slots = token_slots(s)
        ts = tri(s)
        if cb is not None:                                    # reference only: the ceiling of a per-character mix
            tot["chart + trigram (KT), reference"] += kt_mix(wins.setdefault("trigram", [0, 0]), [cb, sum(ts)])
        for i, (t, x) in enumerate(zip(list(s) + [EOS], ts)):
            kind = "end" if t == EOS else ("operator" if t in OPERATORS else
                                           ("component (whole)" if slots[i] is None else "component"))
            tri_acc[f"trigram bits, {kind}"] += x
    out = {k: v / (n_ok if k.startswith("chart") else len(test)) for k, v in tot.items()}
    out["KT wins (chart, causal)"] = dict(wins)
    for c, M in coders.items():
        out[f"{c}: best depth (Bayes)"] = min(DEPTHS, key=lambda d: M.cum[d])
        out[f"{c}: depth bits (Bayes, per position)"] = {
            d: round(M.cum[d] / M.ACC["count, positions"], 3) for d in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 63)}
    acc = Counter(CC.ACC)
    acc.update(tri_acc)
    diag = {k: v / len(test) for k, v in sorted(acc.items()) if not k.startswith("count")}
    diag.update({k: v / len(test) for k, v in sorted(acc.items()) if k.startswith("count")})
    diag["mean path depth"] = acc["path depth (sum)"] / acc["count, positions"]
    for k in [k for k in acc if k.startswith("l1 ")]:          # per position
        diag[k.replace(" (sum)", " (mean)")] = acc[k] / acc["count, positions"]
        diag.pop(k, None)
    diag["causal contexts sorted"] = len(CC.paths)
    return out, diag


def chart_diag(CG, test, chart, seen, n_samples=8, seed=0):
    """DIAG: the chart's bits per character split over posterior-sampled derivations: the start
    concept, the n - 1 rules P(A B | C), the n emissions P(w | A); E[bits(tree)] - bits = the
    credit of summing over derivations. Rules: how often the drawn pair was stored under C's own
    concept. Then the chart's own length distribution, P(n) = Z over n free positions."""
    rng = random.Random(seed)
    acc, n = Counter(), 0
    for s, cb in zip(test, chart):
        if cb is None:
            continue
        toks = list(s)
        ins = CG.inside(toks)
        m = len(toks)
        v, _ = ins[(0, m)]
        top = CG.start * v
        for _ in range(n_samples):
            C = int(np.searchsorted(np.cumsum(top / top.sum()), rng.random() * (1 - 1e-12)))
            bits = {"start": -math.log2(CG.start[C]), "rules": 0.0, "emissions": 0.0}
            stack, own = [(0, m, C)], 0
            while stack:
                i, j, C = stack.pop()
                if j - i == 1:
                    bits["emissions"] += -math.log2(CG._leaf(toks[i])[C])
                    continue
                k, A, B, p, at_own = draw_split(CG, ins, i, j, C, rng)
                bits["rules"] += -math.log2(p)
                own += at_own
                stack += [(i, k, A), (k, j, B)]
            for key, x in bits.items():
                acc[key] += x / n_samples
            acc["E bits(tree)"] += sum(bits.values()) / n_samples
            acc["rules seen at own concept"] += own / n_samples / max(m - 1, 1)
        acc["bits"] += cb - sum(ESC for t in s if t not in seen)
        n += 1
    out = {k: v / n for k, v in acc.items()}
    out["ambiguity credit"] = out["E bits(tree)"] - out["bits"]
    lens = Counter(len(s) for s in test)
    pz = {}
    for m in range(2, 12):
        z = CG.logZ(CG.inside([None] * m), m)
        pz[m] = math.exp(z) if z > -math.inf else 0.0
    out["chart P(length)"] = {m: round(p, 4) for m, p in pz.items()}
    out["chart P(length <= 11)"] = sum(pz.values())
    out["length bits (chart)"] = sum(-math.log2(pz[len(s)]) for s in test if pz.get(len(s))) / len(test)
    out["test length share"] = {m: round(c / len(test), 4) for m, c in sorted(lens.items())}
    return out


def draw_split(CG, ins, i, j, C, rng):
    """One split of span (i, j) under C drawn from the posterior (as ChartGen._draw_tree), with
    P(A B | C) of the drawn pair and whether the pair was stored under C's own concept."""
    parts_A, parts_B, parts_w, carry, own = [], [], [], 1.0, []
    for d, (r_ix, s_ix, N, lamv) in enumerate(reversed(CG.levels)):
        k = s_ix[C]
        mk = r_ix == k
        parts_A.append(CG.rA[mk]); parts_B.append(CG.rB[mk])
        parts_w.append(carry * CG.rw[mk] / (N[k] + lamv[k]))
        own.append(np.full(int(mk.sum()), d == 0))
        carry *= lamv[k] / (N[k] + lamv[k])
    parts_A.append(CG.gA); parts_B.append(CG.gB); parts_w.append(carry * CG.gw)
    own.append(np.zeros(len(CG.gA), dtype=bool))
    As, Bs, base, own = (np.concatenate(parts_A), np.concatenate(parts_B), np.concatenate(parts_w),
                         np.concatenate(own))
    ks, ws, ls = [], [], []
    for k in range(i + 1, j):
        (a, la), (b, lb) = ins[(i, k)], ins[(k, j)]
        if la == -math.inf or lb == -math.inf:
            continue
        ks.append(k)
        ws.append(base * a[As] * b[Bs])
        ls.append(la + lb)
    top = max(ls)
    p = np.concatenate([w * math.exp(l - top) for w, l in zip(ws, ls)])
    r = int(np.searchsorted(np.cumsum(p / p.sum()), rng.random() * (1 - 1e-12)))
    k, A, B = ks[r // len(As)], int(As[r % len(As)]), int(Bs[r % len(As)])
    same = (As == A) & (Bs == B)
    return k, A, B, float(base[same].sum()), bool((same & own).any())


if cfg.pop("seq_causal", False):          # the learner's context tree holds causal descriptions
    from trellis2.trellis_causal import CausalTrellis
    L = CausalTrellis(seed=seed, **cfg)
else:
    L = Trellis3(seed=seed, **cfg)
if MIX == "self":                        # one learner (r171): the chart and the causal reader read the same seq_causal learner
    LC = L
elif MIX:                                # the causal reader, learned on the same stream (r169)
    from trellis2.trellis_causal import CausalTrellis
    LC = CausalTrellis(seed=seed, **json.loads(MIX))
rows, done, t0 = [], 0, time.time()
for n in checkpoints:
    for c in chars[done:n]:
        L.learn(c.tokens)
        if MIX and LC is not L:
            LC.learn(c.tokens)
    done = n
    train = [c.tokens for c in chars[:n]]
    seen = {t for s in train for t in s}
    row = {"seed": seed, "train": n, "test": len(test), "learn s": round(time.time() - t0),
           "unseen-token share": sum(t not in seen for s in test for t in s) / sum(len(s) for s in test),
           "tokens per character": sum(len(s) for s in test) / len(test)}
    mean = lambda xs: sum(xs) / len(xs)
    row["unigram"] = mean([unigram(train)(s) for s in test])
    row["bigram"] = mean([ngram(train, 2)(s) for s in test])
    row["trigram"] = mean([ngram(train, 3)(s) for s in test])
    bb, nm = bpe(train)
    row["bpe"], row["bpe merges"] = mean([bb(s) for s in test]), nm
    row["xz"] = mean(xz_bits(train, test[:200]))
    for k, ch in enumerate(CHARTS):
        t1 = time.time()
        CG = ChartGen(L, **ch)
        bs = [chart_bits(CG, seen, s) for s in test]
        ok = [b for b in bs if b is not None]
        row[f"trellis {k}"] = {"chart": ch, "bits": mean(ok) if ok else None, "coded": len(ok) / len(bs),
                               "stats": CG.stats, "s": round(time.time() - t1)}
        if LENGTH:
            # the chart's own length distribution leaks mass onto lengths characters never have (even ones):
            # code the length from the stored characters' lengths (add-half over 1..64), then the string given
            # its length, P_chart(s) / Z_n with Z_n the chart's mass on length n
            lc = Counter(len(s) for s in train)
            p_len = lambda m: (lc.get(m, 0) + 0.5) / (len(train) + 0.5 * 64)
            zn = {}
            for s, b in zip(test, bs):
                if b is not None and len(s) not in zn:
                    zn[len(s)] = CG.logZ(CG.inside([None] * len(s)), len(s))
            bl = [None if b is None or zn[len(s)] == -math.inf else
                  b + zn[len(s)] / LN2 - math.log2(p_len(len(s))) for s, b in zip(test, bs)]
            okl = [b for b in bl if b is not None]
            row[f"trellis {k} + stored lengths"] = mean(okl) if okl else None
            if MIX and k == 0:
                t2 = time.time()
                row["mix"] = mix_rows(CG, LC, train, seen, test, bl)
                row["mix"]["s"] = round(time.time() - t2)
        if k == 0 and (CAUSAL or DIAG):                        # the first chart is the one mixed and diagnosed
            CG0, bs0 = CG, bs
    if CAUSAL:
        t1 = time.time()
        row["causal"], row["causal diag"] = causal_rows(L, train, seen, bs0, test)
        row["causal"]["s"] = round(time.time() - t1)
    if DIAG:
        t1 = time.time()
        row["chart diag"] = chart_diag(CG0, test, bs0, seen)
        row["chart diag"]["s"] = round(time.time() - t1)
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
