"""Does sentence structure pay on real text?

Code lengths of the Penn Treebank sample (NLTK; every sentence, punctuation
and empty elements removed: 3,901 sentences, 82,356 tags) under descriptions
with and without sentence structure. Every number is the length of an actual
message in bits: each table row is coded with its Dirichlet-multinomial
predictive (the Bayesian mixture code of ``trellis2/mdl.py``), with the
concentration chosen by code length from a small grid.

A description that sends a tree can be scored two ways:

* derivation: the message sends the analysis, -log2 P(s, t);
* total probability: the message sends only the sentence, -log2 P(s), the sum
  over all trees, which bits-back coding achieves (Hinton & van Camp 1993;
  Townsend et al. 2019). The difference, -log2 P(t | s), is the price of
  naming one tree. It is computed with the inside algorithm, under the
  parameters estimated from the gold trees.

Heads follow Collins (1999). Dependency trees are generated head-outward
(Klein & Manning 2004): per head and direction, continue/stop decisions given
whether a dependent was taken, and each dependent's tag given the head's tag
and the direction (first order) or also the previous sibling's tag (second
order). Word codes are sequential hierarchical Pitman-Yor codes (interpolated
Kneser-Ney as a code) that back off from the richest context to the tag.

Usage:
    python experiments/v2/treebank_codes.py --out experiments/v2/results/treebank/structure_codes.md
    # --em: EM for the first-order model on WSJ10 (needs torch)
    # --trellis: TRELLIS v2's own code for whole-tree analyses of WSJ10
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.mdl_search import code_bits  # noqa: E402
from trellis2.treebank import PUNCTUATION, _parse_sexpr, default_ptb_root  # noqa: E402

LN2 = math.log(2)
ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0)

HEAD_RULES = {   # Collins (1999), table A.1: search direction and priority list
    "ADJP": ("L", "NNS QP NN $ ADVP JJ VBN VBG ADJP JJR NP JJS DT FW RBR RBS SBAR RB"),
    "ADVP": ("R", "RB RBR RBS FW ADVP TO CD JJR JJ IN NP JJS NN"),
    "CONJP": ("R", "CC RB IN"), "FRAG": ("R", ""), "INTJ": ("L", ""), "LST": ("R", "LS :"),
    "NAC": ("L", "NN NNS NNP NNPS NP NAC EX $ CD QP PRP VBG JJ JJS JJR ADJP FW"),
    "PP": ("R", "IN TO VBG VBN RP FW"), "PRN": ("L", ""), "PRT": ("R", "RP"),
    "QP": ("L", "$ IN NNS NN JJ RB DT CD NCD QP JJR JJS"), "RRC": ("R", "VP NP ADVP ADJP PP"),
    "S": ("L", "TO IN VP S SBAR ADJP UCP NP"),
    "SBAR": ("L", "WHNP WHPP WHADVP WHADJP IN DT S SQ SINV SBAR FRAG"),
    "SBARQ": ("L", "SQ S SINV SBARQ FRAG"), "SINV": ("L", "VBZ VBD VBP VB MD VP S SINV ADJP NP"),
    "SQ": ("L", "VBZ VBD VBP VB MD VP SQ"), "UCP": ("R", ""),
    "VP": ("L", "TO VBD VBN MD VBZ VB VBG VBP VP ADJP NN NNS NP"),
    "WHADJP": ("L", "CC WRB JJ ADJP"), "WHADVP": ("R", "CC WRB"),
    "WHNP": ("L", "WDT WP WP$ WHADJP WHPP WHNP"), "WHPP": ("R", "IN TO FW"),
}
NP_GROUPS = (("NN", "NNP", "NNPS", "NNS", "NX", "POS", "JJR"), ("$", "ADJP", "PRN"), ("CD",),
             ("JJ", "JJS", "RB", "QP"))


# Data ------------------------------------------------------------------- #
@dataclass
class Sentence:
    words: List[str]                       # lowercased
    tags: List[str]
    heads: List[int]                       # head of each word, -1 for the root
    units: List[Tuple[int, int, int]]      # base phrases and lone words: (start, end, head)
    tree: tuple                            # gold tree, right-binarized, as a plain-PCFG node


def _clean(t, toks):
    """(label, children) for a phrase, (tag, index) for a kept word, or None."""
    label, kids = t[0], t[1:]
    if len(kids) == 1 and isinstance(kids[0], str):
        if label in PUNCTUATION:
            return None
        toks.append((kids[0].lower(), label))
        return (label, len(toks) - 1)
    out = [c for c in (_clean(k, toks) for k in kids) if c is not None]
    return ((re.split(r"[-=]", label)[0] or label) if label else "S", out) if out else None


def _head_child(label, labels) -> int:
    if label in HEAD_RULES:
        direction, priority = HEAD_RULES[label]
        order = range(len(labels)) if direction == "L" else range(len(labels) - 1, -1, -1)
        for cat in priority.split():
            for i in order:
                if labels[i] == cat:
                    return i
        return 0 if direction == "L" else len(labels) - 1
    if labels[-1] == "POS":                       # NP and anything unlisted
        return len(labels) - 1
    for group in NP_GROUPS[:1]:
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] in group:
                return i
    for i, lab in enumerate(labels):
        if lab == "NP":
            return i
    for group in NP_GROUPS[1:]:
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] in group:
                return i
    return len(labels) - 1


def _heads(node, heads) -> int:
    """Head word of ``node``; sets heads[i] for every non-head word below it."""
    label, body = node
    if isinstance(body, int):
        return body
    kid_heads = [_heads(c, heads) for c in body]
    h = kid_heads[_head_child(label, [c[0] for c in body])]
    for k in kid_heads:
        if k != h:
            heads[k] = h
    return h


def _units(node, out):
    label, body = node
    if isinstance(body, int):
        out.append((body, body + 1, body))
    elif len(body) >= 2 and all(isinstance(c[1], int) for c in body):
        out.append((body[0][1], body[-1][1] + 1, _heads(node, {})))
    else:
        for c in body:
            _units(c, out)


def _binarized(node):
    """Unary chains collapse to their lowest node; children c1 ... ck become
    c1 (@X: c2 (... ck))."""
    label, body = node
    if isinstance(body, int):
        return None
    while len(body) == 1 and not isinstance(body[0][1], int):
        label, body = body[0]
    if len(body) == 1:
        return None                                # a unary chain down to a word

    def leaf_or_tree(c):
        b = _binarized(c)
        if b is not None:
            return b
        while not isinstance(c[1], int):
            c = c[1][0]
        return (c[0], c[0])

    kids = [leaf_or_tree(c) for c in body]

    def chain(cs, name):
        if len(cs) == 2:
            return (name, (cs[0], cs[1]))
        return (name, (cs[0], chain(cs[1:], "@" + label)))
    return chain(kids, label)


def load_sentences() -> List[Sentence]:
    out = []
    for path in sorted(glob.glob(os.path.join(default_ptb_root(), "*.mrg"))):
        with open(path) as f:
            blocks = _parse_sexpr(f.read())
        for block in blocks:
            t = block[1] if block[0] is None and len(block) == 2 else block
            toks = []
            node = _clean(t, toks)
            if node is None or len(toks) < 2:
                continue
            heads = [None] * len(toks)
            heads[_heads(node, heads)] = -1
            units = []
            _units(node, units)
            out.append(Sentence([w for w, _ in toks], [p for _, p in toks], heads,
                                sorted(units), _binarized(node)))
    return out


# Codes ------------------------------------------------------------------ #
def dm_bits(counts, alphabet: int, alpha: float) -> float:
    c = np.array([v for v in counts if v > 0], dtype=float)
    if c.size == 0:
        return 0.0
    a = alphabet * alpha
    return float(gammaln(c.sum() + a) - gammaln(a) - np.sum(gammaln(c + alpha) - gammaln(alpha))) / LN2


def table_bits(rows: Dict, alphabet: int, alpha: float) -> float:
    return sum(dm_bits(r.values(), alphabet, alpha) for r in rows.values())


def tagset(sents) -> List[str]:
    return sorted({t for s in sents for t in s.tags})


def ngram_bits(sents, order: int) -> float:
    rows = defaultdict(Counter)
    for s in sents:
        seq = ["<s>"] * (order - 1) + s.tags + ["</s>"]
        for i in range(order - 1, len(seq)):
            rows[tuple(seq[i - order + 1:i])][seq[i]] += 1
    A = len(tagset(sents)) + 1
    return min(table_bits(rows, A, a) for a in ALPHAS)


class HeadOutward:
    """Head-outward dependency model of tag sequences, first or second order,
    with parameters estimated from the gold trees."""

    def __init__(self, sents, order: int):
        self.order = order
        self.T = len(tagset(sents))
        self.stop, self.attach, self.root = defaultdict(Counter), defaultdict(Counter), Counter()
        for s in sents:
            for table, key, outcome in self.events(s.tags, s.heads):
                {"stop": self.stop, "attach": self.attach, "root": self.root}[table]
                if table == "root":
                    self.root[outcome] += 1
                else:
                    getattr(self, table)[key][outcome] += 1
        self.alpha = min(ALPHAS, key=self._derivation_bits)
        self.derivation_bits = self._derivation_bits(self.alpha)

    def _key(self, head, direction, prev):
        return (head, direction, prev) if self.order == 2 else (head, direction)

    def events(self, tags, heads):
        deps = defaultdict(list)
        for i, h in enumerate(heads):
            if h == -1:
                yield "root", None, tags[i]
            else:
                deps[h].append(i)
        for h in range(len(tags)):
            for d, ds in (("L", sorted([x for x in deps[h] if x < h], reverse=True)),
                          ("R", sorted(x for x in deps[h] if x > h))):
                prev = "<none>"
                for k, x in enumerate(ds):
                    yield "stop", (tags[h], d, k == 0), "go"
                    yield "attach", self._key(tags[h], d, prev), tags[x]
                    prev = tags[x]
                yield "stop", (tags[h], d, not ds), "stop"

    def _derivation_bits(self, a):
        return (table_bits(self.stop, 2, a) + table_bits(self.attach, self.T, a)
                + dm_bits(self.root.values(), self.T, a))

    # plug-in log probabilities (posterior means)
    def lp_stop(self, head, d, adj, go):
        row = self.stop.get((head, d, adj), Counter())
        return math.log((row["go" if go else "stop"] + self.alpha) / (sum(row.values()) + 2 * self.alpha))

    def lp_attach(self, head, d, prev, tag):
        row = self.attach.get(self._key(head, d, prev), Counter())
        return math.log((row[tag] + self.alpha) / (sum(row.values()) + self.T * self.alpha))

    def lp_root(self, tag):
        return math.log((self.root[tag] + self.alpha) / (sum(self.root.values()) + self.T * self.alpha))

    def tree_lp(self, tags, heads) -> float:
        lp = 0.0
        for table, key, outcome in self.events(tags, heads):
            if table == "root":
                lp += self.lp_root(outcome)
            elif table == "stop":
                lp += self.lp_stop(*key, outcome == "go")
            else:
                lp += self.lp_attach(key[0], key[1], key[2] if self.order == 2 else None, outcome)
        return lp

    def inside(self, tags) -> float:
        """log P(tags), summed over projective trees (Eisner's second-order
        split-head algorithm; the first-order model ignores the sibling)."""
        n, t = len(tags), tags
        NEG = -np.inf
        Rd = np.full((n, n), NEG); Ld = np.full((n, n), NEG)    # right half of h to e; left half b to h
        IR = np.full((n, n), NEG); IL = np.full((n, n), NEG)    # h has taken dependent d
        B = np.full((n, n), NEG)                                 # Rd[x, m] + Ld[m+1, y], summed over m
        go = {(h, d, a, g): self.lp_stop(h, d, a, g) for h in set(t) for d in "LR"
              for a in (True, False) for g in (True, False)}
        att: Dict[tuple, float] = {}

        def A(head, d, prev, tag):
            if self.order == 1:
                prev = None
            key = (head, d, prev, tag)
            if key not in att:
                att[key] = self.lp_attach(head, d, prev, tag)
            return att[key]

        for h in range(n):
            Rd[h, h] = go[(t[h], "R", True, False)]
            Ld[h, h] = go[(t[h], "L", True, False)]
        for w in range(1, n):
            for x in range(n - w):
                y = x + w
                m = np.arange(x, y)
                B[x, y] = np.logaddexp.reduce(Rd[x, m] + Ld[m + 1, y])
            for h in range(n - w):
                d = h + w      # right dependent d of h, then left dependent h of head d
                terms = [go[(t[h], "R", True, True)] + A(t[h], "R", "<none>", t[d]) + Ld[h + 1, d]]
                terms += [IR[h, s] + go[(t[h], "R", False, True)] + A(t[h], "R", t[s], t[d]) + B[s, d]
                          for s in range(h + 1, d)]
                IR[h, d] = np.logaddexp.reduce(terms)
                terms = [go[(t[d], "L", True, True)] + A(t[d], "L", "<none>", t[h]) + Rd[h, d - 1]]
                terms += [IL[d, s] + go[(t[d], "L", False, True)] + A(t[d], "L", t[s], t[h]) + B[h, s]
                          for s in range(h + 1, d)]
                IL[d, h] = np.logaddexp.reduce(terms)
            for h in range(n - w):
                e = h + w
                ds = np.arange(h + 1, e + 1)
                Rd[h, e] = np.logaddexp.reduce(IR[h, ds] + Rd[ds, e]) + go[(t[h], "R", False, False)]
                ds = np.arange(h, e)
                Ld[h, e] = np.logaddexp.reduce(IL[e, ds] + Ld[h, ds]) + go[(t[e], "L", False, False)]
        return float(np.logaddexp.reduce([self.lp_root(t[r]) + Ld[0, r] + Rd[r, n - 1] for r in range(n)]))

    def bits_back(self, sents) -> float:
        return sum(self.inside(s.tags) - self.tree_lp(s.tags, s.heads) for s in sents) / LN2


def projective_trees(n):
    """Every projective dependency tree on n words (brute force, for checks)."""
    for heads in itertools.product(range(-1, n), repeat=n):
        if sum(h == -1 for h in heads) != 1 or any(h == i for i, h in enumerate(heads)):
            continue

        def ancestor(x, a):
            seen = set()
            while x != -1 and x not in seen:
                if x == a:
                    return True
                seen.add(x)
                x = heads[x]
            return x == a

        if any(not ancestor(i, -1) for i in range(n)):
            continue                               # a cycle
        if all(ancestor(k, h) for i, h in enumerate(heads) if h != -1
               for k in range(min(i, h) + 1, max(i, h))):
            yield list(heads)


def headed_chunk_bits(sents) -> float:
    """Base phrases (and lone words) as headed units: each unit's category is
    its head's tag (and whether it is a phrase), sent given the previous
    unit's category; the unit's other tags are sent as dependents of its
    head, head-outward."""
    T = len(tagset(sents))
    top, stop, attach = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for s in sents:
        prev = "<s>"
        for i, j, h in s.units:
            cat = (s.tags[h], j - i > 1)
            top[prev][cat] += 1
            prev = cat
            for d, ds in (("L", s.tags[i:h][::-1]), ("R", s.tags[h + 1:j])):
                for k, tag in enumerate(ds):
                    stop[(cat, d, k == 0)]["go"] += 1
                    attach[(cat, d)][tag] += 1
                stop[(cat, d, not ds)]["stop"] += 1
        top[prev]["</s>"] += 1
    C = len({c for r in top.values() for c in r})
    return min(table_bits(top, C, a) + table_bits(stop, 2, a) + table_bits(attach, T, a) for a in ALPHAS)


class PitmanYor:
    """Sequential hierarchical Pitman-Yor code of words: P(w | c_1) backs off
    to P(w | c_2), ..., down to a Dirichlet over the vocabulary (one table per
    word type and context: interpolated Kneser-Ney as a code)."""

    def __init__(self, V: int, levels):
        self.V, self.levels = V, levels                   # levels: [(concentration, discount)]
        self.n = [defaultdict(Counter) for _ in levels]
        self.tot = [Counter() for _ in levels]
        self.base, self.base_tot, self.bits = Counter(), 0, 0.0

    def prob(self, w, ctxs, k=0):
        if k == len(ctxs):
            return (self.base[w] + 0.01) / (self.base_tot + self.V * 0.01)
        row, (b, d) = self.n[k][ctxs[k]], self.levels[k]
        return (max(row[w] - d, 0.0) + (b + d * len(row)) * self.prob(w, ctxs, k + 1)) / (self.tot[k][ctxs[k]] + b)

    def code(self, w, ctxs):
        self.bits -= math.log2(self.prob(w, ctxs))
        for k, c in enumerate(ctxs):
            seen = self.n[k][c][w] > 0
            self.n[k][c][w] += 1
            self.tot[k][c] += 1
            if seen:
                return
        self.base[w] += 1
        self.base_tot += 1


def word_bits(sents, contexts) -> float:
    """Words given their tags and a context, best over a small grid."""
    V = len({w for s in sents for w in s.words})
    best = math.inf
    for b, d in itertools.product((0.5, 2.0), (0.8, 0.9)):
        levels = None
        coder = None
        for s in sents:
            for i, w in enumerate(s.words):
                ctxs = contexts(s, i)
                if coder is None:
                    levels = [(b, d)] * (len(ctxs) - 2) + [(1.0, 0.8), (1.0, 0.7)]
                    coder = PitmanYor(V, levels)
                coder.code(w, ctxs)
        best = min(best, coder.bits)
    return best


def _prev(s, i):
    return (s.words[i - 1], s.tags[i - 1]) if i else ("<s>", "<s>")


def _head(s, i):
    h = s.heads[i]
    return (s.words[h], s.tags[h], "L" if h > i else "R") if h >= 0 else ("<root>", "<root>", "")


WORD_CONTEXTS = {
    "tag only": lambda s, i: [(s.tags[i],), (s.tags[i],)],
    "previous word": lambda s, i: [(s.tags[i],) + _prev(s, i), (s.tags[i], _prev(s, i)[1]), (s.tags[i],)],
    "head word": lambda s, i: [(s.tags[i],) + _head(s, i), (s.tags[i],) + _head(s, i)[1:], (s.tags[i],)],
    "previous word and head word": lambda s, i: [(s.tags[i],) + _prev(s, i) + _head(s, i),
                                                 (s.tags[i],) + _prev(s, i), (s.tags[i], _prev(s, i)[1]),
                                                 (s.tags[i],)],
}


class LabelledInside:
    """Inside probabilities of the plain PCFG read off labelled gold trees
    (maximum-likelihood rules), for the bits back of TRELLIS's representation."""

    def __init__(self, trees):
        from trellis2.mdl_search import _nodes
        self._nodes = _nodes
        self.rows, self.start = defaultdict(Counter), Counter()
        for top in trees:
            self.start[top[0]] += 1
            for lab, body in _nodes(top):
                self.rows[lab][body if isinstance(body, str) else (body[0][0], body[1][0])] += 1
        cats = sorted(self.rows, key=str)
        self.ix = {c: i for i, c in enumerate(cats)}
        self.K = len(cats)
        self.emit = defaultdict(lambda: np.zeros(self.K))
        A, Bc, C, P = [], [], [], []
        for lab, r in self.rows.items():
            n = sum(r.values())
            for key, c in r.items():
                if isinstance(key, str):
                    self.emit[key][self.ix[lab]] += c / n
                else:
                    A.append(self.ix[lab]); Bc.append(self.ix[key[0]]); C.append(self.ix[key[1]]); P.append(c / n)
        self.A, self.B, self.C, self.P = map(np.array, (A, Bc, C, P))
        self.p_start = np.zeros(self.K)
        for c, v in self.start.items():
            self.p_start[self.ix[c]] = v / len(trees)

    def tree_lp(self, top) -> float:
        lp = math.log(self.start[top[0]] / sum(self.start.values()))
        for lab, body in self._nodes(top):
            r = self.rows[lab]
            lp += math.log(r[body if isinstance(body, str) else (body[0][0], body[1][0])] / sum(r.values()))
        return lp

    def inside(self, tags) -> float:
        n = len(tags)
        ins = np.zeros((n, n + 1, self.K))
        for i, t in enumerate(tags):
            ins[i, i + 1] = self.emit[t]
        for w in range(2, n + 1):
            for i in range(n - w + 1):
                j = i + w
                left = np.stack([ins[i, k] for k in range(i + 1, j)])
                right = np.stack([ins[k, j] for k in range(i + 1, j)])
                vals = (left[:, self.B] * right[:, self.C]).sum(axis=0) * self.P
                ins[i, j] = np.bincount(self.A, weights=vals, minlength=self.K)
        return math.log(float(ins[0, n] @ self.p_start))


# Optional parts ---------------------------------------------------------- #
EM_STARTS = ([("the gold trees' parameters", 0), ("harmonic (Klein & Manning 2004)", 0),
              ("right-branching chains", 0), ("left-branching chains", 0), ("uniform", 0)]
             + [("random trees", seed) for seed in range(7)]
             + [("random parameters", seed) for seed in range(6)])
EM_ITERATIONS, EM_ALPHA = 40, 0.1


def _em_inside(q, ls, la, lr, arc=None):
    """log P(tags) under the first-order model, differentiable (torch): the
    gradient with respect to a log-parameter is its expected count."""
    import torch
    n = len(q)
    Ro, Lo, Rd, Ld, IR, IL = {}, {}, {}, {}, {}, {}
    for h in range(n):
        Ro[h, h] = Lo[h, h] = torch.zeros(())
        Rd[h, h], Ld[h, h] = ls[q[h], 1, 1, 1], ls[q[h], 0, 1, 1]

    def lse(xs):
        return torch.logsumexp(torch.stack(xs), 0)
    for w in range(1, n):
        for i in range(n - w):
            j = i + w
            IR[i, j] = lse([Ro[i, k] + ls[q[i], 1, int(k == i), 0] + Ld[k + 1, j] for k in range(i, j)]) \
                + la[q[i], 1, q[j]] + (arc[i, j] if arc is not None else 0)
            IL[j, i] = lse([Lo[j, k] + ls[q[j], 0, int(k == j), 0] + Rd[i, k - 1] for k in range(i + 1, j + 1)]) \
                + la[q[j], 0, q[i]] + (arc[j, i] if arc is not None else 0)
            Ro[i, j] = lse([IR[i, d] + Rd[d, j] for d in range(i + 1, j + 1)])
            Rd[i, j] = Ro[i, j] + ls[q[i], 1, 0, 1]
            Lo[j, i] = lse([IL[j, d] + Ld[i, d] for d in range(i, j)])
            Ld[i, j] = Lo[j, i] + ls[q[j], 0, 0, 1]
    return lse([lr[q[r]] + Ld[0, r] + Rd[r, n - 1] for r in range(n)])


def _em_run(job):
    """One EM run of the first-order model on WSJ10 from one start. Returns
    the code (-log2 P(s) under the plug-in parameters, plus the Occam factor
    of the expected counts) and the directed accuracy of the posterior-argmax
    heads."""
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    start, seed = job
    sents = [s for s in load_sentences() if len(s.tags) <= 10]
    tags = tagset(sents)
    T, ix, a = len(tags), {t: i for i, t in enumerate(tags)}, EM_ALPHA
    seqs = [[ix[t] for t in s.tags] for s in sents]
    rng = random.Random(seed)

    def counts_of(heads_list):
        st, at, rt = torch.zeros(T, 2, 2, 2), torch.zeros(T, 2, T), torch.zeros(T)
        for q, hs in zip(seqs, heads_list):
            deps = defaultdict(list)
            for i, h in enumerate(hs):
                if h == -1:
                    rt[q[i]] += 1
                else:
                    deps[h].append(i)
            for h in range(len(q)):
                for dr, side in ((0, sorted([d for d in deps[h] if d < h], reverse=True)),
                                 (1, sorted(d for d in deps[h] if d > h))):
                    for k, d in enumerate(side):
                        st[q[h], dr, int(k == 0), 0] += 1
                        at[q[h], dr, q[d]] += 1
                    st[q[h], dr, int(not side), 1] += 1
        return st, at, rt

    def random_tree(n):
        heads = [None] * n

        def build(i, j):
            if j - i == 1:
                return i
            k = rng.randint(i + 1, j - 1)
            left, right = build(i, k), build(k, j)
            if rng.random() < 0.5:
                heads[right] = left
                return left
            heads[left] = right
            return right
        heads[build(0, n)] = -1
        return heads

    if start.startswith("the gold"):
        st, at, rt = counts_of([s.heads for s in sents])
    elif start.startswith("harmonic"):
        st, at, rt = torch.ones(T, 2, 2, 2), torch.zeros(T, 2, T), torch.zeros(T)
        for q in seqs:
            n = len(q)
            for i in range(n):
                rt[q[i]] += 1.0 / n
                z = sum(1.0 / abs(i - j) for j in range(n) if j != i)
                for j in range(n):
                    if j != i:
                        at[q[j], int(i > j), q[i]] += 1.0 / abs(i - j) / z
    elif start.startswith("right"):
        st, at, rt = counts_of([[i - 1 for i in range(len(q))] for q in seqs])
    elif start.startswith("left"):
        st, at, rt = counts_of([[i + 1 if i + 1 < len(q) else -1 for i in range(len(q))] for q in seqs])
    elif start == "uniform":
        st, at, rt = torch.ones(T, 2, 2, 2), torch.ones(T, 2, T), torch.ones(T)
    elif start == "random trees":
        st, at, rt = counts_of([random_tree(len(q)) for q in seqs])
    else:
        g = torch.Generator().manual_seed(seed)
        st, at, rt = (torch.rand(T, 2, 2, 2, generator=g) * 10, torch.rand(T, 2, T, generator=g) * 10,
                      torch.rand(T, generator=g) * 10)

    def occam(st, at, rt):
        bits = 0.0
        for rows, A in ((st.reshape(-1, 2), 2), (at.reshape(-1, T), T), (rt.reshape(1, -1), T)):
            for r in rows.numpy():
                n = r.sum()
                if n > 0:
                    nz = r[r > 0]
                    dm = gammaln(n + A * a) - gammaln(A * a) - np.sum(gammaln(r + a) - gammaln(a))
                    bits += (dm + np.sum(nz * np.log(nz / n))) / LN2
        return bits

    for it in range(EM_ITERATIONS + 1):
        ls = torch.log((st + a) / (st + a).sum(-1, keepdim=True)).requires_grad_()
        la = torch.log((at + a) / (at + a).sum(-1, keepdim=True)).requires_grad_()
        lr = torch.log((rt + a) / (rt + a).sum()).requires_grad_()
        total = sum(_em_inside(q, ls, la, lr) for q in seqs)
        total.backward()
        bits = -total.item() / LN2 + occam(ls.grad, la.grad, lr.grad)
        if it == EM_ITERATIONS:
            break
        st, at, rt = ls.grad.detach(), la.grad.detach(), lr.grad.detach()
    correct = 0
    for q, s in zip(seqs, sents):
        n = len(q)
        arc = torch.zeros(n, n, requires_grad=True)
        _em_inside(q, ls.detach(), la.detach(), lr.detach(), arc).backward()
        post = arc.grad
        for d in range(n):
            best = max([(1 - post[:, d].sum().item(), -1)] + [(post[h, d].item(), h) for h in range(n) if h != d])
            correct += int(best[1] == s.heads[d])
    return start, seed, bits, correct / sum(map(len, seqs))


def em_first_order(workers: int):
    """EM for the first-order model on WSJ10 from every start of ``EM_STARTS``."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_em_run, EM_STARTS))


def trellis_codes():
    """TRELLIS v2's full code for whole-tree analyses of the WSJ10 training
    sentences (seed 13): the learner's forests, the same forests completed
    above their chunks, purely right- and left-branching trees, gold trees."""
    from trellis2 import Trellis2
    from trellis2.data import Tree
    from trellis2.treebank import load_wsj, split
    from trellis2.unsupervised import UnsupervisedLearner

    train, _ = split(load_wsj(default_ptb_root()), 13)
    learner = UnsupervisedLearner(seed=13)
    for s in train:
        learner.observe(s.tags)
    learner.sleep()
    forests = [Tree(t.n, t.split, {}, t.roots) for t in learner.trees]

    def complete(tree, side):
        split_, roots, n = dict(tree.split), tree.roots, tree.n
        for a in range(len(roots) - 1):
            if side == "right":
                split_[(roots[a][0], n)] = roots[a][1]
            else:
                split_[(0, roots[len(roots) - 1 - a][1])] = roots[len(roots) - 1 - a][0]
        return Tree(n, split_)

    variants = {
        "the learner's forests": forests,
        "the learner's chunks, right-branching above them": [complete(t, "right") for t in forests],
        "the learner's chunks, left-branching above them": [complete(t, "left") for t in forests],
        "right-branching trees": [Tree(s.tree.n, {(i, s.tree.n): i + 1 for i in range(s.tree.n - 1)}) for s in train],
        "left-branching trees": [Tree(s.tree.n, {(0, j): j - 1 for j in range(2, s.tree.n + 1)}) for s in train],
        "gold trees (right-binarized)": [s.tree for s in train],
    }
    out = {"learner's night (its own code)": learner.history[-1]["bits"]}
    for name, trees in variants.items():
        m = Trellis2(seed=13)
        for s, t in zip(train, trees):
            m.learn(s.tags, t)
        out[name] = m.consolidate().info["total bits"]
    return out


# Report ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--em", action="store_true")
    ap.add_argument("--trellis", action="store_true")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    lines = []

    def emit(text=""):
        print(text, flush=True)
        lines.append(text)

    sents = load_sentences()
    n_tags = sum(len(s.tags) for s in sents)
    wsj10 = [s for s in sents if len(s.tags) <= 10]
    for order in (1, 2):                      # the inside algorithm sums over every tree
        m = HeadOutward(wsj10, order)
        for s in [s for s in wsj10 if 3 <= len(s.tags) <= 6][:8]:
            brute = np.logaddexp.reduce([m.tree_lp(s.tags, h) for h in projective_trees(len(s.tags))])
            assert abs(brute - m.inside(s.tags)) < 1e-9

    emit(f"Penn Treebank sample: {len(sents)} sentences, {n_tags} tags (punctuation removed); "
         f"WSJ10: {len(wsj10)} sentences.\n")
    bigram = ngram_bits(sents, 2)
    emit("| Description of the tags | Sends | Bits | Against the tag bigram |")
    emit("|---|---|---|---|")

    def row(name, sends, bits, ref=bigram):
        emit(f"| {name} | {sends} | {bits:,.0f} | {(bits - ref) / ref:+.1%} |")

    for order, name in ((1, "tag unigram"), (2, "tag bigram"), (3, "tag trigram")):
        row(name, "–", ngram_bits(sents, order))
    for order, name in ((1, "dependency trees, first order"), (2, "dependency trees, second order (sibling)")):
        m = HeadOutward(sents, order)
        back = m.bits_back(sents)
        row(f"{name}, gold", "derivation", m.derivation_bits)
        row(f"{name}, gold", "total probability", m.derivation_bits - back)
        emit(f"|  (bits back: {back / len(sents):.1f} per sentence) | | | |")
    row("headed base-phrase chunks, Markov over heads, gold", "derivation", headed_chunk_bits(sents))

    emit("\nWSJ10, TRELLIS's representation (plain PCFG over the treebank's labels):\n")
    bigram10 = ngram_bits(wsj10, 2)
    trees = [s.tree for s in wsj10]
    V = len({t for s in wsj10 for t in s.tags}) + 1
    derivation = min(code_bits([[t] for t in trees], V, a) for a in (0.001, 0.01, 0.1, 0.3))
    pcfg = LabelledInside(trees)
    back = sum(pcfg.inside(s.tags) - pcfg.tree_lp(t) for s, t in zip(wsj10, trees)) / LN2
    m1 = HeadOutward(wsj10, 1)
    emit("| Description of the tags (WSJ10) | Sends | Bits | Against the tag bigram |")
    emit("|---|---|---|---|")
    row("tag bigram", "–", bigram10, bigram10)
    row("labelled gold trees, plain PCFG", "derivation", derivation, bigram10)
    row("labelled gold trees, plain PCFG", "total probability", derivation - back, bigram10)
    emit(f"|  (bits back: {back / len(wsj10):.1f} per sentence) | | | |")
    row("dependency trees, first order, gold", "total probability",
        m1.derivation_bits - m1.bits_back(wsj10), bigram10)

    if args.em:
        from scipy.stats import spearmanr
        runs = em_first_order(args.workers)
        emit(f"\nWSJ10, the first-order dependency model after {EM_ITERATIONS} EM iterations "
             "(total probability), by start:\n")
        emit("| Start | Bits | Against the tag bigram | Heads right |")
        emit("|---|---|---|---|")
        for start, seed, bits, acc in sorted(runs, key=lambda r: r[2]):
            name = f"{start}, seed {seed}" if start.startswith("random") else start
            emit(f"| {name} | {bits:,.0f} | {(bits - bigram10) / bigram10:+.1%} | {acc:.1%} |")
        rho = spearmanr([r[2] for r in runs], [r[3] for r in runs]).correlation
        emit(f"\nRank correlation between code length and heads right: {rho:.2f}")

    emit("\nThe gap of the first-order dependency code (total probability) shrinks slowly with data:\n")
    emit("| Sentences | Tags | Against the tag bigram |")
    emit("|---|---|---|")
    for frac in (0.125, 0.25, 0.5, 1.0):
        gaps, toks = [], []
        for rep in range(3 if frac < 1 else 1):
            sub = random.Random(rep).sample(sents, int(frac * len(sents)))
            m = HeadOutward(sub, 1)
            b = ngram_bits(sub, 2)
            gaps.append((m.derivation_bits - m.bits_back(sub) - b) / b)
            toks.append(sum(len(s.tags) for s in sub))
        emit(f"| {int(frac * len(sents)):,} | {int(np.mean(toks)):,} | {np.mean(gaps):+.1%} |")

    emit("\nWords given their tags (lowercased; sequential Pitman-Yor codes):\n")
    emit("| Context of each word | Bits |")
    emit("|---|---|")
    for name, ctx in WORD_CONTEXTS.items():
        emit(f"| {name} | {word_bits(sents, ctx):,.0f} |")

    if args.trellis:
        emit("\nTRELLIS v2's own code for analyses of the WSJ10 training sentences (seed 13):\n")
        emit("| Analyses | Bits |")
        emit("|---|---|")
        for name, bits in trellis_codes().items():
            emit(f"| {name} | {bits:,.0f} |")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
