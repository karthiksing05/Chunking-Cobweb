"""Chart generation: inside-outside over concepts instead of stored chunks.

Generation by expansion fills a gap only with one stored instance whose span fits
it exactly, so "the * the dog" fails when no stored chunk is the [N V] the gap
needs. Here the gap is covered by parts of several constituents. Memory is read
as a grammar over concepts:

  symbols    a word's learned category (its window name), and a composite's
             content concept at depth ``cut`` of the content tree
  rules      P(A B | C) from the stored composites' own parts, each counted with
             its stored weight, backed off to the pairs of all composites:
             (n_C(A B) + lam g(A B)) / (n_C + lam)
  emission   P(w | category) from the stored words
  start      P(C) from the whole experiences (SENT)

A prompt is a list of words and gaps ("*": one or more words, "?": exactly one).
For every expansion of the gaps (up to ``gap_max`` words each) the inside pass
sums over every tree whose yield matches, the prompt's words fixed and each free
position summed over all words, so Z(expansion) is the grammar's own probability
of that shape. An expansion is drawn by Z, a decomposition is drawn top-down from
the inside scores and frozen, and each free position is filled left to right by
Cobweb prediction from its surroundings (``_gap_fills`` on the words placed so far,
the prompt's fixed words and, with a story facet, the previous sentence), among
words of the category the chart gave it.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .trellis3 import SENT, Elem

HOLE = "_"


class ChartGen:
    def __init__(self, L, cut: int = 3, lam: float = 1.0, alpha: float = 0.1, gap_max: int = 4,
                 length=(3, 8)):
        self.L, self.cut, self.lam, self.alpha, self.gap_max, self.length = L, cut, lam, alpha, gap_max, length
        self._cat: Dict[str, object] = {}
        self._build()

    # ------------------------------------------------------------ grammar
    def cat(self, tok: str):
        if tok not in self._cat:
            bag = self.L._window_name(tok)
            self._cat[tok] = ("w", max(bag.items(), key=lambda kv: kv[1])[0]) if bag else ("t", tok)
        return self._cat[tok]

    def _concept(self, e):
        node = e.cleaf
        if node is None:
            return None
        path = [node]
        while path[-1].parent is not None:
            path.append(path[-1].parent)
        if path[-1] is not self.L.cnt.root:                   # a stale leaf: sort it again
            path = list(reversed(self.L.cnt.categorize(e.cnt)))
        path.reverse()
        return ("c", path[min(self.cut, len(path) - 1)].id)

    def sym_of(self, e):
        return self.cat(e.token) if e.token is not None else self._concept(e)

    def _build(self):
        L = self.L
        rules, emit, start = Counter(), defaultdict(Counter), Counter()
        for e in L.records:
            if e.token is not None:
                emit[self.cat(e.token)][e.token] += 1.0
                continue
            if e.left is None or e.right is None or e.weight <= 0:
                continue
            c, a, b = self._concept(e), self.sym_of(e.left), self.sym_of(e.right)
            if c is not None and a is not None and b is not None:
                rules[(a, b, c)] += e.weight
        for rid, n in L.cnt.passive(L.cnt.root, SENT).items():
            c = self._concept(L.records[rid])
            if c is not None:
                start[c] += n
        syms = sorted({s for k in rules for s in k} | set(emit) | set(start), key=repr)
        self.syms, self.ix = syms, {s: i for i, s in enumerate(syms)}
        S = len(syms)
        self.n_words = len(emit)
        self.is_word = np.array([s[0] != "c" for s in syms])
        keys = list(rules)
        self.rA = np.array([self.ix[a] for a, _, _ in keys], dtype=np.int64)
        self.rB = np.array([self.ix[b] for _, b, _ in keys], dtype=np.int64)
        self.rC = np.array([self.ix[c] for _, _, c in keys], dtype=np.int64)
        self.rw = np.array([rules[k] for k in keys])
        self.N = np.bincount(self.rC, weights=self.rw, minlength=S)
        g = Counter()
        for (a, b, _), w in rules.items():
            g[(a, b)] += w
        gk = list(g)
        self.gA = np.array([self.ix[a] for a, _ in gk], dtype=np.int64)
        self.gB = np.array([self.ix[b] for _, b in gk], dtype=np.int64)
        self.gw = np.array([g[k] for k in gk]) / max(sum(g.values()), 1e-12)
        self.denom = np.where(self.is_word, np.inf, self.N + self.lam)
        st = np.zeros(S)
        for c, n in start.items():
            st[self.ix[c]] = n
        comp = ~self.is_word
        st[comp] += 0.5
        self.start = st / st.sum()
        self.emit = {self.ix[s]: c for s, c in emit.items()}
        self.vocab = sorted({w for c in emit.values() for w in c})
        self.stats = {"symbols": S, "word categories": int(self.is_word.sum()), "concepts": int(comp.sum()),
                      "rules": len(keys), "part pairs": len(gk)}

    # ------------------------------------------------------------ inside
    def _leaf(self, tok: Optional[str]):
        v = np.zeros(len(self.syms))
        if tok is None:                                       # a free position: every word, summed
            v[self.is_word] = 1.0
            return v
        V = len(self.vocab) + 1
        for i, c in self.emit.items():
            v[i] = (c.get(tok, 0.0) + self.alpha / V) / (sum(c.values()) + self.alpha)
        return v

    def _split(self, a, b):
        """Unnormalised vector over parents C of sum over rules of P(A B | C) a_A b_B."""
        obs = np.bincount(self.rC, weights=self.rw * a[self.rA] * b[self.rB], minlength=len(self.syms))
        glob = float(np.dot(self.gw, a[self.gA] * b[self.gB]))
        return (obs + self.lam * glob) / self.denom

    def inside(self, toks: Sequence[Optional[str]]):
        n = len(toks)
        ins = {}
        for i, t in enumerate(toks):
            v = self._leaf(t)
            m = v.max()
            ins[(i, i + 1)] = (v / m if m > 0 else v, math.log(m) if m > 0 else -math.inf)
        for span in range(2, n + 1):
            for i in range(0, n - span + 1):
                j = i + span
                parts = []
                for k in range(i + 1, j):
                    (a, la), (b, lb) = ins[(i, k)], ins[(k, j)]
                    if la == -math.inf or lb == -math.inf:
                        continue
                    parts.append((self._split(a, b), la + lb))
                if not parts:
                    ins[(i, j)] = (np.zeros(len(self.syms)), -math.inf)
                    continue
                top = max(l for _, l in parts)
                v = sum(p * math.exp(l - top) for p, l in parts)
                m = v.max()
                ins[(i, j)] = (v / m, top + math.log(m)) if m > 0 else (v, -math.inf)
        return ins

    def logZ(self, ins, n):
        v, l = ins[(0, n)]
        z = float(np.dot(self.start, v))
        return l + math.log(z) if z > 0 and l > -math.inf else -math.inf

    # ------------------------------------------------------------ sampling
    def _draw_tree(self, ins, i, j, C, rng, leaves, tree):
        if j - i == 1:
            leaves[i] = C
            tree.append((i, j, self.syms[C]))
            return
        tree.append((i, j, self.syms[C]))
        mask_obs = self.rC == C
        A_o, B_o, w_o = self.rA[mask_obs], self.rB[mask_obs], self.rw[mask_obs]
        As, Bs = np.concatenate([A_o, self.gA]), np.concatenate([B_o, self.gB])
        base = np.concatenate([w_o, self.lam * self.gw])
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
        p = p / p.sum()
        r = int(np.searchsorted(np.cumsum(p), rng.random() * (1 - 1e-12)))
        k, A, B = ks[r // len(As)], int(As[r % len(As)]), int(Bs[r % len(As)])
        self._draw_tree(ins, i, k, A, rng, leaves, tree)
        self._draw_tree(ins, k, j, B, rng, leaves, tree)

    def _expansions(self, pattern):
        gaps = [i for i, t in enumerate(pattern) if t in ("*", "?")]
        if pattern == ["*"]:
            lens = [list(range(self.length[0], self.length[1] + 1))]
        else:
            lens = [[1] if pattern[g] == "?" else list(range(1, self.gap_max + 1)) for g in gaps]
        for combo in itertools.product(*lens):
            toks, it = [], iter(combo)
            for t in pattern:
                toks.extend([None] * next(it) if t in ("*", "?") else [t])
            yield toks

    def generate(self, pattern: Sequence[str], rng: random.Random, prev: Optional[List[str]] = None,
                 info: Optional[dict] = None) -> Optional[List[str]]:
        pattern = list(pattern)
        options = []
        for toks in self._expansions(pattern):
            ins = self.inside(toks)
            z = self.logZ(ins, len(toks))
            if z > -math.inf:
                options.append((toks, ins, z))
        if not options:
            return None
        top = max(z for _, _, z in options)
        toks, ins, _ = options[rng.choices(range(len(options)), weights=[math.exp(z - top) for _, _, z in options])[0]]
        n = len(toks)
        v, _ = ins[(0, n)]
        root = rng.choices(range(len(self.syms)), weights=self.start * v)[0]
        leaves, tree = {}, []
        self._draw_tree(ins, 0, n, root, rng, leaves, tree)
        out = self._fill(toks, leaves, rng, prev)
        if info is not None:
            info.update({"tree": tree, "length": n})
        return out

    # ------------------------------------------------------------ filling
    def _fill(self, toks, leaves, rng, prev):
        L = self.L
        keep_pv = L.__dict__.get("_pv")
        if getattr(L, "story_facet", False):
            L._pv = list(prev) if prev else None
        out = [t if t is not None else HOLE for t in toks]
        try:
            for i, t in enumerate(toks):
                if t is not None:
                    continue
                want = self.syms[leaves[i]]
                e = Elem((i, i + 1), token=HOLE)
                e.token = None
                raw = L._surface(out, i, i + 1)
                try:
                    q = L._context(out, e, analysed=False)
                except Exception:
                    q = dict(raw)
                q = {a: v for a, v in q.items() if not (a in raw and raw[a] == HOLE)}
                fills = L._gap_fills(q)
                cand = {f[1]: w for f, w in fills.items() if f[0] == "w" and w > 0 and self.cat(f[1]) == want}
                if not cand:                                  # the context knows none: the category's own words
                    cand = dict(self.emit.get(leaves[i], {}))
                if not cand:
                    return None
                words, ws = zip(*cand.items())
                out[i] = rng.choices(words, weights=ws)[0]
        finally:
            L._pv = keep_pv
        return out
