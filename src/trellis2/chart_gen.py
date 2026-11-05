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
  start      P(C) from the whole experiences (SENT); with ``follow`` and a previous
             sentence, P(C | prev) from the experiences that followed sentences
             like it (NEXT), by hierarchical back-off down prev's content path
             (as _pv_lift): P(C | node) = (n_node(C) + P(C | parent)) / (n_node + 1),
             starting from P(C)

A prompt is a list of words and gaps ("*": one or more words, "?": exactly one).
For every expansion of the gaps (up to ``gap_max`` words each) the inside pass
sums over every tree whose yield matches, the prompt's words fixed and each free
position summed over all words, so Z(expansion) is the grammar's own probability
of that shape. An expansion is drawn by Z, a decomposition is drawn top-down from
the inside scores (or, with ``decomp="max"``, the likeliest split at each node) and
frozen, and each free position is filled left to right by
Cobweb prediction from its surface window (``_gap_fills`` on the placed words within
two places), among words of the category the chart gave it. With ``fill`` "chart" or
"both" no tree is frozen: see the constructor.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .trellis3 import NEXT, SENT, Elem

HOLE = "_"


class ChartGen:
    def __init__(self, L, cut: int = 3, lam: float = 1.0, alpha: float = 0.1, gap_max: int = 4,
                 length=(3, 8), word_sym: str = "category", ctx_cut: int = 3, follow: bool = False,
                 decomp: str = "sample", cut_by: str = "depth", topic: Optional[float] = None,
                 fill: str = "cobweb", topic_by: str = "symbol", topic_mean: bool = False):
        self.L, self.cut, self.lam, self.alpha, self.gap_max, self.length = L, cut, lam, alpha, gap_max, length
        # cut_by "bits": cut and ctx_cut are information levels (-log2 of a concept's share
        # of its tree); a symbol is the deepest ancestor within that many bits, so the
        # grammar's grain does not hang on how deep or bushy the tree grew; cut_by "count":
        # the deepest ancestor holding at least ``cut`` instances (a generalization covers
        # more than one instance at cut 2), so the grain does not hang on the tree's size
        self.cut_by = cut_by
        # topic=beta (generalization as a function, y = G(x)): with a previous sentence x
        # and word_sym "context", a free position's word symbol is itself a context
        # concept G, so it counts by the lift P(x | G) / P(x | root) (L._pv_lift, back-off
        # along the path) raised to beta / n, n the free positions: the sentence's
        # geometric mean, so x is counted once for the whole sentence, not once per word
        self.topic = topic        # a number, or "fit": beta chosen by likelihood on stored pairs (fit_topic)
        # topic_by "symbol": x read at G itself; "best": at whichever of G's ancestors (or G)
        # explains x best, so the grain x is read at is chosen per symbol, not fixed by the cut
        self.topic_by = topic_by
        self.topic_mean = topic_mean   # the lift per word of x (its geometric mean) rather than the sum
        # fill "cobweb": a drawn tree's leaves filled left to right by Cobweb prediction among
        # each leaf symbol's words; "chart": no tree is frozen, each free position in turn
        # draws its word from the chart's marginal given the words placed so far (the rest
        # summed out: an outside pass), so a head's complement is read after the head is
        # chosen; "both": that marginal times the Cobweb prediction
        self.fill = fill
        self.follow = follow
        self.decomp = decomp      # "sample": each split drawn by its share of the inside score; "max": the likeliest split
        self.word_sym, self.ctx_cut = word_sym, ctx_cut   # "category" (window name), "token" (the word), or
        # "context": each occurrence named by its context concept at depth ctx_cut (where it stood)
        self._cat: Dict[str, object] = {}
        self._build()

    # ------------------------------------------------------------ grammar
    def cat(self, tok: str):
        if tok not in self._cat and self.word_sym == "token":
            self._cat[tok] = ("t", tok)
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
        path = self._cut(path, self.L.cnt, self.cut)
        sym = ("c", path[-1].id)
        self._path[sym] = [n.id for n in path]                 # root ... the symbol's own concept
        return sym

    def _ctx_concept(self, e):
        """The context concept the grammar states ``e`` over: its context leaf cut at ctx_cut."""
        node = e.leaf
        if node is None:
            return None
        path = [node]
        while path[-1].parent is not None:
            path.append(path[-1].parent)
        if path[-1] is not self.L.ctx.root:
            path = list(reversed(self.L.ctx.categorize(e.ctx)))
        path.reverse()
        return self._cut(path, self.L.ctx, self.ctx_cut)[-1]

    def _word_context(self, e):
        node = self._ctx_concept(e)
        if node is None:
            return None
        self._xnode[("x", node.id)] = node
        return ("x", node.id)

    def _cut(self, path, tree, cut):
        """The root-first path down to the symbol's concept: ``cut`` levels deep, or
        with cut_by "bits", every concept within ``cut`` bits; with "count", every concept
        holding at least ``cut`` instances."""
        if self.cut_by == "depth":
            return path[:cut + 1]
        if self.cut_by == "count":
            return [n for n in path if n.count >= cut] or path[:1]
        tot = max(tree.root.count, 1e-9)
        keep = [n for n in path if -math.log2(max(n.count, 1e-9) / tot) <= cut]
        return keep or path[:1]

    def sym_of(self, e):
        if e.token is None:
            return self._concept(e)
        if self.word_sym == "context":
            return self._word_context(e)
        return self.cat(e.token)

    def _build(self):
        L = self.L
        self._path = {}
        rules, emit, start = Counter(), defaultdict(Counter), Counter()
        self._xnode = {}                                      # each context word symbol's concept (for topic)
        for e in L.records:
            if e.token is not None:
                s = self.sym_of(e)
                if s is not None:
                    emit[s][e.token] += 1.0
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
        # back-off along the content tree: a symbol's rules, then those of its ancestors
        # at each depth up to the root, then the pairs of every composite (P_glob):
        # P_l(AB | C) = (n_{anc_l(C)}(AB) + lam P_{l-1}(AB | C)) / (n_{anc_l(C)} + lam)
        comp_syms = [s for s in syms if s[0] == "c"]
        self.levels = []                                      # per depth: (anc index of each rule, of each symbol, totals)
        depth = self.cut if self.cut_by == "depth" else max((len(p) for p in self._path.values()), default=1) - 1
        for d in range(depth + 1):
            anc = lambda s: self._path[s][min(d, len(self._path[s]) - 1)]
            ids = sorted({anc(s) for s in comp_syms})
            pos = {x: i for i, x in enumerate(ids)}
            r_ix = np.array([pos[anc(c)] for _, _, c in keys], dtype=np.int64)
            s_ix = np.full(S, -1, dtype=np.int64)
            for s in comp_syms:
                s_ix[self.ix[s]] = pos[anc(s)]
            N = np.bincount(r_ix, weights=self.rw, minlength=len(ids))
            self.levels.append((r_ix, s_ix, N))
        self.comp_ix = np.array([self.ix[s] for s in comp_syms], dtype=np.int64)
        g = Counter()
        for (a, b, _), w in rules.items():
            g[(a, b)] += w
        gk = list(g)
        self.gA = np.array([self.ix[a] for a, _ in gk], dtype=np.int64)
        self.gB = np.array([self.ix[b] for _, b in gk], dtype=np.int64)
        self.gw = np.array([g[k] for k in gk]) / max(sum(g.values()), 1e-12)
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
        prod = self.rw * a[self.rA] * b[self.rB]
        prev = np.full(len(self.syms), float(np.dot(self.gw, a[self.gA] * b[self.gB])))
        for r_ix, s_ix, N in self.levels:
            obs = np.bincount(r_ix, weights=prod, minlength=len(N))
            val = np.zeros(len(self.syms))
            c = self.comp_ix
            val[c] = (obs[s_ix[c]] + self.lam * prev[c]) / (N[s_ix[c]] + self.lam)
            prev = val
        return prev

    def inside(self, toks: Sequence[Optional[str]], lift=None, lifted=None):
        n = len(toks)
        ins = {}
        lifted = lifted or [t is None for t in toks]           # the positions x weighs (the free ones by default)
        free = sum(lifted)
        for i, t in enumerate(toks):
            v = self._leaf(t)
            if lifted[i] and lift is not None:              # P(x | G)^(beta / free), G the symbol
                v = v * np.exp(lift / free)                 # unshifted, so Z stays comparable across lengths
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

    def start_given(self, prev: Optional[List[str]]):
        """P(C | prev): the top concepts of the experiences that followed sentences like
        ``prev`` (NEXT at each concept on prev's content path), each level backed off
        to the one above it, the root to the unconditioned start."""
        if not (self.follow and prev):
            return self.start
        L = self.L
        r = L.parse(list(prev))
        n = len(prev)
        el = next((r["comps"][k] for k in r["chosen"] if k[0] == 0 and k[2] == n), None)
        if el is None:
            return self.start
        p = self.start
        for node in L.cnt.categorize(L._jx(el) if L.hierarchy == "joint" else L._content(el)):      # root ... leaf
            cnt = np.zeros(len(self.syms))
            for rid, w in L.cnt.passive(node, NEXT).items():
                c = self._concept(L.records[rid]) if L.records[rid].cleaf is not None else None
                if c is not None and c in self.ix:
                    cnt[self.ix[c]] += w
            p = (cnt + p) / (cnt.sum() + 1.0)
        return p

    def topic_lift(self, prev: Optional[List[str]]):
        """log P(prev | G) / P(prev | root) for each context word symbol G (zero elsewhere),
        times ``topic``; None without a topic, a previous sentence or context symbols."""
        L = self.L
        if not (self.topic and prev and getattr(L, "story_facet", False)) or not self._xnode:
            return None
        lift = L._pv_lift(prev)
        out = np.zeros(len(self.syms))
        for s, node in self._xnode.items():
            if s in self.ix:
                if self.topic_by == "best":                   # the generalization on G's path that explains x best
                    path = [node]
                    while path[-1].parent is not None:
                        path.append(path[-1].parent)
                    out[self.ix[s]] = max(lift(a) for a in path)
                else:
                    out[self.ix[s]] = lift(node)
        if self.topic_mean:                                   # x's words as one observation, not |x| independent ones
            out = out / len(prev)
        return self.topic * out

    def fit_topic(self, pairs, grid=(0, 0.5, 1, 2, 4, 8, 16)):
        """beta by likelihood: the mean over stored (previous, next) experiences of
        log P(next | previous, its length) = log Z_beta(next) - log Z_beta(n free positions),
        x weighing every position; the best beta on the grid is kept. Returns each beta's score."""
        self.topic = 1.0
        data = []
        for x, y in pairs:
            ell = self.topic_lift(x)
            if ell is not None:
                data.append((list(y), ell, self.start_given(x)))
        if not data:
            return {}
        score = {}
        for b in grid:
            tot = 0.0
            for y, ell, st in data:
                n = len(y)
                tot += (self.logZ(self.inside(y, b * ell, lifted=[True] * n), n, st)
                        - self.logZ(self.inside([None] * n, b * ell), n, st))
            score[b] = tot / len(data)
        self.topic = max(score, key=score.get)
        return score

    def logZ(self, ins, n, start=None):
        v, l = ins[(0, n)]
        z = float(np.dot(self.start if start is None else start, v))
        return l + math.log(z) if z > 0 and l > -math.inf else -math.inf

    # ------------------------------------------------------------ sampling
    def _draw_tree(self, ins, i, j, C, rng, leaves, tree):
        if j - i == 1:
            leaves[i] = C
            tree.append((i, j, self.syms[C]))
            return
        tree.append((i, j, self.syms[C]))
        # P(AB | C) unrolled: each depth's rules times the share left to it by the deeper ones
        parts_A, parts_B, parts_w, carry = [], [], [], 1.0
        for r_ix, s_ix, N in reversed(self.levels):
            k = s_ix[C]
            m = r_ix == k
            parts_A.append(self.rA[m]); parts_B.append(self.rB[m])
            parts_w.append(carry * self.rw[m] / (N[k] + self.lam))
            carry *= self.lam / (N[k] + self.lam)
        parts_A.append(self.gA); parts_B.append(self.gB); parts_w.append(carry * self.gw)
        As, Bs, base = np.concatenate(parts_A), np.concatenate(parts_B), np.concatenate(parts_w)
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
        if self.decomp == "max":
            r = int(np.argmax(p))
        else:
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
        start = self.start_given(prev)
        lift = self.topic_lift(prev)                         # free positions read through x's generalizations
        options = []
        for toks in self._expansions(pattern):
            ins = self.inside(toks, lift)
            z = self.logZ(ins, len(toks), start)
            if z > -math.inf:
                options.append((toks, ins, z))
        if not options:
            return None
        top = max(z for _, _, z in options)
        toks, ins, _ = options[rng.choices(range(len(options)), weights=[math.exp(z - top) for _, _, z in options])[0]]
        n = len(toks)
        if self.fill != "cobweb":
            if info is not None:
                info.update({"length": n, "lengths": [(len(t), round(z, 3)) for t, _, z in options], "steps": []})
            return self._fill_chart(toks, rng, prev, lift, start, info)
        v, _ = ins[(0, n)]
        root = rng.choices(range(len(self.syms)), weights=start * v)[0]
        leaves, tree = {}, []
        self._draw_tree(ins, 0, n, root, rng, leaves, tree)
        out = self._fill(toks, leaves, rng, prev)
        if info is not None:
            info.update({"tree": tree, "length": n})
        return out

    # ------------------------------------------------------------ filling
    def _cobweb_fills(self, out, i):
        """Cobweb's prediction for position i from its surface window alone (the placed
        words within two places): {word: weight}. It sees no chunks and not the previous
        sentence; chunk context and the topic reach generation through the chart."""
        q = {a: v for a, v in self.L._surface(out, i, i + 1).items() if v != HOLE}
        return {f[1]: w for f, w in self.L._gap_fills(q).items() if f[0] == "w" and w > 0}

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
                members = self.emit.get(leaves[i], {})
                cand = {w: x for w, x in self._cobweb_fills(out, i).items() if w in members}
                if not cand:                                  # the context knows none: the category's own words
                    cand = dict(self.emit.get(leaves[i], {}))
                if not cand:
                    return None
                words, ws = zip(*cand.items())
                out[i] = rng.choices(words, weights=ws)[0]
        finally:
            L._pv = keep_pv
        return out

    def _fill_chart(self, toks, rng, prev, lift, start, info=None):
        """Each free position, left to right, draws its word from P(w at i | placed words):
        sum over symbols s of mu_i(s) P(w | s), mu_i the chart's posterior of s at i (the
        gradient of log Z in i's leaf), times Cobweb's prediction with fill "both"."""
        L = self.L
        keep_pv = L.__dict__.get("_pv")
        if getattr(L, "story_facet", False):
            L._pv = list(prev) if prev else None
        lifted = [t is None for t in toks]                    # x weighs the symbols of the free positions
        out = [t if t is not None else HOLE for t in toks]
        try:
            for i in [k for k, f in enumerate(lifted) if f]:
                mu = self._marginal([None if w == HOLE else w for w in out], lifted, lift, start, i)
                if mu is None:
                    return None
                pc = defaultdict(float)
                for k in np.nonzero(mu > mu.max() * 1e-6)[0]:
                    c = self.emit.get(int(k))
                    if c:
                        tot = sum(c.values())
                        for w, x in c.items():
                            pc[w] += mu[k] * x / tot
                cand = dict(pc)
                if self.fill == "both":
                    cf = self._cobweb_fills(out, i)
                    cand = {w: x * cf[w] for w, x in pc.items() if w in cf} or cand
                if not cand:
                    return None
                words, ws = zip(*cand.items())
                out[i] = rng.choices(words, weights=ws)[0]
                if info is not None:                          # what the explorer shows: this position's sources
                    cf_ = cf if self.fill == "both" else {}
                    norm = lambda d: {w: x / max(sum(d.values()), 1e-300) for w, x in d.items()}
                    p_io, p_w, p_all = norm(pc), norm(cf_), norm(cand)
                    best = sorted(p_all, key=p_all.get, reverse=True)[:8]
                    if out[i] not in best:
                        best.append(out[i])
                    syms = [(self.syms[int(k)], float(mu[k])) for k in np.argsort(-mu)[:5] if mu[k] > 0]
                    info["steps"].append({"i": i, "word": out[i],
                                          "words": [(w, round(p_io.get(w, 0.0), 4), round(p_w.get(w, 0.0), 4), round(p_all[w], 4)) for w in best],
                                          "symbols": [(s[1], round(m, 4), [w for w, _ in Counter(self.emit.get(self.ix[s], {})).most_common(6)]) for s, m in syms]})
        finally:
            L._pv = keep_pv
        return out

    def _marginal(self, toks, lifted, lift, start, i):
        """The chart's posterior over symbols at position i (fixed words emitted, free
        positions summed over every word): leaf_i times its outside score, over Z."""
        S, c = len(self.syms), self.comp_ix
        n, free = len(toks), sum(lifted)
        ins = {}                                              # unscaled: n is small, float64 holds it
        for p, tok in enumerate(toks):
            v = self._leaf(tok)
            if lifted[p] and lift is not None:
                v = v * np.exp(lift / free)
            ins[(p, p + 1)] = v
        for span in range(2, n + 1):
            for a in range(0, n - span + 1):
                b = a + span
                ins[(a, b)] = sum(self._split(ins[(a, k)], ins[(k, b)]) for k in range(a + 1, b))
        z = float(np.dot(start, ins[(0, n)]))
        if not z > 0:
            return None
        out = defaultdict(lambda: np.zeros(S))
        out[(0, n)] = np.asarray(start, dtype=float)
        for span in range(n, 1, -1):                          # outside, top down: _split transposed
            for a in range(0, n - span + 1):
                b = a + span
                g = out[(a, b)]
                g_prod, g_glob = np.zeros(len(self.rw)), 0.0
                for r_ix, s_ix, N in reversed(self.levels):
                    sc = s_ix[c]
                    d = g[c] / (N[sc] + self.lam)
                    g_prod += np.bincount(sc, weights=d, minlength=len(N))[r_ix]
                    g = np.zeros(S)
                    g[c] = self.lam * d
                g_glob = g.sum()                              # the first level's back-off is one shared scalar
                for k in range(a + 1, b):
                    L_, R_ = ins[(a, k)], ins[(k, b)]
                    out[(a, k)] += (np.bincount(self.rA, weights=g_prod * self.rw * R_[self.rB], minlength=S)
                                    + np.bincount(self.gA, weights=g_glob * self.gw * R_[self.gB], minlength=S))
                    out[(k, b)] += (np.bincount(self.rB, weights=g_prod * self.rw * L_[self.rA], minlength=S)
                                    + np.bincount(self.gB, weights=g_glob * self.gw * L_[self.gA], minlength=S))
        return ins[(i, i + 1)] * out[(i, i + 1)] / z

    def describe(self, top: int = 12) -> dict:
        """Each word symbol's most frequent words, and the commonest rules (a diagnostic)."""
        cats = {repr(self.syms[i]): [w for w, _ in c.most_common(top)] + [f"({int(sum(c.values()))})"]
                for i, c in sorted(self.emit.items(), key=lambda kv: -sum(kv[1].values()))}
        order = np.argsort(-self.rw)[:30]
        rules = [f"{self.syms[self.rC[k]]} -> {self.syms[self.rA[k]]} {self.syms[self.rB[k]]} ({self.rw[k]:.1f})" for k in order]
        return {"word symbols": cats, "top rules": rules}
