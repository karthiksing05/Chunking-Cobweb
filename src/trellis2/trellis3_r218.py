"""Round 218: semi-incremental (windowed, depth-limited) inside-outside parsing (opt-in subclass; trellis3.py is untouched).

Trellis3R218(win_W=None, win_stride=None, io_depth_D=None, win_learn=False) is Trellis3R216 (and so Trellis3) unless a knob is set; with
win_W and io_depth_D both None, `parse` is the parent's, byte-identical (full inside-outside over the whole sentence). A spectrum from greedy
(commit every merge at once) to full IO (commit the whole tree after seeing the whole sentence):

  win_W    window, in SYMBOLS (a symbol is a token or a frozen chunk). The sentence is parsed in a sequence of windows of W symbols.
  win_stride  how far the window slides (default W: disjoint windows). Only the first `stride` symbols of a window are committed; the
           other W - stride symbols are look-ahead that the next window re-parses.
  io_depth_D  the number of levels of higher-order chunks committed per IO run: of a window's best parse only nodes of height <= D
           (a node over symbols has height 1 + max of its children's heights, a symbol 0) are frozen.
  io_depth_mode  "freeze" (default): IO over the whole window, then only nodes of height <= D are frozen (a staged COMMIT: the whole chart is still
           computed, so nothing is saved). "chart": the chart itself is capped, so a run only evaluates composites over at most L = 2**D
           symbols (a binary tree of height <= D has at most 2**D leaves; trees of any shape on that many symbols are allowed, a relaxation);
           the analyses are then FORESTS of such trees (a symbol alone is a tree: factor 1), inside-outside over forests (`_forest_io`),
           the MBR forest's trees are frozen, the next pass runs over those trees as symbols. A window of at most L symbols is plain full IO.
  win_learn  False: learn() uses the full-IO parse (windowing only at evaluation); True: learn() uses the windowed parse too, so the
           stored chunks, their posteriors and the stored statistics come from it.

How a parse is made (`_r218_parse`). The sentence starts as a list U of symbols (the words). One PASS over U:
  windows start at s = 0, S, 2S, ..., and the window at s holds U[s : s + W]; the window that reaches the end of U is the last.
  1. Inside-outside over the window (a CKY chart of the same composites as `_chart`, built over the window's symbols: a composite's span
     is the token span of its symbols, its two parts are the best sub-analyses of the window; same _surface/_content/_context/_factor,
     so a composite over the same parts has the same gain as in full IO; the sentence's raw tokens are always visible as context, so the
     chunk context of a composite at the window's edge still names the tokens just outside the window). logZ of the window and the
     posterior of every composite of the window come from `_inside_outside` unchanged.
  2. The window's best tree is the MBR tree (`_mbr` on the window's posteriors, as in full IO).
  3. FREEZING. The commit region is the first S symbols of the window (the whole window if it is the last one). The maximal nodes of the MBR
     tree that lie entirely inside the commit region and have height <= D are frozen: each becomes one symbol of the next list. A symbol
     in the region that no frozen node covers stays a symbol. A node that crosses the region's right edge is not frozen (its parts inside
     the region are). The frozen symbol is the chart's composite for that span (Elem with the chart's best-part children, as in full IO any
     chunk is built on the best sub-analyses); every MBR node inside a frozen node is added to the final tree's `chosen` list; nothing
     inside a frozen node is parsed again.
  4. The next window starts at the first symbol after the commit region.
After the pass the symbols are the frozen symbols in order; passes repeat on that list (a window then holds frozen chunks and tokens, so
a higher chunk is built on frozen chunks) until one symbol is left, whose span is the sentence. Every pass commits at least one merge
(the last window has two or more symbols and its region is the whole window, so it freezes at least one cherry; D >= 1).
W >= the sentence length with D = None is the full-IO parse (tested: identical chosen tree, logZ, posteriors, gains).

The returned dict is the parent's: `tree`, `words`, `comps` (every composite computed in any window, the last one computed for a span key),
`post` (its posterior in the window that last computed it; a frozen node's is its posterior in the window that froze it, normalised
within that window, not over the sentence), `chosen` (the frozen nodes = a binary tree over the sentence, preorder within each window),
`logZ` = the sum of the windows' logZ (equal to the sentence logZ when there is one window; with several windows the look-ahead
overlap counts twice, so scores are comparable across cells only through `logw`), plus `logw` = the chosen tree's log weight
sum(gain) * ln2 / tau and `stats` (windows, passes, IO triples, composite evaluations, seconds). The learner (win_learn True) stores
whatever `comps`/`post`/`chosen` say (store="posterior": every computed composite with its latest window posterior, nodes of windows
that were later superseded included).

Cost: the expensive unit is one composite evaluation (`_factor`: two Cobweb tree queries); a composite over the same two parts in a later
window is reused from a per-parse memo and not counted (`stats["factors"]` counts evaluations, `stats["triples"]` the IO triples).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

from .data import Tree
from .trellis3 import LN2, Elem, logsumexp
from .trellis3_r216 import Trellis3R216


class Trellis3R218(Trellis3R216):
    win_W: Optional[int] = None
    win_stride: Optional[int] = None
    io_depth_D: Optional[int] = None
    win_learn: bool = False

    def __init__(self, *args, win_W: Optional[int] = None, win_stride: Optional[int] = None, io_depth_D: Optional[int] = None,
                 win_learn: bool = False, io_depth_mode: str = "freeze", **kw):
        super().__init__(*args, **kw)
        assert win_W is None or win_W >= 2
        assert io_depth_D is None or io_depth_D >= 1
        assert win_stride is None or win_stride >= 1
        assert io_depth_mode in ("freeze", "chart")
        self.win_W, self.win_stride, self.io_depth_D, self.win_learn = win_W, win_stride, io_depth_D, win_learn
        self.io_depth_mode = io_depth_mode
        self._r218_nf = 0
        self._r218_learning = False
        self.r218_stats: dict = {}

    # ---- bookkeeping ----------------------------------------------------------------------------------------------
    def _factor(self, e) -> float:
        self._r218_nf += 1
        return super()._factor(e)

    def learn(self, tokens: Sequence[str], follows: bool = False) -> Tree:
        self._r218_learning = True
        try:
            return super().learn(tokens, follows)
        finally:
            self._r218_learning = False

    @property
    def r218_active(self) -> bool:
        return self.win_W is not None or self.io_depth_D is not None

    # ---- parse ----------------------------------------------------------------------------------------------------
    def parse(self, tokens: Sequence[str], search: Optional[str] = None) -> dict:
        tokens = list(tokens)
        n = len(tokens)
        use = self.r218_active and (self.win_learn or not self._r218_learning) and n >= 2 and (search or self.search) != "beam"
        t0, f0 = time.time(), self._r218_nf
        if not use:
            r = super().parse(tokens, search)
            if n >= 2 and "comps" in r:
                r["stats"] = {"windows": 1, "passes": 1, "triples": len(r["comps"]), "factors": self._r218_nf - f0,
                              "seconds": time.time() - t0, "forced": 0}
            else:
                r["stats"] = {"windows": 0, "passes": 0, "triples": 0, "factors": 0, "seconds": 0.0, "forced": 0}
            r["logw"] = sum(r["comps"][c].gain for c in r["chosen"]) * math.log(2) / self.tau if r.get("chosen") else 0.0
            self.r218_stats = r["stats"]
            return r
        self._wc_cache = {}
        self._rel_cache = {}
        self._nb_cache = {}
        assert not self.neighbour_chunks, "r218: neighbour_chunks is not supported"
        if self.r216:
            self._r216_D = self.r216_table(tokens)
        try:
            r = self._r218_parse(tokens)
        finally:
            self.__dict__.pop("_r216_D", None)
        r["stats"]["seconds"] = time.time() - t0
        r["stats"]["factors"] = self._r218_nf - f0
        self.r218_stats = r["stats"]
        return r

    def _r218_parse(self, tokens: List[str]) -> dict:
        n = len(tokens)
        words = []
        for i, w in enumerate(tokens):
            e = Elem((i, i + 1), token=w)
            e.ctx = self._context(tokens, e, analysed=False)
            words.append(e)
        acc = {"comps": {}, "post": {}, "chosen": [], "logZ": 0.0, "triples": 0, "windows": 0, "memo": {}}
        U = list(words)
        passes, forced = 0, 0
        full = False
        while len(U) > 1:
            m = len(U)
            W = m if (full or self.win_W is None) else max(2, min(self.win_W, m))
            S = W if self.win_stride is None else max(1, min(self.win_stride, W))
            D = None if full else self.io_depth_D
            newU: List[Elem] = []
            s = 0
            while s < m:
                e_ = min(s + W, m)
                last = e_ == m
                win = U[s:e_]
                if len(win) == 1:
                    newU.extend(win)
                    break
                hi = len(win) if last else S
                newU.extend(self._r218_window(tokens, win, hi, D, acc))
                if last:
                    break
                s += S
            passes += 1
            if len(newU) >= m:                        # cannot happen (see the module doc); the safe fallback is a full-IO pass
                forced += 1
                full = True
            U = newU
        assert len(U) == 1 and U[0].span == (0, n), (len(U), n)
        chosen = acc["chosen"]
        assert len(chosen) == n - 1
        split = {(i, j): k for i, k, j in chosen}
        logw = sum(acc["comps"][c].gain for c in chosen) * math.log(2) / self.tau
        return {"tree": Tree(n, split, {}, [(0, n)]), "words": words, "comps": acc["comps"], "post": acc["post"], "chosen": chosen,
                "logZ": acc["logZ"], "logw": logw,
                "stats": {"windows": acc["windows"], "passes": passes, "triples": acc["triples"], "forced": forced}}

    def _r218_window(self, tokens, win: List[Elem], hi: int, D: Optional[int], acc: dict) -> List[Elem]:
        """IO over the symbols `win`, then freeze inside the first `hi` symbols; returns the window's new symbols for that region (frozen
        nodes and uncovered symbols, in order).
        io_depth_mode "freeze": full IO over the window, freeze the maximal MBR nodes of height <= D.
        io_depth_mode "chart": the chart is capped at spans of L = 2**D symbols (fewer composites evaluated); if the window has more than L
        symbols the analyses are forests of trees of at most L symbols (`_forest_io`), the MBR forest's trees are all frozen (those inside
        the region); a window of at most L symbols is the plain full IO."""
        m = len(win)
        chart = D is not None and self.io_depth_mode == "chart"
        cap = (2 ** D if D < 30 else m) if chart else None
        forest = cap is not None and cap < m
        top = cap if forest else m
        hD = None if chart else D
        bnd = [w.span[0] for w in win] + [win[-1].span[1]]
        best = {(i, i + 1): (0.0, win[i]) for i in range(m)}
        comps_u: Dict[tuple, Elem] = {}
        memo = acc["memo"]
        for length in range(2, top + 1):
            for i in range(0, m - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    a, b = best[(i, k)][1], best[(k, j)][1]
                    mk = (bnd[i], bnd[k], bnd[j], id(a), id(b))
                    e = memo.get(mk)
                    if e is None:
                        e = Elem((bnd[i], bnd[j]), split=bnd[k], left=a, right=b)
                        e.ctx = self._surface(tokens, bnd[i], bnd[j])
                        e.cnt = self._content(e)
                        e.ctx = self._context(tokens, e, analysed=False)
                        e.gain = self._factor(e)
                        memo[mk] = e
                    comps_u[(i, k, j)] = e
                    v = best[(i, k)][0] + best[(k, j)][0] + e.gain
                    if (i, j) not in best or v > best[(i, j)][0]:
                        best[(i, j)] = (v, e)
        if forest:
            post, logZ = self._forest_io(m, comps_u, cap)
            tops, chosen = self._forest_mbr(m, post, cap)
        else:
            post, logZ = self._inside_outside(m, comps_u)
            _, chosen = self._mbr(m, post)
            tops = [(0, m)]
        acc["windows"] += 1
        acc["triples"] += len(comps_u)
        acc["logZ"] += logZ
        for (i, k, j), e in comps_u.items():
            key = (bnd[i], bnd[k], bnd[j])
            acc["comps"][key] = e
            acc["post"][key] = post[(i, k, j)]
        split = {(i, j): k for i, k, j in chosen}
        height: Dict[tuple, int] = {}

        def hgt(i, j):
            if j - i == 1:
                return 0
            if (i, j) not in height:
                k = split[(i, j)]
                height[(i, j)] = 1 + max(hgt(i, k), hgt(k, j))
            return height[(i, j)]

        def cover(i, j):
            if i >= hi:
                return []
            if j <= hi and (hD is None or hgt(i, j) <= hD):
                return [(i, j)]
            k = split[(i, j)]
            return cover(i, k) + cover(k, j)

        def collect(i, j):                                  # preorder, as `_mbr`'s walk
            if j - i < 2:
                return
            k = split[(i, j)]
            acc["chosen"].append((bnd[i], bnd[k], bnd[j]))
            collect(i, k)
            collect(k, j)

        out = []
        for ti, tj in tops:
            for i, j in cover(ti, tj):
                if j - i == 1:
                    out.append(win[i])
                else:
                    collect(i, j)
                    out.append(comps_u[(i, split[(i, j)], j)])
        return out

    def _forest_io(self, m: int, comps, L: int):
        """Inside-outside over forests of the m symbols whose trees each hold at most L symbols (a symbol alone is a tree of one: factor 1).
        Weight of a forest = product of the factors 2^(gain/tau) of its composites. Returns the posterior of every composite (i, k, j) (the
        probability that the composite is in the forest) and log Z (the log sum of the forests' weights)."""
        f = {key: e.gain * LN2 / self.tau for key, e in comps.items()}
        ins = {(i, i + 1): 0.0 for i in range(m)}
        for length in range(2, min(L, m) + 1):
            for i in range(0, m - length + 1):
                j = i + length
                ins[(i, j)] = logsumexp([f[(i, k, j)] + ins[(i, k)] + ins[(k, j)] for k in range(i + 1, j)])
        a = [0.0] + [-math.inf] * m
        for j in range(1, m + 1):
            a[j] = logsumexp([a[i] + ins[(i, j)] for i in range(max(0, j - L), j)])
        b = [-math.inf] * m + [0.0]
        for i in range(m - 1, -1, -1):
            b[i] = logsumexp([ins[(i, j)] + b[j] for j in range(i + 1, min(m, i + L) + 1)])
        logZ = a[m]
        out = {(i, j): a[i] + b[j] for (i, j) in ins}
        for length in range(min(L, m), 1, -1):
            for i in range(0, m - length + 1):
                j = i + length
                o = out[(i, j)]
                for k in range(i + 1, j):
                    t = o + f[(i, k, j)]
                    out[(i, k)] = logsumexp([out[(i, k)], t + ins[(k, j)]])
                    out[(k, j)] = logsumexp([out[(k, j)], t + ins[(i, k)]])
        post = {key: math.exp(f[key] + ins[(key[0], key[1])] + ins[(key[1], key[2])] + out[(key[0], key[2])] - logZ) for key in comps}
        return post, logZ

    @staticmethod
    def _forest_mbr(m: int, post, L: int):
        """The forest (trees of at most L symbols) maximising the expected number of correct composites, as `_mbr`: per span the best
        sub-tree by span posterior, then the partition of the symbols into trees of the largest summed value (ties: the larger tree).
        Returns the trees' spans and the composites (i, k, j) of all of them. At least one tree has two symbols."""
        span_post: Dict[tuple, float] = {}
        for (i, k, j), p in post.items():
            span_post[(i, j)] = span_post.get((i, j), 0.0) + p
        best = {(i, i + 1): (0.0, None) for i in range(m)}
        for length in range(2, min(L, m) + 1):
            for i in range(0, m - length + 1):
                j = i + length
                k = max(range(i + 1, j), key=lambda k: best[(i, k)][0] + best[(k, j)][0] + post[(i, k, j)])
                best[(i, j)] = (best[(i, k)][0] + best[(k, j)][0] + span_post[(i, j)], k)
        G, back = [0.0] + [0.0] * m, [0] * (m + 1)
        for j in range(1, m + 1):
            bv, bi = -math.inf, j - 1
            for i in range(max(0, j - L), j):
                v = G[i] + (best[(i, j)][0] if j - i >= 2 else 0.0)
                if v > bv + 1e-12:
                    bv, bi = v, i
            G[j], back[j] = bv, bi
        tops, j = [], m
        while j > 0:
            tops.append((back[j], j))
            j = back[j]
        tops.reverse()
        if all(j - i == 1 for i, j in tops):           # progress is guaranteed: join the likeliest pair
            i = max(range(m - 1), key=lambda i: span_post[(i, i + 2)])
            tops = [(x, x + 1) for x in range(i)] + [(i, i + 2)] + [(x, x + 1) for x in range(i + 2, m)]
        split, chosen = {}, []

        def walk(i, j):
            if j - i < 2:
                return
            k = best[(i, j)][1] if best[(i, j)][1] is not None else i + 1
            chosen.append((i, k, j))
            walk(i, k)
            walk(k, j)
        for i, j in tops:
            walk(i, j)
        return tops, chosen
