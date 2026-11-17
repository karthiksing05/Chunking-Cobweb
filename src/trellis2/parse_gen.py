"""Round 202: generate through the parse (opt-in; nothing here changes Trellis3).

The causal story generator reads the next word from the window l1..lW alone. Here the generation state is the
learner's own parse of the prefix: the chart of every (span, split) of the words placed so far, with the factors
2^(gain / tau) that inside-outside uses (Trellis3._chart, Trellis3._factor, Trellis3._inside_outside), read with
the right edge of the prefix OPEN (the window slots past the last word are the unwritten future: left out of each
description, as Cobweb sorts partial instances) or CLOSED (end markers, exactly as learning describes a sentence).

A candidate next word w (with "go on" or "end here") is weighted by the window read (the proposal, unchanged) times
the inside score of the extended prefix,

    P(w, go)  propto  q(w, go)  * Z_open(prefix + w)
    P(w, end) propto  q(w, end) * Z_closed(prefix + w)

so w is judged by how the words before it are analysed once it stands to their right: its identity enters every
element whose right window reaches it (the context concept a pivot sorts to with w as its r1, the chunks over
the pivot, the content concepts of those chunks), not only through the window read. The end is the same read with
the edge closed: Z_closed / Z_open is how much the chart's concepts expect the end markers where they would stand
(the closure of the spanning analysis), so a sentence ends where its analysis is a complete one. One read, both
decisions; no new constant (tau is the parser's own, 1).

PrefixChart memoises every element by its local token window (span plus W words each side, the open / closed flag
when the window reaches the edge): an element's description depends on nothing else, so prefixes shared across
draws and candidates are parsed once.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .trellis3 import BOS, EOS, LN2, Elem, logsumexp


class PrefixChart:
    """Inside scores of token prefixes under a learned Trellis3's chart, right edge open or closed."""

    def __init__(self, L, drop: Sequence[str] = ("pv",), cap: int = 250000):
        bad = [k for k, v in (("children", L.children), ("spine", L.spine), ("neighbour_chunks", L.neighbour_chunks),
                              ("neighbour_concepts", L.neighbour_concepts), ("neighbour_types", L.neighbour_types),
                              ("window_names", L.window_names), ("relational", L.relational),
                              ("conj", L.__dict__.get("conj")), ("edge_once", L.__dict__.get("edge_once")),
                              ("search beam", L.search == "beam"), ("factor", L.factor != "gain"),
                              ("hierarchy", L.hierarchy != "two"), ("boundary", L.boundary != "marker"),
                              ("donate_own", L.donate_own))
               if v]
        if bad:      # descriptions that read beyond the element's local window, or a different parser
            raise NotImplementedError("PrefixChart supports the r184+ story recipe only; set: %s" % bad)
        self.L, self.W, self.drop, self.cap = L, L.window, tuple(drop), cap
        self.memo: Dict[tuple, Tuple[float, Elem, float]] = {}
        self.tops: Dict[tuple, tuple] = {}      # span key -> (inside terms, composites) of its splits
        self._s: Dict[int, float] = {}
        self.calls = self.built = 0

    # -- descriptions ------------------------------------------------------------------------------------
    def _ctx(self, toks: List[str], e: Elem, closed: bool) -> dict:
        x = self.L._context(toks, e, analysed=False)
        n, j = len(toks), e.span[1]
        if not closed:                   # the future: window slots past the last word are not observed
            for k in range(1, self.W + 1):
                if j + k - 1 >= n:
                    x.pop(f"r{k}", None)
        for a in self.drop:              # the previous sentence (story facet): generation is unconditioned
            x.pop(a, None)
        return x

    def _key(self, toks: List[str], i: int, j: int, closed: bool) -> tuple:
        n = len(toks)
        a, b = max(0, i - self.W), min(n, j + self.W)
        return (tuple(toks[a:b]), i - a, j - i, bool(closed) if j + self.W > n else None)

    def span(self, toks: List[str], i: int, j: int, closed: bool) -> Tuple[float, Elem, float]:
        """(Viterbi score, best element, log inside) of span (i, j), as Trellis3._chart / _inside_outside."""
        key = self._key(toks, i, j, closed)
        got = self.memo.get(key)
        if got is not None:
            return got
        L = self.L
        if j - i == 1:
            e = Elem((i, j), token=toks[i])
            e.ctx = self._ctx(toks, e, closed)
            res = (0.0, e, 0.0)
        else:
            best: Optional[Tuple[float, Elem]] = None
            terms, comps = [], []
            for k in range(i + 1, j):
                vl, a, il = self.span(toks, i, k, closed)
                vr, b, ir = self.span(toks, k, j, closed)
                e = Elem((i, j), split=k, left=a, right=b)
                e.ctx = L._surface(toks, i, j)
                e.cnt = L._content(e)
                e.ctx = self._ctx(toks, e, closed)
                e.gain = L._factor(e)
                self.built += 1
                terms.append(e.gain * LN2 / L.tau + il + ir)
                comps.append(e)
                v = vl + vr + e.gain
                if best is None or v > best[0]:
                    best = (v, e)
            res = (best[0], best[1], logsumexp(terms))
            self.tops[key] = (terms, comps)
        if len(self.memo) >= self.cap:
            self.memo.clear()
            self.tops.clear()
        self.memo[key] = res
        return res

    def logZ(self, toks: Sequence[str], closed: bool) -> float:
        """log Z of the prefix (natural log; 0 for a single word: no composite)."""
        toks = list(toks)
        self.calls += 1
        if len(toks) < 2:
            return 0.0
        return self.span(toks, 0, len(toks), closed)[2]

    def sent_share(self, node) -> float:
        """s(C): the share of a content concept's stored composite mass that is a whole committed sentence (SENT),
        backed off along the path with one pseudo-count, s(C) = (sent_C + s(parent)) / (count_C + 1), from the
        root's own share."""
        got = self._s.get(node.id)
        if got is None:
            sent = sum(self.L.cnt.passive(node, "sent").values())
            if node.parent is None:
                got = min(1.0, sent / max(node.count, 1e-9))
            else:
                got = min(1.0, (sent + self.sent_share(node.parent)) / (node.count + 1.0))   # SENT counts 1 per sentence, the concept its posterior mass
            self._s[node.id] = got
        return got

    def closure(self, toks: Sequence[str]) -> float:
        """r202 v2: the closure probability of a whole candidate sentence: over the closed chart's spanning
        composites (one per top split, weighted by its share of Z), the share s(C) of their content concept C
        that memory committed as whole sentences. A spanning analysis built of parts that never make a sentence
        together (an open determiner phrase, a dangling preposition) sorts to concepts of non-sentence chunks."""
        toks = list(toks)
        if len(toks) < 2:
            return self.sent_share(self.L.cnt.root)
        key = self._key(toks, 0, len(toks), True)
        if key not in self.tops:
            self.memo.pop(key, None)
            self.span(toks, 0, len(toks), True)
            if key not in self.tops:      # cleared at the cap while being built: build once more
                self.memo.pop(key, None)
                self.span(toks, 0, len(toks), True)
        terms, comps = self.tops[key]
        top = max(terms)
        ws = [math.exp(t - top) for t in terms]
        out = 0.0
        for w, e in zip(ws, comps):
            if "sent" not in e.memo:
                e.memo["sent"] = self.sent_share(self.L.cnt.categorize(e.cnt)[-1])
            out += w * e.memo["sent"]
        return out / sum(ws)


def parse_step(PC: PrefixChart, out: Sequence[str], vs: List[tuple], ws: List[float], words_of, rng, temp: float = 1.0,
               M: int = 16, stats: Optional[dict] = None, mode: str = "parse"):
    """One generation step through the parse: M proposals from the window read's weights ws (already raised to
    1 / temp), each distinct candidate (v, end) reweighted by Z(out + words(v)) ** (1 / temp) with the edge closed
    for an end, open otherwise; one is drawn by sampling-importance-resampling. Returns the chosen (v, end).
    mode "close" (ablation: the closure gate alone): a go-on candidate keeps weight 1 and an end candidate is
    weighted by (Z_closed / Z_open)(out + words(v)) ** (1 / temp), the chart's odds for closing the sentence there."""
    idx = rng.choices(range(len(vs)), weights=ws, k=M)
    cnt: Dict[int, int] = {}
    for j in idx:
        cnt[j] = cnt.get(j, 0) + 1
    cands = sorted(cnt)
    if mode == "close":
        lz = []
        for j in cands:
            if vs[j][1]:
                toks = list(out) + list(words_of(vs[j][0]))
                lz.append(PC.logZ(toks, closed=True) - PC.logZ(toks, closed=False))
            else:
                lz.append(0.0)
    elif mode in ("parse2", "close2"):
        # r202 v2: the end read as a closure probability c (PrefixChart.closure) against its base rate c0 (the
        # content root's sentence share): end x c / c0, go on x (1 - c) / (1 - c0); "parse2" also keeps the open
        # chart's Z for every candidate (both the same edge, so end and go are compared on equal descriptions)
        c0 = PC.sent_share(PC.L.cnt.root)
        lz = []
        for j in cands:
            toks = list(out) + list(words_of(vs[j][0]))
            c = min(max(PC.closure(toks), 1e-12), 1 - 1e-12)
            z = math.log(c / c0) if vs[j][1] else math.log((1 - c) / (1 - c0))
            if mode == "parse2":
                z += PC.logZ(toks, closed=False)
            lz.append(z)
    else:
        lz = [PC.logZ(list(out) + list(words_of(vs[j][0])), closed=bool(vs[j][1])) for j in cands]
    top = max(lz)
    wts = [cnt[j] * math.exp((z - top) / temp) for j, z in zip(cands, lz)]
    pick = rng.choices(cands, weights=wts)[0]
    if stats is not None:
        stats["steps"] = stats.get("steps", 0) + 1
        stats["distinct"] = stats.get("distinct", 0) + len(cands)
        first = max(cands, key=lambda j: cnt[j])
        stats["moved"] = stats.get("moved", 0) + (pick != first)
        stats["spread"] = stats.get("spread", 0.0) + (top - min(lz))
    return vs[pick]
