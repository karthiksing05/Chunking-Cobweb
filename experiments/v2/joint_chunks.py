"""Joint chunk moves: a nested composite applied at all its occurrences at once.

The greedy barrier (TRELLIS v2, characters from sequences alone): the
structure search finds an operator with its first part (``[⿰ 讠]``) but not
the join of that chunk with the second part, because each binary join, made
one at a time, gives a fresh chunk category that is rare at the top level,
whose start code costs more than the join saves. Track 1 met the same wall
with category splits and passed it with a coordinated move scored once by
the exact code. This module is the analogue for chunks.

**The move.** A *nested composite* is a binary tree whose leaves are
categories of existing elements and whose internal nodes are chunks, for
example ((A B) C). Applied, every non-overlapping occurrence of its leaves,
in reading order, is grouped as the composite says, all at once. Each
internal node takes a category: an existing chunk category whose row already
holds that pair of child categories (``reuse``; the most frequent such), or
a fresh one, one per distinct pair of child categories within the move.

* Proposals (``mine_windows``): frequent windows of 3..``max_leaves``
  adjacent top-level elements by category (the ``top_windows`` most frequent
  with at least ``min_count`` occurrences), each with every binary
  bracketing. Nothing domain-specific: categories and adjacency only.
* Scoring: exact. Inside the search (``JointState``) a composite changes a
  few rows, the start row, the continue/stop row and the alphabet size, so
  its plain code is computed from cached counts like the chunk and merge
  moves (``tests/trellis2/test_joint_chunks.py`` checks it against a full
  recomputation). As final-round refinement (``refine_final``) a composite
  may re-bracket whole trees (``restructure``: required brackets in, the
  brackets crossing them out, what is left right-branching), so candidates
  are ranked by the recomputed plain code, and the best few refitted and
  accepted only if the night's full code shrinks.

Two placements (``install``):

* ``search``: the composite is a third move of the night's beam search
  (``joint_chunk_and_merge``), next to chunk and merge, scored by the plain
  code like them.
* ``final``: after the night, composites over adjacent nodes of the final
  analyses, plain code ranks, full code (consolidation and re-analysis, as
  the night's own candidates) accepts, while one pays.

**Interface for other domains** (e.g. ``chess.BoardSearch``). The move needs
three things of a domain's search state, and nothing else:

1. ``elements``: per experience, its top-level elements, each with a
   category; and an adjacency between elements (here: next in reading
   order; on a board: the relation, a direction and distance, from one
   piece to another). A composite's leaves are categories, its internal
   nodes say which two parts join (and, with typed relations, by which
   relation: the pattern tree's nodes then carry a relation).
2. ``occurrences(pattern)``: the non-overlapping places the pattern's leaves
   occur in the stated adjacency, chosen greedily in a fixed order.
3. ``delta(pattern, n)``: the exact change of the domain's plain code if
   all n occurrences are grouped (new rows for fresh categories, counts added
   to reused rows, the top level's changes); ``apply(pattern)``.

``JointState`` implements these for sequences on ``mdl_search._State``;
``composite_delta`` is written against the counts any Dirichlet-multinomial
row code keeps (rows, row sizes, the start row, continuations), so a domain
whose search keeps such counts can reuse it unchanged.
"""
from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

from trellis2.mdl_search import LN2, Node, _phi, _State, code_bits, from_tree, to_tree

CONFIG = {
    "max_leaves": 4,        # composites of 3..max_leaves leaves (2 is the chunk move)
    "min_count": 3,         # occurrences a window needs to be proposed
    "top_windows": 60,      # most frequent windows proposed per state
    "reuse": True,          # internal nodes may take an existing chunk category
    # final-round refinement
    "final_candidates": 4,  # refitted per step (the best by plain code)
    "final_steps": 8,       # at most this many accepted composites
    "final_workers": 4,
    "log": True,
}

Span = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Shapes: binary bracketings of m leaves
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def shapes(lo: int, hi: int) -> Tuple:
    """Every binary tree over leaves lo..hi-1: a leaf is its index, an internal
    node a pair."""
    if hi - lo == 1:
        return (lo,)
    out = []
    for k in range(lo + 1, hi):
        for a in shapes(lo, k):
            for b in shapes(k, hi):
                out.append((a, b))
    return tuple(out)


def shape_spans(shape, start: int = 0) -> List[Span]:
    """The internal nodes of a shape as leaf-index spans (bottom-up order)."""
    out = []

    def rec(s):
        if isinstance(s, int):
            return (s, s + 1)
        a, b = rec(s[0]), rec(s[1])
        out.append((a[0], b[1]))
        return (a[0], b[1])
    rec(shape)
    return out


# --------------------------------------------------------------------------- #
# The exact plain-code change of a composite, from row counts
# --------------------------------------------------------------------------- #
def plan_labels(window: Tuple, shape, rows: Dict[Hashable, Counter], reuse: bool, fresh_name,
                holders: Optional[Dict] = None):
    """Categories of a composite's internal nodes, bottom up: a list of
    (label, left label, right label, is_new), and the root's label. A node
    takes the existing category whose row holds its pair most often
    (``reuse``), else a fresh one (one per distinct pair in the composite)."""
    nodes, new = [], {}
    best_holder = (holders if holders is not None else _holders(rows)) if reuse else {}

    def rec(s):
        if isinstance(s, int):
            return window[s]
        lab_l, lab_r = rec(s[0]), rec(s[1])
        o = ("p", lab_l, lab_r)
        if (lab_l, lab_r) in new:
            lab, is_new = new[(lab_l, lab_r)], True
        elif o in best_holder:
            lab, is_new = best_holder[o], False
        else:
            lab = fresh_name(len(new))
            new[(lab_l, lab_r)] = lab
            is_new = True
        nodes.append((lab, lab_l, lab_r, is_new))
        return lab
    root = rec(shape)
    return nodes, root


def _holders(rows) -> Dict[tuple, Hashable]:
    """Pair outcome -> the category whose row holds it most often (ties: the
    first in name order)."""
    best: Dict[tuple, Tuple[int, Hashable]] = {}
    for lab in sorted(rows, key=str):
        for o, c in rows[lab].items():
            if o[0] == "p" and (o not in best or c > best[o][0]):
                best[o] = (c, lab)
    return {o: lab for o, (c, lab) in best.items()}


def composite_delta(st: _State, window: Tuple, nodes, root, n: int, alphabet_cache: dict) -> float:
    """Nats after applying a composite with n occurrences to state ``st`` (the
    code of ``mdl_search._State``, computed from the changed rows only)."""
    a, lg, phi = st.alpha, math.lgamma, st._phi_one
    new_labels = []
    added: Dict[Hashable, Counter] = defaultdict(Counter)
    for lab, l, r, is_new in nodes:
        added[lab][("p", l, r)] += n
        if is_new and lab not in new_labels:
            new_labels.append(lab)
    K = len(st.rows) + len(new_labels)
    if K not in alphabet_cache:
        alphabet_cache[K] = st._alphabet_term(K)
    aa = (st.n_tokens + K * K) * a
    alphabet, s_total = alphabet_cache[K], st.s_total
    for lab, add in added.items():
        if lab in st.rows:
            row, n0 = st.rows[lab], st.row_n[lab]
            n1 = n0 + sum(add.values())
            alphabet += lg(n1 + aa) - lg(n0 + aa)
            s_total += sum(phi(row.get(o, 0) + c) - phi(row.get(o, 0)) for o, c in add.items())
        else:
            alphabet += lg(sum(add.values()) + aa)
            s_total += sum(phi(c) for c in add.values())
    # The top level: m leaves leave it, the root joins it, per occurrence.
    m = len(window)
    change = Counter()
    for x in window:
        change[x] -= n
    change[root] += n
    start_s = st.start_s
    for x, d in change.items():
        if d:
            c0 = st.start.get(x, 0)
            start_s += phi(c0 + d) - phi(c0)
    return st._nats(K, alphabet, s_total, st.start_n - n * (m - 1), start_s, st.cont - n * (m - 1))


# --------------------------------------------------------------------------- #
# Proposals: frequent windows of top-level categories
# --------------------------------------------------------------------------- #
def mine_windows(label_seqs: Sequence[Sequence[Hashable]], max_leaves: int, min_count: int,
                 top: int) -> List[Tuple[Tuple, int]]:
    """The ``top`` most frequent windows of 3..max_leaves adjacent categories
    (counted with overlaps) with at least ``min_count`` non-overlapping
    occurrences, as (window, non-overlapping count), most frequent first."""
    cnt: Counter = Counter()
    for labs in label_seqs:
        L = len(labs)
        for m in range(3, min(max_leaves, L) + 1):
            for i in range(L - m + 1):
                cnt[tuple(labs[i:i + m])] += 1
    cands = [w for w, c in cnt.items() if c >= min_count]
    cands.sort(key=lambda w: (-cnt[w], str(w)))
    out = []
    for w in cands[:top * 2]:
        n = sum(len(match_positions(labs, w)) for labs in label_seqs)
        if n >= min_count:
            out.append((w, n))
        if len(out) == top:
            break
    return out


def match_positions(labs: Sequence[Hashable], window: Tuple) -> List[int]:
    """Non-overlapping occurrences of a window in a category sequence, left to
    right."""
    m, out, i = len(window), [], 0
    while i + m <= len(labs):
        if tuple(labs[i:i + m]) == window:
            out.append(i)
            i += m
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Inside the search: a third move of the beam
# --------------------------------------------------------------------------- #
class JointState(_State):
    """``mdl_search._State`` with composite moves next to chunk and merge."""

    max_leaves = 4
    min_count = 3
    top_windows = 60
    reuse = True

    def _child(self) -> "JointState":
        c = _State._child(self)
        c.__class__ = type(self)
        return c

    def scored_moves(self):
        yield from _State.scored_moves(self)
        yield from self.composite_moves()

    def composite_moves(self):
        seqs = [self._labels(t) for t in self.tops]
        cache: dict = {}
        holders = _holders(self.rows) if self.reuse else {}
        for window, n in mine_windows(seqs, self.max_leaves, self.min_count, self.top_windows):
            for shape in shapes(0, len(window)):
                nodes, root = plan_labels(window, shape, self.rows, self.reuse,
                                          lambda k: ("chunk", self.fresh + k), holders)
                yield composite_delta(self, window, nodes, root, n, cache), ("composite", window, shape)

    def apply(self, move):
        if move[0] != "composite":
            return _State.apply(self, move)
        _, window, shape = move
        nodes, root = plan_labels(window, shape, self.rows, self.reuse, lambda k: ("chunk", self.fresh + k))
        label_of = {}                      # internal span (leaf indices) -> label
        for (lab, _, _, _), span in zip(nodes, shape_spans(shape)):
            label_of[span] = lab
        c = self._child()
        n_new = len({lab for lab, _, _, is_new in nodes if is_new})
        c.fresh += n_new
        tops, n = [], 0
        for sentence in self.tops:
            labs = self._labels(sentence)
            pos = match_positions(labs, window)
            if not pos:
                tops.append(sentence)
                continue
            out, t = [], 0
            for i in pos:
                out.extend(sentence[t:i])
                out.append(_build(shape, sentence[i:i + len(window)], label_of))
                t = i + len(window)
                n += 1
            out.extend(sentence[t:])
            tops.append(out)
        c.tops = tops
        added: Dict[Hashable, Counter] = defaultdict(Counter)
        for lab, l, r, _ in nodes:
            added[lab][("p", l, r)] += n
        for lab, add in added.items():
            row = Counter(self.rows.get(lab, Counter()))
            old_s = self.row_s.get(lab, 0.0)
            row.update(add)
            c.rows[lab] = row
            c.row_n[lab] = self.row_n.get(lab, 0) + sum(add.values())
            c.row_s[lab] = self._phis(row.values())
            c.s_total += c.row_s[lab] - old_s
            for o in add:
                for z in (o[1], o[2]):
                    c.parents[z] = set(c.parents.get(z, ())) | {lab}
        change = Counter()
        for x in window:
            change[x] -= n
        change[root] += n
        for x, d in change.items():
            if d:
                c0 = c.start.get(x, 0)
                c.start_s += self._phi_one(c0 + d) - self._phi_one(c0)
                c.start[x] = c0 + d
        c.start_n -= n * (len(window) - 1)
        c.cont -= n * (len(window) - 1)
        c.start = Counter({k: v for k, v in c.start.items() if v})
        c._set_nats()
        return c


def _build(shape, leaves: Sequence[Node], label_of: Dict[Span, Hashable]) -> Node:
    def rec(s):
        if isinstance(s, int):
            return (s, s + 1), leaves[s]
        (i, k), a = rec(s[0])
        (_, j), b = rec(s[1])
        return (i, j), (label_of[(i, j)], (a, b))
    return rec(shape)[1]


def _describe(move) -> str:
    if move[0] == "composite":
        return f"composite({' '.join(map(str, move[1]))} as {move[2]})"
    return f"{move[0]}({move[1]}, {move[2]})"


def joint_chunk_and_merge(analyses: List[List[Node]], n_tokens: int, alpha: float, beam: int = 1,
                          patience: int = 0, max_steps: int = 500, log=None,
                          state_class=JointState) -> Tuple[List[List[Node]], float]:
    """``mdl_search.chunk_and_merge`` with composite moves in the beam (the
    same search otherwise: exact scores, the beam's best distinct
    successors, Stolcke's lookahead)."""
    best = state_class(analyses, n_tokens, alpha)
    frontier, stale = [best], 0
    for step in range(max_steps):
        scored = sorted(((nats, i, move) for i, st in enumerate(frontier)
                         for nats, move in st.scored_moves()), key=lambda x: round(x[0], 7))
        successors = []
        for nats, i, move in scored:
            if len(successors) == beam:
                break
            child = frontier[i].apply(move)
            ties = [s for s, _ in successors if abs(s.nats - child.nats) < 1e-7]
            if any(s.signature() == child.signature() for s in ties):
                continue
            successors.append((child, move))
        if not successors:
            break
        frontier = [c for c, _ in successors]
        if frontier[0].nats < best.nats - 1e-9 * LN2:
            best, stale = frontier[0], 0
            if log:
                log(step, _describe(successors[0][1]), best.bits)
        else:
            stale += 1
            if stale > patience:
                break
    return best.analyses, best.bits


def _search_joint(job):
    """One structure search of the night with composite moves (a process's job)."""
    start, n_tokens, alpha, beam, patience = job[:5]
    cfg = job[5] if len(job) > 5 else CONFIG
    state_class = type("JointStateCfg", (JointState,), {
        "max_leaves": cfg["max_leaves"], "min_count": cfg["min_count"],
        "top_windows": cfg["top_windows"], "reuse": cfg["reuse"]})
    moves = Counter()

    def log(step, move, bits):
        moves[move.split("(")[0]] += 1
    t0 = time.time()
    analyses, bits = joint_chunk_and_merge(start, n_tokens, alpha, beam=beam, patience=patience,
                                           state_class=state_class, log=log)
    if cfg.get("log"):
        print(f"    joint search: {bits:.1f} bits, improving moves {dict(moves)}, "
              f"{time.time() - t0:.0f}s", flush=True)
    return analyses, bits


_search_joint.is_search = True


# --------------------------------------------------------------------------- #
# Re-bracketing analyses (final round; the oracle diagnostic)
# --------------------------------------------------------------------------- #
def flatten(tops: List[Node]):
    """(tokens, labels by span, composite spans, roots) of a symbolic analysis."""
    toks, lab, comp, roots = [], {}, set(), []

    def rec(node, i):
        l, body = node
        if isinstance(body, str):
            toks.append(body)
            lab[(i, i + 1)] = l
            return i + 1
        k = rec(body[0], i)
        j = rec(body[1], k)
        lab[(i, j)] = l
        comp.add((i, j))
        return j
    pos = 0
    for t in tops:
        j = rec(t, pos)
        roots.append((pos, j))
        pos = j
    return toks, lab, comp, roots


def crosses(a: Span, b: Span) -> bool:
    return a[0] < b[0] < a[1] < b[1] or b[0] < a[0] < b[1] < a[1]


class Labeler:
    """Fresh categories for new nodes: one per distinct pair of child
    categories (shared by every analysis it labels)."""

    def __init__(self, prefix="jc"):
        self.prefix, self.table = prefix, {}

    def __call__(self, l, r):
        return self.table.setdefault((l, r), (self.prefix, len(self.table)))


def restructure(tops: List[Node], required: Set[Span], labeler) -> List[Node]:
    """The analysis with every required span a constituent: the existing
    brackets crossing a required one are dropped, and any node left with more
    than two parts is joined right-branching (inner joins keep the node's
    category, as ``mdl_search.joined`` does; a new node's category comes from
    ``labeler(left, right)``). Required spans must not cross each other."""
    toks, lab, comp, roots = flatten(tops)
    req = {s for s in required if s[1] - s[0] >= 2}
    keep = {s for s in comp if not any(crosses(s, r) for r in req)}
    family = keep | req | {(i, i + 1) for i in range(len(toks))}
    children: Dict[Span, List[Span]] = defaultdict(list)
    top: List[Span] = []
    stack: List[Span] = []
    for s in sorted(family, key=lambda s: (s[0], -s[1])):
        while stack and stack[-1][1] <= s[0]:
            stack.pop()
        (children[stack[-1]] if stack else top).append(s)
        stack.append(s)

    def node(s) -> Node:
        if s[1] - s[0] == 1:
            return (lab[s], toks[s[0]])
        parts = [node(c) for c in children[s]]
        own = lab.get(s) if s in keep else None
        right = parts[-1]
        for p in reversed(parts[1:-1]):
            right = (own if own is not None else labeler(p[0], right[0]), (p, right))
        return (own if own is not None else labeler(parts[0][0], right[0]), (parts[0], right))
    return [node(s) for s in top]


def window_occurrences(tops: List[Node], max_leaves: int, top_only: bool = False):
    """Every window of 3..max_leaves adjacent existing nodes (any level, or
    top-level elements only) of an analysis: (leaf spans, leaf categories)."""
    toks, lab, comp, roots = flatten(tops)
    starting: Dict[int, List[Span]] = defaultdict(list)
    for s in (roots if top_only else sorted(lab)):
        starting[s[0]].append(s)
    n, out = len(toks), []

    def extend(seq):
        if len(seq) >= 3:
            out.append((tuple(seq), tuple(lab[s] for s in seq)))
        if len(seq) == max_leaves or seq[-1][1] >= n:
            return
        for s in starting[seq[-1][1]]:
            extend(seq + [s])
    for i in range(n):
        for s in starting[i]:
            extend([s])
    return out


def required_of(leaf_spans: Sequence[Span], shape) -> Set[Span]:
    return {(leaf_spans[a][0], leaf_spans[b - 1][1]) for a, b in shape_spans(shape)}


def apply_tree_composite(analyses: List[List[Node]], window: Tuple, shape, labeler, max_leaves: int,
                         top_only: bool = False):
    """A composite over adjacent nodes of (possibly whole) analyses: per
    analysis, its occurrences left to right, each kept if its brackets cross
    none kept before; returns the new analyses and the occurrences applied."""
    out, n = [], 0
    for tops in analyses:
        req: Set[Span] = set()
        have = None
        for spans, labs in window_occurrences(tops, max_leaves, top_only):
            if labs != window:
                continue
            r = required_of(spans, shape)
            if have is None:
                _, _, comp, roots = flatten(tops)
                have = comp | set(roots)
            # An occurrence already so bracketed is not one the move makes.
            if r <= have or any(crosses(a, b) for a in r for b in req) or r <= req:
                continue
            req |= r
            n += 1
        out.append(restructure(tops, req, labeler) if req else tops)
    return out, n


# --------------------------------------------------------------------------- #
# Final-round refinement
# --------------------------------------------------------------------------- #
def tree_candidates(analyses, n_tokens: int, alpha: float, cfg: dict, round_no: int):
    """Composites over adjacent nodes of the analyses, ranked by the exact
    plain code (recomputed)."""
    cnt = Counter()
    for tops in analyses:
        seen = set()
        for spans, labs in window_occurrences(tops, cfg["max_leaves"]):
            if (spans[0][0], labs) not in seen:
                seen.add((spans[0][0], labs))
                cnt[labs] += 1
    windows = [w for w, c in cnt.items() if c >= cfg["min_count"]]
    windows.sort(key=lambda w: (-cnt[w], str(w)))
    base = code_bits(analyses, n_tokens, alpha)
    ranked = []
    for w in windows[:cfg["top_windows"]]:
        for shape in shapes(0, len(w)):
            lab = Labeler(prefix=f"jf{round_no}")
            new, n = apply_tree_composite(analyses, w, shape, lab, cfg["max_leaves"])
            if n < cfg["min_count"]:
                continue
            ranked.append((code_bits(new, n_tokens, alpha) - base, w, shape, n, new))
    ranked.sort(key=lambda r: r[0])
    return base, ranked


def _consolidate_job(job):
    learner, analyses = job
    _, bits, trees, steps = learner._consolidate([to_tree(a) for a in analyses])
    return bits, trees, steps


def refine_final(learner, cfg: Optional[dict] = None, log=print) -> List[dict]:
    """Composite moves on the night's final analyses: plain code ranks, the
    full code (consolidation and re-analysis, as the night's candidates)
    accepts, while one pays. Replaces the learner's model and analyses."""
    cfg = dict(CONFIG, **(cfg or {}))
    n_tokens = len({w for s in learner.experiences for w in s}) + 1
    model, bits, trees = learner.model, learner.model.grammar.info["total bits"], learner.trees
    analyses = learner.analyses
    history = [{"step": -1, "full bits": bits}]
    tried: Set = set()
    for step in range(cfg["final_steps"]):
        t0 = time.time()
        base, ranked = tree_candidates(analyses, n_tokens, learner.alpha, cfg, step)
        ranked = [r for r in ranked if (r[1], r[2]) not in tried][:cfg["final_candidates"]]
        if not ranked:
            break
        worker = learner._for_worker()
        jobs = [(worker, r[4]) for r in ranked]
        # In parallel only where the night itself may be (``workers``).
        n_workers = min(cfg["final_workers"], max(learner.workers, 1), len(jobs))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                outs = list(pool.map(_consolidate_job, jobs))
        else:
            outs = [_consolidate_job(j) for j in jobs]
        best = None
        for r, (b, t, steps) in zip(ranked, outs):
            tried.add((r[1], r[2]))
            rec = {"step": step, "window": [str(x) for x in r[1]], "shape": str(r[2]), "occurrences": r[3],
                   "plain delta": r[0], "full delta": b - bits, "seconds": time.time() - t0}
            history.append(rec)
            if log:
                log(f"    final composite {rec}")
            if b < bits - 1e-6 and (best is None or b < best[0]):
                best = (b, t, rec)
        if best is None:
            break
        bits, trees, rec = best
        rec["accepted"] = True
        model = learner.fit(trees)
        bits = model.grammar.info["total bits"]
        analyses = learner._in_categories(model, trees)
    learner.model, learner.trees, learner.analyses = model, trees, analyses
    return history


# --------------------------------------------------------------------------- #
# Installing a placement in this process
# --------------------------------------------------------------------------- #
def install(variant: str, config: Optional[dict] = None) -> List[str]:
    """``search``: the night's structure searches use ``joint_chunk_and_merge``.
    ``final``: ``refine_final`` after the night (the night itself unchanged)."""
    import trellis2.unsupervised as U
    CONFIG.update(config or {})
    if variant == "search":
        U._search = _SEARCH_FN
        _SEARCH_CFG.clear()
        _SEARCH_CFG.update(CONFIG)
        return ["trellis2.unsupervised._search (joint_chunks._search_with_config)"]
    if variant == "candidate":
        _SEARCH_CFG.clear()
        _SEARCH_CFG.update(CONFIG)
        _CANDIDATE[0] = True
        return ["UnsupervisedLearner.sleep (joint_chunks.candidate_sleep: searches with and without "
                "the joint chunk move, both kinds consolidated, the full code chooses)"]
    if variant == "final":
        _FINAL[0] = True
        return ["UnsupervisedLearner.sleep (+ joint_chunks.refine_final on the final analyses)"]
    raise ValueError(variant)


_SEARCH_CFG: dict = {}
_FINAL = [False]
_CANDIDATE = [False]


def _search_either(job):
    """One structure search of the night, without (``plain``) or with
    (``joint``) the joint chunk move (a process's job)."""
    kind, job = job
    if kind == "joint":
        return _search_with_config(job)
    import trellis2.unsupervised as U
    return U._search(job)


_search_either.is_search = True


def candidate_sleep(self):
    """``UnsupervisedLearner.sleep`` with the joint chunk move as a candidate:
    every start is searched without and with the move, the best few distinct
    results of each kind (and each with its forests joined) are consolidated,
    and the shortest full code chooses among all of them, as the night always
    does. Everything after the choice is the night unchanged."""
    import trellis2.unsupervised as U
    from trellis2.mdl_search import joined, word_classes
    t0 = time.time()
    night = self.nights
    n_tokens = len({w for s in self.experiences for w in s}) + 1

    def log(stage, move, bits, seconds=None):
        self.history.append({"night": night, "stage": stage, "move": move, "bits": bits,
                             "sentences": len(self.experiences),
                             "seconds": time.time() - t0 if seconds is None else seconds})

    path = word_classes(self.experiences, self.alpha)
    starts = [(f"{len(set(cls.values()))} word classes",
               [[(("w", cls[w]), w) for w in s] for s in self.experiences])
              for cls in path[-self.levels:]]
    if self.model is not None:
        starts.append(("perceived analyses", self.analyses))
    flat = [[(("w", path[-1][w]), w) for w in s] for s in self.experiences]
    log("flat", "start", code_bits(flat, n_tokens, self.alpha))
    kinds = ("plain", "joint")
    jobs = [(k, (start, n_tokens, self.alpha, self.beam, self.patience)) for k in kinds for _, start in starts]
    searched = self._map(_search_either, jobs)
    search_seconds = time.time() - t0
    candidates = []
    for k in kinds:
        results = sorted([(bits, name + ("" if k == "plain" else ", with joint chunks"), analyses)
                          for (kk, _), (name, _), (analyses, bits) in
                          zip(jobs, starts * len(kinds), searched) if kk == k], key=lambda r: r[0])
        mine = []
        for r in results:
            if all(abs(r[0] - c[0]) > 1e-6 for c in mine):
                mine.append(r)
            if len(mine) == self.consolidations:
                break
        candidates += mine
    if self.join_forests:
        candidates += [(code_bits(whole, n_tokens, self.alpha), f"{name}, forests joined", whole)
                       for _, name, analyses in list(candidates) if any(len(a) > 1 for a in analyses)
                       for whole in [[joined(a) for a in analyses]]]
    if self.workers > 1 and len(candidates) > 1:
        worker = self._for_worker()
        outcomes = self._map(U._consolidated, [(worker, [to_tree(a) for a in analyses])
                                               for _, _, analyses in candidates])
    else:
        outcomes = (self._consolidate([to_tree(a) for a in analyses]) for _, _, analyses in candidates)
    best = None
    self.candidate_codes = []
    for outcome, (bits_plain, name, _) in zip(outcomes, candidates):
        self.candidate_codes.append({"candidate": name, "plain bits": bits_plain, "full bits": outcome[1]})
        if best is None or outcome[1] < best[1] - 1e-6:
            best = outcome + (name, bits_plain)
    model, bits, trees, steps, name, bits_plain = best
    if model is None:
        model = self.fit(trees)
    log("structure", f"chunk and merge from {name}", bits_plain, search_seconds)
    log("concepts", "consolidate", steps[0])
    for b in steps[1:]:
        log("re-analysis", "viterbi", b)
    if self.context_free_steps and model.memory.contexts() is not None:
        plain = self._for_worker()
        plain.trellis_kwargs = dict(self.trellis_kwargs, previous_word=False)
        plain.reanalysis_steps = self.context_free_steps
        _, _, found, plain_steps = plain._consolidate(trees)
        if len(plain_steps) > 1:
            m2, b2, t2, s2 = self._consolidate(found)
            if b2 < bits - 1e-6:
                for b in plain_steps[1:]:
                    log("re-analysis", "viterbi without the read's context (its code)", b)
                log("concepts", "consolidate with the read's context", s2[0])
                for b in s2[1:]:
                    log("re-analysis", "viterbi", b)
                model, bits, trees = m2, b2, t2
    if self.sampling:
        model, bits, trees = self._sampled(model, bits, trees, log)
    self.model, self.trees = model, trees
    self.analyses = self._in_categories(model, trees)
    self.nights += 1
    print(f"    night {night}: winner '{name}'; candidates "
          + "; ".join(f"{c['candidate']}: plain {c['plain bits']:.0f}, full {c['full bits']:.0f}"
                      for c in self.candidate_codes), flush=True)
    return model.grammar


def _search_with_config(job):
    return _search_joint(tuple(job[:5]) + (dict(CONFIG, **_SEARCH_CFG),))


_search_with_config.is_search = True
_SEARCH_FN = _search_with_config


def LEARNER_CLASS(base):
    """The learner class for the installed placement: with ``final``, the
    night is followed by ``refine_final``."""
    if _CANDIDATE[0]:
        name = f"Candidate_{base.__name__}"
        if name not in globals():
            cls = type(name, (base,), {"sleep": candidate_sleep, "candidate_codes": None,
                                       "__module__": __name__})
            cls.__qualname__ = name
            globals()[name] = cls
        return globals()[name]
    if not _FINAL[0]:
        return base

    name = f"FinalRefined_{base.__name__}"
    if name not in globals():
        def sleep(self):
            base.sleep(self)
            self.joint_log = refine_final(self)
            return self.model.grammar
        # A module-level class, so that the learner pickles into worker processes.
        cls = type(name, (base,), {"joint_log": None, "sleep": sleep, "__module__": __name__})
        cls.__qualname__ = name
        globals()[name] = cls
    return globals()[name]


# --------------------------------------------------------------------------- #
# Step 1: the oracle diagnostic (gold IDS subtrees, evaluation only)
# --------------------------------------------------------------------------- #
def run_diagnostic(args):
    from run_joint_chunks_diag import run
    run(args)
