"""TRELLIS v3: v1's two hierarchies with chunk context, representations donated
between them, and inside-outside parsing.

Iterates directly on TRELLIS v1 (ACS paper): every element has a context
description (how it sits in the experience) and, if composite, a content
description (what it is made of); each kind is organised by its own Cobweb
hierarchy. v3 strengthens what each description holds and lets each hierarchy
name things in the other's terms:

* context instance: the surface window (two tokens either side, over the
  element's outermost positions); chunk context — the context concepts of its
  two children and, when the element is in the experience's analysis, of its
  parent and sibling; and its own content concept (donated from the content
  hierarchy);
* content instance: each part's context concept (as in v1), each part's content
  concept (donated back), and each part's height (v1's complexity tag).

Concepts are named at depths 1..D of their hierarchy, so a description can
match coarsely or finely. A composite's parts are the best sub-analyses of
their spans, found bottom-up.

Parsing is inside-outside over every analysis: each composite (span and split)
carries a factor 2^(gain / tau), its gain being how much better its content and
context descriptions are coded by the concepts they sort to than by the roots;
the posterior of every composite follows, and the reported analysis maximises
the expected number of correct composites. Descriptions that need an analysis
(parent, sibling) are left out while parsing: Cobweb sorts partial instances.

Learning stores, per experience, every word and (``store``) every composite
weighted by its posterior, every composite equally, or the best analysis only.
Stored descriptions are refreshed as the concepts they name move (bounded per
experience).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Tree
from .incremental_cobweb import CobwebNode, IncrementalCobwebTree

BOS, EOS, BLANK = "<s>", "</s>", "-"
LN2 = math.log(2.0)
CONST = "const"      # passive: constituent (posterior) mass of the stored elements
WORD = "word"        # passive: the word a context concept's word elements are (v1's content reference)
# generation statistics, held by the concepts themselves (passive: never sorted on)
FILL = "fill"        # at a context concept: the committed instances that stood there (("w", token) | ("c", content leaf id))
LPART, RPART = "lpart", "rpart"   # at a content concept: where its parts stood ((context leaf id, is primitive))
SENT = "sent"        # at a content concept: committed whole experiences below it (content leaf id)


def log_prob(node: CobwebNode, x: dict, root: CobwebNode, alpha: float) -> float:
    """ln P(x | node), Laplace-smoothed, attributes independent; x may be partial.
    alpha < 0: a total pseudo-count |alpha| spread over the attribute's values
    (|alpha| / size each), so smoothing does not swamp small concepts when an
    attribute has many values."""
    total = 0.0
    for a, v in x.items():
        d = node.av.get(a, {})
        size = len(root.av.get(a, {})) + 1
        al = alpha if alpha >= 0 else -alpha / size
        if isinstance(v, dict):        # a bag: the expected log probability of its members
            total += sum(p * math.log((d.get(u, 0.0) + al) / (node.count + al * size)) for u, p in v.items())
        else:
            total += math.log((d.get(v, 0.0) + al) / (node.count + al * size))
    return total


def _bag(v) -> Dict[str, float]:
    if isinstance(v, dict):
        t = sum(v.values()) or 1.0
        return {u: p / t for u, p in v.items()}
    return {v: 1.0}


def _drift(old: dict, new: dict) -> float:
    """Mean total-variation distance between two descriptions, over attributes."""
    keys = set(old) | set(new)
    tot = 0.0
    for k in keys:
        a, b = _bag(old.get(k, {})), _bag(new.get(k, {}))
        tot += 0.5 * sum(abs(a.get(u, 0.0) - b.get(u, 0.0)) for u in set(a) | set(b))
    return tot / max(len(keys), 1)


def logsumexp(xs: List[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Elem:
    __slots__ = ("span", "split", "token", "left", "right", "ctx", "cnt", "leaf", "cleaf", "weight",
                 "height", "gain", "par", "sib", "memo", "const", "used", "rid", "pas", "nbt", "win", "placed")

    def __init__(self, span, split=None, token=None, left=None, right=None):
        self.span, self.split, self.token, self.left, self.right = span, split, token, left, right
        self.ctx = self.cnt = None
        self.leaf = self.cleaf = None
        self.weight = 0.0
        self.height = 0 if token is not None else 1 + max(left.height, right.height)
        self.gain = 0.0
        self.par = self.sib = None          # context concept ids of parent / sibling in the analysis
        self.memo = {}                      # concept ids found by sorting (valid while unstored)
        self.const = 0.0                    # its posterior of being a constituent (store "ccm")
        self.used = 0.0                     # 1 if it is in the analysis memory chose for its experience
        self.rid = None                     # its index in memory (stable; statistics refer to it)
        self.pas = []                       # the passive counts it contributed: (tree, attr, value, w)
        self.nbt = None                     # the words just outside it (neighbour_types), for refresh
        self.win = None                     # its raw surface window (window_names), for refresh
        self.placed = None                  # descriptions it was last placed or verified by (verify-drift)


class Trellis3:
    def __init__(self, depth: int = 3, alpha: float = 1.0, seed: int = 0, window: int = 2,
                 store: str = "posterior", tau: float = 1.0, chunk_context: bool = True,
                 donate: bool = True, refresh_budget: Optional[int] = 3000, min_weight: float = 1e-3,
                 bags: bool = True, children: Optional[bool] = None, spine: Optional[bool] = None,
                 neighbour_concepts: bool = False, donate_ctx: Optional[bool] = None,
                 donate_cnt: Optional[bool] = None, factor: str = "gain", support: float = 3.0,
                 naming: str = "depth", heights: bool = True, donate_own: bool = False,
                 refresh: str = "refit", readout: str = "mbr", cc_naming: Optional[str] = None,
                 word_cc: bool = False, boundary: str = "marker", type_names: bool = False, neighbour_types: bool = False,
                 type_top: int = 40, type_k: int = 8, type_key: str = "tokens",
                 word_content: Optional[str] = None, word_cnt: bool = False, restructure: str = "always", drift_tau: float = 0.25, word_name_bits: Optional[float] = None, tree: str = "python", tree_alpha: float = 1e-4, tree_fast: bool = True, band_bits: Sequence[float] = (1, 2, 4, 6, 8, 12), relational: Optional[str] = None,
                 rel_alpha: float = 0.5, rel_rounds: int = 2, nb_bits: Optional[float] = None,
                 nb_rare: Optional[int] = None, window_names: Optional[str] = None,
                 name_level: Optional[str] = None, name_mix_k: Optional[float] = None,
                 part_names: Optional[str] = None, verify_weight: str = "one",
                 ref_names: bool = False, repair_reached: int = 0):
        self.depth, self.alpha, self.window, self.tau = depth, alpha, window, tau
        self.store, self.chunk_context, self.donate = store, chunk_context, donate
        self.refresh_budget, self.min_weight = refresh_budget, min_weight
        # bags: each facet that names a concept (a part, a child, the parent, the
        # sibling, the own content) is ONE attribute whose value is the bag of its
        # concept ids at depths 1..D, so every facet counts once in category
        # utility however many levels it names (else one attribute per depth).
        self.bags = bags
        # chunk context in two facets: the children's context concepts (fixed by
        # the chunk itself) and the spine, parent and sibling (from the analysis)
        self.children = chunk_context if children is None else children
        self.spine = chunk_context if spine is None else spine
        # the window at the level of categories: the context concepts of the
        # tokens just left and right of the span (known while parsing)
        self.neighbour_concepts = neighbour_concepts
        # donation's two directions: the element's own content concept into its
        # context; the parts' content concepts into the content
        self.donate_ctx = donate if donate_ctx is None else donate_ctx
        self.donate_cnt = donate if donate_cnt is None else donate_cnt
        self._word_ids: Dict[Tuple[tuple, int], List[str]] = {}
        # a chunk's factor in inside-outside: "gain" (against the roots) or
        # "code": log2 of the chunk's probability under memory, max over its
        # sorting path of share(C) P(x|C), in both hierarchies. Every full binary
        # analysis has n - 1 chunks, so a tree's weight is then the joint
        # probability of its chunks' descriptions, with no per-chunk baseline.
        self.factor = factor
        # "ccm" (Klein & Manning's constituent-context contrast, over Cobweb's
        # concepts): log2 [(constituent share of C) / (non-constituent share of
        # C)] at the deepest concept on the description's path holding at least
        # ``support`` elements, in both hierarchies; words count as constituents
        self.support = support
        # how a facet names a concept: "depth" (its concepts at depths 1..D) or
        # "path" (every concept on its path below the root, the own leaf left
        # out): shared ancestry at every granularity, no fixed depth; or "leaf":
        # the D concepts just above its own leaf (the substitutable granularity)
        self.naming = naming
        # the donated own-content facet in the context may be named differently
        # from the parts (it keeps fine detail; parts take one clean name)
        self.cc_naming = cc_naming or naming
        # a word's own content is the word itself: with word_cc its identity is
        # donated into its context description as the cc facet, as a chunk's
        # content concept is (otherwise a word's cc is blank)
        self.word_cc = word_cc
        # with word_cnt a word part's content facet (LC/RC) is the word itself, so
        # its identity lives on the content side (what a chunk is made of) and the
        # context tree can describe a word only by where it stands (word_cc off)
        self.word_cnt = word_cnt
        # naming="bands": a path is named by the first concept reaching each of these
        # information levels (-log2 of its share of the tree), so a name holds a
        # few concepts at matched resolution, not every concept on the path
        self.band_bits = tuple(band_bits)
        # the sentence boundary in a window: "marker" (<s> / </s>, a value many
        # instances share) or "missing" (the slot is simply not observed)
        self.boundary = boundary
        # type-level names (distributional, as in word2vec): a part is named by
        # the aggregate of the context paths of every stored instance of its
        # type (its token sequence; a word is the one-token case), not by the
        # one context it occurred in. Substitutable types then share names
        # wherever they occurred. The aggregate keeps its ``type_top`` heaviest
        # concepts.
        self.type_names, self.type_top = type_names, type_top
        # type_names="sample": a type keeps a weighted reservoir of ``type_k`` of
        # the context leaves its instances stood at; its name is the bag of the
        # concepts on those leaves' current paths (a sampled bag of concepts,
        # never stale). type_key="concept": a composite's type is its content
        # concept (chunk type) and not its exact token string.
        self.type_k, self.type_key = type_k, type_key
        self.type_res: Dict[tuple, List[Tuple[float, CobwebNode]]] = {}
        self._type_rng = random.Random(seed + 7)
        self.neighbour_types = neighbour_types
        # a neighbour's learned name keeps only the bands up to nb_bits (its
        # category; None: every band, which nearly identifies the word)
        self.nb_bits = nb_bits
        # nb_rare=n (brief T6): the neighbour facet only for elements whose own type has
        # been seen fewer than n times; a frequent type is described by its own window
        # (its evidence), a rare one borrows its neighbours' learned names
        self.nb_rare = nb_rare
        # window_names: the surface window at the level of learned categories. Each
        # neighbouring word is written as its learned name (bands up to nb_bits)
        # instead of its token; "near": the adjacent slots only, "all": every slot,
        # "far": every slot but the adjacent ones (those stay tokens).
        # A word with no name yet stays its token; the sentence markers stay markers.
        self.window_names = window_names
        # name_level="parent": a type's sampled name is the distribution of the parent
        # concepts of its sampled context leaves (the finest category the tree forms
        # around each instance, read from the current tree) instead of their band names
        self.name_level = name_level
        # name_mix_k=k (interdependence at readout): a word's name is its own sampled
        # name with weight n / (n + k) (n: how often it has been seen) plus, with the
        # rest, the names of the words that stood next to it, tagged by side. What is
        # stored in the trees is unchanged; rare words lean on their neighbours.
        self.name_mix_k = name_mix_k
        # part_names="bands": a chunk names its parts by their band names even when the
        # word's own representation (name_level / name_mix_k) is finer, so the content
        # tree generalises over parts while words are identified by the finer name
        self.part_names = part_names
        self.type_seen: Dict[tuple, float] = {}
        self.type_bag: Dict[tuple, Dict[str, float]] = {}
        # a word's content name: what it builds. Each word keeps a weighted
        # reservoir of ``type_k`` content leaves of the composites it is a direct
        # part of, tagged by side; its name is the bag of side-tagged concepts on
        # those leaves' current paths. "parts": a word part's LC/RC carry it;
        # "cc": the word's own content in its context is it; "cc+token": half
        # the token, half the bag.
        self.word_content = word_content
        self.word_res: Dict[str, List[Tuple[float, str, CobwebNode]]] = {}
        # relational names (docs/RELATIONAL.md): a word's anchor v (where it
        # stands) and context role c (how it works as a neighbour) at the fixed
        # point v = a P + (1-a) A c, c = a P + (1-a) A^T v, read from the trees:
        # P from the word's sampled context leaves, A from those leaves' window
        # counts, A^T from a reservoir of the words that stood next to it.
        # "names": a word part's L/R is v; "window": a context also gets lc/rc,
        # the context roles of its outside neighbours; "both".
        self.relational, self.rel_alpha, self.rel_rounds = relational, rel_alpha, rel_rounds
        self.nb_res: Dict[str, List[Tuple[float, str, str]]] = {}
        # the parts' heights in the content (v1's complexity tag); without them a
        # part that recurses (an N' where an N stands) is named only by its concepts
        self.heights = heights
        # donation of the element's OWN context concept into its content (the
        # mirror of ``cc``): the context concept of its surface window, so a
        # content concept groups chunks by makeup and by where they occur
        self.donate_own = donate_own
        # how a stored instance whose named concepts moved is re-described:
        # "refit" (taken out and re-inserted) or "relabel" (its counts updated
        # in place on its own path; it stays where it was stored)
        self.refresh = refresh
        # refresh="drift" (lazy repair): a stored record whose description drifted by
        # more than drift_tau (mean total variation over attributes) is reassigned
        # (moved to where it now sorts); smaller drift is relabelled in place
        self.drift_tau = drift_tau
        # verify_weight="own": a reassignment check sorts the record at its own
        # (posterior) weight, the weight ifit placed it with; "one": at weight 1
        self.verify_weight = verify_weight
        # ref_names (update on read, docs/UPDATE_ON_READ.md): a stored record's
        # name-copying fields (parts' names L/R, parts' content LC/RC, a chunk's own
        # content cc) hold references into a shared name table that TRELLIS updates
        # after each experience; the trees apply name changes to a concept's counts
        # when a sort reaches it, so stored descriptions never go stale and no
        # refresh re-describes them (refresh_budget then only checks placement)
        self.ref_names = ref_names
        # repair_reached=k (repair where there is activity): after an experience,
        # the records stored at the leaves beside where its records were sorted (the
        # concepts its sorts just changed) are checked and moved if misplaced, at
        # most k per hierarchy
        self.repair_reached = repair_reached
        self._at: Dict[str, Dict[int, List[Elem]]] = {"ctx": {}, "cnt": {}}
        # word_cc="name": a word's own-content facet keeps only the bands up to
        # word_name_bits (its category-level representation; None: all bands)
        self.word_name_bits = word_name_bits
        # the analysis read out of an experience: "mbr" (most expected correct
        # composites) or "regen" (the analysis whose chunks memory regenerates
        # most probably: the sum over its composites of log2 P(parts | content
        # concept), so unpacking the chunks gives back the experience)
        self.readout = readout
        d = [""] if bags else [str(k) for k in range(1, depth + 1)]
        surface = self._surface_names(window)
        chunk = (tuple(f"{p}{k}" for p in ("cl", "cr") for k in d) if self.children else ()) + \
            (tuple(f"{p}{k}" for p in ("par", "sib") for k in d) if self.spine else ())
        own = tuple(f"cc{k}" for k in d) if self.donate_ctx else ()
        neigh = self._neigh_names(d) \
            if (neighbour_concepts or neighbour_types or relational in ("window", "both")) else ()
        self.ctx_attrs = surface + neigh + chunk + own + (("le", "re") if boundary == "edge" else ())
        parts = tuple(f"{p}{k}" for p in ("L", "R") for k in d)
        given = tuple(f"{p}{k}" for p in ("LC", "RC") for k in d) if self.donate_cnt else ()
        self.cnt_attrs = parts + given + (("LH", "RH") if heights else ()) + \
            (tuple(f"oc{k}" for k in d) if donate_own else ()) + self._extra_cnt_names()
        if tree == "discrete":
            # cobweb-private's CobwebDiscreteTree (entropy category utility, alpha
            # pseudo-counts per value) in place of the pure-Python Fisher Cobweb
            from .discrete_cobweb import DiscreteCobwebTree
            self.ctx = DiscreteCobwebTree(self.ctx_attrs, seed=seed, alpha=tree_alpha)
            self.cnt = DiscreteCobwebTree(self.cnt_attrs, seed=seed + 1, alpha=tree_alpha)
        elif tree == "cu":
            # cobweb_cu (compiled Fisher Cobweb): the same trees as the Python one;
            # tree_fast=False gives its reference arithmetic (bit-identical to the
            # Python tree with tree_fast=False), True its fast path (same scores
            # up to rounding)
            from .cu_cobweb import CUCobwebTree
            self.ctx = CUCobwebTree(self.ctx_attrs, seed=seed, fast=tree_fast)
            self.cnt = CUCobwebTree(self.cnt_attrs, seed=seed + 1, fast=tree_fast)
        elif ref_names:
            from .ref_cobweb import NameTable, RefCobwebTree
            self.names = NameTable()
            self.ref_keys: set = set()
            self.ctx = RefCobwebTree(self.ctx_attrs, seed=seed, table=self.names)
            self.cnt = RefCobwebTree(self.cnt_attrs, seed=seed + 1, table=self.names)
            if not tree_fast:
                self.ctx.fast_ec = self.cnt.fast_ec = False
        else:
            self.ctx = IncrementalCobwebTree(self.ctx_attrs, seed=seed)
            self.cnt = IncrementalCobwebTree(self.cnt_attrs, seed=seed + 1)
            if not tree_fast:
                self.ctx.fast_ec = self.cnt.fast_ec = False
        self.ctx.partial = self.cnt.partial = (boundary in ("missing", "edge"))
        # restructure="sample": a span stored with posterior weight w merges or
        # splits concepts with probability w (see IncrementalCobwebTree)
        self.ctx.sample_restructure = self.cnt.sample_restructure = (restructure == "sample")
        self.rng = random.Random(seed)
        self.records: List[Elem] = []
        self._rr = 0
        self.stats = {"experiences": 0, "words stored": 0, "composites stored": 0, "refreshed": 0}

    # ------------------------------------------------------------------ #
    # Descriptions
    # ------------------------------------------------------------------ #
    # attribute names (a relational domain names its own; trellis_graph.py)
    @staticmethod
    def _surface_names(window: int) -> tuple:
        return tuple(f"l{k}" for k in range(window, 0, -1)) + tuple(f"r{k}" for k in range(1, window + 1))

    @staticmethod
    def _neigh_names(d) -> tuple:
        return tuple(f"{p}{k}" for p in ("lc", "rc") for k in d)

    @staticmethod
    def _extra_cnt_names() -> tuple:
        return ()

    def _surface(self, tokens, i, j) -> dict:
        x = {}
        for k in range(1, self.window + 1):
            if i - k >= 0:
                x[f"l{k}"] = tokens[i - k]
            elif self.boundary == "marker":
                x[f"l{k}"] = BOS
            if j + k - 1 < len(tokens):
                x[f"r{k}"] = tokens[j + k - 1]
            elif self.boundary == "marker":
                x[f"r{k}"] = EOS
        if self.boundary == "edge":
            # whether a neighbour exists on each side, once per side; the
            # neighbour slots past an edge stay unobserved
            x["le"] = "edge" if i == 0 else "in"
            x["re"] = "edge" if j == len(tokens) else "in"
        return x

    def is_sentence(self, e: Elem) -> bool:
        """A sentence-level element: nothing on either side."""
        if self.boundary == "edge":
            return e.ctx.get("le") == "edge" and e.ctx.get("re") == "edge"
        if self.boundary == "missing":
            return "l1" not in e.ctx and "r1" not in e.ctx
        return e.ctx.get("l1") == BOS and e.ctx.get("r1") == EOS

    def _ids(self, path: List[CobwebNode]) -> List[str]:
        ids = [f"c{n.id}" for n in path[1:self.depth + 1]]
        return ids + [BLANK] * (self.depth - len(ids))

    def _ctx_ids(self, e: Elem) -> List[str]:
        """An element's context concept, now: from its stored leaf, or by sorting."""
        if e.leaf is not None:
            return self._ids(self.ctx.ancestors(e.leaf)[::-1])
        if "ctx" not in e.memo:
            e.memo["ctx"] = self._ids(self.ctx.categorize(e.ctx))
        return e.memo["ctx"]

    def _cnt_ids(self, e: Elem) -> List[str]:
        if e.token is not None:
            return [BLANK] * self.depth
        if e.cleaf is not None:
            return self._ids(self.cnt.ancestors(e.cleaf)[::-1])
        if "cnt" not in e.memo:
            e.memo["cnt"] = self._ids(self.cnt.categorize(e.cnt))
        return e.memo["cnt"]

    def _ctx_name(self, e: Elem, part: bool = False) -> List[str]:
        """The ids a facet names an element's context concept by (``part``: as a
        chunk's part, which ``part_names`` may name more coarsely)."""
        if part and self.part_names == "bands" and (self.name_level or self.name_mix_k):
            keep = (self.name_level, self.name_mix_k)
            self.name_level = self.name_mix_k = None
            try:
                return self._ctx_name(e)
            finally:
                self.name_level, self.name_mix_k = keep
        if self.naming == "depth":
            return self._ctx_ids(e)
        if e.token is not None and self.relational in ("names", "both"):
            bag = self._rel_v(e.token, self.rel_rounds)
            if bag:
                return bag
        if self.type_names == "sample":
            bag = self._mixed_name(e.token) if (e.token is not None and self.name_mix_k) else self._res_bag(self._sample_key(e))
            if bag:
                return bag
        elif self.type_names:
            bag = self.type_bag.get(self._type_key(e))
            if bag:
                return dict(sorted(bag.items(), key=lambda kv: -kv[1])[:self.type_top])
        if e.leaf is not None:
            path = self.ctx.ancestors(e.leaf)[::-1]
        else:
            if "ctxp" not in e.memo:
                e.memo["ctxp"] = self.ctx.categorize(e.ctx)
            path = e.memo["ctxp"]
        return self._path_ids(path)

    def _res_bag(self, key, max_bits: Optional[float] = None) -> Optional[Dict[str, float]]:
        """P: the bag of concepts on a type's sampled context leaves' current paths
        (with ``max_bits``, under naming="bands", only the bands up to that level)."""
        res = self.type_res.get(key)
        if not res:
            return None
        bag: Dict[str, float] = {}
        keep = self.band_bits
        if max_bits is not None:
            self.band_bits = tuple(b for b in keep if b <= max_bits)
        for _, leaf in res:
            if self.name_level == "parent":
                par = leaf.parent if not leaf.children else leaf
                ids = [f"c{par.id}"] if par is not None else [BLANK]
            else:
                ids = self._path_ids(self.ctx.ancestors(leaf)[::-1])
            for v in ids:
                bag[v] = bag.get(v, 0.0) + 1.0 / (len(ids) * len(res))
        self.band_bits = keep
        return dict(sorted(bag.items(), key=lambda kv: -kv[1])[:self.type_top])

    def _rel_mix(self, own: Optional[Dict[str, float]], nbrs: Dict[Tuple[str, str], float], role, k: int) -> Dict[str, float]:
        """a * own + (1 - a) * sum of the neighbours' bags (side-tagged, last hop only)."""
        a = self.rel_alpha if own else 0.0
        out: Dict[str, float] = {}
        for v, w in (own or {}).items():
            out[v] = out.get(v, 0.0) + a * w
        tot = sum(nbrs.values())
        for (side, u), w in nbrs.items():
            for v, x in role(u, k).items():
                key = side + ">" + v.split(">")[-1]
                out[key] = out.get(key, 0.0) + (1 - a) * x * w / tot
        s = sum(out.values())
        top = sorted(out.items(), key=lambda kv: -kv[1])[:self.type_top]
        return {v: w / s for v, w in top} if s > 0 else {}

    def _rel_v(self, w: str, k: int) -> Dict[str, float]:
        """Anchor name of word w after k rounds: where it stands, plus its neighbours' context roles."""
        cache = self.__dict__.setdefault("_rel_cache", {})
        if ("v", w, k) in cache:
            return cache[("v", w, k)]
        own = self._res_bag((w,)) or {}
        nbrs: Dict[Tuple[str, str], float] = {}
        if k > 0:
            res = self.type_res.get((w,), [])
            for _, leaf in res:                            # A: w's neighbours, from its leaves' window counts
                for a, side in (("l1", "L"), ("r1", "R")):
                    d = leaf.av.get(a)
                    if not isinstance(d, dict):
                        continue
                    for u, c in d.items():
                        if u not in (BOS, EOS) and isinstance(u, str):
                            nbrs[(side, u)] = nbrs.get((side, u), 0.0) + c / max(leaf.count, 1e-9) / len(res)
        out = self._rel_mix(own, nbrs, self._rel_c, k - 1) if nbrs else own
        cache[("v", w, k)] = out
        return out

    def _rel_c(self, u: str, k: int) -> Dict[str, float]:
        """Context role of word u after k rounds: its own placement plus the anchors of words next to it."""
        cache = self.__dict__.setdefault("_rel_cache", {})
        if ("c", u, k) in cache:
            return cache[("c", u, k)]
        own = self._res_bag((u,)) or {}
        nbrs: Dict[Tuple[str, str], float] = {}
        if k > 0:
            for _, side, w in self.nb_res.get(u, []):      # A^T: the words that stood next to u
                nbrs[(side, w)] = nbrs.get((side, w), 0.0) + 1.0
        out = self._rel_mix(own, nbrs, self._rel_v, k - 1) if nbrs else own
        cache[("c", u, k)] = out
        return out

    def _add_neighbour(self, u: str, side: str, w: str) -> None:
        """Word w stood on ``side`` of word u: a reservoir of type_k (side, w) per u."""
        res = self.nb_res.setdefault(u, [])
        key = self._type_rng.random()
        if len(res) < self.type_k:
            res.append((key, side, w))
        else:
            i = min(range(len(res)), key=lambda t: res[t][0])
            if key > res[i][0]:
                res[i] = (key, side, w)

    @staticmethod
    def _type_key(e: Elem) -> tuple:
        out, stack = [], [e]
        while stack:
            x = stack.pop()
            if x.token is not None:
                out.append(x.token)
            else:
                stack += [x.right, x.left]
        return tuple(out)

    def _sample_key(self, e: Elem) -> tuple:
        """A primitive's type is its token; a composite's is its token string, or
        with type_key="concept" its content concept (where it is or would be stored)."""
        if e.token is not None or self.type_key != "concept":
            return self._type_key(e)
        if e.cleaf is not None:
            return ("c", e.cleaf.id)
        if "cntp" not in e.memo:
            e.memo["cntp"] = self.cnt.categorize(e.cnt)
        return ("c", e.memo["cntp"][-1].id)

    def _add_type(self, e: Elem) -> None:
        """Add a stored instance's context to its type: with type_names="sample"
        into a weighted reservoir of ``type_k`` context leaves (A-Res: key
        u ** (1 / weight), keep the largest), else into the full aggregate."""
        if self.type_names == "sample":
            if e.weight <= 0:
                return
            k = self._sample_key(e)
            self.type_seen[k] = self.type_seen.get(k, 0.0) + e.weight
            res = self.type_res.setdefault(self._sample_key(e), [])
            key = self._type_rng.random() ** (1.0 / e.weight)
            if len(res) < self.type_k:
                res.append((key, e.leaf))
            else:
                i = min(range(len(res)), key=lambda t: res[t][0])
                if key > res[i][0]:
                    res[i] = (key, e.leaf)
            return
        bag = self.type_bag.setdefault(self._type_key(e), {})
        ids = self._path_ids(self.ctx.ancestors(e.leaf)[::-1])
        for v in ids:
            bag[v] = bag.get(v, 0.0) + e.weight / max(len(ids), 1)

    def _add_word_content(self, e: Elem) -> None:
        """A stored composite adds its content leaf to the reservoirs of its word parts."""
        if e.weight <= 0 or e.cleaf is None:
            return
        for side, part in (("L", e.left), ("R", e.right)):
            if part.token is None:
                continue
            res = self.word_res.setdefault(part.token, [])
            key = self._type_rng.random() ** (1.0 / e.weight)
            if len(res) < self.type_k:
                res.append((key, side, e.cleaf))
            else:
                i = min(range(len(res)), key=lambda t: res[t][0])
                if key > res[i][0]:
                    res[i] = (key, side, e.cleaf)

    def _word_content_bag(self, token: str) -> Optional[Dict[str, float]]:
        """The bag of side-tagged content concepts a word's sampled composites sit under."""
        cache = getattr(self, "_wc_cache", None)
        if cache is not None and token in cache:
            return cache[token]
        res = self.word_res.get(token)
        if not res:
            return None
        bag: Dict[str, float] = {}
        for _, side, leaf in res:
            ids = self._path_ids(self.cnt.ancestors(leaf)[::-1])
            for v in ids:
                bag[side + v] = bag.get(side + v, 0.0) + 1.0 / (len(ids) * len(res))
        out = dict(sorted(bag.items(), key=lambda kv: -kv[1])[:self.type_top])
        if cache is not None:
            cache[token] = out
        return out

    def _cnt_name(self, e: Elem, naming: Optional[str] = None) -> List[str]:
        naming = naming or self.naming
        if naming == "depth":
            return self._cnt_ids(e)
        if e.token is not None:
            return [f"w:{e.token}"] if self.word_cnt else [BLANK]
        if e.cleaf is not None:
            path = self.cnt.ancestors(e.cleaf)[::-1]
        else:
            if "cntp" not in e.memo:
                e.memo["cntp"] = self.cnt.categorize(e.cnt)
            path = e.memo["cntp"]
        return self._path_ids(path, naming)

    def _path_ids(self, path, naming: Optional[str] = None) -> List[str]:
        naming = naming or self.naming
        inner = path[1:-1] if len(path) > 1 and not path[-1].children else path[1:]
        if naming == "leaf":
            # the D concepts nearest the element's own leaf (finest first)
            inner = inner[::-1][:self.depth]
        if naming == "basic" and inner:
            # the basic level (Corter & Gluck 1992): the concept on the path
            # maximising P(c) [sum P(v|c)^2 - sum P(v|root)^2], one categorical name
            root = path[0]
            base = root.sstot / max(root.count, 1e-9) ** 2

            def cu(n):
                return (n.count / max(root.count, 1e-9)) * (n.sstot / max(n.count, 1e-9) ** 2 - base)
            return [f"c{max(inner, key=cu).id}"]
        if naming == "bands" and inner:
            total, out, k = max(path[0].count, 1e-9), [], 0
            for n in inner:
                b = -math.log2(max(n.count, 1e-9) / total)
                while k < len(self.band_bits) and b >= self.band_bits[k]:
                    out.append(f"c{n.id}")
                    k += 1
            return out or [BLANK]
        if naming == "ipath" and inner:
            # each concept weighted by its information, -log2 of its share of
            # the tree in whole bits: shared coarse ancestry counts little
            total = max(path[0].count, 1e-9)
            return [f"c{n.id}" for n in inner
                    for _ in range(max(1, round(-math.log2(max(n.count, 1e-9) / total))))]
        return [f"c{n.id}" for n in inner] or [BLANK]

    def _cc_ids(self, e: Elem):
        if e.token is not None and self.word_content in ("cc", "cc+token"):
            bag = self._word_content_bag(e.token)
            if bag:
                if self.word_content == "cc":
                    return bag
                tot = sum(bag.values())
                out = {v: 0.5 * w / tot for v, w in bag.items()}
                out[f"w:{e.token}"] = 0.5
                return out
        if e.token is not None and self.word_cc == "cat":
            # a word identified by its learned category: one stable value, its most
            # common coarse concept (bands up to nb_bits), never its token
            bag = self._window_name(e.token)
            return [max(bag.items(), key=lambda kv: kv[1])[0]] if bag else [BLANK]
        if e.token is not None and self.word_cc == "name":
            # a word identified by its learned representation: its type name (where
            # its type has stood, read through the current context tree), not its token
            bag = self._res_bag(self._sample_key(e), self.word_name_bits)
            return bag if bag else [BLANK]
        if e.token is not None and self.word_cc:
            return [f"w:{e.token}"]
        if self.cc_naming == "tokens":
            # what a chunk is made of, in the same vocabulary as a word's own
            # content: the bag of its tokens (a word is the one-token case)
            out, stack = [], [e]
            while stack:
                x = stack.pop()
                if x.token is not None:
                    out.append(f"w:{x.token}")
                else:
                    stack += [x.right, x.left]
            return out
        return self._cnt_name(e, self.cc_naming)

    def _facet(self, x: dict, name: str, ids: List[str]) -> None:
        """Write a facet's concept ids: one bag attribute, or one attribute per depth."""
        if self.bags and isinstance(ids, dict):            # a weighted aggregate (type names)
            tot = sum(ids.values())
            x[name] = {v: w / tot for v, w in ids.items()} if tot > 0 else BLANK
        elif self.bags:
            real = [v for v in ids if v != BLANK]
            bag: Dict[str, float] = {}
            for v in real:
                bag[v] = bag.get(v, 0.0) + 1.0 / len(real)
            x[name] = bag if real else BLANK
        else:
            ids = (list(ids) + [BLANK] * self.depth)[:self.depth]
            for k, v in enumerate(ids, 1):
                x[f"{name}{k}"] = v

    def _own_ctx(self, e: Elem, tokens=None) -> List[str]:
        """The context concept of the element's surface window (no donated facet)."""
        if "own" not in e.memo:
            x = {a: v for a, v in e.ctx.items() if a[0] in "lr" and a[1:].isdigit()}
            path = self.ctx.categorize(x)
            e.memo["own"] = self._path_ids(path) if self.naming in ("path", "ipath", "bands", "leaf", "basic") else self._ids(path)
        return e.memo["own"]

    def _content(self, e: Elem) -> dict:
        x = {}
        if self.donate_own:
            self._facet(x, "oc", self._own_ctx(e))
        for p, part in (("L", e.left), ("R", e.right)):
            self._facet(x, p, self._ctx_name(part, part=True))
            if self.donate_cnt:
                bag = self._word_content_bag(part.token) if (part.token is not None and self.word_content in ("parts", "cc", "cc+token")) else None
                self._facet(x, f"{p}C", bag if bag else self._cnt_name(part))
            if self.heights:
                x[f"{p}H"] = str(min(part.height, 4))
        return x

    def _context(self, tokens, e: Elem, analysed: bool) -> dict:
        """The context description; with ``analysed`` False the parts that need an
        analysis (parent, sibling) are left out (a partial instance)."""
        x = self._surface(tokens, *e.span)
        if self.window_names:
            e.win = {a: v for a, v in x.items() if a[0] in "lr" and a[1:].isdigit()}
            x.update(self._named_window(e.win))
        if self.neighbour_types:
            # the window at the level of types: each neighbouring word named by
            # the aggregate of where its type has occurred
            i, j = e.span
            e.nbt = tuple(tokens[pos] if 0 <= pos < len(tokens) else None for pos in (i - 1, j))
            self._nb_facets(x, e)
        elif self.relational in ("window", "both"):
            # the window at the level of roles: each outside neighbour named by its
            # context role (how it works as a neighbour), a same-level element
            i, j = e.span
            for p, pos in (("lc", i - 1), ("rc", j)):
                bag = self._rel_c(tokens[pos], self.rel_rounds) if 0 <= pos < len(tokens) else None
                self._facet(x, p, bag if bag else [BLANK])
        elif self.neighbour_concepts:
            i, j = e.span
            for p, pos in (("lc", i - 1), ("rc", j)):
                self._facet(x, p, self._word_concept(tokens, pos))
        if self.children:
            for p, part in (("cl", e.left), ("cr", e.right)):
                self._facet(x, p, self._ctx_ids(part) if part is not None else [BLANK] * self.depth)
        if self.spine:
            if analysed:
                for p, ids in (("par", e.par), ("sib", e.sib)):
                    self._facet(x, p, ids or [BLANK] * self.depth)
        if self.donate_ctx:
            self._facet(x, "cc", self._cc_ids(e))
        return x

    def _window_name(self, token: str) -> Optional[Dict[str, float]]:
        """A window slot's learned name: a category, so the band names up to nb_bits
        even when words are represented by finer names (name_level / name_mix_k)."""
        cache = self.__dict__.setdefault("_nb_cache", {})
        if ("win", token) not in cache:
            keep = self.name_level
            self.name_level = None
            try:
                cache[("win", token)] = self._res_bag((token,), self.nb_bits)
            finally:
                self.name_level = keep
        return cache[("win", token)]

    def _named_window(self, win: dict) -> dict:
        out = {}
        for a, v in win.items():
            if v in (BOS, EOS) or (self.window_names == "near" and a[1:] != "1") or \
                    (self.window_names == "far" and a[1:] == "1"):
                out[a] = v
                continue
            bag = self._window_name(v)
            # one categorical value: the neighbour's most common coarse concept (a bag
            # re-sampled at every reading would make every stored context unique)
            out[a] = max(bag.items(), key=lambda kv: kv[1])[0] if bag else v
        return out

    def _mixed_name(self, token: str) -> Optional[Dict[str, float]]:
        cache = self.__dict__.setdefault("_nb_cache", {})
        if ("mix", token) in cache:
            return cache[("mix", token)]
        own = self._res_bag((token,)) or {}
        nbrs = self.nb_res.get(token, [])
        borrowed: Dict[str, float] = {}
        for _, side, w in nbrs:
            for v, x in (self._res_bag((w,)) or {}).items():
                key = side + ">" + v
                borrowed[key] = borrowed.get(key, 0.0) + x / len(nbrs)
        n = self.type_seen.get((token,), 0.0)
        a = n / (n + self.name_mix_k) if own else 0.0
        out: Dict[str, float] = {}
        for v, x in own.items():
            out[v] = out.get(v, 0.0) + a * x / (sum(own.values()) or 1.0)
        tb = sum(borrowed.values())
        for v, x in borrowed.items():
            out[v] = out.get(v, 0.0) + (1 - a) * x / tb
        out = out or None
        cache[("mix", token)] = out
        return out

    def _rare(self, e: Elem) -> bool:
        return self.nb_rare is None or self.type_seen.get(self._sample_key(e), 0.0) < self.nb_rare

    def _nb_facets(self, x: dict, e: Elem) -> None:
        if not self._rare(e):
            for p in ("lc", "rc"):
                self._facet(x, p, [BLANK])
            return
        for p, tok in zip(("lc", "rc"), e.nbt):
            bag = self._nb_name(tok) if tok is not None else None
            self._facet(x, p, bag if bag else [BLANK])

    def _nb_name(self, token: str) -> Optional[Dict[str, float]]:
        """A neighbouring word's learned name: with type_names="sample" the bag of
        concepts on its sampled context leaves' current paths (read once per
        experience), else the aggregate of where its type has occurred."""
        if self.type_names == "sample":
            cache = self.__dict__.setdefault("_nb_cache", {})
            if token not in cache:
                cache[token] = self._res_bag((token,), self.nb_bits)
            return cache[token]
        bag = self.type_bag.get((token,))
        return dict(sorted(bag.items(), key=lambda kv: -kv[1])[:self.type_top]) if bag else None

    def _word_concept(self, tokens, pos: int) -> List[str]:
        """The context concept of the token at ``pos`` (sorting its surface window);
        blank outside the experience."""
        if pos < 0 or pos >= len(tokens):
            return [BLANK] * self.depth
        x = self._surface(tokens, pos, pos + 1)
        return self._ids(self.ctx.categorize(x))

    def _full(self, x: dict, attrs) -> dict:
        """Complete an instance for storage (missing attributes blank; with
        boundary="missing" the surface slots beyond the experience stay out)."""
        if self.boundary in ("missing", "edge"):
            return {a: x.get(a, BLANK) for a in attrs if a in x or not (a[0] in "lr" and a[1:].isdigit())}
        return {a: x.get(a, BLANK) for a in attrs}

    # ------------------------------------------------------------------ #
    # Scoring and inside-outside
    # ------------------------------------------------------------------ #
    def _code(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """max over x's sorting path (root included) of log2 [share(C) P(x|C)]."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        best = -math.inf
        for node in tree.categorize(x):
            if node.count <= 0:
                continue
            best = max(best, (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha)) / LN2)
        return best

    def _contrast(self, tree: IncrementalCobwebTree, x: dict) -> float:
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = root
        for n in tree.categorize(x)[1:]:
            if n.count >= self.support:
                node = n
        c = lambda n: sum(n.pav.get(CONST, {}).values())
        cC, cR = c(node), c(root)
        dC, dR = max(node.count - cC, 0.0), max(root.count - cR, 0.0)
        eps = 0.5
        return math.log2(((cC + eps) / (cR + eps)) / ((dC + eps) / (dR + eps)))

    def _climb(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """v1's recognition at the climbing ancestor: log2 P(x|A) / P(x|root),
        A the deepest concept on x's sorting path holding at least ``support``
        instances (the root when none does: an unrecognised chunk scores 0)."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = root
        for n in tree.categorize(x)[1:]:
            if n.count >= self.support:
                node = n
        if node is root:
            return 0.0
        return (log_prob(node, x, root, self.alpha) - log_prob(root, x, root, self.alpha)) / LN2

    def _gain(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """Best over x's sorting path of log2 [share(C) P(x|C) / P(x|root)]; 0 at the root."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        base = log_prob(root, x, root, self.alpha)
        best = 0.0
        for node in tree.categorize(x)[1:]:
            if node.count <= 0:
                continue
            g = (math.log(node.count / root.count) + log_prob(node, x, root, self.alpha) - base) / LN2
            best = max(best, g)
        return best

    def _chart(self, tokens: List[str]):
        """All elements of an experience, bottom-up: words, then every composite
        (span, split) with its parts the best (Viterbi) sub-analyses; each
        composite's gain from both hierarchies."""
        n = len(tokens)
        words = []
        for i, w in enumerate(tokens):
            e = Elem((i, i + 1), token=w)
            e.ctx = self._context(tokens, e, analysed=False)
            words.append(e)
        best: Dict[Tuple[int, int], Tuple[float, Elem]] = {(i, i + 1): (0.0, words[i]) for i in range(n)}
        comps: Dict[Tuple[int, int, int], Elem] = {}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    a, b = best[(i, k)][1], best[(k, j)][1]
                    e = Elem((i, j), split=k, left=a, right=b)
                    e.ctx = self._surface(tokens, i, j)
                    e.cnt = self._content(e)
                    e.ctx = self._context(tokens, e, analysed=False)
                    if self.factor == "ccm":
                        e.gain = self._contrast(self.cnt, e.cnt) + self._contrast(self.ctx, e.ctx)
                    elif self.factor == "regen":
                        # content scored by how probably memory unpacks it
                        e.gain = self._regen_bits(e) + self._gain(self.ctx, e.ctx)
                    elif self.factor == "deriv":
                        e.gain = self._deriv_bits(e)
                    elif self.factor == "regen2":
                        e.gain = self._regen_bits(e) + self._regen_tree(self.ctx, e.ctx)
                    elif self.factor == "climb":
                        e.gain = self._climb(self.cnt, e.cnt) + self._climb(self.ctx, e.ctx)
                    elif self.factor == "code":
                        e.gain = self._code(self.cnt, e.cnt) + self._code(self.ctx, e.ctx)
                    else:
                        e.gain = self._gain(self.cnt, e.cnt) + self._gain(self.ctx, e.ctx)
                    comps[(i, k, j)] = e
                    v = best[(i, k)][0] + best[(k, j)][0] + e.gain
                    if (i, j) not in best or v > best[(i, j)][0]:
                        best[(i, j)] = (v, e)
        return words, comps

    def _inside_outside(self, n: int, comps):
        """Log inside/outside over binary analyses with factors 2^(gain/tau);
        returns the posterior of every composite and log Z."""
        f = {key: e.gain * LN2 / self.tau for key, e in comps.items()}
        ins = {(i, i + 1): 0.0 for i in range(n)}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                ins[(i, j)] = logsumexp([f[(i, k, j)] + ins[(i, k)] + ins[(k, j)] for k in range(i + 1, j)])
        out = {(0, n): 0.0}
        for length in range(n, 1, -1):
            for i in range(0, n - length + 1):
                j = i + length
                o = out.get((i, j), -math.inf)
                for k in range(i + 1, j):
                    t = o + f[(i, k, j)]
                    for span, other in (((i, k), ins[(k, j)]), ((k, j), ins[(i, k)])):
                        out[span] = logsumexp([out.get(span, -math.inf), t + other])
        logZ = ins[(0, n)]
        post = {key: math.exp(f[key] + ins[(key[0], key[1])] + ins[(key[1], key[2])] + out.get((key[0], key[2]), -math.inf) - logZ)
                for key in comps}
        return post, logZ

    def _regen_bits(self, e: Elem) -> float:
        """log2 P(e's content | its content concept): how probably memory
        unpacks this chunk into these parts (the deepest concept on its sorting
        path holding at least ``support`` instances; the root when none does)."""
        if "regen" not in e.memo:
            root = self.cnt.root
            if root.count <= 0:
                e.memo["regen"] = 0.0
            else:
                node = root
                for c in self.cnt.categorize(e.cnt)[1:]:
                    if c.count >= self.support:
                        node = c
                e.memo["regen"] = log_prob(node, e.cnt, root, self.alpha) / LN2
        return e.memo["regen"]

    def _deriv_bits(self, e: Elem) -> float:
        """One node of a derivation through both trees: log2 P(parts | content
        concept) plus log2 P(own content concept | context concept), the second
        read through the donated cc facet (the context concept sorted from the
        rest of the context instance, at the deepest supported level)."""
        if "deriv" not in e.memo:
            bits = self._regen_bits(e)
            cc = {a: v for a, v in e.ctx.items() if a.startswith("cc")}
            root = self.ctx.root
            if cc and root.count > 0:
                rest = {a: v for a, v in e.ctx.items() if not a.startswith("cc")}
                node = root
                for c in self.ctx.categorize(rest)[1:]:
                    if c.count >= self.support:
                        node = c
                bits += log_prob(node, cc, root, self.alpha) / LN2
            e.memo["deriv"] = bits
        return e.memo["deriv"]

    def _regen_tree(self, tree: IncrementalCobwebTree, x: dict) -> float:
        """log2 P(x | the deepest concept on x's path with at least ``support`` instances)."""
        root = tree.root
        if root.count <= 0:
            return 0.0
        node = root
        for c in tree.categorize(x)[1:]:
            if c.count >= self.support:
                node = c
        return log_prob(node, x, root, self.alpha) / LN2

    def _regen(self, n: int, comps, score=None) -> Tuple[Tree, List[Tuple[int, int, int]]]:
        """The analysis maximising the summed regeneration log probability of its chunks."""
        score = score or self._regen_bits
        best = {(i, i + 1): (0.0, None) for i in range(n)}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                k = max(range(i + 1, j), key=lambda k: best[(i, k)][0] + best[(k, j)][0] + score(comps[(i, k, j)]))
                best[(i, j)] = (best[(i, k)][0] + best[(k, j)][0] + score(comps[(i, k, j)]), k)
        split, chosen = {}, []

        def walk(i, j):
            if j - i < 2:
                return
            k = best[(i, j)][1]
            split[(i, j)] = k
            chosen.append((i, k, j))
            walk(i, k)
            walk(k, j)
        walk(0, n)
        return Tree(n, split, {}, [(0, n)]), chosen

    def _mbr(self, n: int, post) -> Tuple[Tree, List[Tuple[int, int, int]]]:
        """The analysis maximising the expected number of correct composites."""
        span_post: Dict[Tuple[int, int], float] = {}
        for (i, k, j), p in post.items():
            span_post[(i, j)] = span_post.get((i, j), 0.0) + p
        best = {(i, i + 1): (0.0, None) for i in range(n)}
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                k = max(range(i + 1, j), key=lambda k: best[(i, k)][0] + best[(k, j)][0] + post[(i, k, j)])
                best[(i, j)] = (best[(i, k)][0] + best[(k, j)][0] + span_post[(i, j)], k)
        split, chosen = {}, []

        def walk(i, j):
            if j - i < 2:
                return
            k = best[(i, j)][1]
            split[(i, j)] = k
            chosen.append((i, k, j))
            walk(i, k)
            walk(k, j)
        walk(0, n)
        return Tree(n, split, {}, [(0, n)]), chosen

    def parse(self, tokens: Sequence[str]) -> dict:
        tokens = list(tokens)
        n = len(tokens)
        self._wc_cache = {}                 # word content names, read once per experience
        self._rel_cache = {}                # relational names likewise
        self._nb_cache = {}                 # neighbours' learned names likewise
        words, comps = self._chart(tokens)
        if n < 2:
            return {"tree": Tree(n, {}, {}, [(0, n)] if n else []), "words": words, "comps": comps,
                    "post": {}, "chosen": []}
        post, logZ = self._inside_outside(n, comps)
        if self.readout in ("regen", "deriv"):
            tree, chosen = self._regen(n, comps, self._deriv_bits if self.readout == "deriv" else self._regen_bits)
        else:
            tree, chosen = self._mbr(n, post)
        return {"tree": tree, "words": words, "comps": comps, "post": post, "chosen": chosen, "logZ": logZ}

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def learn(self, tokens: Sequence[str]) -> Tree:
        tokens = list(tokens)
        r = self.parse(tokens)
        words, comps, post, chosen = r["words"], r["comps"], r["post"], set(r["chosen"])
        # chunk context from the analysis: parent and sibling of its elements
        node_of = {(e.span[0], e.span[0] + 1): e for e in words}
        for key in chosen:
            node_of[(key[0], key[2])] = comps[key]
            comps[key].used = 1.0
        for w in words:
            w.used = 1.0
        for (i, k, j) in chosen:
            parent = comps[(i, k, j)]
            for child, sib in ((node_of[(i, k)], node_of[(k, j)]), (node_of[(k, j)], node_of[(i, k)])):
                child.par, child.sib = self._ctx_ids(parent), self._ctx_ids(sib)
        if self.store == "ccm":
            # each span is one observation (as in the constituent-context model):
            # weight 1 shared among its splits by P(split | span), of which the
            # constituent share is the span's posterior
            span_post: Dict[Tuple[int, int], float] = {}
            for (i, k, j), p in post.items():
                span_post[(i, j)] = span_post.get((i, j), 0.0) + p
            weights = {}
            for (i, k, j), e in comps.items():
                sp = span_post.get((i, j), 0.0)
                weights[(i, k, j)] = post[(i, k, j)] / sp if sp > 1e-12 else 1.0 / (j - i - 1)
                e.const = post.get((i, k, j), 0.0)
        elif self.store == "posterior":
            weights = {key: p for key, p in post.items()}
        elif self.store == "all":
            weights = {key: 1.0 for key in comps}
        else:
            weights = {key: 1.0 for key in chosen}
        # words first, then composites bottom-up: content (so a context can name it), then context
        for e in words:
            e.weight = 1.0
            e.ctx = self._full(self._context(tokens, e, analysed=True), self.ctx_attrs)
            e.leaf = self.ctx.ifit(e.ctx, 1.0)
            e.rid = len(self.records)
            self._at_add(e, "ctx")
            self._pas(e, "ctx", WORD, e.token, 1.0)
            if self.store == "ccm":
                e.const = 1.0
                self._pas(e, "ctx", CONST, "c", 1.0)
            if self.type_names or self.neighbour_types:
                self._add_type(e)
            if self.relational or self.name_mix_k:
                i = e.span[0]
                if i > 0:
                    self._add_neighbour(tokens[i - 1], "R", e.token)
                if i + 1 < len(tokens):
                    self._add_neighbour(tokens[i + 1], "L", e.token)
            self.records.append(e)
            self.stats["words stored"] += 1
        first = len(self.records)
        for key in sorted(comps, key=lambda t: (t[2] - t[0], t)):
            w = weights.get(key, 0.0)
            if w < self.min_weight:
                continue
            e = comps[key]
            e.weight = w
            e.rid = len(self.records)
            e.cnt = self._content(e)
            if self.ref_names:
                e.cnt = self._ref_content(e, e.cnt)
            e.cleaf = self.cnt.ifit(e.cnt, w)
            e.ctx = self._full(self._context(tokens, e, analysed=key in chosen), self.ctx_attrs)
            if self.ref_names and self.donate_ctx and self.bags:
                e.ctx["cc"] = self._ref(("cc", e.rid), e.ctx["cc"])
            e.leaf = self.ctx.ifit(e.ctx, w)
            self._at_add(e, "ctx")
            self._at_add(e, "cnt")
            if e.const > 0:
                self._pas(e, "cnt", CONST, "c", e.const)
                self._pas(e, "ctx", CONST, "c", e.const)
            if self.type_names or self.neighbour_types:
                self._add_type(e)
            if self.word_content:
                self._add_word_content(e)
            self.records.append(e)
            self.stats["composites stored"] += 1
        # what generation reads: the committed analysis as statistics of concepts
        for e in words:
            self._pas(e, "ctx", FILL, ("w", e.token), 1.0)
        for key in chosen:
            e = comps[key]
            if e.cleaf is None or e.leaf is None:
                continue
            # statistics name records (stable), not leaves (which reassignment moves)
            self._pas(e, "ctx", FILL, ("c", e.rid), 1.0)
            for a, part in ((LPART, e.left), (RPART, e.right)):
                if part.leaf is not None:
                    self._pas(e, "cnt", a, (part.rid, part.token is not None), 1.0)
            if e.span == (0, len(tokens)):
                self._pas(e, "cnt", SENT, e.rid, 1.0)
        self.stats["experiences"] += 1
        if self.repair_reached:
            self._repair(words + [comps[k] for k in comps if comps[k].rid is not None])
        self._refresh(first, tokens)
        return r["tree"]

    def _refresh(self, first: int, tokens) -> None:
        """Re-describe stored instances whose named concepts moved: the newest
        and a round-robin slice of the rest (``refresh_budget``)."""
        recs = self.records
        self._wc_cache = {}                 # re-read word content names after this experience
        self._rel_cache = {}
        self._nb_cache = {}
        if self.ref_names:
            self._update_names()
        b = self.refresh_budget
        if b is None or first <= b:
            todo = recs[first:] if b is not None and first > b else recs
        else:
            old = recs[:first]
            s = self._rr % len(old)
            todo = recs[first:] + (old[s:] + old[:s])[:b]
            self._rr = (s + b) % len(old)
        lazy = self.refresh in ("relabel", "verify", "verify-changed", "verify-events", "verify-drift")
        if self.refresh == "verify-events":
            # event-driven repair: besides the records whose description changed,
            # check those stored under a concept a merge or split just changed
            # (at most refresh_budget of them, the rest wait for a later event)
            hit = []
            for which, tree in (("cnt", self.cnt), ("ctx", self.ctx)):
                dirty = getattr(tree, "restructured", None)
                if not dirty:
                    continue
                leaves = set()
                for d in dirty:
                    stack = [d]
                    while stack:
                        n = stack.pop()
                        if n.children:
                            stack.extend(n.children)
                        else:
                            leaves.add(id(n))
                dirty.clear()
                for e in recs[:first]:
                    if id(e.cleaf if which == "cnt" else e.leaf) in leaves:
                        hit.append((which, e))
            if b is not None and len(hit) > b:
                hit = random.Random(len(recs)).sample(hit, b)
            for which, e in hit:
                if which == "cnt" and e.token is None and e.cleaf is not None and self._misplaced(self.cnt, e.cleaf, e.cnt, e.weight):
                    self._move(e, "cnt", e.cnt)
                    self.stats["reassigned"] = self.stats.get("reassigned", 0) + 1
                elif which == "ctx" and e.leaf is not None and self._misplaced(self.ctx, e.leaf, e.ctx, e.weight):
                    self._move(e, "ctx", e.ctx)
                    self.stats["reassigned"] = self.stats.get("reassigned", 0) + 1
        for e in todo:
            moved_c = moved_x = False
            if self.refresh == "verify-drift" and e.placed is None:
                e.placed = {"cnt": e.cnt, "ctx": e.ctx}
            if self.ref_names:
                pass                        # descriptions are references: nothing to re-describe
            elif e.token is None:
                e.memo.pop("own", None)
                new = self._content(e)
                moved_c = new != e.cnt
                if new != e.cnt and (lazy or
                                     (self.refresh == "drift" and _drift(e.cnt, new) <= self.drift_tau)):
                    self.cnt.relabel(e.cleaf, e.cnt, new, e.weight)
                    e.cnt = new
                    self.stats["refreshed"] += 1
                elif new != e.cnt:
                    self._move(e, "cnt", new)
                    self.stats["refreshed"] += 1
            # the context's named concepts (children, own content) follow too
            new = dict(e.ctx)
            if self.ref_names:
                new = e.ctx
            if self.children and e.token is None:
                for p, part in (("cl", e.left), ("cr", e.right)):
                    self._facet(new, p, self._ctx_ids(part))
            if self.donate_ctx and not self.ref_names:
                self._facet(new, "cc", self._cc_ids(e))
            if self.neighbour_types and e.nbt is not None:
                self._nb_facets(new, e)
            if self.window_names and e.win is not None:
                new.update(self._named_window(e.win))
            moved_x = new != e.ctx
            if new != e.ctx and (lazy or
                                 (self.refresh == "drift" and _drift(e.ctx, new) <= self.drift_tau)):
                self.ctx.relabel(e.leaf, e.ctx, new, e.weight)
                e.ctx = new
                self.stats["refreshed"] += 1
            elif new != e.ctx:
                self._move(e, "ctx", new)
                self.stats["refreshed"] += 1
            if self.refresh in ("verify", "verify-changed", "verify-events"):
                # lazy repair by placement: move a record only when the concepts at its
                # information bands are no longer those its current description sorts to
                # ("verify-changed": checked only when its description just changed)
                chk = self.refresh == "verify"     # "verify-events": changed descriptions here
                if (chk or moved_c) and e.token is None and e.cleaf is not None and self._misplaced(self.cnt, e.cleaf, e.cnt, e.weight):
                    self._move(e, "cnt", e.cnt)
                    self.stats["reassigned"] = self.stats.get("reassigned", 0) + 1
                if (chk or moved_x) and e.leaf is not None and self._misplaced(self.ctx, e.leaf, e.ctx, e.weight):
                    self._move(e, "ctx", e.ctx)
                    self.stats["reassigned"] = self.stats.get("reassigned", 0) + 1
            elif self.refresh == "verify-drift":
                # descriptions follow every visit (cheap); placement is checked only once
                # a description has drifted by more than drift_tau since it was last placed
                # or checked, so small shifts in name weights cost no sort
                for which, tree, leaf in (("cnt", self.cnt, e.cleaf), ("ctx", self.ctx, e.leaf)):
                    cur = e.cnt if which == "cnt" else e.ctx
                    if leaf is None or (which == "cnt" and e.token is not None):
                        continue
                    if _drift(e.placed[which], cur) <= self.drift_tau:
                        continue
                    self.stats["checked"] = self.stats.get("checked", 0) + 1
                    if self._misplaced(tree, leaf, cur, e.weight):
                        self._move(e, which, cur)
                        self.stats["reassigned"] = self.stats.get("reassigned", 0) + 1
                    e.placed[which] = cur

    # ------------------------------------------------------------------ #
    # Reference names (ref_names)
    # ------------------------------------------------------------------ #
    def _fv(self, ids):
        x = {}
        self._facet(x, "t", ids)
        return x["t"]

    def _ref(self, key, value):
        """A reference to ``key`` whose current value is ``value`` (entered in the table)."""
        self.names.set(key, value)
        self.ref_keys.add(key)
        from .ref_cobweb import Ref
        return Ref(key)

    def _part_key(self, part: Elem, kind: str):
        if part.token is not None:
            return (kind, "w", part.token) if kind == "P" else None
        return (kind, part.rid) if part.rid is not None and part.cleaf is not None else None

    def _ref_content(self, e: Elem, x: dict) -> dict:
        """The content description with each part's name (and a chunk part's content) as references."""
        if not self.bags:
            return x
        x = dict(x)
        for p, part in (("L", e.left), ("R", e.right)):
            k = self._part_key(part, "P")
            if k is not None:
                x[p] = self._ref(k, x[p])
            if self.donate_cnt:
                k = self._part_key(part, "C")
                if k is not None:
                    x[f"{p}C"] = self._ref(k, x[f"{p}C"])
        return x

    def _key_value(self, key):
        kind = key[0]
        if kind == "P" and key[1] == "w":
            return self._fv(self._ctx_name(Elem((0, 1), token=key[2]), part=True))
        e = self.records[key[1]]
        if kind == "P":
            return self._fv(self._ctx_name(e, part=True))
        if kind == "C":
            return self._fv(self._cnt_name(e))
        return self._fv(self._cc_ids(e))                        # "cc"

    def _update_names(self) -> None:
        """Enter every referenced name's current value (after an experience)."""
        n0 = self.names.version
        for key in self.ref_keys:
            self.names.set(key, self._key_value(key))
        self.stats["name changes"] = self.stats.get("name changes", 0) + self.names.version - n0

    def _at_add(self, e: Elem, which: str) -> None:
        if self.repair_reached:
            leaf = e.leaf if which == "ctx" else e.cleaf
            self._at[which].setdefault(id(leaf), []).append(e)

    def _at_remove(self, e: Elem, which: str) -> None:
        if self.repair_reached:
            leaf = e.leaf if which == "ctx" else e.cleaf
            lst = self._at[which].get(id(leaf), [])
            if e in lst:
                lst.remove(e)
            if not lst:
                self._at[which].pop(id(leaf), None)

    def _repair(self, new: List[Elem]) -> None:
        """Check the records stored beside the experience's records (the leaves under
        the concepts its sorts reached) and move the misplaced ones."""
        for which, tree in (("cnt", self.cnt), ("ctx", self.ctx)):
            mine = {id(e) for e in new}
            todo, seen = [], set()
            for e in new:
                leaf = e.leaf if which == "ctx" else e.cleaf
                if leaf is None or leaf.parent is None:
                    continue
                for c in leaf.parent.children:
                    if c.children or id(c) in seen:
                        continue
                    seen.add(id(c))
                    todo.extend(r for r in self._at[which].get(id(c), []) if id(r) not in mine)
            if len(todo) > self.repair_reached:
                todo = self.rng.sample(todo, self.repair_reached)
            for r in todo:
                leaf = r.leaf if which == "ctx" else r.cleaf
                x = r.ctx if which == "ctx" else r.cnt
                if leaf is None or (which == "cnt" and r.token is not None):
                    continue
                self.stats["repair checked"] = self.stats.get("repair checked", 0) + 1
                if self._misplaced(tree, leaf, x, r.weight):
                    self._move(r, which, x)
                    self.stats["repaired"] = self.stats.get("repaired", 0) + 1

    def _misplaced(self, tree, leaf, x: dict, w: float = 1.0) -> bool:
        """Do the band-level concepts above a stored leaf differ from those its
        description sorts to now? (verify_weight="own": sorted at the record's own
        weight, as ifit placed it; else at weight 1)"""
        stored = tree.ancestors(leaf)[::-1][:-1]
        now = tree.categorize(x, w if self.verify_weight == "own" else 1.0)[:-1]
        return self._path_ids(stored + [leaf], "bands") != self._path_ids(now + [leaf], "bands")

    def _pas(self, e: Elem, which: str, attr: str, value, w: float) -> None:
        """Add a passive count at the record's leaf in one hierarchy and remember
        it, so that reassigning the record carries its statistics along."""
        tree, leaf = (self.ctx, e.leaf) if which == "ctx" else (self.cnt, e.cleaf)
        tree.add_passive(leaf, attr, value, w)
        e.pas.append((which, attr, value, w))

    def _move(self, e: Elem, which: str, new: dict) -> None:
        """Reassignment: take the record out of one hierarchy and insert it again
        with its current description, its passive counts moving with it."""
        mine = [(a, v, w) for t, a, v, w in e.pas if t == which]
        self._at_remove(e, which)
        if which == "ctx":
            for a, v, w in mine:
                self.ctx.add_passive(e.leaf, a, v, -w)
            self.ctx.unfit(e.leaf, e.ctx, e.weight)
            e.ctx, e.leaf = new, self.ctx.ifit(new, e.weight)
            for a, v, w in mine:
                self.ctx.add_passive(e.leaf, a, v, w)
        else:
            for a, v, w in mine:
                self.cnt.add_passive(e.cleaf, a, v, -w)
            self.cnt.unfit(e.cleaf, e.cnt, e.weight)
            e.cnt, e.cleaf = new, self.cnt.ifit(new, e.weight)
            for a, v, w in mine:
                self.cnt.add_passive(e.cleaf, a, v, w)
        self._at_add(e, which)

    # ------------------------------------------------------------------ #
    # Generation: top-down through the two hierarchies (no word-to-word step)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _index(tree):
        idx, stack = {}, [(tree.root, 0)]
        while stack:
            n, d = stack.pop()
            idx[f"c{n.id}"] = (n, d)
            stack.extend((c, d + 1) for c in n.children)
        return idx

    def _pick(self, rng, counts: Dict, idx=None, skip=()):
        """Sample a value by its count; concept ids only at their deepest named depth."""
        items = [(v, m) for v, m in counts.items() if m > 0 and v not in skip]
        if idx is not None:
            items = [(v, m) for v, m in items if v in idx]
            if items:
                deep = max(idx[v][1] for v, _ in items)
                items = [(v, m) for v, m in items if idx[v][1] == deep]
        if not items:
            return None
        r = rng.random() * sum(m for _, m in items)
        for v, m in items:
            r -= m
            if r <= 0:
                return v
        return items[-1][0]

    def generate(self, rng: Optional[random.Random] = None, max_depth: int = 12) -> Optional[List[str]]:
        """A sentence from memory, top-down: the sentence context (BOS ... EOS)
        names, through its donated own-content facet, the content concepts that
        fill it; a content concept names its parts' context concepts, content
        concepts and heights; a part of height 0 is a word drawn from its
        context concept's word counts, a taller part expands its content concept."""
        rng = rng or self.rng
        if not self.donate_ctx or not self.donate_cnt:
            raise ValueError("generation reads the donated facets (donate_ctx and donate_cnt)")
        cidx, nidx = self._index(self.ctx), self._index(self.cnt)
        x = {f"l{k}": BOS for k in range(1, self.window + 1)}
        x.update({f"r{k}": EOS for k in range(1, self.window + 1)})
        path = self.ctx.categorize(x)
        node = next((n for n in reversed(path) if self._pick(rng, n.av.get("cc", {}), nidx, (BLANK,))), None)
        if node is None:
            return None
        top = self._pick(rng, node.av.get("cc", {}), nidx, (BLANK,))
        out: List[str] = []

        def word(ctx_id):
            w = self._pick(rng, self.ctx.passive(cidx[ctx_id][0], WORD)) if ctx_id in cidx else None
            if w is None:
                raise LookupError
            out.append(w)

        def expand(cnt_id, d):
            c = nidx[cnt_id][0]
            for p in ("L", "R"):
                h = self._pick(rng, c.av.get(f"{p}H", {}))
                sub = self._pick(rng, c.av.get(f"{p}C", {}), nidx, (BLANK,))
                if h == "0" or sub is None or d >= max_depth:
                    word(self._pick(rng, c.av.get(p, {}), cidx, (BLANK,)))
                else:
                    expand(sub, d + 1)
        try:
            expand(top, 0)
        except LookupError:
            return None
        return out

    def _members(self, tree: IncrementalCobwebTree, which: str):
        """concept id -> (stored records under it, their weights), for one hierarchy."""
        at: Dict[int, List[Elem]] = {}
        for e in self.records:
            leaf = e.leaf if which == "ctx" else e.cleaf
            if leaf is not None:
                at.setdefault(id(leaf), []).append(e)
        out: Dict[str, List[Elem]] = {}
        # post-order without recursion (trees can be deeper than Python's stack)
        stack, order = [tree.root], []
        while stack:
            n = stack.pop()
            order.append(n)
            stack.extend(n.children)
        for n in reversed(order):
            out[f"c{n.id}"] = (list(at.get(id(n), [])) if not n.children
                               else [r for c in n.children for r in out[f"c{c.id}"]])
        return out

    def generate_sub(self, rng: Optional[random.Random] = None, depth: Optional[int] = None,
                     via: str = "ctx", max_depth: int = 12, members=None,
                     up: Optional[int] = None) -> Optional[List[str]]:
        """A sentence by substitution (v1's substitutability): a stored sentence-
        spanning element is expanded part by part, each part replaced by a stored
        element drawn (by weight) from the part's concept at ``depth``: its
        context concept (``via='ctx'``), or for a composite part its content
        concept (``via='cnt'``); drawn words are emitted, drawn composites expand.
        With ``up`` the concept is ``up`` levels above the part's own leaf
        (leaf-relative granularity) instead of at a fixed depth."""
        rng = rng or self.rng
        d = depth or self.depth
        if members is None:
            members = (self._members(self.ctx, "ctx"), self._members(self.cnt, "cnt"))
        mctx, mcnt = members
        roots = [e for e in self.records if e.token is None and self.is_sentence(e)]
        if not roots:
            return None

        def draw(pool):
            pool = [r for r in pool if r.weight > 0]
            if not pool:
                return None
            t = rng.random() * sum(r.weight for r in pool)
            for r in pool:
                t -= r.weight
                if t <= 0:
                    return r
            return pool[-1]
        out: List[str] = []

        def concept(tree, leaf, x):
            path = tree.ancestors(leaf)[::-1] if leaf is not None else tree.categorize(x)
            if up is not None:
                return f"c{path[max(len(path) - 1 - up, 1 if len(path) > 1 else 0)].id}"
            return f"c{path[min(d, len(path) - 1)].id}"

        def expand(e, k):
            for part in (e.left, e.right):
                if via == "cnt" and part.token is None:
                    pool = mcnt.get(concept(self.cnt, part.cleaf, part.cnt), [])
                else:
                    pool = mctx.get(concept(self.ctx, part.leaf, part.ctx), [])
                r = draw(pool) if k < max_depth else None
                if r is None:
                    r = part
                if r.token is not None:
                    out.append(r.token)
                elif r.left is None:
                    raise LookupError
                else:
                    expand(r, k + 1)
        try:
            expand(draw(roots), 0)
        except (LookupError, RecursionError):
            return None
        return out

    def generate_expand(self, rng: Optional[random.Random] = None, members=None, up: int = 0,
                        max_depth: int = 12, kind: bool = True, by: str = "weight",
                        right_hint: bool = False, min_members: Optional[int] = None) -> Optional[List[str]]:
        """Expansive generation. Sample a sentence-level chunk from memory as the
        root, then expand top-down, left to right. Each slot carries metadata
        (its context concept, ``up`` levels above the stored part's leaf, and
        whether the part was a word or a chunk) and chunk context (the tokens
        already generated to its left; the sentence end when the slot is last).
        Every stored member of the slot's concept is weighted by its stored
        weight times the probability its concept gives those surroundings, and
        one is drawn in proportion (probability matching), then expanded the
        same way: a word is emitted, a chunk's parts become new slots. With
        by="used" a member counts only if it is in an analysis memory chose
        (the chunks it committed to encode its experiences)."""
        rng = rng or self.rng
        if members is None:
            members = self._members(self.ctx, "ctx")
        wt = (lambda m: m.used) if by == "used" else (lambda m: m.weight)
        roots = [e for e in self.records if e.token is None and wt(e) > 0 and self.is_sentence(e)]
        if not roots:
            return None
        sizes = {a: max(len(self.ctx.root.av.get(a, {})), 1) for a in self.ctx_attrs}
        out: List[str] = []

        def surround(m, at_end, hint=None):
            node = m.leaf.parent if m.leaf is not None and m.leaf.parent is not None else m.leaf
            if node is None:
                return 1.0
            p = 1.0
            if hint is not None and not at_end:
                # the right neighbour the chunk being expanded leads us to expect
                d = node.av.get("r1", {})
                p *= (d.get(hint, 0.0) + self.alpha) / (node.count + self.alpha * sizes["r1"])
            if self.boundary == "edge":
                known = [(f"l{k}", out[-k]) for k in range(1, min(self.window, len(out)) + 1)]
                known += [("le", "edge" if not out else "in"), ("re", "edge" if at_end else "in")]
            elif self.boundary == "missing":
                known = [(f"l{k}", out[-k]) for k in range(1, min(self.window, len(out)) + 1)]
                # a slot past an edge is unobserved: weight by how often the
                # member's concept leaves that slot unobserved
                edge = [f"l{k}" for k in range(len(out) + 1, self.window + 1)] + \
                       ([f"r{k}" for k in range(1, self.window + 1)] if at_end else [])
                for a in edge:
                    seen_a = sum(node.av.get(a, {}).values())
                    p *= (max(node.count - seen_a, 0.0) + self.alpha) / (node.count + 2 * self.alpha)
            else:
                known = ([(f"l{k}", out[-k] if k <= len(out) else BOS) for k in range(1, self.window + 1)] +
                         ([(f"r{k}", EOS) for k in range(1, self.window + 1)] if at_end else []))
            for a, v in known:
                d = node.av.get(a, {})
                p *= (d.get(v, 0.0) + self.alpha) / (node.count + self.alpha * sizes[a])
            return p

        def draw(pool, at_end, hint=None):
            scored = [(m, wt(m) * surround(m, at_end, hint)) for m in pool if wt(m) > 0]
            tot = sum(w for _, w in scored)
            if tot <= 0:
                return None
            t = rng.random() * tot
            for m, w in scored:
                t -= w
                if t <= 0:
                    return m
            return scored[-1][0]

        def concept(part):
            path = self.ctx.ancestors(part.leaf)[::-1] if part.leaf is not None else self.ctx.categorize(part.ctx)
            return f"c{path[max(len(path) - 1 - up, 1 if len(path) > 1 else 0)].id}"

        def first(x):
            while x.token is None and x.left is not None:
                x = x.left
            return x.token

        def expand(e, d, at_end, right=None):
            for side, part in (("L", e.left), ("R", e.right)):
                last = at_end and side == "R"
                # metadata of the chunk being expanded: a left part was followed by
                # the first word of its stored sibling; a right part by what follows
                # its parent
                hint = None
                if right_hint:
                    hint = first(e.right) if side == "L" else right
                word = part.token is not None or d >= max_depth
                if min_members:
                    # the deepest concept on the part's path whose committed
                    # members of the part's kind hold ``min_members`` distinct
                    # types: fine where evidence is dense, broader where sparse
                    path = self.ctx.ancestors(part.leaf)[::-1] if part.leaf is not None \
                        else self.ctx.categorize(part.ctx)
                    for node in reversed(path[1:] if len(path) > 1 else path):
                        pool = [m for m in members.get(f"c{node.id}", [])
                                if wt(m) > 0 and (not kind or (m.token is not None) == word)]
                        if len({self._type_key(m) for m in pool}) >= min_members:
                            break
                else:
                    pool = members.get(concept(part), [])
                    if kind:
                        pool = [m for m in pool if (m.token is not None) == word]
                m = draw(pool, last, hint) or part
                if m.token is not None:
                    out.append(m.token)
                elif m.left is None or d >= max_depth:
                    raise LookupError
                else:
                    expand(m, d + 1, last, hint)
        try:
            expand(draw(roots, True), 0, True)
        except (LookupError, RecursionError):
            return None
        return out

    def concept_index(self) -> Tuple[Dict[int, CobwebNode], Dict[int, CobwebNode]]:
        """node id -> node, for each hierarchy (generation addresses concepts by id)."""
        out = []
        for tree in (self.ctx, self.cnt):
            idx, stack = {}, [tree.root]
            while stack:
                n = stack.pop()
                idx[n.id] = n
                stack.extend(n.children)
            out.append(idx)
        return out[0], out[1]

    def generate_tree(self, rng: Optional[random.Random] = None, index=None, level: int = 0,
                      choose: str = "leaf", beta: float = 1.0, kind: Optional[int] = None,
                      primitive_leaf: bool = False, max_depth: int = 12,
                      trace: Optional[list] = None, fill_name=False, fill_peer: bool = False,
                      pattern: Optional[Sequence[str]] = None, ctx_fill: bool = False,
                      budget: int = 4000, gap: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
                      gap_beta: float = 1.0, gap_mode: str = "window") -> Optional[List[str]]:
        """Expansive generation from the two hierarchies alone. Every step reads
        statistics held by concepts: a whole experience is drawn from the content
        root (SENT); a content concept gives where its parts stood (LPART, RPART:
        a context concept, and whether the part was primitive); a context concept
        gives the committed instances that stood there (FILL: a primitive, or a
        content concept to expand). Each draw is in proportion to the counts
        (probability matching).

        Which context concept on a part's path supplies the instances:
        ``choose="leaf"`` the part's own leaf, ``level`` concepts above it;
        ``"best"`` the concept that best describes where the part stands, the
        maximum over the path of log2 share(C) + log2 P(surroundings | C) (the
        quantity learning weighs chunks by; it peaks at the largest concept whose
        members share these surroundings); ``"match"`` a concept drawn in
        proportion to 2 ** that score.

        Prompting: ``pattern`` constrains the output to a template of words and
        wildcards ("*": one or more words, "?": one word), e.g. a prefix
        ["tom", "and", "*"] or an infill ["the", "*", "was", "happy"]. Candidate
        fillers of a slot are drawn without replacement and the derivation
        backtracks when the words so far cannot complete the template (at most
        ``budget`` candidate tries). ``ctx_fill``: a slot's fillers come from the
        context concept its actual surroundings sort to (the words generated on its
        left, the stored slot's right neighbours) instead of the stored slot's own
        leaf, so what was generated (or prompted) conditions what follows.
        ``gap=(left words, right words)``: generate only the missing piece between
        them (infill; a prefix completion is right = []): the gap's surroundings are
        given, and a committed instance (a word, or a chunk then expanded) is drawn
        by its posterior of standing there (``_gap_fills``). ``gap_mode="sibling"``
        (a prefix, right = []): the prompt is parsed into one chunk, a stored whole
        experience is drawn by the posterior that its left part is such a chunk
        (``_sibling_fills``: its part name and content), and only its right part is
        generated, so the continuation is conditioned on the whole prompt chunk."""
        fill_predict = isinstance(fill_name, str) and fill_name.endswith("+predict")
        if fill_predict:
            # "...+predict": a word slot also weights each candidate by the probability
            # of its learned category (word_cc="cat") at this part of the composite
            fill_name = fill_name[:-len("+predict")] or False
        if isinstance(fill_name, str) and fill_name.endswith("+peer"):
            # "both+peer": compare by the fine names, the slot word counting as its closest peer
            fill_name, fill_peer = fill_name[:-len("+peer")], True
        rng = rng or self.rng
        window = [a for a in self.ctx_attrs if a[0] in "lr" and a[1:].isdigit()]
        content = choose.endswith("+content")
        choose = choose.split("+")[0]
        cc_size = len(self.ctx.root.av.get("cc", {})) + 1
        root_n = max(self.ctx.root.count, 1e-9)
        sizes = {a: len(self.ctx.root.av.get(a, {})) + 1 for a in window}
        scored: Dict[int, List[Tuple[float, CobwebNode]]] = {}

        def path_scores(leaf: CobwebNode) -> List[Tuple[float, CobwebNode]]:
            if leaf.id not in scored:
                x = {a: max(leaf.av[a].items(), key=lambda kv: kv[1])[0] for a in window
                     if isinstance(leaf.av.get(a), dict) and leaf.av[a]}
                # with choose="...+content" the part's own content (the cc facet,
                # what stands there) counts too: the leaf's distribution of it
                own = {u: c / max(leaf.count, 1e-9) for u, c in leaf.av.get("cc", {}).items()} \
                    if content else {}
                out, node = [], leaf
                while node is not None:
                    s = math.log2(max(node.count, 1e-9) / root_n)
                    for a, v in x.items():
                        d = node.av.get(a, {})
                        s += math.log2((d.get(v, 0.0) + self.alpha) / (node.count + self.alpha * sizes[a]))
                    if own:
                        d = node.av.get("cc", {})
                        s += sum(p * math.log2((d.get(u, 0.0) + self.alpha) / (node.count + self.alpha * cc_size))
                                 for u, p in own.items())
                    out.append((s, node))
                    node = node.parent
                scored[leaf.id] = out
            return scored[leaf.id]

        def concept_for(leaf: CobwebNode, primitive: bool = False) -> CobwebNode:
            if choose == "backoff":
                # hierarchical back-off (Dirichlet smoothing along the path): stay at
                # a concept holding n committed instances of this kind with
                # probability n / (n + beta), else step up to its parent
                node = leaf
                while node.parent is not None:
                    n = sum(c for f, c in self.ctx.passive(node, FILL).items() if (f[0] == "w") == primitive)
                    if n > 0 and rng.random() < n / (n + beta):
                        break
                    node = node.parent
                return node
            if choose == "best":
                return max(path_scores(leaf), key=lambda t: t[0])[1]
            if choose == "match":
                ps = path_scores(leaf)
                m = max(s for s, _ in ps)
                return draw({id(n): 2.0 ** (s - m) for s, n in ps}, {id(n): n for _, n in ps})
            node = leaf
            for _ in range(level):
                node = node.parent or node
            return node

        def draw(d, items=None):
            tot = sum(v for v in d.values() if v > 0)
            if tot <= 0:
                return None
            t = rng.random() * tot
            for k, v in d.items():
                if v > 0:
                    t -= v
                    if t <= 0:
                        return items[k] if items else k
            return items[k] if items else k

        names: Dict[str, Dict[str, float]] = {}

        builds: Dict[str, Dict[str, float]] = {}

        def build_of(tok: str) -> Dict[str, float]:
            if tok not in builds:
                bag = self._word_content_bag(tok) or {}
                tot = sum(bag.values()) or 1.0
                builds[tok] = {u: p / tot for u, p in bag.items()}
            return builds[tok]

        def name_of(tok: str) -> Dict[str, float]:
            if tok not in names:
                # a slot accepts members of its category: compared as parts are named
                bag = self._ctx_name(Elem((0, 1), token=tok), part=not fill_peer) or {}
                if isinstance(bag, list):
                    real = [u for u in bag if u != BLANK]
                    bag = {u: real.count(u) / len(real) for u in set(real)} if real else {}
                tot = sum(bag.values()) or 1.0
                names[tok] = {u: p / tot for u, p in bag.items()}
            return names[tok]

        out: List[str] = []
        pat = list(pattern) if pattern is not None else None
        window_attrs = [a for a in window if a in ("l1", "l2", "r1", "r2")]
        tries = [0]

        def states(toks) -> set:
            st = {0}
            for t in toks:
                nx = set()
                for q in st:
                    if q < len(pat):
                        if pat[q] == "*":
                            nx.update((q, q + 1))
                        elif pat[q] == "?" or pat[q] == t:
                            nx.add(q + 1)
                st = nx
                if not st:
                    break
            return st

        def viable() -> bool:
            return pat is None or bool(states(out))

        def surroundings_leaf(slot: Elem) -> Optional[List[CobwebNode]]:
            q = {}
            for a in window_attrs:
                k = int(a[1:])
                if a[0] == "l":
                    q[a] = out[-k] if len(out) >= k else BOS
                elif slot.ctx is not None and a in slot.ctx:
                    q[a] = slot.ctx[a]
            return self._query_path(q) if q else None

        def expand(x: CobwebNode, d: int) -> bool:
            for a in (LPART, RPART):
                if not part(x, a, d):
                    return False
            return True

        def part(x: CobwebNode, a: str, d: int) -> bool:
            if True:
                where = draw(self.cnt.passive(x, a))
                if where is None or self.records[where[0]].leaf is None:
                    raise LookupError
                leaf, primitive = self.records[where[0]].leaf, where[1] or d >= max_depth
                node = leaf if (primitive and primitive_leaf) else concept_for(leaf, primitive)
                fills = {f: c for f, c in self.ctx.passive(node, FILL).items() if (f[0] == "w") == primitive}
                if ctx_fill:
                    path = surroundings_leaf(self.records[where[0]])
                    for nd in reversed(path or []):
                        got = {f: c for f, c in self.ctx.passive(nd, FILL).items() if (f[0] == "w") == primitive}
                        if got:
                            node, fills = nd, got
                            break
                if kind is not None and not primitive and node is not leaf:
                    # the AND side agrees: keep composites of the same kind as the
                    # leaf's own, i.e. under the content concepts ``kind`` levels
                    # above the content concepts of the leaf's composite instances
                    kinds = set()
                    for g in self.ctx.passive(leaf, FILL):
                        if g[0] == "c" and self.records[g[1]].cleaf is not None:
                            y = self.records[g[1]].cleaf
                            for _ in range(kind):
                                y = y.parent or y
                            kinds.add(y.id)

                    def under(cid):
                        y = self.records[cid].cleaf
                        while y is not None:
                            if y.id in kinds:
                                return True
                            y = y.parent
                        return False
                    fills = {f: c for f, c in fills.items() if under(f[1])}
                if not fills:                      # the chosen concept holds none of this kind
                    node = leaf
                    fills = {f: c for f, c in self.ctx.passive(node, FILL).items() if (f[0] == "w") == primitive}
                if fill_name and primitive and self.records[where[0]].token is not None:
                    # a word slot: weight each candidate word by how much its learned
                    # representation overlaps that of the word stored in the slot: its
                    # context name, or with fill_name="build" what it builds (the AND
                    # side: the content concepts of the composites it is a part of)
                    # with fill_name="both", the product of the two overlaps (a candidate
                    # must match the slot word in what it builds and where it stands)
                    reps = {"build": [build_of], "both": [name_of, build_of]}.get(fill_name, [name_of])
                    slot_tok = self.records[where[0]].token
                    sim = {}
                    for f in fills:
                        v = 1.0
                        for rep in reps:
                            own = rep(slot_tok)
                            v *= sum(min(p, rep(f[1]).get(u, 0.0)) for u, p in own.items())
                        sim[f] = v
                    if fill_peer and ("w", slot_tok) in sim:
                        # the slot word is one member of its category: it counts as much
                        # as its closest peer, not more for matching its own name
                        peers = [v for f, v in sim.items() if f[1] != slot_tok]
                        if peers and max(peers) > 0:
                            sim[("w", slot_tok)] = min(sim[("w", slot_tok)], max(peers))
                    weighted = {f: c * sim[f] for f, c in fills.items()}
                    if any(v > 0 for v in weighted.values()):
                        fills = weighted
                if fill_predict and primitive and fills:
                    # Cobweb prediction from the AND side: the categories of the words that
                    # stood in this part of the composite concept, each read as it is now
                    def cat_of(tok):
                        bag = self._window_name(tok)
                        return max(bag.items(), key=lambda kv: kv[1])[0] if bag else None
                    cats: Dict[str, float] = {}
                    for (rid, _), c in self.cnt.passive(x, a).items():
                        tok = self.records[rid].token
                        if tok is not None:
                            k = cat_of(tok)
                            cats[k] = cats.get(k, 0.0) + c
                    tot = sum(cats.values())
                    if tot > 0:
                        size = len(cats) + 1
                        weighted = {f: c * (cats.get(cat_of(f[1]), 0.0) + self.alpha) / (tot + self.alpha * size)
                                    for f, c in fills.items()}
                        fills = weighted
                if pat is None:
                    f = draw(fills)
                    entry = None
                    if trace is not None:
                        entry = {"depth": d, "composite": x.id, "part": "left" if a == LPART else "right",
                                 "leaf": leaf.id, "concept": node.id, "primitive": primitive,
                                 "instances": fills, "drawn": f, "slot": where[0], "start": len(out)}
                        trace.append(entry)
                    if f is None:
                        raise LookupError
                    if f[0] == "w":
                        out.append(f[1])
                    elif self.records[f[1]].cleaf is not None:
                        expand(self.records[f[1]].cleaf, d + 1)
                    else:
                        raise LookupError
                    if entry is not None:
                        entry["end"] = len(out)
                    return True
                # prompted: fillers without replacement, backtracking on a dead end
                left = dict(fills)
                while left:
                    tries[0] += 1
                    if tries[0] > budget:
                        raise LookupError
                    f = draw(left)
                    if f is None:
                        break
                    del left[f]
                    mark = len(out)
                    if f[0] == "w":
                        out.append(f[1])
                        if viable():
                            return True
                    elif self.records[f[1]].cleaf is not None:
                        try:
                            if expand(self.records[f[1]].cleaf, d + 1) and viable():
                                return True
                        except LookupError:
                            if tries[0] > budget:
                                raise
                    del out[mark:]
                return False

        if gap is not None and gap_mode == "sibling" and not gap[1]:
            lw = list(gap[0])
            post = self._sibling_fills(lw, gap_beta)
            if trace is not None:
                top = sorted(post.items(), key=lambda kv: -kv[1])[:10]
                trace.append({"sibling sentences": [(" ".join(self._type_key(self.records[r])), round(v, 4)) for r, v in top]})
            for _ in range(20):
                r = draw(post)
                if r is None:
                    return None
                out.clear()
                try:
                    if part(self.records[r].cleaf, RPART, 1):
                        return lw + list(out)
                except (LookupError, RecursionError):
                    continue
            return None
        if gap is not None:
            lw, rw = list(gap[0]), list(gap[1])
            q = {}
            for a in window_attrs:
                k = int(a[1:])
                if a[0] == "l":
                    q[a] = lw[-k] if len(lw) >= k else BOS
                else:
                    q[a] = rw[k - 1] if len(rw) >= k else EOS
            fills = self._gap_fills(q, gap_beta)
            if trace is not None:
                top = sorted(fills.items(), key=lambda kv: -kv[1])[:10]
                trace.append({"gap fillers": [(f, round(v, 4)) for f, v in top]})
            for _ in range(20):
                f = draw(fills)
                if f is None:
                    return None
                out.clear()
                try:
                    if f[0] == "w":
                        out.append(f[1])
                    elif self.records[f[1]].cleaf is not None:
                        expand(self.records[f[1]].cleaf, 1)
                    else:
                        continue
                except (LookupError, RecursionError):
                    continue
                return lw + list(out) + rw
            return None
        if pat is None:
            root = draw(self.cnt.passive(self.cnt.root, SENT))
            if root is None or self.records[root].cleaf is None:
                return None
            try:
                expand(self.records[root].cleaf, 0)
            except (LookupError, RecursionError):
                return None
            return out
        roots = {r: c for r, c in self.cnt.passive(self.cnt.root, SENT).items() if self.records[r].cleaf is not None}
        while roots:
            r = draw(roots)
            del roots[r]
            out.clear()
            try:
                if expand(self.records[r].cleaf, 0) and len(pat) in states(out):
                    return list(out)
            except LookupError:
                if tries[0] > budget:
                    return None
            except RecursionError:
                return None
        return None

    def _gap_fills(self, q: dict, beta: float = 1.0) -> Dict[tuple, float]:
        """P(filler | surroundings q) over the committed instances: proportional to
        prod over q's attributes of P(value | the instance's context leaf), each
        estimated along the leaf's path with hierarchical Dirichlet back-off
        P(v | C) = (n_C(v) + beta P(v | parent)) / (n_C + beta) from a Laplace root,
        so a chunk is scored by how probably it stands in these surroundings and
        unseen combinations borrow from the concepts above its leaf."""
        root = self.ctx.root
        sizes = {a: len(root.av.get(a, {})) + 1 for a in q}
        memo: Dict[int, Dict[str, float]] = {}

        def probs(node) -> Dict[str, float]:
            if node.id in memo:
                return memo[node.id]
            if node.parent is None:
                out = {a: (node.av.get(a, {}).get(v, 0.0) + self.alpha) / (node.count + self.alpha * sizes[a])
                       for a, v in q.items()}
            else:
                up = probs(node.parent)
                out = {a: (node.av.get(a, {}).get(v, 0.0) + beta * up[a]) / (node.count + beta)
                       for a, v in q.items()}
            memo[node.id] = out
            return out

        cache = self.__dict__.setdefault("_gap_cache", {})
        key = (tuple(sorted(q.items())), len(self.records), self.stats.get("experiences", 0), beta)
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        mass: Dict[tuple, float] = {}
        for e in self.records:
            if e.leaf is None:
                continue
            for which, attr, value, w in e.pas:
                if which == "ctx" and attr == FILL:
                    lp = sum(math.log(max(v, 1e-300)) for v in probs(e.leaf).values())
                    mass[value] = mass.get(value, 0.0) + w * math.exp(lp)
        cache[key] = mass
        return mass

    def _sibling_fills(self, left: Sequence[str], beta: float = 1.0) -> Dict[int, float]:
        """P(committed composite | its left part is a chunk on the prompt's right
        edge): the prompt is parsed; a continuation attaches to one of the units
        ending at the prompt's end (the last word, each chosen composite ending
        there, up to the whole prompt), each described as a left part is (part name
        L, content LC). A committed composite is scored by the expected
        log-probability of such a description at its content leaf (hierarchical
        Dirichlet back-off along the leaf's path, as ``_gap_fills``), summed over the
        attachment points; its right part is then what the gap generates."""
        left = list(left)
        cache = self.__dict__.setdefault("_sib_cache", {})
        key = (tuple(left), len(self.records), self.stats.get("experiences", 0), beta)
        if key in cache:
            return cache[key]
        if len(cache) > 256:
            cache.clear()
        n = len(left)
        units = [Elem((n - 1, n), token=left[-1])]
        if n > 1:
            r = self.parse(left)
            units += [r["comps"][k] for k in sorted(r["chosen"], key=lambda k: -k[0]) if k[2] == n]
        post: Dict[int, float] = {}
        for el in units:
            for rid, v in self._left_part_post(el, beta).items():
                post[rid] = post.get(rid, 0.0) + v
        cache[key] = post
        return post

    def _left_part_post(self, el: Elem, beta: float) -> Dict[int, float]:
        """Committed composites scored by P(their left part is ``el``) (see _sibling_fills)."""
        x = {}
        self._facet(x, "L", self._ctx_name(el, part=True))
        if self.donate_cnt:
            self._facet(x, "LC", self._cnt_name(el))
        q = {a: (v if isinstance(v, dict) else {v: 1.0}) for a, v in x.items()}
        tree = self.cnt
        sizes = {a: len(tree.root.av.get(a, {})) + 1 for a in q}
        memo: Dict[int, Dict[str, Dict]] = {}

        def probs(node):
            if node.id in memo:
                return memo[node.id]
            out = {}
            up = probs(node.parent) if node.parent is not None else None
            for a, bag in q.items():
                d = node.av.get(a, {})
                if up is None:
                    out[a] = {v: (d.get(v, 0.0) + self.alpha) / (node.count + self.alpha * sizes[a]) for v in bag}
                else:
                    out[a] = {v: (d.get(v, 0.0) + beta * up[a][v]) / (node.count + beta) for v in bag}
            memo[node.id] = out
            return out

        post: Dict[int, float] = {}
        for e in self.records:
            if e.cleaf is None:
                continue
            for which, attr, value, w in e.pas:
                if which == "ctx" and attr == FILL:
                    pr = probs(e.cleaf)
                    lp = sum(p * math.log(max(pr[a][v], 1e-300)) for a, bag in q.items() for v, p in bag.items())
                    post[e.rid] = post.get(e.rid, 0.0) + w * math.exp(lp)
        return post

    def _query_path(self, x: dict) -> List[CobwebNode]:
        """Sort a partial description by probability: at each concept the child
        maximising log share + sum over the given attributes of log P(value | child)
        (Laplace-smoothed with ``alpha``; attributes not given are not scored), so a
        query that leaves out some attributes still lands among instances that
        share the given ones."""
        tree = self.ctx
        node, path = tree.root, [tree.root]
        sizes = {a: len(tree.root.av.get(a, {})) + 1 for a in x}
        while node.children:
            def score(c):
                s = math.log(max(c.count, 1e-9))
                for a, v in x.items():
                    d = c.av.get(a, {})
                    s += math.log((d.get(v, 0.0) + self.alpha) / (c.count + self.alpha * sizes[a]))
                return s
            node = max(node.children, key=score)
            path.append(node)
        return path

    def generate_best(self, rng: Optional[random.Random] = None, members=None, k: int = 8,
                      up: int = 0, by: str = "weight") -> Optional[List[str]]:
        """Expand ``k`` candidate sentences, then draw one in proportion to how
        well memory recognises it: exp of the parser's log total weight over all
        analyses, per composite (so length does not decide). The parser acts as
        the critic of the generator; drawing in proportion keeps it novel."""
        rng = rng or self.rng
        if members is None:
            members = self._members(self.ctx, "ctx")
        cands = [s for s in (self.generate_expand(rng, members=members, up=up, by=by) for _ in range(k)) if s]
        if not cands:
            return None
        scores = []
        for s in cands:
            if len(s) < 2:
                scores.append(0.0)
                continue
            r = self.parse(s)
            scores.append(r["logZ"] / (len(s) - 1))
        m = max(scores)
        w = [math.exp(v - m) for v in scores]
        t = rng.random() * sum(w)
        for s, v in zip(cands, w):
            t -= v
            if t <= 0:
                return s
        return cands[-1]
