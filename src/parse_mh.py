"""

Primary module for multi-hierarchy theory! We're programming the whole thing from scratch,
going to try to leverage some of the old code but the parse trees need a large rework
especially regarding their visualization (context-hierarchy and content-hierarchy both need
to be worked).

See implementation details in MULTIHIERARCHY.md!

Key difference from parse.py (single-hierarchy):
    - TWO Cobweb hierarchies: one for content, one for context.
    - Content instances use **leaf pointer encoding** with just 2 attributes
      (Methodology 4.0):
          content_instance = {
              0: {left_label_path: 1},   # context-hierarchy leaf concept for left child
              1: {right_label_path: 1},  # context-hierarchy leaf concept for right child
          }
      Soft similarity between values is computed via LCA depth in the context
      hierarchy (the ref_tree), so that nodes sharing a deep common ancestor
      are treated as more similar than unrelated nodes.
    - Context instances carry sliding-window context + complexity + content-ref:
          context_instance = {0..ctx_len-1: ctx_before,       (slot mode)
                              ctx_len..2*ctx_len-1: ctx_after, (slot mode)
                              -2: complexity (hidden, stored as {C{X}_vid: 1}),
                              2*ctx_len: content-ref (visible)}
          In BOW mode:      {0: before_bag, 1: after_bag,
                             -2: complexity (hidden, stored as {C{X}_vid: 1}),
                             2: content-ref (visible)}
          Complexity uses per-level vocab identifiers C1, C2, C3 … so the key
          itself encodes the complexity value (count is always 1).
      Content-ref is at the first positive index after all context attributes
      (2*ctx_len in slot mode, 2 in BOW mode) so Cobweb includes it in entropy
      calculations.
    - Primitive labels: {word_id: 1}.
    - Composite labels: {context_leaf_concept_id: 1}.
    - Only frozen/accepted chunks are added to BOTH hierarchies; unfrozen
      candidates go only to the content hierarchy.
    - Scoring is recognition-based (tree-wide log-probability).
"""
import uuid
import os
import json
import asyncio
from playwright.async_api import async_playwright
import re
from cobweb.cobweb_discrete import CobwebDiscreteTree, CobwebDiscreteNode
from viz import HTMLCobwebDrawer
from typing import List, Tuple, Optional, Dict, Any
from sortedcontainers import SortedList
import heapq
import time
import math
import random
from pprint import pprint


# ---------------------------------------------------------------------------
# Weighting helper
# ---------------------------------------------------------------------------

def _context_weight(j: int, mode: str = "binary") -> float:
    """Return the weight for context slot at distance *j* (0-indexed).

    Parameters
    ----------
    j : int
        0-indexed distance from the target word (0 = immediate neighbour).
    mode : str
        ``'binary'``   – 1 / 2^(j+1)   (default, original behaviour)
        ``'harmonic'`` – 1 / (j+1)
        ``'constant'`` – 1
    """
    if mode == "harmonic":
        return 1.0 / (j + 2)
    elif mode == "constant":
        return 1.0
    else:  # 'binary' (default / fallback)
        return 1.0 / (2 ** (j + 1))


# ---------------------------------------------------------------------------
# Categorization helpers (ported from parse.py, hierarchy-agnostic)
# ---------------------------------------------------------------------------

def _categorize_dfs(inst: dict, tree: CobwebDiscreteTree,
                    stochastic: bool = False):
    """
    DFS categorization down a CobwebDiscreteTree, returning
    ``(leaf_node, path_strings, node_path)``.
    ``path_strings`` is ``["CONCEPT-<hash>", ...]`` from root to leaf.

    Parameters
    ----------
    stochastic : bool
        If True, sample from the child probability distribution at each
        level instead of always picking the best child.  This introduces
        variety during generation so that identical inputs can produce
        different outputs.
    """
    path: List[str] = []
    node_path: List[CobwebDiscreteNode] = []

    try:
        node = tree.root
    except Exception:
        try:
            leaf = tree.categorize(inst)
            return leaf, [f"CONCEPT-{leaf.concept_hash()}"], [leaf]
        except Exception:
            return None, [], []

    try:
        path.append(f"CONCEPT-{node.concept_hash()}")
        node_path.append(node)
        last_node = node
    except Exception:
        return None, [], []

    while True:
        try:
            child_scores = node.log_prob_children_given_instance(inst)
        except Exception:
            break

        if not child_scores:
            break

        # Convert log-probs to usable floats
        scores = []
        for v in child_scores:
            try:
                val = float(v)
                if math.isnan(val):
                    val = -float("inf")
            except Exception:
                val = -float("inf")
            scores.append(val)

        if all(s == -float("inf") for s in scores):
            break

        if stochastic:
            # Convert log-probs to probabilities via softmax and sample
            max_s = max(scores)
            weights = [math.exp(s - max_s) if s != -float("inf") else 0.0
                       for s in scores]
            total = sum(weights)
            if total <= 0:
                break
            chosen_idx = random.choices(range(len(weights)), weights=weights, k=1)[0]
        else:
            chosen_idx = None
            best_val = -float("inf")
            for i, val in enumerate(scores):
                if val > best_val:
                    best_val = val
                    chosen_idx = i
            if chosen_idx is None:
                break

        try:
            node = node.children[chosen_idx]
        except Exception:
            try:
                node = node.children[chosen_idx][1]
            except Exception:
                break

        try:
            path.append(f"CONCEPT-{node.concept_hash()}")
            node_path.append(node)
            last_node = node
        except Exception:
            break

    return last_node, path, node_path


# ---------------------------------------------------------------------------
# BFS categorization helpers (Python ports of C++ predict / predict_pmi)
# ---------------------------------------------------------------------------

def _logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _categorize_bfs(inst: dict, tree: CobwebDiscreteTree):
    """
    BFS (best-first) categorization down a CobwebDiscreteTree.

    Mirrors the C++ ``CobwebDiscreteTree::predict`` method but terminates
    as soon as a **leaf** node is popped from the priority queue.

    Returns ``(leaf_node, path_strings, node_path, depth_dists)``
    where *depth_dists* is a list of dicts ``[{concept_str: weight}, ...]``
    (one per depth level) containing **all** BFS-explored nodes at that
    depth with normalised probability weights.
    """
    try:
        root = tree.root
    except Exception:
        return None, [], [], []

    root_ll = root.log_prob_instance(inst)
    if math.isnan(root_ll) or root_ll == 0:
        root_ll = -1e8

    # Max-heap via negation; tie-break on id to avoid comparing nodes
    heap = [(-root_ll, id(root), root)]

    total_weight = 0.0
    depth_raw: dict = {}      # depth → [(concept_str, log_score, node)]
    node_path: List[CobwebDiscreteNode] = []
    leaf_node = None

    while heap:
        neg_score, _, curr = heapq.heappop(heap)
        curr_score = -neg_score

        # running logsumexp accumulation
        if total_weight == 0.0:
            total_weight = curr_score
        else:
            total_weight = _logsumexp(total_weight, curr_score)

        d = curr.depth()
        concept_str = f"CONCEPT-{curr.concept_hash()}"

        if d not in depth_raw:
            depth_raw[d] = []
        depth_raw[d].append((concept_str, curr_score, curr))
        node_path.append(curr)

        # --- terminate at first leaf ---
        if not curr.children:
            leaf_node = curr
            break

        # expand children
        for child in curr.children:
            child_ll = child.log_prob_instance(inst)
            if math.isnan(child_ll) or child_ll == 0:
                child_ll = -1e8
            heapq.heappush(heap, (-child_ll, id(child), child))

    # ---- build per-depth normalised distributions ----
    max_d = max(depth_raw.keys()) if depth_raw else 0
    depth_dists: List[dict] = []
    path_strings: List[str] = []

    for d in range(max_d + 1):
        entries = depth_raw.get(d, [])
        if not entries:
            depth_dists.append({})
            path_strings.append("EMPTYNULL")
            continue
        # softmax within depth level
        scores = [e[1] for e in entries]
        max_s = max(scores)
        weights = [math.exp(s - max_s) for s in scores]
        total = sum(weights)
        dist: dict = {}
        for (cstr, _, _nd), w in zip(entries, weights):
            norm_w = w / total if total > 0 else 0
            dist[cstr] = dist.get(cstr, 0) + norm_w
        depth_dists.append(dist)
        best = max(dist, key=dist.get)
        path_strings.append(best)

    return leaf_node, path_strings, node_path, depth_dists


def _categorize_bfs_pmi(inst: dict, tree: CobwebDiscreteTree):
    """
    BFS-PMI categorization – like ``_categorize_bfs`` but each node's
    priority includes a PMI-inspired entropy + cross-entropy term
    (mirrors C++ ``predict_pmi``).

    The PMI contribution is summed across **all** attributes present in
    the tree so that no single attribute needs to be specified.

    Terminates at the first leaf, same as ``_categorize_bfs``.

    Returns ``(leaf_node, path_strings, node_path, depth_dists)``.
    """
    try:
        root = tree.root
    except Exception:
        return None, [], [], []

    root_ll = root.log_prob_instance(inst)
    if math.isnan(root_ll) or root_ll == 0:
        root_ll = -1e8

    # Instance-conditioned predictions (equivalent to calling predict once)
    p_attrs_given_instance = tree.predict(inst, 250, False)

    def _ent_ce(node_pred, p_given_inst):
        """H(node) + CE(node ‖ instance) summed over shared attrs."""
        result = 0.0
        for attr in node_pred:
            pg = p_given_inst.get(attr)
            if pg is None:
                continue
            for val, p in node_pred[attr].items():
                q = pg.get(val, 1e-10)
                if p > 0 and q > 0:
                    result += p * math.log(p) + p * math.log(q)
        return result

    root_pred = root.predict_probs()
    root_ent_ce = _ent_ce(root_pred, p_attrs_given_instance)
    initial_score = root_ll + root_ent_ce

    heap = [(-initial_score, id(root), root)]
    total_weight = 0.0
    depth_raw: dict = {}
    node_path: List[CobwebDiscreteNode] = []
    leaf_node = None

    while heap:
        neg_score, _, curr = heapq.heappop(heap)
        curr_score = -neg_score

        if total_weight == 0.0:
            total_weight = curr_score
        else:
            total_weight = _logsumexp(total_weight, curr_score)

        d = curr.depth()
        concept_str = f"CONCEPT-{curr.concept_hash()}"

        if d not in depth_raw:
            depth_raw[d] = []
        depth_raw[d].append((concept_str, curr_score, curr))
        node_path.append(curr)

        if not curr.children:
            leaf_node = curr
            break

        for child in curr.children:
            child_ll = child.log_prob_instance(inst)
            if math.isnan(child_ll) or child_ll == 0:
                child_ll = -1e8
            child_pred = child.predict_probs()
            child_ent_ce = _ent_ce(child_pred, p_attrs_given_instance)
            child_score = child_ll + child_ent_ce
            heapq.heappush(heap, (-child_score, id(child), child))

    # ---- build per-depth normalised distributions ----
    max_d = max(depth_raw.keys()) if depth_raw else 0
    depth_dists: List[dict] = []
    path_strings: List[str] = []

    for d in range(max_d + 1):
        entries = depth_raw.get(d, [])
        if not entries:
            depth_dists.append({})
            path_strings.append("EMPTYNULL")
            continue
        scores = [e[1] for e in entries]
        max_s = max(scores)
        weights = [math.exp(s - max_s) for s in scores]
        total = sum(weights)
        dist: dict = {}
        for (cstr, _, _nd), w in zip(entries, weights):
            norm_w = w / total if total > 0 else 0
            dist[cstr] = dist.get(cstr, 0) + norm_w
        depth_dists.append(dist)
        best = max(dist, key=dist.get)
        path_strings.append(best)

    return leaf_node, path_strings, node_path, depth_dists


def _categorize(inst: dict, tree: CobwebDiscreteTree,
                mode: str = 'dfs', stochastic: bool = False):
    """
    Unified dispatcher for categorization strategies.

    Returns ``(leaf_node, path_strings, node_path, depth_dists)``.
    For ``'dfs'``, *depth_dists* is ``None``.
    """
    if mode == 'bfs':
        return _categorize_bfs(inst, tree)
    elif mode == 'bfs_pmi':
        return _categorize_bfs_pmi(inst, tree)
    else:  # 'dfs' or default
        leaf, path, node_path = _categorize_dfs(inst, tree, stochastic)
        return leaf, path, node_path, None


def _score_along_path(
    node_path: List[CobwebDiscreteNode],
    instance: dict,
    tree: CobwebDiscreteTree,
    debug: bool = False,
    eval_alpha: float = None,
    _basic_cache: dict = None,
    instance_basic_level: bool = False,
) -> dict:
    """
    Compute recognition statistics along a categorization path.
    Mirrors FiniteParseTree._score_function from parse.py.
    Returns a dict of score metrics!

    FROM NOTES WITH CHRIS - basic level **count** is the cost/score used for
    thresholding.  A count of -1 means the basic-level node collapsed back to
    the root (not enough evidence to form a real category), which should always
    fail the threshold.  A positive count N means the basic-level node has been
    reinforced N times; pass once N > threshold.

    _basic_cache: optional dict for caching get_basic() results by leaf
    concept hash. Useful when the tree is read-only (e.g. during build()).
    """
    raw_log_probs = []
    path_counts = []

    for node in node_path:
        lp = node.log_prob_instance(instance)
        raw_log_probs.append(lp)
        path_counts.append(node.count)

    tree_log_prob = tree.log_prob(instance, 200, False)
    tree_class_log_prob = tree.log_prob_class_given_instance(instance, 200, False)

    _bl_eval_alpha = eval_alpha if eval_alpha is not None else -1.0
    leaf_hash = node_path[-1].concept_hash()
    _basic_key = (id(tree), leaf_hash, _bl_eval_alpha, instance_basic_level) if _basic_cache is not None else None
    if _basic_cache is not None and _basic_key in _basic_cache:
        basic_level_node = _basic_cache[_basic_key]
    else:
        if instance_basic_level:
            basic_level_node = node_path[-1].get_basic_instance_pmi(instance, debug=True, eval_alpha=_bl_eval_alpha)
        else:
            basic_level_node = node_path[-1].get_basic(200, 100, debug=True, eval_alpha=_bl_eval_alpha)
        if _basic_cache is not None:
            _basic_cache[_basic_key] = basic_level_node
    basic_level_log_prob = basic_level_node.log_prob_instance(instance)
    basic_level_class_log_prob = basic_level_node.log_prob_class_given_instance(instance)
    # if basic level node is root node immediately rule this out, we don't have enough evidence!!
    basic_level_count = basic_level_node.count if basic_level_node.concept_hash() != node_path[0].concept_hash() else -1

    score_data = {
        # raw data
        "raw_node_log_probs": str(raw_log_probs),
        "candidate_counts": str(path_counts),

        # basic level stuff
        "basic_level_count": basic_level_count,
        "basic_level_log_prob": basic_level_log_prob,
        "basic_level_class_log_prob": basic_level_class_log_prob,
        "cost": basic_level_count,

        # tree stuff
        "tree_log_prob": tree_log_prob,
        "tree_class_log_prob": tree_class_log_prob,

        # additional info
        "root_log_prob": raw_log_probs[0],
        "leaf_log_prob": raw_log_probs[-1],
    }

    if debug:
        print("-" * 60)
        print("Scoring statistics:")
        pprint(score_data)
        print("-" * 60)

    return score_data


# ---------------------------------------------------------------------------
# Complexity vocab helper
# ---------------------------------------------------------------------------

def _get_or_register_cplx_vid(complexity: int, id_to_value: list, value_to_id: dict) -> int:
    """Return the vocab ID for the complexity identifier ``C{complexity}``.

    Each distinct complexity level gets its own vocab entry (e.g. ``C1``,
    ``C2``, ``C3`` ...) so the identity of the key — rather than the count
    stored as its value — encodes how complex the node is.
    Registers the identifier in the vocabulary on first use.
    """
    cplx_str = f"C{complexity}"
    vid = value_to_id.get(cplx_str)
    if vid is None:
        vid = len(id_to_value)
        id_to_value.append(cplx_str)
        value_to_id[cplx_str] = vid
    return vid


# ---------------------------------------------------------------------------
# label_path helper (single leaf pointer)
# ---------------------------------------------------------------------------

def _build_label_from_ctx_leaf(ctx_leaf, value_to_id: dict) -> int:
    """
    Return the single context-hierarchy leaf concept vocab ID.

    Under Methodology 4.0 the multi-depth label_path is replaced by a
    single leaf pointer — the most-specific concept that a node was
    categorized into.
    """
    if ctx_leaf is None:
        return 0
    concept_str = f"CONCEPT-{ctx_leaf.concept_hash()}"
    return value_to_id.get(concept_str, 0)



# ---------------------------------------------------------------------------
# PrimitiveParseNode
# ---------------------------------------------------------------------------

class PrimitiveParseNode(object):
    """
    Represents a single word/token in the parse tree.

    Every node carries two facets:
        context_instance  – what the context hierarchy sees
                            (sliding-window context + complexity)
        label             – discrete identity {word_id: 1} for primitives
        label_path        – single leaf pointer (context concept vocab ID)
                            used to build content instances.

    Attributes
    ----------
    parent : CompositeParseNode | None
    children : SortedList          always empty for primitives
    position_idx : int             word position in the sentence
    title : str                    unique random id
    context_instance : dict        instance dict for the context hierarchy
    label : dict                   {word_id: 1}
    label_path : int               context leaf concept vocab ID
    complexity : int               always 1 for primitives
    word_id : int                  vocabulary id of the raw token
    score_data : dict              scoring statistics from the context hierarchy
    stable : bool                  whether this primitive passed the threshold
    """

    def __init__(self, context_instance: dict, label: dict, position_idx: int, word_id: int):
        self.parent: Optional['CompositeParseNode'] = None
        self.children: SortedList = SortedList()  # always empty for primitives

        self.title: str = uuid.uuid4().hex[:10]
        self.position_idx: int = position_idx

        self.context_instance: dict = context_instance
        self.label: dict = label  # discrete identity: {word_id: 1} for primitives
        self.label_path: int = 0  # single leaf pointer (context concept vid)
        self.word_id: int = word_id

        self.complexity: int = 1
        self.score_data: dict = {}
        self.context_path_hashes: list = []
        self.stable: bool = False

    def __lt__(self, other):
        return id(self) < id(other)

    # ------------------------------------------------------------------
    @staticmethod
    def create_node(context_instance: dict, label: dict, position_idx: int, word_id: int) -> 'PrimitiveParseNode':
        """Factory method mirroring the old code's static constructors."""
        return PrimitiveParseNode(context_instance, label, position_idx, word_id)

    # ------------------------------------------------------------------
    def set_parent(self, node: 'CompositeParseNode'):
        """Attach this node under *node*, updating both sides of the link."""
        try:
            self.parent.children.remove((self.position_idx, self))
        except (AttributeError, ValueError):
            pass
        self.parent = node
        node.children.add((self.position_idx, self))

    # ------------------------------------------------------------------
    def get_context_instance(self) -> dict:
        """
        Return the context-focused instance for this node.
        Returns a shallow copy of the raw context_instance dict (all int keys)
        so it can be passed directly to CobwebDiscreteTree.ifit().
        """
        return dict(self.context_instance)

    # ------------------------------------------------------------------
    def get_label(self) -> dict:
        """Return the discrete identity dict {word_id: 1} for primitives."""
        return dict(self.label)


# ---------------------------------------------------------------------------
# CompositeParseNode
# ---------------------------------------------------------------------------

class CompositeParseNode(object):
    """
    Represents a merged chunk (two children) in the parse tree.

    Carries two distinct instance facets:
        content_instance  – multi-depth path-encoded attrs for the content hierarchy
        context_instance  – sliding-window context + complexity, for the context hierarchy

    The node's *label* is the weighted categorize path through the context hierarchy
    (same pattern as PrimitiveParseNode.label) so that parent chunks can reference
    this node's identity via its context semantics.
    """

    def __init__(self):
        self.is_global_root: bool = False
        self.position_idx: Optional[float] = None

        self.parent: Optional['CompositeParseNode'] = None
        self.children: SortedList = SortedList()

        self.title: str = uuid.uuid4().hex[:10]

        # content facet
        self.content_instance: Optional[dict] = None  # {0..cpd-1: left depths, cpd..2*cpd-1: right depths}

        # context facet
        self.context_instance: Optional[dict] = None
        self.context_before: Optional[List[dict]] = None
        self.context_after: Optional[List[dict]] = None

        self.label: Optional[dict] = None  # {concept_leaf_id: 1}
        self.label_path: int = 0  # single leaf pointer (context concept vid)
        self.categorize_path: Optional[List] = None  # raw path strings

        self.context_length: int = 0
        self.complexity: int = 0

        self.concept_label = None  # vocab id of the concept
        self.frozen: bool = False  # whether this chunk has been "accepted"

    def __lt__(self, other):
        return id(self) < id(other)

    # ------------------------------------------------------------------
    @staticmethod
    def create_global_root() -> 'CompositeParseNode':
        """Create the sentinel root that owns all top-level parse nodes."""
        node = CompositeParseNode()
        node.is_global_root = True
        return node

    # ------------------------------------------------------------------
    @staticmethod
    def create_content_instance(left_node, right_node) -> dict:
        """
        Build the content-hierarchy instance from two children using
        single leaf pointers (Methodology 4.0).

        Layout (2 attributes):
          Attr 0 : left leaf pointer  {left_label_path: 1}
          Attr 1 : right leaf pointer {right_label_path: 1}

        label_path is a single int (context concept vocab ID).
        """
        left_lp = getattr(left_node, 'label_path', 0)
        right_lp = getattr(right_node, 'label_path', 0)

        # Fallback for nodes without label_path set
        if not left_lp:
            lbl = left_node.get_label()
            left_lp = next(iter(lbl.keys()), 0)
        if not right_lp:
            lbl = right_node.get_label()
            right_lp = next(iter(lbl.keys()), 0)

        return {
            0: {left_lp: 1} if left_lp != 0 else {0: 0},
            1: {right_lp: 1} if right_lp != 0 else {0: 0},
        }

    # ------------------------------------------------------------------
    @staticmethod
    def create_context_instance(left_node, right_node, context_length: int,
                                content_ref_id: int = None,
                                cplx_vocab_pair: tuple = (None, None),
                                bow: bool = False,
                                weighting: str = 'binary',
                                empty_weighting: bool = False,
                                chunk_context_before: list = None,
                                chunk_context_after: list = None) -> dict:
        """
        Build the context-hierarchy instance from two children.
        Attributes:
            0 .. context_length-1       : context_before (from left_node)
            context_length .. 2*ctx-1   : context_after  (from right_node)
            -2                          : complexity – stored as
                                          {C{X}_vid: 1} where X encodes the level
                                          (hidden, negative index)
            2*context_length            : content-ref (content hierarchy leaf
                                          concept vocab id, for generation)
                                          Visible (positive index) so it
                                          participates in Cobweb categorization.
                                          In BOW mode the index is 2 instead.

        Child expansion information is stored in the **content** hierarchy
        via path attributes (label_path of left/right children).  No
        child-ref hidden attributes are needed in the context instance.

        Parameters
        ----------
        cplx_vocab_pair : tuple of (id_to_value, value_to_id)
            Vocabulary lists used to look up or register ``C{X}`` complexity
            identifiers on demand.
        chunk_context_before : list | None
            When provided, overrides ``left_node.context_before`` with a list
            of ``{label_path: 1}`` dicts built from neighboring top-level
            nodes (chunk context mode).
        chunk_context_after : list | None
            Same as *chunk_context_before* but overrides
            ``right_node.context_after``.
        """
        ctx_inst: dict = {}

        left_c = getattr(left_node, "complexity", 1)
        right_c = getattr(right_node, "complexity", 1)
        complexity = max(left_c, right_c) + 1

        left_ctx_before = chunk_context_before if chunk_context_before is not None else (getattr(left_node, "context_before", None) or [])
        right_ctx_after = chunk_context_after if chunk_context_after is not None else (getattr(right_node, "context_after", None) or [])

        if bow:
            # BOW: collapse all before slots into key 0, all after into key 1.
            # Values are distance-weighted and summed per word.
            # EMPTYNULL (0) slots are omitted entirely.
            before_bag: dict = {}
            for j in range(context_length):
                if j < len(left_ctx_before) and left_ctx_before[j]:
                    weight = _context_weight(j, weighting)
                    for k in left_ctx_before[j]:
                        if k != 0:  # skip EMPTYNULL
                            before_bag[k] = before_bag.get(k, 0) + weight
            if before_bag:
                ctx_inst[0] = before_bag

            after_bag: dict = {}
            for j in range(context_length):
                if j < len(right_ctx_after) and right_ctx_after[j]:
                    weight = _context_weight(j, weighting)
                    for k in right_ctx_after[j]:
                        if k != 0:  # skip EMPTYNULL
                            after_bag[k] = after_bag.get(k, 0) + weight
            if after_bag:
                ctx_inst[1] = after_bag

            ctx_inst[-2] = {_get_or_register_cplx_vid(complexity, cplx_vocab_pair[0], cplx_vocab_pair[1]): 1}  # complexity hidden at -2
        else:
            _empty_val = 1 if empty_weighting else 0
            # Slot-per-position: one attribute per context window position.
            for j in range(context_length):
                if j < len(left_ctx_before) and left_ctx_before[j]:
                    ctx_inst[j] = {k: _context_weight(j, weighting) for k in left_ctx_before[j]}
                    ctx_inst[j][0] = 0
                else:
                    ctx_inst[j] = {0: _empty_val}

            for j in range(context_length):
                attr_key = context_length + j
                if j < len(right_ctx_after) and right_ctx_after[j]:
                    ctx_inst[attr_key] = {k: _context_weight(j, weighting) for k in right_ctx_after[j]}
                    ctx_inst[attr_key][0] = 0
                else:
                    ctx_inst[attr_key] = {0: _empty_val}

            ctx_inst[-2] = {_get_or_register_cplx_vid(complexity, cplx_vocab_pair[0], cplx_vocab_pair[1]): 1}

        # content-ref attribute: for composites this is the content
        # hierarchy leaf concept id; for primitives it's the word_id
        # (set separately in build_primitives).
        # Visible (positive index) so Cobweb includes it in entropy.
        if content_ref_id is not None:
            _content_ref_attr = 2 if bow else 2 * context_length
            ctx_inst[_content_ref_attr] = {content_ref_id: 1}

        return ctx_inst

    # ------------------------------------------------------------------
    @staticmethod
    def create_node(
        content_instance: dict,
        context_instance: dict,
        label: dict,
        categorize_path: list,
        position_idx: float,
        context_length: int,
        complexity: int,
        concept_label=None,
    ) -> 'CompositeParseNode':
        """Full factory: create a composite node with all attributes set."""
        node = CompositeParseNode()

        node.content_instance = content_instance
        node.context_instance = context_instance
        node.label = label
        node.categorize_path = categorize_path
        node.position_idx = position_idx
        node.context_length = context_length
        node.complexity = complexity
        node.concept_label = concept_label

        # derive before/after lists from context_instance for visualization
        node.context_before = []
        for j in range(context_length):
            node.context_before.append(dict(context_instance.get(j, {0: 0})))
        node.context_after = []
        for j in range(context_length):
            node.context_after.append(dict(context_instance.get(context_length + j, {0: 0})))

        return node

    # ------------------------------------------------------------------
    def set_parent(self, node: 'CompositeParseNode'):
        """Attach this node under *node*, updating both sides."""
        try:
            self.parent.children.remove((self.position_idx, self))
        except (AttributeError, ValueError):
            pass
        self.parent = node
        node.children.add((self.position_idx, self))

    # ------------------------------------------------------------------
    def get_context_instance(self) -> dict:
        """
        Return the context-focused instance (all int keys) suitable for
        CobwebDiscreteTree.ifit().
        """
        return dict(self.context_instance) if self.context_instance else {}

    # ------------------------------------------------------------------
    def get_content_instance(self) -> dict:
        """Return the content-focus instance (no metadata)."""
        return dict(self.content_instance) if self.content_instance else {}

    # ------------------------------------------------------------------
    def get_label(self) -> dict:
        """Weighted path dict for use as content in parent chunks."""
        return dict(self.label) if self.label else {}


# ---------------------------------------------------------------------------
# FiniteParseTree  (multi-hierarchy version)
# ---------------------------------------------------------------------------

class FiniteParseTree(object):
    """
    Short-term parse tree for a single sentence/window, backed by two
    Cobweb hierarchies (content + context) stored in a LongTermMemory object.

    The API mirrors parse.py's FiniteParseTree as closely as possible so that
    downstream tools (GUI, tests) can swap implementations.
    """

    def __init__(self, ltm: 'LongTermMemory', context_length: int = 3):
        self.ltm = ltm
        self.context_length = context_length

        self.global_root_node = CompositeParseNode.create_global_root()
        self.window: Optional[str] = None
        self.nodes: List = []

        self.action_log: List[dict] = []
        self._undo_stack: List[dict] = []

    # ---- helpers --------------------------------------------------------

    @property
    def id_to_value(self):
        return self.ltm.id_to_value

    @property
    def value_to_id(self):
        return self.ltm.value_to_id

    def _safe_lookup(self, idx):
        if isinstance(idx, str):
            return idx
        try:
            if idx is not None and isinstance(idx, int) and 0 <= idx < len(self.id_to_value):
                return self.id_to_value[idx]
        except Exception:
            pass
        return "None"

    def ctx_list(self, ctx: dict, draw_zeros: bool = False, max_size: int = 7) -> list:
        if not ctx:
            return []
        items = sorted(ctx.items(), key=lambda kv: (-kv[1], kv[0]))
        if not draw_zeros:
            non_zero = [(k, v) for k, v in items if k != 0]
            if non_zero:
                items = non_zero
            else:
                # All entries are EMPTYNULL (key 0) — show it explicitly
                # so the fractional weight is visible.
                items = [(k, v) for k, v in items]
        if len(items) > max_size:
            items = items[:max_size]
        return [{"key": self._safe_lookup(k), "val": float(v)} for k, v in items]

    def _find_root_child_by_index(self, position_idx):
        for wi, ch in self.global_root_node.children:
            if wi == position_idx:
                if isinstance(ch, CompositeParseNode) or getattr(ch, "stable", False):
                    return ch
        return None

    # ---- primitive layer ------------------------------------------------

    def build_primitives(self, window: str, threshold=0, debug: bool = False):
        """
        Tokenize *window* and create PrimitiveParseNode objects.
        Each primitive is categorized in the context hierarchy to obtain
        its label ({word_id: 1}) and label_path (multi-depth ancestor list).
        """
        self.window = window

        elements = re.findall(r"[\w']+|[.,!?;]", window)
        word_ids = [self.value_to_id[e] for e in elements]

        _bow = getattr(self.ltm, 'bow', False)
        _weighting = getattr(self.ltm, 'weighting', 'binary')
        _empty_wt = getattr(self.ltm, 'empty_weighting', False)
        _empty_val = lambda j: _context_weight(j, _weighting) if _empty_wt else 0
        _ctx_leaves = []  # saved for chunk_context pass 2
        for i, wid in enumerate(word_ids):
            # build sliding-window context instance for context hierarchy
            ctx_inst: dict = {}

            if _bow:
                # BOW: key 0 = before bag, key 1 = after bag, key 2 = complexity.
                # Distance-weighted, summed if repeated; EMPTYNULL omitted.
                before_bag: dict = {}
                for j in range(self.context_length):
                    src_idx = i - (j + 1)
                    if 0 <= src_idx < len(word_ids):
                        w = word_ids[src_idx]
                        before_bag[w] = before_bag.get(w, 0) + _context_weight(j, _weighting)
                if before_bag:
                    ctx_inst[0] = before_bag

                after_bag: dict = {}
                for j in range(self.context_length):
                    src_idx = i + (j + 1)
                    if 0 <= src_idx < len(word_ids):
                        w = word_ids[src_idx]
                        after_bag[w] = after_bag.get(w, 0) + _context_weight(j, _weighting)
                if after_bag:
                    ctx_inst[1] = after_bag

                ctx_inst[-2] = {_get_or_register_cplx_vid(1, self.id_to_value, self.value_to_id): 1}  # complexity hidden at -2
            else:
                # context_before
                for j in range(self.context_length):
                    src_idx = i - (j + 1)
                    if 0 <= src_idx < len(word_ids):
                        cw = word_ids[src_idx]
                        ctx_inst[j] = {cw: _context_weight(j, _weighting)}
                        ctx_inst[j][0] = 0
                    else:
                        ctx_inst[j] = {0: _empty_val(j)}

                # context_after
                for j in range(self.context_length):
                    src_idx = i + (j + 1)
                    attr_key = self.context_length + j
                    if 0 <= src_idx < len(word_ids):
                        cw = word_ids[src_idx]
                        ctx_inst[attr_key] = {cw: _context_weight(j, _weighting)}
                        ctx_inst[attr_key][0] = 0
                    else:
                        ctx_inst[attr_key] = {0: _empty_val(j)}

                # complexity = 1 for primitives – C1 identifier, count = 1
                ctx_inst[-2] = {_get_or_register_cplx_vid(1, self.id_to_value, self.value_to_id): 1}

            # word identity attribute – enables generation to recover
            # the actual word from a context hierarchy leaf.
            # Visible (positive index) so Cobweb includes it in entropy.
            _content_ref_attr = self.ltm.content_ref_attr
            ctx_inst[_content_ref_attr] = {wid: 1}

            # categorize in context hierarchy to get label path
            _cat_mode = getattr(self.ltm, 'categorization_mode', 'dfs')
            leaf_node, path_strs, node_path, depth_dists = _categorize(
                ctx_inst, self.ltm.context_hierarchy, mode=_cat_mode)
            _ctx_leaves.append(leaf_node)

            # Discrete single-identity label: primitive's identity is its word_id
            label = {wid: 1}

            # Single leaf pointer: the context leaf concept this node categorized into
            label_path = _build_label_from_ctx_leaf(leaf_node, self.value_to_id)
            # Register for LCA similarity in content hierarchy
            if label_path:
                self.ltm.content_hierarchy.register_ref_val(label_path, leaf_node)
                # Also register on context hierarchy in pass 1 so that the
                # iterative chunk_context loop can resolve these VIDs via LCA
                # from the very first iteration.
                if getattr(self.ltm, 'chunk_context', False):
                    self.ltm.context_hierarchy.register_ref_val(label_path, leaf_node)

            node = PrimitiveParseNode.create_node(ctx_inst, label, position_idx=i, word_id=wid)
            # For BFS modes, include ALL explored node hashes for viz
            if depth_dists is not None:
                node.context_path_hashes = [n.concept_hash() for n in node_path]
            else:
                node.context_path_hashes = [p[8:] if p.startswith("CONCEPT-") else p for p in path_strs]
            node.label_path = label_path

            # build context_before / context_after lists of dicts for visualization
            # (also used by create_context_instance for composites)
            cb = []
            for j in range(self.context_length):
                src_idx = i - (j + 1)
                if 0 <= src_idx < len(word_ids):
                    cb.append({word_ids[src_idx]: 1})
                else:
                    cb.append({0: _empty_val(j)})
            node.context_before = cb

            ca = []
            for j in range(self.context_length):
                src_idx = i + (j + 1)
                if 0 <= src_idx < len(word_ids):
                    ca.append({word_ids[src_idx]: 1})
                else:
                    ca.append({0: _empty_val(j)})
            node.context_after = ca

            # score
            score_data = _score_along_path(node_path, ctx_inst, self.ltm.context_hierarchy,
                                            eval_alpha=getattr(self.ltm, 'context_bl_alpha', None),
                                            instance_basic_level=getattr(self.ltm, 'instance_basic_level', False))
            node.score_data = score_data

            if threshold == "converge":
                node.stable = True
            else:
                # cost is basic_level_count: -1 (root=no evidence) or a
                # positive integer. Fallback -1 mirrors the "no evidence"
                # sentinel so that an absent key never spuriously passes.
                node.stable = score_data.get("cost", -1) > threshold

            node.set_parent(self.global_root_node)
            self.nodes.append(node)

        # -- chunk_context iterative passes: rebuild context instances using
        #    neighbors' label_paths, re-categorize, repeat until convergence
        #    or context_n_iterations is reached --
        if getattr(self.ltm, 'chunk_context', False):
            _max_iters = getattr(self.ltm, 'context_n_iterations', 0)
            primitive_nodes = [x[1] for x in self.global_root_node.children]
            _cat_mode = getattr(self.ltm, 'categorization_mode', 'dfs')
            _cref_attr = self.ltm.content_ref_attr
            _iteration = 0

            while True:
                _iteration += 1
                _any_changed = False

                for idx, node in enumerate(primitive_nodes):
                    # context_before from left neighbors' label_paths
                    cb = []
                    for j in range(self.context_length):
                        src = idx - (j + 1)
                        if 0 <= src:
                            lp = getattr(primitive_nodes[src], 'label_path', 0)
                            cb.append({lp: 1} if lp else {})
                        else:
                            cb.append({})
                    node.context_before = cb

                    # context_after from right neighbors' label_paths
                    ca = []
                    for j in range(self.context_length):
                        src = idx + (j + 1)
                        if src < len(primitive_nodes):
                            lp = getattr(primitive_nodes[src], 'label_path', 0)
                            ca.append({lp: 1} if lp else {})
                        else:
                            ca.append({})
                    node.context_after = ca

                    # Rebuild context_instance from chunk context
                    ctx_inst = {}
                    if _bow:
                        before_bag = {}
                        for j in range(self.context_length):
                            if j < len(cb) and cb[j]:
                                weight = _context_weight(j, _weighting)
                                for k in cb[j]:
                                    if k != 0:
                                        before_bag[k] = before_bag.get(k, 0) + weight
                        if before_bag:
                            ctx_inst[0] = before_bag

                        after_bag = {}
                        for j in range(self.context_length):
                            if j < len(ca) and ca[j]:
                                weight = _context_weight(j, _weighting)
                                for k in ca[j]:
                                    if k != 0:
                                        after_bag[k] = after_bag.get(k, 0) + weight
                        if after_bag:
                            ctx_inst[1] = after_bag

                        ctx_inst[-2] = {_get_or_register_cplx_vid(1, self.id_to_value, self.value_to_id): 1}
                    else:
                        _empty_v = 1 if _empty_wt else 0
                        for j in range(self.context_length):
                            if j < len(cb) and cb[j]:
                                ctx_inst[j] = {k: _context_weight(j, _weighting) for k in cb[j]}
                                ctx_inst[j][0] = 0
                            else:
                                ctx_inst[j] = {0: _empty_v}
                        for j in range(self.context_length):
                            attr_key = self.context_length + j
                            if j < len(ca) and ca[j]:
                                ctx_inst[attr_key] = {k: _context_weight(j, _weighting) for k in ca[j]}
                                ctx_inst[attr_key][0] = 0
                            else:
                                ctx_inst[attr_key] = {0: _empty_v}
                        ctx_inst[-2] = {_get_or_register_cplx_vid(1, self.id_to_value, self.value_to_id): 1}

                    # content-ref = label_path (context hierarchy leaf pointer)
                    ctx_inst[_cref_attr] = {node.label_path: 1} if node.label_path else {node.word_id: 1}

                    node.context_instance = ctx_inst

                    # Re-categorize (non-modifying) with the updated context instance
                    leaf_node, path_strs, node_path, depth_dists = _categorize(
                        ctx_inst, self.ltm.context_hierarchy, mode=_cat_mode)
                    _ctx_leaves[idx] = leaf_node
                    new_label_path = _build_label_from_ctx_leaf(leaf_node, self.value_to_id)

                    if new_label_path != node.label_path:
                        _any_changed = True
                        node.label_path = new_label_path
                        # Update content-ref in the instance to reflect new label
                        ctx_inst[_cref_attr] = {new_label_path: 1} if new_label_path else {node.word_id: 1}
                        node.context_instance = ctx_inst

                    # Register VID → node on context hierarchy immediately so
                    # subsequent nodes in this pass can resolve it via LCA.
                    if node.label_path and leaf_node is not None:
                        self.ltm.context_hierarchy.register_ref_val(
                            node.label_path, leaf_node)

                # Check termination: converged or hit iteration cap
                if not _any_changed:
                    break
                if _max_iters > 0 and _iteration >= _max_iters:
                    break

            # Register final label_paths on context hierarchy for self-ref LCA
            for idx, node in enumerate(primitive_nodes):
                if node.label_path and idx < len(_ctx_leaves):
                    self.ltm.content_hierarchy.register_ref_val(node.label_path, _ctx_leaves[idx])
                    self.ltm.context_hierarchy.register_ref_val(
                        node.label_path, _ctx_leaves[idx]
                    )

    # ---- pair enumeration -----------------------------------------------

    def get_parentless_pairs(self) -> List[dict]:
        """
        Return consecutive pairs of root-level children (left-to-right).
        """
        pairs = []
        parentless = [x[1] for x in self.global_root_node.children]
        for i in range(len(parentless) - 1):
            left = parentless[i]
            right = parentless[i + 1]

            # extract representative key for labelling
            def _first_key(node):
                if isinstance(node, PrimitiveParseNode):
                    return node.word_id
                elif node.content_instance:
                    cl = node.content_instance.get(0, {})
                    return next(iter(cl.keys()), None) if cl else None
                return None

            pairs.append({
                "left_word_index": left.position_idx,
                "right_word_index": right.position_idx,
                "left_title": left.title,
                "right_title": right.title,
                "left_label": self._safe_lookup(_first_key(left)),
                "right_label": self._safe_lookup(_first_key(right)),
            })
        return pairs

    # ---- evaluation -----------------------------------------------------

    def evaluate_pair(self, left_word_index, right_word_index, debug=False,
                      _basic_cache=None, _child_ctx_cache=None) -> dict:
        """
        Evaluate merging two root-level children.
        Builds *both* content and context instances, categorizes each in its
        respective hierarchy, and returns scoring data.

        THIS IS WHERE THE MAGIC HAPPENS!!

        _basic_cache: optional dict for caching get_basic() results.
        _child_ctx_cache: optional dict for caching per-child context scores.
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        # Step 1: Build and categorize context WITHOUT -1 content-ref
        # (matching add_parse_tree behavior)

        # When chunk_context is enabled, build context from top-level
        # nodes' label_paths instead of children's sliding-window context.
        _chunk_cb = None
        _chunk_ca = None
        if getattr(self.ltm, 'chunk_context', False):
            parentless = [x[1] for x in self.global_root_node.children]
            left_pos = next((i for i, n in enumerate(parentless) if n is left_node), None)
            right_pos = next((i for i, n in enumerate(parentless) if n is right_node), None)
            if left_pos is not None and right_pos is not None:
                _chunk_cb = []
                for j in range(self.context_length):
                    src = left_pos - (j + 1)
                    if 0 <= src:
                        lp = getattr(parentless[src], 'label_path', 0)
                        _chunk_cb.append({lp: 1} if lp else {})
                    else:
                        _chunk_cb.append({})
                _chunk_ca = []
                for j in range(self.context_length):
                    src = right_pos + (j + 1)
                    if src < len(parentless):
                        lp = getattr(parentless[src], 'label_path', 0)
                        _chunk_ca.append({lp: 1} if lp else {})
                    else:
                        _chunk_ca.append({})

        context_inst = CompositeParseNode.create_context_instance(
            left_node, right_node, self.context_length,
            content_ref_id=None,  # no -1 during categorization
            cplx_vocab_pair=(self.id_to_value, self.value_to_id),
            bow=getattr(self.ltm, 'bow', False),
            weighting=getattr(self.ltm, 'weighting', 'binary'),
            empty_weighting=getattr(self.ltm, 'empty_weighting', False),
            chunk_context_before=_chunk_cb,
            chunk_context_after=_chunk_ca,
        )

        # categorize in context hierarchy (for identity / label)
        _cat_mode = getattr(self.ltm, 'categorization_mode', 'dfs')
        ctx_leaf, ctx_path, ctx_node_path, ctx_depth_dists = _categorize(
            context_inst, self.ltm.context_hierarchy, mode=_cat_mode)

        # Step 2: Build and categorize content (for scoring)
        content_inst = CompositeParseNode.create_content_instance(left_node, right_node)
        cnt_leaf, cnt_path, cnt_node_path, cnt_depth_dists = _categorize(
            content_inst, self.ltm.content_hierarchy, mode=_cat_mode)

        content_score_data = _score_along_path(cnt_node_path, content_inst, self.ltm.content_hierarchy,
                                               eval_alpha=getattr(self.ltm, 'content_bl_alpha', None),
                                               _basic_cache=_basic_cache,
                                               instance_basic_level=getattr(self.ltm, 'instance_basic_level', False))
        context_score_data = _score_along_path(ctx_node_path, context_inst, self.ltm.context_hierarchy,
                                               eval_alpha=getattr(self.ltm, 'context_bl_alpha', None),
                                               _basic_cache=_basic_cache,
                                               instance_basic_level=getattr(self.ltm, 'instance_basic_level', False))

        # IMPORTANT STUFF HERE!!!
        score = content_score_data["cost"]

        # Score each child's own context instance individually so the UI
        # can display per-chunk context scores alongside the merged score.
        _cref_attr = self.ltm.content_ref_attr
        def _score_child_ctx(child):
            child_id = id(child)
            if _child_ctx_cache is not None and child_id in _child_ctx_cache:
                return _child_ctx_cache[child_id]
            ctx = child.get_context_instance()
            ctx.pop(_cref_attr, None)  # strip content-ref, matches add_parse_tree
            if not ctx:
                result = {}
            else:
                _, _, _path, _ = _categorize(ctx, self.ltm.context_hierarchy, mode=_cat_mode)
                result = _score_along_path(_path, ctx, self.ltm.context_hierarchy,
                                           eval_alpha=getattr(self.ltm, 'context_bl_alpha', None),
                                           _basic_cache=_basic_cache,
                                           instance_basic_level=getattr(self.ltm, 'instance_basic_level', False))
            if _child_ctx_cache is not None:
                _child_ctx_cache[child_id] = result
            return result

        left_ctx_score_data  = _score_child_ctx(left_node)
        right_ctx_score_data = _score_child_ctx(right_node)

        if debug:
            print(f"Score for pair: {score:.4f}")

        # build label (weighted path from context hierarchy)
        ctx_path_ids = []
        for pstr in ctx_path:
            vid = self.value_to_id.get(pstr)
            if vid is not None:
                ctx_path_ids.append(vid)

        ctx_hash = ctx_leaf.concept_hash() if ctx_leaf else "unknown"
        ctx_concept_id = self.value_to_id.get(f"CONCEPT-{ctx_hash}")

        # For BFS modes, path hashes include ALL explored nodes for viz
        _is_bfs = ctx_depth_dists is not None
        if _is_bfs:
            content_path_hashes = [n.concept_hash() for n in cnt_node_path]
            context_path_hashes = [n.concept_hash() for n in ctx_node_path]
        else:
            content_path_hashes = [p[8:] if p.startswith("CONCEPT-") else p for p in cnt_path]
            context_path_hashes = [p[8:] if p.startswith("CONCEPT-") else p for p in ctx_path]

        res = {
            "content_inst": content_inst,
            "context_inst": context_inst,
            "categorize_path": ctx_path_ids,
            "candidate_concept_hash": ctx_hash,
            "candidate_concept_id": ctx_concept_id,
            "score": score,
            "left_word_index": left_word_index,
            "right_word_index": right_word_index,
            "left_title": left_node.title,
            "right_title": right_node.title,
            # Path hashes for path visualization
            "content_path_hashes": content_path_hashes,
            "context_path_hashes": context_path_hashes,
            "content_score_data": content_score_data,
            "context_score_data": context_score_data,
            # Per-child context scores (visualized in the evaluate-pair modal)
            "left_context_score_data": left_ctx_score_data,
            "right_context_score_data": right_ctx_score_data,
            "categorization_mode": _cat_mode,
        }

        if debug:
            _lc = res.get("left_context_score_data", {})
            _rc = res.get("right_context_score_data", {})
            _cnt = res.get("content_score_data", {})
            _ctx = res.get("context_score_data", {})
            print(
                f"  pair ({res['left_word_index']}, {res['right_word_index']}):\n"
                f"    left  ctx  | bl_count={_lc.get('basic_level_count', 'N/A')!s:>6}  tree_lp={_lc.get('tree_log_prob', float('nan')):.4f}\n"
                f"    right ctx  | bl_count={_rc.get('basic_level_count', 'N/A')!s:>6}  tree_lp={_rc.get('tree_log_prob', float('nan')):.4f}\n"
                f"    cand  ctx  | bl_count={_ctx.get('basic_level_count', 'N/A')!s:>6}  tree_lp={_ctx.get('tree_log_prob', float('nan')):.4f}\n"
                f"    cand  cnt  | bl_count={_cnt.get('basic_level_count', 'N/A')!s:>6}  cnt_tree_lp={_cnt.get('tree_log_prob', float('nan')):.4f} cxt_tree_lp={_ctx.get('tree_log_prob', float('nan')):.4f}"
            )

        return res

    # ---- application ----------------------------------------------------

    def apply_candidate(self, left_word_index, right_word_index, frozen: bool = True) -> dict:
        """
        Apply a candidate merge: create a CompositeParseNode and re-parent children.

        Order matches add_parse_tree: context-first WITHOUT -1, then content.
        The -1 content-ref will be written later in add_parse_tree step 4.

        If *frozen* is True the chunk is considered accepted and will eventually
        be added to both hierarchies; otherwise only the content hierarchy.
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        # Step 1: Build and categorize context WITHOUT -1
        # (matching add_parse_tree step 1)

        # When chunk_context is enabled, build context from top-level
        # nodes' label_paths.
        _chunk_cb = None
        _chunk_ca = None
        if getattr(self.ltm, 'chunk_context', False):
            parentless = [x[1] for x in self.global_root_node.children]
            left_pos = next((i for i, n in enumerate(parentless) if n is left_node), None)
            right_pos = next((i for i, n in enumerate(parentless) if n is right_node), None)
            if left_pos is not None and right_pos is not None:
                _chunk_cb = []
                for j in range(self.context_length):
                    src = left_pos - (j + 1)
                    if 0 <= src:
                        lp = getattr(parentless[src], 'label_path', 0)
                        _chunk_cb.append({lp: 1} if lp else {})
                    else:
                        _chunk_cb.append({})
                _chunk_ca = []
                for j in range(self.context_length):
                    src = right_pos + (j + 1)
                    if src < len(parentless):
                        lp = getattr(parentless[src], 'label_path', 0)
                        _chunk_ca.append({lp: 1} if lp else {})
                    else:
                        _chunk_ca.append({})

        context_inst = CompositeParseNode.create_context_instance(
            left_node, right_node, self.context_length,
            content_ref_id=None,  # no -1 during categorization
            cplx_vocab_pair=(self.id_to_value, self.value_to_id),
            bow=getattr(self.ltm, 'bow', False),
            weighting=getattr(self.ltm, 'weighting', 'binary'),
            empty_weighting=getattr(self.ltm, 'empty_weighting', False),
            chunk_context_before=_chunk_cb,
            chunk_context_after=_chunk_ca,
        )

        # categorize in context hierarchy
        _cat_mode = getattr(self.ltm, 'categorization_mode', 'dfs')
        ctx_leaf, ctx_path, _, ctx_depth_dists = _categorize(
            context_inst, self.ltm.context_hierarchy, mode=_cat_mode)

        # Step 2: Build content instance
        # (matching add_parse_tree step 3, but we don't categorize here —
        # add_parse_tree will rebuild content instances from refreshed labels)
        content_inst = CompositeParseNode.create_content_instance(left_node, right_node)

        left_c = getattr(left_node, "complexity", 1)
        right_c = getattr(right_node, "complexity", 1)
        complexity = max(left_c, right_c) + 1

        ctx_hash = ctx_leaf.concept_hash() if ctx_leaf else "unknown"
        concept_label_str = f"CONCEPT-{ctx_hash}"
        self.ltm.add_to_vocab(concept_label_str)
        concept_label = self.value_to_id.get(concept_label_str)

        # Discrete single-identity label
        label = {concept_label: 1} if concept_label is not None else {0: 1}
        path_ids = list(label.keys())

        # Single leaf pointer
        _label_path = _build_label_from_ctx_leaf(ctx_leaf, self.value_to_id)
        # Register for LCA similarity in content hierarchy
        if _label_path:
            self.ltm.content_hierarchy.register_ref_val(_label_path, ctx_leaf)
            # Also register on context hierarchy when chunk_context (self-ref)
            if getattr(self.ltm, 'chunk_context', False):
                self.ltm.context_hierarchy.register_ref_val(_label_path, ctx_leaf)

        new_node = CompositeParseNode.create_node(
            content_instance=content_inst,
            context_instance=context_inst,
            label=label,
            categorize_path=path_ids,
            position_idx=0.5 * (left_node.position_idx + right_node.position_idx),
            context_length=self.context_length,
            complexity=complexity,
            concept_label=concept_label,
        )
        new_node.label_path = _label_path
        new_node.frozen = frozen

        self.nodes.append(new_node)
        new_node.set_parent(self.global_root_node)
        left_node.set_parent(new_node)
        right_node.set_parent(new_node)

        # undo support
        undo_entry = {
            "action": "apply_candidate",
            "added_node_title": new_node.title,
            "added_node_word_index": new_node.position_idx,
            "left_word_index": left_node.position_idx,
            "right_word_index": right_node.position_idx,
            "timestamp": time.time(),
        }
        self._undo_stack.append(undo_entry)

        log_entry = {
            "timestamp": time.time(),
            "type": "apply_candidate",
            "description": (
                f"Applied chunk {left_node.title} ({left_node.position_idx}) "
                f"+ {right_node.title} ({right_node.position_idx}) "
                f"-> CONCEPT-{ctx_hash}"
            ),
            "payload": {
                "left": {"title": left_node.title, "position_idx": left_node.position_idx},
                "right": {"title": right_node.title, "position_idx": right_node.position_idx},
                "new_node": {"title": new_node.title, "position_idx": new_node.position_idx, "concept_id": concept_label},
            },
        }
        self.action_log.append(log_entry)

        return {
            "ok": True,
            "added_node": {
                "title": new_node.title,
                "position_idx": new_node.position_idx,
                "concept_id": concept_label,
            },
            "action_log_entry": log_entry,
        }

    # ---- undo -----------------------------------------------------------

    def undo(self) -> dict:
        if not self._undo_stack:
            return {"ok": False, "reason": "Nothing to undo"}

        entry = self._undo_stack.pop()
        if entry["action"] != "apply_candidate":
            return {"ok": False, "reason": "Unsupported undo action"}

        added_title = entry["added_node_title"]
        added_node = next((n for n in self.nodes if n.title == added_title), None)
        if added_node is None:
            return {"ok": False, "reason": "Added node not found"}

        # re-parent children to global root
        try:
            for wi, ch in list(added_node.children):
                ch.set_parent(self.global_root_node)
        except Exception:
            pass

        # remove added node
        try:
            self.global_root_node.children.remove((added_node.position_idx, added_node))
            self.nodes.remove(added_node)
        except ValueError:
            pass

        # trim action log
        for i in range(len(self.action_log) - 1, -1, -1):
            if (
                self.action_log[i]["type"] == "apply_candidate"
                and self.action_log[i]["payload"]["new_node"]["title"] == added_title
            ):
                self.action_log.pop(i)
                break

        return {"ok": True, "undone": added_title}

    # ---- build (automatic) ---------------------------------------------

    def build(self, window: str, end_behavior="converge", debug=False) -> bool:
        """
        Fully automatic parse: build primitives then greedily merge best pairs.

        Selection strategy (two-stage):
            1. Threshold gate: keep only candidates whose basic-level count
               exceeds *end_behavior* (numeric) or is > -1 (``"converge"``).
               A count of -1 means the basic-level node collapsed to the root,
               i.e. there is no real evidence for the chunk.
            2. Tie-break by content ``tree_log_prob``: among all candidates
               that pass the gate, the one with the highest tree log-probability
               is chosen.  If no candidates pass the gate, the loop terminates.
        """
        self.window = window
        self.build_primitives(window, threshold=end_behavior, debug=debug)

        # Determine the numeric count threshold for stage-1 gating.
        # "converge" means accept any candidate with at least some evidence
        # (basic_level_count > -1).  A numeric end_behavior is used directly.
        count_threshold = -1 if end_behavior == "converge" else end_behavior

        # --- pair score cache: avoid re-evaluating unchanged pairs ---
        _pair_cache: dict = {}  # (left_idx, right_idx) → (sum_tree_lps, res)
        # --- caches for expensive sub-computations (trees are read-only during build) ---
        _basic_cache: dict = {}   # (tree_id, leaf_hash, alpha) → basic_level_node
        _child_ctx_cache: dict = {}  # child_id → context score dict

        while True:
            pairs = self.get_parentless_pairs()
            if not pairs:
                break

            candidates = []
            for p in pairs:
                pair_key = (p["left_word_index"], p["right_word_index"])

                # Use cached result if available
                if pair_key in _pair_cache:
                    cached_lps, cached_res = _pair_cache[pair_key]
                    if cached_lps is not None:  # passed threshold last time
                        candidates.append((cached_lps, cached_res))
                    continue

                try:
                    res = self.evaluate_pair(p["left_word_index"], p["right_word_index"], debug=debug,
                                            _basic_cache=_basic_cache, _child_ctx_cache=_child_ctx_cache)
                except Exception as e:
                    if debug:
                        print(f"evaluate_pair failed: {e}")
                    _pair_cache[pair_key] = (None, None)
                    continue

                basic_level_count = res.get("score", -float("inf"))  # == content_score_data["cost"]
                cnt_tree_log_prob = res.get("content_score_data", {}).get("tree_log_prob", -float("inf"))
                sum_tree_lps = cnt_tree_log_prob

                # Stage 1: threshold gate
                if basic_level_count <= count_threshold:
                    _pair_cache[pair_key] = (None, None)
                    continue

                _pair_cache[pair_key] = (sum_tree_lps, res)
                candidates.append((sum_tree_lps, res)) # SECONDARY SCORE

            if not candidates:
                break

            # Stage 2: pick the candidate with the highest tree_log_prob
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0][1]

            chosen_left = chosen["left_word_index"]
            chosen_right = chosen["right_word_index"]

            try:
                self.apply_candidate(
                    chosen_left,
                    chosen_right,
                    frozen=True,
                )
            except Exception as e:
                if debug:
                    print(f"apply_candidate failed: {e}")
                break

            # Invalidate cache entries involving the merged nodes.
            # After merge, the new composite replaces (chosen_left, chosen_right)
            # and adjacency changes only for their neighbors.
            stale_keys = [k for k in _pair_cache
                          if chosen_left in k or chosen_right in k]
            for k in stale_keys:
                del _pair_cache[k]

            if end_behavior == "converge" and len(self.global_root_node.children) <= 1:
                break

        return True

    # ---- instance collection -------------------------------------------

    def get_parsed_instances(self) -> Tuple[List[dict], List[dict]]:
        """
        Return (content_instances, context_instances) for all nodes in the tree.
        Primitives yield context instances only (they have no content_instance).
        """
        content_insts: List[dict] = []
        context_insts: List[dict] = []

        def dfs(node):
            if isinstance(node, PrimitiveParseNode):
                context_insts.append(node.get_context_instance())
            elif isinstance(node, CompositeParseNode) and not node.is_global_root:
                content_insts.append(node.get_content_instance())
                context_insts.append(node.get_context_instance())
            for _, ch in getattr(node, "children", []):
                dfs(ch)

        for _, ch in self.global_root_node.children:
            dfs(ch)

        return content_insts, context_insts

    def get_unparsed_instances(self) -> Tuple[List[dict], List[dict]]:
        """
        Return (content_instances, context_instances) for candidate pairs
        that were NOT merged.
        """
        content_insts: List[dict] = []
        context_insts: List[dict] = []
        pairs = self.get_parentless_pairs()

        for p in pairs:
            left = self._find_root_child_by_index(p["left_word_index"])
            right = self._find_root_child_by_index(p["right_word_index"])
            if left and right:
                ci = CompositeParseNode.create_content_instance(left, right)
                content_insts.append(ci)

        return content_insts, context_insts

    def get_all_instances(self) -> Tuple[List[dict], List[dict]]:
        """
        Combine parsed and unparsed instances.
        Per multi-hierarchy theory:
          - ALL candidate content instances go to the content hierarchy
          - Only frozen/accepted context instances go to the context hierarchy
        """
        p_content, p_context = self.get_parsed_instances()
        u_content, u_context = self.get_unparsed_instances()
        return p_content + u_content, p_context + u_context

    # ---- visualization --------------------------------------------------

    def _draw_node_to_dict(self, node, draw_zeros=False) -> dict:
        """Build a JSON-serialisable dict for D3 parse-tree rendering.

        Returns a dict with content and context data split into separate
        left/right and before/after groups for side-by-side display:
        ``content_rows`` (primitives), ``content_left_rows``,
        ``content_right_rows``, ``context_before_rows``,
        ``context_after_rows``, ``context_other_rows``.
        """

        def _rows_from_ctx(ctx: dict, attr_name: str) -> list:
            """Turn one context-instance attribute dict into flat rows."""
            items = self.ctx_list(ctx, draw_zeros)
            if not items:
                return [{"attr": attr_name, "val": "empty", "count": ""}]
            rows = []
            for idx_j, kv in enumerate(items):
                rows.append({
                    "attr": attr_name if idx_j == 0 else "",
                    "val": kv["key"],
                    "count": f"{kv['val']:.2f}",
                })
            return rows

        def _score_rows_from_node(n):
            """Build score annotation rows if the node has log-prob or basic-count attributes."""
            rows = []
            clp = getattr(n, '_content_log_prob', None)
            xlp = getattr(n, '_context_log_prob', None)
            cbc = getattr(n, '_content_basic_count', None)
            xbc = getattr(n, '_context_basic_count', None)
            if clp is not None:
                rows.append({"attr": "CntLP", "val": f"{clp:.3f}", "count": ""})
            if xlp is not None:
                rows.append({"attr": "CtxLP", "val": f"{xlp:.3f}", "count": ""})
            if cbc is not None:
                rows.append({"attr": "CntBC", "val": f"{cbc:.0f}", "count": ""})
            if xbc is not None:
                rows.append({"attr": "CtxBC", "val": f"{xbc:.0f}", "count": ""})
            return rows

        if isinstance(node, PrimitiveParseNode):
            # Content rows: just the word identity
            content_rows = [{"attr": "Word", "val": self._safe_lookup(node.word_id), "count": "1.00"}]

            # Context rows: split into before / after / other
            context_before_rows = []
            context_after_rows = []
            context_other_rows = []
            if node.context_instance:
                cl = self.context_length
                _bow = getattr(self.ltm, 'bow', False)
                _cref = 2 * cl
                for attr_key in sorted(k for k in node.context_instance if k >= 0 and k != _cref):
                    if _bow:
                        if attr_key == 0:
                            context_before_rows.extend(_rows_from_ctx(node.context_instance[attr_key], "CtxBefore"))
                        elif attr_key == 1:
                            context_after_rows.extend(_rows_from_ctx(node.context_instance[attr_key], "CtxAfter"))
                        else:
                            context_other_rows.extend(_rows_from_ctx(node.context_instance[attr_key], f"Attr{attr_key}"))
                    else:
                        if attr_key < cl:
                            hdr = f"CtxBefore{attr_key}"
                            context_before_rows.extend(_rows_from_ctx(node.context_instance[attr_key], hdr))
                        elif attr_key < 2 * cl:
                            hdr = f"CtxAfter{attr_key - cl}"
                            context_after_rows.extend(_rows_from_ctx(node.context_instance[attr_key], hdr))
                        else:
                            context_other_rows.extend(_rows_from_ctx(node.context_instance[attr_key], f"Attr{attr_key}"))
                if -2 in node.context_instance:
                    context_other_rows.extend(_rows_from_ctx(node.context_instance[-2], "Complexity"))
                _cref = 2 * cl
                if _cref in node.context_instance:
                    context_other_rows.extend(_rows_from_ctx(node.context_instance[_cref], "ContentRef"))

            return {
                "title": node.title,
                "content_rows": content_rows,
                "content_left_rows": [],
                "content_right_rows": [],
                "context_before_rows": context_before_rows,
                "context_after_rows": context_after_rows,
                "context_other_rows": context_other_rows,
                "score_rows": _score_rows_from_node(node),
                "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
            }

        elif isinstance(node, CompositeParseNode):
            if node.is_global_root:
                return {
                    "title": "ROOT",
                    "content_rows": [],
                    "content_left_rows": [],
                    "content_right_rows": [],
                    "context_before_rows": [],
                    "context_after_rows": [],
                    "context_other_rows": [],
                    "score_rows": [],
                    "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
                }

            # Content rows: left (attr 0) and right (attr 1)
            content_left_rows = []
            content_right_rows = []
            if node.content_instance:
                content_left_rows.extend(_rows_from_ctx(node.content_instance.get(0, {}), "Left"))
                content_right_rows.extend(_rows_from_ctx(node.content_instance.get(1, {}), "Right"))

            # Context rows: split into before / after / other
            context_before_rows = []
            context_after_rows = []
            context_other_rows = []
            if node.context_instance:
                cl = self.context_length
                _bow = getattr(self.ltm, 'bow', False)
                _cref = 2 * cl
                for attr_key in sorted(k for k in node.context_instance if k >= 0 and k != _cref):
                    if _bow:
                        if attr_key == 0:
                            context_before_rows.extend(_rows_from_ctx(node.context_instance[attr_key], "CtxBefore"))
                        elif attr_key == 1:
                            context_after_rows.extend(_rows_from_ctx(node.context_instance[attr_key], "CtxAfter"))
                        else:
                            context_other_rows.extend(_rows_from_ctx(node.context_instance[attr_key], f"Attr{attr_key}"))
                    else:
                        if attr_key < cl:
                            hdr = f"CtxBefore{attr_key}"
                            context_before_rows.extend(_rows_from_ctx(node.context_instance[attr_key], hdr))
                        elif attr_key < 2 * cl:
                            hdr = f"CtxAfter{attr_key - cl}"
                            context_after_rows.extend(_rows_from_ctx(node.context_instance[attr_key], hdr))
                        else:
                            context_other_rows.extend(_rows_from_ctx(node.context_instance[attr_key], f"Attr{attr_key}"))
                if -2 in node.context_instance:
                    context_other_rows.extend(_rows_from_ctx(node.context_instance[-2], "Complexity"))
                _cref2 = 2 * cl
                if _cref2 in node.context_instance:
                    context_other_rows.extend(_rows_from_ctx(node.context_instance[_cref2], "ContentRef"))

            return {
                "title": node.title,
                "content_rows": [],
                "content_left_rows": content_left_rows,
                "content_right_rows": content_right_rows,
                "context_before_rows": context_before_rows,
                "context_after_rows": context_after_rows,
                "context_other_rows": context_other_rows,
                "score_rows": _score_rows_from_node(node),
                "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
            }
        else:
            raise TypeError(f"Unknown node type {type(node)}")

    def _draw_tree_to_json(self) -> dict:
        return self._draw_node_to_dict(self.global_root_node)

    def visualize(self, out_base="parse_tree", render_png=True):
        d3_json = json.dumps(self._draw_tree_to_json())
        html = self._build_html(d3_json)

        html_path = f"{out_base}.html"
        png_path = f"{out_base}.png"
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        if render_png:
            asyncio.run(self._html_to_png(html_path, png_path))
            return html_path, png_path
        return html_path

    async def _html_to_png(self, html_path, png_path):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("file://" + os.path.abspath(html_path))
            await page.wait_for_selector("#tree svg")
            await page.evaluate("""
                () => new Promise(resolve => {
                    requestAnimationFrame(() => requestAnimationFrame(resolve));
                })
            """)
            bb = await page.evaluate("""
                () => {
                    const c = document.querySelector('#tree-container');
                    return { width: Math.ceil(c.scrollWidth)+20, height: Math.ceil(c.scrollHeight)+20 };
                }
            """)
            await page.set_viewport_size({"width": bb["width"], "height": bb["height"]})
            svg_elem = await page.query_selector("#tree svg")
            await svg_elem.screenshot(path=png_path, scale="css")
            await browser.close()

    def _build_html(self, d3_data_json, node_w=600, node_h=130, h_gap=80, v_gap=150):
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Parse Tree (Multi-Hierarchy)</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
#tree-container {{ display: inline-block; }}
.link {{ fill: none; stroke: #9aa1a9; stroke-width: 1.5px; }}
.node-box {{ stroke: #444; fill: #fff; rx: 8; ry: 8; filter: drop-shadow(1px 2px 2px rgba(0,0,0,0.15)); }}
.node-fo table {{ border-collapse: collapse; font-size: 12px; margin: 4px 0; }}
.node-fo th, .node-fo td {{ border: 1px solid #888; padding: 2px 6px; }}
.node-fo th {{ background: #f3f5f7; font-weight: 600; }}
.section-header {{ font-weight: 700; font-size: 13px; color: #333; margin: 8px 0 4px 0; border-bottom: 1px solid #ccc; padding-bottom: 2px; }}
.side-by-side {{ display: flex; gap: 8px; align-items: flex-start; }}
.side-by-side > div {{ flex: 1; min-width: 0; }}
.sub-title {{ font-weight: 600; font-size: 11px; color: #555; margin-bottom: 2px; }}
</style>
</head>
<body>
<div id="tree-container"><div id="tree"></div></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const data = {d3_data_json};
const nodeW={node_w}, nodeH={node_h}, hGap={h_gap}, vGap={v_gap};

const root = d3.hierarchy(data);
const layout = d3.tree()
  .nodeSize([nodeW+hGap, nodeH+vGap])
  .separation((a,b) => a.parent===b.parent ? 1.0 : 1.4);
layout(root);

const svg = d3.select("#tree").append("svg").attr("width",1).attr("height",1);
const g = svg.append("g");

const linkGen = d3.linkVertical().x(d=>d.x).y(d=>d.y);
const link = g.selectAll("path.link").data(root.links()).join("path")
  .attr("class","link").attr("d",linkGen);

const node = g.selectAll("g.node").data(root.descendants()).join("g")
  .attr("transform", d=>`translate(${{d.x}},${{d.y}})`);

node.append("rect").attr("class","node-box")
  .attr("x",-nodeW/2).attr("y",0).attr("width",nodeW).attr("height",nodeH);

node.append("foreignObject").attr("class","node-fo")
  .attr("x",-nodeW/2+6).attr("y",6).attr("width",nodeW-12).attr("height",1000)
  .html(d => nodeHTML(d.data));

// measure actual content heights
node.selectAll("foreignObject").each(function(d){{
  const fo=d3.select(this), div=fo.select("div").node();
  const h=div.getBoundingClientRect().height+12;
  d._nodeHeight = h;
  fo.attr("height",h);
  d3.select(this.parentNode).select("rect").attr("height",h+12);
}});

// compute per-depth max height and reposition rows
const depthMaxHeight = new Map();
root.each(d => {{
  const h = d._nodeHeight || nodeH;
  const existing = depthMaxHeight.get(d.depth) || 0;
  if (h > existing) depthMaxHeight.set(d.depth, h);
}});
const maxDepth = Math.max(...Array.from(depthMaxHeight.keys()));
const depthOffsets = [];
for (let i = 0; i <= maxDepth; i++) {{
  const prev = i === 0 ? 0 : depthOffsets[i-1] + (depthMaxHeight.get(i-1)||nodeH) + vGap;
  depthOffsets.push(prev);
}}
root.each(d => {{ d.y = depthOffsets[d.depth]; }});
node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
link.attr("d", linkGen);

// fit viewBox
let x0=Infinity, x1=-Infinity, y0=Infinity, y1=-Infinity;
root.each(d => {{
  const halfW = nodeW/2, h = d._nodeHeight || nodeH;
  if (d.x-halfW-30 < x0) x0 = d.x-halfW-30;
  if (d.x+halfW+30 > x1) x1 = d.x+halfW+30;
  if (d.y-30 < y0) y0 = d.y-30;
  if (d.y+h+30 > y1) y1 = d.y+h+30;
}});
const width=x1-x0, height=y1-y0;
svg.attr("width",width).attr("height",height)
  .attr("viewBox",[x0,y0,width,height].join(" "));

function threeColTable(title, rows) {{
  if (!rows || rows.length === 0) return "";
  const body = rows.map(r =>
    `<tr><td>${{r.attr}}</td><td>${{r.val}}</td><td>${{r.count}}</td></tr>`
  ).join("");
  return `<div class="sub-title">${{title}}</div>
    <table>
      <tr><th>Attr</th><th>Value</th><th>Count</th></tr>
      ${{body}}
    </table>`;
}}

function nodeHTML(d){{
  // --- Content section ---
  let contentHTML = "";
  const hasLeft = d.content_left_rows && d.content_left_rows.length > 0;
  const hasRight = d.content_right_rows && d.content_right_rows.length > 0;
  const hasSingle = d.content_rows && d.content_rows.length > 0;

  if (hasLeft || hasRight) {{
    // Composite node: left and right side by side
    contentHTML = `<div class="section-header">Content</div>
      <div class="side-by-side">
        <div>${{threeColTable("Left", d.content_left_rows)}}</div>
        <div>${{threeColTable("Right", d.content_right_rows)}}</div>
      </div>`;
  }} else if (hasSingle) {{
    // Primitive node: single content table
    contentHTML = `<div class="section-header">Content</div>${{threeColTable("", d.content_rows)}}`;
  }}

  // --- Context section ---
  let contextHTML = "";
  const hasBefore = d.context_before_rows && d.context_before_rows.length > 0;
  const hasAfter = d.context_after_rows && d.context_after_rows.length > 0;
  const hasOther = d.context_other_rows && d.context_other_rows.length > 0;

  if (hasBefore || hasAfter || hasOther) {{
    let ctxInner = "";
    if (hasBefore || hasAfter) {{
      ctxInner += `<div class="side-by-side">
        <div>${{threeColTable("Before", d.context_before_rows)}}</div>
        <div>${{threeColTable("After", d.context_after_rows)}}</div>
      </div>`;
    }}
    if (hasOther) {{
      ctxInner += threeColTable("Other", d.context_other_rows);
    }}
    contextHTML = `<div class="section-header">Context</div>${{ctxInner}}`;
  }}

  // --- Score section ---
  let scoreHTML = "";
  const hasScores = d.score_rows && d.score_rows.length > 0;
  if (hasScores) {{
    const scoreBody = d.score_rows.map(r =>
      `<tr><td>${{r.attr}}</td><td style="font-weight:600">${{r.val}}</td><td>${{r.count}}</td></tr>`
    ).join("");
    scoreHTML = `<div class="section-header" style="color:#2b6cb0;">Scores</div>
      <table>
        <tr><th>Metric</th><th>Value</th><th></th></tr>
        ${{scoreBody}}
      </table>`;
  }}

  return `<div class="node-fo">
    <table><tr><th colspan="3">${{d.title}}</th></tr></table>
    ${{contentHTML}}
    ${{contextHTML}}
    ${{scoreHTML}}
  </div>`;
}}
</script>
</body>
</html>"""

    # ---- primitive score data (for editor sidebar) --------------------

    def get_primitive_score_data(self) -> list:
        """Collect score_data from all PrimitiveParseNodes for display in the editor."""
        primitives = []

        def walk(node):
            if isinstance(node, PrimitiveParseNode):
                primitives.append(node)
            for _, ch in getattr(node, "children", []):
                walk(ch)

        walk(self.global_root_node)
        primitives.sort(key=lambda n: (n.position_idx if n.position_idx is not None else float("inf"), n.title))

        return [
            {"title": n.title, "position_idx": n.position_idx, "score_data": n.score_data or {},
             "context_path_hashes": getattr(n, "context_path_hashes", [])}
            for n in primitives
        ]

    # ---- export (wraps to_json + action log) ---------------------------

    def export_json(self, filepath=None) -> dict:
        res = self.to_json(filepath=filepath)
        log_entry = {
            "timestamp": time.time(),
            "type": "export",
            "description": f"Exported parse tree to {filepath or 'json-string'}",
            "payload": {"filepath": filepath},
        }
        self.action_log.append(log_entry)
        return {"ok": True, "export_result": res, "action_log_entry": log_entry}

    # ---- editor HTML (interactive) -------------------------------------

    def editor_build_html(self, d3_data_json, node_w=600, node_h=130, h_gap=80, v_gap=150):
        sentence_str = self.window or ""
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Parse Tree Editor (Multi-Hierarchy)</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
#editor-container {{ display: flex; flex-direction: row; height: 100vh; }}
#tree-panel {{ flex: 3; overflow: auto; border-right: 1px solid #ccc; padding: 12px; }}
#sidebar {{ flex: 1; overflow-y: auto; padding: 12px; background: #f9f9f9; }}
#primitive-scores {{ margin-top: 12px; }}
#primitive-score-buttons button {{ width: 100%; text-align: left; margin: 4px 0; padding: 4px 8px; font-size: 12px; }}
#primitive-score-view {{ margin-top: 8px; font-size: 12px; }}
#header {{ padding: 12px; border-bottom: 1px solid #ccc; }}
button {{ margin: 4px; padding: 4px 8px; font-size: 12px; }}
#pair-buttons {{ margin-bottom: 12px; }}
ul {{ list-style: none; padding-left: 0; font-size: 12px; }}
li {{ margin-bottom: 6px; }}
.modal {{
display: none; position: fixed; z-index: 1000; left:0; top:0; width:100%; height:100%;
overflow:auto; background-color: rgba(0,0,0,0.4);
}}
.modal-content {{
background-color: #fff; margin: 4% auto; padding: 20px; border: 1px solid #888;
min-width: 600px; width: fit-content; max-width: 95vw; border-radius:8px;
}}
.close {{ float:right; font-size: 18px; cursor: pointer; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
th, td {{ border: 1px solid #888; padding: 4px; font-size: 12px; }}
th {{ background: #f3f5f7; font-weight: 600; }}
.path-tab-btn {{ padding: 4px 12px; font-size: 12px; cursor: pointer; border: 1px solid #aaa; background: #fff; border-radius: 4px; margin-right: 4px; }}
.path-tab-btn.active-tab {{ background: #dbeafe; font-weight: 700; border-color: #3b82f6; }}
#path-viz-container {{ width: 560px; height: 420px; overflow: hidden; border: 1px solid #ccc; background: #fafafa; margin-top: 6px; }}
#path-viz-svg {{ display: block; }}
.section-header {{ font-weight: 700; font-size: 13px; color: #333; margin: 8px 0 4px 0; border-bottom: 1px solid #ccc; padding-bottom: 2px; }}
.side-by-side {{ display: flex; gap: 8px; align-items: flex-start; }}
.side-by-side > div {{ flex: 1; min-width: 0; }}
.sub-title {{ font-weight: 600; font-size: 11px; color: #555; margin-bottom: 2px; }}
</style>
</head>
<body>
<div id="header">
    <h2>Parse Tree Editor (Multi-Hierarchy)</h2>
    <h4>Current sentence: <span id="sentence-text">{sentence_str}</span></h4>
    <button id="undo-btn">Undo Last Chunk</button>
    <button id="export-btn">Export Tree</button>
    <button id="export-ltm-btn">Export LTM</button>
</div>
<div id="editor-container">
    <div id="tree-panel"><div id="tree"></div></div>
    <div id="sidebar">
        <div id="pair-buttons"><strong>Candidate Pairs:</strong></div>
        <div id="primitive-scores">
            <strong>Primitive Scores:</strong>
            <div id="primitive-score-buttons"></div>
            <div id="primitive-path-viz-container" style="display:none;margin-top:8px;border:1px solid #ccc;background:#fafafa;border-radius:4px;">
                <div style="font-size:11px;color:#555;padding:4px 6px 2px;"><strong>Context Score Data:</strong></div>
                <div id="primitive-score-view" style="padding:2px 6px 6px;font-size:12px;max-height:200px;overflow-y:auto;"></div>
                <div id="primitive-path-viz-sub" style="display:none;">
                    <div style="font-size:11px;color:#555;padding:4px 6px 0;border-top:1px solid #e0e0e0;"><strong>Context Categorize Path:</strong></div>
                    <svg id="primitive-path-svg" width="100%" height="300"></svg>
                </div>
            </div>
        </div>
        <div><strong>Action Log:</strong><ul id="action-log"></ul></div>
    </div>
</div>
<div id="candidate-modal" class="modal">
<div class="modal-content">
    <span class="close">&times;</span>
    <h3>Candidate Chunk</h3>
    <p><strong>Title:</strong> <span id="candidate-title"></span></p>
    <p><strong>Score:</strong> <span id="candidate-score"></span></p>
    <div style="margin-top:14px;">
        <strong>Categorize Path:</strong>
        <div style="margin:6px 0 4px;">
            <button id="tab-btn-content" class="path-tab-btn active-tab" onclick="switchPathTab('content')">Content Hierarchy</button>
            <button id="tab-btn-context" class="path-tab-btn" onclick="switchPathTab('context')">Context Hierarchy</button>
        </div>
        <div id="path-viz-container">
            <div id="path-tab-score" style="font-size:12px;max-height:180px;overflow-y:auto;margin-bottom:6px;"></div>
            <svg id="path-viz-svg" width="560" height="420"></svg>
        </div>
    </div>
    <button id="apply-candidate-btn" style="margin-top:12px;">Apply Chunk</button>
</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
let treeData = {d3_data_json};
let currentLeft=null, currentRight=null;
let _pathVizData = {{content: null, context: null}};
let _currentPathTab = 'content';
let _primitiveContextTree = null;

function switchPathTab(tab) {{
    _currentPathTab = tab;
    document.getElementById('tab-btn-content').classList.toggle('active-tab', tab === 'content');
    document.getElementById('tab-btn-context').classList.toggle('active-tab', tab === 'context');
    renderPathViz(tab);
}}

function renderPathViz(tab) {{
    const entry = _pathVizData[tab];
    const scoreEl = document.getElementById('path-tab-score');
    if (!entry || !entry.tree) {{ if(scoreEl) scoreEl.innerHTML=''; return; }}

    // Merged-pair score
    let scoreHTML = buildScoreTable(entry.scoreData || {{}});

    // Per-chunk context scores (context tab only)
    if (tab === 'context') {{
        const hasLeft  = entry.leftScoreData  && Object.keys(entry.leftScoreData).length  > 0;
        const hasRight = entry.rightScoreData && Object.keys(entry.rightScoreData).length > 0;
        if (hasLeft || hasRight) {{
            scoreHTML += `<div class="section-header" style="margin-top:10px;">Per-Chunk Context Scores</div>
<div class="side-by-side">
  <div>
    <div class="sub-title">${{entry.leftTitle || "Left"}}</div>
    ${{buildScoreTable(entry.leftScoreData || {{}})}}
  </div>
  <div>
    <div class="sub-title">${{entry.rightTitle || "Right"}}</div>
    ${{buildScoreTable(entry.rightScoreData || {{}})}}
  </div>
</div>`;
        }}
    }}

    if (scoreEl) scoreEl.innerHTML = scoreHTML;
    const treeD  = entry.tree;
    const pathArr = entry.path || [];
    const pathNodes = new Set(pathArr);
    // Reconstruct edges from tree structure (works for both DFS and BFS)
    const pathEdges = new Set();
    function findEdges(node) {{
        if (!node.children) return;
        for (const child of node.children) {{
            if (pathNodes.has(node.id) && pathNodes.has(child.id)) {{
                pathEdges.add(node.id + ">>>" + child.id);
            }}
            findEdges(child);
        }}
    }}
    findEdges(treeD);
    const W = 560, H = 420;
    const svgSel = d3.select("#path-viz-svg");
    svgSel.selectAll("*").remove();
    svgSel.attr("width", W).attr("height", H);
    const g = svgSel.append("g").attr("class", "pviz-root");
    const root = d3.hierarchy(treeD);
    const layout = d3.tree().size([W - 80, H - 60]);
    layout(root);
    // Custom Y: full spacing for first 7 layers, compressed beyond
    const FULL_LAYERS = 7;
    let maxDepth = 0;
    root.each(d => {{ if (d.depth > maxDepth) maxDepth = d.depth; }});
    function depthY(depth) {{
        const fullGap = 50, compressGap = 12;
        if (depth <= FULL_LAYERS) return depth * fullGap;
        return FULL_LAYERS * fullGap + (depth - FULL_LAYERS) * compressGap;
    }}
    root.each(d => {{ d.x += 40; d.y = depthY(d.depth) + 30; }});
    svgSel.call(d3.zoom().scaleExtent([0.02, 20])
        .on("zoom", e => d3.select("#path-viz-svg .pviz-root").attr("transform", e.transform)));
    g.selectAll("line.pviz-e")
        .data(root.links()).join("line").attr("class", "pviz-e")
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y)
        .attr("stroke", d => pathEdges.has(d.source.data.id+">>>"+d.target.data.id) ? "#e53e3e" : "#ccc")
        .attr("stroke-width", d => pathEdges.has(d.source.data.id+">>>"+d.target.data.id) ? 2.5 : 1);
    g.selectAll("circle.pviz-n")
        .data(root.descendants()).join("circle").attr("class", "pviz-n")
        .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4)
        .attr("fill", d => pathNodes.has(d.data.id) ? "#e53e3e" : "#888")
        .attr("stroke", "none")
        .append("title").text(d => d.data.id);
}}
const nodeW={node_w}, nodeH={node_h}, hGap={h_gap}, vGap={v_gap};

function renderTree(data){{
    d3.select("#tree").selectAll("*").remove();
    const root = d3.hierarchy(data);
    const layout = d3.tree().nodeSize([nodeW+hGap, nodeH+vGap])
        .separation((a,b) => a.parent===b.parent ? 1.0 : 1.4);
    layout(root);
    const svg=d3.select("#tree").append("svg").attr("width",1).attr("height",1);
    const g=svg.append("g");
    const linkGen=d3.linkVertical().x(d=>d.x).y(d=>d.y);
    const link=g.selectAll("path.link").data(root.links()).join("path").attr("class","link")
        .attr("fill","none").attr("stroke","#9aa1a9").attr("stroke-width",1.5)
        .attr("d",linkGen);
    const node=g.selectAll("g.node").data(root.descendants()).join("g")
        .attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
    node.append("rect").attr("class","node-box").attr("x",-nodeW/2).attr("y",0)
        .attr("width",nodeW).attr("height",nodeH).attr("stroke","#444").attr("fill","#fff").attr("rx",8).attr("ry",8);
    node.append("foreignObject").attr("class","node-fo").attr("x",-nodeW/2+6).attr("y",6)
        .attr("width",nodeW-12).attr("height",1000).html(d=>nodeHTML(d.data));
    // measure actual heights
    node.selectAll("foreignObject").each(function(d){{
        const fo=d3.select(this),div=fo.select("div").node();
        const h=div.getBoundingClientRect().height+12;
        d._nodeHeight=h;
        fo.attr("height",h);
        d3.select(this.parentNode).select("rect").attr("height",h+12);
    }});
    // depth-based layout to prevent overlap
    const depthMaxHeight=new Map();
    root.each(d=>{{const h=d._nodeHeight||nodeH;const ex=depthMaxHeight.get(d.depth)||0;if(h>ex) depthMaxHeight.set(d.depth,h);}});
    const maxD=Math.max(...Array.from(depthMaxHeight.keys()));
    const depthOffsets=[];
    for(let i=0;i<=maxD;i++){{const prev=i===0?0:depthOffsets[i-1]+(depthMaxHeight.get(i-1)||nodeH)+vGap;depthOffsets.push(prev);}}
    root.each(d=>{{d.y=depthOffsets[d.depth];}});
    node.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
    link.attr("d",linkGen);
    // fit viewBox
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
    root.each(d=>{{const halfW=nodeW/2,h=d._nodeHeight||nodeH;if(d.x-halfW-30<x0)x0=d.x-halfW-30;if(d.x+halfW+30>x1)x1=d.x+halfW+30;if(d.y-30<y0)y0=d.y-30;if(d.y+h+30>y1)y1=d.y+h+30;}});
    const width=x1-x0,height=y1-y0;
    svg.attr("width",width).attr("height",height).attr("viewBox",[x0,y0,width,height].join(" "));
}}

function threeColTable(title, rows) {{
    if (!rows || rows.length === 0) return "";
    const body = rows.map(r =>
        `<tr><td>${{r.attr}}</td><td>${{r.val}}</td><td>${{r.count}}</td></tr>`
    ).join("");
    return `<div class="sub-title">${{title}}</div>
        <table>
          <tr><th>Attr</th><th>Value</th><th>Count</th></tr>
          ${{body}}
        </table>`;
}}

function nodeHTML(d){{
  // --- Content section ---
  let contentHTML = "";
  const hasLeft = d.content_left_rows && d.content_left_rows.length > 0;
  const hasRight = d.content_right_rows && d.content_right_rows.length > 0;
  const hasSingle = d.content_rows && d.content_rows.length > 0;

  if (hasLeft || hasRight) {{
    contentHTML = `<div class="section-header">Content</div>
      <div class="side-by-side">
        <div>${{threeColTable("Left", d.content_left_rows)}}</div>
        <div>${{threeColTable("Right", d.content_right_rows)}}</div>
      </div>`;
  }} else if (hasSingle) {{
    contentHTML = `<div class="section-header">Content</div>${{threeColTable("", d.content_rows)}}`;
  }}

  // --- Context section ---
  let contextHTML = "";
  const hasBefore = d.context_before_rows && d.context_before_rows.length > 0;
  const hasAfter = d.context_after_rows && d.context_after_rows.length > 0;
  const hasOther = d.context_other_rows && d.context_other_rows.length > 0;

  if (hasBefore || hasAfter || hasOther) {{
    let ctxInner = "";
    if (hasBefore || hasAfter) {{
      ctxInner += `<div class="side-by-side">
        <div>${{threeColTable("Before", d.context_before_rows)}}</div>
        <div>${{threeColTable("After", d.context_after_rows)}}</div>
      </div>`;
    }}
    if (hasOther) {{
      ctxInner += threeColTable("Other", d.context_other_rows);
    }}
    contextHTML = `<div class="section-header">Context</div>${{ctxInner}}`;
  }}

  return `<div class="node-fo">
    <table><tr><th colspan="3">${{d.title}}</th></tr></table>
    ${{contentHTML}}
    ${{contextHTML}}
  </div>`;
}}

function updateLog(log){{
    const ul=document.getElementById("action-log");ul.innerHTML="";
    log.forEach(e=>{{ul.innerHTML+=`<li>[${{new Date(e.timestamp*1000).toLocaleTimeString()}}] ${{e.description}}</li>`;}});
}}
function formatScoreValue(val){{
    if(val===null||typeof val==="undefined") return "null";
    if(typeof val==="number") return Number.isFinite(val)?val.toFixed(3):String(val);
    return val;
}}
function buildScoreTable(score){{
    if(!score||Object.keys(score).length===0) return "<i>No score data</i>";
    let rows="";
    for(const [k,v] of Object.entries(score)){{if(typeof v==="object"&&v!==null) continue;rows+=`<tr><td>${{k}}</td><td>${{formatScoreValue(v)}}</td></tr>`;}}
    return `<table><tr><th>Metric</th><th>Value</th></tr>${{rows}}</table>`;
}}
function renderPrimitivePath(pathHashes){{
    const sub=document.getElementById("primitive-path-viz-sub");
    if(!_primitiveContextTree||!pathHashes||pathHashes.length===0){{if(sub)sub.style.display="none";return;}}
    if(sub)sub.style.display="";
    const pathArr=pathHashes;
    const pathNodes=new Set(pathArr);
    // Reconstruct edges from tree structure (works for both DFS and BFS)
    const pathEdges=new Set();
    function findEdges(node){{
        if(!node.children)return;
        for(const child of node.children){{
            if(pathNodes.has(node.id)&&pathNodes.has(child.id)){{
                pathEdges.add(node.id+">>>"+child.id);
            }}
            findEdges(child);
        }}
    }}
    findEdges(_primitiveContextTree);
    const svgEl=document.getElementById("primitive-path-svg");
    const W=svgEl?svgEl.getBoundingClientRect().width||380:380;
    const H=280;
    const svgSel=d3.select("#primitive-path-svg");
    svgSel.selectAll("*").remove();
    svgSel.attr("width",W).attr("height",H);
    const g=svgSel.append("g").attr("class","ppviz-root");
    const root=d3.hierarchy(_primitiveContextTree);
    const layout=d3.tree().size([W-80,H-60]);
    layout(root);
    // Custom Y: full spacing for first 7 layers, compressed beyond
    const PP_FULL=7;
    let ppMaxD=0;root.each(d=>{{if(d.depth>ppMaxD)ppMaxD=d.depth;}});
    function ppDepthY(depth){{const fg=40,cg=10;if(depth<=PP_FULL)return depth*fg;return PP_FULL*fg+(depth-PP_FULL)*cg;}}
    root.each(d=>{{d.x+=40;d.y=ppDepthY(d.depth)+30;}});
    svgSel.call(d3.zoom().scaleExtent([0.02,20])
        .on("zoom",e=>d3.select("#primitive-path-svg .ppviz-root").attr("transform",e.transform)));
    g.selectAll("line.ppviz-e")
        .data(root.links()).join("line").attr("class","ppviz-e")
        .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
        .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y)
        .attr("stroke",d=>pathEdges.has(d.source.data.id+">>>"+d.target.data.id)?"#e53e3e":"#ccc")
        .attr("stroke-width",d=>pathEdges.has(d.source.data.id+">>>"+d.target.data.id)?2.5:1);
    g.selectAll("circle.ppviz-n")
        .data(root.descendants()).join("circle").attr("class","ppviz-n")
        .attr("cx",d=>d.x).attr("cy",d=>d.y).attr("r",4)
        .attr("fill",d=>pathNodes.has(d.data.id)?"#e53e3e":"#888")
        .attr("stroke","none")
        .append("title").text(d=>d.data.id);
}}
function renderPrimitiveScores(primitives){{
    const btnContainer=document.getElementById("primitive-score-buttons");
    if(!btnContainer) return;
    btnContainer.innerHTML="";
    const container=document.getElementById("primitive-path-viz-container");
    const view=document.getElementById("primitive-score-view");
    const sub=document.getElementById("primitive-path-viz-sub");
    if(container)container.style.display="none";
    if(sub)sub.style.display="none";
    primitives.forEach(p=>{{
        const btn=document.createElement("button");btn.textContent=p.title;
        btn.onclick=()=>{{
            if(view)view.innerHTML=buildScoreTable(p.score_data||{{}});
            if(container)container.style.display="";
            renderPrimitivePath(p.context_path_hashes||[]);
        }};
        btnContainer.appendChild(btn);
    }});
}}
function loadPairs(){{
    fetch("/api/tree").then(r=>r.json()).then(data=>{{
        const container=document.getElementById("pair-buttons");
        container.innerHTML="<strong>Candidate Pairs:</strong>";
        const s=document.getElementById("sentence-text");
        if(s&&data.sentence) s.textContent=data.sentence;
        if(data.context_tree) _primitiveContextTree=data.context_tree;
        renderPrimitiveScores(data.primitive_scores||[]);
        data.pairs.forEach(p=>{{
            const btn=document.createElement("button");
            btn.textContent=`${{p.left_title}} + ${{p.right_title}}`;
            btn.onclick=()=>evaluatePair(p.left_word_index,p.right_word_index);
            container.appendChild(btn);
        }});
        updateLog(data.action_log);renderTree(data.tree);
    }});
}}
function evaluatePair(left,right){{
    currentLeft=left;currentRight=right;
    fetch("/api/evaluate",{{method:"POST",headers:{{"Content-Type":"application/json"}},
        body:JSON.stringify({{left_word_index:left,right_word_index:right,debug:true}})}})
    .then(r=>r.json()).then(res=>{{
        if(res.ok) showCandidateModal(res.result, res.content_tree, res.context_tree);
        else alert(res.error);
    }});
}}
const modal=document.getElementById("candidate-modal");
const spanClose=modal.querySelector(".close");
spanClose.onclick=()=>modal.style.display="none";
window.onclick=e=>{{if(e.target==modal) modal.style.display="none";}};
function showCandidateModal(result, contentTree, contextTree){{
    document.getElementById("candidate-title").textContent=result.candidate_concept_id||result.candidate_concept_hash;
    var modeLabel = result.categorization_mode ? (' [' + result.categorization_mode.toUpperCase() + ']') : '';
    document.getElementById("candidate-score").textContent=result.score.toFixed(3) + modeLabel;
    // store path data for both hierarchies
    _pathVizData = {{
        content: {{ tree: contentTree, path: result.content_path_hashes || [], scoreData: result.content_score_data || {{}} }},
        context: {{
            tree: contextTree,
            path: result.context_path_hashes || [],
            scoreData: result.context_score_data || {{}},
            leftTitle: result.left_title || "Left",
            rightTitle: result.right_title || "Right",
            leftScoreData: result.left_context_score_data || {{}},
            rightScoreData: result.right_context_score_data || {{}},
        }},
    }};
    // reset tab to content
    _currentPathTab = 'content';
    document.getElementById('tab-btn-content').classList.add('active-tab');
    document.getElementById('tab-btn-context').classList.remove('active-tab');
    modal.style.display="block";
    // render after modal is visible so SVG has correct dimensions
    requestAnimationFrame(() => renderPathViz('content'));
}}
document.getElementById("apply-candidate-btn").onclick=()=>{{
    if(currentLeft===null||currentRight===null) return;
    if(!confirm("Confirm applying this chunk?")) return;
    fetch("/api/apply",{{method:"POST",headers:{{"Content-Type":"application/json"}},
        body:JSON.stringify({{left_word_index:currentLeft,right_word_index:currentRight}})}})
    .then(r=>r.json()).then(res=>{{if(res.ok){{loadPairs();modal.style.display="none";}}else alert(res.error);}});
}};
document.getElementById("undo-btn").onclick=()=>{{
    fetch("/api/undo",{{method:"POST"}}).then(r=>r.json()).then(res=>{{
        if(res.ok) loadPairs(); else alert(res.reason||"Undo failed");
    }});
}};
document.getElementById("export-btn").onclick=()=>{{
    const fp=prompt("Enter filepath to export (optional):","");
    fetch("/api/export",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{filepath:fp}})}})
    .then(r=>r.json()).then(res=>{{
        if(res.ok){{alert("Parse tree exported and LTM updated!");
            if(res.refresh){{const s=document.getElementById("sentence-text");
                if(s&&res.new_sentence) s.textContent=res.new_sentence;
                setTimeout(()=>location.reload(),800);
            }}else loadPairs();
        }}else alert(res.error||"Export failed");
    }}).catch(err=>alert("Network error: "+err));
}};
document.getElementById("export-ltm-btn").onclick=()=>{{
    const fp=prompt("Enter filepath to save LTM (optional):","");
    fetch("/api/export_ltm",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{filepath:fp}})}})
    .then(r=>r.json()).then(res=>{{
        if(res.ok) alert("LTM exported!"+(res.filepath?" Saved to: "+res.filepath:""));
        else alert("Export failed: "+(res.error||"Unknown error"));
    }}).catch(err=>alert("Network error: "+err));
}};
loadPairs();
</script>
</body>
</html>"""

    # ---- serialization --------------------------------------------------

    def to_json(self, filepath=None):
        def serialize_node(node, index_map):
            if isinstance(node, PrimitiveParseNode):
                return {
                    "node_type": "primitive",
                    "title": node.title,
                    "position_idx": node.position_idx,
                    "word_id": node.word_id,
                    "context_instance": node.context_instance,
                    "label": node.label,
                    "complexity": node.complexity,
                    "stable": node.stable,
                    "score_data": node.score_data,
                    "context_before": getattr(node, "context_before", None),
                    "context_after": getattr(node, "context_after", None),
                    "parent": index_map.get(node.parent),
                    "children": [],
                }
            elif isinstance(node, CompositeParseNode):
                return {
                    "node_type": "composite",
                    "title": node.title,
                    "position_idx": node.position_idx,
                    "is_global_root": node.is_global_root,
                    "content_instance": node.content_instance,
                    "context_instance": node.context_instance,
                    "label": node.label,
                    "categorize_path": node.categorize_path,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "context_length": node.context_length,
                    "complexity": node.complexity,
                    "concept_label": node.concept_label,
                    "frozen": node.frozen,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            else:
                raise TypeError(f"Unknown node type {type(node)}")

        index_map = {node: i for i, node in enumerate([self.global_root_node] + self.nodes)}
        data = {
            "window": self.window,
            "context_length": self.context_length,
            "nodes": [serialize_node(n, index_map) for n in [self.global_root_node] + self.nodes],
        }
        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return filepath
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(data, ltm: 'LongTermMemory', filepath=False) -> 'FiniteParseTree':
        if filepath:
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(data, str):
            data = json.loads(data)

        tree = FiniteParseTree(ltm, context_length=data.get("context_length", 3))
        tree.window = data.get("window")

        def restore_dict_keys(d):
            if d is None:
                return None
            if isinstance(d, dict):
                return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()}
            return d

        node_objs = []
        for n in data["nodes"]:
            if n["node_type"] == "primitive":
                ctx_inst = restore_dict_keys(n.get("context_instance")) or {}
                label = restore_dict_keys(n.get("label")) or {}
                pn = PrimitiveParseNode(ctx_inst, label, n.get("position_idx", 0), n.get("word_id", 0))
                pn.title = n.get("title", pn.title)
                pn.complexity = n.get("complexity", 1)
                pn.stable = n.get("stable", False)
                pn.score_data = n.get("score_data", {})
                pn.context_before = [restore_dict_keys(d) for d in (n.get("context_before") or [])]
                pn.context_after = [restore_dict_keys(d) for d in (n.get("context_after") or [])]
                node_objs.append(pn)
            elif n["node_type"] == "composite":
                cn = CompositeParseNode()
                cn.title = n.get("title", cn.title)
                cn.is_global_root = n.get("is_global_root", False)
                cn.position_idx = n.get("position_idx")
                cn.content_instance = restore_dict_keys(n.get("content_instance"))
                cn.context_instance = restore_dict_keys(n.get("context_instance"))
                cn.label = restore_dict_keys(n.get("label"))
                cn.categorize_path = n.get("categorize_path")
                cn.context_before = [restore_dict_keys(d) for d in (n.get("context_before") or [])]
                cn.context_after = [restore_dict_keys(d) for d in (n.get("context_after") or [])]
                cn.context_length = n.get("context_length", 0)
                cn.complexity = n.get("complexity", 0)
                cn.concept_label = n.get("concept_label")
                cn.frozen = n.get("frozen", False)
                node_objs.append(cn)
            else:
                raise ValueError(f"Unknown node_type {n['node_type']}")

        # restore parent/child
        for idx, n in enumerate(data["nodes"]):
            obj = node_objs[idx]
            pi = n.get("parent")
            if pi is not None:
                obj.parent = node_objs[pi]
            obj.children = SortedList(
                [(node_objs[ci].position_idx, node_objs[ci]) for ci in n.get("children", [])]
            )

        tree.global_root_node = node_objs[0]
        tree.nodes = node_objs[1:]
        return tree


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory(object):
    """
    Holds TWO Cobweb hierarchies (content + context) plus corpus/vocabulary
    management.

    Content hierarchy  – instances are {0..cpd-1: left_path_depths, cpd..2*cpd-1: right_path_depths}.
    Context hierarchy (non-BOW) – instances are {0..ctx_len-1: ctx_before,
                          ctx_len..2*ctx_len-1: ctx_after,
                          -2: complexity (hidden),
                          2*ctx_len: content-ref (word_id for primitives,
                                       content leaf concept_id for composites)}.

    Context hierarchy (BOW) – instances are {0: before_bag, 1: after_bag,
                          -2: complexity (hidden), 2: content-ref}.
                          Words in bag attributes are distance-weighted
                          (1/2^(j+1)) and summed if repeated.
    """

    def __init__(self, value_corpus: list, context_length: int = 3, alpha: float = 1e-4,
                 content_alpha: float = None, context_alpha: float = None,
                 content_bl_alpha: float = None, context_bl_alpha: float = None,
                 bow: bool = False,
                 categorization_mode: str = 'dfs', weighting: str = 'binary', empty_weighting: bool = False,
                 chunk_context: bool = False, context_n_iterations: int = 0,
                 depth_max_content: int = 1000, depth_max_context: int = 1000,
                 branch_max_content: int = 1000, branch_max_context: int = 1000,
                 instance_basic_level: bool = False,
                 content_attr_weights: dict = None, context_attr_weights: dict = None):
        _content_alpha = content_alpha if content_alpha is not None else alpha
        _context_alpha = context_alpha if context_alpha is not None else alpha
        self.content_alpha = _content_alpha
        self.context_alpha = _context_alpha
        self.content_bl_alpha = content_bl_alpha
        self.context_bl_alpha = context_bl_alpha
        self.chunk_context = chunk_context
        self.context_n_iterations = context_n_iterations

        # Create context hierarchy first (it serves as ref_tree for content)
        self.context_hierarchy = CobwebDiscreteTree(_context_alpha, weight_attr=False, depth_max=depth_max_context, branch_max=branch_max_context, attr_weights=context_attr_weights or {})
        # Content hierarchy uses context hierarchy as ref_tree for LCA similarity
        self.content_hierarchy = CobwebDiscreteTree(_content_alpha, weight_attr=False, depth_max=depth_max_content, branch_max=branch_max_content, ref_tree=self.context_hierarchy, attr_weights=content_attr_weights or {})
        # Mark content attrs 0 (left) and 1 (right) as ref attrs for soft matching
        self.content_hierarchy.set_ref_attr(0)
        self.content_hierarchy.set_ref_attr(1)

        # When chunk_context is enabled, context hierarchy uses itself as
        # ref_tree so that label_path values get LCA similarity.
        if chunk_context:
            self.context_hierarchy.set_ref_tree(self.context_hierarchy)
            if bow:
                self.context_hierarchy.set_ref_attr(0)  # before_bag
                self.context_hierarchy.set_ref_attr(1)  # after_bag
            else:
                for j in range(2 * context_length):
                    self.context_hierarchy.set_ref_attr(j)
            # content-ref attr also uses label_paths
            _cref = 2 if bow else 2 * context_length
            self.context_hierarchy.set_ref_attr(_cref)

        # vocabulary: index 0 is always EMPTYNULL
        self.id_to_value: List[str] = ["EMPTYNULL"]
        for x in value_corpus:
            self.id_to_value.append(x)
        # C{X} complexity identifiers are registered dynamically on first use.
        self.value_to_id: Dict[str, int] = {w: i for i, w in enumerate(self.id_to_value)}
        self.id_count: int = len(self.id_to_value) - 1

        self.context_length = context_length
        self.bow = bow
        self.categorization_mode = categorization_mode
        self.weighting = weighting  # 'binary', 'harmonic', or 'constant'
        self.empty_weighting = empty_weighting  # True: EMPTYNULL uses count 1
        self.instance_basic_level = instance_basic_level

        # register root concepts of both hierarchies
        self._register_concept(self.content_hierarchy.root)
        self._register_concept(self.context_hierarchy.root)

        # drawer for content hierarchy visualization
        # Single leaf pointer per side (Methodology 4.0)
        content_headers = ["Left", "Right"]
        self.content_drawer = HTMLCobwebDrawer(
            content_headers,
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id,
        )

        # drawer for context hierarchy visualization
        if bow:
            context_headers = ["CtxBefore", "CtxAfter"]
        else:
            context_headers = (
                [f"Context-Before{i}" for i in range(context_length)]
                + [f"Context-After{i}" for i in range(context_length)]
            )
        content_ref_attr_idx = len(context_headers)
        context_headers = context_headers + ["Content-Ref"]
        def _content_ref_display(val_id):
            if val_id is not None and 0 <= val_id < len(self.id_to_value):
                name = self.id_to_value[val_id]
            else:
                name = f"?{val_id}"
            if isinstance(name, str) and name.startswith("CONCEPT-"):
                return "C-" + name[8:20] + "…"
            return name
        self.context_drawer = HTMLCobwebDrawer(
            context_headers,
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id,
            attr_value_fn={content_ref_attr_idx: _content_ref_display},
            attr_name_overrides={content_ref_attr_idx: "Content-Ref", -2: "Complexity"},
        )

    # ---- property helpers -----------------------------------------------

    @property
    def content_ref_attr(self) -> int:
        """Attribute index for the content-ref in context instances.

        The first positive index after all context attributes:
            Normal slot mode:         2 * context_length
            BOW mode:                 2
        """
        return 2 if self.bow else 2 * self.context_length

    # ---- vocabulary helpers ---------------------------------------------

    def _register_concept(self, node: CobwebDiscreteNode):
        new_vocab = f"CONCEPT-{node.concept_hash()}"
        self.add_to_vocab(new_vocab)

    def add_to_vocab(self, new_vocab: str) -> bool:
        if new_vocab not in self.value_to_id:
            vid = len(self.id_to_value)   # always correct even if _get_or_register_cplx_vid
                                          # inserted entries without updating id_count
            self.id_to_value.append(new_vocab)
            self.id_count = vid
            self.value_to_id[new_vocab] = vid
            return True
        return False

    # ---- instance conversion helpers ------------------------------------

    def get_content_instance_statistics(self, content_inst: dict, debug=False) -> dict:
        """
        Categorize a content instance in the content hierarchy and return scoring data.
        """
        _cat_mode = getattr(self, 'categorization_mode', 'dfs')
        leaf, path, node_path, _ = _categorize(
            content_inst, self.content_hierarchy, mode=_cat_mode)
        return _score_along_path(node_path, content_inst, self.content_hierarchy, debug=debug,
                                 eval_alpha=self.content_bl_alpha,
                                 instance_basic_level=getattr(self, 'instance_basic_level', False))

    def get_context_instance_statistics(self, context_inst: dict, debug=False) -> dict:
        """
        Categorize a context instance in the context hierarchy and return scoring data.
        """
        _cat_mode = getattr(self, 'categorization_mode', 'dfs')
        leaf, path, node_path, _ = _categorize(
            context_inst, self.context_hierarchy, mode=_cat_mode)
        return _score_along_path(node_path, context_inst, self.context_hierarchy, debug=debug,
                                 eval_alpha=self.context_bl_alpha,
                                 instance_basic_level=getattr(self, 'instance_basic_level', False))

    # ---- learning (ifit + vocab management) -----------------------------

    def _ifit_and_update_vocab(self, instance: dict, tree: CobwebDiscreteTree, debug=False):
        """
        Call ifit on a hierarchy, process the resulting actions (vocab updates
        + splits).  Returns ``(leaf_node, rewrite_rules)``.
        """
        leaf, actions = tree.ifit(instance, debug=True)
        actions = [json.loads(x) for x in actions]

        rewrite_rules = []
        for act in actions:
            if act["action"] == "NEW":
                self.add_to_vocab(f"CONCEPT-{act['node']}")
            elif act["action"] == "MERGE":
                self.add_to_vocab(f"CONCEPT-{act['new_node']}")
            elif act["action"] == "SPLIT":
                rewrite_rules.append((act["deleted"], act["parent"]))

        return leaf, rewrite_rules

    def _apply_rewrite_rules(self, tree: CobwebDiscreteTree, rewrite_rules: list):
        """
        BFS through *tree* and replace split-deleted concept vocab IDs in
        av_counts.

        *rewrite_rules* is a list of ``(deleted_hash, parent_hash)`` where each
        hash is the raw concept hash string from a SPLIT action.  We convert
        these to integer vocab IDs and walk the tree, replacing every occurrence
        of the deleted ID with the parent ID in every node's av_count.
        """
        # Convert hash-based rules to vocab-ID-based rules.
        vid_rules: list = []
        for deleted_hash, parent_hash in rewrite_rules:
            old_vid = self.value_to_id.get(f"CONCEPT-{deleted_hash}")
            new_vid = self.value_to_id.get(f"CONCEPT-{parent_hash}")
            if old_vid is not None and new_vid is not None:
                vid_rules.append((old_vid, new_vid))

        if not vid_rules:
            return

        def av_replacement(av):
            replaced = False
            for attr in av.keys():
                for old_vid, new_vid in vid_rules:
                    if old_vid in av[attr]:
                        av[attr][new_vid] = av[attr].get(new_vid, 0) + av[attr][old_vid]
                        del av[attr][old_vid]
                        replaced = True
            return av, replaced

        to_visit = [tree.root]
        while to_visit:
            curr = to_visit.pop(0)
            new_av, replaced = av_replacement(curr.av_count)
            if replaced:
                curr.set_av_count(new_av)
                to_visit.extend(curr.children)

    def add_parse_tree(self, parse_tree: 'FiniteParseTree', debug: bool = False, shuffle: bool = True):
        """
        Learn from a completed parse tree.

        Order of operations (context-first per MULTIHIERARCHY.md):
          1. **Fit context instances** (WITHOUT -1 content-ref).
             This builds/updates the context hierarchy structure.
             Record which context leaf each node lands in.
          2. **Compute labels from context leaves** – use the freshly-fitted
             context hierarchy paths to build label_paths for content instances.
          3. **Build & fit content instances** using the fresh label_paths.
             Record which content leaf each composite lands in.
             Propagate any content-hierarchy splits to the context hierarchy.
          4. **Write content-refs back** – for each node, update its context
             leaf's av_count to include -1 = content_leaf reference, using
             set_av_count(). This makes content-refs available for generation.
        """
        # -- Step 0: collect nodes ----------------------------------------
        all_nodes: list = []  # every PrimitiveParseNode and CompositeParseNode

        def _collect(node):
            if isinstance(node, PrimitiveParseNode):
                all_nodes.append(node)
            elif isinstance(node, CompositeParseNode) and not node.is_global_root:
                all_nodes.append(node)
            for _, ch in getattr(node, "children", []):
                _collect(ch)

        for _, ch in parse_tree.global_root_node.children:
            _collect(ch)

        if debug:
            print(f"Adding parse tree for window: \"{parse_tree.window}\"")
            print(f"  nodes collected: {len(all_nodes)}")

        # -- Step 1: fit context instances ------------------------------------
        ctx_leaf_map: dict = {}   # id(node) → context-hierarchy leaf
        _cref_attr = self.content_ref_attr

        # When chunk_context is enabled the context hierarchy is its own
        # ref_tree; invalidate its ref cache before fitting new instances.
        if self.chunk_context:
            self.context_hierarchy.invalidate_ref_cache()

        # Optionally shuffle instances before fitting to hierarchies
        ctx_nodes = list(all_nodes)
        if shuffle:
            random.shuffle(ctx_nodes)

        for node in ctx_nodes:
            ctx_inst = node.get_context_instance()
            # Only strip content-ref for composites; primitives keep it so the
            # word-identity attribute participates in Cobweb clustering and is
            # registered in tree.attr_vals before any step-4 increment_attr_value.
            if isinstance(node, CompositeParseNode):
                ctx_inst.pop(_cref_attr, None)
            leaf, rewrites = self._ifit_and_update_vocab(
                ctx_inst, self.context_hierarchy, debug=debug)
            ctx_leaf_map[id(node)] = leaf
            # Propagate context splits to content hierarchy using simple rewrite
            if rewrites:
                self._apply_rewrite_rules(self.content_hierarchy, rewrites)

        if debug:
            print(f"  context instances fitted: {len(all_nodes)}")

        # -- Step 2: compute labels from fresh context hierarchy ----------

        def _refresh_labels_bottom_up(node):
            """Bottom-up DFS: refresh children first, then this node."""
            for _, ch in getattr(node, "children", []):
                _refresh_labels_bottom_up(ch)

            ctx_leaf = ctx_leaf_map.get(id(node))
            if ctx_leaf is None:
                return

            if isinstance(node, PrimitiveParseNode):
                node.label = {node.word_id: 1}
                node.label_path = _build_label_from_ctx_leaf(ctx_leaf, self.value_to_id)
                # Register the label_path VID → context leaf for LCA similarity
                if node.label_path:
                    self.content_hierarchy.register_ref_val(node.label_path, ctx_leaf)
                    if self.chunk_context:
                        self.context_hierarchy.register_ref_val(node.label_path, ctx_leaf)

            elif isinstance(node, CompositeParseNode) and not node.is_global_root:
                ctx_hash = ctx_leaf.concept_hash()
                concept_label_str = f"CONCEPT-{ctx_hash}"
                self.add_to_vocab(concept_label_str)
                new_concept_label = self.value_to_id.get(concept_label_str)
                node.concept_label = new_concept_label
                node.label = {new_concept_label: 1} if new_concept_label else {0: 1}

                node.label_path = _build_label_from_ctx_leaf(ctx_leaf, self.value_to_id)
                # Register the label_path VID → context leaf for LCA similarity
                if node.label_path:
                    self.content_hierarchy.register_ref_val(node.label_path, ctx_leaf)
                    if self.chunk_context:
                        self.context_hierarchy.register_ref_val(node.label_path, ctx_leaf)

                # Rebuild content_instance from children's refreshed labels
                children_sorted = list(node.children)
                if len(children_sorted) == 2:
                    left_child = children_sorted[0][1]
                    right_child = children_sorted[1][1]
                    node.content_instance = CompositeParseNode.create_content_instance(
                        left_child, right_child
                    )

        for _, ch in parse_tree.global_root_node.children:
            _refresh_labels_bottom_up(ch)

        if debug:
            print(f"  labels refreshed from context hierarchy")

        # -- Step 3: fit content instances --------------------------------
        # Invalidate cached ref_tree max_depth since context hierarchy may
        # have changed structure during step 1.
        self.content_hierarchy.invalidate_ref_cache()
        content_leaf_map: dict = {}   # id(comp_node) → content-hierarchy leaf

        # Prepare composite nodes for content fitting; optionally shuffle
        content_nodes = [n for n in all_nodes if isinstance(n, CompositeParseNode) and not n.is_global_root]
        if shuffle:
            random.shuffle(content_nodes)
        for node in content_nodes:
            ci = node.get_content_instance()
            if ci:
                leaf, rewrites = self._ifit_and_update_vocab(
                    ci, self.content_hierarchy, debug=debug)
                content_leaf_map[id(node)] = leaf
                if rewrites:
                    self._apply_rewrite_rules(self.context_hierarchy, rewrites)

        # Also fit orphan candidate pairs (parentless adjacent pairs)
        pairs = parse_tree.get_parentless_pairs()
        if shuffle:
            random.shuffle(pairs)
        for p in pairs:
            left = parse_tree._find_root_child_by_index(p["left_word_index"])
            right = parse_tree._find_root_child_by_index(p["right_word_index"])
            if left and right:
                ci = CompositeParseNode.create_content_instance(
                    left, right)
                _leaf, rewrites = self._ifit_and_update_vocab(
                    ci, self.content_hierarchy, debug=debug)
                if rewrites:
                    self._apply_rewrite_rules(self.context_hierarchy, rewrites)

        if debug:
            print(f"  content instances fitted: {len(content_leaf_map)}")

        # -- Step 4: write content-refs back to context hierarchy ---------
        # For each composite node, write a single content-leaf pointer into its
        # context-hierarchy leaf.  Primitives already have their content-ref
        # (word_id or label_path) written via ifit in step 1.
        for node in all_nodes:
            if isinstance(node, PrimitiveParseNode):
                continue  # already in tree via step-1 ifit

            ctx_leaf = ctx_leaf_map.get(id(node))
            if ctx_leaf is None:
                continue

            if isinstance(node, CompositeParseNode) and not node.is_global_root:
                if self.chunk_context:
                    # chunk_context: content-ref = context hierarchy leaf pointer
                    vid = node.label_path
                    if not vid:
                        continue
                else:
                    cnt_leaf = content_leaf_map.get(id(node))
                    if cnt_leaf is None:
                        continue
                    # Single pointer: register a CONCEPT-<hash> value and write it
                    concept_str = f"CONCEPT-{cnt_leaf.concept_hash()}"
                    self.add_to_vocab(concept_str)
                    vid = self.value_to_id.get(concept_str, 0)
                    if vid == 0:
                        continue
                node.context_instance[_cref_attr] = {vid: 1}
                ctx_leaf.increment_attr_value(_cref_attr, vid, 1)
            else:
                continue

        if debug:
            print(f"  content-refs written to context hierarchy leaves")

        return True

    # ---- visualization --------------------------------------------------

    def visualize_content_hierarchy(self, out_base="content_hierarchy", max_depth=1e9):
        self.content_drawer.draw_tree(self.content_hierarchy.root, out_base, max_depth=max_depth)

    def visualize_context_hierarchy(self, out_base="context_hierarchy", max_depth=1e9):
        self.context_drawer.draw_tree(self.context_hierarchy.root, out_base, max_depth=max_depth)

    # ---- basic-level node retrieval -------------------------------------

    def get_basic_level_nodes(self, n_samples: int = 200, max_nodes: int = 100) -> dict:
        """
        Walk every leaf node in both hierarchies, call .get_basic() on each,
        and return a deduplicated list of (hash, node, freq) tuples where
        freq is the number of leaf nodes that claimed that node as their
        basic-level node.

        Returns
        -------
        {"content": [(hash, node, freq), ...], "context": [(hash, node, freq), ...]}
        """
        def _collect(root, bl_alpha=None):
            seen = {}   # hash -> node
            freq = {}   # hash -> int count
            stack = [root]
            while stack:
                curr = stack.pop()
                if not curr.children:
                    _bl_eval_alpha = bl_alpha if bl_alpha is not None else -1.0
                    if getattr(self, 'instance_basic_level', False):
                        _inst = {a: {max(v, key=v.get): 1.0} for a, v in curr.av_count.items()} if curr.av_count else {}
                        basic = curr.get_basic_instance_pmi(_inst, eval_alpha=_bl_eval_alpha)
                    else:
                        basic = curr.get_basic(n_samples, max_nodes, debug=False,
                                              eval_alpha=_bl_eval_alpha)
                    h = basic.concept_hash()
                    if h not in seen:
                        seen[h] = basic
                        freq[h] = 0
                    freq[h] += 1
                else:
                    for child in curr.children:
                        stack.append(child)
            return [(h, seen[h], freq[h]) for h in seen]

        return {
            "content": _collect(self.content_hierarchy.root,
                                bl_alpha=getattr(self, 'content_bl_alpha', None)),
            "context": _collect(self.context_hierarchy.root,
                                bl_alpha=getattr(self, 'context_bl_alpha', None)),
        }

    # ---- save / load ----------------------------------------------------

    def save_state(self, dirpath: str) -> dict:
        os.makedirs(dirpath, exist_ok=True)

        meta = {
            "context_length": self.context_length,
            "content_alpha": self.content_alpha,
            "context_alpha": self.context_alpha,
            "content_bl_alpha": getattr(self, 'content_bl_alpha', None),
            "context_bl_alpha": getattr(self, 'context_bl_alpha', None),
            "bow": self.bow,
            "categorization_mode": getattr(self, 'categorization_mode', 'dfs'),
            "weighting": getattr(self, 'weighting', 'binary'),
            "empty_weighting": getattr(self, 'empty_weighting', False),
            "depth_max_content": self.content_hierarchy.depth_max,
            "depth_max_context": self.context_hierarchy.depth_max,
            "branch_max_content": self.content_hierarchy.branch_max,
            "branch_max_context": self.context_hierarchy.branch_max,
            "id_count": self.id_count,
            "id_to_value": self.id_to_value,
            "value_to_id": self.value_to_id,
        }

        meta_path = os.path.join(dirpath, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # content hierarchy
        content_path = os.path.join(dirpath, "content_tree.json")
        try:
            self.content_hierarchy.dump_json(content_path)
        except Exception:
            pass

        # context hierarchy
        context_path = os.path.join(dirpath, "context_tree.json")
        try:
            self.context_hierarchy.dump_json(context_path)
        except Exception:
            pass

        return {"ok": True, "meta": meta_path, "content_tree": content_path, "context_tree": context_path}

    @staticmethod
    def load_state(dirpath: str) -> 'LongTermMemory':
        meta_path = os.path.join(dirpath, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {dirpath}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        ltm = LongTermMemory(
            [], context_length=meta.get("context_length", 3),
            content_alpha=meta.get("content_alpha", 1e-4),
            context_alpha=meta.get("context_alpha", 1e-4),
            content_bl_alpha=meta.get("content_bl_alpha", None),
            context_bl_alpha=meta.get("context_bl_alpha", None),
            bow=meta.get("bow", False),
            categorization_mode=meta.get("categorization_mode", "dfs"),
            weighting=meta.get("weighting", "binary"),
            empty_weighting=meta.get("empty_weighting", False),
            depth_max_content=meta.get("depth_max_content", 1000),
            depth_max_context=meta.get("depth_max_context", 1000),
            branch_max_content=meta.get("branch_max_content", 1000),
            branch_max_context=meta.get("branch_max_context", 1000),
        )
        ltm.id_to_value = meta.get("id_to_value", ltm.id_to_value)
        ltm.value_to_id = meta.get("value_to_id", ltm.value_to_id)
        ltm.id_count = meta.get("id_count", ltm.id_count)

        # load hierarchies
        content_path = os.path.join(dirpath, "content_tree.json")
        if os.path.exists(content_path):
            ltm.content_hierarchy.load_json(content_path)

        context_path = os.path.join(dirpath, "context_tree.json")
        if os.path.exists(context_path):
            ltm.context_hierarchy.load_json(context_path)

        # rebuild drawers — delegate to a re-init of the drawer only;
        # LongTermMemory.__init__ already ran the right logic via the
        # constructor call above, so the drawers are already correct.
        # We only need to refresh stale id_to_value / value_to_id refs.
        ltm.content_drawer.id_to_value = ltm.id_to_value
        ltm.content_drawer.value_to_id = ltm.value_to_id
        ltm.context_drawer.id_to_value = ltm.id_to_value
        ltm.context_drawer.value_to_id = ltm.value_to_id
        return ltm


# ---------------------------------------------------------------------------
# WEBSTER  (primary orchestrator)
# ---------------------------------------------------------------------------

class WEBSTER(object):
    """
    Primary class that orchestrates all parsing and learning logic.
    All other classes (PrimitiveParseNode, CompositeParseNode, FiniteParseTree,
    LongTermMemory) serve as data classes with helper methods; WEBSTER handles
    the overall flow.

    Named per MULTIHIERARCHY.md's specification.
    """

    def __init__(self, value_corpus: list, context_length: int = 3, alpha: float = 1e-4,
                 content_alpha: float = None, context_alpha: float = None,
                 content_bl_alpha: float = None, context_bl_alpha: float = None,
                 threshold=5, bow: bool = False,
                 categorization_mode: str = 'dfs', weighting: str = 'binary', empty_weighting: bool = False,
                 chunk_context: bool = False, context_n_iterations: int = 0,
                 depth_max_content: int = 1000, depth_max_context: int = 1000,
                 branch_max_content: int = 1000, branch_max_context: int = 1000,
                 instance_basic_level: bool = False,
                 content_attr_weights: dict = None, context_attr_weights: dict = None):
        """
        Parameters
        ----------
        value_corpus : list
            Initial vocabulary (list of word strings).
        context_length : int
            Number of context-window slots on each side (before/after).
        alpha : float
            Default Cobweb smoothing parameter (used for both hierarchies
            unless overridden by *content_alpha* / *context_alpha*).
        content_alpha : float | None
            Smoothing parameter for the content hierarchy. Falls back to
            *alpha* when ``None``.
        context_alpha : float | None
            Smoothing parameter for the context hierarchy. Falls back to
            *alpha* when ``None``.
        content_bl_alpha : float | None
            Smoothing (eval_alpha) used **only during EPMI evaluation** in
            ``get_basic_instance_pmi`` calls on the content hierarchy.
            Decouples the basic-level detector's sharpness from the tree's
            structural alpha.  Falls back to the tree's own alpha when ``None``.
        context_bl_alpha : float | None
            Same as *content_bl_alpha* but for the context hierarchy.
        threshold : int | float
            Minimum **basic-level count** required to accept a chunk merge.
            Default ``5``.
        bow : bool
            If True, context instances use a bag-of-words representation.
        categorization_mode : str
            Categorization strategy: ``'dfs'``, ``'bfs'``, or ``'bfs_pmi'``.
        weighting : str
            Context distance-weighting scheme:
            ``'binary'``, ``'harmonic'``, or ``'constant'``.
        empty_weighting : bool
            If True, EMPTYNULL context slots use count 1 instead of 0.
        chunk_context : bool
            If True, context instances use neighbors' concept labels
            (label_paths) instead of raw word IDs.
        context_n_iterations : int
            Maximum number of iterative re-categorization passes when
            ``chunk_context`` is True.  ``0`` means iterate until
            convergence (labels stop changing).  A positive integer
            caps the loop at that many passes.
        depth_max_content : int
            Maximum depth for content Cobweb tree. Default ``1000``.
        depth_max_context : int
            Maximum depth for context Cobweb tree. Default ``1000``.
        branch_max_content : int
            Maximum children per node in content tree. Default ``1000``.
        branch_max_context : int
            Maximum children per node in context tree. Default ``1000``.
        instance_basic_level : bool
            If True, use ``get_basic_instance_pmi()`` (instance PMI proxy) instead
            of the Monte Carlo ``get_basic()`` for basic-level selection. Fast
            analytical approximation; no sampling required. Default ``False``.
        """
        self.ltm = LongTermMemory(
            value_corpus, context_length=context_length,
            alpha=alpha,
            content_alpha=content_alpha, context_alpha=context_alpha,
            content_bl_alpha=content_bl_alpha, context_bl_alpha=context_bl_alpha,
            bow=bow,
            categorization_mode=categorization_mode,
            weighting=weighting,
            empty_weighting=empty_weighting,
            chunk_context=chunk_context, context_n_iterations=context_n_iterations,
            depth_max_content=depth_max_content, depth_max_context=depth_max_context,
            branch_max_content=branch_max_content, branch_max_context=branch_max_context,
            instance_basic_level=instance_basic_level,
            content_attr_weights=content_attr_weights,
            context_attr_weights=context_attr_weights,
        )
        self.context_length = context_length
        self.threshold = threshold
        self.bow = bow
        self.categorization_mode = categorization_mode
        self.weighting = weighting
        self.empty_weighting = empty_weighting
        self.chunk_context = chunk_context
        self.context_n_iterations = context_n_iterations
        self.content_bl_alpha = content_bl_alpha
        self.context_bl_alpha = context_bl_alpha
        self.instance_basic_level = instance_basic_level

    # ---- accessors ------------------------------------------------------

    @property
    def id_to_value(self):
        return self.ltm.id_to_value

    @property
    def value_to_id(self):
        return self.ltm.value_to_id

    def get_long_term_memory(self) -> LongTermMemory:
        return self.ltm

    def get_basic_level_nodes(self, n_samples: int = 200, max_nodes: int = 100) -> dict:
        """Delegate to LongTermMemory.get_basic_level_nodes."""
        return self.ltm.get_basic_level_nodes(n_samples=n_samples, max_nodes=max_nodes)

    # ---- primary parsing ------------------------------------------------

    def parse_sentence(
        self,
        sentence: str,
        threshold=None,
        new_vocab: bool = True,
        learning: bool = False,
        debug: bool = False,
    ) -> FiniteParseTree:
        """
        Create a parse tree for *sentence*.

        Parameters
        ----------
        sentence : str
            The text to parse.
        threshold : float | "converge" | None
            Score threshold for accepting chunks. None falls back to self.threshold.
        new_vocab : bool
            If True, add previously unseen words to the vocabulary.
        learning : bool
            If True, add the resulting parse tree to the long-term memory after parsing.
        debug : bool
            Verbose output.

        Returns
        -------
        FiniteParseTree
        """
        if threshold is None:
            threshold = self.threshold

        # optionally register new words
        if new_vocab:
            tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
            for tok in tokens:
                self.ltm.add_to_vocab(tok)

        parse_tree = FiniteParseTree(self.ltm, context_length=self.context_length)
        parse_tree.build(sentence, end_behavior=threshold, debug=debug)

        if learning:
            self.ltm.add_parse_tree(
                parse_tree, shuffle=False, debug=debug,
            )

        return parse_tree

    def parse_input(
        self,
        windows: List[str],
        end_behavior="converge",
        learning: bool = True,
        debug: bool = False,
    ) -> List[FiniteParseTree]:
        """
        Parse a list of sentences/windows and optionally learn from each.
        """
        trees = []
        for window in windows:
            if debug:
                print(f"BUILDING PARSE TREE FOR WINDOW: {window}")
            pt = self.parse_sentence(
                window,
                threshold=end_behavior,
                new_vocab=True,
                learning=learning,
                debug=debug,
            )
            trees.append(pt)
            if debug:
                print("-" * 100)
        return trees

    # ---- generation -----------------------------------------------------

    def _find_node_by_hash(self, root, hash_str):
        """Find a node in a hierarchy tree by its concept hash string."""
        if root is None:
            return None
        if str(root.concept_hash()) == str(hash_str):
            return root
        for child in root.children:
            found = self._find_node_by_hash(child, hash_str)
            if found:
                return found
        return None

    def generate_sentence(self, masked_sentence: str = "", debug: bool = False) -> List:
        """
        Generate or complete a sentence.

        Algorithm (per MULTIHIERARCHY.md line 83):
          1. Sample a complex context instance (sentence root — empty context,
             high complexity).
          2. Read its content-ref (-1) → find that content-hierarchy leaf.
          3. get_basic() on that content leaf → sample a new leaf from the
             basic-level subtree.
          4. Read the sampled content leaf's left/right path attrs (deepest
             non-empty depth, i.e. most specific concept).
             These are references to CONTEXT hierarchy nodes (for composites)
             or words (for primitives).
          5. For each path_vid:
             - word → PrimitiveParseNode (BASE CASE, terminal)
             - CONCEPT-<hash> → find that CONTEXT node → read ITS -1
               (content-ref) → go to step 2 (RECURSE)
          6. Repeat until all leaves are words (sentence complete).

        Returns ``[generated_text, FiniteParseTree]``.
        """

        _cl  = self.context_length
        _ref_attr  = self.ltm.content_ref_attr  # content-ref attribute index
        _cplx_attr = -2                   # complexity (hidden, negative key)

        # ── helpers ───────────────────────────────────────────────────────

        def _name(vid):
            """Vocab ID → string."""
            if isinstance(vid, str):
                return vid
            try:
                if vid is not None and 0 <= vid < len(self.id_to_value):
                    return self.id_to_value[vid]
            except Exception:
                pass
            return None

        def _is_concept(vid):
            n = _name(vid)
            return n and isinstance(n, str) and n.startswith("CONCEPT-")

        def _is_word(vid):
            n = _name(vid)
            if n is None or n == "EMPTYNULL":
                return False
            return not (isinstance(n, str) and n.startswith("CONCEPT-"))

        # ── hash → node index (built once per call) ──────────────────────

        def _build_index(root):
            out = {}
            def walk(n):
                out[str(n.concept_hash())] = n
                for c in n.children:
                    walk(c)
            walk(root)
            return out

        ctx_index = _build_index(self.ltm.context_hierarchy.root)
        cnt_index = _build_index(self.ltm.content_hierarchy.root)

        # Also build node-ID index for resilience against hash changes
        ctx_id_index = {}
        for h, n in ctx_index.items():
            nid = h.rsplit('_', 1)[-1]
            ctx_id_index[nid] = n
        cnt_id_index = {}
        for h, n in cnt_index.items():
            nid = h.rsplit('_', 1)[-1]
            cnt_id_index[nid] = n

        def _find_ctx(concept_str):
            """CONCEPT-<hash> → context hierarchy node."""
            if not concept_str or not concept_str.startswith("CONCEPT-"):
                return None
            h = concept_str[len("CONCEPT-"):]
            node = ctx_index.get(h)
            if node is None:
                nid = h.rsplit('_', 1)[-1]
                node = ctx_id_index.get(nid)
            return node

        def _find_cnt(concept_str):
            """CONCEPT-<hash> → content hierarchy node."""
            if not concept_str or not concept_str.startswith("CONCEPT-"):
                return None
            h = concept_str[len("CONCEPT-"):]
            node = cnt_index.get(h)
            if node is None:
                nid = h.rsplit('_', 1)[-1]
                node = cnt_id_index.get(nid)
            return node

        # ── read content-ref from a context node ─────────────────────────

        def _read_content_ref(ctx_node, prefer_concept=False):
            """Read the dominant content-ref string from ctx_node's -1 attr.

            Args:
                ctx_node: Context hierarchy node to read from.
                prefer_concept: If True, only sample from CONCEPT refs
                    (composite level). Falls back to parent nodes if the
                    leaf has no CONCEPT refs. This ensures basic-level
                    sampling is used for masked completion.

            Returns (ref_string, is_word) or (None, False)."""
            if ctx_node is None:
                return None, False

            node = ctx_node
            # When prefer_concept, walk up to find a node with CONCEPT refs
            max_walk = 20
            for _ in range(max_walk):
                rd = (node.av_count or {}).get(_ref_attr, {})
                pool = {}
                for vid, w in rd.items():
                    if vid == 0:
                        continue
                    n = _name(vid)
                    if n is None or n == "EMPTYNULL":
                        continue
                    if prefer_concept and not (isinstance(n, str) and
                                               n.startswith("CONCEPT-")):
                        continue  # skip word-level refs
                    pool[vid] = w
                if pool:
                    break
                # No suitable refs at this node — try parent
                if prefer_concept and hasattr(node, 'parent') and node.parent:
                    node = node.parent
                else:
                    break

            if not pool:
                # Fallback: if prefer_concept found nothing, try without filter
                if prefer_concept:
                    return _read_content_ref(ctx_node, prefer_concept=False)
                return None, False

            chosen = random.choices(
                list(pool.keys()),
                weights=[max(x, 1e-12) for x in pool.values()],
                k=1)[0]
            n = _name(chosen)
            is_w = not (isinstance(n, str) and n.startswith("CONCEPT-"))
            return n, is_w

        # ── read complexity ──────────────────────────────────────────────

        def _read_cplx(ctx_node):
            """Decode the dominant complexity level from a context node.

            Complexity is stored as ``{C{X}_vid: count}`` at key ``_cplx_attr``;
            we pick the C{X} identifier with the highest observed count and
            return X (the integer complexity value).
            """
            if ctx_node is None:
                return 1
            av = ctx_node.av_count or {}
            cplx_data = av.get(_cplx_attr, {})
            if not cplx_data:
                return 1
            best_cplx, best_weight = 1, 0
            for vid, weight in cplx_data.items():
                if vid == 0:
                    continue
                cplx_str = _name(vid)
                if cplx_str and isinstance(cplx_str, str) and cplx_str.startswith("C"):
                    try:
                        c = int(cplx_str[1:])
                        if weight > best_weight:
                            best_weight = weight
                            best_cplx = c
                    except ValueError:
                        pass
            return max(best_cplx, 1)

        # ── sample from content hierarchy via basic-level ────────────────

        def _basic_sample(cnt_node):
            """get_basic / get_basic_instance_pmi → sample a leaf from the basic-level subtree."""
            _bl_eval_alpha = self.content_bl_alpha if self.content_bl_alpha is not None else -1.0
            if getattr(self, 'instance_basic_level', False):
                _inst = {a: {max(v, key=v.get): 1.0} for a, v in cnt_node.av_count.items()} if cnt_node.av_count else {}
                basic = cnt_node.get_basic_instance_pmi(_inst, debug=True, eval_alpha=_bl_eval_alpha)
            else:
                basic = cnt_node.get_basic(100, 1000, debug=True, eval_alpha=_bl_eval_alpha)
            # basic = cnt_node.get_best(cnt_node.av_count)
            # basic = cnt_node.tree.categorize(cnt_node.av_count).get_best(cnt_node.av_count)
            # basic = cnt_node
            cnt_root_h = str(self.ltm.content_hierarchy.root.concept_hash())
            if str(basic.concept_hash()) == cnt_root_h:
                raise ValueError("BASIC LEVEL = ROOT????")
            leaf = basic
            while leaf.children:
                leaf = random.choices(
                    leaf.children,
                    weights=[max(c.count, 1) for c in leaf.children],
                    k=1)[0]
            return leaf

        def _sample_path(cnt_leaf, attr_idx):
            """Sample a single path vid from content leaf at attr_idx (0=left, 1=right)."""
            rd = (cnt_leaf.av_count or {}).get(attr_idx, {})
            pool = {v: w for v, w in rd.items() if v != 0}
            if not pool:
                return 0
            return random.choices(
                list(pool.keys()),
                weights=[max(x, 1e-12) for x in pool.values()],
                k=1)[0]

        # ── context instance builders ────────────────────────────────────

        _wt_mode = getattr(self.ltm, 'weighting', 'binary')
        _ew = getattr(self.ltm, 'empty_weighting', False)
        _ev = lambda j: _context_weight(j, _wt_mode) if _ew else 0

        def _empty_ctx(cplx=1):
            ctx = {}
            for j in range(_cl):
                ctx[j]       = {0: _ev(j)}
                ctx[_cl + j] = {0: _ev(j)}
            ctx[_cplx_attr] = {_get_or_register_cplx_vid(cplx, self.ltm.id_to_value, self.ltm.value_to_id): 1}
            return ctx

        def _seeded_ctx(pos, known, cplx):
            ctx = {}
            for j in range(_cl):
                s = pos - (j + 1)
                if 0 <= s < len(known) and known[s]:
                    ctx[j] = {known[s]: _context_weight(j, _wt_mode)}; ctx[j][0] = 0
                else:
                    ctx[j] = {0: _ev(j)}
            for j in range(_cl):
                s = pos + (j + 1)
                if 0 <= s < len(known) and known[s]:
                    ctx[_cl+j] = {known[s]: _context_weight(j, _wt_mode)}; ctx[_cl+j][0] = 0
                else:
                    ctx[_cl+j] = {0: _ev(j)}
            ctx[_cplx_attr] = {_get_or_register_cplx_vid(cplx, self.ltm.id_to_value, self.ltm.value_to_id): 1}
            return ctx

        # ── flatten to word list ─────────────────────────────────────────

        def _words(root):
            ps = []
            def dfs(n):
                if isinstance(n, PrimitiveParseNode):
                    ps.append(n)
                for _, ch in getattr(n, "children", []):
                    dfs(ch)
            dfs(root)
            ps.sort(key=lambda p: (p.position_idx or 0))
            return [(_name(p.word_id) or "?") for p in ps]

        # ══════════════════════════════════════════════════════════════════
        #  _expand: content-ref → recursively build parse tree
        #
        #  This is the core generation loop. Given a content-ref (from a
        #  context node's -1 attribute), it:
        #    1. Finds the content node
        #    2. get_basic → sample a content leaf
        #    3. Reads left/right path_vids from that leaf
        #    4. For each path_vid:
        #       - word → PrimitiveParseNode (terminal)
        #       - CONCEPT-hash → find CONTEXT node → read ITS -1 → recurse
        # ══════════════════════════════════════════════════════════════════

        def _expand(content_ref_str, position, depth=0, max_depth=40):
            """Expand a content-ref string into a parse subtree.

            Args:
                content_ref_str: Either a word (terminal) or CONCEPT-<hash>
                    referencing a content-hierarchy node.
                position: The position index for this node.
                depth: Current recursion depth.

            Returns:
                (node, [all_nodes]) where node is the root of the subtree.
            """
            if depth > max_depth:
                raise RuntimeError(
                    f"Recursion depth {depth} > {max_depth}. "
                    f"Cycle in content-refs? ref={content_ref_str}")

            # ── BASE CASE: content-ref is a word ──
            if not (isinstance(content_ref_str, str) and
                    content_ref_str.startswith("CONCEPT-")):
                wid = self.value_to_id.get(content_ref_str, 0)
                if wid == 0:
                    raise RuntimeError(
                        f"Word '{content_ref_str}' not in vocabulary")
                if debug:
                    print(f"  {'  '*depth}→ word: {content_ref_str}")
                prim = PrimitiveParseNode.create_node(
                    context_instance=_empty_ctx(1),
                    label={wid: 1},
                    position_idx=position,
                    word_id=wid)
                return prim, [prim]

            # ── RECURSIVE CASE: content-ref is CONCEPT-<hash> ──
            cnt_node = _find_cnt(content_ref_str)
            if cnt_node is None:
                raise RuntimeError(
                    f"Content node not found for {content_ref_str[:50]}. "
                    f"Stale content-ref.")

            # get_basic → sample a leaf
            sampled_leaf = _basic_sample(cnt_node)

            # Read left (attr 0) and right (attr 1) from sampled content leaf
            left_pv  = _sample_path(sampled_leaf, 0)
            right_pv = _sample_path(sampled_leaf, 1)

            left_name  = _name(left_pv)
            right_name = _name(right_pv)

            if debug:
                print(f"  {'  '*depth}expand: L={left_name}  R={right_name}")

            # Create composite node for this level
            comp = CompositeParseNode()
            comp.position_idx = position
            comp.complexity = depth + 2
            comp.context_length = _cl
            all_nodes = [comp]

            # ── Resolve LEFT child ──
            left_child_ref = _resolve_path_vid(left_pv, depth + 1)
            left_node, left_sub = _expand(
                left_child_ref, position - 0.25 / (2 ** depth),
                depth=depth + 1)
            left_node.set_parent(comp)
            all_nodes.extend(left_sub)

            # ── Resolve RIGHT child ──
            right_child_ref = _resolve_path_vid(right_pv, depth + 1)
            right_node, right_sub = _expand(
                right_child_ref, position + 0.25 / (2 ** depth),
                depth=depth + 1)
            right_node.set_parent(comp)
            all_nodes.extend(right_sub)

            return comp, all_nodes

        def _resolve_path_vid(path_vid, depth):
            """Resolve a path_vid to a content-ref string.

            path_vid is from a content leaf's attribute. It's either:
              - A word vocab ID → return the word string (terminal)
              - A CONCEPT-<hash> → this references a CONTEXT node.
                Read that context node's -1 (content-ref) and return it.
            """
            ref_name = _name(path_vid) if path_vid else None

            if not ref_name or ref_name == "EMPTYNULL" or path_vid == 0:
                raise RuntimeError(
                    f"Invalid path_vid={path_vid} at depth={depth}. "
                    f"Incomplete training data.")

            # Word → return directly
            if not (isinstance(ref_name, str) and ref_name.startswith("CONCEPT-")):
                return ref_name

            # CONCEPT → find context node → read its content-ref
            ctx_node = _find_ctx(ref_name)
            if ctx_node is None:
                raise RuntimeError(
                    f"Context node not found for path {ref_name[:50]}. "
                    f"Split propagation bug or stale vocabulary.")

            # If this context node has children (was split after training),
            # go to a leaf for more specific content-ref
            if ctx_node.children:
                cat_leaf = self.ltm.context_hierarchy.categorize(
                    _empty_ctx(depth))
                if cat_leaf and not cat_leaf.children:
                    ctx_node = cat_leaf

            content_ref, is_word = _read_content_ref(ctx_node)
            if content_ref is None:
                raise RuntimeError(
                    f"No content-ref on context node "
                    f"...{str(ctx_node.concept_hash())[-12:]}. "
                    f"Train with threshold='converge'.")

            if debug:
                kind = "word" if is_word else "concept"
                print(f"  {'  '*depth}ctx→content-ref: {content_ref[:40]} ({kind})")

            return content_ref

        # ── collect sentence-root context leaves ─────────────────────────

        def _sentence_root_ctx_leaves():
            """Return [(node, complexity)] for context leaves whose context
            slots are all-empty (sentence roots)."""
            results = []
            def walk(n):
                if not n.children:
                    av = n.av_count or {}
                    is_sent = all(
                        not any(v != 0 for v in av.get(a, {}))
                        for a in range(2 * _cl))
                    if not is_sent:
                        return
                    # Must have a content-ref
                    rd = av.get(_ref_attr, {})
                    has_ref = any(v != 0 and _name(v) and _name(v) != "EMPTYNULL"
                                 for v in rd)
                    if not has_ref:
                        return
                    cplx = _read_cplx(n)
                    results.append((n, cplx))
                for c in n.children:
                    walk(c)
            walk(self.ltm.context_hierarchy.root)
            return results

        # ══════════════════════════════════════════════════════════════════
        #  MASKED SENTENCE COMPLETION
        #
        #  Per MULTIHIERARCHY.md:
        #    "For generation with masked language, we do the same generation
        #     process above for each masked token (but we use surrounding
        #     context to find the initial context-hierarchy node as well)"
        #    "build a new composite node next to the child node, and then
        #     use the context from previous parse to predict what that new
        #     composite node's content is, and then decompose it in the
        #     same way we would decompose a new generation from scratch"
        #
        #  1. Parse known (non-mask) tokens to build structure.
        #  2. For each [mask], determine composite-level complexity from
        #     adjacent parsed structure.
        #  3. Categorize in context hierarchy at that complexity (NOT
        #     primitive complexity=1) → get a CONCEPT content-ref.
        #  4. Expand with _expand (basic-level sampling — same as
        #     from-scratch generation).
        # ══════════════════════════════════════════════════════════════════
        if masked_sentence:
            tokens = re.findall(r"[\w']+|[.,!?;]|\[mask\]", masked_sentence)
            mask_pos = [i for i, t in enumerate(tokens) if t == "[mask]"]
            if debug:
                print(f"Tokens: {tokens}\nMask positions: {mask_pos}")

            # word-id lookup for known positions (used for context slots)
            known = [None if t == "[mask]" else self.value_to_id.get(t, 0)
                     for t in tokens]

            # ── Step 1: Parse known tokens to understand structure ────────
            known_only = [t for t in tokens if t != "[mask]"]
            partial_tree = None
            if known_only:
                try:
                    partial_tree = self.parse_sentence(
                        " ".join(known_only),
                        threshold="converge",
                        learning=False,
                        debug=False)
                except Exception:
                    pass

            # Map original token positions → top-level parsed node
            orig_pos_to_top: Dict[int, Any] = {}
            if partial_tree:
                _prims = []
                def _collect_pt_prims(n):
                    if isinstance(n, PrimitiveParseNode):
                        _prims.append(n)
                    for _, ch in getattr(n, "children", []):
                        _collect_pt_prims(ch)
                _collect_pt_prims(partial_tree.global_root_node)
                _prims.sort(key=lambda p: p.position_idx)

                known_indices = [i for i, t in enumerate(tokens)
                                 if t != "[mask]"]
                for k_idx, prim in enumerate(_prims):
                    if k_idx < len(known_indices):
                        orig_pos = known_indices[k_idx]
                        # Walk up to the top-level ancestor
                        node = prim
                        while (hasattr(node, "parent") and node.parent and
                               not getattr(node.parent, "is_global_root",
                                           False)):
                            node = node.parent
                        orig_pos_to_top[orig_pos] = node

            # ── Step 2: For each mask, determine complexity & generate ───
            results: Dict[int, List[str]] = {}
            for mi in mask_pos:
                if debug:
                    print(f"\n--- [mask] at {mi} ---")

                # Check whether the mask has known tokens on BOTH sides
                # (mid-sentence) vs only one side (end/start of sentence).
                has_left_ctx  = mi > 0 and known[mi - 1] is not None
                has_right_ctx = (mi < len(known) - 1
                                 and known[mi + 1] is not None)
                is_mid_sentence = has_left_ctx and has_right_ctx

                if is_mid_sentence:
                    # Mid-sentence mask: the mask most likely replaces a
                    # small number of words.  Use primitive-level
                    # complexity so the context hierarchy returns a
                    # word-level content-ref (not a high-complexity
                    # CONCEPT that expands into a whole phrase).
                    target_cplx = 1
                    use_prefer_concept = False
                else:
                    # End/start of sentence (or adjacent to another mask):
                    # use adjacent parsed structure to infer how large the
                    # generated subtree should be.
                    adj_cplx = 1
                    for delta in [-1, 1]:
                        adj = mi + delta
                        if adj in orig_pos_to_top:
                            c = getattr(orig_pos_to_top[adj],
                                        "complexity", 1)
                            adj_cplx = max(adj_cplx, c)
                    target_cplx = adj_cplx + 1
                    use_prefer_concept = True

                # Build context (word-ids for context slots, composite
                # complexity) — same structure as training context instances.
                ctx = _seeded_ctx(mi, known, target_cplx)

                if debug:
                    print(f"  Target complexity: {target_cplx}"
                          f"  (mid-sentence: {is_mid_sentence})")

                # Categorize in context hierarchy
                ctx_leaf = self.ltm.context_hierarchy.categorize(ctx)
                if ctx_leaf is None:
                    results[mi] = ["?"]
                    continue

                ref, is_word = _read_content_ref(ctx_leaf,
                                               prefer_concept=use_prefer_concept)
                if ref is None:
                    results[mi] = ["?"]
                    continue

                if debug:
                    kind = "word" if is_word else "concept"
                    print(f"  content-ref: {ref[:40]} ({kind})")

                # Expand with _expand — same basic-level sampling as
                # from-scratch generation.
                try:
                    mr = CompositeParseNode.create_global_root()
                    node, sub = _expand(ref, float(mi), depth=0)
                    node.set_parent(mr)
                    results[mi] = _words(mr)
                except RuntimeError as e:
                    if debug:
                        print(f"  Failed: {e}")
                    results[mi] = ["?"]

            out = []
            for i, t in enumerate(tokens):
                if t == "[mask]":
                    out.extend(results.get(i, ["?"]))
                else:
                    out.append(t)
            gen_text = " ".join(out)
            pt = self.parse_sentence(gen_text, threshold="converge",
                                     learning=False, debug=debug)
            return [gen_text, pt]

        # ══════════════════════════════════════════════════════════════════
        #  GENERATION FROM SCRATCH
        #
        #  Per MULTIHIERARCHY.md line 83:
        #  "sample a complex context instance, find the leaf which
        #   corresponds to its content-ref, find the basic level node of
        #   that leaf, sample a new leaf from that node, expand its two
        #   content elements as new nodes by using PATH INFORMATION to
        #   traverse the CONTEXT HIERARCHY, and repeat this process until
        #   words terminate as sentences!!!"
        # ══════════════════════════════════════════════════════════════════
        else:
            if debug:
                print("=== GENERATION FROM SCRATCH ===")

            # Find sentence-root context leaves (empty context, has content-ref)
            sent_roots = _sentence_root_ctx_leaves()
            if not sent_roots:
                raise RuntimeError(
                    "No sentence-root context leaves found. "
                    "Train on more data before generating.")

            # Prefer high-complexity roots (full sentences)
            nodes, complexities = zip(*sent_roots)
            weights = [c ** 2 for c in complexities]

            chosen_idx = random.choices(
                range(len(nodes)),
                weights=[max(w, 1e-12) for w in weights],
                k=1)[0]
            chosen_ctx = nodes[chosen_idx]
            chosen_cplx = complexities[chosen_idx]

            if debug:
                print(f"  Sampled context leaf: "
                      f"...{str(chosen_ctx.concept_hash())[-12:]} "
                      f"complexity={chosen_cplx}")

            # Read content-ref from the chosen context leaf
            content_ref, is_word = _read_content_ref(chosen_ctx)
            if content_ref is None:
                raise RuntimeError(
                    f"No content-ref on chosen context leaf. "
                    f"Train with threshold='converge'.")

            if debug:
                kind = "word" if is_word else "concept"
                print(f"  content-ref: {content_ref[:40]} ({kind})")

            # Expand: unpack the content-ref recursively
            global_root = CompositeParseNode.create_global_root()
            try:
                root_node, all_nodes = _expand(content_ref, 0.5, depth=0)
                root_node.set_parent(global_root)
            except RuntimeError as e:
                if debug:
                    print(f"  Expansion failed: {e}")
                raise

            gen_text = " ".join(_words(global_root))
            if debug:
                print(f"\nGenerated: {gen_text}")

            fp = FiniteParseTree(self.ltm, self.context_length)
            fp.window = gen_text
            fp.global_root_node = global_root
            fp.nodes = all_nodes
            return [gen_text, fp]


    # ---- visualization --------------------------------------------------

    def visualize_ltm(self, out_base="ltm", max_depth=1e9):
        """Visualize both hierarchies."""
        self.ltm.visualize_content_hierarchy(f"{out_base}_content", max_depth=max_depth)
        self.ltm.visualize_context_hierarchy(f"{out_base}_context", max_depth=max_depth)

    # ---- save / load ----------------------------------------------------

    def save_state(self, dirpath: str) -> dict:
        os.makedirs(dirpath, exist_ok=True)
        meta = {
            "context_length": self.context_length,
            "threshold": self.threshold,
            "primitive_threshold": getattr(self, 'primitive_threshold', None),
            "content_bl_alpha": getattr(self, 'content_bl_alpha', None),
            "context_bl_alpha": getattr(self, 'context_bl_alpha', None),
            "bow": self.bow,
            "categorization_mode": getattr(self, 'categorization_mode', 'dfs'),
            "weighting": getattr(self, 'weighting', 'binary'),
            "empty_weighting": getattr(self, 'empty_weighting', False),
            "use_observation_buffer": getattr(self, 'use_observation_buffer', False),
            "obs_buffer_flush_every": getattr(self, 'obs_buffer_flush_every', 5),
        }
        meta_path = os.path.join(dirpath, "webster_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        ltm_dir = os.path.join(dirpath, "ltm")
        ltm_result = self.ltm.save_state(ltm_dir)

        return {"ok": True, "webster_meta": meta_path, "ltm": ltm_result}

    @staticmethod
    def load_state(dirpath: str) -> 'WEBSTER':
        meta_path = os.path.join(dirpath, "webster_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"webster_meta.json not found in {dirpath}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        ltm_dir = os.path.join(dirpath, "ltm")
        ltm = LongTermMemory.load_state(ltm_dir)

        w = WEBSTER.__new__(WEBSTER)
        w.ltm = ltm
        w.context_length = meta.get("context_length", 3)
        w.threshold = meta.get("threshold", 5)
        w.primitive_threshold = meta.get("primitive_threshold", None)
        w.content_bl_alpha = meta.get("content_bl_alpha", None)
        w.context_bl_alpha = meta.get("context_bl_alpha", None)
        w.bow = meta.get("bow", False)
        w.categorization_mode = meta.get("categorization_mode", "dfs")
        w.weighting = meta.get("weighting", "binary")
        w.empty_weighting = meta.get("empty_weighting", False)
        w.use_observation_buffer = meta.get("use_observation_buffer", False)
        w.obs_buffer_flush_every = meta.get("obs_buffer_flush_every", 5)
        w._obs_buffer = None
        w.ltm.categorization_mode = w.categorization_mode
        w.ltm.weighting = w.weighting
        w.ltm.empty_weighting = w.empty_weighting
        return w
