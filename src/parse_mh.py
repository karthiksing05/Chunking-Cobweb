"""

Primary module for multi-hierarchy theory! We're programming the whole thing from scratch,
going to try to leverage some of the old code but the parse trees need a large rework
especially regarding their visualization (context-hierarchy and content-hierarchy both need
to be worked).

See implementation details in MULTIHIERARCHY.md!

Key difference from parse.py (single-hierarchy):
    - TWO Cobweb hierarchies: one for content, one for context.
    - Content instances use **multi-attribute path encoding** with
      `content_path_depth` (default 3) depth levels per side:
          content_instance = {
              0: {left_depth0: 1},  # ROOT / most general (context root concept)
              1: {left_depth1: 1},  # deeper toward leaf (more specific)
              2: {left_depth2: 1},  # most specific (leaf-level concept)
              3: {right_depth0: 1},
              4: {right_depth1: 1},
              5: {right_depth2: 1},
          }
      Each attribute has exactly one value with count 1 so that
      log_prob_instance produces clean, comparable scores.
      Words sharing a context-hierarchy ancestor get identical values
      at that depth level, enabling category-level generalization.
    - Context instances carry sliding-window context + complexity:
          context_instance = {0..ctx_len-1: ctx_before,
                              ctx_len..2*ctx_len-1: ctx_after,
                              2*ctx_len: complexity,
                              2*ctx_len+1: content-ref}
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


def _score_along_path(
    node_path: List[CobwebDiscreteNode],
    instance: dict,
    tree: CobwebDiscreteTree,
    debug: bool = False,
) -> dict:
    """
    Compute recognition statistics along a categorization path.
    Mirrors FiniteParseTree._score_function from parse.py.
    Returns a dict of score metrics; the primary one is 'cost' (tree-wide log-prob).
    """
    raw_log_probs = []
    avg_log_probs = []
    path_counts = []
    best_lp = -float("inf")
    best_lp_idx = 0

    for i, node in enumerate(node_path):
        lp = node.log_prob_instance(instance)
        if math.isnan(lp) or lp == 0:
            lp = -1e8

        node_complexity = sum(
            cnt for attr_dict in node.av_count.values() for cnt in attr_dict.values()
        )
        inst_complexity = sum(
            cnt for attr_dict in instance.values() for cnt in attr_dict.values()
        )
        path_counts.append(node.count)

        avg_lp = lp / inst_complexity if inst_complexity else -1e8

        if lp > best_lp:
            best_lp = lp
            best_lp_idx = i

        raw_log_probs.append(lp)
        avg_log_probs.append(avg_lp)

    # weighted cost (leaf-biased)
    cost = 0.0
    coef = 1.0
    for lp in reversed(raw_log_probs):
        cost += lp * coef
        coef *= 0.33

    tree_log_prob = tree.log_prob(instance, 250, False)
    if tree_log_prob == 0:
        tree_log_prob = -1e9

    score_data = {
        "raw_node_log_probs": str(raw_log_probs),
        "candidate_counts": str(path_counts),
        "normed_log_prob": cost,
        "best_log_prob_idx": best_lp_idx,
        "cost": tree_log_prob,
        "tree_log_prob": tree_log_prob,
        "best_log_prob": best_lp,
    }

    if debug:
        print("-" * 60)
        print("Scoring statistics:")
        pprint(score_data)
        print("-" * 60)

    return score_data


# ---------------------------------------------------------------------------
# Shared helpers for multi-depth label_path construction
# ---------------------------------------------------------------------------

def _build_label_path_from_ctx(path_strs: list, value_to_id: dict,
                               content_path_depth: int) -> list:
    """
    Build a multi-depth label_path from a context-hierarchy categorization
    path.

    Every entry is a context-hierarchy concept vocab ID — **never** a raw
    word ID.  Words are stored separately in the context hierarchy's
    content-ref attribute (``2*ctx_len + 1``); the content hierarchy
    organises purely by structural concept paths.

    Parameters
    ----------
    path_strs : list[str]
        The categorization path from root to leaf (e.g. ["CONCEPT-aaa",
        "CONCEPT-bbb", "CONCEPT-ccc"]).
    value_to_id : dict
        Vocabulary mapping (str → int).
    content_path_depth : int
        How many depth levels to include.

    Returns
    -------
    list[int] of length *content_path_depth*.
        [depth_0, depth_1, depth_2, ...] from most-general (root) to
        most-specific (towards leaf).  Padded with 0 (EMPTYNULL) if the
        path is shorter than *content_path_depth*.
    """
    # Root-first: depth-0 = root (most general), deeper = more specific.
    cpd = content_path_depth

    lp = []
    for s in path_strs[:cpd]:
        v = value_to_id.get(s)
        lp.append(v if v is not None else 0)

    # Pad to exactly content_path_depth
    while len(lp) < cpd:
        lp.append(0)
    return lp[:cpd]


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
        label_path        – multi-depth list [word_id, parent_concept, grandparent, ...]
                            used to build content instances with multi-attribute
                            path encoding for category-level generalization.

    Attributes
    ----------
    parent : CompositeParseNode | None
    children : SortedList          always empty for primitives
    position_idx : int             word position in the sentence
    title : str                    unique random id
    context_instance : dict        instance dict for the context hierarchy
    label : dict                   {word_id: 1}
    label_path : list              [word_id, ctx_parent_vid, ctx_grandparent_vid, ...]
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
        self.label_path: list = []  # multi-depth path for content hierarchy
        self.word_id: int = word_id

        self.complexity: int = 1
        self.score_data: dict = {}
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
        self.label_path: list = []  # [concept_leaf, parent, grandparent, ...]
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
    def create_content_instance(left_node, right_node, content_path_depth: int = 1) -> dict:
        """
        Build the content-hierarchy instance from two children using
        multi-attribute path encoding.

        Layout (2 * content_path_depth attributes):
          Attrs  0 .. cpd-1   : left side  (depth 0 = ROOT / most general,
                                             deeper = more specific toward leaf)
          Attrs cpd .. 2*cpd-1: right side  (same depth ordering)

        For primitives, the deepest non-zero depth holds the most specific
        CONCEPT-<hash> (leaf-level context concept).
        For composites, same structure — all CONCEPT refs, no raw words.

        Each attribute has exactly one value with count 1, keeping
        log_prob_instance clean and comparable.
        """
        left_path = getattr(left_node, 'label_path', None)
        right_path = getattr(right_node, 'label_path', None)

        # Fallback for nodes without label_path (backward compat)
        if not left_path:
            lbl = left_node.get_label()
            left_path = [next(iter(lbl.keys()), 0)]
        if not right_path:
            lbl = right_node.get_label()
            right_path = [next(iter(lbl.keys()), 0)]

        cpd = content_path_depth
        inst = {}
        for i in range(cpd):
            val_l = left_path[i] if i < len(left_path) else 0
            inst[i] = {val_l: 1}

            val_r = right_path[i] if i < len(right_path) else 0
            inst[cpd + i] = {val_r: 1}

        return inst

    # ------------------------------------------------------------------
    @staticmethod
    def create_context_instance(left_node, right_node, context_length: int,
                                content_ref_id: int = None,
                                complexity_vid: int = 0) -> dict:
        """
        Build the context-hierarchy instance from two children.
        Attributes:
            0 .. context_length-1       : context_before (from left_node)
            context_length .. 2*ctx-1   : context_after  (from right_node)
            2*context_length            : complexity – stored as
                                          {complexity_vid: actual_complexity_int}
            -1                          : content-ref (content hierarchy leaf
                                          concept vocab id, for generation)
                                          Hidden (negative index) so it does
                                          not influence Cobweb categorization.

        Child expansion information is stored in the **content** hierarchy
        via path attributes (label_path of left/right children).  No
        child-ref hidden attributes are needed in the context instance.

        Parameters
        ----------
        complexity_vid : int
            Vocab ID for the "COMPLEXITY" sentinel.
        """
        ctx_inst: dict = {}

        # context_before from the left child
        left_ctx_before = getattr(left_node, "context_before", None) or []
        for j in range(context_length):
            if j < len(left_ctx_before) and left_ctx_before[j]:
                ctx_inst[j] = {k: 1.0 / (2 ** (j + 1)) for k in left_ctx_before[j]}
                ctx_inst[j][0] = 0
            else:
                ctx_inst[j] = {0: 1.0 / (2 ** (j + 1))}

        # context_after from the right child
        right_ctx_after = getattr(right_node, "context_after", None) or []
        for j in range(context_length):
            attr_key = context_length + j
            if j < len(right_ctx_after) and right_ctx_after[j]:
                ctx_inst[attr_key] = {k: 1.0 / (2 ** (j + 1)) for k in right_ctx_after[j]}
                ctx_inst[attr_key][0] = 0
            else:
                ctx_inst[attr_key] = {0: 1.0 / (2 ** (j + 1))}

        # complexity attribute – single COMPLEXITY sentinel, count = actual complexity
        left_c = getattr(left_node, "complexity", 1)
        right_c = getattr(right_node, "complexity", 1)
        complexity = max(left_c, right_c) + 1
        ctx_inst[2 * context_length] = {complexity_vid: complexity}

        # content-ref attribute: for composites this is the content
        # hierarchy leaf concept id; for primitives it's the word_id
        # (set separately in build_primitives)
        if content_ref_id is not None:
            ctx_inst[-1] = {content_ref_id: 1}

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
            node.context_before.append(context_instance.get(j, {0: 0}))
        node.context_after = []
        for j in range(context_length):
            node.context_after.append(context_instance.get(context_length + j, {0: 0}))

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

    def build_primitives(self, window: str, threshold=-7):
        """
        Tokenize *window* and create PrimitiveParseNode objects.
        Each primitive is categorized in the context hierarchy to obtain
        its label ({word_id: 1}) and label_path (multi-depth ancestor list).
        """
        self.window = window

        elements = re.findall(r"[\w']+|[.,!?;]", window)
        word_ids = [self.value_to_id[e] for e in elements]

        for i, wid in enumerate(word_ids):
            # build sliding-window context instance for context hierarchy
            ctx_inst: dict = {}

            # context_before
            for j in range(self.context_length):
                src_idx = i - (j + 1)
                if 0 <= src_idx < len(word_ids):
                    ctx_inst[j] = {word_ids[src_idx]: 1.0 / (2 ** (j + 1))}
                    ctx_inst[j][0] = 0
                else:
                    ctx_inst[j] = {0: 1.0 / (2 ** (j + 1))}

            # context_after
            for j in range(self.context_length):
                src_idx = i + (j + 1)
                attr_key = self.context_length + j
                if 0 <= src_idx < len(word_ids):
                    ctx_inst[attr_key] = {word_ids[src_idx]: 1.0 / (2 ** (j + 1))}
                    ctx_inst[attr_key][0] = 0
                else:
                    ctx_inst[attr_key] = {0: 1.0 / (2 ** (j + 1))}

            # complexity = 1 for primitives – single COMPLEXITY value, count = 1
            _cplx_vid = self.value_to_id.get("COMPLEXITY", 0)
            ctx_inst[2 * self.context_length] = {_cplx_vid: 1}

            # word identity attribute – enables generation to recover
            # the actual word from a context hierarchy leaf.
            # Hidden (negative index) so it doesn't influence categorization.
            ctx_inst[-1] = {wid: 1}

            # categorize in context hierarchy to get label path
            leaf_node, path_strs, node_path = _categorize_dfs(ctx_inst, self.ltm.context_hierarchy)

            # Discrete single-identity label: primitive's identity is its word_id
            label = {wid: 1}

            # Build label_path: root-first [CONCEPT-root, ..., CONCEPT-leaf]
            # All entries are context-hierarchy concept refs — NO raw words.
            # The word itself is stored in the context hierarchy's -1 attr.
            label_path = _build_label_path_from_ctx(
                path_strs, self.value_to_id, self.ltm.content_path_depth
            )

            node = PrimitiveParseNode.create_node(ctx_inst, label, position_idx=i, word_id=wid)
            node.label_path = label_path

            # build context_before / context_after lists of dicts for visualization
            cb = []
            for j in range(self.context_length):
                src_idx = i - (j + 1)
                if 0 <= src_idx < len(word_ids):
                    cb.append({word_ids[src_idx]: 1})
                else:
                    cb.append({0: 1})
            node.context_before = cb

            ca = []
            for j in range(self.context_length):
                src_idx = i + (j + 1)
                if 0 <= src_idx < len(word_ids):
                    ca.append({word_ids[src_idx]: 1})
                else:
                    ca.append({0: 1})
            node.context_after = ca

            # score
            score_data = _score_along_path(node_path, ctx_inst, self.ltm.context_hierarchy)
            node.score_data = score_data

            if threshold == "converge":
                node.stable = True
            else:
                node.stable = score_data.get("cost", -1e8) > threshold

            node.set_parent(self.global_root_node)
            self.nodes.append(node)

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

    def evaluate_pair(self, left_word_index, right_word_index, debug=False) -> dict:
        """
        Evaluate merging two root-level children.
        Builds *both* content and context instances, categorizes each in its
        respective hierarchy, and returns scoring data.

        Order matches add_parse_tree: context-first WITHOUT -1, then content.
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        # Step 1: Build and categorize context WITHOUT -1 content-ref
        # (matching add_parse_tree behavior)
        _cplx_vid = self.value_to_id.get("COMPLEXITY", 0)
        context_inst = CompositeParseNode.create_context_instance(
            left_node, right_node, self.context_length,
            content_ref_id=None,  # no -1 during categorization
            complexity_vid=_cplx_vid,
        )

        # categorize in context hierarchy (for identity / label)
        ctx_leaf, ctx_path, ctx_node_path = _categorize_dfs(context_inst, self.ltm.context_hierarchy)

        # Step 2: Build and categorize content (for scoring)
        content_inst = CompositeParseNode.create_content_instance(left_node, right_node, self.ltm.content_path_depth)
        cnt_leaf, cnt_path, cnt_node_path = _categorize_dfs(content_inst, self.ltm.content_hierarchy)

        # Score using content hierarchy tree-wide log-probability.
        # Multi-attribute path encoding gives each attribute exactly one
        # value with count 1, so log_prob produces clean, comparable scores.
        score = self.ltm.content_hierarchy.log_prob(content_inst, 250, False)
        if score == 0:
            score = -1e9
        score_data = {
            "cost": score,
            "tree_log_prob": score,
        }
        if debug:
            print(f"Content score for pair: {score:.4f}")
            print(f"  content_inst: {content_inst}")

        # build label (weighted path from context hierarchy)
        ctx_path_ids = []
        for pstr in ctx_path:
            vid = self.value_to_id.get(pstr)
            if vid is not None:
                ctx_path_ids.append(vid)

        ctx_hash = ctx_leaf.concept_hash() if ctx_leaf else "unknown"
        ctx_concept_id = self.value_to_id.get(f"CONCEPT-{ctx_hash}")

        return {
            "content_inst": content_inst,
            "context_inst": context_inst,
            "categorize_path": ctx_path_ids,
            "candidate_concept_hash": ctx_hash,
            "candidate_concept_id": ctx_concept_id,
            "score": score,
            "debug": score_data,
            "left_word_index": left_word_index,
            "right_word_index": right_word_index,
            "left_title": left_node.title,
            "right_title": right_node.title,
        }

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
        _cplx_vid = self.value_to_id.get("COMPLEXITY", 0)
        context_inst = CompositeParseNode.create_context_instance(
            left_node, right_node, self.context_length,
            content_ref_id=None,  # no -1 during categorization
            complexity_vid=_cplx_vid,
        )

        # categorize in context hierarchy
        ctx_leaf, ctx_path, _ = _categorize_dfs(context_inst, self.ltm.context_hierarchy)

        # Step 2: Build content instance
        # (matching add_parse_tree step 3, but we don't categorize here —
        # add_parse_tree will rebuild content instances from refreshed labels)
        content_inst = CompositeParseNode.create_content_instance(left_node, right_node, self.ltm.content_path_depth)

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

        # Multi-depth label_path via helper
        _label_path = _build_label_path_from_ctx(
            ctx_path, self.value_to_id, self.ltm.content_path_depth
        )

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
        """
        self.window = window
        self.build_primitives(window, threshold=end_behavior)

        while True:
            pairs = self.get_parentless_pairs()
            if not pairs:
                break

            best = None
            for p in pairs:
                try:
                    res = self.evaluate_pair(p["left_word_index"], p["right_word_index"], debug=debug)
                except Exception as e:
                    if debug:
                        print(f"evaluate_pair failed: {e}")
                    continue
                score = res.get("score", -float("inf"))
                if best is None or score > best[0]:
                    best = (score, res)

            if best is None:
                break

            if isinstance(end_behavior, (int, float)):
                if best[0] < end_behavior:
                    break

            chosen = best[1]
            try:
                self.apply_candidate(
                    chosen["left_word_index"],
                    chosen["right_word_index"],
                    frozen=True,
                )
            except Exception as e:
                if debug:
                    print(f"apply_candidate failed: {e}")
                break

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
                ci = CompositeParseNode.create_content_instance(left, right, self.ltm.content_path_depth)
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

        Returns a dict with ``content_rows`` and ``context_rows``, each a
        flat list of ``{attr, val, count}`` dicts suitable for a single
        three-column table.
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

        if isinstance(node, PrimitiveParseNode):
            # Content rows: just the word identity
            content_rows = [{"attr": "Word", "val": self._safe_lookup(node.word_id), "count": "1.00"}]

            # Context rows: flatten context_instance attributes
            context_rows = []
            if node.context_instance:
                ctx_hdrs = (
                    [f"CtxBefore{i}" for i in range(self.context_length)]
                    + [f"CtxAfter{i}" for i in range(self.context_length)]
                    + ["Complexity"]
                )
                for attr_key in sorted(k for k in node.context_instance if k >= 0):
                    hdr = ctx_hdrs[attr_key] if attr_key < len(ctx_hdrs) else f"Attr{attr_key}"
                    context_rows.extend(_rows_from_ctx(node.context_instance[attr_key], hdr))
                # Hidden attr -1 (content-ref)
                if -1 in node.context_instance:
                    context_rows.extend(_rows_from_ctx(node.context_instance[-1], "ContentRef"))

            return {
                "title": node.title,
                "content_rows": content_rows,
                "context_rows": context_rows,
                "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
            }

        elif isinstance(node, CompositeParseNode):
            if node.is_global_root:
                return {
                    "title": "ROOT",
                    "content_rows": [],
                    "context_rows": [],
                    "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
                }

            _cpd = self.ltm.content_path_depth if self.ltm else 1

            # Content rows: Left0..N, Right0..N
            content_rows = []
            if node.content_instance:
                for i in range(_cpd):
                    content_rows.extend(_rows_from_ctx(node.content_instance.get(i, {}), f"Left{i}"))
                for i in range(_cpd):
                    content_rows.extend(_rows_from_ctx(node.content_instance.get(_cpd + i, {}), f"Right{i}"))

            # Context rows: flatten context_instance attributes
            context_rows = []
            if node.context_instance:
                ctx_hdrs = (
                    [f"CtxBefore{i}" for i in range(self.context_length)]
                    + [f"CtxAfter{i}" for i in range(self.context_length)]
                    + ["Complexity"]
                )
                for attr_key in sorted(k for k in node.context_instance if k >= 0):
                    hdr = ctx_hdrs[attr_key] if attr_key < len(ctx_hdrs) else f"Attr{attr_key}"
                    context_rows.extend(_rows_from_ctx(node.context_instance[attr_key], hdr))
                if -1 in node.context_instance:
                    context_rows.extend(_rows_from_ctx(node.context_instance[-1], "ContentRef"))

            return {
                "title": node.title,
                "content_rows": content_rows,
                "context_rows": context_rows,
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

    def _build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
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
.section-title {{ font-weight: bold; margin-top: 6px; font-size: 11px; color: #555; }}
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
  return `<div class="section-title">${{title}}</div>
    <table>
      <tr><th>Attr</th><th>Value</th><th>Count</th></tr>
      ${{body}}
    </table>`;
}}

function nodeHTML(d){{
  return `<div class="node-fo">
    <table><tr><th colspan="3">${{d.title}}</th></tr></table>
    ${{threeColTable("Content Instance", d.content_rows)}}
    ${{threeColTable("Context Instance", d.context_rows)}}
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
            {"title": n.title, "position_idx": n.position_idx, "score_data": n.score_data or {}}
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

    def editor_build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
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
background-color: #fff; margin: 10% auto; padding: 20px; border: 1px solid #888; width: 400px; border-radius:8px;
}}
.close {{ float:right; font-size: 18px; cursor: pointer; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
th, td {{ border: 1px solid #888; padding: 4px; font-size: 12px; }}
th {{ background: #f3f5f7; font-weight: 600; }}
.section-title {{ font-weight: bold; margin-top: 6px; font-size: 11px; color: #555; }}
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
            <div id="primitive-score-view"><i>Select a primitive to view its score data.</i></div>
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
    <table id="candidate-debug"></table>
    <button id="apply-candidate-btn">Apply Chunk</button>
</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
let treeData = {d3_data_json};
let currentLeft=null, currentRight=null;
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
    return `<div class="section-title">${{title}}</div>
        <table>
          <tr><th>Attr</th><th>Value</th><th>Count</th></tr>
          ${{body}}
        </table>`;
}}

function nodeHTML(d){{
    return `<div class="node-fo">
        <table><tr><th colspan="3">${{d.title}}</th></tr></table>
        ${{threeColTable("Content Instance", d.content_rows)}}
        ${{threeColTable("Context Instance", d.context_rows)}}
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
    for(const [k,v] of Object.entries(score)){{rows+=`<tr><td>${{k}}</td><td>${{formatScoreValue(v)}}</td></tr>`;}}
    return `<table><tr><th>Metric</th><th>Value</th></tr>${{rows}}</table>`;
}}
function renderPrimitiveScores(primitives){{
    const btnContainer=document.getElementById("primitive-score-buttons");
    const view=document.getElementById("primitive-score-view");
    if(!btnContainer||!view) return;
    btnContainer.innerHTML="";view.innerHTML="<i>Select a primitive to view its score data.</i>";
    primitives.forEach(p=>{{
        const btn=document.createElement("button");btn.textContent=p.title;
        btn.onclick=()=>{{view.innerHTML=buildScoreTable(p.score_data);}};
        btnContainer.appendChild(btn);
    }});
}}
function loadPairs(){{
    fetch("/api/tree").then(r=>r.json()).then(data=>{{
        const container=document.getElementById("pair-buttons");
        container.innerHTML="<strong>Candidate Pairs:</strong>";
        const s=document.getElementById("sentence-text");
        if(s&&data.sentence) s.textContent=data.sentence;
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
    .then(r=>r.json()).then(res=>{{if(res.ok) showCandidateModal(res.result); else alert(res.error);}});
}}
const modal=document.getElementById("candidate-modal");
const spanClose=modal.querySelector(".close");
spanClose.onclick=()=>modal.style.display="none";
window.onclick=e=>{{if(e.target==modal) modal.style.display="none";}};
function showCandidateModal(result){{
    document.getElementById("candidate-title").textContent=result.candidate_concept_id||result.candidate_concept_hash;
    document.getElementById("candidate-score").textContent=result.score.toFixed(3);
    const dbg=document.getElementById("candidate-debug");dbg.innerHTML="";
    function ctxTable(ctx,title){{
        if(!ctx||ctx.length===0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
        const rows=ctx.map(kv=>`<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
        return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
    }}
    function buildDebugHTML(debugObj){{
        let html=`<div class="subtable"><b>Debug Stats</b><table><tr><th>Stat</th><th>Value</th></tr>`;
        for(const [k,v] of Object.entries(debugObj)){{html+=`<tr><td>${{k}}</td><td>${{v===null?"null":v.toFixed?v.toFixed(3):v}}</td></tr>`;}}
        html+=`</table></div>`;return html;
    }}
    dbg.innerHTML=buildDebugHTML(result.debug);
    modal.style.display="block";
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
    Context hierarchy   – instances are {0..ctx_len-1: ctx_before,
                          ctx_len..2*ctx_len-1: ctx_after,
                          2*ctx_len: complexity,
                          2*ctx_len+1: content-ref (word_id for primitives,
                                       content leaf concept_id for composites)}.
    """

    def __init__(self, value_corpus: list, context_length: int = 3, content_path_depth: int = 3, alpha: float = 1e-4):
        self.content_hierarchy = CobwebDiscreteTree(alpha)
        self.context_hierarchy = CobwebDiscreteTree(alpha)

        # vocabulary: index 0 is always EMPTYNULL
        self.id_to_value: List[str] = ["EMPTYNULL"]
        for x in value_corpus:
            self.id_to_value.append(x)
        # Reserve a special vocab entry for the COMPLEXITY sentinel value
        self.id_to_value.append("COMPLEXITY")
        self.value_to_id: Dict[str, int] = {w: i for i, w in enumerate(self.id_to_value)}
        self.id_count: int = len(self.id_to_value) - 1

        self.context_length = context_length
        self.content_path_depth = content_path_depth

        # register root concepts of both hierarchies
        self._register_concept(self.content_hierarchy.root)
        self._register_concept(self.context_hierarchy.root)

        # drawer for content hierarchy visualization
        # Multi-attribute path encoding: one attr per depth level per side
        content_headers = (
            [f"Left-Depth{i}" for i in range(content_path_depth)]
            + [f"Right-Depth{i}" for i in range(content_path_depth)]
        )
        self.content_drawer = HTMLCobwebDrawer(
            content_headers,
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id,
        )

        # drawer for context hierarchy visualization
        context_headers = (
            [f"Context-Before{i}" for i in range(context_length)]
            + [f"Context-After{i}" for i in range(context_length)]
            + ["Complexity"]
        )
        content_ref_attr_idx = -1
        def _content_ref_display(val_id):
            if val_id is not None and 0 <= val_id < len(self.id_to_value):
                name = self.id_to_value[val_id]
            else:
                name = f"?{val_id}"
            if isinstance(name, str) and name.startswith("CONCEPT-"):
                return "C-" + name[8:20] + "…"
            return name
        context_attr_value_fn = {
            content_ref_attr_idx: _content_ref_display,
        }
        self.context_drawer = HTMLCobwebDrawer(
            context_headers,
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id,
            attr_value_fn=context_attr_value_fn,
            attr_name_overrides={content_ref_attr_idx: "Content-Ref"},
        )

    # ---- vocabulary helpers ---------------------------------------------

    def _register_concept(self, node: CobwebDiscreteNode):
        new_vocab = f"CONCEPT-{node.concept_hash()}"
        self.add_to_vocab(new_vocab)

    def add_to_vocab(self, new_vocab: str) -> bool:
        if new_vocab not in self.value_to_id:
            self.id_to_value.append(new_vocab)
            self.id_count += 1
            self.value_to_id[new_vocab] = self.id_count
            return True
        return False

    # ---- instance conversion helpers ------------------------------------

    def get_content_instance_statistics(self, content_inst: dict, debug=False) -> dict:
        """
        Categorize a content instance in the content hierarchy and return scoring data.
        """
        leaf, path, node_path = _categorize_dfs(content_inst, self.content_hierarchy)
        return _score_along_path(node_path, content_inst, self.content_hierarchy, debug=debug)

    def get_context_instance_statistics(self, context_inst: dict, debug=False) -> dict:
        """
        Categorize a context instance in the context hierarchy and return scoring data.
        """
        leaf, path, node_path = _categorize_dfs(context_inst, self.context_hierarchy)
        return _score_along_path(node_path, context_inst, self.context_hierarchy, debug=debug)

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

    def add_parse_tree(self, parse_tree: 'FiniteParseTree', debug=False):
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

        # -- Step 1: fit context instances WITHOUT -1 ---------------------
        ctx_leaf_map: dict = {}   # id(node) → context-hierarchy leaf

        for node in all_nodes:
            ctx_inst = node.get_context_instance()
            # Remove -1 if present (we'll add it in step 4)
            ctx_inst.pop(-1, None)
            leaf, rewrites = self._ifit_and_update_vocab(
                ctx_inst, self.context_hierarchy, debug=debug)
            ctx_leaf_map[id(node)] = leaf
            # Propagate context splits to content hierarchy
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
                # Primitive label_path: root-first CONCEPT path.
                # depth-0 = root (most general), deeper = more specific.
                # The word itself lives in the context hierarchy's -1 attr.
                lp_rev = []
                p = ctx_leaf
                while p is not None:
                    concept_str = f"CONCEPT-{p.concept_hash()}"
                    self.add_to_vocab(concept_str)
                    v = self.value_to_id.get(concept_str)
                    if v is not None:
                        lp_rev.append(v)
                    p = getattr(p, 'parent', None)
                # lp_rev is leaf→root; reverse to root→leaf (root-first).
                lp = list(reversed(lp_rev))
                # Keep root end, truncate deepest if too long.
                lp = lp[:self.content_path_depth]
                while len(lp) < self.content_path_depth:
                    lp.append(0)
                node.label_path = lp

            elif isinstance(node, CompositeParseNode) and not node.is_global_root:
                ctx_hash = ctx_leaf.concept_hash()
                concept_label_str = f"CONCEPT-{ctx_hash}"
                self.add_to_vocab(concept_label_str)
                new_concept_label = self.value_to_id.get(concept_label_str)
                node.concept_label = new_concept_label
                node.label = {new_concept_label: 1} if new_concept_label else {0: 1}

                # Composite label_path: root-first CONCEPT path.
                # depth-0 = root (most general), deeper = more specific.
                lp_rev = []
                p = ctx_leaf
                while p is not None:
                    concept_str = f"CONCEPT-{p.concept_hash()}"
                    self.add_to_vocab(concept_str)
                    v = self.value_to_id.get(concept_str)
                    if v is not None:
                        lp_rev.append(v)
                    p = getattr(p, 'parent', None)
                # lp_rev is leaf→root; reverse to root→leaf (root-first).
                lp = list(reversed(lp_rev))
                # Keep root end, truncate deepest if too long.
                lp = lp[:self.content_path_depth]
                while len(lp) < self.content_path_depth:
                    lp.append(0)
                node.label_path = lp

                # Rebuild content_instance from children's refreshed labels
                children_sorted = list(node.children)
                if len(children_sorted) == 2:
                    left_child = children_sorted[0][1]
                    right_child = children_sorted[1][1]
                    node.content_instance = CompositeParseNode.create_content_instance(
                        left_child, right_child, self.content_path_depth
                    )

        for _, ch in parse_tree.global_root_node.children:
            _refresh_labels_bottom_up(ch)

        if debug:
            print(f"  labels refreshed from context hierarchy")

        # -- Step 3: fit content instances --------------------------------
        content_leaf_map: dict = {}   # id(comp_node) → content-hierarchy leaf

        for node in all_nodes:
            if isinstance(node, CompositeParseNode) and not node.is_global_root:
                ci = node.get_content_instance()
                if ci:
                    leaf, rewrites = self._ifit_and_update_vocab(
                        ci, self.content_hierarchy, debug=debug)
                    content_leaf_map[id(node)] = leaf
                    # Propagate content splits to context hierarchy
                    if rewrites:
                        self._apply_rewrite_rules(self.context_hierarchy, rewrites)

        # Also fit orphan candidate pairs (parentless adjacent pairs)
        pairs = parse_tree.get_parentless_pairs()
        for p in pairs:
            left = parse_tree._find_root_child_by_index(p["left_word_index"])
            right = parse_tree._find_root_child_by_index(p["right_word_index"])
            if left and right:
                ci = CompositeParseNode.create_content_instance(
                    left, right, self.content_path_depth)
                _leaf, rewrites = self._ifit_and_update_vocab(
                    ci, self.content_hierarchy, debug=debug)
                if rewrites:
                    self._apply_rewrite_rules(self.context_hierarchy, rewrites)

        if debug:
            print(f"  content instances fitted: {len(content_leaf_map)}")

        # -- Step 4: write content-refs back to context hierarchy ---------
        # For each node, update its context leaf's av_count to include
        # -1 = content-ref. Use set_av_count() to write directly.
        for node in all_nodes:
            ctx_leaf = ctx_leaf_map.get(id(node))
            if ctx_leaf is None:
                continue

            # Determine the content-ref value
            if isinstance(node, PrimitiveParseNode):
                # Primitive: content-ref = word_id
                ref_vid = node.word_id
            elif isinstance(node, CompositeParseNode) and not node.is_global_root:
                cnt_leaf = content_leaf_map.get(id(node))
                if cnt_leaf is None:
                    continue
                cnt_ref_str = f"CONCEPT-{cnt_leaf.concept_hash()}"
                self.add_to_vocab(cnt_ref_str)
                ref_vid = self.value_to_id.get(cnt_ref_str, 0)
                if ref_vid == 0:
                    continue
            else:
                continue

            # Update the context leaf's av_count to include -1
            current_av = dict(ctx_leaf.av_count) if ctx_leaf.av_count else {}
            # Deep copy each attribute's value dict
            new_av = {}
            for k, v in current_av.items():
                new_av[k] = dict(v)
            # Add or update -1 attribute
            if -1 not in new_av:
                new_av[-1] = {}
            ref_counts = dict(new_av[-1])
            ref_counts[ref_vid] = ref_counts.get(ref_vid, 0) + 1
            new_av[-1] = ref_counts
            ctx_leaf.set_av_count(new_av)

        if debug:
            print(f"  content-refs written to context hierarchy leaves")

        return True

    # ---- visualization --------------------------------------------------

    def visualize_content_hierarchy(self, out_base="content_hierarchy", max_depth=1e9):
        self.content_drawer.draw_tree(self.content_hierarchy.root, out_base, max_depth=max_depth)

    def visualize_context_hierarchy(self, out_base="context_hierarchy", max_depth=1e9):
        self.context_drawer.draw_tree(self.context_hierarchy.root, out_base, max_depth=max_depth)

    # ---- save / load ----------------------------------------------------

    def save_state(self, dirpath: str) -> dict:
        os.makedirs(dirpath, exist_ok=True)

        meta = {
            "context_length": self.context_length,
            "content_path_depth": self.content_path_depth,
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
            content_path_depth=meta.get("content_path_depth", 3)
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

        # rebuild drawers
        content_headers = (
            [f"Left-Depth{i}" for i in range(ltm.content_path_depth)]
            + [f"Right-Depth{i}" for i in range(ltm.content_path_depth)]
        )
        ltm.content_drawer = HTMLCobwebDrawer(
            content_headers,
            id_to_value=ltm.id_to_value,
            value_to_id=ltm.value_to_id,
        )
        context_headers = (
            [f"Context-Before{i}" for i in range(ltm.context_length)]
            + [f"Context-After{i}" for i in range(ltm.context_length)]
            + ["Complexity"]
        )
        content_ref_attr_idx = -1
        def _content_ref_display(val_id):
            if val_id is not None and 0 <= val_id < len(ltm.id_to_value):
                name = ltm.id_to_value[val_id]
            else:
                name = f"?{val_id}"
            if isinstance(name, str) and name.startswith("CONCEPT-"):
                return "C-" + name[8:20] + "…"
            return name
        context_attr_value_fn = {
            content_ref_attr_idx: _content_ref_display,
        }
        ltm.context_drawer = HTMLCobwebDrawer(
            context_headers,
            id_to_value=ltm.id_to_value,
            value_to_id=ltm.value_to_id,
            attr_value_fn=context_attr_value_fn,
            attr_name_overrides={content_ref_attr_idx: "Content-Ref"},
        )
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

    def __init__(self, value_corpus: list, context_length: int = 3, content_length: int = 3, alpha: float = 1e-4, threshold=-5.0):
        """
        Parameters
        ----------
        value_corpus : list
            Initial vocabulary (list of word strings).
        context_length : int
            Number of context-window slots on each side (before/after).
        content_length : int
            Number of depth-level attributes per side in content instances.
            Total content attributes = 2 * content_length (left depths + right depths).
        alpha : float
            Cobweb smoothing parameter.
        threshold : float
            Default score threshold for accepting chunk merges.
        """
        self.ltm = LongTermMemory(value_corpus, context_length=context_length, content_path_depth=content_length, alpha=alpha)
        self.context_length = context_length
        self.content_length = content_length
        self.threshold = threshold

    # ---- accessors ------------------------------------------------------

    @property
    def id_to_value(self):
        return self.ltm.id_to_value

    @property
    def value_to_id(self):
        return self.ltm.value_to_id

    def get_long_term_memory(self) -> LongTermMemory:
        return self.ltm

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
            self.ltm.add_parse_tree(parse_tree, debug=debug)

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

    # ---- chunk evaluation (called by external tools / GUI) --------------

    def evaluate_chunk(self, content_instance: dict, context_instance: dict, debug=False) -> dict:
        """
        Given content and context instances for a candidate chunk, return
        recognition scores from both hierarchies.
        """
        content_stats = self.ltm.get_content_instance_statistics(content_instance, debug=debug)
        context_stats = self.ltm.get_context_instance_statistics(context_instance, debug=debug)

        return {
            "content_score": content_stats.get("cost", -1e8),
            "context_score": context_stats.get("cost", -1e8),
            "content_stats": content_stats,
            "context_stats": context_stats,
        }

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

        _cpd = self.ltm.content_path_depth
        _cl  = self.context_length
        _cplx_vid  = self.ltm.value_to_id.get("COMPLEXITY", 0)
        _ref_attr  = -1                   # content-ref (hidden)
        _cplx_attr = 2 * _cl              # complexity attribute

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
            if ctx_node is None:
                return 1
            av = ctx_node.av_count or {}
            s = av.get(_cplx_attr, {}).get(_cplx_vid, 0)
            n = sum(av.get(_ref_attr, {}).values()) or 1
            return max(int(round(s / n)), 1)

        # ── sample from content hierarchy via basic-level ────────────────

        def _basic_sample(cnt_node):
            """get_basic → sample a leaf from the basic-level subtree."""
            basic = cnt_node.get_basic(1000, 1000) # TODO REVISE THIS!!! n_nodes, n_samples
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
            """Sample a single path vid from content leaf at attr_idx."""
            rd = (cnt_leaf.av_count or {}).get(attr_idx, {})
            pool = {v: w for v, w in rd.items() if v != 0}
            if not pool:
                return 0
            return random.choices(
                list(pool.keys()),
                weights=[max(x, 1e-12) for x in pool.values()],
                k=1)[0]

        def _deepest_attr(cnt_leaf, side_offset):
            """Find the deepest (most specific) non-empty attr index for a side.

            With root-first ordering, depth-0 = root, depth-(cpd-1) = most
            specific.  We scan from deepest to shallowest and return the
            first attr index that has a non-zero value.
            """
            for d in range(_cpd - 1, -1, -1):
                attr = side_offset + d
                rd = (cnt_leaf.av_count or {}).get(attr, {})
                if any(v != 0 and c > 0 for v, c in rd.items()):
                    return attr
            return side_offset  # fallback to depth-0

        # ── context instance builders ────────────────────────────────────

        def _empty_ctx(cplx=1):
            ctx = {}
            for j in range(_cl):
                ctx[j]       = {0: 1.0 / (2 ** (j + 1))}
                ctx[_cl + j] = {0: 1.0 / (2 ** (j + 1))}
            ctx[_cplx_attr] = {_cplx_vid: cplx}
            return ctx

        def _seeded_ctx(pos, known, cplx):
            ctx = {}
            for j in range(_cl):
                s = pos - (j + 1)
                if 0 <= s < len(known) and known[s]:
                    ctx[j] = {known[s]: 1.0 / (2 ** (j + 1))}; ctx[j][0] = 0
                else:
                    ctx[j] = {0: 1.0 / (2 ** (j + 1))}
            for j in range(_cl):
                s = pos + (j + 1)
                if 0 <= s < len(known) and known[s]:
                    ctx[_cl+j] = {known[s]: 1.0 / (2 ** (j + 1))}; ctx[_cl+j][0] = 0
                else:
                    ctx[_cl+j] = {0: 1.0 / (2 ** (j + 1))}
            ctx[_cplx_attr] = {_cplx_vid: cplx}
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

            # Read left/right path_vids from sampled content leaf
            # Root-first ordering: deepest non-empty attr = most specific
            left_attr  = _deepest_attr(sampled_leaf, 0)
            right_attr = _deepest_attr(sampled_leaf, _cpd)
            left_pv  = _sample_path(sampled_leaf, left_attr)
            right_pv = _sample_path(sampled_leaf, right_attr)

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
            "content_length": self.content_length,
            "threshold": self.threshold,
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
        w.content_length = meta.get("content_length", 3)
        w.threshold = meta.get("threshold", -7)
        return w
