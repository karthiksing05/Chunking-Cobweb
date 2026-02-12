"""

Primary module for multi-hierarchy theory! We're programming the whole thing from scratch,
going to try to leverage some of the old code but the parse trees need a large rework
especially regarding their visualization (context-hierarchy and content-hierarchy both need
to be worked).

See implementation details in MULTIHIERARCHY.md!

Key difference from parse.py (single-hierarchy):
    - TWO Cobweb hierarchies: one for content (left+right path), one for context (surrounding
      context windows + complexity).  In the old code a single LTM stored a flat instance
      {0: left, 1: right, 2..N: ctx_before, N+1..2N: ctx_after, 2N+1: primitive_content}.
      Here those are split:
          content_instance = {0: left_path, 1: right_path}
          context_instance = {0..context_length-1: ctx_before, context_length..2*ctx_len-1: ctx_after,
                              2*ctx_len: complexity}
    - Primitive labels come from categorizing in the *context* hierarchy.
    - Composite labels come from categorizing in the *context* hierarchy (full path,
      weighted leaf→root via 1/2^i).
    - Only frozen/accepted chunks are added to BOTH hierarchies; unfrozen candidates
      go only to the content hierarchy.
    - Scoring is recognition-based (tree-wide log-probability), matching parse.py's
      current approach.
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

def _categorize_dfs(inst: dict, tree: CobwebDiscreteTree):
    """
    DFS categorization down a CobwebDiscreteTree, returning (leaf_node, path_strings, node_path).
    path_strings is ["CONCEPT-<hash>", ...] from root to leaf.
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

        best_idx = None
        best_val = -float("inf")
        for i, v in enumerate(child_scores):
            try:
                val = float(v)
                if math.isnan(val):
                    val = -float("inf")
            except Exception:
                val = -float("inf")
            if val > best_val:
                best_val = val
                best_idx = i

        if best_idx is None or best_val == -float("inf"):
            break

        try:
            node = node.children[best_idx]
        except Exception:
            try:
                node = node.children[best_idx][1]
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
# PrimitiveParseNode
# ---------------------------------------------------------------------------

class PrimitiveParseNode(object):
    """
    Represents a single word/token in the parse tree.

    In the multi-hierarchy theory every node carries two facets:
        context_instance  – what the context hierarchy sees
                            (sliding-window context + complexity attribute)
        label             – the full categorization path through the context
                            hierarchy, weighted leaf→root via 1/2^i.  This is
                            what higher-level nodes use as "content" when
                            building composite chunks.

    Attributes
    ----------
    parent : CompositeParseNode | None
    children : SortedList          always empty for primitives
    position_idx : int             word position in the sentence
    title : str                    unique random id
    context_instance : dict        the instance dict for the context hierarchy
    label : dict                   weighted path dict {concept_id: weight}
    complexity : int               always 1 for primitives
    word_id : int                  the vocabulary id of the raw token
    score_data : dict              scoring statistics from the context hierarchy
    stable : bool                  whether this primitive passed the threshold
    """

    def __init__(self, context_instance: dict, label: dict, position_idx: int, word_id: int):
        self.parent: Optional['CompositeParseNode'] = None
        self.children: SortedList = SortedList()  # always empty for primitives

        self.title: str = uuid.uuid4().hex[:10]
        self.position_idx: int = position_idx

        self.context_instance: dict = context_instance
        self.label: dict = label  # weighted categorize path {concept_vocab_id: 1/2^i}
        self.word_id: int = word_id

        self.complexity: int = 1
        self.score_data: dict = {}
        self.stable: bool = False

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
        """Return the weighted path dict used as 'content' in higher-level nodes."""
        return dict(self.label)


# ---------------------------------------------------------------------------
# CompositeParseNode
# ---------------------------------------------------------------------------

class CompositeParseNode(object):
    """
    Represents a merged chunk (two children) in the parse tree.

    Carries two distinct instance facets:
        content_instance  – {0: left_label, 1: right_label} for the content hierarchy
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
        self.content_instance: Optional[dict] = None  # {0: left_label, 1: right_label}

        # context facet
        self.context_instance: Optional[dict] = None
        self.context_before: Optional[List[dict]] = None
        self.context_after: Optional[List[dict]] = None

        self.label: Optional[dict] = None  # weighted path from context hierarchy
        self.categorize_path: Optional[List] = None  # raw path strings

        self.context_length: int = 0
        self.complexity: int = 0

        self.concept_label = None  # vocab id of the concept
        self.frozen: bool = False  # whether this chunk has been "accepted"

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
        Build the content-hierarchy instance from two children.
        Content is strictly {0: left_label, 1: right_label} – no context.
        Labels are the weighted categorize paths of each child.
        """
        left_label = left_node.get_label()
        right_label = right_node.get_label()

        content_inst = {
            0: dict(left_label),
            1: dict(right_label),
        }
        # Ensure EMPTYNULL placeholder
        content_inst[0].setdefault(0, 0)
        content_inst[1].setdefault(0, 0)
        return content_inst

    # ------------------------------------------------------------------
    @staticmethod
    def create_context_instance(left_node, right_node, context_length: int,
                                content_ref_id: int = None) -> dict:
        """
        Build the context-hierarchy instance from two children.
        Attributes:
            0 .. context_length-1       : context_before (from left_node)
            context_length .. 2*ctx-1   : context_after  (from right_node)
            2*context_length            : complexity (max child complexity + 1)
            2*context_length + 1        : content-ref (content hierarchy leaf
                                          concept vocab id, for generation)
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

        # complexity attribute
        left_c = getattr(left_node, "complexity", 1)
        right_c = getattr(right_node, "complexity", 1)
        complexity = max(left_c, right_c) + 1
        ctx_inst[2 * context_length] = {complexity: 1}

        # content-ref attribute: for composites this is the content
        # hierarchy leaf concept id; for primitives it's the word_id
        # (set separately in build_primitives)
        if content_ref_id is not None:
            ctx_inst[2 * context_length + 1] = {content_ref_id: 1}

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
            items = [(k, v) for k, v in items if k != 0]
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
        Each primitive is categorized in the **context hierarchy** to obtain
        its label (weighted path).
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

            # complexity = 1 for primitives
            ctx_inst[2 * self.context_length] = {1: 1}

            # word identity attribute – enables generation to recover
            # the actual word from a context hierarchy leaf
            ctx_inst[2 * self.context_length + 1] = {wid: 1}

            # categorize in context hierarchy to get label path
            leaf_node, path_strs, node_path = _categorize_dfs(ctx_inst, self.ltm.context_hierarchy)

            # convert path strings to vocab ids and build weighted label
            label: dict = {}
            path_ids: list = []
            for idx_p, pstr in enumerate(path_strs):
                vid = self.value_to_id.get(pstr)
                if vid is not None:
                    label[vid] = 1.0 / (2 ** (idx_p + 1))
                    path_ids.append(vid)
            label[0] = 0  # EMPTYNULL placeholder

            node = PrimitiveParseNode.create_node(ctx_inst, label, position_idx=i, word_id=wid)

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
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        content_inst = CompositeParseNode.create_content_instance(left_node, right_node)

        # categorize in content hierarchy first to get the leaf reference
        cnt_leaf, cnt_path, cnt_node_path = _categorize_dfs(content_inst, self.ltm.content_hierarchy)

        # store content hierarchy leaf reference in context instance
        cnt_hash = cnt_leaf.concept_hash() if cnt_leaf else "unknown"
        cnt_ref_str = f"CONCEPT-{cnt_hash}"
        self.ltm.add_to_vocab(cnt_ref_str)
        cnt_ref_id = self.value_to_id.get(cnt_ref_str, 0)

        context_inst = CompositeParseNode.create_context_instance(
            left_node, right_node, self.context_length,
            content_ref_id=cnt_ref_id,
        )

        # categorize in context hierarchy (for identity / label)
        ctx_leaf, ctx_path, ctx_node_path = _categorize_dfs(context_inst, self.ltm.context_hierarchy)

        # score from content hierarchy (recognition)
        score_data = _score_along_path(cnt_node_path, content_inst, self.ltm.content_hierarchy, debug=debug)
        score = score_data.get("cost", -1e8)

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

        If *frozen* is True the chunk is considered accepted and will eventually
        be added to both hierarchies; otherwise only the content hierarchy.
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        content_inst = CompositeParseNode.create_content_instance(left_node, right_node)

        # categorize in content hierarchy first to get the leaf reference
        cnt_leaf, cnt_path, _ = _categorize_dfs(content_inst, self.ltm.content_hierarchy)
        cnt_hash = cnt_leaf.concept_hash() if cnt_leaf else "unknown"
        cnt_ref_str = f"CONCEPT-{cnt_hash}"
        self.ltm.add_to_vocab(cnt_ref_str)
        cnt_ref_id = self.value_to_id.get(cnt_ref_str, 0)

        context_inst = CompositeParseNode.create_context_instance(
            left_node, right_node, self.context_length,
            content_ref_id=cnt_ref_id,
        )

        # categorize in context hierarchy
        ctx_leaf, ctx_path, _ = _categorize_dfs(context_inst, self.ltm.context_hierarchy)

        # build label (weighted path)
        label: dict = {}
        path_ids: list = []
        for idx_p, pstr in enumerate(ctx_path):
            vid = self.value_to_id.get(pstr)
            if vid is not None:
                label[vid] = 1.0 / (2 ** (idx_p + 1))
                path_ids.append(vid)
        label[0] = 0

        left_c = getattr(left_node, "complexity", 1)
        right_c = getattr(right_node, "complexity", 1)
        complexity = max(left_c, right_c) + 1

        ctx_hash = ctx_leaf.concept_hash() if ctx_leaf else "unknown"
        concept_label = self.value_to_id.get(f"CONCEPT-{ctx_hash}")

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
        if isinstance(node, PrimitiveParseNode):
            left_list = [{"key": self._safe_lookup(node.word_id), "val": 1.0}]
            before_list = [self.ctx_list(d or {}, draw_zeros) for d in (getattr(node, "context_before", None) or [])]
            after_list = [self.ctx_list(d or {}, draw_zeros) for d in (getattr(node, "context_after", None) or [])]
            # context instance attributes for primitives
            ctx_attrs = []
            if node.context_instance:
                for attr_key in sorted(node.context_instance.keys()):
                    ctx_attrs.append(self.ctx_list(node.context_instance[attr_key], draw_zeros))
            return {
                "title": node.title,
                "left": left_list,
                "right": [],
                "before": before_list,
                "after": after_list,
                "ctx_attrs": ctx_attrs,
                "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
            }
        elif isinstance(node, CompositeParseNode):
            if node.is_global_root:
                return {
                    "title": "ROOT",
                    "left": [],
                    "right": [],
                    "before": [],
                    "after": [],
                    "ctx_attrs": [],
                    "children": [self._draw_node_to_dict(ch[1], draw_zeros) for ch in node.children],
                }
            left_list = self.ctx_list(node.content_instance.get(0, {}) if node.content_instance else {}, draw_zeros)
            right_list = self.ctx_list(node.content_instance.get(1, {}) if node.content_instance else {}, draw_zeros)
            before_list = [self.ctx_list(d or {}, draw_zeros) for d in (node.context_before or [])]
            after_list = [self.ctx_list(d or {}, draw_zeros) for d in (node.context_after or [])]
            # context instance attributes for composites
            ctx_attrs = []
            if node.context_instance:
                for attr_key in sorted(node.context_instance.keys()):
                    ctx_attrs.append(self.ctx_list(node.context_instance[attr_key], draw_zeros))
            return {
                "title": node.title,
                "left": left_list,
                "right": right_list,
                "before": before_list,
                "after": after_list,
                "ctx_attrs": ctx_attrs,
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
        ctx_headers_json = json.dumps(
            [f"CtxBefore{i}" for i in range(self.context_length)]
            + [f"CtxAfter{i}" for i in range(self.context_length)]
            + ["Complexity"]
        )
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
.subtable b {{ display: inline-block; margin: 6px 0 2px; }}
.subtable table {{ border-collapse: collapse; }}
.subtable td {{ border: 1px solid #bbb; padding: 1px 4px; }}
</style>
</head>
<body>
<div id="tree-container"><div id="tree"></div></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const data = {d3_data_json};
const ctxHeaders = {ctx_headers_json};
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

function nodeHTML(d){{
  const ctxTable=(ctx,title)=>{{
    if(!ctx||ctx.length===0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
    const rows=ctx.map(kv=>`<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
    return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
  }};
  const ctxMulti=(arr,base)=>{{
    if(!arr||arr.length===0) return `<div class="subtable"><i>${{base}}: empty</i></div>`;
    if(Array.isArray(arr)&&arr.length>0&&arr[0]&&typeof arr[0].key!=='undefined') return ctxTable(arr,base);
    let out=""; arr.forEach((c,i)=>{{ out+=ctxTable(c,`${{base}}${{i}}`); }}); return out;
  }};
  // content instance
  let contentHTML="";
  const lH=Array.isArray(d.left)&&d.left.length>0, rH=Array.isArray(d.right)&&d.right.length>0;
  if(rH) contentHTML=`<div class="section-title">Content Instance</div>${{ctxTable(d.left,"Left")}}${{ctxTable(d.right,"Right")}}`;
  else if(lH) contentHTML=`<div class="section-title">Content Instance</div>${{ctxTable(d.left,"Word")}}`;
  else contentHTML=``;
  // context instance
  let contextHTML="";
  if(d.ctx_attrs && d.ctx_attrs.length>0){{
    contextHTML=`<div class="section-title">Context Instance</div>`;
    d.ctx_attrs.forEach((attr,i)=>{{
      const hdr = i < ctxHeaders.length ? ctxHeaders[i] : `Attr${{i}}`;
      contextHTML += ctxTable(attr, hdr);
    }});
  }}
  return `<div class="node-fo">
    <table><tr><th colspan="2">${{d.title}}</th></tr></table>
    ${{contentHTML}}
    ${{contextHTML}}
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
        ctx_headers_json = json.dumps(
            [f"CtxBefore{i}" for i in range(self.context_length)]
            + [f"CtxAfter{i}" for i in range(self.context_length)]
            + ["Complexity"]
        )
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
const ctxHeaders = {ctx_headers_json};

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

function nodeHTML(d){{
    const ctxTable=(ctx,title)=>{{
        if(!ctx||ctx.length===0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
        const rows=ctx.map(kv=>`<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
        return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
    }};
    // content instance
    let contentHTML="";
    const lH=Array.isArray(d.left)&&d.left.length>0,rH=Array.isArray(d.right)&&d.right.length>0;
    if(rH) contentHTML=`<div class="section-title">Content Instance</div>${{ctxTable(d.left,"Left")}}${{ctxTable(d.right,"Right")}}`;
    else if(lH) contentHTML=`<div class="section-title">Content Instance</div>${{ctxTable(d.left,"Word")}}`;
    else contentHTML=``;
    // context instance
    let contextHTML="";
    if(d.ctx_attrs && d.ctx_attrs.length>0){{
        contextHTML=`<div class="section-title">Context Instance</div>`;
        d.ctx_attrs.forEach((attr,i)=>{{
            const hdr=i<ctxHeaders.length?ctxHeaders[i]:`Attr${{i}}`;
            contextHTML+=ctxTable(attr,hdr);
        }});
    }}
    return `<div class="node-fo">
        <table><tr><th colspan="2">${{d.title}}</th></tr></table>
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
# RollingParseTree (placeholder for streaming)
# ---------------------------------------------------------------------------

@DeprecationWarning
class RollingParseTree(object):
    """Placeholder for future streaming / rolling-window parse tree."""
    pass


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory(object):
    """
    Holds TWO Cobweb hierarchies (content + context) plus corpus/vocabulary
    management.

    Content hierarchy  – instances are {0: left_label, 1: right_label}.
    Context hierarchy   – instances are {0..ctx_len-1: ctx_before,
                          ctx_len..2*ctx_len-1: ctx_after,
                          2*ctx_len: complexity,
                          2*ctx_len+1: content-ref (word_id for primitives,
                                       content leaf concept_id for composites)}.
    """

    def __init__(self, value_corpus: list, context_length: int = 3, alpha: float = 1e-4):
        self.content_hierarchy = CobwebDiscreteTree(alpha)
        self.context_hierarchy = CobwebDiscreteTree(alpha)

        # vocabulary: index 0 is always EMPTYNULL
        self.id_to_value: List[str] = ["EMPTYNULL"]
        for x in value_corpus:
            self.id_to_value.append(x)
        self.value_to_id: Dict[str, int] = {w: i for i, w in enumerate(self.id_to_value)}
        self.id_count: int = len(self.id_to_value) - 1

        self.context_length = context_length

        # register root concepts of both hierarchies
        self._register_concept(self.content_hierarchy.root)
        self._register_concept(self.context_hierarchy.root)

        # drawer for content hierarchy visualization
        content_headers = ["Content-Left", "Content-Right"]
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
            + ["Content-Ref"]
        )
        # Complexity attribute should display raw integer values,
        # not vocab lookups.
        complexity_attr_idx = 2 * context_length
        context_attr_value_fn = {
            complexity_attr_idx: lambda val_id: str(val_id),
        }
        self.context_drawer = HTMLCobwebDrawer(
            context_headers,
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id,
            attr_value_fn=context_attr_value_fn,
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

    def _ifit_and_update_vocab(self, instance: dict, tree: CobwebDiscreteTree, debug=False) -> list:
        """
        Call ifit on a hierarchy, process the resulting actions (vocab updates + splits).
        Returns the list of raw actions.
        """
        _, actions = tree.ifit(instance, debug=True)
        actions = [json.loads(x) for x in actions]

        rewrite_rules = []
        for act in actions:
            if act["action"] == "NEW":
                self.add_to_vocab(f"CONCEPT-{act['node']}")
            elif act["action"] == "MERGE":
                self.add_to_vocab(f"CONCEPT-{act['new_node']}")
            elif act["action"] == "SPLIT":
                rewrite_rules.append((act["deleted"], act["parent"]))

        # apply rewrite rules via BFS
        if rewrite_rules:
            self._apply_rewrite_rules(tree, rewrite_rules)

        return actions

    def _apply_rewrite_rules(self, tree: CobwebDiscreteTree, rewrite_rules: list):
        """BFS through tree and replace split-deleted concept hashes in av_counts."""
        def av_replacement(av):
            replaced = False
            for k in av.keys():
                for concept_hash in list(av[k].keys()):
                    for old, new in rewrite_rules:
                        if f"CONCEPT-{concept_hash}" == old:
                            av[k].setdefault(f"CONCEPT-{new}", 0)
                            av[k][f"CONCEPT-{new}"] += av[k][old]
                            del av[k][old]
                            replaced = True
            return av, replaced

        to_visit = [tree.root]
        while to_visit:
            curr = to_visit.pop(0)
            new_av, replaced = av_replacement(curr.av_count)
            curr.set_av_count(new_av)
            if replaced:
                to_visit.extend(curr.children)

    def add_parse_tree(self, parse_tree: 'FiniteParseTree', debug=False):
        """
        Learn from a completed parse tree.

        Per multi-hierarchy theory:
          - ALL content instances (parsed + unparsed candidates) → content hierarchy
          - Only context instances from parsed/frozen nodes → context hierarchy
        """
        content_insts, context_insts = parse_tree.get_all_instances()

        if debug:
            print(f"Adding parse tree for window: \"{parse_tree.window}\"")
            print(f"  content instances: {len(content_insts)}")
            print(f"  context instances: {len(context_insts)}")

        for ci in content_insts:
            self._ifit_and_update_vocab(ci, self.content_hierarchy, debug=debug)

        for xi in context_insts:
            self._ifit_and_update_vocab(xi, self.context_hierarchy, debug=debug)

        return True

    # ---- update vocabulary after external changes -----------------------

    def update_vocabulary(self, actions: list):
        """Process a list of Cobweb actions to update vocabulary accordingly."""
        for act in actions:
            if act.get("action") in ("NEW", "MERGE"):
                node_hash = act.get("node") or act.get("new_node")
                if node_hash:
                    self.add_to_vocab(f"CONCEPT-{node_hash}")

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

        ltm = LongTermMemory([], context_length=meta.get("context_length", 3))
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
        content_headers = ["Content-Left", "Content-Right"]
        ltm.content_drawer = HTMLCobwebDrawer(
            content_headers,
            id_to_value=ltm.id_to_value,
            value_to_id=ltm.value_to_id,
        )
        context_headers = (
            [f"Context-Before{i}" for i in range(ltm.context_length)]
            + [f"Context-After{i}" for i in range(ltm.context_length)]
            + ["Complexity"]
            + ["Content-Ref"]
        )
        complexity_attr_idx = 2 * ltm.context_length
        context_attr_value_fn = {
            complexity_attr_idx: lambda val_id: str(val_id),
        }
        ltm.context_drawer = HTMLCobwebDrawer(
            context_headers,
            id_to_value=ltm.id_to_value,
            value_to_id=ltm.value_to_id,
            attr_value_fn=context_attr_value_fn,
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

    def __init__(self, value_corpus: list, context_length: int = 3, alpha: float = 1e-4, threshold=-7):
        self.ltm = LongTermMemory(value_corpus, context_length=context_length, alpha=alpha)
        self.context_length = context_length
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

    def generate_sentence(self, masked_sentence: str = "", debug: bool = False) -> List:
        """
        Generate or complete a sentence.

        **Masked completion** (``"the [mask] dog [mask] the park"``):
            1. Tokenize, build primitives for known words (with context).
            2. For each ``[mask]``, use the surrounding context to predict
               the most likely word from the context hierarchy.

        **From-scratch generation** (empty string):
            1. Sample a high-level content node (no context → high complexity).
            2. Recursively expand composites via the content hierarchy until
               every leaf is a primitive (complexity 1).
            3. At each expansion step, the content hierarchy's leaf node
               determines the left and right children's content; the context
               hierarchy determines whether complexity is > 1 (composite) or
               == 1 (primitive / freeze).

        Returns ``[generated_text, FiniteParseTree]``.
        """

        # ---- shared helpers ------------------------------------------------

        def _probabilistic_subset(d: dict, k: int = 20) -> dict:
            """Sample *k* keys from *d* weighted by value."""
            if not d:
                return {}
            keys = list(d.keys())
            weights = [max(v, 1e-12) for v in d.values()]
            selected = random.choices(keys, weights=weights, k=k)
            return {sk: d[sk] for sk in selected}

        def _safe_lookup(idx):
            if isinstance(idx, str):
                return idx
            try:
                if idx is not None and 0 <= idx < len(self.id_to_value):
                    return self.id_to_value[idx]
            except Exception:
                pass
            return None

        def _is_word_token(vid: int) -> bool:
            """True when *vid* is a genuine word (not EMPTYNULL, not CONCEPT-…)."""
            if vid is None or vid == 0:
                return False
            name = _safe_lookup(vid)
            if name is None or name == "EMPTYNULL":
                return False
            if isinstance(name, str) and name.startswith("CONCEPT-"):
                return False
            return True

        def _best_word_from_distribution(dist: dict) -> int:
            """Weighted random pick of the best word id from a value distribution."""
            candidates = {vid: w for vid, w in dist.items() if _is_word_token(vid)}
            if not candidates:
                return 0
            return random.choices(
                list(candidates.keys()),
                weights=[max(v, 1e-12) for v in candidates.values()],
                k=1,
            )[0]

        def _traverse_to_leaf(label_dict: dict) -> Optional[CobwebDiscreteNode]:
            """Follow a weighted label path through the content hierarchy to a leaf."""
            curr = self.ltm.content_hierarchy.root
            for item_key, _ in sorted(label_dict.items(), key=lambda x: -x[1]):
                concept_str = _safe_lookup(item_key)
                if concept_str is None or concept_str == "EMPTYNULL":
                    continue
                if not curr.children:
                    return curr
                found = False
                for child in curr.children:
                    try:
                        if f"CONCEPT-{child.concept_hash()}" == concept_str:
                            curr = child
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    return curr
            return curr

        def _traverse_context_leaf(label_dict: dict) -> Optional[CobwebDiscreteNode]:
            """Follow a weighted label path through the **context** hierarchy.

            Labels stored in the content hierarchy are weighted paths of
            CONCEPT-… IDs that originate from the context hierarchy.  This
            function traverses the context hierarchy matching those concept
            hashes and returns the deepest reachable node.
            """
            curr = self.ltm.context_hierarchy.root
            for item_key, _ in sorted(label_dict.items(), key=lambda x: -x[1]):
                concept_str = _safe_lookup(item_key)
                if concept_str is None or concept_str == "EMPTYNULL":
                    continue
                if not isinstance(concept_str, str) or not concept_str.startswith("CONCEPT-"):
                    continue
                if not curr.children:
                    return curr
                found = False
                for child in curr.children:
                    try:
                        if f"CONCEPT-{child.concept_hash()}" == concept_str:
                            curr = child
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    return curr
            return curr

        def _word_from_context_leaf(ctx_node: CobwebDiscreteNode) -> int:
            """Extract the most likely word_id from a context hierarchy node
            by reading its word-identity attribute (2*ctx_len+1)."""
            word_attr = 2 * self.context_length + 1
            av = ctx_node.av_count if ctx_node else {}
            word_dist = av.get(word_attr, {})
            candidates = {vid: w for vid, w in word_dist.items() if _is_word_token(vid)}
            if not candidates:
                return 0
            return random.choices(
                list(candidates.keys()),
                weights=[max(v, 1e-12) for v in candidates.values()],
                k=1,
            )[0]

        def _content_ref_from_leaf(ctx_node: CobwebDiscreteNode) -> int:
            """Extract the most likely content-ref vocab id from a context
            hierarchy node's content-ref attribute (2*ctx_len+1).
            Returns a word vocab id for primitives or a CONCEPT-… vocab id
            for composites, or 0 if nothing useful is found."""
            ref_attr = 2 * self.context_length + 1
            av = ctx_node.av_count if ctx_node else {}
            ref_dist = av.get(ref_attr, {})
            candidates = {}
            for vid, w in ref_dist.items():
                if vid == 0:
                    continue
                name = _safe_lookup(vid)
                if name is None or name == "EMPTYNULL":
                    continue
                candidates[vid] = w
            if not candidates:
                return 0
            return random.choices(
                list(candidates.keys()),
                weights=[max(v, 1e-12) for v in candidates.values()],
                k=1,
            )[0]

        def _find_content_node_by_hash(target_hash_str: str) -> Optional[CobwebDiscreteNode]:
            """BFS search the content hierarchy for a node whose
            ``concept_hash()`` matches *target_hash_str* (as a string)."""
            queue = [self.ltm.content_hierarchy.root]
            while queue:
                node = queue.pop(0)
                try:
                    if str(node.concept_hash()) == target_hash_str:
                        return node
                except Exception:
                    pass
                queue.extend(node.children)
            return None

        def _make_empty_ctx(complexity_val: int = 1) -> dict:
            """Build an empty context instance with a given complexity value."""
            ctx = {}
            for j in range(self.context_length):
                ctx[j] = {0: 0}
                ctx[self.context_length + j] = {0: 0}
            ctx[2 * self.context_length] = {complexity_val: 1}
            return ctx

        def _sample_content_node(hint: dict = None):
            """
            Sample a leaf from the content hierarchy.

            If *hint* is given it should be a partial content instance
            ``{0: …, 1: …}``; missing attrs are predicted.  Returns
            ``(leaf_node, path_strs, node_path)``.
            """
            partial = dict(hint) if hint else {}
            prediction = self.ltm.content_hierarchy.predict(partial, random.randint(100, 500), False)
            if 0 not in partial or not partial[0]:
                partial[0] = _probabilistic_subset(prediction.get(0, {}), k=20)
            if 1 not in partial or not partial[1]:
                partial[1] = _probabilistic_subset(prediction.get(1, {}), k=20)
            partial[0][0] = 0
            partial[1][0] = 0
            return _categorize_dfs(partial, self.ltm.content_hierarchy)

        def _predict_complexity(ctx_inst: dict) -> int:
            """
            Use the context hierarchy to predict the most likely complexity
            for a given context instance.  Returns the integer complexity.
            """
            complexity_attr = 2 * self.context_length
            prediction = self.ltm.context_hierarchy.predict(ctx_inst, random.randint(100, 500), False)
            complexity_dist = prediction.get(complexity_attr, {})
            if not complexity_dist:
                return 1
            # pick the complexity value with highest weight
            best_c = max(complexity_dist, key=complexity_dist.get)
            return max(int(best_c), 1)

        def _build_label_from_path(path_strs: list) -> dict:
            """Convert a categorize path into a weighted label dict."""
            label: dict = {}
            for idx_p, pstr in enumerate(path_strs):
                vid = self.value_to_id.get(pstr)
                if vid is not None:
                    label[vid] = 1.0 / (2 ** (idx_p + 1))
            label[0] = 0
            return label

        # ---- expand a single composite into two children -------------------

        def _expand_node(node: CompositeParseNode) -> Tuple[Any, Any]:
            """
            Expand a composite parse node into two children.

            For each child (left, right):
                1. Traverse the *context* hierarchy using the side label from
                   the parent's content_instance to reach a context leaf.
                2. Read the content-ref attribute (``2*ctx_len+1``) from that
                   leaf.

                   - If the ref is a word  → create a ``PrimitiveParseNode``.
                   - If the ref is a ``CONCEPT-<hash>`` → find the matching
                     content hierarchy node, use its ``av_count`` for
                     sub-content, and create a ``CompositeParseNode`` for
                     further expansion.

                3. Fall back to content hierarchy traversal + complexity
                   prediction when the content-ref is unavailable.
            """
            if not node.content_instance:
                return None, None

            left_label = node.content_instance.get(0, {})
            right_label = node.content_instance.get(1, {})
            content_ref_attr = 2 * self.context_length + 1

            results = []
            for side_idx, side_label in enumerate([left_label, right_label]):
                # 1. Traverse context hierarchy to get a context leaf
                ctx_leaf = _traverse_context_leaf(side_label)

                # 2. Read content-ref from context leaf
                content_ref_id = _content_ref_from_leaf(ctx_leaf) if ctx_leaf else 0
                ref_name = _safe_lookup(content_ref_id) if content_ref_id else None

                if debug:
                    print(f"  Side {side_idx}: ctx_leaf_hash="
                          f"{ctx_leaf.concept_hash() if ctx_leaf else 'None'}, "
                          f"content_ref={ref_name} (id={content_ref_id})")

                # --- PRIMITIVE: content-ref is a word ---
                if content_ref_id and _is_word_token(content_ref_id):
                    word_id = content_ref_id
                    child_ctx = _make_empty_ctx(1)
                    child_ctx[content_ref_attr] = {word_id: 1}
                    _, ctx_path, _ = _categorize_dfs(child_ctx, self.ltm.context_hierarchy)
                    child_label = _build_label_from_path(ctx_path)
                    child_node = PrimitiveParseNode.create_node(
                        context_instance=child_ctx,
                        label=child_label,
                        position_idx=node.position_idx,
                        word_id=word_id,
                    )
                    results.append(child_node)
                    continue

                # --- COMPOSITE: content-ref is a CONCEPT-<hash> ---
                if ref_name and isinstance(ref_name, str) and ref_name.startswith("CONCEPT-"):
                    target_hash = ref_name[len("CONCEPT-"):]
                    content_node = _find_content_node_by_hash(target_hash)
                    if content_node:
                        cnt_av = content_node.av_count or {}
                        child_content = {
                            0: _probabilistic_subset(dict(cnt_av.get(0, {0: 1})), k=15),
                            1: _probabilistic_subset(dict(cnt_av.get(1, {0: 1})), k=15),
                        }
                        child_content[0][0] = 0
                        child_content[1][0] = 0
                        parent_ctx = (dict(node.context_instance)
                                      if node.context_instance
                                      else _make_empty_ctx(node.complexity))
                        child_complexity = max(_predict_complexity(parent_ctx) - 1, 2)
                        child_ctx = _make_empty_ctx(child_complexity)
                        _, ctx_path, _ = _categorize_dfs(child_ctx, self.ltm.context_hierarchy)
                        child_label = _build_label_from_path(ctx_path)
                        child_node = CompositeParseNode.create_node(
                            content_instance=child_content,
                            context_instance=child_ctx,
                            label=child_label,
                            categorize_path=[],
                            position_idx=node.position_idx,
                            context_length=self.context_length,
                            complexity=child_complexity,
                        )
                        results.append(child_node)
                        continue

                # --- FALLBACK: content hierarchy traversal + complexity ---
                leaf = _traverse_to_leaf(side_label)
                if leaf is None:
                    results.append(None)
                    continue

                leaf_av = leaf.av_count or {}
                left_dist = dict(leaf_av.get(0, {0: 1}))
                right_dist = dict(leaf_av.get(1, {0: 1}))

                parent_ctx = (dict(node.context_instance)
                              if node.context_instance
                              else _make_empty_ctx(node.complexity))
                child_complexity = max(_predict_complexity(parent_ctx) - 1, 1)

                if debug:
                    leaf_hash = leaf.concept_hash() if leaf else "None"
                    print(f"  Fallback side {side_idx}: leaf={leaf_hash}, "
                          f"child_complexity={child_complexity}")

                if child_complexity > 1:
                    child_content = {
                        0: _probabilistic_subset(left_dist, k=15),
                        1: _probabilistic_subset(right_dist, k=15),
                    }
                    child_content[0][0] = 0
                    child_content[1][0] = 0
                    child_ctx = _make_empty_ctx(child_complexity)
                    _, ctx_path, _ = _categorize_dfs(child_ctx, self.ltm.context_hierarchy)
                    child_label = _build_label_from_path(ctx_path)
                    child_node = CompositeParseNode.create_node(
                        content_instance=child_content,
                        context_instance=child_ctx,
                        label=child_label,
                        categorize_path=[],
                        position_idx=node.position_idx,
                        context_length=self.context_length,
                        complexity=child_complexity,
                    )
                    results.append(child_node)
                else:
                    # last resort: predict word from global context hierarchy
                    fallback_ctx = _make_empty_ctx(1)
                    prediction = self.ltm.context_hierarchy.predict(
                        fallback_ctx, random.randint(100, 500), False
                    )
                    word_dist = prediction.get(content_ref_attr, {})
                    word_id = _best_word_from_distribution(word_dist)
                    child_ctx = _make_empty_ctx(1)
                    child_ctx[content_ref_attr] = {word_id: 1}
                    _, ctx_path, _ = _categorize_dfs(child_ctx, self.ltm.context_hierarchy)
                    child_label = _build_label_from_path(ctx_path)
                    child_node = PrimitiveParseNode.create_node(
                        context_instance=child_ctx,
                        label=child_label,
                        position_idx=node.position_idx,
                        word_id=word_id,
                    )
                    results.append(child_node)

            return results[0], results[1]

        # ---- flatten primitives to word list -------------------------------

        def _flatten_to_words(root_node) -> List[str]:
            """DFS-collect primitives in position order, return word list."""
            primitives = []

            def dfs(n):
                if isinstance(n, PrimitiveParseNode):
                    primitives.append(n)
                for _, ch in getattr(n, "children", []):
                    dfs(ch)

            dfs(root_node)
            primitives.sort(key=lambda p: p.position_idx if p.position_idx is not None else 0)
            words = []
            for p in primitives:
                w = _safe_lookup(p.word_id)
                words.append(w if w and w != "EMPTYNULL" else "?")
            return words

        # =====================================================================
        # MASKED SENTENCE COMPLETION
        # =====================================================================
        if masked_sentence:
            tokens = re.findall(r"[\w']+|[.,!?;]|\[mask\]", masked_sentence)
            mask_positions = [i for i, t in enumerate(tokens) if t == "[mask]"]

            if debug:
                print(f"Tokens: {tokens}")
                print(f"Mask positions: {mask_positions}")

            # Resolve known tokens to vocab ids
            word_ids = []
            for tok in tokens:
                if tok == "[mask]":
                    word_ids.append(None)  # placeholder
                else:
                    vid = self.value_to_id.get(tok, 0)
                    word_ids.append(vid)

            # Fill each [mask] using context hierarchy predictions
            for mi in mask_positions:
                ctx_inst = {}

                # context_before: words to the left of the mask
                for j in range(self.context_length):
                    src_idx = mi - (j + 1)
                    if 0 <= src_idx < len(word_ids) and word_ids[src_idx] is not None and word_ids[src_idx] != 0:
                        ctx_inst[j] = {word_ids[src_idx]: 1.0 / (2 ** (j + 1))}
                        ctx_inst[j][0] = 0
                    else:
                        ctx_inst[j] = {0: 1.0 / (2 ** (j + 1))}

                # context_after: words to the right of the mask
                for j in range(self.context_length):
                    src_idx = mi + (j + 1)
                    attr_key = self.context_length + j
                    if 0 <= src_idx < len(word_ids) and word_ids[src_idx] is not None and word_ids[src_idx] != 0:
                        ctx_inst[attr_key] = {word_ids[src_idx]: 1.0 / (2 ** (j + 1))}
                        ctx_inst[attr_key][0] = 0
                    else:
                        ctx_inst[attr_key] = {0: 1.0 / (2 ** (j + 1))}

                # complexity = 1 (we want a primitive word)
                ctx_inst[2 * self.context_length] = {1: 1}

                # predict from context hierarchy – specifically the
                # word-identity attribute (2*ctx_len+1) which stores
                # actual word IDs in primitive context instances
                prediction = self.ltm.context_hierarchy.predict(ctx_inst, random.randint(100, 500), False)

                word_attr = 2 * self.context_length + 1
                word_dist = prediction.get(word_attr, {})
                combined = {vid: w for vid, w in word_dist.items() if _is_word_token(vid)}

                if combined:
                    chosen_id = random.choices(
                        list(combined.keys()),
                        weights=[max(v, 1e-12) for v in combined.values()],
                        k=1,
                    )[0]
                else:
                    chosen_id = 0

                word_ids[mi] = chosen_id

                if debug:
                    print(f"  [mask] at {mi}: chose '{_safe_lookup(chosen_id)}' (id={chosen_id})")

            # rebuild token list
            result_tokens = []
            for i, tok in enumerate(tokens):
                if tok == "[mask]":
                    w = _safe_lookup(word_ids[i])
                    result_tokens.append(w if w and w != "EMPTYNULL" else "?")
                else:
                    result_tokens.append(tok)

            generated_text = " ".join(result_tokens)

            # Build a parse tree for the completed sentence
            pt = self.parse_sentence(generated_text, threshold="converge", learning=False, debug=debug)
            return [generated_text, pt]

        # =====================================================================
        # GENERATION FROM SCRATCH
        # =====================================================================
        else:
            global_root = CompositeParseNode.create_global_root()

            # 1. Sample a high-level content node (no context → sentence-level)
            sampled_leaf, sampled_path, _ = _sample_content_node()

            if debug:
                leaf_hash = sampled_leaf.concept_hash() if sampled_leaf else "None"
                print(f"Sampled initial content leaf: {leaf_hash}")
                print(f"  Path: {sampled_path}")

            label = _build_label_from_path(sampled_path)

            # Use context hierarchy to predict appropriate starting complexity
            seed_ctx = _make_empty_ctx(1)  # start with minimal complexity hint
            initial_complexity = _predict_complexity(seed_ctx)
            # ensure we start with at least complexity 2 so expansion happens
            initial_complexity = max(initial_complexity, 2)

            if debug:
                print(f"  Initial complexity: {initial_complexity}")

            # Build seed content from sampled leaf's av_count
            sampled_av = sampled_leaf.av_count if sampled_leaf else {}
            sampled_content = {
                0: _probabilistic_subset(dict(sampled_av.get(0, {0: 1})), k=20),
                1: _probabilistic_subset(dict(sampled_av.get(1, {0: 1})), k=20),
            }
            sampled_content[0][0] = 0
            sampled_content[1][0] = 0

            ctx_inst = _make_empty_ctx(initial_complexity)

            initial_node = CompositeParseNode.create_node(
                content_instance=sampled_content,
                context_instance=ctx_inst,
                label=label,
                categorize_path=[],
                position_idx=0.0,
                context_length=self.context_length,
                complexity=initial_complexity,
            )
            initial_node.set_parent(global_root)

            # 2. Expand recursively via a priority queue (highest complexity first)
            frontier = [(-initial_complexity, id(initial_node), initial_node)]
            all_nodes = [initial_node]
            max_expansions = 100  # safety limit

            expansions = 0
            while frontier and expansions < max_expansions:
                neg_complexity, _, node_to_expand = heapq.heappop(frontier)

                if not isinstance(node_to_expand, CompositeParseNode):
                    continue

                if debug:
                    print(f"\nExpansion {expansions}: node at pos={node_to_expand.position_idx}, "
                          f"complexity={node_to_expand.complexity}")

                left_child, right_child = _expand_node(node_to_expand)

                if left_child is not None:
                    left_child.position_idx = node_to_expand.position_idx - 0.5 / (2 ** expansions)
                    left_child.set_parent(node_to_expand)
                    all_nodes.append(left_child)
                    if isinstance(left_child, CompositeParseNode):
                        heapq.heappush(frontier, (
                            -left_child.complexity, id(left_child), left_child
                        ))

                if right_child is not None:
                    right_child.position_idx = node_to_expand.position_idx + 0.5 / (2 ** expansions)
                    right_child.set_parent(node_to_expand)
                    all_nodes.append(right_child)
                    if isinstance(right_child, CompositeParseNode):
                        heapq.heappush(frontier, (
                            -right_child.complexity, id(right_child), right_child
                        ))

                expansions += 1

            # 3. Flatten to words
            words = _flatten_to_words(global_root)
            generated_text = " ".join(words) if words else ""

            if debug:
                print(f"\nGenerated: {generated_text}")
                print(f"Total expansions: {expansions}, total nodes: {len(all_nodes)}")

            final_parse = FiniteParseTree(self.ltm, self.context_length)
            final_parse.window = generated_text
            final_parse.global_root_node = global_root
            final_parse.nodes = all_nodes

            return [generated_text, final_parse]

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
        w.threshold = meta.get("threshold", -7)
        return w
