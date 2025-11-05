import uuid
import os
import json
import asyncio
from playwright.async_api import async_playwright
import re
from cobweb.cobweb_discrete import CobwebTree, CobwebNode
from viz import HTMLCobwebDrawer
from typing import List
from sortedcontainers import SortedList
import heapq
import time
import math
import random

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class PrimitiveParseNode:

    def __init__(self, content, context_before, context_after, word_index):

        self.global_root = False
        self.word_index = word_index

        self.parent = None
        self.children = SortedList()

        self.title = uuid.uuid4().hex[:10] # random id

        self.content = content
        self.context_before = context_before
        self.context_after = context_after

        self.concept_label = None

    def set_parent(self, node):
        """
        Helper method to set the parent of the current node!
        We're going to adjust both parents and children with this method, to
        allow for efficient designation.

        This method also deletes the existing parent-child connection, if it
        already exists (to efficiently manage the root node changes).
        """

        try:
            self.parent.children.remove((self.word_index, self))
        except AttributeError: # trying to assign parentship for the first time
            # print("Assigning current node's parent as the global root node for brevity")
            pass
        except ValueError: # this should never happen
            print("Parent does not exist or parent does not include the current node as its child")

        self.parent = node
        node.children.add((self.word_index, self))

    def get_as_instance(self):
        """
        Helper method to get the current parse node as an instance description!

        Note that with this method, we don't have to worry about an empty set for the content because it carries forth
        """

        return {
            "content": self.content,
            "context-before": self.context_before,
            "context-after": self.context_after
        }

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class CompositeParseNode:

    def __init__(self):

        self.global_root = False
        self.word_index = None

        self.parent = None
        self.children = SortedList()

        self.title = uuid.uuid4().hex[:10] # random id

        self.content_left = None
        self.content_right = None
        self.context_before = None
        self.context_after = None

        self.concept_label = None
        self.categorize_path = None # NEW THING FOR THE PATHS! A list representing the path taken to get to the given concept label!

    @staticmethod
    def create_global_root():
        """
        A static method that creates an empty root node to act as the navigation
        for all parents. This will be extremely useful for keeping track of
        partial parses with our cutoff.
        """

        node = CompositeParseNode()

        node.global_root = True

        return node

    @staticmethod
    def create_node(instance_dict, closest_concept_id, categorize_path, word_index):
        """
        One of the method-based constructors to create the correct version
        of our parse nodes at the leaf level - this will create a parse node
        from a directly-parsed instance from the input.
        """

        node = CompositeParseNode()

        node.content_left = instance_dict[0]
        node.content_right = instance_dict[1]

        node.context_before = instance_dict[2]
        node.context_after = instance_dict[3]

        node.concept_label = closest_concept_id
        node.categorize_path = categorize_path

        node.word_index = word_index

        return node

    @staticmethod
    def create_merge_instance(node_left, node_right):
        """
        One of the method-based constructors to create the correct version
        of our parse nodes at the not leaf level - it should properly reference
        context and construct a parse node based on two given parse nodes.

        NOTE: this is not going to actively create a node yet, it's a helper
        method to merge instances. We'll take the best of these instances (from
        the categorization) and then add it to our parse tree.

        CONTEXT CLARIFICATION:
        I would assume that the label for a given node is in terms of the
        concept or instance that it references within the code. Most initial
        instances will be pretty useless / have useless concept labels, but
        eventually we'll be able to track stronger correlations!

        CONTENT CLARIFICATION:
        For content, we have one of two clarifications (which I've asked Pat
        about).
        - https://docs.google.com/presentation/d/1k1PNL73OuZC2lCdqy-q-OfNPMiYJlft04t_kOJTNUZ0/edit?slide=id.g3719c2a0e40_0_26#slide=id.g3719c2a0e40_0_26
        The above URL summarizes it best - currently working with Option 2,
        which we're going to implement, but either should be solid. For brevity,
        Option 2 involves using only primitive instances to represent content
        and context, which is easily done for our tree.
        """

        new_inst_dict = {}

        if type(node_left) == PrimitiveParseNode:
            new_inst_dict[0] = {node_left.content: 1}
        else:
            new_inst_dict[0] = dict([(key, 1) for key in node_left.categorize_path])

        if type(node_right) == PrimitiveParseNode:
            new_inst_dict[1] = {node_right.content: 1}
        else:
            new_inst_dict[1] = dict([(key, 1) for key in node_right.categorize_path])

        new_inst_dict[0][0] = 0
        new_inst_dict[1][0] = 0

        new_inst_dict[2] = node_left.context_before
        new_inst_dict[3] = node_right.context_after

        return new_inst_dict

    def set_parent(self, node):
        """
        Helper method to set the parent of the current node!
        We're going to adjust both parents and children with this method, to
        allow for efficient designation.

        This method also deletes the existing parent-child connection, if it
        already exists (to efficiently manage the root node changes).
        """

        try:
            self.parent.children.remove((self.word_index, self))
        except AttributeError: # trying to assign parentship for the first time
            # print("Assigning current node's parent as the global root node for brevity")
            pass
        except ValueError: # this should never happen
            print("Parent does not exist or parent does not include the current node as its child")

        self.parent = node
        node.children.add((self.word_index, self))

    def get_as_instance(self):
        """
        Helper method to get the current parse node as an instance description!
        """

        return {
            0: self.content_left,
            1: self.content_right,
            2: self.context_before,
            3: self.context_after
        }


"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

def custom_categorize_match_score(inst, tree):
    """
    A helper function that expands comprehensively based on match score through a heap-based
    best-first-search. The most likely candidate is the leaf that best matches the passed
    instance.
    """
    def match_score(candidate, inst):
        """
        A helper function that calculates a score by the matched instances. For simplicity, we'll
        just do a direct and normalized "match_count" - check all attributes and their corresponding
        values for matched values in the candidate instance.

        Returns a normalized score denoting the "overlap" between the score and the value.
        """

        sim = 0.0
        for attr, vals in candidate.items():
            if attr in inst:
                for k, v in vals.items():
                    sim += v * inst[attr].get(k, 0)
        denom = sum(sum(vals.values()) for vals in candidate.values()) + 1e-9
        return sim / denom

    try:
        root = tree.root
    except Exception:
        try:
            leaf = tree.categorize(inst)
            return leaf, [f"CONCEPT-{leaf.concept_hash()}"]
        except Exception:
            return None, []

    def sim_of(node):
        child_av = getattr(node, 'av_count', {}) or {}
        return match_score(child_av, inst)

    def iter_children(n):
        try:
            for ch in n.children:
                if isinstance(ch, (list, tuple)) and len(ch) >= 2:
                    yield ch[1]
                else:
                    yield ch
        except Exception:
            return

    heap = []
    root_score = sim_of(root)
    root_path = [f"CONCEPT-{root.concept_hash()}"]
    heapq.heappush(heap, (-root_score + random.random() * 1e-6, root, root_path))

    seen = set()
    expansions = 0
    last_node = root
    last_path = root_path

    while heap:
        score, node, path = heapq.heappop(heap)
        expansions += 1

        try:
            chash = node.concept_hash()
        except Exception:
            chash = None
        if chash in seen:
            continue
        if chash is not None:
            seen.add(chash)

        try:
            children_list = list(iter_children(node))
            has_children = bool(children_list)
        except Exception:
            children_list = []
            has_children = False

        last_node = node
        last_path = path

        if not has_children:
            return node, path

        for child in children_list:
            try:
                child_hash = child.concept_hash()
            except Exception:
                child_hash = None
            child_sim = sim_of(child)
            child_path = path + ([f"CONCEPT-{child_hash}"] if child_hash is not None else [])
            heapq.heappush(heap, (-child_sim + random.random() * 1e-6, child, child_path))

    return last_node, last_path

def custom_categorize_dfs(inst, tree):
    """
    A helper function that categorizes down the Cobweb Tree and saves the concept hashes of all
    nodes to a list ["CONCEPT-{hash}", ...] and returns that list.

    Contains extensive error-catching, but the goal is that we return the whole list of the path
    traversed when categorizing the node.
    """
    path = []

    try:
        node = tree.root
    except Exception:
        try:
            leaf = tree.categorize(inst)
            return [f"CONCEPT-{leaf.concept_hash()}"]
        except Exception:
            return []

    # Append root concept
    try:
        path.append(f"CONCEPT-{node.concept_hash()}")
        lastNode = node
    except Exception:
        # if concept_hash is not available, return empty path
        return None, []

    while True:
        try:
            # log probabilities for each child (in order corresponding to node.children)
            child_scores = node.prob_children_given_instance(inst)
            # child_scores = []
            # for child in node.children:
            #     child_scores.append(node.pu_for_insert(child, inst) - child.pu_for_new_child(inst))
        except Exception as e:
            pass

        if not child_scores:
            break

        # print(child_scores)

        # find best child index (handle NaN by treating as -inf)
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

        # descend to selected child
        try:
            # Expect children to be a sequence of CobwebNode objects (as exposed by the C++ binding)
            node = node.children[best_idx]
        except Exception:
            # Some bindings may expose children as list of (attr, node) pairs -- try to handle that
            try:
                node = node.children[best_idx][1]
            except Exception:
                break

        try:
            path.append(f"CONCEPT-{node.concept_hash()}")
            lastNode = node
        except Exception:
            break

    # categorizeNode = tree.categorize(inst)

    return lastNode, path

def custom_categorize_bfs(inst, tree, max_expansions: int = 10000):
    """
    Priority-BFS categorization: explore nodes by priority until a leaf is found.

    By default the scorer attempts to use FiniteParseTree._score_function (which
    produces a 'cost' where smaller is better). If `use_log_probs=True` the BFS
    will instead score nodes by their average log-probability for `inst` (higher
    is better), computed from `node.log_prob_instance(inst)` normalized by total
    instance weight. The heap ordering is handled so both modes pick the "best"
    candidate first.

    Returns: (leaf_node, path)
    """
    try:
        root = tree.root
    except Exception:
        try:
            leaf = tree.categorize(inst)
            return leaf, [f"CONCEPT-{leaf.concept_hash()}"]
        except Exception:
            return None, []

    def sim_of(node):
        sd = FiniteParseTree._score_function(node, inst)
        return -1 * float(sd.get('cost', 0.0))

    def iter_children(n):
        try:
            for ch in n.children:
                if isinstance(ch, (list, tuple)) and len(ch) >= 2:
                    yield ch[1]
                else:
                    yield ch
        except Exception:
            return

    heap = []
    root_score = sim_of(root)
    root_path = [f"CONCEPT-{root.concept_hash()}"]
    heapq.heappush(heap, (-root_score + random.random() * 1e-6, root, root_path))

    seen = set()
    expansions = 0
    best_node = root
    best_path = root_path

    while heap and expansions < max_expansions:
        score, node, path = heapq.heappop(heap)
        expansions += 1

        try:
            chash = node.concept_hash()
        except Exception:
            chash = None
        if chash in seen:
            continue
        if chash is not None:
            seen.add(chash)

        try:
            children_list = list(iter_children(node))
            has_children = bool(children_list)
        except Exception:
            children_list = []
            has_children = False

        if not has_children:
            return node, path

        for child in children_list:
            try:
                child_hash = child.concept_hash()
            except Exception:
                child_hash = None
            child_sim = sim_of(child)
            child_path = path + ([f"CONCEPT-{child_hash}"] if child_hash is not None else [])
            heapq.heappush(heap, (-child_sim + random.random() * 1e-6, child, child_path))

    return best_node, best_path

def custom_categorize(inst, tree):
    return custom_categorize_bfs(inst, tree)

class FiniteParseTree:

    def __init__(self, ltm_hierarchy, id_to_value, value_to_id, context_length=3):
        self.ltm_hierarchy = ltm_hierarchy

        self.id_to_value = id_to_value
        self.value_to_id = value_to_id

        self.context_length = context_length

        self.global_root_node = CompositeParseNode.create_global_root()

        self.window = None

        self.nodes = []

        self.action_log = []
        self._undo_stack = []

        self._ensure_editor_state()

    def _ensure_editor_state(self):
        # call in __init__ of FiniteParseTree:
        if not hasattr(self, "action_log"):
            self.action_log = []
        if not hasattr(self, "_undo_stack"):
            self._undo_stack = []

    @staticmethod
    def _score_function(node: CobwebNode, instance: dict, lam=0.0, debug=False):
        """
        Compute a symmetric, scaled similarity between two attribute-value dictionaries.
        Returns a value in (0, 1].
        """

        # 1. Compute raw log-probability (uses node’s built-in method)
        log_prob = node.log_prob_instance(instance)

        # 2. Compute total weight (sum of all value frequencies)
        total_weight = sum(
            cnt for vAttr in instance.values() for cnt in vAttr.values()
        )
        if total_weight <= 0:
            total_weight = 1.0  # avoid division by zero

        # 3. Normalize log-probability
        avg_log_prob = log_prob / total_weight

        # 4. Compute node complexity (sum of all av_count entries)
        complexity = sum(
            cnt for attr_dict in node.av_count.values() for cnt in attr_dict.values()
        )

        # 5. Compute soft penalty term
        penalty = lam * math.log(1.0 + complexity)

        # 6. Define cost (higher = better)
        cost = -avg_log_prob + penalty

        score_data = {
            'raw_log_prob': log_prob,
            'avg_log_prob': avg_log_prob,
            'complexity': complexity,
            'cost': cost
        }

        if debug:
            print("-" * 60)
            print("Scoring statistics:")
            print(score_data)
            print("-" * 60)

        return score_data

    """
    -----------------------------------
    HELPER FUNCTIONS FOR INCREMENTAL BUILDING PROCESS!
    -----------------------------------
    """

    def build_primitives(self, window):
        """
        A custom function that we add (not given by GPT) to set up the primitive nodes to be combined!
        """
        self.window = window

        elements = re.findall(r"[\w']+|[.,!?;]", window)
        elements = [self.value_to_id[element] for element in elements]

        # Creating first layer of primitive nodes
        for i in range(len(elements)):

            content = elements[i]

            context_before_lst = elements[max(0, i - self.context_length):(i)][::-1]
            context_after_lst = elements[(i + 1):min(len(elements), i + self.context_length + 1)]

            # have to compute the dictionaries through a for loop so that we can
            # add multiple weights for multiple instances of the word

            context_before_dict = {0: 0} # initializing with the empty set!
            context_after_dict = {0: 0}

            for j in range(len(context_before_lst)):
                context_before_dict.setdefault(context_before_lst[j], 0)
                context_before_dict[context_before_lst[j]] += 1 / (j + 1) # this works because we reversed it above

            for j in range(len(context_after_lst)):
                context_after_dict.setdefault(context_after_lst[j], 0)
                context_after_dict[context_after_lst[j]] += 1 / (j + 1)

            node = PrimitiveParseNode(content, context_before_dict, context_after_dict, i)

            node.set_parent(self.global_root_node)

            self.nodes.append(node)


    def get_parentless_pairs(self):
        """
        Return a list of consecutive parentless pairs (as dicts) for the root-level
        nodes (the nodes that are children of global_root_node), in left-to-right order.
        Each dict: {"left_word_index": int, "right_word_index": int, "left_label": str, "right_label": str}
        """
        pairs = []
        parentless = [x[1] for x in self.global_root_node.children]
        for i in range(len(parentless) - 1):
            left = parentless[i]
            right = parentless[i + 1]
            left_label = self._safe_lookup(next(iter(left.content_left.keys())) if getattr(left, "content_left", None) else getattr(left, "content", None))
            # _safe_lookup handles None; for primitives we use content; for composite we used content_left first-key
            right_label = self._safe_lookup(next(iter(right.content_left.keys())) if getattr(right, "content_left", None) else getattr(right, "content", None))
            pairs.append({
                "left_word_index": left.word_index,
                "right_word_index": right.word_index,
                "left_title": left.title,
                "right_title": right.title,
                "left_label": left_label,
                "right_label": right_label
            })
        return pairs

    def _find_root_child_by_index(self, word_index):
        for wi, ch in self.global_root_node.children:
            if wi == word_index:
                return ch
        return None

    def evaluate_pair(self, left_word_index, right_word_index, debug=False):
        """
        Evaluate the best candidate for the pair of root-level nodes specified by their word indexes.
        Returns a serializable dict with:
        - merge_inst: instance dict produced by create_merge_instance (0..3 keys)
        - candidate_concept_hash / id
        - score (float)
        - debug: numeric debug stats
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        merge_inst = CompositeParseNode.create_merge_instance(left_node, right_node)
        # compute the categorize_path for this merged instance (list of concept ids)

        candidate_concept, categorize_path = custom_categorize(merge_inst, self.ltm_hierarchy)

        if debug:
            print("Categorization Path:")
            print(categorize_path)

        try:
            categorize_path = [self.value_to_id.get(key) for key in categorize_path]
        except Exception:
            categorize_path = []

        candidate_hash = candidate_concept.concept_hash()
        candidate_id = self.value_to_id.get(f"CONCEPT-{candidate_hash}")

        score_data = FiniteParseTree._score_function(candidate_concept, merge_inst, debug=debug)
        score = score_data["cost"]

        # want to visualize candidate_concept.av_count, need to use draw_dict:
        if len(candidate_concept.av_count) == 0:
            candidate_draw_inst = {
                "title": candidate_hash,
                "left": [],
                "right": [],
                "before": [],
                "after":  [],
                "children": []
            }
        else:
            candidate_draw_inst = {
                "title": candidate_hash,
                "left": self.ctx_list(candidate_concept.av_count[0]),
                "right": self.ctx_list(candidate_concept.av_count[1]),
                "before": self.ctx_list(candidate_concept.av_count[2]),
                "after":  self.ctx_list(candidate_concept.av_count[3]),
                "children": []
            }

        returnData = {
            "merge_inst": merge_inst,
            "categorize_path": categorize_path,
            "candidate_concept_hash": candidate_hash,
            "candidate_concept_id": candidate_id,
            "candidate_inst": candidate_draw_inst,
            "score": score,
            "debug": score_data,
            "left_word_index": left_word_index,
            "right_word_index": right_word_index,
            "left_title": left_node.title,
            "right_title": right_node.title
        }
                
        # print(returnData)

        return returnData

    def apply_candidate(self, left_word_index, right_word_index, candidate_concept_hash=None):
        """
        Apply the best candidate merge for the pair and modify the tree.
        If candidate_concept_hash is None, recompute via categorize (same as evaluate_pair).
        Returns the added node's serialized info (and appends to action_log).
        Pushes undo information to _undo_stack.
        """
        left_node = self._find_root_child_by_index(left_word_index)
        right_node = self._find_root_child_by_index(right_word_index)
        if left_node is None or right_node is None:
            raise ValueError("Left or right node not found among root's children")

        merge_inst = CompositeParseNode.create_merge_instance(left_node, right_node)

        candidate_concept, categorize_path = custom_categorize(merge_inst, self.ltm_hierarchy)

        # compute categorize_path for the merged instance and store on node
        try:
            categorize_path = [self.value_to_id.get(key) for key in categorize_path]
        except Exception:
            categorize_path = []

        if candidate_concept_hash is not None and candidate_concept.concept_hash() != candidate_concept_hash:
            # if user gave an explicit hash, try to find that concept in hierarchy -
            # but default behavior uses categorize(...) basic level, same as build()
            pass
        candidate_id = self.value_to_id.get(f"CONCEPT-{candidate_concept.concept_hash()}")

        # create the composite parse node consistent with build() behavior
        add_parse_node = CompositeParseNode.create_node(
            merge_inst,
            candidate_id,
            categorize_path,
            0.5 * (left_node.word_index + right_node.word_index)
        )

        # Update tree structures (append node and re-parent children)
        self.nodes.append(add_parse_node)
        add_parse_node.set_parent(self.global_root_node)
        left_node.set_parent(add_parse_node)
        right_node.set_parent(add_parse_node)

        # Prepare undo entry: we will store enough to revert the change simply
        undo_entry = {
            "action": "apply_candidate",
            "added_node_title": add_parse_node.title,
            "added_node_word_index": add_parse_node.word_index,
            "left_word_index": left_node.word_index,
            "right_word_index": right_node.word_index,
            "timestamp": time.time()
        }
        self._undo_stack.append(undo_entry)

        # Append action log entry
        log_entry = {
            "timestamp": time.time(),
            "type": "apply_candidate",
            "description": f"Applied chunk combining {left_node.title} ({left_node.word_index}) + {right_node.title} ({right_node.word_index}) -> concept CONCEPT-{candidate_concept.concept_hash()}",
            "payload": {
                "left": {"title": left_node.title, "word_index": left_node.word_index},
                "right": {"title": right_node.title, "word_index": right_node.word_index},
                "new_node": {"title": add_parse_node.title, "word_index": add_parse_node.word_index, "concept_id": candidate_id}
            }
        }
        self.action_log.append(log_entry)

        return {
            "ok": True,
            "added_node": {
                "title": add_parse_node.title,
                "word_index": add_parse_node.word_index,
                "concept_id": candidate_id
            },
            "action_log_entry": log_entry
        }

    def undo(self):
        """
        Undo the last apply_candidate action. Returns True/False for success.
        Current implementation only supports undoing the last apply_candidate (stack).
        """
        if not self._undo_stack:
            return {"ok": False, "reason": "Nothing to undo!"}

        entry = self._undo_stack.pop()
        if entry["action"] != "apply_candidate":
            return {"ok": False, "reason": "Unsupported undo action!"}

        # find the added node by title and remove it
        added_title = entry["added_node_title"]
        added_node = next((n for n in self.nodes if n.title == added_title), None)
        if added_node is None:
            return {"ok": False, "reason": "Added node not found!"}

        # re-parent its children back to the global root
        left_w = entry["left_word_index"]
        right_w = entry["right_word_index"]
        left_node = self._find_root_child_by_index(left_w) or next((n for n in self.nodes + [self.global_root_node] if n.word_index == left_w), None)
        right_node = self._find_root_child_by_index(right_w) or next((n for n in self.nodes + [self.global_root_node] if n.word_index == right_w), None)

        # If left/right_node currently have parent == added_node, reparent to global_root_node
        try:
            for wi, ch in list(added_node.children):
                ch.set_parent(self.global_root_node)
        except Exception:
            pass

        # remove added_node from nodes list
        try:
            # added_node.set_parent(None) # hopefully a thing
            self.global_root_node.children.remove((added_node.word_index, added_node))
            self.nodes.remove(added_node)
        except ValueError:
            pass

        # remove matching action_log entry (last matching apply_candidate) if exists
        for i in range(len(self.action_log)-1, -1, -1):
            if self.action_log[i]["type"] == "apply_candidate" and self.action_log[i]["payload"]["new_node"]["title"] == added_title:
                removed = self.action_log.pop(i)
                break

        return {"ok": True, "undone": added_title}

    def export_json(self, filepath=None):
        """
        Export parse tree as JSON; if filepath provided, save there. Also append action_log entry.
        """
        res = self.to_json(filepath=filepath)
        log_entry = {
            "timestamp": time.time(),
            "type": "export",
            "description": f"Exported parse tree to {filepath or 'json-string'}",
            "payload": {"filepath": filepath}
        }
        self.action_log.append(log_entry)
        return {"ok": True, "export_result": res, "action_log_entry": log_entry}
    
    """
    REGULAR FUNCTIONS CONTINUE BELOW
    """

    def build(self, window, end_behavior="converge", debug=False):
        """
        Primary method of construction that returns all available nonterminals
        as instances ready to be passed into the long-term hierarchy. The
        following process takes place:
        *   First, all chunk candidates are proposed based on the current set of
            non-terminals without parents.
        *   Each candidate is categorized within the Cobweb hierarchy, and the
            candidate that finds the best fit is added to the parse tree.
            *   I believe Chris summarized this well as the chunk that finds the
                concept that has the highest probability of generating that chunk is
                added to the parse tree.

        IMPORTANT PARAM: "end_behavior"
        *   Can be "converge" to represent the tree converging on one root
        *   A float to represent the tree continually updating until no candidates
            proposed have a valuable enough addition.

        IMPORTANT PARAM: "content_ids" --> can be "longterm" to represent that composite nodes,
        after content, can inherit from localized IDs

        Process for creating a non-terminal parse node:
        *   initialize the parse node based on an instance dictionary
        *   assign it its respective concept label
        *   because it's a new parse node, connect it to the global root node and
            its two children nodes

        Returns:
        *   the list of new concept labels that are required
        """

        self.window = window

        self.build_primitives(window)

        while True:
            parentless_pairs = self.get_parentless_pairs()
            if not parentless_pairs:
                break

            best = None  # tuple (score, eval_result)

            for p in parentless_pairs:
                try:
                    res = self.evaluate_pair(p["left_word_index"], p["right_word_index"], debug=debug)
                except Exception as e:
                    if debug:
                        print(f"evaluate_pair failed for {p}: {e}")
                    continue

                score = res.get("score", float("-inf"))
                if best is None or score > best[0]:
                    best = (score, res)

            if best is None:
                break

            if isinstance(end_behavior, (int, float)):
                if best[0] < end_behavior:
                    break

            # Apply the chosen candidate
            chosen = best[1]
            left_idx = chosen["left_word_index"]
            right_idx = chosen["right_word_index"]
            candidate_hash = chosen.get("candidate_concept_hash")

            try:
                self.apply_candidate(left_idx, right_idx, candidate_concept_hash=candidate_hash)
            except Exception as e:
                if debug:
                    print(f"apply_candidate failed for {left_idx},{right_idx}: {e}")
                break

            if end_behavior == "converge":
                if len(self.global_root_node.children) <= 1:
                    break

        return True
        

    def get_chunk_instances(self):
        """
        Primary method that returns all available nonterminals as instances
        ready to be passed into the long-term hierarchy
        """

        instances = []

        def dfs_insts(node):
            if type(node) == PrimitiveParseNode:
                return

            instances.append(node.get_as_instance())

            for _, child in node.children:
                dfs_insts(child)

        for _, child in self.global_root_node.children:
            dfs_insts(child)

        return instances

    def visualize(self, out_base="parse_tree", render_png=True):
        """
        Render the parse tree into an HTML file and optionally a PNG screenshot.
        The PNG height automatically adjusts to fit the tree, including all nodes.
        """

        # Convert tree to JSON and build HTML
        d3_json = json.dumps(self._draw_tree_to_json())
        html = self._build_html(d3_json)

        html_path = f"{out_base}.html"
        png_path = f"{out_base}.png"

        # Ensure output directories exist
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

        # Write HTML file
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        if render_png:
            asyncio.run(self._html_to_png(html_path, png_path))
            return html_path, png_path
        else:
            return html_path

    async def _html_to_png(self, html_path, png_path):
        """
        Convert the HTML tree to a PNG screenshot using Playwright.
        Automatically adjusts the PNG size to fit the full tree height.
        """

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("file://" + os.path.abspath(html_path))

            # Wait until the tree SVG exists
            await page.wait_for_selector("#tree svg")

            # Wait for layout of dynamic node content to stabilize
            await page.evaluate("""
                () => new Promise(resolve => {
                    requestAnimationFrame(() => requestAnimationFrame(resolve));
                })
            """)

            # Measure the wrapper container including all overflow from foreignObjects
            bounding_box = await page.evaluate("""
                () => {
                    const container = document.querySelector('#tree-container');
                    return {
                        width: Math.ceil(container.scrollWidth) + 20,
                        height: Math.ceil(container.scrollHeight) + 20
                    };
                }
            """)

            # Set viewport to actual content size
            await page.set_viewport_size({
                "width": bounding_box["width"],
                "height": bounding_box["height"]
            })

            # Take a screenshot of the SVG
            svg_elem = await page.query_selector("#tree svg")
            await svg_elem.screenshot(path=png_path, scale="css")

            await browser.close()

    def _safe_lookup(self, idx):
        # If idx is already a string (e.g. "CONCEPT-..."), return it directly
        if isinstance(idx, str):
            return idx

        # If idx is an integer index into id_to_value, return the mapped value
        try:
            if idx is not None and isinstance(idx, int) and 0 <= idx < len(self.id_to_value):
                return self.id_to_value[idx]
        except Exception:
            pass

        # fallback
        return "None"

    # new logic for 0:0 claims
    def ctx_list(self, ctx, draw_zeros=False, max_size=7):
        if not ctx:
            return []
        # sort by descending value then key (matching prior behavior)
        items = sorted(ctx.items(), key=lambda kv: (-kv[1], kv[0]))
        # filter out the EMPTYNULL key (0) unless requested, BEFORE truncating to max_size
        if not draw_zeros:
            items = [(k, v) for (k, v) in items if k != 0]
        # respect the max_size parameter (was hard-coded to 7 previously)
        if len(items) > max_size:
            items = items[:max_size]
        x = [{"key": self._safe_lookup(k), "val": float(v)} for k, v in items]
        return x

    def _draw_node_to_dict(self, node, children_getter, draw_zeros=False):
        """
        Note: because this is purely for drawing, we should be able to remove the EMPTYNULL here!
        """

        if isinstance(node, PrimitiveParseNode):
            # represent primitive content as a single-entry list to keep a consistent shape
            left_list = [{"key": self._safe_lookup(node.content), "val": 1.0}]
            return {
                "title": node.title,
                "left": left_list,
                "right": [],  # primitives have no right content
                "before": self.ctx_list(node.context_before, draw_zeros),
                "after":  self.ctx_list(node.context_after, draw_zeros),
                "children": [self._draw_node_to_dict(ch[1], children_getter) for ch in children_getter(node)]
            }

        elif isinstance(node, CompositeParseNode):
            # content_left/content_right are dicts -> convert to same list-of-{key,val} shape
            left_list  = self.ctx_list(node.content_left, draw_zeros)
            right_list = self.ctx_list(node.content_right, draw_zeros)

            return {
                "title": node.title,
                "left": left_list,
                "right": right_list,
                "before": self.ctx_list(node.context_before, draw_zeros),
                "after":  self.ctx_list(node.context_after, draw_zeros),
                "children": [self._draw_node_to_dict(ch[1], children_getter) for ch in children_getter(node)]
            }

        else:
            raise TypeError(f"Unknown node type {type(node)}")


    def _draw_tree_to_json(self):
        def children_getter(n):
            for wi, ch in getattr(n, "children", []):
                yield (wi, ch)
        return self._draw_node_to_dict(self.global_root_node, children_getter)

    def _build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>Parse Tree</title>
    <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    #tree-container {{ display: inline-block; }}
    .link {{ fill: none; stroke: #9aa1a9; stroke-width: 1.5px; }}
    .node-box {{ stroke: #444; fill: #fff; rx: 8; ry: 8; filter: drop-shadow(1px 2px 2px rgba(0,0,0,0.15)); }}
    .node-fo table {{ border-collapse: collapse; font-size: 12px; margin: 4px 0; }}
    .node-fo th, .node-fo td {{ border: 1px solid #888; padding: 2px 6px; }}
    .node-fo th {{ background: #f3f5f7; font-weight: 600; }}
    .section-title {{ font-weight: bold; margin-top: 4px; }}
    .section {{ margin-top: 10px; margin-bottom: 10px; }}
    .subtable b {{ display: inline-block; margin: 6px 0 2px; }}
    .subtable table {{ border-collapse: collapse; }}
    .subtable td {{ border: 1px solid #bbb; padding: 1px 4px; }}
    </style>
    </head>
    <body>
    <div id="tree-container">
    <div id="tree"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
    const data = {d3_data_json};
    const nodeW  = {node_w};
    const nodeH  = {node_h};
    const hGap   = {h_gap};
    const vGap   = {v_gap};

    const root = d3.hierarchy(data);
    const layout = d3.tree().nodeSize([nodeW + hGap, nodeH + vGap]);
    layout(root);

    // compute bounds
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    root.each(d => {{
        if (d.x < x0) x0 = d.x;
        if (d.x > x1) x1 = d.x;
        if (d.y < y0) y0 = d.y;
        if (d.y > y1) y1 = d.y;
    }});
    const width  = x1 - x0 + nodeW + 320;
    const height = y1 - y0 + nodeH + 320; // THIS IS THE HEIGHT MODIFIER

    const svg = d3.select("#tree").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [x0 - nodeW/2, y0 - nodeH/2, width, height].join(" "));

    const g = svg.append("g");

    // links
    g.selectAll("path.link")
    .data(root.links())
    .join("path")
    .attr("class", "link")
    .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

    // nodes
    const node = g.selectAll("g.node")
    .data(root.descendants())
    .join("g")
    .attr("transform", d => `translate(${{d.x}},${{d.y}})`);

    // node rect
    node.append("rect")
    .attr("class", "node-box")
    .attr("x", -nodeW/2)
    .attr("y", 0)
    .attr("width", nodeW)
    .attr("height", nodeH);

    // node HTML via foreignObject
    node.append("foreignObject")
    .attr("class", "node-fo")
    .attr("x", -nodeW/2 + 6)
    .attr("y", 6)
    .attr("width", nodeW - 12)
    .attr("height", 1000)
    .html(d => nodeHTML(d.data));

    // shrink foreignObjects to actual content height
    node.selectAll("foreignObject").each(function() {{
    const fo = d3.select(this);
    const div = fo.select("div").node();
    const h = div.getBoundingClientRect().height + 6;
    fo.attr("height", h);
    d3.select(this.parentNode).select("rect").attr("height", h + 12);
    }});

    function nodeHTML(d) {{
        const ctxTable = (ctx, title) => {{
            if (!ctx || ctx.length === 0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
            const rows = ctx.map(kv => `<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
            return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
        }};

        // content: left and right are now lists of {{key, val}}
        let contentHTML = "";
        const leftHas = Array.isArray(d.left) && d.left.length > 0;
        const rightHas = Array.isArray(d.right) && d.right.length > 0;

        if (rightHas) {{
            // composite-style node: show both left and right content tables
            contentHTML = `<div class="section">${{ctxTable(d.left, "Content-Left")}}${{ctxTable(d.right, "Content-Right")}}</div>`;
        }} else if (leftHas) {{
            // primitive or single-sided composite: show a single content table
            // label it "Content" for primitives
            const label = (d.right && d.right.length > 0) ? "Content-Left" : "Content";
            contentHTML = `<div class="section">${{ctxTable(d.left, label)}}</div>`;
        }} else {{
            contentHTML = `<div class="section"><i>Content: empty</i></div>`;
        }}

        return `
        <div class="node-fo">
            <table><tr><th colspan="2">${{d.title}}</th></tr></table>
            ${{contentHTML}}
            ${{ctxTable(d.before, "Context-Before")}}
            ${{ctxTable(d.after,  "Context-After")}}
        </div>`;
    }}

    </script>
    </body>
    </html>
    """

    def editor_build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
        sentence_str = self.window  # current sentence display
        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>Parse Tree Editor</title>
    <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    #editor-container {{ display: flex; flex-direction: row; height: 100vh; }}
    #tree-panel {{ flex: 3; overflow: auto; border-right: 1px solid #ccc; padding: 12px; }}
    #sidebar {{ flex: 1; overflow-y: auto; padding: 12px; background: #f9f9f9; }}
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
    </style>
    </head>
    <body>
    <div id="header">
        <h2>Parse Tree Editor</h2>
        <h4>Current sentence: <span id="sentence-text">{sentence_str}</span></h4>
        <button id="undo-btn">Undo Last Chunk</button>
        <button id="export-btn">Export Tree</button>
        <button id="export-ltm-btn">Export LTM</button>
    </div>
    <div id="editor-container">
        <div id="tree-panel">
            <div id="tree"></div>
        </div>
        <div id="sidebar">
            <div id="pair-buttons"><strong>Candidate Pairs:</strong></div>
            <div><strong>Action Log:</strong><ul id="action-log"></ul></div>
        </div>
    </div>

    <!-- Candidate Modal -->
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

    // --- D3 tree rendering ---
    const nodeW  = {node_w};
    const nodeH  = {node_h};
    const hGap   = {h_gap};
    const vGap   = {v_gap};

    function renderTree(data){{
        d3.select("#tree").selectAll("*").remove();
        const root = d3.hierarchy(data);
        const layout = d3.tree().nodeSize([nodeW+hGap, nodeH+vGap]);
        layout(root);

        let x0=Infinity, x1=-Infinity, y0=Infinity, y1=-Infinity;
        root.each(d=>{{ x0=Math.min(x0,d.x); x1=Math.max(x1,d.x); y0=Math.min(y0,d.y); y1=Math.max(y1,d.y); }});
        const width=x1-x0+nodeW+320, height=y1-y0+nodeH+320;

        const svg = d3.select("#tree").append("svg")
            .attr("width", width).attr("height", height)
            .attr("viewBox",[x0-nodeW/2, y0-nodeH/2, width, height].join(" "));

        const g = svg.append("g");

        g.selectAll("path.link")
            .data(root.links())
            .join("path")
            .attr("class","link")
            .attr("fill","none")
            .attr("stroke","#9aa1a9")
            .attr("stroke-width",1.5)
            .attr("d", d3.linkVertical().x(d=>d.x).y(d=>d.y));

        const node = g.selectAll("g.node")
            .data(root.descendants())
            .join("g")
            .attr("transform", d=>`translate(${{d.x}},${{d.y}})`);

        node.append("rect")
            .attr("class","node-box")
            .attr("x",-nodeW/2)
            .attr("y",0)
            .attr("width",nodeW)
            .attr("height",nodeH)
            .attr("stroke","#444")
            .attr("fill","#fff")
            .attr("rx",8)
            .attr("ry",8);

        node.append("foreignObject")
            .attr("class","node-fo")
            .attr("x",-nodeW/2+6)
            .attr("y",6)
            .attr("width",nodeW-12)
            .attr("height",1000)
            .html(d=>nodeHTML(d.data));

        node.selectAll("foreignObject").each(function(){{
            const fo=d3.select(this);
            const div=fo.select("div").node();
            const h=div.getBoundingClientRect().height+6;
            fo.attr("height",h);
            d3.select(this.parentNode).select("rect").attr("height",h+12);
        }});
    }}

    function nodeHTML(d){{
        const ctxTable=(ctx,title)=>{{
            if(!ctx||ctx.length===0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
            const rows = ctx.map(kv=>`<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
            return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
        }};
        let contentHTML="";
        const leftHas=Array.isArray(d.left)&&d.left.length>0;
        const rightHas=Array.isArray(d.right)&&d.right.length>0;
        if(rightHas) {{
            contentHTML = `<div class="section">${{ctxTable(d.left,"Content-Left")}}${{ctxTable(d.right,"Content-Right")}}</div>`;
        }} else if(leftHas) contentHTML=`<div class="section">${{ctxTable(d.left,d.right&&d.right.length>0?"Content-Left":"Content")}}</div>`;
        else contentHTML=`<div class="section"><i>Content: empty</i></div>`;
        return `<div class="node-fo">
            <table><tr><th colspan="2">${{d.title}}</th></tr></table>
            ${{contentHTML}}
            ${{ctxTable(d.before,"Context-Before")}}
            ${{ctxTable(d.after,"Context-After")}}
        </div>`;
    }}

    // --- Sidebar action log update ---
    function updateLog(log){{
        const ul = document.getElementById("action-log");
        ul.innerHTML="";
        log.forEach(e=>{{ ul.innerHTML += `<li>[${{new Date(e.timestamp*1000).toLocaleTimeString()}}] ${{e.description}}</li>`; }});
    }}

    // --- Pair buttons ---
    function loadPairs(){{
        fetch("/api/tree").then(r=>r.json()).then(data=>{{
            const container = document.getElementById("pair-buttons");
            container.innerHTML="<strong>Candidate Pairs:</strong>";
            const s = document.getElementById("sentence-text");
            if(s && data.sentence) s.textContent = data.sentence;
            data.pairs.forEach(p=>{{
                const btn=document.createElement("button");
                btn.textContent=`${{p.left_title}} + ${{p.right_title}}`;
                btn.onclick=()=> evaluatePair(p.left_word_index,p.right_word_index);
                container.appendChild(btn);
            }});
            updateLog(data.action_log);
            renderTree(data.tree);
        }});
    }}

    // --- Evaluate a pair ---
    function evaluatePair(left,right){{
        currentLeft=left; currentRight=right;
        fetch("/api/evaluate", {{
            method:"POST", headers:{{"Content-Type":"application/json"}},
            body: JSON.stringify({{left_word_index:left,right_word_index:right,debug:true}})
        }}).then(r=>r.json()).then(res=>{{
            if(res.ok) showCandidateModal(res.result);
            else alert(res.error);
        }});
    }}

    // --- Candidate Modal ---
    const modal = document.getElementById("candidate-modal");
    const spanClose = modal.querySelector(".close");
    spanClose.onclick = ()=> modal.style.display="none";
    window.onclick = e=> {{ if(e.target==modal) modal.style.display="none"; }};
    function showCandidateModal(result){{
        const candidate = result.candidate_inst; // candidate node data
        document.getElementById("candidate-title").textContent = result.candidate_concept_id || result.candidate_concept_hash;
        document.getElementById("candidate-score").textContent = result.score.toFixed(3);

        const dbg = document.getElementById("candidate-debug");
        dbg.innerHTML = ""; // clear previous content

        // --- Build node-style table ---
        function ctxTable(ctx, title) {{
            if(!ctx || ctx.length===0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
            const rows = ctx.map(kv => `<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
            return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
        }}

        function buildCandidateHTML(d){{
            let contentHTML="";
            const leftHas = Array.isArray(d.left) && d.left.length>0;
            const rightHas = Array.isArray(d.right) && d.right.length>0;
            if(rightHas){{
                contentHTML = `<div class="section">${{ctxTable(d.left,"Content-Left")}}${{ctxTable(d.right,"Content-Right")}}</div>`;
            }} else if(leftHas){{
                contentHTML = `<div class="section">${{ctxTable(d.left,d.right && d.right.length>0?"Content-Left":"Content")}}</div>`;
            }} else {{
                contentHTML = `<div class="section"><i>Content: empty</i></div>`;
            }}
            return `
                <table><tr><th colspan="2">${{d.title}}</th></tr></table>
                ${{contentHTML}}
                ${{ctxTable(d.before,"Context-Before")}}
                ${{ctxTable(d.after,"Context-After")}}
            `;
        }}

        // --- Build debug table ---
        function buildDebugHTML(debugObj){{
            let html = `<div class="subtable"><b>Debug Stats</b><table><tr><th>Stat</th><th>Value</th></tr>`;
            for(const [k,v] of Object.entries(debugObj)){{
                html += `<tr><td>${{k}}</td><td>${{v===null ? "null" : v.toFixed ? v.toFixed(3) : v}}</td></tr>`;
            }}
            html += `</table></div>`;
            return html;
        }}

        // Render both: candidate table + debug stats
        dbg.innerHTML = buildCandidateHTML(candidate) + buildDebugHTML(result.debug);

        modal.style.display = "block";
    }}

    // --- Apply candidate ---
    document.getElementById("apply-candidate-btn").onclick = ()=>{{
        if(currentLeft===null||currentRight===null) return;
        if(!confirm("Confirm applying this chunk?")) return;
        fetch("/api/apply", {{
            method:"POST", headers:{{"Content-Type":"application/json"}},
            body: JSON.stringify({{left_word_index:currentLeft,right_word_index:currentRight}})
        }}).then(r=>r.json()).then(res=>{{
            if(res.ok){{ loadPairs(); modal.style.display="none"; }}
            else alert(res.error);
        }});
    }}

    // --- Undo ---
    document.getElementById("undo-btn").onclick = ()=>{{
        fetch("/api/undo",{{method:"POST"}}).then(r=>r.json()).then(res=>{{
            if(res.ok) loadPairs();
            else alert(res.reason||"Undo failed");
        }});
    }}

    // --- Export ---
    document.getElementById("export-btn").onclick = ()=>{{
        const fp = prompt("Enter filepath to export (optional):","");
        fetch("/api/export",{{
            method:"POST",
            headers:{{"Content-Type":"application/json"}},
            body: JSON.stringify({{filepath:fp}})
        }})
        .then(r=>r.json())
        .then(res=>{{
            if(res.ok){{
                alert("Parse tree exported and LTM updated!");
                if(res.refresh){{
                    // update sentence text live
                    const s = document.getElementById("sentence-text");
                    if(s && res.new_sentence) s.textContent = res.new_sentence;
                    // slight delay before refreshing to new tree
                    setTimeout(()=> location.reload(), 800);
                }} else {{
                    loadPairs();
                }}
            }} else {{
                alert(res.error || "Export failed");
            }}
        }})
        .catch(err => alert("Network error: " + err));
    }};

    // --- Export LTM ---
    document.getElementById("export-ltm-btn").onclick = () => {{
        const fp = prompt("Enter filepath to save LTM JSON (optional):","");
        fetch("/api/export_ltm", {{
            method: "POST",
            headers: {{"Content-Type":"application/json"}},
            body: JSON.stringify({{filepath: fp}})
        }})
        .then(r => r.json())
        .then(res => {{
            if(res.ok){{
                alert("LTM exported!" + (res.filepath ? " Saved to: " + res.filepath : ""));
            }} else {{
                alert("Export failed: " + (res.error || "Unknown error"));
            }}
        }})
        .catch(err => alert("Network error: " + err));
    }};


    // --- Initial load ---
    loadPairs();

    </script>
    </body>
    </html>
    """



    def to_json(self, filepath=None):
        """
        Serialize the ParseTree into JSON. Optionally save to `filepath`.
        """

        def serialize_node(node, index_map):
            if isinstance(node, PrimitiveParseNode):
                return {
                    "node_type": "primitive",
                    "title": node.title,
                    "word_index":node.word_index,
                    "content": node.content,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "concept_label": node.concept_label,
                    "global_root": node.global_root,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            elif isinstance(node, CompositeParseNode):
                return {
                    "node_type": "composite",
                    "title": node.title,
                    "word_index":node.word_index,
                    "content_left": node.content_left,
                    "content_right": node.content_right,
                    "categorize_path": node.categorize_path,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "concept_label": node.concept_label,
                    "global_root": node.global_root,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            else:
                raise TypeError(f"Unknown node type {type(node)}")

        index_map = {node: i for i, node in enumerate([self.global_root_node] + self.nodes)}
        data = {
            "window": self.window,
            "context_length": self.context_length,
            "id_to_value": self.id_to_value,
            "value_to_id": self.value_to_id,
            "nodes": [serialize_node(node, index_map) for node in [self.global_root_node] + self.nodes],
        }

        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return filepath
        else:
            return json.dumps(data, indent=2)

    @staticmethod
    def from_json(data, ltm_hierarchy, filepath=False):
        """
        Deserialize a ParseTree from JSON. Requires the same ltm_hierarchy instance.
        """
        if filepath:
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(data, str):
            data = json.loads(data)

        tree = FiniteParseTree(
            ltm_hierarchy,
            id_to_value=data["id_to_value"],
            value_to_id=data["value_to_id"],
            context_length=data["context_length"],
        )
        tree.window = data["window"]

        def restore_dict_keys(d):
            if d is None:
                return None
            return {int(k): v for k, v in d.items()}

        node_objs = []
        for n in data["nodes"]:
            if n["node_type"] == "primitive":
                node = PrimitiveParseNode(
                    content=n["content"],
                    context_before=restore_dict_keys(n["context_before"]),
                    context_after=restore_dict_keys(n["context_after"]),
                    word_index=n["word_index"],
                )
            elif n["node_type"] == "composite":
                node = CompositeParseNode()
                node.content_left = restore_dict_keys(n["content_left"])
                node.content_right = restore_dict_keys(n["content_right"])
                # restore categorize_path if present (list of concept ids)
                node.categorize_path = n.get("categorize_path")
                node.context_before = restore_dict_keys(n["context_before"])
                node.context_after = restore_dict_keys(n["context_after"])
                node.word_index = n["word_index"]

            else:
                raise ValueError(f"Unknown node_type {n['node_type']}")

            node.title = n["title"]
            node.concept_label = n["concept_label"]
            node.global_root = n["global_root"]
            node_objs.append(node)

        # restore parent/child relations
        for idx, n in enumerate(data["nodes"]):
            node = node_objs[idx]
            parent_idx = n["parent"]
            if parent_idx is not None:
                node.parent = node_objs[parent_idx]
            node.children = [(node_objs[ch].word_index, node_objs[ch]) for ch in n["children"]]

        tree.global_root_node = node_objs[0]
        tree.nodes = node_objs[1:]

        return tree

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

@DeprecationWarning
class IncrementalParseTree:
    """
    A class extremely similar to the parse tree given above, but one that takes in a stream of
    data continuously and decides to add to the parse tree until no longer possible by either
    of the end conditions mentioned above (nothing left to merge or ran out of data). In this
    implementation, we fix the size of the working memory, and for each merge, we add the 

    Note that we also need to add a function to the LanguageChunkingParser that supports a
    continuous stream of input (no defined length).

    Important design decision to confirm:
    *   What is the termination quality of this streaming-based method?
        *   The termination quality should either be the end of the corpus (which is chill)
            or the point at which nothing useful can be merged to reveal higher quality, at
            which point everything from the working memory is dumped and a new set is read in.
        *   I'm much more a fan of the latter right now, but we can work through both as we
            proceed. The former will probably result in unstable queries
    *   When are instances added to the parse tree?
        *   Should chunk candidates be added as soon as they are wrapped into the continuous
            parse tree or should they be added at termination point?
        *   If the criteria we choose for the first decision is termination of a single parse tree
            once the full corpus has been parsed, that'll lead to increasingly painful and unstable
            initial trees.
    *   How do we determine the most optimal candidate?!?
        *   This is an unsolved question from the FiniteParseTree situation - once we can better
            attribute the additions through GUI analysis, we can probably iterate better on this.
        *   We'll go by a best-fit candidate and a 

    Code Analysis:
    *   We initialize a tree-based structure similar to the previous, but we initialize a "working
        memory manager" and make the following changes:
        *   Save all possible changes (potentially create a separate datastructure for this)
            with the score, the categorized ID, and the two children indexes for which we've 
            categorized the combination of the pairwise nodes
        *   The "working memory" represents the topmost level of parentless nodes, which will all be
            connected to the global root node and their pairwise combinations will be 
    *   Pseudocode:
        *   Fill the working memory with the first slew of primitive instances (and related datastructures)
        *   While the priority queue does not empty:
            *   Add to working memory until working memory has reached capacity
            *   Evaluate any new chunk candidates (based on the strictly new instances) and add them to the
                priority queue
            *   Select the best candidate according to the priority queue, pop it, and add it to the tree
                and add it to the long-term memory
    *   The important thing to consider is that the global root node becomes infinitely more important
        in this case than in the finite case

    """
    def __init__(self):
        pass

    def build(self, corpus):
        pass

    def get_chunk_instances(self):
        pass

    def visualize(self, out_base="parse_tree", render_png=True):
        """
        Render the parse tree into an HTML file and optionally a PNG screenshot.
        The PNG height automatically adjusts to fit the tree, including all nodes.
        """

        # Convert tree to JSON and build HTML
        d3_json = json.dumps(self._tree_to_json())
        html = self._build_html(d3_json)

        html_path = f"{out_base}.html"
        png_path = f"{out_base}.png"

        # Ensure output directories exist
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

        # Write HTML file
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        if render_png:
            asyncio.run(self._html_to_png(html_path, png_path))
            return html_path, png_path
        else:
            return html_path

    async def _html_to_png(self, html_path, png_path):
        """
        Convert the HTML tree to a PNG screenshot using Playwright.
        Automatically adjusts the PNG size to fit the full tree height.
        """

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("file://" + os.path.abspath(html_path))

            # Wait until the tree SVG exists
            await page.wait_for_selector("#tree svg")

            # Wait for layout of dynamic node content to stabilize
            await page.evaluate("""
                () => new Promise(resolve => {
                    requestAnimationFrame(() => requestAnimationFrame(resolve));
                })
            """)

            # Measure the wrapper container including all overflow from foreignObjects
            bounding_box = await page.evaluate("""
                () => {
                    const container = document.querySelector('#tree-container');
                    return {
                        width: Math.ceil(container.scrollWidth) + 20,
                        height: Math.ceil(container.scrollHeight) + 20
                    };
                }
            """)

            # Set viewport to actual content size
            await page.set_viewport_size({
                "width": bounding_box["width"],
                "height": bounding_box["height"]
            })

            # Take a screenshot of the SVG
            svg_elem = await page.query_selector("#tree svg")
            await svg_elem.screenshot(path=png_path, scale="css")

            await browser.close()

    def _safe_lookup(self, idx):
        if (idx is not None and 0 <= idx < len(self.id_to_value)):
            return self.id_to_value[idx]
        else:
            # print("index", idx)
            return "None"

    def _node_to_dict(self, node, children_getter):
        def ctx_list(ctx):
            if not ctx:
                return []
            items = sorted(ctx.items(), key=lambda kv: (-kv[1], kv[0]))
            return [{"key": self._safe_lookup(k), "val": float(v)} for k, v in items]

        if isinstance(node, PrimitiveParseNode):
            return {
                "title": node.title,
                "left": self._safe_lookup(node.content),
                "right": None,
                "before": ctx_list(node.context_before),
                "after": ctx_list(node.context_after),
                "children": [self._node_to_dict(ch[1], children_getter) for ch in children_getter(node)]
            }

        elif isinstance(node, CompositeParseNode):
            left_id  = None if not node.content_left  else next(iter(node.content_left.keys()))
            right_id = None if not node.content_right else next(iter(node.content_right.keys()))
            left  = self._safe_lookup(left_id)
            right = self._safe_lookup(right_id)

            return {
                "title": node.title,
                "left": left,
                "right": right,
                "before": ctx_list(node.context_before),
                "after":  ctx_list(node.context_after),
                "children": [self._node_to_dict(ch[1], children_getter) for ch in children_getter(node)]
            }

        else:
            raise TypeError(f"Unknown node type {type(node)}")


    def _tree_to_json(self):
        def children_getter(n):
            for wi, ch in getattr(n, "children", []):
                yield (wi, ch)
        return self._node_to_dict(self.global_root_node, children_getter)

    def _build_html(self, d3_data_json, node_w=280, node_h=130, h_gap=80, v_gap=150):
        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>Parse Tree</title>
    <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    #tree-container {{ display: inline-block; }}
    .link {{ fill: none; stroke: #9aa1a9; stroke-width: 1.5px; }}
    .node-box {{ stroke: #444; fill: #fff; rx: 8; ry: 8; filter: drop-shadow(1px 2px 2px rgba(0,0,0,0.15)); }}
    .node-fo table {{ border-collapse: collapse; font-size: 12px; margin: 4px 0; }}
    .node-fo th, .node-fo td {{ border: 1px solid #888; padding: 2px 6px; }}
    .node-fo th {{ background: #f3f5f7; font-weight: 600; }}
    .section-title {{ font-weight: bold; margin-top: 4px; }}
    .section {{ margin-top: 10px; margin-bottom: 10px; }}
    .subtable b {{ display: inline-block; margin: 6px 0 2px; }}
    .subtable table {{ border-collapse: collapse; }}
    .subtable td {{ border: 1px solid #bbb; padding: 1px 4px; }}
    </style>
    </head>
    <body>
    <div id="tree-container">
    <div id="tree"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script>
    const data = {d3_data_json};
    const nodeW  = {node_w};
    const nodeH  = {node_h};
    const hGap   = {h_gap};
    const vGap   = {v_gap};

    const root = d3.hierarchy(data);
    const layout = d3.tree().nodeSize([nodeW + hGap, nodeH + vGap]);
    layout(root);

    // compute bounds
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    root.each(d => {{
        if (d.x < x0) x0 = d.x;
        if (d.x > x1) x1 = d.x;
        if (d.y < y0) y0 = d.y;
        if (d.y > y1) y1 = d.y;
    }});
    const width  = x1 - x0 + nodeW + 320;
    const height = y1 - y0 + nodeH + 320; // THIS IS THE HEIGHT MODIFIER

    const svg = d3.select("#tree").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [x0 - nodeW/2, y0 - nodeH/2, width, height].join(" "));

    const g = svg.append("g");

    // links
    g.selectAll("path.link")
    .data(root.links())
    .join("path")
    .attr("class", "link")
    .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

    // nodes
    const node = g.selectAll("g.node")
    .data(root.descendants())
    .join("g")
    .attr("transform", d => `translate(${{d.x}},${{d.y}})`);

    // node rect
    node.append("rect")
    .attr("class", "node-box")
    .attr("x", -nodeW/2)
    .attr("y", 0)
    .attr("width", nodeW)
    .attr("height", nodeH);

    // node HTML via foreignObject
    node.append("foreignObject")
    .attr("class", "node-fo")
    .attr("x", -nodeW/2 + 6)
    .attr("y", 6)
    .attr("width", nodeW - 12)
    .attr("height", 1000)
    .html(d => nodeHTML(d.data));

    // shrink foreignObjects to actual content height
    node.selectAll("foreignObject").each(function() {{
    const fo = d3.select(this);
    const div = fo.select("div").node();
    const h = div.getBoundingClientRect().height + 6;
    fo.attr("height", h);
    d3.select(this.parentNode).select("rect").attr("height", h + 12);
    }});

    function nodeHTML(d) {{
        const ctxTable = (ctx, title) => {{
            if (!ctx || ctx.length === 0) return `<div class="subtable"><i>${{title}}: empty</i></div>`;
            const rows = ctx.map(kv => `<tr><td>${{kv.key}}</td><td>${{kv.val.toFixed(2)}}</td></tr>`).join("");
            return `<div class="subtable"><b>${{title}}</b><table><tbody>${{rows}}</tbody></table></div>`;
        }};

        let contentRows = "";
        if (d.right && d.right !== "None") {{
            // Composite node with left and right
            contentRows = `
                <tr><td>Content-Left</td><td>${{d.left}}</td></tr>
                <tr><td>Content-Right</td><td>${{d.right}}</td></tr>`;
        }} else {{
            // Primitive node with single content
            contentRows = `<tr><td>Content</td><td>${{d.left}}</td></tr>`;
        }}

        return `
        <div class="node-fo">
            <table><tr><th colspan="2">${{d.title}}</th></tr></table>
            <table>
            ${{contentRows}}
            </table>
            ${{ctxTable(d.before, "Context-Before")}}
            ${{ctxTable(d.after,  "Context-After")}}
        </div>`;
    }}

    </script>
    </body>
    </html>
    """


    def to_json(self, filepath=None):
        """
        Serialize the ParseTree into JSON. Optionally save to `filepath`.
        """

        def serialize_node(node, index_map):
            if isinstance(node, PrimitiveParseNode):
                return {
                    "node_type": "primitive",
                    "title": node.title,
                    "word_index":node.word_index,
                    "content": node.content,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "concept_label": node.concept_label,
                    "global_root": node.global_root,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            elif isinstance(node, CompositeParseNode):
                return {
                    "node_type": "composite",
                    "title": node.title,
                    "word_index":node.word_index,
                    "content_left": node.content_left,
                    "content_right": node.content_right,
                    "context_before": node.context_before,
                    "context_after": node.context_after,
                    "concept_label": node.concept_label,
                    "global_root": node.global_root,
                    "parent": index_map.get(node.parent),
                    "children": [index_map[ch[1]] for ch in node.children],
                }
            else:
                raise TypeError(f"Unknown node type {type(node)}")

        index_map = {node: i for i, node in enumerate([self.global_root_node] + self.nodes)}
        data = {
            "window": self.window,
            "context_length": self.context_length,
            "id_to_value": self.id_to_value,
            "value_to_id": self.value_to_id,
            "nodes": [serialize_node(node, index_map) for node in [self.global_root_node] + self.nodes],
        }

        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return filepath
        else:
            return json.dumps(data, indent=2)

    @staticmethod
    def from_json(data, ltm_hierarchy, filepath=False):
        """
        Deserialize a ParseTree from JSON. Requires the same ltm_hierarchy instance.
        """
        if filepath:
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(data, str):
            data = json.loads(data)

        tree = FiniteParseTree(
            ltm_hierarchy,
            id_to_value=data["id_to_value"],
            value_to_id=data["value_to_id"],
            context_length=data["context_length"],
        )
        tree.window = data["window"]

        def restore_dict_keys(d):
            if d is None:
                return None
            return {int(k): v for k, v in d.items()}

        node_objs = []
        for n in data["nodes"]:
            if n["node_type"] == "primitive":
                node = PrimitiveParseNode(
                    content=n["content"],
                    context_before=restore_dict_keys(n["context_before"]),
                    context_after=restore_dict_keys(n["context_after"]),
                    word_index=n["word_index"],
                )
            elif n["node_type"] == "composite":
                node = CompositeParseNode()
                node.content_left = restore_dict_keys(n["content_left"])
                node.content_right = restore_dict_keys(n["content_right"])
                node.context_before = restore_dict_keys(n["context_before"])
                node.context_after = restore_dict_keys(n["context_after"])
                node.word_index = n["word_index"]

            else:
                raise ValueError(f"Unknown node_type {n['node_type']}")

            node.title = n["title"]
            node.concept_label = n["concept_label"]
            node.global_root = n["global_root"]
            node_objs.append(node)

        # restore parent/child relations
        for idx, n in enumerate(data["nodes"]):
            node = node_objs[idx]
            parent_idx = n["parent"]
            if parent_idx is not None:
                node.parent = node_objs[parent_idx]
            node.children = [(node_objs[ch].word_index, node_objs[ch]) for ch in n["children"]]

        tree.global_root_node = node_objs[0]
        tree.nodes = node_objs[1:]

        return tree

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class LanguageChunkingParser:

    """
    This is only for language right now - lots of changes will need to be made
    to generalize, such as properly representing context.

    Standardizations:
    *   id-0 => content-left
    *   id-1 => content-right
    *   id-2 => context-before
    *   id-3 => context-after
    """

    def __init__(self, value_corpus, context_length=3, merge_split=True):

        self.ltm_hierarchy = CobwebTree(alpha=1)

        self.id_to_value = ["EMPTYNULL"]
        for x in value_corpus:
            self.id_to_value.append(x)
        self.value_to_id = dict([(w, i) for i, w in enumerate(self.id_to_value)])
        self.id_count = len(value_corpus)

        # adding root node to dictionary! edge case not properly counted
        hsh = self.ltm_hierarchy.root.concept_hash()
        self.id_to_value.append(f"CONCEPT-{hsh}")
        self.id_count += 1
        self.value_to_id[f"CONCEPT-{hsh}"] = self.id_count

        self.cobweb_drawer = HTMLCobwebDrawer(
            ["Content-Left", "Content-Right", "Context-Before", "Context-After"],
            id_to_value=self.id_to_value,
            value_to_id=self.value_to_id
        )

        self.context_length = context_length
        self.merge_split = merge_split

    def get_long_term_memory(self):
        return self.ltm_hierarchy

    def parse_input(self, windows, end_behavior="converge", debug=False) -> List[FiniteParseTree]:
        """
        Primary method for parsing input (a list of sliding windows, for now these are
        sentences) and updating the long-term-memory hierarchy using a parse tree.

        Returns the Parse Tree!
        """

        parse_trees = []

        for window in windows:
            if debug:
                print(f"BUILDING PARSE TREE FOR WINDOW {window}")
            parse_tree = FiniteParseTree(self.ltm_hierarchy, self.id_to_value, self.value_to_id)
            parse_tree.build(window, end_behavior, debug)

            parse_trees.append(parse_tree)

            if debug:
                print("-" * 100)
                print()

        return parse_trees

    def add_parse_tree(self, parse_tree, debug=False):
        """
        Method to add the parse tree to the long-term hierarchy.

        Important note here is that because the parse tree doesn't contain the
        primitive instances in code (the leaves of the code tree are composite
        by nature), we can literally just add all nodes of the parse tree as
        defined by the code.

        Pipeline:
        *   Save tree structure before adding all new nodes (for each node,
            what is its parent and what are all its children)
        *   Add all instances of the parse tree as children via Cobweb's ifit
        *   Now, we iterate over the new tree and the old tree and keep track of
            all fundamentally applied actions.
        *   We initialize an empty list, "rewrite_rules", for updating all
            instance dictionaries after properly logging all actions.
        *   We process each Cobweb action in an effort to keep an updated as
            follows:
            *   Any "Add" operations don't change the tree's node structure and
                are not necessary to process.
            *   Any "Create" operation creates a new node (typically the last
                action) - these nodes must be added via the pattern
                "CONCEPT-{node.concept_hash()}" to the self.value_to_id and
                self.id_to_value dictionaries (also update self.id_count).
            *   Any "Split" operation deletes a node and promotes all children
                to become parents - I'm PRETTY SURE there's no vocabulary
                changes, but we will need to save a rewrite rule binding the old
                node vocabulary id to it's parent's id, (deleted_id, parent_id).
            *   Any "Merge" operation groups two children and creates a common
                parent node for these two nodes. In this case, we should add the
                new node to the vocabulary similar to the "Create" action.
        *   Finally, we iterate through the tree and apply all rewrite rules
            (which I'm pretty sure are only SPLIT rules, so this should be
            fairly easy).

        ---
        With this new iteration of the method, we have a couple administrative
        actions we need to complete, but this should greatly reduce the size of
        the code by a large amount (and the lasting complexity as well).

        After adding to the tree, we'll have a stack-trace of actions. We need
        to conduct the following two administrative actions:
        *   All new nodes need to be added to the vocabulary
        *   All deleted nodes need to be transferred to a rewrite-rules list 
            and a new method needs to handle all rewrites to the cobweb tree
            recursively.

        We will implement both of these in here so as not to touch the
        administrative Cobweb.

        For created nodes:
        *   If the node is created:
            *   Add it to the vocabulary!
        For deleted nodes:
        *   If the node is deleted:
            *   Iterate through all nodes and replace the node's id with its
                parent's id.

        For replacement, we program some cool BFS from the root that adds the
        probability of old_key to new_key and then removes old_key, and then
        traverses down all children, stopping the traversal if old_key doesn't
        exist.
        """

        if debug:
            print(f"Adding parse tree for window, \"{parse_tree.window}\"")

        # adding all new instances
        insts = parse_tree.get_chunk_instances()

        def print_actions(actions):
            if not debug:
                return
            print()
            print("PRINTING ALL ACTIONS TAKEN OVER THIS PASS:")
            for i, act in enumerate(actions, 1):
                a = act["action"]
                node = act.get("node")
                parent = act.get("parent")
                extras = act.get("extra_nodes_created", [])
                absorbed = act.get("absorbed", False)

                if a == "NEW":
                    if absorbed:
                        print(f"{i:02d}. [NEW/ABSORB] Node {node} absorbed instance "
                            f"(parent={parent or 'ROOT'})")
                    elif extras:
                        print(f"{i:02d}. [NEW] Node {node} (parent={parent}) "
                            f"(extra created: {', '.join(extras)})")
                    else:
                        print(f"{i:02d}. [NEW] Node {node} (parent={parent})")
                elif a == "BEST":
                    print(f"{i:02d}. [BEST] Descend into node {node} (parent={parent})")
                elif a == "MERGE":
                    kids = ", ".join(act.get("children", []))
                    print(f"{i:02d}. [MERGE] New node {act['new_node']} under {act['parent']} "
                        f"merged children [{kids}]")
                elif a == "SPLIT":
                    kids = ", ".join(act.get("promoted_children", []))
                    print(f"{i:02d}. [SPLIT] Deleted {act['deleted']} under {act['parent']}, "
                        f"promoted [{kids}]")
                elif a == "FALLBACK":
                    print(f"{i:02d}. [FALLBACK] Went to {node} (parent={parent})")
                else:
                    print(f"{i:02d}. [UNKNOWN] {act}")

            
            print("---")

        def add_node(node_list):
            for n in node_list:
                new_vocab = f"CONCEPT-{n}"
                self.add_to_vocab(new_vocab)

        all_actions = []
        mode = 0 if self.merge_split else 4

        for inst in insts:
            if debug:
                print("Adding instance to CobwebTree:", inst)
            _, _, actions = self.ltm_hierarchy.ifit(inst, mode=mode, debug=True) # debug flag for saving logs!

            # print(actions)

            # json loading!!
            actions = [json.loads(x) for x in actions]

            # print(actions)

            all_actions += actions
            print_actions(actions)

        # handle new, merge, and split actions
        rewrite_rules = []
        for action in all_actions:
            if action["action"] == "NEW":
                add_node([action["node"]])
            elif action["action"] == "MERGE":
                add_node([action["new_node"]])
            elif action["action"] == "SPLIT":
                rewrite_rules.append((action["deleted"], action["parent"]))

        # BFS/DFS throughout tree and edit av_counts where applicable
        if self.merge_split and rewrite_rules:
            def av_replacement(inst):
                replaced = False
                for k in inst.keys():
                    for concept_hash in list(inst[k].keys()):
                        for old, new in rewrite_rules:
                            if f"CONCEPT-{concept_hash}" == old:
                                inst[k].setdefault(f"CONCEPT-{new}", 0)
                                inst[k][f"CONCEPT-{new}"] += inst[k][old]
                                del inst[k][old]
                                replaced = True
                return inst, replaced

            to_visit = [self.ltm_hierarchy.root]
            while to_visit:
                curr = to_visit.pop(0)
                new_av_count, replaced = av_replacement(curr.av_count)
                curr.set_av_count(new_av_count)
                if replaced:
                    to_visit.extend(curr.children)

        if debug:
            print("-" * 60)

        return True
    
    def add_to_vocab(self, new_vocab):
        """
        Helper function to add new word to vocabulary!

        Should return True if the vocab was successfully added, False otherwise.
        """
        if new_vocab not in self.value_to_id:
            self.id_to_value.append(new_vocab)
            self.id_count += 1
            self.value_to_id[new_vocab] = self.id_count
            return True
        return False

    def visualize_ltm(self, out_base="cobweb_tree"):
        """
        We had a rudimentary CobwebDrawer before but I'd very much enjoy if we
        could expand on this and create an HTML-drawing Cobweb method before we
        continue tests - it would be both easier to explain and certainly easy
        to verify.
        """
        self.cobweb_drawer.draw_tree(self.ltm_hierarchy.root, out_base)

    def save_state(self, dirpath: str):
        """
        Save the LanguageChunkingParser state to a directory.

        Writes a `meta.json` for simple attributes (id_to_value, value_to_id,
        context_length, merge_split, id_count, window if present) and uses
        the CobwebTree.write_json_stream to write the LTM to `tree.json` in the
        same directory.
        """
        os.makedirs(dirpath, exist_ok=True)

        # meta data
        meta = {
            "context_length": self.context_length,
            "merge_split": self.merge_split,
            "id_count": getattr(self, "id_count", None),
            "id_to_value": self.id_to_value,
            "value_to_id": self.value_to_id,
        }

        meta_path = os.path.join(dirpath, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # LTM tree
        tree_path = os.path.join(dirpath, "tree.json")
        # CobwebTree provides a write_json_stream(filename) method
        try:
            self.ltm_hierarchy.write_json_stream(tree_path)
        except Exception:
            # fallback: if write_json_stream not present, try to use to_json
            try:
                with open(tree_path, "w", encoding="utf-8") as f:
                    json.dump(self.ltm_hierarchy.to_json(), f)
            except Exception as e:
                raise

        return {"ok": True, "meta": meta_path, "tree": tree_path}

    @staticmethod
    def load_state(dirpath: str) -> "LanguageChunkingParser":
        """
        Static loader that creates and returns a LanguageChunkingParser instance
        initialized from the given directory (created by `save_state`).

        Usage:
            parser = LanguageChunkingParser.load_state('/path/to/dir')

        Expects `meta.json` and `tree.json` under `dirpath`.
        """
        meta_path = os.path.join(dirpath, "meta.json")
        tree_path = os.path.join(dirpath, "tree.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {dirpath}")
        if not os.path.exists(tree_path):
            raise FileNotFoundError(f"tree.json not found in {dirpath}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Create a new parser with an empty value_corpus, then override
        parser = LanguageChunkingParser([], context_length=meta.get("context_length", 3), merge_split=meta.get("merge_split", True))

        # restore simple attrs (overwrite what constructor set)
        parser.context_length = meta.get("context_length", parser.context_length)
        parser.merge_split = meta.get("merge_split", parser.merge_split)
        parser.id_count = meta.get("id_count", getattr(parser, "id_count", None))
        parser.id_to_value = meta.get("id_to_value", parser.id_to_value)
        parser.value_to_id = meta.get("value_to_id", parser.value_to_id)

        # load LTM via CobwebTree.load_json_stream or read_json_stream
        try:
            if hasattr(parser.ltm_hierarchy, "load_json_stream"):
                parser.ltm_hierarchy.load_json_stream(tree_path)
            elif hasattr(parser.ltm_hierarchy, "read_json_stream"):
                parser.ltm_hierarchy.read_json_stream(tree_path)
            else:
                if hasattr(parser.ltm_hierarchy, "from_json"):
                    new_tree = parser.ltm_hierarchy.from_json(tree_path)
                    parser.ltm_hierarchy = new_tree
                else:
                    with open(tree_path, "r", encoding="utf-8") as tf:
                        tree_data = json.load(tf)
                    if hasattr(parser.ltm_hierarchy, "load_json"):
                        parser.ltm_hierarchy.load_json(tree_data)
                    else:
                        parser.ltm_hierarchy._loaded_json = tree_data
        except Exception:
            raise

        # recreate cobweb drawer to reflect loaded vocabulary
        parser.cobweb_drawer = HTMLCobwebDrawer(
            ["Content-Left", "Content-Right", "Context-Before", "Context-After"],
            id_to_value=parser.id_to_value,
            value_to_id=parser.value_to_id
        )

        return parser