from __future__ import annotations
import math
import json
import random
from typing import Dict, List, Tuple, Optional, Set, DefaultDict
from collections import defaultdict
import uuid

# Types
AVCount = Dict[int, Dict[int, float]]   # {attr: {val: count}}
AttrVals = Dict[int, Set[int]]


class CobwebTree:
    def __init__(self,
                 alpha: float = 0.01,
                 weight_attr: bool = False,
                 objective: int = 0,
                 children_norm: bool = False,
                 norm_attributes: bool = True):
        """
        alpha: Laplace smoothing parameter
        weight_attr: scale per-attribute information by root frequency
        objective: 0 / 1 / 2 (different normalizations)
        children_norm: if True divide PU by number of children
        norm_attributes: control calculation strategy (mirrors C++ flag)
        """
        self.alpha = float(alpha)
        self.weight_attr = bool(weight_attr)
        self.objective = int(objective)
        self.children_norm = bool(children_norm)
        self.norm_attributes = bool(norm_attributes)

        self.attr_vals: AttrVals = defaultdict(set)
        self.root = CobwebNode(tree=self, parent=None)

    # ---------------- Management ----------------
    def clear(self) -> None:
        """Reset the tree (clear root and seen attributes)."""
        self.root = CobwebNode(tree=self, parent=None)
        self.attr_vals = defaultdict(set)

    def _ensure_attr_vals(self, instance: AVCount) -> None:
        """Register attribute-value combinations seen in an instance."""
        for a, vm in instance.items():
            if a < 0:
                continue
            for v in vm.keys():
                self.attr_vals[a].add(v)

    def ifit(self, instance: AVCount, mode: int = 0) -> Tuple["CobwebNode", Dict[str, float]]:
        """
        Fit a single instance using COBWEB ops.
        mode:
          0 = full (choose best op among BEST/NEW/MERGE/SPLIT)
          1 = BEST only
          2 = random decisions among allowed ops
          3 = epsilon-greedy BEST (1% random exploration)
          4 = BEST and NEW only (explicitly ignore MERGE and SPLIT)
        Returns (node_reached, stats dict — currently unused placeholder).
        """
        self._ensure_attr_vals(instance)
        current = self.root
        stats: Dict[str, float] = {}

        while True:
            # Case: leaf and empty or exact match -> absorb and stop
            if not current.children and (current.count == 0 or current.is_exact_match(instance)):
                current.increment_counts(instance)
                break

            # Fringe split: leaf but not exact match
            if not current.children:
                if current.parent is None:
                    # Case: current is root → mutate in place, don’t replace object
                    # Create two children: one for existing contents, one for new instance
                    old_child = CobwebNode(tree=self, parent=current)
                    old_child.update_counts_from_node(current)

                    new_child = CobwebNode(tree=self, parent=current)
                    new_child.increment_counts(instance)

                    current.children = [old_child, new_child]
                    current.clear_counts()
                    current.update_counts_from_node(old_child)
                    current.update_counts_from_node(new_child)

                    current = new_child
                    break
                else:
                    # Case: non-root fringe split → same as before
                    new_parent = CobwebNode(tree=self, parent=current.parent)
                    new_parent.update_counts_from_node(current)

                    p = current.parent
                    p.children = [new_parent if c is current else c for c in p.children]

                    current.parent = new_parent
                    new_parent.children.append(current)

                    sibling = CobwebNode(tree=self, parent=new_parent)
                    sibling.increment_counts(instance)
                    new_parent.increment_counts(instance)
                    new_parent.children.append(sibling)
                    current = sibling
                    break


            # Internal node: find two best children
            best1_score, best1, best2 = current.two_best_children(instance)

            # choose action according to mode
            if mode == 1:
                action = CobwebNode.BEST
            elif mode == 2:
                choices = [CobwebNode.BEST, CobwebNode.NEW]
                if best2 is not None:
                    choices += [CobwebNode.MERGE, CobwebNode.SPLIT]
                action = random.choice(choices)
            elif mode == 3:
                action = CobwebNode.BEST
                if best1 is not None and best2 is not None and random.random() < 1e-2:
                    _, action = current.get_best_operation(instance, best1, best2, best1_score)
            elif mode == 4:
                # only allow BEST and NEW
                _, action = current.get_best_operation(instance, best1, best2, best1_score)
                if action not in (CobwebNode.BEST, CobwebNode.NEW):
                    action = CobwebNode.BEST
            else:
                _, action = current.get_best_operation(instance, best1, best2, best1_score)

            # Execute chosen action
            if action == CobwebNode.BEST and best1 is not None:
                current.increment_counts(instance)
                current = best1
                continue

            if action == CobwebNode.NEW:
                current.increment_counts(instance)
                new_child = CobwebNode(tree=self, parent=current)
                new_child.increment_counts(instance)
                current.children.append(new_child)
                current = new_child
                break

            if action == CobwebNode.MERGE and best1 is not None and best2 is not None:
                current.increment_counts(instance)
                merged = CobwebNode(tree=self, parent=current)
                merged.update_counts_from_node(best1)
                merged.update_counts_from_node(best2)

                best1.parent = merged
                best2.parent = merged
                merged.children.extend([best1, best2])

                current.children = [c for c in current.children if c not in (best1, best2)]
                current.children.append(merged)
                current = merged
                continue

            if action == CobwebNode.SPLIT and best1 is not None:
                current.children = [c for c in current.children if c is not best1]
                for ch in best1.children:
                    ch.parent = current
                    ch.tree = self
                    current.children.append(ch)
                continue

            # fallback: descend to best (if present), otherwise just stop
            current.increment_counts(instance)
            if best1 is None:
                break
            current = best1

        return current, stats


    def fit(self, instances: List[AVCount], mode: int, iterations: int = 1, randomizeFirst: bool = True) -> None:
        """
        Batch training:
          - iterations: number of passes over data
          - randomizeFirst: shuffle instances on first iteration if True
        """
        if iterations < 1:
            iterations = 1
        for it in range(iterations):
            if it == 0 and randomizeFirst:
                random.shuffle(instances)
            for inst in instances:
                self.ifit(inst, mode)
            # shuffle between iterations to reduce ordering bias
            random.shuffle(instances)

    # ---------------- Categorization & I/O ----------------
    def categorize(self, instance: AVCount) -> "CobwebNode":
        """Greedy categorize by descending to child with highest posterior."""
        self._ensure_attr_vals(instance)
        node = self.root
        while node.children:
            node = max(node.children, key=lambda c: c.log_prob_class_given_instance(instance, use_root_counts=False))
        return node

    def dump_json(self) -> str:
        """Serialize tree to JSON string (human readable)."""
        return json.dumps({
            "alpha": self.alpha,
            "weight_attr": self.weight_attr,
            "objective": self.objective,
            "children_norm": self.children_norm,
            "norm_attributes": self.norm_attributes,
            "root": self.root.node_to_dict()
        }, indent=2)

    def output_json(self) -> str:
        """Alias for dump_json() for API parity with C++."""
        return self.dump_json()


class CobwebNode:
    # operation codes
    BEST, NEW, MERGE, SPLIT = 0, 1, 2, 3

    def __init__(self, tree: CobwebTree, parent: Optional["CobwebNode"] = None, concept_hash: Optional[str] = None):
        self.tree = tree
        self.parent = parent
        self.children: List["CobwebNode"] = []

        self.count: float = 0.0
        self.a_count: DefaultDict[int, float] = defaultdict(float)
        # av_count[attr][val] -> count
        self.av_count: DefaultDict[int, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        # cache for entropy calculations: for attr -> sum_{val} (n_val + alpha) * log(n_val + alpha)
        self.sum_n_logn: DefaultDict[int, float] = defaultdict(float)

        if concept_hash:
            self._concept_hash = concept_hash
        else:
            self._concept_hash = uuid.uuid4().hex[:10]

    # ---------------- Count updates ----------------
    def increment_counts(self, instance: AVCount) -> None:
        """Add instance counts into this node and update cached sum_n_logn."""
        alpha = self.tree.alpha
        self.count += 1.0
        for attr, vm in instance.items():
            for val, cnt in vm.items():
                # subtract old tf*log(tf) if present
                if attr > 0 and val in self.av_count[attr]:
                    old_tf = self.av_count[attr][val] + alpha
                    self.sum_n_logn[attr] -= old_tf * math.log(old_tf)
                self.a_count[attr] += cnt
                self.av_count[attr][val] += cnt
                if attr > 0:
                    new_tf = self.av_count[attr][val] + alpha
                    self.sum_n_logn[attr] += new_tf * math.log(new_tf)

    def update_counts_from_node(self, node: "CobwebNode") -> None:
        """Accumulate another node's counts into this node (used for merges/fringe copies)."""
        alpha = self.tree.alpha
        self.count += node.count
        for attr, vm in node.av_count.items():
            self.a_count[attr] += node.a_count.get(attr, 0.0)
            for val, cnt in vm.items():
                if attr > 0 and val in self.av_count[attr]:
                    old_tf = self.av_count[attr][val] + alpha
                    self.sum_n_logn[attr] -= old_tf * math.log(old_tf)
                self.av_count[attr][val] += cnt
                if attr > 0:
                    new_tf = self.av_count[attr][val] + alpha
                    self.sum_n_logn[attr] += new_tf * math.log(new_tf)

    # ---------------- Matching / traversal helpers ----------------
    def is_exact_match(self, instance: AVCount) -> bool:
        """Return True if this node contains the exact same set of attributes and values."""
        attrs_here = {a for a in self.av_count.keys() if a >= 0}
        attrs_inst = {a for a in instance.keys() if a >= 0}
        if attrs_here != attrs_inst:
            return False
        for a in attrs_here:
            if set(self.av_count[a].keys()) != set(instance[a].keys()):
                return False
        return True

    def depth(self) -> int:
        d = 0
        cur = self
        while cur.parent is not None:
            d += 1
            cur = cur.parent
        return d

    def num_concepts(self) -> int:
        total = 1
        for c in self.children:
            total += c.num_concepts()
        return total

    def is_parent(self, other: "CobwebNode") -> bool:
        cur = other
        while cur is not None:
            if cur.parent is self:
                return True
            cur = cur.parent
        return False

    # ---------------- Hash / representation ----------------
    def _hash(self) -> int:
        """Deterministic hash based on av_count contents."""
        parts = []
        for attr in sorted(self.av_count.keys()):
            vals = sorted(self.av_count[attr].items())
            parts.append(f"{attr}:" + ",".join(f"{v}:{int(c) if float(c).is_integer() else c}" for v, c in vals))
        s = "|".join(parts)
        return hash(s)

    def concept_hash(self) -> str:
        """
        Unique 10-char hexadecimal ID for this node,
        fixed at instantiation and independent of AV counts.
        """
        return self._concept_hash

    def __str__(self) -> str:
        return f"<CobwebNode depth={self.depth()} count={self.count} children={len(self.children)}>"

    def pretty_print(self, depth: int = 0) -> str:
        pad = "  " * depth
        lines = [f"{pad}{self.__str__()} -> {self.concept_hash()}"]
        for c in self.children:
            lines.append(c.pretty_print(depth + 1))
        return "\n".join(lines)

    def node_to_dict(self) -> Dict:
        """Return a JSON-serializable dict for this node and its subtree."""
        return {
            "count": self.count,
            "a_count": {str(k): v for k, v in self.a_count.items()},
            "sum_n_logn": {str(k): v for k, v in self.sum_n_logn.items()},
            "av_count": {str(k): {str(val): cnt for val, cnt in vm.items()} for k, vm in self.av_count.items()},
            "children": [c.node_to_dict() for c in self.children]
        }

    def avcounts_to_json(self) -> str:
        return json.dumps({str(k): {str(v): c for v, c in vm.items()} for k, vm in self.av_count.items()}, indent=2)

    def ser_avcounts(self) -> str:
        return self.avcounts_to_json()

    def a_count_to_json(self) -> str:
        return json.dumps({str(k): v for k, v in self.a_count.items()}, indent=2)

    def sum_n_logn_to_json(self) -> str:
        return json.dumps({str(k): v for k, v in self.sum_n_logn.items()}, indent=2)

    def dump_json(self) -> str:
        return json.dumps(self.node_to_dict(), indent=2)

    def output_json(self) -> str:
        return self.dump_json()

    # ---------------- Probability helpers ----------------
    def probability(self, attr: int, val: int) -> float:
        """P(val | attr, this node) with Laplace smoothing."""
        if attr not in self.tree.attr_vals or len(self.tree.attr_vals[attr]) == 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals = len(self.tree.attr_vals[attr])
        numerator = self.av_count[attr].get(val, 0.0) + alpha
        denom = self.a_count.get(attr, 0.0) + num_vals * alpha
        return numerator / (denom + 1e-15)

    def predict_probs(self) -> Dict[int, Dict[int, float]]:
        """Return P(val|attr) for all known attributes and values using Laplace smoothing."""
        out: Dict[int, Dict[int, float]] = {}
        for attr, valset in self.tree.attr_vals.items():
            out[attr] = {}
            num_vals = len(valset)
            denom = self.a_count.get(attr, 0.0) + num_vals * self.tree.alpha
            if denom <= 0:
                # fallback to uniform
                uniform = 1.0 / max(1, num_vals)
                for v in valset:
                    out[attr][v] = uniform
                continue
            for v in valset:
                out[attr][v] = (self.av_count[attr].get(v, 0.0) + self.tree.alpha) / (denom + 1e-15)
        return out

    def predict_log_probs(self) -> Dict[int, Dict[int, float]]:
        return {a: {v: math.log(p) if p > 0 else -float("inf") for v, p in vm.items()} for a, vm in self.predict_probs().items()}

    def get_weighted_values(self, attr: int, allowNone: bool = True) -> List[Tuple[int, float]]:
        if attr not in self.tree.attr_vals:
            return []
        weights = [(v, self.av_count[attr].get(v, 0.0) + self.tree.alpha) for v in sorted(self.tree.attr_vals[attr])]
        if allowNone and not weights:
            return []
        return weights

    # ---------------- Log-prob functions ----------------
    def log_prob_instance(self, instance: AVCount) -> float:
        """log P(instance | node) using observed attributes only."""
        logp = 0.0
        alpha = self.tree.alpha
        for attr, vm in instance.items():
            if attr < 0 or attr not in self.tree.attr_vals:
                continue
            num_vals = len(self.tree.attr_vals[attr])
            denom = self.a_count.get(attr, 0.0) + num_vals * alpha
            for v, cnt in vm.items():
                if v not in self.tree.attr_vals[attr]:
                    numer = alpha
                else:
                    numer = self.av_count[attr].get(v, 0.0) + alpha
                logp += cnt * (math.log(numer + 1e-15) - math.log(denom + 1e-15))
        return logp

    def log_prob_instance_missing(self, instance: AVCount) -> float:
        """
        log P(instance | node) when considering missing attributes.
        This implementation uses observed attributes and approximates missing ones by ignoring them,
        which matches the conservative approach used in the port.
        """
        logp = 0.0
        alpha = self.tree.alpha
        for attr, valset in self.tree.attr_vals.items():
            if attr < 0:
                continue
            if attr in instance:
                num_vals = len(valset)
                denom = self.a_count.get(attr, 0.0) + num_vals * alpha
                for v, cnt in instance[attr].items():
                    numer = self.av_count[attr].get(v, 0.0) + alpha
                    logp += cnt * (math.log(numer + 1e-15) - math.log(denom + 1e-15))
            else:
                # missing attribute: we do not add anything (conservative)
                pass
        return logp

    def log_prob_class_given_instance(self, instance: AVCount, use_root_counts: bool = False) -> float:
        """
        log P(node | instance) ∝ log P(instance | node) + log prior(node).
        If use_root_counts True prior denominator uses root.count; otherwise uses parent.count (if present).
        """
        ll = self.log_prob_instance(instance)
        prior_num = max(self.count, 0.0)
        if self.parent is None:
            prior_den = max(self.tree.root.count, 1e-15)
        else:
            prior_den = max(self.parent.count if not use_root_counts else self.tree.root.count, 1e-15)
        prior = math.log((prior_num + 1e-15) / (prior_den + 1e-15))
        return ll + prior
    
    def log_prob_children_given_instance(self, instance: AVCount) -> float:
        """
        log P(instance | children of this node).
        That is, mixture over children: sum_c P(c|parent) * P(instance|c).
        """
        if not self.children:
            return -float("inf")
        total = 0.0
        for c in self.children:
            prior = (c.count + 1e-15) / (self.count + 1e-15)
            likelihood = math.exp(c.log_prob_instance(instance))
            total += prior * likelihood
        return math.log(total + 1e-15)

    # ---------------- Entropy & PU math ----------------
    def entropy_attr(self, attr: int) -> float:
        if attr < 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals_total = len(self.tree.attr_vals[attr])
        ratio = 1.0
        if self.tree.weight_attr and self.tree.root.count > 0:
            ratio = (self.tree.root.a_count.get(attr, 0.0) / self.tree.root.count)
        attr_count = self.a_count.get(attr, 0.0)
        num_vals_in_c = len(self.av_count.get(attr, {}))
        sum_n_logn = self.sum_n_logn.get(attr, 0.0)
        n0 = num_vals_total - num_vals_in_c
        denom = attr_count + num_vals_total * alpha
        info = -ratio * ((sum_n_logn + n0 * alpha * math.log(alpha + 1e-15)) / (denom + 1e-15) - math.log(denom + 1e-15))
        return info

    def entropy(self) -> float:
        total = 0.0
        for attr in self.tree.attr_vals.keys():
            total += self.entropy_attr(attr)
        return total

    def entropy_attr_insert(self, attr: int, instance: AVCount) -> float:
        if attr < 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals_total = len(self.tree.attr_vals[attr])
        ratio = 1.0
        if self.tree.weight_attr and self.tree.root.count > 0:
            ratio = (self.tree.root.a_count.get(attr, 0.0) / self.tree.root.count)
        attr_count = self.a_count.get(attr, 0.0)
        num_vals_in_c = len(self.av_count.get(attr, {}))
        sum_n_logn = self.sum_n_logn.get(attr, 0.0)

        if attr in instance:
            for v, cnt in instance[attr].items():
                prior = self.av_count[attr].get(v, 0.0)
                if prior > 0:
                    tf_old = prior + alpha
                    sum_n_logn -= tf_old * math.log(tf_old)
                else:
                    num_vals_in_c += 1
                tf_new = prior + cnt + alpha
                sum_n_logn += tf_new * math.log(tf_new)
                attr_count += cnt

        n0 = num_vals_total - num_vals_in_c
        denom = attr_count + num_vals_total * alpha
        info = -ratio * ((sum_n_logn + n0 * alpha * math.log(alpha + 1e-15)) / (denom + 1e-15) - math.log(denom + 1e-15))
        return info

    def entropy_insert(self, instance: AVCount) -> float:
        info = 0.0
        seen = set()
        for attr in self.av_count.keys():
            info += self.entropy_attr_insert(attr, instance)
            seen.add(attr)
        for attr in instance.keys():
            if attr not in seen:
                info += self.entropy_attr_insert(attr, instance)
        return info

    def entropy_attr_merge(self, attr: int, other: "CobwebNode", instance: AVCount) -> float:
        if attr < 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals_total = len(self.tree.attr_vals[attr])
        ratio = 1.0
        if self.tree.weight_attr and self.tree.root.count > 0:
            ratio = (self.tree.root.a_count.get(attr, 0.0) / self.tree.root.count)

        attr_count = self.a_count.get(attr, 0.0)
        num_vals_in_c = len(self.av_count.get(attr, {}))
        sum_n_logn = self.sum_n_logn.get(attr, 0.0)

        if attr in other.av_count:
            for v, other_cnt in other.av_count[attr].items():
                inst_cnt = instance.get(attr, {}).get(v, 0.0)
                prior = self.av_count[attr].get(v, 0.0)
                if prior > 0:
                    tf_old = prior + alpha
                    sum_n_logn -= tf_old * math.log(tf_old)
                else:
                    num_vals_in_c += 1
                tf_new = prior + other_cnt + inst_cnt + alpha
                sum_n_logn += tf_new * math.log(tf_new)
                attr_count += other_cnt + inst_cnt

        if attr in instance:
            for v, inst_cnt in instance[attr].items():
                if v in other.av_count.get(attr, {}):
                    continue
                prior = self.av_count[attr].get(v, 0.0)
                if prior > 0:
                    tf_old = prior + alpha
                    sum_n_logn -= tf_old * math.log(tf_old)
                else:
                    num_vals_in_c += 1
                tf_new = prior + inst_cnt + alpha
                sum_n_logn += tf_new * math.log(tf_new)
                attr_count += inst_cnt

        n0 = num_vals_total - num_vals_in_c
        denom = attr_count + num_vals_total * alpha
        info = -ratio * ((sum_n_logn + n0 * alpha * math.log(alpha + 1e-15)) / (denom + 1e-15) - math.log(denom + 1e-15))
        return info

    def entropy_merge(self, other: "CobwebNode", instance: AVCount) -> float:
        info = 0.0
        seen = set()
        for attr in self.av_count.keys():
            info += self.entropy_attr_merge(attr, other, instance)
            seen.add(attr)
        for attr in other.av_count.keys():
            if attr not in seen:
                info += self.entropy_attr_merge(attr, other, instance)
                seen.add(attr)
        for attr in instance.keys():
            if attr not in seen:
                info += self.entropy_attr_merge(attr, other, instance)
        return info

    # ---------------- Partition Utility ----------------
    def _normalize_obj(self, parent_entropy: float, children_entropy: float, concept_entropy: float, child_count_norm: Optional[int]) -> float:
        obj = (parent_entropy - children_entropy)
        if self.tree.objective == 1:
            obj = obj / (parent_entropy + 1e-15)
        elif self.tree.objective == 2:
            obj = obj / (children_entropy + concept_entropy + 1e-15)
        if self.tree.children_norm and child_count_norm and child_count_norm > 0:
            obj = obj / child_count_norm
        return obj

    def pu_for_insert(self, child: "CobwebNode", instance: AVCount) -> float:
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = 0.0
            for attr in self.tree.attr_vals.keys():
                parent_entropy += self.entropy_attr_insert(attr, instance)
            for c in self.children:
                if c is child:
                    p = (c.count + 1.0) / (self.count + 1.0)
                    concept_entropy -= p * math.log(p + 1e-15)
                    for attr in self.tree.attr_vals.keys():
                        children_entropy += p * c.entropy_attr_insert(attr, instance)
                else:
                    p = (c.count) / (self.count + 1.0)
                    concept_entropy -= p * math.log(p + 1e-15)
                    for attr in self.tree.attr_vals.keys():
                        children_entropy += p * c.entropy_attr(attr)
            return self._normalize_obj(parent_entropy, children_entropy, concept_entropy, len(self.children))

        total = 0.0
        for attr in self.tree.attr_vals.keys():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                if c is child:
                    p = (c.count + 1.0) / (self.count + 1.0)
                    children_entropy += p * c.entropy_attr_insert(attr, instance)
                    concept_entropy -= p * math.log(p + 1e-15)
                else:
                    p = (c.count) / (self.count + 1.0)
                    children_entropy += p * c.entropy_attr(attr)
                    concept_entropy -= p * math.log(p + 1e-15)
            parent_entropy = self.entropy_attr_insert(attr, instance)
            total += self._normalize_obj(parent_entropy, children_entropy, concept_entropy, len(self.children))
        return total

    def pu_for_new_child(self, instance: AVCount) -> float:
        tmp = CobwebNode(tree=self.tree, parent=self)
        tmp.increment_counts(instance)
        p_new = 1.0 / (self.count + 1.0)

        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = -p_new * math.log(p_new + 1e-15)
            for attr in self.tree.attr_vals.keys():
                parent_entropy += self.entropy_attr_insert(attr, instance)
                children_entropy += p_new * tmp.entropy_attr(attr)
            for c in self.children:
                p = c.count / (self.count + 1.0)
                concept_entropy -= p * math.log(p + 1e-15)
                for attr in self.tree.attr_vals.keys():
                    children_entropy += p * c.entropy_attr(attr)
            return self._normalize_obj(parent_entropy, children_entropy, concept_entropy, len(self.children) + 1)

        total = 0.0
        for attr in self.tree.attr_vals.keys():
            children_entropy = p_new * tmp.entropy_attr(attr)
            concept_entropy = -p_new * math.log(p_new + 1e-15)
            for c in self.children:
                p = c.count / (self.count + 1.0)
                children_entropy += p * c.entropy_attr(attr)
                concept_entropy -= p * math.log(p + 1e-15)
            parent_entropy = self.entropy_attr_insert(attr, instance)
            total += self._normalize_obj(parent_entropy, children_entropy, concept_entropy, len(self.children) + 1)
        return total

    def pu_for_merge(self, best1: "CobwebNode", best2: "CobwebNode", instance: AVCount) -> float:
        if not self.tree.norm_attributes:
            p_merged = (best1.count + best2.count + 1.0) / (self.count + 1.0)
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = -p_merged * math.log(p_merged + 1e-15)
            for attr in self.tree.attr_vals.keys():
                parent_entropy += self.entropy_attr_insert(attr, instance)
                children_entropy += p_merged * best1.entropy_attr_merge(attr, best2, instance)
            for c in self.children:
                if c in (best1, best2):
                    continue
                p = c.count / (self.count + 1.0)
                concept_entropy -= p * math.log(p + 1e-15)
                for attr in self.tree.attr_vals.keys():
                    children_entropy += p * c.entropy_attr(attr)
            return self._normalize_obj(parent_entropy, children_entropy, concept_entropy, max(1, len(self.children) - 1))

        total = 0.0
        for attr in self.tree.attr_vals.keys():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                if c in (best1, best2):
                    continue
                p = c.count / (self.count + 1.0)
                children_entropy += p * c.entropy_attr(attr)
                concept_entropy -= p * math.log(p + 1e-15)
            p = (best1.count + best2.count + 1.0) / (self.count + 1.0)
            children_entropy += p * best1.entropy_attr_merge(attr, best2, instance)
            concept_entropy -= p * math.log(p + 1e-15)
            parent_entropy = self.entropy_attr_insert(attr, instance)
            total += self._normalize_obj(parent_entropy, children_entropy, concept_entropy, max(1, len(self.children) - 1))
        return total

    def pu_for_split(self, best: "CobwebNode") -> float:
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = 0.0
            for attr in self.tree.attr_vals.keys():
                parent_entropy += self.entropy_attr(attr)
            for c in self.children:
                if c is best:
                    continue
                p = c.count / max(self.count, 1e-15)
                concept_entropy -= p * math.log(p + 1e-15)
                for attr in self.tree.attr_vals.keys():
                    children_entropy += p * c.entropy_attr(attr)
            for c in best.children:
                p = c.count / max(self.count, 1e-15)
                concept_entropy -= p * math.log(p + 1e-15)
                for attr in self.tree.attr_vals.keys():
                    children_entropy += p * c.entropy_attr(attr)
            return self._normalize_obj(parent_entropy, children_entropy, concept_entropy, max(1, len(self.children) - 1 + len(best.children)))

        total = 0.0
        for attr in self.tree.attr_vals.keys():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                if c is best:
                    continue
                p = c.count / max(self.count, 1e-15)
                children_entropy += p * c.entropy_attr(attr)
                concept_entropy -= p * math.log(p + 1e-15)
            for c in best.children:
                p = c.count / max(self.count, 1e-15)
                children_entropy += p * c.entropy_attr(attr)
                concept_entropy -= p * math.log(p + 1e-15)
            parent_entropy = self.entropy_attr(attr)
            total += self._normalize_obj(parent_entropy, children_entropy, concept_entropy, max(1, len(self.children) - 1 + len(best.children)))
        return total

    def partition_utility(self) -> float:
        if not self.children:
            return 0.0
        entropy = 0.0
        for attr in self.tree.attr_vals.keys():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                p = c.count / max(self.count, 1e-15)
                children_entropy += p * c.entropy_attr(attr)
                concept_entropy -= p * math.log(p + 1e-15)
            parent_entropy = self.entropy_attr(attr)
            entropy += self._normalize_obj(parent_entropy, children_entropy, concept_entropy, max(1, len(self.children)))
        return entropy

    def category_utility(self) -> float:
        return self.partition_utility()

    # ---------------- Decision helpers ----------------
    def two_best_children(self, instance: AVCount) -> Tuple[float, Optional["CobwebNode"], Optional["CobwebNode"]]:
        best1 = None
        best2 = None
        best1_s = -float("inf")
        best2_s = -float("inf")
        for c in self.children:
            s = c.log_prob_class_given_instance(instance, use_root_counts=False)
            if s > best1_s:
                best2, best2_s = best1, best1_s
                best1, best1_s = c, s
            elif s > best2_s:
                best2, best2_s = c, s
        return best1_s, best1, best2

    def get_best_operation(self, instance: AVCount, best1: Optional["CobwebNode"], best2: Optional["CobwebNode"], _best1_pu: float) -> Tuple[float, int]:
        candidates: List[Tuple[float, int]] = []
        if best1 is not None:
            candidates.append((self.pu_for_insert(best1, instance), CobwebNode.BEST))
        candidates.append((self.pu_for_new_child(instance), CobwebNode.NEW))
        if best1 is not None and best2 is not None:
            candidates.append((self.pu_for_merge(best1, best2, instance), CobwebNode.MERGE))
        if best1 is not None and best1.children:
            candidates.append((self.pu_for_split(best1), CobwebNode.SPLIT))
        if not candidates:
            return float("-inf"), CobwebNode.BEST
        best_score, action = max(candidates, key=lambda t: t[0])
        return best_score, action

    # ---------------- Predict mixture helpers ----------------
    def predict_weighted_probs(self, instance: AVCount) -> Dict[int, Dict[int, float]]:
        """
        Weighted mixture of this node and immediate children based on log-likelihood + log prior.
        Produces probabilities for each attribute-value.
        """
        comps = []
        comps.append((self, self.log_prob_instance(instance) + math.log(self.count + 1e-15)))
        for c in self.children:
            comps.append((c, c.log_prob_instance(instance) + math.log(c.count + 1e-15)))
        maxs = max(s for _, s in comps)
        exps = [math.exp(s - maxs) for _, s in comps]
        total = sum(exps)
        weights = [e / (total + 1e-15) for e in exps]

        out: Dict[int, Dict[int, float]] = {}
        for attr in self.tree.attr_vals.keys():
            out[attr] = {v: 0.0 for v in self.tree.attr_vals[attr]}
            for (comp_node, _), w in zip(comps, weights):
                probs = comp_node.predict_probs()
                for v in self.tree.attr_vals[attr]:
                    out[attr][v] += w * probs.get(attr, {}).get(v, 0.0)
        return out

    def predict_weighted_leaves_probs(self, instance: AVCount) -> Dict[int, Dict[int, float]]:
        return self.predict_weighted_probs(instance)

    def predict(self, attr: int, choiceFn: str = "most likely", allowNone: bool = True) -> Optional[int]:
        probs = self.predict_probs().get(attr, {})
        if not probs:
            return None
        if choiceFn == "most likely":
            return max(probs.items(), key=lambda t: t[1])[0]
        elif choiceFn == "weighted":
            items = list(probs.items())
            vals, weights = zip(*items)
            total = sum(weights)
            if total <= 0:
                return random.choice(vals)
            r = random.random() * total
            acc = 0.0
            for v, w in items:
                acc += w
                if r <= acc:
                    return v
            return vals[-1]
        elif choiceFn == "random":
            return random.choice(list(probs.keys()))
        else:
            return max(probs.items(), key=lambda t: t[1])[0]

    # ---------------- Basic / Best level helpers ----------------
    def get_basic_level(self) -> "CobwebNode":
        """
        Find the 'basic level' by following children that improve category utility.
        """
        best = self
        best_cu = self.category_utility()

        current = self
        improved = True
        while improved and current.children:
            improved = False
            for c in current.children:
                cu = c.category_utility()
                if cu > best_cu:
                    best = c
                    best_cu = cu
                    current = c
                    improved = True
                    break
        return best

    def get_best_level(self, instance: AVCount) -> "CobwebNode":
        """
        Walk down by posterior probability until a leaf (C++'s get_best_level uses log_prob_class_given_instance).
        """
        current = self
        best = self
        best_ll = self.log_prob_class_given_instance(instance, use_root_counts=True)

        # climb up to root and then return best along path? The C++ method checks parents,
        # but a common usage is to descend by posterior; implement both semantics if needed.
        # Here we'll follow C++ logic: climb to parent checking posterior (best among ancestors).
        curr = self
        while curr.parent is not None:
            curr = curr.parent
            curr_ll = curr.log_prob_class_given_instance(instance, use_root_counts=True)
            if curr_ll > best_ll:
                best_ll = curr_ll
                best = curr
        return best

    # ---------------- Other small helpers ----------------
    def get_weighted_values_list(self, attr: int, allowNone: bool = True) -> List[Tuple[int, float]]:
        return self.get_weighted_values(attr, allowNone)

    def clear_counts(self) -> None:
        """
        Reset all statistical counts on this node, but keep identity/hash
        and structural pointers intact.
        """
        self.count = 0.0
        self.a_count.clear()
        self.av_count.clear()
        self.sum_n_logn.clear()

