"""
Python port of my_cobweb_discrete.cpp (discrete Cobweb implementation).
This file implements CobwebTree and CobwebNode with equivalent behavior for
counting, entropy, categorize/fit, predict, and json dump/load.

Notes / assumptions:
- ATTR_TYPE and VALUE_TYPE are ints in C++ file; we use ints in Python too.
- Instances are represented as dict[int, dict[int, float]] mapping attr -> val -> count
- Some C++ features (thread pool, rapidjson streaming handler, nanobind bindings)
  are adapted to Python idioms.
- The goal is functional parity for core algorithms (increment_counts, entropy,
  pu_for_insert/new/merge/split, categorize, cobweb fit logic, predict probs).
- A few utility functions (logsumexp, eff_logsumexp) are implemented.

This is a conservative, well-typed implementation intended for direct import.
"""
from __future__ import annotations

import math
import random
import time
import json
import itertools
from typing import Dict, Tuple, List, Optional, Set, Any
from collections import defaultdict

ATTR_TYPE = int
VALUE_TYPE = int
COUNT_TYPE = float
INSTANCE_TYPE = Dict[int, Dict[int, COUNT_TYPE]]
VAL_COUNT_TYPE = Dict[int, COUNT_TYPE]
AV_COUNT_TYPE = Dict[int, VAL_COUNT_TYPE]
AV_KEY_TYPE = Dict[int, Set[int]]
ATTR_COUNT_TYPE = Dict[int, COUNT_TYPE]
OPERATION_TYPE = Tuple[float, int]

# Operation codes
BEST = 0
NEW = 1
MERGE = 2
SPLIT = 3
BEST_NEW = 4

# Simple global attribute name -> id map copied from C++ (same numeric keys)
ATTRIBUTE_MAP: Dict[str, int] = {
    "alpha": 1000000,
    "weight_attr": 10000001,
    "objective": 10000002,
    "children_norm": 10000003,
    "norm_attributes": 10000004,
    "root": 10000005,
    "count": 100000011,
    "a_count": 100000012,
    "sum_n_logn": 100000013,
    "av_count": 100000014,
    "children": 100000015,
    "concept_hash": 100000020,
}

GLOBAL_CONCEPT_COUNTER = itertools.count(start=1)

def generate_concept_hash() -> str:
    ns = int(time.time() * 1e9)
    cid = next(GLOBAL_CONCEPT_COUNTER)
    return f"{ns}_{cid}"


def custom_rand() -> float:
    return random.random()


def logsumexp(vals: List[float]) -> float:
    if not vals:
        return -math.inf
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    s = sum(math.exp(v - m) for v in vals)
    return m + math.log(s)


def eff_logsumexp(a: float, b: float) -> float:
    # numerically stable log-sum-exp for two values
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


class CobwebNode:
    def __init__(self, other: Optional["CobwebNode"] = None):
        # If other provided, clone structure
        self.tree: Optional[CobwebTree] = None
        self.parent: Optional["CobwebNode"] = None
        self.children: List["CobwebNode"] = []
        self.concept_id: str = generate_concept_hash()
        self.count: COUNT_TYPE = 0.0
        self.a_count: ATTR_COUNT_TYPE = defaultdict(float)
        self.sum_n_logn: ATTR_COUNT_TYPE = defaultdict(float)
        self.av_count: AV_COUNT_TYPE = defaultdict(lambda: defaultdict(float))

        if other is not None:
            # clone counts and children
            self.parent = other.parent
            self.tree = other.tree
            self.update_counts_from_node(other)
            for c in other.children:
                cc = CobwebNode(c)
                cc.parent = self
                self.children.append(cc)
            self.concept_id = generate_concept_hash()

    def increment_counts(self, instance: AV_COUNT_TYPE) -> None:
        self.count += 1.0
        for attr, val_map in instance.items():
            for val, cnt in val_map.items():
                self.a_count[attr] = self.a_count.get(attr, 0.0) + cnt
                if attr > 0:
                    if attr in self.av_count and val in self.av_count[attr]:
                        tf = self.av_count[attr][val] + self.tree.alpha
                        self.sum_n_logn[attr] -= tf * math.log(tf)
                self.av_count[attr][val] = self.av_count[attr].get(val, 0.0) + cnt
                if attr > 0:
                    tf = self.av_count[attr][val] + self.tree.alpha
                    self.sum_n_logn[attr] = self.sum_n_logn.get(attr, 0.0) + tf * math.log(tf)

    def update_counts_from_node(self, node: "CobwebNode") -> None:
        self.count += node.count
        for attr, val_map in node.av_count.items():
            self.a_count[attr] = self.a_count.get(attr, 0.0) + node.a_count.get(attr, 0.0)
            for val, cnt in val_map.items():
                if attr > 0:
                    if attr in self.av_count and val in self.av_count[attr]:
                        tf = self.av_count[attr][val] + self.tree.alpha
                        self.sum_n_logn[attr] -= tf * math.log(tf)
                self.av_count[attr][val] = self.av_count[attr].get(val, 0.0) + cnt
                if attr > 0:
                    tf = self.av_count[attr][val] + self.tree.alpha
                    self.sum_n_logn[attr] = self.sum_n_logn.get(attr, 0.0) + tf * math.log(tf)

    def entropy_attr_insert(self, attr: ATTR_TYPE, instance: AV_COUNT_TYPE) -> float:
        if attr < 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals_total = len(self.tree.attr_vals[attr])
        num_vals_in_c = 0
        attr_count = 0.0
        ratio = 1.0
        if self.tree.weight_attr:
            ratio = (1.0 * self.tree.root.a_count[attr]) / (self.tree.root.count)
        if attr in self.av_count:
            attr_count = self.a_count.get(attr, 0.0)
            num_vals_in_c = len(self.av_count[attr])
        sum_n_logn = self.sum_n_logn.get(attr, 0.0)
        if attr in instance:
            for val, cnt in instance[attr].items():
                attr_count += cnt
                prior_av_count = 0.0
                if attr in self.av_count and val in self.av_count[attr]:
                    prior_av_count = self.av_count[attr][val]
                    tf = prior_av_count + self.tree.alpha
                    sum_n_logn -= tf * math.log(tf)
                else:
                    num_vals_in_c += 1
                tf = prior_av_count + cnt + self.tree.alpha
                sum_n_logn += tf * math.log(tf)
        n0 = num_vals_total - num_vals_in_c
        info = -ratio * ((1.0 / (attr_count + num_vals_total * alpha)) * (sum_n_logn + n0 * alpha * math.log(alpha)) - math.log(attr_count + num_vals_total * alpha))
        return info

    def entropy_insert(self, instance: AV_COUNT_TYPE) -> float:
        info = 0.0
        for attr in self.av_count.keys():
            if attr < 0:
                continue
            info += self.entropy_attr_insert(attr, instance)
        for attr in instance.keys():
            if attr < 0:
                continue
            if attr in self.av_count:
                continue
            info += self.entropy_attr_insert(attr, instance)
        return info

    def entropy_attr_merge(self, attr: ATTR_TYPE, other: "CobwebNode", instance: AV_COUNT_TYPE) -> float:
        if attr < 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals_total = len(self.tree.attr_vals[attr])
        num_vals_in_c = 0
        attr_count = 0.0
        ratio = 1.0
        if self.tree.weight_attr:
            ratio = (1.0 * self.tree.root.a_count[attr]) / (self.tree.root.count)
        if attr in self.av_count:
            attr_count = self.a_count.get(attr, 0.0)
            num_vals_in_c = len(self.av_count[attr])
        sum_n_logn = self.sum_n_logn.get(attr, 0.0)
        if attr in other.av_count:
            for val, other_av_count in other.av_count[attr].items():
                instance_av_count = 0.0
                if attr in instance and val in instance[attr]:
                    instance_av_count = instance[attr][val]
                attr_count += other_av_count + instance_av_count
                prior_av_count = 0.0
                if attr in self.av_count and val in self.av_count[attr]:
                    prior_av_count = self.av_count[attr][val]
                    tf = prior_av_count + alpha
                    sum_n_logn -= tf * math.log(tf)
                else:
                    num_vals_in_c += 1
                new_tf = prior_av_count + other_av_count + instance_av_count + alpha
                sum_n_logn += new_tf * math.log(new_tf)
        if attr in instance:
            for val, instance_av_count in instance[attr].items():
                if attr in other.av_count and val in other.av_count[attr]:
                    continue
                other_av_count = 0.0
                attr_count += other_av_count + instance_av_count
                prior_av_count = 0.0
                if attr in self.av_count and val in self.av_count[attr]:
                    prior_av_count = self.av_count[attr][val]
                    tf = prior_av_count + alpha
                    sum_n_logn -= tf * math.log(tf)
                else:
                    num_vals_in_c += 1
                new_tf = prior_av_count + other_av_count + instance_av_count + alpha
                sum_n_logn += new_tf * math.log(new_tf)
        n0 = num_vals_total - num_vals_in_c
        info = -ratio * ((1.0 / (attr_count + num_vals_total * alpha)) * (sum_n_logn + n0 * alpha * math.log(alpha)) - math.log(attr_count + num_vals_total * alpha))
        return info

    def entropy_merge(self, other: "CobwebNode", instance: AV_COUNT_TYPE) -> float:
        info = 0.0
        for attr in self.av_count.keys():
            if attr < 0:
                continue
            info += self.entropy_attr_merge(attr, other, instance)
        for attr in other.av_count.keys():
            if attr < 0:
                continue
            if attr in self.av_count:
                continue
            info += self.entropy_attr_merge(attr, other, instance)
        for attr in instance.keys():
            if attr < 0:
                continue
            if attr in self.av_count:
                continue
            if attr in other.av_count:
                continue
            info += self.entropy_attr_merge(attr, other, instance)
        return info

    def get_best_level(self, instance: INSTANCE_TYPE) -> "CobwebNode":
        curr = self
        best = self
        best_ll = self.log_prob_class_given_instance(instance, True)
        while curr.parent is not None:
            curr = curr.parent
            curr_ll = curr.log_prob_class_given_instance(instance, True)
            if curr_ll > best_ll:
                best = curr
                best_ll = curr_ll
        return best

    def get_basic_level(self) -> "CobwebNode":
        curr = self
        best = self
        best_cu = self.category_utility()
        while curr.parent is not None:
            curr = curr.parent
            curr_cu = curr.category_utility()
            if curr_cu > best_cu:
                best = curr
                best_cu = curr_cu
        return best

    def entropy_attr(self, attr: ATTR_TYPE) -> float:
        if attr < 0:
            return 0.0
        alpha = self.tree.alpha
        num_vals_total = len(self.tree.attr_vals[attr])
        num_vals_in_c = 0
        attr_count = 0.0
        if attr in self.av_count:
            attr_count = self.a_count.get(attr, 0.0)
            num_vals_in_c = len(self.av_count[attr])
        ratio = 1.0
        if self.tree.weight_attr:
            ratio = (1.0 * self.tree.root.a_count[attr]) / (self.tree.root.count)
        sum_n_logn = self.sum_n_logn.get(attr, 0.0)
        n0 = num_vals_total - num_vals_in_c
        info = -ratio * ((1.0 / (attr_count + num_vals_total * alpha)) * (sum_n_logn + n0 * alpha * math.log(alpha)) - math.log(attr_count + num_vals_total * alpha))
        return info

    def entropy(self) -> float:
        info = 0.0
        for attr in self.av_count.keys():
            if attr < 0:
                continue
            info += self.entropy_attr(attr)
        return info

    def get_best_operation(self, instance: AV_COUNT_TYPE, best1: "CobwebNode", best2: Optional["CobwebNode"], best1_pu: float) -> OPERATION_TYPE:
        if best1 is None:
            raise ValueError("Need at least one best child.")
        ops: List[Tuple[float, float, int]] = []
        ops.append((best1_pu, custom_rand(), BEST))
        ops.append((self.pu_for_new_child(instance), custom_rand(), NEW))
        if len(self.children) > 2 and best2 is not None:
            ops.append((self.pu_for_merge(best1, best2, instance), custom_rand(), MERGE))
        if len(best1.children) > 0:
            ops.append((self.pu_for_split(best1), custom_rand(), SPLIT))
        ops.sort(key=lambda x: (x[0], x[1]), reverse=True)
        bestOp = (ops[0][0], ops[0][2])
        return bestOp

    def two_best_children(self, instance: AV_COUNT_TYPE) -> Tuple[float, "CobwebNode", Optional["CobwebNode"]]:
        if not self.children:
            raise ValueError("No children!")
        if self.tree.objective == 0:
            relative_pu: List[Tuple[float, float, float, CobwebNode]] = []
            for child in self.children:
                relative_pu.append(((child.count * child.entropy()) - ((child.count + 1) * child.entropy_insert(instance)), child.count, custom_rand(), child))
            relative_pu.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            best1 = relative_pu[0][3]
            best1_pu = 0.0
            best2 = relative_pu[1][3] if len(relative_pu) > 1 else None
            return (best1_pu, best1, best2)
        else:
            pus: List[Tuple[float, float, float, CobwebNode]] = []
            for child in self.children:
                pus.append((self.pu_for_insert(child, instance), child.count, custom_rand(), child))
            pus.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            best1 = pus[0][3]
            best1_pu = 0.0
            best2 = pus[1][3] if len(pus) > 1 else None
            return (best1_pu, best1, best2)

    def partition_utility(self) -> float:
        if not self.children:
            return 0.0
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = 0.0
            for attr, val_set in self.tree.attr_vals.items():
                parent_entropy += self.entropy_attr(attr)
            for child in self.children:
                p_of_child = (1.0 * child.count) / self.count
                concept_entropy -= p_of_child * math.log(p_of_child)
                for attr, val_set in self.tree.attr_vals.items():
                    children_entropy += p_of_child * child.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= len(self.children)
            return obj
        entropy = 0.0
        for attr, val_set in self.tree.attr_vals.items():
            children_entropy = 0.0
            concept_entropy = 0.0
            for child in self.children:
                p_of_child = (1.0 * child.count) / self.count
                children_entropy += p_of_child * child.entropy_attr(attr)
                concept_entropy -= p_of_child * math.log(p_of_child)
            parent_entropy = self.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= len(self.children)
            entropy += obj
        return entropy

    def pu_for_insert(self, child: "CobwebNode", instance: AV_COUNT_TYPE) -> float:
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = 0.0
            for attr, val_set in self.tree.attr_vals.items():
                parent_entropy += self.entropy_attr_insert(attr, instance)
            for c in self.children:
                if c == child:
                    p_of_child = (c.count + 1.0) / (self.count + 1.0)
                    concept_entropy -= p_of_child * math.log(p_of_child)
                    for attr, val_set in self.tree.attr_vals.items():
                        children_entropy += p_of_child * c.entropy_attr_insert(attr, instance)
                else:
                    p_of_child = (1.0 * c.count) / (self.count + 1.0)
                    concept_entropy -= p_of_child * math.log(p_of_child)
                    for attr, val_set in self.tree.attr_vals.items():
                        children_entropy += p_of_child * c.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= len(self.children)
            return obj
        entropy = 0.0
        for attr, val_set in self.tree.attr_vals.items():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                if c == child:
                    p_of_child = (c.count + 1.0) / (self.count + 1.0)
                    children_entropy += p_of_child * c.entropy_attr_insert(attr, instance)
                    concept_entropy -= p_of_child * math.log(p_of_child)
                else:
                    p_of_child = (1.0 * c.count) / (self.count + 1.0)
                    children_entropy += p_of_child * c.entropy_attr(attr)
                    concept_entropy -= p_of_child * math.log(p_of_child)
            parent_entropy = self.entropy_attr_insert(attr, instance)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children))
            entropy += obj
        return entropy

    def pu_for_new_child(self, instance: AV_COUNT_TYPE) -> float:
        new_child = CobwebNode()
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance)
        p_of_new_child = 1.0 / (self.count + 1.0)
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = -p_of_new_child * math.log(p_of_new_child)
            for attr, val_set in self.tree.attr_vals.items():
                children_entropy += p_of_new_child * new_child.entropy_attr(attr)
                parent_entropy += self.entropy_attr_insert(attr, instance)
            for child in self.children:
                p_of_child = (1.0 * child.count) / (self.count + 1.0)
                concept_entropy -= p_of_child * math.log(p_of_child)
                for attr, val_set in self.tree.attr_vals.items():
                    children_entropy += p_of_child * child.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children) + 1)
            return obj
        entropy = 0.0
        for attr, val_set in self.tree.attr_vals.items():
            children_entropy = p_of_new_child * new_child.entropy_attr(attr)
            concept_entropy = -p_of_new_child * math.log(p_of_new_child)
            for c in self.children:
                p_of_child = (1.0 * c.count) / (self.count + 1.0)
                children_entropy += p_of_child * c.entropy_attr(attr)
                concept_entropy -= p_of_child * math.log(p_of_child)
            parent_entropy = self.entropy_attr_insert(attr, instance)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children) + 1)
            entropy += obj
        return entropy

    def pu_for_merge(self, best1: "CobwebNode", best2: "CobwebNode", instance: AV_COUNT_TYPE) -> float:
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            p_of_merged = (best1.count + best2.count + 1.0) / (self.count + 1.0)
            concept_entropy = -p_of_merged * math.log(p_of_merged)
            for attr, val_set in self.tree.attr_vals.items():
                parent_entropy += self.entropy_attr_insert(attr, instance)
                children_entropy += p_of_merged * best1.entropy_attr_merge(attr, best2, instance)
            for child in self.children:
                if child == best1 or child == best2:
                    continue
                p_of_child = (1.0 * child.count) / (self.count + 1.0)
                concept_entropy -= p_of_child * math.log(p_of_child)
                for attr, val_set in self.tree.attr_vals.items():
                    children_entropy += p_of_child * child.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children) - 1)
            return obj
        entropy = 0.0
        for attr, val_set in self.tree.attr_vals.items():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                if c == best1 or c == best2:
                    continue
                p_of_child = (1.0 * c.count) / (self.count + 1.0)
                children_entropy += p_of_child * c.entropy_attr(attr)
                concept_entropy -= p_of_child * math.log(p_of_child)
            p_of_child = (best1.count + best2.count + 1.0) / (self.count + 1.0)
            children_entropy += p_of_child * best1.entropy_attr_merge(attr, best2, instance)
            concept_entropy -= p_of_child * math.log(p_of_child)
            parent_entropy = self.entropy_attr_insert(attr, instance)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children) - 1)
            entropy += obj
        return entropy

    def pu_for_split(self, best: "CobwebNode") -> float:
        if not self.tree.norm_attributes:
            parent_entropy = 0.0
            children_entropy = 0.0
            concept_entropy = 0.0
            for attr, val_set in self.tree.attr_vals.items():
                parent_entropy += self.entropy_attr(attr)
            for child in self.children:
                if child == best:
                    continue
                p_of_child = (1.0 * child.count) / self.count
                concept_entropy -= p_of_child * math.log(p_of_child)
                for attr, val_set in self.tree.attr_vals.items():
                    children_entropy += p_of_child * child.entropy_attr(attr)
            for child in best.children:
                p_of_child = (1.0 * child.count) / self.count
                concept_entropy -= p_of_child * math.log(p_of_child)
                for attr, val_set in self.tree.attr_vals.items():
                    children_entropy += p_of_child * child.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children) - 1 + len(best.children))
            return obj
        entropy = 0.0
        for attr, val_set in self.tree.attr_vals.items():
            children_entropy = 0.0
            concept_entropy = 0.0
            for c in self.children:
                if c == best:
                    continue
                p_of_child = (1.0 * c.count) / self.count
                children_entropy += p_of_child * c.entropy_attr(attr)
                concept_entropy -= p_of_child * math.log(p_of_child)
            for c in best.children:
                p_of_child = (1.0 * c.count) / self.count
                children_entropy += p_of_child * c.entropy_attr(attr)
                concept_entropy -= p_of_child * math.log(p_of_child)
            parent_entropy = self.entropy_attr(attr)
            obj = (parent_entropy - children_entropy)
            if self.tree.objective == 1:
                obj /= parent_entropy
            elif self.tree.objective == 2:
                obj /= (children_entropy + concept_entropy)
            if self.tree.children_norm:
                obj /= (len(self.children) - 1 + len(best.children))
            entropy += obj
        return entropy

    def is_exact_match(self, instance: AV_COUNT_TYPE) -> bool:
        all_attrs: Set[int] = set()
        for attr in instance.keys():
            all_attrs.add(attr)
        for attr in self.av_count.keys():
            all_attrs.add(attr)
        for attr in all_attrs:
            if attr < 0:
                continue
            if attr in instance and attr not in self.av_count:
                return False
            if attr in self.av_count and attr not in instance:
                return False
            if attr in self.av_count and attr in instance:
                instance_attr_count = 0.0
                all_vals: Set[int] = set()
                for val in self.av_count[attr].keys():
                    all_vals.add(val)
                for val, cnt in instance[attr].items():
                    all_vals.add(val)
                    instance_attr_count += cnt
                for val in all_vals:
                    if val in instance[attr] and val not in self.av_count[attr]:
                        return False
                    if val in self.av_count[attr] and val not in instance[attr]:
                        return False
                    instance_prob = (1.0 * instance[attr][val]) / instance_attr_count
                    concept_prob = (1.0 * self.av_count[attr][val]) / self.a_count[attr]
                    if abs(instance_prob - concept_prob) > 1e-5:
                        return False
        return True

    def _hash(self) -> int:
        return hash(self.concept_id)

    def __str__(self) -> str:
        return self.pretty_print()

    def concept_hash(self) -> str:
        return self.concept_id

    def pretty_print(self, depth: int = 0) -> str:
        ret = "\t" * depth + "|-" + self.avcounts_to_json() + "\n"
        for c in self.children:
            ret += c.pretty_print(depth + 1)
        return ret

    def depth(self) -> int:
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def is_parent(self, otherConcept: "CobwebNode") -> bool:
        temp = otherConcept
        while temp is not None:
            if temp == self:
                return True
            temp = temp.parent
        return False

    def num_concepts(self) -> int:
        childrenCount = 0
        for c in self.children:
            childrenCount += c.num_concepts()
        return 1 + childrenCount

    def avcounts_to_json(self) -> str:
        ret = "{"
        ret += '"_category_utility": {"#ContinuousValue#": {"mean": ' + str(self.category_utility()) + ', "std": 1, "n": 1}},\n'
        c = 0
        for attr, vAttr in self.av_count.items():
            ret += f'"{attr}": {{'
            inner_items = []
            for val, cnt in vAttr.items():
                inner_items.append(f'"{val}": {double_to_string(cnt)}')
            ret += ", ".join(inner_items)
            ret += "}"
            c += 1
            if c != len(self.av_count):
                ret += ", "
        ret += "}"
        return ret

    def ser_avcounts(self) -> str:
        ret = "{"
        c = 0
        for attr, vAttr in self.av_count.items():
            ret += f'"{attr}": {{'
            inner_items = []
            for val, cnt in vAttr.items():
                inner_items.append(f'"{val}": {double_to_string(cnt)}')
            ret += ", ".join(inner_items)
            ret += "}"
            c += 1
            if c != len(self.av_count):
                ret += ", "
        ret += "}"
        return ret

    def a_count_to_json(self) -> str:
        ret = "{"
        first = True
        for attr, cnt in self.a_count.items():
            if not first:
                ret += ",\n"
            first = False
            ret += f'"{attr}": {double_to_string(cnt)}'
        ret += "}"
        return ret

    def sum_n_logn_to_json(self) -> str:
        ret = "{"
        first = True
        for attr, cnt in self.sum_n_logn.items():
            if not first:
                ret += ",\n"
            first = False
            ret += f'"{attr}": {double_to_string(cnt)}'
        ret += "}"
        return ret

    def dump_json(self) -> str:
        output = "{"
        output += f'"concept_hash": "{self.concept_id}",\n'
        output += f'"count": {double_to_string(self.count)},\n'
        output += '"a_count": ' + self.a_count_to_json() + ',\n'
        output += '"sum_n_logn": ' + self.sum_n_logn_to_json() + ',\n'
        output += '"av_count": ' + self.ser_avcounts() + ',\n'
        output += '"children": [\n'
        first = True
        for c in self.children:
            if not first:
                output += ","
            else:
                first = False
            output += c.dump_json()
        output += ']\n'
        output += '}'
        return output

    def output_json(self) -> str:
        output = '{'
        output += f'"name": "Concept{self.concept_id}",\n'
        output += f'"size": {int(self.count)},\n'
        output += '"children": [\n'
        first = True
        for c in self.children:
            if not first:
                output += ","
            else:
                first = False
            output += c.output_json()
        output += '],\n'
        output += '"counts": ' + self.avcounts_to_json() + ',\n'
        output += '"attr_counts": ' + self.a_count_to_json() + '\n'
        output += '}'
        return output

    def predict_weighted_leaves_probs(self, instance: INSTANCE_TYPE) -> Dict[int, Dict[int, float]]:
        concept_weights = 0.0
        out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        curr = self
        while curr.parent is not None:
            prev = curr
            curr = curr.parent
            for child in curr.children:
                if child == prev:
                    continue
                c_prob = math.exp(child.log_prob_class_given_instance(instance, True))
                concept_weights += c_prob
                for attr, val_set in self.tree.attr_vals.items():
                    num_vals = len(self.tree.attr_vals[attr])
                    alpha = self.tree.alpha
                    attr_count = child.a_count.get(attr, 0.0)
                    for val in val_set:
                        av_count = child.av_count.get(attr, {}).get(val, 0.0)
                        p = ((av_count + alpha) / (attr_count + num_vals * alpha))
                        out[attr][val] += p * c_prob
        for attr, val_set in self.tree.attr_vals.items():
            for val in val_set:
                out[attr][val] = out[attr][val] / concept_weights if concept_weights != 0 else 0.0
        return out

    def predict_weighted_probs(self, instance: INSTANCE_TYPE) -> Dict[int, Dict[int, float]]:
        concept_weights = 0.0
        out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        curr = self
        while curr is not None:
            c_prob = math.exp(curr.log_prob_class_given_instance(instance, True))
            concept_weights += c_prob
            for attr, val_set in self.tree.attr_vals.items():
                num_vals = len(self.tree.attr_vals[attr])
                alpha = self.tree.alpha
                attr_count = curr.a_count.get(attr, 0.0)
                for val in val_set:
                    av_count = curr.av_count.get(attr, {}).get(val, 0.0)
                    p = ((av_count + alpha) / (attr_count + num_vals * alpha))
                    out[attr][val] += p * c_prob
            curr = curr.parent
        for attr, val_set in self.tree.attr_vals.items():
            for val in val_set:
                out[attr][val] = out[attr][val] / concept_weights if concept_weights != 0 else 0.0
        return out

    def predict_log_probs(self) -> Dict[int, Dict[int, float]]:
        out: Dict[int, Dict[int, float]] = defaultdict(dict)
        for attr, val_set in self.tree.attr_vals.items():
            num_vals = len(self.tree.attr_vals[attr])
            alpha = self.tree.alpha
            attr_count = 0.0
            if attr in self.a_count:
                attr_count = self.a_count[attr]
                second_term = math.log(attr_count + num_vals * alpha)
            else:
                second_term = math.log(num_vals) + self.tree.log_alpha
            for val in val_set:
                av_count = self.av_count.get(attr, {}).get(val, 0.0)
                if av_count > 0 or True:
                    out[attr][val] = math.log(av_count + alpha) - second_term
        return out

    def predict_probs(self) -> Dict[int, Dict[int, float]]:
        out: Dict[int, Dict[int, float]] = defaultdict(dict)
        for attr, val_set in self.tree.attr_vals.items():
            num_vals = len(self.tree.attr_vals[attr])
            alpha = self.tree.alpha
            attr_count = self.a_count.get(attr, 0.0)
            for val in val_set:
                av_count = self.av_count.get(attr, {}).get(val, 0.0)
                p = ((av_count + alpha) / (attr_count + num_vals * alpha))
                out[attr][val] = p
        return out

    def get_weighted_values(self, attr: ATTR_TYPE, allowNone: bool = True) -> List[Tuple[int, float]]:
        choices: List[Tuple[int, float]] = []
        if attr not in self.av_count:
            choices.append((-1, 1.0))
            return choices
        valCount = 0.0
        for val, tmp in self.av_count[attr].items():
            count = tmp
            choices.append((val, (1.0 * count) / self.count))
            valCount += count
        if allowNone:
            choices.append((-1, ((1.0 * (self.count - valCount)) / self.count)))
        return choices

    def predict(self, attr: ATTR_TYPE, choiceFn: str = "most likely", allowNone: bool = True) -> int:
        if choiceFn in ("most likely", "m"):
            choose = most_likely_choice
        elif choiceFn in ("sampled", "s"):
            choose = weighted_choice
        else:
            raise ValueError("Unknown choice_fn")
        if attr not in self.av_count:
            return -1
        choices = self.get_weighted_values(attr, allowNone)
        return choose(choices)

    def probability(self, attr: ATTR_TYPE, val: VALUE_TYPE) -> float:
        if val == -1:
            c = 0.0
            if attr in self.av_count:
                for a, vAttr in self.av_count.items():
                    for v, cnt in vAttr.items():
                        c += cnt
                return (1.0 * (self.count - c)) / self.count
        if attr in self.av_count and val in self.av_count[attr]:
            return (1.0 * self.av_count[attr][val]) / self.count
        return 0.0

    def category_utility(self) -> float:
        root_entropy = 0.0
        child_entropy = 0.0
        p_of_child = (1.0 * self.count) / self.tree.root.count if self.tree.root.count > 0 else 0.0
        for attr, val_set in self.tree.attr_vals.items():
            root_entropy += self.tree.root.entropy_attr(attr)
            child_entropy += self.entropy_attr(attr)
        return p_of_child * (root_entropy - child_entropy)

    def log_prob_children_given_instance_ext(self, instance: INSTANCE_TYPE) -> List[float]:
        return self.log_prob_children_given_instance(instance)

    def log_prob_children_given_instance(self, instance: AV_COUNT_TYPE) -> List[float]:
        raw_log_probs: List[float] = []
        for child in self.children:
            raw_log_probs.append(child.log_prob_class_given_instance(instance, False))
        log_p_of_x = logsumexp(raw_log_probs)
        return [lp - log_p_of_x for lp in raw_log_probs]

    def prob_children_given_instance_ext(self, instance: INSTANCE_TYPE) -> List[float]:
        return self.prob_children_given_instance(instance)

    def prob_children_given_instance(self, instance: AV_COUNT_TYPE) -> List[float]:
        raw_probs: List[float] = []
        for child in self.children:
            p = math.exp(child.log_prob_class_given_instance(instance, False))
            raw_probs.append(p)
        s = sum(raw_probs)
        return [p / s for p in raw_probs]

    def log_prob_class_given_instance_ext(self, instance: INSTANCE_TYPE, use_root_counts: bool = False) -> float:
        return self.log_prob_class_given_instance(instance, use_root_counts)

    def log_prob_class_given_instance(self, instance: AV_COUNT_TYPE, use_root_counts: bool = False) -> float:
        lp = self.log_prob_instance(instance)
        if use_root_counts:
            if self.tree.root.count > 0:
                lp += math.log((1.0 * self.count) / self.tree.root.count)
        else:
            if self.parent is None or self.parent.count == 0:
                lp += 0.0
            else:
                lp += math.log((1.0 * self.count) / self.parent.count)
        return lp

    def log_prob_instance_ext(self, instance: INSTANCE_TYPE) -> float:
        return self.log_prob_instance(instance)

    def log_prob_instance(self, instance: AV_COUNT_TYPE) -> float:
        log_prob = 0.0
        for attr, vAttr in instance.items():
            hidden = (attr < 0)
            if hidden or attr not in self.tree.attr_vals:
                continue
            num_vals = len(self.tree.attr_vals[attr])
            for val, cnt in vAttr.items():
                if val not in self.tree.attr_vals[attr]:
                    continue
                alpha = self.tree.alpha
                av_count = alpha
                if attr in self.av_count and val in self.av_count[attr]:
                    av_count += self.av_count[attr][val]
                a_count = num_vals * alpha + (self.a_count.get(attr, 0.0) if attr in self.a_count else 0.0)
                log_prob += cnt * (math.log(av_count) - math.log(a_count))
        return log_prob

    def log_prob_instance_missing_ext(self, instance: INSTANCE_TYPE) -> float:
        return self.log_prob_instance_missing(instance)

    def log_prob_instance_missing(self, instance: AV_COUNT_TYPE) -> float:
        log_prob = 0.0
        for attr, val_set in self.tree.attr_vals.items():
            if attr < 0:
                continue
            num_vals = len(self.tree.attr_vals[attr])
            alpha = self.tree.alpha
            if attr in instance:
                for val, cnt in instance[attr].items():
                    if val not in self.tree.attr_vals[attr]:
                        continue
                    av_count = alpha
                    if attr in self.av_count and val in self.av_count[attr]:
                        av_count += self.av_count[attr][val]
                    a_count = num_vals * alpha + (self.a_count.get(attr, 0.0) if attr in self.a_count else 0.0)
                    log_prob += cnt * (math.log(av_count) - math.log(a_count))
            else:
                cnt = 1.0
                if self.tree.weight_attr and self.tree.root.count > 0:
                    cnt = (1.0 * self.tree.root.a_count.get(attr, 0.0)) / self.tree.root.count
                num_vals_in_c = 0
                if attr in self.av_count:
                    attr_count = self.a_count.get(attr, 0.0)
                    num_vals_in_c = len(self.av_count[attr])
                    for val, av_count in self.av_count[attr].items():
                        p = ((av_count + alpha) / (attr_count + num_vals * alpha))
                        log_prob += cnt * p * math.log(p)
                n0 = num_vals - num_vals_in_c
                p_missing = alpha / (num_vals * alpha)
                log_prob += cnt * n0 * p_missing * math.log(p_missing)
        return log_prob

    def set_av_count(self, new_av_count: AV_COUNT_TYPE) -> None:
        # Replace av_count and recompute dependents
        self.av_count = defaultdict(lambda: defaultdict(float))
        for attr, val_map in new_av_count.items():
            for val, cnt in val_map.items():
                self.av_count[attr][val] = cnt
        self.a_count.clear()
        self.sum_n_logn.clear()
        self.count = 0.0
        for attr, val_map in self.av_count.items():
            attr_sum = 0.0
            for val, cnt in val_map.items():
                self.a_count[attr] = self.a_count.get(attr, 0.0) + cnt
                attr_sum += cnt
                self.count += cnt
                if cnt > 0:
                    self.sum_n_logn[attr] = self.sum_n_logn.get(attr, 0.0) + (cnt * math.log(cnt))


class CobwebTree:
    def __init__(self, alpha: float = 1.0, weight_attr: bool = False, objective: int = 0, children_norm: bool = True, norm_attributes: bool = False):
        self.alpha = alpha
        self.log_alpha = math.log(alpha) if alpha > 0 else float('-inf')
        self.weight_attr = weight_attr
        self.objective = objective
        self.children_norm = children_norm
        self.norm_attributes = norm_attributes
        self.root = CobwebNode()
        self.root.tree = self
        self.attr_vals: AV_KEY_TYPE = defaultdict(set)

    def partial_categorize(self, instance: INSTANCE_TYPE, k: int) -> CobwebNode:
        if k <= 0:
            return self.root
        curr = self.root
        steps = 0
        while True:
            if not curr.children:
                return curr
            if steps >= k:
                return curr
            parent = curr
            curr = None
            best_logp = None
            for child in parent.children:
                logp = child.log_prob_class_given_instance_ext(instance, False)
                if curr is None or logp > best_logp:
                    best_logp = logp
                    curr = child
            steps += 1
        return curr

    def __str__(self) -> str:
        return self.root.__str__()

    def dump_json(self) -> str:
        obj = {
            "alpha": self.alpha,
            "weight_attr": self.weight_attr,
            "objective": self.objective,
            "children_norm": self.children_norm,
            "norm_attributes": self.norm_attributes,
            "root": json.loads(self.root.dump_json())
        }
        return json.dumps(obj)

    def write_json_stream(self, save_path: str) -> None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(self.dump_json())

    def load_json_stream(self, json_model_path: str) -> None:
        with open(json_model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # naive loader: recursively build nodes
        def build_node(d: Dict[str, Any], parent: Optional[CobwebNode] = None) -> CobwebNode:
            node = CobwebNode()
            node.tree = self
            node.parent = parent
            node.concept_id = d.get("concept_hash", generate_concept_hash())
            node.count = d.get("count", 0.0)
            node.a_count = defaultdict(float, {int(k): float(v) for k, v in d.get("a_count", {}).items()})
            node.sum_n_logn = defaultdict(float, {int(k): float(v) for k, v in d.get("sum_n_logn", {}).items()})
            avc = d.get("av_count", {})
            for ak, av in avc.items():
                a_k = int(ak)
                for vk, vv in av.items():
                    node.av_count[a_k][int(vk)] = float(vv)
            for attr, val_map in node.av_count.items():
                for val in val_map.keys():
                    self.attr_vals[attr].add(val)
            for child_d in d.get("children", []):
                child = build_node(child_d, node)
                node.children.append(child)
            return node
        root_node = build_node(data.get("root", {}), None)
        self.root = root_node

    def clear(self) -> None:
        self.root = CobwebNode()
        self.root.tree = self
        self.attr_vals = defaultdict(set)

    def ifit_helper(self, instance: INSTANCE_TYPE, mode: int, debug: bool = False):
        return self.cobweb(instance, mode, debug)

    def ifit(self, instance: INSTANCE_TYPE, mode: int, debug: bool = False):
        return self.ifit_helper(instance, mode, debug)

    def fit(self, instances: List[INSTANCE_TYPE], mode: int, iterations: int = 1, randomizeFirst: bool = True) -> None:
        for i in range(iterations):
            if i == 0 and randomizeFirst:
                random.shuffle(instances)
            for instance in instances:
                self.ifit(instance, mode)
            random.shuffle(instances)

    def chooseRandomAction(self) -> int:
        return random.randint(BEST, SPLIT)

    def cobweb(self, instance: AV_COUNT_TYPE, mode: int, debug: bool = False):
        for attr, val_map in instance.items():
            for val in val_map.keys():
                self.attr_vals[attr].add(val)
        current = self.root
        operation_stats: Dict[str, float] = {}
        debug_logs: List[str] = []
        while True:
            if not current.children and (current.count == 0 or current.is_exact_match(instance)):
                if debug:
                    debug_logs.append(json.dumps({"action": "NEW", "node": current.concept_hash(), "parent": current.parent.concept_hash() if current.parent else None}))
                current.increment_counts(instance)
                break
            elif not current.children:
                # fringe split
                old_current_clone = CobwebNode()
                old_current_clone.tree = self
                old_current_clone.parent = current
                old_current_clone.count = current.count
                old_current_clone.a_count = defaultdict(float, current.a_count)
                old_current_clone.sum_n_logn = defaultdict(float, current.sum_n_logn)
                for attr, val_map in current.av_count.items():
                    for val, cnt in val_map.items():
                        old_current_clone.av_count[attr][val] = cnt
                old_current_clone.children = current.children[:]
                for child in old_current_clone.children:
                    child.parent = old_current_clone
                current.children = []
                new_node = CobwebNode()
                new_node.tree = self
                new_node.parent = current
                current.children.append(old_current_clone)
                current.children.append(new_node)
                new_node.increment_counts(instance)
                current.increment_counts(instance)
                if current.parent is None:
                    self.root = current
                break
            else:
                # four ops
                _, best1, best2 = current.two_best_children(instance)
                bestAction = 0
                if mode == 0:
                    _, action = current.get_best_operation(instance, best1, best2, 0.0)
                    bestAction = action
                elif mode == 1:
                    bestAction = BEST
                elif mode == 2:
                    if best1 is None:
                        bestAction = random.randint(0, 1)
                    else:
                        if best2 is None:
                            bestAction = random.randint(0, 2)
                            if bestAction == 2:
                                bestAction = 3
                        else:
                            bestAction = random.randint(0, 3)
                elif mode == 3:
                    bestAction = BEST
                    if best1 is not None and best2 is not None:
                        epsilon = 1e-2
                        p = random.random()
                        if p < epsilon:
                            _, action = current.get_best_operation(instance, best1, best2, 0.0)
                            bestAction = action
                elif mode == BEST_NEW:
                    if best1 is None:
                        bestAction = NEW
                    else:
                        best_pu = current.pu_for_insert(best1, instance)
                        new_pu = current.pu_for_new_child(instance)
                        if best_pu > new_pu:
                            bestAction = BEST
                        elif new_pu > best_pu:
                            bestAction = NEW
                        else:
                            bestAction = BEST if random.randint(0, 1) == 0 else NEW
                else:
                    bestAction = 0
                if bestAction == BEST:
                    current.increment_counts(instance)
                    current = best1
                elif bestAction == NEW:
                    current.increment_counts(instance)
                    new_child = CobwebNode()
                    new_child.parent = current
                    new_child.tree = self
                    new_child.increment_counts(instance)
                    current.children.append(new_child)
                    current = new_child
                    break
                elif bestAction == MERGE:
                    current.increment_counts(instance)
                    new_child = CobwebNode()
                    new_child.parent = current
                    new_child.tree = self
                    new_child.update_counts_from_node(best1)
                    new_child.update_counts_from_node(best2)
                    best1.parent = new_child
                    best2.parent = new_child
                    new_child.children.append(best1)
                    new_child.children.append(best2)
                    current.children = [c for c in current.children if c not in (best1, best2)]
                    current.children.append(new_child)
                    current = new_child
                elif bestAction == SPLIT:
                    current.children = [c for c in current.children if c != best1]
                    for c in best1.children:
                        c.parent = current
                        c.tree = self
                        current.children.append(c)
                    # In C++ delete best1; in Python GC will collect if no refs remain
                else:
                    raise ValueError(f"Best action choice {bestAction} not recognized")
        return current, operation_stats, debug_logs

    def _cobweb_categorize(self, instance: AV_COUNT_TYPE) -> CobwebNode:
        current = self.root
        while True:
            if not current.children:
                return current
            parent = current
            current = None
            best_logp = None
            for child in parent.children:
                logp = child.log_prob_class_given_instance(instance, False)
                if current is None or logp > best_logp:
                    best_logp = logp
                    current = child
        return current

    def categorize_helper(self, instance: INSTANCE_TYPE) -> CobwebNode:
        return self._cobweb_categorize(instance)

    def categorize(self, instance: INSTANCE_TYPE) -> CobwebNode:
        return self.categorize_helper(instance)

    def predict_probs_mixture_helper(self, instance: AV_COUNT_TYPE, ll_path: float, max_nodes: int, greedy: bool, missing: bool):
        out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: -math.inf))
        for attr, val_set in self.attr_vals.items():
            for val in val_set:
                out[attr][val] = -math.inf
        operation_stats: Dict[str, float] = {}
        nodes_expanded = 0
        total_weight = 0.0
        first_weight = True
        root_ll_inst = 0.0
        if missing:
            root_ll_inst = self.root.log_prob_instance_missing(instance)
        else:
            root_ll_inst = self.root.log_prob_instance(instance)
        import heapq
        # use max-heap via negatives
        heap: List[Tuple[float, float, CobwebNode]] = [(-(root_ll_inst), 0.0, self.root)]
        start = time.time()
        while heap:
            _, curr_ll, curr = heapq.heappop(heap)
            curr_ll = -_[0] if isinstance(_, tuple) else curr_ll
            nodes_expanded += 1
            if greedy:
                heap = []
            # accumulate total_weight via logsumexp
            # for simplicity compute score as curr_ll
            curr_score = curr_ll
            if first_weight:
                total_weight = curr_score
                first_weight = False
            else:
                total_weight = eff_logsumexp(total_weight, curr_score)
            # predict log probs
            curr_log_probs = curr.predict_log_probs()
            for attr, val_set in curr_log_probs.items():
                for val, log_p in val_set.items():
                    out[attr][val] = eff_logsumexp(out[attr][val], curr_score + log_p)
            if nodes_expanded >= max_nodes:
                break
            log_children_probs = curr.log_prob_children_given_instance(instance)
            for i, child in enumerate(curr.children):
                if missing:
                    child_ll_inst = child.log_prob_instance_missing(instance)
                else:
                    child_ll_inst = child.log_prob_instance(instance)
                child_ll_given_parent = log_children_probs[i]
                child_ll = child_ll_given_parent + curr_ll
                heapq.heappush(heap, (-(child_ll_inst + child_ll), child_ll, child))
        # exponentiate normalized
        for attr, val_set in out.items():
            for val, p in val_set.items():
                out[attr][val] = math.exp(p - total_weight) if total_weight != 0 else 0.0
        return operation_stats, out

    def predict_probs_mixture(self, instance: INSTANCE_TYPE, max_nodes: int, greedy: bool, missing: bool):
        return self.predict_probs_mixture_helper(instance, 0.0, max_nodes, greedy, missing)

    def predict_probs_mixture_parallel(self, instances: List[INSTANCE_TYPE], max_nodes: int, greedy: bool, missing: bool, num_threads: int):
        # simple serial implementation (threading optional)
        out = []
        for inst in instances:
            _, probs = self.predict_probs_mixture(inst, max_nodes, greedy, missing)
            out.append(probs)
        return out


# Helper functions used in class methods

def double_to_string(x: float) -> str:
    return f"{x:.12g}"


def most_likely_choice(choices: List[Tuple[int, float]]) -> int:
    vals: List[Tuple[float, float, int]] = []
    for val, prob in choices:
        if prob < 0:
            print("most_likely_choice: all weights must be greater than or equal to 0")
        vals.append((prob, custom_rand(), val))
    vals.sort(reverse=True)
    return vals[0][2]


def weighted_choice(choices: List[Tuple[int, float]]) -> int:
    # fallback simple weighted sample
    total = sum(w for _, w in choices)
    if total <= 0:
        return choices[0][0]
    r = random.random() * total
    upto = 0.0
    for val, w in choices:
        if upto + w >= r:
            return val
        upto += w
    return choices[-1][0]


# expose main classes for import
__all__ = ["CobwebTree", "CobwebNode"]
