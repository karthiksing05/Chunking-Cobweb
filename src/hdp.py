"""
Hierarchical Dirichlet Process (HDP) for discrete attribute-value data.

Theory
------
This module implements a *Nested Chinese Restaurant Process* (nCRP/HDP) tree
that satisfies the four data-structure requirements from MULTIHIERARCHY.md:

  1. **Incremental / online** — ``ifit(instance)`` adds one instance at a
     time via the Chinese Restaurant Franchise (CRF) without replaying the
     full corpus.  Optional Gibbs refinement sweeps push toward the MAP
     posterior.

  2. **Discrete attribute-value input** — instances are plain Python dicts
     of the form ``{attr: {val: count}, ...}``, identical to the format
     used by CobwebDiscreteTree throughout parse_mh.py.  Attributes whose
     key is a *negative integer* are treated as **hidden** and excluded from
     main likelihood computations (same convention as the rest of the
     codebase).

  3. **Recognition score** — defined in MULTIHIERARCHY.md as
          score = log p(x | c_basic) + log(count_basic / count_root)
     where ``c_basic`` is the *basic-level* node for the instance (see
     below).  Returned by ``recognition_score(instance)``.

  4. **Adaptive basic level** — the basic-level node is the node along the
     categorisation path (root → leaf) that *maximises* the recognition
     score.  As more data arrives the partition shifts, so the basic level
     adapts automatically.

nCRP Dynamics
-------------
The tree is grown via the nested CRP.  Every ``HDPNode`` owns a local CRP
(concentration ``alpha``).  When an instance is inserted:

  * Start at the root.
  * At each node, assign the instance to an **existing child** with
    probability  count_k / (node.count - 1 + alpha)  (CRP),
    or spawn a **new child** with probability  alpha / (node.count-1+alpha).
  * Continue until reaching a leaf (a node whose depth equals the
    ``max_depth`` cap, or one with no children when the instance could not
    improve any child).
  * The instance is counted at *every* node along its path (hierarchical
    counts), so ancestors always have count ≥ descendants.

Likelihood: Dirichlet-Categorical
----------------------------------
Within each node the predictive likelihood for an attribute-value pair is

    p(x_attr = v | node) = (n_{node,attr,v} + beta) / (n_{node,attr} + beta*|V_attr|)

where ``n_{node,attr,v}`` is the *local* count of value v for attr in that
node, ``n_{node,attr}`` = Σ_v n_{node,attr,v}, and ``|V_attr|`` is the
global vocabulary size for attr.  This is the standard collapsed
Dirichlet-Categorical predictive posterior.

Gibbs Sampling
--------------
``gibbs_sweep()`` performs one full pass over all stored instances,
removing and resampling their entire root-to-leaf path.  Call it explicitly
or set ``n_gibbs_init`` on the constructor to run sweeps automatically after
each ``ifit``.  ``fit(instances, n_passes)`` runs batch insertion followed
by ``n_passes * len(instances)`` collapsed Gibbs steps.

Serialisation
-------------
``to_dict()`` / ``from_dict()`` round-trip the full tree to a JSON-
compatible dict so it can be saved alongside the LTM JSON files already
produced by parse_mh.py.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_hidden(attr) -> bool:
    """Return True for hidden attributes (negative int keys)."""
    try:
        return int(attr) < 0
    except (ValueError, TypeError):
        return False


def _logsumexp_pair(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _logsumexp(values) -> float:
    """log-sum-exp over an iterable of floats."""
    vals = list(values)
    if not vals:
        return -math.inf
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals))


# ---------------------------------------------------------------------------
# HDPNode
# ---------------------------------------------------------------------------

class HDPNode:
    """
    A single node in the nCRP / HDP tree.

    Each ``HDPNode`` stores:
    * `av_count`   — {attr: {val: int}}  local sufficient statistics
                     (counts only from instances *directly* assigned to
                     the subtree rooted here — the hierarchy rolls up
                     automatically because each instance is counted at
                     *every* ancestor).
    * `count`      — total number of instances in this subtree.
    * `children`   — ordered list of child ``HDPNode`` objects.
    * `parent`     — reference to parent node (``None`` for root).
    * `node_id`    — stable UUID string, used in serialisation.
    """

    __slots__ = (
        "node_id", "count", "av_count", "children", "parent", "_depth_cache"
    )

    def __init__(
        self,
        node_id: Optional[str] = None,
        parent: Optional["HDPNode"] = None,
    ):
        self.node_id: str = node_id or str(uuid.uuid4())
        self.count: int = 0
        # {attr: {val: int}}
        self.av_count: Dict[Any, Dict[Any, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.children: List["HDPNode"] = []
        self.parent: Optional["HDPNode"] = parent
        self._depth_cache: Optional[int] = None

    # ------------------------------------------------------------------
    # Tree navigation
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Distance from the root (root = depth 0)."""
        if self._depth_cache is not None:
            return self._depth_cache
        d = 0
        n = self
        while n.parent is not None:
            d += 1
            n = n.parent
        self._depth_cache = d
        return d

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, child: "HDPNode"):
        child.parent = self
        child._depth_cache = None  # invalidate
        self.children.append(child)

    def path_from_root(self) -> List["HDPNode"]:
        """Return [root, ..., self]."""
        nodes = []
        n: Optional[HDPNode] = self
        while n is not None:
            nodes.append(n)
            n = n.parent
        nodes.reverse()
        return nodes

    # ------------------------------------------------------------------
    # Sufficient statistics
    # ------------------------------------------------------------------

    def _add_instance_counts(self, instance: dict):
        """Increment local av_count for *instance* (skips hidden attrs)."""
        for attr, val_dict in instance.items():
            if _is_hidden(attr):
                continue
            for val, cnt in val_dict.items():
                self.av_count[attr][val] += cnt

    def _remove_instance_counts(self, instance: dict):
        """Decrement local av_count for *instance* (skips hidden attrs)."""
        for attr, val_dict in instance.items():
            if _is_hidden(attr):
                continue
            for val, cnt in val_dict.items():
                self.av_count[attr][val] -= cnt
                if self.av_count[attr][val] <= 0:
                    del self.av_count[attr][val]

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def log_prob_instance(
        self,
        instance: dict,
        attr_value_vocab: Dict[Any, Set],
        beta: float = 1.0,
    ) -> float:
        """
        Compute  log p(instance | node)  under Dirichlet-Categorical.

        For each visible (non-hidden) attribute present in *instance*:

            p(x_attr = v | node) = (n_{node,attr,v} + β) /
                                   (n_{node,attr} + β * |V_attr|)

        Attributes not present in *attr_value_vocab* use a vocab size of 1.
        """
        log_prob = 0.0
        for attr, val_dict in instance.items():
            if _is_hidden(attr):
                continue
            vocab_size = max(len(attr_value_vocab.get(attr, ())), 1)
            n_attr = sum(self.av_count[attr].values())
            denom = n_attr + beta * vocab_size
            if denom <= 0:
                denom = 1.0
            for val, cnt in val_dict.items():
                n_av = self.av_count[attr].get(val, 0)
                p = (n_av + beta) / denom
                p = max(p, 1e-300)
                log_prob += cnt * math.log(p)
        return log_prob

    # ------------------------------------------------------------------
    # Dictionary serialisation
    # ------------------------------------------------------------------

    def _to_dict_shallow(self) -> dict:
        """Export node fields (no recursion into children)."""
        return {
            "node_id": self.node_id,
            "count": self.count,
            "av_count": {
                str(attr): {str(v): c for v, c in vd.items()}
                for attr, vd in self.av_count.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"HDPNode(id={self.node_id[:8]}…, "
            f"depth={self.depth()}, count={self.count}, "
            f"children={len(self.children)})"
        )


# ---------------------------------------------------------------------------
# HDP — the full nCRP tree
# ---------------------------------------------------------------------------

class HDP:
    """
    Hierarchical Dirichlet Process tree (nCRP).

    Parameters
    ----------
    alpha : float
        CRP concentration at every level.  Higher values grow wider trees
        (more children per node); lower values grow deeper, thinner trees.
        Default: 1.0.
    beta : float
        Symmetric Dirichlet prior for the Categorical likelihood at each
        node.  Larger beta smooths probabilities toward the uniform;
        smaller beta sharpens them.  Default: 0.1.
    max_depth : int or None
        Hard cap on tree depth (root is depth 0).  ``None`` = unlimited.
        Default: None.
    n_gibbs_init : int
        Number of Gibbs sweeps to run after every ``ifit`` call. 0 = no
        refinement (pure online, fastest).  Default: 0.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        max_depth: Optional[int] = None,
        n_gibbs_init: int = 0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.max_depth = max_depth
        self.n_gibbs_init = n_gibbs_init

        self.root: HDPNode = HDPNode(node_id="ROOT")

        # Flat index of all nodes for fast lookup
        self._nodes: Dict[str, HDPNode] = {"ROOT": self.root}

        # All stored instances with their full path (list of node_ids)
        # stored_instances[i] = (instance_dict, [node_id_root, ..., node_id_leaf])
        self._stored: List[Tuple[dict, List[str]]] = []

        # Global vocabulary: {attr: set(vals)}
        self.attr_value_vocab: Dict[Any, Set] = defaultdict(set)

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def _update_vocab(self, instance: dict):
        for attr, val_dict in instance.items():
            if _is_hidden(attr):
                continue
            for val in val_dict:
                self.attr_value_vocab[attr].add(val)

    # ------------------------------------------------------------------
    # CRP path sampling
    # ------------------------------------------------------------------

    def _crp_log_prior_existing(self, child: HDPNode, parent: HDPNode) -> float:
        """
        CRP prior for routing to an *existing* child:
            log( count_child / (count_parent - 1 + alpha) )
        where count_parent is *before* adding the current instance.
        """
        denom = max(parent.count - 1, 0) + self.alpha
        if child.count <= 0 or denom <= 0:
            return -math.inf
        return math.log(child.count / denom)

    def _crp_log_prior_new(self, parent: HDPNode) -> float:
        """
        CRP prior for creating a *new* child:
            log( alpha / (count_parent - 1 + alpha) )
        """
        denom = max(parent.count - 1, 0) + self.alpha
        if denom <= 0:
            return math.log(self.alpha) if self.alpha > 0 else -math.inf
        return math.log(self.alpha / denom)

    def _sample_child(
        self, instance: dict, node: HDPNode, depth: int
    ) -> Optional[HDPNode]:
        """
        Sample the next child for *instance* at *node* via CRP + likelihood.

        Returns the chosen child, or ``None`` if a new child should be
        spawned (but does NOT create the node — the caller does that).
        Returns ``None`` (= leaf / terminate) when ``max_depth`` is hit.
        """
        at_max_depth = self.max_depth is not None and depth >= self.max_depth

        log_scores: List[float] = []
        candidates: List[Optional[HDPNode]] = []

        for child in node.children:
            lp = (
                self._crp_log_prior_existing(child, node)
                + child.log_prob_instance(
                    instance, self.attr_value_vocab, self.beta
                )
            )
            log_scores.append(lp)
            candidates.append(child)

        if not at_max_depth:
            lp_new = (
                self._crp_log_prior_new(node)
                # Empty new node: use flat Dirichlet prior as pseudo-likelihood
                + self._new_cluster_log_likelihood(instance)
            )
            log_scores.append(lp_new)
            candidates.append(None)  # sentinel for "new child"

        if not candidates:
            return None  # leaf

        # Softmax sample
        max_s = max(log_scores)
        weights = [math.exp(s - max_s) for s in log_scores]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        chosen = candidates[-1]
        for cand, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                chosen = cand
                break

        return chosen  # None → caller creates new child

    def _new_cluster_log_likelihood(self, instance: dict) -> float:
        """
        Likelihood of *instance* under a brand-new empty cluster.
        With no counts, the Dirichlet-Cat collapses to uniform: 1/|V_attr|.
        """
        log_prob = 0.0
        for attr, val_dict in instance.items():
            if _is_hidden(attr):
                continue
            vocab_size = max(len(self.attr_value_vocab.get(attr, ())), 1)
            for _, cnt in val_dict.items():
                log_prob += cnt * math.log(1.0 / vocab_size)
        return log_prob

    # ------------------------------------------------------------------
    # Path assignment (add / remove from tree)
    # ------------------------------------------------------------------

    def _assign_path(self, instance: dict) -> List[str]:
        """
        Traverse from root to leaf, sampling children via CRP + likelihood.
        Increment counts along the entire path and update av_counts.
        Returns the list of node_ids traversed (root first, leaf last).
        """
        path_ids: List[str] = []
        node = self.root
        depth = 0

        while True:
            # Count this instance at the current node
            node.count += 1
            node._add_instance_counts(instance)
            path_ids.append(node.node_id)

            # Decide where to go next
            chosen = self._sample_child(instance, node, depth)

            if chosen is None and (
                self.max_depth is None or depth < self.max_depth
            ):
                # Spawn a new child
                new_node = HDPNode(parent=node)
                node.add_child(new_node)
                self._nodes[new_node.node_id] = new_node
                node = new_node
                depth += 1
                # Count at the new child too
                node.count += 1
                node._add_instance_counts(instance)
                path_ids.append(node.node_id)
                break  # leaf reached after spawning
            elif chosen is None:
                # At max_depth or truly no children — this node is the leaf
                break
            else:
                node = chosen
                depth += 1

        return path_ids

    def _unassign_path(self, instance: dict, path_ids: List[str]):
        """
        Undo a path assignment: decrement counts along *path_ids*.
        """
        for nid in path_ids:
            node = self._nodes.get(nid)
            if node is None:
                continue
            node.count -= 1
            node._remove_instance_counts(instance)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def ifit(self, instance: dict):
        """
        Incremental online update: add one instance.

        Steps
        -----
        1. Update global vocabulary.
        2. Sample a root-to-leaf path via nCRP + likelihood.
        3. Record the instance and its path.
        4. Optionally run ``n_gibbs_init`` Gibbs refinement sweeps.
        5. Prune empty nodes.
        """
        self._update_vocab(instance)
        path_ids = self._assign_path(instance)
        self._stored.append((instance, path_ids))

        for _ in range(self.n_gibbs_init):
            self.gibbs_sweep()

        self._prune_empty_nodes()

    def fit(self, instances: List[dict], n_passes: int = 5):
        """
        Batch fit: insert all *instances* then run Gibbs sampling.

        Parameters
        ----------
        instances : list[dict]
            Each element is an attribute-value instance.
        n_passes : int
            Number of full Gibbs passes after initial insertion.
        """
        for inst in instances:
            self._update_vocab(inst)
            path_ids = self._assign_path(inst)
            self._stored.append((inst, path_ids))

        for _ in range(n_passes):
            self.gibbs_sweep()

        self._prune_empty_nodes()

    # ------------------------------------------------------------------
    # Gibbs sampling
    # ------------------------------------------------------------------

    def gibbs_sweep(self):
        """
        One full collapsed Gibbs pass over all stored instances.

        For each instance:
          1. Remove from its current path.
          2. Resample a new root-to-leaf path.
          3. Re-assign.
        """
        indices = list(range(len(self._stored)))
        random.shuffle(indices)

        for idx in indices:
            inst, old_path = self._stored[idx]
            self._unassign_path(inst, old_path)
            new_path = self._assign_path(inst)
            self._stored[idx] = (inst, new_path)

        self._prune_empty_nodes()

    def _prune_empty_nodes(self):
        """
        Remove leaf nodes with count == 0 (can appear after Gibbs).
        Walk bottom-up so we cascade removals upward.
        """
        changed = True
        while changed:
            changed = False
            to_remove: List[str] = []
            for nid, node in list(self._nodes.items()):
                if nid == "ROOT":
                    continue
                if node.count <= 0 and node.is_leaf():
                    to_remove.append(nid)

            for nid in to_remove:
                node = self._nodes.pop(nid, None)
                if node and node.parent:
                    if node in node.parent.children:
                        node.parent.children.remove(node)
                changed = True

    # ------------------------------------------------------------------
    # Categorisation
    # ------------------------------------------------------------------

    def categorize_path(
        self, instance: dict
    ) -> Tuple[HDPNode, List[HDPNode]]:
        """
        Greedily categorise *instance* from root to leaf (MAP descent).
        Does **not** modify the model.

        Returns
        -------
        leaf : HDPNode
            The leaf node reached.
        path : list[HDPNode]
            Full path from root (index 0) to leaf (index -1).
        """
        path: List[HDPNode] = []
        node = self.root

        while True:
            path.append(node)
            if not node.children:
                break

            best_child: Optional[HDPNode] = None
            best_score = -math.inf

            for child in node.children:
                score = (
                    self._crp_log_prior_existing(child, node)
                    + child.log_prob_instance(
                        instance, self.attr_value_vocab, self.beta
                    )
                )
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break

            # Only go deeper if the best child is an improvement over staying
            node = best_child

        return path[-1], path

    def categorize(self, instance: dict) -> HDPNode:
        """Return the MAP leaf node for *instance*."""
        leaf, _ = self.categorize_path(instance)
        return leaf

    # ------------------------------------------------------------------
    # Recognition & scoring (from MULTIHIERARCHY.md)
    # ------------------------------------------------------------------

    def _recognition_score_at(self, instance: dict, node: HDPNode) -> float:
        """
        Recognition score at a specific node:
            log p(x | node) + log(count_node / count_root)
        """
        if node.count <= 0 or self.root.count <= 0:
            return -math.inf
        log_lik = node.log_prob_instance(
            instance, self.attr_value_vocab, self.beta
        )
        log_freq = math.log(node.count / self.root.count)
        return log_lik + log_freq

    def basic_level(self, instance: dict) -> Optional[HDPNode]:
        """
        Return the node along the MAP categorisation path that maximises
        the recognition score:
            score = log p(x | node) + log(count_node / count_root)

        This is the *adaptive* basic level — it shifts as data grows.
        """
        _, path = self.categorize_path(instance)
        if not path:
            return None

        best_node = path[0]
        best_score = self._recognition_score_at(instance, path[0])

        for node in path[1:]:
            s = self._recognition_score_at(instance, node)
            if s > best_score:
                best_score = s
                best_node = node

        return best_node

    def recognition_score(self, instance: dict) -> float:
        """
        Primary recognition score (MULTIHIERARCHY.md §Revising our scoring):
            score = log p(x | c_basic) + log(count_basic / count_root)

        Returns ``-inf`` if the model is empty.
        """
        c_basic = self.basic_level(instance)
        if c_basic is None:
            return -math.inf
        return self._recognition_score_at(instance, c_basic)

    def log_prob_instance(self, instance: dict) -> float:
        """
        Marginal log-likelihood  log p(x) = log Σ_k p(k) p(x|k),
        summing over *all* nodes (leaves only, weighted by their CRP prior).

        Useful for model comparison / evaluation.
        """
        leaves = [n for n in self._nodes.values() if n.is_leaf() and n.count > 0]
        if not leaves:
            return -math.inf

        total = self.root.count
        log_probs = []
        for node in leaves:
            log_prior = math.log(node.count / total) if total > 0 else -math.inf
            log_lik = node.log_prob_instance(
                instance, self.attr_value_vocab, self.beta
            )
            log_probs.append(log_prior + log_lik)

        return _logsumexp(log_probs)

    def score_along_path(
        self, instance: dict
    ) -> List[Dict[str, Any]]:
        """
        Return per-node scoring statistics along the MAP path.

        Each entry in the returned list is a dict with:
          * ``node``             — the HDPNode
          * ``depth``            — node depth
          * ``count``            — node.count
          * ``log_likelihood``   — log p(x | node)
          * ``log_frequency``    — log(count / root.count)
          * ``recognition``      — log_likelihood + log_frequency
        """
        _, path = self.categorize_path(instance)
        stats = []
        for node in path:
            ll = node.log_prob_instance(
                instance, self.attr_value_vocab, self.beta
            )
            lf = (
                math.log(node.count / self.root.count)
                if self.root.count > 0 and node.count > 0
                else -math.inf
            )
            stats.append(
                {
                    "node": node,
                    "depth": node.depth(),
                    "count": node.count,
                    "log_likelihood": ll,
                    "log_frequency": lf,
                    "recognition": ll + lf,
                }
            )
        return stats

    # ------------------------------------------------------------------
    # Prediction (fill in missing attributes)
    # ------------------------------------------------------------------

    def predict_best_value(
        self,
        instance: dict,
        query_attr: Any,
        candidates: Optional[List[Any]] = None,
        use_basic_level: bool = True,
    ) -> Optional[Any]:
        """
        Given a partial *instance* (missing *query_attr*), predict the
        most probable value for that attribute.

        Parameters
        ----------
        instance : dict
            Partial attribute-value dict (may include some values for
            *query_attr* if you want, they will be ignored in lookup).
        query_attr : any
            The attribute whose value is unknown.
        candidates : list, optional
            Restrict prediction to these values.  Defaults to all values
            seen for *query_attr* in the global vocabulary.
        use_basic_level : bool
            If True, predict from the basic-level node instead of the leaf.
            Basic-level tends to produce more generalisable predictions.
        """
        node = (
            self.basic_level(instance)
            if use_basic_level
            else self.categorize(instance)
        )
        if node is None:
            return None

        vocab = candidates or list(self.attr_value_vocab.get(query_attr, []))
        if not vocab:
            return None

        vocab_size = max(len(self.attr_value_vocab.get(query_attr, ())), 1)
        n_attr = sum(node.av_count[query_attr].values())
        denom = n_attr + self.beta * vocab_size
        if denom <= 0:
            denom = 1.0

        best_val = max(
            vocab,
            key=lambda v: (node.av_count[query_attr].get(v, 0) + self.beta) / denom,
        )
        return best_val

    def predict_distribution(
        self,
        instance: dict,
        query_attr: Any,
        use_basic_level: bool = True,
    ) -> Dict[Any, float]:
        """
        Return the full predictive posterior distribution over all known
        values for *query_attr*, given *instance*.

        Returns a dict ``{val: probability}`` (sums to 1).
        """
        node = (
            self.basic_level(instance)
            if use_basic_level
            else self.categorize(instance)
        )
        if node is None:
            return {}

        vocab = list(self.attr_value_vocab.get(query_attr, []))
        if not vocab:
            return {}

        vocab_size = len(vocab)
        n_attr = sum(node.av_count[query_attr].values())
        denom = n_attr + self.beta * vocab_size
        if denom <= 0:
            denom = 1.0

        dist = {}
        for val in vocab:
            n_av = node.av_count[query_attr].get(val, 0)
            dist[val] = (n_av + self.beta) / denom

        return dist

    # ------------------------------------------------------------------
    # Generation (sample instances from the model)
    # ------------------------------------------------------------------

    def sample_instance(
        self,
        node: Optional[HDPNode] = None,
        stochastic_path: bool = True,
    ) -> dict:
        """
        Sample a new instance from the model.

        Parameters
        ----------
        node : HDPNode, optional
            If given, sample from this specific node's Dirichlet posterior.
            If ``None``, first sample a leaf proportional to its count,
            then sample attribute values.
        stochastic_path : bool
            If True (default), sample the traversal stochastically rather
            than greedily — introduces variety in generation.
        """
        if node is None:
            leaves = [n for n in self._nodes.values() if n.is_leaf() and n.count > 0]
            if not leaves:
                return {}
            weights = [n.count for n in leaves]
            total = sum(weights)
            r = random.random() * total
            cumulative = 0.0
            node = leaves[-1]
            for leaf, w in zip(leaves, weights):
                cumulative += w
                if r <= cumulative:
                    node = leaf
                    break

        instance = {}
        for attr, val_set in self.attr_value_vocab.items():
            vocab = list(val_set)
            if not vocab:
                continue
            vocab_size = len(vocab)
            n_attr = sum(node.av_count[attr].values())
            denom = n_attr + self.beta * vocab_size
            if denom <= 0:
                denom = 1.0
            probs = [
                (node.av_count[attr].get(v, 0) + self.beta) / denom
                for v in vocab
            ]
            chosen_val = random.choices(vocab, weights=probs, k=1)[0]
            instance[attr] = {chosen_val: 1}

        return instance

    # ------------------------------------------------------------------
    # Summarisation / inspection
    # ------------------------------------------------------------------

    def node_summary(self, node: HDPNode, top_k: int = 5) -> Dict[str, Any]:
        """
        Human-readable summary of *node*: top-k attribute-value counts.
        """
        summary: Dict[str, Any] = {
            "node_id": node.node_id[:12],
            "depth": node.depth(),
            "count": node.count,
            "proportion": node.count / max(self.root.count, 1),
            "n_children": len(node.children),
            "attributes": {},
        }
        for attr, val_counts in node.av_count.items():
            total = sum(val_counts.values())
            if total == 0:
                continue
            top = sorted(val_counts.items(), key=lambda x: -x[1])[:top_k]
            summary["attributes"][attr] = [
                {"value": v, "count": c, "prob": c / total}
                for v, c in top
            ]
        return summary

    def all_node_summaries(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return summaries for all non-root nodes with count > 0,
        sorted by depth then by count descending.
        """
        nodes = [
            n
            for nid, n in self._nodes.items()
            if nid != "ROOT" and n.count > 0
        ]
        nodes.sort(key=lambda n: (n.depth(), -n.count))
        return [self.node_summary(n, top_k) for n in nodes]

    def tree_stats(self) -> Dict[str, Any]:
        """High-level statistics about the current tree."""
        all_nodes = list(self._nodes.values())
        leaves = [n for n in all_nodes if n.is_leaf()]
        depths = [n.depth() for n in all_nodes]
        return {
            "n_nodes": len(all_nodes),
            "n_leaves": len(leaves),
            "n_instances": self.root.count,
            "n_stored": len(self._stored),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "n_attrs": len(self.attr_value_vocab),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def print_tree(self, node: Optional[HDPNode] = None, indent: int = 0):
        """Print a text representation of the tree (for debugging)."""
        if node is None:
            node = self.root
        prefix = "  " * indent
        marker = "ROOT" if node.node_id == "ROOT" else node.node_id[:8] + "…"
        print(f"{prefix}[{marker}] count={node.count} children={len(node.children)}")
        for child in node.children:
            self.print_tree(child, indent + 1)

    def leaves(self) -> List[HDPNode]:
        """Return all leaf nodes with count > 0."""
        return [n for n in self._nodes.values() if n.is_leaf() and n.count > 0]

    def nodes_at_depth(self, depth: int) -> List[HDPNode]:
        """Return all nodes at a given depth with count > 0."""
        return [
            n for n in self._nodes.values()
            if n.depth() == depth and n.count > 0
        ]

    # ------------------------------------------------------------------
    # Hyperparameter utilities
    # ------------------------------------------------------------------

    def set_alpha(self, alpha: float):
        """Update concentration parameter (takes effect on next assignment)."""
        self.alpha = alpha

    def set_beta(self, beta: float):
        """Update Dirichlet prior (takes effect on next likelihood call)."""
        self.beta = beta

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the full tree to a JSON-compatible dict.

        Format:
          {
            "alpha": …, "beta": …, "max_depth": …, "n_gibbs_init": …,
            "attr_value_vocab": {attr: [val, …], …},
            "nodes": {node_id: {node_id, count, av_count}, …},
            "tree": {node_id: [child_node_id, …], …},
            "stored": [[instance, [path_ids…]], …]
          }
        """
        nodes_out: Dict[str, dict] = {}
        tree_out: Dict[str, list] = {}

        for nid, node in self._nodes.items():
            nodes_out[nid] = node._to_dict_shallow()
            tree_out[nid] = [c.node_id for c in node.children]

        vocab_out = {
            str(attr): list(vals)
            for attr, vals in self.attr_value_vocab.items()
        }

        stored_out = []
        for inst, path in self._stored:
            # Serialise instance: keys may be ints; convert to str
            inst_ser = {str(k): {str(vk): vc for vk, vc in vd.items()}
                        for k, vd in inst.items()}
            stored_out.append([inst_ser, path])

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "max_depth": self.max_depth,
            "n_gibbs_init": self.n_gibbs_init,
            "attr_value_vocab": vocab_out,
            "nodes": nodes_out,
            "tree": tree_out,
            "stored": stored_out,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HDP":
        """
        Reconstruct an ``HDP`` from a dict produced by ``to_dict()``.
        """
        hdp = cls(
            alpha=data["alpha"],
            beta=data["beta"],
            max_depth=data.get("max_depth"),
            n_gibbs_init=data.get("n_gibbs_init", 0),
        )

        # Rebuild vocabulary (keys stay as strings since JSON forces it)
        for attr_str, vals in data["attr_value_vocab"].items():
            # Attempt int conversion for numeric keys
            try:
                attr = int(attr_str)
            except (ValueError, TypeError):
                attr = attr_str
            hdp.attr_value_vocab[attr] = set(vals)

        # Rebuild nodes (without children yet)
        hdp._nodes = {}
        node_map: Dict[str, HDPNode] = {}
        for nid, ndata in data["nodes"].items():
            node = HDPNode(node_id=nid)
            node.count = ndata["count"]
            for attr_str, vd in ndata["av_count"].items():
                try:
                    attr = int(attr_str)
                except (ValueError, TypeError):
                    attr = attr_str
                for val_str, cnt in vd.items():
                    node.av_count[attr][val_str] = cnt
            node_map[nid] = node
            hdp._nodes[nid] = node

        # Wire up parent-child relationships
        for nid, children_ids in data["tree"].items():
            parent_node = node_map[nid]
            for cid in children_ids:
                child_node = node_map[cid]
                child_node.parent = parent_node
                child_node._depth_cache = None
                parent_node.children.append(child_node)

        hdp.root = node_map.get("ROOT", HDPNode(node_id="ROOT"))

        # Restore stored instances
        hdp._stored = []
        for inst_ser, path in data.get("stored", []):
            inst: Dict[Any, Dict[Any, int]] = {}
            for attr_str, vd in inst_ser.items():
                try:
                    attr = int(attr_str)
                except (ValueError, TypeError):
                    attr = attr_str
                inst[attr] = {vk: vc for vk, vc in vd.items()}
            hdp._stored.append((inst, path))

        return hdp

    def to_json(self, path: str):
        """Save the model to a JSON file at *path*."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "HDP":
        """Load a model from a JSON file produced by ``to_json()``."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of stored instances."""
        return len(self._stored)

    def __repr__(self) -> str:
        s = self.tree_stats()
        return (
            f"HDP(alpha={self.alpha}, beta={self.beta}, "
            f"n_nodes={s['n_nodes']}, n_leaves={s['n_leaves']}, "
            f"n_instances={s['n_instances']})"
        )
