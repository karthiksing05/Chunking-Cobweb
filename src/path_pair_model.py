"""
path_pair_model.py
==================
Streaming probabilistic model over ordered pairs of variable-length categorical
paths (nodeID sequences).

The model maintains:
  - Exponential time-decay counts (lazy evaluation).
  - Left and right marginal prefix tries.
  - Top-K joint count tables per left path.
  - A hierarchical prototype abstraction layer sharing common left/right prefixes.

All paths are stored internally as tuples.  No external dependencies beyond
the Python standard library.

Probability math
----------------
All probabilities use Laplace (additive) smoothing with hyperparameter α:

  P(L)    = (count(L)   + α) / (N_L   + α · |V_L|)
  P(R)    = (count(R)   + α) / (N_R   + α · |V_R|)
  P(L,R)  = (count(L,R) + α) / (N     + α · |V_LR|)

  PMI(L,R)  = log( P(L,R) / (P(L) · P(R)) )
  NPMI(L,R) = PMI(L,R) / -log(P(L,R))       ∈ [-1, +1]

Decay formula (applied lazily)
-------------------------------
  C_t = C_prev · exp(-λ · Δt) + 1
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Centralized decay helper
# ---------------------------------------------------------------------------

def _decay(count: float, last_update: float, lam: float, now: float) -> float:
    """Return *count* after applying exponential time decay.

    The pure decay (before adding a new increment) is:

        C_decayed = C_prev · exp(-λ · Δt)

    where Δt = now - last_update.

    Parameters
    ----------
    count:
        The previously stored count value.
    last_update:
        UNIX timestamp of when *count* was last written.
    lam:
        Decay rate λ ≥ 0.  λ = 0 disables decay entirely.
    now:
        Current UNIX timestamp.

    Returns
    -------
    float
        The decayed count, ready for a new +1 increment to be added by the caller.
    """
    if lam == 0.0:
        return count
    delta = now - last_update
    if delta <= 0.0:
        return count
    return count * math.exp(-lam * delta)


# ---------------------------------------------------------------------------
# Trie (prefix statistics)
# ---------------------------------------------------------------------------

class TrieNode:
    """A single node in the marginal prefix trie.

    Attributes
    ----------
    children:
        Mapping from path element to child TrieNode.
    count:
        Lazily-decayed cumulative count of observations that passed through
        this node.
    last_update:
        UNIX timestamp of the last write (used for lazy decay).
    """

    __slots__ = ("children", "count", "last_update")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.count: float = 0.0
        self.last_update: float = time.time()


class Trie:
    """Prefix trie for tracking decayed marginal counts of path sequences.

    Each call to :meth:`add` increments the count at every prefix node along
    the path, applying lazy decay before the increment.

    Parameters
    ----------
    lam:
        Decay rate λ forwarded to :func:`_decay`.
    """

    def __init__(self, lam: float = 0.0) -> None:
        self._root = TrieNode()
        self._lam = lam

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_decayed(self, node: TrieNode, now: float) -> float:
        """Return node's lazily-decayed count without mutating state."""
        return _decay(node.count, node.last_update, self._lam, now)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, path: Tuple[str, ...]) -> None:
        """Record one observation of *path*, incrementing all prefix nodes.

        Applies lazy decay before the +1 increment at every node.

        Time complexity: O(len(path))
        """
        now = time.time()
        node = self._root
        node.count = _decay(node.count, node.last_update, self._lam, now) + 1.0
        node.last_update = now
        for element in path:
            if element not in node.children:
                node.children[element] = TrieNode()
            node = node.children[element]
            node.count = _decay(node.count, node.last_update, self._lam, now) + 1.0
            node.last_update = now

    def get_count(self, path: Tuple[str, ...]) -> float:
        """Return the decayed count at the terminal node of *path*.

        Returns 0.0 if the path has never been added.

        Time complexity: O(len(path))
        """
        now = time.time()
        node = self._root
        for element in path:
            if element not in node.children:
                return 0.0
            node = node.children[element]
        return _decay(node.count, node.last_update, self._lam, now)

    def add_with_weight(self, path: Tuple[str, ...], weight: float, now: float) -> None:
        """Add *weight* to every prefix node along *path*, using *now* as timestamp.

        Unlike :meth:`add`, this does **not** apply lazy decay — it is intended
        for rebuilding tries from a snapshot where all counts have already been
        decayed to a common reference time *now*.

        Time complexity: O(len(path))
        """
        node = self._root
        node.count += weight
        node.last_update = now
        for element in path:
            if element not in node.children:
                node.children[element] = TrieNode()
            node = node.children[element]
            node.count += weight
            node.last_update = now

    def total_count(self) -> float:
        """Return the decayed total observation count (stored at the root)."""
        now = time.time()
        return _decay(self._root.count, self._root.last_update, self._lam, now)


# ---------------------------------------------------------------------------
# Prototype (hierarchical abstraction cluster)
# ---------------------------------------------------------------------------

def _lcp(seqs: List[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Compute the longest common prefix of a non-empty list of tuples.

    Returns an empty tuple if there is no common prefix, or if *seqs*
    is empty.
    """
    if not seqs:
        return ()
    prefix = list(seqs[0])
    for seq in seqs[1:]:
        new_len = min(len(prefix), len(seq))
        prefix = prefix[:new_len]
        for i in range(new_len):
            if prefix[i] != seq[i]:
                prefix = prefix[:i]
                break
        if not prefix:
            break
    return tuple(prefix)


class Prototype:
    """A cluster of concrete (left_path, right_path) pairs sharing a common
    left prefix and a common right prefix.

    Matching rule
    -------------
    A prototype matches an incoming pair (L, R) iff:
      - L starts with ``left_prefix``, AND
      - R starts with ``right_prefix``.

    Promotion
    ---------
    Once the number of distinct members exceeds *promotion_threshold*, the
    left/right prefixes are recomputed as the LCP of all member paths, which
    may broaden the prototype's matching region.

    Sampling
    --------
    :meth:`sample_member` performs cumulative weighted random selection
    proportional to decayed weights.

    Parameters
    ----------
    left_prefix:
        Initial left-side prefix (typically the full path of the first member).
    right_prefix:
        Initial right-side prefix.
    lam:
        Decay rate λ for member weight decay.
    """

    __slots__ = (
        "left_prefix",
        "right_prefix",
        "lam",
        "members",
        "member_last_update",
        "total_weight",
        "last_update",
    )

    def __init__(
        self,
        left_prefix: Tuple[str, ...],
        right_prefix: Tuple[str, ...],
        lam: float,
    ) -> None:
        self.left_prefix: Tuple[str, ...] = left_prefix
        self.right_prefix: Tuple[str, ...] = right_prefix
        self.lam: float = lam
        # (left_path, right_path) → stored (pre-decay) weight
        self.members: Dict[Tuple[Tuple, Tuple], float] = {}
        # (left_path, right_path) → timestamp of last write
        self.member_last_update: Dict[Tuple[Tuple, Tuple], float] = {}
        self.total_weight: float = 0.0
        self.last_update: float = time.time()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def matches(self, left: Tuple[str, ...], right: Tuple[str, ...]) -> bool:
        """Return True iff *left* starts with ``left_prefix`` and *right*
        starts with ``right_prefix``."""
        lp = self.left_prefix
        rp = self.right_prefix
        return (
            left[: len(lp)] == lp
            and right[: len(rp)] == rp
        )

    @property
    def combined_prefix_length(self) -> int:
        """Sum of left and right prefix lengths (used to rank alternative matches)."""
        return len(self.left_prefix) + len(self.right_prefix)

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _member_weight(self, key: Tuple, now: float) -> float:
        """Lazily-decayed weight for a member key."""
        return _decay(self.members[key], self.member_last_update[key], self.lam, now)

    def get_total_weight(self) -> float:
        """Recompute and cache the total decayed weight across all members."""
        now = time.time()
        total = sum(self._member_weight(k, now) for k in self.members)
        self.total_weight = total
        return total

    # ------------------------------------------------------------------
    # Observation update
    # ------------------------------------------------------------------

    def add_observation(
        self, left: Tuple[str, ...], right: Tuple[str, ...]
    ) -> None:
        """Record one observation of (left, right) into this prototype.

        Applies lazy decay to any existing weight before adding +1.
        """
        now = time.time()
        key = (left, right)
        if key in self.members:
            self.members[key] = self._member_weight(key, now) + 1.0
        else:
            self.members[key] = 1.0
        self.member_last_update[key] = now
        self.last_update = now

    def add_observation_weighted(
        self,
        left: Tuple[str, ...],
        right: Tuple[str, ...],
        weight: float,
        now: float,
    ) -> None:
        """Add *weight* for (left, right) using *now* as timestamp.

        Like :meth:`add_observation` but bypasses lazy decay — used during a
        full rebuild where counts have already been snapshotted.
        """
        key = (left, right)
        if key in self.members:
            self.members[key] += weight
        else:
            self.members[key] = weight
        self.member_last_update[key] = now
        self.last_update = now

    # ------------------------------------------------------------------
    # Promotion / re-abstraction
    # ------------------------------------------------------------------

    def recompute_prefixes(self) -> None:
        """Recompute left/right prefixes as the LCP of all current members.

        Called after the promotion threshold is exceeded; may widen the
        matching region if member paths have diverged.
        """
        if not self.members:
            return
        left_paths: List[Tuple[str, ...]] = [k[0] for k in self.members]
        right_paths: List[Tuple[str, ...]] = [k[1] for k in self.members]
        self.left_prefix = _lcp(left_paths)
        self.right_prefix = _lcp(right_paths)
        self.last_update = time.time()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_member(self) -> Optional[Tuple[Tuple, Tuple]]:
        """Sample a concrete (left_path, right_path) weighted by decayed count
        **and** proximity to the prototype's current prefixes.

        Sampling weight
        ---------------
        For each member m = (L, R)::

            extra_steps = (len(L) - len(left_prefix)) + (len(R) - len(right_prefix))
            proximity   = 1 / (1 + extra_steps)
            weight(m)   = decayed_count(m) * proximity

        Interpretation: members whose paths are the shortest extension of the
        prototype's abstract prefixes are considered closest to the prototype
        and receive a higher sampling probability relative to their raw count.

        Uses cumulative weighted random selection (O(|members|)).

        Returns None if the prototype has no members.
        Degrades to uniform sampling if all combined weights are ≤ 0.
        """
        if not self.members:
            return None
        now = time.time()
        lp_len = len(self.left_prefix)
        rp_len = len(self.right_prefix)
        keys = list(self.members.keys())
        weights: List[float] = []
        for k in keys:
            left_k, right_k = k
            decayed_w = self._member_weight(k, now)
            extra = (len(left_k) - lp_len) + (len(right_k) - rp_len)
            # extra >= 0 always (members must start with the prefix)
            proximity = 1.0 / (1.0 + extra)
            weights.append(decayed_w * proximity)
        total = sum(weights)
        if total <= 0.0:
            # All weights have decayed to ~0; fall back to uniform selection.
            return random.choice(keys)
        threshold = random.uniform(0.0, total)
        cumulative = 0.0
        for key, w in zip(keys, weights):
            cumulative += w
            if cumulative >= threshold:
                return key
        return keys[-1]  # Numerical safety fallback


# ---------------------------------------------------------------------------
# PathPairModel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Path transform helpers
# ---------------------------------------------------------------------------

def _delete_from_path(path: Tuple[str, ...], node_id: str) -> Tuple[str, ...]:
    """Return *path* with every occurrence of *node_id* removed.

    Semantics: the deleted node's children are "shifted up" and directly
    follow the deleted node's parent in the resulting path.

    Example
    -------
    >>> _delete_from_path(("root", "A", "B", "C"), "A")
    ('root', 'B', 'C')
    """
    return tuple(e for e in path if e != node_id)


def _insert_into_path(
    path: Tuple[str, ...],
    parent_id: str,
    new_node_id: str,
    children: Set[str],
) -> Tuple[str, ...]:
    """Return *path* with *new_node_id* inserted between *parent_id* and any
    immediately following element that is in *children*.

    Semantics: the new node is interposed between a parent and a subset of its
    current children, lengthening affected paths by one step.

    Example
    -------
    >>> _insert_into_path(("root", "A", "B"), "root", "X", {"A"})
    ('root', 'X', 'A', 'B')
    """
    result: List[str] = []
    for i, element in enumerate(path):
        result.append(element)
        if (
            element == parent_id
            and i + 1 < len(path)
            and path[i + 1] in children
        ):
            result.append(new_node_id)
    return tuple(result)


class PathPairModel:
    """Streaming probabilistic model over ordered pairs of variable-length
    categorical paths (nodeID sequences).

    Core data structures
    --------------------
    * **Left trie** – prefix trie of decayed marginal counts for left paths.
    * **Right trie** – same for right paths.
    * **Joint table** – ``{left_path: {right_path: (count, timestamp)}}``.
      Each left path retains at most *top_k* partners; excess entries are
      evicted at observation time by lowest decayed count.
    * **Prototype store** – ``{(left_prefix, right_prefix): Prototype}``.
      Prototypes are keyed by their current prefix pair; re-keyed on promotion.

    Parameters
    ----------
    lam:
        Exponential decay rate λ ≥ 0 (default 0.01).  Use 0 for no decay.
    top_k:
        Maximum number of right-path partners retained per left path (default 50).
    alpha:
        Laplace smoothing pseudo-count α > 0 (default 0.1).
    promotion_threshold:
        Distinct members needed before a prototype's prefixes are re-abstracted
        via LCP (default 5).
    """

    def __init__(
        self,
        lam: float = 0.01,
        top_k: int = 50,
        alpha: float = 0.1,
        promotion_threshold: int = 5,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.lam = lam
        self.top_k = top_k
        self.alpha = alpha
        self.promotion_threshold = promotion_threshold

        # Marginal tries
        self._left_trie: Trie = Trie(lam=lam)
        self._right_trie: Trie = Trie(lam=lam)

        # Joint table: left_path -> {right_path -> (count, last_update)}
        self._joint: Dict[
            Tuple[str, ...], Dict[Tuple[str, ...], Tuple[float, float]]
        ] = {}

        # Prototype store: (left_prefix, right_prefix) -> Prototype
        self._prototypes: Dict[
            Tuple[Tuple[str, ...], Tuple[str, ...]], Prototype
        ] = {}

    # ------------------------------------------------------------------
    # Conversion helper
    # ------------------------------------------------------------------

    @staticmethod
    def _t(path: List[str]) -> Tuple[str, ...]:
        """Convert a list-form path to its canonical tuple representation."""
        return tuple(path)

    # ------------------------------------------------------------------
    # Joint count access
    # ------------------------------------------------------------------

    def _read_joint(
        self, left: Tuple[str, ...], right: Tuple[str, ...], now: float
    ) -> float:
        """Return the lazily-decayed joint count for (left, right).

        Returns 0.0 if the pair has never been observed or was evicted from
        the top-K table.
        """
        partners = self._joint.get(left)
        if partners is None:
            return 0.0
        entry = partners.get(right)
        if entry is None:
            return 0.0
        raw, ts = entry
        return _decay(raw, ts, self.lam, now)

    def _write_joint(
        self, left: Tuple[str, ...], right: Tuple[str, ...], now: float
    ) -> None:
        """Increment the joint count for (left, right) by 1.

        Applies lazy decay to the existing value first.  If this would push
        the partner set beyond *top_k*, the entry with the smallest decayed
        count is evicted.

        Time complexity: O(K) in the worst case (linear eviction scan).
        """
        if left not in self._joint:
            self._joint[left] = {}
        partners = self._joint[left]

        if right in partners:
            raw, ts = partners[right]
            new_count = _decay(raw, ts, self.lam, now) + 1.0
            partners[right] = (new_count, now)
        else:
            # Need room for the new entry
            if len(partners) >= self.top_k:
                # Evict the partner with the smallest current decayed count.
                # This is O(K) but K is bounded and small relative to typical
                # path cardinalities.
                min_right = min(
                    partners,
                    key=lambda r: _decay(
                        partners[r][0], partners[r][1], self.lam, now
                    ),
                )
                del partners[min_right]
            partners[right] = (1.0, now)

    # ------------------------------------------------------------------
    # Prototype routing
    # ------------------------------------------------------------------

    def _find_best_prototype(
        self,
        left: Tuple[str, ...],
        right: Tuple[str, ...],
    ) -> Optional[Prototype]:
        """Return the prototype with the longest combined prefix matching (left, right).

        Among all prototypes P where left starts with P.left_prefix AND
        right starts with P.right_prefix, choose the one maximising
        len(P.left_prefix) + len(P.right_prefix).

        Returns None if no prototype matches.

        Time complexity: O(|prototypes|).  In practice the prototype set grows
        sublinearly because pairs sharing structural resemblance are merged.
        """
        best: Optional[Prototype] = None
        best_len: int = -1
        for proto in self._prototypes.values():
            if proto.matches(left, right):
                cpl = proto.combined_prefix_length
                if cpl > best_len:
                    best_len = cpl
                    best = proto
        return best

    def _route_to_prototype(
        self,
        left: Tuple[str, ...],
        right: Tuple[str, ...],
    ) -> None:
        """Route the observed pair to the best matching prototype.

        If no prototype matches, a new one is created with full path prefixes.
        After adding the observation, checks for promotion and re-keys the
        prototype dictionary if prefixes change.
        """
        proto = self._find_best_prototype(left, right)
        if proto is None:
            # Bootstrap with full paths as prefixes
            init_key: Tuple[Tuple, Tuple] = (left, right)
            if init_key not in self._prototypes:
                self._prototypes[init_key] = Prototype(left, right, self.lam)
            proto = self._prototypes[init_key]

        old_key: Tuple[Tuple, Tuple] = (proto.left_prefix, proto.right_prefix)
        proto.add_observation(left, right)

        # Check promotion threshold
        if len(proto.members) > self.promotion_threshold:
            proto.recompute_prefixes()
            new_key: Tuple[Tuple, Tuple] = (proto.left_prefix, proto.right_prefix)
            if new_key != old_key:
                # Re-key: remove old entry, insert under new key.
                self._prototypes.pop(old_key, None)
                self._prototypes[new_key] = proto
        else:
            # Ensure the prototype is reachable under its current key.
            if old_key not in self._prototypes:
                self._prototypes[old_key] = proto

    # ------------------------------------------------------------------
    # Internal rebuild (used by path-update operations)
    # ------------------------------------------------------------------

    def _snapshot_joint(self) -> List[Tuple[Tuple[str, ...], Tuple[str, ...], float]]:
        """Snapshot every (left, right, decayed_count) entry from the joint table.

        Returns a list of triples with counts decayed to *now*.
        Entries whose decayed count rounds to ≤ 0 are still included so that
        structural transforms don't silently drop pairs.
        """
        now = time.time()
        entries: List[Tuple[Tuple[str, ...], Tuple[str, ...], float]] = []
        for left, partners in self._joint.items():
            for right, (raw, ts) in partners.items():
                decayed = _decay(raw, ts, self.lam, now)
                entries.append((left, right, max(decayed, 0.0)))
        return entries

    def _rebuild_all(
        self,
        entries: List[Tuple[Tuple[str, ...], Tuple[str, ...], float]],
    ) -> None:
        """Rebuild every internal structure from a flat list of weighted pairs.

        Parameters
        ----------
        entries:
            Each element is ``(left_path, right_path, weight)``.
            Duplicate (left, right) keys are merged (weights summed).

        Effects
        -------
        * Left & right marginal tries are recreated from scratch.
        * Joint table is repopulated (top-K eviction applied normally).
        * Prototype store is rebuilt via the standard routing logic.

        Complexity: O(N · (max_path_len + K + |prototypes|))
        where N = len(entries).

        Note
        ----
        All rebuilt counts share a single ``now`` timestamp, which "snaps"
        the decay clock.  This is an unavoidable side-effect of structural
        mutations — the relative magnitudes of decayed counts are preserved,
        but the per-entry timestamps are unified.  Subsequent streaming
        updates continue lazily from that point.
        """
        now = time.time()

        # --- Merge duplicates ------------------------------------------------
        merged: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float] = {}
        for left, right, weight in entries:
            key = (left, right)
            merged[key] = merged.get(key, 0.0) + weight

        # --- Fresh tries -----------------------------------------------------
        self._left_trie = Trie(lam=self.lam)
        self._right_trie = Trie(lam=self.lam)

        # --- Fresh joint table -----------------------------------------------
        self._joint = {}

        # --- Fresh prototypes ------------------------------------------------
        self._prototypes = {}

        # --- Repopulate ------------------------------------------------------
        for (left, right), weight in merged.items():
            if weight <= 0.0:
                continue
            # Tries (additive, not +1)
            self._left_trie.add_with_weight(left, weight, now)
            self._right_trie.add_with_weight(right, weight, now)

            # Joint table — store directly (no decay needed, same timestamp)
            if left not in self._joint:
                self._joint[left] = {}
            partners = self._joint[left]
            if right in partners:
                old_w, _ = partners[right]
                partners[right] = (old_w + weight, now)
            else:
                if len(partners) >= self.top_k:
                    min_right = min(partners, key=lambda r: partners[r][0])
                    if partners[min_right][0] < weight:
                        del partners[min_right]
                        partners[right] = (weight, now)
                    # else: drop this entry (smaller than everything in table)
                else:
                    partners[right] = (weight, now)

            # Prototypes — weighted routing
            proto = self._find_best_prototype(left, right)
            if proto is None:
                init_key: Tuple[Tuple, Tuple] = (left, right)
                if init_key not in self._prototypes:
                    self._prototypes[init_key] = Prototype(left, right, self.lam)
                proto = self._prototypes[init_key]

            old_key: Tuple[Tuple, Tuple] = (proto.left_prefix, proto.right_prefix)
            proto.add_observation_weighted(left, right, weight, now)

            if len(proto.members) > self.promotion_threshold:
                proto.recompute_prefixes()
                new_key: Tuple[Tuple, Tuple] = (proto.left_prefix, proto.right_prefix)
                if new_key != old_key:
                    self._prototypes.pop(old_key, None)
                    self._prototypes[new_key] = proto
            else:
                if old_key not in self._prototypes:
                    self._prototypes[old_key] = proto

    # ------------------------------------------------------------------
    # Path-update operations
    # ------------------------------------------------------------------

    def delete_node(self, node_id: str) -> int:
        """Remove *node_id* from every stored path, shifting descendants up.

        When a node is deleted from a hierarchy, any path containing that node
        has it spliced out — the node's children become direct children of the
        node's parent in each affected path.

        Example
        -------
        If the model contains the path ``["root", "A", "B", "C"]`` and
        ``delete_node("A")`` is called, the path becomes
        ``["root", "B", "C"]``.

        Effects
        -------
        * All internal structures (tries, joint table, prototypes) are rebuilt
          after applying the transform.
        * If two formerly-distinct paths collapse to the same path after
          deletion, their decayed counts are summed.
        * Vocabulary sizes may shrink.

        Complexity: O(N · (max_path_len + K)) — a full rebuild.

        Parameters
        ----------
        node_id:
            The node identifier to remove from all paths.

        Returns
        -------
        int
            Number of (left, right) entries whose paths were modified.
        """
        snapshot = self._snapshot_joint()
        affected = 0
        transformed: List[Tuple[Tuple[str, ...], Tuple[str, ...], float]] = []
        for left, right, weight in snapshot:
            new_left = _delete_from_path(left, node_id)
            new_right = _delete_from_path(right, node_id)
            if new_left != left or new_right != right:
                affected += 1
            # Skip entries that collapse to empty paths
            if new_left and new_right:
                transformed.append((new_left, new_right, weight))
        self._rebuild_all(transformed)
        return affected

    def insert_node(
        self,
        parent_id: str,
        new_node_id: str,
        children: List[str],
    ) -> int:
        """Insert *new_node_id* between *parent_id* and its specified *children*
        in every stored path.

        When a new node is added to a hierarchy, it is interposed between an
        existing parent and a subset of that parent's current children.  Any
        path that transitions from *parent_id* directly to one of *children*
        gains *new_node_id* in between.

        Example
        -------
        If the model contains ``["root", "A", "B"]`` and
        ``insert_node("root", "X", ["A"])`` is called, the path becomes
        ``["root", "X", "A", "B"]``.

        Effects
        -------
        * All internal structures are rebuilt after applying the transform.
        * No merging is expected (insertion always lengthens paths), but if
          duplicates arise they are summed.

        Complexity: O(N · (max_path_len + K)) — a full rebuild.

        Parameters
        ----------
        parent_id:
            Node ID of the existing parent under which *new_node_id* is inserted.
        new_node_id:
            The new node to insert.
        children:
            List of existing child node IDs that should become children of
            *new_node_id* (rather than direct children of *parent_id*).

        Returns
        -------
        int
            Number of (left, right) entries whose paths were modified.
        """
        child_set: Set[str] = set(children)
        snapshot = self._snapshot_joint()
        affected = 0
        transformed: List[Tuple[Tuple[str, ...], Tuple[str, ...], float]] = []
        for left, right, weight in snapshot:
            new_left = _insert_into_path(left, parent_id, new_node_id, child_set)
            new_right = _insert_into_path(right, parent_id, new_node_id, child_set)
            if new_left != left or new_right != right:
                affected += 1
            transformed.append((new_left, new_right, weight))
        self._rebuild_all(transformed)
        return affected

    # ------------------------------------------------------------------
    # Core observation
    # ------------------------------------------------------------------

    def observe_pair(
        self, left_path: List[str], right_path: List[str]
    ) -> None:
        """Record one observation of the ordered pair (left_path, right_path).

        Updates
        -------
        * Left marginal trie.
        * Right marginal trie.
        * Joint count table (with top-K eviction if necessary).
        * Prototype abstraction layer.

        Time complexity: amortized O(|L| + |R| + K)
        where K is the top-K cap.

        Parameters
        ----------
        left_path:
            Sequence of node IDs for the left path.
        right_path:
            Sequence of node IDs for the right path.
        """
        left = self._t(left_path)
        right = self._t(right_path)
        now = time.time()

        self._left_trie.add(left)
        self._right_trie.add(right)
        self._write_joint(left, right, now)
        self._route_to_prototype(left, right)

    # ------------------------------------------------------------------
    # Vocabulary helpers (vocabulary sizes for Laplace smoothing)
    # ------------------------------------------------------------------

    def _vocab_left(self) -> int:
        """Number of distinct left paths ever observed (|V_L|)."""
        return max(len(self._joint), 1)

    def _vocab_right(self) -> int:
        """Number of distinct right paths across all left partners (|V_R|).

        Computed by walking at most K entries per left path once.
        Complexity: O(|left_vocab| · K).
        """
        seen: Set[Tuple[str, ...]] = set()
        for partners in self._joint.values():
            seen.update(partners.keys())
        return max(len(seen), 1)

    def _vocab_joint(self) -> int:
        """Number of distinct (left, right) pairs stored (|V_LR|)."""
        return max(sum(len(p) for p in self._joint.values()), 1)

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------

    def p_marginal_left(self, left_path: List[str]) -> float:
        """Laplace-smoothed marginal probability P(L).

        P(L) = (count(L) + α) / (N_L + α · |V_L|)

        where N_L is the total left observation count and |V_L| the number
        of distinct left paths.
        """
        left = self._t(left_path)
        c = self._left_trie.get_count(left)
        total = self._left_trie.total_count()
        v = self._vocab_left()
        return (c + self.alpha) / (total + self.alpha * v)

    def p_marginal_right(self, right_path: List[str]) -> float:
        """Laplace-smoothed marginal probability P(R).

        P(R) = (count(R) + α) / (N_R + α · |V_R|)
        """
        right = self._t(right_path)
        c = self._right_trie.get_count(right)
        total = self._right_trie.total_count()
        v = self._vocab_right()
        return (c + self.alpha) / (total + self.alpha * v)

    def p_joint(self, left_path: List[str], right_path: List[str]) -> float:
        """Laplace-smoothed joint probability P(L, R).

        P(L,R) = (count(L,R) + α) / (N + α · |V_LR|)

        where N is the total number of observations (equal to N_L == N_R since
        every call to observe_pair increments both) and |V_LR| the number of
        distinct observed pairs.

        Note
        ----
        count(L,R) may be 0 if the pair was never observed *or* if it was
        evicted from the top-K table.  In both cases Laplace smoothing yields
        a small but non-zero probability, which is intentional.
        """
        left = self._t(left_path)
        right = self._t(right_path)
        now = time.time()
        c = self._read_joint(left, right, now)
        total = self._left_trie.total_count()
        v = self._vocab_joint()
        return (c + self.alpha) / (total + self.alpha * v)

    def pmi(self, left_path: List[str], right_path: List[str]) -> float:
        """Pointwise Mutual Information.

        PMI(L, R) = log( P(L,R) / (P(L) · P(R)) )

        Positive → positive association; negative → repulsion.
        Returns ``-math.inf`` for degenerate cases (should not occur with
        Laplace smoothing, but guarded defensively).
        """
        p_lr = self.p_joint(left_path, right_path)
        p_l = self.p_marginal_left(left_path)
        p_r = self.p_marginal_right(right_path)
        denom = p_l * p_r
        if denom <= 0.0 or p_lr <= 0.0:
            return -math.inf
        return math.log(p_lr / denom)

    def npmi(self, left_path: List[str], right_path: List[str]) -> float:
        """Normalized Pointwise Mutual Information.

        NPMI(L, R) = PMI(L, R) / -log( P(L, R) )   ∈ [-1, +1]

        Interpretation
        --------------
          +1 → perfect co-occurrence (L and R always appear together).
           0 → statistical independence.
          -1 → mutual exclusion (L and R never co-occur).

        Returns -1.0 for zero joint probability (pair never or rarely seen)
        and 0.0 for the degenerate case where P(L,R) = 1.
        """
        raw_pmi = self.pmi(left_path, right_path)
        if raw_pmi == -math.inf:
            return -1.0
        p_lr = self.p_joint(left_path, right_path)
        if p_lr <= 0.0:
            return -1.0
        normalizer = -math.log(p_lr)
        if normalizer == 0.0:
            # P(L,R) == 1.0 → degenerate; normalizer vanishes.
            return 0.0
        return raw_pmi / normalizer

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_right_candidates(
        self, left_path: List[str]
    ) -> List[Tuple[float, List[str]]]:
        """Rank retained right-path partners for *left_path* by NPMI.

        Only the right paths currently in the top-K table for *left_path*
        are considered (i.e., evicted entries are not re-scored).

        Parameters
        ----------
        left_path:
            The left path whose partners should be ranked.

        Returns
        -------
        List[Tuple[float, List[str]]]
            List of ``(npmi_score, right_path_as_list)`` sorted in descending
            order of NPMI.  Empty list if *left_path* has no recorded partners.
        """
        left = self._t(left_path)
        partners = self._joint.get(left)
        if not partners:
            return []
        results: List[Tuple[float, List[str]]] = []
        for right_tuple in partners:
            score = self.npmi(left_path, list(right_tuple))
            results.append((score, list(right_tuple)))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Prototype interface
    # ------------------------------------------------------------------

    def get_prototype(
        self, left_path: List[str], right_path: List[str]
    ) -> Optional[Prototype]:
        """Return the best-matching Prototype for (left_path, right_path).

        Uses the longest combined-prefix matching rule: among all prototypes
        P where left_path starts with P.left_prefix AND right_path starts with
        P.right_prefix, return the one with the greatest
        len(P.left_prefix) + len(P.right_prefix).

        Returns None if no prototype matches the pair.
        """
        left = self._t(left_path)
        right = self._t(right_path)
        return self._find_best_prototype(left, right)

    def sample_exemplar_from_prototype(
        self, left_path: List[str], right_path: List[str]
    ) -> Optional[Tuple[List[str], List[str]]]:
        """Sample a concrete (left, right) exemplar from the best matching prototype.

        Steps
        -----
        1. Find the best matching prototype using the longest-prefix rule.
        2. Perform cumulative weighted random selection over prototype members
           proportional to their decayed weights.

        Parameters
        ----------
        left_path:
            Left path used to identify the matching prototype.
        right_path:
            Right path used to identify the matching prototype.

        Returns
        -------
        Optional[Tuple[List[str], List[str]]]
            A sampled ``(left_path_list, right_path_list)`` pair, or None if
            no prototype matches the query pair.
        """
        proto = self.get_prototype(left_path, right_path)
        if proto is None:
            return None
        sampled = proto.sample_member()
        if sampled is None:
            return None
        left_t, right_t = sampled
        return list(left_t), list(right_t)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PathPairModel — streaming probabilistic path-pair model")
    print("=" * 60)

    model = PathPairModel(
        lam=0.001,
        top_k=20,
        alpha=0.1,
        promotion_threshold=3,
    )

    # ------------------------------------------------------------------
    # 1. Build a prototype that undergoes genuine prefix abstraction
    # ------------------------------------------------------------------
    print("\n--- Phase 1: observe seed pair ---")
    model.observe_pair(["animal"], ["behaviour"])
    print("  observed  ['animal']  ->  ['behaviour']")

    print("\n--- Phase 2: observe depth-1 extensions (trigger promotion) ---")
    depth1_pairs = [
        (["animal", "mammal"],  ["behaviour", "social"]),
        (["animal", "bird"],    ["behaviour", "migration"]),
        (["animal", "reptile"], ["behaviour", "thermoregulation"]),
        (["animal", "fish"],    ["behaviour", "schooling"]),
    ]
    for lp, rp in depth1_pairs:
        model.observe_pair(lp, rp)
        print(f"  observed  {lp}  ->  {rp}")

    print("\n--- Phase 3: observe depth-2 extensions ---")
    depth2_pairs = [
        (["animal", "mammal",  "primate"],   ["behaviour", "social",    "grooming"]),
        (["animal", "mammal",  "carnivore"],  ["behaviour", "social",    "pack_hunt"]),
        (["animal", "bird",    "raptor"],     ["behaviour", "migration", "soaring"]),
        (["animal", "reptile", "crocodile"],  ["behaviour", "thermoregulation", "basking"]),
    ]
    for lp, rp in depth2_pairs:
        model.observe_pair(lp, rp)
        print(f"  observed  {lp}  ->  {rp}")

    # Repeat seed + one depth-1 pair for higher counts
    for _ in range(3):
        model.observe_pair(["animal"], ["behaviour"])
        model.observe_pair(["animal", "mammal"], ["behaviour", "social"])

    # Unrelated cluster
    print("\n--- Phase 4: unrelated cluster ---")
    for lp, rp in [
        (["vehicle", "car"],   ["action", "drive"]),
        (["vehicle", "plane"], ["action", "fly"]),
    ]:
        model.observe_pair(lp, rp)
        print(f"  observed  {lp}  ->  {rp}")

    # ------------------------------------------------------------------
    # 2. Inspect the promoted prototype
    # ------------------------------------------------------------------
    def _print_prototype(label: str, proto: Optional[Prototype]) -> None:
        """Helper to pretty-print a prototype."""
        print(f"\n--- {label} ---")
        if proto is None:
            print("  No matching prototype found.")
            return
        print(f"  left_prefix   = {proto.left_prefix}")
        print(f"  right_prefix  = {proto.right_prefix}")
        print(f"  # members     = {len(proto.members)}")
        print(f"  total_weight  = {proto.get_total_weight():.3f}")
        lp_len = len(proto.left_prefix)
        rp_len = len(proto.right_prefix)
        print(f"  {'Member (L, R)':<64}  {'wt':>6}  {'extra':>5}  {'prox':>5}")
        print("  " + "-" * 85)
        for (ml, mr), wraw in sorted(
            proto.members.items(),
            key=lambda kv: (len(kv[0][0]) + len(kv[0][1])),
        ):
            extra = (len(ml) - lp_len) + (len(mr) - rp_len)
            prox  = 1.0 / (1.0 + extra)
            label_m = f"({list(ml)}, {list(mr)})"
            print(f"  {label_m:<64}  {wraw:>6.2f}  {extra:>5d}  {prox:>5.3f}")

    proto = model.get_prototype(["animal", "mammal"], ["behaviour", "social"])
    _print_prototype("Prototype BEFORE any path update", proto)

    # ------------------------------------------------------------------
    # 3. DELETE NODE demo
    # ------------------------------------------------------------------
    # Remove the intermediate "mammal" node.  Any path containing "mammal"
    # has that element spliced out, so e.g.
    #   ["animal", "mammal", "primate"] → ["animal", "primate"]
    #   ["animal", "mammal"]            → ["animal"]
    # The latter merges with the existing seed pair ["animal"] → ["behaviour"]
    # (counts summed).
    print("\n" + "=" * 60)
    print("DELETE NODE: removing 'mammal' from all paths")
    print("=" * 60)
    n_del = model.delete_node("mammal")
    print(f"  Affected entries: {n_del}")

    proto = model.get_prototype(["animal", "primate"], ["behaviour", "social", "grooming"])
    _print_prototype("Prototype AFTER delete_node('mammal')", proto)

    # Show that formerly-separate paths merged
    print("\n--- Joint table entries for left=('animal',) ---")
    left_key = ("animal",)
    partners = model._joint.get(left_key, {})
    for rk, (w, ts) in sorted(partners.items(), key=lambda x: -x[1][0]):
        print(f"  {str(rk):<50}  count={w:.2f}")

    # ------------------------------------------------------------------
    # 4. INSERT NODE demo
    # ------------------------------------------------------------------
    # Insert a new "vertebrate" node between "animal" and its children
    # ["bird", "reptile", "fish", "primate", "carnivore"].
    # Paths like ["animal", "bird", ...] become ["animal", "vertebrate", "bird", ...].
    print("\n" + "=" * 60)
    print("INSERT NODE: adding 'vertebrate' between 'animal' and its children")
    print("=" * 60)
    n_ins = model.insert_node(
        parent_id="animal",
        new_node_id="vertebrate",
        children=["bird", "reptile", "fish", "primate", "carnivore"],
    )
    print(f"  Affected entries: {n_ins}")

    proto = model.get_prototype(
        ["animal", "vertebrate", "bird"], ["behaviour", "migration"]
    )
    _print_prototype("Prototype AFTER insert_node('vertebrate')", proto)

    # ------------------------------------------------------------------
    # 5. NPMI and ranking after structural changes
    # ------------------------------------------------------------------
    print("\n--- NPMI scores after both updates (left = ['animal','vertebrate','bird']) ---")
    left_query = ["animal", "vertebrate", "bird"]
    for right_query in [
        ["behaviour", "migration"],
        ["behaviour", "migration", "soaring"],
        ["behaviour", "social"],
        ["action", "drive"],
    ]:
        score = model.npmi(left_query, right_query)
        print(f"  NPMI({left_query}, {right_query}) = {score:+.4f}")

    print(f"\n--- Ranked right candidates for {left_query} ---")
    ranked = model.rank_right_candidates(left_query)
    for score, rp in ranked:
        print(f"  npmi={score:+.4f}  right={rp}")

    # ------------------------------------------------------------------
    # 6. Proximity-weighted sampling after structural changes
    # ------------------------------------------------------------------
    print("\n--- 30 proximity-weighted exemplar samples from updated prototype ---")
    tally: Dict[str, int] = {}
    for _ in range(30):
        result = model.sample_exemplar_from_prototype(
            ["animal", "vertebrate", "bird"], ["behaviour", "migration"]
        )
        if result is not None:
            label = f"({result[0]}, {result[1]})"
            tally[label] = tally.get(label, 0) + 1
    for label, count in sorted(tally.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"  {count:>3}x  {bar:<32}  {label}")

    # ------------------------------------------------------------------
    # 7. Probabilities for a merged pair
    # ------------------------------------------------------------------
    print("\n--- P/PMI/NPMI for the seed pair (counts merged after deletion) ---")
    lp, rp = ["animal"], ["behaviour"]
    print(f"  P(L)    = {model.p_marginal_left(lp):.6f}")
    print(f"  P(R)    = {model.p_marginal_right(rp):.6f}")
    print(f"  P(L,R)  = {model.p_joint(lp, rp):.6f}")
    print(f"  PMI     = {model.pmi(lp, rp):+.4f}")
    print(f"  NPMI    = {model.npmi(lp, rp):+.4f}")
