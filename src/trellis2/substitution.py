"""One memory, read three ways, packaged as a small learner.

The memory holds (1) the stored experiences (token lists, and which experience directly
followed which), (2) two trees over token types, one by what stands after each type and one by
what stands before it (SideTrees), and (3) counts of what followed what (n-gram tables).
Learning is one pass, one experience at a time; nothing is rebuilt.

Generation retrieves the stored experience that followed the stored experience most like the
prompt, then replaces one token of it. A replacement is offered where two readings of the
trees agree: the frame vote (types that stood between the same neighbours as the token) and
the pair licences (the new neighbour pairs, when not stored, must be licensed by the trees).
The slot with the largest ratio of offer to the standing token's own vote is changed.

Coding is a Witten-Bell n-gram code whose back-off, at the order-2 step, borrows from the trees:
half the lower-order probability and half the licence's distribution over the next token.

Hand-set: `order` (n-gram order) and the half-and-half mix in the coder.
"""
import math
import random
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Optional, Sequence

from trellis2.type_trees import SideTrees

BOS, EOS = "<s>", "</s>"


class SubstitutionLearner:
    def __init__(self, seed: int = 0, order: int = 3, refit: str = "set"):
        self.seed, self.order = seed, order
        self.trees = SideTrees(seed, refit=refit)
        self._seqs: List[list] = []
        self._pairs_stored: List[tuple] = []          # (previous experience, next experience)
        self._pair_sets: List[set] = []               # set of the previous experience's tokens
        self._df: Counter = Counter()                 # experiences containing each token
        self._adj: set = set()                        # adjacent pairs, boundary marks included
        self._F_lr: Dict[tuple, set] = defaultdict(set)
        self._F_wide: Dict[tuple, set] = defaultdict(set)
        self._counts = [defaultdict(Counter) for _ in range(order)]   # context -> next counts
        self._seen: set = set()
        self._clear_caches()

    def _clear_caches(self) -> None:
        self._idf: Optional[dict] = None
        self._V: dict = {}          # (vote mode, lr key, wide key) -> candidate scores
        self._va: dict = {}         # (table, key) -> votes_all
        self._lic: dict = {}        # (a, b) -> licence
        self._nd: dict = {}         # prev token -> next_distribution

    # ------------------------------------------------------------------ learning
    def learn(self, seq: Sequence[Hashable], follows: bool = False) -> None:
        """One experience. follows=True: it directly followed the previously learned one."""
        s = list(seq)
        if follows and self._seqs:
            self._pairs_stored.append((self._seqs[-1], s))
            self._pair_sets.append(set(self._seqs[-1]))
        self._seqs.append(s)
        self.trees.observe(s)
        self._df.update(set(s))
        t = [BOS, BOS] + s + [EOS, EOS]               # token i sits at t[i + 2]
        for i in range(len(s)):
            self._F_lr[(t[i + 1], t[i + 3])].add(s[i])
            self._F_wide[(t[i], t[i + 1], t[i + 3], t[i + 4])].add(s[i])
        u = [BOS] + s + [EOS]
        self._adj.update(zip(u, u[1:]))
        seq_ = [BOS] * (self.order - 1) + s + [EOS]
        for i in range(self.order - 1, len(seq_)):
            for k in range(self.order):
                self._counts[k][tuple(seq_[i - k:i])][seq_[i]] += 1
        self._seen.update(s)
        self._clear_caches()

    # ---------------------------------------------------------------- generation
    def _retrieve(self, prev, rng) -> list:
        if self._pairs_stored and prev is not None:
            if self._idf is None:
                n = len(self._seqs)
                self._idf = {w: math.log(n / c) for w, c in self._df.items()}
            ps = set(prev)
            sc = [sum(self._idf.get(w, 0.0) for w in ps & a) + 1e-6 * rng.random() for a in self._pair_sets]
            return list(self._pairs_stored[max(range(len(sc)), key=sc.__getitem__)][1])
        return list(rng.choice(self._seqs))

    def _votes(self, vote: str, k_lr: tuple, k_w: tuple) -> dict:
        V = self._V.get((vote, k_lr, k_w))
        if V is None:
            V = defaultdict(float)
            for key, F, tab in ((k_lr, self._F_lr, "lr"), (k_w, self._F_wide, "w")):
                if (tab, key) not in self._va:
                    self._va[(tab, key)] = self.trees.votes_all(F[key]) if key in F else {}
                for y, v in self._va[(tab, key)].items():
                    V[y] += v
            if vote == "product":
                a_, b_ = self._va[("lr", k_lr)], self._va[("w", k_w)]
                V = {y: v * b_[y] for y, v in a_.items() if y in b_}
            self._V[(vote, k_lr, k_w)] = V
        return V

    def _once(self, s: list, rng, require_new: bool, vote: str, min_len, max_len) -> Optional[list]:
        t = [BOS, BOS] + s + [EOS, EOS]
        offers = []
        for i in range(len(s)):
            if len(s) <= 1:                           # replacing the only token leaves nothing standing
                break
            own = s[i]
            l, r = t[i + 1], t[i + 3]
            V = self._votes(vote, (l, r), (t[i], l, r, t[i + 4]))
            ov = V.get(own, 0.0)
            if ov <= 0:
                continue
            best, cand = 0.0, []
            for y, v in V.items():
                if y == own or (min_len is not None and len(s) < min_len) or (max_len is not None and len(s) > max_len):
                    continue
                sc_, unstored, ok = v, 0, True
                for x_, z_ in ((l, y), (y, r)):
                    if (x_, z_) in self._adj:
                        continue
                    unstored += 1
                    if x_ == BOS or z_ == EOS:
                        ok = False
                        break
                    f = self._lic.get((x_, z_))
                    if f is None:
                        f = self._lic[(x_, z_)] = self.trees.licence(x_, z_)
                    if f <= 0:
                        ok = False
                        break
                    sc_ *= f
                if not ok or (require_new and not unstored):
                    continue
                if sc_ > best * (1 - 1e-12):
                    if sc_ > best:
                        best = sc_
                    cand.append((sc_, y))
            if not cand:
                continue
            tied = sorted(y for sc_, y in cand if sc_ >= best * (1 - 1e-12))
            offers.append((i, rng.choice(tied), best / ov))
        if not offers:
            return None
        offers.sort(key=lambda o: (o[0], o[1]))
        top = max(o[2] for o in offers)
        i, y, _ = rng.choice([o for o in offers if o[2] >= top * (1 - 1e-12)])
        return s[:i] + [y] + s[i + 1:]

    def generate(self, prev=None, rng: Optional[random.Random] = None, require_new: bool = False,
                 vote: str = "sum", min_len=None, max_len=None) -> Optional[list]:
        """Retrieve, then replace one token; retried (fresh retrieval) up to 200 times. None if
        the memory is empty or no try finds a slot with an offer."""
        rng = rng or random.Random(self.seed)
        if not self._seqs:
            return None
        for _ in range(200):
            x = self._once(self._retrieve(prev, rng), rng, require_new, vote, min_len, max_len)
            if x:
                return x
        return None

    # -------------------------------------------------------------------- coding
    def _q(self, prev, t) -> Optional[float]:
        if prev not in self._seen:                    # boundaries and unknown tokens
            return None
        if prev not in self._nd:
            self._nd[prev] = self.trees.next_distribution(prev)
        d = self._nd[prev]
        return d.get(t, 0.0) if d else None

    def _p(self, ctx: list, t, V: int) -> float:
        pr = 1.0 / V
        for k in range(self.order):                   # Witten-Bell, low to high order
            if k == 1:
                q = self._q(ctx[-1], t)
                if q is not None:
                    pr = 0.5 * pr + 0.5 * q
            c = self._counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
            if not c:
                continue
            tot, types = sum(c.values()), len(c)
            pr = (c.get(t, 0) + types * pr) / (tot + types)
        return pr

    def bits(self, seq: Sequence[Hashable], escape: float = 0.0, vocab_size: Optional[int] = None) -> float:
        """Code length of seq in bits; `escape` is added for each token never learned.
        vocab_size overrides V (default: token types learned + 1)."""
        V = vocab_size if vocab_size is not None else len(self._seen) + 1
        s = list(seq)
        u = [BOS] * (self.order - 1) + s + [EOS]
        n = self.order
        return sum(-math.log2(self._p(u[i - n + 1:i], u[i], V)) for i in range(n - 1, len(u))) \
            + sum(escape for t in s if t not in self._seen)
