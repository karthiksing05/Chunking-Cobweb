"""Inside-outside parsing over the factored grammar.

A sentence is one binary tree, or, when the grammar cannot derive it whole,
a forest of two or more top-level pieces (a partial analysis). The inside
pass fills, for every span, the probability that each symbol derives it
("the frontier of valid parses going up"). A forward-backward pass over the
top level sums over the whole tree and every way of cutting the sentence
into pieces. A top-down pass then gives posteriors: ``mu[i, j, A]`` is the probability that
span (i, j) is a chunk of category A given the whole sentence, i.e. given
its content (inside) and all of its context (outside) at once. The
minimum-Bayes-risk decoder picks the binary tree with the largest expected
number of correct spans ("the best non-intersecting set going down").

Inside vectors are stored normalised with a per-span log scale, so long
sentences do not underflow. The factorisation
    P(A -> B C) = sum_c U[A,c] (1 - pk[c]) Lt[c,B] Rt[c,C]
keeps a span's work at O(n * M) instead of O(n * K^3).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from .data import Span, Tree
from .grammar import Grammar
from .memory import BOS


class Chart:
    def __init__(self, grammar: Grammar, tokens: Sequence[str]):
        g = grammar
        self.g = g
        self.tokens = list(tokens)
        ids = g.token_ids(tokens)
        n = self.n = len(ids)
        K, M = g.K, g.M
        a = self.a = np.zeros((n + 1, n + 1, K))
        la = self.la = np.full((n + 1, n + 1), -np.inf)
        lam = self.lam = np.zeros((n + 1, n + 1, M))
        rho = self.rho = np.zeros((n + 1, n + 1, M))
        # A span's rule choice is made in the light of the word before it.
        self.context = [self.tokens[i - 1] if i else BOS for i in range(n)]
        by_context = {x: g.rules(x) for x in set(self.context)}
        U = self.U = [by_context[x] for x in self.context]
        for i in range(n):
            v = g.lexical(int(ids[i]), self.tokens[i - 1] if i else BOS)
            s = v.sum()
            if s > 0:
                a[i, i + 1] = v / s
                la[i, i + 1] = np.log(s)
                lam[i, i + 1] = g.Lt @ a[i, i + 1]
                rho[i, i + 1] = g.Rt @ a[i, i + 1]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                scale = la[i, i + 1:j] + la[i + 1:j, j]
                top = scale.max()
                if not np.isfinite(top):
                    continue
                pair = lam[i, i + 1:j] * rho[i + 1:j, j]          # (splits, M)
                gamma = g.qk * (np.exp(scale - top) @ pair)        # (M,)
                v = U[i] @ gamma                                   # (K,)
                s = v.sum()
                if s <= 0:
                    continue
                a[i, j] = v / s
                la[i, j] = top + np.log(s)
                lam[i, j] = g.Lt @ a[i, j]
                rho[i, j] = g.Rt @ a[i, j]

        # Top level: the sentence is one tree (log_whole, root symbol from S),
        # or a forest (log_forest) of two or more pieces whose symbols come
        # from S_piece; after its second piece a forest stops (log_stop) or
        # continues (log_cont). top[i, j] = log sum_A S_piece[A] inside(i, j, A).
        with np.errstate(divide="ignore"):
            self.top = la + np.log(np.einsum("ijk,k->ij", a, g.S_piece))
            whole = la[0, n] + np.log(a[0, n] @ g.S) if n else -np.inf
        self.whole = g.log_whole + whole
        F1 = self.F1 = np.full(n + 1, -np.inf)   # one piece covering [0, j)
        H = self.H = np.full(n + 1, -np.inf)     # two or more pieces covering [0, j)
        if n:
            F1[1:n] = g.log_forest + self.top[0, 1:n]
        for j in range(2, n + 1):
            H[j] = logsumexp(np.logaddexp(F1[1:j], g.log_cont + H[1:j]) + self.top[1:j, j])
        G = self.G = np.full(n + 1, -np.inf)     # complete [j, n) after two or more pieces
        G1 = self.G1 = np.full(n + 1, -np.inf)   # complete [j, n) after exactly one piece
        if n:
            G[n] = g.log_stop
        for j in range(n - 1, 0, -1):
            G1[j] = logsumexp(self.top[j, j + 1:] + G[j + 1:])
            G[j] = g.log_cont + G1[j]
        self.forest = H[n] + g.log_stop if n else -np.inf
        self.log_prob = float(np.logaddexp(self.whole, self.forest)) if n else 0.0
        self._mu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def _top_posteriors(self) -> Tuple[float, np.ndarray]:
        """(P(the sentence is one tree), P(span (i, j) is a piece of a forest))."""
        n = self.n
        pieces = np.zeros((n + 1, n + 1))
        if n == 0 or not np.isfinite(self.log_prob):
            return 0.0, pieces
        lp = np.full((n + 1, n + 1), -np.inf)
        lp[0, 1:n] = self.F1[1:n] + self.G1[1:n]
        before = np.logaddexp(self.F1, self.g.log_cont + self.H)
        for i in range(1, n):
            lp[i, i + 1:] = before[i] + self.top[i, i + 1:] + self.G[i + 1:]
        with np.errstate(invalid="ignore"):
            pieces = np.exp(lp - self.log_prob)
        return float(np.exp(self.whole - self.log_prob)), np.nan_to_num(pieces)

    def top_level_posteriors(self) -> np.ndarray:
        """P(span (i, j) is a top-level chunk | sentence): the whole sentence as
        one tree, or a piece of a forest."""
        p_whole, tops = self._top_posteriors()
        if self.n:
            tops[0, self.n] += p_whole
        return tops

    def label_posteriors(self) -> np.ndarray:
        """mu[i, j, A] = P(span (i, j) is a chunk of category A | sentence)."""
        if self._mu is not None:
            return self._mu
        g, n, a, la = self.g, self.n, self.a, self.la
        mu = np.zeros_like(a)
        if n == 0 or not np.isfinite(self.log_prob):
            self._mu = mu
            return mu
        p_whole, pieces = self._top_posteriors()
        weighted = a * g.S_piece[None, None, :]
        norm = weighted.sum(axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu += np.where(norm > 0, weighted / norm, 0.0) * pieces[:, :, None]
        root = a[0, n] * g.S
        if root.sum() > 0:
            mu[0, n] += p_whole * root / root.sum()
        for length in range(n, 1, -1):
            for i in range(n - length + 1):
                j = i + length
                if mu[i, j].sum() <= 0:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    w = np.where(a[i, j] > 0, mu[i, j] / a[i, j], 0.0)
                nu = (w @ self.U[i]) * g.qk                             # (M,)
                lam_l = self.lam[i, i + 1:j]                            # (splits, M)
                rho_r = self.rho[i + 1:j, j]
                rel = np.exp(la[i, i + 1:j] + la[i + 1:j, j] - la[i, j])
                post = nu[None, :] * lam_l * rho_r * rel[:, None]       # P(split k, rule c)
                with np.errstate(divide="ignore", invalid="ignore"):
                    left = np.where(lam_l > 0, post / lam_l, 0.0)
                    right = np.where(rho_r > 0, post / rho_r, 0.0)
                mu[i, i + 1:j] += a[i, i + 1:j] * (left @ g.Lt)
                mu[i + 1:j, j] += a[i + 1:j, j] * (right @ g.Rt)
        self._mu = mu
        return mu

    def span_posteriors(self) -> np.ndarray:
        """P(span (i, j) is a chunk | sentence), summed over categories."""
        return self.label_posteriors().sum(axis=2)

    def mbr_tree(self, threshold: float = 0.0) -> Tree:
        """Binary tree over the whole sentence maximising the expected number
        of correct spans.

        Each span contributes ``mu(i, j) - threshold``; every binary tree has
        the same number of spans, so the threshold only matters for callers
        that read off the chosen spans' margins.
        """
        n = self.n
        if n < 2:
            return Tree(n, {})
        post = self.span_posteriors()
        mu = self.label_posteriors()
        best = np.zeros((n + 1, n + 1))
        choice: Dict[Span, int] = {}
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                ks = np.arange(i + 1, j)
                vals = best[i, ks] + best[ks, j]
                pick = int(np.argmax(vals))
                best[i, j] = post[i, j] - threshold + vals[pick]
                choice[(i, j)] = int(ks[pick])
        split: Dict[Span, int] = {}
        label: Dict[Span, int] = {}
        stack = [(0, n)]
        while stack:
            i, j = stack.pop()
            label[(i, j)] = int(np.argmax(mu[i, j]))
            if j - i < 2:
                continue
            k = choice[(i, j)]
            split[(i, j)] = k
            stack.extend([(i, k), (k, j)])
        return Tree(n, split, label)

    def viterbi_tree(self) -> Tree:
        """The single most probable labelled analysis (a forest when the
        grammar prefers several top-level chunks).

        Given the grammar, this is the analysis with the shortest derivation
        code, which is why unsupervised learning re-parses with it. Rule
        classes are summed out: P(A -> B C) = sum_c U[A,c] (1-pk[c]) Lt[c,B] Rt[c,C].
        """
        g, n = self.g, self.n
        if n == 0:
            return Tree(0, {})
        with np.errstate(divide="ignore"):
            # One table of binary rules per context (the word before a span).
            log_rule = [g.log_binary(x) for x in self.context]
            ids = g.token_ids(self.tokens)
            lex = np.log(np.stack([self.U[i] @ (g.pk * g.E[:, ids[i]]) for i in range(n)]))   # (n, K)
            log_start = np.log(g.S)
        K = g.K
        best = np.full((n + 1, n + 1, K), -np.inf)
        back: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(n):
            best[i, i + 1] = lex[i]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                scores = np.full((j - i - 1, K, K, K), -np.inf)
                for kk, k in enumerate(range(i + 1, j)):
                    pair = best[i, k][:, None] + best[k, j][None, :]       # (B, C)
                    scores[kk] = log_rule[i] + pair[None, :, :]
                flat = scores.transpose(1, 0, 2, 3).reshape(K, -1)         # A x (k, B, C)
                arg = np.argmax(flat, axis=1)
                best[i, j] = flat[np.arange(K), arg]
                back[(i, j)] = arg
        # The best whole tree, against the best forest of two or more pieces.
        A0 = int(np.argmax(log_start + best[0, n]))
        whole = g.log_whole + log_start[A0] + best[0, n, A0]
        with np.errstate(divide="ignore"):
            chunk = np.log(g.S_piece)[None, None, :] + best                 # (i, j, A)
        piece_A, piece_v = chunk.argmax(axis=2), chunk.max(axis=2)
        one = np.full(n + 1, -np.inf)       # one piece covering [0, j)
        many = np.full(n + 1, -np.inf)      # two or more pieces covering [0, j)
        prev: Dict[int, Tuple[int, bool]] = {}
        one[1:n] = g.log_forest + piece_v[0, 1:n]
        for j in range(2, n + 1):
            for i in range(1, j):
                for after_one, base in ((True, one[i]), (False, g.log_cont + many[i])):
                    if base + piece_v[i, j] > many[j]:
                        many[j], prev[j] = base + piece_v[i, j], (i, after_one)
        if whole >= many[n] + g.log_stop:
            roots, stack = [(0, n)], [(0, n, A0)]
        else:
            roots, stack, j = [], [], n
            while True:
                i, after_one = prev[j]
                roots.append((i, j))
                stack.append((i, j, int(piece_A[i, j])))
                if after_one:
                    roots.append((0, i))
                    stack.append((0, i, int(piece_A[0, i])))
                    break
                j = i
            roots.reverse()
        split: Dict[Span, int] = {}
        label: Dict[Span, int] = {}
        while stack:
            i, j, A = stack.pop()
            label[(i, j)] = A
            if j - i < 2:
                continue
            kk, B, C = np.unravel_index(back[(i, j)][A], (j - i - 1, K, K))
            k = i + 1 + int(kk)
            split[(i, j)] = k
            stack.extend([(i, k, int(B)), (k, j, int(C))])
        return Tree(n, split, label, roots)

    def confident_spans(self, threshold: float = 0.5):
        """Composite spans whose posterior exceeds ``threshold``. With
        threshold >= 0.5 these never cross, so they can be learned as chunks."""
        post = self.span_posteriors()
        n = self.n
        return [(i, j) for i in range(n) for j in range(i + 2, n + 1)
                if post[i, j] > threshold]

    def sample_tree(self, rng: np.random.Generator) -> Tree:
        """Draw one analysis from the posterior over analyses."""
        g, n, a, la = self.g, self.n, self.a, self.la
        if n == 0:
            return Tree(0, {})

        def pick(logits):
            p = np.exp(logits - logsumexp(logits))
            return int(rng.choice(len(p), p=p / p.sum()))

        if rng.random() < np.exp(self.whole - self.log_prob):
            w = g.S * a[0, n]
            roots, stack = [(0, n)], [(0, n, int(rng.choice(g.K, p=w / w.sum())))]
        else:
            # The first piece, then each next one; the stop at n is in G[n].
            ends = [1 + pick(self.F1[1:n] + self.G1[1:n])]
            while ends[-1] < n:
                i = ends[-1]
                ends.append(i + 1 + pick(self.top[i, i + 1:] + self.G[i + 1:]))
            roots, stack = [], []
            for i, j in zip([0] + ends[:-1], ends):
                w = g.S_piece * a[i, j]
                roots.append((i, j))
                stack.append((i, j, int(rng.choice(g.K, p=w / w.sum()))))
        split: Dict[Span, int] = {}
        label: Dict[Span, int] = {}
        while stack:
            i, j, A = stack.pop()
            label[(i, j)] = A
            if j - i < 2:
                continue
            lam_l = self.lam[i, i + 1:j]
            rho_r = self.rho[i + 1:j, j]
            rel = np.exp(la[i, i + 1:j] + la[i + 1:j, j] - la[i, j])
            weight = (self.U[i][A] * g.qk)[None, :] * lam_l * rho_r * rel[:, None]
            flat = weight.ravel()
            pick = int(rng.choice(flat.size, p=flat / flat.sum()))
            kk, c = divmod(pick, g.M)
            k = i + 1 + kk
            split[(i, j)] = k
            pb = g.Lt[c] * a[i, k]
            pc = g.Rt[c] * a[k, j]
            stack.append((k, j, int(rng.choice(g.K, p=pc / pc.sum()))))
            stack.append((i, k, int(rng.choice(g.K, p=pb / pb.sum()))))
        return Tree(n, split, label, roots)


def parse(grammar: Grammar, tokens: Sequence[str]) -> Tuple[Tree, Chart]:
    """Minimum-Bayes-risk parse; also returns the chart for inspection."""
    chart = Chart(grammar, tokens)
    return chart.mbr_tree(), chart
