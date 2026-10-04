"""Inside-outside parsing over the factored grammar.

A sentence is a sequence of top-level chunks (a single one when the grammar
always stops after one, ``p_stop = 1``), each a binary tree. The inside pass
fills, for every span, the probability that each symbol derives it ("the
frontier of valid parses going up"). A forward-backward pass over the top
level sums over every way of cutting the sentence into top-level chunks. A
top-down pass then gives posteriors: ``mu[i, j, A]`` is the probability that
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
        for i in range(n):
            v = g.lexical(int(ids[i]))
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
                v = g.U @ gamma                                    # (K,)
                s = v.sum()
                if s <= 0:
                    continue
                a[i, j] = v / s
                la[i, j] = top + np.log(s)
                lam[i, j] = g.Lt @ a[i, j]
                rho[i, j] = g.Rt @ a[i, j]

        # Top level. top[i, j] = log sum_A S[A] inside(i, j, A); a chunk that
        # does not start the sentence first pays log(1 - p_stop).
        with np.errstate(divide="ignore"):
            self.top = la + np.log(np.einsum("ijk,k->ij", a, g.S))
        self.cont = np.full(n + 1, g.log_cont)
        if n:
            self.cont[0] = 0.0
        F = self.F = np.full(n + 1, -np.inf)   # all ways to cover [0, j)
        G = self.G = np.full(n + 1, -np.inf)   # all ways to cover [j, n), with the stop
        F[0] = 0.0
        for j in range(1, n + 1):
            F[j] = logsumexp(F[:j] + self.cont[:j] + self.top[:j, j])
        G[n] = g.log_stop
        for i in range(n - 1, -1, -1):
            G[i] = logsumexp(self.cont[i] + self.top[i, i + 1:] + G[i + 1:])
        self.log_prob = float(G[0]) if n else 0.0
        self._mu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def top_level_posteriors(self) -> np.ndarray:
        """P(span (i, j) is a top-level chunk | sentence)."""
        n = self.n
        with np.errstate(invalid="ignore"):
            lp = (self.F[:, None] + self.cont[:, None] + self.top + self.G[None, :]
                  - self.log_prob)
        lp[~np.isfinite(lp)] = -np.inf
        return np.exp(lp) if n else np.zeros((1, 1))

    def label_posteriors(self) -> np.ndarray:
        """mu[i, j, A] = P(span (i, j) is a chunk of category A | sentence)."""
        if self._mu is not None:
            return self._mu
        g, n, a, la = self.g, self.n, self.a, self.la
        mu = np.zeros_like(a)
        if n == 0 or not np.isfinite(self.log_prob):
            self._mu = mu
            return mu
        tops = self.top_level_posteriors()
        weighted = a * g.S[None, None, :]
        norm = weighted.sum(axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu += np.where(norm > 0, weighted / norm, 0.0) * tops[:, :, None]
        for length in range(n, 1, -1):
            for i in range(n - length + 1):
                j = i + length
                if mu[i, j].sum() <= 0:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    w = np.where(a[i, j] > 0, mu[i, j] / a[i, j], 0.0)
                nu = (w @ g.U) * g.qk                                   # (M,)
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
            log_rule = np.log(np.einsum("ac,c,cb,cd->abd", g.U, g.qk, g.Lt, g.Rt))
            ids = g.token_ids(self.tokens)
            lex = np.log(g.U @ (g.pk[:, None] * g.E[:, ids])).T          # (n, K)
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
                    scores[kk] = log_rule + pair[None, :, :]
                flat = scores.transpose(1, 0, 2, 3).reshape(K, -1)         # A x (k, B, C)
                arg = np.argmax(flat, axis=1)
                best[i, j] = flat[np.arange(K), arg]
                back[(i, j)] = arg
        # Best way to cut the sentence into top-level chunks.
        chunk = log_start[None, None, :] + best                            # (i, j, A)
        V = np.full(n + 1, -np.inf)
        V[0] = 0.0
        prev: Dict[int, Tuple[int, int]] = {}
        for j in range(1, n + 1):
            for i in range(j):
                A = int(np.argmax(chunk[i, j]))
                cand = V[i] + self.cont[i] + chunk[i, j, A]
                if cand > V[j]:
                    V[j], prev[j] = cand, (i, A)
        roots, stack, j = [], [], n
        while j > 0:
            i, A = prev[j]
            roots.append((i, j))
            stack.append((i, j, A))
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
        roots, stack, i = [], [], 0
        while i < n:
            logits = self.cont[i] + self.top[i, i + 1:] + self.G[i + 1:]
            p = np.exp(logits - logsumexp(logits))
            j = i + 1 + int(rng.choice(len(p), p=p / p.sum()))
            w = g.S * a[i, j]
            roots.append((i, j))
            stack.append((i, j, int(rng.choice(g.K, p=w / w.sum()))))
            i = j
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
            weight = (g.U[A] * g.qk)[None, :] * lam_l * rho_r * rel[:, None]
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
