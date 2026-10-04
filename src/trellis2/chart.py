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

        # Top level: the sentence is one tree (log_whole, root symbol from S),
        # or a forest (log_forest) of two or more pieces: the first piece's
        # symbol from S_piece, each next one's from T_piece[previous symbol],
        # and from the second piece on the forest ends (log_stop_v) or goes
        # on (log_cont_v) given the last piece's symbol.
        # ins[i, j, A] = log inside(i, j, A).
        K = g.K
        with np.errstate(divide="ignore"):
            ins = self.ins = la[:, :, None] + np.log(a)
            whole = la[0, n] + np.log(a[0, n] @ g.S) if n else -np.inf
            log_first = np.log(g.S_piece)
        self.whole = g.log_whole + whole
        F1 = self.F1 = np.full((n + 1, K), -np.inf)   # one piece covering [0, j), its symbol
        H = self.H = np.full((n + 1, K), -np.inf)     # two or more pieces covering [0, j), the last one's symbol
        P = self.P = np.full((n + 1, K), -np.inf)     # a further piece starting at i, its symbol
        if n:
            F1[1:n] = g.log_forest + log_first[None, :] + ins[0, 1:n]
        for i in range(1, n):
            before = np.logaddexp(F1[i], H[i] + g.log_cont_v)
            P[i] = logsumexp(before[:, None] + g.log_T, axis=0)
            H[i + 1:] = np.logaddexp(H[i + 1:], P[i][None, :] + ins[i, i + 1:])
        G = self.G = np.full((n + 1, K), -np.inf)     # complete [j, n) after a later piece of symbol B
        G1 = self.G1 = np.full((n + 1, K), -np.inf)   # complete [j, n) after the first piece
        if n:
            G[n] = g.log_stop_v
        for j in range(n - 1, 0, -1):
            nxt = logsumexp(ins[j, j + 1:] + G[j + 1:], axis=0)          # next piece's symbol C
            G1[j] = logsumexp(g.log_T + nxt[None, :], axis=1)
            G[j] = g.log_cont_v + G1[j]
        self.forest = float(logsumexp(H[n] + g.log_stop_v)) if n else -np.inf
        self.log_prob = float(np.logaddexp(self.whole, self.forest)) if n else 0.0
        self._mu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def _top_posteriors(self) -> Tuple[float, np.ndarray]:
        """(P(the sentence is one tree), P(span (i, j) is a piece of symbol A
        of a forest)), the latter of shape (n+1, n+1, K)."""
        n, K = self.n, self.g.K
        pieces = np.zeros((n + 1, n + 1, K))
        if n == 0 or not np.isfinite(self.log_prob):
            return 0.0, pieces
        lp = np.full((n + 1, n + 1, K), -np.inf)
        lp[0, 1:n] = self.F1[1:n] + self.G1[1:n]
        for i in range(1, n):
            lp[i, i + 1:] = self.P[i][None, :] + self.ins[i, i + 1:] + self.G[i + 1:]
        with np.errstate(invalid="ignore"):
            pieces = np.exp(lp - self.log_prob)
        return float(np.exp(self.whole - self.log_prob)), np.nan_to_num(pieces)

    def top_level_posteriors(self) -> np.ndarray:
        """P(span (i, j) is a top-level chunk | sentence): the whole sentence as
        one tree, or a piece of a forest."""
        p_whole, pieces = self._top_posteriors()
        tops = pieces.sum(axis=2)
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
        mu += pieces
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
        # The best whole tree, against the best forest of two or more pieces.
        A0 = int(np.argmax(log_start + best[0, n]))
        whole = g.log_whole + log_start[A0] + best[0, n, A0]
        with np.errstate(divide="ignore"):
            log_first = np.log(g.S_piece)
        one = np.full((n + 1, K), -np.inf)       # one piece covering [0, j), its symbol
        many = np.full((n + 1, K), -np.inf)      # two or more, the last one's symbol
        prev: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
        one[1:n] = g.log_forest + log_first[None, :] + best[0, 1:n]
        for i in range(1, n):
            cand = np.stack([one[i], many[i] + g.log_cont_v])           # (2, A)
            score = cand[:, :, None] + g.log_T[None, :, :]              # (2, A, B)
            flat = score.reshape(-1, K)
            arg = flat.argmax(axis=0)                                    # best (phase, A) per B
            enter = flat[arg, np.arange(K)]
            for jj in range(i + 1, n + 1):
                val = enter + best[i, jj]
                better = val > many[jj]
                for B in np.flatnonzero(better):
                    many[jj, B] = val[B]
                    ph, A = divmod(int(arg[B]), K)
                    prev[(jj, int(B))] = (i, A, ph == 0)
        end = many[n] + g.log_stop_v
        Bn = int(np.argmax(end)) if n else 0
        if n < 2 or whole >= end[Bn]:
            roots, stack = [(0, n)], [(0, n, A0)]
        else:
            roots, stack, jj, B = [], [], n, Bn
            while True:
                i, A, from_one = prev[(jj, B)]
                roots.append((i, jj))
                stack.append((i, jj, B))
                if from_one:
                    roots.append((0, i))
                    stack.append((0, i, A))
                    break
                jj, B = i, A
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
            K = g.K
            first = pick((self.F1[1:n] + self.G1[1:n]).ravel())
            jj, A = 1 + first // K, first % K
            roots, stack = [(0, jj)], [(0, jj, A)]
            while jj < n:
                # The next piece (jj, k) of symbol C; the stop at n is in G[n].
                nxt = pick((g.log_T[A][None, :] + self.ins[jj, jj + 1:] + self.G[jj + 1:]).ravel())
                k, C = jj + 1 + nxt // K, nxt % K
                roots.append((jj, k))
                stack.append((jj, k, C))
                jj, A = k, C
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
