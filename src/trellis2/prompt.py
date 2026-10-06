"""Prompting: complete the beginning of an experience with the grammar.

Given a prompt (the first tokens of a sentence, in the domain's reading
order), the grammar completes it in two steps.

1. A scaffolded parse. The prompt is analysed as the beginning of an
   experience: chunks that finish inside it, and a right frontier of open
   chunks, each covering the end of the prompt and going on past it, with the
   categories of the parts it still needs. The scaffold is drawn from the
   grammar's posterior given the prompt, which needs, for every category and
   prompt position, the probability that the category derives a string
   beginning with the rest of the prompt (prefix probabilities; Jelinek &
   Lafferty 1991, Stolcke 1995). A chunk that covers the end of the prompt
   is open through its left part, whose left part may be open again (the
   left-corner closure, one small linear solve per position), or through its
   right part after a left part that finishes inside the prompt.
2. Completion. The open chunks' missing parts are generated innermost first,
   chunk by chunk, each decomposed down to tokens in the light of the words
   read so far, until the parse closes (and a forest has as many pieces as its
   layout draws). At temperature 1 the completions are drawn from
   P(experience | it begins with the prompt); below 1, each choice favours
   its likelier chunks.

Sentences only (the domain read left to right, as one tree or a forest);
pieces read afresh are not supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from .chart import Chart
from .data import Span, Tree
from .grammar import Grammar


@dataclass
class Completion:
    tokens: List[str]          # the prompt and its completion
    tree: Tree                 # the whole analysis (one tree or a forest)
    prompt_length: int
    open_spans: List[Span]     # the scaffold's open chunks, as completed
    log_prob: float            # ln P(experience | it begins with the prompt)


class PromptChart:
    """Prefix probabilities of a prompt under a grammar, and draws of its
    scaffold and completions."""

    def __init__(self, grammar: Grammar, prompt: Sequence[str]):
        g = self.g = grammar
        if g.fresh_pieces:
            raise ValueError("prompting reads a forest's pieces in the light of the words before them")
        self.prompt = list(prompt)
        m = self.m = len(self.prompt)
        self.chart = Chart(g, self.prompt) if m else None
        K = g.K
        # pv[i] * exp(pl[i]) = P(A derives a string beginning with prompt[i:]).
        self.pv = np.zeros((m + 1, K))
        self.pl = np.full(m + 1, -np.inf)
        self.pv[m], self.pl[m] = 1.0 / K, np.log(K)       # nothing left to match
        ids = g.token_ids(self.prompt)
        c = self.chart
        for i in range(m - 1, -1, -1):
            U = g.rules(g.read(self.prompt, i))
            # Terms with a left part that finishes inside the prompt, [i, k),
            # k < m, and a right part that starts at k and covers the rest.
            ks = np.arange(i + 1, m)
            if len(ks):
                scale = c.la[i, ks] + self.pl[ks]
                top = scale.max() if np.isfinite(scale).any() else -np.inf
            else:
                top = -np.inf
            if i == m - 1:                                # the last token itself
                b = U @ (g.pk * g.E[:, ids[i]])
                shift = 0.0
            elif np.isfinite(top):
                weights = np.exp(scale - top)[:, None]
                gamma = (c.lam[i, ks] * (self.pv[ks] @ g.Rt.T) * weights).sum(axis=0)
                b = U @ (g.qk * gamma)
                shift = top
            else:
                continue
            # The left-corner closure: a left part open at i again.
            R = (U * g.qk) @ g.Lt
            v = np.linalg.solve(np.eye(K) - R, b)
            v = np.maximum(v, 0.0)
            s = v.sum()
            if s > 0 and np.isfinite(s):
                self.pv[i], self.pl[i] = v / s, shift + np.log(s)
        self._top_terms()

    # ------------------------------------------------------------------ #
    def _top_terms(self) -> None:
        """ln P of each way the prompt can begin an experience: one tree whose
        root covers it (and may end with it), or a forest whose piece holding
        the prompt's last token starts at k, after none, one or more finished
        pieces. Their sum is the prompt's prefix probability."""
        g, m = self.g, self.m
        with np.errstate(divide="ignore"):
            terms: List[Tuple[float, tuple]] = []
            if m == 0:
                self.terms, self.log_prefix = [(0.0, ("empty",))], 0.0
                return
            c = self.chart
            log_S, log_P = np.log(self.pv[0] @ g.S), np.log(g.S_piece @ self.pv.T)
            terms.append((g.log_whole + log_S + self.pl[0], ("whole",)))
            terms.append((g.log_forest + log_P[0] + self.pl[0], ("open piece", 0, "none")))
            for k in range(1, m):
                one = g.log_forest + c.top[0, k]
                terms.append((one + log_P[k] + self.pl[k], ("open piece", k, "one")))
                if k >= 2:
                    terms.append((g.log_cont + c.H[k] + log_P[k] + self.pl[k], ("open piece", k, "many")))
        self.terms = [(float(v), how) for v, how in terms]
        self.log_prefix = float(logsumexp([v for v, _ in self.terms]))

    # ------------------------------------------------------------------ #
    def complete(self, rng: np.random.Generator, temperature: float = 1.0,
                 max_len: int = 30) -> Optional[Completion]:
        """One completion: a scaffold drawn from the posterior given the
        prompt, then its open parts generated. None if the experience would
        exceed ``max_len`` tokens (callers draw again)."""
        g, m = self.g, self.m
        if not np.isfinite(self.log_prefix):
            raise ValueError("the grammar cannot begin an experience with this prompt")
        self._nodes: List[list] = []     # [symbol, rule class, token, left, right], depth first
        self._pending: List[Tuple[int, int, int]] = []    # (symbol, parent, side), outermost first
        self._open: List[int] = []
        how = self._pick([v for v, _ in self.terms], rng)
        how = self.terms[how][1]
        roots: List[int] = []
        pieces_after = None                      # forests: pieces drawn so far, if the layout goes on
        if how[0] == "empty":
            out = g.sample(rng, max_len=max_len)
            if out is None:
                return None
            tokens, tree = out
            return Completion(tokens, tree, 0, [], self._conditional(tokens))
        if how[0] == "whole":
            A = self._choose(np.log(g.S) + np.log(self.pv[0]), rng)
            roots.append(self._open_node(0, A, rng))
        else:
            _, k, before = how
            roots.extend(self._pieces(0, k, before, rng) if k else [])
            A = self._choose(np.log(g.S_piece) + np.log(self.pv[k]), rng)
            roots.append(self._open_node(k, A, rng))
            pieces_after = len(roots)
        # Completion: the open chunks' missing parts, innermost first, then
        # any further pieces the forest's layout draws.
        tokens = list(self.prompt)
        for sym, parent, side in reversed(self._pending):
            if not self._expand(sym, parent, side, tokens, rng, temperature, max_len):
                return None
        if pieces_after is not None:
            while pieces_after < 2 or rng.random() >= g.p_stop:
                A = int(rng.choice(g.K, p=_temper(g.S_piece, temperature)))
                root = len(self._nodes)
                if not self._expand(A, -1, 0, tokens, rng, temperature, max_len):
                    return None
                roots.append(root)
                pieces_after += 1
        tree, spans = self._tree(roots, len(tokens))
        return Completion(tokens, tree, m, [spans[i] for i in self._open], self._conditional(tokens))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick(logits: Sequence[float], rng: np.random.Generator) -> int:
        x = np.asarray(logits, dtype=float)
        p = np.exp(x - logsumexp(x))
        return int(rng.choice(len(p), p=p / p.sum()))

    def _choose(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        with np.errstate(divide="ignore"):
            return self._pick(np.where(np.isfinite(logits), logits, -np.inf), rng)

    def _pieces(self, i: int, j: int, how: str, rng: np.random.Generator) -> List[int]:
        """Finished pieces covering prompt[i:j] (``i`` = 0): one piece, or two
        or more, drawn as the chart's forest recursion weighs them."""
        c, g = self.chart, self.g
        if how == "one":
            return [self._finished_piece(0, j, rng)]
        # Two or more pieces ending at j: the last piece [s, j) follows either
        # one piece or, after a continue, two or more.
        ends = [j]
        while True:
            s_opts = np.arange(1, ends[-1])
            one = c.F1[s_opts] + c.top[s_opts, ends[-1]]
            many = g.log_cont + c.H[s_opts] + c.top[s_opts, ends[-1]]
            pick = self._pick(np.concatenate([one, many]), rng)
            s = int(s_opts[pick % len(s_opts)])
            ends.append(s)
            if pick < len(s_opts):        # the piece before is the first
                break
        bounds = [0] + ends[::-1]
        return [self._finished_piece(a, b, rng) for a, b in zip(bounds, bounds[1:])]

    def _finished_piece(self, i: int, j: int, rng: np.random.Generator) -> int:
        A = self._choose(np.log(self.g.S_piece) + np.log(self.chart.a[i, j]), rng)
        return self._finished(i, j, A, rng)

    def _finished(self, i: int, j: int, A: int, rng: np.random.Generator) -> int:
        """A chunk that finishes inside the prompt, drawn from its inside
        posterior (its node index)."""
        g, c = self.g, self.chart
        idx = len(self._nodes)
        if j - i == 1:
            self._nodes.append([A, -1, self.prompt[i], -1, -1])
            return idx
        lam_l, rho_r = c.lam[i, i + 1:j], c.rho[i + 1:j, j]
        rel = np.exp(c.la[i, i + 1:j] + c.la[i + 1:j, j] - c.la[i, j])
        weight = (c.U[i][A] * g.qk)[None, :] * lam_l * rho_r * rel[:, None]
        flat = weight.ravel()
        kk, rule = divmod(int(rng.choice(flat.size, p=flat / flat.sum())), g.M)
        k = i + 1 + kk
        B = int(rng.choice(g.K, p=_norm(g.Lt[rule] * c.a[i, k])))
        C = int(rng.choice(g.K, p=_norm(g.Rt[rule] * c.a[k, j])))
        self._nodes.append([A, rule, None, -1, -1])
        self._nodes[idx][3] = self._finished(i, k, B, rng)
        self._nodes[idx][4] = self._finished(k, j, C, rng)
        return idx

    def _open_node(self, i: int, A: int, rng: np.random.Generator) -> int:
        """A chunk of category A open at prompt position i: it covers the rest
        of the prompt and goes on (its node index)."""
        g, c, m = self.g, self.chart, self.m
        U = g.rules(g.read(self.prompt, i))[A]
        options: List[Tuple[float, tuple]] = []
        with np.errstate(divide="ignore"):
            if i == m - 1:                      # the prompt's last token, as A's whole yield
                w = g.token_ids([self.prompt[i]])[0]
                lex = U * g.pk * g.E[:, w]
                options += [(float(np.log(lex[r])), ("token", r)) for r in np.flatnonzero(lex > 0)]
            for k in range(i + 1, m):           # a finished left part [i, k), the right part open at k
                part = U * g.qk * c.lam[i, k] * (g.Rt @ self.pv[k])
                logs = np.log(part) + c.la[i, k] + self.pl[k]
                options += [(float(logs[r]), ("finished", r, k)) for r in np.flatnonzero(part > 0)]
            part = U * g.qk * (g.Lt @ self.pv[i])   # the left part open at i, the right part to come
            logs = np.log(part) + self.pl[i]
            options += [(float(logs[r]), ("open", r)) for r in np.flatnonzero(part > 0)]
        how = options[self._pick([v for v, _ in options], rng)][1]
        idx = len(self._nodes)
        self._open.append(idx)
        if how[0] == "token":
            self._nodes.append([A, how[1], self.prompt[i], -1, -1])
            return idx
        rule = how[1]
        self._nodes.append([A, rule, None, -1, -1])
        if how[0] == "finished":
            k = how[2]
            B = int(rng.choice(g.K, p=_norm(g.Lt[rule] * c.a[i, k])))
            C = int(rng.choice(g.K, p=_norm(g.Rt[rule] * self.pv[k])))
            self._nodes[idx][3] = self._finished(i, k, B, rng)
            self._nodes[idx][4] = self._open_node(k, C, rng)
        else:
            B = int(rng.choice(g.K, p=_norm(g.Lt[rule] * self.pv[i])))
            C = int(rng.choice(g.K, p=g.Rt[rule]))
            self._pending.append((C, idx, 1))
            self._nodes[idx][3] = self._open_node(i, B, rng)
        return idx

    def _expand(self, sym: int, parent: int, side: int, tokens: List[str], rng: np.random.Generator,
                temperature: float, max_len: int) -> bool:
        """Generate a chunk of category ``sym`` after ``tokens`` (as the
        grammar's sampler does), attaching it to ``parent``; False if the
        experience grows past ``max_len`` tokens."""
        g = self.g
        stack = [(sym, parent, side)]
        while stack:
            sym, parent, side = stack.pop()
            rule = int(rng.choice(g.M, p=_temper(g.rules(g.read(tokens, len(tokens)))[sym], temperature)))
            idx = len(self._nodes)
            if parent >= 0:
                self._nodes[parent][3 + side] = idx
            if rng.random() < _temper(np.array([g.pk[rule], g.qk[rule]]), temperature)[0]:
                w = int(rng.choice(len(g.vocab), p=_temper(g.E[rule], temperature)))
                self._nodes.append([sym, rule, g.vocab[w], -1, -1])
                tokens.append(g.vocab[w])
                if len(tokens) > max_len:
                    return False
            else:
                self._nodes.append([sym, rule, None, -1, -1])
                B = int(rng.choice(g.K, p=_temper(g.Lt[rule], temperature)))
                C = int(rng.choice(g.K, p=_temper(g.Rt[rule], temperature)))
                stack.append((C, idx, 1))
                stack.append((B, idx, 0))
        return True

    def _tree(self, roots: List[int], n: int) -> Tuple[Tree, Dict[int, Span]]:
        """Spans of every node: leaves come left to right in depth-first order."""
        split, label, spans = {}, {}, {}
        cursor = [0]

        def span_of(idx):
            sym, _, tok, li, ri = self._nodes[idx]
            if tok is not None:
                i = cursor[0]
                cursor[0] += 1
                spans[idx] = (i, i + 1)
                label[(i, i + 1)] = sym
                return i, i + 1
            i, k = span_of(li)
            _, j = span_of(ri)
            split[(i, j)] = k
            label[(i, j)] = sym
            spans[idx] = (i, j)
            return i, j

        root_spans = [span_of(r) for r in roots]
        assert cursor[0] == n
        return Tree(n, split, label, root_spans), spans

    def _conditional(self, tokens: List[str]) -> float:
        """ln P(experience | it begins with the prompt)."""
        return float(Chart(self.g, tokens).log_prob - self.log_prefix)


def _norm(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1.0 / len(p))


def _temper(p: np.ndarray, temperature: float) -> np.ndarray:
    """A distribution sharpened (temperature < 1) or flattened (> 1)."""
    if temperature == 1.0:
        return p / p.sum()
    with np.errstate(divide="ignore"):
        logp = np.log(p) / temperature
    logp -= logp.max()
    q = np.exp(logp)
    return q / q.sum()


def complete(grammar: Grammar, prompt: Sequence[str], n: int = 1, rng: Optional[np.random.Generator] = None,
             temperature: float = 1.0, max_len: int = 30, max_tries: int = 100) -> List[Completion]:
    """``n`` completions of ``prompt`` (drawn again when one exceeds ``max_len``)."""
    rng = rng or np.random.default_rng()
    pc = PromptChart(grammar, prompt)
    out: List[Completion] = []
    tries = 0
    while len(out) < n:
        c = pc.complete(rng, temperature, max_len)
        if c is None:
            tries += 1
            if tries > max_tries * n:
                raise RuntimeError("completions keep exceeding max_len")
            continue
        out.append(c)
    return out
