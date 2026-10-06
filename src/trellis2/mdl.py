"""Description lengths in bits.

The learner's objective is the length of an actual message: the bits needed
to transmit the training sentences. A sentence is sent as its analysis
(derivation), coded event by event with the Dirichlet-multinomial predictive
of each grammar table, i.e. the Bayesian mixture over the table's parameters.
Arithmetic coding with that adaptive predictor achieves the length

    L(D) = sum over tables  -log2 DM(counts; alpha)

whatever the order of the events, so it is a proper prequential code
(Rissanen 1984; Dawid 1984). The receiver additionally needs the grammar's
size (number of symbols and rule classes), sent with Elias' universal code
for the integers. Nothing else is free: a chunk type pays for itself only by
making the analyses that use it cheaper to send.

The total splits into data bits, the cost under the best-fitting parameters,
-log2 P(D | theta_ML), and model bits, the remainder
log2 [P(D | theta_ML) / P_DM(D)] >= 0, which is the price of learning the
parameters (the parametric complexity, or Occam factor).
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

LN2 = math.log(2.0)


def dm_code(groups: np.ndarray, keys: np.ndarray, weights: np.ndarray,
            alphabet: int, alpha: float) -> float:
    """-log marginal likelihood (nats) of ``keys`` under one Dirichlet-multinomial
    per group, each over ``alphabet`` outcomes with concentration ``alpha``."""
    if groups.size == 0:
        return 0.0
    stride = np.int64(alphabet)
    gk = groups.astype(np.int64) * stride + keys.astype(np.int64)
    n_groups = int(groups.max()) + 1
    if n_groups * alphabet <= 16 * gk.size + 4096:
        # A small table: count every cell directly instead of sorting (the
        # same cells, counts and sums, in the same order).
        uniq = np.flatnonzero(np.bincount(gk, minlength=n_groups * alphabet))
        cnt = np.bincount(gk, weights=weights, minlength=n_groups * alphabet)[uniq]
        g = uniq // stride
        totals = np.bincount(g, weights=cnt, minlength=n_groups)[np.bincount(g, minlength=n_groups) > 0]
    else:
        uniq, inv = np.unique(gk, return_inverse=True)
        cnt = np.bincount(inv, weights=weights)
        _, ginv = np.unique(uniq // stride, return_inverse=True)
        totals = np.bincount(ginv, weights=cnt)
    a_tot = alphabet * alpha
    return float(np.sum(gammaln(totals + a_tot) - gammaln(a_tot))
                 - np.sum(gammaln(cnt + alpha) - gammaln(alpha)))


def _before(keys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """For each event, the weight of the earlier events with the same key."""
    order = np.argsort(keys, kind="stable")
    k, w = keys[order], weights[order]
    total = np.cumsum(w)
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    first = np.repeat(total[starts] - w[starts], np.diff(np.r_[starts, len(k)]))
    out = np.empty_like(total)
    out[order] = total - w - first
    return out


def backoff_code(groups: np.ndarray, contexts: np.ndarray, keys: np.ndarray, weights: np.ndarray,
                 alphabet: int, alpha: float, beta: float) -> float:
    """Prequential code (nats) of ``keys``, each predicted from the events
    before it given its group and its context, backing off to the group:

        P(k | g, x) = (n(g, x, k) + beta P(k | g)) / (n(g, x) + beta),
        P(k | g)    = (n(g, k) + alpha) / (n(g) + alphabet alpha).

    The code is that of arithmetic coding with this predictive, events in
    the given order. A context seen with the group for the first time is
    predicted by the group alone."""
    if groups.size == 0:
        return 0.0
    return backoff_coder(groups, contexts, weights)(keys, alphabet, alpha, beta)


def backoff_coder(groups: np.ndarray, contexts: np.ndarray, weights: np.ndarray):
    """``backoff_code`` for fixed groups and contexts, as a function of
    (keys, alphabet, alpha, beta): what depends only on the groups and
    contexts is counted once (a cut search codes many keyings of the same
    events)."""
    g, x = groups.astype(np.int64), contexts.astype(np.int64)
    gx = g * np.int64(x.max() + 1) + x
    n_g = _before(g, weights)
    n_gx = _before(gx, weights)

    def code(keys: np.ndarray, alphabet: int, alpha: float, beta: float) -> float:
        k, span_k = keys.astype(np.int64), np.int64(alphabet)
        n_gk = _before(g * span_k + k, weights)
        n_gxk = _before(gx * span_k + k, weights)
        p_g = (n_gk + alpha) / (n_g + alphabet * alpha)
        p = (n_gxk + beta * p_g) / (n_gx + beta)
        return float(-np.sum(weights * np.log(p)))
    return code


def rows_nats(n: np.ndarray, alpha: float) -> float:
    """Code (nats) of count rows (one per row of ``n``), each a
    Dirichlet-multinomial over its columns."""
    a = n.shape[-1] * alpha
    return float(np.sum(gammaln(n.sum(axis=-1) + a) - gammaln(a))
                 - np.sum(gammaln(n + alpha) - gammaln(alpha)))


def elias_delta_bits(n: int) -> float:
    """Length of Elias' delta code for a positive integer (a universal code)."""
    if n < 1:
        raise ValueError("Elias delta codes positive integers")
    L = n.bit_length()
    return float(L - 1 + 2 * (L.bit_length() - 1) + 1)


def dm_row_nats(counts: Iterable[float], alphabet: int, alpha: float) -> float:
    """-ln DM(counts) for one table row over ``alphabet`` outcomes."""
    c = np.fromiter(counts, dtype=float)
    if c.size == 0:
        return 0.0
    a_tot = alphabet * alpha
    return float(gammaln(c.sum() + a_tot) - gammaln(a_tot)
                 - np.sum(gammaln(c + alpha) - gammaln(alpha)))


def ml_row_nats(counts: Iterable[float]) -> float:
    """-ln P(counts | maximum-likelihood parameters) for one table row."""
    c = np.fromiter(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    return float(-np.sum(c * np.log(c / c.sum())))


def split_bits(tables: Sequence[Tuple[Sequence[Mapping], int]], alpha: float
               ) -> Tuple[float, float]:
    """(model bits, data bits) for tables given as (rows, alphabet size)."""
    dm = ml = 0.0
    for rows, alphabet in tables:
        for row in rows:
            vals = [v for v in row.values() if v > 0]
            dm += dm_row_nats(vals, alphabet, alpha)
            ml += ml_row_nats(vals)
    return (dm - ml) / LN2, ml / LN2
