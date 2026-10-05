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
    uniq, inv = np.unique(gk, return_inverse=True)
    cnt = np.bincount(inv, weights=weights)
    _, ginv = np.unique(uniq // stride, return_inverse=True)
    totals = np.bincount(ginv, weights=cnt)
    a_tot = alphabet * alpha
    return float(np.sum(gammaln(totals + a_tot) - gammaln(a_tot))
                 - np.sum(gammaln(cnt + alpha) - gammaln(alpha)))


def beta_nats(n: np.ndarray, alpha: float) -> float:
    """Code (nats) of the counts of a two-outcome row (Beta-binomial)."""
    return float(gammaln(n.sum() + 2 * alpha) - gammaln(2 * alpha)
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
