"""Prompting: prefix probabilities and completions checked exactly, on small
grammars whose sentences are one tree or a forest, read with no context, one
word or two."""
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from test_chart import context_grammar, forest_grammar, pair_grammar
from trellis2.chart import Chart
from trellis2.prompt import PromptChart, complete

GRAMMARS = {"forest": lambda: forest_grammar(3, p_stop=0.5, p_whole=0.4),
            "context": lambda: context_grammar(4, p_stop=0.5, p_whole=0.4),
            "pairs": lambda: pair_grammar(5, p_stop=0.5, p_whole=0.4)}


@pytest.mark.parametrize("make", list(GRAMMARS))
@pytest.mark.parametrize("prompt", [[], ["w0"], ["w1", "w0"], ["w2", "w2", "w1"], ["w1", "<unk>", "w0", "w0"]])
def test_beginning_with_a_prompt_is_ending_there_or_reading_one_more_token(make, prompt):
    """P(begins with p) = P(is exactly p) + the sum over tokens w of
    P(begins with p w), with no truncation; and nothing begins with less
    than certainty."""
    g = GRAMMARS[make]()
    z = PromptChart(g, prompt).log_prefix
    exact = Chart(g, prompt).log_prob if prompt else -np.inf
    more = logsumexp([PromptChart(g, prompt + [w]).log_prefix for w in g.vocab])
    assert math.isclose(z, float(np.logaddexp(exact, more)), rel_tol=1e-9, abs_tol=1e-12)
    if not prompt:
        assert z == 0.0


@pytest.mark.parametrize("make", list(GRAMMARS))
@pytest.mark.parametrize("prompt", [["w1"], ["w0", "w2"]])
def test_completions_are_drawn_as_often_as_the_grammar_says(make, prompt):
    """Each completed sentence s is drawn with probability P(s) / P(begins
    with the prompt), its analysis is a valid tree over it, and the
    scaffold's open chunks cover the end of the prompt."""
    g = GRAMMARS[make]()
    pc = PromptChart(g, prompt)
    rng = np.random.default_rng(0)
    n, counts = 6000, {}
    for _ in range(n):
        c = pc.complete(rng, max_len=10)
        if c is None:
            continue
        assert c.tokens[:len(prompt)] == prompt and c.tree.is_valid() and c.tree.n == len(c.tokens)
        assert all(i < len(prompt) <= j for i, j in c.open_spans)
        counts[tuple(c.tokens)] = counts.get(tuple(c.tokens), 0) + 1
    for tokens, k in sorted(counts.items(), key=lambda x: -x[1])[:6]:
        expected = n * math.exp(Chart(g, list(tokens)).log_prob - pc.log_prefix)
        assert abs(k - expected) <= 4 * math.sqrt(expected) + 2, (tokens, k, expected)


def test_a_low_temperature_completes_with_likelier_sentences():
    g = pair_grammar(5, p_stop=0.5, p_whole=0.4)
    rng = np.random.default_rng(1)
    warm = np.mean([c.log_prob for c in complete(g, ["w1"], 400, rng)])
    cool = np.mean([c.log_prob for c in complete(g, ["w1"], 400, rng, temperature=0.3)])
    assert cool > warm
