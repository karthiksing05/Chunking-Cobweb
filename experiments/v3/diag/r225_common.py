"""Round 225 shared helpers: the held-out sentences for the compression measurements and the code-length arithmetic.

Data (same stream and seeds as r212 / r215 / r216, `r215_common.stream`):
  train  the first n kept sentences of the seed's stream (3..8 words, all in the 250-word vocabulary)
  dev    the first DEV_N kept sentences of the stories right after the training stories (the LM early-stopping set of r212/r215); used here
         ONLY to select a free parameter of a code (TRELLIS eps floor, BPE merge count, n-gram discount grid, LM architecture / epoch)
  test   the first TEST_N kept sentences of the seed's 1500 test stories (stories disjoint from every training story); every number
         in the tables is on these
A sentence's code length is -log2 of the model's probability of the sentence AS A SEQUENCE OF 250-WORD-VOCABULARY WORDS PLUS THE END
(</s> for the LMs and n-grams; the end flag of the last word's outcome for TRELLIS's generator), so every model codes the same events
(a word at a time over the same alphabet and the length). Bits per token = bits per sentence / (words + 1) (words and the end);
bits per word = bits / words.
"""
import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402

OUT = "experiments/v3/results/story_gen"
RAW = os.path.join(OUT, "r225_raw")
SEEDS = (13, 17)
NS = tuple(int(x) for x in os.environ.get("R225_NS", "250,500,1000,2000,4000").split(","))
TEST_N, DEV_N = 1000, 300


def data(seed, n):
    train, held, test, vocab = C.stream(seed, n)
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    tests = [list(s) for st in test for s in st if keep(s)][:TEST_N]
    dev = [list(s) for s in held][:DEV_N]
    sents = [list(s) for s, _ in train]
    return sents, dev, tests, vocab


def bits_of_steps(steps, eps, K):
    """TRELLIS generator code: -log2 of prod_i (p_i + eps) / (1 + eps K), p_i the pair-gated normalised probability of the true outcome
    (word, ends-here) at step i; a step with no mass anywhere (Z = 0) costs log2 K (uniform over the K = 2 |vocab| outcomes)."""
    b = 0.0
    for p, _, _, _ in steps:
        p = 1.0 / K if p is None else p
        b -= math.log2((p + eps) / (1 + eps * K))
    return b
