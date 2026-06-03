"""
met6 diagnostic — how far is UNSUPERVISED from SUPERVISED?

Head-to-head on the SAME corpus + SAME WEBSTER config:
  - SUPERVISED   : train by replaying gold merges (the acs-26 hollow_learn
                   recipe) — the reference.
  - UNSUPERVISED : train on sentences only, parse_sentence(learning=True),
                   τ drives chunk commitment (no gold merges).

Both evaluated identically on a held-out test fold:
  - parse F1 / EM vs gold brackets (the supervised benchmark metric)
  - generation from-scratch grammaticality / novelty (CYK legality oracle)

Goal: find unsupervised settings whose parse F1 + gen-grammaticality
approach the supervised reference. Sweeps grammar × n_train × τ.

Usage:
    PYTHONHASHSEED=0 python tests/met6/diag_supervised_gap.py
"""
import os, sys, random, re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
from util.cfg import (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL,
                      TEST_GRAMMAR_MED, TEST_CORPUS_MED,
                      generate_with_merges)
from parse_mh import WEBSTER, FiniteParseTree, PrimitiveParseNode
from unittests.learning_curves_test import _build_cyk_recognizer, _bracket_set

MATURITY_GATE = ("root_log_prob", -12.0)
CONTEXT_LENGTH = 3
CONTENT_ALPHA = 1e-4
SEED = 13

CONFIGS = [
    # (grammar, corpus, label, n_train, flatten)
    ("SMALL", TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL, 150, ("VP",)),
    ("MED",   TEST_GRAMMAR_MED,   TEST_CORPUS_MED,   150, ("VP",)),
]
TAUS = [2, 5]
N_EPOCHS_UNSUP = 6
N_GEN = 60


def make_webster(corpus):
    return WEBSTER(
        corpus, context_length=CONTEXT_LENGTH, threshold=30,
        content_alpha=CONTENT_ALPHA, context_alpha=1e-4,
        content_bl_alpha=10, context_bl_alpha=10,
        bow=False, empty_weighting=True, chunk_context=False,
        weighting="binary", categorization_mode="dfs",
        depth_max_content=1000, depth_max_context=1000,
        branch_max_content=1000, branch_max_context=1000,
        content_top_k=7, content_pool_depth=4,
    )


def gen_corpus(grammar, n, seed, flatten):
    random.seed(seed)
    seen, out = set(), []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        text, merges = generate_with_merges("S", grammar, flatten_at_parent=flatten)
        text = text.strip()
        if not text or text in seen or len(re.findall(r"[\w']+|[.,!?;]", text)) < 3:
            continue
        seen.add(text)
        out.append({"sentence": text, "merges": merges})
    return out


def gold_brackets(webster, h):
    gt = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    gt.build_primitives(h["sentence"], threshold="converge")
    for m in h["merges"]:
        try: gt.apply_candidate(m["left"], m["right"])
        except Exception: pass
    return _bracket_set(gt)


def eval_model(webster, test, gold, tau, recog, word_to_pos, train_set, trained_trees):
    # parse F1 / EM vs gold
    tp = fp = fn = em = n = 0
    for h in test:
        s = h["sentence"]
        pt = webster.parse_sentence(s, threshold=tau, climb_count_threshold=tau,
                                    new_vocab=False, learning=False,
                                    maturity_gate=MATURITY_GATE, gate_mode="skip")
        pred = _bracket_set(pt); g = gold[s]
        tp += len(g & pred); fp += len(pred - g); fn += len(g - pred)
        if g == pred and len(g) > 0: em += 1
        n += 1
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-12); EM = em / max(n, 1)

    # generation grammaticality / novelty
    webster.learn_leaf_transitions(trained_trees)
    webster.learn_chunk_records(trained_trees)
    webster.ltm.chunk_pool_weight = 5.0
    made = gram = novel = 0
    if getattr(webster, "sentence_root_chunks", None):
        for _ in range(N_GEN):
            try: text, _ = webster.generate_via_chunk_replay()
            except Exception: continue
            toks = text.split()
            if not toks: continue
            made += 1
            if all(t in word_to_pos for t in toks) and recog(toks): gram += 1
            if text.strip() not in train_set: novel += 1
    return {"F1": F, "EM": EM, "gen_gram": gram / max(made, 1),
            "gen_novel": novel / max(made, 1)}


def train_supervised(corpus, train):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = make_webster(corpus)
    for h in train:
        for tok in re.findall(r"[\w']+|[.,!?;]", h["sentence"]):
            web.ltm.add_to_vocab(tok)
    trees = []
    for h in train:
        t = FiniteParseTree(web.ltm, context_length=CONTEXT_LENGTH)
        t.build_primitives(h["sentence"], threshold="converge")
        for m in h["merges"]:
            try: t.apply_candidate(m["left"], m["right"])
            except Exception: pass
        web.ltm.add_parse_tree(t, shuffle=True)
        trees.append(t)
    return web, trees


def train_unsupervised(corpus, train, tau, epochs):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = make_webster(corpus)
    trees = []
    for ep in range(epochs):
        trees = []
        for h in train:
            pt = web.parse_sentence(h["sentence"], threshold=tau, climb_count_threshold=tau,
                                    new_vocab=(ep == 0), learning=True,
                                    maturity_gate=MATURITY_GATE, gate_mode="skip")
            trees.append(pt)
    return web, trees


def main():
    print("=== met6 supervised-gap probe — parse F1 / EM / gen-gram, same corpus+config ===\n")
    print(f"{'grammar':>7} {'mode':>14} {'τ':>3} {'F1':>6} {'EM':>6} {'genGram':>8} {'genNovel':>8}")
    for gname, grammar, corpus, n_train, flatten in CONFIGS:
        recog, word_to_pos = _build_cyk_recognizer(grammar)
        data = gen_corpus(grammar, n_train + 40, seed=SEED, flatten=flatten)
        random.seed(SEED); random.shuffle(data)
        train, test = data[:n_train], data[n_train:n_train + 40]
        train_set = {h["sentence"].strip() for h in train}

        # gold brackets need a vocab; build once on a scratch webster
        scratch = make_webster(corpus)
        for h in data:
            for tok in re.findall(r"[\w']+|[.,!?;]", h["sentence"]):
                scratch.ltm.add_to_vocab(tok)
        gold = {h["sentence"]: gold_brackets(scratch, h) for h in test}

        # supervised reference
        web_s, trees_s = train_supervised(corpus, train)
        r = eval_model(web_s, test, gold, 30, recog, word_to_pos, train_set, trees_s)
        print(f"{gname:>7} {'SUPERVISED':>14} {'—':>3} {100*r['F1']:>5.1f} "
              f"{100*r['EM']:>5.1f} {100*r['gen_gram']:>7.1f} {100*r['gen_novel']:>7.1f}", flush=True)

        # unsupervised at each τ
        for tau in TAUS:
            web_u, trees_u = train_unsupervised(corpus, train, tau, N_EPOCHS_UNSUP)
            r = eval_model(web_u, test, gold, tau, recog, word_to_pos, train_set, trees_u)
            print(f"{gname:>7} {'UNSUPERVISED':>14} {tau:>3} {100*r['F1']:>5.1f} "
                  f"{100*r['EM']:>5.1f} {100*r['gen_gram']:>7.1f} {100*r['gen_novel']:>7.1f}", flush=True)
        print()


if __name__ == "__main__":
    main()
