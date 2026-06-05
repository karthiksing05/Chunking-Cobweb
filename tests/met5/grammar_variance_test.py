"""
Variance test — run the SOTA config N times and report the spread of
parse metrics. The cobweb C++ library seeds its RNG with std::random_device
at module load (see cobweb-private/src/helper.cpp:11-14), so every run
produces a different tree structure even with Python's seed fixed. This
script quantifies how much the parse metrics actually move from run to run.
"""

import os, sys, csv, json, glob, random, shutil, re
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import TRELLIS, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed

HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"
CONTEXT_LENGTH    = 3
PRIMITIVES_FIRST  = 200
SEED              = 13
N_RUNS            = 5

SOTA = {
    "content_alpha": 1e-6, "context_alpha": 1e-6,
    "content_bl_alpha": 10, "context_bl_alpha": 10,
    "weight_attr": False,
    "content_top_k": 7, "content_pool_depth": 4,
    "threshold": 30,
}

# Shared helpers
def _chunk_span(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            out.append(int(n.position_idx)); return
        for _, c in getattr(n, "children", []): w(c)
    w(node)
    return (min(out), max(out)) if out else (None, None)

def _walk_composites(node):
    if isinstance(node, PrimitiveParseNode): return
    if not getattr(node, "is_global_root", False): yield node
    for _, c in getattr(node, "children", []): yield from _walk_composites(c)

def _bracket_set(tree):
    return {(_chunk_span(c)[0], _chunk_span(c)[1])
            for c in _walk_composites(tree.global_root_node)
            if _chunk_span(c) != (None, None) and _chunk_span(c)[0] != _chunk_span(c)[1]}

# Load corpus once
hollow_corpus: list = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p) as f:
        try: data = json.load(f)
        except json.JSONDecodeError: continue
    if "sentence" in data and "merges" in data: hollow_corpus.append(data)

random.seed(SEED); np.random.seed(SEED)
random.shuffle(hollow_corpus)
_split = int(0.8 * len(hollow_corpus))
TRAIN, TEST = hollow_corpus[:_split], hollow_corpus[_split:]

random.seed(SEED + 1)
PRIM_SENTS = [generate("S", TEST_GRAMMAR1) for _ in range(PRIMITIVES_FIRST)]

def _run_once(seed_offset):
    random.seed(SEED + seed_offset); np.random.seed(SEED + seed_offset)
    cobweb_set_seed(SEED + seed_offset)   # ← seed cobweb's C++ RNG too
    cfg = SOTA
    trellis = TRELLIS(
        TEST_CORPUS1, context_length=CONTEXT_LENGTH,
        threshold=cfg["threshold"],
        content_alpha=cfg["content_alpha"], context_alpha=cfg["context_alpha"],
        content_bl_alpha=cfg["content_bl_alpha"], context_bl_alpha=cfg["context_bl_alpha"],
        bow=False, empty_weighting=True, chunk_context=False,
        weighting="binary", categorization_mode="dfs",
        content_top_k=cfg["content_top_k"], content_pool_depth=cfg["content_pool_depth"],
        content_weight_attr=cfg["weight_attr"], context_weight_attr=cfg["weight_attr"],
    )
    for s in PRIM_SENTS:
        trellis.parse_sentence(s, threshold=1e9, new_vocab=True, learning=True, debug=False)
    for hollow in TRAIN:
        tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
        tree.build_primitives(hollow["sentence"], threshold=cfg["threshold"])
        for m in hollow["merges"]:
            try: tree.apply_candidate(m["left"], m["right"])
            except: pass
        trellis.ltm.add_parse_tree(tree, shuffle=True, debug=False)

    total_tp = total_fp = total_fn = 0; exact = n_sents = 0
    for hollow in TEST:
        sentence = hollow["sentence"]
        if len(re.findall(r"[\w']+|[.,!?;]", sentence)) < 2: continue
        gold_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except: pass
        gold = _bracket_set(gold_tree)
        pred_tree = trellis.parse_sentence(
            sentence, threshold=cfg["threshold"], new_vocab=False, learning=False, debug=False)
        pred = _bracket_set(pred_tree)
        total_tp += len(gold & pred); total_fp += len(pred - gold); total_fn += len(gold - pred)
        if gold == pred and gold: exact += 1
        n_sents += 1

    P = total_tp / max(total_tp + total_fp, 1)
    R = total_tp / max(total_tp + total_fn, 1)
    F1 = 2*P*R / max(P+R, 1e-12)
    EM = exact / max(n_sents, 1)
    return {"P": P, "R": R, "F1": F1, "EM": EM}

print(f"Running SOTA config {N_RUNS} times to quantify variance...\n")
print(f"{'run':>4} {'P':>8} {'R':>8} {'F1':>8} {'EM':>8}")
runs = []
for i in range(N_RUNS):
    r = _run_once(i * 17)
    runs.append(r)
    print(f"{i+1:>4} {100*r['P']:>7.1f}% {100*r['R']:>7.1f}% "
          f"{100*r['F1']:>7.1f}% {100*r['EM']:>7.1f}%")

print()
print("Aggregate statistics:")
for k in ["P", "R", "F1", "EM"]:
    vals = [r[k] for r in runs]
    print(f"  {k:>3}: mean={100*statistics.mean(vals):.1f}%  "
          f"std={100*statistics.stdev(vals):.1f}pp  "
          f"min={100*min(vals):.1f}%  max={100*max(vals):.1f}%  "
          f"range={100*(max(vals)-min(vals)):.1f}pp")
