"""
Learning-curve test for WEBSTER. Trains incrementally on the supplied
hollow corpus, snapshotting evaluation metrics every ``EVAL_EVERY``
training sentences:

    • Parse F1 on held-out test fold
    • From-scratch generation: grammaticality rate
    • From-scratch generation: novelty rate (fraction not in training)

Writes ``learning_curves.csv`` and ``learning_curves.png`` to ``OUT_DIR``.
Designed to share configuration with ``hollow_learn_test_mh.py`` (same
hyperparameters, same 80/20 split convention).

Usage:
    python unittests/learning_curves_test.py

Or programmatically:
    from unittests.learning_curves_test import run_learning_curves
    run_learning_curves(
        corpus_dir="data/cfg_grammar_1",
        out_dir="confs/acs-26/grammar_1",
        grammar=TEST_GRAMMAR1, corpus=TEST_CORPUS1,
    )
"""
import os, sys, json, random, glob, re, shutil
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.cfg import (generate, generate_with_merges,
                      TEST_GRAMMAR1, TEST_CORPUS1,
                      TEST_GRAMMAR3, TEST_CORPUS3)
from parse_mh import WEBSTER, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed


# Default hyperparameters — matched to hollow_learn_test_mh.py.
CONTEXT_LENGTH   = 3
THRESHOLD        = 30
PRIMITIVES_FIRST = 200
EVAL_EVERY       = 10
N_GEN_PER_EVAL   = 30   # smaller than hollow_learn's 50 for speed


# ---------- helpers reused from hollow_learn (re-declared here to avoid
# importing the script body, which still does heavy work at module top) --
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
    for _, c in getattr(node, "children", []):
        yield from _walk_composites(c)


def _bracket_set(tree):
    s = set()
    for c in _walk_composites(tree.global_root_node):
        a, b = _chunk_span(c)
        if a is not None and a != b: s.add((a, b))
    return s


def _build_cyk_recognizer(grammar):
    """Same CYK that hollow_learn_test_mh uses to score generation."""
    from collections import defaultdict
    POS_LIST = ["Det", "N", "Adj", "V", "P", "RelPro"]
    WORD_TO_POS = {}
    for pos in POS_LIST:
        for prod in grammar.get(pos, []):
            for w in prod:
                WORD_TO_POS[w] = pos

    def recog(tokens, start="S"):
        n = len(tokens)
        if n == 0: return False
        term_lhs = defaultdict(set)
        for lhs, prods in grammar.items():
            for prod in prods:
                if len(prod) == 1 and (prod[0] in WORD_TO_POS or prod[0] not in grammar):
                    term_lhs[prod[0]].add(lhs)
        chart = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
        def _unary(s):
            changed = True
            while changed:
                changed = False
                for lhs, prods in grammar.items():
                    for prod in prods:
                        if len(prod) == 1 and prod[0] in s and lhs not in s:
                            s.add(lhs); changed = True
        for i, t in enumerate(tokens):
            chart[i][i+1] = set(term_lhs.get(t, set()))
            _unary(chart[i][i+1])
        for span in range(2, n+1):
            for i in range(n - span + 1):
                j = i + span
                for k in range(i+1, j):
                    L, R = chart[i][k], chart[k][j]
                    if not L or not R: continue
                    for lhs, prods in grammar.items():
                        for prod in prods:
                            if (len(prod) == 2 and prod[0] in L and prod[1] in R):
                                chart[i][j].add(lhs)
                if span >= 3:
                    for k2 in range(i+1, j-1):
                        for k in range(k2+1, j):
                            a, b, c = chart[i][k2], chart[k2][k], chart[k][j]
                            if not a or not b or not c: continue
                            for lhs, prods in grammar.items():
                                for prod in prods:
                                    if (len(prod) == 3
                                            and prod[0] in a and prod[1] in b
                                            and prod[2] in c):
                                        chart[i][j].add(lhs)
                _unary(chart[i][j])
        return start in chart[0][n]

    return recog, WORD_TO_POS


def run_learning_curves(
    corpus_dir: str = "data/test_hollow_grammar_1",
    out_dir:    str = "unittests/learning_curves_test",
    grammar:    dict = TEST_GRAMMAR1,
    corpus:     list = TEST_CORPUS1,
    seed:       int  = 13,
    eval_every: int  = EVAL_EVERY,
    n_gen_per_eval: int = N_GEN_PER_EVAL,
):
    """Train incrementally; eval every ``eval_every`` training sentences."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed); np.random.seed(seed); cobweb_set_seed(seed)

    # ── Load corpus ──
    paths = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))
    corpus_data = []
    for p in paths:
        with open(p) as f:
            try: d = json.load(f)
            except: continue
        if "sentence" in d and "merges" in d: corpus_data.append(d)
    print(f"Loaded {len(corpus_data)} parse trees from {corpus_dir}")
    if not corpus_data:
        print("[ERROR] No data found.")
        return

    random.shuffle(corpus_data)
    sp = int(0.8 * len(corpus_data))
    train_h, test_h = corpus_data[:sp], corpus_data[sp:]
    print(f"  Split: train={len(train_h)}  test={len(test_h)}")

    # Set of training sentences for novelty checking.
    train_sentences = {h["sentence"].strip() for h in train_h}

    webster = WEBSTER(
        corpus, context_length=CONTEXT_LENGTH, threshold=THRESHOLD,
        content_alpha=1e-4, context_alpha=1e-4,
        content_bl_alpha=10, context_bl_alpha=10,
        bow=False, empty_weighting=True, chunk_context=False,
        weighting="binary", categorization_mode="dfs",
        depth_max_content=1000, depth_max_context=1000,
        branch_max_content=1000, branch_max_context=1000,
        content_top_k=7, content_pool_depth=4,
    )

    print(f"\nPHASE 1: PRIMITIVES ({PRIMITIVES_FIRST} random sentences)")
    for i in range(PRIMITIVES_FIRST):
        s = generate("S", grammar)
        webster.parse_sentence(s, threshold=1e9, new_vocab=True,
                               learning=True, debug=False)

    recog, WORD_TO_POS = _build_cyk_recognizer(grammar)

    # Precompute gold brackets for the test fold (once).
    gold = {}
    for h in test_h:
        sent = h["sentence"]
        if len(re.findall(r"[\w']+|[.,!?;]", sent)) < 2: continue
        gt = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        gt.build_primitives(sent, threshold="converge")
        for m in h["merges"]:
            try: gt.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold[sent] = _bracket_set(gt)

    def _eval_parse():
        tp = fp = fn = em = n = 0
        for sent, g in gold.items():
            pt = webster.parse_sentence(sent, threshold=THRESHOLD,
                                        new_vocab=False, learning=False,
                                        debug=False)
            pred = _bracket_set(pt)
            tp += len(g & pred); fp += len(pred - g); fn += len(g - pred)
            if g == pred and len(g) > 0: em += 1
            n += 1
        P = tp / max(tp+fp, 1); R = tp / max(tp+fn, 1)
        F = 2*P*R / max(P+R, 1e-12)
        return P, R, F, em / max(n, 1)

    def _eval_gen(n_samples=n_gen_per_eval):
        gen_ok = lex_ok = novel = total = 0
        if not getattr(webster, "sentence_root_chunks", None):
            return 0.0, 0.0, 0.0, 0
        for _ in range(n_samples):
            try:
                text, _ = webster.generate_via_chunk_replay()
            except Exception:
                continue
            toks = text.split()
            if not toks: continue
            l_ok = all(t in WORD_TO_POS for t in toks)
            g_ok = l_ok and recog(toks)
            n_ok = text.strip() not in train_sentences
            total += 1
            if l_ok:  lex_ok += 1
            if g_ok:  gen_ok += 1
            if n_ok:  novel  += 1
        return (gen_ok / max(total, 1),
                novel  / max(total, 1),
                lex_ok / max(total, 1),
                total)

    # ── Phase 2: incremental train + eval ──
    print(f"\nPHASE 2: INCREMENTAL TRAINING ({len(train_h)} sentences, "
          f"eval every {eval_every})")
    trained_trees = []
    rows = []   # learning curve: list of dicts
    # Initial eval at n_trained=0 (no chunks yet, gen unavailable).
    P0, R0, F0, em0 = _eval_parse()
    rows.append({"n_trained": 0,
                 "parse_F1": F0, "parse_P": P0, "parse_R": R0, "parse_EM": em0,
                 "gen_gram": 0.0, "gen_novel": 0.0, "gen_lex": 0.0,
                 "gen_n": 0})
    print(f"  n=  0  F1={100*F0:5.1f}%  EM={100*em0:5.1f}%  gen=n/a")

    for i, hollow in enumerate(train_h):
        sent  = hollow["sentence"]
        merges= hollow["merges"]
        tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        tree.build_primitives(sent, threshold=THRESHOLD)
        for m in merges:
            try: tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)
        trained_trees.append(tree)

        if (i + 1) % eval_every == 0 or (i + 1) == len(train_h):
            # Re-mine leaf transitions + chunk records on current trees.
            webster.learn_leaf_transitions(trained_trees)
            webster.learn_chunk_records(trained_trees)
            webster.ltm.chunk_pool_weight = 5.0
            P, R, F, em = _eval_parse()
            g_gram, g_novel, g_lex, g_n = _eval_gen()
            rows.append({
                "n_trained": i + 1,
                "parse_F1": F, "parse_P": P, "parse_R": R, "parse_EM": em,
                "gen_gram": g_gram, "gen_novel": g_novel, "gen_lex": g_lex,
                "gen_n": g_n,
            })
            print(f"  n={i+1:>3}  F1={100*F:5.1f}%  EM={100*em:5.1f}%  "
                  f"gen_gram={100*g_gram:5.1f}%  "
                  f"gen_novel={100*g_novel:5.1f}%  (n={g_n})")

    # ── Persist CSV + chart ──
    csv_path = f"{out_dir}/learning_curves.csv"
    with open(csv_path, "w") as f:
        f.write("n_trained,parse_F1,parse_P,parse_R,parse_EM,"
                "gen_gram,gen_novel,gen_lex,gen_n\n")
        for r in rows:
            f.write(f"{r['n_trained']},{r['parse_F1']:.4f},"
                    f"{r['parse_P']:.4f},{r['parse_R']:.4f},"
                    f"{r['parse_EM']:.4f},{r['gen_gram']:.4f},"
                    f"{r['gen_novel']:.4f},{r['gen_lex']:.4f},"
                    f"{r['gen_n']}\n")
    print(f"\nCurves CSV  → {csv_path}")

    xs = [r["n_trained"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Parse panel
    axes[0].plot(xs, [100*r["parse_F1"] for r in rows], "o-", color="#1f77b4", label="Parse F1")
    axes[0].plot(xs, [100*r["parse_EM"] for r in rows], "s-", color="#9467bd", label="Exact-match")
    axes[0].set_xlabel("# training sentences")
    axes[0].set_ylabel("% on held-out test")
    axes[0].set_title("Parse accuracy")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3); axes[0].legend()
    # Gen-gram panel
    axes[1].plot(xs, [100*r["gen_gram"] for r in rows], "o-", color="#bcbd22", label="Grammatical")
    axes[1].plot(xs, [100*r["gen_lex"]  for r in rows], "s-", color="#17becf", label="In-lexicon")
    axes[1].set_xlabel("# training sentences")
    axes[1].set_ylabel("% of generated outputs")
    axes[1].set_title("Generation grammaticality")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3); axes[1].legend()
    # Gen-novelty panel
    axes[2].plot(xs, [100*r["gen_novel"] for r in rows], "o-", color="#d62728", label="Novel")
    axes[2].set_xlabel("# training sentences")
    axes[2].set_ylabel("% of generated outputs")
    axes[2].set_title("Generation novelty (not in train)")
    axes[2].set_ylim(0, 105)
    axes[2].grid(alpha=0.3); axes[2].legend()
    fig.suptitle(
        f"Learning curves — {corpus_dir.split('/')[-1]} (eval every {eval_every})",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{out_dir}/learning_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Curves PNG  → {out_dir}/learning_curves.png")


if __name__ == "__main__":
    run_learning_curves()
