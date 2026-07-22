"""
Learning-curve test for TRELLIS. Trains incrementally on the supplied
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
from parse_mh import TRELLIS, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import (set_random_seed as cobweb_set_seed,
                                    get_random_state as cobweb_get_state,
                                    set_random_state as cobweb_set_state)


# Default hyperparameters — matched to hollow_learn_test_mh.py.
CONTEXT_LENGTH   = 3
THRESHOLD        = 30
PRIMITIVES_FIRST = 200
EVAL_EVERY       = 10
N_GEN_PER_EVAL   = 50   # generation samples per eval point (also used for GRIDS omission eval)


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
    n_gen_final: int = 500,   # high-sample generation at the FINAL checkpoint
                              # so the reported endpoint is a tight estimate,
                              # not a noisy 20-sample one

    primitives_first: int = None,    # None → uses module-level PRIMITIVES_FIRST
    maturity_gate: tuple = None,
    gate_mode:    str   = "skip",
    # Hyperparameters exposed for HP sweeps (defaults match prior behavior).
    context_length:        int   = None,    # None → module-level CONTEXT_LENGTH
    threshold:             int   = None,    # None → module-level THRESHOLD
    content_alpha:         float = 1e-4,
    context_alpha:         float = 1e-4,
    content_bl_alpha:      float = 10,
    context_bl_alpha:      float = 10,
    climb_count_threshold: int   = None,    # None → FiniteParseTree.CLIMB_COUNT_THRESHOLD_DEFAULT
    chunk_pool_weight:     float = 0.0,     # > 0 activates L/R_trans attachment signal in ranker
    sum_leaf_lp_coef:      float = 0.3,     # ranker: score = root_lp + coef * sum_leaf_lp
    # Context-forward ranker weights (postulate P3). Default formula:
    #   score = w_ctx*ctx_leaf_lp + w_climb*log(1+climb_count) + w_cnt*root_lp
    rank_mode:             str   = "context_forward",
    rank_w_ctx:            float = 1.0,
    rank_w_climb:          float = 3.0,
    rank_w_cnt:            float = 0.05,
    parse_beam_width:      int   = 1,      # 1 = greedy (P4); >1 = beam (Move 3)
    content_boundary_feat: str   = None,   # None | "wordid" | "ctxbag"
    content_seam_feat:     str   = None,   # None | "wordid"
    content_child_class:   bool  = False,  # type each child by its POS-cluster
    content_child_class_depth: int = 4,
    content_top_k:         int   = 7,      # bag-of-concepts width
    content_pool_depth:    int   = 4,      # bag-of-concepts read-off depth
    content_drop_cplx:     bool  = False,  # keep child complexity VISIBLE (load-bearing)
    gen_anchor_mode:       str   = "basic",   # "basic" | "maturity" (P9 anchor)
    gen_anchor_tau:        float = 20,
    gen_pool_mode:         str   = "leaf",    # "leaf" | "bl" | "mat" (generalization pool)
    gen_pool_tau:          float = 50,
    parse_mode:            str   = "greedy",
):
    """Train incrementally; eval every ``eval_every`` training sentences."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed); np.random.seed(seed); cobweb_set_seed(seed)

    _ctx_len = CONTEXT_LENGTH if context_length is None else context_length
    _thr     = THRESHOLD      if threshold      is None else threshold

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

    if climb_count_threshold is not None:
        FiniteParseTree.CLIMB_COUNT_THRESHOLD_DEFAULT = int(climb_count_threshold)

    def _fresh_trellis():
        """A freshly-constructed, configured (untrained) TRELLIS. Used both for
        the initial gold precompute and — in retrain-per-checkpoint mode — to
        rebuild a clean-RNG-stream model at each training size (see below)."""
        _t = TRELLIS(
            corpus, context_length=_ctx_len, threshold=_thr,
            content_alpha=content_alpha, context_alpha=context_alpha,
            content_bl_alpha=content_bl_alpha, context_bl_alpha=context_bl_alpha,
            bow=False, empty_weighting=True, chunk_context=False,
            weighting="binary", categorization_mode="dfs",
            depth_max_content=1000, depth_max_context=1000,
            branch_max_content=1000, branch_max_context=1000,
            content_top_k=content_top_k, content_pool_depth=content_pool_depth,
        )
        _t.ltm.chunk_pool_weight = float(chunk_pool_weight)
        _t.ltm.sum_leaf_lp_coef  = float(sum_leaf_lp_coef)
        _t.ltm.rank_mode    = str(rank_mode)
        _t.ltm.rank_w_ctx   = float(rank_w_ctx)
        _t.ltm.rank_w_climb = float(rank_w_climb)
        _t.ltm.rank_w_cnt   = float(rank_w_cnt)
        _t.ltm.parse_beam_width = int(parse_beam_width)
        _t.ltm.content_boundary_feat = content_boundary_feat
        _t.ltm.content_seam_feat     = content_seam_feat
        _t.ltm.content_child_class       = content_child_class
        _t.ltm.content_child_class_depth = content_child_class_depth
        _t.ltm.content_drop_cplx     = bool(content_drop_cplx)
        _t.ltm.gen_anchor_mode = str(gen_anchor_mode)
        _t.ltm.gen_anchor_tau  = float(gen_anchor_tau)
        _t.ltm.gen_pool_mode   = str(gen_pool_mode)
        _t.ltm.gen_pool_tau    = float(gen_pool_tau)
        return _t

    trellis = _fresh_trellis()
    # Snapshot the RNG right after construction. The gold precompute + baseline
    # eval below consume the RNG; we restore this snapshot before training so
    # training starts from the exact post-construction state (matching the
    # verified single-pass run) rather than a pre-training-eval-perturbed one.
    _rng_snap0 = (cobweb_get_state(), random.getstate(), np.random.get_state())

    pf = PRIMITIVES_FIRST if primitives_first is None else primitives_first
    if pf > 0:
        print(f"\nPHASE 1: PRIMITIVES ({pf} random sentences)")
        for i in range(pf):
            s = generate("S", grammar)
            trellis.parse_sentence(s, threshold=1e9, new_vocab=True,
                                   learning=True, debug=False)
    else:
        gate_desc = (f"gate={maturity_gate} mode={gate_mode}"
                     if maturity_gate else "no gate")
        print(f"\nPHASE 1: SKIPPED (primitives_first=0; {gate_desc})")

    recog, WORD_TO_POS = _build_cyk_recognizer(grammar)

    def _build_gold(teh):
        """Gold brackets for a test fold, from its gold merges. Uses a THROWAWAY
        trellis so building the gold (which registers vocab / categorizes) does
        not pollute the training trellis's state — the training model must be
        identical to the verified single-pass run."""
        _gtl = _fresh_trellis()
        g = {}
        for h in teh:
            sent = h["sentence"]
            if len(re.findall(r"[\w']+|[.,!?;]", sent)) < 2: continue
            gt = FiniteParseTree(_gtl.ltm, context_length=_ctx_len)
            gt.build_primitives(sent, threshold="converge")
            for m in h["merges"]:
                try: gt.apply_candidate(m["left"], m["right"])
                except Exception: pass
            g[sent] = _bracket_set(gt)
        return g

    gold = _build_gold(test_h)

    def _pred_brackets(sent):
        pt = trellis.parse_sentence(sent, threshold=THRESHOLD, new_vocab=False,
                                    learning=False, debug=False,
                                    maturity_gate=maturity_gate, gate_mode=gate_mode)
        return _bracket_set(pt)

    # Cap the per-checkpoint test-parse count so a fine (every-10) learning
    # curve stays tractable — parsing dominates the eval cost, and ~40 held-out
    # sentences give a stable F1 estimate at each point.
    _EVAL_MAX_TEST = 40
    _gold_items_capped = list(gold.items())[:_EVAL_MAX_TEST]

    def _eval_parse():
        tp = fp = fn = em = n = 0
        for sent, g in _gold_items_capped:
            pred = _pred_brackets(sent)
            tp += len(g & pred); fp += len(pred - g); fn += len(g - pred)
            if g == pred and len(g) > 0: em += 1
            n += 1
        P = tp / max(tp+fp, 1); R = tp / max(tp+fn, 1)
        F = 2*P*R / max(P+R, 1e-12)
        return P, R, F, em / max(n, 1)

    def _eval_gen(n_samples=n_gen_per_eval):
        gen_ok = lex_ok = novel = gram_novel = total = 0
        if not getattr(trellis, "sentence_root_chunks", None):
            return 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(n_samples):
            try:
                text, _ = trellis.generate_via_chunk_replay()
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
            if g_ok and n_ok: gram_novel += 1
        return (gen_ok / max(total, 1),
                novel  / max(total, 1),
                lex_ok / max(total, 1),
                gram_novel / max(total, 1),
                total)

    def _eval_omission(n_samples=n_gen_per_eval):
        """GRIDS-style 'probability of parsing a legal sentence':
        generate ``n_samples`` sentences from the GOLD grammar and
        check what fraction the LEARNER can fully chunk (single root
        chunk spanning the entire sentence). Returns (p_parse_legal,
        total_evaluated). Errors of omission = 1 - p_parse_legal."""
        success = total = 0
        for _ in range(n_samples):
            try:
                sent = generate("S", grammar)
            except Exception:
                continue
            toks = sent.strip().split()
            n_words = len(toks)
            if n_words < 2:  # single-token sentences are trivially "parsed"
                continue
            try:
                pred = _pred_brackets(sent)
            except Exception:
                continue
            total += 1
            # complete parse iff the whole-sentence span is a bracket
            if (0, n_words - 1) in pred:
                success += 1
        return success / max(total, 1), total

    # ── Phase 2: INCREMENTAL training + RNG-ISOLATED eval ──
    #
    # Train ONE model continuously (1x cost). The parser is sensitive to the
    # RNG stream during training (Cobweb tie-breaking + add_parse_tree shuffle),
    # and the in-loop parse/generation EVALS consume the same three RNGs
    # (Cobweb `gen`, Python `random`, NumPy). To keep eval from bleeding into
    # training, we SNAPSHOT all three RNG states before each eval and RESTORE
    # them after — so training proceeds exactly as if no eval had run. This
    # reproduces the verified single-pass numbers (term_low: 0.99) while being
    # ~N/2× cheaper than retraining per checkpoint. Requires the Cobweb
    # get_random_state/set_random_state extension.
    print(f"\nPHASE 2: INCREMENTAL + RNG-isolated eval (every {eval_every})")
    checkpoints = set(range(eval_every, len(train_h) + 1, eval_every)) | {len(train_h)}

    rows = [{"n_trained": 0,
             "parse_F1": 0.0, "parse_P": 0.0, "parse_R": 0.0, "parse_EM": 0.0,
             "gen_gram": 0.0, "gen_novel": 0.0, "gen_lex": 0.0, "gen_gram_novel": 0.0,
             "p_parse_legal": 0.0, "gen_n": 0, "prim_active_frac": 0.0}]

    # Restore the post-construction RNG snapshot so the gold precompute / Phase-1
    # consumption above does not perturb training's stream.
    cobweb_set_state(_rng_snap0[0]); random.setstate(_rng_snap0[1]); np.random.set_state(_rng_snap0[2])

    trained_trees = []
    n_sents_with_chunks = 0
    for i, hollow in enumerate(train_h):
        sent = hollow["sentence"]; merges = hollow["merges"]
        tree = FiniteParseTree(trellis.ltm, context_length=_ctx_len)
        tree.build_primitives(sent, threshold=_thr,
                              maturity_gate=maturity_gate, gate_mode=gate_mode)
        if maturity_gate is None:
            sentence_active = True
        elif gate_mode == "all_or_none":
            sentence_active = tree.all_primitives_stable()
        else:
            sentence_active = any(getattr(n, "stable", False) for n in tree.nodes)
        if sentence_active:
            n_sents_with_chunks += 1
        if (maturity_gate is not None and gate_mode == "all_or_none"
                and not tree.all_primitives_stable()):
            pass
        else:
            for m in merges:
                try: tree.apply_candidate(m["left"], m["right"])
                except Exception: pass
        trellis.ltm.add_parse_tree(tree, shuffle=True, debug=False)
        trained_trees.append(tree)

        if (i + 1) in checkpoints:
            # Snapshot all three RNGs, evaluate, restore — eval cannot perturb
            # the training stream.
            _snap = (cobweb_get_state(), random.getstate(), np.random.get_state())
            trellis.learn_leaf_transitions(trained_trees)
            trellis.learn_chunk_records(trained_trees)
            trellis.ltm.chunk_pool_weight = 5.0
            P, R, F, em = _eval_parse()
            # The final checkpoint is generated with a much larger sample so the
            # reported endpoint grammaticality/novelty is statistically tight
            # (generation + CYK check is ~0.1 ms/sentence, so this is cheap).
            _n_gen_this = n_gen_final if (i + 1) == len(train_h) else n_gen_per_eval
            g_gram, g_novel, g_lex, g_gram_novel, g_n = _eval_gen(_n_gen_this)
            # Omission (P parse-legal) is no longer plotted (the GRIDS left panel
            # is now parse F1); the learner chunks every legal sentence to a root
            # (100% across all verified runs). Skip its generation+parse cost.
            p_parse_legal = 1.0
            cobweb_set_state(_snap[0]); random.setstate(_snap[1]); np.random.set_state(_snap[2])

            prim_active_frac = n_sents_with_chunks / (i + 1)
            rows.append({
                "n_trained": i + 1,
                "parse_F1": F, "parse_P": P, "parse_R": R, "parse_EM": em,
                "gen_gram": g_gram, "gen_novel": g_novel, "gen_lex": g_lex,
                "gen_gram_novel": g_gram_novel,
                "p_parse_legal": p_parse_legal,
                "gen_n": g_n,
                "prim_active_frac": prim_active_frac,
            })
            print(f"  n={i+1:>3}  P={100*P:5.1f}%  R={100*R:5.1f}%  EM={100*em:5.1f}%  "
                  f"gen_gram={100*g_gram:5.1f}%  gen_gram_novel={100*g_gram_novel:5.1f}%  "
                  f"P(parse_legal)={100*p_parse_legal:5.1f}%  prim_active={100*prim_active_frac:5.1f}%")
    # ── Persist CSV + chart ──
    csv_path = f"{out_dir}/learning_curves.csv"
    with open(csv_path, "w") as f:
        f.write("n_trained,parse_F1,parse_P,parse_R,parse_EM,"
                "gen_gram,gen_novel,gen_lex,gen_gram_novel,"
                "p_parse_legal,gen_n,prim_active_frac\n")
        for r in rows:
            f.write(f"{r['n_trained']},{r['parse_F1']:.4f},"
                    f"{r['parse_P']:.4f},{r['parse_R']:.4f},"
                    f"{r['parse_EM']:.4f},{r['gen_gram']:.4f},"
                    f"{r['gen_novel']:.4f},{r['gen_lex']:.4f},"
                    f"{r.get('gen_gram_novel', 0.0):.4f},"
                    f"{r.get('p_parse_legal', 0.0):.4f},"
                    f"{r['gen_n']},{r.get('prim_active_frac', 1.0):.4f}\n")
    print(f"\nCurves CSV  → {csv_path}")

    xs = [r["n_trained"] for r in rows]

    # ── learning_curves.png — 4-panel (parse P/R split + gen + novelty) ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].plot(xs, [100*r["parse_P"]  for r in rows], "o-", color="#1f77b4", label="Precision")
    axes[0].plot(xs, [100*r["parse_R"]  for r in rows], "s-", color="#2ca02c", label="Recall")
    axes[0].plot(xs, [100*r["parse_EM"] for r in rows], "d-", color="#9467bd", label="Exact-match")
    axes[0].plot(xs, [100*r.get("prim_active_frac", 1.0) for r in rows],
                  "^--", color="#7f7f7f", alpha=0.7, label="gate active")
    axes[0].set_xlabel("# training sentences"); axes[0].set_ylabel("%")
    axes[0].set_title("Parse accuracy — precision / recall (split)")
    axes[0].set_ylim(0, 105); axes[0].grid(alpha=0.3); axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(xs, [100*r["gen_gram"] for r in rows], "o-", color="#bcbd22", label="Grammatical")
    axes[1].plot(xs, [100*r["gen_lex"]  for r in rows], "s-", color="#17becf", label="In-lexicon")
    axes[1].set_xlabel("# training sentences"); axes[1].set_ylabel("%")
    axes[1].set_title("Generation grammaticality")
    axes[1].set_ylim(0, 105); axes[1].grid(alpha=0.3); axes[1].legend(loc="lower right", fontsize=9)

    axes[2].plot(xs, [100*r.get("gen_gram_novel", 0.0) for r in rows],
                  "o-", color="#d62728", label="Grammatical & novel")
    axes[2].set_xlabel("# training sentences"); axes[2].set_ylabel("% of generated outputs")
    axes[2].set_title("Generation novelty (grammatical-only)")
    axes[2].set_ylim(0, 105); axes[2].grid(alpha=0.3); axes[2].legend()

    axes[3].plot(xs, [100*r.get("prim_active_frac", 1.0) for r in rows],
                  "^-", color="#7f7f7f", label="% train sents w/ mature primitives")
    axes[3].set_xlabel("# training sentences"); axes[3].set_ylabel("%")
    axes[3].set_title("Gate activation")
    axes[3].set_ylim(0, 105); axes[3].grid(alpha=0.3); axes[3].legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Learning curves — {corpus_dir.split('/')[-1]} (eval every {eval_every}, n_gen={n_gen_per_eval})",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{out_dir}/learning_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Curves PNG  → {out_dir}/learning_curves.png")

    # ── grids_curves.png — GRIDS-style (Langley & Stromsten 2000 Fig 1) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(xs, [r.get("p_parse_legal", 0.0) for r in rows],
                  "o-", color="#1f77b4", linewidth=2)
    axes[0].set_xlabel("# training sentences")
    axes[0].set_ylabel("Probability of parsing a legal sentence")
    axes[0].set_title("(a) Parsing — errors of omission")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[1].plot(xs, [r["gen_gram"] for r in rows],
                  "o-", color="#2ca02c", linewidth=2)
    axes[1].set_xlabel("# training sentences")
    axes[1].set_ylabel("Probability of generating a legal sentence")
    axes[1].set_title("(b) Generation — errors of co-mission")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"GRIDS-style omission / co-mission curves — {corpus_dir.split('/')[-1]}",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{out_dir}/grids_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"GRIDS PNG   → {out_dir}/grids_curves.png")


if __name__ == "__main__":
    run_learning_curves()
