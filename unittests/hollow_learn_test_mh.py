"""
Hollow Learning Test (Multi-Hierarchy) — trains WEBSTER from a corpus of
human-annotated "hollow" parse trees (merge recipes), then evaluates
the UNSUPERVISED "climbing-ancestor" parse strategy:

    Stage 1 (gate): For each candidate pair, descend the content tree
        to its leaf, then walk UP until finding the most-specific
        ancestor whose count > THRESHOLD. If the walk reaches root,
        the bag isn't well-supported anywhere — reject.

    Stage 2 (rank): Argmax of the climbed ancestor's
        log_prob_instance(bag) — how well the candidate fits the
        well-supported cluster it would commit to.

    Stage 3 (stop): When no candidate passes the gate, the parse
        terminates ("create chunks until we can no longer do so").

This matches the user's mental model: representations + threshold over
them + ample-count gate + chunk-until-done. Fully unsupervised.

The supervised chunk-class probe in
`tests/met5/grammar_decoding_test.py` Phase 3a hits 99% — that
confirms the cobweb-tree representations ARE separable by chunk class.
The climbing-ancestor mechanism is the unsupervised counterpart: it
finds the well-supported cluster (any depth, not necessarily basic-
level) without needing class labels.

  (1)  Ground-truth PARSE accuracy on held-out hollow sentences.
       Bracket P/R/F1 between WEBSTER's auto-parse and the gold
       brackets. Order-independent (bracket SET).

  (1b) STEP-PICK accuracy. At each gold-merge step, run the climbing-
       ancestor gate + ranker and ask: is the top pick a gold
       candidate? Threshold-test reference for Phase 4b (no climbing
       gate): ~93% step-pick.

  (2) GRAMMAR (chunk-class) accuracy via a Cobweb-Discrete probe.
      Same protocol as tests/met5/grammar_decoding_test.py Phase 3a:
      train a probe on chunk content_instance bags from the train fold,
      predict chunk classes on the held-out test fold, report per-class
      precision/recall/F1.  This is the "strong decoding style" — it
      treats each bag as a discrete attribute → value-set instance and
      lets Cobweb cluster classes from the representations alone.

  (3) GENERATION accuracy under that same decoding style.
      Generation already flows through FiniteParseTree._resolve_bag's
      frontier-categorize routine (see src/parse_mh.py:_resolve_bag),
      which scores every candidate content-ref by bag-weighted log-prob
      across the K canonical context nodes — same posterior-weighted
      decoding the probe in (2) certifies as discriminative.  We then
      score each generation by:
        - From-scratch: grammaticality against TEST_GRAMMAR1 (CYK).
        - Single-mask completion: exact-token recovery vs the gold
          held-out token.
"""

import sys, os, shutil, json, random, glob, re
from collections import Counter, defaultdict

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import WEBSTER, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import CobwebDiscreteTree, set_random_seed as cobweb_set_seed


def run_hollow_learn(
    corpus_dir: str = "data/test_hollow_grammar_1",
    out_dir:    str = "unittests/hollow_learn_test_mh",
    grammar:    dict = TEST_GRAMMAR1,
    corpus:     list = TEST_CORPUS1,
    seed:       int  = 13,
    primitives_first: int = 200,
):
    """Run the full hollow_learn pipeline against ``corpus_dir`` /
    ``grammar`` / ``corpus``, writing outputs to ``out_dir``.

    Refactored from the original script body so that ``confs/acs-26``
    can call this with synthetic CFG-derived datasets.
    """
    # ── Configuration ──────────────────────────────────────────────────────────
    OUT_DIR = out_dir
    HOLLOW_CORPUS_DIR = corpus_dir
    VIZ_INTERMEDIATES = True    # per-step intermediate parse / LTM viz

    CONTEXT_LENGTH    = 3
    THRESHOLD         = 30
    PRIMITIVES_FIRST  = primitives_first
    SEED              = seed
    PROBE_ALPHA       = 1e-3

    random.seed(SEED)
    np.random.seed(SEED)
    cobweb_set_seed(SEED)   # seed the C++ cobweb RNG for reproducibility

    # corpus injected via parameter
    # grammar injected via parameter

    # Word → POS for chunk classification.
    # Derive POS classes from the grammar: any non-terminal whose
    # productions are all single-element terminal lists. Lets the
    # test work with grammars beyond TEST_GRAMMAR1 (e.g. TEST_GRAMMAR3
    # adds RelPro). Falls back to the canonical 5-POS list if the
    # grammar's surface productions don't fit this rule.
    POS_LIST = []
    for sym, prods in grammar.items():
        if not prods:
            continue
        if all(len(p) == 1 and p[0] not in grammar for p in prods):
            POS_LIST.append(sym)
    if not POS_LIST:
        POS_LIST = ["Det", "N", "Adj", "V", "P"]
    WORD_TO_POS = {}
    for pos in POS_LIST:
        for prod in grammar[pos]:
            for w in prod:
                WORD_TO_POS[w] = pos
    CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]
    ALL_LABELS   = POS_LIST + CHUNK_LABELS + ["OTHER"]

    # ── Setup ──────────────────────────────────────────────────────────────────
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load hollow corpus ─────────────────────────────────────────────────────
    hollow_paths = sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json")))
    hollow_corpus: list[dict] = []
    for p in hollow_paths:
        with open(p, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON: {p}")
                continue
        if "sentence" in data and "merges" in data:
            hollow_corpus.append(data)

    print(f"Loaded {len(hollow_corpus)} hollow parse trees from {HOLLOW_CORPUS_DIR}")
    if not hollow_corpus:
        print("[ERROR] No hollow parse trees found.")
        sys.exit(1)

    # 80/20 sentence-level split (matches grammar_threshold_test / grammar_decoding_test).
    random.shuffle(hollow_corpus)
    _split = int(0.8 * len(hollow_corpus))
    train_hollow = hollow_corpus[:_split]
    test_hollow  = hollow_corpus[_split:]
    print(f"  Split: train={len(train_hollow)}  test={len(test_hollow)}")

    # ── Train/test memorization audit ─────────────────────────────
    # Verify the test fold is genuinely held out (no test sentence
    # appears verbatim in train), then quantify STRUCTURAL novelty
    # (test sentences whose POS-tag sequence was NEVER seen in
    # training). Structural novelty matters because a parser trained
    # on, say, "the X V the Y" sentences should generalize to NEW
    # surface forms with the same POS structure.
    _train_sents = {h["sentence"].strip() for h in train_hollow}
    _test_sents  = {h["sentence"].strip() for h in test_hollow}
    _overlap = _train_sents & _test_sents
    assert not _overlap, (
        f"Train/test split has {len(_overlap)} duplicate sentences "
        f"— memorization risk! Examples: {list(_overlap)[:3]}")
    print(f"  Train∩Test exact overlap: 0 sentences (verified disjoint)")

    def _pos_seq(text):
        return tuple(WORD_TO_POS.get(w, "?") for w in text.split())

    train_pos = {_pos_seq(s) for s in _train_sents}
    test_pos  = {_pos_seq(s) for s in _test_sents}
    pos_novel_test = sum(1 for s in _test_sents
                         if _pos_seq(s) not in train_pos)
    print(f"  Test sentences with novel POS structure: "
          f"{pos_novel_test}/{len(_test_sents)} "
          f"({100*pos_novel_test/max(len(_test_sents),1):.1f}%)")
    print(f"  Unique POS sequences in train: {len(train_pos)}, "
          f"test: {len(test_pos)}")

    # ── Initialise WEBSTER (identical hyperparameters to grammar_threshold_test) ──
    webster = WEBSTER(
        corpus,
        context_length=CONTEXT_LENGTH,
        threshold=THRESHOLD,
        # alpha=1e-4 chosen via tests/met5/grammar_param_sweep_test.py — tied
        # for top F1 with 1e-6 but better EM (65.2% vs 60.9%) and step-pick
        # (98.3% vs 96.6%). bl_alpha=10 is a hard floor: 1.0 breaks EPMI from
        # the leaf side, 100 collapses it.
        content_alpha=1e-4,
        context_alpha=1e-4,
        content_bl_alpha=10,
        context_bl_alpha=10,
        bow=False,
        empty_weighting=True,
        chunk_context=False,
        weighting="binary",
        categorization_mode="dfs",
        depth_max_content=1000,
        depth_max_context=1000,
        branch_max_content=1000,
        branch_max_context=1000,
        content_top_k=7,
        content_pool_depth=4,
    )

    # ── Phase 1: primitives-only on random sentences ───────────────────────────
    print(f"\n=== PHASE 1: PRIMITIVES ONLY ({PRIMITIVES_FIRST} random sentences) ===")
    for i in range(PRIMITIVES_FIRST):
        sentence = generate("S", grammar)
        parse_tree = webster.parse_sentence(
            sentence, threshold=1e9, new_vocab=True,
            learning=True, debug=False)
        if VIZ_INTERMEDIATES and i % 25 == 0:
            parse_tree.visualize(f"{OUT_DIR}/train_trees/primitives_tree{i}")
            webster.visualize_ltm(f"{OUT_DIR}/ltms/primitives_ltm{i}", max_depth=3)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{PRIMITIVES_FIRST}]")

    # ── Phase 2: replay TRAIN hollow trees with merges ─────────────────────────
    print(f"\n=== PHASE 2: HOLLOW CORPUS TRAINING (train fold, size = {len(train_hollow)}) ===")
    trained_trees = []   # keep for unsupervised transition-map mining below
    for i, hollow in enumerate(train_hollow):
        sentence = hollow["sentence"]
        merges   = hollow["merges"]

        tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        tree.build_primitives(sentence, threshold=THRESHOLD)
        for m in merges:
            try:
                tree.apply_candidate(m["left"], m["right"])
            except Exception as e:
                print(f"  [WARN] Merge ({m['left']}, {m['right']}) failed on "
                      f"sentence \"{sentence}\": {e}")
        webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)
        trained_trees.append(tree)

        if VIZ_INTERMEDIATES and i % 10 == 0:
            tree.visualize(f"{OUT_DIR}/train_trees/train_parse_tree{i}")
            webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=3)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(train_hollow)}]")

    # Learn unsupervised structural transitions on the content tree:
    # for each composite chunk in training, record which content-tree
    # leaves its CHILDREN landed at. This builds a self-supervised
    # leaf→child-leaves map used by generate_sentence to keep
    # unpacking class-coherent (e.g. left child of S → NP-class leaf,
    # right child → VP-class leaf — except instead of labels, it uses
    # the leaves themselves as identifiers). No POS dictionary, no gold
    # class labels — pure unsupervised structural pattern.
    webster.learn_leaf_transitions(trained_trees)
    webster.learn_chunk_records(trained_trees)
    print(f"\nLearned {len(webster.content_leaf_transitions)} "
          f"content-tree leaf transitions (unsupervised)")
    print(f"Recorded {sum(len(v) for v in webster.leaf_to_chunks.values())} "
          f"chunk records across {len(webster.leaf_to_chunks)} leaves; "
          f"{len(webster.sentence_root_chunks)} sentence-root chunks "
          f"available as gen seeds")

    # Turn ON the subtree-exchange parsing heuristic. The ranker now uses
    # cnt_root_lp + λ · [log(1+joint) + log(1+L_marg) + log(1+R_marg)]
    # at every merge step. Joint = exact (L_leaf, R_leaf) pair was seen at
    # this parent leaf in training; L_marg / R_marg = either side seen
    # alone as a left / right child of this parent. This is the same
    # subtree-exchange lesson that drove generation from 48% → 100%
    # grammatical, applied as a training-attestation prior on greedy
    # parsing.
    webster.ltm.chunk_pool_weight = 5.0
    print(f"Subtree-exchange parsing heuristic ON: "
          f"chunk_pool_weight = {webster.ltm.chunk_pool_weight}")

    # ── Save state + visualize final LTM ───────────────────────────────────────
    SAVE_DIR = f"{OUT_DIR}/final_ltm_data"
    webster.save_state(SAVE_DIR)
    print(f"\nSaved Final LTM to \"{SAVE_DIR}\"")
    webster.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=3)

    # =============================================================================
    # Shared helpers (lifted from grammar_decoding_test.py / grammar_threshold_test.py)
    # =============================================================================
    def _chunk_span(node):
        out = []
        def w(n):
            if isinstance(n, PrimitiveParseNode):
                out.append(int(n.position_idx)); return
            for _, c in getattr(n, "children", []):
                w(c)
        w(node)
        if not out: return None, None
        return min(out), max(out)

    def _chunk_yield(node):
        out = []
        def w(n):
            if isinstance(n, PrimitiveParseNode):
                wid = getattr(n, "word_id", None)
                if wid is None or wid < 0 or wid >= len(webster.ltm.id_to_value):
                    return
                pos = WORD_TO_POS.get(webster.ltm.id_to_value[wid])
                if pos: out.append(pos)
                return
            for _, c in sorted(getattr(n, "children", []),
                               key=lambda x: x[0] if x[0] is not None else 0):
                w(c)
        w(node)
        return out

    def _walk_composites(node):
        if isinstance(node, PrimitiveParseNode): return
        if not getattr(node, "is_global_root", False):
            yield node
        for _, c in getattr(node, "children", []):
            yield from _walk_composites(c)

    def classify_chunk(node, sentence_len):
        """Head-based chunk classification; S only for the root chunk."""
        pos_seq = _chunk_yield(node)
        if not pos_seq:                                     return None
        if len(pos_seq) == 1:                               return pos_seq[0]
        s, e = _chunk_span(node)
        if s == 0 and e == sentence_len - 1:                return "S"
        if "V" in pos_seq:                                  return "VP"
        if pos_seq[0] == "P":                               return "PP"
        if all(p == "Adj" for p in pos_seq):                return "AdjP"
        if pos_seq[0] == "Adj" and "N" in pos_seq:          return "AdjP"
        if "N" in pos_seq or pos_seq[0] == "Det":           return "NP"
        return "OTHER"

    def _clean_bag(bag):
        """Drop EMPTYNULL (vid 0) sentinels from each attr's value-set."""
        out = {}
        for a, vm in (bag or {}).items():
            cleaned = {v: c for v, c in (vm or {}).items() if v != 0}
            if cleaned: out[a] = cleaned
        return out

    # =============================================================================
    # (1) GROUND-TRUTH PARSE ACCURACY
    # =============================================================================
    # For each held-out hollow sentence:
    #   • gold brackets  = the set of {(left_word_idx, right_word_idx)} spans
    #                      induced by replaying the human merge sequence.
    #   • pred brackets  = the same set induced by WEBSTER's auto-parse via
    #                      parse_sentence(threshold=THRESHOLD) — i.e. the
    #                      4-stage build() from src/parse_mh.py.
    # Report micro-averaged precision / recall / F1 across the held-out fold.
    print("\n=== (1) GROUND-TRUTH PARSE ACCURACY ===")

    def _bracket_set(tree):
        """Return set of (start, end) word-index spans for every composite chunk."""
        brackets = set()
        for comp in _walk_composites(tree.global_root_node):
            s, e = _chunk_span(comp)
            if s is not None and e is not None and s != e:
                brackets.add((s, e))
        return brackets

    parse_rows = []
    total_tp = total_fp = total_fn = 0
    exact_match_count = 0

    for hollow in test_hollow:
        sentence = hollow["sentence"]
        sent_len = len(re.findall(r"[\w']+|[.,!?;]", sentence))
        if sent_len < 2:
            continue

        # gold brackets via replaying merges
        gold_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold = _bracket_set(gold_tree)

        # predicted brackets via WEBSTER auto-parse. The new build() uses
        # the climbing-ancestor count gate on the content tree: only
        # candidates whose categorization leaf has an ancestor with
        # count > THRESHOLD are admitted. THRESHOLD here flows into both
        # primitive stability AND the climbing gate.
        pred_tree = webster.parse_sentence(
            sentence, threshold=THRESHOLD, new_vocab=False,
            learning=False, debug=False)
        pred = _bracket_set(pred_tree)

        tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
        total_tp += tp; total_fp += fp; total_fn += fn
        if gold == pred and len(gold) > 0:
            exact_match_count += 1
        parse_rows.append({"sentence": sentence, "gold": sorted(gold),
                           "pred": sorted(pred), "tp": tp, "fp": fp, "fn": fn})

        # Per-sentence parse-tree viz: gold (replayed merges) + pred (auto-parse).
        idx = len(parse_rows) - 1
        gold_tree.visualize(f"{OUT_DIR}/test_trees/test_gold_tree{idx}")
        pred_tree.visualize(f"{OUT_DIR}/test_trees/test_pred_tree{idx}")

    # Bonus: parse + visualize random fake-word strings to expose the
    # negative-input rejection behavior of the sum_class_lp gate.
    print("\n  Parsing 10 fake (random-word) sentences for inspection...")
    fake_sentences = [
        " ".join([random.choice(corpus) for _ in range(random.randint(3, 8))])
        for _ in range(10)
    ]
    for i, fake in enumerate(fake_sentences):
        fake_tree = webster.parse_sentence(
            fake, threshold=THRESHOLD, new_vocab=False,
            learning=False, debug=False)
        fake_tree.visualize(f"{OUT_DIR}/fake_trees/fake_parse_tree{i}")
        n_chunks = sum(1 for _ in _walk_composites(fake_tree.global_root_node))
        print(f"    [{i+1:>2}] \"{fake}\"   chunks formed: {n_chunks}")

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    exact_pct = exact_match_count / max(len(parse_rows), 1)

    print(f"  Test sentences: {len(parse_rows)}")
    print(f"  Gold brackets: {total_tp + total_fn}   Pred brackets: {total_tp + total_fp}")
    print(f"  Bracket Precision : {100*precision:5.1f}%   "
          f"Recall: {100*recall:5.1f}%   F1: {100*f1:5.1f}%")
    print(f"  Exact-match parses (gold == pred): "
          f"{exact_match_count}/{len(parse_rows)} ({100*exact_pct:.1f}%)")

    with open(f"{OUT_DIR}/parse_accuracy.csv", "w") as f:
        f.write("sentence,gold_brackets,pred_brackets,tp,fp,fn\n")
        for r in parse_rows:
            f.write(f"\"{r['sentence']}\",\"{r['gold']}\",\"{r['pred']}\","
                    f"{r['tp']},{r['fp']},{r['fn']}\n")

    # =============================================================================
    # (1b) STEP-PICK ACCURACY (climbing-ancestor protocol)
    # =============================================================================
    # At each step of the gold replay on each held-out hollow sentence,
    # evaluate every parentless pair and apply build()'s exact strategy:
    #   Stage 1 (gate): climbing-ancestor count > THRESHOLD (content tree)
    #   Stage 2 (rank): argmax ancestor.log_prob_instance(bag)
    # Then ask: does the top-ranked pair's resulting span lie in the gold
    # bracket set?
    print("\n=== (1b) STEP-PICK ACCURACY (climbing-ancestor) ===")

    n_step_correct = n_step_total = n_step_no_cand = 0
    cand_heur_log: list = []   # per-candidate heuristics for histograms
    step_rows = []
    for hollow in test_hollow:
        sentence = hollow["sentence"]
        # 1. Gold bracket set (replay merges on a fresh tree).
        gold_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        gold_tree.build_primitives(sentence, threshold="converge")
        for m in hollow["merges"]:
            try: gold_tree.apply_candidate(m["left"], m["right"])
            except Exception: pass
        gold = _bracket_set(gold_tree)
        if not gold:
            continue

        # 2. Step-by-step replay alongside gold merges.
        step_tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
        step_tree.build_primitives(sentence, threshold="converge")

        for step_idx, m in enumerate(hollow["merges"]):
            pairs = step_tree.get_parentless_pairs()
            if not pairs:
                break

            # Evaluate every parentless pair. Mirror build()'s strategy:
            # climbing-ancestor count gate × argmax cnt_root_lp.
            admitted = []   # [(score, span), ...]
            for p in pairs:
                try:
                    res = step_tree.evaluate_pair(
                        p["left_word_index"], p["right_word_index"],
                        climb_count_threshold=THRESHOLD)
                except Exception:
                    continue
                csd = res.get("content_score_data", {})
                if csd.get("climb_hit_root", True):
                    continue   # gate rejects
                left_node  = step_tree._find_root_child_by_index(p["left_word_index"])
                right_node = step_tree._find_root_child_by_index(p["right_word_index"])
                if left_node is None or right_node is None:
                    continue
                ls, _  = _chunk_span(left_node)
                _, re_ = _chunk_span(right_node)
                # Mirror build()'s ranker:
                #   cnt_root_lp + 0.3·sum_leaf_lp + λ·attestation-boost
                _sc = csd.get("root_log_prob", -float("inf"))
                _ctx_sd = res.get("context_score_data", {})
                _cnt_leaf_lp = csd.get("leaf_log_prob", None)
                _ctx_leaf_lp = _ctx_sd.get("leaf_log_prob", None)
                if _cnt_leaf_lp is not None and _ctx_leaf_lp is not None:
                    _sc = _sc + 0.3 * (float(_cnt_leaf_lp) + float(_ctx_leaf_lp))
                _w = getattr(webster.ltm, "chunk_pool_weight", 0.0) or 0.0
                if _w > 0:
                    import math as _m
                    _sc = _sc + _w * (
                        _m.log(1.0 + csd.get("chunk_pool_match", 0))
                        + _m.log(1.0 + csd.get("L_trans_count", 0))
                        + _m.log(1.0 + csd.get("R_trans_count", 0)))
                admitted.append((_sc, (int(ls), int(re_))))

                # Per-candidate heuristic log for the gold-vs-non-gold
                # histogram panel (see step_pick_histograms.png). We
                # record the candidate's key scores plus whether its
                # merged span is in the gold bracket set.
                _ctx_root_lp = _ctx_sd.get("root_log_prob", None)
                _ctx_bl_lp   = _ctx_sd.get("basic_level_log_prob", None)
                _cnt_bl_lp   = csd.get("basic_level_log_prob", None)
                _is_gold_cand = (int(ls), int(re_)) in gold
                cand_heur_log.append({
                    "is_gold": _is_gold_cand,
                    "cnt_root_lp":  csd.get("root_log_prob", float("nan")),
                    "sum_root_lp":  (csd.get("root_log_prob", 0.0)
                                     + (_ctx_root_lp if _ctx_root_lp is not None else 0.0)),
                    "sum_leaf_lp":  ((float(_cnt_leaf_lp) + float(_ctx_leaf_lp))
                                     if _cnt_leaf_lp is not None and _ctx_leaf_lp is not None
                                     else float("nan")),
                    "sum_bl_lp":    ((float(_cnt_bl_lp) + float(_ctx_bl_lp))
                                     if _cnt_bl_lp is not None and _ctx_bl_lp is not None
                                     else float("nan")),
                    "cur_score":    _sc,
                    "chunk_pool_match": csd.get("chunk_pool_match", 0),
                    "L_trans":      csd.get("L_trans_count", 0),
                    "R_trans":      csd.get("R_trans_count", 0),
                })

            n_step_total += 1
            if not admitted:
                n_step_no_cand += 1
                top_span = None
                is_gold = False
            else:
                admitted.sort(key=lambda x: x[0], reverse=True)
                top_span = admitted[0][1]
                is_gold = top_span in gold
                if is_gold:
                    n_step_correct += 1

            step_rows.append({
                "sentence": sentence, "step": step_idx,
                "n_pairs": len(pairs), "n_admitted": len(admitted),
                "top_span": top_span, "is_gold": int(is_gold),
            })

            # Advance state by applying the gold merge.
            try:
                step_tree.apply_candidate(m["left"], m["right"])
            except Exception:
                break

    step_acc = n_step_correct / max(n_step_total, 1)
    print(f"  Steps evaluated:    {n_step_total}")
    print(f"  Steps with NO admissible candidate "
          f"(climbing gate rejected all): {n_step_no_cand}  "
          f"({100*n_step_no_cand/max(n_step_total,1):.1f}%)")
    print(f"  Step-pick accuracy: "
          f"{n_step_correct}/{n_step_total} ({100*step_acc:.1f}%)")
    print(f"  (THRESHOLD = {THRESHOLD} = climbing-ancestor count threshold)")

    with open(f"{OUT_DIR}/step_pick_accuracy.csv", "w") as f:
        f.write("sentence,step,n_pairs,n_admitted,top_span,is_gold\n")
        for r in step_rows:
            f.write(f"\"{r['sentence']}\",{r['step']},{r['n_pairs']},"
                    f"{r['n_admitted']},\"{r['top_span']}\",{r['is_gold']}\n")

    # =============================================================================
    # (2) GRAMMAR (CHUNK-CLASS) ACCURACY via Cobweb-Discrete probe
    # =============================================================================
    # Train a CobwebDiscreteTree probe on (content_instance bag → gold chunk class)
    # pairs from the TRAIN fold; predict + evaluate on the TEST fold.
    # This is the same "strong decoding style" used in
    # tests/met5/grammar_decoding_test.py Phase 3a.
    print("\n=== (2) GRAMMAR / CHUNK-CLASS ACCURACY (Cobweb-Discrete probe) ===")

    _CLASS_ATTR = -1000   # special attr slot for the gold-label

    def _train_probe(train_bags, train_labels):
        label_ids = {lbl: i + 1 for i, lbl in enumerate(sorted(set(train_labels)))}
        id_labels = {i: lbl for lbl, i in label_ids.items()}
        probe = CobwebDiscreteTree(alpha=PROBE_ALPHA, weight_attr=True)
        for bag, lbl in zip(train_bags, train_labels):
            inst = _clean_bag(bag)
            inst[_CLASS_ATTR] = {label_ids[lbl]: 1}
            probe.ifit(inst)
        return probe, label_ids, id_labels

    def _predict_probe(probe, bag, id_labels):
        inst = _clean_bag(bag)
        n = probe.root
        while n.children:
            n = max(n.children, key=lambda c: c.log_prob_instance(inst))
        while n is not None:
            dist = (n.av_count or {}).get(_CLASS_ATTR, {})
            winning = [(v, c) for v, c in (dist or {}).items() if v != 0]
            if winning:
                return id_labels.get(max(winning, key=lambda kv: kv[1])[0])
            n = getattr(n, "parent", None)
        return None

    # Build train + test bags from chunk content_instances.
    def _collect_chunk_bags(hollow_set):
        bags, ys = [], []
        for hollow in hollow_set:
            sentence = hollow["sentence"]; sent_toks = sentence.split()
            n_words  = len(sent_toks)
            tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
            tree.build_primitives(sentence, threshold="converge")
            for m in hollow["merges"]:
                try: tree.apply_candidate(m["left"], m["right"])
                except Exception: pass
            for comp in _walk_composites(tree.global_root_node):
                ci = comp.get_content_instance()
                if not ci: continue
                gold = classify_chunk(comp, n_words)
                if gold is None: continue
                bags.append(ci); ys.append(gold)
        return bags, ys

    train_bags, train_y = _collect_chunk_bags(train_hollow)
    test_bags,  test_y  = _collect_chunk_bags(test_hollow)
    print(f"  Train chunks: {len(train_bags)}    Test chunks: {len(test_bags)}")

    chunk_probe, _, chunk_id_labels = _train_probe(train_bags, train_y)
    preds = [_predict_probe(chunk_probe, b, chunk_id_labels) or "UNKNOWN"
             for b in test_bags]

    chunk_correct = sum(1 for p, g in zip(preds, test_y) if p == g)
    chunk_acc = chunk_correct / max(len(test_y), 1)
    print(f"  Overall chunk-class accuracy: "
          f"{chunk_correct}/{len(test_y)} ({100*chunk_acc:.1f}%)")

    # Per-class precision/recall/F1
    gold_by = Counter(test_y); pred_by = Counter(preds)
    tp_by   = Counter(g for p, g in zip(preds, test_y) if p == g)
    print(f"  {'class':<6} {'n':>5} {'TP':>4} {'P':>7} {'R':>7} {'F1':>7}")
    prf_rows = []
    for cls in sorted(set(train_y) | set(test_y)):
        n  = gold_by.get(cls, 0)
        tp = tp_by.get(cls, 0)
        pp = pred_by.get(cls, 0)
        prec = tp / pp if pp else 0.0
        rec  = tp / n  if n  else 0.0
        f1c  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if n == 0 and pp == 0: continue
        prf_rows.append((cls, n, tp, prec, rec, f1c))
        print(f"  {cls:<6} {n:>5} {tp:>4} {100*prec:>6.1f}% "
              f"{100*rec:>6.1f}% {100*f1c:>6.1f}%")

    with open(f"{OUT_DIR}/chunk_class_accuracy.csv", "w") as f:
        f.write("class,n_gold,tp,precision,recall,f1\n")
        for cls, n, tp, p, r, f1c in prf_rows:
            f.write(f"{cls},{n},{tp},{p:.4f},{r:.4f},{f1c:.4f}\n")
        f.write(f"OVERALL,{len(test_y)},{chunk_correct},,{chunk_acc:.4f},\n")

    # =============================================================================
    # (3) GENERATION ACCURACY (under the strong decoding style)
    # =============================================================================
    # 3a. FROM-SCRATCH: generate sentences and score grammaticality with a
    #     CYK recognizer over TEST_GRAMMAR1.  Generation already routes
    #     through FiniteParseTree._resolve_bag's bag-weighted frontier-
    #     categorize (same posterior-weighted decoding the probe in (2)
    #     validates).
    # 3b. MASKED COMPLETION: hide one token in each held-out sentence, run
    #     generation, score (i) exact-token recovery and (ii) POS-class
    #     recovery of the filled-in token.
    print("\n=== (3) GENERATION ACCURACY ===")

    # ── CYK-style recognizer over TEST_GRAMMAR1 ──
    # Handles binary, ternary, unary productions on a chart[i][j] of
    # derived nonterminals. Each span is filled by:
    #   - splitting at every k between binary RHS pairs
    #   - splitting at every (k2, k) pair between ternary RHS triples
    #   - applying unary closure until fixed point
    def _grammar_recognize(tokens, start="S"):
        n = len(tokens)
        if n == 0: return False
        term_lhs = defaultdict(set)
        for lhs, prods in grammar.items():
            for prod in prods:
                if len(prod) == 1 and (prod[0] in WORD_TO_POS or prod[0] not in grammar):
                    term_lhs[prod[0]].add(lhs)

        chart = [[set() for _ in range(n + 1)] for _ in range(n + 1)]

        def _unary_closure(s):
            changed = True
            while changed:
                changed = False
                for lhs, prods in grammar.items():
                    for prod in prods:
                        if len(prod) == 1 and prod[0] in s and lhs not in s:
                            s.add(lhs); changed = True

        for i, tok in enumerate(tokens):
            chart[i][i + 1] = set(term_lhs.get(tok, set()))
            _unary_closure(chart[i][i + 1])

        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span
                # Binary productions: try every split point.
                for k in range(i + 1, j):
                    left, right = chart[i][k], chart[k][j]
                    if not left or not right: continue
                    for lhs, prods in grammar.items():
                        for prod in prods:
                            if (len(prod) == 2
                                    and prod[0] in left and prod[1] in right):
                                chart[i][j].add(lhs)
                # Ternary productions: try every (k2, k) split. Doesn't
                # require chart[i][k] to be non-empty (intentional — the
                # 3-way split bypasses chart[i][k] entirely).
                if span >= 3:
                    for k2 in range(i + 1, j - 1):
                        for k in range(k2 + 1, j):
                            a_set = chart[i][k2]
                            b_set = chart[k2][k]
                            c_set = chart[k][j]
                            if not a_set or not b_set or not c_set: continue
                            for lhs, prods in grammar.items():
                                for prod in prods:
                                    if (len(prod) == 3
                                            and prod[0] in a_set
                                            and prod[1] in b_set
                                            and prod[2] in c_set):
                                        chart[i][j].add(lhs)
                _unary_closure(chart[i][j])
        return start in chart[0][n]

    # ── 3a. From-scratch generation ──
    print("\n  --- (3a) From-scratch generation ---")
    N_GEN = 50
    n_gen_ok = n_lex_ok = n_total = 0
    generations = []
    for i in range(N_GEN):
        try:
            gen_text, gen_parse = webster.generate_via_chunk_replay()
        except Exception as e:
            gen_text, gen_parse = f"<failed: {e}>", None
        n_total += 1
        toks = gen_text.split()
        lex_ok = all(t in WORD_TO_POS for t in toks) if toks else False
        gram_ok = lex_ok and _grammar_recognize(toks)
        if lex_ok:  n_lex_ok  += 1
        if gram_ok: n_gen_ok  += 1
        generations.append({"text": gen_text, "lex_ok": lex_ok, "gram_ok": gram_ok})
        flag = "✓" if gram_ok else ("L" if lex_ok else "x")
        print(f"    [{i+1:>3}] {flag} \"{gen_text}\"")
        if gen_parse is not None and i < 20:
            try:
                gen_parse.visualize(f"{OUT_DIR}/generated_trees/generated_parse_tree{i}")
            except Exception:
                pass

    print(f"  In-lexicon : {n_lex_ok}/{n_total} ({100*n_lex_ok/max(n_total,1):.1f}%)")
    print(f"  Grammatical: {n_gen_ok}/{n_total} ({100*n_gen_ok/max(n_total,1):.1f}%)")

    # ── Uniqueness / novelty metrics ──────────────────────────────────────────
    # Two complementary measures:
    #   • novelty_rate  = fraction of generations whose TEXT was never seen
    #                     verbatim during training. A pure replay system
    #                     scores 0%; pure hallucination scores 100%. We want
    #                     this HIGH (genuinely creating new sentences) and
    #                     in concert with grammaticality (not just garbage).
    #   • diversity     = unique outputs / total outputs. Measures how
    #                     varied generation is — a system that emits the
    #                     same sentence 50 times scores 0.02; one that
    #                     always emits something different scores 1.0.
    # Together they detect: are we MEMORIZING (low novelty + low diversity),
    # REPEATING ONE NOVEL (low novelty + low diversity but novel sample),
    # or TRULY GENERATING (high novelty + high diversity).
    train_set = {h["sentence"].strip() for h in train_hollow}
    unique_texts = {g["text"].strip() for g in generations}
    novel_count = sum(1 for g in generations if g["text"].strip() not in train_set)
    gram_novel_count = sum(1 for g in generations
                           if g["gram_ok"] and g["text"].strip() not in train_set)
    novelty_rate = novel_count / max(n_total, 1)
    diversity    = len(unique_texts) / max(n_total, 1)
    gram_novel_rate = gram_novel_count / max(n_total, 1)
    # Structural novelty (POS-sequence) — a stronger memorization test
    # than surface-form novelty. An output that happens to use the same
    # POS sequence as a training sentence (e.g. "Det N V Det N") but
    # with different words is still STRUCTURALLY known. A genuinely new
    # POS combination is evidence the model has learned to combine
    # chunks beyond what training showed.
    gen_pos_novel = sum(1 for g in generations
                        if _pos_seq(g["text"].strip()) not in train_pos)
    pos_novel_rate = gen_pos_novel / max(n_total, 1)
    print(f"  Novelty (not in train): {novel_count}/{n_total} "
          f"({100*novelty_rate:.1f}%)")
    print(f"  POS-structural novelty: {gen_pos_novel}/{n_total} "
          f"({100*pos_novel_rate:.1f}%)")
    print(f"  Diversity (unique/total): {len(unique_texts)}/{n_total} "
          f"({100*diversity:.1f}%)")
    print(f"  Grammatical AND novel: {gram_novel_count}/{n_total} "
          f"({100*gram_novel_rate:.1f}%)")

    with open(f"{OUT_DIR}/generation_from_scratch.csv", "w") as f:
        f.write("idx,in_lexicon,grammatical,novel,text\n")
        for i, g in enumerate(generations):
            is_novel = int(g["text"].strip() not in train_set)
            f.write(f"{i},{int(g['lex_ok'])},{int(g['gram_ok'])},"
                    f"{is_novel},\"{g['text']}\"\n")

    # ── 3b. Single-token masked completion (POS + exact recovery) ──
    print("\n  --- (3b) Single-token masked completion ---")
    mask_rows = []
    exact_hits = pos_hits = mask_total = 0
    test_sents = [h["sentence"] for h in test_hollow if len(h["sentence"].split()) >= 3]

    for i, sent in enumerate(test_sents[:30]):
        toks = sent.split()
        mid  = len(toks) // 2
        gold_tok = toks[mid]
        gold_pos = WORD_TO_POS.get(gold_tok, "OTHER")
        masked = " ".join(toks[:mid] + ["[mask]"] + toks[mid + 1:])
        try:
            completed, comp_parse = webster.generate_sentence(
                masked_sentence=masked, debug=False)
        except Exception as e:
            completed, comp_parse = f"<failed: {e}>", None
        if comp_parse is not None and i < 20:
            try:
                comp_parse.visualize(f"{OUT_DIR}/mask_trees/mask_tree{i}")
            except Exception:
                pass
        comp_toks = completed.split()
        # Pull out what filled the mask: take the token at position `mid`
        # (or "?" if completion is shorter).
        filled = comp_toks[mid] if mid < len(comp_toks) else "?"
        filled_pos = WORD_TO_POS.get(filled, "OTHER")

        exact = (filled == gold_tok)
        pos_ok = (filled_pos == gold_pos)
        if exact:  exact_hits += 1
        if pos_ok: pos_hits   += 1
        mask_total += 1

        mark = "=" if exact else ("~" if pos_ok else "x")
        print(f"    [{i+1:>3}] {mark} gold=\"{gold_tok}\" ({gold_pos}) "
              f"→ filled=\"{filled}\" ({filled_pos})   completed=\"{completed}\"")
        mask_rows.append({
            "sentence": sent, "mask_pos": mid,
            "gold": gold_tok, "gold_pos": gold_pos,
            "filled": filled, "filled_pos": filled_pos,
            "completed": completed,
            "exact": int(exact), "pos_ok": int(pos_ok),
        })

    print(f"  Exact-token recovery: {exact_hits}/{mask_total} "
          f"({100*exact_hits/max(mask_total,1):.1f}%)")
    print(f"  POS-class recovery  : {pos_hits}/{mask_total} "
          f"({100*pos_hits/max(mask_total,1):.1f}%)")
    print(f"  (chance: lexicon={100/len(WORD_TO_POS):.1f}%, "
          f"POS={100/len(POS_LIST):.0f}%)")

    with open(f"{OUT_DIR}/generation_masked.csv", "w") as f:
        f.write("idx,sentence,mask_pos,gold,gold_pos,filled,filled_pos,"
                "completed,exact,pos_ok\n")
        for i, r in enumerate(mask_rows):
            f.write(f"{i},\"{r['sentence']}\",{r['mask_pos']},{r['gold']},"
                    f"{r['gold_pos']},{r['filled']},{r['filled_pos']},"
                    f"\"{r['completed']}\",{r['exact']},{r['pos_ok']}\n")

    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — flying colors check")
    print("=" * 70)
    print(f"  (1)  Parse bracket P/R/F1     : "
          f"{100*precision:.1f}% / {100*recall:.1f}% / {100*f1:.1f}%")
    print(f"       Exact-match parses        : {100*exact_pct:.1f}%")
    print(f"  (1b) Step-pick (climbing-ancestor): {100*step_acc:.1f}%  "
          f"(of {n_step_total} steps; {n_step_no_cand} gated out)")
    print(f"  (2)  Chunk-class accuracy     : {100*chunk_acc:.1f}%")
    print(f"  (3a) From-scratch grammatical : {100*n_gen_ok/max(n_total,1):.1f}%")
    print(f"  (3a) From-scratch in-lexicon  : {100*n_lex_ok/max(n_total,1):.1f}%")
    print(f"  (3a) From-scratch novelty     : {100*novelty_rate:.1f}%  "
          f"(unique outputs: {100*diversity:.1f}%)")
    print(f"  (3a) Grammatical AND novel    : {100*gram_novel_rate:.1f}%")
    print(f"  (3b) Mask exact recovery      : "
          f"{100*exact_hits/max(mask_total,1):.1f}%")
    print(f"  (3b) Mask POS recovery        : "
          f"{100*pos_hits/max(mask_total,1):.1f}%")
    print()
    print(f"Artefacts in {OUT_DIR}/:")
    print(f"  parse_accuracy.csv, chunk_class_accuracy.csv,")
    print(f"  generation_from_scratch.csv, generation_masked.csv,")
    print(f"  final_ltm_data/ (saved WEBSTER state)")
    print(f"  performance_summary.png (overview graphic)")

    # =============================================================================
    # GOLD vs NON-GOLD HEURISTIC HISTOGRAMS  (parse-time discriminability)
    # =============================================================================
    # For each candidate evaluated along a gold trajectory we logged
    # ``cand_heur_log``: { is_gold, cnt_root_lp, sum_root_lp, sum_leaf_lp,
    # sum_bl_lp, cur_score, chunk_pool_match, L_trans, R_trans }.
    # Plot density histograms of (gold-candidates vs non-gold) for each
    # heuristic side-by-side so we can see where the parse-time signal
    # actually separates the two populations. Mirrors the gold/non-gold
    # density plot in tests/met5/grammar_threshold_test.py Phase 2.
    print("\n=== Building gold-vs-non-gold heuristic histograms ===")
    _hist_heurs = ["cnt_root_lp", "sum_root_lp", "sum_leaf_lp",
                   "sum_bl_lp", "cur_score",
                   "chunk_pool_match", "L_trans", "R_trans"]
    _gold_vals    = {h: [r[h] for r in cand_heur_log if r["is_gold"]] for h in _hist_heurs}
    _nongold_vals = {h: [r[h] for r in cand_heur_log if not r["is_gold"]] for h in _hist_heurs}
    print(f"  {len(cand_heur_log)} total candidates: "
          f"{sum(1 for r in cand_heur_log if r['is_gold'])} gold, "
          f"{sum(1 for r in cand_heur_log if not r['is_gold'])} non-gold")

    # Persist the raw candidate heuristic log so multi-seed pipelines
    # can pool across seeds and rebuild an aggregate histogram.
    _cand_csv = f"{OUT_DIR}/cand_heur_log.csv"
    with open(_cand_csv, "w") as _f:
        _f.write("is_gold," + ",".join(_hist_heurs) + "\n")
        for _r in cand_heur_log:
            _f.write(f"{int(bool(_r['is_gold']))},"
                     + ",".join(f"{_r[h]}" for h in _hist_heurs) + "\n")
    print(f"  Per-candidate heuristic log → {_cand_csv}")

    n_h = len(_hist_heurs)
    n_cols = 4
    n_rows = (n_h + n_cols - 1) // n_cols
    fig_h, axes_h = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 3 * n_rows))
    axes_h = axes_h.flatten() if n_rows > 1 else [axes_h] if n_cols == 1 else axes_h
    for i, h in enumerate(_hist_heurs):
        ax = axes_h[i]
        g_arr = np.array([v for v in _gold_vals[h] if np.isfinite(v)])
        n_arr = np.array([v for v in _nongold_vals[h] if np.isfinite(v)])
        if len(g_arr) == 0 and len(n_arr) == 0:
            ax.set_title(f"{h}\n(no data)")
            continue
        # Shared bin edges so the histograms are directly comparable.
        all_v = np.concatenate([g_arr, n_arr]) if len(n_arr) > 0 else g_arr
        lo, hi = float(np.min(all_v)), float(np.max(all_v))
        if lo == hi:
            lo -= 0.5; hi += 0.5
        bins = np.linspace(lo, hi, 30)
        # Y-axis = raw candidate COUNT per bin (not density-normalised).
        # The two populations have different totals (gold ~half non-gold)
        # so showing counts directly is more honest than densities for
        # a quick eyeball of "which scores have how many candidates".
        ax.hist(n_arr, bins=bins, alpha=0.5, color="#d62728",
                label=f"non-gold (n={len(n_arr)})")
        ax.hist(g_arr, bins=bins, alpha=0.7, color="#2ca02c",
                label=f"gold (n={len(g_arr)})")
        # Effect size: standardized mean difference (Cohen's d).
        if len(g_arr) > 1 and len(n_arr) > 1:
            pooled_std = float(np.sqrt(
                (np.var(g_arr, ddof=1) + np.var(n_arr, ddof=1)) / 2))
            d = ((float(np.mean(g_arr)) - float(np.mean(n_arr)))
                 / max(pooled_std, 1e-9))
            ax.set_title(f"{h}  (d={d:+.2f})", fontsize=10)
        else:
            ax.set_title(h, fontsize=10)
        ax.set_ylabel("# candidates", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)
    # Hide unused subplots
    for j in range(n_h, len(axes_h)):
        axes_h[j].set_visible(False)
    fig_h.suptitle(
        "Parse-time heuristic distributions — gold vs non-gold candidates "
        f"(d = Cohen's effect size; SEED={SEED})",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT_DIR}/step_pick_histograms.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Histograms → {OUT_DIR}/step_pick_histograms.png")

    # =============================================================================
    # SUMMARY GRAPHIC — performance across all evaluated tasks
    # =============================================================================
    # A four-panel figure:
    #   • Panel A: bracket-level P / R / F1 + exact-match (Phase 1).
    #   • Panel B: per-chunk-class P / R / F1 (Phase 2), with overall accuracy
    #               called out in the title.
    #   • Panel C: from-scratch generation rates (in-lexicon, grammatical).
    #   • Panel D: masked-completion recovery (exact-token, POS-class) with
    #               chance baselines.
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "WEBSTER — Hollow Learning Test Performance Summary  "
        f"(SEED={SEED}, train={len(train_hollow)}, test={len(test_hollow)})  "
        f"[strategy: climbing-ancestor count gate (THRESHOLD={THRESHOLD}) × "
        "argmax cnt_root_lp]",
        fontsize=12, fontweight="bold")

    # Panel A — Parse bracket P/R/F1 + exact-match.
    axA = fig.add_subplot(2, 3, 1)
    parse_metrics = ["Precision", "Recall", "F1", "Exact-match"]
    parse_values  = [precision, recall, f1, exact_pct]
    parse_colors  = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    bars = axA.bar(parse_metrics, parse_values,
                   color=parse_colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, parse_values):
        axA.text(b.get_x() + b.get_width()/2, v + 0.02,
                 f"{100*v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axA.set_ylim(0, 1.15)
    axA.set_ylabel("Score")
    axA.set_title(
        f"(A) End-to-end Parse Accuracy  (n={len(parse_rows)})", fontsize=11)
    axA.grid(axis="y", alpha=0.3)

    # Panel B — Step-pick accuracy: climbing-ancestor gate.
    axB = fig.add_subplot(2, 3, 2)
    step_chance = (1 / max(np.mean([r["n_pairs"] for r in step_rows]) or 1, 1)
                   if step_rows else 0)
    gate_pass_rate = ((n_step_total - n_step_no_cand) / max(n_step_total, 1)
                      if n_step_total else 0)
    sp_labels = ["Step-pick\n(of admitted)",
                 "Gate pass-rate\n(climb cleared)",
                 "chance"]
    sp_values = [step_acc, gate_pass_rate, step_chance]
    sp_colors = ["#d62728", "#2ca02c", "lightgray"]
    sp_bars = axB.bar(sp_labels, sp_values,
                      color=sp_colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(sp_bars, sp_values):
        axB.text(b.get_x() + b.get_width()/2, v + 0.02,
                 f"{100*v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axB.set_ylim(0, 1.15)
    axB.set_ylabel("Accuracy")
    axB.set_title(
        f"(B) Step-pick — Climbing-Ancestor  (n={n_step_total} steps)",
        fontsize=11)
    axB.tick_params(axis="x", labelsize=8)
    axB.grid(axis="y", alpha=0.3)

    # Panel C — Per-chunk-class P / R / F1 grouped bars.
    axC = fig.add_subplot(2, 3, 3)
    if prf_rows:
        cls_names = [r[0] for r in prf_rows]
        P  = [r[3] for r in prf_rows]
        R  = [r[4] for r in prf_rows]
        F1 = [r[5] for r in prf_rows]
        n_g = [r[1] for r in prf_rows]
        x = np.arange(len(cls_names)); w = 0.27
        axC.bar(x - w, P,  w, label="Precision", color="#1f77b4")
        axC.bar(x,     R,  w, label="Recall",    color="#2ca02c")
        axC.bar(x + w, F1, w, label="F1",        color="#d62728")
        for i in range(len(cls_names)):
            axC.text(x[i] - w, P[i]  + 0.02, f"{100*P[i]:.0f}",  ha="center", fontsize=7)
            axC.text(x[i],     R[i]  + 0.02, f"{100*R[i]:.0f}",  ha="center", fontsize=7)
            axC.text(x[i] + w, F1[i] + 0.02, f"{100*F1[i]:.0f}", ha="center", fontsize=7)
        axC.set_xticks(x)
        axC.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cls_names, n_g)],
                           fontsize=9)
        axC.set_ylim(0, 1.15); axC.set_ylabel("Score")
        axC.set_title(
            f"(C) Chunk-Class Probe   overall {100*chunk_acc:.1f}%   (n={len(test_y)})",
            fontsize=11)
        axC.legend(loc="lower right", fontsize=8)
        axC.grid(axis="y", alpha=0.3)

    # Panel D — From-scratch generation rates (incl. uniqueness/novelty).
    axD = fig.add_subplot(2, 3, 4)
    gen_metrics = ["In-lexicon", "Grammatical", "Novel", "Unique", "Gram & Novel"]
    gen_values  = [n_lex_ok / max(n_total, 1),
                   n_gen_ok / max(n_total, 1),
                   novelty_rate,
                   diversity,
                   gram_novel_rate]
    gen_colors  = ["#17becf", "#bcbd22", "#9467bd", "#8c564b", "#2ca02c"]
    bars = axD.bar(gen_metrics, gen_values,
                   color=gen_colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, gen_values):
        axD.text(b.get_x() + b.get_width()/2, v + 0.02,
                 f"{100*v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axD.set_ylim(0, 1.15); axD.set_ylabel("Rate")
    axD.set_title(
        f"(D) From-scratch Generation  (n={n_total})", fontsize=11)
    axD.tick_params(axis="x", labelsize=8, rotation=15)
    axD.grid(axis="y", alpha=0.3)

    # Panel E — Mask completion: exact + POS recovery, with chance baselines.
    axE = fig.add_subplot(2, 3, 5)
    mask_metrics = ["Exact-token", "POS-class"]
    mask_values  = [exact_hits / max(mask_total, 1),
                    pos_hits   / max(mask_total, 1)]
    mask_chance  = [1 / max(len(WORD_TO_POS), 1),
                    1 / max(len(POS_LIST), 1)]
    mask_colors  = ["#ff7f0e", "#e377c2"]
    x = np.arange(len(mask_metrics)); w = 0.35
    bars = axE.bar(x - w/2, mask_values, w, color=mask_colors,
                   edgecolor="black", linewidth=0.5, label="WEBSTER")
    chance_bars = axE.bar(x + w/2, mask_chance, w,
                          color="lightgray", edgecolor="black",
                          linewidth=0.5, label="chance")
    for b, v in zip(bars, mask_values):
        axE.text(b.get_x() + b.get_width()/2, v + 0.02,
                 f"{100*v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(chance_bars, mask_chance):
        axE.text(b.get_x() + b.get_width()/2, v + 0.02,
                 f"{100*v:.1f}%", ha="center", fontsize=8, color="#444")
    axE.set_xticks(x); axE.set_xticklabels(mask_metrics)
    axE.set_ylim(0, 1.15); axE.set_ylabel("Recovery rate")
    axE.set_title(
        f"(E) Single-token Masked Completion  (n={mask_total})", fontsize=11)
    axE.legend(loc="upper right", fontsize=9)
    axE.grid(axis="y", alpha=0.3)

    # Panel F — Headline numbers / overall scorecard.
    axF = fig.add_subplot(2, 3, 6)
    axF.axis("off")
    scorecard = [
        ("Parse F1",                  f"{100*f1:.1f}%"),
        ("Step-pick (climbing)",      f"{100*step_acc:.1f}%"),
        ("Chunk-class accuracy",      f"{100*chunk_acc:.1f}%"),
        ("Gen grammatical",           f"{100*n_gen_ok/max(n_total,1):.1f}%"),
        ("Gen novelty",               f"{100*novelty_rate:.1f}%"),
        ("Gen gram&novel",            f"{100*gram_novel_rate:.1f}%"),
        ("Mask POS recovery",         f"{100*pos_hits/max(mask_total,1):.1f}%"),
    ]
    axF.set_title("(F) Scorecard", fontsize=11)
    y_top = 0.95
    for i, (label, val) in enumerate(scorecard):
        y = y_top - i * 0.14
        axF.text(0.05, y, label, fontsize=12, va="center",
                 transform=axF.transAxes)
        axF.text(0.95, y, val, fontsize=14, va="center", ha="right",
                 fontweight="bold", color="#1f77b4",
                 transform=axF.transAxes)
        axF.plot([0.04, 0.96], [y - 0.07, y - 0.07],
                 color="#ddd", linewidth=0.6, transform=axF.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT_DIR}/performance_summary.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Summary graphic → {OUT_DIR}/performance_summary.png")



if __name__ == "__main__":
    run_hollow_learn()