"""
General Learning Test (Multi-Hierarchy) — UNSUPERVISED counterpart to
unittests/hollow_learn_test_mh.py.

Where hollow_learn_test trains on HUMAN-ANNOTATED hollow merges
(supervised structural signal), this test trains *only* on raw
sentences. TRELLIS's own ``parse_sentence(learning=True)`` decides
the chunks via build()'s climbing-ancestor gate + cnt_root_lp ranker
and learns from its OWN parses.

Evaluation uses the SAME held-out hollow corpus as hollow_learn_test
so the two tests are directly comparable:
    hollow_learn → upper bound (supervised structural signal)
    gen_learn    → unsupervised lower bound

Scores (matched to hollow_learn_test_mh.py):

  (1)  Ground-truth PARSE accuracy on held-out hollow sentences
       (P / R / F1, exact-match). Order-independent (bracket SETS).
  (1b) STEP-PICK accuracy — at each gold-merge step, does build()'s
       selection rule pick a gold candidate?
  (2)  CHUNK-CLASS accuracy via Cobweb-Discrete probe — trains on
       (content_instance, gold-class) from a hollow train fold,
       predicts test-fold classes. Tells us if the UNSUPERVISED
       representations are still class-discriminative.
  (3a) From-scratch GENERATION — 50 outputs, in-lexicon and
       CYK-grammatical rates.
  (3b) Single-token MASKED COMPLETION — exact-token and POS-class
       recovery.

See unittests/README.md for the full metric reference.
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
from parse_mh import TRELLIS, FiniteParseTree, PrimitiveParseNode
from cobweb.cobweb_discrete import CobwebDiscreteTree, set_random_seed as cobweb_set_seed

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR           = "unittests/gen_learn_test_mh"
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"
VIZ_INTERMEDIATES = True

CONTEXT_LENGTH    = 3
THRESHOLD         = 30
PRIMITIVES_FIRST  = 200
SEED              = 13
PROBE_ALPHA       = 1e-3

# Number of UNSUPERVISED training sentences (random CFG-derived).
# We mirror hollow_learn_test's data scale: PRIMITIVES_FIRST (200) for
# primitives-only learning + ~90 sentences for chunk learning.
N_UNSUP_TRAIN     = 90

random.seed(SEED)
np.random.seed(SEED)
cobweb_set_seed(SEED)

corpus  = TEST_CORPUS1
grammar = TEST_GRAMMAR1

POS_LIST = ["Det", "N", "Adj", "V", "P"]
WORD_TO_POS = {}
for pos in POS_LIST:
    for prod in grammar[pos]:
        for w in prod:
            WORD_TO_POS[w] = pos
CHUNK_LABELS = ["NP", "AdjP", "PP", "VP", "S"]

# ── Setup ──────────────────────────────────────────────────────────────────
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load hollow corpus (used ONLY for evaluation) ─────────────────────────
hollow_corpus: list[dict] = []
for p in sorted(glob.glob(os.path.join(HOLLOW_CORPUS_DIR, "*.json"))):
    with open(p, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON: {p}")
            continue
    if "sentence" in data and "merges" in data:
        hollow_corpus.append(data)
print(f"Loaded {len(hollow_corpus)} hollow parse trees from {HOLLOW_CORPUS_DIR}")

# Same 80/20 sentence-level split as hollow_learn_test for direct comparability.
random.shuffle(hollow_corpus)
_split = int(0.8 * len(hollow_corpus))
train_hollow = hollow_corpus[:_split]
test_hollow  = hollow_corpus[_split:]
print(f"  Hollow split: train={len(train_hollow)}  test={len(test_hollow)}  "
      f"(train used ONLY for chunk-class probe; eval is test fold)")

# Generate UNSUPERVISED training data (random sentences from the CFG).
random.seed(SEED + 1)
unsup_sentences = [generate("S", grammar) for _ in range(PRIMITIVES_FIRST + N_UNSUP_TRAIN)]
print(f"  Unsupervised training sentences: {len(unsup_sentences)} "
      f"(first {PRIMITIVES_FIRST} primitives-only, rest with build())")

# Reset seed before TRELLIS construction so the training run is reproducible.
random.seed(SEED)
np.random.seed(SEED)
cobweb_set_seed(SEED)

# ── Initialise TRELLIS (same hyperparams as hollow_learn_test) ────────────
trellis = TRELLIS(
    corpus,
    context_length=CONTEXT_LENGTH,
    threshold=THRESHOLD,
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
    sentence = unsup_sentences[i]
    parse_tree = trellis.parse_sentence(
        sentence, threshold=1e9, new_vocab=True,
        learning=True, debug=False)
    if VIZ_INTERMEDIATES and i % 50 == 0:
        parse_tree.visualize(f"{OUT_DIR}/train_trees/primitives_tree{i}")
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{PRIMITIVES_FIRST}]")

# ── Phase 2: UNSUPERVISED chunk learning ───────────────────────────────────
# This is the KEY difference vs hollow_learn_test: build() decides the
# chunks here (no gold merges). The climbing-ancestor gate + cnt_root_lp
# ranker have to discover chunking on their own.
print(f"\n=== PHASE 2: UNSUPERVISED CHUNK LEARNING ({N_UNSUP_TRAIN} sentences) ===")
trained_trees = []   # keep so subtree-exchange generation can mine them
for i in range(N_UNSUP_TRAIN):
    sentence = unsup_sentences[PRIMITIVES_FIRST + i]
    parse_tree = trellis.parse_sentence(
        sentence, threshold=THRESHOLD, new_vocab=True,
        learning=True, debug=False)
    trained_trees.append(parse_tree)
    if VIZ_INTERMEDIATES and i % 10 == 0:
        parse_tree.visualize(f"{OUT_DIR}/train_trees/unsup_tree{i}")
    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{N_UNSUP_TRAIN}]")

# Mine the unsupervised parse trees for chunk-replay data. Even when
# build() doesn't always fire at the sentence-root level (the gate may
# stop short of S), per-leaf chunk pools are still populated by every
# composite TRELLIS did form during training — so generation can replay
# subtrees out of whatever was learned.
trellis.learn_leaf_transitions(trained_trees)
trellis.learn_chunk_records(trained_trees)
print(f"\nLearned {len(trellis.content_leaf_transitions)} "
      f"content-tree leaf transitions (unsupervised)")
print(f"Recorded {sum(len(v) for v in trellis.leaf_to_chunks.values())} "
      f"chunk records across {len(trellis.leaf_to_chunks)} leaves; "
      f"{len(trellis.sentence_root_chunks)} sentence-root chunks "
      f"available as gen seeds")

# Turn ON the subtree-exchange parsing heuristic — same knob as
# hollow_learn. Mined from TRELLIS's OWN parses (no gold merges) so
# the boost reinforces self-consistent training patterns rather than
# pulling toward human-preferred ones.
trellis.ltm.chunk_pool_weight = 5.0
print(f"Subtree-exchange parsing heuristic ON: "
      f"chunk_pool_weight = {trellis.ltm.chunk_pool_weight}")

# ── Save state ─────────────────────────────────────────────────────────────
SAVE_DIR = f"{OUT_DIR}/final_ltm_data"
trellis.save_state(SAVE_DIR)
print(f"\nSaved Final LTM to \"{SAVE_DIR}\"")
trellis.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=3)

# =============================================================================
# Shared helpers
# =============================================================================
def _chunk_span(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            out.append(int(n.position_idx)); return
        for _, c in getattr(n, "children", []): w(c)
    w(node)
    if not out: return None, None
    return min(out), max(out)

def _chunk_yield(node):
    out = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            wid = getattr(n, "word_id", None)
            if wid is None or wid < 0 or wid >= len(trellis.ltm.id_to_value):
                return
            pos = WORD_TO_POS.get(trellis.ltm.id_to_value[wid])
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
    out = {}
    for a, vm in (bag or {}).items():
        cleaned = {v: c for v, c in (vm or {}).items() if v != 0}
        if cleaned: out[a] = cleaned
    return out

def _bracket_set(tree):
    brackets = set()
    for comp in _walk_composites(tree.global_root_node):
        s, e = _chunk_span(comp)
        if s is not None and e is not None and s != e:
            brackets.add((s, e))
    return brackets

# =============================================================================
# (1) GROUND-TRUTH PARSE ACCURACY
# =============================================================================
print("\n=== (1) GROUND-TRUTH PARSE ACCURACY (held-out hollow) ===")

parse_rows = []
total_tp = total_fp = total_fn = 0
exact_match_count = 0

for hollow in test_hollow:
    sentence = hollow["sentence"]
    sent_len = len(re.findall(r"[\w']+|[.,!?;]", sentence))
    if sent_len < 2:
        continue

    gold_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
    gold_tree.build_primitives(sentence, threshold="converge")
    for m in hollow["merges"]:
        try: gold_tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    gold = _bracket_set(gold_tree)

    pred_tree = trellis.parse_sentence(
        sentence, threshold=THRESHOLD, new_vocab=False,
        learning=False, debug=False)
    pred = _bracket_set(pred_tree)

    tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
    total_tp += tp; total_fp += fp; total_fn += fn
    if gold == pred and len(gold) > 0:
        exact_match_count += 1
    parse_rows.append({"sentence": sentence, "gold": sorted(gold),
                       "pred": sorted(pred), "tp": tp, "fp": fp, "fn": fn})

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
print("\n=== (1b) STEP-PICK ACCURACY (climbing-ancestor) ===")

n_step_correct = n_step_total = n_step_no_cand = 0
step_rows = []
for hollow in test_hollow:
    sentence = hollow["sentence"]
    gold_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
    gold_tree.build_primitives(sentence, threshold="converge")
    for m in hollow["merges"]:
        try: gold_tree.apply_candidate(m["left"], m["right"])
        except Exception: pass
    gold = _bracket_set(gold_tree)
    if not gold:
        continue

    step_tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
    step_tree.build_primitives(sentence, threshold="converge")

    for step_idx, m in enumerate(hollow["merges"]):
        pairs = step_tree.get_parentless_pairs()
        if not pairs:
            break

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
            _w = getattr(trellis.ltm, "chunk_pool_weight", 0.0) or 0.0
            if _w > 0:
                import math as _m
                _sc = _sc + _w * (
                    _m.log(1.0 + csd.get("chunk_pool_match", 0))
                    + _m.log(1.0 + csd.get("L_trans_count", 0))
                    + _m.log(1.0 + csd.get("R_trans_count", 0)))
            admitted.append((_sc, (int(ls), int(re_))))

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

        try:
            step_tree.apply_candidate(m["left"], m["right"])
        except Exception:
            break

step_acc = n_step_correct / max(n_step_total, 1)
print(f"  Steps evaluated:    {n_step_total}")
print(f"  Steps with NO admissible candidate (climbing gate rejected all): "
      f"{n_step_no_cand}  ({100*n_step_no_cand/max(n_step_total,1):.1f}%)")
print(f"  Step-pick accuracy: "
      f"{n_step_correct}/{n_step_total} ({100*step_acc:.1f}%)")
print(f"  (THRESHOLD = {THRESHOLD} = climbing-ancestor count threshold)")

with open(f"{OUT_DIR}/step_pick_accuracy.csv", "w") as f:
    f.write("sentence,step,n_pairs,n_admitted,top_span,is_gold\n")
    for r in step_rows:
        f.write(f"\"{r['sentence']}\",{r['step']},{r['n_pairs']},"
                f"{r['n_admitted']},\"{r['top_span']}\",{r['is_gold']}\n")

# =============================================================================
# (2) CHUNK-CLASS ACCURACY via Cobweb-Discrete probe
# =============================================================================
print("\n=== (2) CHUNK-CLASS ACCURACY (Cobweb-Discrete probe) ===")

_CLASS_ATTR = -1000

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

def _collect_chunk_bags(hollow_set):
    """Collect (bag, class) pairs from hollow gold parses. Used by the
    probe whether TRELLIS was trained supervised (hollow) or
    unsupervised (random). Gold class comes from the hollow merges, not
    from TRELLIS's internal state — so the probe always sees ground
    truth labels."""
    bags, ys = [], []
    for hollow in hollow_set:
        sentence = hollow["sentence"]; sent_toks = sentence.split()
        n_words  = len(sent_toks)
        tree = FiniteParseTree(trellis.ltm, context_length=CONTEXT_LENGTH)
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
# (3) GENERATION ACCURACY
# =============================================================================
print("\n=== (3) GENERATION ACCURACY ===")

def _grammar_recognize(tokens, start="S"):
    n = len(tokens)
    if n == 0: return False
    term_lhs = defaultdict(set)
    for lhs, prods in grammar.items():
        for prod in prods:
            if len(prod) == 1 and (prod[0] in WORD_TO_POS or prod[0] not in grammar):
                term_lhs[prod[0]].add(lhs)
    chart = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    def _uc(s):
        c = True
        while c:
            c = False
            for lhs, prods in grammar.items():
                for prod in prods:
                    if len(prod) == 1 and prod[0] in s and lhs not in s:
                        s.add(lhs); c = True
    for i, tok in enumerate(tokens):
        chart[i][i + 1] = set(term_lhs.get(tok, set()))
        _uc(chart[i][i + 1])
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                L, R = chart[i][k], chart[k][j]
                if not L or not R: continue
                for lhs, prods in grammar.items():
                    for prod in prods:
                        if len(prod) == 2 and prod[0] in L and prod[1] in R:
                            chart[i][j].add(lhs)
            if span >= 3:
                for k2 in range(i + 1, j - 1):
                    for k in range(k2 + 1, j):
                        a, b, c = chart[i][k2], chart[k2][k], chart[k][j]
                        if not a or not b or not c: continue
                        for lhs, prods in grammar.items():
                            for prod in prods:
                                if (len(prod) == 3 and prod[0] in a
                                        and prod[1] in b and prod[2] in c):
                                    chart[i][j].add(lhs)
            _uc(chart[i][j])
    return start in chart[0][n]

# ── 3a. From-scratch generation ──
print("\n  --- (3a) From-scratch generation ---")
N_GEN = 50
n_gen_ok = n_lex_ok = n_total = 0
generations = []
# Subtree-exchange generation needs at least one sentence-root chunk to seed.
# Unsupervised training may not produce any (the climbing gate can stop before
# the top-level S), so fall back to the old _resolve_bag path if it's empty.
_use_replay = bool(getattr(trellis, "sentence_root_chunks", None))
print(f"    [gen mode: {'subtree-replay' if _use_replay else 'resolve_bag fallback'}]")
for i in range(N_GEN):
    try:
        if _use_replay:
            gen_text, gen_parse = trellis.generate_via_chunk_replay()
        else:
            gen_text, gen_parse = trellis.generate_sentence(debug=False)
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

# ── Uniqueness / novelty metrics (see unittests/README.md (3a) section) ──
# train_set = the unsupervised training sentences (Phase 2 only — the
# 90 sentences TRELLIS actually formed chunks on; the 200 primitives-
# only sentences don't contribute chunks, but we union them in too so
# novelty isn't inflated by primitive-only repeats).
train_set = set(s.strip() for s in unsup_sentences)
unique_texts = {g["text"].strip() for g in generations}
novel_count = sum(1 for g in generations if g["text"].strip() not in train_set)
gram_novel_count = sum(1 for g in generations
                       if g["gram_ok"] and g["text"].strip() not in train_set)
novelty_rate = novel_count / max(n_total, 1)
diversity    = len(unique_texts) / max(n_total, 1)
gram_novel_rate = gram_novel_count / max(n_total, 1)
print(f"  Novelty (not in train): {novel_count}/{n_total} "
      f"({100*novelty_rate:.1f}%)")
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

# ── 3b. Single-token masked completion ──
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
        completed, comp_parse = trellis.generate_sentence(
            masked_sentence=masked, debug=False)
    except Exception as e:
        completed, comp_parse = f"<failed: {e}>", None
    if comp_parse is not None and i < 20:
        try:
            comp_parse.visualize(f"{OUT_DIR}/mask_trees/mask_tree{i}")
        except Exception:
            pass
    comp_toks = completed.split()
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
print("SUMMARY — UNSUPERVISED HEURISTIC EVALUATION")
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
print(f"  step_pick_accuracy.csv,")
print(f"  generation_from_scratch.csv, generation_masked.csv,")
print(f"  final_ltm_data/ (saved TRELLIS state)")
print(f"  performance_summary.png (overview graphic)")

# =============================================================================
# SUMMARY GRAPHIC (mirrors hollow_learn_test's six-panel layout)
# =============================================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    f"TRELLIS — gen_learn (UNSUPERVISED) Performance Summary  "
    f"(SEED={SEED}, unsup_train={N_UNSUP_TRAIN}, "
    f"eval_test={len(test_hollow)})  "
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
axA.set_ylim(0, 1.15); axA.set_ylabel("Score")
axA.set_title(f"(A) End-to-end Parse Accuracy  (n={len(parse_rows)})", fontsize=11)
axA.grid(axis="y", alpha=0.3)

# Panel B — Step-pick.
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
axB.set_ylim(0, 1.15); axB.set_ylabel("Accuracy")
axB.set_title(
    f"(B) Step-pick — Climbing-Ancestor  (n={n_step_total} steps)",
    fontsize=11)
axB.tick_params(axis="x", labelsize=8)
axB.grid(axis="y", alpha=0.3)

# Panel C — Chunk-class per-class P/R/F1.
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
    axC.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cls_names, n_g)], fontsize=9)
    axC.set_ylim(0, 1.15); axC.set_ylabel("Score")
    axC.set_title(
        f"(C) Chunk-Class Probe   overall {100*chunk_acc:.1f}%   (n={len(test_y)})",
        fontsize=11)
    axC.legend(loc="lower right", fontsize=8)
    axC.grid(axis="y", alpha=0.3)

# Panel D — From-scratch generation (incl. uniqueness/novelty).
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
axD.set_title(f"(D) From-scratch Generation  (n={n_total})", fontsize=11)
axD.tick_params(axis="x", labelsize=8, rotation=15)
axD.grid(axis="y", alpha=0.3)

# Panel E — Mask completion.
axE = fig.add_subplot(2, 3, 5)
mask_metrics = ["Exact-token", "POS-class"]
mask_values  = [exact_hits / max(mask_total, 1),
                pos_hits   / max(mask_total, 1)]
mask_chance  = [1 / max(len(WORD_TO_POS), 1),
                1 / max(len(POS_LIST), 1)]
mask_colors  = ["#ff7f0e", "#e377c2"]
x = np.arange(len(mask_metrics)); w = 0.35
bars = axE.bar(x - w/2, mask_values, w, color=mask_colors,
               edgecolor="black", linewidth=0.5, label="TRELLIS")
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
axE.set_title(f"(E) Single-token Masked Completion  (n={mask_total})", fontsize=11)
axE.legend(loc="upper right", fontsize=9)
axE.grid(axis="y", alpha=0.3)

# Panel F — Scorecard.
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
