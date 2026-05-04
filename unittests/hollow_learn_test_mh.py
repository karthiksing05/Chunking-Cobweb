"""
Hollow Learning Test (Multi-Hierarchy) — trains WEBSTER from a corpus of
human-annotated "hollow" parse trees (merge recipes), then evaluates
learning + generation quality.

Hollow JSON format:
  { "sentence": "the dog chased the cat",
    "merges": [{"left": 0, "right": 1}, ...] }

Workflow:
  1. Load hollow JSONs from HOLLOW_CORPUS_DIR
  2. For each hollow tree, build primitives and replay merges via WEBSTER
  3. Learn from each completed tree
  4. Evaluate on held-out test sentences (auto-parsed)
  5. Run generation tests (from-scratch, masked, multi-mask, prefix)
"""

import sys, os

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1, TEST_GRAMMAR2, TEST_CORPUS2
from parse_mh import WEBSTER, FiniteParseTree
import shutil
import json
import random
import glob

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = "unittests/hollow_learn_test_mh"
HOLLOW_CORPUS_DIR = "data/test_hollow_grammar_1"  # where hollow JSONs live
VIZ_INTERMEDIATES = True

CONTEXT_LENGTH = 3
THRESHOLD = 30
PRIMITIVES_FIRST = 200  # first N trees train with infinite threshold (primitives only)

corpus = TEST_CORPUS1
grammar = TEST_GRAMMAR1

# ── Setup ──────────────────────────────────────────────────────────────────
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# ── Load hollow corpus ────────────────────────────────────────────────────
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
    else:
        print(f"[WARN] Skipping non-hollow JSON (missing sentence/merges): {p}")

print(f"Loaded {len(hollow_corpus)} hollow parse trees from {HOLLOW_CORPUS_DIR}")

if not hollow_corpus:
    print("[ERROR] No hollow parse trees found. Create some with the hollow editor first.")
    sys.exit(1)

# ── Initialise WEBSTER ────────────────────────────────────────────────────
webster = WEBSTER(
    corpus,
    instance_basic_level=True,
    context_length=CONTEXT_LENGTH,
    threshold=THRESHOLD,
    content_alpha=1e-3,
    context_alpha=1e-3,
    content_bl_alpha=1e-1,
    context_bl_alpha=1,
    bow=False,
    empty_weighting=True,
    chunk_context=False,
    # context_n_iterations=5,
    weighting="binary",
    categorization_mode="dfs",
    depth_max_content=10,
    depth_max_context=10,
    branch_max_content=10,
    branch_max_context=10,
    context_attr_weights={6: 2.0},   # attr 6 = content-ref when context_length=3
    content_attr_weights={0: 1.0, 1: 1.0},  # boost left & right child attrs
)

# ── Phase 1: primitives-only on random sentences ──────────────────────────
print(f"\n=== PHASE 1: PRIMITIVES ONLY ({PRIMITIVES_FIRST} random sentences) ===")
for i in range(PRIMITIVES_FIRST):
    sentence = generate("S", grammar)
    parse_tree = webster.parse_sentence(
        sentence,
        threshold=1e9,
        new_vocab=True,
        learning=True,
        debug=False,
    )

    if i % 5 == 0 and VIZ_INTERMEDIATES:
        parse_tree.visualize(f"{OUT_DIR}/train_trees/primitives_tree{i}")
        webster.visualize_ltm(f"{OUT_DIR}/ltms/primitives_ltm{i}", max_depth=3)

    print(f"  [{i+1}/{PRIMITIVES_FIRST}] Primitives: \"{sentence}\"")

# ── Phase 2: replay hollow trees with merges ──────────────────────────────
print(f"\n=== PHASE 2: HOLLOW CORPUS TRAINING (size = {len(hollow_corpus)}) ===")
for i, hollow in enumerate(hollow_corpus):
    sentence = hollow["sentence"]
    merges = hollow["merges"]

    # Build primitives for this sentence
    tree = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    tree.build_primitives(sentence, threshold=THRESHOLD)

    # Replay the human-annotated merge sequence
    for m in merges:
        try:
            tree.apply_candidate(m["left"], m["right"])
        except Exception as e:
            print(f"  [WARN] Merge ({m['left']}, {m['right']}) failed on sentence "
                  f"\"{sentence}\": {e}")

    # Learn from the completed tree
    webster.ltm.add_parse_tree(tree, shuffle=True, debug=False)

    if i % 5 == 0 and VIZ_INTERMEDIATES:
        tree.visualize(f"{OUT_DIR}/train_trees/train_parse_tree{i}")
        webster.visualize_ltm(f"{OUT_DIR}/ltms/ltm{i}", max_depth=3)

    print(f"  [{i+1}/{len(hollow_corpus)}] Trained on: \"{sentence}\"  "
          f"({len(merges)} merges)")

# ── Save state ─────────────────────────────────────────────────────────────
SAVE_DIR = f"{OUT_DIR}/final_ltm_data"
webster.save_state(SAVE_DIR)
print(f"\nSaved Final LTM to \"{SAVE_DIR}\"!")
webster.visualize_ltm(f"{OUT_DIR}/final_ltm", max_depth=3)

# ── Test: auto-parse held-out sentences ────────────────────────────────────
print("\n=== AUTO-PARSE TEST SENTENCES ===")
num_test = 20
test_documents = [generate("S", grammar) for _ in range(num_test)]

for i, test in enumerate(test_documents):
    parse_tree = webster.parse_sentence(
        test,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=False,
        debug=True,
    )
    parse_tree.visualize(f"{OUT_DIR}/test_trees/test_parse_tree{i}")
    print(f"  Test tree {i}: \"{test}\"")

# ── Test: random / fake sentences ──────────────────────────────────────────
print("\n=== FAKE SENTENCE PARSING ===")
fake_sentences = [
    " ".join([random.choice(corpus) for _ in range(random.randint(3, 8))])
    for _ in range(10)
]
for i, fake in enumerate(fake_sentences):
    parse_tree = webster.parse_sentence(
        fake,
        threshold=THRESHOLD,
        new_vocab=True,
        learning=False,
        debug=True,
    )
    parse_tree.visualize(f"{OUT_DIR}/fake_trees/fake_parse_tree{i}")
    print(f"  Fake tree {i}: \"{fake}\"")

# ── Generation: from-scratch ──────────────────────────────────────────────
print("\n=== FROM-SCRATCH GENERATION ===")
gen_path = f"{OUT_DIR}/generated_sentences.txt"
os.makedirs(os.path.dirname(gen_path), exist_ok=True)
with open(gen_path, "w") as gf:
    for i in range(10):
        sentence, parse = webster.generate_sentence(debug=True)
        print(f"  Generated [{i}]: \"{sentence}\"")
        gf.write(f"[{i}] {sentence}\n")
        if parse and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/generated_trees/generated_parse_tree{i}")
print(f"Saved generated sentences to \"{gen_path}\"")

# ── Generation: single-token mask ─────────────────────────────────────────
print("\n=== MASKED COMPLETION (single token) ===")
for i in range(min(5, len(test_documents))):
    tokens = test_documents[i].split()
    if len(tokens) > 2:
        mask_idx = random.randint(1, len(tokens) - 1)
        tokens[mask_idx] = "[mask]"
    masked = " ".join(tokens)
    print(f"  Masked:    \"{masked}\"")
    completed, parse = webster.generate_sentence(masked_sentence=masked, debug=True)
    print(f"  Completed: \"{completed}\"\n")

# ── Generation: mid-sentence mask ─────────────────────────────────────────
print("\n=== MID-SENTENCE MASKED COMPLETION ===")
mid_mask_path = f"{OUT_DIR}/mid_mask_results.txt"
os.makedirs(os.path.dirname(mid_mask_path), exist_ok=True)
with open(mid_mask_path, "w") as mf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 3:
            continue
        mid = len(tokens) // 2
        original_token = tokens[mid]
        masked_tokens = tokens[:mid] + ["[mask]"] + tokens[mid + 1:]
        masked = " ".join(masked_tokens)
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"  (replaced \"{original_token}\" at pos {mid})")
        mf.write(f"[{i}] original: {test_documents[i]}\n")
        mf.write(f"    masked:   {masked}\n")
        mf.write(f"    replaced: \"{original_token}\" at pos {mid}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=True)
        print(f"  Completed: \"{completed}\"\n")
        mf.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/mid_mask_trees/mid_mask_tree{i}")
print(f"Saved mid-sentence mask results to \"{mid_mask_path}\"")

# ── Generation: multi-token mid mask ──────────────────────────────────────
print("\n=== MULTI-TOKEN MID-SENTENCE MASK ===")
multi_mid_path = f"{OUT_DIR}/multi_mid_mask_results.txt"
os.makedirs(os.path.dirname(multi_mid_path), exist_ok=True)
with open(multi_mid_path, "w") as mmf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 5:
            continue
        max_remove = min(4, len(tokens) - 2)
        num_remove = random.randint(2, max(2, max_remove))
        mid = max(1, (len(tokens) - num_remove) // 2)
        removed = tokens[mid:mid + num_remove]
        masked_tokens = tokens[:mid] + ["[mask]"] + tokens[mid + num_remove:]
        masked = " ".join(masked_tokens)
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"  (removed {num_remove} tokens: "
              f"\"{' '.join(removed)}\" at pos {mid}-{mid+num_remove-1})")
        mmf.write(f"[{i}] original: {test_documents[i]}\n")
        mmf.write(f"    masked:   {masked}\n")
        mmf.write(f"    removed: \"{' '.join(removed)}\" ({num_remove} tokens) "
                  f"at pos {mid}-{mid+num_remove-1}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=True)
        print(f"  Completed: \"{completed}\"\n")
        mmf.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/multi_mid_trees/multi_mid_tree{i}")
print(f"Saved multi-token mid mask results to \"{multi_mid_path}\"")

# ── Generation: prefix prediction (expand second half) ────────────────────
print("\n=== MASKED PREDICTION (expand second half) ===")
mask_pred_path = f"{OUT_DIR}/masked_prediction_results.txt"
os.makedirs(os.path.dirname(mask_pred_path), exist_ok=True)
with open(mask_pred_path, "w") as mpf:
    for i in range(min(10, len(test_documents))):
        tokens = test_documents[i].split()
        if len(tokens) < 2:
            continue
        split_point = len(tokens) // 2
        prefix = tokens[:split_point]
        masked = " ".join(prefix + ["[mask]"])
        print(f"  Original:  \"{test_documents[i]}\"")
        print(f"  Masked:    \"{masked}\"")
        mpf.write(f"[{i}] original: {test_documents[i]}\n")
        mpf.write(f"    masked:   {masked}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=True)
        print(f"  Completed: \"{completed}\"\n")
        mpf.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, "visualize"):
            parse.visualize(f"{OUT_DIR}/masked_pred_trees/masked_pred_tree{i}")
print(f"Saved masked prediction results to \"{mask_pred_path}\"")
