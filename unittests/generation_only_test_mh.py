"""
Generation-only script for a saved WEBSTER LTM.

This script loads a previously saved LTM (from
`unittests/gen_learn_test_mh/final_ltm_data`) and runs generation-only
procedures: from-scratch generation, masked completions, and masked
prediction. Results (parse-tree visuals and text outputs) are written to
`unittests/generation_only_test_mh`.
"""

import os
import shutil
import random

from parse_mh import WEBSTER
from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1

LOAD_LTM_DIR = "data/test_grammar2/ltm_194c09e0"
OUT_DIR = "unittests/generation_only_test_mh"


if not os.path.exists(LOAD_LTM_DIR):
    print(f"Saved LTM not found at {LOAD_LTM_DIR}. Aborting generation-only run.")
    raise SystemExit(1)

# reset output directory
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

webster = WEBSTER.load_state(LOAD_LTM_DIR)

# FROM-SCRATCH GENERATION
print("\n--- FROM-SCRATCH GENERATION ---")
gen_results_path = os.path.join(OUT_DIR, "generated_sentences.txt")
os.makedirs(os.path.join(OUT_DIR, "generated_trees"), exist_ok=True)
with open(gen_results_path, "w", encoding="utf-8") as gen_f:
    for i in range(10):
        sentence, parse = webster.generate_sentence(debug=False)
        print(f"Generated sentence [{i}]: \"{sentence}\"")
        gen_f.write(f"[{i}] {sentence}\n")
        if parse and hasattr(parse, "visualize"):
            parse.visualize(os.path.join(OUT_DIR, "generated_trees", f"generated_parse_tree{i}"))
print(f"Saved generated sentences to \"{gen_results_path}\"")

# Generate fresh random sentences using the grammar. These will be used as inputs for masked
# completion/prediction tests.
random.seed(42)
num_sentences = 100
generated = [generate("S", TEST_GRAMMAR1) for _ in range(num_sentences)]

# MASKED COMPLETION (single token)
print("\n--- MASKED COMPLETION (single token) ---")
for i in range(min(5, len(generated))):
    tokens = generated[i].split()
    if len(tokens) > 2:
        mask_idx = random.randint(1, len(tokens) - 1)
        tokens[mask_idx] = "[mask]"
    masked = " ".join(tokens)
    print(f"  Masked input: \"{masked}\"")
    completed, parse = webster.generate_sentence(masked_sentence=masked, debug=False)
    print(f"  Completed:    \"{completed}\"\n")
    if parse and hasattr(parse, "visualize"):
        os.makedirs(os.path.join(OUT_DIR, "masked_trees"), exist_ok=True)
        parse.visualize(os.path.join(OUT_DIR, "masked_trees", f"masked_parse_tree{i}"))

# MID-SENTENCE MASKED COMPLETION
print("\n--- MID-SENTENCE MASKED COMPLETION ---")
mid_mask_path = os.path.join(OUT_DIR, "mid_mask_results.txt")
os.makedirs(os.path.dirname(mid_mask_path), exist_ok=True)
with open(mid_mask_path, "w", encoding="utf-8") as mid_f:
    for i in range(min(10, len(generated))):
        tokens = generated[i].split()
        if len(tokens) < 3:
            continue
        mid = len(tokens) // 2
        original_token = tokens[mid]
        masked_tokens = tokens[:mid] + ["[mask]"] + tokens[mid + 1 :]
        masked = " ".join(masked_tokens)
        print(f"  Original:  \"{generated[i]}\"")
        print(f"  Masked:    \"{masked}\"  (replaced \"{original_token}\" at pos {mid})")
        mid_f.write(f"[{i}] original: {generated[i]}\n")
        mid_f.write(f"    masked:   {masked}\n")
        mid_f.write(f"    replaced: \"{original_token}\" at pos {mid}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=False)
        print(f"  Completed: \"{completed}\"\n")
        mid_f.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, "visualize"):
            os.makedirs(os.path.join(OUT_DIR, "mid_mask_trees"), exist_ok=True)
            parse.visualize(os.path.join(OUT_DIR, "mid_mask_trees", f"mid_mask_tree{i}"))
print(f"Saved mid-sentence mask results to \"{mid_mask_path}\"")

# MULTI-TOKEN MID-SENTENCE MASK
print("\n--- MULTI-TOKEN MID-SENTENCE MASK ---")
multi_mid_path = os.path.join(OUT_DIR, "multi_mid_mask_results.txt")
os.makedirs(os.path.dirname(multi_mid_path), exist_ok=True)
with open(multi_mid_path, "w", encoding="utf-8") as multi_f:
    for i in range(min(10, len(generated))):
        tokens = generated[i].split()
        if len(tokens) < 5:
            continue
        max_remove = min(4, len(tokens) - 2)
        num_remove = random.randint(2, max(2, max_remove))
        mid = max(1, (len(tokens) - num_remove) // 2)
        removed = tokens[mid : mid + num_remove]
        masked_tokens = tokens[:mid] + ["[mask]"] + tokens[mid + num_remove :]
        masked = " ".join(masked_tokens)
        print(f"  Original:  \"{generated[i]}\"")
        print(f"  Masked:    \"{masked}\"  (removed {num_remove} tokens: \"{' '.join(removed)}\" at pos {mid}-{mid+num_remove-1})")
        multi_f.write(f"[{i}] original: {generated[i]}\n")
        multi_f.write(f"    masked:   {masked}\n")
        multi_f.write(f"    removed: \"{' '.join(removed)}\" ({num_remove} tokens) at pos {mid}-{mid+num_remove-1}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=False)
        print(f"  Completed: \"{completed}\"\n")
        multi_f.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, "visualize"):
            os.makedirs(os.path.join(OUT_DIR, "multi_mid_trees"), exist_ok=True)
            parse.visualize(os.path.join(OUT_DIR, "multi_mid_trees", f"multi_mid_tree{i}"))
print(f"Saved multi-token mid mask results to \"{multi_mid_path}\"")

# MASKED PREDICTION (expand second half)
print("\n--- MASKED PREDICTION (expand second half) ---")
mask_pred_path = os.path.join(OUT_DIR, "masked_prediction_results.txt")
os.makedirs(os.path.dirname(mask_pred_path), exist_ok=True)
with open(mask_pred_path, "w", encoding="utf-8") as mask_f:
    for i in range(min(10, len(generated))):
        tokens = generated[i].split()
        if len(tokens) < 2:
            continue
        split_point = len(tokens) // 2
        prefix = tokens[:split_point]
        masked = " ".join(prefix + ["[mask]"])
        print(f"  Original:  \"{generated[i]}\"")
        print(f"  Masked:    \"{masked}\"")
        mask_f.write(f"[{i}] original: {generated[i]}\n")
        mask_f.write(f"    masked:   {masked}\n")
        completed, parse = webster.generate_sentence(masked_sentence=masked, debug=False)
        print(f"  Completed: \"{completed}\"\n")
        mask_f.write(f"    completed: {completed}\n\n")
        if parse and hasattr(parse, "visualize"):
            os.makedirs(os.path.join(OUT_DIR, "masked_pred_trees"), exist_ok=True)
            parse.visualize(os.path.join(OUT_DIR, "masked_pred_trees", f"masked_pred_tree{i}"))
print(f"Saved masked prediction results to \"{mask_pred_path}\"")

# Visualize LTM
webster.visualize_ltm(os.path.join(OUT_DIR, "final_ltm"), max_depth=3)

print(f"Generation-only outputs written to '{OUT_DIR}'")
