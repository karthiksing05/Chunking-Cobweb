import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parse_mh import LongTermMemory
from viz import CategorizePathVisualizer

# --- Set the LTM directory to visualize ---
LTM_FILENAME = "unittests/gen_learn_test_mh/final_ltm_data/ltm"

# --- Load LTM ---
ltm = LongTermMemory.load_state(LTM_FILENAME)

# --- Visualize content hierarchy (no red paths) ---
content_html = CategorizePathVisualizer.build_standalone_html(
    ltm.content_hierarchy.root,
    path_hashes=[],
)
content_out = "sandbox_content_hierarchy.html"
with open(content_out, "w", encoding="utf-8") as f:
    f.write(content_html)
print(f"Content hierarchy saved to: {content_out}")

# --- Visualize context hierarchy (no red paths) ---
context_html = CategorizePathVisualizer.build_standalone_html(
    ltm.context_hierarchy.root,
    path_hashes=[],
)
context_out = "sandbox_context_hierarchy.html"
with open(context_out, "w", encoding="utf-8") as f:
    f.write(context_html)
print(f"Context hierarchy saved to: {context_out}")
