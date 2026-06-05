"""
Minimally Reproducible Crash: SIGSEGV in content_hierarchy.ifit()

Root cause (confirmed):
  `ensure_val_ancestors()` in cobweb_discrete_tree.cpp iterates
  `val_to_node` and walks each stored node's parent chain:

      CobwebDiscreteNode* p = leaf_node;
      while (p != nullptr) {
          path.push_back(p);
          p = p->parent;  // use-after-free if leaf_node is a freed pointer
      }

  `lca_similarity()` also walks `val_to_node` pointers directly before
  our fix:

      CobwebDiscreteNode* p = node1;
      while (p != nullptr) { ancestors1.insert(p); p = p->parent; }

  A `val_to_node` entry can point to a freed/stale node via two paths:

  Path A — ensure_val_ancestors (stale after SPLIT):
    When the context hierarchy undergoes a SPLIT operation, an internal node
    I is deleted.  If `val_to_node` holds a pointer to I (because I was
    previously a leaf and registered via register_ref_val before fringe-split
    promoted it to internal), then the next `ensure_val_ancestors()` call
    dereferences a freed pointer → SIGSEGV.

  Path B — lca_similarity (stale after SPLIT, directly):
    `lca_similarity()` was called without refreshing val_to_node first.
    If the context hierarchy had changed (SPLIT) between the last
    ensure_val_ancestors() call and this lca_similarity() call, val_to_node
    may still contain stale pointers.

Fix applied in cobweb_discrete_tree.cpp:
  1. `ensure_val_ancestors()`: Before walking any val_to_node pointer, build
     a live-node set from the ref_tree and evict any val_to_node entries
     whose pointer is not in the live set.
  2. `lca_similarity()`: Call `ensure_val_ancestors()` at entry to guarantee
     val_to_node is clean before dereferencing stored node pointers.

Crash confirmed at:
  ~sentence 63 originally, then ~sentence 114 after partial fix, then
  resolved (all 157 sentences complete without crash) after the full fix.

Expected behaviour (after fix):
  The test completes without any signal, exit code 0.

Run:
    python tests/pivot/test_mrp_segfault_ref_split.py    # should print "OK"
    pytest -s tests/pivot/test_mrp_segfault_ref_split.py  # should pass
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import TRELLIS


def test_no_segfault_after_context_split():
    """
    Parse 157 randomly generated sentences with learning=True, using the same
    parameters as gen_learn_test_mh.py (the original crash reproducer).

    The crash path:
      - First 100 sentences: primitives-only mode (threshold=1e9)
        Context hierarchy grows and undergoes SPLIT operations, which delete
        intermediate nodes.  val_to_node in the content hierarchy may now hold
        pointers to freed context nodes.
      - Sentences 100+: threshold=30 enables composite learning.
        content_hierarchy.ifit() → cobweb() → increment_counts() →
        log_prob_instance() → lca_similarity() → ensure_val_ancestors() →
        walks val_to_node pointer chain → use-after-free → SIGSEGV.

    Confirmed exit code 139 (SIGSEGV) at sentence ~114 before fix.
    """
    sentences = [generate("S", TEST_GRAMMAR1) for _ in range(157)]

    trellis = TRELLIS(
        TEST_CORPUS1,
        context_length=3,
        threshold=30,
        content_alpha=1e-3,
        context_alpha=1e-3,
        content_bl_alpha=1e-1,
        context_bl_alpha=1,
        bow=False,
        empty_weighting=True,
        weighting="binary",
        categorization_mode="dfs",
        depth_max_content=1000,
        depth_max_context=1000,
        branch_max_content=1000,
        branch_max_context=1000,
    )

    PRIMITIVES_FIRST = 100
    for i, sentence in enumerate(sentences):
        p_threshold = 1e9 if i < PRIMITIVES_FIRST else 30
        trellis.parse_sentence(
            sentence,
            threshold=p_threshold,
            new_vocab=True,
            learning=True,
            debug=False,
        )

    # If we reach here without a segfault the fix is working.


if __name__ == "__main__":
    test_no_segfault_after_context_split()
    print("OK")
