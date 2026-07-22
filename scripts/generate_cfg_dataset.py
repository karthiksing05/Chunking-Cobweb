"""
Generate synthetic ground-truth trees for TEST_GRAMMAR_{SMALL,MED,LARGE}
in the hollow-corpus JSON format.

The hollow corpus under ``data/test_hollow_grammar_1/`` was hand-
annotated; this script derives equivalent annotations directly from
the CFG via ``generate_with_merges``, which binarizes the derivation
tree left-to-right. Output JSON matches the hollow format
({"sentence": str, "merges": list[dict[left, right]]}) so
``hollow_learn_test_mh.py`` can train on it without any code change.

Usage:
    python scripts/generate_cfg_dataset.py
"""
import os, sys, json, random, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from util.cfg import (generate_with_merges,
                      TEST_GRAMMAR_SMALL, TEST_GRAMMAR_MED, TEST_GRAMMAR_LARGE)

# (out_dir, grammar, seed, n_target, max_words, flatten_at_parent).
#
# Per-grammar flatten choices:
#   SMALL — flatten ``VP`` only. The grammar has no VPobj nonterminal.
#   MED / LARGE — flatten ``VP`` and ``VPobj``. Both nonterminals get
#     bubbled up to the running S-chunk so that V / NP / PP merge
#     incrementally (matching the hand-annotated hollow corpus
#     convention: ``(0,2) "a park admired"`` brackets — V glued into
#     the running S, no VP constituent). RelClause in LARGE is NOT
#     flattened: it is a real constituent that the parser should
#     learn as a chunk type.
CONFIGS = [
    # SMALL: minimal grammar (S→NP VP; VP=V NP | V; 4 N, 4 V, 2 Det),
    # tighter length cap because the surface forms are short.
    ("data/cfg_grammar_small", TEST_GRAMMAR_SMALL, 13, 300,  8, ("VP",)),
    ("data/cfg_grammar_med",   TEST_GRAMMAR_MED,   13, 300, 10, ("VP", "VPobj")),
    # LARGE gets 2x the training data and a tighter length cap.
    # Rationale: TEST_GRAMMAR_LARGE adds RelClauses on top of MED's
    # productions. The parser needs proportionally more examples per
    # chunk type to converge. Tighter length (10) also kills the deep
    # RelClause-in-RelClause cases that dominate the long tail.
    ("data/cfg_grammar_large", TEST_GRAMMAR_LARGE, 23, 400, 10, ("VP", "VPobj")),
]


def main():
    for out_dir, grammar, seed, n_target, max_words, flatten in CONFIGS:
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= n_target:
            print(f"  {out_dir}: already has ≥{n_target} files, skipping")
            continue
        os.makedirs(out_dir, exist_ok=True)
        random.seed(seed)
        seen: set[str] = set()
        n_written = 0
        max_attempts = n_target * 60
        attempts = 0
        n_too_long = 0
        while n_written < n_target and attempts < max_attempts:
            attempts += 1
            text, merges = generate_with_merges(
                "S", grammar, flatten_at_parent=flatten)
            text = text.strip()
            if not text or text in seen:
                continue
            n_words = len(text.split())
            if n_words > max_words:
                n_too_long += 1
                continue
            seen.add(text)
            short_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
            payload = {"sentence": text, "merges": merges}
            path = os.path.join(out_dir, f"cfg_{short_hash}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            n_written += 1
        print(f"  {out_dir}: wrote {n_written} unique sentences "
              f"(max_words={max_words}, flatten={flatten}; "
              f"{attempts} attempts, {n_too_long} rejected for length)")


if __name__ == "__main__":
    main()
